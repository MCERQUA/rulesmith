"""A choice over the task's own labels, written '[task labels]' instead of restated in full."""

import json
import re

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_cli import invoke, write
from test_graph import task_data

from rulesmith.graph import Plan, Task
from rulesmith.optimize import SearchConfig, optimize_plan, proposer_prompt
from rulesmith.rules import RuleError, compile_rules, parts, render_rules, whole

SHORT = "[task labels]"


def described() -> Task:
    return Task.model_validate(task_data() | {"descriptions": {"accept": "It can go ahead."}})


def choice_plan(criteria: dict[str, str]) -> Plan:
    return compile_rules(
        "always: ask "
        + json.dumps("Which is it?")
        + " "
        + " ".join(f"[{label}: {json.dumps(text)}]" for label, text in criteria.items())
    )


def question(plan: Plan):
    (call,) = [node for node in plan.nodes.values() if node.type == "call"]
    return call.question


def test_every_label_is_described_and_an_undescribed_one_by_its_own_words():
    task = Task.model_validate(
        task_data()
        | {"labels": ["accept", "needs_review"], "descriptions": {"accept": "It can go ahead."}}
        | {"validation": [{"id": "b", "state": "request B", "label": "accept"}]}
    )
    assert task.criteria() == {"accept": "It can go ahead.", "needs_review": "needs review"}


def test_the_short_form_asks_over_every_task_label_as_the_task_describes_it():
    criteria = described().criteria()
    plan = compile_rules(
        f'ask verdict = "Which is it?" {SHORT}\nwhen verdict == "review": review\nalways: accept',
        criteria,
    )
    asked = question(plan)
    assert asked.type == "choice" and asked.criteria == criteria


def test_the_short_form_without_a_task_is_refused():
    with pytest.raises(RuleError, match="task"):
        compile_rules(f'always: ask "Which is it?" {SHORT}')


LABEL = st.from_regex(r"[a-z][a-z_]{0,7}", fullmatch=True)
DESCRIPTION = st.from_regex(r"[A-Za-z][A-Za-z ,.']{0,20}[a-z.]", fullmatch=True)


@given(st.dictionaries(LABEL, DESCRIPTION, min_size=2, max_size=8))
def test_a_choice_over_exactly_the_task_labels_round_trips_in_the_short_form(criteria):
    plan = choice_plan(criteria)
    text = render_rules(plan, criteria)
    assert SHORT in text
    assert render_rules(compile_rules(text, criteria), criteria) == text
    assert whole(parts(plan, criteria), criteria) == plan


@pytest.mark.parametrize(
    "change",
    [
        lambda criteria: criteria | {"accept": "It may go ahead after a look."},
        lambda criteria: criteria | {"escalate": "Someone senior must see it."},
    ],
    ids=["a description differs", "the task has a label the question lacks"],
)
def test_a_choice_that_differs_from_the_task_labels_is_written_in_full(change):
    asked = described().criteria()
    plan = choice_plan(asked)
    task_criteria = change(dict(asked))
    text = render_rules(plan, task_criteria)
    assert SHORT not in text
    # Read back against the same task, the full form is the same question.
    assert question(compile_rules(text, task_criteria)).criteria == asked
    assert whole(parts(plan, task_criteria), task_criteria) == plan


def test_the_proposer_is_shown_a_short_form_that_compiles_against_the_task():
    task = described()
    prompt = proposer_prompt(task, SearchConfig())
    (form,) = re.findall(r"'(ask \w+ = \"[^\"]*\" \[task labels\])'", prompt)
    name = form.split()[1]
    plan = compile_rules(
        f'{form}\nwhen {name} == "review": review\nalways: accept', task.criteria()
    )
    assert question(plan).criteria == task.criteria()


def test_search_hands_the_proposer_the_short_form_and_compiles_its_answer(provider, client):
    task = described()
    criteria = task.criteria()

    def answer_every_question(request):
        # The seed answers wrongly, so search has a failure to reflect on.
        provider.answers = {
            key: {
                "type": "choice",
                "choice": "review" if asked["instructions"].startswith("Classify") else "accept",
                "confidence": 0.9,
                "probabilities": {"accept": 0.5, "review": 0.5},
            }
            for key, asked in request["questions"].items()
        }

    provider.on_request = answer_every_question
    seed = compile_rules(f'always: ask "Classify the request." {SHORT}', criteria)
    shown = []

    def reflect(prompt):
        program = prompt.split("The whole program as it stands:\n", 1)[1]
        shown.append(program.split("\n\nYour previous proposal", 1)[0])
        if prompt.rsplit("Return the part named ", 1)[1].startswith("readings"):
            return ""
        return f'always: ask "Can this request go ahead as it stands?" {SHORT}'

    best, report = optimize_plan(
        task,
        seed,
        client,
        "fixture-model",
        reflect,
        SearchConfig(max_metric_calls=4, reflection_batch=1, repair_attempts=1),
    )
    assert shown and all(SHORT in part for part in shown)
    assert "It can go ahead." not in "".join(shown)
    # An answer in the short form compiles as it is, with nothing sent back to repair.
    assert report["repairs"] == 0 and report["invalid_candidate_evaluations"] == 0
    assert question(best).criteria == criteria


def test_a_rules_file_may_ask_over_the_task_labels(tmp_path):
    task = write(tmp_path / "task.json", task_data())
    rules = tmp_path / "policy.rules"
    rules.write_text(f'always: ask "Which is it?" {SHORT}\n')
    result = invoke("validate", task, rules)
    assert result.returncode == 0, result.stderr
