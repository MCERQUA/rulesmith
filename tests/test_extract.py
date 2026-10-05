"""Extraction: a generative model reads free text into a few typed fields, and rules read those
fields the way they read an input's own, so a decision on free text gets fields to test."""

import json

import dspy
import pytest
from conftest import chat_reply
from test_cli import invoke, write
from test_round_trip import round_trips

from rulesmith.extract import Extractor
from rulesmith.graph import Plan, Task
from rulesmith.optimize import GraphAdapter, SearchConfig, proposer_prompt
from rulesmith.rules import compile_rules, parts
from rulesmith.runtime import DecisionProgram, Models

REFUNDS = (
    'extract claim = "Read the refund request." '
    "{amount in [0, 10000], refund_asked one of [yes, no]}\n"
    'when claim.refund_asked == "yes" and claim.amount < 50: auto_refund\n'
    "always: review"
)


def extractor(provider) -> Extractor:
    return Extractor(
        dspy.LM(
            "openai/extractor-model",
            api_base=provider.url + "/v1",
            api_key="test-placeholder",
            cache=False,
            num_retries=0,
        )
    )


def chats(provider) -> list[dict]:
    return [request for path, request in provider.requests if path == "/v1/chat/completions"]


def decide(provider, completion: str, text: str = "Please refund the $30 I was double charged."):
    provider.completion = completion
    program = DecisionProgram(
        compile_rules(REFUNDS), None, "none", models=Models(extractor=extractor(provider))
    )
    return program(state=text)


def test_an_extraction_and_the_rules_on_its_fields_read_back_the_same():
    plan = round_trips(
        "read channel one of [email, chat]\n"
        'extract claim = "Read the request." {amount in [0, 10000], refund_asked one of [yes, no]} '
        'when channel == "email"\n'
        'when claim.refund_asked == "yes" and claim.amount < 50: auto_refund\n'
        "always: review"
    )
    reads = {
        node.path[0]: node for node in plan.nodes.values() if node.type == "read" and node.source
    }
    assert set(reads) == {"amount", "refund_asked"}
    assert {node.source for node in reads.values()} == {"claim"}
    assert plan.extracts()


def test_rules_decide_on_the_fields_a_model_extracted(provider):
    decided = decide(provider, chat_reply(amount=30, refund_asked="yes"))
    assert decided.label == "auto_refund"
    (asked,) = chats(provider)
    assert "Read the refund request." in json.dumps(asked["messages"])
    # An extraction is a model call, counted and priced like a judge's.
    assert decided.calls["claim"]["model"] == "openai/extractor-model"
    assert decided.calls["claim"]["extracted"] == ["amount", "refund_asked"]


@pytest.mark.parametrize(
    "completion",
    [
        chat_reply(amount=99999, refund_asked="yes"),  # outside the declared bounds
        chat_reply(amount=30, refund_asked="maybe"),  # not one of the categories
        chat_reply(amount=30, refund_asked="None"),  # the model could not say
    ],
)
def test_a_field_the_model_could_not_fill_within_its_domain_is_absent(provider, completion):
    assert decide(provider, completion).label == "review"


def test_a_reply_that_does_not_parse_leaves_every_field_absent_and_says_why(provider):
    decided = decide(provider, "I cannot help with that.")
    assert decided.label == "review"
    assert decided.calls["claim"]["error"]


def test_the_same_text_is_extracted_once(provider):
    provider.completion = chat_reply(amount=30, refund_asked="yes")
    program = DecisionProgram(
        compile_rules(REFUNDS), None, "none", models=Models(extractor=extractor(provider))
    )
    program(state="Refund my $30.")
    program(state="Refund my $30.")
    assert len(chats(provider)) == 1


def test_a_graph_that_extracts_needs_an_extractor():
    with pytest.raises(ValueError, match="extract"):
        DecisionProgram(compile_rules(REFUNDS), None, "none")


def test_only_an_extraction_has_fields_to_read():
    with pytest.raises(ValueError, match="extraction"):
        Plan.model_validate(
            {
                "nodes": {
                    "q": {
                        "type": "call",
                        "question": {"type": "noul", "instructions": "Is it urgent?"},
                        "inputs": {"text": {"type": "state"}},
                    },
                    "field": {"type": "read", "source": "q", "path": ["x"], "bounds": [0, 1]},
                },
                "outputs": {"label": {"type": "node", "node": "field"}},
            }
        )


def test_search_prices_an_extraction_and_is_told_how_to_write_one(provider):
    task = Task.model_validate(
        {
            "description": "Refund or review.",
            "labels": ["auto_refund", "review"],
            "train": [{"id": "a", "state": "Refund my $30.", "label": "auto_refund"}],
            "validation": [{"id": "b", "state": "x", "label": "review"}],
            "test": [{"id": "c", "state": "y", "label": "review"}],
        }
    )
    provider.completion = chat_reply(amount=30, refund_asked="yes")
    config = SearchConfig(extract_cost=0.1)
    adapter = GraphAdapter(task, None, "none", config, models=Models(extractor=extractor(provider)))
    batch = adapter.evaluate(task.train, parts(compile_rules(REFUNDS)), capture_traces=True)
    assert batch.scores == [pytest.approx(0.9)]
    assert "extract" in batch.trajectories[0]["Feedback"]
    assert "extract NAME = " not in proposer_prompt(task, config)
    assert "extract NAME = " in proposer_prompt(task, config, extracting=True)


def test_the_command_line_extracts_with_the_model_it_is_given(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    provider.completion = chat_reply(amount=30, refund_asked="yes")
    rules = tmp_path / "graph.rules"
    rules.write_text(REFUNDS + "\n")
    state = write(tmp_path / "state.json", "Refund my $30.")
    model = [
        "--extractor-model",
        "openai/extractor-model",
        "--extractor-base-url",
        provider.url + "/v1",
    ]
    result = invoke("run", rules, state, *model)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["label"] == "auto_refund"

    missing = invoke("run", rules, state)
    assert missing.returncode != 0
    assert "--extractor-model" in missing.stderr


def test_a_graph_that_extracts_validates_and_draws_without_a_model(tmp_path):
    task = write(
        tmp_path / "task.json",
        {
            "description": "Refund or review.",
            "labels": ["auto_refund", "review"],
            "train": [{"id": "a", "state": "Refund my $30.", "label": "auto_refund"}],
            "validation": [{"id": "b", "state": "x", "label": "review"}],
            "test": [{"id": "c", "state": "y", "label": "review"}],
        },
    )
    rules = tmp_path / "graph.rules"
    rules.write_text(REFUNDS + "\n")
    assert invoke("validate", task, rules).returncode == 0
    drawn = invoke("draw", rules)
    assert drawn.returncode == 0, drawn.stderr
    # The extraction feeds each field the rules test.
    assert "claim.amount" in drawn.stdout and "extract" in drawn.stdout
