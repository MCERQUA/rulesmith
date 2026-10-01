"""A briefing: what a judge reads before the input, which is where guidance works."""

from test_graph import task_data

from rulesmith.graph import Plan
from rulesmith.optimize import SearchConfig, proposer_prompt
from rulesmith.rules import compile_rules, render_rules
from rulesmith.runtime import DecisionProgram

BRIEFED = (
    'ask kind = "What kind of answer does this question ask for?" '
    '[number: "A count or a measurement."] [place: "Somewhere on a map."] '
    'briefed "Classify the ANSWER, not the subject. \\"How deep\\" asks for a number."\n'
    'when kind == "number": numeric\n'
    "always: location"
)


def test_a_briefing_round_trips_and_keeps_its_place_among_the_clauses():
    once = render_rules(compile_rules(BRIEFED))
    assert (
        'briefed "Classify the ANSWER, not the subject. \\"How deep\\" asks for a number."' in once
    )
    assert render_rules(compile_rules(once)) == once
    plan = compile_rules(once)
    assert plan.nodes["kind"].briefing.startswith("Classify the ANSWER")
    # With the other clauses too, each in its place: options, briefing, temperature, given.
    full = BRIEFED.replace('number."\n', 'number." calibrated 1.5 given [observation]\n', 1)
    assert render_rules(compile_rules(full)) == render_rules(
        compile_rules(render_rules(compile_rules(full)))
    )
    assert compile_rules(full).nodes["kind"].temperature == 1.5


def test_the_judge_reads_the_briefing_before_the_input(provider, client):
    provider.body["answers"] = {
        "kind": {
            "type": "choice",
            "choice": "number",
            "probabilities": {"number": 0.9, "place": 0.1},
            "confidence": 0.9,
        }
    }
    run = DecisionProgram(compile_rules(BRIEFED), client, "fixture")("How deep is the ocean?")
    assert run.label == "numeric"
    state = provider.requests[-1][1]["state"]
    # The briefing is the first thing in the state, ahead of the input, and the question itself
    # is unchanged: what the judge is shown is where guidance works, not what it is asked.
    assert list(state) == ["briefing", "observation"]
    assert state["briefing"].startswith("Classify the ANSWER") and state["observation"].startswith(
        "How deep"
    )
    assert (
        provider.requests[-1][1]["questions"]["kind"]["instructions"]
        == "What kind of answer does this question ask for?"
    )


def test_a_briefing_survives_json_and_the_proposer_is_told_about_it():
    plan = compile_rules(BRIEFED)
    again = Plan.model_validate_json(plan.model_dump_json())
    assert again.nodes["kind"].briefing == plan.nodes["kind"].briefing
    from rulesmith.graph import Task

    prompt = proposer_prompt(Task.model_validate(task_data()), SearchConfig())
    assert "briefed" in prompt


def test_a_clause_on_its_own_line_belongs_to_the_judgment_above_it():
    # A long briefing invites a line break before it; the proposer wrote half its briefings so.
    split = BRIEFED.replace(' briefed "', '\nbriefed "', 1)
    assert "\nbriefed" in split
    assert render_rules(compile_rules(split)) == render_rules(compile_rules(BRIEFED))
    # Any of a judgment's trailing clauses may continue so, and line numbers still name the rule.
    lines = BRIEFED.splitlines()
    lines[0] += " calibrated 1.5"
    joined = render_rules(compile_rules("\n".join(lines)))
    apart = lines[0].replace(" calibrated 1.5", "\ncalibrated 1.5")
    assert render_rules(compile_rules("\n".join([apart, *lines[1:]]))) == joined
