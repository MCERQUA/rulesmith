"""Watching a decision in production: its traces say which rule decided each case and which model
each call went to, and a rule deciding a very different share of real traffic than of the examples
it was counted on is flagged before its accuracy is known to have moved."""

import json

from test_cli import invoke

from rulesmith.rules import compile_rules
from rulesmith.runtime import DecisionProgram, Judge, Models

# Counted on 100 examples: rule_1 decided 10 of them, rule_2 the other 90.
MEASURED = (
    "read amount in [0, 10000]\n"
    "when amount >= 500: review calibrated 10/10 of 100\n"
    "always: refund calibrated 80/90 of 100\n"
)


def traffic(graph, trace, amounts):
    program = DecisionProgram(compile_rules(graph), None, "none", trace=trace)
    for amount in amounts:
        program(state={"amount": amount})


def test_a_trace_names_the_model_each_call_went_to(tmp_path, client, larger, provider):
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    provider.body["answers"] = {"team": {"type": "noul", "noul": 0.1}}
    larger.body = {**larger.body, "answers": {"rule_2": {"type": "noul", "noul": 0.9}}}
    trace = tmp_path / "decisions.jsonl"
    graph = (
        'ask team = "Is it urgent?"\n'
        "when team >= 0.5: urgent\n"
        'always: ask "Is it urgent?" using large'
    )
    with TypeSafeClient(base_url=larger.url, api_key="k", retry=RetryPolicy(max_retries=0)) as big:
        program = DecisionProgram(
            compile_rules(graph),
            client,
            "small",
            trace=trace,
            models=Models({"large": Judge(big, "big")}),
        )
        program(state="Server down!")
    (record,) = [json.loads(line) for line in trace.read_text().splitlines()]
    assert record["called"] == {"team": "the judge", "rule_2": "judge large"}


def test_a_rule_deciding_far_more_real_traffic_than_it_was_counted_on_is_flagged(tmp_path):
    graph = tmp_path / "graph.rules"
    graph.write_text(MEASURED)
    traces = tmp_path / "traces"
    traces.mkdir()
    # Eight in ten real requests are large, where the examples had one in ten.
    traffic(MEASURED, traces / "a.jsonl", [900] * 40 + [50] * 10)
    # Decisions by another graph are no evidence about this one.
    traffic("always: refund", traces / "b.jsonl", [50] * 30)
    result = invoke("drift", graph, traces)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["decisions"] == 50
    assert {moved["rule"] for moved in report["drift"]} == {"rule_1", "rule_2"}


def test_undecided_traffic_is_counted_apart_from_the_rules(tmp_path):
    graph = tmp_path / "graph.rules"
    partial = "read amount in [0, 10000]\nwhen amount >= 500: review calibrated 10/10 of 100\n"
    graph.write_text(partial)
    trace = tmp_path / "decisions.jsonl"
    traffic(partial, trace, [900] * 10 + [50] * 10)
    result = invoke("drift", graph, trace)
    report = json.loads(result.stdout)
    assert report["decisions"] == 20 and report["undecided"] == 10
    # Half the traffic was decided by no rule, which the rule's own share does not cover.
    assert [moved["rule"] for moved in report["drift"]] == ["rule_1"]


def test_traffic_like_the_examples_raises_nothing(tmp_path):
    graph = tmp_path / "graph.rules"
    graph.write_text(MEASURED)
    trace = tmp_path / "decisions.jsonl"
    traffic(MEASURED, trace, [900] * 5 + [50] * 45)
    result = invoke("drift", graph, trace)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["drift"] == []
