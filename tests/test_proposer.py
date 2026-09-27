import json

import dspy
from conftest import chat_reply
from test_graph import binary_plan

from rulesmith.label import Labeler
from rulesmith.optimize import GraphProposal, GraphProposer, text_adapter


def proposer(provider):
    return GraphProposer(
        dspy.LM(
            "openai/fixture-model",
            api_base=provider.url + "/v1",
            api_key="test-placeholder",
            cache=False,
            num_retries=0,
        )
    )


def test_a_part_comes_back_as_the_text_the_model_wrote():
    rules = "read target.x in [-1, 1]\nwhen target.x >= 0.1: TURN_RIGHT\nalways: MOVE_FORWARD"
    assert text_adapter().parse(GraphProposal, chat_reply(graph=rules)) == {"graph": rules}


def test_whatever_the_model_answers_becomes_the_candidate(provider):
    # Rules the compiler cannot read are reported by it, with the line, not hidden here.
    provider.completion = "when nonsense: MOVE_FORWARD"
    assert proposer(provider)("Propose rules.") == "when nonsense: MOVE_FORWARD"
    assert len(provider.requests) == 1


def test_a_fenced_answer_loses_its_fence(provider):
    provider.completion = "```\nalways: review\n```"
    assert proposer(provider)("Propose rules.") == "always: review"


def test_a_graph_wrapped_in_prose_is_still_read(provider):
    # Small models explain themselves before answering; the graph is still in there.
    graph = binary_plan()
    provider.completion = f"Based on the error I fixed the output.\n\n{json.dumps(graph)}\n\nDone."
    assert json.loads(proposer(provider)("Propose a decision graph.")) == graph


def test_proposer_and_labeler_work_on_servers_without_json_mode(provider):
    provider.json_mode = False
    lm = dspy.LM(
        "openai/fixture-model",
        api_base=provider.url + "/v1",
        api_key="test-placeholder",
        cache=False,
        num_retries=0,
    )
    graph = binary_plan()
    provider.completion = chat_reply(graph=graph)
    assert json.loads(GraphProposer(lm)("Propose a decision graph.")) == graph
    provider.completion = chat_reply(label="review")
    assert Labeler(lm, "Classify requests.", ["accept", "review"])("request A") == "review"
