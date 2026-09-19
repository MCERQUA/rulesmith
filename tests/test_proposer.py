import json

import dspy
import pytest
from conftest import chat_reply
from test_graph import binary_plan

from decision_compose.label import Labeler
from decision_compose.optimize import GraphProposal, GraphProposer, text_adapter


def test_signature_preserves_open_node_ids_and_wiring():
    graph = binary_plan()
    result = text_adapter().parse(GraphProposal, chat_reply(**graph))
    assert result == graph


@pytest.mark.parametrize(
    "completion",
    [
        "not a graph",
        '{"nodes": [], "outputs": {}}',
        '{"nodes": {"x": {"type": "value", "value": 1e309}}, "outputs": {}}',
    ],
)
def test_malformed_generation_is_reported_as_an_invalid_candidate(provider, completion):
    provider.completion = completion
    proposer = GraphProposer(
        dspy.LM(
            "openai/fixture-model",
            api_base=provider.url + "/v1",
            api_key="test-placeholder",
            cache=False,
            num_retries=0,
        )
    )
    candidate = json.loads(proposer("Propose a decision graph."))
    assert candidate["generation_error"]
    assert len(provider.requests) == 1


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
    provider.completion = chat_reply(**graph)
    assert json.loads(GraphProposer(lm)("Propose a decision graph.")) == graph
    provider.completion = chat_reply(label="review")
    assert Labeler(lm, "Classify requests.", ["accept", "review"])("request A") == "review"
