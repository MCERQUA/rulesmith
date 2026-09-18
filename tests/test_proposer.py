import json

import dspy
import pytest
from test_graph import binary_plan

from decision_compose.optimize import GraphProposal, GraphProposer


def test_signature_preserves_open_node_ids_and_wiring():
    graph = binary_plan()
    result = dspy.JSONAdapter().parse(GraphProposal, json.dumps(graph))
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
    assert provider.requests[0][1]["response_format"] == {"type": "json_object"}
