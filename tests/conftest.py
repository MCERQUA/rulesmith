"""Local HTTP protocol fixtures; these do not emulate Jev's intelligence."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from rulesmith.graph import Plan


def pytest_addoption(parser):
    parser.addoption(
        "--live", action="store_true", help="Allow paid hosted-model integration tests"
    )
    parser.addoption(
        "--ruling-url", help="Run local-model integration tests against this Ruling server"
    )
    parser.addoption("--ruling-reflection-model", help="DSPy model ID for real graph generation")
    parser.addoption("--ruling-reflection-url", help="Explicit local text-generation API base URL")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("live") is not None and not config.getoption("--live"):
            item.add_marker(pytest.mark.skip(reason="paid live check requires --live"))
        if item.get_closest_marker("ruling") is not None and not config.getoption("--ruling-url"):
            item.add_marker(pytest.mark.skip(reason="local model check requires --ruling-url"))


def chat_reply(**fields):
    """A completion in DSPy's ChatAdapter field format."""
    body = "".join(
        f"[[ ## {name} ## ]]\n{value if isinstance(value, str) else json.dumps(value)}\n\n"
        for name, value in fields.items()
    )
    return body + "[[ ## completed ## ]]"


@pytest.fixture
def mixed_plan():
    return Plan.model_validate(
        {
            "nodes": {
                "team": {
                    "type": "call",
                    "question": {
                        "type": "choice",
                        "instructions": "Which team should handle this request?",
                        "criteria": {"billing": "Charges and refunds", "technical": "Product bugs"},
                    },
                    "inputs": {"state": {"type": "state"}},
                },
                "urgency": {
                    "type": "call",
                    "question": {
                        "type": "score",
                        "instructions": "How urgent is this request?",
                        "criteria": ["Routine request", "Immediate assistance needed"],
                    },
                    "inputs": {"state": {"type": "state"}},
                },
                "refund": {
                    "type": "call",
                    "question": {"type": "noul", "instructions": "The customer requests a refund."},
                    "inputs": {"state": {"type": "state"}},
                },
                "decision": {
                    "type": "threshold",
                    "source": "refund",
                    "cutoff": 0.5,
                    "below": "review",
                    "above": "accept",
                },
            },
            "outputs": {
                "label": {"type": "node", "node": "decision"},
                "team": {"type": "node", "node": "team"},
                "urgency": {"type": "node", "node": "urgency"},
            },
        }
    )


@pytest.fixture
def provider():
    fixture = SimpleNamespace(
        requests=[],
        answers=None,
        on_request=None,
        on_chat=None,
        status=200,
        completion=None,
        # LM Studio, for one, rejects OpenAI's JSON mode.
        json_mode=True,
        body={
            "model": "fixture-model",
            "answers": {},
            "usage": {"input_tokens": 12, "output_tokens": 0},
        },
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fixture.requests.append((self.path, request))
            response = fixture.body
            if self.path == "/v1/systemone":
                if fixture.on_request is not None:
                    fixture.on_request(request)
                if fixture.answers is not None:
                    response = {
                        **fixture.body,
                        "answers": {
                            key: fixture.answers[key]
                            for key in request["questions"]
                            if key in fixture.answers
                        },
                    }
            if self.path == "/v1/chat/completions":
                if fixture.on_chat is not None:
                    fixture.on_chat(request)
                response = {
                    "id": "fixture-completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": fixture.completion},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            status = fixture.status
            if not fixture.json_mode and request.get("response_format", {}).get("type") == (
                "json_object"
            ):
                status = 400
                response = {"error": "'response_format.type' must be 'json_schema' or 'text'"}
            body = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    fixture.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield fixture
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def client(provider):
    with TypeSafeClient(
        base_url=provider.url, api_key="test-placeholder", retry=RetryPolicy(max_retries=0)
    ) as client:
        yield client
