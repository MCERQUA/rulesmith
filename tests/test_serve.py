"""Serving decisions over HTTP, so a workflow in any language can ask one and get rulesmith's
answer, with the rule that decided and a trace of it."""

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from rulesmith.rules import compile_rules
from rulesmith.runtime import DecisionProgram
from rulesmith.serve import make_server

TRIAGE = "read order.total in [0, 1000000]\nwhen order.total >= 500: hold\nalways: ship\n"
PARTIAL = "read order.total in [0, 1000000]\nwhen order.total >= 500: hold\n"


def post(url: str, body) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.fixture
def served(tmp_path):
    programs = {
        "triage": DecisionProgram(compile_rules(TRIAGE), None, "none", trace=tmp_path / "t.jsonl"),
        "partial": DecisionProgram(compile_rules(PARTIAL), None, "none"),
    }
    server = make_server(programs, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", tmp_path / "t.jsonl"
    server.shutdown()
    server.server_close()
    thread.join()


def test_a_decision_answers_what_run_would_and_is_traced(served):
    url, trace = served
    status, decided = post(f"{url}/decisions/triage", {"order": {"total": 900}})
    assert status == 200
    assert decided["label"] == "hold" and decided["decided_by"] == "rule_1"
    (record,) = [json.loads(line) for line in trace.read_text().splitlines()]
    assert record["label"] == "hold"


def test_the_decisions_served_are_listed(served):
    url, _ = served
    with urllib.request.urlopen(f"{url}/decisions") as response:
        assert sorted(json.loads(response.read())["decisions"]) == ["partial", "triage"]


def test_a_request_the_server_cannot_answer_says_why(served):
    url, _ = served
    status, body = post(f"{url}/decisions/payroll", {})
    assert status == 404 and "payroll" in body["error"]
    request = urllib.request.Request(f"{url}/decisions/triage", data=b"{not json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request)
    assert refused.value.code == 400
    # A graph that could not answer this input is the caller's to handle, with what it did decide.
    status, body = post(f"{url}/decisions/partial", {"order": {"total": 10}})
    assert status == 422
    assert "label" in body["error"] and body["result"]["skipped"]


def test_the_command_line_serves_decision_folders(tmp_path):
    folder = tmp_path / "triage"
    folder.mkdir()
    (folder / "graph.rules").write_text(TRIAGE)
    trace = tmp_path / "decisions.jsonl"
    server = subprocess.Popen(
        [sys.executable, "-m", "rulesmith.cli", "serve", folder, "--port", "0", "--trace", trace],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # The first line names where it listens, so a caller never guesses when it is ready.
        ready = server.stdout.readline()
        url = json.loads(ready)["url"]
        status, decided = post(f"{url}/decisions/triage", {"order": {"total": 20}})
        assert status == 200 and decided["label"] == "ship"
        assert json.loads(trace.read_text())["label"] == "ship"
    finally:
        server.terminate()
        server.wait(timeout=10)
