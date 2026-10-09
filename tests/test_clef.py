"""Serving Clef from its own release code: the HTTP layer, with a stand-in for the loaded model,
and, given --clef-url, a real graph decided by the real model."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from conftest import post
from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

from rulesmith.clef import PATH, make_server
from rulesmith.graph import Task

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {"outage": {"type": "noul", "instructions": "Is a service down?"}}


def echo(request):
    """What the release's systemone returns, shaped from the request it was given."""
    if request["questions"].get("outage", {}).get("type") != "noul":
        raise ValueError("outage: type must be noul, choice, or score")
    answers = {key: {"type": "noul", "noul": 0.9} for key in request["questions"]}
    return {"model": request["model"], "answers": answers, "usage": {"input_tokens": 7}}


@pytest.fixture
def serving():
    """A server around `answer`, which a test may replace before asking."""
    held = {"answer": echo}
    server = make_server(lambda request: held["answer"](request), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", server, held
    server.shutdown()
    server.server_close()
    thread.join()


def test_a_systemone_request_is_answered_by_the_model(serving):
    url, _, _ = serving
    status, body = post(
        f"{url}{PATH}", {"model": "clef-flash", "state": "x", "questions": QUESTIONS}
    )
    assert status == 200 and body["answers"]["outage"]["noul"] == 0.9
    status, body = post(f"{url}/v1/chat/completions", {})
    assert status == 404 and "error" in body


def test_the_sdk_reads_a_refusal_as_its_reason(serving):
    # The SDK is what rulesmith asks through; a bad question must reach it as the model's reason.
    url, _, _ = serving
    questions = {"outage": {"type": "score", "criteria": ["a"]}}
    with TypeSafeClient(base_url=url, api_key="x", retry=RetryPolicy(max_retries=0)) as client:
        with pytest.raises(TypeSafeError, match="type must be noul"):
            client.system_one(state="x", questions=questions, model="clef-flash")


def test_a_body_that_is_not_a_request_is_refused_without_reaching_the_model(serving):
    url, _, held = serving
    held["answer"] = lambda request: pytest.fail("the model was asked")
    for body in ([1, 2], "state"):
        status, refused = post(f"{url}{PATH}", body)
        assert status == 400 and refused["error"]


def test_a_model_failure_is_a_server_error_and_the_server_keeps_serving(serving):
    url, _, held = serving

    def out_of_memory(request):
        raise RuntimeError("MPS backend out of memory")

    held["answer"] = out_of_memory
    status, body = post(f"{url}{PATH}", {"model": "clef-flash", "state": "x", "questions": {}})
    assert status == 500 and "out of memory" in body["error"]
    held["answer"] = echo
    status, _ = post(f"{url}{PATH}", {"model": "clef-flash", "state": "x", "questions": QUESTIONS})
    assert status == 200


def test_the_model_is_asked_one_request_at_a_time(serving):
    url, server, held = serving
    # The first request holds the model until a second has arrived: either at the lock, which
    # the watched lock reports, or inside the model, which is the failure this test catches.
    second_arrived = threading.Event()
    inside, overlapped = [], []

    class Watched:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                second_arrived.set()
                self.lock.acquire()

        def __exit__(self, *exception):
            self.lock.release()

    def held_model(request):
        inside.append(request["state"])
        if len(inside) > 1:
            overlapped.append(True)
            second_arrived.set()
        else:
            second_arrived.wait()
        result = echo(request)
        inside.remove(request["state"])
        return result

    server.lock = Watched()
    held["answer"] = held_model
    statuses = []

    def ask(state):
        body = {"model": "clef-flash", "state": state, "questions": QUESTIONS}
        statuses.append(post(f"{url}{PATH}", body)[0])

    askers = [threading.Thread(target=ask, args=(state,)) for state in ("a", "b")]
    for asker in askers:
        asker.start()
    for asker in askers:
        asker.join()
    assert statuses == [200, 200] and not overlapped


STAND_IN = """
from rulesmith import clef
from rulesmith.cli import main

def load(weights, device):
    # transformers 5.10.2 prints docstring warnings to stdout as it imports Qwen3.5.
    print("[ERROR] `loss` is part of Qwen3_5CausalLMOutputWithPast.__init__'s signature")
    return (lambda request: {"model": weights, "answers": {}, "usage": {}}), "stand-in"

clef.load = load
main()
"""


def test_the_command_line_says_where_it_listens_once_the_model_is_loaded():
    server = subprocess.Popen(
        [sys.executable, "-c", STAND_IN, "clef-serve", "--port", "0", "--weights", "w"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # The first line names where it listens, so a caller never guesses when it is ready.
        ready = json.loads(server.stdout.readline())
        assert ready["device"] == "stand-in"
        status, body = post(f"{ready['url']}{PATH}", {"model": "m", "state": "x"})
        assert status == 200 and body["model"] == "w"
    finally:
        server.terminate()
        server.wait(timeout=10)


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="torch is installed")
def test_without_the_clef_extra_the_command_says_what_to_install():
    result = subprocess.run(
        [sys.executable, "-m", "rulesmith.cli", "clef-serve", "--port", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "rulesmith[clef]" in result.stderr


@pytest.mark.clef
def test_a_real_graph_is_decided_by_the_real_model(request, tmp_path):
    task = Task.model_validate_json((ROOT / "examples/support-routing/task.json").read_text())
    environment = dict(os.environ)
    environment.pop("TYPESAFE_API_KEY", None)
    # Prevent DSPy's transitive dotenv support from restoring a hosted credential.
    environment["PYTHON_DOTENV_DISABLED"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rulesmith.cli",
            "run",
            str(ROOT / "examples/support-routing/graph.rules"),
            str(ROOT / "examples/support-routing/input.json"),
            "--backend",
            "clef",
            "--base-url",
            request.config.getoption("--clef-url"),
            "--max-workers",
            "1",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    decided = json.loads(result.stdout)
    assert decided["label"] in task.labels and decided["models"] == ["clef-flash"]
    probabilities = decided["answers"]["team"]["probabilities"]
    assert set(probabilities) == {"billing", "technical", "sales"}
    assert sum(probabilities.values()) == pytest.approx(1, abs=1e-3)
    assert decided["input_tokens"] > 0
