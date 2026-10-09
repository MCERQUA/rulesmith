import json
import math
import string

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_cli import invoke, write
from test_judges import ESCALATING, team
from typesafe_sdk import RetryPolicy

from rulesmith.chat_judge import (
    ABSENT_MARGIN,
    TOP_LOGPROBS,
    ChatClient,
    ChatJudgeError,
    letter_probabilities,
)
from rulesmith.cli import BACKENDS
from rulesmith.runtime import DecisionProgram


def top(**logprobs):
    """The first token's log-probabilities, as OpenAI returns them with logprobs on."""
    tokens = [{"token": token, "logprob": value} for token, value in logprobs.items()]
    return {"content": [{"token": tokens[0]["token"], "logprob": 0, "top_logprobs": tokens}]}


def answering(provider, by_instructions):
    """Reply to each question with the log-probabilities given for its instructions."""

    def reply(request):
        prompt = request["messages"][-1]["content"]
        provider.logprobs = next(
            value for instructions, value in by_instructions.items() if instructions in prompt
        )

    provider.on_chat = reply


@pytest.fixture
def chat(provider):
    with ChatClient(
        base_url=f"{provider.url}/v1",
        api_key="test-placeholder",
        timeout=5,
        retry=RetryPolicy(max_retries=0),
    ) as client:
        yield client


logprob = st.floats(min_value=-30, max_value=0)


@given(st.lists(logprob, min_size=2, max_size=20))
def test_probabilities_are_the_softmax_of_the_option_letters(values):
    tokens = [
        {"token": letter, "logprob": v}
        for letter, v in zip(string.ascii_uppercase[: len(values)], values, strict=True)
    ]
    probabilities = letter_probabilities(tokens, len(values))
    total = sum(math.exp(v) for v in values)
    assert probabilities == pytest.approx([math.exp(v) / total for v in values])


@given(st.lists(logprob, min_size=1, max_size=5), st.integers(min_value=1, max_value=5))
def test_a_letter_missing_from_the_top_list_ranks_below_every_listed_one(values, missing):
    count = len(values) + missing
    tokens = [
        {"token": letter, "logprob": v}
        for letter, v in zip(string.ascii_uppercase[: len(values)], values, strict=True)
    ]
    probabilities = letter_probabilities(tokens, count)
    assert sum(probabilities) == pytest.approx(1)
    listed, absent = probabilities[: len(values)], probabilities[len(values) :]
    assert max(absent) < min(listed)
    assert absent[0] / min(listed) == pytest.approx(math.exp(-ABSENT_MARGIN))


def test_spellings_of_one_letter_pool_their_probability():
    # Tokenizers split " A" from "A", and mlx_lm.server spells the space as byte-level BPE's
    # "Ġ" or SentencePiece's "▁"; each is the model answering A.
    tokens = [
        {"token": "A", "logprob": math.log(0.2)},
        {"token": " A", "logprob": math.log(0.1)},
        {"token": "ĠA", "logprob": math.log(0.1)},
        {"token": "▁B", "logprob": math.log(0.2)},
        {"token": "Sure", "logprob": math.log(0.2)},
    ]
    assert letter_probabilities(tokens, 2) == pytest.approx([2 / 3, 1 / 3])


def test_no_option_letter_among_the_top_tokens_is_an_error():
    with pytest.raises(ChatJudgeError, match="none of the option letters"):
        letter_probabilities([{"token": "Sure", "logprob": -0.1}], 2)


def test_each_question_is_one_deterministic_single_token_request(provider, chat):
    provider.logprobs = top(A=math.log(0.9), B=math.log(0.1))
    state = {"ticket": "Refund me", "amount": 40}
    questions = {
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "Charges", "technical": "Bugs"},
        },
        "refund": {"type": "noul", "instructions": "A refund is requested."},
    }
    chat.system_one(state=state, questions=questions, model="judge-model")
    assert [path for path, _ in provider.requests] == ["/v1/chat/completions"] * 2
    for _, request in provider.requests:
        assert request["model"] == "judge-model"
        assert request["max_tokens"] == 1
        assert request["temperature"] == 0
        assert request["logprobs"] is True and request["top_logprobs"] == TOP_LOGPROBS
        prompt = request["messages"][-1]["content"]
        assert json.dumps(state, indent=2) in prompt
    assert all(h["Authorization"] == "Bearer test-placeholder" for h in provider.headers)
    choice = provider.requests[0][1]["messages"][-1]["content"]
    assert "A. billing: Charges" in choice and "B. technical: Bugs" in choice


def test_a_string_state_is_shown_as_written(provider, chat):
    provider.logprobs = top(A=-0.1, B=-2.5)
    chat.system_one(
        state="Refund me", questions={"q": {"type": "noul", "instructions": "Refund?"}}, model="m"
    )
    prompt = provider.requests[0][1]["messages"][-1]["content"]
    assert "Refund me" in prompt and '"Refund me"' not in prompt


def test_a_graph_runs_on_chat_answers(provider, chat, mixed_plan):
    answering(
        provider,
        {
            "Which team": top(A=math.log(0.2), B=math.log(0.8)),
            "How urgent": top(A=math.log(0.25), B=math.log(0.75)),
            "refund": top(A=math.log(0.7), B=math.log(0.3)),
        },
    )
    result = DecisionProgram(mixed_plan, chat, "judge-model")(state={"ticket": "Refund me"})
    assert result.answers["team"]["value"] == "technical"
    assert result.answers["team"]["probabilities"] == pytest.approx(
        {"billing": 0.2, "technical": 0.8}
    )
    # TypeSafe's choice confidence: how far the leader stands above uniform.
    assert result.answers["team"]["confidence"] == pytest.approx((0.8 - 0.5) / 0.5)
    assert result.answers["urgency"]["value"] == pytest.approx(0.75)
    assert result.answers["refund"]["value"] == pytest.approx(0.7)
    assert result.label == "accept"
    # Each call is its own request, and the fixture bills 10 prompt and 10 completion tokens.
    assert result.input_tokens == 30 and result.output_tokens == 30
    assert result.models == ["judge-model"]


def test_a_model_without_log_probabilities_is_named_in_the_error(provider, chat):
    provider.completion = "A"
    with pytest.raises(ChatJudgeError, match="reasoning-model.*log-probabilities"):
        chat.system_one(
            state="x",
            questions={"q": {"type": "noul", "instructions": "?"}},
            model="reasoning-model",
        )


def test_a_choice_with_more_options_than_letters_is_refused(chat):
    criteria = {f"option{i}": "" for i in range(27)}
    with pytest.raises(ChatJudgeError, match="26"):
        chat.system_one(
            state="x",
            questions={"q": {"type": "choice", "instructions": "?", "criteria": criteria}},
            model="m",
        )


@pytest.mark.parametrize("retries,succeeds", [(0, False), (1, True)])
def test_retries_ride_out_a_server_error(provider, retries, succeeds):
    provider.logprobs = top(A=-0.1, B=-2.5)

    def fail_once(request):
        provider.status = 503 if len(provider.requests) == 1 else 200

    provider.on_chat = fail_once
    policy = RetryPolicy(max_retries=retries, backoff_initial=0)
    question = {"q": {"type": "noul", "instructions": "?"}}
    with ChatClient(base_url=f"{provider.url}/v1", api_key="k", timeout=5, retry=policy) as client:
        if succeeds:
            assert client.system_one(state="x", questions=question, model="m").answers["q"].noul
        else:
            with pytest.raises(ChatJudgeError, match="503"):
                client.system_one(state="x", questions=question, model="m")


def test_the_openai_backend_runs_a_graph_from_its_environment(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", f"{provider.url}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    provider.logprobs = top(A=math.log(0.8), B=math.log(0.2))
    rules = tmp_path / "graph.rules"
    rules.write_text('ask urgent = "Is it urgent?"\nwhen urgent >= 0.5: page\nalways: queue\n')
    state = write(tmp_path / "state.json", {"message": "Server down"})
    trace = tmp_path / "trace.jsonl"
    result = invoke("run", rules, state, "--backend", "openai", "--trace", trace)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["label"] == "page"
    assert provider.requests[0][1]["model"] == BACKENDS["openai"].model
    assert provider.headers[0]["Authorization"] == "Bearer test-placeholder"
    assert json.loads(trace.read_text())["model"] == BACKENDS["openai"].model


def test_a_named_judge_can_be_an_openai_compatible_api(tmp_path, provider, larger):
    provider.body["answers"] = {"team": team("billing", 0.55)}
    larger.logprobs = top(A=math.log(0.1), B=math.log(0.9))
    rules = tmp_path / "graph.rules"
    rules.write_text(ESCALATING + "\n")
    state = write(tmp_path / "state.json", "It crashes, or maybe I was charged?")
    judge = f"large=openai:large-model@{larger.url}/v1"
    result = invoke(
        "run", rules, state, "--backend", "ruling", "--base-url", provider.url, "--judge", judge
    )
    assert result.returncode == 0, result.stderr
    decided = json.loads(result.stdout)
    assert decided["label"] == "technical"
    assert larger.requests[0][1]["model"] == "large-model"
    escalated = [call for call in decided["calls"].values() if call.get("judge") == "large"]
    assert [call["model"] for call in escalated] == ["large-model"]


def test_an_unreachable_chat_judge_is_reported_in_a_line(tmp_path):
    rules = tmp_path / "graph.rules"
    rules.write_text('always: ask "Is it urgent?"\n')
    state = write(tmp_path / "state.json", "Server down")
    url = "http://127.0.0.1:1/v1"
    result = invoke("run", rules, state, "--backend", "openai", "--base-url", url)
    assert result.returncode == 2
    assert "unreachable" in result.stderr and "Traceback" not in result.stderr
