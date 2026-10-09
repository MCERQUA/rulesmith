"""A judge reached through any OpenAI-compatible chat API, read from its log-probabilities.

Each question is asked on its own, with its options lettered A, B, C and so on, for a single
token. The probability of each option is the softmax of its letter's log-probability among the
first token's most likely candidates, so an answer carries the same distribution a TypeSafe
judge returns, and the runtime reads it the same way.
"""

import json
import math
import string
import time
from types import TracebackType
from typing import Self

import httpx
from typesafe_sdk import (
    ChoiceAnswer,
    NoulAnswer,
    RetryPolicy,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

LETTERS = string.ascii_uppercase
# How many first-token candidates to ask for. OpenAI and vLLM allow 20, but mlx_lm.server
# refuses more than 11 by dropping the connection; ten covers every score rubric and most
# choices, and a letter beyond them is bounded by ABSENT_MARGIN.
TOP_LOGPROBS = 10
# A letter missing from the top candidates is less likely than every listed one by an unknown
# amount. It is taken to sit this many nats below the least likely listed token: low enough to
# rank under all of them, while still keeping a little probability, as the model would.
ABSENT_MARGIN = 1.0
SYSTEM = "You are a careful judge. Answer with exactly one letter, and nothing else."


class ChatJudgeError(Exception):
    """The chat API failed, or answered in a form a judgment cannot be read from."""


def letter_probabilities(candidates: list[dict], count: int) -> list[float]:
    """The probabilities of the first `count` letters, from the first token's top candidates."""
    letters = LETTERS[:count]
    found: dict[str, list[float]] = {}
    for candidate in candidates:
        # Some servers return raw tokenizer pieces, with the leading space spelled as byte-level
        # BPE's "Ġ" or SentencePiece's "▁".
        letter = candidate["token"].strip().lstrip("Ġ▁")
        if letter in letters:
            found.setdefault(letter, []).append(candidate["logprob"])
    if not found:
        likeliest = max(candidates, key=lambda candidate: candidate["logprob"])["token"]
        raise ChatJudgeError(
            "none of the option letters is among the model's most likely first tokens; it "
            f"began with {likeliest!r}, as a model that thinks before answering does"
        )
    floor = min(candidate["logprob"] for candidate in candidates) - ABSENT_MARGIN
    logits = [logsumexp(found[letter]) if letter in found else floor for letter in letters]
    peak = max(logits)
    weights = [math.exp(logit - peak) for logit in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


# Confidence follows the formulas TypeSafe publishes in its MIT-licensed system-one-adapter (as
# Ruling's calibration.py reimplements them), so it reads the same as a hosted judge's.
def choice_confidence(probabilities: list[float]) -> float:
    """How far the leading option stands above a uniform distribution, from 0 to 1."""
    uniform = 1 / len(probabilities)
    return (max(probabilities) - uniform) / (1 - uniform)


def score_confidence(probabilities: list[float]) -> float:
    """How tightly a score's mass sits around its modal level, from 0 to 1."""
    mode = probabilities.index(max(probabilities))
    spread = sum(p * abs(level - mode) for level, p in enumerate(probabilities))
    middle = (len(probabilities) - 1) / 2
    uniform_spread = sum(abs(level - middle) for level in range(len(probabilities)))
    return max(0.0, 1 - spread * len(probabilities) / uniform_spread)


def options(question: dict) -> list[str]:
    """The options a question offers, in the order they are lettered."""
    if question["type"] == "noul":
        return ["True", "False"]
    if question["type"] == "choice":
        return [f"{label}: {meaning}" for label, meaning in question["criteria"].items()]
    return list(question["criteria"])


def prompt(state, question: dict) -> str:
    shown = state if isinstance(state, str) else json.dumps(state, indent=2)
    lettered = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(options(question)))
    heading = {
        "noul": "Is the statement true of the input?",
        "choice": "Options:",
        "score": "Levels, from lowest to highest:",
    }[question["type"]]
    return (
        f"Input:\n{shown}\n\n{question['instructions']}\n\n{heading}\n{lettered}\n\n"
        "Answer with exactly one letter."
    )


def answer(question: dict, probabilities: list[float]):
    if question["type"] == "noul":
        return NoulAnswer(noul=probabilities[0])
    if question["type"] == "choice":
        labels = list(question["criteria"])
        return ChoiceAnswer(
            choice=labels[probabilities.index(max(probabilities))],
            confidence=choice_confidence(probabilities),
            probabilities=dict(zip(labels, probabilities, strict=True)),
        )
    return ScoreAnswer(
        score=sum(level * p for level, p in enumerate(probabilities)),
        confidence=score_confidence(probabilities),
        legend=dict(enumerate(question["criteria"])),
        probabilities=dict(enumerate(probabilities)),
    )


class ChatClient:
    """Answers TypeSafe's System One questions through an OpenAI-compatible chat API."""

    def __init__(self, *, base_url: str, api_key: str | None, timeout: float, retry: RetryPolicy):
        # A local server needs no key, and an absent one is better than a made-up one.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        self.retry = retry

    def system_one(self, *, state, questions: dict[str, dict], model: str) -> SystemOneResponse:
        answers, served, input_tokens, output_tokens = {}, model, 0, 0
        for key, question in questions.items():
            count = len(options(question))
            if count > len(LETTERS):
                raise ChatJudgeError(
                    f"question {key!r} has {count} options, and the chat judge letters at most "
                    f"{len(LETTERS)}"
                )
            body = self.complete(model, prompt(state, question))
            choice = body["choices"][0]
            content = (choice.get("logprobs") or {}).get("content")
            if not content or not content[0].get("top_logprobs"):
                raise ChatJudgeError(
                    f"{model} returned no log-probabilities; the openai backend needs a model "
                    "that returns them, which reasoning models and some providers do not"
                )
            answers[key] = answer(question, letter_probabilities(content[0]["top_logprobs"], count))
            served = body.get("model", served)
            usage = body.get("usage") or {}
            input_tokens += usage.get("prompt_tokens", 0)
            output_tokens += usage.get("completion_tokens", 0)
        return SystemOneResponse(
            model=served,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            answers=answers,
        )

    def complete(self, model: str, text: str) -> dict:
        request = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
        }
        attempt = 0
        while True:
            last = attempt == self.retry.max_retries
            try:
                response = self.http.post("/chat/completions", json=request)
            except httpx.TransportError as error:
                if last:
                    raise ChatJudgeError(f"{self.http.base_url} is unreachable: {error}") from error
            else:
                if response.is_success:
                    return response.json()
                if last or response.status_code not in self.retry.http_statuses:
                    raise ChatJudgeError(
                        f"{self.http.base_url} answered {response.status_code}: {response.text}"
                    )
            time.sleep(min(self.retry.backoff_initial * 2**attempt, self.retry.backoff_max))
            attempt += 1

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
