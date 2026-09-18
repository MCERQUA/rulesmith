"""Execute a decision tree as a DSPy module using TypeSafe's official client."""

import math

import dspy
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer, TypeSafeClient

from jev_compose.graph import Choice, Noul, Plan, Question, Score, State


class ResponseError(ValueError):
    """The provider returned an answer that violates the requested contract."""


def in_range(value: float, upper: float) -> bool:
    return math.isfinite(value) and 0 <= value <= upper


def read_answer(question: Question, answer: NoulAnswer | ChoiceAnswer | ScoreAnswer) -> dict:
    expected = {Noul: NoulAnswer, Choice: ChoiceAnswer, Score: ScoreAnswer}
    if not isinstance(answer, expected[type(question)]):
        raise ResponseError("answer type does not match question type")
    if isinstance(answer, NoulAnswer):
        if not in_range(answer.noul, 1):
            raise ResponseError("Noul probability must be finite and between 0 and 1")
        return {"type": "noul", "value": answer.noul}

    keys = (
        set(question.criteria)
        if isinstance(question, Choice)
        else set(range(len(question.criteria)))
    )
    probabilities = answer.probabilities
    if set(probabilities) != keys or not all(in_range(p, 1) for p in probabilities.values()):
        raise ResponseError(
            "probabilities must cover exactly the declared options and lie in [0, 1]"
        )
    # The hosted service may round individual probabilities in its response.
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.02):
        raise ResponseError("probabilities do not sum to 1 within rounding tolerance")
    if not in_range(answer.confidence, 1):
        raise ResponseError("confidence must be finite and between 0 and 1")

    if isinstance(answer, ChoiceAnswer):
        if answer.choice not in keys:
            raise ResponseError("selected option is outside the declared options")
        value = answer.choice
    else:
        if not in_range(answer.score, len(question.criteria) - 1):
            raise ResponseError("score is outside the declared rubric")
        if answer.legend != dict(enumerate(question.criteria)):
            raise ResponseError("returned rubric differs from the requested rubric")
        value = answer.score
    return {
        "type": question.type,
        "value": value,
        "probabilities": probabilities,
        "confidence": answer.confidence,
    }


class JevProgram(dspy.Module):
    def __init__(self, plan: Plan, client: TypeSafeClient, model: str):
        super().__init__()
        self.plan = plan
        self.client = client
        self.model = model

    def forward(self, state: State) -> dspy.Prediction:
        answers = {}
        model, input_tokens, output_tokens = None, 0, 0
        if self.plan.questions:
            response = self.client.system_one(
                state=state,
                questions={
                    key: question.model_dump() for key, question in self.plan.questions.items()
                },
                model=self.model,
            )
            if set(response.answers) != set(self.plan.questions):
                raise ResponseError("answer IDs do not match the requested question IDs")
            answers = {
                key: read_answer(question, response.answers[key])
                for key, question in self.plan.questions.items()
            }
            model = response.model
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        label, path = self.plan.decide({key: answer["value"] for key, answer in answers.items()})
        return dspy.Prediction(
            label=label,
            path=path,
            answers=answers,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
