"""Grade a graph's answers with a stronger model, where no label says what is right."""

import json
import math

import dspy

from rulesmith.graph import Example, State
from rulesmith.optimize import Outcome, text_adapter
from rulesmith.runtime import DecisionProgram, GraphExecutionError


def model_name(model: str) -> str:
    """A model without its provider or publisher prefix, which two services write differently."""
    return model.rsplit("/", 1)[-1].lower()


class Grader(dspy.Module):
    """A model's grade, from 0 to 1, for what a graph answered, with the reason for it. Each
    answer to each input is graded once: search scores the same answers over and over."""

    def __init__(self, lm: dspy.LM, description: str):
        super().__init__()
        self.lm = lm
        signature = dspy.make_signature(
            {
                "state": (State, dspy.InputField()),
                "outputs": (dict, dspy.InputField()),
                "reference": (
                    str | None,
                    dspy.InputField(desc="The label a person judged right, when there is one"),
                ),
                "score": (float, dspy.OutputField(desc="0 is wrong, 1 is right")),
                "reason": (str, dspy.OutputField(desc="What makes the outputs right or wrong")),
            },
            f"Grade the outputs a decision program gave for this input. The task: {description}",
        )
        self.predict = dspy.Predict(signature)
        self.grades: dict[str, tuple[float, str]] = {}

    def forward(self, state: State, outputs: dict, reference: str | None) -> tuple[float, str]:
        key = json.dumps([state, outputs, reference], sort_keys=True)
        if key not in self.grades:
            with dspy.context(lm=self.lm, adapter=text_adapter()):
                graded = self.predict(state=state, outputs=outputs, reference=reference)
            if not (math.isfinite(graded.score) and 0 <= graded.score <= 1):
                raise ValueError(f"the grader scored {graded.score}, outside 0 to 1")
            self.grades[key] = (graded.score, graded.reason)
        return self.grades[key]


class Graded:
    """Scores an answer by a grader's judgment. A graph that fails to answer scores zero
    without being graded."""

    def __init__(self, grader: Grader):
        self.grader = grader

    def __call__(self, program: DecisionProgram, example: Example) -> Outcome:
        try:
            prediction, error = program(state=example.state), None
        except GraphExecutionError as failure:
            prediction, error = failure.prediction, f"Graph execution failed: {failure}"
        record = {"id": example.id, "expected": example.label, **dict(prediction)}
        trace = {"Inputs": example.state, "Generated Outputs": dict(prediction)}
        if error:
            return Outcome(0.0, record, trace | {"Feedback": error}, prediction.models, error)
        # A graph that asks the grader's own model can learn to ask it the grader's question,
        # which scores perfectly and means nothing.
        grading = model_name(self.grader.lm.model)
        if grading in {model_name(model) for model in prediction.models}:
            raise ValueError(
                f"the grader {self.grader.lm.model} is the model the graph asks; "
                "grade with a different, stronger model"
            )
        score, reason = self.grader(
            state=example.state, outputs=prediction.outputs, reference=example.label
        )
        return Outcome(
            score=score,
            record=record | {"grade": score, "reason": reason},
            trace=trace | {"Feedback": f"A grader scored these outputs {score:g} of 1: {reason}"},
            models=prediction.models,
        )
