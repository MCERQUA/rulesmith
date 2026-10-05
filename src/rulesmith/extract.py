"""Reading free text into typed fields with a generative model, for rules to test."""

import json
import re

import dspy
from dspy.utils.exceptions import AdapterParseError

from rulesmith.graph import Read, State
from rulesmith.optimize import text_adapter

# A field the model wrote as anything but a plain number is not one, and is left absent.
NUMBER = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?")


def described(field: Read) -> str:
    if field.categories is not None:
        return f"One of: {', '.join(field.categories)}. None if the record does not say."
    low, high = field.bounds
    return f"A number from {low:g} to {high:g}. None if the record does not say."


class Extractor:
    """A model asked to fill a few fields from a record, each answer kept as text: whether it is
    a number in bounds or one of the categories is the read's to decide, as for any input. Each
    record is read once per set of fields, since search runs a graph on the same examples again
    and again."""

    def __init__(self, lm: dspy.LM):
        self.lm = lm
        self.found: dict[str, tuple[dict, dict]] = {}

    def __call__(
        self, instructions: str, fields: dict[str, Read], record: State
    ) -> tuple[dict, dict]:
        key = json.dumps(
            [instructions, {name: described(field) for name, field in fields.items()}, record],
            sort_keys=True,
        )
        if key not in self.found:
            self.found[key] = self.extract(instructions, fields, record)
        return self.found[key]

    def extract(
        self, instructions: str, fields: dict[str, Read], record: State
    ) -> tuple[dict, dict]:
        signature = dspy.make_signature(
            {"record": (State, dspy.InputField())}
            | {
                name: (str, dspy.OutputField(desc=described(field)))
                for name, field in fields.items()
            },
            instructions,
        )
        usage = {
            "model": self.lm.model,
            "input_tokens": 0,
            "output_tokens": 0,
            "extracted": list(fields),
        }
        with dspy.context(lm=self.lm, adapter=text_adapter(), track_usage=True):
            try:
                predicted = dspy.Predict(signature)(record=record)
            except AdapterParseError as error:
                # A reply with no fields in it fills none: the rules see every field absent.
                return {}, usage | {"error": str(error)}
        for counted in predicted.get_lm_usage().values():
            usage["input_tokens"] += counted.get("prompt_tokens") or 0
            usage["output_tokens"] += counted.get("completion_tokens") or 0
        values = {}
        for name, field in fields.items():
            text = str(getattr(predicted, name)).strip()
            if field.categories is not None:
                values[name] = text
            elif NUMBER.fullmatch(text):
                values[name] = float(text)
        return values, usage
