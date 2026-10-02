# Use from Python

A graph runs as a `dspy.Module`:

```python
from pathlib import Path

from typesafe_sdk import TypeSafeClient

from rulesmith.graph import Plan
from rulesmith.runtime import DecisionProgram

plan = Plan.model_validate_json(Path("runs/my-task/plan.json").read_text())
with TypeSafeClient(base_url="http://127.0.0.1:8010", api_key="local-no-auth") as client:
    program = DecisionProgram(plan, client, model="default")
    result = program(state="Please correct the duplicate charge on my invoice.")
    print(result.label, result.answers)
```

Output (probabilities rounded):

```text
billing {'intent': {'type': 'choice', 'value': 'billing', 'probabilities': {'billing': 0.992, 'technical': 0.008, 'sales': 4.5e-05}, 'confidence': 0.989}}
```

Against Jev, the client reads `TYPESAFE_API_KEY` and TypeSafe's endpoint by
itself: `with TypeSafeClient() as client:` and `model="jev-latest"`.

`result.answers` holds what each judgment said, so a decision can be logged or
audited, not just acted on. Rules load the same way, and a graph of reads alone
needs no judge at all:

```python
from rulesmith.rules import compile_rules, render_rules

plan = compile_rules(Path("examples/order-review/graph.rules").read_text())
print(render_rules(plan))  # and back again: the graph reads as the rules it came from

triage = compile_rules("""
read order.total in [0, 1000000]
when order.total >= 500: hold_for_review
always: ship
""")
print(DecisionProgram(triage, None, "none")(state={"order": {"total": 900}}).label)
```

Output:

```text
read order.total in [0, 1000000]
read order.country one of [domestic, eu, rest]
read customer.orders in [0, 10000]
ask suspicious = "This order reads like card testing or resale fraud." when order.total >= 500
when customer.orders >= 5 and order.total < 500: ship
when order.total < 50: ship
when suspicious >= 0.7: hold_for_review
when suspicious >= 0.3 and order.country == "rest": hold_for_review
always: ship
hold_for_review
```

A map whose units live in files of their own loads with `rulesmith.maps.load(path)`,
which embeds each unit and returns an ordinary `Plan`; see
[Composing graphs](graphs.md#composing-graphs).

Search runs from Python too; see [Optimization](optimization.md#from-python).
