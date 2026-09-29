"""A plan as a Mermaid flowchart, for looking at a graph rules cannot express."""

from textwrap import shorten

from rulesmith.graph import (
    Call,
    Coalesce,
    Confidence,
    Merge,
    NodeInput,
    Plan,
    Read,
    Sum,
    Switch,
    Threshold,
    Value,
    Vote,
    tested,
)
from rulesmith.rules import GENERATED, number, path_of, render_condition

SHAPES = {
    Read: ("[/", "/]"),
    Call: ("{{", "}}"),
    Threshold: ("[", "]"),
    Switch: ("[", "]"),
    Merge: ("[", "]"),
    Coalesce: ("([", "])"),
    Vote: ("[/", "\\]"),
    Confidence: ("[", "]"),
    Sum: ("[", "]"),
    Value: ("((", "))"),
}
# Colors by what a node costs to run, with dark text so each reads on a light or dark page: the
# judge is the one place a model is asked, whichever backend runs the graph.
KINDS = {
    "judge": (Call,),
    "read": (Read,),
    "rule": (Value, Threshold),
}
STYLES = {
    "judge": "fill:#fcd34d,stroke:#b45309,stroke-width:2px,color:#1c1917",
    "condition": "fill:#e5e7eb,stroke:#6b7280,color:#1c1917",
    "read": "fill:#bfdbfe,stroke:#1d4ed8,color:#1c1917",
    "rule": "fill:#bbf7d0,stroke:#15803d,color:#1c1917",
}


def label(name: str, node) -> str:
    """What a box says: enough to recognize the node, not its whole definition."""
    if isinstance(node, Read):
        return f"read {path_of(node)}"
    if isinstance(node, Call):
        # Named as the rules test it, and marked as the model call it is.
        named = "judge" if GENERATED.fullmatch(name) else f"judge: {name}"
        question = shorten(node.question.instructions, 48, placeholder="...")
        return f"{named}<br/>{node.question.type}: {question}"
    if isinstance(node, Threshold):
        return f"{name}: >= {number(node.cutoff)} ? {node.above} : {node.below}"
    if isinstance(node, Value):
        # A named condition holds a bare true; its name is what the rules test.
        return name if node.value is True else str(node.value)
    if isinstance(node, Switch):
        return f"{name}: switch on {node.source}"
    if isinstance(node, Vote):
        return f"{name}: {node.rule} of {len(node.sources)}"
    if isinstance(node, Coalesce):
        return f"{name}<br/>first that answers"
    if isinstance(node, Confidence):
        return f"how sure {node.source} was"
    if isinstance(node, Sum):
        return f"{name}: sum"
    if isinstance(node, Merge):
        return f"{name}<br/>merge"
    return name


def safe(text: str) -> str:
    """Mermaid takes the label between its shape markers, so anything that closes one has to go.
    Labels are quoted, and quoted text may hold parentheses, which a guard needs to keep its
    meaning."""
    for character in '"[]{}|;\n':
        text = text.replace(character, " ")
    return " ".join(text.split())


def mermaid(plan: Plan) -> str:
    """The plan as a Mermaid flowchart: boxes for nodes, arrows for what feeds what, and the
    condition on a guarded node written along the arrow that reaches it."""
    lines = ["flowchart TD"]
    reads = {name: node for name, node in plan.nodes.items() if isinstance(node, Read)}
    # Mermaid joins boxes that share an id, and a node may be named like an output or anything
    # else, so ids are positions and the names go in the labels.
    ids = {name: f"n{index}" for index, name in enumerate(plan.nodes)}
    conditions: list[str] = []
    for name, node in plan.nodes.items():
        open_shape, close_shape = SHAPES[type(node)]
        lines.append(f'    {ids[name]}{open_shape}"{safe(label(name, node))}"{close_shape}')
    for name, node in plan.nodes.items():
        if isinstance(node, Coalesce):
            # Tried in order, so the order is the point: a judge last is asked only when every
            # rule ahead of it declined.
            for rank, source in enumerate(node.sources, start=1):
                lines.append(f'    {ids[source]} -->|"{rank}"| {ids[name]}')
            continue
        for source in sorted(feeds(node)):
            lines.append(f"    {ids[source]} --> {ids[name]}")
        # A rule is often a bare value with a condition on it, so the guard has no data edge to
        # ride on. It is a box of its own, fed by each value it reads, with a dashed arrow to
        # what it gates: written on one arrow, a condition over two values looked as if it
        # belonged to one of them.
        if node.when is not None:
            guard = safe(render_condition(node.when, reads, plan))
            sources = sorted(tested(node.when))
            box = f"c{len(conditions)}"
            conditions.append(box)
            lines.append(f'    {box}["{guard}"]')
            lines += [f"    {ids[source]} --> {box}" for source in sources]
            lines.append(f"    {box} -.-> {ids[name]}")
    for index, (output, source) in enumerate(plan.outputs.items()):
        lines.append(f'    o{index}(("{safe(output)}"))')
        # Thick, so what the graph answers stands apart from what feeds what.
        if isinstance(source, NodeInput):
            lines.append(f"    {ids[source.node]} ==> o{index}")
    styled = {
        kind: [ids[name] for name, node in plan.nodes.items() if isinstance(node, types)]
        for kind, types in KINDS.items()
    } | {"condition": conditions}
    for kind, members in styled.items():
        if members:
            lines.append(f"    classDef {kind} {STYLES[kind]}")
            lines.append(f"    class {','.join(members)} {kind}")
    return "\n".join(lines)


def feeds(node) -> set[str]:
    """The nodes whose values this one consumes, ignoring the ones its guard mentions."""
    if isinstance(node, (Call, Merge)):
        return {source.node for source in node.inputs.values() if isinstance(source, NodeInput)}
    if isinstance(node, (Coalesce, Vote)):
        return set(node.sources)
    if isinstance(node, (Switch, Threshold, Confidence)):
        return {node.source}
    if isinstance(node, Sum):
        return {term.node for term in node.terms}
    return set()
