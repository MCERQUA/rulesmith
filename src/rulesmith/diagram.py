"""A plan as a Mermaid flowchart, for looking at a graph rules cannot express."""

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
from rulesmith.rules import number, path_of, render_condition

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


def label(name: str, node) -> str:
    """What a box says: enough to recognize the node, not its whole definition."""
    if isinstance(node, Read):
        return f"read {path_of(node)}"
    if isinstance(node, Call):
        return f"ask {node.question.type}: {node.question.instructions[:40]}"
    if isinstance(node, Threshold):
        return f"{name}: >= {number(node.cutoff)} ? {node.above} : {node.below}"
    if isinstance(node, Value):
        # A named condition holds a bare true; its name is what the rules test.
        return name if node.value is True else str(node.value)
    if isinstance(node, Switch):
        return f"{name}: switch on {node.source}"
    if isinstance(node, Vote):
        return f"{name}: {node.rule} of {len(node.sources)}"
    if isinstance(node, Confidence):
        return f"how sure {node.source} was"
    if isinstance(node, Sum):
        return f"{name}: sum"
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
    for name, node in plan.nodes.items():
        open_shape, close_shape = SHAPES[type(node)]
        lines.append(f'    {ids[name]}{open_shape}"{safe(label(name, node))}"{close_shape}')
    for name, node in plan.nodes.items():
        for source in sorted(feeds(node)):
            lines.append(f"    {ids[source]} --> {ids[name]}")
        # A rule is often a bare value with a condition on it, so the guard has no data edge to
        # ride on. Draw it as its own dashed arrow from whatever the condition reads.
        if node.when is not None:
            guard = safe(render_condition(node.when, reads, plan))
            for source in sorted(tested(node.when)):
                lines.append(f'    {ids[source]} -.->|"{guard}"| {ids[name]}')
    for index, (output, source) in enumerate(plan.outputs.items()):
        lines.append(f'    o{index}(("{safe(output)}"))')
        if isinstance(source, NodeInput):
            lines.append(f"    {ids[source.node]} --> o{index}")
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
