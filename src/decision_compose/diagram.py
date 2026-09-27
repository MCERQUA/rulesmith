"""A plan as a Mermaid flowchart, for looking at a graph rules cannot express."""

from decision_compose.graph import (
    Call,
    Coalesce,
    Confidence,
    Merge,
    NodeInput,
    Plan,
    Read,
    Switch,
    Threshold,
    Value,
    Vote,
    tested,
)
from decision_compose.rules import path_of, render_condition

SHAPES = {
    Read: ("[/", "/]"),
    Call: ("{{", "}}"),
    Threshold: ("[", "]"),
    Switch: ("[", "]"),
    Merge: ("[", "]"),
    Coalesce: ("([", "])"),
    Vote: ("[/", "\\]"),
    Confidence: ("[", "]"),
    Value: ("((", "))"),
}


def label(name: str, node) -> str:
    """What a box says: enough to recognize the node, not its whole definition."""
    if isinstance(node, Read):
        return f"read {path_of(node)}"
    if isinstance(node, Call):
        return f"ask {node.question.type}: {node.question.instructions[:40]}"
    if isinstance(node, Threshold):
        return f"{name}: >= {node.cutoff:g} ? {node.above} : {node.below}"
    if isinstance(node, Value):
        # A named condition holds a bare true; its name is what the rules test.
        return name if node.value is True else str(node.value)
    if isinstance(node, Switch):
        return f"{name}: switch on {node.source}"
    if isinstance(node, Vote):
        return f"{name}: {node.rule} of {len(node.sources)}"
    if isinstance(node, Confidence):
        return f"how sure {node.source} was"
    return name


def safe(text: str) -> str:
    """Mermaid takes the label between its shape markers, so anything that closes one has to go."""
    for character in '"[]{}()|;\n':
        text = text.replace(character, " ")
    return " ".join(text.split())


def mermaid(plan: Plan) -> str:
    """The plan as a Mermaid flowchart: boxes for nodes, arrows for what feeds what, and the
    condition on a guarded node written along the arrow that reaches it."""
    lines = ["flowchart TD"]
    reads = {name: node for name, node in plan.nodes.items() if isinstance(node, Read)}
    for name, node in plan.nodes.items():
        open_shape, close_shape = SHAPES[type(node)]
        lines.append(f'    {name}{open_shape}"{safe(label(name, node))}"{close_shape}')
    for name, node in plan.nodes.items():
        for source in sorted(feeds(node)):
            lines.append(f"    {source} --> {name}")
        # A rule is often a bare value with a condition on it, so the guard has no data edge to
        # ride on. Draw it as its own dashed arrow from whatever the condition reads.
        if node.when is not None:
            guard = safe(render_condition(node.when, reads))
            for source in sorted(tested(node.when)):
                lines.append(f'    {source} -.->|"{guard}"| {name}')
    for output, source in plan.outputs.items():
        lines.append(f'    {output}(("{safe(output)}"))')
        if isinstance(source, NodeInput):
            lines.append(f"    {source.node} --> {output}")
    return "\n".join(lines)


def feeds(node) -> set[str]:
    """The nodes whose values this one consumes, ignoring the ones its guard mentions."""
    if isinstance(node, (Call, Merge)):
        return {source.node for source in node.inputs.values() if isinstance(source, NodeInput)}
    if isinstance(node, (Coalesce, Vote)):
        return set(node.sources)
    if isinstance(node, (Switch, Threshold, Confidence)):
        return {node.source}
    return set()
