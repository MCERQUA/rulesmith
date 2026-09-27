"""A plan as a Mermaid flowchart, for looking at a graph rules cannot express."""

from decision_compose.graph import (
    All,
    Any,
    Call,
    Coalesce,
    Compare,
    Merge,
    Not,
    NodeInput,
    Plan,
    Present,
    Read,
    Switch,
    Threshold,
    Value,
)
from decision_compose.rules import path_of, render_condition

SHAPES = {
    Read: ("[/", "/]"),
    Call: ("{{", "}}"),
    Threshold: ("[", "]"),
    Switch: ("[", "]"),
    Merge: ("[", "]"),
    Coalesce: ("([", "])"),
    Value: ("((", "))"),
}


def label(name: str, node) -> str:
    """What a box says: enough to recognise the node, not its whole definition."""
    if isinstance(node, Read):
        return f"read {path_of(node)}"
    if isinstance(node, Call):
        return f"ask {node.question.type}: {node.question.instructions[:40]}"
    if isinstance(node, Threshold):
        return f"{name}: >= {node.cutoff:g} ? {node.above} : {node.below}"
    if isinstance(node, Value):
        return str(node.value)
    if isinstance(node, Switch):
        return f"{name}: switch on {node.source}"
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
        lines.append(f"    {name}{open_shape}\"{safe(label(name, node))}\"{close_shape}")
    for name, node in plan.nodes.items():
        for source in sorted(feeds(node)):
            lines.append(f"    {source} --> {name}")
        # A rule is often a bare value with a condition on it, so the guard has no data edge to
        # ride on. Draw it as its own dashed arrow from whatever the condition reads.
        if node.when is not None:
            guard = safe(render_condition(node.when, reads))
            for source in sorted(gates(node.when)):
                lines.append(f"    {source} -.->|\"{guard}\"| {name}")
    for output, source in plan.outputs.items():
        lines.append(f"    {output}((\"{safe(output)}\"))")
        if isinstance(source, NodeInput):
            lines.append(f"    {source.node} --> {output}")
    return "\n".join(lines)


def gates(condition) -> set[str]:
    """The nodes a condition tests."""
    if isinstance(condition, (Compare, Present)):
        return {condition.node}
    if isinstance(condition, Not):
        return gates(condition.condition)
    if isinstance(condition, (All, Any)):
        return {node for part in condition.conditions for node in gates(part)}
    return set()


def feeds(node) -> set[str]:
    """The nodes whose values this one consumes, ignoring the ones its guard mentions."""
    if isinstance(node, (Call, Merge)):
        return {source.node for source in node.inputs.values() if isinstance(source, NodeInput)}
    if isinstance(node, Coalesce):
        return set(node.sources)
    if isinstance(node, (Switch, Threshold)):
        return {node.source}
    return set()
