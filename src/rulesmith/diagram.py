"""A plan as a Mermaid flowchart, for looking at a graph rules cannot express."""

from textwrap import shorten

from rulesmith.graph import (
    Call,
    Coalesce,
    Confidence,
    Graph,
    Merge,
    NodeInput,
    Plan,
    Present,
    Read,
    Sum,
    Switch,
    Threshold,
    Value,
    Vote,
    tested,
)
from rulesmith.rules import GENERATED, name_of, number, path_of, render_condition

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
    "unit": (Graph,),
}
STYLES = {
    "judge": "fill:#fcd34d,stroke:#b45309,stroke-width:2px,color:#1c1917",
    "unit": "fill:#ede9fe,stroke:#7c3aed,color:#1c1917",
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


def mermaid(plan: Plan, closed: bool = False) -> str:
    """The plan as a Mermaid flowchart: boxes for nodes, arrows for what feeds what, a box of
    its own for each condition, and a unit's graph drawn inside a box of its own, or, closed,
    as one box, so a map of many units reads as a map."""
    lines = ["flowchart TD"]
    styled: dict[str, list[str]] = {}
    draw(plan, "", lines, styled, closed)
    for kind, members in styled.items():
        if members:
            lines.append(f"    classDef {kind} {STYLES[kind]}")
            lines.append(f"    class {','.join(members)} {kind}")
    return "\n".join(lines)


def guard(condition, reads: dict[str, Read], plan: Plan) -> str:
    """A condition as its box says it. In rules a bare name tests that the node answered, which
    reads fine after `when`; alone in a box it needs the verb."""
    if isinstance(condition, Present):
        return f"{name_of(condition.node, reads, plan)} is present"
    return render_condition(condition, reads, plan)


def draw(
    plan: Plan,
    prefix: str,
    lines: list[str],
    styled: dict[str, list[str]],
    closed: bool = False,
    exit: tuple[str, str] | None = None,
) -> None:
    """One plan's nodes, edges and outputs, with ids under `prefix` so a unit's graph and the
    graph around it never share an id. `exit` names the output that is a unit's answer and the
    id of the unit's exit node, which stands in for that output's circle."""
    reads = {name: node for name, node in plan.nodes.items() if isinstance(node, Read)}
    # Mermaid joins boxes that share an id, and a node may be named like an output or anything
    # else, so ids are positions and the names go in the labels.
    ids = {name: f"{prefix}n{index}" for index, name in enumerate(plan.nodes)}
    conditions: list[str] = []
    # What a unit's gate and input attach to: the unit's whole box, since the condition decides
    # whether any of it runs.
    whole: dict[str, str] = {}
    for name, node in plan.nodes.items():
        if isinstance(node, Graph):
            if closed:
                # The double-edged box is the usual sign for a step that is a program of its
                # own, which a closed unit is.
                lines.append(f'    {ids[name]}[["{safe(name)}<br/>a graph of its own"]]')
                whole[name] = ids[name]
            else:
                whole[name] = f"{ids[name]}_unit"
                lines.append(f'    subgraph {whole[name]}["{safe(name)}: a graph of its own"]')
                answers = f"{safe(name)}<br/>answers with its {node.output}"
                lines.append(f'    {ids[name]}(["{answers}"])')
                draw(node.plan, f"{ids[name]}_", lines, styled, closed, (node.output, ids[name]))
                lines.append("    end")
            if isinstance(node.input, NodeInput):
                lines.append(f"    {ids[node.input.node]} --> {whole[name]}")
            continue
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
            sources = sorted(tested(node.when))
            box = f"{prefix}c{len(conditions)}"
            conditions.append(box)
            lines.append(f'    {box}["{safe(guard(node.when, reads, plan))}"]')
            lines += [f"    {ids[source]} --> {box}" for source in sources]
            lines.append(f"    {box} -.-> {whole.get(name, ids[name])}")
    for index, (output, source) in enumerate(plan.outputs.items()):
        # A unit's answer leaves through its exit node; every other output is a circle.
        if exit is not None and output == exit[0]:
            target = exit[1]
        else:
            target = f"{prefix}o{index}"
            lines.append(f'    {target}(("{safe(output)}"))')
        # Thick, so what the graph answers stands apart from what feeds what.
        if isinstance(source, NodeInput):
            lines.append(f"    {ids[source.node]} ==> {target}")
    for kind, types in KINDS.items():
        styled.setdefault(kind, []).extend(
            ids[name] for name, node in plan.nodes.items() if isinstance(node, types)
        )
    styled.setdefault("condition", []).extend(conditions)


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
