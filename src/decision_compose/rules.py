"""A rule list: the shape every searched graph takes, in a form a model can write.

Rules compile to the graph language and render back from it, so search can propose rules
while the runtime keeps executing graphs. Graphs that do more than try rules in order --
merges, switches, several outputs -- have no rule form and stay JSON.
"""

import json
import re

from decision_compose.graph import (
    All,
    Call,
    Coalesce,
    Compare,
    Condition,
    NodeInput,
    Not,
    Plan,
    Present,
    Read,
    Threshold,
    Value,
    references,
)

OPERATORS = {">=": "ge", "<": "lt", "==": "eq"}
SYMBOLS = {name: symbol for symbol, name in OPERATORS.items()}
LABEL = r"[A-Za-z_][\w+.]*"
SECTIONS = ("readings", "rules")
JOINERS = ((" or ", "any"), (" and ", "all"))


def parts(plan: Plan) -> dict[str, str]:
    """The rules split into what the graph can know and what it decides, for search to revise
    one at a time. A graph beyond rules is carried whole, as JSON, in its rules part."""
    try:
        lines = render_rules(plan).splitlines()
    except RuleError:
        return {"readings": "", "rules": plan.model_dump_json(indent=2)}
    declared = [line for line in lines if line.startswith(("read ", "ask "))]
    readings, rules = SECTIONS
    return {readings: "\n".join(declared), rules: "\n".join(lines[len(declared) :])}


def whole(candidate: dict[str, str]) -> Plan:
    """One plan from the parts a proposer returned: readings first, so rules can name them."""
    order = [part for part in SECTIONS if part in candidate]
    order += [part for part in candidate if part not in SECTIONS]
    written = [candidate[part] for part in order if candidate[part]]
    # An answer in whole-graph JSON is the program entire, not a part to join to the rules.
    graph = [text for text in written if text.lstrip().startswith("{")]
    return compile_rules(graph[-1] if graph else "\n".join(written))


class RuleError(ValueError):
    """A rule list that cannot be read, or a graph that cannot be written as one."""


def path_of(read: Read) -> str:
    return ".".join(str(part) for part in read.path)


def render_condition(condition: Condition, reads: dict[str, Read]) -> str:
    """One condition as rule text; reads appear as the paths they take."""
    if isinstance(condition, Compare):
        value = condition.value
        return f"{name_of(condition.node, reads)} {SYMBOLS[condition.op]} {json.dumps(value)}"
    if isinstance(condition, Present):
        return name_of(condition.node, reads)
    if isinstance(condition, Not):
        return f"not {render_condition(condition.condition, reads)}"
    joiner = " and " if condition.type == "all" else " or "
    return joiner.join(render_condition(part, reads) for part in condition.conditions)


def name_of(node: str, reads: dict[str, Read]) -> str:
    """A read appears as its path; anything else by the name the rules gave it."""
    return path_of(reads[node]) if node in reads else node


def render_rules(plan: Plan) -> str:
    """The plan as rules, or a RuleError naming what has no rule form."""
    if set(plan.outputs) != {"label"}:
        raise RuleError(f"rules produce one label, not {sorted(plan.outputs)}")
    (output,) = plan.outputs.values()
    if not isinstance(output, NodeInput):
        return "always: " + json.dumps(output.value).strip('"')
    reads = {name: node for name, node in plan.nodes.items() if isinstance(node, Read)}
    rules = ordered(plan, output.node)
    acting = {name for name, _ in rules}
    lines = []
    for read in reads.values():
        domain = (
            f"in [{read.bounds[0]:g}, {read.bounds[1]:g}]"
            if read.bounds
            else "one of [" + ", ".join(read.categories or []) + "]"
        )
        lines.append(f"read {path_of(read)} {domain}")
    # A judgement a rule asks about, rather than acts on, is declared before the rules.
    for name, node in plan.nodes.items():
        if isinstance(node, Call) and name not in acting:
            gate = f" when {render_condition(node.when, reads)}" if node.when is not None else ""
            lines.append(f"ask {name} = {render_action(node, reads).removeprefix('ask ')}{gate}")
    for name, guard in rules:
        node = plan.nodes[name]
        condition = render_condition(guard, reads) if guard else None
        lines.append(
            f"{'when ' + condition if condition else 'always'}: {render_action(node, reads)}"
        )
    return "\n".join(lines)


def ordered(plan: Plan, top: str, guard: Condition | None = None) -> list[tuple[str, Condition]]:
    """Every rule the graph tries, in order, with the conditions that reach it."""
    node = plan.nodes[top]
    tests: list[Condition] = []
    for outer in (guard, node.when):
        # A group's guard and its rules often repeat a test; say it once.
        parts = outer.conditions if isinstance(outer, All) else [outer] if outer else []
        tests += [test for test in parts if test not in tests]
    combined = tests[0] if len(tests) == 1 else (All(conditions=tests) if tests else None)
    if isinstance(node, Coalesce):
        return [rule for source in node.sources for rule in ordered(plan, source, combined)]
    if isinstance(node, (Value, Call, Threshold)):
        return [(top, combined)]
    raise RuleError(f"{node.type} node {top!r} has no rule form")


def render_action(node, reads: dict[str, Read]) -> str:
    if isinstance(node, Value):
        return json.dumps(node.value).strip('"')
    if isinstance(node, Threshold):
        return f"{name_of(node.source, reads)} >= {node.cutoff:g} ? {node.above} : {node.below}"
    if len(node.inputs) != 1:
        raise RuleError(f"a judgement over {sorted(node.inputs)} has no rule form")
    question = node.question
    if question.type == "noul":
        return f"ask {json.dumps(question.instructions)}"
    if question.type == "score":
        levels = " ".join(f"[{level}]" for level in question.criteria)
        return f"ask {json.dumps(question.instructions)} {levels}"
    criteria = " ".join(f"[{k}: {json.dumps(v)}]" for k, v in question.criteria.items())
    return f"ask {json.dumps(question.instructions)} {criteria}"


def compile_rules(text: str) -> Plan:
    """A plan from rule text, or from whole-graph JSON for what rules cannot say."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            graph = json.loads(stripped)
            return pruned(graph.get("nodes") or {}, graph.get("outputs") or {})
        except ValueError as error:
            raise RuleError(str(error)) from error
    nodes: dict[str, dict] = {}
    names: dict[str, str] = {}
    sources: list[str] = []
    for number, line in enumerate(stripped.splitlines(), start=1):
        line = line.strip()
        # Asked for one part, a proposer tends to answer with the whole program: fenced, and with
        # the section headings written back out. Neither says anything about the rules themselves.
        if not line or line.startswith(("#", "```")) or line.rstrip(":") in SECTIONS:
            continue
        try:
            read_rule(line, nodes, names, sources)
        except RuleError as error:
            raise RuleError(f"line {number}: {error}") from error
    if not sources:
        raise RuleError("no rules")
    only = nodes[sources[0]]
    if len(sources) == 1 and only["type"] == "value" and "when" not in only:
        # A policy of one unconditional action is a constant, and says so.
        return Plan.constant(only["value"])
    nodes["decision"] = {"type": "coalesce", "sources": sources}
    return pruned(nodes, {"label": {"type": "node", "node": "decision"}})


def pruned(nodes: dict, outputs: dict) -> Plan:
    """The plan those nodes describe, less anything no output depends on: a reading nobody
    asks about, or a rule a rewrite stopped naming, decides nothing."""
    if not isinstance(nodes, dict):
        raise RuleError("nodes must be an object")
    if not isinstance(outputs, dict):
        raise RuleError("outputs must be an object")
    wanted: set[str] = set()
    pending = list(set().union(*(references(o) for o in outputs.values())) if outputs else set())
    while pending:
        node = pending.pop()
        if node not in wanted and node in nodes:
            wanted.add(node)
            pending.extend(references(nodes[node]))
    try:
        return Plan.model_validate(
            {"nodes": {k: v for k, v in nodes.items() if k in wanted}, "outputs": outputs}
        )
    except ValueError as error:
        raise RuleError(str(error)) from error


def read_rule(line: str, nodes: dict, names: dict, sources: list) -> None:
    if line.startswith("read "):
        declare_read(line[5:].strip(), nodes, names)
        return
    if line.startswith("ask ") and "=" in line.split("[", 1)[0]:
        name, _, question = line[4:].partition("=")
        question, gate = split_gate(question.strip())
        node = parse_action(f"ask {question}", names)
        if gate:
            node["when"] = parse_condition(gate, names)
        nodes[name.strip()] = node
        names[name.strip()] = name.strip()
        return
    head, _, action = line.partition(":")
    if not action:
        raise RuleError("a rule reads 'when <condition>: <action>' or 'always: <action>'")
    head = head.strip()
    guard = None if head == "always" else parse_condition(head.removeprefix("when").strip(), names)
    name = f"rule_{len(sources) + 1}"
    nodes[name] = parse_action(action.strip(), names) | ({"when": guard} if guard else {})
    sources.append(name)


def split_gate(text: str) -> tuple[str, str | None]:
    """A judgement from the condition it is asked under, past any criteria that quote 'when'."""
    criteria = text.rfind("]")
    instructions = re.match(r'\s*"(?:[^"\\]|\\.)*"', text)
    tail = max(criteria + 1, instructions.end() if instructions else 0)
    head, _, gate = text[tail:].partition(" when ")
    return text[:tail] + head, gate.strip() or None


def declare_read(text: str, nodes: dict, names: dict) -> None:
    path, _, domain = text.partition(" ")
    node: dict = {"type": "read", "path": [int(p) if p.isdigit() else p for p in path.split(".")]}
    if domain.startswith("in "):
        low, high = json.loads(domain[3:].replace("[", "[").strip())
        node["bounds"] = [low, high]
    elif domain.startswith("one of "):
        node["categories"] = [c.strip() for c in domain[7:].strip(" []").split(",")]
    else:
        raise RuleError(f"read {path} needs 'in [low, high]' or 'one of [a, b]'")
    name = "read_" + re.sub(r"\W", "_", path)
    nodes[name] = node
    names[path] = name


def parse_condition(text: str, reads: dict) -> dict:
    # 'or' binds loosest, so it splits first and each side may still be a run of 'and'.
    for joiner, kind in JOINERS:
        if joiner in text:
            sides = [parse_condition(side.strip(), reads) for side in text.split(joiner)]
            return {"type": kind, "conditions": sides}
    return parse_test(text.strip(), reads)


def parse_test(text: str, reads: dict) -> dict:
    if text.startswith("not "):
        return {"type": "not", "condition": parse_test(text[4:].strip(), reads)}
    for symbol in ("<=", ">"):
        if f" {symbol} " in text:
            path, _, value = text.partition(f" {symbol} ")
            # A graph compares with '<', '>=' and '=='; the other two orderings are made of those.
            below = {
                "type": "any",
                "conditions": [
                    parse_test(f"{path.strip()} < {value.strip()}", reads),
                    parse_test(f"{path.strip()} == {value.strip()}", reads),
                ],
            }
            return below if symbol == "<=" else {"type": "not", "condition": below}
    for symbol, op in OPERATORS.items():
        if f" {symbol} " in text:
            path, _, value = text.partition(f" {symbol} ")
            try:
                compared = json.loads(value.strip())
            except json.JSONDecodeError as error:
                raise RuleError(f"{value.strip()!r} is not a number or a quoted label") from error
            return {
                "type": "compare",
                "node": named(path.strip(), reads),
                "op": op,
                "value": compared,
            }
    return {"type": "present", "node": named(text, reads)}


def named(path: str, names: dict) -> str:
    if path not in names:
        raise RuleError(
            f"nothing named {path!r}; declare it with 'read {path} ...' or 'ask {path} = ...'"
        )
    return names[path]


def parse_action(text: str, names: dict) -> dict:
    if text.startswith("ask "):
        body = text[4:].strip()
        instructions, bracket, criteria = body.partition("[")
        question: dict = {"instructions": json.loads(instructions.strip())}
        pairs = re.findall(r"\[([^:\]]+):\s*(\"(?:[^\"\\\\]|\\\\.)*\")\]", bracket + criteria)
        levels = re.findall(r"\[([^:\]]+)\]", bracket + criteria)
        if pairs:
            question |= {"type": "choice", "criteria": {k.strip(): json.loads(v) for k, v in pairs}}
        elif levels:
            question |= {"type": "score", "criteria": [level.strip() for level in levels]}
        else:
            question |= {"type": "noul"}
        return {
            "type": "call",
            "question": question,
            "inputs": {"observation": {"type": "state"}},
        }
    if "?" in text and ":" in text:
        test, _, arms = text.partition("?")
        above, _, below = arms.partition(":")
        path, _, cutoff = test.strip().partition(" >= ")
        return {
            "type": "threshold",
            "source": named(path.strip(), names),
            "cutoff": float(cutoff),
            "above": above.strip(),
            "below": below.strip(),
        }
    return {"type": "value", "value": text}
