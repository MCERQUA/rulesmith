"""A rule list: the shape every searched graph takes, in a form a model can write.

Rules compile to the graph language and render back from it, so search can propose rules
while the runtime keeps executing graphs. Graphs that do more than try rules in order --
merges, switches, several outputs -- have no rule form and stay JSON.
"""

import json
import math
import re
from itertools import takewhile

from rulesmith.graph import (
    All,
    Any,
    Call,
    Coalesce,
    Compare,
    Condition,
    Confidence,
    LiteralInput,
    NodeInput,
    Not,
    Plan,
    Present,
    Read,
    Relation,
    StateInput,
    Sum,
    Threshold,
    Value,
    Vote,
    references,
)

OPERATORS = {">=": "ge", "<": "lt", "==": "eq"}
SYMBOLS = {name: symbol for symbol, name in OPERATORS.items()}
LABEL = r"[A-Za-z_][\w+.]*"
NAME = re.compile(r"[A-Za-z_]\w*")
# Arithmetic sets its operators apart with spaces, which keeps a minus from reading as a sign
# and leaves no way to confuse it with a condition.
ARITHMETIC = re.compile(r" [-+*] ")
NUMBER = re.compile(r"-?\d+(\.\d+)?(e-?\d+)?")
COMPARING = re.compile(r"[<>=]|\band\b|\bor\b|\bnot\b")
# The compiler names what the rules leave unnamed, so a written name must not look like one.
GENERATED = re.compile(r"(read|rule|decision)_\w*|decision")
# The output a task scores, and the one rules decide unless a section names another.
LABEL_OUTPUT = "label"
OUTPUT = re.compile(r"output (\w+):")
SECTIONS = ("readings", "rules")
# What a judgment asked about the whole input calls it.
WHOLE = "observation"
JOINERS = ((" or ", "any"), (" and ", "all"))


def parts(plan: Plan) -> dict[str, str]:
    """The rules split into what the graph can know and what it decides, for search to revise
    one at a time. A graph beyond rules is carried whole, as JSON, in its rules part."""
    try:
        lines = render_rules(plan).splitlines()
    except RuleError:
        return {"readings": "", "rules": plan.model_dump_json(indent=2)}
    # A judgment asked under a named condition follows that name, so it and everything after it
    # belong to the rules part, which may declare what its rules need.
    declared = list(takewhile(lambda line: line.startswith(("read ", "ask ")), lines))
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


def number(value: float | str) -> str:
    """A value as rule text: whole numbers without a trailing '.0', and every other number in
    full, since a rounded cutoff is a different graph."""
    whole = isinstance(value, float) and value.is_integer()
    return json.dumps(int(value) if whole else value)


def path_of(read: Read) -> str:
    return ".".join(str(part) for part in read.path)


def render_condition(condition: Condition, reads: dict[str, Read], plan: Plan | None = None) -> str:
    """One condition as rule text; reads appear as the paths they take."""
    if isinstance(condition, (Compare, Relation)):
        against = (
            name_of(condition.other, reads, plan)
            if isinstance(condition, Relation)
            else number(condition.value)
        )
        return f"{name_of(condition.node, reads, plan)} {SYMBOLS[condition.op]} {against}"
    if isinstance(condition, Present):
        return name_of(condition.node, reads, plan)
    if isinstance(condition, Not):
        return f"not {group(condition.condition, reads, plan)}"
    joiner = " and " if condition.type == "all" else " or "
    return joiner.join(group(part, reads, plan) for part in condition.conditions)


def group(condition: Condition, reads: dict[str, Read], plan: Plan | None = None) -> str:
    """One part of a longer condition. A group of its own needs parentheses, or reading it back
    would join it to its neighbors: 'a and (b or c)' is not 'a and b, or c'."""
    text = render_condition(condition, reads, plan)
    return f"({text})" if isinstance(condition, (All, Any)) else text


def name_of(node: str, reads: dict[str, Read], plan: Plan | None = None) -> str:
    """An unnamed read appears as its path, a confidence as the judgment it belongs to, and
    anything else by the name the rules gave it."""
    if node in reads and GENERATED.fullmatch(node):
        return path_of(reads[node])
    if plan is not None and isinstance(plan.nodes.get(node), Confidence):
        return f"{plan.nodes[node].source}.confidence"
    return node


def render_rules(plan: Plan) -> str:
    """The plan as rules, or a RuleError naming what has no rule form. The label's rules come
    first; every other output follows under its own 'output <name>:' heading."""
    reads = {name: node for name, node in plan.nodes.items() if isinstance(node, Read)}
    acting: set[str] = set()
    for output, source in plan.outputs.items():
        if isinstance(source, NodeInput):
            if source.view != "value" or source.optional:
                raise RuleError(f"output {output!r} has no rule form")
            acting |= spine(plan, source.node)
        elif not isinstance(source, LiteralInput):
            raise RuleError(f"output {output!r} is the whole input, which has no rule form")
    lines = []
    for name, read in reads.items():
        domain = (
            f"in [{number(read.bounds[0])}, {number(read.bounds[1])}]"
            if read.bounds
            else "one of [" + ", ".join(read.categories or []) + "]"
        )
        named = "" if GENERATED.fullmatch(name) else f"{name} = "
        lines.append(f"read {named}{path_of(read)} {domain}")
    lines += declarations(plan, acting, reads)
    for output in sorted(plan.outputs, key=lambda output: output != LABEL_OUTPUT):
        if output != LABEL_OUTPUT:
            lines.append(f"output {output}:")
        source = plan.outputs[output]
        if isinstance(source, LiteralInput):
            lines.append("always: " + json.dumps(source.value).strip('"'))
            continue
        for name, guard in ordered(plan, source.node):
            action = render_action(plan.nodes[name], reads, plan)
            condition = render_condition(guard, reads, plan) if guard else None
            lines.append(f"{'when ' + condition if condition else 'always'}: {action}")
    return "\n".join(lines)


def spine(plan: Plan, top: str) -> set[str]:
    """The rules the graph tries and the chain of fallbacks that joins them."""
    node = plan.nodes[top]
    if isinstance(node, Coalesce):
        return {top}.union(*(spine(plan, source) for source in node.sources))
    return {top}


def declarations(plan: Plan, acting: set[str], reads: dict[str, Read]) -> list[str]:
    """A line for every name the rules test, each after the names it is built from. Judgments
    come first and votes after them, as the reader expects, unless a name has to go earlier."""
    dependencies, written, lines = plan.dependencies(), set(), []

    def declare(name: str) -> None:
        if name in written or name in reads or name in acting:
            return
        written.add(name)
        for needed in sorted(dependencies[name]):
            declare(needed)
        node = plan.nodes[name]
        # A confidence is written where it is tested, as the judgment's name and '.confidence'.
        if not isinstance(node, Confidence):
            lines.append(declaration(name, node, reads, plan))

    rank = {Call: 0, Vote: 1}
    for name in sorted(plan.nodes, key=lambda name: rank.get(type(plan.nodes[name]), 2)):
        declare(name)
    return lines


def named_condition(node) -> bool:
    """A 'let' condition: a bare true that exists only when its guard holds."""
    return isinstance(node, Value) and node.value is True and node.when is not None


def declaration(name: str, node, reads: dict[str, Read], plan: Plan) -> str:
    if isinstance(node, Call):
        gate = f" when {render_condition(node.when, reads, plan)}" if node.when is not None else ""
        return f"ask {name} = {render_action(node, reads, plan).removeprefix('ask ')}{gate}"
    if isinstance(node, Vote):
        return f"vote {name} = {node.rule} of [{', '.join(node.sources)}]"
    if named_condition(node):
        return f"let {name} = {render_condition(node.when, reads, plan)}"
    if isinstance(node, Sum) and node.when is None:
        return f"let {name} = {render_sum(node, reads, plan)}"
    raise RuleError(f"{node.type} node {name!r} has no rule form")


def render_sum(node: Sum, reads: dict[str, Read], plan: Plan) -> str:
    """A sum as arithmetic: terms in order, a weight only where it is not one, the constant
    last, and each sign carried by the operator before it."""
    text = ""
    for index, term in enumerate(node.terms):
        weight = abs(term.weight)
        operand = name_of(term.node, reads, plan)
        operand = operand if weight == 1 else f"{number(weight)} * {operand}"
        sign = "-" if term.weight < 0 else "+"
        text += (f" {sign} " if index else "-" if sign == "-" else "") + operand
    if node.constant:
        text += f" {'-' if node.constant < 0 else '+'} {number(abs(node.constant))}"
    return text


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


def render_action(node, reads: dict[str, Read], plan: Plan) -> str:
    if isinstance(node, Value):
        return json.dumps(node.value).strip('"')
    if isinstance(node, Threshold):
        source = name_of(node.source, reads, plan)
        return f"{source} >= {number(node.cutoff)} ? {node.above} : {node.below}"
    question = node.question
    asked = f"ask {json.dumps(question.instructions)}"
    if question.type == "score":
        asked += " " + " ".join(f"[{level}]" for level in question.criteria)
    elif question.type == "choice":
        asked += " " + " ".join(f"[{k}: {json.dumps(v)}]" for k, v in question.criteria.items())
    return asked + render_given(node, reads, plan)


def render_given(node: Call, reads: dict[str, Read], plan: Plan) -> str:
    """What a judgment is shown: nothing to say for the whole input, which rules call
    'observation', and otherwise each value by the name that is also its key."""
    if node.inputs == {WHOLE: StateInput()}:
        return ""
    for key, source in node.inputs.items():
        if isinstance(source, StateInput):
            written = WHOLE
        elif isinstance(source, NodeInput) and source.view == "value" and not source.optional:
            written = name_of(source.node, reads, plan)
        else:
            written = None
        # The key is what the judge reads, so a rule form under another key would change it.
        if written != key:
            raise RuleError(f"a judgment's input {key!r} has no rule form")
    return f" given [{', '.join(node.inputs)}]"


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
    sections: dict[str, list[str]] = {LABEL_OUTPUT: []}
    for number, line in enumerate(stripped.splitlines(), start=1):
        line = line.strip()
        # Asked for one part, a proposer tends to answer with the whole program: fenced, and with
        # the section headings written back out. Neither says anything about the rules themselves.
        if not line or line.startswith(("#", "```")) or line.rstrip(":") in SECTIONS:
            continue
        heading = OUTPUT.fullmatch(line)
        if heading and heading.group(1) in sections:
            raise RuleError(f"line {number}: output {heading.group(1)!r} is declared twice")
        if heading:
            sections[heading.group(1)] = []
            continue
        try:
            read_rule(line, nodes, names, list(sections.values())[-1])
        except RuleError as error:
            raise RuleError(f"line {number}: {error}") from error
    if not any(sections.values()):
        raise RuleError("no rules")
    outputs = {}
    for output, rules in sections.items():
        # Rules may decide only other outputs, and a task that needs a label says so itself.
        if not rules and output == LABEL_OUTPUT:
            continue
        if not rules:
            raise RuleError(f"output {output!r} has no rules")
        only = nodes[rules[0]]
        if len(rules) == 1 and only["type"] == "value" and "when" not in only:
            # An output of one unconditional action is a constant, and says so.
            outputs[output] = {"type": "literal", "value": only["value"]}
            continue
        joined = "decision" if output == LABEL_OUTPUT else f"decision_{output}"
        nodes[joined] = {"type": "coalesce", "sources": rules}
        outputs[output] = {"type": "node", "node": joined}
    return pruned(nodes, outputs)


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
        node = parse_action(f"ask {question}", names, nodes)
        if gate:
            node["when"] = parse_condition(gate, names, nodes)
        claim(name.strip(), node, nodes, names)
        return
    if line.startswith("let ") and "=" in line:
        name, _, body = line[4:].partition("=")
        body = body.strip()
        if ARITHMETIC.search(f" {body}") and not COMPARING.search(body):
            node = parse_sum(body, names, nodes)
        else:
            # A named condition holds when its condition does: a value that exists only then,
            # which a rule tests by naming it.
            node = {"type": "value", "value": True, "when": parse_condition(body, names, nodes)}
        claim(name.strip(), node, nodes, names)
        return
    if line.startswith("vote ") and "=" in line:
        name, _, body = line[5:].partition("=")
        rule, _, listed = body.strip().partition(" of ")
        if rule.strip() not in ("majority", "agree"):
            raise RuleError(
                f"a vote is 'majority of [...]' or 'agree of [...]', not {rule.strip()!r}"
            )
        sources = [named(part.strip(), names, nodes) for part in listed.strip(" []").split(",")]
        if len(sources) < 2:
            raise RuleError("a vote needs at least two judgments")
        claim(
            name.strip(), {"type": "vote", "rule": rule.strip(), "sources": sources}, nodes, names
        )
        return
    head, _, action = line.partition(":")
    if not action:
        raise RuleError("a rule reads 'when <condition>: <action>' or 'always: <action>'")
    head = head.strip()
    guard = (
        None
        if head == "always"
        else parse_condition(head.removeprefix("when").strip(), names, nodes)
    )
    # Numbered across every output, so each output's rules keep distinct names.
    name = f"rule_{sum(key.startswith('rule_') for key in nodes) + 1}"
    nodes[name] = parse_action(action.strip(), names, nodes) | ({"when": guard} if guard else {})
    sources.append(name)


def parse_sum(text: str, names: dict, nodes: dict) -> dict:
    """'2 * gain - worst + 1' as weighted terms and a constant."""
    pieces = re.split(r" ([+-]) ", text)
    signed = [(1, pieces[0])] + [
        (1 if sign == "+" else -1, piece)
        for sign, piece in zip(pieces[1::2], pieces[2::2], strict=True)
    ]
    terms, constant = [], 0.0
    for sign, piece in signed:
        if piece.startswith("-") and not NUMBER.match(piece):
            sign, piece = -sign, piece[1:]
        factors = [factor.strip() for factor in piece.split(" * ")]
        operands = [factor for factor in factors if not NUMBER.fullmatch(factor)]
        weight = sign * math.prod(float(f) for f in factors if NUMBER.fullmatch(f))
        if len(operands) > 1:
            raise RuleError(f"{piece!r} multiplies two values; a weight must be a number")
        if operands:
            terms.append({"node": named(operands[0], names, nodes), "weight": weight})
        else:
            constant += weight
    if not terms:
        raise RuleError(f"{text!r} names no value to compute from")
    return {"type": "sum", "terms": terms, "constant": constant}


def claim(name: str, node: dict, nodes: dict, names: dict) -> None:
    """Declare a name. A rules part may restate a judgment the readings part declared, and the
    later one stands; a name that would mean a different kind of thing is refused, since the
    rules already written against it expect the first."""
    if not NAME.fullmatch(name):
        raise RuleError(f"{name!r} is not a name; use letters, digits and underscores")
    if GENERATED.fullmatch(name):
        raise RuleError(f"{name!r} is reserved: the compiler names read_, rule_ and decision")
    earlier = nodes.get(name) if names.get(name) == name else None
    if (name in names or name in nodes) and (earlier is None or earlier["type"] != node["type"]):
        raise RuleError(f"{name!r} is already declared")
    nodes[name] = node
    names[name] = name


def split_gate(text: str) -> tuple[str, str | None]:
    """A judgment from the condition it is asked under, past any criteria that quote 'when'."""
    criteria = text.rfind("]")
    instructions = re.match(r'\s*"(?:[^"\\]|\\.)*"', text)
    tail = max(criteria + 1, instructions.end() if instructions else 0)
    head, _, gate = text[tail:].partition(" when ")
    return text[:tail] + head, gate.strip() or None


def declare_read(text: str, nodes: dict, names: dict) -> None:
    alias = re.match(r"(\S+) = ", text)
    if alias:
        text = text[alias.end() :]
    path, _, domain = text.partition(" ")
    node: dict = {"type": "read", "path": [int(p) if p.isdigit() else p for p in path.split(".")]}
    if domain.startswith("in "):
        low, high = json.loads(domain[3:].replace("[", "[").strip())
        node["bounds"] = [low, high]
    elif domain.startswith("one of "):
        node["categories"] = [c.strip() for c in domain[7:].strip(" []").split(",")]
    else:
        raise RuleError(f"read {path} needs 'in [low, high]' or 'one of [a, b]'")
    if alias:
        claim(alias.group(1), node, nodes, names)
        return
    name = "read_" + re.sub(r"\W", "_", path)
    nodes[name] = node
    names[path] = name


def parse_condition(text: str, reads: dict, nodes: dict | None = None) -> dict:
    text = ungrouped(text.strip())
    # 'or' binds loosest, so it splits first and each side may still be a run of 'and'.
    for joiner, kind in JOINERS:
        sides = split_outside(text, joiner)
        if len(sides) > 1:
            return {
                "type": kind,
                "conditions": [parse_condition(side, reads, nodes) for side in sides],
            }
    return parse_test(text, reads, nodes)


def ungrouped(text: str) -> str:
    """The condition inside a pair of parentheses that wraps the whole of it."""
    while text.startswith("(") and depths(text)[-1] == 0 and min(depths(text)[:-1]) > 0:
        text = text[1:-1].strip()
    return text


def depths(text: str) -> list[int]:
    """How deep in parentheses each character leaves the text, after that character."""
    depth, reached = 0, []
    for character in text:
        depth += (character == "(") - (character == ")")
        reached.append(depth)
    return reached


def split_outside(text: str, joiner: str) -> list[str]:
    """The text split on a joiner, ignoring the ones inside a group."""
    sides, start = [], 0
    for index, depth in enumerate(depths(text)):
        if depth == 0 and text.startswith(joiner, index + 1):
            sides.append(text[start : index + 1])
            start = index + 1 + len(joiner)
    return [side.strip() for side in sides + [text[start:]]]


def parse_test(text: str, reads: dict, nodes: dict | None = None) -> dict:
    if text.startswith("not "):
        # What follows may be a group of its own, so read it as a whole condition.
        return {"type": "not", "condition": parse_condition(text[4:], reads, nodes)}
    for symbol in ("<=", ">"):
        if f" {symbol} " in text:
            path, _, value = text.partition(f" {symbol} ")
            # A graph compares with '<', '>=' and '=='; the other two orderings are made of those.
            below = {
                "type": "any",
                "conditions": [
                    parse_test(f"{path.strip()} < {value.strip()}", reads, nodes),
                    parse_test(f"{path.strip()} == {value.strip()}", reads, nodes),
                ],
            }
            return below if symbol == "<=" else {"type": "not", "condition": below}
    for symbol, op in OPERATORS.items():
        if f" {symbol} " in text:
            path, _, value = text.partition(f" {symbol} ")
            node = named(path.strip(), reads, nodes)
            try:
                compared = json.loads(value.strip())
            except json.JSONDecodeError:
                try:
                    other = named(value.strip(), reads, nodes)
                except RuleError as error:
                    raise RuleError(
                        f"{value.strip()!r} is not a number, a quoted label or a declared name"
                    ) from error
                return {"type": "relation", "node": node, "op": op, "other": other}
            return {"type": "compare", "node": node, "op": op, "value": compared}
    return {"type": "present", "node": named(text, reads, nodes)}


def named(path: str, names: dict, nodes: dict | None = None) -> str:
    """A name a condition tests. '<judgment>.confidence' is how sure that judgment was, and
    the node that reports it is made on first mention."""
    if path not in names and path.endswith(".confidence") and nodes is not None:
        asked = path.removesuffix(".confidence")
        # Only a judgment has a confidence. A reading is a number the input already carried.
        if asked in names and nodes.get(names[asked], {}).get("type") == "call":
            name = f"{names[asked]}_confidence"
            nodes.setdefault(name, {"type": "confidence", "source": names[asked]})
            names[path] = name
    if path not in names:
        raise RuleError(
            f"nothing named {path!r}; declare it with 'read {path} ...', 'ask {path} = ...' "
            f"or 'let {path} = ...'"
        )
    return names[path]


def parse_action(text: str, names: dict, nodes: dict) -> dict:
    if text.startswith("ask "):
        body = text[4:].strip()
        given = re.search(r"\s+given\s+\[([^\]]*)\]\s*$", body)
        inputs = {WHOLE: {"type": "state"}}
        if given:
            body = body[: given.start()]
            listed = [name.strip() for name in given.group(1).split(",")]
            inputs = {
                name: {"type": "state"}
                if name == WHOLE
                else {"type": "node", "node": named(name, names, nodes)}
                for name in listed
            }
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
            "inputs": inputs,
        }
    if "?" in text and ":" in text:
        test, _, arms = text.partition("?")
        above, _, below = arms.partition(":")
        path, _, cutoff = test.strip().partition(" >= ")
        return {
            "type": "threshold",
            "source": named(path.strip(), names, nodes),
            "cutoff": float(cutoff),
            "above": above.strip(),
            "below": below.strip(),
        }
    return {"type": "value", "value": text}
