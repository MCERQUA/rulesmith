"""Validate, execute, evaluate, and evolve a saved decision graph."""

import argparse
import json
import os
import signal
import sys
from collections import Counter
from contextlib import ExitStack, nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import dspy
import imageio_ffmpeg
import vizdoom as vzd
from PIL import Image
from typesafe_sdk import RetryPolicy, TypeSafeClient

from rulesmith.ablate import effects
from rulesmith.arena import Duels, Entrant, duel, tournament
from rulesmith.chess import ChessTask, Engine, Games
from rulesmith.diagram import mermaid
from rulesmith.doom import DoomEnvironment, DoomTask, Episodes
from rulesmith.grade import Graded, Grader, model_name
from rulesmith.graph import Call, Plan, Task
from rulesmith.label import Labeler, Unlabeled, draft_task
from rulesmith.optimize import (
    GraphProposer,
    Labels,
    SearchConfig,
    described_labels,
    evaluate_plan,
    optimize_plan,
)
from rulesmith.rules import compile_rules
from rulesmith.runtime import DecisionProgram, ExecutionConfig


@dataclass(frozen=True)
class Backend:
    model: str
    base_url_env: str
    api_key_env: str
    base_url: str | None = None
    api_key: str | None = None


BACKENDS = {
    "typesafe": Backend("jev-latest", "TYPESAFE_BASE_URL", "TYPESAFE_API_KEY"),
    # The SDK requires a nonempty key even when the local server disables auth.
    "ruling": Backend(
        "default", "RULING_BASE_URL", "RULING_API_KEY", "http://127.0.0.1:8010", "local-no-auth"
    ),
}


def resolve_backend(name: str, model: str | None = None, base_url: str | None = None):
    backend = BACKENDS[name]
    endpoint = (base_url or os.environ.get(backend.base_url_env, "")).strip() or backend.base_url
    key = os.environ.get(backend.api_key_env, "").strip() or backend.api_key
    return model or backend.model, {"base_url": endpoint, "api_key": key}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    helps = {
        "validate": "Check a graph against a task without calling any model",
        "evaluate": "Score a graph on one split of a task",
        "optimize": "Search for the graph with the best validation score",
        "record": "Play a graph in Doom and save the video",
    }
    for name, summary in helps.items():
        command = commands.add_parser(name, help=summary)
        command.add_argument("task", type=Path)
        if name == "optimize":
            command.add_argument(
                "plan", type=Path, nargs="?", help="Seed graph; defaults to the majority label"
            )
        else:
            command.add_argument("plan", type=Path)
        command.add_argument("--max-nodes", type=int, default=SearchConfig().max_nodes)
        command.add_argument("--max-depth", type=int, default=SearchConfig().max_depth)
        if name == "validate":
            continue
        provider_arguments(command)
        command.add_argument("--output", type=Path, required=True)
        if name in ("evaluate", "optimize"):
            command.add_argument(
                "--parallel-examples",
                type=positive,
                default=SearchConfig().parallel_examples,
                help="Examples or episodes scored at once; deathmatch episodes can run together",
            )
        if name in ("evaluate", "optimize", "record"):
            command.add_argument(
                "--opponent",
                type=Path,
                help="Graph a deathmatch task's episodes are played against",
            )
        if name in ("evaluate", "optimize"):
            command.add_argument(
                "--engine",
                help="A UCI engine, such as stockfish, to play a chess task against instead of "
                "a graph",
            )
            command.add_argument(
                "--engine-skill", type=int, help="The engine's Skill Level (default 0, weakest)"
            )
            command.add_argument(
                "--engine-seconds", type=float, help="The engine's time per move (default 0.05)"
            )
            command.add_argument(
                "--grader-model",
                help="DSPy model identifier that grades answers instead of the task's labels",
            )
            command.add_argument("--grader-base-url")
        if name == "record":
            command.add_argument(
                "--seed", type=int, nargs="+", required=True, help="Episode seeds, played in order"
            )
        elif name == "evaluate":
            command.add_argument("--split", choices=("train", "validation", "test"), default="test")
            command.add_argument(
                "--visible", action="store_true", help="Show the game window for a Doom task"
            )
        else:
            command.add_argument(
                "--reflection-model",
                required=True,
                help="DSPy model identifier for the generative proposer",
            )
            command.add_argument("--reflection-base-url")
            command.add_argument("--reflection-max-tokens", type=int, default=4096)
            command.add_argument(
                "--reflection-timeout", type=float, help="Seconds before a proposal request fails"
            )
            command.add_argument(
                "--max-metric-calls", type=int, default=SearchConfig().max_metric_calls
            )
            command.add_argument(
                "--reflection-batch", type=int, default=SearchConfig().reflection_batch
            )
            command.add_argument(
                "--repair-attempts", type=int, default=SearchConfig().repair_attempts
            )
            command.add_argument("--merges", type=int, default=SearchConfig().merges)
            command.add_argument(
                "--mine",
                action="store_true",
                help="Mine rules a field alone settles, and start there if they score better",
            )
            command.add_argument(
                "--significance",
                type=float,
                help="Keep the seed unless the winner's gain passes a sign test at this level",
            )
            command.add_argument(
                "--call-cost",
                type=float,
                default=SearchConfig().call_cost,
                help="Score off an example per judge question, to prefer graphs that ask less",
            )
            command.add_argument("--seed", type=int, default=SearchConfig().seed)
    draft = commands.add_parser("draft-labels", help="Label raw inputs for review as a task")
    draft.add_argument("inputs", type=Path, help="JSON with description, labels, and inputs")
    draft.add_argument("--output", type=Path, required=True)
    draft.add_argument(
        "--labeler-model", required=True, help="DSPy model identifier for the labeler"
    )
    draft.add_argument("--labeler-base-url")
    draft.add_argument("--validation", type=int, required=True, help="Validation example count")
    draft.add_argument("--test", type=int, required=True, help="Test example count")
    draft.add_argument("--seed", type=int, default=0, help="Split shuffle seed")
    arena = commands.add_parser("arena", help="Rank Doom graphs by Elo from deathmatch duels")
    arena.add_argument("task", type=Path, help="Doom task whose episodes are the matches")
    arena.add_argument("plans", type=Path, nargs="+", help="Entrants, named by file stem")
    arena.add_argument("--split", choices=("train", "validation", "test"), default="test")
    arena.add_argument("--max-nodes", type=int, default=SearchConfig().max_nodes)
    arena.add_argument("--max-depth", type=int, default=SearchConfig().max_depth)
    arena.add_argument("--output", type=Path, required=True)
    provider_arguments(arena)
    draw = commands.add_parser("draw", help="Write a plan as a Mermaid flowchart")
    draw.add_argument("plan", type=Path)
    ablation = commands.add_parser(
        "ablate", help="Measure what each reading, judgment and name in a graph is worth"
    )
    ablation.add_argument("task", type=Path)
    ablation.add_argument("plan", type=Path)
    ablation.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    ablation.add_argument("--output", type=Path)
    provider_arguments(ablation)
    run = commands.add_parser("run", help="Run a graph on one input and print what it decided")
    run.add_argument("plan", type=Path)
    run.add_argument("state", type=Path, help="JSON file containing the input state")
    provider_arguments(run)
    return root


def provider_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--backend", choices=tuple(BACKENDS), default="typesafe")
    command.add_argument("--model", help="Model ID; defaults to the selected backend alias")
    command.add_argument("--base-url", help="Override the selected backend endpoint")
    command.add_argument("--timeout", type=float, default=60)
    command.add_argument(
        "--retries", type=int, default=0, help="Retries for failed or timed-out judge requests"
    )
    command.add_argument("--max-workers", type=positive, default=ExecutionConfig().max_workers)


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {value}")
    return value


def judge(args: argparse.Namespace, needed: bool):
    """The judge model, and a client for it when some graph calls it."""
    model, connection = resolve_backend(args.backend, args.model, args.base_url)
    if not needed:
        return model, nullcontext()
    retry = RetryPolicy(max_retries=args.retries)
    return model, TypeSafeClient(**connection, timeout=args.timeout, retry=retry)


def ablation(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """What each input is worth, over the states the task itself supplies."""
    task = load_task(cli, args.task)
    if args.output and args.output.exists():
        cli.error(f"output already exists: {args.output}")
    # A plan is structurally valid once it loads; size limits bound what search may write, and
    # have nothing to say about measuring a graph that already exists.
    plan = read_plan(cli, args.plan, task)
    examples = getattr(task, args.split)
    if not all(hasattr(example, "state") for example in examples):
        cli.error(
            "ablate reads states from a task's examples; a Doom or chess task keeps seeds, "
            "so measure those graphs with rulesmith.ablate.effects over recorded states"
        )
    states = [example.state for example in examples]
    calls = any(isinstance(node, Call) for node in plan.nodes.values())
    model, judge_client = judge(args, calls)
    with judge_client as client:
        execution = ExecutionConfig(max_workers=args.max_workers)
        found = effects(plan, states, client if calls else None, model, execution)
    report = {
        "states": len(states),
        "effects": [
            {
                "node": effect.node,
                "kind": effect.kind,
                "changed": effect.changed,
                "share": round(effect.share, 4),
                "reason": effect.reason,
            }
            for effect in found
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(dump(report))
    print(dump(report), end="")


def arena(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    task = load_task(cli, args.task)
    if not isinstance(task, DoomTask) or not task.deathmatch:
        cli.error("arena requires a Doom task with deathmatch set")
    if args.output.exists():
        cli.error(f"output already exists: {args.output}")
    plans = {}
    for path in args.plans:
        if path.stem in plans:
            cli.error(f"entrants share the name {path.stem}; rename a plan file")
        plans[path.stem] = read_plan(cli, path, task)
        plans[path.stem].check(task.labels, args.max_nodes, args.max_depth)
    if len(plans) < 2:
        cli.error("the arena needs at least two entrants")
    calls = any(isinstance(node, Call) for plan in plans.values() for node in plan.nodes.values())
    model, judge_client = judge(args, calls)
    execution = ExecutionConfig(max_workers=args.max_workers)
    with judge_client as client:
        entrants = [
            Entrant(name, DecisionProgram(plan, client, model, execution))
            for name, plan in plans.items()
        ]
        report = tournament(task, entrants, getattr(task, args.split))
    report["split"] = args.split
    report["backend"] = args.backend
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dump(report))
    print(dump({"ratings": report["ratings"], "output": str(args.output)}), end="")


def dump(value: dict) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def draft_labels(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.output.exists():
        cli.error(f"output already exists: {args.output}")
    unlabeled = Unlabeled.model_validate_json(args.inputs.read_text())
    lm_options = {"api_base": args.labeler_base_url} if args.labeler_base_url else {}
    labeler = Labeler(
        dspy.LM(args.labeler_model, cache=False, num_retries=0, **lm_options),
        unlabeled.description,
        unlabeled.labels,
    )
    task = draft_task(
        unlabeled.description,
        unlabeled.labels,
        unlabeled.inputs,
        labeler,
        args.validation,
        args.test,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(task.model_dump_json(indent=2) + "\n")
    counts = Counter(e.label for e in task.train + task.validation + task.test)
    print(dump({"labels": dict(counts), "output": str(args.output)}), end="")


def save_video(frames: list, fps: int, path: Path) -> None:
    if path.suffix == ".mp4":
        height, width, _ = frames[0].shape
        writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps)
        writer.send(None)
        for frame in frames:
            writer.send(frame)
        writer.close()
        return
    first, *rest = (Image.fromarray(frame) for frame in frames)
    # GIF delays are whole centiseconds, so round to 10 ms rather than truncate.
    delay = round(100 / fps) * 10
    first.save(path, save_all=True, append_images=rest, duration=delay, loop=0)


def scorer(args: argparse.Namespace, task):
    """What scores a labeled task's answers: its labels, or a grader when one is named. A game
    task's scorer is its games, chosen where the games are set up."""
    if not isinstance(task, Task):
        return None
    if not args.grader_model:
        return Labels(task)
    lm_options = {"api_base": args.grader_base_url} if args.grader_base_url else {}
    lm = dspy.LM(args.grader_model, cache=False, num_retries=0, **lm_options)
    return Graded(Grader(lm, task.description))


def scored_by(args: argparse.Namespace, task) -> str:
    """How a report's scores were made, since a grader's score is not accuracy."""
    if getattr(args, "grader_model", None):
        return f"grader {args.grader_model}"
    return "labels" if isinstance(task, Task) else "games"


def opponent_of(args: argparse.Namespace) -> str | None:
    """Who a played task's scores were earned against, since a score means nothing without it."""
    if getattr(args, "engine", None):
        return f"engine {args.engine} at skill {args.engine_skill or 0}"
    return str(args.opponent) if getattr(args, "opponent", None) else None


def read_plan(
    cli: argparse.ArgumentParser, path: Path, task: Task | DoomTask | ChessTask | None = None
) -> Plan:
    """A graph from its JSON, or from the rules it is written as, which may ask over the labels
    of the task the command names. A file that does not hold one is a mistake in the command,
    reported as such rather than as a crash."""
    text = path.read_text()
    try:
        if path.suffix == ".rules":
            return compile_rules(text, described_labels(task) if task else None)
        return Plan.model_validate_json(text)
    except ValueError as error:
        cli.error(f"{path}: {error}")


def load_task(cli: argparse.ArgumentParser, path: Path) -> Task | DoomTask | ChessTask:
    try:
        data = json.loads(path.read_text())
        kind = (
            ChessTask if data.get("game") == "chess" else DoomTask if "scenario" in data else Task
        )
        return kind.model_validate(data)
    except ValueError as error:
        cli.error(f"{path}: {error}")


def main() -> None:
    # The default SIGTERM action skips cleanup, which leaves ViZDoom engines running (they
    # ignore SIGTERM themselves); exiting normally closes every environment on the way out.
    signal.signal(signal.SIGTERM, lambda number, _: sys.exit(128 + number))
    cli = parser()
    args = cli.parse_args()
    if args.command == "draft-labels":
        draft_labels(cli, args)
        return
    if args.command == "draw":
        print(mermaid(read_plan(cli, args.plan)))
        return
    if args.command == "arena":
        arena(cli, args)
        return
    if args.command == "ablate":
        ablation(cli, args)
        return
    task = None if args.command == "run" else load_task(cli, args.task)
    plan = read_plan(cli, args.plan, task) if args.plan else task.baseline_plan()
    if args.command != "run":
        plan.check(task.labels, args.max_nodes, args.max_depth)
    if args.command == "validate":
        print(dump({"nodes": len(plan.nodes), "labels": task.labels}), end="")
        return
    if getattr(args, "visible", False) and not isinstance(task, DoomTask):
        cli.error("--visible requires a Doom task")
    deathmatch = args.command != "run" and isinstance(task, DoomTask) and task.deathmatch
    played = deathmatch or (args.command != "run" and isinstance(task, ChessTask))
    engine = getattr(args, "engine", None)
    tuned = [flag for flag in ("engine_skill", "engine_seconds") if getattr(args, flag, None)]
    if tuned and not engine:
        cli.error("--engine-skill and --engine-seconds set the strength of an --engine")
    if engine and not isinstance(task, ChessTask):
        cli.error("--engine plays a chess task; other games are played against a graph")
    if engine and args.opponent:
        cli.error("a chess task is played against --opponent or --engine, not both")
    against = bool(args.opponent or engine) if hasattr(args, "opponent") else False
    if args.command in ("evaluate", "optimize", "record") and played != against:
        cli.error("--opponent is required for, and only for, tasks played against another graph")
    opponent = None
    if played and not engine:
        opponent = read_plan(cli, args.opponent, task)
        opponent.check(task.labels, args.max_nodes, args.max_depth)
    if getattr(args, "parallel_examples", 1) > 1 and isinstance(task, DoomTask) and not deathmatch:
        cli.error("--parallel-examples needs a deathmatch task; other Doom episodes share one game")
    if args.command == "record" and not isinstance(task, DoomTask):
        cli.error("record requires a Doom task")
    if args.command == "record" and args.output.suffix not in (".gif", ".mp4"):
        cli.error("record writes .gif or .mp4")
    grading = getattr(args, "grader_model", None)
    if getattr(args, "grader_base_url", None) and not grading:
        cli.error("--grader-base-url needs a --grader-model to send grades to")
    if grading and not isinstance(task, Task):
        cli.error("--grader-model grades a labeled task's examples; games score themselves")
    if grading and model_name(grading) == model_name(resolve_backend(args.backend, args.model)[0]):
        # Search would learn to ask the grader's own question and score perfectly for nothing.
        cli.error("--grader-model is the judge the graph asks; grade with a different model")
    if getattr(args, "visible", False) and deathmatch:
        cli.error("--visible shows a single-player Doom game; a deathmatch has no one window")
    if args.command in ("evaluate", "optimize") and isinstance(task, Task) and not grading:
        splits = ("train", "validation") if args.command == "optimize" else (args.split,)
        unlabeled = [e.id for split in splits for e in getattr(task, split) if e.label is None]
        if unlabeled:
            cli.error(f"examples {unlabeled} have no label; label them or pass a --grader-model")
    if args.command in ("evaluate", "optimize", "record"):
        if args.output.exists():
            cli.error(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "optimize":
        config = SearchConfig(
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
            max_metric_calls=args.max_metric_calls,
            reflection_batch=args.reflection_batch,
            repair_attempts=args.repair_attempts,
            seed=args.seed,
            parallel_examples=args.parallel_examples,
            merges=args.merges,
            call_cost=args.call_cost,
            significance=args.significance,
            mine=args.mine,
        )
        lm_options = {"api_base": args.reflection_base_url} if args.reflection_base_url else {}
        if args.reflection_timeout is not None:
            lm_options["timeout"] = args.reflection_timeout
        reflection = dspy.LM(
            args.reflection_model,
            temperature=1,
            max_tokens=args.reflection_max_tokens,
            cache=False,
            num_retries=0,
            **lm_options,
        )

        reflect = GraphProposer(reflection)

    execution = ExecutionConfig(max_workers=args.max_workers)
    model, judge_client = judge(
        args,
        args.command == "optimize"
        or any(
            isinstance(node, Call)
            for graph in (plan, opponent)
            if graph is not None
            for node in graph.nodes.values()
        ),
    )
    with ExitStack() as stack:
        client = stack.enter_context(judge_client)
        score = scorer(args, task) if args.command in ("evaluate", "optimize") else None
        if deathmatch:
            score = Duels(task, DecisionProgram(opponent, client, model, execution))
        elif engine:
            skill = args.engine_skill or 0
            rival = Engine(engine, skill, args.engine_seconds or 0.05)
            score = Games(task, stack.enter_context(rival))
        elif args.command != "run" and isinstance(task, ChessTask):
            score = Games(task, DecisionProgram(opponent, client, model, execution))
        elif args.command != "run" and isinstance(task, DoomTask):
            env = DoomEnvironment(
                task, visible=getattr(args, "visible", False), record=args.command == "record"
            )
            score = Episodes(task, stack.enter_context(env))
        if args.command == "run":
            prediction = DecisionProgram(plan, client, model, execution)(
                state=json.loads(args.state.read_text())
            )
            print(dump(dict(prediction)), end="")
        elif args.command == "record":
            program = DecisionProgram(plan, client, model, execution)
            frames, episodes = [], []
            for seed in args.seed:
                if deathmatch:
                    played, _ = duel(task, program, score.opponent, seed, True, frames)
                    episodes.append(played.record)
                else:
                    episodes.append(score(program, seed).record)
                    frames += env.frames
            save_video(frames, vzd.DEFAULT_TICRATE, args.output)
            print(dump({"episodes": episodes, "output": str(args.output)}), end="")
        elif args.command == "evaluate":
            report = evaluate_plan(
                plan,
                getattr(task, args.split),
                client,
                model,
                score,
                execution,
                args.parallel_examples,
            )
            report["split"] = args.split
            report["backend"] = args.backend
            report["model"] = model
            report["scored_by"] = scored_by(args, task)
            report["opponent"] = opponent_of(args)
            args.output.write_text(dump(report))
            print(dump({"score": report["score"], "output": str(args.output)}), end="")
        else:
            with redirect_stdout(sys.stderr):
                best, report = optimize_plan(
                    task, plan, client, model, reflect, config, execution, score
                )
            report["backend"] = args.backend
            report["reflection_model"] = args.reflection_model
            report["scored_by"] = scored_by(args, task)
            report["opponent"] = opponent_of(args)
            args.output.mkdir()
            (args.output / "plan.json").write_text(best.model_dump_json(indent=2) + "\n")
            (args.output / "report.json").write_text(dump(report))
            print(dump(report), end="")


if __name__ == "__main__":
    main()
