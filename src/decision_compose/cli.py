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

from decision_compose.arena import Duels, Entrant, duel, tournament
from decision_compose.doom import DoomEnvironment, DoomTask, Episodes
from decision_compose.graph import Call, Plan, Task
from decision_compose.label import Labeler, Unlabeled, draft_task
from decision_compose.optimize import (
    GraphProposer,
    SearchConfig,
    classify,
    evaluate_plan,
    optimize_plan,
)
from decision_compose.runtime import DecisionProgram, ExecutionConfig


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
    for name in ("validate", "evaluate", "optimize", "record"):
        command = commands.add_parser(name)
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
        if name in ("evaluate", "optimize", "record"):
            command.add_argument(
                "--parallel-examples",
                type=int,
                default=SearchConfig().parallel_examples,
                help="Examples or episodes scored at once; deathmatch episodes can run together",
            )
            command.add_argument(
                "--opponent",
                type=Path,
                help="Graph a deathmatch task's episodes are played against",
            )
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
    run = commands.add_parser("run")
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
    command.add_argument("--max-workers", type=int, default=ExecutionConfig().max_workers)


def judge(args: argparse.Namespace, needed: bool):
    """The judge model, and a client for it when some graph calls it."""
    model, connection = resolve_backend(args.backend, args.model, args.base_url)
    if not needed:
        return model, nullcontext()
    retry = RetryPolicy(max_retries=args.retries)
    return model, TypeSafeClient(**connection, timeout=args.timeout, retry=retry)


def arena(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    task = load_task(args.task)
    if not isinstance(task, DoomTask) or not task.deathmatch:
        cli.error("arena requires a Doom task with deathmatch set")
    if args.output.exists():
        cli.error(f"output already exists: {args.output}")
    plans = {}
    for path in args.plans:
        if path.stem in plans:
            cli.error(f"entrants share the name {path.stem}; rename a plan file")
        plans[path.stem] = Plan.model_validate_json(path.read_text())
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


def load_task(path: Path) -> Task | DoomTask:
    data = json.loads(path.read_text())
    return (DoomTask if "scenario" in data else Task).model_validate(data)


def main() -> None:
    # The default SIGTERM action skips cleanup, which leaves ViZDoom engines running (they
    # ignore SIGTERM themselves); exiting normally closes every environment on the way out.
    signal.signal(signal.SIGTERM, lambda number, _: sys.exit(128 + number))
    cli = parser()
    args = cli.parse_args()
    if args.command == "draft-labels":
        draft_labels(cli, args)
        return
    if args.command == "arena":
        arena(cli, args)
        return
    if args.command != "run":
        task = load_task(args.task)
    plan = Plan.model_validate_json(args.plan.read_text()) if args.plan else task.baseline_plan()
    if args.command != "run":
        plan.check(task.labels, args.max_nodes, args.max_depth)
    if args.command == "validate":
        print(dump({"nodes": len(plan.nodes), "labels": task.labels}), end="")
        return
    if getattr(args, "visible", False) and not isinstance(task, DoomTask):
        cli.error("--visible requires a Doom task")
    deathmatch = args.command != "run" and isinstance(task, DoomTask) and task.deathmatch
    if args.command in ("evaluate", "optimize", "record") and deathmatch != bool(args.opponent):
        cli.error("--opponent is required for, and only for, deathmatch tasks")
    if getattr(args, "parallel_examples", 1) > 1 and isinstance(task, DoomTask) and not deathmatch:
        cli.error("--parallel-examples needs a deathmatch task; other Doom episodes share one game")
    if args.command == "record" and not isinstance(task, DoomTask):
        cli.error("record requires a Doom task")
    if args.command == "record" and args.output.suffix not in (".gif", ".mp4"):
        cli.error("record writes .gif or .mp4")
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
        args.command == "optimize" or any(isinstance(node, Call) for node in plan.nodes.values()),
    )
    with ExitStack() as stack:
        client = stack.enter_context(judge_client)
        score = classify
        if deathmatch:
            opponent = Plan.model_validate_json(args.opponent.read_text())
            opponent.check(task.labels, args.max_nodes, args.max_depth)
            score = Duels(task, DecisionProgram(opponent, client, model, execution))
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
                execution,
                score,
                args.parallel_examples,
            )
            report["split"] = args.split
            report["backend"] = args.backend
            args.output.write_text(dump(report))
            print(dump({"score": report["score"], "output": str(args.output)}), end="")
        else:
            with redirect_stdout(sys.stderr):
                best, report = optimize_plan(
                    task, plan, client, model, reflect, config, execution, score
                )
            report["backend"] = args.backend
            report["reflection_model"] = args.reflection_model
            args.output.mkdir()
            (args.output / "plan.json").write_text(best.model_dump_json(indent=2) + "\n")
            (args.output / "report.json").write_text(dump(report))
            print(dump(report), end="")


if __name__ == "__main__":
    main()
