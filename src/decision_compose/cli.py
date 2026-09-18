"""Validate, execute, evaluate, and evolve a saved decision tree."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import dspy
from typesafe_sdk import RetryPolicy, TypeSafeClient

from decision_compose.graph import Plan, Task
from decision_compose.optimize import SearchConfig, evaluate_plan, optimize_plan
from decision_compose.runtime import DecisionProgram


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("validate", "evaluate", "optimize"):
        command = commands.add_parser(name)
        command.add_argument("task", type=Path)
        command.add_argument("plan", type=Path)
        command.add_argument("--max-questions", type=int, default=SearchConfig().max_questions)
        command.add_argument("--max-depth", type=int, default=SearchConfig().max_depth)
        if name == "validate":
            continue
        provider_arguments(command)
        command.add_argument("--output", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--split", choices=("train", "validation", "test"), default="test")
        else:
            command.add_argument(
                "--reflection-model",
                required=True,
                help="DSPy model identifier for the generative proposer",
            )
            command.add_argument("--reflection-base-url")
            command.add_argument("--reflection-max-tokens", type=int, default=4096)
            command.add_argument(
                "--max-metric-calls", type=int, default=SearchConfig().max_metric_calls
            )
            command.add_argument(
                "--reflection-batch", type=int, default=SearchConfig().reflection_batch
            )
            command.add_argument("--seed", type=int, default=SearchConfig().seed)
    run = commands.add_parser("run")
    run.add_argument("plan", type=Path)
    run.add_argument("state", type=Path, help="JSON file containing the input state")
    provider_arguments(run)
    return root


def provider_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--model", default="jev-latest")
    command.add_argument("--base-url", help="Override TYPESAFE_BASE_URL")
    command.add_argument("--timeout", type=float, default=60)


def dump(value: dict) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    plan = Plan.model_validate_json(args.plan.read_text())
    if args.command != "run":
        task = Task.model_validate_json(args.task.read_text())
        plan.check(task.labels, args.max_questions, args.max_depth)
    if args.command == "validate":
        print(dump({"questions": len(plan.questions), "labels": task.labels}), end="")
        return
    if args.command in ("evaluate", "optimize"):
        if args.output.exists():
            cli.error(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "optimize":
        config = SearchConfig(
            max_questions=args.max_questions,
            max_depth=args.max_depth,
            max_metric_calls=args.max_metric_calls,
            reflection_batch=args.reflection_batch,
            seed=args.seed,
        )
        lm_options = {"api_base": args.reflection_base_url} if args.reflection_base_url else {}
        reflection = dspy.LM(
            args.reflection_model,
            temperature=1,
            max_tokens=args.reflection_max_tokens,
            cache=False,
            num_retries=0,
            **lm_options,
        )

        def reflect(prompt):
            return (
                reflection(messages=prompt)[0]
                if isinstance(prompt, list)
                else reflection(prompt)[0]
            )

    with TypeSafeClient(
        base_url=args.base_url, timeout=args.timeout, retry=RetryPolicy(max_retries=0)
    ) as client:
        if args.command == "run":
            prediction = DecisionProgram(plan, client, args.model)(
                state=json.loads(args.state.read_text())
            )
            print(dump(dict(prediction)), end="")
        elif args.command == "evaluate":
            report = evaluate_plan(plan, getattr(task, args.split), client, args.model)
            report["split"] = args.split
            args.output.write_text(dump(report))
            print(dump({"accuracy": report["accuracy"], "output": str(args.output)}), end="")
        else:
            with redirect_stdout(sys.stderr):
                best, report = optimize_plan(task, plan, client, args.model, reflect, config)
            report["reflection_model"] = args.reflection_model
            args.output.mkdir()
            (args.output / "plan.json").write_text(best.model_dump_json(indent=2) + "\n")
            (args.output / "report.json").write_text(dump(report))
            print(dump(report), end="")


if __name__ == "__main__":
    main()
