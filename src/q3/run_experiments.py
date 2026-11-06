#!/usr/bin/env python3
"""
Run the pre-registered Q3 ANN experiment plan one run at a time.

- Reads the plan CSV and validates required fields.
- Skips previously logged experiment IDs unless --rerun is supplied.
- Gives every repeated run a separate attempt number and run name.
- Logs completed and crashed executions.
- Keeps execution status separate from scientific interpretation.
- Uses validation metrics only. This script does not evaluate the test split.

Example:
  python src/q3/run_experiments.py --config configs/config.yaml --plan configs/q3_plan.csv --max-runs 2
  python src/q3/run_experiments.py --plan configs/q3_plan.csv --only E03
  python src/q3/run_experiments.py --plan configs/q3_plan.csv --dry-run

Commit or otherwise preserve the final experiment plan before formal runs.
"""

import argparse
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Make both src/ and src/q3 importable.
_SRC_DIR = Path(__file__).resolve().parent.parent
_Q3_DIR = Path(__file__).resolve().parent
for _directory in (str(_SRC_DIR), str(_Q3_DIR)):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

from common import load_config
from q3_utils import (
    append_log,
    git_commit,
    hardware_string,
    log_path,
    logged_ids,
    next_attempt,
    now,
    summarise_run,
)


REQUIRED = [
    "id",
    "group",
    "what_changed",
    "hypothesis",
    "weights",
    "epochs",
    "imgsz",
    "batch",
    "optimizer",
    "lr0",
    "seed",
]


def validate_plan(plan):
    """Check required columns and basic row completeness."""

    missing = [column for column in REQUIRED if column not in plan.columns]

    if missing:
        raise SystemExit(f"Plan is missing required columns: {missing}")

    if plan.empty:
        raise SystemExit("The selected experiment plan contains no rows.")

    if plan["id"].astype(str).str.strip().eq("").any():
        raise SystemExit("The plan contains an empty experiment ID.")

    duplicated = plan.loc[plan["id"].duplicated(), "id"].tolist()

    if duplicated:
        raise SystemExit(
            "The plan contains duplicate experiment IDs: "
            + ", ".join(sorted(set(duplicated)))
        )

    return plan


def build_command(python_executable, train_script, cfg_path, row, run_name):
    """Build the YOLO training command from one pre-registered plan row."""

    command = [
        python_executable,
        str(train_script),
        "--config",
        str(cfg_path),
        "--weights",
        str(row["weights"]),
        "--epochs",
        str(row["epochs"]),
        "--imgsz",
        str(row["imgsz"]),
        "--batch",
        str(row["batch"]),
        "--optimizer",
        str(row["optimizer"]),
        "--lr0",
        str(row["lr0"]),
        "--seed",
        str(row["seed"]),
        "--name",
        run_name,
    ]

    extra_args = str(row.get("extra_args", "")).strip()

    if extra_args:
        command.extend(shlex.split(extra_args))

    return command


def collect_run_directory(line):
    """Extract the run directory if the training script prints it."""

    match = re.match(r"Run directory:\s*(.+)", line.strip())

    if match:
        return Path(match.group(1).strip())

    return None


def execute_training(command):
    """
    Run the training process while displaying its output.

    Returns:
        return_code, run_directory, output_tail, duration_minutes, error_text
    """

    started = time.time()
    output_tail = []
    run_dir = None
    process = None
    error_text = ""

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )

        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
                output_tail.append(line.rstrip())
                output_tail = output_tail[-40:]

                found_dir = collect_run_directory(line)

                if found_dir is not None:
                    run_dir = found_dir

        return_code = process.wait()

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"

        if process is not None and process.poll() is None:
            process.kill()
            process.wait()

        return_code = -1

    duration_minutes = (time.time() - started) / 60

    return (
        return_code,
        run_dir,
        output_tail,
        duration_minutes,
        error_text,
    )


def make_log_row(row, attempt, run_name, duration_minutes, run_dir):
    """Create the common fields for an ANN experiment log entry."""

    return {
        "id": str(row["id"]),
        "attempt": attempt,
        "date": now(),
        "family": "yolo",
        "group": row["group"],
        "hypothesis": row["hypothesis"],
        "what_changed": row["what_changed"],
        "seed": row["seed"],
        "epochs": row["epochs"],
        "hyperparameters": (
            f"weights={row['weights']} "
            f"imgsz={row['imgsz']} "
            f"batch={row['batch']} "
            f"optimizer={row['optimizer']} "
            f"lr0={row['lr0']} "
            f"run_name={run_name} "
            f"{row.get('extra_args', '')}"
        ).strip(),
        "train_time_min": round(duration_minutes, 2),
        "git_commit": git_commit(),
        "run_dir": str(run_dir or ""),
        "hardware": hardware_string(),
        "scientific_outcome": "pending_review",
        "what_was_learned": "",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=None)
    parser.add_argument("--plan", default="configs/q3_plan.csv")
    parser.add_argument(
        "--train-script",
        default=str(Path(__file__).resolve().parent / "train_yolo.py"),
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these experiment IDs.",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--time-budget-min",
        type=float,
        default=None,
        help="Stop launching runs when the estimated next run exceeds this budget.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run IDs already in the log and record a new attempt number.",
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.max_runs is not None and args.max_runs < 1:
        raise SystemExit("--max-runs must be at least 1.")

    if args.time_budget_min is not None and args.time_budget_min <= 0:
        raise SystemExit("--time-budget-min must be greater than 0.")

    cfg = load_config(args.config)
    config_path = cfg["_path"]

    plan_path = Path(args.plan)

    if not plan_path.exists():
        raise SystemExit(f"Experiment plan not found: {plan_path}")

    train_script = Path(args.train_script)

    if not train_script.exists():
        raise SystemExit(f"Training script not found: {train_script}")

    plan = pd.read_csv(
        plan_path,
        dtype=str,
        keep_default_na=False,
    )

    plan = validate_plan(plan)

    if args.only:
        requested_ids = set(args.only)
        available_ids = set(plan["id"])
        missing_ids = requested_ids - available_ids

        if missing_ids:
            raise SystemExit(
                "Requested experiment IDs are not in the plan: "
                + ", ".join(sorted(missing_ids))
            )

        plan = plan[plan["id"].isin(requested_ids)]

    experiment_log = log_path(cfg)
    done = logged_ids(experiment_log)

    started = time.time()
    durations = []
    launched = 0

    for row in plan.to_dict("records"):

        experiment_id = str(row["id"]).strip()

        # Do not execute incomplete or placeholder plan rows.
        row_text = " ".join(str(value) for value in row.values())

        if "TODO" in row_text.upper():
            print(f"[skip] {experiment_id}: plan row still contains TODO")
            continue

        if experiment_id in done and not args.rerun:
            print(f"[skip] {experiment_id}: already recorded in the experiment log")
            continue

        if args.max_runs is not None and launched >= args.max_runs:
            print("[stop] Maximum number of runs reached.")
            break

        if args.time_budget_min is not None and durations:
            elapsed_minutes = (time.time() - started) / 60
            average_minutes = sum(durations) / len(durations)

            if elapsed_minutes + average_minutes > args.time_budget_min:
                print("[stop] Estimated next run would exceed the time budget.")
                break

        attempt = next_attempt(experiment_log, experiment_id)

        # Use a distinct output name for repeated attempts to avoid overwriting runs.
        run_name = (
            experiment_id
            if attempt == 1
            else f"{experiment_id}_attempt{attempt:02d}"
        )

        command = build_command(
            sys.executable,
            train_script,
            config_path,
            row,
            run_name,
        )

        print(
            "\n"
            + "=" * 100
            + f"\n{experiment_id} | attempt {attempt} | [{row['group']}]"
            + f"\nChange: {row['what_changed']}"
            + f"\nHypothesis: {row['hypothesis']}"
            + "\n$ "
            + subprocess.list2cmdline(command)
        )

        if args.dry_run:
            print("[dry-run] Command displayed. No training was started and nothing was logged.")
            continue

        (
            return_code,
            run_dir,
            output_tail,
            duration_minutes,
            execution_error,
        ) = execute_training(command)

        durations.append(duration_minutes)
        launched += 1

        log_row = make_log_row(
            row,
            attempt,
            run_name,
            duration_minutes,
            run_dir,
        )

        results_file_exists = (
            run_dir is not None
            and (run_dir / "results.csv").exists()
        )

        if return_code != 0 or not results_file_exists:
            diagnostic = []

            if return_code != 0:
                diagnostic.append(f"process_exit_code_{return_code}")

            if not results_file_exists:
                diagnostic.append("no_results_csv")

            if execution_error:
                diagnostic.append("launch_or_execution_error")

            log_row.update({
                "outcome": "crashed",
                "execution_status": "crashed",
                "scientific_outcome": "not_assessed",
                "auto_flags": ";".join(diagnostic),
                "what_was_learned": (
                    "Execution failed. Review the error and final console output: "
                    + " | ".join(output_tail[-5:])
                    + (f" | Error: {execution_error}" if execution_error else "")
                )[:1000],
            })

        else:
            log_row.update({
                "outcome": "completed",
                "execution_status": "completed",
            })

            try:
                summary = summarise_run(
                    run_dir,
                    planned_epochs=row["epochs"],
                )

                log_row.update({
                    "val_mAP50": summary["val_mAP50"],
                    "val_mAP50_95": summary["val_mAP50_95"],
                    "best_epoch": summary["best_epoch"],
                    "auto_flags": summary["flags"],
                    "epochs": summary["epochs_run"],
                })

            except Exception as exc:
                log_row.update({
                    "auto_flags": "results_summary_error",
                    "what_was_learned": (
                        "Training process completed, but results could not be summarised. "
                        f"Review results.csv. Error: {type(exc).__name__}: {exc}"
                    ),
                })

        # Log each attempt, including failed runs.
        append_log(experiment_log, log_row)

        # Update the in-memory set so duplicate IDs in a future plan edit
        # cannot accidentally run again during this same process.
        done.add(experiment_id)

        print(
            f"\nLogged {experiment_id}, attempt {attempt}: "
            f"execution={log_row.get('execution_status')}, "
            f"mAP50={log_row.get('val_mAP50', '')}, "
            f"flags={log_row.get('auto_flags', '')}"
        )

    print("\nExperiment plan processing finished.")
    print(f"Runs launched: {launched}")
    print(f"Experiment log: {experiment_log}")
    print(
        "Review each completed run and fill in what_was_learned "
        "and scientific_outcome using validation results and diagnostic evidence."
    )


if __name__ == "__main__":
    main()
