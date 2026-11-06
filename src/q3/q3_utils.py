"""Shared helpers for Q3 scripts."""

import json
import platform
import subprocess
import sys

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# Make Q3 and common project modules importable.
# Make both src/ and src/q3 importable.
_SRC_DIR = Path(__file__).resolve().parent.parent
_Q3_DIR = Path(__file__).resolve().parent
for _directory in (str(_SRC_DIR), str(_Q3_DIR)):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

from common import load_config, paths, read_yolo_labels, SPLITS  # noqa: E402
from prepare_data import make_image_id  # noqa: E402


# Experiment log columns.
# Existing columns are retained for compatibility.
LOG_COLUMNS = [
    "id",
    "attempt",
    "date",
    "family",
    "group",
    "hypothesis",
    "what_changed",
    "hyperparameters",
    "seed",
    "epochs",
    "val_mAP50",
    "val_mAP50_95",
    "best_epoch",
    "train_time_min",
    "outcome",
    "execution_status",
    "scientific_outcome",
    "auto_flags",
    "what_was_learned",
    "run_dir",
    "git_commit",
    "hardware",
]


def q3_dirs(cfg):
    """Return the standard Q3 output directories."""

    root = Path(cfg["outputs_dir"]) / "q3"

    return {
        "root": root,
        "preds": root / "preds",
        "eval": root / "eval",
        "err": root / "error_analysis",
        "baseline": root / "baseline",
        "figs": root / "figures",
        "tables": root / "tables",
        "lock": root / "TEST_EVALUATED.lock",
    }


def log_path(cfg):
    """Return the experiment log location."""

    return Path(cfg.get("experiment_log", "logs/experiments.csv"))


def image_table(cfg):
    """
    Create one row per image.

    Includes image_id, stem, split, dimensions,
    colour and group information.
    """

    P = paths(cfg)

    man = pd.read_csv(P["manifest"])
    sp = pd.read_csv(P["split_csv"])

    sp["image_id"] = sp["path"].map(
        lambda p: make_image_id(p, cfg["raw_root"])
    )

    keep = [
        c for c in ("image_id", "colour", "group")
        if c in sp.columns
    ]

    return man.merge(sp[keep], on="image_id", how="left")


def load_gt(cfg, split):
    """
    Load ground truth for one split.

    Returns:
        g: Ground-truth bounding boxes.
        imgs: Image information for the selected split.
    """

    if split not in SPLITS:
        raise ValueError(
            f"Unknown split '{split}'. Expected one of: {SPLITS}"
        )

    imgs = image_table(cfg)

    imgs = imgs[
        imgs["split"] == split
    ].reset_index(drop=True)

    lab = read_yolo_labels(cfg["resized_dir"])

    lab = lab[lab["split"] == split]

    g = lab.merge(
        imgs[["image_id", "rW", "rH", "colour"]],
        on="image_id",
        how="left",
    )

    g["x1"] = (g.cx - g.w / 2).clip(0, 1)
    g["y1"] = (g.cy - g.h / 2).clip(0, 1)
    g["x2"] = (g.cx + g.w / 2).clip(0, 1)
    g["y2"] = (g.cy + g.h / 2).clip(0, 1)

    g["short_px"] = np.minimum(
        g.w * g.rW,
        g.h * g.rH,
    )

    return g.reset_index(drop=True), imgs


def resolve_conf_thr(conf_thr, threshold_json):
    """
    Resolve the confidence threshold.

    A threshold JSON file takes priority.
    Threshold selection must be based on validation data.
    """

    if threshold_json:
        with open(threshold_json, "r", encoding="utf-8") as f:
            threshold_data = json.load(f)

        if "conf_thr" not in threshold_data:
            raise ValueError(
                f"'conf_thr' is missing from {threshold_json}"
            )

        return float(threshold_data["conf_thr"])

    return 0.25 if conf_thr is None else float(conf_thr)


def guard_test(cfg, split, allow_test):
    """
    Prevent accidental access to the test split.

    The test split should only be evaluated during final evaluation.
    """

    if split == "test" and not allow_test:
        raise SystemExit(
            "Refusing to touch the TEST split. "
            "Test is evaluated once by final_eval.py. "
            "error_analysis.py may read its saved predictions afterwards."
        )


def git_commit():
    """Return the current Git commit, if available."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

    except Exception:
        return "no-git"


def hardware_string():
    """Return available hardware information."""

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

            return (
                f"GPU: {gpu_name}; "
                f"CUDA: {torch.version.cuda}"
            )

    except Exception:
        pass

    return (
        f"CPU: {platform.processor() or platform.machine()}; "
        f"System: {platform.system()}"
    )


def append_log(path, row):
    """
    Append one experiment record.

    Existing experiment records and additional columns are preserved.
    New logging columns are added automatically.
    """

    p = Path(path)

    p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    old = pd.DataFrame()

    if p.exists() and p.stat().st_size > 0:
        try:
            old = pd.read_csv(
                p,
                dtype=str,
                keep_default_na=False,
            )

        except pd.errors.EmptyDataError:
            old = pd.DataFrame()

    # Add missing fields to the new record.
    new_data = {
        column: ""
        for column in LOG_COLUMNS
    }

    new_data.update(row)

    new_row = pd.DataFrame([
        {
            key: "" if value is None else str(value)
            for key, value in new_data.items()
        }
    ])

    # Preserve existing columns and their data.
    df = pd.concat(
        [old, new_row],
        ignore_index=True,
    ).fillna("")

    # Keep standard columns first.
    cols = (
        [c for c in LOG_COLUMNS if c in df.columns]
        + [c for c in df.columns if c not in LOG_COLUMNS]
    )

    df[cols].to_csv(
        p,
        index=False,
    )


def logged_ids(path):
    """
    Return experiment IDs already recorded.

    This keeps the existing interface used by the Q3 scripts.
    """

    p = Path(path)

    if not p.exists() or p.stat().st_size == 0:
        return set()

    try:
        df = pd.read_csv(
            p,
            dtype=str,
            keep_default_na=False,
        )

        if "id" not in df.columns:
            return set()

        return set(df["id"])

    except Exception:
        return set()


def next_attempt(path, experiment_id):
    """
    Return the next attempt number for an experiment ID.

    Example:
        First run  -> 1
        Second run -> 2
        Third run  -> 3
    """

    p = Path(path)

    if not p.exists() or p.stat().st_size == 0:
        return 1

    try:
        df = pd.read_csv(
            p,
            dtype=str,
            keep_default_na=False,
        )

        if "id" not in df.columns:
            return 1

        matching = df[df["id"] == str(experiment_id)]

        if matching.empty or "attempt" not in matching.columns:
            return 1

        attempts = pd.to_numeric(
            matching["attempt"],
            errors="coerce",
        ).dropna()

        if attempts.empty:
            return 1

        return int(attempts.max()) + 1

    except Exception:
        return 1


def read_results_csv(run_dir):
    """
    Read Ultralytics results.csv.

    Removes spaces from column names and converts
    numeric fields safely.
    """

    result_file = Path(run_dir) / "results.csv"

    if not result_file.exists():
        raise FileNotFoundError(
            f"Training results file not found: {result_file}"
        )

    df = pd.read_csv(
        result_file,
        skipinitialspace=True,
    )

    df.columns = [
        column.strip()
        for column in df.columns
    ]

    return df.apply(
        pd.to_numeric,
        errors="coerce",
    )


def summarise_run(run_dir, planned_epochs=None):
    """
    Summarise an Ultralytics training run.

    Returns:
        best_epoch
        val_mAP50
        val_mAP50_95
        epochs_run
        flags
    """

    df = read_results_csv(run_dir)

    if df.empty:
        raise ValueError(
            f"No training results found in {run_dir}"
        )

    m50 = "metrics/mAP50(B)"
    m95 = "metrics/mAP50-95(B)"

    if m50 not in df.columns:
        raise ValueError(
            f"Required validation metric '{m50}' is missing."
        )

    if m95 not in df.columns:
        raise ValueError(
            f"Required validation metric '{m95}' is missing."
        )

    valid_m50 = df[m50].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if valid_m50.notna().sum() == 0:
        raise ValueError(
            "No valid validation mAP50 values were recorded."
        )

    best_index = valid_m50.idxmax()

    best_m50 = float(df.loc[best_index, m50])
    best_m95 = float(df.loc[best_index, m95])

    flags = []

    # Check training losses.
    loss_cols = [
        c for c in df.columns
        if c.endswith("_loss")
    ]

    if loss_cols:
        loss_values = df[loss_cols].to_numpy(dtype=float)

        if not np.isfinite(loss_values).all():
            flags.append("nan_or_inf_loss")

    # Check validation metrics.
    metric_values = df[
        [m50, m95]
    ].to_numpy(dtype=float)

    if not np.isfinite(metric_values).all():
        flags.append("nan_or_inf_validation_metric")

    if best_m50 < 0.02:
        flags.append("map50_near_zero")

    n = len(df)

    if n >= 5 and best_index >= n - 3:
        flags.append("still_improving_at_end")

    # Check whether validation loss rises.
    val_loss_cols = [
        c for c in df.columns
        if c.startswith("val/") and c.endswith("_loss")
    ]

    if val_loss_cols and n >= 8:

        val_loss = df[val_loss_cols].sum(axis=1)

        if (
            val_loss.notna().any()
            and val_loss.iloc[-1] > 1.1 * val_loss.min()
            and int(val_loss.idxmin()) < 0.7 * n
        ):
            flags.append("val_loss_rising")

    # Check early stopping.
    if planned_epochs and n < int(planned_epochs):
        flags.append(
            f"stopped_early({n}/{planned_epochs})"
        )

    if "epoch" in df.columns:
        epoch_value = df.loc[best_index, "epoch"]

        best_epoch = (
            int(epoch_value)
            if pd.notna(epoch_value)
            else int(best_index) + 1
        )

    else:
        best_epoch = int(best_index) + 1

    return {
        "best_epoch": best_epoch,
        "val_mAP50": round(best_m50, 4),
        "val_mAP50_95": (
            round(best_m95, 4)
            if np.isfinite(best_m95)
            else ""
        ),
        "epochs_run": n,
        "flags": ";".join(flags),
    }


def now():
    """Return the current local date and time."""

    return datetime.now().isoformat(
        timespec="seconds"
    )