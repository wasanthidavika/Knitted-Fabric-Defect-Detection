"""Shared helpers for the ISL-Knit project."""
import os, random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SPLITS = ["train", "val", "test"]
DEV_SPLITS = ["train", "val"]          # test is NEVER used for analysis or decisions


def load_config(path=None):
    path = path or os.environ.get("ISL_CONFIG") or "configs/config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["class_names"] = {int(k): str(v) for k, v in cfg["class_names"].items()}
    cfg["_path"] = str(path)
    return cfg


def paths(cfg):
    audit, res, out = Path(cfg["audit_dir"]), Path(cfg["resized_dir"]), Path(cfg["outputs_dir"])
    return {"audit": audit, "split_csv": audit / "proposed_splits.csv", "audit_report": audit / "report.json",
            "resized": res, "manifest": res / "manifest.csv", "out": out, "q1": out / "q1", "q2": out / "q2"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_yolo_labels(resized_dir):
    """Read YOLO labels using unique image IDs."""
    rows = []

    for sp in SPLITS:
        d = Path(resized_dir) / "labels" / sp

        if not d.is_dir():
            continue

        for f in sorted(d.glob("*.txt")):
            image_id = f.stem

            for ln in f.read_text().splitlines():
                t = ln.split()

                if len(t) == 5:
                    rows.append(
                        (sp, image_id, int(float(t[0])), *map(float, t[1:]))
                    )

    return pd.DataFrame(
        rows,
        columns=["split", "image_id", "cls", "cx", "cy", "w", "h"]
    )