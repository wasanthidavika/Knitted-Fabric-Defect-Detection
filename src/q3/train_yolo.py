#!/usr/bin/env python3
"""Train COCO-pretrained YOLOv8 on the frozen ISL-Knit train/validation split.

Run from the project root:
    python src/train_yolo.py --config configs/config.yaml

This script uses train/validation paths only. It never passes data_test.yaml
into training. Defaults are an initial 30-epoch screening configuration,
not a claimed optimum.

CHANGES vs the previous version (Q3): new optional flags --freeze, --cls, --box, --momentum, --cos-lr, --lrf,
--warmup-epochs (only passed to Ultralytics when set, so defaults are unchanged); the recorded "model" now comes
from the weights file name (yolov8n/s/m) instead of being hard-coded to YOLOv8s.
"""
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO, __version__ as ultralytics_version

# Make both src/ and src/q3 importable.
_SRC_DIR = Path(__file__).resolve().parent.parent
_Q3_DIR = Path(__file__).resolve().parent
for _directory in (str(_SRC_DIR), str(_Q3_DIR)):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)
from common import load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on ISL-Knit.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--weights", default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="", help="Empty lets Ultralytics choose; e.g. 0 or cpu.")
    p.add_argument("--optimizer", default="AdamW", choices=["SGD", "AdamW", "Adam", "auto"])
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--project", default=None)
    p.add_argument("--name", default="yolov8s_screening_seed42")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--hsv-h", type=float, default=0.015)
    p.add_argument("--hsv-s", type=float, default=0.7)
    p.add_argument("--hsv-v", type=float, default=0.4)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--flipud", type=float, default=0.0)
    p.add_argument("--cache", action="store_true")
    # --- added for Q3 (optional; passed only when set) ---
    p.add_argument("--freeze", type=int, default=None, help="freeze the first N layers (10 = backbone)")
    p.add_argument("--cls", type=float, default=None, help="classification loss gain")
    p.add_argument("--box", type=float, default=None, help="box loss gain")
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--cos-lr", action="store_true")
    p.add_argument("--lrf", type=float, default=None, help="final lr fraction")
    p.add_argument("--warmup-epochs", type=float, default=None)
    return p.parse_args()


def make_training_yaml(resized_dir, class_names, output_dir):
    train_images = resized_dir / "images" / "train"
    val_images = resized_dir / "images" / "val"
    required = [train_images, val_images,
                resized_dir / "labels" / "train",
                resized_dir / "labels" / "val"]
    missing = [str(x) for x in required if not x.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Prepared YOLO folders are missing:\n  " + "\n  ".join(missing)
            + "\nRun src/prepare_data.py successfully before training."
        )
    n_train = len(list(train_images.glob("*")))
    n_val = len(list(val_images.glob("*")))
    if not n_train or not n_val:
        raise RuntimeError(f"Empty split: train={n_train}, val={n_val}")

    # Deliberately excludes the test split.
    data = {
        "path": str(resized_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {int(k): str(v) for k, v in class_names.items()},
        "nc": len(class_names),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "data_train_val.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"Training images: {n_train}")
    print(f"Validation images: {n_val}")
    print(f"Classes: {len(class_names)}")
    print(f"Train/validation YAML: {yaml_path}")
    return yaml_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"] if args.seed is None else args.seed)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    root = Path.cwd()
    resized_dir = Path(cfg["resized_dir"])
    if not resized_dir.is_absolute():
        resized_dir = root / resized_dir
    outputs_dir = Path(cfg["outputs_dir"])
    if not outputs_dir.is_absolute():
        outputs_dir = root / outputs_dir
    project_dir = Path(args.project) if args.project else outputs_dir / "q3"
    if not project_dir.is_absolute():
        project_dir = root / project_dir

    data_yaml = make_training_yaml(resized_dir, cfg["class_names"], project_dir / "configs")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("epochs and batch must be >= 1; imgsz must be >= 32.")

    optional = {}
    if args.freeze is not None:
        optional["freeze"] = args.freeze
    if args.cls is not None:
        optional["cls"] = args.cls
    if args.box is not None:
        optional["box"] = args.box
    if args.momentum is not None:
        optional["momentum"] = args.momentum
    if args.cos_lr:
        optional["cos_lr"] = True
    if args.lrf is not None:
        optional["lrf"] = args.lrf
    if args.warmup_epochs is not None:
        optional["warmup_epochs"] = args.warmup_epochs

    settings = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model": Path(args.weights).stem,
        "weights": args.weights,
        "pretrained": True,
        "data_yaml": str(data_yaml.resolve()),
        "test_split_used": False,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device or "Ultralytics auto",
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "amp": not args.no_amp,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "cache": args.cache,
        "optional_overrides": optional,
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics_version,
    }
    project_dir.mkdir(parents=True, exist_ok=True)
    config_record = project_dir / f"{args.name}_config.json"
    config_record.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print("Training settings:")
    print(json.dumps(settings, indent=2))
    print("The test split is excluded from training and validation.")

    # Default weights are COCO-pretrained YOLOv8. Ultralytics downloads them
    # on first use if they are not already in its weights cache.
    model = YOLO(args.weights)
    results = model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers,
        device=args.device if args.device else None,
        optimizer=args.optimizer, lr0=args.lr0,
        weight_decay=args.weight_decay, patience=args.patience,
        seed=seed, deterministic=True, pretrained=True,
        amp=not args.no_amp, project=str(project_dir), name=args.name,
        exist_ok=False, save=True, save_period=5, plots=True, val=True,
        cache=args.cache, mosaic=args.mosaic, close_mosaic=args.close_mosaic,
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
        fliplr=args.fliplr, flipud=args.flipud,
        **optional,
    )
    save_dir = Path(getattr(results, "save_dir", project_dir / args.name))
    print("\nTraining finished.")
    print(f"Run directory: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last weights: {save_dir / 'weights' / 'last.pt'}")
    print(f"Training results: {save_dir / 'results.csv'}")
    print(f"Run settings: {config_record}")


if __name__ == "__main__":
    main()
