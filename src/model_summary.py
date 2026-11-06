#!/usr/bin/env python3
"""
Q2.a evidence generator: builds YOLOv8s (randomly initialised, no download) with nc=7 and records
 * every layer: index, input link(s), type, repeats, parameters, output shape at imgsz
 * parameter counts (nc=7 and the COCO nc=80 reference), GFLOPs, strides, Detect-head details
 * module inventory (Conv2d / BatchNorm2d / SiLU ...) so claims about normalisation and activation are VERIFIED
Writes outputs/q2/{layers.csv, param_summary.json, q2_architecture_table.md}.
Usage:  python src/model_summary.py --config configs/config.yaml [--cfg yolov8s.yaml] [--imgsz 640]
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, paths, set_seed


def shape_str(o):
    if isinstance(o, torch.Tensor):
        return "x".join(map(str, o.shape[1:]))
    if isinstance(o, (list, tuple)):
        return "[" + "; ".join(shape_str(x) for x in o) + "]"
    return type(o).__name__


def nparams(m, trainable_only=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--cfg", default="yolov8s.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    P = paths(cfg)
    P["q2"].mkdir(parents=True, exist_ok=True)
    nc = len(cfg["class_names"])

    model = DetectionModel(a.cfg, ch=3, nc=nc, verbose=False)
    ref80 = DetectionModel(a.cfg, ch=3, nc=80, verbose=False)
    n_back = len(model.yaml["backbone"])
    n_head = len(model.yaml["head"])           # neck layers + final Detect
    shapes, hooks = {}, []
    for m in model.model:
        hooks.append(m.register_forward_hook(lambda mod, i, o, idx=m.i: shapes.__setitem__(idx, shape_str(o))))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, a.imgsz, a.imgsz))
    for h in hooks:
        h.remove()

    rows = []
    for m in model.model:
        reps = len(m.m) if hasattr(m, "m") and isinstance(m.m, nn.ModuleList) else 1
        section = "backbone" if m.i < n_back else ("detect head" if m.i == len(model.model) - 1 else "neck")
        rows.append({"idx": m.i, "from_layers": str(m.f), "type": m.type.split(".")[-1], "repeats": reps,
                     "params": nparams(m), "out_shape": shapes.get(m.i, ""), "section": section})
    layers = pd.DataFrame(rows)
    layers.to_csv(P["q2"] / "layers.csv", index=False)

    det = model.model[-1]
    modcount = Counter(type(m).__name__ for m in model.modules())
    acts = sorted({type(m).__name__ for m in model.modules() if isinstance(m, (nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Hardswish, nn.Sigmoid))})
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = float(get_flops(model, a.imgsz))
    except Exception as e:
        gflops = None
        print("FLOPs not computed:", e)
    summary = {
        "cfg": a.cfg, "imgsz": a.imgsz, "nc": nc,
        "total_params_nc7": nparams(model), "total_params_nc80_coco_reference": nparams(ref80),
        "params_by_section": layers.groupby("section")["params"].sum().to_dict(),
        "gflops": gflops, "strides": [float(s) for s in det.stride],
        "detect": {"nc": det.nc, "reg_max": det.reg_max, "outputs_per_location": det.no,
                   "box_branch_params": nparams(det.cv2), "cls_branch_params": nparams(det.cv3), "dfl_params": nparams(det.dfl)},
        "module_counts": {k: modcount[k] for k in ("Conv2d", "BatchNorm2d", "SiLU", "Upsample", "MaxPool2d", "Dropout") if k in modcount},
        "activation_types_found": acts,
        "yaml_depth_multiple": model.yaml.get("depth_multiple"), "yaml_width_multiple": model.yaml.get("width_multiple"),
    }
    json.dump(summary, open(P["q2"] / "param_summary.json", "w"), indent=2)

    md = "| idx | from | type | x n | params | output (C x H x W) | section |\n|---|---|---|---|---:|---|---|\n"
    for r in layers.itertuples():
        md += f"| {r.idx} | {r.from_layers} | {r.type} | {r.repeats} | {r.params:,} | {r.out_shape} | {r.section} |\n"
    md += f"\nTotal parameters (nc={nc}): **{summary['total_params_nc7']:,}**; COCO reference (nc=80): {summary['total_params_nc80_coco_reference']:,}; GFLOPs @ {a.imgsz}: {gflops}\n"
    (P["q2"] / "q2_architecture_table.md").write_text(md)
    print(md)
    print(json.dumps(summary, indent=2))
    print("\nVERIFY in the report: activation types found =", acts, "| BatchNorm2d count =", modcount.get("BatchNorm2d"))


if __name__ == "__main__":
    main()
