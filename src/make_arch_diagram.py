#!/usr/bin/env python3
"""
Q2.a architecture diagram, drawn from outputs/q2/layers.csv (so labels come from the real model).
Writes outputs/q2/architecture.png and an editable architecture.mmd (Mermaid).
Usage:  python src/make_arch_diagram.py --config configs/config.yaml [--layers path/to/layers.csv]
"""
import argparse, ast, json, sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, paths

COLOURS = {"Conv": "#cfe2f3", "C2f": "#d9ead3", "SPPF": "#fce5cd", "Upsample": "#eeeeee", "Concat": "#ead1dc", "Detect": "#f4cccc"}
BW, BH = 4.4, 0.62


def parse_from(s, i):
    v = ast.literal_eval(s)
    v = v if isinstance(v, (list, tuple)) else [v]
    return [(i + x if x < 0 else x) for x in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--layers", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    P = paths(cfg)
    lp = Path(a.layers) if a.layers else P["q2"] / "layers.csv"
    L = pd.read_csv(lp).rename(columns={"from": "from_layers"})
    sj = P["q2"] / "param_summary.json"
    S = json.load(open(sj)) if sj.exists() else {}

    pos, col_x = {}, {"backbone": 0.0, "neck": 5.6, "detect head": 11.2}
    for sec in ("backbone", "neck", "detect head"):
        sub = L[L.section == sec]
        for k, r in enumerate(sub.itertuples()):
            y = -k * 0.95 if sec != "detect head" else -(len(L[L.section == "neck"]) * 0.95) / 2
            pos[r.idx] = (col_x[sec], y)
    fig, ax = plt.subplots(figsize=(17, 12.5))
    for r in L.itertuples():
        x, y = pos[r.idx]
        txt = f"{r.idx}: {r.type}" + (f" x{r.repeats}" if r.repeats > 1 else "") + f"\n{r.out_shape}  |  {r.params:,} p"
        h = BH if r.type != "Detect" else BH * 3.2
        ax.add_patch(FancyBboxPatch((x, y - h / 2), BW, h, boxstyle="round,pad=0.03", fc=COLOURS.get(r.type, "#ffffff"), ec="#444"))
        ax.text(x + BW / 2, y, txt, ha="center", va="center", fontsize=7.2)
    for r in L.itertuples():
        for j in parse_from(r.from_layers, r.idx):
            if j not in pos:
                continue
            (x0, y0), (x1, y1) = pos[j], pos[r.idx]
            same_col = abs(x0 - x1) < 1e-6
            if same_col and r.idx - j == 1:
                a0, a1 = (x0 + BW / 2, y0 - BH / 2), (x1 + BW / 2, y1 + BH / 2)
                rad = 0.0
            else:
                a0, a1 = (x0 + BW, y0) if x1 > x0 else (x0 + BW, y0), (x1 + (0 if x1 > x0 else BW), y1)
                rad = -0.25 if same_col else 0.0
                if same_col:
                    a0, a1 = (x0 + BW, y0), (x1 + BW, y1)
            ax.add_patch(FancyArrowPatch(a0, a1, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#555", connectionstyle=f"arc3,rad={rad}"))
    ax.text(0 + BW / 2, 1.0, "BACKBONE (feature extractor)", ha="center", fontweight="bold")
    ax.text(5.6 + BW / 2, 1.0, "NECK (PAN-FPN: top-down + bottom-up fusion)", ha="center", fontweight="bold")
    ax.text(11.2 + BW / 2, 1.0, "HEAD (decoupled, anchor-free)", ha="center", fontweight="bold")
    title = (f"{S['cfg']} | input {S['imgsz']}x{S['imgsz']}x3 | nc={S['nc']} | params {S['total_params_nc7']:,} | GFLOPs {S['gflops']}"
             if S else "YOLOv8s architecture (run model_summary.py first for parameter totals)")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.5, 16.2)
    ax.set_ylim(-(max(len(L[L.section == 'backbone']), len(L[L.section == 'neck'])) * 0.95) - 0.2, 1.6)
    ax.axis("off")
    P["q2"].mkdir(parents=True, exist_ok=True)
    fig.savefig(P["q2"] / "architecture.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    mm = ["flowchart TD", '  IN["Input 640x640x3"]']
    for r in L.itertuples():
        mm.append(f'  L{r.idx}["{r.idx}: {r.type}{" x" + str(r.repeats) if r.repeats > 1 else ""}<br/>{r.out_shape}"]')
    for r in L.itertuples():
        srcs = parse_from(r.from_layers, r.idx)
        for j in srcs:
            mm.append(f"  {'IN' if j < 0 else 'L' + str(j)} --> L{r.idx}")
    (P["q2"] / "architecture.mmd").write_text("\n".join(mm))
    print("Wrote", P["q2"] / "architecture.png", "and architecture.mmd")


if __name__ == "__main__":
    main()
