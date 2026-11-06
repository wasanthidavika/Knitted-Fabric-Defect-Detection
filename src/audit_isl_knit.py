#!/usr/bin/env python3
"""
ISL-Knit dataset audit  ->  tables, plots, report.json and a filled-in addendum draft.

Usage (Kaggle notebook, dataset attached via "Add Input"):
    !python audit_isl_knit.py --root /kaggle/input/isl-knit-defect --out /kaggle/working/audit

Options:
    --thumb 64        thumbnail size used for hashing/colour (default 64)
    --hash-thr 12     max Hamming distance (of 256 bits) to call two images near-duplicates
    --no-md5          skip exact-duplicate hashing (faster)
    --seed 42

Outputs (in --out):
    report.json, addendum_draft.md, proposed_splits.csv, pairs_sample.csv,
    tables/*.csv, plots/*.png, colour_check.png
Everything in addendum_draft.md marked [EDIT] or [FILL] needs YOUR decision/input.
Nothing here is invented: every number comes from the files on disk.
"""
import argparse, json, re, hashlib, warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}
SPLITS = ["train", "val", "test"]
TBL = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ----------------------------------------------------------------- discovery
def discover(root):
    root = Path(root)
    dirs = [root] + [p for p in root.rglob("*") if p.is_dir() and len(p.relative_to(root).parts) <= 4]
    found = {}
    for d in dirs:
        if d.name.lower() != "images":
            continue
        lab = d.parent / "labels"
        if not lab.is_dir():
            continue
        alias = SPLIT_ALIASES.get(d.parent.name.lower())
        if alias:                                   # <split>/images , <split>/labels
            found[alias] = (d, lab)
        else:                                       # images/<split> , labels/<split>
            for sub in d.iterdir():
                a = SPLIT_ALIASES.get(sub.name.lower())
                if sub.is_dir() and a and (lab / sub.name).is_dir():
                    found[a] = (sub, lab / sub.name)
    return found


def load_names(root):
    root = Path(root)
    for y in list(root.rglob("*.yaml"))[:20] + list(root.rglob("*.yml"))[:20]:
        try:
            import yaml
            d = yaml.safe_load(y.read_text())
            n = d.get("names") if isinstance(d, dict) else None
            if isinstance(n, dict):
                return {int(k): str(v) for k, v in n.items()}
            if isinstance(n, list):
                return {i: str(v) for i, v in enumerate(n)}
        except Exception:
            continue
    for t in list(root.rglob("classes.txt"))[:3]:
        try:
            return {i: l.strip() for i, l in enumerate(t.read_text().splitlines()) if l.strip()}
        except Exception:
            continue
    return {}


# ----------------------------------------------------------------- labels
def parse_label(path):
    rows, issues = [], Counter()
    try:
        lines = Path(path).read_text().strip().splitlines()
    except Exception:
        issues["unreadable"] += 1
        return rows, issues
    for ln in lines:
        t = ln.split()
        if not t:
            continue
        try:
            c = float(t[0])
            if c != int(c):
                issues["non_integer_class"] += 1
                continue
            c = int(c)
            v = [float(x) for x in t[1:]]
        except ValueError:
            issues["malformed"] += 1
            continue
        if len(v) == 4:
            cx, cy, w, h = v
        elif len(v) >= 6 and len(v) % 2 == 0:        # polygon (segmentation) line
            xs, ys = v[0::2], v[1::2]
            issues["polygon_lines"] += 1
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            issues["malformed"] += 1
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            issues["out_of_range"] += 1
        rows.append((c, cx, cy, w, h))
    return rows, issues


# ----------------------------------------------------------------- image probe
def probe(args):
    path, thumb, do_md5 = args
    out = {"path": str(path)}
    try:
        with Image.open(path) as im:
            out["W"], out["H"] = im.size
            try:
                im.draft("RGB", (thumb * 4, thumb * 4))   # fast JPEG decode
            except Exception:
                pass
            t = im.convert("RGB").resize((thumb, thumb), Image.BILINEAR)
        a = np.asarray(t, dtype=np.float32) / 255.0
        g = (a.mean(2) * 255).astype(np.uint8)
        gs = np.asarray(Image.fromarray(g).resize((17, 16), Image.BILINEAR), dtype=np.int16)
        bits = (gs[:, 1:] > gs[:, :-1]).flatten()          # 256-bit dHash
        out["hash"] = np.packbits(bits).tobytes()
        mx, mn = a.max(2), a.min(2)
        out["sat"] = float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())
        out["val"] = float(mx.mean())
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["err"] = str(e)[:80]
    if do_md5:
        try:
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out["md5"] = h.hexdigest()
        except Exception:
            out["md5"] = None
    return out


def group_stem(stem):
    s = re.sub(r"\.rf\.[0-9a-f]+$", "", stem)             # Roboflow export suffix
    s = re.sub(r"_(jpg|jpeg|png)$", "", s, flags=re.I)
    s = re.sub(r"[_\-](aug|augmented|copy|flip|rot)\d*$", "", s, flags=re.I)
    return s


# ----------------------------------------------------------------- near duplicates
def near_dups(packed, thr):
    n = len(packed)
    edges, nn, nn_idx = [], np.full(n, 999), np.full(n, -1)
    for s in range(0, n, 256):
        blk = packed[s:s + 256]
        d = TBL[np.bitwise_xor(blk[:, None, :], packed[None, :, :])].sum(2, dtype=np.int16)
        for i in range(len(blk)):
            d[i, s + i] = 999
        nn[s:s + 256] = d.min(1)
        nn_idx[s:s + 256] = d.argmin(1)
        ii, jj = np.where(d <= thr)
        edges += [(s + a, int(b), int(d[a, b])) for a, b in zip(ii, jj) if s + a < b]
    return edges, nn, nn_idx


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def otsu(v):
    hist, edges = np.histogram(v, bins=64)
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    m = np.cumsum(p * edges[:-1])
    sb = (m[-1] * w - m) ** 2 / (w * (1 - w) + 1e-12)
    return float(edges[:-1][np.argmax(sb)])


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="audit_out")
    ap.add_argument("--thumb", type=int, default=64)
    ap.add_argument("--hash-thr", type=int, default=12)
    ap.add_argument("--no-md5", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True); (out / "plots").mkdir(exist_ok=True)
    rep = {"root": a.root, "seed": a.seed, "hash_thr": a.hash_thr}

    # 1. structure --------------------------------------------------------
    found = discover(a.root)
    if not found:
        raise SystemExit("No <split>/images + <split>/labels structure found under --root. "
                         "Check the path (ls the folder) and adjust discover().")
    print("Splits found:", {k: (str(v[0]), str(v[1])) for k, v in found.items()})
    names = load_names(a.root)
    rep["class_names_file"] = names

    rows = []
    for sp, (idir, ldir) in found.items():
        lab_index = {p.stem: p for p in Path(ldir).rglob("*.txt")}
        for p in sorted(Path(idir).rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                rows.append({"split": sp, "stem": p.stem, "path": str(p),
                             "label_path": str(lab_index.get(p.stem, "")),
                             "label_exists": p.stem in lab_index})
        img_stems = {r["stem"] for r in rows if r["split"] == sp}
        rep.setdefault("orphan_labels", {})[sp] = len(set(lab_index) - img_stems)
    df = pd.DataFrame(rows)
    print(f"{len(df)} images found")

    # 2. labels -----------------------------------------------------------
    boxes, issues = [], Counter()
    for r in df.itertuples():
        if not r.label_exists:
            continue
        bx, iss = parse_label(r.label_path)
        issues.update(iss)
        boxes += [(r.split, r.stem, *b) for b in bx]
    bdf = pd.DataFrame(boxes, columns=["split", "stem", "cls", "cx", "cy", "w", "h"])
    rep["label_issues"] = dict(issues)
    if not names:
        names = {c: f"class_{c}" for c in sorted(bdf["cls"].unique())}
    bdf["cls_name"] = bdf["cls"].map(lambda c: names.get(c, f"class_{c}"))

    # 3. probe images (size, hash, colour, md5) ----------------------------
    print("Probing images (size / hash / colour / md5) ...")
    jobs = [(p, a.thumb, not a.no_md5) for p in df["path"]]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(probe, jobs, chunksize=8))
    pr = pd.DataFrame(res).rename(columns={"path": "path"})
    df = df.merge(pr, on="path", how="left")
    rep["unreadable_images"] = int((~df["ok"].fillna(False).astype(bool)).sum())
    df = df[df["ok"].fillna(False).astype(bool)].reset_index(drop=True)

    cnt = bdf.groupby("stem").agg(n_boxes=("cls", "size"), classes=("cls", lambda s: tuple(sorted(set(s)))))
    df = df.merge(cnt, left_on="stem", right_index=True, how="left")
    df["n_boxes"] = df["n_boxes"].fillna(0).astype(int)
    df["classes"] = df["classes"].apply(lambda x: x if isinstance(x, tuple) else ())
    df["has_defect"] = df["n_boxes"] > 0

    # 4. size / split summary ---------------------------------------------
    summ = df.groupby("split").agg(images=("stem", "size"), with_label_file=("label_exists", "sum"),
                                   no_defect_images=("has_defect", lambda s: int((~s).sum())),
                                   mean_W=("W", "mean"), mean_H=("H", "mean"),
                                   min_W=("W", "min"), max_W=("W", "max"))
    summ["boxes"] = bdf.groupby("split").size()
    summ = summ.reindex([s for s in SPLITS if s in summ.index])
    summ.to_csv(out / "tables/split_summary.csv")
    rep["split_summary"] = summ.round(1).to_dict("index")
    rep["image_size_counts_top5"] = {f"{k[0]}x{k[1]}": int(v) for k, v in
                                     Counter(zip(df.W, df.H)).most_common(5)}

    # 5. class distribution ------------------------------------------------
    inst = bdf.pivot_table(index="cls_name", columns="split", values="cx", aggfunc="size", fill_value=0)
    inst = inst.reindex(columns=[s for s in SPLITS if s in inst.columns])
    inst["total"] = inst.sum(1)
    inst.to_csv(out / "tables/class_instances.csv")
    img_cls = []
    for r in df.itertuples():
        for c in r.classes:
            img_cls.append((r.split, names.get(c, f"class_{c}")))
    icl = pd.DataFrame(img_cls, columns=["split", "cls_name"])
    imgc = icl.pivot_table(index="cls_name", columns="split", values="split", aggfunc="size", fill_value=0)
    imgc = imgc.reindex(columns=[s for s in SPLITS if s in imgc.columns])
    imgc["total"] = imgc.sum(1)
    imgc.to_csv(out / "tables/class_images.csv")
    rep["class_instances"] = inst.to_dict("index")
    rep["class_images"] = imgc.to_dict("index")
    multi = df[df["classes"].apply(len) > 1]
    rep["multi_class_images"] = len(multi)
    rep["multi_class_frac_of_defect_images"] = round(len(multi) / max(df.has_defect.sum(), 1), 4)
    rep["no_defect_frac"] = round(float((~df.has_defect).mean()), 4)
    tot = imgc["total"].sort_values(ascending=False)
    rep["imbalance_ratio_images"] = round(float(tot.iloc[0] / max(tot.iloc[-1], 1)), 2) if len(tot) else None
    # split-vs-class distribution shift
    try:
        from scipy.stats import chi2_contingency
        rep["chi2_class_vs_split_p"] = float(chi2_contingency(imgc.drop(columns="total").values)[1])
    except Exception:
        rep["chi2_class_vs_split_p"] = None

    # 6. box size vs input resolution -------------------------------------
    bdf["area"] = bdf.w * bdf.h
    res_tab = {}
    for R in (224, 384, 512, 640):
        short = np.minimum(bdf.w, bdf.h) * R
        res_tab[R] = {"median_short_side_px": round(float(short.median()), 1),
                      "frac_boxes_short_side_lt_8px": round(float((short < 8).mean()), 4)}
    rep["box_size_vs_resolution"] = res_tab
    rep["box_rel_area_quantiles"] = bdf["area"].quantile([.05, .25, .5, .75, .95]).round(5).to_dict()
    rec = next((R for R in (224, 384, 512, 640) if res_tab[R]["frac_boxes_short_side_lt_8px"] <= 0.10), 640)
    rep["suggested_min_resolution"] = rec
    rep["vit_tokens"] = {R: (R // 16) ** 2 for R in (224, 384, 512)}

    # 7. duplicates & leakage ---------------------------------------------
    packed = np.stack([np.frombuffer(h, dtype=np.uint8) for h in df["hash"]])
    if len(df) > 30000:
        print("WARNING: >30k images, near-duplicate search will be slow")
    edges, nn, nn_idx = near_dups(packed, a.hash_thr)
    df["nn_dist"] = nn
    df["nn_split"] = df["split"].values[nn_idx]
    cross = [(i, j, d) for i, j, d in edges if df.split[i] != df.split[j]]
    within = len(edges) - len(cross)
    rep["near_dup_pairs_total"] = len(edges)
    rep["near_dup_pairs_cross_split"] = len(cross)
    rep["near_dup_pairs_within_split"] = within
    leaky = {}
    for sp in SPLITS:
        idx = set()
        for i, j, d in cross:
            if df.split[i] == sp: idx.add(i)
            if df.split[j] == sp: idx.add(j)
        n_sp = int((df.split == sp).sum())
        leaky[sp] = {"images_with_near_dup_elsewhere": len(idx), "frac": round(len(idx) / max(n_sp, 1), 4)}
    rep["leak_by_split"] = leaky
    if not a.no_md5:
        g = df.dropna(subset=["md5"]).groupby("md5")["split"].agg(list)
        dups = g[g.apply(len) > 1]
        rep["exact_dup_groups"] = int(len(dups))
        rep["exact_dup_groups_cross_split"] = int(sum(len(set(x)) > 1 for x in dups))
    df["gstem"] = df["stem"].map(group_stem)
    sg = df.groupby("gstem")["split"].agg(lambda s: len(set(s)))
    rep["stem_groups_cross_split"] = int((sg > 1).sum())
    rep["stem_groups_multi_image"] = int((df.groupby("gstem").size() > 1).sum())
    # sample pairs for eyeballing
    pd.DataFrame([{"a": df.path[i], "a_split": df.split[i], "b": df.path[j], "b_split": df.split[j], "hamming": d}
                  for i, j, d in sorted(cross, key=lambda x: x[2])[:50]]).to_csv(out / "pairs_sample.csv", index=False)
    # roll / lot metadata?
    meta = [p.name for p in Path(a.root).rglob("*") if p.suffix.lower() in {".csv", ".json", ".xlsx", ".txt", ".md"}
            and p.name.lower() != "classes.txt" and "labels" not in {q.name.lower() for q in p.parents}][:30]
    rep["metadata_files_found"] = meta

    # 8. grey vs dyed ------------------------------------------------------
    kw = df["path"].str.lower().apply(lambda s: "grey" if re.search(r"gr[ae]y", s) else ("dyed" if "dye" in s else None))
    rep["colour_keyword_coverage"] = round(float(kw.notna().mean()), 4)
    thr = otsu(df["sat"].values)
    heur = np.where(df["sat"] > thr, "dyed", "grey")
    df["colour"] = np.where(kw.notna(), kw, heur)
    df["colour_source"] = np.where(kw.notna(), "path_keyword", "saturation_otsu(HEURISTIC)")
    rep["colour_sat_threshold"] = round(thr, 4)
    rep["colour_counts"] = df["colour"].value_counts().to_dict()
    rep["colour_source_counts"] = df["colour_source"].value_counts().to_dict()
    cc = []
    for r in df.itertuples():
        for c in r.classes:
            cc.append((names.get(c, f"class_{c}"), r.colour))
    cc = pd.DataFrame(cc, columns=["cls_name", "colour"])
    ctab = cc.pivot_table(index="cls_name", columns="colour", values="colour", aggfunc="size", fill_value=0)
    ctab["dyed_share"] = (ctab.get("dyed", 0) / ctab.sum(1)).round(3)
    ctab.to_csv(out / "tables/class_by_colour.csv")
    rep["class_by_colour"] = ctab.to_dict("index")
    rep["no_defect_by_colour"] = df[~df.has_defect]["colour"].value_counts().to_dict()
    try:
        from scipy.stats import chi2_contingency
        t = ctab.drop(columns="dyed_share").values
        rep["chi2_class_vs_colour_p"] = float(chi2_contingency(t)[1]) if t.shape[1] > 1 else None
    except Exception:
        rep["chi2_class_vs_colour_p"] = None
    colour_split = pd.crosstab(df["split"], df["colour"])
    colour_split.to_csv(out / "tables/colour_by_split.csv")
    rep["colour_by_split"] = colour_split.to_dict("index")

    # contact sheet to verify colour heuristic by eye
    rng = np.random.default_rng(a.seed)
    tiles = []
    for col in ("grey", "dyed"):
        sub = df[df.colour == col]
        pick = sub.sample(min(6, len(sub)), random_state=a.seed) if len(sub) else sub
        for p in pick["path"]:
            with Image.open(p) as im:
                try: im.draft("RGB", (400, 400))
                except Exception: pass
                tiles.append((col, im.convert("RGB").resize((160, 160))))
    if tiles:
        fig, axs = plt.subplots(2, 6, figsize=(12, 4.4))
        for ax in axs.ravel(): ax.axis("off")
        for ax, (col, im) in zip(axs.ravel(), tiles):
            ax.imshow(im); ax.set_title(col, fontsize=9)
        fig.suptitle("Predicted colour (top: grey, bottom: dyed) - CHECK BY EYE"); fig.tight_layout()
        fig.savefig(out / "colour_check.png", dpi=110); plt.close(fig)

    # 9. proposed group-aware split (only use if leakage was found) --------
    uf = UF(len(df))
    for i, j, d in edges: uf.union(i, j)
    first = {}
    for i, gs in enumerate(df["gstem"]):
        if gs in first: uf.union(i, first[gs])
        else: first[gs] = i
    if not a.no_md5:
        first = {}
        for i, m in enumerate(df["md5"]):
            if m is None or (isinstance(m, float) and np.isnan(m)): continue
            if m in first: uf.union(i, first[m])
            else: first[m] = i
    df["group"] = [uf.find(i) for i in range(len(df))]
    rep["n_groups"] = int(df["group"].nunique())
    freq = Counter(c for cl in df.classes for c in cl)
    df["strat"] = df.classes.apply(lambda cl: min(cl, key=lambda c: freq[c]) if cl else -1)
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgk = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=a.seed)
        df["proposed_split"] = "train"
        for k, (_, te) in enumerate(sgk.split(df, df["strat"], df["group"])):
            df.loc[te, "proposed_split"] = "train" if k < 14 else ("val" if k < 17 else "test")
        df[["path", "split", "proposed_split", "group", "colour", "has_defect"]].to_csv(out / "proposed_splits.csv", index=False)
    except Exception as e:
        print("Could not build proposed split:", e)

    # 10. plots ------------------------------------------------------------
    inst.drop(columns="total").plot(kind="bar", figsize=(9, 4), title="Defect instances per class and split")
    plt.tight_layout(); plt.savefig(out / "plots/class_instances.png", dpi=110); plt.close()
    ctab.drop(columns="dyed_share").plot(kind="bar", stacked=True, figsize=(9, 4), title="Images per class: grey vs dyed")
    plt.tight_layout(); plt.savefig(out / "plots/class_by_colour.png", dpi=110); plt.close()
    plt.figure(figsize=(6, 4)); plt.hist(np.log10(bdf.area.clip(1e-6)), bins=50)
    plt.xlabel("log10(box area / image area)"); plt.title("Relative defect size"); plt.tight_layout()
    plt.savefig(out / "plots/box_area.png", dpi=110); plt.close()
    plt.figure(figsize=(6, 4)); plt.hist(df.sat, bins=50); plt.axvline(thr, color="r")
    plt.xlabel("mean saturation"); plt.title("Colour heuristic (red = Otsu threshold)"); plt.tight_layout()
    plt.savefig(out / "plots/saturation.png", dpi=110); plt.close()
    plt.figure(figsize=(6, 4)); plt.hist(np.clip(df.nn_dist, 0, 80), bins=40); plt.axvline(a.hash_thr, color="r")
    plt.xlabel("Hamming distance to nearest other image (256-bit dHash)"); plt.tight_layout()
    plt.savefig(out / "plots/nn_hash_distance.png", dpi=110); plt.close()

    df.drop(columns=["hash"]).to_csv(out / "tables/image_table.csv", index=False)
    json.dump(rep, open(out / "report.json", "w"), indent=2, default=str)
    (out / "addendum_draft.md").write_text(build_addendum(rep, summ, inst, imgc, ctab, names, a), encoding="utf-8")
    print(f"\nDone. Open {out/'addendum_draft.md'} and resolve every [EDIT]/[FILL].")
    print(json.dumps({k: rep[k] for k in ("near_dup_pairs_cross_split", "stem_groups_cross_split",
                                           "colour_counts", "suggested_min_resolution", "no_defect_frac")}, indent=1, default=str))


# ----------------------------------------------------------------- addendum text
def md_table(df):
    d = df.reset_index()
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in d.values) + "\n"


def build_addendum(rep, summ, inst, imgc, ctab, names, a):
    ncls = len(imgc)
    tot_imgs = int(summ["images"].sum())
    cross = rep["near_dup_pairs_cross_split"]
    leak_txt = (f"{cross} near-duplicate image pairs (dHash Hamming distance <= {rep['hash_thr']}/256) cross split boundaries; "
                f"the share of images with a near-duplicate in another split is "
                + ", ".join(f"{sp}: {v['frac']*100:.1f}%" for sp, v in rep["leak_by_split"].items()) + ". "
                if cross else "No near-duplicate pairs crossing split boundaries were found at this threshold. ")
    if rep.get("exact_dup_groups") is not None:
        leak_txt += f"Exact (MD5) duplicate groups: {rep['exact_dup_groups']} ({rep['exact_dup_groups_cross_split']} spanning splits). "
    leak_txt += (f"{rep['stem_groups_cross_split']} filename-derived groups (augmentation/export suffixes removed) appear in more than one split. ")
    if rep["metadata_files_found"]:
        meta_txt = "Metadata files found: " + ", ".join(rep["metadata_files_found"][:8]) + ". [EDIT: state whether any contain roll/dyeing-lot IDs]"
    else:
        meta_txt = "No metadata file with fabric-roll or dyeing-lot identifiers was found, so roll-level grouping cannot be enforced directly."
    need_resplit = cross > 0 or rep["stem_groups_cross_split"] > 0
    split_decision = (
        "Because related images cross the supplied split boundaries, I will NOT use the supplied split. I will build one fixed, "
        f"group-aware split (groups = connected components of near-duplicate, exact-duplicate and same-filename-stem links; {rep.get('n_groups')} groups), "
        "stratified by rarest defect class, using StratifiedGroupKFold (seed 42, about 70/15/15). It is frozen before any model tuning."
        if need_resplit else
        "No cross-split leakage was detected, so I will keep the supplied train/val/test split as the single fixed split. "
        "Residual risk: images from the same roll may be near-identical in ways a perceptual hash misses; I will state this limitation in the report.")
    p = rep["chi2_class_vs_colour_p"]
    col_shift = f"chi-square test of class vs colour p = {p:.3g}" if p is not None else "chi-square test not computed (scipy missing)"
    colour_src = rep["colour_source_counts"]
    heur_note = ("Grey/dyed labels came from file paths." if "saturation_otsu(HEURISTIC)" not in colour_src else
                 f"Grey/dyed was inferred ({colour_src}). Only {rep['colour_keyword_coverage']*100:.0f}% of paths carry a colour keyword, "
                 f"so the rest uses a mean-saturation threshold ({rep['colour_sat_threshold']}); verified by eye on colour_check.png [EDIT: confirm].")
    rs = rep["box_size_vs_resolution"]
    rec = rep["suggested_min_resolution"]
    multi_frac = rep["multi_class_frac_of_defect_images"]
    task_kind = "multi-label" if multi_frac > 0.05 else "single-label multi-class"
    top_cls = imgc["total"].idxmax(); bot_cls = imgc["total"].idxmin()

    return f"""# Addendum: Dataset Audit and Firm Decisions
**Project:** AI-Based Knitted Fabric Defect Detection and Explainable Quality Inspection
**Dataset:** ISL-Knit (Kaggle), audited with a script included in my submission (audit_isl_knit.py, seed {a.seed}).

## 1. Verified dataset facts
Licence: [FILL: copy exactly from the Kaggle page / paper]. Source paper: [FILL: citation].
{tot_imgs} images in total; labels are YOLO bounding boxes, {int(summ['boxes'].sum())} annotated instances, {ncls} defect classes ({', '.join(imgc.index)}).
Most common image sizes: {rep['image_size_counts_top5']}.

{md_table(summ[['images','with_label_file','no_defect_images','boxes']])}
Images per class (an image counts once per class it contains):

{md_table(imgc)}
Label-quality issues found: {rep['label_issues'] or 'none'}. Unreadable images: {rep['unreadable_images']}. Labels without images: {rep['orphan_labels']}.
Images with no defect annotation: {rep['no_defect_frac']*100:.1f}% of all images. Images with more than one defect class: {rep['multi_class_images']} ({multi_frac*100:.1f}% of defective images).

## 2. Leakage and split strategy
{leak_txt}
{meta_txt}

**Decision:** {split_decision}

## 3. Grey vs dyed fabric
{heur_note} Counts: {rep['colour_counts']}. Class-by-colour image counts:

{md_table(ctab)}
{col_shift}. [EDIT: one sentence interpreting this: is any class almost only grey or only dyed? If yes, state it is a shortcut risk.]
**Decision:** I will (a) stratify the split by class and report colour balance per split, (b) train a colour-only logistic regression on mean RGB/HSV as a shortcut check; if it scores well above the trivial benchmark, colour is a confound, and (c) report error analysis separately for grey and dyed fabric.

## 4. Three findings that constrain modelling
1. **Class imbalance:** the most frequent class ({top_cls}, {int(imgc.loc[top_cls,'total'])} images) is {rep['imbalance_ratio_images']}x the rarest ({bot_cls}, {int(imgc.loc[bot_cls,'total'])}). Accuracy is therefore unsuitable; I will use class-weighted loss/sampling and macro-averaged metrics.
2. **Defect size vs input resolution:** median defect short side is {rs[224]['median_short_side_px']} px at 224x224 and {rs[512]['median_short_side_px']} px at 512x512; the share of defects with a short side under 8 px is {rs[224]['frac_boxes_short_side_lt_8px']*100:.1f}% at 224, {rs[384]['frac_boxes_short_side_lt_8px']*100:.1f}% at 384 and {rs[512]['frac_boxes_short_side_lt_8px']*100:.1f}% at 512. The audit suggests an input of at least {rec} px. [EDIT: confirm after viewing sample images]
3. **Leakage / confounds:** {'cross-split near-duplicates exist, so a group-aware split is mandatory' if need_resplit else 'the supplied split shows no detected cross-split duplicates, but residual roll-level leakage cannot be excluded'}; class-vs-colour association is documented in section 3; {rep['no_defect_frac']*100:.1f}% of images contain no defect, which defines how the "no defect" outcome is handled.

## 5. Firm decisions
- **Task:** image-level {task_kind} defect classification derived from the YOLO boxes (class present / not present; images without boxes are "no defect"). Localisation is provided by Grad-CAM, not by a detector. [EDIT: change if you decide otherwise, and say why]
- **User and decision:** a fabric quality inspector decides whether a fabric image is passed or sent for manual re-inspection, and which defect type to look for.
- **Error costs:** a missed defect (false negative) lets faulty fabric proceed to dyeing/cutting; a false alarm costs only an extra manual check. False negatives are therefore costlier. [EDIT: confirm with your own reasoning]
- **Primary metric:** macro-F1 on the held-out test set (per-class recall also reported); the decision threshold is chosen on validation data only.
- **Classical baseline:** Gabor filter bank + GLCM + LBP features, concatenated and standardised, into an RBF-SVM tuned by grid search over C and gamma [FILL: grid, number of configurations]; plus gradient boosting on the same features as a second classical model. The ANN gets a tuning budget of the same order of magnitude [FILL: number of configurations for each].
- **ANN:** pretrained CNN fine-tuned (ResNet18 first candidate) at input resolution [FILL: 224/384/512 from finding 2] with class-weighted cross-entropy{' (or per-class sigmoid/BCE, because images can contain several defects)' if task_kind == 'multi-label' else ''}.
- **Vision Transformer (transformer treatment):** a ViT-B/16 receives the same images. At 224x224 with 16x16 patches the sequence is (224/16)^2 = {rep['vit_tokens'][224]} patch tokens (+1 [CLS] = {rep['vit_tokens'][224]+1}); at 384x384 it would be {rep['vit_tokens'][384]} tokens (+1 = {rep['vit_tokens'][384]+1}). {'Given finding 2, I will also evaluate the higher resolution because 224 px may erase small defects.' if rec > 224 else ''} Attention cost grows quadratically with token count, which is the main practical limit.
- **Question 6:** a pretrained vision-language model is applied to the same held-out test images with the same macro-F1; prompts are selected on validation data only; model/version, prompts, cost and latency are documented. [EDIT: name the model]
- **Explainability:** Grad-CAM on the CNN, validated by occluding/perturbing highlighted regions and checking the prediction falls; incorrect predictions analysed.

## 6. Question 2.b prediction (made before any results)
[EDIT: write your own: which family do you expect to win on macro-F1, and why? Use the audit: dataset size, texture-dominated images, defect size, imbalance.]

## 7. How my work differs from the dataset paper
[EDIT: after reading the paper, state its task, baselines and metrics, and how your task, split, baseline and evaluation differ.]
"""


if __name__ == "__main__":
    main()
