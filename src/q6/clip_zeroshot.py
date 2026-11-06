#!/usr/bin/env python3
"""
CLIP zero-shot defect presence (Q6.a). Image-level, multi-label, NO training on the fabric images.

Pipeline: image (640 px long side, as everywhere) -> tiles -> CLIP image embeddings (cached on disk) ->
cosine similarity to class text embeddings (+ a "normal fabric" prompt) -> softmax over 8 prompts -> per-class score per tile ->
aggregate over tiles (max or top-3 mean) -> image-level score per class.

Everything that is "decided" (prompt set, variant global/tiled, aggregator, per-class thresholds) is decided on VALIDATION images only.
Selection criterion: macro average precision on validation (threshold-free). All combinations tried are logged.

  python src/clip_zeroshot.py --stage select            # val only: embeds, tries every prompt-set x variant x aggregator
  python src/clip_zeroshot.py --stage scores --split val
  python src/clip_zeroshot.py --stage latency
Test-set scoring happens only in q6_final_eval.py (guarded). Set Q6_FAKE=1 to use a fake scorer (pipeline tests only).
"""
import argparse, hashlib, itertools, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config
from image_level import metrics, multi_hot, tune_thresholds
from q3_utils import append_log, git_commit, guard_test, hardware_string, image_table, load_gt, log_path, now

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


# ------------------------------------------------------------------ scorers
class Scorer:
    name = "base"

    def encode_images(self, imgs):            # list of 224x224x3 uint8 RGB -> (n, d) L2-normalised
        raise NotImplementedError

    def encode_texts(self, texts):            # list of str -> (m, d) L2-normalised
        raise NotImplementedError


class HFClipScorer(Scorer):
    """Hugging Face CLIP. NOT run in the authoring sandbox (no torch/transformers): first real run may need small fixes."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None, batch=64):
        import torch
        from transformers import CLIPModel, CLIPTokenizer
        self.torch, self.batch, self.name = torch, batch, model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(model_name).to(self.device, dtype=self.dtype).eval()
        self.tok = CLIPTokenizer.from_pretrained(model_name)

    def _feat(self, out):
        return out if isinstance(out, self.torch.Tensor) else out.pooler_output

    def encode_images(self, imgs):
        t, res = self.torch, []
        for i in range(0, len(imgs), self.batch):
            x = (np.stack(imgs[i:i + self.batch]).astype(np.float32) / 255.0 - MEAN) / STD
            x = t.from_numpy(x).permute(0, 3, 1, 2).to(self.device, dtype=self.dtype)
            with t.no_grad():
                f = self._feat(self.model.get_image_features(pixel_values=x))
            res.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        return np.concatenate(res)

    def encode_texts(self, texts):
        t = self.torch
        tok = self.tok(list(texts), padding=True, return_tensors="pt").to(self.device)
        with t.no_grad():
            f = self._feat(self.model.get_text_features(**tok))
        return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()


class FakeScorer(Scorer):
    """TEST DOUBLE ONLY: deterministic embeddings from simple image statistics / hashed text. Tests the pipeline, says nothing about CLIP."""
    name = "fake"
    D = 16

    def __init__(self):
        self.P = np.random.default_rng(0).normal(size=(9, self.D))

    def encode_images(self, imgs):
        F = []
        for im in imgs:
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            F.append(np.r_[im.reshape(-1, 3).mean(0) / 255, im.reshape(-1, 3).std(0) / 255, np.abs(np.diff(g, axis=1)).mean(), g.std(), g.mean()])
        E = np.asarray(F) @ self.P
        return E / np.linalg.norm(E, axis=1, keepdims=True)

    def encode_texts(self, texts):
        E = np.stack([np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16)).normal(size=self.D) for t in texts])
        return E / np.linalg.norm(E, axis=1, keepdims=True)


def get_scorer(model_name):
    return FakeScorer() if os.environ.get("Q6_FAKE") == "1" else HFClipScorer(model_name)


# ------------------------------------------------------------------ tiling and embeddings
def tile_boxes(H, W, size=224, stride=112):
    xs = list(range(0, max(W - size, 0) + 1, stride))
    ys = list(range(0, max(H - size, 0) + 1, stride))
    if xs[-1] != max(W - size, 0):
        xs.append(max(W - size, 0))
    if ys[-1] != max(H - size, 0):
        ys.append(max(H - size, 0))
    return [(x, y, min(size, W), min(size, H)) for y in ys for x in xs]


def make_tiles(img, variant, size=224, stride=112):
    if variant == "global":
        return [cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)]
    H, W = img.shape[:2]
    out = []
    for x, y, w, h in tile_boxes(H, W, size, stride):
        c = img[y:y + h, x:x + w]
        out.append(c if (w, h) == (size, size) else cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA))
    return out


def load_img(cfg, split, iid):
    im = cv2.imread(str(Path(cfg["resized_dir"]) / "images" / split / f"{iid}.jpg"))
    return im[..., ::-1].copy()


def model_tag(name):
    return name.replace("/", "_")


def embed_split(cfg, scorer, split, variant, allow_test=False, limit=None):
    """{image_id: (n_tiles, d)} for a split; cached in outputs/q6/clip_cache/."""
    guard_test(cfg, split, allow_test)
    cache = Path(cfg["outputs_dir"]) / "q6" / "clip_cache" / f"{model_tag(scorer.name)}_{split}_{variant}.npz"
    ids = sorted(image_table(cfg).query("split == @split")["image_id"])[: limit or None]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = dict(zip(z["ids"].tolist(), np.split(z["emb"], np.cumsum(z["counts"])[:-1])))
        if set(ids) <= set(have):
            return {i: have[i] for i in ids}
    emb = {}
    for k, iid in enumerate(ids):
        emb[iid] = scorer.encode_images(make_tiles(load_img(cfg, split, iid), variant))
        if (k + 1) % 100 == 0:
            print(f"  embedded {k + 1}/{len(ids)} ({split}, {variant})")
    if limit is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ids=np.array(ids), counts=np.array([len(emb[i]) for i in ids]), emb=np.concatenate([emb[i] for i in ids]))
    return emb


# ------------------------------------------------------------------ prompts and scoring
def load_prompts(path=None):
    p = Path(path or Path(__file__).resolve().parents[1] / "configs" / "q6_prompts.yaml")
    return yaml.safe_load(open(p))


def text_matrix(scorer, prompts, template_set):
    """(n_classes + 1, d): prompt-ensembled class embeddings, last row = 'normal fabric'."""
    groups = [prompts["class_descriptions"][c] for c in sorted(prompts["class_descriptions"])] + [prompts["normal"]]
    rows = []
    for descs in groups:
        texts = [t.format(d) for t in prompts["templates"][template_set] for d in descs]
        e = scorer.encode_texts(texts).mean(0)
        rows.append(e / np.linalg.norm(e))
    return np.stack(rows)


def aggregate(tile_probs, agg):
    if agg == "max":
        return tile_probs.max(0)
    k = min(3, len(tile_probs))
    return np.sort(tile_probs, axis=0)[-k:].mean(0)


def image_scores(emb, T, temperature, agg, class_ids):
    rows = {}
    for iid, E in emb.items():
        logit = temperature * E @ T.T
        p = np.exp(logit - logit.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        rows[iid] = aggregate(p[:, :-1], agg)
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(class_ids))


def select_config(cfg, scorer, prompts, limit=None):
    """VALIDATION only: try every prompt set x variant x aggregator, log all, return the best by macro-AP."""
    class_ids = sorted(cfg["class_names"])
    gt, imgs = load_gt(cfg, "val")
    ids = sorted(imgs["image_id"])[: limit or None]
    y = multi_hot(gt, ids, class_ids)
    embs = {v: embed_split(cfg, scorer, "val", v, limit=limit) for v in prompts["variants"]}
    rows = []
    for ts, v, agg in itertools.product(prompts["templates"], prompts["variants"], prompts["aggregators"]):
        T = text_matrix(scorer, prompts, ts)
        s = image_scores(embs[v], T, prompts["temperature"], agg, class_ids).loc[ids]
        thr = tune_thresholds(s, y)
        m = metrics(s, y, thr)
        cid = f"C_{ts}_{v}_{agg}"
        rows.append({"id": cid, "prompt_set": ts, "variant": v, "aggregator": agg, "val_macro_AP": m["macro_ap"], "val_macro_F1": m["macro_f1"]})
        append_log(log_path(cfg), {"id": cid, "date": now(), "family": "clip", "group": "clip_prompt_selection",
                                   "hypothesis": "Tiled scoring and descriptive prompts beat whole-image scoring and bare class names",
                                   "what_changed": f"prompt_set={ts} variant={v} aggregator={agg}", "seed": "n/a",
                                   "hyperparameters": f"model={scorer.name} temperature={prompts['temperature']}", "outcome": "completed",
                                   "val_macro_AP": round(m["macro_ap"], 4), "val_macro_F1": round(m["macro_f1"], 4), "what_was_learned": "",
                                   "run_dir": "outputs/q6", "git_commit": git_commit()})
        print(rows[-1])
    sel = pd.DataFrame(rows).sort_values("val_macro_AP", ascending=False)
    out = Path(cfg["outputs_dir"]) / "q6"
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / "clip_selection.csv", index=False)
    best = sel.iloc[0]
    chosen = {"model": scorer.name, "prompt_set": best.prompt_set, "variant": best.variant, "aggregator": best.aggregator,
              "val_macro_AP": float(best.val_macro_AP), "selected_on": "validation only", "n_combinations_tried": len(sel)}
    json.dump(chosen, open(out / "clip_chosen.json", "w"), indent=2)
    return chosen, sel


def clip_scores(cfg, scorer, prompts, chosen, split, allow_test=False, limit=None):
    class_ids = sorted(cfg["class_names"])
    emb = embed_split(cfg, scorer, split, chosen["variant"], allow_test=allow_test, limit=limit)
    T = text_matrix(scorer, prompts, chosen["prompt_set"])
    return image_scores(emb, T, prompts["temperature"], chosen["aggregator"], class_ids)


def measure_latency(cfg, scorer, prompts, chosen, n=30, split="val"):
    imgs = sorted(image_table(cfg).query("split == @split")["image_id"])[:n]
    arrs = [load_img(cfg, split, i) for i in imgs]
    T = text_matrix(scorer, prompts, chosen["prompt_set"])
    ts = []
    for k, im in enumerate(arrs):
        t0 = time.perf_counter()
        E = scorer.encode_images(make_tiles(im, chosen["variant"]))
        logit = prompts["temperature"] * E @ T.T
        p = np.exp(logit - logit.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        aggregate(p[:, :-1], chosen["aggregator"])
        if k >= 2:
            ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(ts)), "mean_ms": float(np.mean(ts)), "p95_ms": float(np.percentile(ts, 95)), "n": len(ts),
            "hardware": hardware_string(), "scope": "tiling + CLIP image encoder + scoring per image, excludes decoding and text encoding"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", required=True, choices=["select", "scores", "latency"])
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N images")
    a = ap.parse_args()
    cfg = load_config(a.config)
    prompts = load_prompts(a.prompts)
    out = Path(cfg["outputs_dir"]) / "q6"
    scorer = get_scorer(a.model)
    if a.stage == "select":
        chosen, sel = select_config(cfg, scorer, prompts, a.limit)
        print(sel.round(4).to_string(index=False))
        print("\nchosen on VALIDATION:", json.dumps(chosen), "\nRecord ALL rows in the report, not just the winner.")
    else:
        chosen = json.load(open(out / "clip_chosen.json"))
        if a.stage == "scores":
            s = clip_scores(cfg, scorer, prompts, chosen, a.split, limit=a.limit)
            s.to_csv(out / f"clip_scores_{a.split}.csv", index_label="image_id")
            print(f"-> {out / f'clip_scores_{a.split}.csv'}")
        else:
            lat = measure_latency(cfg, scorer, prompts, chosen)
            json.dump(lat, open(out / "clip_latency.json", "w"), indent=2)
            print(json.dumps(lat, indent=2))


if __name__ == "__main__":
    main()
