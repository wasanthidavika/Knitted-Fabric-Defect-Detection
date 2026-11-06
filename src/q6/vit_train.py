#!/usr/bin/env python3
"""
Fine-tune a pretrained Vision Transformer (ViT-B/16) for IMAGE-LEVEL multi-label defect presence (Q6.a, transformer treatment).

Why image level: a plain ViT classifier has no boxes, so it is compared with YOLO / the SVM / CLIP by image-level macro-F1
(see image_level.py), not by mAP@0.5.
Input: image (640 px long side, as everywhere) -> letterbox to S x S (aspect ratio kept, grey padding) -> 16x16 patches.
  S=384: 24x24 = 576 patch tokens, sequence length 577 with [CLS].  S=224: 196 tokens, 197 with [CLS].
Training uses TRAIN only; the epoch is chosen by validation macro-average-precision; per-class thresholds are tuned on VALIDATION later.
All runs are appended to logs/experiments.csv (family "vit").

  python src/vit_train.py --name V00                                     # full fine-tuning at 384 (initial ViT)
  python src/vit_train.py --name V01 --model vit_base_patch16_224.augreg_in21k_ft_in1k     # 224 ablation: 197 tokens
  python src/vit_train.py --name V02 --freeze-backbone                   # linear probe: is fine-tuning needed?
  python src/vit_train.py --name V03 --lr 1e-5                           # lower lr
  python src/vit_train.py --facts                                        # tokens / parameters only

NOT run in the authoring sandbox (no torch/timm): the first real run may need small fixes (send me the traceback).
Check the exact pretrained tag with:  python -c "import timm; print(timm.list_models('vit_base_patch16_384*', pretrained=True))"
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, set_seed
from image_level import metrics, multi_hot, tune_thresholds
from q3_utils import append_log, git_commit, guard_test, hardware_string, load_gt, log_path, now
from vit_facts import letterbox_square, n_tokens, vit_params


def split_data(cfg, split):
    gt, imgs = load_gt(cfg, split)
    ids = sorted(imgs["image_id"])
    return ids, multi_hot(gt, ids, sorted(cfg["class_names"]))


def build_loader(cfg, split, ids, y, size, mean, std, train, batch, workers):
    import torch

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(ids)

        def __getitem__(self, i):
            im = cv2.imread(str(Path(cfg["resized_dir"]) / "images" / split / f"{ids[i]}.jpg"))[..., ::-1]
            im = letterbox_square(np.ascontiguousarray(im), size)
            if train:
                if np.random.rand() < 0.5:
                    im = im[:, ::-1]
                if np.random.rand() < 0.5:
                    im = im[::-1]
                im = np.clip(im.astype(np.float32) * np.random.uniform(0.9, 1.1) + np.random.uniform(-10, 10), 0, 255)
            x = (np.ascontiguousarray(im).astype(np.float32) / 255.0 - mean) / std
            return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(y.iloc[i].values.astype(np.float32))

    return torch.utils.data.DataLoader(DS(), batch_size=batch, shuffle=train, num_workers=workers, drop_last=False)


def create_model(name, n_classes, pretrained=True):
    import timm
    m = timm.create_model(name, pretrained=pretrained, num_classes=n_classes)
    dc = timm.data.resolve_data_config({}, model=m)
    return m, int(dc["input_size"][1]), np.array(dc["mean"], np.float32), np.array(dc["std"], np.float32)


def predict_scores(model, loader, ids, class_ids, device):
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        for x, _ in loader:
            out.append(torch.sigmoid(model(x.to(device)).float()).cpu().numpy())
    return pd.DataFrame(np.concatenate(out), index=ids, columns=class_ids)


def score_split(cfg, weights, split, allow_test=False, batch=16, device=None, latency_images=30):
    """Image-level scores (DataFrame image_id x class) and batch-1 latency for a saved checkpoint."""
    import torch
    guard_test(cfg, split, allow_test)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    class_ids = ck["classes"]
    model, size, mean, std = create_model(ck["model"], len(class_ids), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    ids, y = split_data(cfg, split)
    scores = predict_scores(model, build_loader(cfg, split, ids, y, size, ck["mean"], ck["std"], False, batch, 0), ids, class_ids, device)
    ts = []
    with torch.no_grad():
        for k, iid in enumerate(ids[: latency_images + 3]):
            im = cv2.imread(str(Path(cfg["resized_dir"]) / "images" / split / f"{iid}.jpg"))[..., ::-1]
            x = torch.from_numpy(((letterbox_square(np.ascontiguousarray(im), size).astype(np.float32) / 255 - ck["mean"]) / ck["std"])).permute(2, 0, 1)[None]
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                model(x.to(device))
            if device == "cuda":
                torch.cuda.synchronize()
            if k >= 3:
                ts.append((time.perf_counter() - t0) * 1000)
    lat = {"median_ms": float(np.median(ts)), "mean_ms": float(np.mean(ts)), "p95_ms": float(np.percentile(ts, 95)), "hardware": hardware_string(),
           "scope": "letterbox + normalise + forward, batch 1, excludes decoding", "input": size}
    return scores, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="V00")
    ap.add_argument("--model", default="vit_base_patch16_384.augreg_in21k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--pos-weight", action="store_true", help="sqrt(neg/pos) class weights in the BCE loss")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--facts", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    class_ids = sorted(cfg["class_names"])
    if a.facts:
        for img in (384, 224):
            print(f"S={img}: {n_tokens(img, 16)} patch tokens, sequence length {n_tokens(img, 16) + 1}, params(7 classes) {vit_params(img, classes=len(class_ids)):,}")
        return
    import torch
    seed = int(cfg["seed"] if a.seed is None else a.seed)
    set_seed(seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, size, mean, std = create_model(a.model, len(class_ids))
    n_params = sum(p.numel() for p in model.parameters())
    if a.freeze_backbone:
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("head")
    model.to(device)
    ids_tr, y_tr = split_data(cfg, "train")
    ids_va, y_va = split_data(cfg, "val")
    dl_tr = build_loader(cfg, "train", ids_tr, y_tr, size, mean, std, True, a.batch, a.workers)
    dl_va = build_loader(cfg, "val", ids_va, y_va, size, mean, std, False, 16, a.workers)
    pw = None
    if a.pos_weight:
        pos = y_tr.values.sum(0).clip(1)
        pw = torch.tensor(np.sqrt((len(y_tr) - pos) / pos), dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    steps, warm = a.epochs * len(dl_tr), int(a.warmup_epochs * len(dl_tr))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    out = Path(cfg["outputs_dir"]) / "q6" / "vit" / a.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"{a.model}: input {size}x{size}, {n_tokens(size, 16)} patch tokens (+CLS), {n_params:,} parameters, device {device}")
    hist, best, bad, t0 = [], (-1.0, None, 0), 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tl = []
        for x, yb in dl_tr:
            x, yb = x.to(device), yb.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(model(x).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl.append(float(loss))
        sc = predict_scores(model, dl_va, ids_va, class_ids, device)
        m = metrics(sc, y_va, tune_thresholds(sc, y_va))
        hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_macro_AP": m["macro_ap"], "val_macro_F1": m["macro_f1"], "lr": sched.get_last_lr()[0]})
        print(hist[-1])
        if m["macro_ap"] > best[0]:
            best, bad = (m["macro_ap"], ep, m["macro_f1"]), 0
            torch.save({"state_dict": model.state_dict(), "model": a.model, "classes": class_ids, "mean": mean, "std": std, "size": size}, out / "best.pt")
            sc.to_csv(out / "val_scores.csv", index_label="image_id")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop")
                break
    minutes = (time.time() - t0) / 60
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    flags = ";".join(f for f, c in (("still_improving_at_end", best[1] == len(hist)), ("early_stopped", len(hist) < a.epochs)) if c)
    append_log(log_path(cfg), {"id": a.name, "date": now(), "family": "vit", "group": "vit", "hypothesis": "Fine-tuned ViT-B/16 detects defect presence at image level",
                               "what_changed": f"model={a.model} freeze_backbone={a.freeze_backbone} lr={a.lr} pos_weight={a.pos_weight}", "seed": seed, "epochs": len(hist),
                               "hyperparameters": f"batch={a.batch} lr={a.lr} wd={a.wd} warmup={a.warmup_epochs} input={size} params={n_params}",
                               "val_macro_AP": round(best[0], 4), "val_macro_F1": round(best[2], 4), "best_epoch": best[1], "train_time_min": round(minutes, 2),
                               "outcome": "completed", "auto_flags": flags, "what_was_learned": "", "run_dir": str(out), "git_commit": git_commit()})
    print(f"best epoch {best[1]}: val macro-AP {best[0]:.4f}, macro-F1 {best[2]:.4f}; weights {out / 'best.pt'}; {minutes:.1f} min")


if __name__ == "__main__":
    main()
