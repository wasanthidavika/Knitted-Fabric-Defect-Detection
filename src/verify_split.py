
#!/usr/bin/env python3
"""
Verify the frozen split (leakage evidence for Q1).

Checks:
- No group spans two partitions.
- Unique image IDs.
- Duplicate original filenames.
- Split sizes and proportions.
- Per-class image counts per split.

Usage: python src/verify_split.py --config configs/config.yaml
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, paths, read_yolo_labels, SPLITS


def make_image_id(src, raw_root):
    relative_path = (
        Path(src).resolve()
        .relative_to(Path(raw_root).resolve())
        .as_posix()
    )
    return hashlib.sha256(relative_path.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    P = paths(cfg)

    sp = pd.read_csv(P["split_csv"])
    col = "proposed_split"

    # Generate the same unique IDs used by prepare_data.py.
    sp["image_id"] = sp["path"].map(
        lambda p: make_image_id(p, cfg["raw_root"])
    )

    rep = {}

    # Group leakage check.
    g = sp.groupby("group")[col].nunique()

    rep["n_images"] = len(sp)
    rep["n_groups"] = int(g.size)
    rep["groups_spanning_splits"] = int((g > 1).sum())

    # Duplicate filename check. Informational only.
    rep["duplicate_stems"] = int(
        sp["path"].map(lambda p: Path(p).stem).duplicated().sum()
    )

    # Unique image ID check.
    rep["duplicate_image_ids"] = int(
        sp["image_id"].duplicated().sum()
    )

    rep["split_counts"] = sp[col].value_counts().to_dict()

    rep["split_shares"] = (
        sp[col].value_counts(normalize=True).round(4).to_dict()
    )

    rep["dyed_by_split"] = (
        sp.groupby(col)["colour"]
        .apply(lambda s: round(float((s == "dyed").mean()), 4))
        .to_dict()
    )

    # Per-class image counts using unique image IDs.
    lab = read_yolo_labels(cfg["resized_dir"])
    warnings = []

    if len(lab):
        names = cfg["class_names"]

        imgs_per = (
            lab.drop_duplicates(["split", "image_id", "cls"])
            .groupby(["cls", "split"])
            .size()
            .unstack(fill_value=0)
        )

        imgs_per = imgs_per.reindex(
            index=sorted(names),
            columns=SPLITS,
            fill_value=0
        )

        imgs_per.index = [names[i] for i in imgs_per.index]

        rep["images_per_class_per_split"] = imgs_per.to_dict("index")

        for c, row in imgs_per.iterrows():
            for s in ("val", "test"):
                if row[s] < 30:
                    warnings.append(
                        f"{c}: only {row[s]} {s} images. "
                        "Metrics for this class will be noisy."
                    )

    # Group size checks.
    gs = sp.groupby("group").size()

    rep["largest_group_images"] = int(gs.max())
    rep["largest_group_share"] = round(
        float(gs.max() / len(sp)), 4
    )

    if rep["largest_group_share"] > 0.02:
        warnings.append(
            "Largest group holds more than 2% of images. "
            "Check for chained near-duplicate groups."
        )

    # Split proportion checks.
    for s_, target in (
        ("train", 0.70),
        ("val", 0.15),
        ("test", 0.15)
    ):
        got = rep["split_shares"].get(s_, 0)

        if abs(got - target) > 0.03:
            warnings.append(
                f"{s_} share is {got * 100:.1f}% "
                f"(target {target * 100:.0f}%)."
            )

    # Expected split count check.
    exp = cfg.get("expected_split_counts")
    rep["matches_addendum_counts"] = None

    if exp:
        rep["matches_addendum_counts"] = all(
            int(rep["split_counts"].get(k, -1)) == int(v)
            for k, v in exp.items()
        )

        if not rep["matches_addendum_counts"]:
            warnings.append(
                f"Split counts {rep['split_counts']} differ "
                f"from expected counts {exp}."
            )

    rep["warnings"] = warnings

    P["q1"].mkdir(parents=True, exist_ok=True)

    with open(P["q1"] / "split_verification.json", "w") as f:
        json.dump(rep, f, indent=2)

    print(json.dumps(rep, indent=2))

    # Final validation.
    if rep["matches_addendum_counts"] is False:
        print("\nFAIL: Split counts do not match configuration!")
        sys.exit(1)

    if rep["groups_spanning_splits"] > 0:
        print("\nFAIL: Some groups cross multiple splits!")
        sys.exit(1)

    if rep["duplicate_image_ids"] > 0:
        print("\nFAIL: Duplicate image IDs detected!")
        sys.exit(1)

    print("\nOK: No group spans multiple partitions.")
    print("OK: All image IDs are unique.")


if __name__ == "__main__":
    main()
