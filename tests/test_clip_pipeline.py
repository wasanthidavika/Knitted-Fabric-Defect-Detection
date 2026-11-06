"""Pipeline tests with the FAKE scorer (no torch/transformers). They check geometry, aggregation, prompt ensembling, neighbour voting and
shapes, NOT CLIP quality. Run: python tests/test_clip_pipeline.py"""
import os, sys
from pathlib import Path
import numpy as np
os.environ["Q6_FAKE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clip_zeroshot as C
from clip_label_audit import crop_with_context, knn_vote

# tiling covers the whole image, tiles are 224x224, borders included
H, W = 480, 640
boxes = C.tile_boxes(H, W)
cov = np.zeros((H, W), bool)
for x, y, w, h in boxes:
    cov[y:y + h, x:x + w] = True
assert cov.all() and len(boxes) == 5 * 4, len(boxes)
img = (np.random.default_rng(0).random((H, W, 3)) * 255).astype(np.uint8)
assert len(C.make_tiles(img, "global")) == 1 and C.make_tiles(img, "global")[0].shape == (224, 224, 3)
t = C.make_tiles(img, "tiled"); assert len(t) == 20 and all(x.shape == (224, 224, 3) for x in t)
small = img[:150, :300]; ts = C.make_tiles(small, "tiled"); assert all(x.shape == (224, 224, 3) for x in ts)       # images smaller than a tile

# aggregation
P = np.array([[.1, .9], [.5, .1], [.3, .2], [.0, .0]])
assert np.allclose(C.aggregate(P, "max"), [.5, .9]) and np.allclose(C.aggregate(P, "top3mean"), [(.5 + .3 + .1) / 3, (.9 + .2 + .1) / 3])

# prompt ensembling and scoring shapes (fake scorer)
sc = C.FakeScorer(); prompts = C.load_prompts()
T = C.text_matrix(sc, prompts, "P2_defect_photo")
assert T.shape == (8, sc.D) and np.allclose(np.linalg.norm(T, axis=1), 1)
emb = {f"i{k}": sc.encode_images(C.make_tiles(img, "tiled")) for k in range(3)}
S = C.image_scores(emb, T, 100.0, "max", list(range(7)))
assert S.shape == (3, 7) and ((S >= 0) & (S <= 1)).all().all()
# probabilities over 8 prompts sum to 1 per tile, so image score per class cannot exceed 1
assert (S.sum(axis=1) <= 7).all()

# neighbour vote: two well separated clusters; mislabelled point is the one disagreeing with its neighbours; same-image neighbours ignored
rng = np.random.default_rng(1)
E = np.concatenate([rng.normal(0, .05, (30, 8)) + np.eye(8)[0], rng.normal(0, .05, (30, 8)) + np.eye(8)[1]])
E /= np.linalg.norm(E, axis=1, keepdims=True)
labels = np.array([0] * 30 + [1] * 30); labels[5] = 1                        # point 5 is mislabelled
images = np.arange(60)
V = knn_vote(E, labels, images, [0, 1], k=7)
assert V[5].argmax() == 0 and V[5, 1] < 0.2 and V[40].argmax() == 1
V2 = knn_vote(E, labels, np.zeros(60, int), [0, 1], k=7)                     # all crops from one image: no neighbours -> uniform vote
assert np.allclose(V2, 0.5)
c = crop_with_context(img, [0.4, 0.4, 0.45, 0.45]); assert c.shape == (224, 224, 3)
print("clip pipeline / label-audit unit tests: all passed")
