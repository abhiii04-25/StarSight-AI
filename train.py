"""
train_model.py
──────────────
Generate synthetic constellation patterns, train KNN, save weights.
Run: python train_model.py
"""
import math
from pathlib import Path
import numpy as np
from star_engine import extract_features
from constellation_data import STAR_CATALOG, get_star_positions_pixels

MODEL_PATH = Path(__file__).with_name("constellation_model.npz")
RNG = np.random.default_rng(42)

def augment(points):
    pts = np.asarray(points, dtype=np.float32)
    c = pts.mean(axis=0)
    pts -= c
    ang = RNG.uniform(0, 2*math.pi)
    sc = RNG.uniform(0.82, 1.18)
    jit = RNG.normal(0, 8.0, pts.shape).astype(np.float32)
    trans = RNG.normal(0, 10.0, 2).astype(np.float32)
    rot = np.array([[math.cos(ang), -math.sin(ang)],[math.sin(ang), math.cos(ang)]], dtype=np.float32)
    pts = (pts @ rot.T) * sc + c + jit + trans
    return [tuple(map(float, r)) for r in pts]

def build_dataset(samples=700):
    feats, labs = [], []
    for name in STAR_CATALOG:
        base = get_star_positions_pixels(name, 500, 60)
        for _ in range(samples):
            f = extract_features(augment(base))
            if f is not None:
                feats.append(f)
                labs.append(name)
    return np.asarray(feats, dtype=np.float32), np.asarray(labs)

def evaluate(x, y, mean, std, classes, k=9):
    xn = (x - mean) / std
    idx = RNG.permutation(len(xn))
    split = int(len(idx)*0.8)
    train, test = xn[idx[:split]], xn[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    correct = 0
    for feat, lab in zip(test, y_test):
        d = np.linalg.norm(train - feat, axis=1)
        nn = np.argsort(d)[:k]
        vote = {n:0.0 for n in classes}
        for i in nn: vote[y_train[i]] += 1.0/(float(d[i])+1e-6)
        if max(vote, key=vote.get) == lab: correct += 1
    return correct / max(len(y_test), 1)

def main():
    x, y = build_dataset()
    classes = sorted(set(y.tolist()))
    mean, std = x.mean(axis=0), x.std(axis=0)+1e-6
    acc = evaluate(x, y, mean, std, classes)
    np.savez_compressed(MODEL_PATH, x=x, y=y, classes=np.asarray(classes),
                        mean=mean.astype(np.float32), std=std.astype(np.float32), k=np.asarray([9], dtype=np.int32))
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"📊 Training samples: {len(x)} | Est. Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()