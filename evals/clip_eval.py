"""Offline accuracy eval for the CLIP dog check-in, used to calibrate the threshold.

It mimics the real flow: a dog is enrolled with a few reference photos (their mean
embedding), and at pickup one new photo is compared with the dog(s) expected.

Two views of the same data:

* identify (``routes.py``): the right dog ranks first among ``--route-size`` dogs,
  and a photo of a dog NOT on the route stays below the threshold.
* verify (bidding check-in): the photo is compared with ONE expected dog; we report
  how many photos of that dog pass (genuine accept) and how many photos of other
  dogs pass (impostor accept) at each threshold.

Data: a folder with one sub-folder of JPGs per dog. DogFaceNet_224resized is made of
aligned face crops (optimistic); DogFaceNet_large is raw web photos (closer to a
phone photo). Re-run on the real demo dogs before trusting any number.

Usage:
    python -I evals/clip_eval.py DATA_DIR --out RESULT.json [--refs 3] [--flip]
Set CLIP_MODEL_PATH to keep the downloaded ONNX model out of ~/.cache.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from cantrack_ai.embeddings import ClipEmbedder

THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0.50 .. 0.99
TARGET_REJECT = 0.90  # pass line fixed before running: unknown dogs rejected
TARGET_TOP1 = 0.90  # pass line fixed before running: right dog ranks first
TARGET_FAR = 0.05  # verify: at most 5% of other-dog photos may pass


def load_dogs(root: Path, max_dogs: int, per_dog: int, refs: int, seed: int) -> dict[str, list[Path]]:
    """Pick up to ``max_dogs`` dogs that have at least ``refs + 1`` photos."""
    rng = random.Random(seed)
    dogs = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        photos = sorted(d.glob("*.jpg"))
        if len(photos) >= refs + 1:
            dogs[d.name] = photos
    names = sorted(dogs)
    rng.shuffle(names)
    return {n: dogs[n][:per_dog] for n in names[:max_dogs]}


def _mirror(data: bytes) -> bytes:
    buf = io.BytesIO()
    ImageOps.mirror(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")).save(buf, "JPEG")
    return buf.getvalue()


def embed_all(dogs: dict[str, list[Path]], cache: Path, flip: bool) -> dict[str, np.ndarray]:
    """Embed every photo once (cached as .npz keyed by file path and flip)."""
    store = {}
    if cache.exists():
        with np.load(cache) as z:
            store = {k: z[k] for k in z.files}
    emb = ClipEmbedder()
    emb.ensure_model()
    out = {}
    for name, photos in dogs.items():
        vecs = []
        for p in photos:
            key = f"{'flip:' if flip else ''}{name}/{p.name}"
            if key not in store:
                data = p.read_bytes()
                v = np.asarray(emb.embed(data), dtype=np.float32)
                if flip:
                    v = unit(v) + unit(np.asarray(emb.embed(_mirror(data)), dtype=np.float32))
                store[key] = v
            vecs.append(store[key])
        out[name] = np.stack(vecs)
    np.savez(cache, **store)
    return out


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def simulate(vecs: dict[str, np.ndarray], refs_n: int, route_size: int, seed: int) -> dict:
    """Return identify records plus verify genuine/impostor similarities."""
    rng = random.Random(seed)
    names = sorted(vecs)
    refs = {n: unit(unit(vecs[n][:refs_n]).mean(axis=0)) for n in names}
    known, unknown, genuine, impostor = [], [], [], []
    for n in names:
        for q in unit(vecs[n][refs_n:]):
            others = [m for m in names if m != n]
            route = [n] + rng.sample(others, route_size - 1)
            sims = np.array([float(q @ refs[m]) for m in route])
            known.append((int(np.argmax(sims)) == 0, float(sims.max())))
            unknown.append(float(max(float(q @ refs[m]) for m in rng.sample(others, route_size))))
            genuine.append(float(q @ refs[n]))
            impostor.extend(float(q @ refs[m]) for m in rng.sample(others, 5))
    return {"known": known, "unknown": unknown, "genuine": genuine, "impostor": impostor}


def summarise(sim: dict) -> dict:
    known, unknown = sim["known"], sim["unknown"]
    gen, imp = sim["genuine"], sim["impostor"]
    rows = []
    for t in THRESHOLDS:
        auto = [(c, s) for c, s in known if s >= t]
        rows.append(
            {
                "threshold": t,
                "unknown_rejected": sum(s < t for s in unknown) / len(unknown),
                "known_auto_confirm_rate": len(auto) / len(known),
                "auto_confirm_precision": (sum(c for c, _ in auto) / len(auto)) if auto else None,
                "genuine_accept": sum(s >= t for s in gen) / len(gen),
                "impostor_accept": sum(s >= t for s in imp) / len(imp),
            }
        )
    return {
        "n_known": len(known),
        "n_impostor_pairs": len(imp),
        "top1": sum(c for c, _ in known) / len(known),
        "rows": rows,
        "chosen_identify": next((r for r in rows if r["unknown_rejected"] >= TARGET_REJECT), None),
        "chosen_verify": next((r for r in rows if r["impostor_accept"] <= TARGET_FAR), None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-dogs", type=int, default=300)
    ap.add_argument("--per-dog", type=int, default=6)
    ap.add_argument("--refs", type=int, default=2, help="reference photos per dog (app allows 1-5)")
    ap.add_argument("--flip", action="store_true", help="average each embedding with its mirror image")
    ap.add_argument("--route-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", type=Path, default=Path("clip_eval_cache.npz"))
    a = ap.parse_args()

    dogs = load_dogs(a.data_dir, a.max_dogs, a.per_dog, a.refs, a.seed)
    vecs = embed_all(dogs, a.cache, a.flip)
    res = summarise(simulate(vecs, a.refs, a.route_size, a.seed))
    res["config"] = {k: str(v) for k, v in vars(a).items()}
    a.out.write_text(json.dumps(res, indent=1))

    print(f"dogs={len(dogs)} refs={a.refs} flip={a.flip} known_queries={res['n_known']} "
          f"top1={res['top1']:.3f} (pass >= {TARGET_TOP1})")
    for r in res["rows"]:
        if r["threshold"] in (0.7, 0.8, 0.85, 0.9):
            print(f"T={r['threshold']:.2f} unknown_rejected={r['unknown_rejected']:.3f} "
                  f"genuine_accept={r['genuine_accept']:.3f} impostor_accept={r['impostor_accept']:.3f}")
    for k in ("chosen_identify", "chosen_verify"):
        c = res[k]
        print(k, None if c is None else
              f"T={c['threshold']} unknown_rejected={c['unknown_rejected']:.3f} "
              f"genuine_accept={c['genuine_accept']:.3f} impostor_accept={c['impostor_accept']:.3f}")


if __name__ == "__main__":
    main()
