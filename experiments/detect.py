"""Matched YOLO11n runs on OpenSAR Insight original VV SLC patches."""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tifffile
import torch
import yaml
import zstandard
from PIL import Image

from benchmark import METHODS, Preprocess, synchronize

SPLITS = ("train", "val", "test")
SIZE = 512


@contextmanager
def open_archive(path):
    with path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            yield archive


def yolo_labels(root):
    rows = []
    for box in root.findall(".//Ship/BoundingBox"):
        # XML coordinates are one-based, inclusive pixel indices.
        left, top = (int(box.findtext(key)) - 1 for key in ("Left", "Top"))
        right, bottom = (int(box.findtext(key)) for key in ("Right", "Bottom"))
        left, top, right, bottom = np.clip([left, top, right, bottom], 0, SIZE)
        if right <= left or bottom <= top:
            raise ValueError("Empty ship bounding box")
        rows.append(f"0 {(right + left) / (2 * SIZE):.8f} {(top + bottom) / (2 * SIZE):.8f}"
                    f" {(right - left) / SIZE:.8f} {(bottom - top) / SIZE:.8f}")
    return rows


def sample_inventory(data, seed):
    with (data / "filtered.csv").open(newline="") as file:
        reviewed = {s: set() for s in SPLITS}
        for row in csv.DictReader(file):
            if match := re.fullmatch(r"DB_OPENSAR_VV_VD_(\d+)\.png", row["file_name"]):
                reviewed[row["split"]].add(match[1])
    rng = np.random.default_rng(seed)
    inventory = {}
    scenes = set()
    for split in SPLITS:
        positive, background, seen = {}, {}, set()
        with open_archive(data / f"{split}_labels.tar.zst") as archive:
            for member in archive:
                match = re.fullmatch(r"DB_OPENSAR_VD_(\d+)\.xml", Path(member.name).name)
                if match is None:
                    continue
                patch_id = match[1]
                root = ET.fromstring(archive.extractfile(member).read())
                scene = root.findtext(".//SARProduct")
                declared = int(root.findtext(".//Number_of_ships"))
                if patch_id in reviewed[split]:
                    seen.add(patch_id)
                    try:
                        rows = yolo_labels(root)
                    except (TypeError, ValueError):
                        continue
                    if len(rows) == declared and rows:
                        positive[patch_id] = (scene, rows)
                elif declared == 0:
                    background[patch_id] = (scene, [])
        if seen != reviewed[split] or not positive:
            raise ValueError(f"Reviewed VV annotations missing for {split}")
        print(f"{split}: excluded {len(seen) - len(positive)} reviewed patches with incomplete boxes", flush=True)
        negatives = rng.choice(sorted(background), size=min(len(positive), len(background)), replace=False)
        inventory[split] = positive | {str(key): background[str(key)] for key in negatives}
        split_scenes = {value[0] for value in inventory[split].values()}
        if scenes & split_scenes:
            raise ValueError("A source scene crosses the official splits")
        scenes.update(split_scenes)
    return inventory


def read_iq(source):
    complex_chip = tifffile.imread(source)
    if complex_chip.shape != (SIZE, SIZE) or not np.iscomplexobj(complex_chip):
        raise ValueError(f"Expected a {SIZE}×{SIZE} complex SLC chip")
    iq = np.stack((complex_chip.real, complex_chip.imag), axis=-1)
    if not np.all((iq >= -32768) & (iq <= 32767) & (iq == np.rint(iq))):
        raise ValueError("I/Q values cannot be losslessly represented as int16")
    return iq.astype(np.int16)


def images(data, split, ids):
    with open_archive(data / f"{split}_patches.tar.zst") as archive:
        for member in archive:
            match = re.fullmatch(r"DB_OPENSAR_VD_(\d+)_SLC_VV\.tiff", Path(member.name).name)
            if match and match[1] in ids:
                yield match[1], read_iq(io.BytesIO(archive.extractfile(member).read()))


def rgb8(tensor):
    return np.rint(tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def validate_cache(args, inventory, pre, allow_new=False):
    dataset_files = ["filtered.csv"] + [
        f"{split}_{kind}.tar.zst" for split in SPLITS for kind in ("labels", "patches")
    ]
    source_root = Path(__file__).resolve().parent
    source_files = [source_root / "benchmark.py", source_root / "detect.py"]
    source_files += sorted((source_root.parent / "merlin_forge").rglob("*.py"))
    fingerprints = {}
    for path in [args.data / name for name in dataset_files] + source_files:
        with path.open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        key = path.name if path.parent == args.data else str(path.relative_to(source_root.parent))
        fingerprints[key] = digest.hexdigest()
    metadata = dict(
        config=pre.forge.cfg, chip_size=pre.size, merlin_device=args.merlin_device,
        merlin_weights_sha256=pre.weights_sha256, fingerprints=fingerprints,
        inventory={split: {key: dict(scene=scene, boxes=boxes)
                           for key, (scene, boxes) in rows.items()}
                   for split, rows in inventory.items()},
    )
    manifest = args.cache / "provenance.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError("Cache provenance differs from this run; use a fresh --cache directory")
    elif not allow_new or any((args.cache / method).exists() for method in METHODS):
        raise ValueError("Cache has no provenance; use a fresh --cache directory and prepare it")
    return metadata


@torch.inference_mode()
def prepare(args, inventory, pre):
    metadata = validate_cache(args, inventory, pre, allow_new=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        root = args.cache / method
        for split in SPLITS:
            for directory in ("images", "labels"):
                (root / directory / split).mkdir(parents=True, exist_ok=True)
        (root / "dataset.yaml").write_text(yaml.safe_dump(
            dict(path=str(root.resolve()), names={0: "ship"}, **{s: f"images/{s}" for s in SPLITS})))
    with (args.out / "split.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("split", "patch_id", "scene", "ships"))
        for split, rows in inventory.items():
            writer.writerows((split, key, scene, len(boxes)) for key, (scene, boxes) in sorted(rows.items()))

    for split, rows in inventory.items():
        found = set()
        for i, (key, iq) in enumerate(images(args.data, split, rows), 1):
            found.add(key)
            if not all((args.cache / m / "images" / split / f"{key}.png").exists() and
                       (args.cache / m / "labels" / split / f"{key}.txt").exists() for m in METHODS):
                for method, image in pre.all_features(iq).items():
                    if not torch.isfinite(image).all():
                        raise ValueError(f"{method} produced non-finite pixels for {key}")
                    image_path = args.cache / method / "images" / split / f"{key}.png"
                    image_tmp = image_path.with_suffix(".tmp.png")
                    Image.fromarray(rgb8(image)).save(image_tmp)
                    image_tmp.replace(image_path)
                    label_path = args.cache / method / "labels" / split / f"{key}.txt"
                    label_tmp = label_path.with_suffix(".tmp.txt")
                    label_tmp.write_text("\n".join(rows[key][1]))
                    label_tmp.replace(label_path)
            if i % 20 == 0 or i == len(rows):
                print(f"Prepared {split}: {i}/{len(rows)}", flush=True)
        if found != set(rows):
            raise ValueError(f"Missing {split} SLC patches: {sorted(set(rows) - found)}")

    manifest_tmp = args.cache / "provenance.tmp.json"
    manifest_tmp.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    manifest_tmp.replace(args.cache / "provenance.json")


def fit(args, inventory, pre):
    provenance = validate_cache(args, inventory, pre)
    os.environ.setdefault("YOLO_CONFIG_DIR", str((args.cache / "yolo-config").resolve()))
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    metadata = dict(
        seed=args.seed, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        torch_threads=args.threads, merlin_device=args.merlin_device, yolo_device=args.yolo_device,
        merlin_weights_sha256=pre.weights_sha256,
        cache_provenance=provenance,
        yolo_initial_weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        subsets={split: dict(images=len(rows), boxes=sum(len(boxes) for _, boxes in rows.values()),
                            scenes=sorted({scene for scene, _ in rows.values()}))
                 for split, rows in inventory.items()},
    )
    (args.out / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metrics = {}
    models = {}
    for method in METHODS:
        for split, rows in inventory.items():
            for folder, suffix in (("images", ".png"), ("labels", ".txt")):
                files = (args.cache / method / folder / split).glob(f"*{suffix}")
                if {p.stem for p in files} != set(rows):
                    raise ValueError(f"Incomplete or stale {method}/{folder}/{split} cache")
        dataset = str((args.cache / method / "dataset.yaml").resolve())
        train_dir = args.cache / "runs" / method
        checkpoint = train_dir / "weights" / "last.pt"
        if not args.eval_only:
            YOLO(str(args.weights)).train(
                data=dataset, project=str(train_dir.parent.resolve()), name=method, exist_ok=True,
                epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, workers=0,
                device=args.yolo_device, seed=args.seed, deterministic=True, amp=False,
                optimizer="AdamW", lr0=0.001, patience=0, plots=False,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, mosaic=0.0, translate=0.0,
                scale=0.0, fliplr=0.0, val=True,
            )
        model = YOLO(str(checkpoint))
        stats = model.val(data=dataset, split="test", imgsz=args.imgsz, batch=args.batch,
                          device=args.yolo_device, workers=0, plots=False, conf=0.001,
                          project=str((args.cache / "eval").resolve()), name=method, exist_ok=True)
        metrics[method] = dict(map50=float(stats.box.map50), map50_95=float(stats.box.map),
                               precision=float(stats.box.mp), recall=float(stats.box.mr))
        models[method] = model
        if not args.eval_only:
            shutil.copyfile(checkpoint, args.out / f"{method}_last.pt")
            shutil.copyfile(train_dir / "results.csv", args.out / f"{method}_training.csv")
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.set_num_threads(args.threads)
    latency(args, inventory, pre, models, metrics)


@torch.inference_mode()
def latency(args, inventory, pre, models, metrics):
    keys = np.random.default_rng(args.seed).choice(sorted(inventory["test"]),
              size=min(args.latency_n, len(inventory["test"])), replace=False)
    chips = list(images(args.data, "test", set(keys)))

    def predict(model, prepared):
        return model.predict(source=np.ascontiguousarray(rgb8(prepared)[..., ::-1]),
                             imgsz=args.imgsz, device=args.yolo_device,
                             conf=0.001, verbose=False)

    reference = pre.all_features(chips[0][1])
    for method, model in models.items():
        for i, (_, iq) in enumerate(chips[:3]):
            prepared = pre.features(iq, method)
            if i == 0:
                torch.testing.assert_close(prepared, reference[method])
            predict(model, prepared)
    latency_rows = []
    for i, (key, iq) in enumerate(chips):
        for method in METHODS[i % len(METHODS):] + METHODS[:i % len(METHODS)]:
            synchronize(args.yolo_device)
            synchronize(args.merlin_device)
            start = time.perf_counter_ns()
            prepared = pre.features(iq, method)
            synchronize(args.merlin_device)
            middle = time.perf_counter_ns()
            predict(models[method], prepared)
            synchronize(args.yolo_device)
            end = time.perf_counter_ns()
            latency_rows.append((method, key, (middle - start) / 1e6, (end - start) / 1e6))
    with (args.out / "latency.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("method", "patch_id", "preprocessing_ms", "end_to_end_ms"))
        writer.writerows(latency_rows)
    for method in METHODS:
        values = [row[2:] for row in latency_rows if row[0] == method]
        q50, q95 = np.percentile(values, [50, 95], axis=0)
        metrics[method].update(preprocessing_p50_ms=q50[0], preprocessing_p95_ms=q95[0],
                               end_to_end_p50_ms=q50[1], end_to_end_p95_ms=q95[1])
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "config", "weights", "cache", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--yolo-device", default="cpu")
    parser.add_argument("--merlin-device", default="cpu")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--latency-n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--prepare-only", action="store_true")
    phase.add_argument("--eval-only", action="store_true")
    phase.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    args.device = "cpu"
    args.merlin_model = "Sentinel-TOPS"
    args.chip_size = SIZE
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    pre = Preprocess(args)
    pre.forge.merlin.adapter._get_model().device = args.merlin_device
    inventory = sample_inventory(args.data, args.seed)
    print({s: dict(Counter(bool(boxes) for _, boxes in rows.values())) for s, rows in inventory.items()}, flush=True)
    if not (args.eval_only or args.train_only):
        prepare(args, inventory, pre)
    if not args.prepare_only:
        fit(args, inventory, pre)
