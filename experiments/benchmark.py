"""Matched SAR preprocessing ablation with an ImageNet ResNet-18 linear probe."""
import argparse
import csv
import hashlib
import json
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18

from merlin_forge.config import load_config
from merlin_forge.forge import MerlinForge

METHODS = ("magnitude", "lowpass_ce", "merlin", "forge")
SHIP_CLASSES = ("Cargo", "Tanker", "Tug")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    group: str
    member: str
    vessel: str = ""

    @property
    def id(self):
        return f"{self.path.name}:{self.member}"


@lru_cache(maxsize=32)
def scene_zip(path):
    return ZipFile(path)


def read_chip(sample):
    payload = scene_zip(sample.path).read(sample.member)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile(payload) as memory, memory.open() as raster:
            if raster.count != 4 or raster.dtypes != ("float32",) * 4:
                raise ValueError(f"{sample.id}: expected the original four-band SLC TIFF")
            iq = raster.read([3, 4])
    # OpenSARShip ReadMe §1.2.2 documents bands 3–4 as real and imaginary VV.
    if not np.all((iq >= -32768) & (iq <= 32767) & (iq == np.rint(iq))):
        raise ValueError(f"{sample.id}: I/Q values cannot be losslessly represented as int16")
    return np.moveaxis(iq.astype(np.int16), 0, -1)


def vessel_ids(archive):
    name = next((name for name in archive.namelist() if name.replace("\\", "/").rsplit("/", 1)[-1] == "Ship.xml"), None)
    if name is None:
        return {}
    ids = {}
    # Several release-2 files have invalid free-text fields. Parse only the
    # numeric coordinate and identity tags within each published ship record.
    for record in re.findall(rb"<ship>(.*?)</ship>", archive.read(name), re.DOTALL):
        x = re.search(rb"<Center_x>\s*(\d+)\s*</Center_x>", record)
        y = re.search(rb"<Center_y>\s*(\d+)\s*</Center_y>", record)
        mmsi = re.search(rb"<MMSI>\s*(\d{9})\s*</MMSI>", record)
        if x and y and mmsi and mmsi[1] != b"000000000":
            ids[f"_x{x[1].decode()}_y{y[1].decode()}"] = mmsi[1].decode()
    return ids


def inventory(args):
    samples = []
    seen = set()
    for folder in args.data:
        for path in sorted(folder.glob("*.zip")):
            archive = scene_zip(path)
            vessels = vessel_ids(archive)
            for name in sorted(archive.namelist()):
                normalized = name.replace("\\", "/")
                basename = normalized.rsplit("/", 1)[-1]
                label = basename.split("_", 1)[0]
                if label not in SHIP_CLASSES or not basename.lower().endswith((".tif", ".tiff")):
                    continue
                # Raw patch names are selected after inspecting the published archive.
                if args.ship_path_filter not in normalized:
                    continue
                key = (path.stem, basename)
                if key not in seen:
                    coordinates = Path(basename).stem[len(label):]
                    samples.append(Sample(path, label, path.stem, name, vessels.get(coordinates, "")))
                    seen.add(key)
    if len({s.group for s in samples}) < 5:
        raise ValueError(f"Need at least 5 distinct scenes, found {len({s.group for s in samples})}")
    labels = [s.label for s in samples]
    groups = [s.group for s in samples]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed)
    train, test = next(splitter.split(np.zeros(len(samples)), labels, groups))
    test_vessels = {samples[i].vessel for i in test if samples[i].vessel}
    return [samples[i] for i in train if samples[i].vessel not in test_vessels], [samples[i] for i in test]


def subsample(samples, cap, seed):
    if cap is None:
        return samples
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted({sample.label for sample in samples}):
        matching = [sample for sample in samples if sample.label == label]
        indices = rng.permutation(len(matching))[:cap]
        selected.extend(matching[i] for i in sorted(indices))
    return selected


class Preprocess:
    def __init__(self, args):
        self.size = args.chip_size
        cfg = load_config(args.config)
        cfg["merlin"].update(
            model_name=args.merlin_model, device=args.device,
            patch_size=self.size, stride_size=self.size - 2,
        )
        self.forge = MerlinForge(cfg)
        # Keep checkpoint loading outside the warm-path measurement.
        denoiser = self.forge.merlin.adapter._get_model()
        denoiser.load_model = lru_cache(maxsize=1)(denoiser.load_model)
        self.weights_sha256 = hashlib.sha256(Path(denoiser.weights_path).read_bytes()).hexdigest()

    def prepare(self, chip):
        height, width = chip.shape[:2]
        top, left = max(0, (height - self.size) // 2), max(0, (width - self.size) // 2)
        chip = chip[top:top + self.size, left:left + self.size]
        dh, dw = self.size - chip.shape[0], self.size - chip.shape[1]
        if dh or dw:
            chip = np.pad(chip, ((dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2), (0, 0)), mode="reflect")
        return self.forge.preprocess(torch.from_numpy(np.ascontiguousarray(chip)), {})

    def features(self, chip, method):
        x, ctx = self.prepare(chip)
        amp = ctx["amplitude"]
        scale = torch.quantile(amp.flatten(), 0.999).clamp_min(1e-12)
        if method == "magnitude":
            channels = amp.unsqueeze(0).expand(3, -1, -1)
        elif method == "lowpass_ce":
            blurred = gaussian_filter(amp.cpu().numpy(), sigma=1.0, mode="reflect")
            weak_strong, _ = self.forge.adaptive(x, ctx)
            channels = torch.stack((torch.from_numpy(blurred).to(amp.device), *weak_strong))
        elif method == "merlin":
            denoised, _ = self.forge.merlin(x, ctx)
            channels = denoised.unsqueeze(0).expand(3, -1, -1)
        else:
            channels, _ = self.forge(x, ctx)
        if method in ("lowpass_ce", "forge"):
            return torch.cat(((channels[:1] / scale).clamp(0, 1), channels[1:]), dim=0)
        return (channels / scale).clamp(0, 1)

    def all_features(self, chip):
        # Reuse the two shared stages during offline extraction.
        x, ctx = self.prepare(chip)
        amp = ctx["amplitude"]
        scale = torch.quantile(amp.flatten(), 0.999).clamp_min(1e-12)
        weak_strong, _ = self.forge.adaptive(x, ctx)
        denoised, _ = self.forge.merlin(x, ctx)
        blurred = torch.from_numpy(gaussian_filter(amp.cpu().numpy(), sigma=1.0, mode="reflect")).to(amp.device)
        def normalize(channel):
            return (channel / scale).clamp(0, 1)
        return dict(zip(METHODS, (
            normalize(amp).expand(3, -1, -1),
            torch.stack((normalize(blurred), *weak_strong)),
            normalize(denoised).expand(3, -1, -1),
            torch.stack((normalize(denoised), *weak_strong)),
        )))


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def paired_ci(pred, ref, truth, groups, seed):
    rng = np.random.default_rng(seed)
    diff = (pred == truth).astype(float) - (ref == truth)
    units = [diff[groups == group] for group in np.unique(groups)]
    totals = np.array([unit.sum() for unit in units])
    counts = np.array([len(unit) for unit in units])
    draws = rng.integers(0, len(units), size=(3000, len(units)))
    estimates = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return np.percentile(estimates, [2.5, 97.5]).tolist()


@torch.inference_mode()
def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train, test = inventory(args)
    train, test = [subsample(rows, args.cap_per_class, args.seed) for rows in (train, test)]
    classes = sorted({s.label for s in train})
    if not train or {s.label for s in test} != set(classes):
        raise ValueError("Every class must occur in both splits")
    args.out.mkdir(parents=True, exist_ok=True)
    scene_files = sorted({sample.path for sample in train + test})
    (args.out / "scene_archives.txt").write_text("".join(
        f"{p.parent.name}/{p.name}\n" for p in scene_files
    ))
    print("Known vessel identities:", sum(bool(s.vessel) for s in train), sum(bool(s.vessel) for s in test), flush=True)
    samples = train + test
    print("Split:", len(train), len(test), Counter(s.label for s in train), Counter(s.label for s in test), flush=True)
    with (args.out / "split.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("split", "id", "label", "group"))
        for split, rows in (("train", train), ("test", test)):
            writer.writerows((split, s.id, s.label, s.group) for s in rows)

    pre = Preprocess(args)
    net = resnet18()
    net.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    net.fc = torch.nn.Identity()
    net = net.to(args.backbone_device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=args.backbone_device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=args.backbone_device)[None, :, None, None]

    def embed(batch):
        batch = F.interpolate(batch.to(args.backbone_device), size=(args.model_size,) * 2,
                              mode="bilinear", align_corners=False, antialias=True)
        return net((batch - mean) / std).cpu().numpy()

    features = {method: [] for method in METHODS}
    for start in range(0, len(samples), args.batch):
        rows = [pre.all_features(read_chip(s)) for s in samples[start:start + args.batch]]
        for method in METHODS:
            features[method].append(embed(torch.stack([row[method] for row in rows])))
        if start % (args.batch * 5) == 0:
            print(f"Embedded {min(start + args.batch, len(samples))}/{len(samples)}", flush=True)

    truth = np.array([s.label for s in test])
    y_train = [s.label for s in train]
    predictions, heads = {}, {}
    for method in METHODS:
        matrix = np.concatenate(features[method])
        train_x, test_x = matrix[:len(train)], matrix[len(train):]
        scaler = StandardScaler().fit(train_x)
        head = LogisticRegression(C=0.01, max_iter=2000, random_state=args.seed)
        head.fit(scaler.transform(train_x), y_train)
        predictions[method] = head.predict(scaler.transform(test_x))
        heads[method] = (scaler, head)
        np.savez(args.out / f"{method}_probe.npz", classes=head.classes_, coef=head.coef_,
                 intercept=head.intercept_, mean=scaler.mean_, scale=scaler.scale_)
    with (args.out / "predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("id", "group", "truth", *METHODS))
        writer.writerows((s.id, s.group, s.label, *(predictions[m][i] for m in METHODS)) for i, s in enumerate(test))

    latency_samples = [read_chip(test[i]) for i in np.random.default_rng(args.seed).choice(
        len(test), size=min(len(test), args.latency_n), replace=False)]
    times = {method: [] for method in METHODS}
    for method in METHODS:
        scaler, head = heads[method]
        for chip in latency_samples[:5]:
            prepared = pre.features(chip, method)
            reference = pre.all_features(chip)[method]
            torch.testing.assert_close(prepared, reference)
            head.predict(scaler.transform(embed(prepared.unsqueeze(0))))
    for i, chip in enumerate(latency_samples):
        for method in METHODS[i % len(METHODS):] + METHODS[:i % len(METHODS)]:
            synchronize(args.device)
            start = time.perf_counter_ns()
            prepared = pre.features(chip, method)
            synchronize(args.device)
            middle = time.perf_counter_ns()
            scaler, head = heads[method]
            head.predict(scaler.transform(embed(prepared.unsqueeze(0))))
            stop = time.perf_counter_ns()
            times[method].append(((middle - start) / 1e6, (stop - start) / 1e6))

    groups = np.array([s.group for s in test])
    metrics = {}
    for method, pred in predictions.items():
        q50, q95 = np.percentile(times[method], [50, 95], axis=0)
        metrics[method] = dict(
            accuracy=accuracy_score(truth, pred), macro_f1=f1_score(truth, pred, average="macro"),
            confusion_matrix=confusion_matrix(truth, pred, labels=classes).tolist(),
            accuracy_delta_vs_magnitude_ci95=paired_ci(pred, predictions["magnitude"], truth, groups, args.seed),
            preprocessing_p50_ms=q50[0], preprocessing_p95_ms=q95[0],
            end_to_end_p50_ms=q50[1], end_to_end_p95_ms=q95[1],
        )
    result = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "data"},
                  merlin_weights_sha256=pre.weights_sha256,
                  backbone_weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  classes=classes, n_train=len(train), n_test=len(test), latency_n=len(latency_samples),
                  train_counts=dict(Counter(y_train)), test_counts=dict(Counter(truth)),
                  test_groups=sorted({s.group for s in test}), metrics=metrics,
                  forge_accuracy_delta_vs_lowpass_ce_ci95=paired_ci(
                      predictions["forge"], predictions["lowpass_ce"], truth, groups, args.seed))
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out / "latency.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("method", "sample_index", "preprocessing_ms", "end_to_end_ms"))
        for method in METHODS:
            writer.writerows((method, i, *times_) for i, times_ in enumerate(times[method]))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("opensar",), required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--merlin-model", choices=("spotlight", "stripmap", "Sentinel-TOPS"), required=True)
    parser.add_argument("--ship-path-filter", default="Patch/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chip-size", type=int, default=128, choices=(32, 64, 128, 256, 512))
    parser.add_argument("--model-size", type=int, default=128)
    parser.add_argument("--cap-per-class", type=int)
    parser.add_argument("--latency-n", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone-device", default="cpu")
    run(parser.parse_args())
