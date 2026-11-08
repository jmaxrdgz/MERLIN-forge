"""Export benchmark test images, ground truth and per-method predictions."""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from benchmark import METHODS, Preprocess, Sample, read_chip
from detect import rgb8


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def method_dirs(out, dataset, exact_inputs=False):
    roots = {method: out / dataset / method for method in METHODS}
    for root in roots.values():
        (root / "images").mkdir(parents=True, exist_ok=True)
        if exact_inputs:
            (root / "inputs").mkdir(exist_ok=True)
    return roots


@torch.inference_mode()
def classification(args):
    dataset = args.dataset
    result = args.results / dataset
    metrics = json.loads((result / "metrics.json").read_text())
    settings = metrics["settings"]
    args.chip_size = settings["chip_size"]
    args.merlin_model = settings["merlin_model"]
    args.device = settings["device"]
    torch.set_num_threads(settings["threads"])
    processor = Preprocess(args)
    if processor.weights_sha256 != metrics["merlin_weights_sha256"]:
        raise ValueError("MERLIN checkpoint differs from the benchmark")
    predictions = read_csv(result / "predictions.csv")
    if len(predictions) != metrics["n_test"]:
        raise ValueError("Prediction count differs from the benchmark test size")
    roots = method_dirs(args.out, dataset, exact_inputs=True)
    archives = {path.name: path for folder in args.data for path in folder.glob("*.zip")}
    exports = {method: [] for method in METHODS}
    for index, row in enumerate(predictions, 1):
        if dataset == "mstar":
            sample = Sample(args.data[0] / "test" / row["id"], row["truth"], row["group"])
        else:
            archive, member = row["id"].split(":", 1)
            sample = Sample(archives[archive], row["truth"], row["group"], member)
        name = f"{index:05d}"
        for method, tensor in processor.all_features(read_chip(sample)).items():
            root = roots[method]
            image = f"images/{name}.png"
            inputs = f"inputs/{name}.npy"
            Image.fromarray(rgb8(tensor)).save(root / image)
            np.save(root / inputs, tensor.cpu().numpy(), allow_pickle=False)
            exports[method].append((row["id"], row["group"], image, inputs, row["truth"], row[method]))
        if index % 100 == 0 or index == len(predictions):
            print(f"{dataset}: exported {index}/{len(predictions)} chips", flush=True)
    for method, rows in exports.items():
        with (roots[method] / "predictions.csv").open("w", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(("id", "group", "image", "input_chw_float32", "truth", "prediction"))
            writer.writerows(rows)


def yolo_boxes(path, with_confidence=False):
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        class_id, x, y, width, height, *scores = line.split()
        box = {"class_id": int(class_id), "xywh_normalized": [float(v) for v in (x, y, width, height)]}
        if with_confidence:
            box["confidence"] = float(scores[0])
        boxes.append(box)
    return boxes


def detection(args):
    os.environ.setdefault("YOLO_CONFIG_DIR", str((args.data[0] / "yolo-config").resolve()))
    from ultralytics import YOLO, settings as yolo_settings

    yolo_settings.update({"sync": False})
    cache = args.data[0]
    result = args.results / "insight"
    settings = json.loads((result / "settings.json").read_text())
    expected = json.loads((result / "metrics.json").read_text())
    test = [row for row in read_csv(result / "split.csv") if row["split"] == "test"]
    if len(test) != settings["subsets"]["test"]["images"]:
        raise ValueError("Test manifest differs from the benchmark test size")
    for method, root in method_dirs(args.out, "insight").items():
        method_cache = cache / method
        image_dir = method_cache / "images" / "test"
        if {p.stem for p in image_dir.glob("*.png")} != {r["patch_id"] for r in test}:
            raise ValueError(f"{method} cached images differ from the test manifest")
        model = YOLO(str(result / f"{method}_last.pt"))
        metrics = model.val(
            data=str((method_cache / "dataset.yaml").resolve()), split="test",
            imgsz=settings["imgsz"], batch=settings["batch"], device=settings["yolo_device"],
            workers=0, plots=False, conf=0.001, save_txt=True, save_conf=True,
            project=str((cache / "export_validation").resolve()), name=method,
        )
        np.testing.assert_allclose(
            [metrics.box.map50, metrics.box.map],
            [expected[method]["map50"], expected[method]["map50_95"]],
            rtol=0, atol=1e-6,
        )
        predictions = Path(metrics.save_dir) / "labels"
        with (root / "predictions.jsonl").open("w") as target:
            for row in test:
                patch = row["patch_id"]
                image = f"images/{patch}.png"
                shutil.copy2(image_dir / f"{patch}.png", root / image)
                target.write(json.dumps({
                    "patch_id": patch,
                    "scene": row["scene"],
                    "image": image,
                    "truth": yolo_boxes(method_cache / "labels" / "test" / f"{patch}.txt"),
                    "predictions": yolo_boxes(predictions / f"{patch}.txt", with_confidence=True),
                }) + "\n")
        print(f"insight/{method}: exported {len(test)} tiles", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mstar", "opensar", "insight"), required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True,
                        help="MSTAR complex root, OpenSARShip archive directories, or YOLO cache")
    parser.add_argument("--results", type=Path, required=True, help="Parent directory of the benchmark results")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    if args.dataset == "insight":
        detection(args)
    else:
        classification(args)
