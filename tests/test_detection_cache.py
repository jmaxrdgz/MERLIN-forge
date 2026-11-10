import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def cache_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    detect = importlib.import_module("detect")
    data = tmp_path / "data"
    data.mkdir()
    for name in ["filtered.csv"] + [
        f"{split}_{kind}.tar.zst" for split in detect.SPLITS for kind in ("labels", "patches")
    ]:
        (data / name).write_bytes(name.encode())
    args = SimpleNamespace(data=data, cache=tmp_path / "cache", out=tmp_path / "out",
                           merlin_device="cpu", eval_only=False)
    inventory = {split: {split: (f"scene-{split}", ["0 0.5 0.5 0.1 0.1"])}
                 for split in detect.SPLITS}
    pre = SimpleNamespace(
        forge=SimpleNamespace(cfg={"adaptive_quantization": {"alpha_weak": 3}}),
        size=512, weights_sha256="checkpoint-a",
        all_features=lambda iq: {method: torch.ones(3, 2, 2) / 2 for method in detect.METHODS},
    )
    monkeypatch.setattr(detect, "images", lambda data, split, rows: ((key, None) for key in rows))
    return detect, args, inventory, pre


def test_cache_preparation_and_reuse(cache_run):
    detect, args, inventory, pre = cache_run
    detect.prepare(args, inventory, pre)
    manifest = json.loads((args.cache / "provenance.json").read_text())
    assert manifest["merlin_weights_sha256"] == "checkpoint-a"

    def unexpected_preprocessing(iq):
        pytest.fail("Matching cache should reuse prepared images")

    pre.all_features = unexpected_preprocessing
    detect.prepare(args, inventory, pre)
    assert detect.validate_cache(args, inventory, pre) == manifest


@pytest.mark.parametrize("change", ["config", "weights", "data", "labels", "inventory", "code"])
@pytest.mark.parametrize("phase", ["prepare", "train", "eval"])
def test_cache_changes_are_rejected_before_outputs_are_overwritten(cache_run, change, phase):
    detect, args, inventory, pre = cache_run
    detect.prepare(args, inventory, pre)
    if change == "config":
        pre.forge.cfg["adaptive_quantization"]["alpha_weak"] = 4
    elif change == "weights":
        pre.weights_sha256 = "checkpoint-b"
    elif change in ("data", "labels"):
        name = "train_patches.tar.zst" if change == "data" else "train_labels.tar.zst"
        (args.data / name).write_bytes(b"different release")
    elif change == "inventory":
        inventory["test"]["test"][1].append("0 0.1 0.1 0.1 0.1")
    else:
        manifest_path = args.cache / "provenance.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["fingerprints"]["experiments/benchmark.py"] = "old-code"
        manifest_path.write_text(json.dumps(manifest))
    settings = args.out / "settings.json"
    settings.write_text("previous results")
    split = (args.out / "split.csv").read_bytes()
    args.eval_only = phase == "eval"
    with pytest.raises(ValueError, match="Cache provenance differs"):
        if phase == "prepare":
            detect.prepare(args, inventory, pre)
        else:
            detect.fit(args, inventory, pre)
    assert settings.read_text() == "previous results"
    assert (args.out / "split.csv").read_bytes() == split


def test_legacy_cache_is_rejected(cache_run):
    detect, args, inventory, pre = cache_run
    (args.cache / "magnitude").mkdir(parents=True)
    with pytest.raises(ValueError, match="Cache has no provenance"):
        detect.prepare(args, inventory, pre)


def test_interrupted_preparation_cannot_be_used_for_training(cache_run, monkeypatch):
    detect, args, inventory, pre = cache_run
    monkeypatch.setattr(detect, "images", lambda *args: iter(()))
    with pytest.raises(ValueError, match="Missing train SLC patches"):
        detect.prepare(args, inventory, pre)
    assert not (args.cache / "provenance.json").exists()
    with pytest.raises(ValueError, match="Cache has no provenance"):
        detect.fit(args, inventory, pre)
