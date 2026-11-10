# MERLIN-forge

MERLIN-forge converts a single complex SAR SLC stored as `.npy` (shape `[H, W, 2]`, real/imag) into a 3-channel amplitude representation for downstream detection models. MERLIN uses the complex SLC's real and imaginary components to produce a despeckled amplitude channel. The weak- and strong-scatter enhancement branches share the magnitude computed from the original SLC.

![MERLIN-forge pipeline. Complex SAR feeds MERLIN directly, and raw amplitude feeds the weak- and strong-scatter enhancement branches.](docs/merlin-forge-pipeline.png)

This work is inspired from channel enhancement and combination (Suo et al., 2024) using Merlin (Dalsasso et al., 2022) as the despeckler for distribution and ease of use purposes. 

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone the repository. MERLIN-forge supports Python 3.10 and later. The setup below uses Python 3.11.

```bash
git clone https://github.com/jmaxrdgz/MERLIN-forge.git
cd MERLIN-forge
uv venv --python 3.11
```

The upstream `deepdespeckling` package imports GDAL for a TIFF reader. For the `.npy` workflow, use the included patch to defer that import.

```bash
git clone https://github.com/hi-paris/deepdespeckling.git .venv/deepdespeckling
git -C .venv/deepdespeckling checkout b9bf54a013933a8dbfa63d5e02cd5b0a7acd7efc
git -C .venv/deepdespeckling apply "$PWD/experiments/deepdespeckling-lazy-gdal.patch"
uv pip install '.[viz,bench]' .venv/deepdespeckling
```

If your environment already has the GDAL Python bindings, install directly from GitHub with `uv pip install 'merlin-forge[viz,bench] @ git+https://github.com/jmaxrdgz/MERLIN-forge.git'`. Use the repository's `config.yaml` or supply your own. The distribution name is `merlin-forge` and the Python import is `merlin_forge`.

## Quick start

Render a sample and benchmark a folder of samples with the installed commands.

```bash
uv run --no-sync merlin-forge-viz --config config.yaml --input path/to/sample.npy --out out.png
uv run --no-sync merlin-forge-bench --config config.yaml --input_dir path/to/dir --n 50
```

## Benchmark results

We compared four three-channel inputs derived from the same complex SAR chips. The magnitude and MERLIN-only baselines repeat raw magnitude or MERLIN-despeckled amplitude across all three channels. The Gaussian control and MERLIN-forge share the same weak- and strong-scatter channels computed from raw magnitude. Their first channel uses Gaussian-smoothed magnitude or MERLIN-despeckled amplitude, respectively.

### Classification

Each method used a frozen ImageNet-pretrained ResNet-18 and a separately fitted logistic regression probe, with matched splits and training settings. OpenSARShip used a scene-based split (1,497 train, 780 test).

| Dataset | Method | Accuracy % | Macro-F1 % |
| --- | --- | ---: | ---: |
| OpenSARShip | Magnitude | 69.1 | 33.1 |
| OpenSARShip | Gaussian + scatter channels | 68.5 | 34.5 |
| OpenSARShip | MERLIN only | 69.9 | 35.7 |
| OpenSARShip | MERLIN-forge | **70.4** | **35.8** |

### Vessel detection

Each method fine-tuned YOLO11n for 10 epochs at 320×320 with the same initial weights, splits, and seed. The OpenSAR Insight evaluation used 196 held-out tiles from three scenes with 236 labeled boxes. Precision and recall are reported at each model's max-F1 operating point. F1 @ 0.30 uses a confidence threshold of 0.30 and one-to-one matches at IoU ≥ 0.50.

| Method | mAP50 % | mAP50–95 % | Precision % | Recall % | F1 @ 0.30 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Magnitude | **53.2** | **19.2** | **80.9** | 48.4 | 59.54 |
| Gaussian + scatter channels | 48.2 | 15.6 | 74.3 | 47.0 | 57.45 |
| MERLIN only | 50.9 | 17.5 | 69.2 | 49.5 | 57.75 |
| MERLIN-forge | 50.3 | 16.1 | 73.8 | **52.5** | **60.87** |

See the [experiment documentation](experiments/README.md) for the full protocol, uncertainty estimates, and reproduction commands.

## References

* **MERLIN Despeckling:** Y. Suo, Y. Wu, T. Miao, W. Diao, X. Sun and K. Fu, "Adaptive SAR Image Enhancement for Aircraft Detection via Speckle Suppression and Channel Combination," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-15, 2024, Art no. 5219415. DOI: 10.1109/TGRS.2024.3435881
* **Channel Enhancement:** E. Dalsasso, L. Denis and F. Tupin, "As if by Magic: Self-Supervised Training of Deep Despeckling Networks with MERLIN," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-13, 2022, Art no. 5236213. DOI: 10.1109/TGRS.2021.3128621

