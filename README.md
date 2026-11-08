# MERLIN-forge

MERLIN-forge converts a single complex SAR SLC stored as `.npy` (shape `[H, W, 2]`, real/imag) into a 3-channel amplitude representation for downstream detection models. MERLIN uses the complex SLC's real and imaginary components to produce a despeckled amplitude channel. The weak- and strong-scatter enhancement branches share the magnitude computed from the original SLC.

![MERLIN-forge pipeline. Complex SAR feeds MERLIN directly, and raw amplitude feeds the weak- and strong-scatter enhancement branches.](docs/merlin-forge-pipeline.png)

This work is inspired from channel enhancement and combination (Suo et al., 2024) using Merlin (Dalsasso et al., 2022) as the despeckler for distribution and ease of use purposes. 

## Setup

1) Create and activate a conda environment (Python >= 3.10):

```bash
conda create -n merlin-forge python=3.10
conda activate merlin-forge
```

2) Install dependencies (PyTorch and deepdespeckling included):

```bash
conda install -y numpy scipy matplotlib scikit-image tqdm nomkl pyyaml
pip install torch torchvision torchaudio

git clone https://github.com/hi-paris/deepdespeckling.git
cd deepdespeckling
pip install -e .
```

## Quick start

- Visualization:

```bash
python -m merlin_forge.viz --config config.yaml --input path/to/sample.npy --out out.png
```
Running this on Mac creates some issues, managed to bypass them adding ```(KMP_DUPLICATE_LIB_OK=TRUE if needed)```before command (not clean but does the job for now).

- Benchmark:

```bash
python -m merlin_forge.bench --config config.yaml --input_dir path/to/dir --n 50
```

## Benchmark results

We compared four three-channel inputs derived from the same complex SAR chips. The magnitude and MERLIN-only baselines repeat raw magnitude or MERLIN-despeckled amplitude across all three channels. The Gaussian control and MERLIN-forge share the same weak- and strong-scatter channels computed from raw magnitude. Their first channel uses Gaussian-smoothed magnitude or MERLIN-despeckled amplitude, respectively.

### Classification

Each method used a frozen ImageNet-pretrained ResNet-18 and a separately fitted logistic regression probe, with matched splits and training settings. OpenSARShip used a scene-based split (1,497 train, 780 test). MSTAR SOC used the standard 17° training and 15° test split (2,747 train, 2,425 test).

| Dataset | Method | Accuracy % | Macro-F1 % |
| --- | --- | ---: | ---: |
| OpenSARShip | Magnitude | 69.1 | 33.1 |
| OpenSARShip | Gaussian + scatter channels | 68.5 | 34.5 |
| OpenSARShip | MERLIN only | 69.9 | 35.7 |
| OpenSARShip | MERLIN-forge | **70.4** | **35.8** |
| MSTAR SOC | Magnitude | **71.3** | **70.6** |
| MSTAR SOC | Gaussian + scatter channels | 67.5 | 66.8 |
| MSTAR SOC | MERLIN only | 36.5 | 33.5 |
| MSTAR SOC | MERLIN-forge | 61.3 | 60.2 |

### Vessel detection

Each method fine-tuned YOLO11n for 10 epochs at 320×320 with the same initial weights, splits, and seed. The OpenSAR Insight evaluation used 196 held-out tiles from three scenes with 236 labeled boxes. Precision and recall are reported at each model's max-F1 operating point. The two F1 columns use fixed confidence thresholds and one-to-one matches at IoU ≥ 0.50.

| Method | mAP50 % | mAP50–95 % | Precision % | Recall % | F1 @ 0.30 % | F1 @ 0.50 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Magnitude | **53.2** | **19.2** | **80.9** | 48.4 | 59.54 | **59.56** |
| Gaussian + scatter channels | 48.2 | 15.6 | 74.3 | 47.0 | 57.45 | 54.70 |
| MERLIN only | 50.9 | 17.5 | 69.2 | 49.5 | 57.75 | 50.00 |
| MERLIN-forge | 50.3 | 16.1 | 73.8 | **52.5** | **60.87** | 57.53 |

See the [experiment documentation](experiments/README.md) for the full protocol, uncertainty estimates, and reproduction commands.

## References

* **MERLIN Despeckling:** Y. Suo, Y. Wu, T. Miao, W. Diao, X. Sun and K. Fu, "Adaptive SAR Image Enhancement for Aircraft Detection via Speckle Suppression and Channel Combination," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-15, 2024, Art no. 5219415. DOI: 10.1109/TGRS.2024.3435881
* **Channel Enhancement:** E. Dalsasso, L. Denis and F. Tupin, "As if by Magic: Self-Supervised Training of Deep Despeckling Networks with MERLIN," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-13, 2022, Art no. 5236213. DOI: 10.1109/TGRS.2021.3128621

