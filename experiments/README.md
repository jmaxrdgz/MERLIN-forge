# MERLIN-forge benchmark results

## Classification

Both benchmarks used frozen ImageNet ResNet-18 features and independently fitted linear probes. Each comparison used the same split, seed and training settings.

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

- **OpenSARShip, 1,497 train and 780 test.** Forge improved accuracy over Gaussian channel enhancement by 1.9 percentage points and macro-F1 by 1.3 points. The paired scene-bootstrap interval for the accuracy difference was +1.1 to +2.9 points across six held-out scenes. Relative to magnitude, the gain was 1.3 accuracy points, with a scene-bootstrap interval of −1.3 to +5.6 points. An always-Cargo classifier would score 71.5% accuracy and 27.8% macro-F1. Only nine test examples were Tug.
- **MSTAR SOC, 2,747 train and 2,425 test.** Forge trailed Gaussian channel enhancement by 6.2 accuracy points and magnitude by 10.0 points. A bootstrap that resamples the ten target-serial groups puts the difference versus Gaussian at −10.4 to −2.0 points. Forge recovered 24.8 accuracy points over MERLIN alone. In a seeded sample of 30 training chips, 66.6% of individual I/Q values had squared values below MERLIN’s internal 0.001 log stabilizer. The vehicle result therefore reflects the chosen spotlight checkpoint and native MSTAR scaling.

## Vessel detection

YOLO11n was fine-tuned for ten epochs at 320×320, using a single seed and 398 training, 24 validation, and 196 test tiles. The held-out set contains 236 labeled boxes across three scenes. MERLIN and YOLO ran on Metal. The remaining preprocessing ran on CPU.

| Method | mAP50 % | mAP50–95 % | Precision % | Recall % | F1 @ 0.30 % | F1 @ 0.50 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Magnitude | **53.2** | **19.2** | **80.9** | 48.4 | 59.54 | **59.56** |
| Gaussian + scatter channels | 48.2 | 15.6 | 74.3 | 47.0 | 57.45 | 54.70 |
| MERLIN only | 50.9 | 17.5 | 69.2 | 49.5 | 57.75 | 50.00 |
| MERLIN-forge | 50.3 | 16.1 | 73.8 | **52.5** | **60.87** | 57.53 |

Forge changed mAP50–95 by +0.5 points versus Gaussian and -3.1 points versus magnitude. Magnitude had the highest mAP50–95 at 19.2%. Forge had the highest reported recall. Ultralytics reports precision and recall at each model’s max-F1 operating point. F1 uses fixed confidence cutoffs of 0.30 and 0.50 with one-to-one matches at IoU ≥ 0.50. Each score pools true positives, false positives, and false negatives across all 196 test tiles. These measurements describe a single training seed and three held-out scenes.

## Scope

The Gaussian and Forge inputs share the two adaptive scatter channels. The first-channel substitution is therefore directly controlled. Suo et al.’s paper uses a learned despeckling network, and the public inference code references a local Swin checkpoint. The measured comparison uses the Gaussian surrogate and these ship and vehicle tasks. [Paper](https://doi.org/10.1109/TGRS.2024.3438560) · [Authors’ inference code](https://github.com/suoyuxi/ChannelEnhancement/blob/57383756ace13da1d655528eb57917f43cd3217e/EnhancementModel/inference.py)

## Reproduce the benchmark

The benchmark used MERLIN-forge commit `67f66a68c59c5dfcf5fa0de47625eb3461410cdb` with the root `config.yaml`. The scripts write per-arm metrics. The classification pipeline also saves fitted probes and held-out predictions. The detection pipeline saves final checkpoints and training curves.

| Method | Classifier or detector input |
| --- | --- |
| `magnitude` | Raw magnitude repeated three times |
| `lowpass_ce` | Gaussian-filtered magnitude (σ = 1 pixel), followed by the two MERLIN-forge adaptive scatter channels |
| `merlin` | MERLIN-despeckled amplitude repeated three times |
| `forge` | MERLIN-despeckled amplitude, followed by the same two adaptive scatter channels |

Suo et al.’s first channel uses learned despeckling. Their public inference script instantiates a Swin Transformer and references a local checkpoint. Their released quantization code contains a single-channel remap and a multi-channel stub. The Gaussian arm uses the Forge scatter mappings as a controlled first-channel substitution. The experiment therefore measures the proposed substitution against a Gaussian control. [Paper](https://doi.org/10.1109/TGRS.2024.3438560) · [Authors’ code](https://github.com/suoyuxi/ChannelEnhancement/tree/57383756ace13da1d655528eb57917f43cd3217e/EnhancementModel)

### Datasets

| Dataset | Data used or access status |
| --- | --- |
| [OpenSARShip 1 and 2](https://opensar.sjtu.edu.cn/) | Cargo, Tanker and Tug from 22 public, dual-polarized Sentinel-1 SLC scenes. Uses original uncalibrated VV chips, bands 3 and 4 as documented in the archive’s `ReadMe.pdf`. The TIFFs store the values in float32. The loader verifies that every used I/Q value is an exact signed 16-bit integer and casts losslessly. |
| [OpenSAR Insight Vessel Detection](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset/tree/e90a0c0465cfb8484b4f356ef968d976054f453a) | Public, MIT-licensed Sentinel-1 benchmark with XML boxes and official scene splits. Uses 512×512 VV SLC tiles, stored as complex64 and checked for lossless conversion to signed int16 I/Q. ESA specifies 16 bits per I and Q for the originating Sentinel-1 SLC product. [Sentinel-1 product documentation](https://documentation.v1.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel1.html) |
| [MSTAR SOC](https://www.sdms.afrl.af.mil/index.php?collection=mstar) | Supplemental X-band vehicle classification. The public mirror’s Phoenix chips store float32 magnitude and phase. The loader reconstructs I/Q from those planes. [Mirror used](https://github.com/omae11/MSTAR-dataset/tree/d99007290d20ce52933c8bf0f8c3f461acd522ee/soc_complex) |
| [SARFish SLC](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish) | Additional native Complex Int16 Sentinel-1 detection and classification candidate. The full paired release is about 3.3 TB compressed and labels are distributed separately. OpenSAR Insight uses some SARFish/xView3 labels, so these benchmarks share provenance. |
| [FAIR-CSAR](https://doi.org/10.57760/sciencedb.radars.00019) | Additional 243 GB Gaofen-3 SLC detection candidate, including aircraft. I/Q storage precision was unverified locally. |
| [CSRSDD](https://doi.org/10.3390/rs17081478) | Gaofen-3 oriented detection candidate with two 16-bit TIFF channels. The published CV-YOLO study describes the dataset as unavailable to the public. |

### Matched protocol

- Classification uses an ImageNet ResNet-18 in evaluation mode as a frozen feature extractor. Each arm fits a standard scaler and an L2 logistic probe on its own training features with fixed `C=0.01`, seed 42 and the same split. Chips are center-cropped or reflection-padded to 128×128 without complex interpolation. MSTAR uses the public 17° train and 15° test split. OpenSARShip uses the first fold of a seeded three-fold stratified scene split and removes training chips with a held-out vessel’s known MMSI.
- Detection uses identical YOLO11n initial weights, settings and seed 42 in all arms. It uses the release’s manually reviewed VV positive selection and a seeded, equal-sized sample of XML-declared empty tiles in each official split. It drops six reviewed tiles with incomplete or invalid boxes. The test evaluation uses the fixed final epoch checkpoint, with the same confidence floor and NMS configuration per arm. Training uses 8-bit RGB PNGs for all four arms, with color, mosaic and spatial augmentation disabled.
- Both pipelines divide the amplitude-based channels by the raw chip magnitude’s 99.9th percentile and clip to [0, 1]. Forge weak/strong channels use the upstream config and already map to [0, 1]. MERLIN uses the published spotlight weights for MSTAR and Sentinel-TOPS weights for both Sentinel-1 benchmarks. The full 128×128 or 512×512 chip is passed once through the fully convolutional network with upstream symmetrization and native input values.

### Reproduce

Run from the repository’s `experiments/` directory. Python 3.11 was used on an Apple M3. Change `mps` to `cpu` on systems without Metal.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-detection.txt
mkdir -p data dependencies
git clone https://github.com/hi-paris/deepdespeckling.git dependencies/deepdespeckling
git -C dependencies/deepdespeckling checkout b9bf54a013933a8dbfa63d5e02cd5b0a7acd7efc
git -C dependencies/deepdespeckling apply "$PWD/deepdespeckling-lazy-gdal.patch"
python -m pip install -e dependencies/deepdespeckling
export PYTHONPATH="$(cd .. && pwd)"
export TQDM_DISABLE=1
export PYTHONDONTWRITEBYTECODE=1
cfg=../config.yaml

git clone https://github.com/omae11/MSTAR-dataset.git data/mstar
git -C data/mstar checkout d99007290d20ce52933c8bf0f8c3f461acd522ee
curl -fL https://download.pytorch.org/models/resnet18-f37072fd.pth -o data/resnet18.pth
python benchmark.py --dataset mstar --data data/mstar/soc_complex \
  --config "$cfg" --weights data/resnet18.pth --out reruns/mstar \
  --merlin-model spotlight --backbone-device mps

# If Python cannot locate its trusted certificates on macOS:
# export SSL_CERT_FILE=/etc/ssl/cert.pem
python fetch_opensar.py data/opensar1 --release 1 --manifest opensar/scene_archives.txt
python fetch_opensar.py data/opensar2 --release 2 --manifest opensar/scene_archives.txt
python benchmark.py --dataset opensar --data data/opensar1 data/opensar2 \
  --config "$cfg" --weights data/resnet18.pth --out reruns/opensar \
  --merlin-model Sentinel-TOPS --backbone-device mps

mkdir -p data/insight
repo=https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset
revision=e90a0c0465cfb8484b4f356ef968d976054f453a
curl -fL "$repo/raw/$revision/yolo_dataset_filtered.csv" -o data/insight/filtered.csv
for split in train val test; do
  for part in labels patches; do
    file="${split}_${part}.tar.zst"
    curl -fL "$repo/resolve/$revision/$split/$file" -o "data/insight/$file"
  done
done
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -o data/yolo11n.pt
PYTORCH_ENABLE_MPS_FALLBACK=1 python detect.py --data data/insight --config "$cfg" \
  --weights data/yolo11n.pt --cache data/yolo-cache --out reruns/insight \
  --merlin-device mps --yolo-device mps --epochs 10 --imgsz 320 --batch 12
```

The dependency patch makes GDAL’s Python binding a lazy import for an unused TIFF loader. MERLIN inference and weights remain upstream. The benchmark reads the SAR imagery with Rasterio or tifffile. [MERLIN paper](https://doi.org/10.1109/TGRS.2021.3128621) · [deepdespeckling source](https://github.com/hi-paris/deepdespeckling/tree/b9bf54a013933a8dbfa63d5e02cd5b0a7acd7efc)

## Held-out inference exports

The repository’s `out/` directory organizes test samples as `<dataset>/<method>/`. The four method folders use the same test IDs within each dataset.

- **MSTAR and OpenSARShip** contain viewable RGB PNGs in `images/`, float32 CHW inputs before ResNet normalization in `inputs/`, and original benchmark labels and predictions in `predictions.csv`. The PNGs are 8-bit previews of the float32 inputs.
- **OpenSAR Insight** contains the exact RGB PNGs used for YOLO evaluation in `images/` and one record per image in `predictions.jsonl`. Each record includes the ground-truth boxes and all exported post-NMS predictions at a confidence floor of 0.001. Boxes use normalized center x, center y, width and height in the original image coordinates.

Run the exporter from the repository root after generating the benchmark results. These paths match the reproduction commands above.

```bash
export PYTHONPATH="$PWD"
export TQDM_DISABLE=1
python experiments/export_test.py --dataset mstar --data experiments/data/mstar/soc_complex \
  --results experiments/reruns
python experiments/export_test.py --dataset opensar --data experiments/data/opensar1 experiments/data/opensar2 \
  --results experiments/reruns
PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/export_test.py --dataset insight \
  --data experiments/data/yolo-cache --results experiments/reruns
```
