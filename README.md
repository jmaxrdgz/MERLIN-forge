# MERLIN-forge

MERLIN-forge converts a single complex SAR SLC stored as `.npy` (shape `[H, W, 2]`, real/imag) into a 3-channel amplitude representation for downstream detection models. It computes amplitude once per sample and generates three channels: MERLIN-despeckled amplitude, weak-scatter enhancement, and strong-scatter enhancement.

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
