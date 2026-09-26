# ctwoodgrad

Derivative based analysis of tomographic images of wood

## Installation

### Pip installation

You can install `ctwoodgrad` directly from GitHub:

```bash
pip install git+https://github.com/vis-florum/ctwoodgrad
```

### Anaconda Installation

If you are using Anaconda, it's best to install `ctwoodgrad` in a dedicated environment:

```bash
conda create -n ctwoodgrad_env python=3.9 numpy pyevtk pip
conda activate ctwoodgrad_env
pip install git+https://github.com/vis-florum/ctwoodgrad
```

If you already have an environment, just install dependencies:

```bash
conda install numpy pyevtk
pip install git+https://github.com/vis-florum/ctwoodgrad
```

### Local development install

If you are developing `ctwoodgrad`, clone the repo and install it locally:

```bash
git clone https://github.com/vis-florum/ctwoodgrad
cd ctwoodgrad
pip install .
```

## Usage

```python
from ctwoodgrad import segmentAir, getFCS
```

### Memory-bounded fibre fields

`getFibreTensorForVoxelSize(image, voxel_spacing_mm)` applies the same
spacing-calibrated gradient and tensor scales used by the CT-Geo pipeline.
For long volumes, use `staged_fibre_tensor_chunks` to compute overlapping
chunks and consume only their non-overlapping direction cores:

```python
from ctwoodgrad import staged_fibre_tensor_chunks

with staged_fibre_tensor_chunks(
    density, voxel_spacing_mm=0.5, axis=0,
    chunk_slices=256, workers=4, memory_budget_gb=6,
    cache_dir="/tmp",
) as chunks:
    for chunk in chunks:
        use(chunk.start, chunk.stop, chunk.radial,
            chunk.tangential, chunk.longitudinal)
```

The arrays retain the input's spatial axis order, followed by a three-component
vector axis. `axis` selects the lengthwise input axis. Each process reads a
shared scan from a temporary memory-mapped file and writes its core to disk.
The temporary files are removed when the context exits. `memory_budget_gb` is
an estimated budget **per worker**, not a hard OS limit; an oversized chunk is
shortened automatically. DIPlib's two Gaussian supports determine the overlap.
