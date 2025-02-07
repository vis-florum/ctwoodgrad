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
