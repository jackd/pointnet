# Pointnet

Based on [this](https://github.com/charlesq34/pointnet).

## Installation

```python
git clone https://github.com/jackd/pointnet.git
```

## Usage

### Single example

```bash
python -m pointnet --gin_file=single/base --gin_params='
    rotate_scheme="pca-xy"
    jitter_positions.stddev=1e-2
    model_dir="~/pointnet_models/single/pca-j1e-2"
'
```
