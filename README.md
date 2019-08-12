# Pointnet

Based on [this](https://github.com/charlesq34/pointnet).

## Installation

```python
git clone https://github.com/jackd/pointnet.git
```

### Changing Process name

```bash
sudo apt-get install build-essential libcap-dev
pip install python-prctl
```

## Usage

### Single example

```bash
python -m pointnet --action=train --config_files=single/base --bindings='
    rotate_scheme="pca-xy"
    jitter_positions.stddev=1e-2
    name="pca-xy_j1e-2"
'
```

## Project Maintenance

Format with [`yapf`](https://github.com/google/yapf), `google` style.

```bash
 yapf --style=google -i -r .
```
