# pyrbo
Python JIT compiler for near-native performance of low-level arithmetic

## Install
These are generic installation instructions.

### To use, permanently
The quickest way to get started is to install the current release from PyPI:
```
pip3 install --user pyrbo
```

### To use, temporarily
If you prefer to keep .local clean, install to a virtualenv:
```
python3 -m venv venvname
venvname/bin/pip install -U pip
venvname/bin/pip install pyrbo
. venvname/bin/activate
```

## API

<a id="pyrbo"></a>

### pyrbo

<a id="pyrbo.leaf"></a>

### pyrbo.leaf

<a id="pyrbo.leaf.turbo"></a>

###### turbo

```python
def turbo(**kwargs)
```

Accelerate the decorated function or method using Cython.
The `types` kwarg is a dict of local variables (including params) to their numpy type.
(If `types` would be the only kwarg, its contents may be provided to `turbo` directly.)

<a id="pyrbo.model"></a>

### pyrbo.model

