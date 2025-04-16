# pyrbo
Python JIT compiler for near-native performance of low-level arithmetic.

## Install
These are generic installation instructions.

### To use, disposably
Install the current release from PyPI to a virtual environment:
```
python3 -m venv venvname
venvname/bin/pip install -U pip
venvname/bin/pip install pyrbo
. venvname/bin/activate
```

### To use, permanently
```
pip3 install --break-system-packages --user pyrbo
```
See `~/.local/bin` for executables.

### To develop
First install venvpool to get the `motivate` command:
```
pip3 install --break-system-packages --user venvpool
```
Get codebase and install executables:
```
git clone git@github.com:combatopera/pyrbo.git
motivate pyrbo
```
Requirements will be satisfied just in time, using sibling projects with matching .egg-info if any.

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

