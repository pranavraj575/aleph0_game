# &aleph;<sub>0</sub> games
Implementation of a few &aleph;<sub>0</sub> games, including '5D Chess with Multiverse Time Travel'

## Setup

Requires Python &ge; 3.11

Clone repository, install

```shell
git clone https://github.com/pranavraj575/aleph0_games
cd aleph0_games
pip install -e .
```

Optionally, install with dev tools:

```shell
pip install -e .[dev]
```
If installed with dev, can test installation, and reformat before pushing:

```shell
python -m pytest
ruff check; ruff format; pyright
```
