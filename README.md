# &aleph;<sub>0</sub> games
Implementation of a few &aleph;<sub>0</sub> games, including '5D Chess with Multiverse Time Travel'

## Setup

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

## Quality control

Format code, lint, and typecheck while editing, and before making a commit:

```shell
ruff check; ruff format; pyright
```

Run tests:

```shell
python3 -m pytest
```
