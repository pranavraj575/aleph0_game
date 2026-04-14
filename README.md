# &aleph;<sub>0</sub> games
Implementation of a few &aleph;<sub>0</sub> games, including 'Jenga' and '5D Chess with Multiverse Time Travel'

![](https://github.com/pranavraj575/aleph0_game/blob/main/images/sample_jenga_game.gif)


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

## Play games

To play a game (either against other humans or against a random player), use `aleph0_game/scripts/play_game.py`

Usage example:

```shell
python aleph0_game/scripts/play_game.py jenga --args players:3 --random_players 1 2
```