# &aleph;<sub>0</sub> games
Implementation of a few &aleph;<sub>0</sub> games, including 'Jenga' and '5D Chess with Multiverse Time Travel'

[//]: <> (python aleph0_game/scripts/play_game.py jenga --args initial_height:5 deterministic:True --save_gif images/sample_jenga_game.gif --duration 1000 --random_players 0 1 --screenshot_dir output/jenga --seed 69 --overwrite --no_render)
[//]: <> (python aleph0_game/scripts/play_game.py chess2d --overwrite --save_gif images/sample_chess2d_game.gif --duration 300 --random_players 0 1 --screenshot_dir output/chess2d --no_render)
[//]: <> (python aleph0_game/scripts/play_game.py chess5d --overwrite --save_gif images/sample_chess5d_game.gif --duration 300 --random_players 0 1 --screenshot_dir output/chess5d --no_render --max_depth 69)
![](https://github.com/pranavraj575/aleph0_game/blob/main/images/sample_jenga_game.gif)

![](https://github.com/pranavraj575/aleph0_game/blob/main/images/sample_chess5d_game.gif)

![](https://github.com/pranavraj575/aleph0_game/blob/main/images/sample_chess2d_game.gif)


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


## Games

### Jenga
![](https://github.com/pranavraj575/aleph0_game/blob/main/images/sample_jenga_game.gif)
### 5D Chess
![](https://github.com/pranavraj575/aleph0_game/blob/main/images/sample_chess5d_game.gif)
### 2D Chess
![](https://github.com/pranavraj575/aleph0_game/blob/main/images/sample_chess2d_game.gif)
