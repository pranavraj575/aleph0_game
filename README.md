# &aleph;<sub>0</sub> games
Implementation of a few &aleph;<sub>0</sub> games, including 'Jenga' and '[5D Chess with Multiverse Time Travel](https://www.5dchesswithmultiversetimetravel.com/)'

[//]: <> (python aleph0_game/scripts/play_game.py jenga --args initial_height:5 deterministic:True --save_gif images/sample_jenga_game.gif --duration 1000 --random_players 0 1 --screenshot_dir output/jenga --seed 69 --overwrite --no_render)
[//]: <> (python aleph0_game/scripts/play_game.py jenga --args initial_height:18 deterministic:True --save_gif images/sample_large_jenga_game.gif --duration 420 --random_players 0 1 --screenshot_dir output/jenga_large --seed 420 --overwrite --no_render)
[//]: <> (python aleph0_game/scripts/play_game.py chess2d --overwrite --save_gif images/sample_chess2d_game.gif --duration 300 --random_players 0 1 --screenshot_dir output/chess2d --no_render)
[//]: <> (python aleph0_game/scripts/play_game.py chess5d --overwrite --save_gif images/sample_chess5d_game.gif --duration 300 --random_players 0 1 --screenshot_dir output/chess5d --no_render --max_depth 69 --seed 420)
![](https://github.com/pranavraj575/aleph0_game/blob/main/images/sample_jenga_game.gif)

![](https://github.com/pranavraj575/aleph0_game/blob/main/images/sample_large_jenga_game.gif)

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

![](https://github.com/pranavraj575/aleph0_game/blob/main/images/sample_large_jenga_game.gif)

Implemented a lightweight approximation of Jenga with no physics simulation.
A player moves by choosing a block to remove, then selecting a target place location.
The block will be placed with some randomness (unless the deterministic flag is enabled, as in the demos).

The following check is used for determining if the tower falls:
* For each height 0<_h_<tower height, consider the convex hull of the blocks at layer _h_.
* Deterministic version: 
  * If the center of mass of the tower segment above _h_ lies outside of the convex hull, the tower is deemed unstable at level _h_.
  * If the tower is unstable at any _h_, it will fall over.
* Stochastic version: 
  * The probability of the tower falling at level _h_ is lower the further the COM is from the convex hull.
  * Probability is 0.5 if COM lies directly above the boundary.
  * The probabilities for each _h_ are multiplied together to compute overall probability of tower falling. 
This check is done twice on each player's turn: after removing a block, and after placing a block.

To replicate the examples shown:

```shell
python aleph0_game/scripts/play_game.py jenga --args initial_height:5 deterministic:True --save_gif images/sample_jenga_game.gif --duration 1000 --random_players 0 1 --screenshot_dir output/jenga --seed 69 --overwrite --no_render
python aleph0_game/scripts/play_game.py jenga --args initial_height:18 deterministic:True --save_gif images/sample_large_jenga_game.gif --duration 420 --random_players 0 1 --screenshot_dir output/jenga_large --seed 420 --overwrite --no_render
```
### 5D Chess
![](https://github.com/pranavraj575/aleph0_game/blob/main/images/sample_chess5d_game.gif)

Implemented [5D chess with multiverse time travel](https://www.5dchesswithmultiversetimetravel.com/), with some caveats:
* The game ends when a king is captured, instead of at checkmate.
  This is done to reduce computation time, since otherwise each turn would have to compute all squares attacked by the opponent.
* The spot an opponent king castled through can also be captured by a piece on the same dimension in the next timestep.
* If the `stalemate_is_win` flag is set to True (which it is by default), no stalemate check is necessary.
  * In this case, the player that captures the opponent king wins.
* If the `stalemate_is_win` flag is set to False, a stalemate check must be done at the conclusion of the game:
  * Assume player _i_ captured the opponent king at turn _t_
  * We consider the state of the game at the start of turn _t_-1 (the start of the opponent's last turn)
  * If the opponent is in [check](https://en.wikipedia.org/wiki/5D_Chess_with_Multiverse_Time_Travel#check:~:text=A%20player%20is%20in%20check), the game ends with a player _i_ win.
  * If there exists a turn (sequence of moves) that the opponent can play on turn _t_-1 to get out of check, this position is NOT a stalemate.
    The game ends in a player _i_ win (since player _i_-1 failed to get out of check).
  * Otherwise, this is a stalemate, and the game ends in a draw.
* Since this check involves considering every sequence of moves that could constitute a turn, it is computationally expensive and turned off by default.


To replicate the example shown:
```shell
python aleph0_game/scripts/play_game.py chess5d --overwrite --save_gif images/sample_chess5d_game.gif --duration 300 --random_players 0 1 --screenshot_dir output/chess5d --no_render --max_depth 69 --seed 420
```

### 2D Chess
![](https://github.com/pranavraj575/aleph0_game/blob/main/images/sample_chess2d_game.gif)

Implemented chess as a subclass of 5d chess. 
Because of this, the game ends on the capture of a king (or upon 'capturing' a square that a king castled through on the previous move). 
Stalemate and checkmate are correctly evalutated upon the game's end (i.e. `stalemate_is_win` flag is False by default).

To replicate the example shown:
```shell
python aleph0_game/scripts/play_game.py chess2d --overwrite --save_gif images/sample_chess2d_game.gif --duration 300 --random_players 0 1 --screenshot_dir output/chess2d --no_render
```
