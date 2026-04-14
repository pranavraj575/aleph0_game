import argparse
import ast

import torch

from aleph0_game import games


def play_game(
    game: games.Game,
    random_players,
):
    canvas = game.get_canvas()
    state = game.init_state()
    terminal = False
    total_rwd = torch.zeros(game.num_agents())
    while not terminal:
        mask = game.action_mask(state)
        player = game.player(state)
        if player in random_players:
            action = game.sample_from_action_mask(mask)
            if game.has_special_actions():
                action = (tuple(map(int, action[0])), int(action[1]))
            else:
                action = tuple(map(int, action))
            print("opponent played action", action)
        else:
            game.render(canvas, state)
            if game.has_special_actions():
                board_actions, special_actions = mask
                board_actions = torch.where(board_actions)
                special_actions = list(map(int, torch.where(special_actions)[0]))
            else:
                board_actions = torch.where(mask)
                special_actions = []
            board_actions = [tuple(map(int, idx)) for idx in list(zip(*board_actions))]
            print("legal board actions:")

            print(*tuple(f"{i}: {idx}" for i, idx in enumerate(board_actions)), sep="\n")
            if special_actions:
                print("legal special actions:")
                print(*tuple(f"{i + len(board_actions)}: {idx}" for i, idx in enumerate(special_actions)), sep="\n")
            while True:
                idx = input("select idx of action: ")
                try:
                    idx = int(idx)
                    if idx >= 0 and idx < len(board_actions) + len(special_actions):
                        break
                except ValueError:
                    pass
            if idx < len(board_actions):
                if game.has_special_actions():
                    action = (board_actions[idx], -1)
                else:
                    action = board_actions[idx]
            else:
                action = (-torch.ones(game.board_action_dim(state=state), dtype=torch.int), special_actions[idx - len(board_actions)])
        state, rwd, terminal, aux = game.step_weak_type(state=state, action=action)
        total_rwd += rwd
    print("game completed, rewards are:")
    print(total_rwd.numpy())


if __name__ == "__main__":
    implemented_games = {
        "chess5d": games.Chess5d,
        "chess2d": games.Chess2d,
        "jenga": games.Jenga,
        "tic-tac-toe": games.TicTacToe,
        "mnk": games.MNK,
        "resign_mnk": games.F_MNK,
    }
    p = argparse.ArgumentParser(description="test game playing/rendering")

    p.add_argument("game", choices=list(implemented_games.keys()), help="game to play")

    p.add_argument("--args", required=False, type=str, default=[], nargs="+", help="keyword arguments of game, in format arg1:value1 arg2:value2 ...")
    p.add_argument("--random_players", required=False, type=int, default=[], nargs="+", help="indices of players that will be making random moves")
    args = p.parse_args()
    Game = implemented_games[args.game]
    game_kwargs = [arg.split(":") for arg in args.args]
    assert all(len(t) == 2 for t in game_kwargs), "--args must be formatted like arg1:value arg2:value ..."
    game_kwargs = {k: ast.literal_eval(v) for k, v in game_kwargs}
    game = Game(**game_kwargs)
    play_game(game=game, random_players=args.random_players)
