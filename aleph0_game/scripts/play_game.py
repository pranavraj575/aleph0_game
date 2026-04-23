import argparse
import ast
import os
import shutil

import torch
from PIL import Image

from aleph0_game import games


def play_game(
    game: games.Game,
    random_players,
    screenshot_dir=None,
    render=True,
    max_depth=float("inf"),
):
    if render:
        canvas = game.get_canvas()
    else:
        canvas = None
    state = game.init_state()
    terminal = False
    total_rwd = torch.zeros(game.num_agents())
    i = 0
    while not terminal:
        if screenshot_dir is not None:
            game.save_screenshot(state, os.path.join(screenshot_dir, str(i)))
        mask = game.action_mask(state)
        player = game.player(state)
        if player in random_players:
            action = game.sample_from_action_mask(mask)
            if game.has_special_actions():
                action = (tuple(map(int, action[0])), int(action[1]))
            else:
                action = tuple(map(int, action))
            print(f"step {i}: opponent played action {action}")
        else:
            if render:
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
        i += 1
        if i > max_depth:
            break
        total_rwd += rwd

    if screenshot_dir is not None:
        game.save_screenshot(state, os.path.join(screenshot_dir, str(i)))
    if render:
        game.close_canvas(canvas)
    print("game completed, rewards are:")
    print(total_rwd.numpy())


def create_gif(image_paths, output_gif_path, duration=200):
    images = [Image.open(image_path) for image_path in image_paths]
    if all(im.size == images[0].size for im in images):
        # no size shenanigans needed, just save as gif
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,  # 0 means infinite loop
        )

    W, H = max([im.width for im in images]), max([im.height for im in images])

    # get background color
    # im.getcolors() returns unsorted list of (count, color)
    colors = images[0].getcolors(maxcolors=images[0].width * images[0].height)
    assert colors is not None
    _, background_color = max(colors, key=lambda x: x[0])

    # resize all images to the maximum image size
    resized_imgs = [Image.new(mode=im.mode, size=(W, H), color=background_color) for im in images]
    for im, canvas_im in zip(images, resized_imgs):
        canvas_im.paste(im)
    resized_imgs[0].save(
        output_gif_path,
        save_all=True,
        append_images=resized_imgs[1:],
        duration=duration,
        loop=0,
    )


if __name__ == "__main__":
    implemented_games = {
        "chess5d": games.Chess5d,
        "chess2d": games.Chess2d,
        "jenga": games.Jenga,
        "tic-tac-toe": games.TicTacToe,
        "mnk": games.MNK,
        "resign_mnk": games.F_MNK,
        "checkers": games.Checkers,
    }
    p = argparse.ArgumentParser(description="test game playing/rendering")

    p.add_argument("game", choices=list(implemented_games.keys()), help="game to play")
    p.add_argument("--args", required=False, type=str, default=[], nargs="+", help="keyword arguments of game, in format arg1:value1 arg2:value2 ...")
    p.add_argument("--no_render", action="store_true", help="dont render the game")
    p.add_argument("--screenshot_dir", required=False, type=str, help="save screenshots to a directory")
    p.add_argument("--overwrite", action="store_true", help="overwrite screenshot dir (if exists)")
    p.add_argument("--save_gif", required=False, type=str, help="save gif of the screenshots to this file")
    p.add_argument("--duration", required=False, type=int, default=200, help="duration (in ms) of each img in gif")
    p.add_argument("--max_depth", required=False, type=int, default=-1, help="maximum number of moves")
    p.add_argument("--random_players", required=False, type=int, default=[], nargs="+", help="indices of players that will be making random moves")
    p.add_argument("--seed", required=False, type=int, default=69, help="random seed for random players")
    args = p.parse_args()
    Game = implemented_games[args.game]
    game_kwargs = [arg.split(":") for arg in args.args]
    assert all(len(t) == 2 for t in game_kwargs), "--args must be formatted like arg1:value arg2:value ..."
    game_kwargs = {k: ast.literal_eval(v) for k, v in game_kwargs}
    game = Game(**game_kwargs)

    if args.screenshot_dir is not None:
        if not args.overwrite:
            assert not os.path.exists(args.screenshot_dir), f"directory {args.screenshot_dir} exists already"
        if os.path.exists(args.screenshot_dir):
            shutil.rmtree(args.screenshot_dir)
        os.makedirs(args.screenshot_dir, exist_ok=True)
    torch.random.manual_seed(args.seed)
    play_game(
        game=game,
        random_players=args.random_players,
        screenshot_dir=args.screenshot_dir,
        render=not args.no_render,
        max_depth=args.max_depth if args.max_depth >= 0 else float("inf"),
    )
    if args.save_gif is not None:
        assert args.screenshot_dir is not None, "if saving gif, screenshot_dir must be specified"
        os.makedirs(os.path.dirname(args.save_gif), exist_ok=True)
        # all filenames should be {i}.xxx
        image_paths = sorted(os.listdir(args.screenshot_dir), key=lambda fn: int(fn.split(".")[0]))
        image_paths = list(map(lambda x: os.path.join(args.screenshot_dir, x), image_paths))
        create_gif(
            image_paths=image_paths,
            output_gif_path=args.save_gif,
            duration=args.duration,
        )
