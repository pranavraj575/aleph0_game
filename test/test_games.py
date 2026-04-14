import pytest
import torch

from aleph0_game.games import F_MNK, MNK, Chess2d, Chess5d, Jenga, TicTacToe

small_games = [
    TicTacToe(),
    MNK(3, 3, 3),
    MNK(5, 5, 3),
    MNK(5, 5, 5),
    Jenga(),
    Jenga(players=3),
    Jenga(initial_height=5),
    Jenga(deterministic=True),
    F_MNK(3, 3, 3),
]
all_games = small_games + [
    Chess5d(),
    Chess2d(),
]


@pytest.mark.parametrize("seed", list(range(13)))
@pytest.mark.parametrize("game", small_games)
def test_game_playthrough(seed, game):
    torch.random.manual_seed(seed)
    s = game.init_state()
    terminal = False
    while not terminal:
        mask = game.action_mask(s)
        action = game.sample_from_action_mask(mask)
        game.agent_observe(s)
        game.critic_observe(s)
        assert game.is_valid(s, action)
        s, _, terminal, _ = game.step(s, action)


@pytest.mark.parametrize("seed", list(range(1)))
@pytest.mark.parametrize("game", [Chess5d()])
def test_game_render(seed, game, depth=200):
    torch.random.manual_seed(seed)
    s = game.init_state()
    terminal = False
    canvas = game.get_canvas()
    while depth >= 0 and not terminal:
        mask = game.action_mask(s)
        action = game.sample_from_action_mask(mask)
        game.render(canvas, s)
        s, _, terminal, _ = game.step(s, action)
        depth -= 1
    game.close_canvas(canvas)


def equality(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    else:
        return all(equality(aa, bb) for aa, bb in zip(a, b))


@pytest.mark.parametrize("seed", list(range(52)))
@pytest.mark.parametrize("game", small_games)
def test_seeded_randomness(seed, game):
    torch.random.manual_seed(seed)
    s = game.init_state()
    torch.random.manual_seed(seed)
    s2 = game.init_state()
    terminal = False
    while not terminal:
        # change seed like this
        seed = seed + 1
        mask = game.action_mask(s)
        action = game.sample_from_action_mask(mask)
        assert equality(game.agent_observe(s), game.agent_observe(s2))
        assert equality(game.critic_observe(s), game.critic_observe(s2))
        torch.random.manual_seed(seed)
        s, _, terminal, _ = game.step(s, action)
        torch.random.manual_seed(seed)
        s2, _, terminal, _ = game.step(s2, action)
