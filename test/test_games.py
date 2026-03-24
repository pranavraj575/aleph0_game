import pytest
import torch

from src.games import MNK, Jenga, TicTacToe

games = [
    TicTacToe(),
    MNK(3, 3, 3),
    MNK(5, 5, 3),
    MNK(5, 5, 5),
    Jenga(),
    Jenga(players=3),
    Jenga(initial_height=5),
    Jenga(deterministic=True),
]


@pytest.mark.parametrize("seed", list(range(13)))
@pytest.mark.parametrize("game", games)
def test_game_playthrough(seed, game):
    torch.random.manual_seed(seed)
    s = game.init_state()
    terminal = False
    while not terminal:
        actions = torch.stack(torch.where(game.action_mask(s)), dim=-1)
        idx = torch.randint(0, len(actions), (1,))
        action = actions[idx].flatten()
        s, _, terminal, _ = game.step(s, action)


@pytest.mark.parametrize("seed", list(range(2)))
@pytest.mark.parametrize("game", games)
def test_game_render(seed, game):
    torch.random.manual_seed(seed)
    s = game.init_state()
    terminal = False
    canvas = game.get_canvas()
    while not terminal:
        actions = torch.stack(torch.where(game.action_mask(s)), dim=-1)
        idx = torch.randint(0, len(actions), (1,))
        action = actions[idx].flatten()
        game.render(canvas, s)
        s, _, terminal, _ = game.step(s, action)
    game.close_canvas(canvas)


def equality(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    else:
        return all(equality(aa, bb) for aa, bb in zip(a, b))


@pytest.mark.parametrize("seed", list(range(52)))
@pytest.mark.parametrize("game", games)
def test_seeded_randomness(seed, game):
    torch.random.manual_seed(seed)
    s = game.init_state()
    torch.random.manual_seed(seed)
    s2 = game.init_state()
    terminal = False
    while not terminal:
        # change seed like this
        seed = seed + 1
        actions = torch.stack(torch.where(game.action_mask(s)), dim=-1)
        idx = torch.randint(0, len(actions), (1,))
        action = actions[idx].flatten()
        assert equality(game.agent_observe(s), game.agent_observe(s2))
        assert equality(game.critic_observe(s), game.critic_observe(s2))
        torch.random.manual_seed(seed)
        s, _, terminal, _ = game.step(s, action)
        torch.random.manual_seed(seed)
        s2, _, terminal, _ = game.step(s2, action)
