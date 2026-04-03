import pytest
import torch

from src.games import F_MNK, MNK, Chess2d, Chess5d, Game, Jenga, TicTacToe

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


def sample_from_action_mask(game: Game, action_mask):
    if game.has_special_actions():
        board_mask, special_mask = action_mask
        assert board_mask.dtype == torch.bool
        assert special_mask.dtype == torch.bool

        # easier if swapped here, since we can test action < len(special_mask)
        combined_mask = torch.concat((special_mask, board_mask.flatten()))
        action = torch.multinomial(combined_mask.to(torch.float), 1, True)
        if action < len(special_mask):
            return (-torch.ones(len(board_mask.shape), dtype=torch.int), action)
        else:
            return (
                torch.cat(torch.unravel_index(action - len(special_mask), board_mask.shape)),
                torch.tensor(-1),
            )
    else:
        # action mask is a tensor
        assert action_mask.dtype == torch.bool
        action = torch.multinomial(action_mask.flatten().to(torch.float), 1, True)
        return torch.cat(torch.unravel_index(action, action_mask.shape))


@pytest.mark.parametrize("seed", list(range(13)))
@pytest.mark.parametrize("game", small_games)
def test_game_playthrough(seed, game):
    torch.random.manual_seed(seed)
    s = game.init_state()
    terminal = False
    while not terminal:
        mask = game.action_mask(s)
        action = sample_from_action_mask(game, mask)
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
        action = sample_from_action_mask(game, mask)
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
        action = sample_from_action_mask(game, mask)
        assert equality(game.agent_observe(s), game.agent_observe(s2))
        assert equality(game.critic_observe(s), game.critic_observe(s2))
        torch.random.manual_seed(seed)
        s, _, terminal, _ = game.step(s, action)
        torch.random.manual_seed(seed)
        s2, _, terminal, _ = game.step(s2, action)
