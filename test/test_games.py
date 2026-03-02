import pytest
import torch

from src.games.tictactoe import MNK


@pytest.mark.parametrize("seed", list(range(52)))
@pytest.mark.parametrize("game", [MNK(3,3,3),
MNK(5,5,3),
MNK(5,5,5),
                                 ])
def test_game_playthrough(seed,game):
    torch.random.manual_seed(seed)
    s = game.init_state()
    terminal=False
    while not terminal:
        actions = torch.stack(torch.where(game.action_mask(s)), dim=-1)
        idx = torch.randint(0, len(actions), (1,))
        action = actions[idx].flatten()
        s, _,terminal, _ = game.step(s, action)
