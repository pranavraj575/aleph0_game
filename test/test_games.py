import pytest
import torch

from src.games.tictactoe import Tictactoe


@pytest.mark.parametrize("seed", list(range(100)))
def test_tictactoe(seed):
    torch.random.manual_seed(seed)
    g = Tictactoe()
    s = g.init_state()
    while not g.is_terminal(s):
        actions = torch.stack(torch.where(g.action_mask(s)), dim=-1)
        idx = torch.randint(0, len(actions), (1,))
        action = actions[idx].flatten()
        s, _ = g.step(s, action)
