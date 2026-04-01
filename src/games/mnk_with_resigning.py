"""
used to test inclusion of a special 'resign' action
Action mask will be (vector, tensor) shaped
    neural network will output logits of the same shape
    when sampling, if the special action is selected, action returned will be tuple
        (0, [-1, -1])
    otherwise, if action (i,j) is selected, action returned will be tuple
        (-1, [i, j],)
"""

import torch

from .mnk import MNK


class F_MNK(MNK):
    HAS_SPECIAL_ACTIONS = True

    def step(self, state, action):
        board, player = state
        special_action, action = action
        if special_action == 0:
            # player resigns (if player 1 resigns, rewards are [-1,1], if player -1 does, rwds are [1,-1])
            rewards = torch.tensor([-player, player])
            return state, rewards, True, dict()

        new_board = board.clone()
        new_board[*action] = player
        game_won = self.check_winner(torch.eq(new_board, player))
        game_won = float(game_won)
        # if no one wan, rewards are [0,0]
        # if player 1 won then player=1, and rewards are [1,-1]
        # if player -1 won then player=-1 and rewards are [-1,1]
        rewards = torch.tensor([game_won * player, -game_won * player])
        terminal = game_won or torch.all(torch.not_equal(new_board, 0))
        new_state = new_board, -player
        return new_state, rewards, terminal, dict()

    def action_mask(self, state):
        board_action_mask = super().action_mask(state)
        # allow an extra action, which will lose the game on the spot
        return (torch.ones(1, dtype=torch.bool), board_action_mask)
