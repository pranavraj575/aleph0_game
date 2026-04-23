import numpy as np
import torch

from .game import Game


class MNK(Game):
    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k

    def num_agents(self):
        return 2

    def board_action_dim(self, state):
        return 2

    def init_state(self):
        # (board, player)
        return torch.zeros((self.m, self.n)), 1

    def step(self, state, action):
        board, player = state
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

    def player(self, state):
        _, player = state
        # state.player is 1 or -1
        # change this to p0 (1) and p1 (-1)
        return int((1 - player) // 2)

    def check_winner(self, board):
        for weights in [
            torch.ones(1, 1, 1, self.k),
            torch.ones(1, 1, self.k, 1),
            torch.eye(self.k).reshape(1, 1, self.k, self.k),
            torch.eye(self.k)[list(range(self.k))[::-1]].reshape(1, 1, self.k, self.k),
        ]:
            conv_res = torch.conv2d(1.0 * board.unsqueeze(0), weights)
            if torch.any(torch.eq(conv_res, self.k)):
                return True
        return False

    def agent_observe(self, state):
        board, player = state
        # for player -1, invert the board selections
        obs_board = board * player
        # observation shaped (3,m,n). observation[0] is the empty squares, observation[1] is the player's squares,
        #  observation[2] is the opponent's squares
        return torch.eq(obs_board.unsqueeze(0), torch.tensor([0, 1, -1]).reshape(-1, 1, 1))

    def action_mask(self, state):
        """
        possible actions to take
        Args:
            state: The state of the environment.
        """
        board, _ = state
        return torch.eq(board, 0)

    def critic_observe(self, state):
        """
        critic observations
        Args:
            state: The state of the environment.
        """
        board, _ = state
        return torch.eq(board.unsqueeze(0), torch.tensor([0, 1, -1]).reshape(-1, 1, 1))

    def get_game_str(self, state):
        board, _ = state
        return (
            "-" * self.n
            + "\n"
            + str(board.numpy().astype(np.int32))
            .replace(" ", "")
            .replace("[", "")
            .replace("]", "")
            .replace("-1", "O")
            .replace("0", ".")
            .replace("1", "X")
            + "\n"
            + "-" * self.n
        )

    def render(self, canvas, state):
        # canvas is not needed, just print it to terminal
        print(self.get_game_str(state))

    def save_screenshot(self, state, output_file, **kwargs):
        ascii_text = self.get_game_str(state=state)
        self.save_screenshot_ascii(ascii_text=ascii_text, output_file=output_file)


class TicTacToe(MNK):
    def __init__(self):
        super().__init__(m=3, n=3, k=3)
