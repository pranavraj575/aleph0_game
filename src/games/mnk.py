import torch

from .game import Game


class MNK(Game):
    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k

    def num_agents(self):
        return 2

    def init_state(self):
        # (board, player)
        return torch.zeros((self.m, self.n)), 1

    def step(self, state, action):
        board, player = state
        new_board = board.clone()
        new_board[action[0], action[1]] = player
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
        return player

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
        return obs_board.unsqueeze(0) == torch.tensor([0, 1, -1]).reshape(-1, 1, 1)

    def action_mask(self, state):
        """
        possible actions to take
        Args:
            state: The state of the environment.
        """
        board, _ = state
        return board == 0

    def critic_observe(self, state):
        """
        critic observations
        Args:
            state: The state of the environment.
        """
        board, _ = state
        return board.unsqueeze(0) == torch.tensor([0, 1, -1]).reshape(-1, 1, 1)

    def display(self, state):
        board, _ = state
        return (
            "---\n"
            + str(board.numpy())
            .replace(" ", "")
            .replace("[", "")
            .replace("]", "")
            .replace("-1", "O")
            .replace("0", ".")
            .replace("1", "O")
            + "\n---"
        )


class TicTacToe(MNK):
    def __init__(self):
        super().__init__(m=3, n=3, k=3)
