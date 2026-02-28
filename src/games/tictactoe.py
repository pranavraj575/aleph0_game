import torch

from .game import Game


class Tictactoe(Game):
    def num_agents(self):
        return 2

    def init_state(self):
        # (board, player)
        return torch.zeros((3, 3)), 1

    def step(self, state, action):
        board, player = state
        new_board = board.clone()
        new_board[action[0], action[1]] = player
        new_state = new_board, -player
        return new_state, dict()

    def player(self, state):
        _, player = state
        return player

    def is_terminal(self, state):
        board, player = state
        # board is terminal if either there are no moves left, or there is a winner
        return torch.all(torch.not_equal(board, torch.tensor(0))) or not torch.all(
            torch.eq(self.get_result(state), 0)
        )

    def get_result(self, state):
        board, _ = state
        for p in 1, -1:
            for k in range(3):
                if torch.all(torch.eq(board[k, :], p)) or torch.all(
                    torch.eq(board[:, k], p)
                ):
                    return torch.tensor([p, -p])
            if torch.all(torch.eq(board[range(3), range(3)], p)) or torch.all(
                torch.eq(board[range(3), [-1 - i for i in range(3)]], p)
            ):
                return torch.tensor([p, -p])
        return torch.zeros(2)

    def agent_observe(self, state):
        board, player = state
        # for player -1, invert the board selections
        obs_board = board * player
        # observation shaped (3,3,3). observation[0] is the empty squares, observation[1] is the player's squares,
        #  observation[2] is the opponent's squares
        return obs_board.reshape(1, 3, 3) == torch.tensor([0, 1, -1]).reshape(-1, 1, 1)

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
        return board.reshape(1, 3, 3) == torch.tensor([0, 1, -1]).reshape(-1, 1, 1)

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
