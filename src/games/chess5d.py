import dataclasses

import torch

from .game import Game


@dataclasses.dataclass
class State:
    board: torch.Tensor
    player: int
    piece_held: int


class Chess5d(Game):
    # reserve a separate index for a blocked board (i.e. board that does not exist yet)
    # this is necessary since knights can jump over blocked boards, but other pieces cannot
    BLOCKED = 18

    EMPTY = 0

    # a pawn cannot be both unmoved and passantable, so we do not need to add that
    PAWN = 1

    UNMOVED_PAWN = 14
    PASSANTABLE_PAWN = 17

    KING = 2
    UNMOVED_KING = 15

    ROOK = 3
    UNMOVED_ROOK = 16

    KNIGHT = 4
    BISHOP = 5
    QUEEN = 6  # attacks on any number of diagonals

    PRINCESS = 10  # combo of rook and bishop
    UNICORN = 11  # attacks on triagonls
    DRAGON = 12  # attacks on quadragonals

    def num_agents(self):
        return 2

    def init_state(self):
        back_rank = torch.tensor(
            [
                self.UNMOVED_ROOK,
                self.KNIGHT,
                self.BISHOP,
                self.QUEEN,
                self.UNMOVED_KING,
                self.BISHOP,
                self.KNIGHT,
                self.UNMOVED_ROOK,
            ]
        )
        board = torch.zeros(8, 8)
        board[0] = back_rank
        board[1] = self.UNMOVED_PAWN
        board[-2] = -self.UNMOVED_PAWN
        board[-1] = -back_rank
        return State(
            board=board,
            player=0,
            piece_held=-1,
        )

    def player(self, state):
        return state.player

    def step(self, state, action):
        new_board = state.board.clone()
        if state.piece_held < 0:
            # pick a square
            piece = state.board[*action]
            new_board[*action] = self.EMPTY
            print(piece)
            pass
        else:
            pass
        raise NotImplementedError

    def agent_observe(self, state):
        # TODO: potentially flip board for black player
        return state.board

    def action_mask(self, state):
        raise NotImplementedError

    def critic_observe(self, state):
        raise NotImplementedError


if __name__ == "__main__":
    c = Chess5d()
    b = c.init_state()
    print(b)
