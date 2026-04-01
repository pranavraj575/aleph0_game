import dataclasses

import torch

from .game import Game


@dataclasses.dataclass
class State:
    board: torch.Tensor
    player: int
    center_timeline: int
    piece_held: int = -1
    held_piece_origin: torch.Tensor = -torch.ones(4, dtype=torch.int)


class Chess5d(Game):
    # reserve a separate index for a blocked board (i.e. board that does not exist yet)
    # this is necessary since knights can jump over blocked boards, but other pieces cannot
    BLOCKED = 19
    REMOVED = 18

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
            center_timeline=0,
        )

    def player(self, state):
        return state.player

    def step(self, state, action):
        special_action, board_action = action
        if special_action == 0:
            # player resigns (if player 1 resigns, rewards are [-1,1], if player -1 does, rwds are [1,-1])
            state = State(
                board=state.board,
                player=1 - state.player,
                center_timeline=state.center_timeline,
            )
            # TODO: check if opponent has no moves here to determine terminality
            return state, torch.zeros(2), False, dict()
        new_board = state.board.clone()

        if state.piece_held < 0:
            # assert special_action<0 # cannot end turn while piece is held
            # pick a square
            piece = state.board[*board_action]
            new_board[*board_action] = self.REMOVED
            return State(
                new_board,
                player=state.player,
                piece_held=piece,
                held_piece_origin=board_action,
                center_timeline=state.center_timeline,
            )
            print(piece)
        else:
            new_board = torch.where(
                torch.eq(new_board, self.REMOVED), self.EMPTY, new_board
            )
            # TODO: check en passant, castling, captures, etc.
            print(new_board)
        raise NotImplementedError

    def agent_observe(self, state):
        # TODO: potentially flip board for black player
        return state.board, torch.tensor([state.piece_held])

    def action_mask(self, state):
        raise NotImplementedError

    def critic_observe(self, state):
        return state.board, torch.tensor([state.piece_held])


if __name__ == "__main__":
    c = Chess5d()
    b = c.init_state()
    print(b)
