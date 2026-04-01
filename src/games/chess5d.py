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
    BOARD_SIZE = 8
    # reserve a separate index for a blocked board (i.e. board that does not exist yet)
    # this is necessary since knights can jump over blocked boards, but other pieces cannot
    BLOCKED = 19
    REMOVED = 18

    EMPTY = 0

    UNMOVED_SHIFT = 13
    # all pieces between these numbers represent unmoved pieces
    LOWEST_UNMOVED = 14
    HIGHEST_UNMOVED = 16

    # a pawn cannot be both unmoved and passantable, so we do not need to add that
    PAWN = 1
    UNMOVED_PAWN = 14
    PASSANTABLE_PAWN = 17
    assert PAWN + UNMOVED_SHIFT == UNMOVED_PAWN
    assert LOWEST_UNMOVED <= UNMOVED_PAWN and UNMOVED_PAWN <= HIGHEST_UNMOVED

    KING = 2
    UNMOVED_KING = 15
    assert KING + UNMOVED_SHIFT == UNMOVED_KING
    assert LOWEST_UNMOVED <= UNMOVED_KING and UNMOVED_KING <= HIGHEST_UNMOVED

    ROOK = 3
    UNMOVED_ROOK = 16
    assert UNMOVED_ROOK + UNMOVED_SHIFT == UNMOVED_ROOK
    assert LOWEST_UNMOVED <= UNMOVED_ROOK and UNMOVED_ROOK <= HIGHEST_UNMOVED

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
                board=new_board,
                player=state.player,
                piece_held=piece,
                held_piece_origin=board_action,
                center_timeline=state.center_timeline,
            )
        else:
            new_board = torch.where(
                torch.eq(new_board, self.REMOVED), self.EMPTY, new_board
            )

            time1, dim1, i1, j1 = state.held_piece_origin
            time2, dim2, i2, j2 = board_action
            if (time1, dim1) == (time2, dim2):
                # Special case, where the move begins and ends on same board

                new_frame = new_board[time1, dim1].clone()
                new_frame[i2, j2] = self.moved_piece(
                    piece=state.piece_held,
                    pick=state.held_piece_origin,
                    place=board_action,
                )
                ident = torch.abs(state.piece_held)
                if ident == self.UNMOVED_KING:  # check for castling
                    # TODO: maybe time travel moves matter for check?
                    diff = j2 - j1
                    if torch.abs(diff) > 1:  # we have castled, move rook as well
                        rook_pick_j = 0 if diff < 0 else self.BOARD_SIZE - 1
                        rook_place_j = int((j2 + j1) / 2)
                        new_frame[time2, dim2, i2, rook_pick_j] = self.EMPTY
                        new_frame[i2, rook_place_j] = self.moved_piece(
                            piece=new_frame[i2, rook_pick_j],
                            pick=torch.tensor((time2, dim2, i2, rook_pick_j)),
                            place=torch.tensor((time2, dim2, i2, rook_place_j)),
                        )
                capture = new_board[*board_action]
                if ident == self.PAWN:  # check for en passant
                    if abs(i2 - i1) == 1 and abs(j2 - j1) == 1:  # captured in xy coords
                        if torch.abs(new_frame[i1, j2]) == self.PASSANTABLE_PAWN:
                            capture = new_frame[i1, j2]
                            new_frame[i1, j2] = self.EMPTY
                new_frame = self.mutate_remove_passantable_pieces(
                    frame=new_frame,
                    keep_idx=(i2, j2),
                )
                new_board = self.mutate_place_frame(
                    board=new_board,
                    frame=new_frame,
                    td_idx=(time1 + 1, dim2),
                )
                print(capture)
            else:
                pass
            return State(
                board=new_board,
                player=state.player,
                center_timeline=state.center_timeline,
            )

    def mutate_place_frame(self, board, frame, td_idx):
        # TODO: fix this
        while td_idx[0] >= len(board):
            # add a time slice of BLOCKED spaces
            blocked_slice = self.BLOCKED * torch.ones(
                (1, *board.shape[1:]), dtype=board.dtype
            )
            board = torch.concatenate((board, blocked_slice), dim=0)
        board[td_idx] = frame

        return board

    def mutate_remove_passantable_pieces(self, frame, keep_idx):
        """
        MUTATES FRAME
        """
        passantable = torch.eq(torch.abs(frame), self.PASSANTABLE_PAWN)
        passantable[*keep_idx] = False
        # change all passantable pawns to normal pawns, with the correct sign
        frame[passantable] = torch.sign(frame[passantable]) * self.PAWN
        return frame

    def moved_piece(self, piece, pick, place):
        ident = torch.abs(piece)
        # if unmoved (pawn, king, rook), it is now moved
        if self.LOWEST_UNMOVED <= ident and ident <= self.HIGHEST_UNMOVED:
            ident = ident - self.UNMOVED_SHIFT
        if ident == self.PAWN or ident == self.PASSANTABLE_PAWN:
            if place[2] == self.BOARD_SIZE - 1 or place[2] == 0:
                # TODO: if we want choice, add an UNKNOWN piece, and special actions for each possible choice
                ident = self.QUEEN
            if torch.abs(pick[2] - place[2]) == 2:
                # pawn moved two spaces, can be captured by enpassant
                ident = self.PASSANTABLE_PAWN
        return ident * torch.sign(piece)

    def agent_observe(self, state):
        # TODO: potentially flip board for black player
        return state.board, torch.concatenate(
            (torch.tensor([state.piece_held]), state.held_piece_origin)
        )

    def action_mask(self, state):
        raise NotImplementedError

    def critic_observe(self, state):
        return state.board, torch.tensor([state.piece_held])


if __name__ == "__main__":
    c = Chess5d()
    b = c.init_state()
    print(b)
