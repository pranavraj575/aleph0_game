import dataclasses

import torch

from .game import Game


@dataclasses.dataclass
class State:
    board: torch.Tensor
    player: int
    center_timeline: int
    piece_held: int = 0
    held_piece_origin: torch.Tensor = -torch.ones(4, dtype=torch.int)


class Chess5d(Game):
    BOARD_SIZE = 8
    # reserve a separate index for a blocked board (i.e. board that does not exist yet)
    # this is necessary since knights can jump over blocked boards, but other pieces cannot
    BLOCKED = 18
    LARGEST_PIECE = BLOCKED - 1

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
    assert ROOK + UNMOVED_SHIFT == UNMOVED_ROOK
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
        board = torch.zeros(8, 8, dtype=torch.int)
        board[0] = back_rank
        board[1] = self.UNMOVED_PAWN
        board[-2] = -self.UNMOVED_PAWN
        board[-1] = -back_rank
        board = board.reshape(1, 1, *board.shape)
        return State(
            board=board,
            player=1,
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
                player=-state.player,
                center_timeline=state.center_timeline,
            )
            # TODO: check if opponent has no moves here to determine terminality
            return state, torch.zeros(2), False, dict()
        new_board = state.board.clone()

        if state.piece_held == 0:
            # assert special_action<0 # cannot end turn while piece is held
            # pick a square
            piece = state.board[*board_action]
            return (
                State(
                    board=new_board,
                    player=state.player,
                    piece_held=piece,
                    held_piece_origin=board_action,
                    center_timeline=state.center_timeline,
                ),
                torch.zeros(2),
                False,
                dict(),
            )
        else:
            new_center_timeline = state.center_timeline

            time1, dim1, i1, j1 = state.held_piece_origin
            time2, dim2, i2, j2 = board_action
            if (time1, dim1) == (time2, dim2):
                # Special case, where the move begins and ends on same board
                new_frame = new_board[time1, dim1].clone()
                new_frame[i1, j1] = self.EMPTY
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
                capture = new_board[*board_action].clone()
                if ident == self.PAWN:  # check for en passant
                    if abs(i2 - i1) == 1 and abs(j2 - j1) == 1:  # captured in xy coords
                        if torch.abs(new_frame[i1, j2]) == self.PASSANTABLE_PAWN:
                            capture = new_frame[i1, j2].clone()
                            new_frame[i1, j2] = self.EMPTY
                new_frame = self.mutate_remove_passantable_pieces(
                    frame=new_frame,
                    keep_idx=(i2, j2),
                )
                new_board, new_center_timeline = self.mutate_add_child_frame(
                    board=new_board,
                    center_timeline=new_center_timeline,
                    frame=new_frame,
                    td_idx=(time2, dim2),
                )
            else:
                new_frame_pick = new_board[time1, dim1].clone()
                new_frame_pick[i1, j1] = self.EMPTY
                # no pieces are enpassantable on this board,
                #  since any pieces that were enpassantable have had a turn pass
                #  and the move is not pawn up 2
                new_frame_pick = self.mutate_remove_passantable_pieces(
                    frame=new_frame_pick, keep_idx=None
                )
                new_board, new_center_timeline = self.mutate_add_child_frame(
                    board=new_board,
                    center_timeline=new_center_timeline,
                    frame=new_frame_pick,
                    td_idx=(time1, dim1),
                )
                new_frame_place = new_board[time2, dim2].clone()
                capture = new_board[*board_action].clone()
                new_frame_place[i2, j2] = self.moved_piece(
                    piece=state.piece_held,
                    pick=state.held_piece_origin,
                    place=board_action,
                )
                new_frame_place = self.mutate_remove_passantable_pieces(
                    frame=new_frame_place, keep_idx=(i2, j2)
                )
                new_board, new_center_timeline = self.mutate_add_child_frame(
                    board=new_board,
                    center_timeline=new_center_timeline,
                    frame=new_frame_place,
                    td_idx=(time2, dim2),
                )

            captured_id = torch.abs(capture)
            if captured_id == self.KING or captured_id == self.UNMOVED_KING:
                term = True
                rwd = torch.tensor([state.player, -state.player])
            else:
                term = False
                rwd = torch.zeros(2)
            return (
                State(
                    board=new_board,
                    player=state.player,
                    center_timeline=new_center_timeline,
                ),
                rwd,
                term,
                dict(),
            )

    def mutate_add_child_frame(self, board, center_timeline, frame, td_idx):
        """
        adds child to the frame specified by td_idx
        :param td_idx: (time, dimension)
        :param board: board to change
        :param frame: frame (nxn) to add as child
        Returns: dimenison spawned (idx)
        """
        time, dim = td_idx
        if self.idx_exists(board=board, td_idx=(time + 1, dim)):
            player = self.player_at(time)
            blocked_slice = self.BLOCKED * torch.ones(
                (board.shape[0], 1, *board.shape[2:]), dtype=board.dtype
            )
            if player == 1:
                new_dim = 0
                board = torch.concatenate((blocked_slice, board), dim=1)
                center_timeline += 1
            else:
                new_dim = board.shape[1]
                board = torch.concatenate((board, blocked_slice), dim=1)
        else:
            new_dim = dim
            while time + 1 >= len(board):
                blocked_slice = self.BLOCKED * torch.ones(
                    (1, *board.shape[1:]), dtype=board.dtype
                )
                board = torch.concatenate((board, blocked_slice), dim=0)
        board[time + 1, new_dim] = frame
        return board, center_timeline

    def idx_exists(self, board, td_idx):
        time, dim = td_idx
        return (time < board.shape[0]) and not torch.any(
            torch.eq(board[time, dim], self.BLOCKED)
        )

    def player_at(self, time):
        """
        1 for first player
        -1 for second player
        """
        return 1 - 2 * (time % 2)

    def mutate_remove_passantable_pieces(self, frame, keep_idx=None):
        """
        MUTATES FRAME
        """
        passantable = torch.eq(torch.abs(frame), self.PASSANTABLE_PAWN)
        if keep_idx is not None:
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
            if abs(pick[2] - place[2]) == 2:
                # pawn moved two spaces, can be captured by enpassant
                ident = self.PASSANTABLE_PAWN
        return ident * torch.sign(piece)

    def agent_observe(self, state):
        # TODO: potentially flip board for opponent
        #  also maybe denote what piece was removed better (instead of just giving index)
        return state.board, torch.concatenate(
            (torch.tensor([state.piece_held]), state.held_piece_origin)
        )

    def action_mask(self, state):
        player_pieces = torch.logical_and(
            torch.eq(torch.sign(state.board), state.player),
            torch.le(state.board, self.LARGEST_PIECE),
        )
        p0 = (state.player + 1) // 2
        players_time = (torch.arange(len(state.board)) + p0) % 2
        action_mask = torch.logical_and(
            player_pieces, players_time.reshape(-1, 1, 1, 1)
        )
        special_moves = torch.ones(1, dtype=torch.bool)
        return special_moves, action_mask

    def critic_observe(self, state):
        return state.board, torch.tensor([state.piece_held])


if __name__ == "__main__":
    c = Chess5d()
    b = c.init_state()
    b = c.step(b, (-1, [0, 0, 1, 0]))[0]
    b = c.step(b, (-1, [0, 0, 3, 0]))[0]
    b = c.step(b, (0, -torch.ones(4)))[0]
    b = c.step(b, (-1, [1, 0, 6, 0]))[0]
    b = c.step(b, (-1, [1, 0, 5, 0]))[0]
    b = c.step(b, (0, -torch.ones(4)))[0]
    b = c.step(b, (-1, [2, 0, 3, 0]))[0]
    b = c.step(b, (-1, [2, 0, 4, 0]))[0]
    b = c.step(b, (0, -torch.ones(4)))[0]
    b = c.step(b, (-1, [3, 0, 6, 1]))[0]
    b = c.step(b, (-1, [3, 0, 4, 1]))[0]
    b = c.step(b, (0, -torch.ones(4)))[0]
    b = c.step(b, (-1, [4, 0, 4, 0]))[0]
    b = c.step(b, (-1, [4, 0, 5, 1]))[0]
    b = c.step(b, (0, -torch.ones(4)))[0]
    print(b)
    print(c.action_mask(b))
