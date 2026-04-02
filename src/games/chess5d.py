import dataclasses
import itertools

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
    BLOCKED = 20
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
    GHOST_KING = 18
    assert KING + UNMOVED_SHIFT == UNMOVED_KING
    assert LOWEST_UNMOVED <= UNMOVED_KING and UNMOVED_KING <= HIGHEST_UNMOVED

    ROOK = 3
    UNMOVED_ROOK = 16
    GHOST_ROOK = 19
    assert ROOK + UNMOVED_SHIFT == UNMOVED_ROOK
    assert LOWEST_UNMOVED <= UNMOVED_ROOK and UNMOVED_ROOK <= HIGHEST_UNMOVED

    KNIGHT = 4
    BISHOP = 5
    QUEEN = 6  # attacks on any number of diagonals

    PRINCESS = 10  # combo of rook and bishop
    UNICORN = 11  # attacks on triagonls
    DRAGON = 12  # attacks on quadragonals

    def __init__(self):
        """
        implemented 5d chess
        SIMPLIFICATIONS:
            players are allowed to castle despite being in check.
                However, if player castles, opponent can 'capture' the squares the king moved through, and win the game
                Thus, this is equivalent to not being able to castle, as doing so loses the game
        """
        super().__init__()

    def has_special_actions(self):
        return True

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
                capture = new_board[*board_action].clone()
                new_frame = new_board[time1, dim1].clone()
                new_frame = self.mutate_remove_temporary_states(frame=new_frame)
                new_frame[i1, j1] = self.EMPTY
                new_frame[i2, j2] = self.moved_piece(
                    piece=state.piece_held,
                    pick=state.held_piece_origin,
                    place=board_action,
                )
                ident = torch.abs(state.piece_held)
                if ident == self.UNMOVED_KING:  # check for castling
                    diff = j2 - j1
                    if torch.abs(diff) > 1:  # we have castled, move rook as well
                        if j2 > j1:
                            new_frame[i1, j1:j2] = self.GHOST_KING * state.player
                        else:
                            new_frame[i1, j2 + 1 : j1 + 1] = self.GHOST_KING * state.player
                        rook_pick_j = 0 if diff < 0 else self.BOARD_SIZE - 1
                        rook_place_j = int((j2 + j1) / 2)

                        new_frame[i2, rook_pick_j] = self.EMPTY
                        # TODO: is there a better way to set player than multiplication here
                        new_frame[i2, rook_place_j] = self.GHOST_ROOK * state.player
                if ident == self.PAWN:  # check for en passant
                    if (abs(i2 - i1) == 1) and (abs(j2 - j1) == 1):  # captured in xy coords
                        if torch.abs(new_board[time1, dim1, i1, j2]) == self.PASSANTABLE_PAWN:
                            capture = new_frame[i1, j2].clone()
                            new_frame[i1, j2] = self.EMPTY
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
                new_frame_pick = self.mutate_remove_temporary_states(frame=new_frame_pick)
                new_board, new_center_timeline = self.mutate_add_child_frame(
                    board=new_board,
                    center_timeline=new_center_timeline,
                    frame=new_frame_pick,
                    td_idx=(time1, dim1),
                )
                new_frame_place = new_board[time2, dim2].clone()
                capture = new_board[*board_action].clone()
                new_frame_place = self.mutate_remove_temporary_states(frame=new_frame_place)
                new_frame_place[i2, j2] = self.moved_piece(
                    piece=state.piece_held,
                    pick=state.held_piece_origin,
                    place=board_action,
                )
                new_board, new_center_timeline = self.mutate_add_child_frame(
                    board=new_board,
                    center_timeline=new_center_timeline,
                    frame=new_frame_place,
                    td_idx=(time2, dim2),
                )

            captured_id = torch.abs(capture)
            # TODO: set containment faster?
            # TODO: stalemate check; maybe keep track of board from last player's move in state
            if captured_id in (
                self.KING,
                self.UNMOVED_KING,
                self.GHOST_KING,
                self.GHOST_ROOK,
            ):
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
            blocked_slice = self.BLOCKED * torch.ones((board.shape[0], 1, *board.shape[2:]), dtype=board.dtype)
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
                blocked_slice = self.BLOCKED * torch.ones((1, *board.shape[1:]), dtype=board.dtype)
                board = torch.concatenate((board, blocked_slice), dim=0)
        board[time + 1, new_dim] = frame
        return board, center_timeline

    def idx_exists(self, board, td_idx, ij_idx=(0, 0)):
        time, dim = td_idx
        i, j = ij_idx
        return (
            (time >= 0)
            and (time < board.shape[0])
            and (dim >= 0)
            and (dim < board.shape[1])
            and (i >= 0)
            and (i < self.BOARD_SIZE)
            and (j >= 0)
            and (j < self.BOARD_SIZE)
            and board[time, dim, i, j] != self.BLOCKED  # if one is blocked, they all are
        )

    def player_at(self, time):
        """
        1 for first player
        -1 for second player
        """
        return 1 - 2 * (time % 2)

    def mutate_remove_temporary_states(self, frame):
        """
        MUTATES FRAME, removes passantable pieces, and record of castling
        """
        ident_frame = torch.abs(frame)
        passantable = torch.eq(ident_frame, self.PASSANTABLE_PAWN)
        # change all passantable pawns to normal pawns, with the correct sign
        frame[passantable] = torch.sign(frame[passantable]) * self.PAWN

        ghost_king = torch.eq(ident_frame, self.GHOST_KING)
        frame[ghost_king] = self.EMPTY

        ghost_rook = torch.eq(ident_frame, self.GHOST_ROOK)
        frame[ghost_rook] = self.ROOK
        return frame

    def moved_piece(self, piece, pick, place):
        ident = torch.abs(piece)
        # if unmoved (pawn, king, rook), it is now moved
        if self.LOWEST_UNMOVED <= ident and ident <= self.HIGHEST_UNMOVED:
            ident = ident - self.UNMOVED_SHIFT
        if (ident == self.PAWN) or (ident == self.PASSANTABLE_PAWN):
            if (place[2] == self.BOARD_SIZE - 1) or (place[2] == 0):
                # TODO: if we want choice, add an UNKNOWN piece, and special actions for each possible choice
                ident = self.QUEEN
            if abs(pick[2] - place[2]) == 2:
                # pawn moved two spaces, can be captured by enpassant
                ident = self.PASSANTABLE_PAWN
        return ident * torch.sign(piece)

    def action_mask(self, state):
        if state.piece_held != 0:
            action_mask = torch.zeros_like(state.board, dtype=torch.bool)
            for idx in self._piece_possible_moves(state.board, state.held_piece_origin):
                action_mask[*idx] = True
        else:
            player_pieces = torch.logical_and(
                torch.eq(torch.sign(state.board), state.player),
                torch.le(state.board, self.LARGEST_PIECE),
            )
            p0 = (state.player + 1) // 2
            players_turn = (torch.arange(len(state.board)) + p0) % 2

            # get 'leaves', which are the last board in a timeline
            #  either last timestep, or all boards after are BLOCKED
            leaves = torch.zeros_like(state.board, dtype=torch.bool)
            leaves[-1] = 1
            leaves[:-1] = torch.eq(state.board[1:], self.BLOCKED)

            action_mask = torch.logical_and(leaves, players_turn.reshape(-1, 1, 1, 1))
            action_mask = torch.logical_and(action_mask, player_pieces)
            for piece_idx in zip(*torch.where(action_mask)):
                piece_idx = torch.tensor(piece_idx)
                if next(self._piece_possible_moves(state.board, piece_idx.clone()), None) is None:
                    action_mask[*piece_idx] = False
        # TODO: what if stalemate? maybe add special move in this case
        if (state.piece_held != 0) or (state.player == self.player_at(self.get_present(board=state.board, center_timeline=state.center_timeline))):
            # player cannot end turn, since they are holding a piece, or have not moved on a board in the 'present'
            special_moves = torch.zeros(1, dtype=torch.bool)
        else:
            # can only end turn if no piece is held, and it is the opponent's turn in the 'present' time
            special_moves = torch.ones(1, dtype=torch.bool)
        return special_moves, action_mask

    def get_present(self, board, center_timeline):
        """
        returns present time of a board
            this is the earliest end of an active timeline
        :return:
        """
        num_dims = board.shape[1]
        num_active_branches = min(center_timeline, num_dims - center_timeline) + 1
        # shaped (num timesteps, num_dims), True where there is a board
        active_mask = torch.not_equal(
            board[:, max(0, center_timeline - num_active_branches) : min(center_timeline + num_active_branches + 1, num_dims), 0, 0], self.BLOCKED
        )

        # shaped (num timesteps, num_dims), timestep index where there is a board, 0 otherwise
        active_indices = active_mask * torch.arange(len(active_mask), dtype=torch.int).unsqueeze(1)

        return torch.min(torch.max(active_indices, dim=0).values)

    def _piece_possible_moves(self, board, piece_idx):
        piece = board[*piece_idx]
        ident = torch.abs(piece)
        player = torch.sign(piece)
        idx_time, idx_dim, idx_i, idx_j = piece_idx

        if ident in (
            self.ROOK,
            self.UNMOVED_ROOK,
            self.BISHOP,
            self.UNICORN,
            self.DRAGON,
            self.PRINCESS,
            self.QUEEN,
            self.KING,
            self.UNMOVED_KING,
        ):  # easy linear moves
            if ident == self.ROOK:
                dims_to_change = itertools.combinations(range(4), 1)
            elif ident == self.BISHOP:
                dims_to_change = itertools.combinations(range(4), 2)
            elif ident == self.UNICORN:
                dims_to_change = itertools.combinations(range(4), 3)
            elif ident == self.DRAGON:
                dims_to_change = itertools.combinations(range(4), 4)
            elif ident == self.PRINCESS:
                # combo of rook and bishop
                dims_to_change = itertools.chain(
                    itertools.combinations(range(4), 1),
                    itertools.combinations(range(4), 2),
                )
            else:
                dims_to_change = itertools.chain(*[itertools.combinations(range(4), k) for k in range(1, 5)])
            for dims in dims_to_change:
                for signs in itertools.product((-1, 1), repeat=len(dims)):
                    pos = piece_idx.clone()
                    vec = torch.tensor((0, 0, 0, 0))
                    for k, dim in enumerate(dims):
                        vec[dim] = signs[k] * ((dim == 0) + 1)  # mult by 2 if dim is time
                    pos += vec
                    while self.idx_exists(board, pos[:2], pos[2:]) and (torch.sign(board[*pos]) != player):
                        yield pos.clone()
                        if (torch.sign(board[*pos]) != 0) or (ident == self.KING) or (ident == self.UNMOVED_KING):
                            # end of the line, or the king which moves single spaces
                            break
                        pos += vec
        if ident == self.KNIGHT:
            dims_to_change = itertools.permutations(range(4), 2)
            for dims in dims_to_change:
                for signs in itertools.product((-1, 1), repeat=len(dims)):
                    pos = piece_idx.clone()
                    for k, dim in enumerate(dims):
                        # multiply one of the dimensions by 1 and one by 2
                        # can do this with *(k+1)
                        pos[dim] += (k + 1) * signs[k] * ((dim == 0) + 1)
                    if self.idx_exists(board, pos[:2], pos[2:]) and (player != torch.sign(board[*pos])):
                        yield pos
        if (ident == self.PAWN) or (ident == self.UNMOVED_PAWN):
            # forward moves, add 'player', which is 1 or -1
            for dim in (2, 1):
                pos = piece_idx.clone()
                for _ in range(1 + (ident == self.UNMOVED_PAWN)):
                    pos[dim] += player
                    if self.idx_exists(board, pos[:2], pos[2:]) and (torch.sign(board[*pos]) == 0):
                        yield pos.clone()
                    else:
                        break
            # diag moves
            for dims in ((2, 3), (1, 0)):
                for aux_sign in (-1, 1):
                    pos = piece_idx.clone()
                    pos[dims[0]] += player
                    pos[dims[1]] += aux_sign
                    if self.idx_exists(board, pos[:2], pos[2:]) and (torch.sign(board[*pos]) == -player):
                        # this MUST be a capture
                        yield pos
            # en passant check
            for other_j in (idx_j + 1, idx_j - 1):
                if self.idx_exists(board, piece_idx[:2], (idx_i, other_j)):
                    other_piece = board[*piece_idx[:3], other_j]
                    if (torch.sign(other_piece) == -player) and (torch.abs(other_piece) == self.PASSANTABLE_PAWN):
                        pos = piece_idx.clone()
                        pos[2] += player
                        pos[3] = other_j
                        yield pos

        # castling check
        if ident == self.UNMOVED_KING:
            # for rook_i in (0, Board.BOARD_SIZE - 1):
            rook_i = idx_i  # rook must be on same rank
            for rook_j in (0, self.BOARD_SIZE - 1):
                # potential rook squares
                rook_maybe = board[idx_time, idx_dim, rook_i, rook_j]
                if rook_maybe == player * self.UNMOVED_ROOK:
                    dir = torch.sign(rook_j - idx_j)
                    if rook_j > idx_j:
                        mid_squares = board[idx_time, idx_dim, rook_i, idx_j + 1 : rook_j]
                    else:
                        mid_squares = board[idx_time, idx_dim, rook_i, rook_j + 1 : idx_j]
                    if torch.all(torch.eq(mid_squares, 0)):
                        pos = piece_idx.clone()
                        pos[3] = idx_j + 2 * dir
                        yield pos
                    else:
                        continue

    def agent_observe(self, state):
        # TODO: potentially flip board for opponent
        #  also maybe denote what piece was removed better (instead of just giving index)
        return state.board, torch.concatenate((torch.tensor([state.piece_held]), state.held_piece_origin))

    def critic_observe(self, state):
        return state.board, torch.tensor([state.piece_held])

    ###
    # RENDERING
    ###
    def render_action_mask(self, canvas, state):
        """
        debug method for printing out action mask
        """
        special_mask, board_mask = self.action_mask(state)
        self.render(
            canvas,
            State(
                board=board_mask.to(torch.int),
                player=state.player,
                center_timeline=state.center_timeline,
                piece_held=state.piece_held,
                held_piece_origin=state.held_piece_origin,
            ),
        )
        print("special_actions:", special_mask.numpy())

    def render(self, canvas, state):
        s = self.get_game_str(state)
        s += f"PLAYER {(1 - state.player) // 2}\n"
        if state.piece_held != 0:
            s += f"PIECE HELD: {self.piece_to_str(state.piece_held)} from {state.held_piece_origin.numpy()}\n"
        print(s)

    def get_game_str(self, state):
        s = ""
        for dim in range(state.board.shape[1] - 1, -1, -1):
            if dim == state.center_timeline:
                s += "(CENTER) "
            s += f"D {dim}:\n"
            for i in range(self.BOARD_SIZE - 1, -1, -1):
                rows = state.board[:, dim, i, :].cpu().numpy()
                row = "||".join("".join(map(self.piece_to_str, row)) for row in rows)

                s += str(row)
                s += "\n"
            s += "\n\n"
        return s

    def piece_to_str(self, piece):
        match abs(piece):
            case self.UNMOVED_PAWN:
                ident_str = "P"
            case self.PAWN:
                ident_str = "P"
            case self.PASSANTABLE_PAWN:
                ident_str = "P"
            case self.KING:
                ident_str = "K"
            case self.UNMOVED_KING:
                ident_str = "K"
            case self.ROOK:
                ident_str = "R"
            case self.UNMOVED_ROOK:
                ident_str = "R"
            case self.GHOST_ROOK:
                ident_str = "R"
            case self.KNIGHT:
                ident_str = "N"
            case self.BISHOP:
                ident_str = "B"
            case self.QUEEN:
                ident_str = "Q"
            case _:
                ident_str = "."
        if piece < 0:
            return ident_str.lower()
        else:
            return ident_str


class Chess2d(Chess5d):
    def has_special_actions(self):
        return False

    def _piece_possible_moves(self, board, piece_idx):
        for idx in super()._piece_possible_moves(board, piece_idx):
            if torch.all(torch.eq(idx[:2], piece_idx[:2])):
                yield idx

    def action_mask(self, state):
        special_action_mask, board_action_mask = super().action_mask(state)
        return board_action_mask[-1, 0]

    def step(self, state, action):
        new_action = (
            torch.tensor(-1),
            torch.concatenate((torch.tensor((len(state.board) - 1, 0)), action)),
        )
        new_state, rewards, terminal, aux = super().step(state, new_action)
        # if we have just placed a piece, it is oppoenents turn
        if (not terminal) and (new_state.piece_held == 0):
            new_state, rewards, terminal, auxp = super().step(new_state, (torch.tensor(0), -torch.ones(4, dtype=torch.int)))
            return new_state, rewards, terminal, aux | auxp
        return new_state, rewards, terminal, aux

    def get_game_str(self, state):
        s = ""

        for i in range(self.BOARD_SIZE - 1, -1, -1):
            row = state.board[-1, 0, i, :].cpu().numpy()
            row = "".join(map(self.piece_to_str, row))

            s += str(row)
            s += "\n"
        s += "\n\n"
        return s
