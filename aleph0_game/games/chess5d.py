import dataclasses
import itertools
from collections import defaultdict

import torch

from .game import Game


@dataclasses.dataclass(frozen=True)
class State:
    # current board
    board: torch.Tensor
    # current player
    player: int
    # '0' timeline
    center_timeline: int
    # track timesteps, a single timestep is a (pick and place) move
    #  i.e. just the pick move does not increment timesteps, and special moves (like end turn) do not increment timesteps
    timestep: int
    # timestep each board was created
    board_spawn_timestep: torch.Tensor
    start_turn_timestep: int
    prev_start_turn_timestep: int
    # we split the game into a pick piece and place piece phase, in the second case, these tracks which piece is picked
    piece_held: torch.Tensor = torch.tensor(0)
    held_piece_origin: torch.Tensor = -torch.ones(4, dtype=torch.int)


def all_subsets(iterable, all_permutations=False):
    for k in range(len(iterable) + 1):
        for subset in itertools.combinations(iterable, k):
            if all_permutations:
                for perm in itertools.permutations(subset):
                    yield perm
            else:
                yield subset


def DAG_subgraphs_w_at_most_one_outgoing_edge(edge_list: dict, used_sources=None, used_vertices=None):
    """
    iterable of all lists of edges that create a DAG such that each vertex is the source of at most one edge
        (we run through all permutations, sort of wasteful)
    the order returned will be in reverse topological order (the correct traversal)

    :param edge_list: dict(vertex -> vertex set), must be copyable
    :param used_sources: set(vertex), sources that were already used as source
    :param used_vertices: set(vertex), vertices that were already used as source or sink (and thus cant be reused as source)
    :return: iterable of (list[(start vertex, end vertex)], used vertices)
    """
    if used_sources is None:
        used_sources = set()
    if used_vertices is None:
        used_vertices = used_sources.copy()
    # given the used items, can either add nothing,
    yield (), used_sources, used_vertices
    # or add some edge (edge must not have a source in used_vertices)
    for source in edge_list:
        if source not in used_vertices:
            for end in edge_list[source]:
                for (
                    subsub,
                    all_source,
                    all_used,
                ) in DAG_subgraphs_w_at_most_one_outgoing_edge(
                    edge_list=edge_list,
                    used_sources=used_sources.union({source}),
                    # if (source, end) move is made, no other moves can be made from eitehr source or end
                    used_vertices=used_vertices.union({source, end}),
                ):
                    yield (((source, end),) + subsub, all_source, all_used)


class Chess5d(Game):
    """
    implemented 5d chess
    SIMPLIFICATIONS:
        game ends upon the capture of a king
            after player i caputures an opponent king, we do a stalemate check (unless specified tha stalemate is a win)
            roll back game to player -i's last turn
            if player -i has no turn that gets them out of check, this is a stalemate, and the game ends in a draw
            otherwise, player i wins (even if there was a move that got player -i out of check that they did not find)
        players are allowed to castle despite being in check.
            However, if player castles, opponent can 'capture' the squares the king moved through, and win the game
            Thus, this is equivalent to not being able to castle, as doing so loses the game
        reasoning for both of these is computation. We only need to test for checks/stalemate once, at the end of the game
            This check is difficult, since it requires considering every possible turn from the losing player
    """

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

    def __init__(self, stalemate_is_win=True):
        """
        :param stalemate_is_win: whether to count stalemate as a win instead of draw
            this save a ton of computation in large games
        """
        super().__init__()
        self.stalemate_is_win = stalemate_is_win

    def has_special_actions(self):
        return True

    def num_agents(self):
        return 2

    def board_action_dim(self, state):
        return 4

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
            timestep=0,
            start_turn_timestep=0,
            prev_start_turn_timestep=-1,
            board_spawn_timestep=torch.zeros_like(board[:, :, 0, 0], dtype=torch.int),
        )

    def player(self, state):
        # state.player is 1 or -1
        # change this to p0 (1) and p1 (-1)
        return int((1 - state.player) // 2)

    def agent_observe(self, state):
        # TODO: potentially flip board for opponent
        #  also maybe denote what piece was removed better (instead of just giving index)
        return state.board, torch.concatenate((torch.tensor([state.piece_held]), state.held_piece_origin))

    def critic_observe(self, state):
        return state.board, torch.tensor([state.piece_held])

    def step(self, state, action):
        """
        calls self._step
        ._step will always take in (board action, special action), and this allows internal logic to be consistent
            (aka it always uses (board move, special move))
        interface with game API can swap to just using board move (in the case of 2d chess)
        """
        return self._step(state=state, action=action)

    def _step(self, state, action):
        """
        does not mutate state
        :param state:
        :param action:
        :return:
        """
        board_action, special_action = action
        if special_action == 0:
            # player ends turn
            state = State(
                board=state.board.clone(),
                player=-state.player,  # dont need to clone, since this is an int
                center_timeline=state.center_timeline,  # same
                timestep=state.timestep,
                start_turn_timestep=state.timestep,
                prev_start_turn_timestep=state.start_turn_timestep,
                board_spawn_timestep=state.board_spawn_timestep,
            )
            # TODO: check if opponent has no moves here to determine terminality
            return state, torch.zeros(2), False, dict()

        if state.piece_held == 0:
            # pick a square
            piece = state.board[*board_action]
            return (
                State(
                    board=state.board.clone(),
                    player=state.player,
                    piece_held=piece,
                    held_piece_origin=board_action,
                    center_timeline=state.center_timeline,
                    timestep=state.timestep,  # just selection step does not increase timestep
                    start_turn_timestep=state.start_turn_timestep,
                    prev_start_turn_timestep=state.prev_start_turn_timestep,
                    board_spawn_timestep=state.board_spawn_timestep,
                ),
                torch.zeros(2),
                False,
                dict(),
            )
        else:
            return self.make_move(state, state.held_piece_origin, board_action)

    def make_move(self, state: State, pick_idx, place_idx):
        new_board = state.board.clone()
        new_board_spawn_timestep = state.board_spawn_timestep.clone()
        new_timestep = state.timestep + 1
        new_center_timeline = state.center_timeline

        time1, dim1, i1, j1 = pick_idx
        time2, dim2, i2, j2 = place_idx
        aux = dict()
        if (time1, dim1) == (time2, dim2):
            # Special case, where the move begins and ends on same board
            capture = new_board[*place_idx].clone()
            new_frame = new_board[time1, dim1].clone()
            new_frame = self.mutate_remove_temporary_states(frame=new_frame)
            new_frame[i1, j1] = self.EMPTY
            new_frame[i2, j2], moved_aux = self.get_moved_piece(
                piece=state.piece_held,
                pick=state.held_piece_origin,
                place=place_idx,
            )
            aux.update(moved_aux)
            ident = torch.abs(state.piece_held)
            if ident == self.UNMOVED_KING:  # check for castling
                diff = j2 - j1
                if torch.abs(diff) > 1:  # we have castled, move rook as well
                    if j2 > j1:
                        new_frame[i1, j1:j2] = self.GHOST_KING * state.player
                        aux["castled"] = "right"
                    else:
                        new_frame[i1, j2 + 1 : j1 + 1] = self.GHOST_KING * state.player
                        aux["castled"] = "left"
                    rook_pick_j = 0 if diff < 0 else self.BOARD_SIZE - 1
                    rook_place_j = int((j2 + j1) / 2)

                    new_frame[i2, rook_pick_j] = self.EMPTY
                    new_frame[i2, rook_place_j] = self.GHOST_ROOK * state.player

            if ident == self.PAWN:  # check for en passant
                if (abs(i2 - i1) == 1) and (abs(j2 - j1) == 1):  # captured in xy coords
                    if torch.abs(new_board[time1, dim1, i1, j2]) == self.PASSANTABLE_PAWN:
                        capture = new_frame[i1, j2].clone()
                        new_frame[i1, j2] = self.EMPTY
            new_board, new_center_timeline, new_board_spawn_timestep = self.mutate_add_child_frame(
                board=new_board,
                center_timeline=new_center_timeline,
                frame=new_frame,
                td_idx=(time2, dim2),
                timestep=new_timestep,
                board_spawn_timestep=new_board_spawn_timestep,
            )
        else:
            new_frame_pick = new_board[time1, dim1].clone()
            new_frame_pick[i1, j1] = self.EMPTY
            # no pieces are enpassantable on this board,
            #  since any pieces that were enpassantable have had a turn pass
            #  and the move is not pawn up 2
            new_frame_pick = self.mutate_remove_temporary_states(frame=new_frame_pick)
            new_board, new_center_timeline, new_board_spawn_timestep = self.mutate_add_child_frame(
                board=new_board,
                center_timeline=new_center_timeline,
                frame=new_frame_pick,
                td_idx=(time1, dim1),
                timestep=new_timestep,
                board_spawn_timestep=new_board_spawn_timestep,
            )

            new_frame_place = new_board[time2, dim2].clone()
            capture = new_board[*place_idx].clone()
            new_frame_place = self.mutate_remove_temporary_states(frame=new_frame_place)
            new_frame_place[i2, j2], moved_aux = self.get_moved_piece(
                piece=state.piece_held,
                pick=state.held_piece_origin,
                place=place_idx,
            )
            aux.update(moved_aux)
            new_board, new_center_timeline, new_board_spawn_timestep = self.mutate_add_child_frame(
                board=new_board,
                center_timeline=new_center_timeline,
                frame=new_frame_place,
                td_idx=(time2, dim2),
                timestep=new_timestep,
                board_spawn_timestep=new_board_spawn_timestep,
            )

        captured_id = torch.abs(capture)
        # TODO: set containment faster?
        # TODO: stalemate check;
        # for some reason, castling is legal even if skipping square attacked by a piece in a different time/dimension
        new_state = State(
            board=new_board,
            player=state.player,
            center_timeline=new_center_timeline,
            timestep=new_timestep,
            start_turn_timestep=state.start_turn_timestep,
            prev_start_turn_timestep=state.prev_start_turn_timestep,
            board_spawn_timestep=new_board_spawn_timestep,
        )
        if captured_id in (self.KING, self.UNMOVED_KING) or ((time1, dim1) == (time2, dim2) and captured_id in (self.GHOST_KING, self.GHOST_ROOK)):
            term = True
            if (not self.stalemate_is_win) and self.is_stalemate(new_state):
                rwd = torch.zeros(2)
            else:
                rwd = torch.tensor([state.player, -state.player])
        else:
            term = False
            rwd = torch.zeros(2)

        if captured_id > 0:
            aux["capture"] = capture
        return (
            new_state,
            rwd,
            term,
            aux,
        )

    def action_mask(self, state: State):
        """
        calls self._action_mask
        ._action_mask will always return (board mask, special mask), and this allows internal logic to be consistent

        in the case of Chess2D, .action_mask only needs to return the board mask.
        """
        return self._action_mask(state=state)

    def _action_mask(self, state):
        if state.piece_held != 0:
            action_mask = torch.zeros_like(state.board, dtype=torch.bool)
            for idx in self._piece_possible_moves(state.board, state.held_piece_origin):
                action_mask[*idx] = True
        else:
            action_mask = self.maybe_movable_piece_mask(state=state)
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
        return action_mask, special_moves

    ###
    # UTILITY FUNCTIONS
    ###

    def maybe_movable_piece_mask(self, state: State):
        """
        mask for all pieces on movable boards of the current player
        :param state:
        :return:
        """
        player_pieces = torch.logical_and(
            torch.eq(torch.sign(state.board), state.player),
            torch.le(state.board, self.LARGEST_PIECE),
        )
        first_players_move = (state.player + 1) // 2
        # 1 if current player is first player, 0 otherwise
        current_players_turn = (torch.arange(len(state.board)) + first_players_move) % 2

        # get 'leaves', which are the last board in a timeline
        #  either last timestep, or all boards after are BLOCKED
        leaves = torch.zeros_like(state.board, dtype=torch.bool)
        leaves[-1] = 1
        leaves[:-1] = torch.eq(state.board[1:], self.BLOCKED)

        # returns the conjuncion, player's pieces that are on a 'leaf' where it is the current player's turn
        action_mask = torch.logical_and(leaves, current_players_turn.reshape(-1, 1, 1, 1))
        return torch.logical_and(action_mask, player_pieces)

    def mutate_add_child_frame(self, board, center_timeline, frame, td_idx, timestep, board_spawn_timestep):
        """
        adds child to the frame specified by td_idx
        :param td_idx: (time, dimension)
        :param board: board to change
        :param frame: frame (nxn) to add as child
        :param board_spawn_timestep: record of timesteps where each board is spawned, new version of this is output
        """
        new_board_spawn_timestep = board_spawn_timestep.clone()
        time, dim = td_idx
        if self.idx_exists(board=board, td_idx=(time + 1, dim)):
            player = self.player_at(time)
            blocked_slice = self.BLOCKED * torch.ones((board.shape[0], 1, *board.shape[2:]), dtype=board.dtype)
            blocked_timestep_slice = -1 * torch.ones((new_board_spawn_timestep.shape[0], 1), dtype=new_board_spawn_timestep.dtype)
            if player == 1:
                new_dim = 0
                board = torch.concatenate((blocked_slice, board), dim=1)
                new_board_spawn_timestep = torch.concatenate((blocked_timestep_slice, new_board_spawn_timestep), dim=1)
                center_timeline += 1
            else:
                new_dim = board.shape[1]
                board = torch.concatenate((board, blocked_slice), dim=1)
                new_board_spawn_timestep = torch.concatenate((new_board_spawn_timestep, blocked_timestep_slice), dim=1)
        else:
            new_dim = dim
            while time + 1 >= len(board):
                blocked_slice = self.BLOCKED * torch.ones((1, *board.shape[1:]), dtype=board.dtype)
                blocked_timestep_slice = -1 * torch.ones((1, *new_board_spawn_timestep.shape[1:]), dtype=new_board_spawn_timestep.dtype)
                board = torch.concatenate((board, blocked_slice), dim=0)
                new_board_spawn_timestep = torch.concatenate((new_board_spawn_timestep, blocked_timestep_slice), dim=0)
        board[time + 1, new_dim] = frame
        new_board_spawn_timestep[time + 1, new_dim] = timestep
        return board, center_timeline, new_board_spawn_timestep

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
        frame[ghost_rook] = torch.sign(frame[ghost_rook]) * self.ROOK
        return frame

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

    def get_moved_piece(self, piece, pick, place):
        ident = torch.abs(piece)
        # if unmoved (pawn, king, rook), it is now moved
        aux = dict()
        if self.LOWEST_UNMOVED <= ident and ident <= self.HIGHEST_UNMOVED:
            ident = ident - self.UNMOVED_SHIFT
        if (ident == self.PAWN) or (ident == self.PASSANTABLE_PAWN):
            if (place[2] == self.BOARD_SIZE - 1) or (place[2] == 0):
                # TODO: if we want choice, add an UNKNOWN piece, and special actions for each possible choice
                ident = self.QUEEN
                aux["promotion"] = "Q"
            if abs(pick[2] - place[2]) == 2:
                # pawn moved two spaces, can be captured by enpassant
                ident = self.PASSANTABLE_PAWN
        return ident * torch.sign(piece), aux

    def get_active_board_range(self, board, center_timeline):
        """
        returns a,b, where [a,b) is the range of active boards
        :param board:
        :param center_timeline:
        :return:
        """
        num_dims = board.shape[1]
        # active branches is the lowest number of branches made by a player plus 1
        num_active_branches = min(center_timeline, (num_dims - 1) - center_timeline) + 1
        return max(0, center_timeline - num_active_branches), min(center_timeline + num_active_branches + 1, num_dims)

    def get_present(self, board, center_timeline):
        """
        returns present time of a board
            this is the earliest end of an active timeline
        :return:
        """
        low, high = self.get_active_board_range(board=board, center_timeline=center_timeline)
        # shaped (num timesteps, num_dims), True where there is a board
        active_mask = torch.not_equal(board[:, low:high, 0, 0], self.BLOCKED)

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
            if ident in (self.ROOK, self.UNMOVED_ROOK):
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
                        if (torch.sign(board[*pos]) != 0) or (ident in (self.KING, self.UNMOVED_KING)):
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
                    pos[dims[1]] += aux_sign * ((dims[1] == 0) + 1)
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

    def rewind_to_timestep(self, state: State, timestep, lossless=False):
        """
        uses timestep data to get state at a particular timestep
        :param state:
        :param lossless: whether to calculate state.start_turn_timestep and state.prev_start_turn_timestep
        :return: state at specified timestep
        """
        assert timestep <= state.timestep

        start_board = state.board.clone()
        # set all timesteps that are too large to -1, then make a mask that only includes numbers that are not -1
        inclusion_mask = torch.where(torch.le(state.board_spawn_timestep, timestep), state.board_spawn_timestep, -1)
        inclusion_mask = torch.not_equal(inclusion_mask, -1)
        new_board_spawn_timestep = torch.where(inclusion_mask, state.board_spawn_timestep, -1)

        center_timeline = state.center_timeline
        # cut off additional timesteps
        while torch.all(torch.logical_not(inclusion_mask[-1])):
            inclusion_mask = inclusion_mask[:-1]
            start_board = start_board[:-1]
            new_board_spawn_timestep = new_board_spawn_timestep[:-1]

        # cut off additional dimensions (from end)
        while torch.all(torch.logical_not(inclusion_mask[:, -1])):
            inclusion_mask = inclusion_mask[:, :-1]
            start_board = start_board[:, :-1]
            new_board_spawn_timestep = new_board_spawn_timestep[:, :-1]

        # cut off additional dimensions (from start)
        while torch.all(torch.logical_not(inclusion_mask[:, 0])):
            inclusion_mask = inclusion_mask[:, 1:]
            start_board = start_board[:, 1:]
            new_board_spawn_timestep = new_board_spawn_timestep[:, 1:]
            center_timeline = center_timeline - 1
        # now filter according to mask
        start_board = torch.where(inclusion_mask.reshape(*inclusion_mask.shape, 1, 1), start_board, self.BLOCKED)

        player = -100
        start_turn_timestep = -1
        prev_start_turn_timestep = -1
        if timestep >= state.start_turn_timestep:
            # the turn has not changed
            start_turn_timestep = state.start_turn_timestep
            prev_start_turn_timestep = state.prev_start_turn_timestep
            player = state.player
        elif timestep >= state.prev_start_turn_timestep:
            # we are on previous turn
            start_turn_timestep = state.prev_start_turn_timestep
            player = -state.player
        if lossless:
            if player == -100:
                # possible since each timestep must spawn at least one board (since only pick + place counts as a timestep)
                time_idx_of_board = torch.where(torch.eq(new_board_spawn_timestep, timestep))[0][0]
                # player 1 has boards 0, 2, ... and player -1 has odd boards
                player = int(1 - 2 * (time_idx_of_board.item() % 2))
            if start_turn_timestep == -1:
                # at the start of players turn, the board with highest timestep is the current timestep
                # any board that is added during the turn is an opponent board
                if player == 1:
                    player_steps = new_board_spawn_timestep[::2]
                else:
                    player_steps = new_board_spawn_timestep[1::2]
                start_turn_timestep = int(torch.max(player_steps))
            if prev_start_turn_timestep == -1:
                # same logic, but we must ignore any boards added by current player
                #  (i.e. after start_turn_timestep)
                temp = torch.where(torch.ge(new_board_spawn_timestep, start_turn_timestep), -1, new_board_spawn_timestep)
                if player == 1:
                    opponent_steps = temp[1::2]
                else:
                    opponent_steps = temp[::2]
                if len(opponent_steps) == 0:
                    prev_start_turn_timestep = -1
                else:
                    prev_start_turn_timestep = int(torch.max(opponent_steps))

        return State(
            board=start_board,
            center_timeline=center_timeline,
            board_spawn_timestep=new_board_spawn_timestep,
            timestep=timestep,
            player=player,
            start_turn_timestep=start_turn_timestep,
            prev_start_turn_timestep=prev_start_turn_timestep,
        )

    def undo_player_turn(self, state: State, prev_turn=False, lossless=False):
        """
        if prev_turn=False, undoes moves the current player made (with no loss of information)
        if prev_turn=True, undoes current and previous player's moves (with loss of information about 2 turns ago)

        undo_player_turn with prev_turn=True cannot be used more than once on a board
        :param state:
        :return: state before player made any moves
        """

        if prev_turn:
            timestep = state.prev_start_turn_timestep
        else:
            timestep = state.start_turn_timestep
        return self.rewind_to_timestep(state=state, timestep=timestep, lossless=lossless)

    def player_in_check(self, state: State):
        """
        returns whether the current player is in check
        check is defined as "A player is in check in a situation where it is the player's turn and,
            if the player were to pass their move on all active boards in the present,
            then the opponent would be able to capture one of the player's kings"
        :param state:
        :return:
        """
        center_timeline = state.center_timeline
        present = self.get_present(board=state.board, center_timeline=center_timeline)
        new_board = state.board.clone()
        if self.player_at(present) == state.player:
            # pass on all active boards in the present

            if len(new_board) == present + 1:
                # timestep present+1 does not exist, add a slice
                blocked_slice = self.BLOCKED * torch.ones((1, *new_board.shape[1:]), dtype=new_board.dtype)
                new_board = torch.concatenate((new_board, blocked_slice), dim=0)

            low, high = self.get_active_board_range(board=state.board, center_timeline=center_timeline)
            # mask of active boards in the present that must be passed on
            # shaped (num active dims,)
            pass_dims = torch.eq(new_board[present + 1, low:high, 0, 0], self.BLOCKED)
            # on the specified dimensions, copy over the previous timestep, otherwise keep the board
            new_board[present + 1, low:high] = torch.where(pass_dims, new_board[present, low:high], new_board[present + 1, low:high])
        dummy_state = State(
            board=new_board,
            player=-state.player,
            center_timeline=center_timeline,
            timestep=-1,
            start_turn_timestep=-1,
            prev_start_turn_timestep=-1,
            board_spawn_timestep=torch.ones(new_board.shape[:2], dtype=torch.int),
        )
        for piece_idx in zip(*torch.where(self.maybe_movable_piece_mask(state=dummy_state))):
            piece_idx = torch.tensor(piece_idx)
            for place_idx in self._piece_possible_moves(new_board, piece_idx):
                dest_piece = torch.abs(new_board[*place_idx])
                if dest_piece in (self.KING, self.UNMOVED_KING):
                    return True
                elif dest_piece in (self.GHOST_KING, self.GHOST_ROOK) and (place_idx[:2] == piece_idx[:2]):
                    return True
        return False

    def is_stalemate(self, terminal_state: State):
        """
        checks if a terminal state is stalemate
        state is terminal if current player just captured a king
        :param terminal_state:
        :return:
        """
        # old_state is the state at start of previous player's turn
        old_state = self.undo_player_turn(state=terminal_state, prev_turn=True)
        if self.player_in_check(old_state):
            return False
        if self.player_has_non_losing_turn(state=old_state):
            return False
        return True

    def player_has_non_losing_turn(self, state: State):
        """
        checks if current player has a turn that ends without them in check
        used in stalemate test
        """
        # TODO: check all possible moves from old state, if any avoid check, then this is not stalemate
        for turn in self.get_all_possible_turns(state=state, all_permutations=False):
            temp_s = state
            for pick_idx, place_idx in turn:
                recenter = torch.tensor([0, temp_s.center_timeline, 0, 0])
                temp_s, _, _, _ = self._step(temp_s, (pick_idx + recenter, torch.tensor(-1, dtype=torch.int)))
                temp_s, _, _, _ = self._step(temp_s, (place_idx + recenter, torch.tensor(-1, dtype=torch.int)))
            if not self.player_in_check(state=temp_s):
                return True
        return False

    def get_all_possible_turns(self, state: State, all_permutations=False):
        """
        iterable of all turns of current player from state
        first, we break moves into equivalence classes based on the (time, dim) of the pick and the place
        move order MOSTLY does not matter, as a board that is spawned cannot be moved to/through (since it is an opponent board)
        the exception is if a move is made to and from a particular board. In this case, the board must be the SOURCE of a move first, then be the SINK
        then we must take care of order when a set of moves contains moves that move TO an active board:
            any move that moves TO a board at (time, dim) must occur after a move that moves FROM (time, dim)
                exception is moves that start and end at same (time,dim) ('self edges'), which must occur first
            Thus, every turn has the following order: ((0) self edges, (1) moves that DONT move TO an active board, (2) moves that do move TO an active board)
            additionally, type (2) can be thought of as a directed graph (vertices are (time,dim) coordinates, edges are the moves)
            this graph cannot have cycles (since this would make it impossible for the "exit" move to occur before the "enter" move on every board of the cycle)
            Thus, this graph must be a DAG, where each vertex is the source of at most one edge
        to generate all possible turns, first consider all possible subsets of moves of type (2) that represent a DAG with that property
            i.e. all vertices are source of at most one edge
        then consider all moves of type (1) or (0) that do not use any 'sources' from the moves already generated
        This will generate all possible subsets of moves that can generate a turn
        a subset is a valid turn if all boards in the 'present' are the source or sink of some move

        once we have a valid subset, we can consider permutations
            type (0) moves have the same result with any permutation
            type (1) moves do have different results with different permutations
            type (2) moves do also (since they do not spawn a new dimension)

        NOTE: turn is centered at the CENTER dimension (to prevent errors resulting from a move changing board indices)
        """
        center_idx = torch.tensor([0, state.center_timeline, 0, 0])
        # assert state.piece_held==0
        board_action_mask, _ = self._action_mask(state=state)
        present = self.get_present(board=state.board, center_timeline=state.center_timeline)
        # boards with any movable pieces
        source_board_mask = torch.any(board_action_mask, dim=(2, 3))
        source_boards = set(tuple(map(int, item)) for item in zip(*torch.where(source_board_mask)))
        source_boards_in_present = {(t, d) for (t, d) in source_boards if t == present}
        # create graph structure
        all_self_edges = set()
        edge_list_to_active = defaultdict(lambda: set())
        edge_list_not_to_active = defaultdict(lambda: set())
        partition = defaultdict(lambda: list())
        for piece_idx in zip(*torch.where(board_action_mask)):
            piece_idx = torch.tensor(piece_idx)
            for place_idx in self._piece_possible_moves(board=state.board, piece_idx=piece_idx):
                start_td_idx, end_td_idx = tuple(map(int, piece_idx[:2])), tuple(map(int, place_idx[:2]))
                partition[(start_td_idx, end_td_idx)].append((piece_idx, place_idx))
                if start_td_idx == end_td_idx:
                    all_self_edges.add(start_td_idx)
                elif end_td_idx in source_boards:
                    edge_list_to_active[start_td_idx].add(end_td_idx)
                else:
                    edge_list_not_to_active[start_td_idx].add(end_td_idx)
        for move_indices, used_sources, used_td_idxs in DAG_subgraphs_w_at_most_one_outgoing_edge(edge_list=edge_list_to_active):
            # move_indices are the moves of type (2)
            # see any self edges that can be added (i.e. moves of type (1))
            possible_self_edges = all_self_edges.difference(used_sources)
            for self_edges in all_subsets(possible_self_edges):
                # self_edge_indices are moves of type (1)
                used_sources_p = used_sources.union(self_edges)
                used_td_idxs_p = used_td_idxs.union(self_edges)
                # see any moves of type (3) that can be added
                possible_type_three_sources = set(edge_list_not_to_active.keys()).difference(used_sources_p)
                for type_three_sources in all_subsets(possible_type_three_sources, all_permutations=all_permutations):
                    used_td_idxs_pp = used_td_idxs_p.union(type_three_sources)
                    # check if this subset advances the present
                    if not source_boards_in_present.issubset(used_td_idxs_pp):
                        continue
                    for type_three_sinks in itertools.product(*(edge_list_not_to_active[td_idx] for td_idx in type_three_sources)):
                        # return turn, done as type (0), (1), then (2) moves

                        equivalnce_class_turn = (
                            tuple((td_idx, td_idx) for td_idx in self_edges) + tuple(zip(type_three_sources, type_three_sinks)) + move_indices
                        )

                        for turn in itertools.product(*(partition[t] for t in equivalnce_class_turn)):
                            yield tuple((start_idx - center_idx, end_idx - center_idx) for (start_idx, end_idx) in turn)

    ###
    # RENDERING
    ###

    def render(self, canvas, state):
        print(self.get_game_str(state))

    def save_screenshot(self, state, output_file, **kwargs):
        ascii_text = self.get_game_str(state=state)
        self.save_screenshot_ascii(ascii_text=ascii_text, output_file=output_file, bold=True)

    def get_game_str(self, state):
        s = ""
        active_rng = self.get_active_board_range(board=state.board, center_timeline=state.center_timeline)
        present = self.get_present(board=state.board, center_timeline=state.center_timeline)

        length_to_place_present = -1
        time_indices = []
        present_token = "%"
        board_sep = "||"
        nl = ""

        for dim in range(state.board.shape[1] - 1, -1, -1):
            if dim == state.center_timeline:
                s += "(CENTER) "
            if dim < active_rng[0] or dim >= active_rng[1]:
                s += "(INACTIVE)"
            s += f"DIM {dim}:\n"
            for t in range(self.BOARD_SIZE - 1, -1, -1):
                rows = state.board[:, dim, t, :].cpu().numpy()
                row = "||".join("".join(map(self.piece_to_str, row)) for row in rows)
                sp = str(row)

                # calculate present and insert present token at right place
                if length_to_place_present == -1:
                    past_rows = state.board[: present + 1, 0, 0, :].cpu().numpy()
                    past_row = "||".join("".join(map(self.piece_to_str, row)) for row in past_rows)
                    length_to_place_present = len(past_row) + 1
                    nl = " " * length_to_place_present + present_token + "\n"
                    for k in range(state.board.shape[0]):
                        past_rows = state.board[:k, 0, 0, :].cpu().numpy()
                        time_indices.append(len(board_sep.join("".join(map(self.piece_to_str, row)) for row in past_rows)))

                if len(sp) < length_to_place_present:
                    sp = sp + " " * (length_to_place_present - len(sp))
                s += sp[:length_to_place_present] + present_token + sp[length_to_place_present:]
                s += "\n"
            s += "\n" + nl

        temp = nl[: -(1 + len(f"PRESENT TIME{present_token}"))] + f"PRESENT TIME{present_token}\n"
        added = ""
        for t in range(len(time_indices)):
            added = added + " " * max(time_indices[t] - len(added), 0)
            added += " " * max((self.BOARD_SIZE + 2 * len(board_sep) - len(str(t))) // 2, 0)
            added += str(t)

        if len(added) < length_to_place_present:
            added = added + " " * (length_to_place_present - len(added))
        added = added[:length_to_place_present] + present_token + added[length_to_place_present:]

        s = temp + added + "\n" + s
        s += f"PLAYER {(1 - state.player) // 2}\n"
        if state.piece_held != 0:
            s += f"PIECE HELD: {self.piece_to_str(state.piece_held)} from {state.held_piece_origin.numpy()}\n"
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

    def render_action_mask(self, canvas, state: State):
        """
        debug method for printing out action mask
        """
        board_mask, special_mask = self.action_mask(state)
        self.render(
            canvas,
            State(
                board=board_mask.to(torch.int),
                player=state.player,
                center_timeline=state.center_timeline,
                piece_held=state.piece_held,
                held_piece_origin=state.held_piece_origin,
                timestep=state.timestep,
                start_turn_timestep=state.start_turn_timestep,
                board_spawn_timestep=state.board_spawn_timestep,
                prev_start_turn_timestep=state.prev_start_turn_timestep,
            ),
        )
        print("special_actions:", special_mask.numpy())


class Chess2d(Chess5d):
    def __init__(self, stalemate_is_win=False):
        super().__init__(stalemate_is_win=stalemate_is_win)

    def has_special_actions(self):
        return False

    def board_action_dim(self, state):
        return 2

    def _piece_possible_moves(self, board, piece_idx):
        for idx in super()._piece_possible_moves(board, piece_idx):
            if torch.all(torch.eq(idx[:2], piece_idx[:2])):
                yield idx

    def action_mask(self, state):
        board_action_mask, special_action_mask = super().action_mask(state)
        return board_action_mask[-1, 0]

    def step(self, state, action):
        new_action = (
            torch.concatenate((torch.tensor((len(state.board) - 1, 0)), action)),
            torch.tensor(-1),
        )
        new_state, rewards, terminal, aux = super().step(state, new_action)
        # if we have just placed a piece, it is oppoenents turn
        if (not terminal) and (new_state.piece_held == 0):
            new_state, rewards, terminal, auxp = super().step(new_state, (-torch.ones(4, dtype=torch.int), torch.tensor(0)))
            return new_state, rewards, terminal, aux | auxp
        return new_state, rewards, terminal, aux

    def get_game_str(self, state):
        s = ""

        for i in range(self.BOARD_SIZE - 1, -1, -1):
            row = state.board[-1, 0, i, :].cpu().numpy()
            row = "".join(map(self.piece_to_str, row))

            s += str(row)
            s += "\n"
        return s[:-1]

    def step_weak_type(self, state, action):
        """
        allow moves using algebraic notation (string)
        CAPITAL LETTERS MATTER
        i.e. imagine the following setting with a bishop and 3 pawns, where the bishop is on the b rank
        B..
        .p.
        P.P
        then the middle pawn (say on c7) can be captured by the bishop, or the pawn on the same rank (or the other pawn, but ignore this)
        if bishop: Bxc7
        if pawn: bxc7
        """
        if type(action) is str:
            pick_idx, place_idx = self.from_algebraic_notation(state=state, action=action)
            state, _, _, aux = self.step(state, pick_idx)
            state, rwd, term, aux_p = self.step(state, place_idx)
            return state, rwd, term, aux | aux_p
        else:
            return super().step_weak_type(state, action)

    def from_algebraic_notation(self, state: State, action: str, ambiguous=False):
        """
        takes a move in algebraic notation and returns the (pick, place) coordinates
        :param state:
        :param action:
        :param ambiguous: if True,the action might be ambiguous.
            returns a list of all actions that match the algebraic notation
        :return:
        """
        # dont need hints of check, checkmate, or capturing
        action = action.replace("#", "").replace("+", "").replace("x", "")
        if "=" in action:
            # TODO: how to promote to a non-queen, maybe have special moves?
            assert action[-1] == "Q", "promotion to non-queen is not handled yet"
            action = action.split("=")[0]
        if action.lower().startswith("o-o") or action.startswith("0-0"):
            if state.player <= 0:
                rank = 7
            else:
                rank = 0
            kings_file = 4
            if len(action) == 3:
                # O-O
                target_file = kings_file + 2
            else:
                # O-O-O
                target_file = kings_file - 2
            return torch.tensor((rank, kings_file)), torch.tensor((rank, target_file))
        place_square = action[-2:].lower()
        place_idx = torch.tensor([int(place_square[1]) - 1, ord(place_square[0]) - 97])
        pick_hint = action[:-2]
        if pick_hint and pick_hint[0].isupper():
            piece_hint = pick_hint[0]
            square_hint = pick_hint[1:]
        else:
            piece_hint = ""
            square_hint = pick_hint

        # 8x8 mask of which pieces can move
        pick_mask = self.action_mask(state)
        if piece_hint == "":
            hint_mask = torch.logical_or(
                torch.eq(state.board[-1, 0], state.player * self.PAWN), torch.eq(state.board[-1, 0], state.player * self.UNMOVED_PAWN)
            )
        elif piece_hint == "K":
            hint_mask = torch.logical_or(
                torch.eq(state.board[-1, 0], state.player * self.KING), torch.eq(state.board[-1, 0], state.player * self.UNMOVED_KING)
            )
        elif piece_hint == "R":
            hint_mask = torch.logical_or(
                torch.eq(state.board[-1, 0], state.player * self.ROOK), torch.eq(state.board[-1, 0], state.player * self.UNMOVED_ROOK)
            )
        else:
            ident = {
                "N": self.KNIGHT,
                "B": self.BISHOP,
                "Q": self.QUEEN,
            }[piece_hint]
            hint_mask = torch.eq(state.board[-1, 0], state.player * ident)
        pick_mask = torch.logical_and(hint_mask, pick_mask)
        if square_hint:
            possible_squares = torch.ones_like(pick_mask, dtype=torch.bool)
            if square_hint[-1].isdigit():
                mask = torch.eq(torch.arange(self.BOARD_SIZE), int(square_hint[-1]) - 1).reshape(-1, 1)
                possible_squares = torch.logical_and(possible_squares, mask)
            if not square_hint[0].isdigit():
                mask = torch.eq(torch.arange(self.BOARD_SIZE), ord(square_hint[0]) - 97).reshape(1, -1)
                possible_squares = torch.logical_and(possible_squares, mask)
            pick_mask = torch.logical_and(pick_mask, possible_squares)
        out = []
        for pick_idx in zip(*torch.where(pick_mask)):
            pick_idx = torch.tensor(pick_idx)
            full_pick_idx = torch.concatenate((torch.tensor([len(state.board) - 1, 0]), pick_idx))
            for option in self._piece_possible_moves(board=state.board, piece_idx=full_pick_idx):
                if torch.equal(option[2:], place_idx):
                    if ambiguous:
                        out.append((pick_idx, place_idx))
                    else:
                        return pick_idx, place_idx

        assert len(out) > 0, f"{self.get_game_str(state)}\n with this board, no moves found to match " + action
        return out

    def to_algebraic_notation(self, state: State, pick_action, place_action):
        place_hint = chr(97 + place_action[1]) + str(place_action[0].item() + 1)
        temp_state, _, _, _ = self.step(state=state, action=pick_action)
        temp_state, _, _, aux = self.step(state=temp_state, action=place_action)
        # TODO: check for checkmate
        if self.player_in_check(temp_state):
            if self.player_has_non_losing_turn(state=temp_state):
                suffix = "+"
            else:
                suffix = "#"
        else:
            suffix = ""

        if "castled" in aux:
            if aux["castled"] == "left":
                return "O-O-O" + suffix
            else:
                return "O-O" + suffix

        if "capture" in aux:
            place_hint = "x" + place_hint
        if "promotion" in aux:
            place_hint = place_hint + "=" + aux["promotion"]

        pick_hint = self.piece_to_str(state.board[-1, 0, *pick_action]).upper()
        if pick_hint == "P":
            pick_hint = ""
            if "capture" in aux:
                # pawn files are included for captures apparently
                pick_hint = chr(97 + pick_action[1])
        if len(self.from_algebraic_notation(state=state, action=pick_hint + place_hint, ambiguous=True)) > 1:
            for disambiguation in (
                pick_hint + chr(97 + pick_action[1]),
                pick_hint + str(pick_action[0].item() + 1),
                pick_hint + chr(97 + pick_action[1]) + str(pick_action[0].item() + 1),
            ):
                if len(self.from_algebraic_notation(state=state, action=disambiguation + place_hint, ambiguous=True)) == 1:
                    pick_hint = disambiguation
                    break

        return pick_hint + place_hint + suffix
