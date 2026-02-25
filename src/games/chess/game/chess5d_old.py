import copy
import itertools

import numpy as np
import torch
from aleph0.examples.chess.game.board import Board
from aleph0.examples.chess.game.multiverse import Multiverse
from aleph0.examples.chess.game.piece import P
from aleph0.examples.chess.game.timeline import Timeline
from aleph0.game import SelectionGame


class Chess5d(SelectionGame):
    END_TURN = 'END_TURN'
    BLOCKED_PIECE = P.TOTAL_PIECES
    MAX_BORING_MOVES = 50  # TODO: that

    # reserve a separate index for a blocked board (i.e. board that does not exist yet)
    # this is necessary since knights can jump over blocked boards, but other pieces cannot

    def __init__(self,
                 initial_multiverse=None,
                 initial_timeline=None,
                 initial_board=None,
                 current_player=P.P0,
                 first_player=P.P0,
                 save_moves=True,
                 term_ev=None,
                 full_stalemate_check=False,  # TODO: That
                 ):
        """
        implemented 5d chess
        tries to initalize like this:
            if initial_multiverse exists, uses this
            if intial_timeline exists, makes a multiverse with just this timeline
            if initial_board exists, makes a timeline with jsut this board
            otherwise, default board

        :param first_player: which player plays on the first board, default 0 (white)
        :param check_validity: whether to run validity check on each move; default false
        :param save_moves: whether to save moves (useful for undoing moves); default True

        NOTE:
            THE WAY TO TEST FOR CHECKMATE/STALEMATE SHOULD BE THE FOLLOWING:
                if the current player cannot advance the present with any set of moves, stalemate
                if the current player captures opponent king:
                    IF on the opponents last turn, there was a king in check:
                        checkmate
                    OTHERWISE:
                        IF on the opponents last turn, there was no possible sequence of moves to avoid this:
                            stalemate
                        OTHERWISE:
                            checkmate

            the reason we do this is because a 'stalemate test' is expensive
                we have to check all possible sequences of opponent moves
                thus, we do this check maybe once at the end of each game to help runtime
        """
        super().__init__(num_players=2,
                         current_player=current_player,
                         subset_size=2,
                         special_moves=[Chess5d.END_TURN],
                         )
        if initial_multiverse is None:
            if initial_timeline is None:
                if initial_board is None:
                    initial_board = Board()
                initial_timeline = Timeline(board_list=[initial_board])
            initial_multiverse = Multiverse(main_timeline=initial_timeline)
        self.save_moves = save_moves
        self.multiverse = initial_multiverse
        self.first_player = first_player
        self.turn_history = [[]]
        self.passed_indices = []
        self.term_ev = term_ev
        self.full_stalemate_check = full_stalemate_check

    @staticmethod
    def _flip_move(move):
        if move == Chess5d.END_TURN:
            return move
        ((time1, dim1, i1, j1), (time2, dim2, i2, j2)) = move
        return ((time1, -dim1, Board.BOARD_SIZE - 1 - i1, Board.BOARD_SIZE - 1 - j1),
                (time2, -dim2, Board.BOARD_SIZE - 1 - i2, Board.BOARD_SIZE - 1 - j2))

    def flipped(self):
        out = Chess5d(
            initial_multiverse=self.multiverse.flipped(),
            first_player=P.flip_player(self.first_player),
            current_player=P.flip_player(self.current_player),
            save_moves=self.save_moves,
            term_ev=self.term_ev,
            full_stalemate_check=self.full_stalemate_check,
        )
        out.turn_history = [[(Chess5d._flip_move(move), [-dim for dim in dims_spawned])
                             for (move, dims_spawned) in turn]
                            for turn in self.turn_history]
        return out

    def _get_active_number(self, dim_range=None):
        """
        returns number of potential active timelines in each direction

        i.e. if return is 5, any timeline at most 5 away is active
        """
        if dim_range is None:
            dim_range = self.multiverse.get_range()
        return min(dim_range[1], -dim_range[0]) + 1

    def _dim_is_active(self, dim):
        return abs(dim) <= self._get_active_number()

    def _add_move_to_history(self, global_move, dimensions_spawned=None):
        """
        records move to undo later
        Args:
            global_move:
            dimensions_spawned: must be specified if move is not END_TURN
        Returns:

        """
        if global_move == Chess5d.END_TURN:
            self.turn_history.append([])
            if not self.save_moves:
                self._prune_history()
        else:
            self.turn_history[-1].append((global_move, dimensions_spawned))

    def _prune_history(self):
        if len(self.turn_history) > 2:
            self.turn_history = self.turn_history[-2:]

    def _mutate_make_move(self, global_move):
        """
        NOTE: MOVE IS GLOBAL, SO ANY MOVES VIEWED AS AN OPPONENT MUST BE CONVERTED BEFORE CALLING THIS METHOD
        moves piece at idx to end idx
        :param global_move = (idx, end_idx)
            idx: (time, dimension, x, y), must be on an existing board
            end_idx: (time, dimension, x, y), must be to an existing board, and
        this will edit the game state to remove the piece at idx, move it to end_idx
        :return (piece captured (empty if None), terminal move)
            NOTE: terminal move does NOT check for stalemate
            it simply checks if either a king was just captured, or the player has no moves left
        """
        if global_move == Chess5d.END_TURN:
            # TODO: if current player is in check, we terminate
            #  roll back this turn. if current player is in check at the start, game is lost
            #   (i.e. either checkmate or currnet player made dumb move)
            #  otherwise, current player moved into check, must do a stalemate test to see if there was another option
            #  Only other way to draw is if a player has no moves.
            #   in that case, roll back turn and do another stalemate test.
            #       if the player had a valid turn, currne player lost
            #       otherwise, stalemate
            self._add_move_to_history(global_move)
            self.current_player = P.flip_player(self.current_player)
            return P.EMPTY, self.no_moves()

        idx, end_idx = global_move
        time1, dim1, i1, j1 = idx
        time2, dim2, i2, j2 = end_idx
        old_board = self.get_board((time1, dim1))

        dimensions_spawned = []
        if (time1, dim1) == (time2, dim2):
            # here we do not create a new board with the piece removed, as it did no time-space hopping
            new_board, piece, capture = old_board.move_piece_on_board(idx=(i1, j1),
                                                                      end_idx=(i2, j2),
                                                                      mutate=False,
                                                                      )
            if P.piece_id(piece) == P.KING:  # check for castling
                movement = np.max(np.abs(np.array(idx) - np.array(end_idx)))
                if movement > 1:
                    # we have castled, move the rook as well
                    king_movement_dir = np.sign(j2 - j1)
                    rook_j = 0 if king_movement_dir == -1 else Board.BOARD_SIZE - 1
                    new_board, _, _ = new_board.move_piece_on_board(idx=(i1, rook_j),
                                                                    end_idx=(i1, j2 - king_movement_dir),
                                                                    mutate=True,
                                                                    )
            if P.piece_id(piece) == P.PAWN:  # check for en passant
                if abs(i2 - i1) + abs(j2 - j1) == 2:  # captured in xy coords
                    if P.en_passantable(self.get_piece((time1, dim1, i1, j2))):
                        new_board, capture = new_board.remove_piece((i1, j2),
                                                                    mutate=True,
                                                                    )
            new_board.mutate_depassant(just_moved=(i2, j2))

            new_dim = self.add_board_child((time2, dim2), new_board)
            dimensions_spawned.append(new_dim)
        else:
            new_board, piece = old_board.remove_piece((i1, j1), mutate=False)
            piece = P.moved(piece)
            # this is the timeline that piece left behind
            new_board.mutate_depassant(just_moved=None)  # there are no just moved pawns

            new_dim_one = self.add_board_child((time1, dim1), new_board)
            dimensions_spawned.append(new_dim_one)
            newer_board, capture = self.get_board((time2, dim2)).add_piece(piece=piece,
                                                                           square=(i2, j2),
                                                                           mutate=False,
                                                                           )

            # even if this is a pawn, enpassant is not possible with timespace jumps
            newer_board.mutate_depassant(just_moved=None)
            new_dim_two = self.add_board_child((time2, dim2), newer_board)
            dimensions_spawned.append(new_dim_two)

        terminal = P.piece_id(capture) == P.KING or self.no_moves()

        # this is a terminal move if we just captured the king, or if the player that just moved has no moves
        # we know the player that just moved has the next move since the last move was not END_TURN
        self._add_move_to_history(global_move=global_move, dimensions_spawned=dimensions_spawned)
        return capture, terminal

    def undo_move(self):
        """
        undoes last move, changes current player if necessary

        returns move, caputure
            (None, None) if no moves were made
        """
        if not self.turn_history:
            print('WARNING: no moves to undo')
            return (None, None)
        turn = self.turn_history[-1]
        if not turn:
            # then the last move was an end turn
            # we must pop it from turn history and return END_TURN
            # we must also change the current player
            self.current_player = P.flip_player(self.current_player)
            self.turn_history.pop()
            return (Chess5d.END_TURN, None)
        move, dims_spawned = turn.pop()
        idx, end_idx = move
        for dim in dims_spawned:
            # no matter what, a new board is created when the piece moves from the board at idx
            # if there is a time dimension jump, another board is created from the board at end_idx
            self.multiverse.remove_board(dim)
        return (move, self.get_piece(end_idx))

    def undo_turn(self, include_boundary=True):
        """
        undoes entire turn of the current player, changes player to opponent if include_boundary is true
        Args:
            include_boundary: whether to include END_TURN
                if true, self.currnet_player will switch to opponent

        returns iterable of (move, captured piece) in correct order
        """
        if not self.turn_history:
            print('WARNING: no turns to undo')
            return None, None
        current_player_moves = len(self.turn_history[-1])
        turn = []
        for _ in range(current_player_moves):
            move, caputure = self.undo_move()
            turn.append((move, caputure))
        if include_boundary:
            self.undo_move()  # this is the END_MOVE token, we can ignore it
        return turn[::-1]  # reverse the history, since we put the moves on in reversed order

    def get_board(self, global_td_idx):
        return self.multiverse.get_board(global_td_idx)

    def get_piece(self, global_idx):
        (time, dim, i, j) = global_idx
        board = self.get_board((time, dim))
        if board is not None:
            return board.get_piece((i, j))

    def idx_exists(self, global_td_idx, global_ij_idx=(0, 0)):
        i, j = global_ij_idx
        if i < 0 or j < 0 or i >= Board.BOARD_SIZE or j >= Board.BOARD_SIZE:
            return False
        return self.multiverse.idx_exists(td_idx=global_td_idx)

    def add_board_child(self, global_td_idx, board):
        """
        adds board as a child to the board specified by td_idx
        :param global_td_idx: (time, dimension)
        :param board: board to add
        Returns: dimenison spawned (idx)
        """
        time, dim = global_td_idx

        if not self.idx_exists((time + 1, dim)):
            # in this case dimension does not change
            new_dim = dim
        else:
            player = self.player_at(time=time)
            overall_range = self.multiverse.get_range()
            if player == P.P0:  # white move
                new_dim = overall_range[0] - 1
            else:  # black move
                new_dim = overall_range[1] + 1

        self.multiverse.add_board((time + 1, new_dim), board)
        return new_dim

    def board_can_be_moved(self, global_td_idx):
        """
        returns if the specified board can be moved from
        """
        time, dim = global_td_idx
        # the board itself must exist
        # if the next board exixts, a move has been made, otherwise, no move was made
        return self.idx_exists((time, dim)) and (not self.idx_exists((time + 1, dim)))

    def boards_with_possible_moves(self):
        """
        iterable of time-dimension coords of boards where a piece can potentially be moved

        equivalent to the leaves of the natural tree

        WILL ALWAYS RETURN BOARDS IN SAME ORDER
        """
        for td_idx in self.multiverse.leaves():
            yield td_idx

    def _players_boards_with_possible_moves(self, player):
        """
        WILL ALWAYS RETURN BOARDS IN SAME ORDER
        """
        for (t, d) in self.boards_with_possible_moves():
            if self.player_at(time=t) == player:
                yield (t, d)

    def _piece_possible_moves(self, global_idx, castling=True):
        """
        returns possible moves of piece at idx
        :param global_idx: (time, dim, i, j)
        :return: iterable of idx candidates

        WILL ALWAYS RETURN MOVES IN SAME ORDER
        """
        idx_time, idx_dim, idx_i, idx_j = global_idx
        piece = self.get_board((idx_time, idx_dim)).get_piece((idx_i, idx_j))
        pid = P.piece_id(piece)
        q_k_dims = itertools.chain(*[itertools.combinations(range(4), k) for k in range(1, 5)])

        if pid in (P.ROOK,
                   P.BISHOP,
                   P.UNICORN,
                   P.DRAGON,
                   P.PRINCESS,
                   P.QUEEN,
                   P.KING,
                   ):  # easy linear moves
            if pid == P.ROOK:
                dims_to_change = itertools.combinations(range(4), 1)
            elif pid == P.BISHOP:
                dims_to_change = itertools.combinations(range(4), 2)
            elif pid == P.UNICORN:
                dims_to_change = itertools.combinations(range(4), 3)
            elif pid == P.DRAGON:
                dims_to_change = itertools.combinations(range(4), 4)
            elif pid == P.PRINCESS:
                # combo of rook and bishop
                dims_to_change = itertools.chain(itertools.combinations(range(4), 1),
                                                 itertools.combinations(range(4), 2))
            else:
                dims_to_change = q_k_dims
            for dims in dims_to_change:
                for signs in itertools.product((-1, 1), repeat=len(dims)):
                    pos = [idx_time, idx_dim, idx_i, idx_j]
                    vec = np.array((0, 0, 0, 0))
                    for k, dim in enumerate(dims):
                        vec[dim] = signs[k]*((dim == 0) + 1)  # mult by 2 if dim is time
                    pos += vec
                    while (self.idx_exists(pos[:2], pos[2:]) and
                           (P.player_of(piece) != P.player_of(self.get_piece(pos)))):
                        yield tuple(tt.item() for tt in pos)
                        if (P.player_of(self.get_piece(pos)) is not None) or pid == P.KING:
                            # end of the line, or the king which moves single spaces
                            break
                        pos += vec
        if pid == P.KNIGHT:
            dims_to_change = itertools.permutations(range(4), 2)
            for dims in dims_to_change:
                for signs in itertools.product((-1, 1), repeat=len(dims)):
                    pos = [idx_time, idx_dim, idx_i, idx_j]
                    for k, dim in enumerate(dims):
                        # multiply one of the dimensions by 1 and one by 2
                        # can do this with *(k+1)
                        pos[dim] += (k + 1)*signs[k]*((dim == 0) + 1)
                    if self.idx_exists(pos[:2], pos[2:]) and (
                            P.player_of(piece) != P.player_of(self.get_piece(pos))):
                        yield tuple(pos)
        if pid == P.PAWN:
            player = self.player_at(time=idx_time)
            dir = P.player_to_sign(player)
            # forward moves
            for dim in (2, 1):
                pos = [idx_time, idx_dim, idx_i, idx_j]
                for _ in range(1 + P.is_unmoved(piece)):
                    pos[dim] += dir
                    if self.idx_exists(pos[:2], pos[2:]) and (P.player_of(self.get_piece(pos)) is None):
                        yield tuple(pos)
                    else:
                        break
            # diag moves
            for dims in ((2, 3), (1, 0)):
                for aux_sign in (-1, 1):
                    pos = [idx_time, idx_dim, idx_i, idx_j]
                    pos[dims[0]] += dir
                    pos[dims[1]] += aux_sign
                    if (self.idx_exists(pos[:2], pos[2:]) and
                            (P.player_of(self.get_piece(pos)) is not None) and
                            (P.player_of(piece) != P.player_of(self.get_piece(pos)))):
                        # this MUST be a capture
                        yield tuple(pos)
            # en passant check
            for other_j in (idx_j + 1, idx_j - 1):
                if self.idx_exists((idx_time, idx_dim), (idx_i, other_j)):
                    other_piece = self.get_piece((idx_time, idx_dim, idx_i, other_j))
                    if P.player_of(other_piece) != P.player_of(piece) and P.en_passantable(other_piece):
                        yield (idx_time, idx_dim, idx_i + dir, other_j)

        # castling check
        if castling and pid == P.KING and P.is_unmoved(piece):
            # for rook_i in (0, Board.BOARD_SIZE - 1):
            rook_i = idx_i  # rook must be on same rank
            for rook_j in (0, Board.BOARD_SIZE - 1):
                # potential rook squares
                rook_maybe = self.get_piece((idx_time, idx_dim, rook_i, rook_j))
                if P.player_of(rook_maybe) == P.player_of(piece):
                    if P.piece_id(rook_maybe) == P.ROOK and P.is_unmoved(rook_maybe):
                        dir = np.sign(rook_j - idx_j)
                        works = True

                        rook_dist = abs(rook_j - idx_j)
                        # if there is a piece blocking a middle square, castling this way is bad
                        if any(P.player_of(self.get_piece((idx_time, idx_dim, rook_i, idx_j + dir*k)))
                               is not None for k in range(1, rook_dist)):
                            continue

                        king_hop_squares = {(idx_time, idx_dim, rook_i, idx_j + dir*k) for k in range(1, 3)}
                        # if there is a piece attacking a square the king hops over
                        # for some reason time travel is ignored in the game
                        for square in self.attacked_squares(player=P.flip_player(P.player_of(piece)),
                                                            time_travel=False):
                            if square in king_hop_squares:
                                works = False
                                break
                        if works:
                            yield (idx_time, idx_dim, idx_i, (idx_j + 2*dir).item())

    def _board_all_possible_moves(self, global_td_idx, castling=True):
        """
        WILL ALWAYS RETURN MOVES IN SAME ORDER
        """
        t, d = global_td_idx
        board = self.get_board(global_td_idx)
        for (i, j) in board.pieces_of(self.player_at(t)):
            idx = (t, d, i, j)
            for end_idx in self._piece_possible_moves(idx, castling=castling):
                yield idx, end_idx

    def _all_possible_selection_moves(self, player, castling=True):
        """
        returns an iterable of all possible moves of the specified player
        if player is None, uses the first player that needs to move
        if the opponent is in check, the player MUST capture the king

        WILL ALWAYS RETURN MOVES IN SAME ORDER
        """
        if player is None:
            player = self.player_at(time=self.present())
        in_check = False
        all_moves = []
        for td_idx in self._players_boards_with_possible_moves(player=player):
            for global_move in self._board_all_possible_moves(global_td_idx=td_idx, castling=castling):
                index, end_idx = global_move
                if P.piece_id(self.get_piece(end_idx)) == P.KING:
                    in_check = True
                    yield global_move
                if not in_check:
                    all_moves.append(global_move)
        # if there are no check moves, just go through all possible moves
        if not in_check:
            for global_move in all_moves:
                yield global_move

    def _all_possible_turn_subsets(self, player=None):
        """
        returns an iterable of all possible turns subsets of the specified player
            if player is None, uses the first player that needs to move

        this problem is equivalent to the following:
            consider the set of all possible moves, ignoring those that do not move to a currently active board
                (i.e. ignore any moves that always spawn a new timeline)
            break these into equivalence classes by their start/end dimensions
            DAG subgraphs in this graph where each vertex is the source of at most one edge are exactly equivalent to
                all subsets of moves that are possible. (we can additionally add back any of the ignored moves so that
                    each board is the source of at most one move)

            first, for any DAG subgraph of this type we can always do all of the moves
                    this is true since we can topologically sort them then do them from the bottom up
                    this works bc a move is impossible if the board it starts from has already been moved to
                    since we are going backwards through the DAG, each starting board cannot have been moved to
            additionally, we can add any subset of the ignored moves to this set of moves and have the same property
                    this is true since we can just do all the ignored moves first.
                        since no ignored move ends at a potential 'starting' board, this will not interfere with anything
            Thus, the moves corresponding to a DAG subgraph of the graph where each vertex is the source of one edge
                unioned with any of the ignored moves is a valid set of moves

            for the other direction, consider a valid set of moves. Since they are valid, they must have some order
                that allows them to be valid in the game
            This order corresponds to a topological sort of the edges in our constructed directed graph. This proves
                that the graph is a DAG since in the game, a board that is the end of a move cannot start a new move
                additionally, each vertex is the source of exactly one edge, as there is only one move per board

        NOTE: it is not true that the result of a set of moves are independent of the order
            if we spawn two new timelines, the order we did the moves determines the numbers of the timelines
            thus, order does matter for the result of the moves

        Thus, for this method, we eventually have to check all possible permutations of these edges

        NOTE: THIS IS ONLY NEEDED TO CALL ONCE PER GAME AS A STALEMATE CHECK,
            TO CHECK IF A CAPTURE OF A KING COULD HAVE BEEN AVOIDED

            PLAYING THE GAME LIKE THIS WILL BE BETTER FOR RUNTIME
        """

        def all_DAG_subgraphs_with_property(edge_list: dict, used_sources=None, used_vertices=None):
            """
            iterable of all lists of edges that create a DAG such that each vertex is the source of at most one edge
                (we run through all permutations, sort of wasteful)
            the order returned will be in reverse topological order (the correct traversal)

            :param edge_list: dict(vertex -> vertex set), must be copyable
            :return: iterable of (list[(start vertex, end vertex)], used vertices)
            """
            if used_sources is None:
                used_sources = set()
            if used_vertices is None:
                used_vertices = set()
            yield (), used_sources, used_vertices
            for source in edge_list:
                if source not in used_vertices:
                    for end in edge_list[source]:
                        for (subsub, all_source, all_used) in all_DAG_subgraphs_with_property(edge_list=edge_list,
                                                                                              used_sources=
                                                                                              used_sources.union(
                                                                                                  {source}),
                                                                                              used_vertices=
                                                                                              used_vertices.union(
                                                                                                  {source, end}),
                                                                                              ):
                            yield (((source, end),) + subsub,
                                   all_source, all_used)

        if player is None:
            player = self.player_at(self.present())

        # we will make a graph as in the description
        # this does not change theoretic runtime, but will in practice probably help a lot

        possible_boards = set(self._players_boards_with_possible_moves(player=player))

        # this will be a list of lists
        # all moves on each board playable board
        partition = dict()  # partitions all moves by start, end board
        edge_list = {td_idx: set() for td_idx in
                     possible_boards}  # edges we care about, just the edges between active boards
        non_edges = {td_idx: set() for td_idx in possible_boards}  # other moves, active -> inactive boards
        for td_idx in possible_boards:
            for move in self._board_all_possible_moves(global_td_idx=td_idx):
                start_idx, end_idx = move
                end_td_idx = end_idx[:2]
                if end_td_idx in possible_boards and end_td_idx != td_idx:
                    edge_list[td_idx].add(end_td_idx)
                else:
                    non_edges[td_idx].add(end_td_idx)
                equiv_class = (td_idx, end_td_idx)
                if equiv_class not in partition:
                    partition[equiv_class] = set()
                partition[equiv_class].add(move)
        for edges, used_sources, used_boards in all_DAG_subgraphs_with_property(edge_list=edge_list):
            # we must also add moves from the non-edges
            other_boards = possible_boards.difference(used_sources)
            for i in range(len(other_boards) + 1):
                for other_initial_boards in itertools.combinations(other_boards, i):
                    # other initial boards is a list of td_idxes
                    for other_final_boards in itertools.product(
                            *(non_edges[td_idx] for td_idx in other_initial_boards)):
                        other_edges = tuple(zip(other_initial_boards, other_final_boards))
                        # this is now a list of (td_idx start, td_idx end)

                        # this is now a list of (td_idx start, td_idx end) in correct order
                        order = other_edges + edges
                        # we will now choose moves of each class, and send them out
                        for move_list in itertools.product(*(partition[equiv_class] for equiv_class in order)):
                            yield move_list

    def all_possible_turn_sets(self, player=None):
        """
        returns all sets of moves that player can take in order to advance present
        """

        if player is None:
            player = self.player_at(self.present())
        sign = P.player_to_sign(player)  # direction of dimensions, 1 for black, -1 for white
        possible_boards = set(self._players_boards_with_possible_moves(player=player))
        possible_presents = set(self.boards_with_possible_moves())

        for moves in self._all_possible_turn_subsets(player=player):
            possible_gifts = possible_presents.copy()  # keep track of possible presents

            dim_range = list(self.multiverse.get_range())
            used_dims = set()

            for (_, dim1, _, _), (time2, dim2, _, _) in moves:
                used_dims.add(dim1)

                if dim2 in used_dims or (time2, dim2) not in possible_boards:
                    # here, we split
                    dim_range[player] += sign
                    possible_gifts.add((time2 + 1, dim_range[player]))
                    used_dims.add(dim_range[player])
                else:
                    used_dims.add(dim2)
            active_number = self._get_active_number(dim_range=dim_range)
            present = min(time for time, dim in possible_gifts if abs(dim) <= active_number)
            success = True
            for time, dim in possible_boards:
                if dim not in used_dims:
                    if abs(dim) <= active_number:
                        if time <= present:
                            success = False
                            break

            if success:
                yield moves

    def no_moves(self):
        """
        returns if no possible moves
        """
        try:
            # why is there no better way to check if a generator is empty
            next(self.get_all_valid_moves())
            return False
        except StopIteration:
            return True

    def attacked_squares(self, player, time_travel=True):
        """
        returns an iterable of all squares attacked by player (with repeats, as there is no fast way to filter these)
        :param time_travel: whether or not to consider time travel (sometimes this is false for castling and stuff)
        """
        # we do not want castling as you cannot capture a piece with that
        # this also causes an infinite loop, as to check for castling, we must use this method
        for move in self._all_possible_selection_moves(player=player, castling=False):
            if move != Chess5d.END_TURN:
                start_idx, end_idx = move
                if time_travel == False and start_idx[:2] == end_idx[:2]:
                    # in this case, we ignore since time travel is not considered
                    continue
                yield end_idx

    def pass_all_present_boards(self):
        """
        does a PASS_TURN on all present boards
        """
        present = self.present()

        for td_idx in self.boards_with_possible_moves():
            time, dim = td_idx
            if time == present and self._dim_is_active(dim=dim):
                board = self.get_board(td_idx)
                self.add_board_child(global_td_idx=td_idx, board=board.clone())
                self.passed_indices.append((time + 1, dim))

    def undo_passed_boards(self):
        """
        undoes all the PASS_TURNs
        """
        for time, dim in self.passed_indices:
            self.multiverse.remove_board(dim)
        self.passed_indices = []

    def present_player_in_check(self):
        """
        returns if current player is in check

        first, the current player must 'pass' on all present boards

        then, this iterates over all opponent attacked squares and returns if any of them are a king
            this is consistent with the definition of check, as a capture of ANY opponent king is a win
        """
        present_player = self.player_at(self.present())
        self.pass_all_present_boards()
        for idx in self.attacked_squares(player=P.flip_player(present_player)):
            if P.piece_id(self.get_piece(global_idx=idx)) == P.KING:
                # if player_of(self.get_piece(idx=idx))==player
                # this check is unnecessary, as opponent cannot move on top of their own king
                self.undo_passed_boards()
                return True
        self.undo_passed_boards()
        return False

    def present_player_can_win(self):
        """
        returns if current player can win
            i.e. if a king is capturable by the current player on the next move
        """
        present_player = self.player_at(self.present())
        for idx in self.attacked_squares(player=present_player):
            if P.piece_id(self.get_piece(global_idx=idx)) == P.KING:
                return True
        return False

    def is_checkmate_or_stalemate(self, player=None):
        """
        returns if it is (checkmate or stalemate) for specified player
            if player is None, uses currnet player
        checks if for all possible moves of the player, the player still ends up in check
        note that this works in the case where player has no moves as well
        """
        if player is None:
            player = self.player_at(self.present())

        for moveset in self.all_possible_turn_sets(player=player):
            # TODO: WE DO NEED TO PERMUTE, BUT IS THERE A WAY TO DO THIS LATER?
            # FOR THE MOST PART, PERMUTATIONS SHOULD NOT HELP GETTING OUT OF CHECK

            # for moves in itertools.permutations(moveset):
            for moves in (moveset,):  # incorrect technically, for correctness use earlier line
                temp_game = self.clone()
                failed = False
                for move in moves:
                    # if we are permuting, it is possible the permutation is invalid. in this case, we do not need to
                    # inspect if the player is in check
                    if temp_game.board_can_be_moved(move[0][:2]):
                        temp_game._mutate_make_move(move)
                    else:
                        failed = True
                        break
                if failed:
                    break
                if not temp_game.present_player_can_win():
                    # then there exists a sequence of moves that keeps the player out of check
                    return False
        return True

    def _terminal_eval(self, mutation=False):
        """
        to be run if game has just ended (king was captured, or no moves left)

        This during terminal evalution one of two things can happen
            The player captures the opponent king:
                In this case, we must undo the current player's turn and the
                opponent's previous turn. We then inspect the game to see if this position is checkmate or stalemate
                    (i.e. if every possible opponent turn leads to check, it is checkmate or stalemate.
                    Otherwise, the opponent made a silly move and lost)
                    the distinction between checkmate and stalemate is whether the opponent is in check
                    (i.e. if the opponent makes no moves, can the current player capture the opponent king)
            The current player has no available moves:
                in this case, we will undo the current player's turn and check for checkmate or stalemate
                    (i.e. if every possible player turn leads to check, it is checkmate or stalemate,
                    and we will return a loss or tie based on if the player is in check)
                If there was a valid turn the player did not find, we will count this as a loss for the player

        in either case, we must undo at most two turns. Thus, the move history needs at most a record of two turns



        :param mutation: whether to let game state be mutated
        :return: (white score, black score)
        """
        if mutation:
            game = self
        else:
            game = self.clone()

        last_player = game.current_player
        opponent = P.flip_player(last_player)
        # no_moves_left = game.no_moves(player=last_player)  # if the last player had remaining moves

        last_turn = game.undo_turn(include_boundary=True)
        king_captured = P.KING in {P.piece_id(captured) for _, captured in last_turn}
        result = [0, 0]
        if king_captured:
            # here the king was captured so either the last player won or drew
            # if the opponent (previous turn) was in check, this is a win
            # if the opponent (previous turn) was not in check:
            #   if there was a turn that got opponent out of check, this is a loss for opponent (win for last player)
            #   otherwise, a draw
            opponent_turn = game.undo_turn(include_boundary=False)
            if game.present_player_in_check():
                # loss for opponent, win for last player
                result[P.player_to_idx(last_player)] = 1
                result[P.player_to_idx(opponent)] = 0
            else:
                if game.is_checkmate_or_stalemate(player=opponent):
                    # opponent was not in check, and all moves led to check, so this is stalemate
                    result[P.player_to_idx(last_player)] = .5
                    result[P.player_to_idx(opponent)] = .5
                else:
                    # there was a move to get out of check, it was unfound, so opponent lost
                    result[P.player_to_idx(last_player)] = 1
                    result[P.player_to_idx(opponent)] = 0
        else:
            # if not no_moves_left:
            #    raise Exception("TERMINAL EVAL CALLED ON NON-TERMINAL STATE")
            # here the king was not captured, so last player must have had no moves left
            # either a loss for last player (there was a valid turn that doesnt lead to check)
            # or a draw (all valid turns lead to check)
            if game.is_checkmate_or_stalemate(player=last_player):
                # there are no valid moves to get out of check
                if game.present_player_in_check():
                    # if last player was in check at the start of their turn, this is a loss
                    # since either last player is in checkmate, or last player moved into check again
                    result[P.player_to_idx(last_player)] = 0
                    result[P.player_to_idx(opponent)] = 1
                else:
                    # if there was no check, this is stalemate, so a draw
                    result[P.player_to_idx(last_player)] = .5
                    result[P.player_to_idx(opponent)] = .5
            else:
                # there was a valid sequence of moves that did not lead to check, unfound by last player
                # thus, last player lost
                result[P.player_to_idx(last_player)] = 0
                result[P.player_to_idx(opponent)] = 1
        return tuple(result)

    def player_at(self, time=None):
        """
        returns which players turn it is
            if a player does not NEED to move, it is not their turn
        0 for first player
        1 for second player
        :param time: if None, uses present
        """
        if time is None:
            time = self.present()
        # if first player is 0, return time%2
        # if first player is 1, return (time+1)%2
        return (time + self.first_player)%2

    def present(self):
        """
        returns the time index of the present
        """
        return min(t for (t, d) in self.boards_with_possible_moves() if self._dim_is_active(d))

    def convert_to_local_move(self, global_move):
        if global_move == Chess5d.END_TURN:
            return Chess5d.END_TURN
        else:
            return tuple(self.convert_to_local_idx(global_idx)
                         for global_idx in global_move)

    def convert_to_global_move(self, local_move):
        if local_move == Chess5d.END_TURN:
            return Chess5d.END_TURN
        else:
            return tuple(self.convert_to_global_idx(local_idx)
                         for local_idx in local_move)

    def convert_to_local_idx(self, global_idx):
        """
        flips index if player is black
        Args:
            global_idx: index to flip
        """
        (t, d, i, j) = global_idx
        overall_range = self.multiverse.get_range()
        if self.current_player == P.P0:
            return (t, d - overall_range[0], i, j)
        else:
            I, J = Board.BOARD_SHAPE
            return (t, overall_range[1] - d, I - i - 1, J - j - 1)

    def convert_to_global_idx(self, local_idx):
        """
        flips index if player is black
        Args:
            local_idx: index to flip
        """
        (t, d, i, j) = local_idx
        overall_range = self.multiverse.get_range()
        if self.current_player == P.P0:
            return (t, d + overall_range[0], i, j)
        else:
            I, J = Board.BOARD_SHAPE
            return (t, overall_range[1] - d, I - i - 1, J - j - 1)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CLASS METHODS                                                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @property
    def permutation_to_standard_pos(self):
        if self.current_player == P.P1:
            return [1, 0]
        else:
            return [0, 1]

    def clone(self):
        game = Chess5d(initial_multiverse=self.multiverse.clone(),
                       first_player=self.first_player,
                       save_moves=self.save_moves,
                       current_player=self.current_player,
                       full_stalemate_check=self.full_stalemate_check
                       )
        game.turn_history = copy.deepcopy(self.turn_history)
        game._prune_history()
        return game

    @property
    def observation_shape(self):
        """
        observation is shapes ((D1,...,DN),(D1,...,DN),(D1,...,DN),(D1,...,DN)), (D1,...,DN, N), T)
        this method returns those shapes
        """
        overall_range = self.multiverse.get_range()
        time_len = self.multiverse.max_length
        dimensions = 1 + overall_range[1] - overall_range[0]
        I, J = Board.BOARD_SHAPE

        return tuple((time_len, dimensions, I, J) for _ in range(4)), (time_len, dimensions, I, J, 4), (0,)

    def valid_selection_moves(self):
        """
        gets all possible moves
        Returns:
            iterable of (self.subset_size tuples of N tuples)
        """
        for global_move in self._all_possible_selection_moves(player=self.current_player, castling=True):
            yield self.convert_to_local_move(global_move=global_move)

    def valid_special_moves(self):
        """
        returns iterable of special moves possible from current position
        MUST BE DETERMINISTIC, always return moves in same order
        Returns: boolean
        """
        if self.player_at(time=self.present()) != self.current_player:
            # the player does not have to move
            yield Chess5d.END_TURN

    @property
    def observation(self):
        """
        observation is shapes ((D1,...,D4),(D1,...,D4),(D1,...,D4),(D1,...,D4)), (D1,...,D4, 4), 0)

        the boards are
            (the pieces (including empty and blocked squares),
            whether each square is in an active timelin,
            whether each square is on a movable board (i.e. a leaf of a timeline),
            the current player of each square,
            )
        indexes are normal indices except for the dimension index, which is centered at the origin dimension
        Returns:

        """
        if self.current_player == P.P1:
            game = self.flipped()
        else:
            game = self.clone()
        board_shapes, index_shape, vec_shape = game.observation_shape
        (time_len, dimensions, xlen, ylen) = index_shape[:4]

        # create boards
        overall_range = self.multiverse.get_range()
        # by default assume each square is blocked
        piece_board = self.BLOCKED_PIECE*torch.ones((time_len, dimensions, xlen, ylen), dtype=torch.long)
        # by default, all timelines are inactive
        active_board = torch.zeros((time_len, dimensions, xlen, ylen), dtype=torch.long)
        # by default no boards are movable
        movable_board = torch.zeros((time_len, dimensions, xlen, ylen), dtype=torch.long)

        for i, dim_idx in enumerate(range(overall_range[0], overall_range[1] + 1)):
            timeline = self.multiverse.get_timeline(dim_idx=dim_idx)
            # set the pieces of the boards in this timeline
            piece_board[timeline.start_idx:timeline.end_time() + 1, i, :, :] = timeline.get_board_as_idxs_stack()
            if self._dim_is_active(dim_idx):
                # set this dimension to active if it is active
                active_board[:, i, :, :] = 1
            # set the leaf to movable
            movable_board[timeline.end_time(), i, :, :] = 1
        players = (self.first_player + torch.arange(time_len).reshape(-1, 1, 1, 1))%2
        player_board = players.broadcast_to((time_len, dimensions, xlen, ylen))

        # create index set
        dim_range = game.multiverse.get_range()
        T, D, X, Y = (
            (torch.arange(time_len),
             torch.arange(dimensions),
             torch.arange(xlen),
             torch.arange(ylen))
        )
        # dim range is (bottom dim, top dim)
        # the true dim index should start at bottom dim, so we need to add that on
        D += dim_range[0]

        T = torch.cat((
            T.view((time_len, 1, 1, 1, 1)),
            torch.zeros((time_len, 1, 1, 1, 3)),
        ), dim=-1)

        D = torch.cat((
            torch.zeros((1, dimensions, 1, 1, 1)),
            D.view((1, dimensions, 1, 1, 1)),
            torch.zeros((1, dimensions, 1, 1, 2)),
        ), dim=-1)
        X = torch.cat((
            torch.zeros((1, 1, xlen, 1, 2)),
            X.view((1, 1, xlen, 1, 1)),
            torch.zeros((1, 1, xlen, 1, 1)),
        ), dim=-1)
        Y = torch.cat((
            torch.zeros((1, 1, 1, ylen, 3)),
            Y.view((1, 1, 1, ylen, 1)),
        ), dim=-1)

        return (piece_board, active_board, movable_board, player_board), T + D + X + Y, torch.zeros(vec_shape)

    @staticmethod
    def num_observation_boards():
        """
        Returns: number of boards in (D1,...,DN, *1),(D1,...,DN, *2),...)
        """
        return 4

    @staticmethod
    def underlying_set_sizes():
        """
        returns number of possible distinct elements of each underlying set, if finite
        """
        return P.TOTAL_PIECES + 1, 2, 2, 2

    @property
    def representation(self):
        """
        Returns: representation of self, likely a tuple of tensors
            often this is the same as self.observation, (i.e. for perfect info games)
        all information required to play the game must be obtainable from representation
        i.e. self.from_represnetation(self.represnetation) must be functionally equivalent to self

        should return clones of any internal variables
        """
        return (self.multiverse.representation,
                self.current_player,
                self.first_player,
                copy.deepcopy(self.turn_history),
                self.term_ev,
                )

    @staticmethod
    def from_representation(representation):
        """
        returns a SubsetGame instance from the output of self.get_representation
        Args:
            representation: output of self.get_representation
        Returns: SubsetGame object
        """
        mult_rep, current_player, first_player, turn_history, term_ev = representation
        game = Chess5d(initial_multiverse=Multiverse.from_representation(mult_rep),
                       current_player=current_player,
                       first_player=first_player,
                       term_ev=term_ev,
                       full_stalemate_check=False,
                       )
        game.turn_history = turn_history
        return game

    def make_move(self, local_move):
        out = self.clone()
        global_move = self.convert_to_global_move(local_move)
        capture, terminal = out._mutate_make_move(global_move)
        if terminal:
            out.term_ev = out._terminal_eval(mutation=False)
        return out

    def is_terminal(self):
        """
        returns if current game has terminated
        CANNOT BE PROBABILISTIC
            if there is a probabilistic element to termination,
                the probabilities must be calculated upon creation of this object and stored
        Returns: boolean
        """
        return self.term_ev is not None

    def get_result(self):
        """
        can only be called on terminal states
        returns an outcome for each player
        Returns: K-tuple of outcomes for each player
            outcomes are generally in the range [0,1] and sum to 1
            i.e. in a 1v1 game, outcomes would be (1,0) for p0 win, (0,1) for p1, and (.5,.5) for a tie
            in team games this can be changed to give teammates the same reward, and have the sum across teams be 1
        """
        return self.term_ev

    def __str__(self):
        return self.multiverse.__str__()

    def render(self):
        if self.current_player == P.P1:
            game = self.flipped()
        else:
            game = self
        print(game.__str__())


if __name__ == '__main__':
    from aleph0.algs import Human, play_game

    chess = Chess5d()
    chess = chess.make_move(next(chess.get_all_valid_moves()))
    chess = chess.make_move(next(chess.get_all_valid_moves()))
    chess = chess.make_move(next(chess.get_all_valid_moves()))

    print(chess)
    rep = chess.representation
    print(Chess5d.from_representation(rep))
    # TODO: check termination eval
    chess = Chess5d()
    chess = chess.make_move(next(chess.get_all_valid_moves()))
    chess = chess.make_move(next(chess.get_all_valid_moves()))
    chess = chess.make_move(next(chess.get_all_valid_moves()))
    chess = chess.make_move(next(chess.get_all_valid_moves()))
    chess = chess.make_move(((2, 0, 2, 0), (0, 0, 4, 0)))
    chess = chess.make_move(next(chess.get_all_valid_moves()))
    print(play_game(game=chess, alg_list=[Human(), Human()]))
    chess = Chess5d()
    chess = chess.make_move(((0, 0, 1, 4), (0, 0, 3, 4)))
    chess = chess.make_move(next(chess.get_all_valid_moves()))
    chess = chess.make_move(next(chess.get_all_valid_moves()))
    chess = chess.make_move(next(chess.get_all_valid_moves()))
    chess = chess.make_move(((2, 0, 0, 3), (2, 0, 4, 7)))

    chess = chess.make_move(next(chess.get_all_valid_moves()))
    chess = chess.make_move(((3, 0, 1, 2), (3, 0, 2, 2)))
    chess = chess.make_move(next(chess.get_all_valid_moves()))
    chess = chess.make_move(((4, 0, 4, 7), (4, 0, 7, 4)))
    chess.render()
    print(chess.is_terminal())
    chess.undo_turn()
    chess.render()
    quit()
