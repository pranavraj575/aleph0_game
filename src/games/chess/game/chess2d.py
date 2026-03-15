import copy

import torch
from aleph0.examples.chess.game.board import Board
from aleph0.examples.chess.game.chess5d import Chess5d
from aleph0.examples.chess.game.multiverse import Multiverse
from aleph0.examples.chess.game.piece import P
from aleph0.examples.chess.game.timeline import Timeline
from aleph0.game import FixedSizeSelectionGame


class Chess2d(Chess5d, FixedSizeSelectionGame):
    def __init__(
        self,
        initial_board=None,
        initial_timeline=None,
        current_player=P.P0,
        first_player=P.P0,
        save_moves=True,
        term_ev=None,
    ):
        super().__init__(
            initial_board=initial_board,
            initial_timeline=initial_timeline,
            current_player=current_player,
            first_player=first_player,
            save_moves=save_moves,
            term_ev=term_ev,
        )

    def _piece_possible_moves(self, global_idx, castling=True):
        # only return moves that do not jump time-dimensions
        for end_idx in super()._piece_possible_moves(global_idx, castling=castling):
            if end_idx[:2] == global_idx[:2]:
                yield end_idx

    def material_draw(self):
        board = self.get_current_board()
        if self.current_player_in_check():
            # then it is not a draw
            return False
        for idx in board.all_pieces():
            piece = board.get_piece(idx)
            if P.piece_id(piece=piece) != P.KING:
                return False
        return True

    def get_current_board(self):
        return self.multiverse.get_board((self.multiverse.max_length - 1, 0))

    def get_current_timeline(self):
        return self.multiverse.get_timeline(0)

    def wrap_move(self, chess2dmove, td_idx=None):
        if td_idx is None:
            td_idx = (self.multiverse.max_length - 1, 0)
        idx, end_idx = chess2dmove
        return td_idx + idx, td_idx + end_idx

    def wrap_idx(self, chess2didx, td_idx=None):
        if td_idx is None:
            td_idx = (self.multiverse.max_length - 1, 0)
        return td_idx + chess2didx

    def unwrap_idx(self, chess5didx):
        return chess5didx[2:]

    def convert_to_local_idx(self, global_idx):
        local_5d_idx = super().convert_to_local_idx(global_idx=global_idx)
        return self.unwrap_idx(chess5didx=local_5d_idx)

    def convert_to_global_idx(self, local_idx):
        local_5d_idx = self.wrap_idx(local_idx)
        return super().convert_to_global_idx(local_idx=local_5d_idx)

    def flipped(self):
        out = Chess2d(
            initial_timeline=self.get_current_timeline().flipped(),
            current_player=P.flip_player(self.current_player),
            first_player=P.flip_player(self.first_player),
            save_moves=self.save_moves,
            term_ev=self.term_ev,
        )
        out.turn_history = [
            [
                (Chess5d._flip_move(move), [-dim for dim in dims_spawned])
                for (move, dims_spawned) in turn
            ]
            for turn in self.turn_history
        ]
        return out

    def prune_timeline(self):
        range = self.get_current_timeline().get_time_range()
        if range[1] - range[0] > 3:
            board_list = self.get_current_timeline().board_list
            start_idx = range[1] - 3
            self.multiverse = Multiverse(
                main_timeline=Timeline(
                    board_list=board_list[-3:],
                    start_idx=start_idx,
                )
            )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CLASS METHODS                                                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def clone(self):
        game = Chess2d(
            initial_timeline=self.get_current_timeline().clone(),
            save_moves=self.save_moves,
            current_player=self.current_player,
            first_player=self.first_player,
            term_ev=self.term_ev,
        )
        game.turn_history = copy.deepcopy(self.turn_history)
        game._prune_history()
        return game

    def render(self):
        if self.current_player == P.P1:
            game = self.flipped()
        else:
            game = self
        print(game.get_current_board().__str__())

    def make_move(self, local_move):
        out = self.clone()
        global_move = self.convert_to_global_move(local_move)
        # this cannot be a terminal move, as the current_player always can END_TURN
        capture, _ = out._mutate_make_move(global_move)
        # this can be terminal, if current_player ends turn in check
        _, terminal = out._mutate_make_move(Chess2d.END_TURN)
        if terminal:
            out.term_ev = out._terminal_eval(mutation=False)
        elif out.material_draw():
            out.term_ev = (0.5, 0.5)  # draw by lack of mating material
        out.prune_timeline()
        return out

    @staticmethod
    def fixed_obs_shape():
        """
        observation is shapes ((D1,D2),), (D1,D2,2),0)
        this method returns those shapes
        """
        return (Board.BOARD_SHAPE,), (Board.BOARD_SHAPE + (2,)), 0

    @property
    def observation_shape(self):
        """
        observation is shapes ((D1,D2),), (D1,D2,2), 0)
        this method returns those shapes
        """
        return self.fixed_obs_shape()

    @property
    def observation(self):
        """
        observation is shapes ((D1,D2),), (D1,D2,2), 0)
        Note that D1 and D2 are fixed

        the board is
            the pieces (including empty squares)
        indexes are normal indices, from the point of view of current player
        Returns:

        """
        if self.current_player == P.P1:
            game = self.flipped()
        else:
            game = self.clone()
        board = game.get_current_board()
        board_obs = board.get_board_as_indices()
        board_shapes, index_shape, vec_shape = game.observation_shape
        (xlen, ylen) = index_shape[:2]

        # create index set
        X = torch.cat(
            (
                torch.arange(xlen).view((xlen, 1, 1)),
                torch.zeros((xlen, 1, 1)),
            ),
            dim=-1,
        )
        Y = torch.cat(
            (
                torch.zeros((1, ylen, 1)),
                torch.arange(ylen).view((1, ylen, 1)),
            ),
            dim=-1,
        )
        return (board_obs,), X + Y, torch.zeros(vec_shape)

    @staticmethod
    def num_observation_boards():
        """
        Returns: number of boards in (D1,...,DN, *1),(D1,...,DN, *2),...)
        """
        return 1

    @staticmethod
    def underlying_set_sizes():
        """
        returns number of possible distinct elements of each underlying set, if finite
        """
        return (P.TOTAL_PIECES,)

    def possible_move_cnt(self):
        """
        all possible choices of two squars, plus an extra unused move
        """
        return Board.BOARD_SQUARES**2 + 1

    def index_to_move(self, idx):
        """
        convert idx into a valid move
        """
        # this should never happen, but just in case
        if idx == self.possible_move_cnt() - 1:
            return Chess2d.END_TURN

        I, J = Board.BOARD_SHAPE
        num_squares = Board.BOARD_SQUARES
        start_idx = idx // num_squares
        start_idx = start_idx // J, start_idx % J
        end_idx = idx % num_squares
        end_idx = end_idx // J, end_idx % J
        return (start_idx, end_idx)

    def move_to_idx(self, move):
        I, J = Board.BOARD_SHAPE
        if move == Chess2d.END_TURN:
            return self.possible_move_cnt() - 1
        (i1, j1), (i2, j2) = move
        return Board.BOARD_SQUARES * (i1 * J + j1) + i2 * J + j2

    @property
    def representation(self):
        """
        Returns: representation of self, likely a tuple of tensors
            often this is the same as self.observation, (i.e. for perfect info games)
        all information required to play the game must be obtainable from representation
        i.e. self.from_represnetation(self.represnetation) must be functionally equivalent to self

        should return clones of any internal variables
        """
        return (
            self.multiverse.main_timeline.representation,
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
        tl_rep, current_player, first_player, turn_history, term_ev = representation
        game = Chess2d(
            initial_board=None,
            initial_timeline=Timeline.from_representation(tl_rep),
            current_player=current_player,
            first_player=first_player,
            term_ev=term_ev,
        )
        game.turn_history = turn_history
        return game

    def __str__(self):
        tl = self.multiverse.main_timeline.clone()
        tl.start_idx = 0
        return tl.__str__()


if __name__ == "__main__":
    from aleph0.algs import Human, play_game

    board = torch.zeros(Board.BOARD_SHAPE, dtype=torch.long)
    I, J = Board.BOARD_SHAPE
    board[I - 1, J - 1] = P.as_player(P.KING, P.P1)
    board[0, 0] = P.as_player(P.KING, P.P0)
    board[0, 1] = P.as_player(P.QUEEN, P.P0)
    board[1, 0] = P.as_player(P.ROOK, P.P0)

    root_game = Chess2d(initial_board=Board(board=board))

    # this should be a draw by insufficient material
    game = root_game.make_move(((0, 1), (6, 7)))
    game = game.make_move(((0, 0), (1, 0)))
    game = game.make_move(((1, 0), (1, 6)))
    game = game.make_move(((1, 0), (0, 0)))
    game = game.make_move(((1, 6), (7, 6)))
    game = game.make_move(((0, 0), (0, 1)))
    print(game)
    assert game.is_terminal()
    print("expected draw:", game.get_result())
    assert game.get_result() == (0.5, 0.5)

    # (1b) in Chess5d._terminal_eval: dumb move, in check
    game = root_game.make_move(((0, 1), (6, 7)))
    game = game.make_move(((0, 0), (1, 1)))
    print(game)
    assert game.is_terminal()
    print(
        "expected p0 win:", game.get_result()
    )  # P0 should win, as P1 could have taken the queen but did not
    assert game.get_result() == (1, 0)

    # (2b) in Chess5d._terminal_eval: dumb move, not in check
    game = root_game.make_move(((0, 1), (0, 6)))
    game = game.make_move(((0, 0), (1, 1)))
    print(game)
    assert game.is_terminal()
    print(
        "expected p0 win:", game.get_result()
    )  # P0 should win, as P1 could have taken the queen but did not
    assert game.get_result() == (1, 0)

    # stalemate
    game = root_game.make_move(((0, 1), (0, 6)))
    game = game.make_move(((0, 0), (1, 0)))
    game = game.make_move(((0, 6), (5, 6)))
    game = game.make_move(((1, 0), (0, 0)))
    game = game.make_move(((0, 0), (1, 1)))  # this position is stalemate
    print(game)
    # however, this will return false, since
    #   we will do a stalemate check next turn, when P1 fails to make a valid turn
    #   this is so we do not need to compute stalemate every turn
    print("expected false:", game.is_terminal())
    assert not game.is_terminal()
    game = game.make_move(next(game.get_all_valid_moves()))
    print("expected true:", game.is_terminal())
    assert game.is_terminal()
    print("expected draw:", game.get_result())
    assert game.get_result() == (0.5, 0.5)

    # checkmate
    game = root_game.make_move(((0, 1), (0, 6)))
    game = game.make_move(((0, 0), (1, 0)))
    game = game.make_move(((1, 0), (1, 7)))

    # again, this is checkmate, but we do the check next turn
    print("expected false:", game.is_terminal())
    assert not game.is_terminal()
    game = game.make_move(next(game.get_all_valid_moves()))
    print(game)
    print("expected true:", game.is_terminal())
    assert game.is_terminal()
    print("expected P0 win:", game.get_result())
    assert game.get_result() == (1, 0)

    quit()

    print(play_game(game, [Human(), Human()], save_histories=False))

    print(Chess2d().possible_move_cnt())
