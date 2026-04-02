import torch
from aleph0.examples.chess.game.piece import P


class Board:
    BOARD_SIZE = 8
    BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)
    BOARD_SQUARES = BOARD_SHAPE[0] * BOARD_SHAPE[1]

    def __init__(self, board=None):
        if board is None:
            board = torch.ones(Board.BOARD_SHAPE, dtype=torch.long) * P.EMPTY
            if Board.BOARD_SHAPE == (8, 8):
                back_rank = [
                    P.UNMOVED_ROOK,
                    P.KNIGHT,
                    P.BISHOP,
                    P.QUEEN,
                    P.UNMOVED_KING,
                    P.BISHOP,
                    P.KNIGHT,
                    P.UNMOVED_ROOK,
                ]
            elif Board.BOARD_SHAPE == (5, 5):
                back_rank = [
                    P.UNMOVED_ROOK,
                    P.KNIGHT,
                    P.BISHOP,
                    P.QUEEN,
                    P.UNMOVED_KING,
                ]
            else:
                raise Exception("NO DEFAULT FOR BOARD SIZE " + str(Board.BOARD_SIZE))
            for i, row in enumerate((back_rank, [P.UNMOVED_PAWN for _ in range(Board.BOARD_SIZE)])):
                board[i] = torch.tensor([P.as_player(piece, P.P0) for piece in row])

                board[Board.BOARD_SIZE - i - 1] = torch.tensor([P.as_player(piece, P.P1) for piece in row])
        self.board = board

    def flipped(self):
        return Board(
            board=P.flip_piece(torch.rot90(self.board, 2)),
        )

    def get_piece(self, idx):
        return self.board[idx].item()

    @property
    def representation(self):
        return self.board.clone()

    @staticmethod
    def from_representation(representation):
        return Board(board=representation)

    def add_piece(self, piece, square, mutate):
        """
        adds piece to specified square
        Args:
            piece:
            square:
            mutate: if true, simply changes current board
                if false, flips player and such

        Returns:

        """
        if mutate:
            new_board = self
        else:
            new_board = self.clone()
        new_board.board[square] = piece
        return new_board, self.get_piece(square)

    def remove_piece(self, square, mutate):
        """
        removes piece at square from board and returns (new board, piece removed)
        Args:
            square:  (i,j)
            mutate: if true, simply changes current board
                if false, flips player and such
        Returns: (new board, piece removed)
        """
        if mutate:
            new_board = self
        else:
            new_board = self.clone()
        new_board.board[square] = P.EMPTY
        return new_board, self.get_piece(square)

    def mutate_depassant(self, just_moved=None):
        """
        removes the enpassant marker from pieces except for the one indicated by just_moved
        :param just_moved: index
        """
        for i in range(Board.BOARD_SIZE):
            for j in range(Board.BOARD_SIZE):
                if (i, j) != just_moved:
                    self.board[i][j] = P.remove_passant(self.board[i][j])

    def all_pieces(self):
        """
        returns iterable of all pieces locations on the board
        :return: iterable of (i,j)
        """
        return zip(*torch.where(self.board != P.EMPTY))

    def pieces_of(self, player):
        """
        WILL ALWAYS RETURN PIECES IN SAME ORDER
        returns iterable of pieces of specified player
        :param player: player
        :return: iterable of (i,j)
        """

        for i, j in zip(*torch.where(torch.eq(torch.sign(self.board), P.player_to_sign(player)))):
            yield i.item(), j.item()

    def clone(self):
        """
        returns copy of self
        """
        return Board(board=self.board.clone())

    def __str__(self):
        s = ""
        for i in range(Board.BOARD_SIZE - 1, -1, -1):
            row = self.board[i]
            s += "|" + ("|".join([P.disp(piece) for piece in row])) + "|"
            s += "\n"
            s += "-" * (len(row) * 2 + 1)
            s += "\n"
        return s

    @staticmethod
    def empty_string():
        s = ""
        for row in range(2 * Board.BOARD_SIZE):
            s += " " * (2 * Board.BOARD_SIZE + 1)
            s += "\n"
        return s

    def get_board_as_indices(self):
        return P.number(self.board)

    def move_piece_on_board(
        self,
        idx,
        end_idx,
        mutate,
    ):
        """
        moves piece from idx to end_idx, returns resulting board
        Args:
            idx:
            end_idx:
        Returns: new board, piece moved, captured piece
        """
        og_piece = self.get_piece(idx)
        piece = P.moved(og_piece)  # since it moved
        i1, j1 = idx
        i2, j2 = end_idx
        if P.piece_id(piece) == P.PAWN:
            if i2 == Board.BOARD_SIZE - 1 or i2 == 0:
                piece = P.as_player(P.PROMOTION, P.player_of(piece))
            if abs(i1 - i2) == 2:
                piece = P.add_passant(piece)
        new_board, capture = self.add_piece(piece, end_idx, mutate=mutate)
        new_board.board[idx] = P.EMPTY
        return new_board, og_piece, capture


if __name__ == "__main__":
    b = Board()
    print(b)
    b, _, _ = b.move_piece_on_board((1, 0), (3, 0), mutate=True)
    print(b)
    assert P.en_passantable(b.get_piece((3, 0)))
