class P:
    P0 = 0
    P1 = 1

    EMPTY = 0

    _UNMOVED_SHIFT = 13
    _UNMOVED_MIN = 14
    _UNMOVED_MAX = 16

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
    BRAWN = 13  # beefy pawn, unimplemented (the en brasant and capturing is annoying)

    PROMOTION = QUEEN

    # asumeing PASSANTABLE PAWN is the largest piece id
    # double this, plus the empty square
    TOTAL_PIECES = PASSANTABLE_PAWN*2 + 1

    @staticmethod
    def flip_player(player):
        return 1 - player

    @staticmethod
    def piece_id(piece):
        """
        returns form of piece that is checkable for identification
        i.e. P.piece_id(piece)==P.PAWN will run true if piece is PAWN, UNMOVED_PAWN, ENPASANTABLE_PAWN
            also if piece is either player's piece
        Args:
            piece: piece
        Returns: piece id
        """
        piece = abs(piece)
        # convert unmoved pieces to moved pieces
        if P.is_unmoved(piece):
            piece = piece - P._UNMOVED_SHIFT
        # specific check for enpassantable pawn
        elif piece == P.PASSANTABLE_PAWN:
            piece = P.PAWN
        return piece

    @staticmethod
    def flip_piece(piece):
        """
        flips piece or array of pieces
        Args:
            piece:
        Returns:
        """
        return -piece

    @staticmethod
    def as_player(piece, player):
        """
        Args:
            piece: piece to mess with
            player: player that owns piece
        Returns:
            piece as player's piece
        """
        return abs(piece)*P.player_to_sign(player)

    @staticmethod
    def is_player(player):
        """
        Args:
            player: player
        Returns:
            piece as player's piece
        """
        return player == P.P0 or player == P.P1

    @staticmethod
    def player_to_sign(player):
        """
        Args:
            player: player
        Returns:
            1 for P0, -1 for P1
        """
        return (1 - 2*player)

    @staticmethod
    def player_to_idx(player):
        """
        Args:
            player: player
        Returns:
            0 for P0, 1 for P1
        """
        return player

    @staticmethod
    def player_of(piece):
        """
        returns player of piece
        """
        if piece is None:
            return None
        elif piece > 0:
            return P.P0
        elif piece < 0:
            return P.P1
        else:
            return None

    @staticmethod
    def moved(piece):
        if P.is_unmoved(piece):
            # go UNMOVED_SHIFT in the direction of 0
            return -P._UNMOVED_SHIFT*P.player_to_sign(P.player_of(piece)) + piece
        if P.en_passantable(piece):
            return P.as_player(P.PAWN, P.player_of(piece))
        return piece

    @staticmethod
    def is_unmoved(piece):
        piece = abs(piece)
        return piece >= P._UNMOVED_MIN and piece <= P._UNMOVED_MAX

    @staticmethod
    def remove_passant(piece):
        player = P.player_of(piece)
        if player is not None and (P.piece_id(piece) == P.PAWN):
            if P.is_unmoved(piece):
                return P.UNMOVED_PAWN*P.player_to_sign(player)
            else:
                return P.PAWN*P.player_to_sign(player)
        else:
            return piece

    @staticmethod
    def add_passant(piece):
        return P.player_to_sign(P.player_of(piece))*P.PASSANTABLE_PAWN

    @staticmethod
    def en_passantable(piece):
        return abs(piece) == P.PASSANTABLE_PAWN

    @staticmethod
    def disp(piece):
        d_piece = ' pkrnbqcudb'[P.piece_id(piece)]
        if P.player_of(piece) == P.P0:
            return d_piece.upper()
        return d_piece

    @staticmethod
    def number(piece):
        """
        converts piece or array of pices to number
        Args:
            piece:

        Returns:

        """
        # lowest possible piece number is -PASSANTABLE_PAWN
        return piece + P.PASSANTABLE_PAWN

    @staticmethod
    def denumber(number):
        return number - P.PASSANTABLE_PAWN


if __name__ == '__main__':
    p = P.PAWN
    print(p)
    print(P.disp(p))
    print(P.disp(P.as_player(p, P.P1)))
