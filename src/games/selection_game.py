import itertools

import numpy as np


class SelectionGame:
    """
    base class for an N-dimensional board game with K players
    each player makes a move by selecting a subset of the board positions of fixed size
        the motivation of this is a 'pick-place' game where the subset size is 2, such as chess or jenga
    this should be IMMUTABLE, all subclass methods like (make move, etc.) should return a copy of self with no shared info
    each observation is associated with
        a tuple of shape (D1,...,DN, *) boards (the * can be different for each board)
            i.e. ((D1,...,DN, *1),(D1,...,DN, *2),...)
            the * represents the shape of the underlying set
                we assume the underlying set is the product of a finite number of underlying sets. this makes
                    it easier to represent some games with potentially mixed input
        a shape (D1,...,DN, N) 'position' board that keeps track of each index's coordinates in each of the N dimensions
        a T dimensional vector with additional game information
    T and N are fixed, while Di can change (sequential data)

    Note that the game need not be deterministic, the only requirement is that the winning strategy is only dependent
        on the current state, as opposed to past values
    for example,
    chess would be represented with
        one shape (8,8) board, where each square contains an integer encoding the identity of the piece
            the integer must also encode relevant information like an unmoved king/rook (for castling), etc
        a shape (8,8,2) position board P where entry P[i,j] simply contains the vector (i,j)
        a T dimensional vector with information such as the current player, the time since last captured piece, etc.
    jenga would be represented with
        one shape (H,3,3,E) board and one shape (H,3,3) board,
            H is the tower height+1, E contains information about the piece,
            such as cartesian position, rotation
            the other board would have binary pieces represneting whether there is a piece in each square
        a shape (H,3,3,3) position board P where entry P[h,i,j] contains [h,i-1,j-1]
            (if we want the 'center' piece to be 0)
        a T dimensional vector with information such as the current player
    """

    COMPRESSABLE = False

    def __init__(
        self,
        num_players,
        current_player,
        subset_size,
        special_moves,
    ):
        """
        Args:
            current_player: player whose move it is
            subset_size: number of indices to pick to complete a normal move
            special_moves: list of special moves (i.e. END_TURN) that are not a selection of indices
                each move must have their own unique index that also cannot be a possible subset of indices
                    i.e. avoid 'special moves' like ((1,2),(2,3)), and instead name them things like 'END_TURN'
        """
        self.current_player = current_player
        self.selection_size = subset_size
        self.special_moves = special_moves
        self.num_players = num_players

    #### must define at least one of the following two
    ### usually get_valid_next_selections is easiest, but can also just directly define valid_selection_moves
    def get_valid_next_selections(self, move_prefix=()):
        """
        NEVER USED OUTSIDE OF valid_selection_moves
        gets valid choices for next index to select
            INDICES ARE INDICES INTO THE OBS ARRAY RETURNED
            MUST BE DETERMINISTIC
            moves must always be returned in the same order
        Args:
            move_prefix: indices selected so far, must be less than self.subsetsize
        Returns:
            iterable of N tuples indicating which additions are valid
        """
        raise NotImplementedError

    def valid_selection_moves(self):
        """
        gets all possible moves
        Returns:
            iterable of (self.subset_size tuples of N tuples)
        """

        def HELP_MEEEE(move_prefix=()):
            if len(move_prefix) == self.selection_size:
                yield move_prefix
            else:
                for next_move in self.get_valid_next_selections(move_prefix=move_prefix):
                    new_prefix = move_prefix + (next_move,)
                    for valid_move in HELP_MEEEE(move_prefix=new_prefix):
                        yield valid_move

        for move in HELP_MEEEE():
            yield move

    def valid_special_moves(self):
        """
        returns iterable of special moves possible from current position
        MUST BE DETERMINISTIC, always return moves in same order
        Returns: boolean
        """
        if not self.special_moves:
            return iter(())
        else:
            raise NotImplementedError

    @property
    def observation(self):
        """
        Returns: (board, position, info vector), as observed by the current player
            of shapes ((D1,...,DN, *1),(D1,...,DN, *2),...), (D1,...,DN, N), (T,))
        should return clones of any internal variables

        This can involve flipping the board and such, if necessary
        note that indices may also need to be flipped
            i.e. the opponent's 0 may be different
        """
        raise NotImplementedError

    @staticmethod
    def num_observation_boards():
        """
        Returns: number of boards in (D1,...,DN, *1),(D1,...,DN, *2),...)
        """
        raise NotImplementedError

    @staticmethod
    def underlying_set_sizes():
        """
        returns number of possible distinct elements of each underlying set, if finite
        """
        raise NotImplementedError

    @property
    def representation(self):
        """
        Returns: representation of self, likely a tuple of tensors
            often this is the same as self.observation, (i.e. for perfect info games)
        all information required to play the game must be obtainable from representation
        i.e. self.from_represnetation(self.represnetation) must be functionally equivalent to self

        should return clones of any internal variables
        """
        raise NotImplementedError

    @staticmethod
    def from_representation(representation):
        """
        returns a SubsetGame instance from the output of self.get_representation
        Args:
            representation: output of self.get_representation
        Returns: SubsetGame object
        """
        raise NotImplementedError

    def make_move(self, local_move):
        """
        gets resulting SubsetGame object of taking specified move from this state
        this may not be deterministic,
        cannot be called on terminal states
        Args:
            local_move: a subset of the possible obs board indices, a tuple of N-tuples
        Returns:
            copy of SubsetGame that represents the result of taking the move
        """
        raise NotImplementedError

    def is_terminal(self):
        """
        returns if current game has terminated
        CANNOT BE PROBABILISTIC
            if there is a probabilistic element to termination,
                the probabilities must be calculated upon creation of this object and stored
        Returns: boolean
        """
        raise NotImplementedError

    def get_result(self):
        """
        can only be called on terminal states
        returns an outcome for each player
        Returns: K-tuple of outcomes for each player (NOT PERMUTED)
            outcomes are generally in the range [0,1] and sum to 1
            i.e. in a 1v1 game, outcomes would be (1,0) for p0 win, (0,1) for p1, and (.5,.5) for a tie
            in team games this can be changed to give teammates the same reward, and have the sum across teams be 1
        """
        raise NotImplementedError

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # NOT REQUIRED TO IMPLEMENT (either extra, or current implementation works fine)        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @property
    def batch_obs(self):
        boards, indices, vec = self.observation
        return (
            tuple(board.unsqueeze(0) for board in boards),
            indices.unsqueeze(0),
            vec.unsqueeze(0),
        )

    @property
    def permutation_to_standard_pos(self):
        """
        return the permutation required to send values to 'normal' values, or None if no perm
        this encodes where each player was sent
        i.e. if we look at the game from player i's perspective,
            permutation_to_standard_pos[i]=0

        if we are playing black in chess, an observation may flip the board, and the network would
            view the game as good for white
            in this case, we would want the permutation to be [1,0] so true_value[i]=output_value[perm[i]]
        """
        return None

    @property
    def sequence_dim(self):
        _, pos_shape, _ = self.observation_shape
        return len(pos_shape) - 1

    @property
    def observation_shape(self):
        """
        observation is shapes ((D1,...,DN, *1),(D1,...,DN, *2),...), (D1,...,DN, N), T)
        this method returns those shapes
        """
        boards, pos, vec = self.observation
        return tuple(b.shape for b in boards), pos.shape, vec.shape.item()

    def get_obs_board_shape(self):
        """
        Returns: (D1,...,DN) for current board
        """
        _, pos_shape, _ = self.observation_shape
        return pos_shape[:-1]

    def get_obs_vector_shape(self):
        """
        returns T for the length of the observation extra vector
        """
        _, _, T = self.observation_shape
        return T

    def get_underlying_set_shapes(self):
        """
        Returns: (*1,*2,...), the encoding shape of the underlying sets
            for board games, this is often empty (), as each piece is represented by a zero-length integer
        we represnet the underlying set as a product of finite underlying sets to make it easier to mix input modes
        """
        obs_shapes, pos_shape, _ = self.observation_shape
        return tuple(tuple(obs_shape)[len(pos_shape) - 1 :] for obs_shape in obs_shapes)

    def clone(self):
        return self.from_representation(self.representation)

    def get_all_valid_moves(self):
        """
        gets all possible moves
        will always return self.valid_selection_moves()
         then self.valid_special_moves()
        Args:
            move_prefix: moves selected so far,
        Returns:
            iterable of (self.subset_size tuples of N tuples)
        """
        for move in self.valid_selection_moves():
            yield move
        for move in self.valid_special_moves():
            yield move

    def symmetries(self, policy_vector):
        """
        Args:
            policy_vector: torch vector of size get_all_valid_moves
        Returns:
            iterable of (SubsetGame, policy_vector) objects that encode equivalent games
            Note that the policy_vector input will likely need to be permuted to match each additional symmetry
            i.e. rotated boards in tictactoe
            used for training algorithm
        """
        yield (self, policy_vector)

    def get_choices_on_dimension(self, dim):
        """
        gets all valid moves on specified dimension
        this is for if on this dimension, valid moves are independent of the previous move choices
            i.e. if 'place' was always a choice of 3 moves like jenga
        this does not need to be implemented, as it is not always true
            (i.e. chess place moves are dependent on which piece was selected)
        Args:
            dim: dimension to inspect
        Returns:
            iterable of N tuples indicating which choices are valid on this dimension
        """
        raise NotImplementedError

    def render(self):
        print(self.__str__())


class FixedSizeSelectionGame(SelectionGame):
    def __init__(self, num_players, current_player, subset_size, special_moves):
        super().__init__(
            num_players=num_players,
            current_player=current_player,
            subset_size=subset_size,
            special_moves=special_moves,
        )

    def get_valid_next_selections(self, move_prefix=()):
        """
        gets valid choices for next index to select
            MUST BE DETERMINISTIC
            moves must always be returned in the same order
        Args:
            move_prefix: indices selected so far, must be less than self.subsetsize
        Returns:
            iterable of N tuples indicating which additions are valid
        """
        raise NotImplementedError

    def valid_special_moves(self):
        """
        returns iterable of special moves possible from current position
        MUST BE DETERMINISTIC, always return moves in same order
        Returns: boolean
        """
        return super().valid_special_moves()

    @property
    def observation(self):
        """
        Returns: (board, position, info vector), as observed by the current player
            of shapes ((D1,...,DN, *1),(D1,...,DN, *2),...), (D1,...,DN, N), (T,))
        should return clones of any internal variables

        This can involve flipping the board and such, if necessary
        """
        raise NotImplementedError

    @staticmethod
    def num_observation_boards():
        """
        Returns: number of boards in (D1,...,DN, *1),(D1,...,DN, *2),...)
        """
        raise NotImplementedError

    @staticmethod
    def underlying_set_sizes():
        """
        returns number of possible distinct elements of each underlying set, if finite
        """
        raise NotImplementedError

    @property
    def representation(self):
        """
        Returns: representation of self, likely a tuple of tensors
            often this is the same as self.observation, (i.e. for perfect info games)
        all information required to play the game must be obtainable from representation
        i.e. self.from_represnetation(self.represnetation) must be functionally equivalent to self

        should return clones of any internal variables
        """
        raise NotImplementedError

    @staticmethod
    def from_representation(representation):
        """
        returns a SubsetGame instance from the output of self.get_representation
        Args:
            representation: output of self.get_representation
        Returns: SubsetGame object
        """
        raise NotImplementedError

    def make_move(self, local_move):
        """
        gets resulting SubsetGame object of taking specified move from this state
        this may not be deterministic,
        cannot be called on terminal states
        Args:
            local_move: a subset of the possible board indices, a tuple of N-tuples
            or END_TURN token for ending turn
        Returns:
            copy of SubsetGame that represents the result of taking the move
        """
        raise NotImplementedError

    def is_terminal(self):
        """
        returns if current game has terminated
        CANNOT BE PROBABILISTIC
            if there is a probabilistic element to termination,
                the probabilities must be calculated upon creation of this object and stored
        Returns: boolean
        """
        raise NotImplementedError

    def get_result(self):
        """
        can only be called on terminal states
        returns an outcome for each player
        Returns: K-tuple of outcomes for each player
            outcomes are generally in the range [0,1] and sum to 1
            i.e. in a 1v1 game, outcomes would be (1,0) for p0 win, (0,1) for p1, and (.5,.5) for a tie
            in team games this can be changed to give teammates the same reward, and have the sum across teams be 1
        """
        raise NotImplementedError

    @staticmethod
    def fixed_obs_shape():
        raise NotImplementedError

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # SHOULD PROBABLY IMPLEMENT (current implementation is bad)                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def index_to_move(self, idx):
        """
        convert idx into a valid move
        """
        if "ind_to_move" not in dir(self):
            self.ind_to_move = []
            _, pos_shape, _ = self.fixed_obs_shape()
            board_shape = pos_shape[:-1]

            all_indices = itertools.product(*[range(t) for t in board_shape])
            # all possible selections of self.subset_size
            self.ind_to_move = list(itertools.product(all_indices, repeat=self.selection_size))
            self.ind_to_move.extend(list(self.special_moves))
        return self.ind_to_move[idx]

    def move_to_idx(self, move):
        for i in range(self.possible_move_cnt()):
            if self.index_to_move(i) == move:
                return i
        return None

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # NOT REQUIRED TO IMPLEMENT (either extra, or current implementation works fine)        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def possible_move_cnt(self):
        """
        return number of possible moves
        by default does the following calculation:
            number of board squares is the product of the board dimensions
            the number of ways to choose self.selection_size squares is (number of board squares)^(self.selection_size)
            the total possible moves is that plus the number of possible special moves
        """
        _, pos_shape, _ = self.fixed_obs_shape()
        board_squares = np.prod(pos_shape[:-1]).item()

        return board_squares**self.selection_size + len(self.special_moves)

    @property
    def observation_shape(self):
        """
        observation is shapes ((D1,...,DN, *1),(D1,...,DN, *2),...), (D1,...,DN, N), T)
        this method returns those shapes
        """
        return self.fixed_obs_shape()
