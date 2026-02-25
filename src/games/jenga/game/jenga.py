"""
contains Tower class
encodes a tower, has helpful methods to get state, as well as make moves
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from aleph0.examples.jenga.game.block import Block
from aleph0.game import SelectionGame
from scipy.spatial import ConvexHull

TOLERANCE = 1e-10


def which_points_in_hull(points, hull, tolerance=TOLERANCE):
    """
    returns if points np array dim (N,2) are in hull

    return N vector of booleans
    """
    # adds one to each row to dot product easier
    aug_points = np.concatenate((points, np.ones((len(points), 1))), axis=1)

    # each column of this is the distance of each projection to all the equations
    # i.e. all_dists[i,j] is the distance of the jth point on the ith equation
    all_dists = np.dot(hull.equations, aug_points.T)

    # maximum distance of each point to the equation line (if postive, that means its outside the line)
    dists = np.max(all_dists, axis=0)

    # returns which of them have all negative distances (i.e. within all the bounds)
    return dists <= tolerance


def hull_score(point, hull, tolerance=TOLERANCE):
    """
    returns the 'score' of the point in the hull
    if the point lies outside of the hull, returns positive number
        score is distance to closest point in hull
        represents how far it is from being stable
    if the point lies inside, returns negative number
        score is -(the distance to the closest bound)
        represents how close it is to being out of bounds
        note that we do not need to check vertices, as the closest point will be to a line
    """

    # hull equations are set up that dot(point, eq[:-1])+eq[-1] is the signed distance from the line
    # if we augment the point with a 1, same as dot(point,eq) for each equation
    # then we can dot the entire matrix for speed
    dists = np.dot(hull.equations, np.concatenate((point, [1])))
    stable = np.all(dists <= 0)

    if stable:
        return np.max(dists)
    else:
        # in this case, the closest point will either be a vertex or a projection onto a line

        # projections to each line is the point minus (error * line vector)
        projections = point - hull.equations[:, :-1]*dists.reshape(-1, 1)

        # filter projections by the ones actually within the hull
        valid_projections = projections[which_points_in_hull(projections, hull, tolerance=tolerance)]

        # points to check are valid projections and the vertices of original hull
        points = np.concatenate((valid_projections, hull.points))

        # distance to all points
        point_dists = np.linalg.norm(points - point, axis=1)

        # return the minimum of these
        return np.min(point_dists)


class Genga(SelectionGame):
    """
    jenga tower representation
    list of layers, each layer is a list of three booleans
    Note: this class does not change after initalization
        all methods like "remove_block" create a new instance of the class
    """
    EMPTY = 0
    FILLED = 1
    PLACABLE = 2

    TOWER_DIAMETER = 2*Block.STD_BLOCK_SPACING + 3*Block.STD_BLOCK_DIM[1]  # theoretical size of a layer

    DEFAULT_INITIAL_SIZE = 5  # number of levels in initial tower

    def __init__(self,
                 subset_size,
                 num_players=2,
                 current_player=0,
                 block_info=None,
                 pos_std=.001,
                 angle_std=.003,
                 fallen=False,
                 initial_size=DEFAULT_INITIAL_SIZE,
                 render_mode='pyplot',
                 ):
        """
        Args:
            block_info: list of block triples, represents each layer
            pos_std: stdev to use for positions if randomly initializing/randomly placing
            angle_std: stdev to use for angles if randomly initializing/randomly placing
            render_mode: 'pyplot' or 'str', how to render tower
        """
        super().__init__(num_players=num_players,
                         current_player=current_player,
                         subset_size=subset_size,
                         special_moves=[],
                         )

        self.pos_std = pos_std
        self.angle_std = angle_std
        if block_info is None:
            block_info = [
                [Block.random_block(L=level, i=i, pos_std=pos_std, angle_std=angle_std) for i in range(3)]
                for level in range(initial_size)
            ]
        self.block_info = block_info

        self.calculated = False  # whether we calculated things like COM
        # compute all of these
        self.update_info()
        self.fallen = fallen
        self.render_mode = render_mode

    def set_fallen(self, log_prob_stable):
        prob = np.exp(log_prob_stable)
        self.fallen = np.random.rand() > prob

    def update_info(self):
        """
        computation of features to avoid having to compute them each time
        """
        self.calculated = True
        # since each block has equal mass, the COM of tower is just the straight average of COMs of blocks
        # we compute COMs at each layer, the COM of 'subtowers' starting from a layer are of interest
        self.Ns = []  # number of blocks above each layer, including that layer
        self.COMs = []  # COM above each layer (inclusive, so COMs[0] is the COM of the tower, and COMs[-1] is the COM of last layer)
        self.hulls = []  # store the convex hull of each layer.

        self.block_hulls = None  # these might not be calculated

        # going backwards so we can accumulate
        N = 0  # running count
        MOMENT = np.zeros(3)  # running moment (sum of positions of blocks, not averaged yet)
        MASS = 0
        for layer in self.block_info[::-1]:
            N += sum([t is not None for t in layer])
            MOMENT += np.sum([b.com()*b.mass for b in layer if b is not None], axis=0)
            MASS += sum([b.mass for b in layer if b is not None])
            self.Ns.append(N)
            self.COMs.append(MOMENT/MASS)

            V = np.concatenate([t.vertices_xy().reshape((4, 2)) for t in layer if t is not None], axis=0)
            self.hulls.append(ConvexHull(V))

        self.Ns.reverse()
        self.COMs.reverse()
        self.hulls.reverse()

    def blocks_on_level(self, L):
        """
        returns blocks on level L
        :param L: level
        :return: int
        """
        if (L == self.height - 1) or (L == -1):
            return self.Ns[-1]
        return self.Ns[L] - self.Ns[L + 1]

    @property
    def height(self):
        """
        :return: height of tower
        """
        return len(self.block_info)

    def com(self):
        """
        center of mass of tower
        :return: np array
        """
        return self.COMs[0]

    def top_layer_filled(self):
        """
        returns if top layer is filled
        """
        return self.blocks_on_level(self.height - 1) == 3

    # FALLING/TERMINAL CHECKS

    def log_prob_stable(self, scale=.001):
        """
        computes log of probability that tower is stable
            sum across layers
                (equivalent to just multiplying the probabilities)
            could also try min across layers
                (equivalent to probability that the least stable layer is stable)
        :return: log(prob)
        """
        return sum(self.log_prob_stable_at_layer(L, scale=scale) for L in range(self.height - 1))

    def log_prob_stable_at_layer(self, L, scale=.0005):
        """
        computes log of probability that tower is stable at layer L
            probability is calculated by sigmoid of signed distance from the convex hull
            scaled by some value so that
        :param L: layer of tower (must be < self.height-1)
        :return: log(prob)
        """
        return -np.log(1 + np.exp(self.raw_score_at_layer(L)/scale))

    def raw_score_at_layer(self, L):
        """
        computes the signed distance from COM at layer L+1 to the convex hull of layer L. Can be thought of as an un-normalized score
        :param L: layer of tower (must be < self.height-1)
        :return: -1 * the signed distance from the projection of the convex hull at layer L+1 to the convex hull of layer L
        """
        hull = self.hulls[L]
        com = self.COMs[L + 1][:2]
        return hull_score(com, hull)

    # VALIDITY CHECKS
    def no_valid_removes(self):
        for L in range(self.height - 2 + self.top_layer_filled()):
            if self.blocks_on_level(L) > 1:
                return False
        return True

    def valid_removes(self):
        """
        returns list of all (L,i) pairs that are allowed to be removed
        """
        for L in range(self.height - 2 + self.top_layer_filled()):
            if self.blocks_on_level(L) > 1:
                for (i, t) in enumerate(self.block_info[L]):
                    if t is not None:
                        yield (L, i)

    def valid_place_blocks(self):
        """
        returns the valid 'moves' to place a block on tower
        :return: non-empty list of indices
        """
        if self.top_layer_filled():
            for i in range(3):
                yield (self.height, i)
        else:
            for i in range(3):
                if self.block_info[-1][i] is None:
                    yield (self.height - 1, i)

    def valid_moves_product(self):
        """
        returns all valid next moves
        :return: (all possible 'remove' steps, all possible 'place' steps)
            Note that any choice from these is a valid next move
        """
        removes = list(self.valid_removes())
        # Note: removing a block does not change the top layer
        # thus, the possible placement moves remain constant after removing a block
        places = list(self.valid_place_blocks())
        return (removes, places)

    # GAME MOVE METHODS (RETURNS MUTATED TOWER)

    def blockinfo_remove_block(self, remove):
        """
        removes specified block
        :param remove: (L,i) tuple
            L: level of block
            i: index of block
        :return: Tower object with specified block removed
        """
        L, i = remove
        bi = copy.deepcopy(self.block_info)
        return [
            [(None if eye == i and ell == L else block) for (eye, block) in enumerate(level)]
            for (ell, level) in enumerate(bi)]

    def blockinfo_place_block(self, place, blk_pos_std=None, blk_angle_std=None):
        """
        places block at specified position
        :param place: position to add block into
        :param blk_pos_std: pos stdev, if different from default
        :param blk_angle_std: angle stdev, if different from default
        :return: Tower with block added
        """
        _, i = place
        if blk_pos_std is None:
            blk_pos_std = self.pos_std
        if blk_angle_std is None:
            blk_angle_std = self.angle_std
        if self.top_layer_filled():
            new_block = Block.random_block(L=self.height, i=i, pos_std=blk_pos_std, angle_std=blk_angle_std)
            return copy.deepcopy(self.block_info) + [[(new_block if eye == i else None) for eye in range(3)]]
        else:
            new_block = Block.random_block(L=self.height - 1, i=i, pos_std=blk_pos_std, angle_std=blk_angle_std)
            bi = copy.deepcopy(self.block_info)
            return [
                [(new_block if eye == i and L == self.height - 1 else block) for (eye, block) in enumerate(level)]
                for (L, level) in enumerate(bi)]

    @staticmethod
    def local_to_global_idx(idx):
        L, I = idx
        return L, I + 1

    def get_indices(self):
        H = self.height + self.top_layer_filled()
        Ht = torch.stack((torch.arange(H).view(-1, 1),
                          torch.zeros((H, 1)),
                          ), dim=-1)
        Wt = torch.stack((torch.zeros((1, 3)),
                          torch.arange(3).view(1, -1) - 1,  # recenter around 0
                          ), dim=-1)
        return Ht + Wt

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ASS METHODS                                                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @property
    def observation(self):
        H = self.height + self.top_layer_filled()
        ident = torch.ones((H, 3), dtype=torch.long)*Jenga.EMPTY

        parity = torch.arange(H)%2
        parity = parity.view(H, 1).broadcast_to(H, 3)

        blocks = torch.zeros((H, 3, Block.vector_size()))
        ident[-1] = Jenga.PLACABLE
        for L, layer in enumerate(self.block_info):
            for i, t in enumerate(layer):
                if t is not None:
                    ident[L, i] = Jenga.FILLED
                    blocks[L, i] = torch.tensor(t.representation)

        return (ident, parity, blocks,), self.get_indices(), torch.zeros(0)

    @staticmethod
    def num_observation_boards():
        """
        Returns: number of boards in (D1,...,DN, *1),(D1,...,DN, *2),...)
        """
        return 3

    @staticmethod
    def underlying_set_sizes():
        """
        returns number of possible distinct elements of each underlying set, if finite
        """
        return [3, 2, None]

    @property
    def permutation_to_standard_pos(self):
        # permutation[current_player] should be 0, as the network should assume current player is 0
        # every other player is in the same order
        return [(i - self.current_player)%self.num_players for i in range(self.num_players)]

    def valid_special_moves(self):
        return iter(())

    def __eq__(self, other):
        return self.block_info == other.block_info

    def __str__(self):
        """
        returns string representation of tower
        binary encoding of each level
        i.e. full layer is a 7, layer with one block is a 1, 2, or 4 depending on which block
        blocks are indexed from -x to +x (even layers) or -y to +y (odd layers)
        """
        s = ''
        for L in self.block_info:
            s_L = 0
            for (i, t) in enumerate(L):
                if t is not None:
                    s_L += int(2**i)
            s += str(s_L)
        return s

    def render_str(self):
        print(self.__str__())

    def render_pyplot(self):
        raise NotImplementedError

    def render(self):
        if self.render_mode == 'pyplot':
            self.render_pyplot()
        else:
            self.render_str()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # outdated methods                                                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def image_of_layer(self, L, resolution=256, radius=None):
        """
        returns an 'image' of density of a layer
        array of 0s or 1s, 1 if there is a block at a point

        :param L: layer of tower
        :param resolution: granularity of the image
        :param radius: distance to look on each dimension
            looks at [-radius,radius]^2

        returns an (resolution x resolution) numpy array 'image'
        """
        if radius is None:
            radius = 1.2*Jenga.TOWER_DIAMETER/2
        grid = (np.arange(resolution)/(resolution - 1) - .5)*2*radius
        xv, yv = np.meshgrid(grid, grid, indexing='xy')
        points_arr = np.stack((xv, yv), axis=2)
        points_arr = points_arr.reshape((resolution*resolution, 2))

        thingies = np.sum(
            [which_points_in_hull(points_arr, hull, tolerance=TOLERANCE) for hull in self.get_block_hulls()[L]], axis=0)

        return thingies.reshape((resolution, resolution))

    def update_special_info(self):
        """
        these probably wont be necessary for EVERY tower, only run if needed
        """
        self.block_hulls = []
        for layer in self.block_info:
            hull_layer = [ConvexHull(t.vertices_xy().reshape((4, 2))) for t in layer if t is not None]
            self.block_hulls.append(hull_layer)

    def get_block_hulls(self):
        if self.block_hulls is None:
            self.update_special_info()
        return self.block_hulls

    def boolean_blocks(self):
        return [[t is not None for t in layer] for layer in self.block_info]

    def num_blocks(self):
        """
        :return: number of blocks
        """
        return self.Ns[0]

    def free_valid_moves(self):
        """
        returns total 'free moves' that are valid to take
            a free move is the max number of moves left before a level is probably fallen
            a full layer has 2 free moves
            a layer with one block has 0 free moves
            a layer with 2 blocks has either 0 or 1 free move depending on the arrangement
        if the top layer is filled, include the (self.height-1)th layer (1 indexed)
        else, go up to (self.height-2), 1 indexed
        """
        layer_types = self.layer_type_count()
        return layer_types[1] + 2*layer_types[3]

    def layer_type_count(self):
        """
        returns number of each layer type, not including layers that we cannot remove blocks from

        types are
            one block
            two block (| || |)
            two block (| |   | |)
            three block
        """
        layer_counts = [0,  # one block layers
                        0,  # two block (| || |) layers
                        0,  # two block (| |   | |) layers
                        0,  # three block layers
                        ]
        for L in range(self.height - 2 + self.top_layer_filled()):
            # if top layer is filled, go one more
            count = self.blocks_on_level(L)
            if count == 1:
                layer_counts[0] += 1
            elif count == 3:
                layer_counts[3] += 1
            else:
                # there are two blocks
                if self.block_info[L][1] is None:
                    # if there is no middle block
                    layer_counts[2] += 1
                else:
                    layer_counts[1] += 1
        return layer_counts


class Jenga(Genga):
    '''
    Jenga implementation
    each move is a pick selection and a place selection
    '''

    def __init__(self,
                 num_players=2,
                 current_player=0,
                 block_info=None,
                 pos_std=.001,
                 angle_std=.003,
                 fallen=False,
                 initial_size=Genga.DEFAULT_INITIAL_SIZE,
                 render_mode='pyplot',
                 ):
        super().__init__(subset_size=2,
                         num_players=num_players,
                         current_player=current_player,
                         block_info=block_info,
                         pos_std=pos_std,
                         angle_std=angle_std,
                         fallen=fallen,
                         initial_size=initial_size,
                         render_mode=render_mode,
                         )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ASS METHODS                                                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def valid_selection_moves(self):
        removes, places = self.valid_moves_product()
        for remove in removes:
            for place in places:
                yield (remove, place)

    @property
    def observation_shape(self):
        H = self.height + self.top_layer_filled()
        return ((H, 3), (H, 3), (H, 3, Block.vector_size())), (H, 3, 2), 0

    @property
    def representation(self):
        """
        Returns: representation of self, likely a tuple of tensors
            often this is the same as self.observation, (i.e. for perfect info games)
        all information required to play the game must be obtainable from representation
        i.e. self.from_represnetation(self.represnetation) must be functionally equivalent to self

        should return clones of any internal variables
        """
        return copy.deepcopy(self.block_info), self.num_players, self.current_player, self.fallen

    @staticmethod
    def from_representation(representation):
        """
        returns a SubsetGame instance from the output of self.get_representation
        Args:
            representation: output of self.get_representation
        Returns: SubsetGame object
        """
        block_info, num_players, current_player, fallen = representation
        return Jenga(block_info=block_info,
                     num_players=num_players,
                     current_player=current_player,
                     fallen=fallen,
                     )

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
        remove, place = local_move

        # remove a block
        remove_blockinfo = self.blockinfo_remove_block(remove=remove)
        removed = Jenga(
            num_players=self.num_players,
            current_player=self.current_player,
            block_info=remove_blockinfo,
            pos_std=self.pos_std,
            angle_std=self.angle_std,
            fallen=False,
            render_mode=self.render_mode
        )

        # place a block
        placed_blockinfo = removed.blockinfo_place_block(place=place)
        placed = Jenga(
            num_players=self.num_players,
            current_player=(self.current_player + 1)%self.num_players,
            block_info=placed_blockinfo,
            pos_std=self.pos_std,
            angle_std=self.angle_std,
            fallen=False,
            render_mode=self.render_mode
        )
        # can calculate both probabilities here, as the probability of both being stable is simply the produt
        placed.set_fallen(log_prob_stable=removed.log_prob_stable() + placed.log_prob_stable())

        return placed

    def get_result(self):
        """
        can only be called on terminal states
        returns an outcome for each player
        Returns: K-tuple of outcomes for each player
            outcomes are generally in the range [0,1] and sum to 1
            i.e. in a 1v1 game, outcomes would be (1,0) for p0 win, (0,1) for p1, and (.5,.5) for a tie
            in team games this can be changed to give teammates the same reward, and have the sum across teams be 1
        """

        results = np.ones(self.num_players)/(self.num_players - 1)
        if self.fallen:
            # previous player knocked over tower, so they lose
            results[self.current_player - 1] = 0
        else:
            # current player has no moves, so they lose
            results[self.current_player] = 0

        return results

    def is_terminal(self):
        """
        returns if current game has terminated
        CANNOT BE PROBABILISTIC
            if there is a probabilistic element to termination,
                the probabilities must be calculated upon creation of this object and stored
        Returns: boolean
        """
        return self.fallen or self.no_valid_removes()

    def render_pyplot(self):
        plt.close()
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_zticks(())

        colors = ['purple', 'red', 'blue', 'orange', 'brown', 'yellow']
        counter = 0

        indices = self.get_indices()
        place_height = self.height + self.top_layer_filled() - 1
        pick_ht_bound = place_height - 1
        for L, (t1, t2) in enumerate(zip(self.block_info + [[None, None, None]], indices)):
            for (i, (b, idx)) in enumerate(zip(t1, t2)):
                label = str(tuple(int(thing.item()) for thing in idx))
                if b is not None:
                    if L >= pick_ht_bound:
                        label = None
                    b.render_to(ax=ax,
                                color=colors[counter%len(colors)],
                                alpha=.9,
                                label=label,
                                )
                else:
                    if L == place_height:
                        b = Block.random_block(L=L, i=i, pos_std=0, angle_std=0)
                        b.render_frame_to(ax=ax,
                                          color='black',
                                          label=label,
                                          linestyle='--',
                                          )
                counter += 1

        plt.ion()
        plt.show()


class JengaOne(Genga):
    """
    version of jenga that is split into a pick move an a place move for each player
    observation vector includes whether it is a pick or place turn for player
    """

    def __init__(self,
                 block_held=False,
                 num_players=2,
                 current_player=0,
                 block_info=None,
                 pos_std=.001,
                 angle_std=.003,
                 fallen=False,
                 initial_size=Genga.DEFAULT_INITIAL_SIZE,
                 render_mode='pyplot',
                 ):
        super().__init__(subset_size=1,
                         num_players=num_players,
                         current_player=current_player,
                         block_info=block_info,
                         pos_std=pos_std,
                         angle_std=angle_std,
                         fallen=fallen,
                         initial_size=initial_size,
                         render_mode=render_mode,
                         )
        self.block_held = block_held

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ASS METHODS                                                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def valid_selection_moves(self):
        if self.block_held:
            for move in self.valid_place_blocks():
                yield (move,)
        else:
            for move in self.valid_removes():
                yield (move,)

    @property
    def observation(self):
        boards, indices, _ = super().observation
        return boards, indices, torch.ones(1)*self.block_held

    @property
    def observation_shape(self):
        H = self.height + self.top_layer_filled()
        return ((H, 3), (H, 3), (H, 3, Block.vector_size())), (H, 3, 2), 1

    @property
    def representation(self):
        """
        Returns: representation of self, likely a tuple of tensors
            often this is the same as self.observation, (i.e. for perfect info games)
        all information required to play the game must be obtainable from representation
        i.e. self.from_represnetation(self.represnetation) must be functionally equivalent to self

        should return clones of any internal variables
        """
        return copy.deepcopy(self.block_info), self.num_players, self.current_player, self.fallen, self.block_held

    @staticmethod
    def from_representation(representation):
        """
        returns a SubsetGame instance from the output of self.get_representation
        Args:
            representation: output of self.get_representation
        Returns: SubsetGame object
        """
        block_info, num_players, current_player, fallen, block_held = representation
        return JengaOne(block_info=block_info,
                        num_players=num_players,
                        current_player=current_player,
                        fallen=fallen,
                        block_held=block_held,
                        )

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
        if self.block_held:
            place, = local_move

            placed_blockinfo = self.blockinfo_place_block(place=place)
            placed = JengaOne(
                block_held=False,
                num_players=self.num_players,
                current_player=(self.current_player + 1)%self.num_players,
                block_info=placed_blockinfo,
                pos_std=self.pos_std,
                angle_std=self.angle_std,
                fallen=False,
                render_mode=self.render_mode
            )
            placed.set_fallen(log_prob_stable=placed.log_prob_stable())
            return placed
        else:
            remove, = local_move
            remove_blockinfo = self.blockinfo_remove_block(remove=remove)
            removed = JengaOne(
                block_held=True,
                num_players=self.num_players,
                current_player=self.current_player,
                block_info=remove_blockinfo,
                pos_std=self.pos_std,
                angle_std=self.angle_std,
                fallen=False,
                render_mode=self.render_mode
            )
            removed.set_fallen(log_prob_stable=removed.log_prob_stable())
            return removed

    def get_result(self):
        """
        can only be called on terminal states
        returns an outcome for each player
        Returns: K-tuple of outcomes for each player
            outcomes are generally in the range [0,1] and sum to 1
            i.e. in a 1v1 game, outcomes would be (1,0) for p0 win, (0,1) for p1, and (.5,.5) for a tie
            in team games this can be changed to give teammates the same reward, and have the sum across teams be 1
        """

        results = np.ones(self.num_players)/(self.num_players - 1)
        # if we hold a block, current player is last player that moved
        # if we are not holding a block, the last player that moved is the previous player
        last_player = self.current_player - 1 + self.block_held
        if self.fallen:
            # previous player knocked over tower, so they lose
            results[last_player] = 0
        else:
            # current player has no moves, so they lose
            results[last_player] = 0
        return results

    def is_terminal(self):
        """
        returns if current game has terminated
        CANNOT BE PROBABILISTIC
            if there is a probabilistic element to termination,
                the probabilities must be calculated upon creation of this object and stored
        Returns: boolean
        """
        # eithher we fell, or we need to pick from a tower that has no valid moves
        return self.fallen or (not self.block_held and self.no_valid_removes())

    def render_pyplot(self):
        plt.close()
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_zticks(())

        colors = ['purple', 'red', 'blue', 'orange', 'brown', 'yellow']
        counter = 0

        indices = self.get_indices()
        place_height = self.height + self.top_layer_filled() - 1
        pick_ht_bound = place_height - 1
        for L, (t1, t2) in enumerate(zip(self.block_info + [[None, None, None]], indices)):
            for (i, (b, idx)) in enumerate(zip(t1, t2)):
                label = str(tuple(int(thing.item()) for thing in idx))
                if b is not None:
                    # if we are holding a block, we do not care about these indices
                    if self.block_held or (L >= pick_ht_bound):
                        label = None
                    b.render_to(ax=ax,
                                color=colors[counter%len(colors)],
                                alpha=.9,
                                label=label,
                                )
                else:
                    if (L == place_height) and self.block_held:
                        b = Block.random_block(L=L, i=i, pos_std=0, angle_std=0)
                        b.render_frame_to(ax=ax,
                                          color='black',
                                          label=label,
                                          linestyle='--',
                                          )
                counter += 1

        plt.ion()
        plt.show()


if __name__ == "__main__":

    from aleph0.algs import MCTS, Human, play_game

    b = Block.random_block(1, 1, pos_std=0.)
    t = Jenga(pos_std=0.001, angle_std=0.005, num_players=3)
    print(t, t.is_terminal(), t.current_player)
    t = t.make_move(((0, 0), (5, 0)))
    print(t, t.is_terminal(), t.current_player)
    t = t.make_move(((1, 0), (5, 1)))
    print(t, t.is_terminal(), t.current_player)
    t = t.make_move(((0, 1), (5, 2)))
    print(t, t.is_terminal(), t.current_player)
    print(t.get_result())

    t = Jenga(pos_std=0.001, angle_std=0.005, initial_size=3)
    t = t.make_move(((0, 0), (3, 2)))
    # the bottom layer has two blocks remaining, MCTS should see that removing one is a really bad idea
    # i.e. the moves starting with (0,1) should have very low probability
    alg = MCTS(num_reads=200)
    print(t)

    policy, val = alg.get_policy_value(game=t)
    for prob, mvoe in zip(policy, t.get_all_valid_moves()):
        print(mvoe, prob.item())
    print('value', val)

    print(play_game(JengaOne(), alg_list=[Human(), alg]))

    quit()
    from PIL import Image

    layer0 = t.image_of_layer(0, resolution=512).astype(np.uint8)
    img = Image.fromarray(layer0*255)
    img.show()
