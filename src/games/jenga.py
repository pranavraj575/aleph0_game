import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import ConvexHull

from .game import Game


def tail_cumsum(arr, dim):
    """
    cumsum, but reversed direction
    the ith index is the sum of i,i+1,... indices
    """
    return torch.sum(arr, dim=dim, keepdim=True) - torch.cumsum(arr, dim=dim) + arr


class Jenga(Game):
    BLOCK_VEC_DIM = 3 + 3 + 1 + 1 + 1 + 8  # position, size, yaw, density,mass, vertices

    def __init__(
        self,
        players=2,
        initial_height=18,
        scale=0.01,
        std_block_size=torch.tensor((0.075, 0.025, 0.015)),
        std_block_spacing=0,  # .005,
        std_wood_density=0.5,
        tolerance=1e-8,
        k=3,
        deterministic=False,
        instability_scale=0.0005,
    ):
        """
        :param players: number of players
        :param initial_height: initial tower height (tower will contain initial_height*3 blocks)
        :param k: number of blocks per layer
        :param scale: how many meters make a 'unit' for observation purposes, .01 means we work in cm scale
        :param std_block_size: size in meters (length, width, height) of a block
        :param std_block_spacing: spacing in meters between blocks on same level
        :param std_wood_density: density of the wood in kg/m^3
        :param tolerance: tolerance (in meters) to use when determining if a point is within a convex hull
        :param deterministic: whether the game is deterministic
            if true, a tower will fall if there is any layer at which the COM of the tower above it is outside the convex hull of that layer
            if false, a tower will fall probabilitically depending on how far the COM is from being within the convex hull of the support layer
        :param instability_scale: distance (in meters) that it will take for a layer to be .75 likely to fall over
            i.e. if COM is this distance outside of the convex hull of the supporting layer, we will judge that level to be .75 likely to fall over
                a scaled sigmoid function will accomplish this
        """
        self.num_players = players
        self.initial_height = initial_height
        self.k = k
        self.scale = scale  # in m/u where u is the unit we use
        self.std_block_size = std_block_size / self.scale  # convert from m to u
        self.std_block_spacing = std_block_spacing / self.scale  # convert from m to u
        self.std_wood_density = (
            std_wood_density * self.scale**3
        )  # convert from kg/m^3 to kg/u^3
        self.tolerance = tolerance / self.scale  # convert from m to u
        self.deterministic = deterministic
        self.instability_scale = instability_scale / self.scale  # convert from m to u
        self.sigmoid_constant = np.log(3) / self.instability_scale
        # calculation of .75=1/(1+exp(-cd)) where d=instability_scale
        #  4/3=1+exp(-cd)
        #  1/3=exp(-cd)
        #  -log(3)=-cd
        #  c=log(3)/d

    def generate_initial_tower(self):
        layer_num = torch.arange(self.initial_height).repeat((self.k, 1)).T.flatten()
        # [0,0,0,1,1,1,...]
        centered_block_num = (
            torch.arange(self.initial_height * self.k) % self.k - (self.k - 1) / 2
        )
        # [-1,0,1,-1,0,1,...]

        mean_z = self.std_block_size[2] * (layer_num + 0.5)
        offsets = centered_block_num * (self.std_block_size[1] + self.std_block_spacing)
        mean_x = offsets * (layer_num % 2)
        # [0,0,0, -.3,0,.3, 0,0,0, ...]
        mean_y = offsets * ((layer_num + 1) % 2)
        # [-.3,0,.3, 0,0,0, -.3,0,.3, ...]
        means = torch.stack((mean_x, mean_y, mean_z), dim=-1)

        yaws = (layer_num % 2) * torch.pi / 2
        # yaws will be [0,0,0, pi/2,pi/2,pi/2,0,0,0,...]
        blocks = self.generate_random_blocks(means=means, yaws=yaws)
        tower = blocks.reshape(-1, self.k, self.BLOCK_VEC_DIM)
        return tower

    def get_means(self, tower):
        return tower[:, :, :3]

    def get_masses(self, tower):
        return tower[:, :, (8,)]

    def get_cumulative_coms(self, tower):
        """
        gets COMs of cumulative tower layers.
            (tower height, self.k) shaped array
                ith entry has COM of the tower starting at level i
        :param tower:
        :return:
        """
        means = self.get_means(tower)
        # (tower height, self.k, 3), means for each block
        masses = self.get_masses(tower)
        # (tower height, self.k, 1), masses of each block
        # block_moments=means*masses
        # layer_moments=torch.sum(block_moments,dim=1)

        layer_moments = torch.sum(means * masses, dim=1)
        # (tower height, 3), moments (mass*com) of each layer
        layer_masses = torch.sum(masses, dim=1)
        # (tower height, 1), masses of each layer
        suffix_COMs = tail_cumsum(layer_moments, dim=0) / tail_cumsum(
            layer_masses, dim=0
        )
        # to get the COM of the tower from layer i onwards, it is the sum of the respective moments divided by the sum of the respecitve masses
        return suffix_COMs

    def check_stability(self, tower):
        """
        returns stability, the probability that the tower remains standing
        :param tower:
        :return:
        """
        coms = self.get_cumulative_coms(tower)
        instability_scores = []
        for layer, com in enumerate(coms[1:]):
            block_idxs = torch.where(tower[layer, :, 8] > 0)[0]
            if len(block_idxs) == 0:
                # all blocks were removed from this layer, the tower is unstable
                return torch.tensor(0.0)
            vertices = tower[layer, block_idxs, 9:17].reshape(-1, 2)
            hull = ConvexHull(vertices)
            eqns = torch.tensor(hull.equations, dtype=torch.float)

            dists = eqns @ torch.concatenate((com[:2], torch.ones(1)))

            if torch.all(dists <= 0):
                instability_scores.append(torch.max(dists))
            else:
                # in this case, the closest point will either be a vertex or a projection onto a line

                # projections to each line is the point minus (error * line vector)
                projections = com[:2] - eqns[:, :-1] * dists.reshape(-1, 1)

                # filter projections by the ones actually within the hull
                # shaped (num_equations, 2)
                aug_proj = torch.concatenate(
                    (projections, torch.ones(projections.shape[0], 1)), dim=-1
                )
                # eqns@aug_proj.T is an (num_eqns,num_eqns) array that represents each projection's distance from each equation
                #  row i is the distance of every point from equation i
                # taking the max over the 0th dimension gets for each point the max distance to all the equaitons
                # only consider points where this value is <=0

                projections = projections[
                    torch.le(torch.max(eqns @ aug_proj.T, dim=0).values, self.tolerance)
                ]

                # points to check are valid projections and the vertices of original hull
                points = torch.concatenate((projections, vertices))

                # min distance to all points
                min_proj_dist = torch.min(torch.linalg.norm(points - com[:2], axis=1))
                instability_scores.append(min_proj_dist)
        if self.deterministic:
            if torch.max(torch.tensor(instability_scores)) >= 0:
                return torch.tensor(0.0)
            else:
                return torch.tensor(1.0)
        else:
            # if instability scores are large, tower is unstable (and 1/(1+e^score) is close to 0)
            #  otherwise, tower is stable (product of many numbers close to 1)

            surviving_probs = 1 / (
                1 + torch.exp(self.sigmoid_constant * torch.tensor(instability_scores))
            )
            return torch.prod(surviving_probs)

    def generate_random_blocks(self, means, yaws, properties=None, pos_std=None):
        """
        generates random blocks, centered at specified means (n,3) and yaws (n,)
        if properties is specified (shaped (n,self.BLOCK_VEC_DIM)),
            will use physical properties (length, mass) from those blocks with no sampling
            will still sample means and yaws
        """
        n = len(yaws)
        if properties is None:
            sizes = self.std_block_size.repeat((n, 1))
            densities = torch.normal(mean=self.std_wood_density, std=0, size=(n, 1))
            volumes = torch.prod(sizes, dim=-1, keepdim=True)
            masses = volumes * densities
        else:
            sizes = properties[:, 3:6]
            densities = properties[:, 7:8]
            masses = properties[:, 8:9]
        if pos_std is None:
            pos_std = 0.002 / self.scale
        means[:, :2] += torch.normal(0, pos_std, means[:, :2].shape)

        # vectors representing the block's length, after the yaw is applied
        # (num_blocks,2)
        block_lengths = torch.stack(
            ((sizes[:, 0] / 2) * torch.cos(yaws), (sizes[:, 0] / 2) * torch.sin(yaws)),
            dim=-1,
        )
        # vector representing the block's width, after the yaw is applied
        block_widths = torch.stack(
            (-(sizes[:, 1] / 2) * torch.sin(yaws), (sizes[:, 1] / 2) * torch.cos(yaws)),
            dim=-1,
        )

        # to get the vertices for each block, we will add the (+ and -) length and width vectors. 2 choices for each makes 4 vertices
        #  block_lengths+block_widths, block_lengths-block_widths, ...
        # do this vectorized by creating 'orientations', which is [1,-1,-1,1],[1,1,-1,-1] before reshaping
        angles = torch.arange(4) * torch.pi / 2 + torch.pi / 4
        orientations = np.sqrt(2) * torch.stack(
            (torch.cos(angles), torch.sin(angles))
        ).reshape(2, 1, 1, 4)
        # shaped (2,4), is [1,-1,-1,1],[1,1,-1,-1]

        block_lw = torch.stack((block_lengths, block_widths), dim=0)
        # (2, num_blocks, 2)
        vertices = torch.sum(block_lw.unsqueeze(-1) * orientations, dim=0).permute(
            0, 2, 1
        )
        # (num_blocks, 4,2), a list of vertices (x,y) for each block
        vertices = vertices + means[:, :2].unsqueeze(1)
        # recenter around xy means (reshaped to be num_blocks, 1, 2)
        vertices = vertices.reshape(-1, 8)

        return torch.concatenate(
            (
                means,  # 0,1,2
                sizes,  # 3,4,5
                yaws.unsqueeze(-1),  # 6
                densities,  # 7
                masses,  # 8
                vertices,  # 9-16
            ),
            dim=-1,
        )

    def num_agents(self):
        return self.num_players

    def init_state(self):
        tower = self.generate_initial_tower()
        player = 0
        phase = 0  # 0 for pick, 1 for place
        stored_block = torch.zeros(self.BLOCK_VEC_DIM)
        return tower, player, phase, stored_block

    def player(self, state):
        _, player, _, _ = state
        return player

    def step(self, state, action):
        tower, player, phase, stored_block = state
        new_tower = tower.clone()
        if action[0] >= len(new_tower):
            new_tower = torch.concatenate(
                (new_tower, torch.zeros(1, self.k, self.BLOCK_VEC_DIM)), dim=0
            )
        if phase == 0:
            block = tower[action[0], action[1]]
            new_tower[action[0], action[1]] = 0
            new_state = new_tower, player, 1, block
        else:
            mean = torch.tensor([0, 0, self.std_block_size[2] * (action[0] + 0.5)])
            # offset the mean in the appropriate direction
            mean[action[0] % 2 + 1] += (action[1] - (self.k - 1) / 2) * (
                self.std_block_size[1] + self.std_block_spacing
            )
            yaw = torch.tensor(((action[0] % 2) * torch.pi / 2,))
            new_tower[action[0], action[1]] = self.generate_random_blocks(
                means=mean.unsqueeze(0), yaws=yaw, properties=stored_block.unsqueeze(0)
            )
            new_state = (
                new_tower,
                (player + 1) % self.num_players,
                0,
                torch.zeros(self.BLOCK_VEC_DIM),
            )
        terminal = False
        rwd = torch.zeros(self.num_players)
        # tower is stable with self.check_stability(new_tower) probability
        #  with 1- this probability, it is unstable
        if torch.rand(1) > self.check_stability(new_tower):
            terminal = True
            rwd[player] = -1.0
        return new_state, rwd, terminal, dict()

    def agent_observe(self, state):
        """
        agent observations
        Args:
            state: The state of the environment.
        """
        tower, _, phase, block = state
        tower_obs = tower[:, :, :9]
        return tower_obs, torch.concatenate((torch.tensor((phase,)), block))

    def action_mask(self, state):
        """
        possible actions to take
        Args:
            state: The state of the environment.
        """

        tower, _, phase, _ = state
        if phase == 0:
            # mass > 0 means block exists there
            action_mask = tower[:, :, 8] > 0
            if not torch.all(action_mask[-1]):
                # if top layer is not full, we cannot draw a block from top two layers
                action_mask[-2:] = False
            else:
                # blocks can never be drawn from top row
                action_mask[-1:] = False
            return action_mask

        else:
            # if any open squares on top level, we can place there
            if torch.any(tower[-1, :, 8] == 0):
                action_mask = tower[:, :, 8] == 0
                action_mask[:-1] = False
                return action_mask
            else:
                # otherwise, add a layer to tower, and we may place anywhere on that layer
                action_mask = torch.zeros(tower.shape[0] + 1, self.k, dtype=torch.bool)
                action_mask[-1] = True
                return action_mask

    def critic_observe(self, state):
        """
        critic observations
        Args:
            state: The state of the environment.
        """
        return self.agent_observe(state)

    def render_pyplot(self, state, ax=None):
        tower, _, phase, stored_block = state
        if ax is None:
            ax = plt.figure().add_subplot(projection="3d")

        ax.set_aspect("equal")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_zticks(())

        colors = ["purple", "red", "blue", "orange", "brown", "yellow"]
        counter = 0
        top_layer_filled = torch.all(tower[-1, :, 8] != 0)

        place_height = len(tower) - 1 + top_layer_filled
        # if h<pick_ht_bound, this is a valid block to pick
        pick_ht_bound = place_height - 1

        for h in range(place_height + 1):
            for i in range(self.k):
                label = str((h, i))

                if h >= len(tower):
                    block_exists = False
                    block = None
                else:
                    block = tower[h, i]
                    block_exists = block[8] > 0
                if block_exists:
                    if phase == 1 or h >= pick_ht_bound:
                        # place phase, we dont care about these labels
                        # or block cannot legally be chosen, we do not need to list label
                        label = None
                    self.render_block(
                        block=block,
                        ax=ax,
                        color=colors[counter % len(colors)],
                        label=label,
                        only_frame=False,
                    )
                    counter += 1
                elif h == place_height and phase == 1:
                    # render the frames of the block placement
                    mean = torch.tensor([0, 0, self.std_block_size[2] * (h + 0.5)])
                    # offset the mean in the appropriate direction
                    mean[h % 2 + 1] += (i - (self.k - 1) / 2) * (
                        self.std_block_size[1] + self.std_block_spacing
                    )
                    yaw = torch.tensor(((h % 2) * torch.pi / 2,))
                    block = self.generate_random_blocks(
                        means=mean.unsqueeze(0),
                        yaws=yaw,
                        properties=stored_block.unsqueeze(0),
                        pos_std=0,
                    )
                    self.render_block(
                        block=block.flatten(),
                        ax=ax,
                        label=label,
                        color="black",
                        linestyle="--",
                        only_frame=True,
                    )

        plt.ion()
        plt.show()

    def render_block(self, block, ax, only_frame, label=None, **plot_kwargs):
        vertices = block[9:17].reshape(-1, 2)
        # 2d projections of block vertices, arranged in a cycle

        # mean z +- 1/2block width
        heights = (torch.arange(2) - 1 / 2) * block[5] + block[2]

        #  vertices.tile(2,1,1) is a 2,4,2 element that is two copies of vertices
        # heights.reshape(2,1,1).tile(1,4,1) is a (2,4,1) array whose 0th element is 4 copies of heights[0]
        vertices = torch.concatenate(
            (vertices.tile((2, 1, 1)), heights.reshape(2, 1, 1).tile(1, 4, 1)), dim=2
        )
        # vertices is now a 2,4,3 array where vertices[0] and vertices[1] are each a cycle of vertices at the lower and upper face of the block
        vertices = vertices.reshape(2, 2, 2, 3)

        # rearrange vertices, so now vertices[0] and vertices[1] each have the same x directions
        #  vertices{:,0] and vertices[:,1] have same y, etc.
        vertices[:, (1,)] = vertices[:, (1,)].flip(2)
        vertices = vertices.permute((1, 2, 0, 3))
        if only_frame:
            for i, j in itertools.product(range(2), repeat=2):
                for line in (
                    vertices[i, j, :, :],
                    vertices[:, i, j, :],
                    vertices[i, :, j, :],
                ):
                    x, y, z = line.T
                    ax.plot(x, y, z, **plot_kwargs)
        else:
            for dir in range(2):
                for square in (
                    vertices[dir, :, :, :],  # z square
                    vertices[:, dir, :, :],  # y square
                    vertices[:, :, dir, :],  # x square
                ):
                    x, y, z = square[:, :, 0], square[:, :, 1], square[:, :, 2]
                    ax.plot_surface(x, y, z, **plot_kwargs)

        if label is not None:
            center = torch.mean(vertices.reshape((-1, 3)), dim=0)
            edge = torch.mean(vertices[:, 0, :, :].reshape((-1, 3)), dim=0)
            vec = edge - center
            label_pos = center + vec * 1.1
            x, y, z = label_pos
            ax.text(x, y, z, label, backgroundcolor="white")


if __name__ == "__main__":
    g = Jenga()
    s = g.init_state()
    s, _, _, _ = g.step(s, torch.tensor([0, 0]))
    s, _, _, _ = g.step(s, torch.tensor([18, 1]))
    s, _, t, _ = g.step(s, torch.tensor([0, 1]))
    print(t)
    g.render_pyplot(s)
    input()
