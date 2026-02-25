import itertools

import numpy as np


class Block:
    STD_BLOCK_DIM = np.array((.075, .025, .015))  # in meters
    STD_BLOCK_SPACING = .005  # in meters

    STD_WOOD_DENSITY = 0.5  # kg/m^3

    def __init__(self,
                 pos,
                 yaw=0.,
                 block_dim=None,
                 density=None,
                 ):
        """
        represents one block
        :param pos: position, np array
        :param yaw: represents orientation

        Note: generally, roll and pitch are 0 and yaw is the only relevant one
        """
        self.pos = pos
        self.yaw = yaw
        if block_dim is None:
            block_dim = Block.STD_BLOCK_DIM
        if density is None:
            density = Block.STD_WOOD_DENSITY
        self.block_dim = block_dim
        self.density = density
        self.mass = np.prod(self.block_dim)*self.density

    @staticmethod
    def vector_size():
        return 3 + 3 + 2

    @property
    def representation(self):
        """
        returns block encoded as a vector

        x,y,z,angle
        """
        return np.concatenate((self.pos, self.block_dim, [self.yaw, self.density]))

    @staticmethod
    def from_representation(representation):
        pos = representation[:3]
        block_dim = representation[3:6]
        yaw, density = representation[6:]
        return Block(pos=pos,
                     yaw=yaw,
                     block_dim=block_dim,
                     density=density,
                     )

    def vertices(self):
        """
        returns vertices of block
        ordered in the following way:
            consider the 2 possiblities for each axis (+x,-x), (+y,-y), (+z,-z)
            vertices_xy()[0,:,:,:] will be all -z points
            vertices_xy()[1,:,:,:] will be all +z points
            vertices_xy()[:,0,:,:] will be all -y points
            vertices_xy()[:,1,:,:] will be all +y points
            vertices_xy()[:,:,0,:] will be all -x points
            vertices_xy()[:,:,1,:] will be all +x points
        :return: 2x2x2x3 array
        """

        dx, dy, dz = self.block_dim

        return np.array([[[
            (X,
             Y,
             self.pos[2] + dz*(z_i - .5),
             ) for X, Y in T]
            for T in self.vertices_xy()]
            for z_i in range(2)])

        return np.array([[
            (X,
             Y,
             self.pos[2] + dz*(z_i - .5),
             )
            for (X, Y) in self.vertices_xy()] for z_i in range(2)]).reshape((8, 3))

    def vertices_xy(self):
        """
        returns xy projected vertices of block
        ordered in the following way:
            consider the 2 possiblities for each axis (+x,-x) and (+y,-y)
            vertices_xy()[0,:,:] will be both -y points
            vertices_xy()[1,:,:] will be both +y points
            vertices_xy()[:,0,:] will be both -x points
            vertices_xy()[:,1,:] will be both +x points

        :return: 2x2x2 array
        """
        dx, dy, _ = self.block_dim
        return self.pos[:2] + np.array([[
            (dx*(x_i - .5)*np.cos(self.yaw) - dy*(y_i - .5)*np.sin(self.yaw),
             dx*(x_i - .5)*np.sin(self.yaw) + dy*(y_i - .5)*np.cos(self.yaw),
             )
            for x_i in range(2)]
            # for x_i in (range(2) if y_i == 0 else range(1, -1, -1))]  # go in reverse for y_i=1
            for y_i in range(2)])  # .reshape((4, 2))

    def com(self):
        """
        :return: np array, center of mass (is just pos)
        """
        return self.pos

    def __eq__(self, other):
        return (np.array_equal(self.pos, other.pos) and
                self.yaw == other.yaw and
                np.array_equal(self.block_dim, other.block_dim) and
                self.density == other.density)

    @staticmethod
    def random_block(L, i, pos_std=0., angle_std=0.):
        """
        creates a randomly placed block at specified level and index
        :param L: level of block in tower
        :param i: index of block in level (increases +x for even levels, +y for odd levels)
            0 is the furthest negative, 2 is the furthest positive
        :param pos_std: std randomness of position
        :param angle_std: std randomness of angle
        :return: block object
        """

        rot = (L%2)*np.pi/2  # rotate yaw if odd level

        pos = np.zeros(3)
        pos += (0, 0, (L + 0.5)*Block.STD_BLOCK_DIM[2])  # height of level + half the height of a block (since COM)

        width = Block.STD_BLOCK_DIM[1]
        if L%2:
            pos += ((i - 1)*(width + Block.STD_BLOCK_SPACING), 0, 0)
        else:
            pos += (0, (i - 1)*(width + Block.STD_BLOCK_SPACING), 0)

        rot = rot + np.random.normal(0, angle_std)
        pos = pos + (np.random.normal(0, pos_std), np.random.normal(0, pos_std), 0.)

        return Block(pos=pos, yaw=rot)

    def render_frame_to(self,
                        ax,
                        label=None,
                        **plot_kwargs,
                        ):
        """
        renders frame to specified pyplot axis
        Args:
            ax: must be created with plt.figure().add_subplot(projection='3d')
            plot_kwargs: kwargs for plot (i.e. color, etc)
        """
        # we want to consider each face separately
        # i.e. all four (+x) points, all four (-x) points
        # there will be 6 faces, 2 for each dim
        vertices = self.vertices()
        for penis, poonis in itertools.product(range(2), repeat=2):
            for line in (vertices[penis, poonis, :, :],
                         vertices[:, penis, poonis, :],
                         vertices[penis,:,  poonis, :],
                         ):
                x, y, z = line.T
                ax.plot(x, y, z, **plot_kwargs)

        center = np.mean(vertices.reshape((-1, 3)), axis=0)
        edge = np.mean(vertices[:, :, 0, :].reshape((-1, 3)), axis=0)
        vec = edge - center
        label_pos = edge + vec*.69
        x, y, z = label_pos
        if label is not None:
            ax.text(x, y, z, label, backgroundcolor='white')

    def render_to(self,
                  ax,
                  label=None,
                  **plot_kwargs,
                  ):
        """
        renders to specified pyplot axis
        Args:
            ax: must be created with plt.figure().add_subplot(projection='3d')
            plot_kwargs: kwargs for plot (i.e. color, etc)
        """
        # we want to consider each face separately
        # i.e. all four (+x) points, all four (-x) points
        # there will be 6 faces, 2 for each dim
        vertices = self.vertices()

        for dir in range(2):
            for square in (vertices[dir, :, :, :],  # z square
                           vertices[:, dir, :, :],  # y square
                           vertices[:, :, dir, :],  # x square
                           ):
                x, y, z = square[:, :, 0], square[:, :, 1], square[:, :, 2]
                ax.plot_surface(x, y, z, **plot_kwargs)
        center = np.mean(vertices.reshape((-1, 3)), axis=0)

        edge = np.mean(vertices[:, :, 0, :].reshape((-1, 3)), axis=0)
        vec = edge - center
        label_pos = edge + vec*.69
        x, y, z = label_pos
        if label is not None:
            ax.text(x, y, z, label, backgroundcolor='white')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    blocks = [[Block.random_block(L, i, pos_std=.001, angle_std=.003) for i in range(3)] for L in range(5)]

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())

    colors = ['purple', 'red', 'blue', 'orange', 'brown', 'yellow']
    counter = 0
    for L, row in enumerate(blocks[:-1]):
        for (i, b) in enumerate(row):
            b.render_to(ax,
                        color=colors[counter%len(colors)],
                        alpha=.9,
                        label=str((L, i)))
            counter += 1
    for (i, b) in enumerate(blocks[-1]):
        b.render_frame_to(ax,
                    color=colors[counter%len(colors)],
                    alpha=.9,
                    label=str((len(blocks) - 1, i)))
        counter += 1
    plt.show()
