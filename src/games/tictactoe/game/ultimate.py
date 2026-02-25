import torch
from aleph0.game import FixedSizeSelectionGame


class UltimateToe(FixedSizeSelectionGame):
    FULL = 3
    EMPTY = 2
    P0 = 0
    P1 = 1

    def __init__(self,
                 current_player=P0,
                 board=None,
                 active_board=None,
                 super_toe=None,
                 ):
        super().__init__(
            num_players=2,
            current_player=current_player,
            subset_size=1,
            special_moves=[],
        )

        if board is None:
            board = self.EMPTY*torch.ones((3, 3, 3, 3), dtype=torch.long)
        self.board = board
        if super_toe is None:
            self.super_toe = torch.ones(3, 3, dtype=torch.long)*self.EMPTY
            for i in range(3):
                for j in range(3):
                    self.reset_super_toe(i, j)
        else:
            self.super_toe = super_toe
        self.active_board = active_board

    def reset_super_toe(self, i, j):
        self.super_toe[i, j] = UltimateToe.test_board(self.board[i, j])

    @staticmethod
    def test_board(bored):
        """
        tests board
        returns EMPTY if game is non-terminal
        returns FULL if game is tied
        returns player that won otherwise
        """
        for player in (UltimateToe.P0, UltimateToe.P1):
            for k in range(3):
                if (torch.all(torch.eq(bored[k, :], player)) or
                        torch.all(torch.eq(bored[:, k], player))
                ):
                    return player
            if (torch.all(torch.eq(bored[range(3), range(3)], player)) or
                    torch.all(torch.eq(bored[range(3), [-1 - i for i in range(3)]], player))):
                return player
        if UltimateToe.EMPTY in bored:
            return UltimateToe.EMPTY
        else:
            return UltimateToe.FULL

    def get_valid_next_selections(self, move_prefix=()):
        if self.active_board is None:
            for (i, j) in zip(*torch.where(torch.eq(self.super_toe, self.EMPTY))):
                for (k, l) in zip(*torch.where(torch.eq(self.board[i, j], self.EMPTY))):
                    yield i.item(), j.item(), k.item(), l.item()
        else:
            i, j = self.active_board
            for (k, l) in zip(*torch.where(torch.eq(self.board[i, j], self.EMPTY))):
                yield i, j, k.item(), l.item()

    @staticmethod
    def fixed_obs_shape():
        return ((
                    (3, 3, 3, 3),
                    (3, 3, 3, 3)
                ),
                (3, 3, 3, 3, 4),
                0)

    @staticmethod
    def underlying_set_sizes():
        return (3,
                4,
                )

    def possible_move_cnt(self):
        return 81

    def index_to_move(self, idx):
        big, small = idx//9, idx%9
        return ((big//3, big%3, small//3, small%3),)

    def move_to_idx(self, move):
        return move[0][0]*27 + move[0][1]*9 + move[0][2]*3 + move[0][3]

    @staticmethod
    def invert_player(player):
        return 1 - player

    @property
    def permutation_to_standard_pos(self):
        if self.current_player == self.P1:
            return [1, 0]
        else:
            return [0, 1]

    @staticmethod
    def get_indices():
        I = torch.cat((torch.arange(3).view((3, 1, 1, 1, 1)),
                       torch.zeros((3, 1, 1, 1, 3)),
                       ), dim=-1)
        J = torch.cat((torch.zeros((1, 3, 1, 1, 1)),
                       torch.arange(3).view((1, 3, 1, 1, 1)),
                       torch.zeros((1, 3, 1, 1, 2)),
                       ), dim=-1)
        K = torch.cat((torch.zeros((1, 1, 3, 1, 2)),
                       torch.arange(3).view((1, 1, 3, 1, 1)),
                       torch.zeros((1, 1, 3, 1, 1)),
                       ), dim=-1)
        L = torch.cat((torch.zeros((1, 1, 1, 3, 3)),
                       torch.arange(3).view((1, 1, 1, 3, 1)),
                       ), dim=-1)
        return I + J + K + L

    @property
    def representation(self):
        return (self.board.clone(),
                self.get_indices(),
                torch.tensor([self.current_player]),
                self.active_board,
                self.super_toe.clone(),
                )

    @property
    def observation(self):
        """
        ignores current player, as it is always assumed to be the X player's move
        """
        if self.current_player == self.P0:
            B, P, T, active_board, super_toe = self.representation
        else:
            B, P, T, active_board, super_toe = self.representation
            p0s = torch.where(torch.eq(B, self.P0))
            p1s = torch.where(torch.eq(B, self.P1))
            B[p0s] = self.P1
            B[p1s] = self.P0

            p0s = torch.where(torch.eq(super_toe, self.P0))
            p1s = torch.where(torch.eq(super_toe, self.P1))
            super_toe[p0s] = self.P1
            super_toe[p1s] = self.P0
        super_toe = super_toe.reshape(3, 3, 1, 1).broadcast_to(3, 3, 3, 3)
        return (
            (
                B,
                super_toe,
            ),
            P,
            torch.zeros(0)
        )

    @staticmethod
    def from_representation(representation):
        board, _, vec, active_board, super_toe = representation
        return UltimateToe(board=board,
                           current_player=vec.item(),
                           active_board=active_board,
                           super_toe=super_toe,
                           )

    def make_move(self, local_move):
        board = self.board.clone()
        board[local_move[0]] = self.current_player
        active_board = local_move[0][2:]
        super_toe = self.super_toe.clone()
        super_toe[local_move[0][:2]] = self.test_board(board[local_move[0][:2]])

        if super_toe[active_board] != UltimateToe.EMPTY:
            active_board = None

        return UltimateToe(current_player=self.invert_player(self.current_player),
                           board=board,
                           active_board=active_board,
                           super_toe=super_toe,
                           )

    def is_terminal(self):
        # board is terminal if the superboard is not empty
        return UltimateToe.test_board(self.super_toe) is not UltimateToe.EMPTY

    def get_result(self):
        teste = UltimateToe.test_board(self.super_toe)
        if teste == UltimateToe.P0:
            return (1., 0.)
        if teste == UltimateToe.P1:
            return (0., 1.)
        return (.5, .5)

    def symmetries(self, policy_vector):
        yield (self, policy_vector)

    def __str__(self):
        stuff = []
        for i in range(3):
            r = []
            for j in range(3):
                if self.super_toe[i, j] != UltimateToe.EMPTY:
                    c = self.super_toe[i, j]
                    if c == UltimateToe.FULL:
                        c = ' '
                    elif c == UltimateToe.P0:
                        c = 'X'
                    elif c == UltimateToe.P1:
                        c = 'O'
                    s = ['   ', ' ' + c + ' ', '   ']
                    r.append(s)
                else:
                    s = []
                    for row in self.board[i, j]:
                        row = row.numpy()
                        s.append(str(row).replace(' ', ''
                                                  ).replace('[', ''
                                                            ).replace(']', ''
                                                                      ).replace('-1', ' '
                                                                                ).replace('0', 'X'
                                                                                          ).replace('1', 'O'
                                                                                                    ).replace('2', '.')
                                 )
                    r.append(s)
            stuff.append(r)
        thing = ''
        for i, row in enumerate(stuff):
            if i > 0:
                thing += '-'*11 + '\n'
            for k in range(3):
                for l, mini in enumerate(row):
                    if l > 0:
                        thing += '|'
                    thing += mini[k]
                thing += '\n'
        return thing
        temp = torch.zeros(9, 9, dtype=torch.long)
        for i in range(3):
            for j in range(3):
                temp[i*3:(i + 1)*3, j*3:(j + 1)*3] = self.board[i, j]
        temp = temp.numpy()
        thing = ''

        for i, row in enumerate(temp):
            s = str(row).replace(' ', ''
                                 ).replace('[', ''
                                           ).replace(']', ''
                                                     ).replace('-1', ' '
                                                               ).replace('0', 'X'
                                                                         ).replace('1', 'O'
                                                                                   ).replace('2', '.')

            thing += s[:3] + '|' + s[3:6] + '|' + s[6:] + '\n'
            if not (i + 1)%3 and i < 8:
                thing += '-'*11 + '\n'

        return thing


class LessUltimateToe(UltimateToe):
    """
    ultimate tic tac toe 2d representation
    all aelse is same
    """

    @staticmethod
    def get_indices():
        I = torch.cat((torch.arange(9).view((9, 1, 1)),
                       torch.zeros((9, 1, 1)),
                       ), dim=-1)
        J = torch.cat((torch.zeros((1, 9, 1)),
                       torch.arange(9).view((1, 9, 1)),
                       ), dim=-1)
        return I + J

    @staticmethod
    def fixed_obs_shape():
    
        return ((
                    (9, 9),
                    (9, 9)
                ),
                (9, 9, 2),
                0)
        B, P, T = UltimateToe.fixed_obs_shape()
        # combine the first and third, second and fourth dimensions
        B = tuple(
            tuple(
                [b[0]*b[2], b[1]*b[3]] + list(b[4:])
            )
            for b in B)
        return B, P, T

    @property
    def observation(self):
        """
        ignores current player, as it is always assumed to be the X player's move
        """
        Bs, _, T = super().observation
        # first swap so we have dimensions in order (big I, mini i, big J, mini j, extra)
        # then flatten (big J mini j) and then (big I, mini i) dimensions
        return tuple(
            B.transpose(1, 2).flatten(2, 3).flatten(0, 1)
            for B in Bs), self.get_indices(), T

    def get_valid_next_selections(self, move_prefix=()):
        for i,j,k,l in super().get_valid_next_selections(move_prefix=move_prefix):
            yield i*3+k,j*3+l
    
    def make_move(self, local_move):
        (i,j),=local_move
        global_move=(i//3,j//3,i%3,j%3),
        
        board = self.board.clone()
        board[global_move[0]] = self.current_player
        active_board = global_move[0][2:]
        super_toe = self.super_toe.clone()
        super_toe[global_move[0][:2]] = self.test_board(board[global_move[0][:2]])

        if super_toe[active_board] != LessUltimateToe.EMPTY:
            active_board = None

        return LessUltimateToe(current_player=self.invert_player(self.current_player),
                           board=board,
                           active_board=active_board,
                           super_toe=super_toe,
                           )
if __name__ == '__main__':
    toe = LessUltimateToe()
    while not toe.is_terminal():
        toe = toe.make_move(next(toe.get_all_valid_moves()))
        print(toe)
    print(toe.get_result())
    toe.possible_move_cnt()
