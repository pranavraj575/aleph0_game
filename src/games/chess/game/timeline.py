import torch
from aleph0.examples.chess.game.board import Board


class Timeline:
    def __init__(self, board_list=None, start_idx=0):
        if board_list is None:
            board_list = []
        self.start_idx = start_idx
        self.board_list = board_list

    @property
    def representation(self):
        return [board.representation for board in self.board_list], self.start_idx

    @staticmethod
    def from_representation(representation):
        board_list, start_idx = representation
        return Timeline(board_list=[Board.from_representation(board_rep) for board_rep in board_list],
                        start_idx=start_idx,
                        )

    def is_empty(self):
        return len(self.board_list) == 0

    def clone(self):
        return Timeline(
            board_list=[board.clone() for board in self.board_list],
            start_idx=self.start_idx,
        )

    def flipped(self):
        return Timeline(board_list=[board.flipped() for board in self.board_list],
                        start_idx=self.start_idx,
                        )

    def get_board(self, real_time: int):
        """
        gets board at time, where time is the 'real' index, relative to 0
        """
        if self.in_range(real_time=real_time):
            return self.board_list[real_time - self.start_idx]
        else:
            return None

    def get_board_as_idxs_stack(self):
        return torch.stack([board.get_board_as_indices() for board in self.board_list], dim=0)

    def get_time_range(self):
        """
        valid time range [start, end)
        """
        return (self.start_idx, self.start_idx + len(self.board_list))

    def end_time(self):
        """
        last possible time
        """
        return self.start_idx + len(self.board_list) - 1

    def in_range(self, real_time):
        start, end = self.get_time_range()
        return start <= real_time and real_time < end

    def append(self, board: Board):
        self.board_list.append(board)

    def pop(self):
        self.board_list.pop()

    def __str__(self):
        s = ''
        interspace = len(str(self.end_time())) + 6

        str_boards = [Board.empty_string()
                      for _ in range(self.start_idx)] + [board.__str__()
                                                         for board in self.board_list]
        str_boards = [s.split('\n') for s in str_boards]
        for row in range(2*Board.BOARD_SIZE):
            if row == Board.BOARD_SIZE:
                midstring = ''
                for time in range(len(str_boards)):
                    midtime = '   t' + str(time) + ': '
                    while len(midtime) < interspace:
                        midtime = ' ' + midtime
                    midstring += midtime
                    midstring += str_boards[time][row]

                s += midstring
            else:
                s += (' '*interspace) + (' '*interspace).join([str_board[row] for str_board in str_boards])
            s += '\n'
        return s


if __name__ == '__main__':
    t = Timeline()
    t.append(Board())
    print(t)
    b: Board = t.get_board(0)
    t.append(b.move_piece_on_board((1, 3), (3, 3), mutate=True)[0])
    print(t)
    print(t.get_board_as_idxs_stack().shape)
