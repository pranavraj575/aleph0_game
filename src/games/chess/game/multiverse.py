from aleph0.examples.chess.game.board import Board
from aleph0.examples.chess.game.timeline import Timeline


class Multiverse:
    def __init__(
        self,
        main_timeline: Timeline,
        up_list: [Timeline] = None,
        down_list: [Timeline] = None,
    ):
        self.main_timeline = main_timeline
        if up_list is None:
            up_list = []
        if down_list is None:
            down_list = []
        self.up_list = up_list
        self.down_list = down_list
        self.max_length = None
        self._set_max_length()

    @property
    def representation(self):
        return (
            self.main_timeline.representation,
            [timeline.representation for timeline in self.up_list],
            [timeline.representation for timeline in self.down_list],
        )

    @staticmethod
    def from_representation(representation):
        main_rep, up_list, down_list = representation
        return Multiverse(
            main_timeline=Timeline.from_representation(main_rep),
            up_list=[Timeline.from_representation(rep) for rep in up_list],
            down_list=[Timeline.from_representation(rep) for rep in down_list],
        )

    def _set_max_length(self):
        self.max_length = self.main_timeline.end_time() + 1
        for listt in self.up_list, self.down_list:
            for timeline in listt:
                self.max_length = max(self.max_length, timeline.end_time() + 1)

    def clone(self):
        return Multiverse(
            main_timeline=self.main_timeline.clone(),
            up_list=[None if timeline is None else timeline.clone() for timeline in self.up_list],
            down_list=[None if timeline is None else timeline.clone() for timeline in self.down_list],
        )

    def flipped(self):
        return Multiverse(
            main_timeline=self.main_timeline.flipped(),
            up_list=[None if timeline is None else timeline.flipped() for timeline in self.down_list],
            down_list=[None if timeline is None else timeline.flipped() for timeline in self.up_list],
        )

    def get_board(self, td_idx):
        time_idx, dim_idx = td_idx
        timeline = self.get_timeline(dim_idx=dim_idx)
        if timeline is None:
            return None
        else:
            return timeline.get_board(time_idx)

    def add_board(self, td_idx, board):
        """
        adds board at specified td_idx
        :param board:
        :return:
        """
        time_idx, dim_idx = td_idx
        if dim_idx == 0:
            self.main_timeline.append(board=board)
            self.max_length = max(self.max_length, self.main_timeline.end_time() + 1)
            return

        if dim_idx > 0:
            dim_idx, listt = dim_idx - 1, self.up_list
        else:
            dim_idx, listt = -dim_idx - 1, self.down_list

        if dim_idx >= len(listt):
            # this should only need to happen once
            timeline = Timeline(start_idx=time_idx)
            listt.append(timeline)
        else:
            timeline = listt[dim_idx]
        timeline.append(board=board)
        self.max_length = max(self.max_length, timeline.end_time() + 1)

    def get_range(self):
        """
        returns range of indices (inclusive)
        """
        return (-len(self.down_list), len(self.up_list))

    def in_range(self, dim):
        bot, top = self.get_range()
        return bot <= dim and dim <= top

    def idx_exists(self, td_idx):
        time, dim = td_idx
        return self.in_range(dim) and self.get_timeline(dim).in_range(time)

    def leaves(self):
        """
        where did you go
        :return: coordinates of all final states

        WILL ALWAYS RETURN COORDS IN SAME ORDER
        """
        overall_range = self.get_range()
        for dim_idx in range(overall_range[0], overall_range[1] + 1):
            yield (self.get_timeline(dim_idx=dim_idx).end_time(), dim_idx)

    def get_timeline(self, dim_idx):
        if dim_idx > 0:
            dim_idx = dim_idx - 1
            if dim_idx < len(self.up_list):
                return self.up_list[dim_idx]
            else:
                return None
        elif dim_idx < 0:
            dim_idx = -dim_idx - 1
            if dim_idx < len(self.down_list):
                return self.down_list[dim_idx]
            else:
                return None
        else:
            return self.main_timeline

    def remove_board(self, dim):
        """
        removes last board at specified dimension, removes timeline if no longer exists
        """
        if self.in_range(dim):
            if dim > 0:
                self.up_list[dim - 1].pop()
                while self.up_list and self.up_list[-1].is_empty():
                    self.up_list.pop()
            elif dim < 0:
                self.down_list[-dim - 1].pop()
                while self.down_list and self.down_list[-1].is_empty():
                    self.down_list.pop()
            else:
                self.main_timeline.pop()
            self._set_max_length()

    def __str__(self):
        s = ""
        overall_range = self.get_range()
        for dim in range(overall_range[1], overall_range[0] - 1, -1):
            timeline = self.get_timeline(dim_idx=dim)
            s += "dimension " + str(dim) + ":\n"
            s += timeline.__str__()
            s += "\n\n"
        return s


if __name__ == "__main__":
    m = Multiverse(main_timeline=Timeline(board_list=[Board()]))
    m.add_board((1, -1), Board())
    print(m)
