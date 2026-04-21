from .chess5d import Chess2d as Chess2d
from .chess5d import Chess5d as Chess5d
from .game import Game as Game
from .jenga import Jenga as Jenga
from .mnk import MNK as MNK
from .mnk import TicTacToe as TicTacToe
from .mnk_with_resigning import F_MNK as F_MNK
from .pyspiel_games import Checkers as Checkers

__all__ = ["Game", "Jenga", "MNK", "F_MNK", "TicTacToe", "Chess5d", "Chess2d", "Checkers"]
