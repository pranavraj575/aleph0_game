import dataclasses

import numpy as np
import pyspiel
import torch

from .game import Game


@dataclasses.dataclass
class State:
    state: str
    index_actions: tuple


class PyspielGame(Game):
    """
    use pyspiel implemented games
    """

    def __init__(self, game_string):
        self.pyspiel_game: pyspiel.Game = pyspiel.load_game(game_string)

    def num_agents(self):
        return self.pyspiel_game.num_players()

    def advance_past_chance(self, pyspiel_state: pyspiel.State):
        rewards = torch.zeros(self.num_agents())
        while pyspiel_state.is_chance_node():
            actions, p = tuple(zip(pyspiel_state.chance_outcomes()))
            pyspiel_state.apply_action(np.random.choice(actions, p=p))
            rewards += pyspiel_state.rewards()
        return pyspiel_state, rewards

    def init_state(self):
        # (board, player)
        pyspiel_state: pyspiel.State = self.pyspiel_game.new_initial_state()
        return State(state=pyspiel_state.serialize(), index_actions=())

    def actions_per_pyspiel_action(self):
        """
        number of 'index choice' actions in each pyspiel action
        """
        raise NotImplementedError

    def convert_to_pyspiel_action(self, state, actions):
        """
        converts a list of (self.actions_per_pyspiel_action()) indices to a single pyspiel action
        :param actions:
        :return:
        """
        raise NotImplementedError

    def step(self, state, action):
        new_index_actions = state.index_actions + (action,)
        rewards = torch.zeros(self.num_agents())
        if len(new_index_actions) == self.actions_per_pyspiel_action():
            pyspiel_state = self.pyspiel_game.deserialize_state(state.state)
            pyspiel_state.apply_action(self.convert_to_pyspiel_action(pyspiel_state, new_index_actions))
            rewards = pyspiel_state.rewards()
            pyspiel_state, r = self.advance_past_chance(pyspiel_state)
            rewards += r
            terminal = pyspiel_state.is_terminal()
            new_serialized = pyspiel_state.serialize()
            new_index_actions = ()
        else:
            terminal = False
            new_serialized = state.state
        return State(state=new_serialized, index_actions=new_index_actions), rewards, terminal, dict()

    def player(self, state):
        pyspiel_state = self.pyspiel_game.deserialize_state(state.state)
        return pyspiel_state.current_player()

    def agent_observe(self, state):
        pyspiel_state: pyspiel.State = self.pyspiel_game.deserialize_state(state.state)
        return torch.tensor(pyspiel_state.observation_tensor()).reshape(self.pyspiel_game.observation_tensor_shape())

    def action_mask(self, state):
        """
        possible actions to take
        Args:
            state: The state of the environment.
        """
        raise NotImplementedError

    def critic_observe(self, state):
        """
        critic observations
        Args:
            state: The state of the environment.
        """
        pyspiel_state: pyspiel.State = self.pyspiel_game.deserialize_state(state.state)
        return torch.tensor(pyspiel_state.observation_tensor()).reshape(self.pyspiel_game.observation_tensor_shape())

    def render(self, canvas, state):
        # canvas is not needed, just print it to terminal
        pyspiel_state: pyspiel.State = self.pyspiel_game.deserialize_state(state.state)
        if pyspiel_state.is_terminal():
            print("TERMINAL:")
            print(pyspiel_state)
        else:
            print(pyspiel_state.observation_string())


class Checkers(PyspielGame):
    def __init__(self):
        super().__init__(game_string="checkers")

    def convert_square_to_idx(self, square):
        return torch.tensor((int(square[1]) - 1, ord(square[0]) - 97))

    def convert_idx_to_square(self, idx):
        return str(chr(idx[1] + 97)) + str(int(idx[0] + 1))

    def action_mask(self, state: State):
        """
        possible actions to take
        Args:
            state: The state of the environment.
        """
        pyspiel_state: pyspiel.State = self.pyspiel_game.deserialize_state(state.state)
        legal_actions = pyspiel_state.legal_actions()
        action_mask = torch.zeros((8, 8), dtype=torch.bool)
        if not state.index_actions:
            for action in legal_actions:
                square = pyspiel_state.action_to_string(action)
                action_mask[*self.convert_square_to_idx(square[:2])] = True
        else:
            square = self.convert_idx_to_square(state.index_actions[0])
            for action in legal_actions:
                if pyspiel_state.action_to_string(action)[:2] == square:
                    place_square = pyspiel_state.action_to_string(action)[2:]
                    action_mask[*self.convert_square_to_idx(place_square)] = True

        return action_mask

    def actions_per_pyspiel_action(self):
        return 2

    def convert_to_pyspiel_action(self, pyspiel_state, actions):
        p, q = actions
        return pyspiel_state.string_to_action(self.convert_idx_to_square(p) + self.convert_idx_to_square(q))
