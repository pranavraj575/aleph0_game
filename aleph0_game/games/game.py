import torch


def tense_cast(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x)


class Game:
    def has_special_actions(self):
        return False

    def num_agents(self):
        raise NotImplementedError

    def init_state(self):
        """
        Initial state of the environment.
        Returns:
            The initial state of the environment.
        """
        raise NotImplementedError

    def player(self, state):
        raise NotImplementedError

    def step_weak_type(self, state, action):
        """
        casts action to tensor then returns step
        """
        if self.has_special_actions():
            board_a, special_a = action
            return self.step(state, (tense_cast(board_a), tense_cast(special_a)))
        else:
            return self.step(state, tense_cast(action))

    def step(self, state, action):
        """
        Update the environment.

        Args:
            state: env state.
            action: The actions taken by the agents.
        Returns:
        A tuple containing:
            (new state,
            rewards (tensor),
            terminal (boolean),
            auxiliary information dictionary)
        """
        raise NotImplementedError

    def action_mask(self, state):
        """
        possible actions to take
        Args:
            state: The state of the environment.
        """
        raise NotImplementedError

    def sample_from_action_mask(self, action_mask):
        if self.has_special_actions():
            board_mask, special_mask = action_mask
            assert board_mask.dtype == torch.bool
            assert special_mask.dtype == torch.bool

            # easier if swapped here, since we can test action < len(special_mask)
            combined_mask = torch.concat((special_mask, board_mask.flatten()))
            action = torch.multinomial(combined_mask.to(torch.float), 1, True)
            if action < len(special_mask):
                return (-torch.ones(len(board_mask.shape), dtype=torch.int), action)
            else:
                return (
                    torch.cat(torch.unravel_index(action - len(special_mask), board_mask.shape)),
                    torch.tensor(-1),
                )
        else:
            # action mask is a tensor
            assert action_mask.dtype == torch.bool
            action = torch.multinomial(action_mask.flatten().to(torch.float), 1, True)
            return torch.cat(torch.unravel_index(action, action_mask.shape))

    def agent_observe(self, state):
        """
        agent observations
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
        raise NotImplementedError

    def board_action_dim(self, state):
        mask = self.action_mask(state=state)
        if self.has_special_actions():
            mask, _ = mask
        return len(mask.shape)

    def example_agent_obs(self):
        # This assumes the environment provides same observation shape to both agents
        state = self.init_state()
        return self.agent_observe(state=state)

    def example_action_mask(self):
        state = self.init_state()
        return self.action_mask(state=state)

    def example_critic_obs(self):
        state = self.init_state()
        return self.critic_observe(state=state)

    def is_valid(self, state, action):
        if self.has_special_actions():
            board_action, special_action = action
            board_mask, special_mask = self.action_mask(state)
            if special_action >= 0:
                return special_mask[special_action]
            else:
                return board_mask[*board_action]

        else:
            return self.action_mask(state)[*action]

    def get_canvas(self):
        """
        :return: canvas object to render game on (if relevant)
        """
        pass

    def render(self, canvas, state):
        """
        render a state
        Args:
            state: The state of the environment.
        """
        raise NotImplementedError

    def close_canvas(self, canvas):
        pass
