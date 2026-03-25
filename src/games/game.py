class Game:
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

    def agent_observe(self, state):
        """
        agent observations
        Args:
            state: The state of the environment.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
