import random

class OpponentBlender:
    """can be used to register baseline agents and provide a convenient way to get a random enemy from the pool """
    def __init__(self):
        self.agents = []

    def register_agent(self, agent_fn):
        """
        Registers an agent function.
        """
        self.agents.append(agent_fn)

    def pick_random_agent(self):
        """
        Returns a randomly selected agent function from registered agents.
        """
        if not self.agents:
            raise ValueError("No agents registered in OpponentBlender.")
        return random.choice(self.agents)
