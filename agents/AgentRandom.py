import random


class AgentRandom:
    def __init__(self):
        pass

    def get_action(self, obs, moves, epsilon=0):
        return random.choice(moves)
