class AgentDQN:
    def __init__(self, agent):
        self.agent = agent

    def get_action(self, obs, moves) -> int:
        return self.agent.get_action(obs, moves)
