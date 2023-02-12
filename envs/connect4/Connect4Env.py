from typing import Tuple, Dict, List

import gym
import numpy as np

from .Connect4 import Connect4

INVALID_ACTION_REWARD = -2


class Connect4Env(gym.Env):

    def __init__(self, config=None) -> None:
        super().__init__()
        self.game = Connect4(config)

        self.action_space = gym.spaces.Discrete(self.game.board_width + 1)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(
            self.game.board_height, self.game.board_width), dtype=np.uint8)
        self.boards: List[np.array] = []

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        info = {}
        self.game = Connect4()
        self.board = np.zeros(
            (self.game.board_height, self.game.board_width), dtype=np.uint8)
        obs = self._get_state(0)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        info = {}
        terminated = False
        truncated = False

        if self.game.is_valid_move(action):
            self.board[self.game.column_counts[action]
                       ][action] = self.game.player + 1
            self.game.move(action)
            terminated = self.game.is_game_over()
            reward = self.game.get_reward(0)
        else:
            truncated = True
            reward = INVALID_ACTION_REWARD

        obs = self._get_state()

        return obs, reward, terminated, truncated, info

    def get_moves(self) -> List[int]:
        return self.game.get_moves()

    def render(self, mode='human') -> None:
        if mode == "human":
            print('  1 2 3 4 5 6 7')
            print(' ---------------')
            print(self._get_state())
            print(' ---------------')
            print('  1 2 3 4 5 6 7')
            if self.game.is_game_over():
                print(self.get_winner_info())

    def _get_state(self, player=None) -> np.ndarray:
        return np.flip(self.board, axis=0).copy()

    def get_winner_info(self) -> str:
        if not self.game.is_game_over():
            return "Game has not ended!"
        winner = (self.game.is_winner(0), self.game.is_winner(1))
        if True in winner:
            return "Player {} won game!".format(1 if winner[0] else 2)
        else:
            return "Draw"

    def close(self) -> None:
        super.close()
