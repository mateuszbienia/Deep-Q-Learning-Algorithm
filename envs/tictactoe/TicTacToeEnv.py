import gym
import numpy as np

from .TicTacToe import TicTacToe
from typing import Tuple, Dict, List

INVALID_ACTION_REWARD = -2


class TicTacToeEnv(gym.Env):
    def __init__(self, config=None) -> None:
        super().__init__()
        self.game = TicTacToe(config)

        self.action_space = gym.spaces.Discrete(
            self.game.board_width * self.game.board_height)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(
            self.game.board_height * self.game.board_width,), dtype=np.uint8)

        self.reward_range = (INVALID_ACTION_REWARD, self.game.reward_win)
        self.board: np.ndarray = np.array([], dtype=np.uint8)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        info = {}
        self.game.reset()
        self.board = np.zeros(self.game.board_height *
                              self.game.board_width, dtype=np.uint8)
        obs = self._get_state()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        info = {}
        terminated = False
        truncated = False
        if self.game.is_valid_move(action):
            self.board[action] = self.game.get_player() + 1
            self.game.move(action)
            terminated = self.game.is_game_over()
            reward = self.game.get_reward()
        else:
            truncated = True
            reward = INVALID_ACTION_REWARD
        obs = self._get_state()
        return obs, reward, terminated, truncated, info

    def render(self, mode="human") -> None:
        if mode == "human":
            for idx, field in enumerate(self._get_state()):
                if (idx+1) % 3 != 0:
                    print("{}|".format(" " if field ==
                          0 else "X" if field == 1 else "O"), end="")
                else:
                    print(" " if field == 0 else "X" if field == 1 else "O")
                    if idx < 7:
                        print("-+-+-")
            print("")
            if self.game.is_game_over():
                print(self.get_winner_info())

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

    def get_moves(self) -> List[int]:
        return self.game.get_moves()

    def _get_state(self) -> np.ndarray:
        return self.board
