import copy
from typing import List

REWARD_LOSE = -1.0
REWARD_WIN = 1.0
REWARD_DRAW = 0.0
REWARD_NEUTRAL = 0.0
BOARD_HEIGHT = 6
BOARD_WIDTH = 7
WIN_LENGTH = 4


class Connect4:
    def __init__(self, config: dict = None):
        self.env_config = dict({
            'board_height': BOARD_HEIGHT,
            'board_width': BOARD_WIDTH,
            'win_length': WIN_LENGTH,
            'reward_win': REWARD_WIN,
            'reward_draw': REWARD_DRAW,
            'reward_lose': REWARD_LOSE,
            'reward_step': REWARD_NEUTRAL,
        }, **config or {})

        self.bitboard = [0, 0]  # bitboard for both players
        self.player = 0  # in {0, 1}

        self.win_conditions = [
            1, (self.board_height + 1), (self.board_height + 1) - 1, (self.board_height + 1) + 1]
        self.empty_indexes = [(self.board_height + 1) *
                              i for i in range(self.board_width)]
        self.column_counts = [0] * self.board_width
        self.top_row = [(x * (self.board_height + 1)) -
                        1 for x in range(1, self.board_width + 1)]

        self.is_winner_cache = [None, None]

    def clone(self):
        clone = Connect4()
        clone.bitboard = copy.deepcopy(self.bitboard)
        clone.empty_indexes = copy.deepcopy(self.empty_indexes)
        clone.column_counts = copy.deepcopy(self.column_counts)
        clone.top_row = copy.deepcopy(self.top_row)
        clone.player = self.player
        return clone

    def move(self, column) -> None:
        self.is_winner_cache = [None, None]
        m2 = 1 << self.empty_indexes[column]
        self.empty_indexes[column] += 1
        self.player ^= 1
        self.bitboard[self.player] ^= m2
        self.column_counts[column] += 1

    def get_reward(self, player=None) -> float:
        if player is None:
            player = self.player

        if self.is_winner(player):
            return self.reward_win
        elif self.is_winner(player ^ 1):
            return self.reward_lose
        elif self.is_draw():
            return self.reward_draw
        else:
            return self.reward_step

    def is_winner(self, player: int = None) -> bool:
        if player is None:
            player = self.player
        player = player ^ 1
        if self.is_winner_cache[player] is not None:
            return self.is_winner_cache[player]

        for d in self.win_conditions:
            bb = self.bitboard[player]
            for i in range(1, self.win_length):
                bb &= self.bitboard[player] >> (i * d)
            if bb != 0:
                self.is_winner_cache[player] = True
                return True
        self.is_winner_cache[player] = False
        return False

    def is_draw(self) -> bool:
        return not (
            self.get_moves()
            or self.is_winner(self.player)
            or self.is_winner(self.player ^ 1)
        )

    def is_game_over(self) -> bool:
        return self.is_winner(self.player) or self.is_winner(self.player ^ 1) or not self.get_moves()

    def get_moves(self) -> List[int]:
        if self.is_winner(self.player) or self.is_winner(self.player ^ 1):
            return []

        return [
            i
            for i in range(self.board_width)
            if self.column_counts[i] < self.board_height
        ]

    def is_valid_move(self, column: int) -> bool:
        return self.empty_indexes[column] != self.top_row[column]

    @ property
    def board_height(self) -> int:
        return self.env_config['board_height']

    @ property
    def board_width(self) -> int:
        return self.env_config['board_width']

    @ property
    def win_length(self) -> int:
        return self.env_config['win_length']

    @ property
    def reward_win(self) -> float:
        return self.env_config['reward_win']

    @ property
    def reward_draw(self) -> float:
        return self.env_config['reward_draw']

    @ property
    def reward_lose(self) -> float:
        return self.env_config['reward_lose']

    @ property
    def reward_step(self) -> float:
        return self.env_config['reward_step']
