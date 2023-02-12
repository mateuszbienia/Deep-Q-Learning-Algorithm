from typing import List

REWARD_LOSE = -1.0
REWARD_WIN = 1.0
REWARD_DRAW = 0.0
REWARD_NEUTRAL = 0.0
BOARD_HEIGHT = 3
BOARD_WIDTH = 3
WIN_LENGTH = 3


class TicTacToe:
    ''' 
    Class that represents Tic Tac Toe game.
    Board is represented as bitboard that may be represented as:
    6|7|8
    -+-+-
    3|4|5
    -+-+-
    0|1|2
    on game board.
    '''

    def __init__(self, config: dict = None):

        self.config = dict({
            'board_height':   BOARD_HEIGHT,
            'board_width':    BOARD_WIDTH,
            'win_length':     WIN_LENGTH,
            'reward_win':     REWARD_WIN,
            'reward_draw':    REWARD_DRAW,
            'reward_lose':    REWARD_LOSE,
            'reward_neutral': REWARD_NEUTRAL,
        }, **config or {})
        self.board = [0, 0]  # bitboard for both players
        self.player = 0  # in {0, 1}

        self.winning_states = [
            # horizontal
            0b111000000,
            0b000111000,
            0b000000111,
            # vertical
            0b100100100,
            0b010010010,
            0b001001001,
            # diagonal
            0b100010001,
            0b001010100,
            # 876543210
        ]
        self.draw = 0b111111111

    def reset(self):
        ''' 
        Reset game to starting state.
        '''
        self.board = [0, 0]
        self.player = 0

    def move(self, pos: int) -> None:
        '''
        Make a move on board.

        :param pos: position on board
        '''
        self.board[self.player] |= 1 << pos
        self.player ^= 1

    def is_winner(self, player: int) -> bool:
        '''
        Check if player won the game.

        :param player: id of player in {0, 1}
        :return: True if player won, False if other
        '''
        for ws in self.winning_states:
            if (self.board[player] & ws) == ws:
                return True
        return False

    def is_draw(self) -> bool:
        '''
        Check if game ended in a draw.

        :return: True if draw, False if other
        '''
        return ((self.board[0] | self.board[1]) & self.draw) == self.draw

    def is_valid_move(self, pos: int) -> bool:
        '''
        Check if move is valid.

        :param pos: position on board
        :return: True if move is valid, False if is not valid
        '''
        shift = 1 << pos
        return not (((self.board[0] | self.board[1]) & (shift)) == (shift))

    def is_game_over(self) -> bool:
        '''
        Check if game is over.

        :return: True if the game ended either by one of the players winning of by a draw, False if the game not ended yet
        '''
        if self.is_winner(self.player):
            return True
        if self.is_winner(self.player ^ 1):
            return True
        if self.is_draw():
            return True
        return False

    def get_reward(self, player: int = None) -> float:
        '''
        Return a reward for a given player.

        :param player: id of player in {0, 1}
        :return: constant that corresponds to the reward based on the game outcome
        '''
        if player is None:
            player = 0
        if self.is_winner(player):
            return self.reward_win
        if self.is_winner(player ^ 1):
            return self.reward_lose
        if self.is_draw():
            return self.reward_draw
        return self.reward_neutral

    def get_moves(self) -> List[int]:
        '''
        Return a list of avalible moves.

        :return: list of avalible moves
        '''
        return [
            i for i in range(self.board_height * self.board_width)
            if self.is_valid_move(i)
        ]

    def get_player(self) -> int:
        '''
        Return the player whose turn  it is now.

        :return: id of player in {0, 1}
        '''
        return self.player

    @ property
    def board_height(self) -> int:
        return self.config['board_height']

    @ property
    def board_width(self) -> int:
        return self.config['board_width']

    @ property
    def win_length(self) -> int:
        return self.config['win_length']

    @ property
    def reward_win(self) -> float:
        return self.config['reward_win']

    @ property
    def reward_draw(self) -> float:
        return self.config['reward_draw']

    @ property
    def reward_lose(self) -> float:
        return self.config['reward_lose']

    @ property
    def reward_neutral(self) -> float:
        return self.config['reward_neutral']
