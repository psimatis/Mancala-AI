import random
from mancala import STORE

class Greedy():
    def __init__(self, name='greedy'):
        self.name = name

    def get_valid_moves(self, game, side):
        return [i for i in range(STORE) if not game.is_pit_empty(side, i)]

    def act(self, game, side):
        valid_moves = self.get_valid_moves(game,side)
        # check bonus round
        for pit in valid_moves:
            if game.board[side][pit] == STORE - pit:
                return pit
        # check capture
        for pit in valid_moves:
            landing = game.move(side, pit, simulate=True)
            if landing['side'] == side and game.is_pit_empty(side, landing['pit']):
                return pit
        # check add to bank
        for pit in valid_moves:
            if game.board[side][pit] > STORE - pit:
                return pit
        return random.choice(valid_moves)