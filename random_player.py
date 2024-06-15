import random
import mancala

random.seed(0)

class Random:
    def __init__(self, name='random'):
        self.name = name

    def get_valid_moves(self, game, side):
        return [i for i in range(mancala.STORE) if not game.is_pit_empty(side, i)]

    def act(self, game, side):
        return random.choice(self.get_valid_moves(game,side))