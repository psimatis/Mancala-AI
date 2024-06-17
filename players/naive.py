import random

random.seed(0)

class Naive:
    def __init__(self, name='random'):
        self.name = name

    def act(self, game):
        return random.choice(game.get_valid_moves(game.current_player))