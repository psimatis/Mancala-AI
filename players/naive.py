import random

random.seed(0)

class Naive:
    def __init__(self, name='naive'):
        self.name = name

    def act(self, game):
        return random.choice(game.get_valid_moves())