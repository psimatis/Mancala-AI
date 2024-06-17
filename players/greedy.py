import random
from mancala import STORE

class Greedy():
    def __init__(self, name='greedy'):
        self.name = name

    def act(self, game):
        player = game.current_player
        valid_moves = game.get_valid_moves(player)
        # check bonus round
        for pit in valid_moves:
            if game.board[player][pit] == STORE - pit:
                return pit
        # check capture
        for pit in valid_moves:
            landing = game.move(player, pit, simulate=True)
            if landing['side'] == player and game.is_pit_empty(player, landing['pit']):
                return pit
        # check add to bank
        for pit in valid_moves:
            if game.board[player][pit] > STORE - pit:
                return pit
        return random.choice(valid_moves)