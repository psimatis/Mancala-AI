import random

class Player:
    def __init__(self, name, strategy=None):
        self.name = name
        self.strategy = strategy

    def act(self, game, side):
        raise NotImplementedError("This method should be overridden by subclasses")

class Random(Player):
    def act(self, game, side):
        valid_moves = [i for i in range(6) if not game.is_slot_empty(side, i)]
        return random.choice(valid_moves)

class Human(Player):
    def act(self, game, side):
        idx = int(input("Enter slot number (0-5): "))
        while idx not in range(6) or game.is_slot_empty(side, idx):
            idx = int(input("Invalid choice. Enter a slot number (0-5) that has pebbles: "))
        return idx

class Greedy(Player):
    def act(self, game, side):
        valid_moves = [i for i in range(6) if not game.is_slot_empty(side, i)]
        #check bonus round
        for idx in valid_moves:
            if game.board[side][idx] == 6 - idx:
                return idx
        #check capture
        for idx in valid_moves:
            landing = game.move(side, idx, calculate_landing=True)
            if landing['side'] == side and game.is_slot_empty(side, landing['idx']):
                return idx
        # add to bank
        for idx in valid_moves:
            if game.board[side][idx] > 6 - idx:
                return idx
        #random
        return random.choice(valid_moves)
    
class Genetic(Player):
    def act(self, game, side):
        scores = [(idx, self.strategy[idx] * game.board[side][idx]) for idx in range(6)]
        scores.sort(key=lambda x: x[1], reverse=True)
        for idx, _ in scores:
            if not game.is_slot_empty(side, idx):
                return idx
        return random.choice([i for i in range(6) if not game.is_slot_empty(side, i)])
    
class DQN(Player):
    def act(self, game, side):
        return self.strategy.act(game.board[1] + game.board[2])