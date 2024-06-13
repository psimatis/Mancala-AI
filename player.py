import random

class Player:
    def __init__(self, name, agent=None):
        self.name = name
        self.agent = agent

    def get_valid_moves(self, game, side):
        return [i for i in range(6) if not game.is_pit_empty(side, i)]

    def act(self, game, side):
        raise NotImplementedError("This method should be overridden by subclasses")

class Random(Player):
    def act(self, game, side):
        return random.choice(self.get_valid_moves(game,side))

class Human(Player):
    def act(self, game, side):
        pit = int(input("Enter pit number (0-5): "))
        while pit not in range(6) or game.is_pit_empty(side, pit):
            pit = int(input("Invalid choice. Enter pit number (0-5) that has stones: "))
        return pit

class Greedy(Player):
    def act(self, game, side):
        valid_moves = self.get_valid_moves(game,side)
        # check bonus round
        for pit in valid_moves:
            if game.board[side][pit] == 6 - pit:
                return pit
        # check capture
        for pit in valid_moves:
            landing = game.move(side, pit, simulate=True)
            if landing['side'] == side and game.is_pit_empty(side, landing['pit']):
                return pit
        # check add to bank
        for pit in valid_moves:
            if game.board[side][pit] > 6 - pit:
                return pit
        return random.choice(valid_moves)

class Genetic(Player):
    def act(self, game, side):
        scores = [(pit, self.agent[pit] * game.board[side][pit]) for pit in range(6)]
        scores.sort(key=lambda x: x[1], reverse=True)
        for pit, _ in scores:
            if not game.is_pit_empty(side, pit):
                return pit
        return random.choice(self.get_valid_moves(game,side))

class DQN(Player):
    def act(self, game, side):
        return self.agent.act(game.board[1] + game.board[2], side)
