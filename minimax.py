import random
import mancala
from random_player import Random
from greedy_player import Greedy
from human_player import Human

random.seed(0)

class MinimaxAgent:
    def __init__(self, name='minimax', depth=5, evaluation='store'):
        self.name = name
        self.depth = depth
        self.evaluation = evaluation

    def get_valid_moves(self, game, side):
        return [i for i in reversed(range(mancala.STORE)) if not game.is_pit_empty(side, i)]

    def simulate_move(self, game, side, pit):
        temp_game = game.copy()
        temp_game.move(side, pit)
        return temp_game

    def evaluate(self, game, side):
        opponent_side = game.switch_side(side)
        if self.evaluation == 'store':
                return game.board[side][mancala.STORE] - game.board[opponent_side][mancala.STORE]
        return game.get_difference(side)

    def minimax(self, game, depth,alpha, beta, maximizing_player, side):
        if depth == 0 or game.is_game_over():
            return self.evaluate(game, side)

        valid_moves = self.get_valid_moves(game, side)
        if maximizing_player:
            max_val = -float('inf')
            for move in valid_moves:
                temp_game = self.simulate_move(game, side, move)
                val = self.minimax(temp_game, depth - 1, alpha, beta, False, game.switch_side(side))
                max_val = max(max_val, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return max_val
        else:
            min_val = float('inf')
            for move in valid_moves:
                temp_game = self.simulate_move(game, side, move)
                val = self.minimax(temp_game, depth - 1, alpha, beta, True, game.switch_side(side))
                min_val = min(min_val, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_val

    def act(self, game, side):
        maximizing_player = True if side == 1 else False
        valid_moves = self.get_valid_moves(game, side)
        best_score = -float('inf')
        best_move = random.choice(valid_moves)

        for move in valid_moves:
            temp_game = self.simulate_move(game, side, move)
            score = self.minimax(temp_game, self.depth, -float('inf'), float('inf'), maximizing_player, game.switch_side(side))
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

if __name__ == "__main__":
    mancala.Game({1: MinimaxAgent(), 2: Human()}).game_loop(verbose=True)