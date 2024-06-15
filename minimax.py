import random
import mancala
from random_player import Random
from greedy_player import Greedy
import time

class MinimaxAgent:
    def __init__(self, name='minimax', depth=6):
        self.name = name
        self.depth = depth

    def get_valid_moves(self, game, side):
        return [i for i in range(mancala.STORE) if not game.is_pit_empty(side, i)]

    def simulate_move(self, game, side, pit):
        temp_game = game.copy()
        temp_game.move(side, pit)
        return temp_game

    def evaluate(self, game, side):
        return game.board[side][mancala.STORE] - game.board[game.switch_side(side)][mancala.STORE]

    def minimax(self, game, depth, alpha, beta, maximizing_player, side):
        if depth == 0 or game.is_game_over():
            return self.evaluate(game, side)

        valid_moves = self.get_valid_moves(game, side)
        if maximizing_player:
            max_eval = -float('inf')
            for move in valid_moves:
                temp_game = self.simulate_move(game, side, move)
                eval = self.minimax(temp_game, depth - 1, alpha, beta, False, game.switch_side(side))
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                temp_game = self.simulate_move(game, side, move)
                eval = self.minimax(temp_game, depth - 1, alpha, beta, True, game.switch_side(side))
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def act(self, game, side):
        valid_moves = self.get_valid_moves(game, side)
        best_score = -float('inf')
        best_move = random.choice(valid_moves)

        for move in valid_moves:
            temp_game = self.simulate_move(game, side, move)
            score = self.minimax(temp_game, self.depth - 1, -float('inf'), float('inf'), False, game.switch_side(side))
            if score > best_score:
                best_score = score
                best_move = move

        return best_move


# Usage example
if __name__ == "__main__":
    start_time = time.time()
    for _ in range(100):
        game = mancala.Game({1: MinimaxAgent(), 2: Greedy()})
        game.game_loop(verbose=True)
    print(time.time() - start_time)