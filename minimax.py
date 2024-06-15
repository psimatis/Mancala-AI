import random
import mancala

class MinimaxAgent:
    def __init__(self, name, depth=6):
        self.name = name
        self.depth = depth

    def get_valid_moves(self, game, side):
        return [i for i in range(6) if not game.is_pit_empty(side, i)]

    def simulate_move(self, game, side, pit):
        temp_game = game.copy()
        temp_game.move(side, pit)
        return temp_game

    def evaluate(self, game, side):
        return game.board[side][mancala.STORE] - game.board[game.switch_side(side)][mancala.STORE]

    def minimax(self, game, depth, maximizing_player, side):
        if depth == 0 or game.is_game_over():
            return self.evaluate(game, side)

        valid_moves = self.get_valid_moves(game, side)
        if maximizing_player:
            max_eval = -float('inf')
            for move in valid_moves:
                temp_game = self.simulate_move(game, side, move)
                eval = self.minimax(temp_game, depth - 1, False, game.switch_side(side))
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                temp_game = self.simulate_move(game, side, move)
                eval = self.minimax(temp_game, depth - 1, True, game.switch_side(side))
                min_eval = min(min_eval, eval)
            return min_eval

    def act(self, game, side):
        valid_moves = self.get_valid_moves(game, side)
        best_score = -float('inf')
        best_move = random.choice(valid_moves)

        for move in valid_moves:
            temp_game = self.simulate_move(game, side, move)
            score = self.minimax(temp_game, self.depth - 1, False, game.switch_side(side))
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

# Usage example
if __name__ == "__main__":
    game = mancala.Game({1: MinimaxAgent(), 2: player.Random('random')})
    game.game_loop(verbose=True)