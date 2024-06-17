import mancala

class Minimax:
    def __init__(self, name='minimax', depth=3):
        self.name = name
        self.depth = depth

    def simulate_move(self, game, side, pit):
        temp_game = game.copy()
        temp_game.move(side, pit)
        return temp_game

    def evaluate(self, game, side):
        opponent_side = game.switch_side(side)
        return game.board[side][mancala.STORE] - game.board[opponent_side][mancala.STORE]

    def minimax(self, game, depth,alpha, beta, maximizing_player, side):
        if depth == 0 or game.is_game_over():
            return self.evaluate(game, side)

        valid_moves = game.get_valid_moves(reverse=True)
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

    def act(self, game):
        maximizing_player = True if game.current_player == 1 else False
        valid_moves = game.get_valid_moves(reverse=True)
        best_score = -float('inf')
        best_move = valid_moves[-1]

        for move in valid_moves:
            temp_game = self.simulate_move(game, game.current_player, move)
            score = self.minimax(temp_game, self.depth, -float('inf'), float('inf'), maximizing_player, game.switch_side(game.current_player))
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
