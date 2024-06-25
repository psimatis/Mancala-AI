import mancala

class Minimax:
    def __init__(self, name='minimax', depth=2):
        self.name = name
        self.depth = depth
        self.player = None

    def simulate_move(self, game, pit):
        temp_game = game.copy()
        temp_game.step(pit)
        return temp_game

    def evaluate(self, game):
        opponent = game.switch_side(self.player)
        return game.board[self.player][mancala.STORE] - game.board[opponent][mancala.STORE]

    def minimax(self, game, depth, alpha, beta):
        if depth == 0 or game.is_game_over():
            return self.evaluate(game)
        player = game.current_player
        valid_moves = game.get_valid_moves()
        valid_moves.reverse()
        if player == 1:
            max_eval = -float('inf')
            for move in valid_moves:
                temp_game = self.simulate_move(game, move)
                eval = self.minimax(temp_game, depth - 1, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                temp_game = self.simulate_move(game, move)
                eval = self.minimax(temp_game, depth - 1, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def act(self, game):
        self.player = game.current_player
        valid_moves = game.get_valid_moves()
        valid_moves.reverse() 
        best_move = valid_moves[0]
        best_score = -float('inf')
        for move in valid_moves:
            temp_game = self.simulate_move(game, move)
            score = self.minimax(temp_game, self.depth - 1, -float('inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
    