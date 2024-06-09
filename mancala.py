import strategy

BANK = 5

class Game:
    board = {}
    logic = {}
    
    def __init__(self, logic):
        self.board = {1: [5,5,5,5,5,0], 2: [5,5,5,5,5,0]}
        self.logic = logic

    def is_side_empty(self):
        return any(sum(self.board[side][:BANK]) == 0 for side in self.board)
    
    def is_slot_empty(self, side, idx):
        return self.board[side][idx] == 0
    
    def switch_side(self, side):
        return 2 if side == 1 else 1
    
    def player_choice(self, player):
        if self.logic[player][0] == 'random':
            return strategy.random_player(self, player)
        elif self.logic[player][0] == 'human':
            return strategy.human_player(self, player)
        elif self.logic[player][0] == 'greedy':
            return strategy.greedy_player(self, player)
        else:
            return strategy.ai_player(self, player)
        
    def calculate_landing(self, side, start_idx):
        player = side
        pebbles = self.board[side][start_idx]
        idx = start_idx
        while pebbles > 0:
            idx = (idx + 1) % (BANK + 1)
            if idx == BANK:
                if player == side:
                    pebbles -= 1
                if pebbles > 0:
                    idx = -1
                    side = self.switch_side(side)
            else:
                pebbles -= 1
        return {'side':side, 'idx':idx}

    def move(self, side, start_idx):
        player = side
        pebbles = self.board[side][start_idx]
        self.board[side][start_idx] = 0
        idx = start_idx
        while pebbles > 0:
            idx = (idx + 1) % (BANK + 1)
            if idx == BANK:
                if player == side:
                    self.board[side][BANK] += 1
                    pebbles -= 1
                if pebbles > 0:
                    idx = -1
                    side = self.switch_side(side)
            else:
                self.board[side][idx] += 1
                pebbles -= 1
        return {'side':side, 'idx':idx}
            
    def capture(self, player, landing):
        if player != landing['side']:
            return False
        if self.board[player][landing['idx']] > 1:
            return False
        opponent = self.switch_side(player)
        pebbles = self.board[opponent][landing['idx']]
        if pebbles > 0:
            self.board[player][BANK] += pebbles + 1
            self.board[opponent][landing['idx']] = 0
            self.board[player][landing['idx']] = 0
            return True
        return False

    def check_bonus_round(self, landing):
        return landing['idx'] == BANK
    
    def get_winner(self):
        p1_pebbles = sum(self.board[1])
        p2_pebbles = sum(self.board[2])
        return 1 if p1_pebbles > p2_pebbles else 2 if p1_pebbles < p2_pebbles else 0
                
    def print_board(self):
        print(self.board[1])
        print(self.board[2])
        print('---------')

    def game_loop(self, verbose=False):
        while not self.is_side_empty():
            player = 1
            while player < 3:
                if verbose: self.print_board()
                slot_picked = self.player_choice(player)
                if verbose: print('Player', player, 'moves', slot_picked)
                landing = self.move(player, slot_picked)
                if self.capture(player, landing):
                    if verbose: print('Capture!')
                if self.check_bonus_round(landing):
                    if verbose: print('Bonus round!')
                else:
                    player += 1
                if self.is_side_empty():
                    break
        if verbose: self.print_board()
        if verbose: print(self.get_winner())
        return(self.get_winner())