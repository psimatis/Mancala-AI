BANK = 6

class Game:   
    def __init__(self, players):
        self.board = {1: [6,6,6,6,6,6,0], 2: [6,6,6,6,6,6,0]}
        self.players = players

    def reset(self):
        self.board = {1: [6,6,6,6,6,6,0], 2: [6,6,6,6,6,6,0]}

    def is_side_empty(self):
        return any(sum(self.board[side][:BANK]) == 0 for side in self.board)
    
    def is_slot_empty(self, side, idx):
        return self.board[side][idx] == 0
    
    def switch_side(self, side):
        return 2 if side == 1 else 1
    
    def player_choice(self, side):
        player = self.players[side]
        return player.act(self, side)
        
    def move(self, side, start_idx, calculate_landing=False):
        player_side = side
        pebbles = self.board[side][start_idx]
        if not calculate_landing: 
            self.board[side][start_idx] = 0
        idx = start_idx
        while pebbles > 0:
            idx = (idx + 1) % (BANK + 1)
            if idx == BANK:
                if player_side == side:
                    if not calculate_landing: 
                        self.board[side][BANK] += 1
                    pebbles -= 1
                if pebbles > 0:
                    idx = -1
                    side = self.switch_side(side)
            else:
                if not calculate_landing: 
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
        if p1_pebbles > p2_pebbles:
            return 1
        elif p1_pebbles < p2_pebbles:
            return 2
        else:
            return 0
                
    def print_board(self):
        print(self.board[1])
        print(self.board[2])
        print('---------')

    def game_step(self, player_side, idx, verbose=True):
        info = {'landing': None, 'capture': False, 'bonus_round': False, 'game_over': False}
        if verbose: 
            print('Player', player_side, 'moves', idx)
        landing = self.move(player_side, idx)
        info['landing'] = landing
        if self.capture(player_side, landing):
            info['capture'] = True
            if verbose: 
                print('Capture!')
        if self.check_bonus_round(landing):
            info['bonus_round'] = True
            if verbose: 
                print('Bonus round!')
        if self.is_side_empty():
            info['game_over'] = True
        return info

    def game_loop(self, verbose=True):
        while not self.is_side_empty():
            player_side = 1
            while player_side < 3:
                if verbose: 
                    self.print_board()
                idx = self.player_choice(player_side)
                info = self.game_step(player_side, idx, verbose)
                if info['game_over']:
                    break
                if not info['bonus_round']:
                    player_side += 1
        if verbose: 
            self.print_board()
            print('Winner:', self.get_winner())
        return(self.get_winner())