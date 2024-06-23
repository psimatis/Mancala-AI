import sys

STORE = 6

class Game:
    def __init__(self, players):
        self.board = {1: [4,4,4,4,4,4,0], 2: [4,4,4,4,4,4,0]}
        self.players = players
        self.current_player = 1

    def copy(self):
        g = Game(self.players)
        g.board = {1: self.board[1][:], 2: self.board[2][:]}
        g.current_player = self.current_player
        return g

    def health_check(self):
        total_stones = sum([sum(self.board[side]) for side in self.board])
        if total_stones != 48:
            print('HEALTH CHECK FAILED. TOTAL STONES:', total_stones)
            self.print_board()
            sys.exit()
    
    def print_board(self, info=None):
        if info: 
            print(info)
        print(self.board[1])
        print(self.board[2])
        print('---------')

    def get_state(self):
        return self.board[1] + self.board[2]
    
    def get_valid_moves(self):
        return [i for i in range(STORE) if not self.is_pit_empty(self.current_player, i)]
    
    def check_bonus_round(self, landing):
        return landing['pit'] == STORE

    def is_game_over(self):
        return any(sum(self.board[side][:STORE]) == 0 for side in self.board)

    def is_pit_empty(self, side, pit):
        return self.board[side][pit] == 0

    def switch_side(self, side):
        return 2 if side == 1 else 1

    def player_choice(self):
        return self.players[self.current_player].act(self)

    def move(self, side, pit, simulate=False):
        stones = self.board[side][pit]
        if not simulate:
            self.board[side][pit] = 0
        start_side = side
        while stones > 0:
            pit = (pit + 1) % (STORE + 1)
            if pit == STORE:
                if start_side == side:
                    if not simulate: 
                        self.board[side][STORE] += 1
                    stones -= 1
                if stones > 0:
                    pit = -1
                    side = self.switch_side(side)
            else:
                if not simulate:
                    self.board[side][pit] += 1
                stones -= 1
        return {'side':side, 'pit':pit}

    def capture(self, side, landing):
        if side != landing['side'] or self.board[side][landing['pit']] > 1 or landing['pit'] == STORE:
            return False
        opponent = self.switch_side(side)
        stones = self.board[opponent][landing['pit']]
        if stones == 0:
            return False
        self.board[side][STORE] += stones + 1
        self.board[opponent][landing['pit']] = 0
        self.board[side][landing['pit']] = 0
        return True
    
    def capture_exposure(self):
        exposures = set()
        opponent = self.switch_side(self.current_player)
        possible_moves = [pit for pit in range(STORE) if self.board[opponent][pit] > 0] 
        for p in possible_moves:
            simulation = self.copy()
            landing = simulation.move(opponent, p)
            if simulation.capture(opponent, landing):
                exposures.add((landing['pit'], self.board[self.current_player][landing['pit']]))
        return sum([s[1] for s in exposures])

    def get_winner(self):
        p1_stones = sum(self.board[1])
        p2_stones = sum(self.board[2])
        if p1_stones > p2_stones:
            return 1
        if p1_stones < p2_stones:
            return 2
        return 0

    def step(self, pit, verbose=False):
        info = {}
        info['player'] = self.current_player
        info['pit'] = pit
        info['landing'] = self.move(self.current_player, pit)
        info['capture'] = self.capture(self.current_player, info['landing'])
        info['bonus_round'] = self.check_bonus_round(info['landing'])
        info['game_over'] = self.is_game_over()
        info['capture_exposure'] = self.capture_exposure()
        if verbose:
            self.print_board(info)
        self.health_check()
        if not info['bonus_round']:
            self.current_player = 2 if self.current_player == 1 else 1
        return info

    def game_loop(self, verbose=False):
        if verbose:
            self.print_board()
        while not self.is_game_over():
            info = self.step(self.player_choice(), verbose)
            if info['game_over']:
                break
        if verbose:
            print('Winner:', self.get_winner())
        return self.get_winner()
