import sys

STORE = 6

class Game:
    def __init__(self, players):
        self.board = {1: [4,4,4,4,4,4,0], 2: [4,4,4,4,4,4,0]}
        self.players = players

    def health_check(self):
        total_stones = sum([sum(self.board[side]) for side in self.board])
        if total_stones != 48:
            print('HEALTH CHECK FAILED. TOTAL STONES:', total_stones)
            self.print_board()
            sys.exit()


    def is_side_empty(self):
        return any(sum(self.board[side][:STORE]) == 0 for side in self.board)

    def is_pit_empty(self, side, pit):
        return self.board[side][pit] == 0

    def switch_side(self, side):
        return 2 if side == 1 else 1

    def player_choice(self, side):
        return self.players[side].act(self, side)

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

    def capture(self, player_side, landing):
        if player_side != landing['side'] or self.board[player_side][landing['pit']] > 1:
            return False
        if landing['pit'] == STORE:
            return False
        opponent = self.switch_side(player_side)
        stones = self.board[opponent][landing['pit']]
        if stones > 0:
            self.board[player_side][STORE] += stones + 1
            self.board[opponent][landing['pit']] = 0
            self.board[player_side][landing['pit']] = 0
            return True
        return False

    def check_bonus_round(self, landing):
        return landing['pit'] == STORE

    def get_winner(self):
        p1_stones = sum(self.board[1])
        p2_stones = sum(self.board[2])
        if p1_stones > p2_stones:
            return 1
        elif p1_stones < p2_stones:
            return 2
        return 0

    def game_step(self, player_side, pit, verbose=False):
        info = {}
        info['player'] = player_side
        info['pit'] = pit
        info['landing'] = self.move(player_side, pit)
        info['capture'] = self.capture(player_side, info['landing'])
        info['bonus_round'] = self.check_bonus_round(info['landing'])
        info['game_over'] = self.is_side_empty()
        if verbose:
            self.print_board()
            print(info)
        self.health_check()
        return info

    def game_loop(self, verbose=False):
        current_player = 1
        while not self.is_side_empty():
            if verbose:
                self.print_board()
            pit = self.player_choice(current_player)
            info = self.game_step(current_player, pit, verbose)
            if info['game_over']:
                break
            if not info['bonus_round']:
                current_player = 2 if current_player == 1 else 1
                self.health_check()
        if verbose:
            print(info)
            self.print_board()
            print('Winner:', self.get_winner())
        return self.get_winner()

    def print_board(self):
        print(self.board[1])
        print(self.board[2])
        print('---------')
