import random

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
        if self.logic[player] == 'random':
            slot_picked = random.randint(0, 4)
            while self.is_slot_empty(player, slot_picked):
                slot_picked = random.randint(0, 4)
        elif self.logic[player] == 'human':
            slot_picked = int(input("Enter slot number (0-4): "))
            while slot_picked not in range(5) or self.is_slot_empty(player, slot_picked):
                slot_picked = int(input("Invalid choice. Enter a slot number (0-4) that has pebbles: "))
        else:
            # TODO: Implement AI logic
            return 0
        return slot_picked

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
        if self.board[1][BANK] > self.board[2][BANK]:
            return 1
        elif self.board[1][BANK] == self.board[2][BANK]:
            return 0
        else:
            return 2
                
    def print_board(self):
        print(self.board[1])
        print(self.board[2])
        print('---------')

def game_loop(players):
    game = Game(players)
    game.print_board()
    while not game.is_side_empty():
        player = 1
        while player < 3:
            slot_picked = game.player_choice(player)
            print('Player', player, 'moves', slot_picked)
            landing = game.move(player, slot_picked)
            game.capture(player, landing)
            if game.check_bonus_round(landing):
                print('Bonus round!')
            else:
                player += 1
            game.print_board()
    print(game.get_winner())