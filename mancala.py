import random

class Game:
    board = {}
    logic = {}
    
    def __init__(this, logic):
        this.board = {1: [5,5,5,5,5,0], 2: [5,5,5,5,5,0]}
        this.logic = logic

    def side_empty(this):
        for p in this.board:
            if sum(this.board[p][:-1]) == 0:
                return True
        return False
    
    def slot_empty(this, p, s):
        return True if this.board[p][s] == 0 else False
    
    def switch_side(this, p):
        return 2 if p == 1 else 1
    
    def player_choice(this, p):
        if this.logic[p] == 'random':
            slot_picked = random.randint(0, 4)
            while this.slot_empty(p, slot_picked):
                slot_picked = random.randint(0, 4)
        elif this.logic[p] == 'human':
            slot_picked = int(input("Enter slot number: "))
        else:
            # TODO AI
            return 0
        return slot_picked

    def move(this, s, idx):
        player = s
        steps = this.board[s][idx]
        this.board[s][idx] = 0
        while steps > 0:
            idx += 1
            if idx == 5:
                if player == s:
                    this.board[s][idx] += 1
                else:
                    steps += 1
                idx = -1
                s = this.switch_side(s)
            else:
                this.board[s][idx] += 1
            steps -= 1
        return {'side':s, 'idx':idx}
            
    def capture(this, player, landing):
        # Must land on current player's side
        if player != landing['side']:
            return
        
        # Must land on an empty slot
        if this.board[player][landing['idx']] > 1:
            return
        
        # Slot on the other side must have pebbles
        opponent = this.switch_side(player)
        pebbles = this.board[opponent][landing['idx']]
        if pebbles > 0:
            this.board[player][5] += pebbles + 1
            this.board[opponent][landing['idx']] = 0
            this.board[player][landing['idx']] = 0
            print('Capture!')

    def check_bonus_round(this, landing):
        return True if landing['idx'] == -1 else False
    
    def get_winner(this):
        if this.board[1][-1] > this.board[2][-1]:
            return 'Player one won!'
        elif this.board[1][-1] == this.board[2][-1]:
            return 'Draw'
        else:
            return 'Player two won!'
                
    def print_board(this):
        print(this.board[1])
        print(this.board[2])
        print('---------')

def game_loop(players):
    game = Game(players)
    game.print_board()
    while not game.side_empty():
        p = 1
        while p < 3:
            slot_picked = game.player_choice(p)
            print('Player', p, 'moves', slot_picked)
            landing = game.move(p, slot_picked)
            game.capture(p, landing)
            if game.check_bonus_round(landing):
                print('Bonus round!')
            else:
                p += 1
            game.print_board()
    print(game.get_winner())