import random

class Game:
    board = {}
    logic = {}
    
    def __init__(this, logic):
        this.board = {'one': [5,5,5,5,5,0], 'two': [5,5,5,5,5,0]}
        this.logic = logic

    def side_empty(this):
        for p in this.board:
            if sum(this.board[p][:-1]) == 0:
                return True
        return False
    
    def slot_empty(this, p, s):
        return True if this.board[p][s] == 0 else False
    
    def player_switch(this, p):
        return 'two' if p == 'one' else 'one'
    
    def player_choice(this, p):
        if this.logic[p] == 'random':
            return random.randint(0, 4)
        elif this.logic[p] == 'human':
            return int(input("Enter slot number: "))
        else:
            # TODO AI
            return 'No AI yet'

    def move(this, p, idx):
        steps = this.board[p][idx]
        this.board[p][idx] = 0
        while steps > 0:
            idx += 1
            this.board[p][idx] += 1
            if idx == 5:
                idx = -1
                p = this.player_switch(p)
            steps -= 1
        return {'side':p, 'idx':idx}
            
    def capture(this, player, landing):
        # Must land on current player's side
        if player != landing['side']:
            return
        
        # Must land on an empty slot
        if this.board[player][landing['idx']] > 1:
            return
        
        # Slot on the other side must have pebbles
        opponent = this.player_switch(player)
        pebbles = this.board[opponent][landing['idx']]
        if pebbles > 0:
            this.board[player][5] += pebbles + 1
            this.board[opponent][landing['idx']] = 0
            this.board[player][landing['idx']] = 0
            print('Capture!')
    
    def get_winner(this):
        if this.board['one'][-1] > this.board['two'][-1]:
            return 'Player one won!'
        elif this.board['one'][-1] == this.board['two'][-1]:
            return 'Draw'
        else:
            return 'Player two won!'
                
    def print_board(this):
        print(this.board['one'])
        print(this.board['two'])

def game_loop(players):
    board = Game(players)
    board.print_board()
    while not board.side_empty():
        for p in ('one', 'two'):
            slot_picked = board.player_choice(p)
            while board.slot_empty(p, slot_picked):
                slot_picked = random.randint(0, 4)
            print('Player', p, 'moves', slot_picked)
            landing = board.move(p, slot_picked)
            board.capture(p, landing)
            board.print_board()


    print(board.get_winner())