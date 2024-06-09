import random

def get_valid_moves(game, player):
    return [i for i in range(6) if not game.is_slot_empty(player, i)]

def random_player(game, player):
    return random.choice(get_valid_moves(game, player))

def human_player(game, player):
    idx = int(input("Enter slot number (0-5): "))
    while idx not in range(6) or game.is_slot_empty(player, idx):
        idx = int(input("Invalid choice. Enter a slot number (0-5) that has pebbles: "))
    return idx

def greedy_player(game, player):
    valid_moves = get_valid_moves(game, player)
    #check bonus round
    for idx in valid_moves:
        if game.board[player][idx] == 6 - idx:
            return idx
    #check capture
    for idx in valid_moves:
        landing = game.move(player, idx, calculate_landing=True)
        if landing['side'] == player and game.is_slot_empty(player, landing['idx']):
            return idx
    # add to bank
    for idx in valid_moves:
        if game.board[player][idx] > 6 - idx:
            return idx
    #random
    return random_player(game, player)

def genetic_ai_player(game, player):
    strategy = game.strategy[player][1]
    scores = [(idx, strategy[idx] * game.board[player][idx]) for idx in range(6)]
    scores.sort(key=lambda x: x[1], reverse=True)
    for idx, _ in scores:
        if not game.is_slot_empty(player, idx):
            return idx
    return None