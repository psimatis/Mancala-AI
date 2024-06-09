import random

def random_player(game, player):
    slot_picked = random.randint(0, 4)
    while game.is_slot_empty(player, slot_picked):
        slot_picked = random.randint(0, 4)
    return slot_picked

def human_player(game, player):
    slot_picked = int(input("Enter slot number (0-4): "))
    while slot_picked not in range(5) or game.is_slot_empty(player, slot_picked):
        slot_picked = int(input("Invalid choice. Enter a slot number (0-4) that has pebbles: "))
    return slot_picked

def greedy_player(game, player):
    valid_moves = [i for i in range(5) if not game.is_slot_empty(player, i)]
    #check bonus round
    for idx in valid_moves:
        if game.board[player][idx] == 5 - idx:
            return idx
    #check capture
    for idx in valid_moves:
        landing = game.calculate_landing(player, idx)
        if landing['side'] == player and game.is_slot_empty(player, landing['idx']):
            return idx
    # add to bank
    for idx in valid_moves:
        if game.board[player][idx] > 5 - idx:
            return idx
    #random
    return random_player(game, player)

def ai_player(game, player):
    strategy = game.logic[player][1]
    scores = [(idx, strategy[idx] * game.board[player][idx]) for idx in range(5)]
    scores.sort(key=lambda x: x[1], reverse=True)
    for idx, _ in scores:
        if not game.is_slot_empty(player, idx):
            return idx
    return None