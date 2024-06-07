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

def ai_player(game, player):
    strategy = game.logic[player][1]
    scores = [(idx, strategy[idx] * (game.board[player][idx] if not game.is_slot_empty(player, idx) else 0)) for idx in range(5)]
    scores.sort(key=lambda x: x[1], reverse=True)
    for idx, _ in scores:
        if not game.is_slot_empty(player, idx):
            return idx
    return 0
