from mancala import Game
from train_genetic import genetic_algorithm

genetic_player = genetic_algorithm()

wins = {0:0, 1:0, 2:0}
for _ in range(1000):
    game = Game({1: ['AI', genetic_player], 2: ['random', None]})
    wins[game.game_loop()] += 1

print(wins)