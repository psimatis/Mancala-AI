import matplotlib.pyplot as plt
from mancala import Game
from train_genetic import genetic_algorithm

genetic_player, history = genetic_algorithm()

plt.figure(figsize=(10, 5))
plt.plot([history[g]['Best Fitness'] for g in history], marker='o', linestyle='-', color='b')
plt.title('Best Fitness Per Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.show()

wins = {0:0, 1:0, 2:0}
for _ in range(1000):
    game = Game({1: ['AI', genetic_player], 2: ['random', None]})
    wins[game.game_loop()] += 1
print(wins)