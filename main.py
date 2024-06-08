import matplotlib.pyplot as plt
from collections import defaultdict 
from mancala import Game
from train_genetic import genetic_algorithm

def plot_training(history):
    plt.figure(figsize=(10, 5))
    plt.plot([history[g]['Best Fitness'] for g in history], marker='o', linestyle='-', color='b')
    plt.title('Best Fitness Per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.show()

print('Training tournament-based genetic algorithm')
genetic_player_tournament, history = genetic_algorithm(generations=10, population_size=100, simulations=1, elitisism=2, tournament=100)
print(f'Best strategy: ', genetic_player_tournament)
plot_training(history)

print('Training random-based genetic algorithm')
genetic_player_random, history = genetic_algorithm(generations=10, population_size=100, simulations=100, elitisism=2, tournament=0)
print(f'Best strategy: ', genetic_player_random)
plot_training(history)

player_profiles = {
    'random': ('random', None),
    'gen_tour': ('AI', genetic_player_tournament),
    'gen_rand': ('AI', genetic_player_random)
}

matches = defaultdict(lambda: defaultdict(int))
matches_number = 1000
for p1 in player_profiles:
    for p2 in player_profiles:
        if p1 > p2:
            continue
        for _ in range(matches_number):
            game = Game({1: player_profiles[p1], 2: player_profiles[p2]})
            winner = game.game_loop()
            match_name = p1 + ' vs ' + p2
            if winner == 0:
                matches[match_name]['draw'] += 1
            elif winner == 1:
                matches[match_name][p1] += 1
            else:
                matches[match_name][p2] += 1

def print_results(matches):
    print(f'Match Results for {matches_number}:\n')
    for match_name, results in matches.items():
        p1, p2 = match_name.split(' vs ')
        p1_win_rate = (results[p1] / matches_number) * 100
        p2_win_rate = (results[p2] / matches_number) * 100
        draw_rate = (results['draw'] / matches_number) * 100
        print(f"{match_name}:")
        print(f"  {p1} Wins: {results[p1]} ({p1_win_rate:.2f}%)")
        print(f"  {p2} Wins: {results[p2]} ({p2_win_rate:.2f}%)")
        print(f"  Draws: {results['draw']} ({draw_rate:.2f}%)")
        print()

print_results(matches)

for p in player_profiles:
    game = Game({1: ('human', None), 2: player_profiles[p]})
    game.game_loop(verbose=True)
    