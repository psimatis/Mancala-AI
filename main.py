import matplotlib.pyplot as plt
from collections import defaultdict 
from mancala import Game
from train_genetic import genetic_algorithm

genetic_player_competition, history = genetic_algorithm(generations=30, population_size=20, simulations=100, elitisism=2)

plt.figure(figsize=(10, 5))
plt.plot([history[g]['Best Fitness'] for g in history], marker='o', linestyle='-', color='b')
plt.title('Best Fitness Per Generation')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.show()

genetic_player_random, history = genetic_algorithm(generations=30, population_size=20, simulations=100, elitisism=2, competition=False)

plt.figure(figsize=(10, 5))
plt.plot([history[g]['Best Fitness'] for g in history], marker='o', linestyle='-', color='b')
plt.title('Best Fitness Per Generation')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.show()

player_profiles = {
    'random': ('random', None),
    'gen_comp': ('AI', genetic_player_competition),
    'gen_rand': ('AI', genetic_player_random)
}

matches = defaultdict(lambda: defaultdict(int))

for p1 in player_profiles:
    for p2 in player_profiles:
        if p1 > p2:
         continue
        for _ in range(1000):
            game = Game({1: player_profiles[p1], 2: player_profiles[p2]})
            winner = game.game_loop()
            match_name = p1 + ' vs ' + p2
            matches[match_name]['total'] += 1
            if winner == 0:
                matches[match_name]['draw'] += 1
            elif winner == 1:
                matches[match_name][p1] += 1
            else:
                matches[match_name][p2] += 1

def print_results(matches):
    print("Match Results:\n")
    for match_name, results in matches.items():
        total_matches = results['total']
        draws = results['draw']
        p1, p2 = match_name.split(' vs ')
        p1_wins = results[p1]
        p2_wins = results[p2]
        p1_win_rate = (p1_wins / total_matches) * 100 if total_matches > 0 else 0
        p2_win_rate = (p2_wins / total_matches) * 100 if total_matches > 0 else 0
        draw_rate = (draws / total_matches) * 100 if total_matches > 0 else 0
        print(f"{match_name}:")
        print(f"  Total Matches: {total_matches}")
        print(f"  {p1} Wins: {p1_wins} ({p1_win_rate:.2f}%)")
        print(f"  {p2} Wins: {p2_wins} ({p2_win_rate:.2f}%)")
        print(f"  Draws: {draws} ({draw_rate:.2f}%)")
        print()


print_results(matches)