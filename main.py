from collections import defaultdict 
from mancala import Game
from genetic_algorithm import run_genetic_algorithm

genetic_player_random, _ = run_genetic_algorithm(generations=10)
genetic_player_tournament, _ = run_genetic_algorithm(generations=10, simulations=1, tournament=100)

player_profiles = {
    'random': ('random', None),
    'greedy': ('greedy', None),
    'genetic_random': ('AI', genetic_player_random),
    'genetic_tournament': ('AI', genetic_player_tournament),
}

matches = defaultdict(lambda: defaultdict(int))
matches_number = 100
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
        if p1 == p2:
            continue
        p1_win_rate = (results[p1] / matches_number) * 100
        p2_win_rate = (results[p2] / matches_number) * 100
        draw_rate = (results['draw'] / matches_number) * 100
        print(f"{match_name}:")
        print(f"  {p1} wins {p1_win_rate:.2f}%")
        print(f"  {p2} wins {p2_win_rate:.2f}%")
        print(f"  Draws {draw_rate:.2f}%")

print_results(matches)

for p in player_profiles:
    game = Game({1: ('human', None), 2: player_profiles[p]})
    game.game_loop(verbose=True)
    