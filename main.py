import pandas as pd
from collections import defaultdict 
from mancala import Game
import player
from genetic_algorithm import train_genetic
import dqn

# Initialize players
random_player = player.Random('random')
greedy_player = player.Greedy('greedy')
human_player = player.Human('human')
genetic_random_player = player.Genetic('genetic_random', train_genetic(generations=2))
genetic_tournament_player = player.Genetic('genetic_tournament', train_genetic(generations=2, simulations=1, tournament=100))
dqn_player = player.DQN('dqn', dqn.train_dqn()) 

player_profiles = (random_player, greedy_player, genetic_random_player, genetic_tournament_player, dqn_player)

def run_experiment(matches_number=100):
    matches = defaultdict(lambda: defaultdict(int))
    for p1 in player_profiles:
        for p2 in player_profiles:
            if p1 == p2:
                continue
            for _ in range(matches_number):
                game = Game({1: p1, 2: p2})
                winner = game.game_loop()
                match_name = p1.name + ' vs ' + p2.name
                if winner == 0:
                    matches[match_name]['draw'] += 1
                elif winner == 1:
                    matches[match_name][p1.name] += 1
                else:
                    matches[match_name][p2.name] += 1
    return matches

def print_results(matches, matches_number):
    results = []
    for match_name, results_dict in matches.items():
        p1, p2 = match_name.split(' vs ')
        p1_wins = results_dict[p1]
        p2_wins = results_dict[p2]
        draws = results_dict['draw']
        p1_win_rate = (p1_wins / matches_number) * 100
        p2_win_rate = (p2_wins / matches_number) * 100
        draw_rate = (draws / matches_number) * 100

        results.append({
            "Player 1": p1,
            "Player 2": p2,
            "Player 1 Win Rate": f"{p1_win_rate:.1f}%",
            "Player 2 Win Rate": f"{p2_win_rate:.1f}%",
            "Draw Rate": f"{draw_rate:.1f}%"
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

def play_mancala():
    for p in player_profiles:
        game = Game({1: human_player, 2: dqn_player})
        game.game_loop(verbose=True)

if __name__ == "__main__":
    matches_number = 10
    matches = run_experiment(matches_number)
    print_results(matches, matches_number)

    play_mancala()