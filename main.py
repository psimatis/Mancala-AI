import random
import pandas as pd
from collections import defaultdict 
from mancala import Game
import player
from genetic_algorithm import train_genetic
from dqn import train_dqn

# Initialize players
player_profiles = (
    player.Human('human'),
    player.Random('random'),
    player.Greedy('greedy'),
   player.Genetic('genetic_random', train_genetic(generations=5)),
   player.Genetic('genetic_tournament', train_genetic(generations=5, simulations=1, tournament=100)),
    player.DQN('dqn_random', train_dqn(opponent_types=(player.Random('random'),))),
    player.DQN('dqn_greedy', train_dqn(opponent_types=(player.Greedy('greedy'),))),
    player.DQN('dqn_mix', train_dqn()),
)

def run_experiment(matches_number=100):
    matches = defaultdict(lambda: defaultdict(int))
    for p1 in player_profiles[1:]:
        for p2 in player_profiles[1:]:
            if p1 == p2:
                continue
            for _ in range(matches_number):
                game = Game({1: p1, 2: p2})
                winner = game.game_loop(verbose=False)
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
    print()

    df['P1 Wins'] = df['Player 1 Win Rate'] > df['Player 2 Win Rate']
    df['P2 Wins'] = df['Player 1 Win Rate'] < df['Player 2 Win Rate']
    for i in ('1','2'):
        grouped = df.groupby('Player ' + i)['P' + i + ' Wins'].sum().reset_index()
        grouped = grouped.sort_values(by='P' + i + ' Wins', ascending=False).reset_index(drop=True)
        print(grouped.to_string(index=False))
        print()

def play_mancala(randomize_start=True):
    for p in player_profiles[-1:]:
        print('Playing against:', p.name)
        if randomize_start and random.random() < 0.5:
            game = Game({1: player_profiles[0], 2: p})
        else:
            game = Game({1: p, 2: player_profiles[0]})
        game.game_loop(verbose=True)

if __name__ == "__main__":
    matches_number = 100
    matches = run_experiment(matches_number)
    print_results(matches, matches_number)

    play_mancala()