from collections import defaultdict
import pandas as pd
from mancala import Game
from human_player import Human
from random_player import Random 
from greedy_player import Greedy
from minimax import MinimaxAgent
import genetic_algorithm as ga
import dqn

MATCHES_NUMBER = 100

players = [
    Human(),
    Random(),
    Greedy(),
    MinimaxAgent('minimax_3_store', 3),
    ga.GeneticAgent('gen_random', ga.train_genetic(generations=5, verbose=False)),
    ga.GeneticAgent('gen_tournament', ga.train_genetic(generations=5, simulations=1, tournament=100, verbose=False)),
    dqn.DQNAgent('dqn_random', verbose=False).train_dqn(),
    dqn.DQNAgent('dqn_greedy', opponents=[Greedy()], verbose=False).train_dqn(),
]

players.append(dqn.DQNAgent('dqn_mix', opponents=players[1:], verbose=False).train_dqn())

def run_experiment():
    print('Running experimenrs')
    matches = defaultdict(lambda: defaultdict(int))
    for p1 in players[1:]:
        for p2 in players[1:]:
            if p1 == p2:
                continue
            match_name = p1.name + ' vs ' + p2.name
            print(match_name)
            for _ in range(MATCHES_NUMBER):
                winner = Game({1: p1, 2: p2}).game_loop()
                if winner == 0:
                    matches[match_name]['draw'] += 1
                elif winner == 1:
                    matches[match_name][p1.name] += 1
                else:
                    matches[match_name][p2.name] += 1
    return matches

def print_results(matches):
    results = []
    for match_name, results_dict in matches.items():
        p1, p2 = match_name.split(' vs ')
        p1_win_rate = (results_dict[p1] / MATCHES_NUMBER) * 100
        p2_win_rate = (results_dict[p2] / MATCHES_NUMBER) * 100
        draw_rate = (results_dict['draw'] / MATCHES_NUMBER) * 100
        results.append({
            "Player 1": p1,
            "Player 2": p2,
            "Player 1 Win Rate": p1_win_rate,
            "Player 2 Win Rate": p2_win_rate,
            "Draw Rate": draw_rate
        })
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()

    df['P1 Wins'] = df['Player 1 Win Rate'] > df['Player 2 Win Rate']
    df['P2 Wins'] = df['Player 1 Win Rate'] < df['Player 2 Win Rate']

    grouped_p1 = df.groupby('Player 1')['P1 Wins'].sum().reset_index()
    grouped_p1.columns = ['Player', 'P1 Wins']

    grouped_p2 = df.groupby('Player 2')['P2 Wins'].sum().reset_index()
    grouped_p2.columns = ['Player', 'P2 Wins']

    combined = pd.merge(grouped_p1, grouped_p2, on='Player', how='outer').fillna(0)
    combined['Total Wins'] = combined['P1 Wins'] + combined['P2 Wins']
    combined = combined.sort_values(by='Total Wins', ascending=False).reset_index(drop=True)
    print(combined.to_string(index=False))
    print()

def play_mancala():
    for p in players[-1:]:
        print('Playing against:', p.name)
        Game({1: p, 2: players[0]}).game_loop(verbose=True)

if __name__ == "__main__":
    print_results(run_experiment())
    play_mancala()
