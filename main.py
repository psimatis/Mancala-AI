from collections import defaultdict
import pandas as pd
from mancala import Game
import player
from genetic_algorithm import train_genetic
import dqn

MATCHES_NUMBER = 100

players = (
    player.Human('human'),
    player.Random('random'),
    player.Greedy('greedy'),
    player.Genetic('gen_random', train_genetic(generations=5)),
    player.Genetic('gen_tournament', train_genetic(generations=5, simulations=1, tournament=100)),
    player.DQN('dqn_random', dqn.Agent().train_dqn(opponents=(player.Random('random'),))),
    player.DQN('dqn_greedy', dqn.Agent().train_dqn(opponents=(player.Greedy('greedy'),))),
    player.DQN('dqn_mix', dqn.Agent().train_dqn(opponents=(player.Greedy(), player.Random())))
)

def run_experiment():
    matches = defaultdict(lambda: defaultdict(int))
    for p1 in players[1:]:
        for p2 in players[1:]:
            if p1 == p2:
                continue
            for _ in range(MATCHES_NUMBER):
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

def play_mancala():
    for p in players[-1:]:
        print('Playing against:', p.name)
        game = Game({1: p, 2: players[0]})
        game.game_loop(verbose=True)

if __name__ == "__main__":
    print_results(run_experiment())
    play_mancala()
