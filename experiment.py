from collections import defaultdict
import pandas as pd
from mancala import Game

def run_experiment(players, games_number=100):
    games = defaultdict(lambda: defaultdict(int))
    if players[0].name == 'human':
        players = players[1:]
    for p1 in players:
        for p2 in players:
            if p1 == p2:
                continue
            game_name = p1.name + ' vs ' + p2.name
            print('Now playing:', game_name)
            for _ in range(games_number):
                winner = Game({1: p1, 2: p2}).game_loop()
                if winner == 0:
                    games[game_name]['draw'] += 1
                elif winner == 1:
                    games[game_name][p1.name] += 1
                else:
                    games[game_name][p2.name] += 1
    return print_results(games, games_number)

def print_results(games, games_number):
    results = []
    for match_name, results_dict in games.items():
        p1, p2 = match_name.split(' vs ')
        p1_win_rate = (results_dict[p1] / games_number) * 100
        p2_win_rate = (results_dict[p2] / games_number) * 100
        draw_rate = (results_dict['draw'] / games_number) * 100
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
    
    dqn_wins = combined.loc[combined['Player'] == 'dqn', 'Total Wins'].sum()
    return combined, dqn_wins