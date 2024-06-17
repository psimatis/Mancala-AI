from mancala import Game
from players.human import Human
from players.naive import Naive
from players.greedy import Greedy
from players.minimax import Minimax
import players.genetic_algorithm as ga
import players.dqn as dqn
from experiment import run_experiment

SHOW_TRAINING = True
MATCHES_NUMBER = 100

def initialize_players(verbose):
    players = [
        Human(),
        Naive(),
        Greedy(),
        Minimax(),
        ga.GeneticAgent('ga_random', ga.train_genetic(generations=5, verbose=verbose)),
        # ga.GeneticAgent('ga_tournament', ga.train_genetic(generations=5, simulations=1, tournament=100, verbose=verbose)),
        dqn.DQNAgent('dqn_random', verbose=verbose).train_dqn(),
        # dqn.DQNAgent('dqn_greedy', opponents=[Greedy()], verbose=verbose).train_dqn(),
    ]
    # players.append(dqn.DQNAgent('dqn_mix', opponents=players[1:], verbose=verbose).train_dqn())
    return players

def play_mancala(players):
    for p in players[-1:]:
        print('Playing against:', p.name)
        Game({1: p, 2: players[0]}).game_loop(verbose=True)
    

if __name__ == "__main__":
    players = initialize_players(verbose=SHOW_TRAINING)
    run_experiment(players, MATCHES_NUMBER)
    play_mancala(players)
