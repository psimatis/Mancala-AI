from mancala import Game
from players.human import Human
from players.naive import Naive
from players.greedy import Greedy
from players.minimax import Minimax
import players.genetic_algorithm as ga
import players.dqn as dqn
from experiment import run_experiment

SHOW_TRAINING = True
GAMES = 10

def initialize_players(verbose):
    players = [
        Human(),
        Naive(),
        Greedy(),
        Minimax(name='mm2', depth=2),
        Minimax(name='mm3', depth=3),
        ga.GeneticAgent('ga_random', ga.train_genetic(generations=5, verbose=verbose)),
        ga.GeneticAgent('ga_tournament', ga.train_genetic(generations=5, simulations=1, tournament=100, verbose=verbose)),
        dqn.DQNAgent('dqn_random', episodes=500, verbose=verbose).train_dqn(),
        dqn.DQNAgent('ddqn_random', episodes=500, double_dqn=True, verbose=verbose).train_dqn(),
        dqn.DQNAgent('dqn_greedy', opponents=[Greedy()], verbose=verbose).train_dqn(),
        dqn.DQNAgent('dqn_mm', opponents=[Minimax(name='mm4', depth=4)], verbose=verbose).train_dqn(),
    ]
    players.append(dqn.DQNAgent('dqn_mix', opponents=players[1:], verbose=verbose).train_dqn())
    return players

def play_mancala(players):
    for p in players[-1:]:
        print('Playing against:', p.name)
        Game({1: p, 2: players[0]}).game_loop(verbose=True)

if __name__ == "__main__":
    players = initialize_players(verbose=SHOW_TRAINING)
    run_experiment(players, GAMES)
    play_mancala(players)
