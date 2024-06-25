from mancala import Game
from players.human import Human
from players.naive import Naive
from players.greedy import Greedy
from players.minimax import Minimax
import players.genetic_algorithm as ga
import players.dqn as dqn
from experiment import run_experiment

SHOW_TRAINING = True
GAMES = 100

def initialize_players():
    players = [
        Human(),
        Greedy(),
        Naive(),
        Minimax(),
        ga.GeneticAgent('ga_vanilla', ga.train_genetic(verbose=SHOW_TRAINING)),
        ga.GeneticAgent('ga_tournament', ga.train_genetic(simulations=1, tournament=100, verbose=SHOW_TRAINING)),
    ]
    players.append(dqn.DQNAgent('dqn', opponents=[Minimax()]).train_dqn(SHOW_TRAINING))
    players.append(dqn.DQNAgent('ddqn', opponents=[Minimax()], double_dqn=True).train_dqn(SHOW_TRAINING))
    return players

def play_mancala(players):
    for p in players[1:]:
        print('Playing against:', p.name)
        Game({1: p, 2: players[0]}).game_loop(verbose=True)

if __name__ == "__main__":
    players = initialize_players()
    run_experiment(players, GAMES)
    play_mancala(players)
