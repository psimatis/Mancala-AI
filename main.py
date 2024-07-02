from mancala import Game
from players.human import Human
from players.naive import Naive
from players.greedy import Greedy
from players.minimax import Minimax
import players.genetic_algorithm as ga
import players.dqn as dqn
from experiment import run_experiment

TRAIN_AGENTS = False
SHOW_TRAINING = True
GAMES = 100

def initialize_agents():
    agents = [
        Human(),
        Greedy(),
        Naive(),
        Minimax('minimax_even'),
        Minimax(name='minimax_odd', depth=3)
    ]
    if TRAIN_AGENTS:
        agents.append(ga.GeneticAgent('ga_vanilla', ga.train_genetic(verbose=SHOW_TRAINING)))
        agents.append(ga.GeneticAgent('ga_tournament', ga.train_genetic(simulations=1, tournament=100, verbose=SHOW_TRAINING)))
        agents.append(dqn.DQNAgent('dqn').train_dqn(SHOW_TRAINING))
        agents.append(dqn.DQNAgent('ddqn', double_dqn=True).train_dqn(SHOW_TRAINING))
    else:
        agents.append(ga.load('best_models/ga_vanilla'))
        agents.append(ga.load('best_models/ga_tournament'))
        agents.append(dqn.DQNAgent().load('best_models/dqn.pth'))
        agents.append(dqn.DQNAgent('ddqn').load('best_models/ddqn.pth'))
    return agents

def play_mancala(agents):
    for p in agents[1:]:
        print('Playing against:', p.name)
        Game({1: p, 2: agents[0]}).game_loop(verbose=True)

if __name__ == "__main__":
    agents = initialize_agents()
    run_experiment(agents, GAMES)
    play_mancala(agents)
