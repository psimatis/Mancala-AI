# Mancala-AI
Mancala[^1] is a turn-based strategy board game where two players compete in stone collection. I keep beating the Nintendo Switch AI, thus I wrote my own. The repository contains the complete game along with several AI players (e.g., DQN, genetic).

## AI Opponents
1. Deep Q-Learning Network (DQN)[^2]: Uses reinforcement learning to develop a strategy based on rewards.
2. Double DQN[^3]: Enhances DQN by reducing overestimation.
1. Genetic Algorithm[^4]: Evolution selects the fittest candidate after multiple generations of simulated games.
1. Tournament Selection[^5]: Selects the fittest candidate through multiple tournaments (i.e., battle royal).
1. Minimax[^6]: Simulates the game down to a specified depth before selecting a move.
1. Greedy: Focuses on immediate gains.
1. Naive: Moves randomly.
1. Human: You play the game.

## Usage
    python main.py
    
The main script:
1. Initializes and trains the AI agents.
2. Runs a tournament between them.
3. Allows the player to compete against the AI.

## Parameters
To facilitate experimentation, the non-trivial opponents are highly configurable.

#### DQN
1. **opponents** (list of player objects): The sparring buddies used by DQN during training. 
1. **episodes** (int): Number of training games.
1. **epsilon_min** (float): Minimum epsilon for the epsilon-greedy policy.
1. **epsilon_decay** (float): Decay rate of epsilon.
1. **batch_size** (int): Size of the minibatch for training.
1. **capacity** (int): Capacity of the replay memory.
1. **gamma** (float): Discount factor for future rewards.
1. **learning_rate** (float): Learning rate for the optimizer.
1. **neurons** (int): Number of neurons in each hidden layer of the neural network.
1. **tau** (float): Soft update parameter for updating the target network.
2. **Double DQN** (boolean): Enables Double DQN.

#### Genetic Algorithm
1. **generations** (int): Number of iterations the algorithm will run.
1. **population_size** (int): Number of individuals in the population.
1. **mutation_rate** (float): Probability of mutation occurring in an individual.
1. **simulations** (int): Number of games each individual plays to evaluate fitness.
1. **elitism** (int): Number of top individuals directly carried over to the next generation.
1. **tournament** (int): Size of the tournament selection pool (0 for random selection).
1. **top** (int): Number of best individuals used for selection and breeding.

#### Minimax
1. **depth** (int): The maximum depth of the game tree that the algorithm will explore.

## A deeper dive for the nerds

#### Project Goal
My goal is to develop challenging AIs using a variety of methods. This project has been immensely fun, and I may add more AI agents in the future.

#### Agent Evaluation
Impartial evaluation is challenging due to varying performance metrics among agents. Furthermore, using myself as a sparring partner isn't ideal. Thus, I opted for tournaments between agents as the most practical and fair method (i.e., the best agent has the most wins).

#### Minimax
Deep explorations may outperform any player (given proper evaluation). However, Minimax is very slow even with alpha-beta pruning[^7]. Evaluation-wise, I considered both the entire player's side of the board and only the stores. Since there was no significant difference, I evaluate solely based on store difference, following Occam's razor.

#### Genetic Algorithm
I often heard in academic circles that *"genetic stuff never works"*. Nevertheless, I decided to give this *underdog* a chance. Both vanilla and tournament selection use the number of wins as fitness to evolve a score distribution for each pit. That score is multiplied by the number of stones, and the pit with the highest value is selected for the next move.

#### DQN
DQN training involves numerous parameters and is notoriously unstable. In addition, designing a dense and *good* reward policy is more of an art than a science. However, the use of Huber loss and soft updates of network weights (Polyak averaging) is beneficial in stabilizing training.

## References
[^1]: Mancala: https://en.wikipedia.org/wiki/Mancala
[^2]: DQN paper: https://arxiv.org/pdf/1312.5602
[^3]: Double DQN paper: https://arxiv.org/pdf/1509.06461v3
[^4]: Genetic Algortithm: https://en.wikipedia.org/wiki/Genetic_algorithm
[^5]: Tournament Selection: https://en.wikipedia.org/wiki/Tournament_selection#:~:text=Tournament%20selection%20is%20a%20method,at%20random%20from%20the%20population.
[^6]: Minimax Algorithm: https://en.wikipedia.org/wiki/Minimax#:~:text=Minmax%20(sometimes%20Minimax%2C%20MM%20or,case%20(maximum%20loss)%20scenario.
[^7]: Alpha-beta pruning: https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
