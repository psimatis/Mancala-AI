# Mancala
Mancala[^1] is a turn-based strategy board game where two players compete on stone collection.
I keep beating the Nintendo Switch AI, thus I wrote my own.
The repository includes the complete game and supports multiple opponent strategies.

## Usage
    python main.py

## Strategies/Opponents supported:
1. Deep Q-Learning Network (DQN)[^2]: Uses reinforcement learning (i.e., learns from rewards) to develop a strategy.
1. Genetic Algorithm[^3]: Evolution selects the fittest candidate after multiple generations of simulated games.
1. Tournament Selection[^4]: Selects the fittest candidate from multiple tournaments (i.e., battle royal).
1. Minimax[^5]: Simulates the game down to specified depth before picking a move.
1. Greedy: Focuses on immediate gains.
1. Naive: Selects moves randomly.
1. Human: You play the game.

## Paramameters
To facilitate experimentation the non-trivial opponents are highly configurable.

### DQN
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

### Genetic Algorithm Parameters
1. **generations** (int): Number of iterations the algorithm will run.
1. **population_size** (int): Number of individuals in the population.
1. **mutation_rate** (float): Probability of mutation occurring in an individual.
1. **simulations** (int): Number of games each individual plays to evaluate fitness.
1. **elitism** (int): Number of top individuals directly carried over to the next generation.
1. **tournament** (int): Size of the tournament selection pool (0 for random selection).
1. **top** (int): Number of best individuals used for selection and breeding.

### Minimax Parameters
1. **depth** (int): The maximum depth of the game tree that the algorithm will explore.

## References
[^1]: Mancala: https://en.wikipedia.org/wiki/Mancala
[^2]: DQN Paper: https://arxiv.org/pdf/1312.5602
[^3]: Genetic Algortithm: https://en.wikipedia.org/wiki/Genetic_algorithm
[^4]: Tournament Selection: https://en.wikipedia.org/wiki/Tournament_selection#:~:text=Tournament%20selection%20is%20a%20method,at%20random%20from%20the%20population.
[^5]: Minimax Algorithm: https://en.wikipedia.org/wiki/Minimax#:~:text=Minmax%20(sometimes%20Minimax%2C%20MM%20or,case%20(maximum%20loss)%20scenario.

