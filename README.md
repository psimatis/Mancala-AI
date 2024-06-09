# Mancala
Mancala [^1] is a turn-based strategy board game where two players compete on pebble collection.
I became too good, beating the Nintendo Switch AI, thus I wrote my own.
The repository includes the complete game logic and supports multiple opponent strategies.

## Strategies/Opponents supported:
1. Random: The moves are random.
1. Greedy: Selects the best possible action for the given moment (e.g., capture, bonus round).
1. Genetic Algorithm [^2]: The fittest candidate is selected after multiple generations of simulated games with randomly generated individuals. 
1. Tournament Genetic Algorithm [^3]: The fittest candidate is selected from multiple tournaments of individuals (think of battle royal).
1. Human: No AI involved (i.e., the user plays the game).

## Genetic Algorithm Parameters
To facilitate experimentation the following parameters can be set:
1. generations (int): Number of iterations the algorithm will run.
2. population_size (int): Number of individuals in the population.
3. mutation_rate (float): Probability of mutation occurring in an individual.
4. simulations (int): Number of games each individual plays to evaluate fitness.
5. elitism (int): Number of top individuals directly carried over to the next generation.
6. tournament (int): Size of the tournament selection pool (0 for random selection).
7. top (int): Number of top individuals used for selection and breeding.
8. verbose (bool): If True, prints detailed information during the algorithm's execution.

## Observations/Todo
* The training of the Tournament Genetic Algorithm is unstable. I assume it is sometimes stuck in local minima.
* Add more AI opponents (e.g., DQN).
* I still beat all the AIs, if someone can give me a challenging AI I will appreciate it.

## References
[^1]: Mancala Wikipedia article: https://en.wikipedia.org/wiki/Mancala
[^2]: Genetic Algortithm Wikipedia article: https://en.wikipedia.org/wiki/Genetic_algorithm
[^3]: Tournament Selection Wikipedia article: https://en.wikipedia.org/wiki/Tournament_selection#:~:text=Tournament%20selection%20is%20a%20method,at%20random%20from%20the%20population.
