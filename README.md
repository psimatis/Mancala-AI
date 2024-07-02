# Mancala-AI
Mancala[^1] is a classic board game where two players compete in stone collection. After beating the Nintendo Switch AI, I decided to write my own. The repository contains the complete game logic (e.g., capture mechanism) along with several AI agents of various difficulties.
 
## Agents
1. **Deep Q-Learning Network (DQN)**[^2]: Deep reinforcement learning develops a strategy based on rewards.
2. **Double DQN**[^3]: Enhances DQN by reducing overestimation.
1. **Genetic Algorithm**[^4]: Evolution selects the fittest candidate after multiple generations of simulated games.
1. **Tournament Selection**[^5]: Selects the fittest candidate through multiple tournaments (i.e., battle royal).
1. **Minimax**[^6]: Simulates the game down to a specified depth before selecting a move.
1. **Greedy**: Focuses on immediate gains.
1. **Naive**: Moves randomly.
1. **Human**: You play the game.
 
## Usage
    python main.py
    
The main script:
1. Initializes the AI agents.
2. Runs a tournament between them.
3. Allows the player to compete against the AI.

The script can be customized using three parameters:
1. **TRAIN_AGENTS** (boolean): If true, the script trains the AI agents from scratch. Otherwise, it loads pretrained agents.
2. **SHOW_TRAINING** (boolean): When true, it displays the training progress, including plots.
3. **GAMES** (int): Specifies the number of games played between each pair of AI agents during the tournament.

## Parameters
To facilitate experimentation, the non-trivial opponents are highly configurable. Furthermore, the **verbose** (boolean) parameter enables printing during training if applicable.
 
#### DQN
1. **opponents** (list of player objects): The sparring buddies the DQN uses during training.
1. **episodes** (int): Number of training games.
1. **epsilon_min** (float): Minimum epsilon for the epsilon-greedy policy.
1. **epsilon_decay** (float): Decay rate of epsilon.
1. **batch_size** (int): Size of the minibatch for training.
1. **capacity** (int): Capacity of the replay memory.
1. **gamma** (float): Discount factor for future rewards.
1. **learning_rate** (float): Learning rate for the optimizer.
1. **neurons** (int): Number of neurons in each hidden layer of the neural network.
1. **tau** (float): Soft update parameter for updating the target network.
2. **double_dqn** (boolean): Enables Double DQN.
 
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
 
## Experiments
Evaluation is challenging due to varying performance metrics among agents, and using myself as a sparring partner isn't ideal. Thus, I opted for tournaments between agents as the most practical and fair method, where the best agent is the one with the most wins. The tournament runs multiple games per agent combination, and an agent wins by beating >50% of the games. Draws are possible but ignored. 

I trained the agents using grid search, and the best performing ones are stored in the *best_models* directory. My training configuration might <ins>not</ins> be optimal so feel free to experiment and *let me know if you can beat my agents*.

The table below demonstrates agent performance in a tournament. Running the code with **TRAIN_AGENT** set to *False* should return the same results. 

<table align="center">
  <tr>
    <th>Player</th>
    <th>Player 1 Wins</th>
    <th>Player 2 Wins</th>
    <th>Total Wins</th>
  </tr>
  <tr>
    <td>Double DQN</td>
    <td align="center">7</td>
    <td align="center">4</td>
    <td align="center">11</td>
  </tr>
  <tr>
    <td>DQN</td>
    <td align="center">7</td>
    <td align="center">3</td>
    <td align="center">10</td>
  </tr>
  <tr>
    <td>Minimax Even</td>
    <td align="center">7</td>
    <td align="center">3</td>
    <td align="center">10</td>
  </tr>
  <tr>
    <td>Minimax Odd</td>
    <td align="center">7</td>
    <td align="center">2</td>
    <td align="center">9</td>
  </tr>
  <tr>
    <td>Greedy</td>
    <td align="center">4</td>
    <td align="center">3</td>
    <td align="center">7</td>
  </tr>
  <tr>
    <td>Vanilla GA</td>
    <td align="center">3</td>
    <td align="center">3</td>
    <td align="center">6</td>
  </tr>
  <tr>
    <td>Tournament GA</td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td align="center">2</td>
  </tr>
  <tr>
    <td>Naive</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
  </tr>
</table>

*Double DQN* achieved the highest total wins, while *DQN* and *Minimax Even* (i.e., depth two) followed closely with 10 wins each. *Minimax Odd* (i.e., depth three) came third, perfoming slightly worse than its even variant. *Greedy* and *Vanilla GA* demonstrated moderate success, being placed fifth and sixth respectively. *Tournament GA* underperformed with only two, just above *Naive*, which lost every game and serves as a good baseline. 

Anecdotaly, I always found playing as Player 2 harder. A notable observation is that Player 1 consistently achieved more wins. This disparity could be due to an inherent first-move advantage in Mancala, which is further supported by the fact that most agents execute a perfect opening[^7]. Nevertheless, I experimented with training the Genetic and DQN agents both as Player 1 and Player 2 to no significant benefit. This can be attributed to the minimal impact of player order on the overall state space. Considering the configuration of 48 stones distributed across 14 pits, the total number of possible states is approximately:
$$\binom{48 + 14 - 1}{14 - 1} = \binom{61}{13}$$
 
#### DQN
DQN training involves numerous parameters and is notoriously unstable. However, the use of Huber loss and soft updates of network weights (Polyak averaging) is beneficial in stabilizing training. Both DQN agents were trained against a single opponent, but there is potential for improvement if trained against a diverse set of strategies. The figures below illustrate the average reward per episode for *Double DQN* (left) and *DQN* (right). The graph was smoothed using running average with the original line faded in the background. 

<p align="center">
 <img src="./plots/ddqn.png" style="width:49%" title="DDQN">
<img src="./plots/dqn.png" style="width:49%" title="DQN">
</p>

Designing a dense and effective reward policy is more of an art than a science. For example, *Double DQN* outperforms *DQN* despite their similar reward. To evaluate the effectiveness of my reward structure in helping the agent win, I plotted the number of wins against the reward received for various training configurations.  The empty upper left corner indicates that agents with low rewards tend to lose. In addition, the dot color denotes the number of steps taken by the agent during an episode. 
<p align="center">
 <img src="./plots/rewards vs wins.png" style="width:70%" title="DDQN">
</p>
 
#### Genetic Algorithm
I often heard in academic circles that *"genetic stuff never works"*. Nevertheless, I decided to give this *underdog* a chance. Both vanilla and tournament selection use the number of wins as fitness to evolve a score distribution for the pits. When the agent acts, the score is multiplied by the number of stones in each pit, and the pit with the highest value is selected as the next move. This method is quite rigid and it does not take into account game mechanics (e.g., bonus round). Furthermore, *Tournament GA* underperformed, likely due to overfitting (i.e., individuals only learned how to beat their peers). The figure below shows the best fitness per generation for *Vanilla GA* (left) and *Tournament GA* (right).

<p align="center">
<img src="./plots/ga_random.png" style="width:49%; height:auto;" title="Vanilla GA">
<img src="./plots/ga_tournament.png" style="width:49%; height:auto;" title="GA Tournament">
</p>
 
#### Minimax
Given proper evaluation, deep explorations can outperform any player. However, the number of possible states grows exponentially, making Minimax slow even with alpha-beta pruning[^8]. An interesting observation is that even depths perform better than odd depths. Intuitively, even depths conclude on the opponent's turn, allowing a safer strategy, assuming optimal play. Conversely, odd depths are riskier since the player does not see the opponent's immediate response.

### Future Work
This project has been immensely fun, and I may add more agents in the future.
 
## References
[^1]: Mancala: https://en.wikipedia.org/wiki/Mancala
[^2]: DQN paper: https://arxiv.org/pdf/1312.5602
[^3]: Double DQN paper: https://arxiv.org/pdf/1509.06461v3
[^4]: Genetic Algortithm: https://en.wikipedia.org/wiki/Genetic_algorithm
[^5]: Tournament Selection: https://en.wikipedia.org/wiki/Tournament_selection#:~:text=Tournament%20selection%20is%20a%20method,at%20random%20from%20the%20population.
[^6]: Minimax Algorithm: https://en.wikipedia.org/wiki/Minimax#:~:text=Minmax%20(sometimes%20Minimax%2C%20MM%20or,case%20(maximum%20loss)%20scenario.
[^7]: Solving Kalah paper: https://naml.us/paper/irving2000_kalah.pdf
[^8]: Alpha-beta pruning: https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
