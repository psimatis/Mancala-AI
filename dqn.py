import random
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import mancala
import player

class DQN(nn.Module):
    def __init__(self, state_size, neurons, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_size, action_size):
        self.neurons = 32
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.01
        self.model = DQN(state_size, self.neurons, action_size)
        self.target_model = DQN(state_size, self.neurons, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.choice([a for a in range(self.action_size) if state[a] > 0])
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        sorted_actions = torch.argsort(q_values, dim=1, descending=True).squeeze()
        for action in sorted_actions:
            if state[0][action] > 0:
                return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0
        for state, action, reward, next_state, game_over in minibatch:
            target = reward
            if not game_over:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.target_model(next_state)).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            adjusted_qvalues = self.model(state).clone()
            adjusted_qvalues[0][action] = target
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, adjusted_qvalues)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / batch_size

def get_state(env):
    return env.board[1] + env.board[2]

def step(env, player, action):
    info = env.game_step(player, action, verbose=False)
    reward = 0
    if info['capture'] or info['bonus_round']:
        reward += 10
    if info['game_over']:
        if env.get_winner() == player:
            reward += 100
        else:
            reward -= 100
    return get_state(env), reward, info['game_over'], info['bonus_round']

def plot_history(history):
    plt.plot(history)
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('DQN Training Loss')
    plt.show()

def train_dqn(episodes=200, batch_size=64, opponent_types=(player.Random('random'), player.Greedy('greedy')), verbose=True):
    """
    Train the DQN agent.

    Args:
        episodes (int): Number of episodes to train.
        batch_size (int): Size of the minibatch for training.

    Returns:
        DQNAgent: Trained DQN agent.
    """
    if verbose: 
        print('Started training DQN player')
    state_size = (mancala.BANK + 1) * 2
    action_size = mancala.BANK
    agent = Agent(state_size, action_size)
    history = []

    for e in range(episodes):
        opponent = random.choice(opponent_types)
        env = mancala.Game({1: player.DQN('dqn', agent), 2: opponent})
        state = get_state(env)
        game_over = False
        loss = -1
        total_reward = 0
        current_player = 1
        while not game_over:
            if current_player == 1:
                action = agent.act(state)
                next_state, reward, game_over, bonus_round = step(env, current_player, action)
                agent.remember(state, action, reward, next_state, game_over)
                total_reward += reward
                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)
            else:
                action = opponent.act(env, current_player)
                next_state, _, game_over, bonus_round = step(env, current_player, action)
            if not bonus_round:
                current_player = 2 if current_player == 1 else 1
            state = next_state
            
        if e % 5 == 0: agent.update_target_model()
        if loss != -1: history.append(loss)
        if verbose: print(f"Episode {e}, Memory: {len(agent.memory)}, Epsilon: {agent.epsilon:.2f}, Loss: {loss:.2f}, Total Reward: {total_reward}")
    #if verbose: plot_history(history)
    return agent