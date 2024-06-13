import random
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import mancala
import player

random.seed(0)
torch.manual_seed(0)

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

class Memory:
    def __init__(self, capacity, prepopulation_size):
        self.memory = deque(maxlen=capacity)
        self.prepopulation_size = prepopulation_size

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, state_size, action_size, memory):
        self.neurons = 64
        self.action_size = action_size
        self.memory = memory
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.00001
        self.model = DQN(state_size, self.neurons, action_size)
        self.target_model = DQN(state_size, self.neurons, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # TODO: Does the agent know which side he is playing on?
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.choice([a for a in range(self.action_size) if state[a] > 0])
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        invalid_actions = [a for a in range(self.action_size) if state[a] <= 0]
        q_values[0][invalid_actions] = float('-inf')
        return torch.argmax(q_values, dim=1).item()

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, game_overs = zip(*minibatch)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        game_overs = torch.tensor(game_overs)

        non_final_mask = ~game_overs
        non_final_next_states = next_states[non_final_mask]

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_state_values = torch.zeros(batch_size)

        if len(non_final_next_states) > 0:
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        expected_q_values = rewards + (next_state_values * self.gamma)

        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

def get_state(env):
    return env.board[1] + env.board[2]

def step(env, player, action):
    init_score = env.board[player][mancala.STORE]
    info = env.game_step(player, action, verbose=False)
    reward = 0

    if info['capture']:
        reward += 10
    if info['bonus_round']:
        reward += 5

    current_score = env.board[player][mancala.STORE]
    reward += current_score - init_score

    opponent = env.switch_side(player)
    opponent_score = env.board[opponent][mancala.STORE]
    reward += current_score - opponent_score

    if info['game_over']:
        if env.get_winner() == player:
            reward += 50
        else:
            reward -= 50
    return get_state(env), reward, info['game_over'], info['bonus_round']

def run_episode(agent, opponent_types, batch_size):
    opponent = random.choice(opponent_types)
    if random.random() < 0.5:
        env = mancala.Game({1: player.DQN('dqn', agent), 2: opponent})
    else:
        env = mancala.Game({1: opponent, 2: player.DQN('dqn', agent)})
    state = get_state(env)
    loss = -1
    total_reward = 0
    game_over = False
    current_player = 1
    while not game_over:
        if current_player == 1:
            action = agent.act(state)
            next_state, reward, game_over, bonus_round = step(env, current_player, action)
            agent.memory.remember(state, action, reward, next_state, game_over)
            total_reward += reward
            if len(agent.memory) > agent.memory.prepopulation_size:
                loss = agent.replay(batch_size)
        else:
            action = opponent.act(env, current_player)
            next_state, _, game_over, bonus_round = step(env, current_player, action)
        if not bonus_round:
            current_player = 2 if current_player == 1 else 1
        state = next_state
    return loss, total_reward

def plot_history(history):
    _, axs = plt.subplots(2, figsize=(10, 10))

    axs[0].plot([l[0] for l in history])
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('DQN Training Loss')

    axs[1].plot([r[1] for r in history])
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Reward')
    axs[1].set_title('DQN Rewards')

    plt.tight_layout()
    plt.show()

def train_dqn(episodes=1000, batch_size=64, opponent_types=(player.Random('random'), player.Greedy('greedy')), verbose=True):
    if verbose:
        print('Started training DQN player')
    state_size = (mancala.STORE + 1) * 2
    action_size = mancala.STORE
    memory = Memory(capacity=8000, prepopulation_size=1000)
    agent = Agent(state_size, action_size, memory)
    history = []

    while len(agent.memory) < agent.memory.prepopulation_size:
        run_episode(agent, opponent_types, batch_size)

    for e in range(episodes):
        loss, reward = run_episode(agent, opponent_types, batch_size)
        if e % 1 == 1000:
            agent.update_target_model()

        if loss != -1:
            history.append((loss, reward))
        if verbose:
            print(f"Episode: {e} Memory {len(agent.memory)} Epsilon: {agent.epsilon:.2f} Loss: {loss:.2f} Reward: {reward}")

    plot_history(history)
    return agent
