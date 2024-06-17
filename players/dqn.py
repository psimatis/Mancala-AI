import random
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import mancala
from players.human import Human
from players.naive import Naive

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
    def __init__(self, capacity, batch_size):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

class EGreedy:
    def __init__(self, epsilon, epsilon_min, decay):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay

    def update(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay

    def explore(self):
        if random.random() <= self.epsilon:
            return True
        return False

class DQNAgent:
    def __init__(self, name='dqn', opponents=[Naive()], episodes=200, epsilon_min=0.01, epsilon_decay=0.99, batch_size=512, capacity=10000, gamma=0.9, learning_rate=0.001, neurons=32, tau=0.01, verbose=True):
        self.name = name
        self.state_size = (mancala.STORE + 1) * 2
        self.action_size = mancala.STORE
        self.e_greedy = EGreedy(1, epsilon_min, epsilon_decay)
        self.memory = Memory(capacity, batch_size)
        self.gamma = gamma
        self.policy_model = DQN(self.state_size, neurons, self.action_size)
        self.target_model = DQN(self.state_size, neurons, self.action_size)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.opponents = opponents
        self.episodes = episodes
        self.tau = tau
        self.verbose = verbose
        self.fixed_states = self.collect_eval_states()

    def collect_eval_states(self, num_states=1000):
        states = []
        while len(states) < num_states:
            env = mancala.Game({1: Naive(), 2: Naive()})
            for _ in range(num_states):
                states.append(env.get_state())
                action = env.players[env.current_player].act(env)
                if env.step(action, verbose=False)['game_over']:
                    break
        return states

    def update_target_model(self):
        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.policy_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_model.load_state_dict(target_net_state_dict)

    def act(self, game):
        state = game.get_state()
        state = state[7:] + state[:7] if game.current_player == 2 else state

        if self.e_greedy.explore():
            return random.choice(game.get_valid_moves(game.current_player))

        invalid_actions = [a for a in range(self.action_size) if state[a] == 0]
        q_values = self.policy_model(torch.FloatTensor(state).unsqueeze(0))
        q_values[0][invalid_actions] = float("-inf")
        return torch.argmax(q_values, dim=1).item()

    def replay(self):
        minibatch = self.memory.sample()
        states, actions, rewards, next_states, game_overs = map(torch.tensor, zip(*minibatch))
        states, next_states = states.float(), next_states.float()

        current_q_values = self.policy_model(states).gather(1, actions.unsqueeze(1))
        next_state_values = torch.zeros(self.memory.batch_size)

        non_final_next_states = next_states[~game_overs]
        if len(non_final_next_states) > 0:
            next_state_values[~game_overs] = (self.target_model(non_final_next_states).max(1)[0].detach())

        expected_q_values = rewards + self.gamma * next_state_values
        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.e_greedy.update()
        return loss.item()

    def step(self, env, action):
        init_score = env.board[env.current_player][mancala.STORE]
        info = env.step(action, verbose=False)

        current_score = env.board[env.current_player][mancala.STORE]
        opponent_score = env.board[env.switch_side(env.current_player)][mancala.STORE]
        reward = 2 * current_score - opponent_score - init_score

        if not info["bonus_round"]:
            reward -= info['capture_exposure']

        if info["game_over"]:
            if env.get_winner() == env.current_player:
                reward += 50
            else:
                reward -= 50

        if info["capture"] or info["bonus_round"]:
            reward += 10
        return env.get_state(), reward, info["game_over"]

    def run_episode(self, opponent_types):
        opponent = random.choice(opponent_types)
        env = mancala.Game({1: DQNAgent('dqn', self), 2: opponent})
        state = env.get_state()
        info = {'avg_loss': 0, 'avg_reward': 0, 'steps': 0}
        game_over = False

        while not game_over:
            info['steps'] += 1
            if 'dqn' in env.players[env.current_player].name:
                action = self.act(env)
                next_state, reward, game_over = self.step(env, action)
                self.memory.remember(state, action, reward, next_state, game_over)
                info['avg_reward'] += reward
                if len(self.memory) > self.memory.batch_size:
                    info['avg_loss'] += self.replay()
            else:
                action = opponent.act(env)
                next_state, _, game_over = self.step(env, action)
            state = next_state
        info['avg_reward'] /= info['steps']
        info['avg_loss'] /= info['steps']
        return info
    
    def compute_average_max_q(self):
        max_q_values = []
        for state in self.fixed_states:
            q_values = self.policy_model(torch.FloatTensor(state).unsqueeze(0))
            max_q_values.append(torch.max(q_values).item())
        return sum(max_q_values)/len(max_q_values)

    def plot_history(self, history):
        _, axs = plt.subplots(3, figsize=(8, 13))
        labels = ('Loss', 'Reward', 'Average Max Q')
        for i in range(len(labels)):
            axs[i].plot([h[i] for h in history])
            axs[i].set_xlabel('Episodes')
            axs[i].set_ylabel(labels[i])
        plt.grid(True)
        plt.show()

    def train_dqn(self):
        if self.verbose:
            print('Training DQN agent against:', [o.name for o in self.opponents])
        steps = 0
        history = []
        for e in range(self.episodes):
            info = self.run_episode(self.opponents)
            steps += info['steps']
            self.update_target_model()
            avg_max_q = self.compute_average_max_q()
            history.append((info['avg_loss'], info['avg_reward'], avg_max_q))
            if self.verbose:
                print(f"Episode: {e} Steps: {steps} Epsilon: {self.e_greedy.epsilon:.2f} Loss: {info['avg_loss']:.2f} Reward: {info['avg_reward']:.2f} Q: {avg_max_q:.2f}")
        if self.verbose:
            self.plot_history(history)
        return self
