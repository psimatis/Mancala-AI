import os
import random
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import mancala
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
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, game_overs = map(torch.tensor, zip(*minibatch))
        return states.float(), actions, rewards, next_states.float(), game_overs

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
    def __init__(self, name='dqn', opponents=[Naive()], episodes=500, epsilon_min=0.01, epsilon_decay=0.99, batch_size=256, capacity=10000, gamma=0.9, learning_rate=0.001, neurons=32, tau=0.01, double_dqn=False):
        self.name = name
        self.state_size = (mancala.STORE + 1) * 2
        self.action_size = mancala.STORE
        self.e_greedy = EGreedy(1, epsilon_min, epsilon_decay)
        self.memory = Memory(capacity, batch_size)
        self.gamma = gamma
        self.neurons = neurons
        self.policy_model = DQN(self.state_size, neurons, self.action_size)
        self.target_model = DQN(self.state_size, neurons, self.action_size)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.opponents = opponents
        self.episodes = episodes
        self.tau = tau
        self.double_dqn = double_dqn
        self.fixed_states = self.collect_eval_states()
        self.history = []

    def collect_eval_states(self, num_states=1000):
        states = []
        while len(states) < num_states:
            env = mancala.Game({1: Naive(), 2: Naive()})
            while True:
                states.append(env.get_state())
                action = env.players[env.current_player].act(env)
                if env.step(action)['game_over']:
                    break
        return states
    
    def compute_average_max_q(self):
        max_q_values = []
        for state in self.fixed_states:
            q_values = self.policy_model(torch.FloatTensor(state).unsqueeze(0))
            max_q_values.append(torch.max(q_values).item())
        return sum(max_q_values)/len(max_q_values)

    def update_target_model(self):
        for target_param, policy_param in zip(self.target_model.parameters(), self.policy_model.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
    
    def load(self, path, neurons=None):
        state_dict = torch.load(path)
        if neurons:
            self.policy_model = DQN(self.state_size, neurons, self.action_size)  
            self.policy_model.load_state_dict(state_dict)
        else:
            self.policy_model = DQN(self.state_size, state_dict['neurons'], self.action_size)
            self.policy_model.load_state_dict(state_dict['model_state_dict'])
        self.policy_model.eval()
        self.e_greedy.epsilon = 0
        return self
    
    def save(self, path):
        model_path = os.path.join(path, f'dqn_model_{self.name}.pth')
        state_dict = {
            'opponents': ' '.join([o.name for o in self.opponents]), 
            'episodes': self.episodes, 
            'epsilon_decay': self.e_greedy.decay, 
            'batch_size': self.memory.batch_size, 
            'capacity': len(self.memory), 
            'neurons': self.neurons,
            'double_dqn': self.double_dqn,
            'model_state_dict': self.policy_model.state_dict()
        }
        torch.save(state_dict, model_path)        

    def act(self, game):
        state = game.get_state()
        state = state[7:] + state[:7] if game.current_player == 2 else state
        if self.e_greedy.explore():
            return random.choice(game.get_valid_moves())
        invalid_actions = [a for a in range(self.action_size) if state[a] == 0]
        q_values = self.policy_model(torch.FloatTensor(state).unsqueeze(0))
        q_values[0][invalid_actions] = float("-inf")
        return torch.argmax(q_values, dim=1).item()

    def replay(self):
        states, actions, rewards, next_states, game_overs = self.memory.sample()
        predicted_q = self.policy_model(states).gather(1, actions.unsqueeze(1))
        next_state_q = torch.zeros(self.memory.batch_size)
        next_states = next_states[~game_overs]
        if len(next_states) > 0:
            if self.double_dqn:
                next_actions = self.policy_model(next_states).max(1)[1].unsqueeze(1)
                next_state_q[~game_overs] = self.target_model(next_states).gather(1, next_actions).squeeze().detach()
            else:
                next_state_q[~game_overs] = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_state_q
        loss = self.criterion(predicted_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.e_greedy.update()
        return loss.item()
    
    def calculate_reward(self, env, player, init_score, info):
        score = env.board[player][mancala.STORE]
        reward = score - init_score
        if score == init_score:
            reward -= 5
        if info['capture']:
            reward += 10
        if info["bonus_round"]:
            reward += 20
        if not info["bonus_round"]:
            reward -= info['capture_exposure']
        if info["game_over"]:
            if env.get_winner() == player:
                reward += 100
            else:
                reward -= 100
        return reward

    def step(self, env, action):
        player = env.current_player
        init_score = env.board[player][mancala.STORE]
        info = env.step(action)
        reward = self.calculate_reward(env, player, init_score, info)
        return env.get_state(), reward, info["game_over"]

    def run_episode(self, opponent_types):
        opponent = random.choice(opponent_types)
        env = mancala.Game({1: self, 2: opponent})
        state = env.get_state()
        info = {'loss': 0, 'reward': 0, 'steps': 0}
        game_over = False
        while not game_over:
            if 'dqn' in env.players[env.current_player].name:
                action = self.act(env)
                next_state, reward, game_over = self.step(env, action)
                self.memory.remember(state, action, reward, next_state, game_over)
                if len(self.memory) > self.memory.batch_size:
                    info['loss'] += self.replay()
                info['reward'] += reward
                info['steps'] += 1
            else:
                action = opponent.act(env)
                next_state, _, game_over = self.step(env, action)
            state = next_state
        info['reward'] /= info['steps']
        info['loss'] /= info['steps']
        info['avg_max_q'] = self.compute_average_max_q()
        return info

    def plot_history(self):
        _, axs = plt.subplots(3, figsize=(8, 13))
        for i, l in enumerate(('Reward', 'Average Q', 'Loss')):
            axs[i].plot([h[i] for h in self.history])
            axs[i].set_xlabel('Episodes')
            axs[i].set_ylabel(l)
            axs[i].grid(True)
        plt.show()

    def train_dqn(self, verbose):
        if verbose:
            print('Training DQN agent against:', [o.name for o in self.opponents])
        for e in range(self.episodes):
            info = self.run_episode(self.opponents)
            self.update_target_model()
            self.history.append((info['reward'], info['avg_max_q'], info['loss'], info['steps']))
            if verbose:
                print(f"Episode: {e} Steps: {info['steps']} Epsilon: {self.e_greedy.epsilon:.2f} Loss: {info['loss']:.2f} Reward: {info['reward']:.2f} Q: {info['avg_max_q']:.2f}")
        if verbose:
            self.plot_history()
        return self
