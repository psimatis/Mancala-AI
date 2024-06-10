import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import mancala

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.target_model(next_state)).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f = target_f.clone()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def get_state(env):
    return env.board[1] + env.board[2]

def step(env, player, action):
    if env.is_slot_empty(player, action):
        return get_state(env), -10, False  # Penalize invalid moves
    landing = env.move(player, action)
    reward = 1 if env.capture(player, landing) else 0
    done = env.is_side_empty()
    reward += 10 if env.check_bonus_round(landing) else 0
    reward += 100 if done and env.get_winner() == player else 0
    reward -= 100 if done and env.get_winner() != player else 0
    return get_state(env), reward, done

def train_dqn(episodes=10, batch_size=32):
    print('Started training DQN player')
    env = mancala.Game({'1': ['AI', None], '2': ['random', None]})
    state_size = len(get_state(env))
    action_size = 6  # 6 possible moves
    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        env.reset()
        state = get_state(env)
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = step(env, 1, action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        agent.update_target_model()
        if e % 10 == 0:
            print(f"Episode {e}/{episodes}, Epsilon: {agent.epsilon:.2f}")

    return agent