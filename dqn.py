import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import mancala
import player

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(self.model(state), dim=1).item()

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
            total_loss += loss
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss

def get_state(env):
    return env.board[1] + env.board[2]

def step(env, player, action):
    if env.is_slot_empty(player, action):
        return get_state(env), -10, False, False  # Penalize invalid moves
    info = env.game_step(player, action, verbose=False)
    reward = 0
    if info['capture']:
        reward += 10
    if info['bonus_round']:
        reward += 10
    if info['game_over']:
        if env.get_winner() == player:
            reward += 100
        else:
            reward -= 100
    return get_state(env), reward, info['game_over'], info['bonus_round']

def train_dqn(episodes=10, batch_size=32, reward_type='', opponent_types=(player.Random('random'), player.Greedy('greedy')), verbose=True):
    """
    Train the DQN agent.

    Args:
        episodes (int): Number of episodes to train.
        batch_size (int): Size of the minibatch for training.

    Returns:
        DQNAgent: Trained DQN agent.
    """
    if verbose: print('Started training DQN player')
    state_size = (mancala.BANK + 1) * 2
    action_size = mancala.BANK
    agent = Agent(state_size, action_size)

    for e in range(episodes):
        opponent = random.choice(opponent_types)
        env = mancala.Game({1: player.DQN('dqn', agent), 2: opponent})
        state = get_state(env)
        game_over = False
        loss = -1
        while not game_over:
            player_side = 1
            while player_side < 3:
                action = agent.act(state)
                next_state, reward, game_over, bonus_round = step(env, player_side, action)
                agent.remember(state, action, reward, next_state, game_over)
                state = next_state
                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)
                if bonus_round or reward == -10:
                    player_side -= 1
                player_side += 1
                if game_over:
                    break
        agent.update_target_model()
        if verbose: print(f"Episode {e}, Epsilon: {agent.epsilon:.2f}, Loss: {loss:.2f}")
    return agent