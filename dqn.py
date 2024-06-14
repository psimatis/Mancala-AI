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
    def __init__(self, epsilon=1.00, epsilon_min=0.01, decay=0.99):
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

class Agent:
    def __init__(self, epsilon_min=0.01, epsilon_decay=0.99, batch_size=512, capacity=10000, gamma=0.9, learning_rate=0.0001, neurons=32):
        self.state_size = (mancala.STORE + 1) * 2
        self.action_size = mancala.STORE
        self.e_greedy = EGreedy(1, epsilon_min, epsilon_decay)
        self.memory = Memory(capacity, batch_size)
        self.gamma = gamma
        self.policy_model = DQN(self.state_size, neurons, self.action_size)
        self.target_model = DQN(self.state_size, neurons, self.action_size)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def act(self, state, side):
        state = state[7:] + state[:7] if side == 2 else state

        if self.e_greedy.explore():
            return random.choice([a for a in range(self.action_size) if state[a] > 0])

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.policy_model(state_tensor)
        invalid_actions = [a for a in range(self.action_size) if state[a] == 0]
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

    def step(self, env, player_side, action):
        init_score = env.board[player_side][mancala.STORE]
        info = env.game_step(player_side, action, verbose=False)

        current_score = env.board[player_side][mancala.STORE]
        opponent_score = env.board[env.switch_side(player_side)][mancala.STORE]
        reward = 2 * current_score - opponent_score - init_score

        if info["game_over"]:
            if env.get_winner() == player_side:
                reward += 10
            else:
                reward -= 10

        if info["capture"] or info["bonus_round"]:
            reward += 5
        return env.get_state(), reward, info["game_over"], info["bonus_round"]

    def run_episode(self, opponent_types):
        opponent = random.choice(opponent_types)
        env = mancala.Game({1: player.DQN("dqn", self), 2: opponent})
        state = env.get_state(env)
        info = {'avg_loss': 0, 'avg_reward': 0, 'steps': 0}
        game_over = False
        current_player = 1

        while not game_over:
            info['steps'] += 1
            if env.players[current_player].name == "dqn":
                action = self.act(state, current_player)
                next_state, reward, game_over, bonus_round = self.step(env, current_player, action)
                self.memory.remember(state, action, reward, next_state, game_over)
                info['avg_reward'] += reward
                if len(self.memory) > self.memory.batch_size:
                    info['avg_loss'] += self.replay()
            else:
                action = opponent.act(env, current_player)
                next_state, _, game_over, bonus_round = self.step(env, current_player, action)
            if not bonus_round:
                current_player = 2 if current_player == 1 else 1
            state = next_state
        info['avg_reward'] /= info['steps']
        info['avg_loss'] /= info['steps']
        return info

    def plot_history(self, history):
        _, axs = plt.subplots(2, figsize=(10, 10))
        axs[0].plot([l[0] for l in history])
        axs[0].set_xlabel("Episodes")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("DQN Training Loss")

        axs[1].plot([r[1] for r in history])
        axs[1].set_xlabel("Episodes")
        axs[1].set_ylabel("Reward")
        axs[1].set_title("DQN Rewards")
        plt.show()

    def train_dqn(self, opponents, episodes=1000, update_frequency=200, verbose=True):
        print('Training DQN agent')
        steps = 0
        history = []
        for e in range(episodes):
            info = self.run_episode(opponents)
            steps += info['steps']
            if e % update_frequency == 0:
                self.update_target_model()
            history.append((info['avg_loss'], info['avg_reward']))
            if verbose:
                print(f"Episode: {e} Steps: {steps} Epsilon: {self.e_greedy.epsilon:.2f} Loss: {info['avg_loss']:.2f} Reward: {info['avg_reward']:.2f}")
        self.plot_history(history)
        return self

if __name__ == "__main__":
    agent = Agent()
    player.DQN('dqn_random', agent.train_dqn(opponents=(player.Random(),)))
