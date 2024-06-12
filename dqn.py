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

class Agent:
    def __init__(self, state_size, action_size):
        self.neurons = 64
        self.action_size = action_size
        self.memory = deque(maxlen=8000)
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.00001
        self.model = DQN(state_size, self.neurons, action_size)
        self.target_model = DQN(state_size, self.neurons, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Using Huber loss
        self.prepopulate_memory = 1000

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.choice([a for a in range(self.action_size) if state[a] > 0])        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        # Mask invalid actions
        invalid_actions = [a for a in range(self.action_size) if state[a] <= 0]
        q_values[0][invalid_actions] = float('-inf')
        return torch.argmax(q_values, dim=1).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
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
    init_score = env.board[player][mancala.BANK]
    info = env.game_step(player, action, verbose=False)
    reward = 0

    if info['capture']:
        reward += 10
    if info['bonus_round']:
        reward += 5 

    current_score = env.board[player][mancala.BANK]
    reward += current_score - init_score

    opponent = env.switch_side(player)
    opponent_score = env.board[opponent][mancala.BANK]
    reward += current_score - opponent_score

    if info['game_over']:
        if env.get_winner() == player:
            reward += 50
        else:
            reward -= 50 
    return get_state(env), reward, info['game_over'], info['bonus_round']

def run_episode(agent, opponent, env, batch_size):
    state = get_state(env)
    loss = -1
    total_reward = 0
    game_over = False
    current_player = 1
    while not game_over:
        if current_player == 1:
            action = agent.act(state)
            next_state, reward, game_over, bonus_round = step(env, current_player, action)
            agent.remember(state, action, reward, next_state, game_over)
            total_reward += reward
            if len(agent.memory) > agent.prepopulate_memory:
                loss = agent.replay(batch_size)
        else:
            action = opponent.act(env, current_player)
            next_state, _, game_over, bonus_round = step(env, current_player, action)
        if not bonus_round:
            current_player = 2 if current_player == 1 else 1
        state = next_state
    return loss, total_reward

def plot_history(history):
    # Calculate the averages for every 5 episodes
    avg_loss = []
    avg_reward = []
    
    for i in range(0, len(history), 5):
        batch = history[i:i+5]
        avg_loss.append(sum([l[0] for l in batch]) / len(batch))
        avg_reward.append(sum([l[1] for l in batch]) / len(batch))
    
    # Plot the averaged history
    fig, axs = plt.subplots(2, figsize=(10, 10))
    
    axs[0].plot(range(0, len(avg_loss) * 5, 5), avg_loss)
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('DQN Training Loss (Averaged)')

    axs[1].plot(range(0, len(avg_reward) * 5, 5), avg_reward)
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Reward')
    axs[1].set_title('DQN Rewards (Averaged)')

    plt.tight_layout()
    plt.show()

def train_dqn(episodes=800, batch_size=64, opponent_types=(player.Random('random'), player.Greedy('greedy')), verbose=True):
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

    while len(agent.memory) < agent.prepopulate_memory:
        opponent = random.choice(opponent_types)
        env = mancala.Game({1: player.DQN('dqn', agent), 2: opponent})
        run_episode(agent, opponent, env, batch_size)

    for e in range(episodes):
        opponent = random.choice(opponent_types)
        env = mancala.Game({1: player.DQN('dqn', agent), 2: opponent})
        loss, reward = run_episode(agent, opponent, env, batch_size)
        if e % 1 == 0: agent.update_target_model()

        if loss != -1: history.append((loss, reward))
        if verbose: print(f"Episode {e}, Memory: {len(agent.memory)}, Epsilon: {agent.epsilon:.2f}, Loss: {loss:.2f}, Reward: {reward}")

    plot_history(history)
    return agent