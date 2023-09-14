import gym
import numpy as np
import pandas as pd
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim

class RecommenderEnv(gym.Env):
    def __init__(self, data, n_items, n_users):
        super(RecommenderEnv, self).__init__()

        self.data = data
        self.n_items = n_items
        self.n_users = n_users

        self.action_space = spaces.Discrete(n_items)
        self.observation_space = spaces.Discrete(n_users)

        self.reset()

    def step(self, action):
        user = self.current_user
        reward = self.data.loc[user, action]
        done = True
        self.reset()

        return self.current_user, reward, done, {}

    def reset(self):
        self.current_user = np.random.randint(self.n_users)
        return self.current_user
    
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def epsilon_greedy_action(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(q_values.shape[-1])
    else:
        return torch.argmax(torch.tensor(q_values)).item()

# Create a toy dataset
data = pd.DataFrame(np.random.randint(0, 2, (5, 5)))
n_items = len(data.columns)
n_users = len(data)

# Initialize the environment
env = RecommenderEnv(data, n_items, n_users)

# Q-learning parameters
n_episodes = 1000
learning_rate = 0.01
gamma = 0.99
epsilon = 0.1

# One-hot encoding for state representation
def one_hot(user_index, n_users):
    one_hot_vector = np.zeros(n_users)
    one_hot_vector[user_index] = 1
    return one_hot_vector

# Create and initialize the Q-network
q_net = QNetwork(n_users, n_items)
optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
for episode in range(n_episodes):
    state = env.reset()
    done = False

    while not done:
        state_array = np.array([one_hot(state, n_users)])  # Convert list to numpy array
        state_tensor = torch.FloatTensor(state_array)  # Create tensor from numpy array
        q_values = q_net(state_tensor)
        action = epsilon_greedy_action(q_values.detach().numpy(), epsilon)

        next_state, reward, done, _ = env.step(action)

        next_state_array = np.array([one_hot(next_state, n_users)])  # Convert list to numpy array
        next_state_tensor = torch.FloatTensor(next_state_array)  # Create tensor from numpy array
        next_q_values = q_net(next_state_tensor)

        target_q_value = reward + gamma * torch.max(next_q_values).item()

        loss = criterion(q_values[0, action], torch.tensor(target_q_value, dtype=torch.float))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state