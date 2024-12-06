import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from env import PendulumEnv


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)




def deep_q_learning(env, device, episodes=500, gamma=0.99, lr=1e-3, batch_size=64, buffer_capacity=10000):
    state_dim = env.observation_space[0]
    action_dim = env.action_space

    q_net = QNetwork(state_dim, action_dim).to(device)  # Move model to device
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    epsilon = 1.0
    epsilon_decay = 0.978
    epsilon_min = 0.01

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(
            0).to(device)
        total_reward = 0

        iter = 0
        max_iter = 5 * env.FPS
        while True:
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space)
            else:
                with torch.no_grad():
                    action = torch.argmax(q_net(state)).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(
                0).to(device)
            buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if done or iter > max_iter:
                break
            iter += 1

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states).to(device)
                actions = torch.tensor(actions).unsqueeze(1).to(device)
                rewards = torch.tensor(
                    rewards, dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.cat(next_states).to(device)
                dones = torch.tensor(
                    dones, dtype=torch.float32).unsqueeze(1).to(device)

                q_values = q_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q_values = target_net(
                        next_states).max(1, keepdim=True)[0]
                    target_q_values = rewards + gamma * \
                        max_next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        target_net.load_state_dict(q_net.state_dict())

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    if not env.do_render:
        env.do_render = True
        env.render_setup()

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    total_reward = 0

    q_net.eval()
    while True:
        with torch.no_grad():
            action = torch.argmax(q_net(state)).item()

        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        state = next_state
        total_reward += reward

        if done:
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            print(f"Total Reward = {total_reward}")
            total_reward = 0


if __name__ == "__main__":
    env = PendulumEnv(do_render=True)
    device = torch.device(
        "mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    deep_q_learning(env, device)
