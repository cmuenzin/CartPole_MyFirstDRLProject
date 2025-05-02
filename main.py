import gymnasium as gym
import torch
import numpy as np
from dqn import DQN, ReplayBuffer
from utils import plot_rewards, save_model, load_model

# Cart Pole DRL Project

# Hyperparameter
ENV_NAME = 'CartPole-v1'
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
LR = 1e-3
BATCH_SIZE = 64
BUFFER_CAPACITY = 10000
TARGET_UPDATE_FREQ = 10
MAX_EPISODES = 500
SOLVED_SCORE = 195


def main():
    # Environment mit Gymnasium
    env = gym.make(ENV_NAME)
    obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Netzwerke und Replay-Buffer
    policy_net = DQN(obs_dim, act_dim)
    target_net = DQN(obs_dim, act_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    epsilon = EPS_START
    rewards = []

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-Greedy Aktion
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.FloatTensor(state))
                    action = q_vals.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Training
            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards_batch, next_states, dones = buffer.sample(BATCH_SIZE)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards_t = torch.FloatTensor(rewards_batch).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones_t = torch.FloatTensor(dones).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    q_next = target_net(next_states).max(1)[0].unsqueeze(1)
                    q_target = rewards_t + GAMMA * q_next * (1 - dones_t)

                loss = torch.nn.MSELoss()(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Ende Episode
        rewards.append(total_reward)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Update Target-Netzwerk
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Ausgabe
        if episode % 20 == 0:
            avg = np.mean(rewards[-20:])
            print(f"Episode {episode:3d} | Ø Reward (letzte 20): {avg:.2f} | ε: {epsilon:.2f}")

        # Abbruch
        if len(rewards) >= 100 and np.mean(rewards[-100:]) >= SOLVED_SCORE:
            print(f"Gelöst in Episode {episode}!")
            break

    # Modelle speichern
    save_model(policy_net, 'policy_net.pth')
    save_model(target_net, 'target_net.pth')

    # Visualisierung
    plot_rewards(rewards)


if __name__ == '__main__':
    main()

