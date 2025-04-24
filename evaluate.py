# evaluate.py
import gymnasium as gym
import torch
from dqn import DQN
from utils import load_model

# ==== Konfiguration ====
ENV_NAME    = 'CartPole-v1'
MODEL_PATH  = 'policy_net.pth'   # Pfad zu deinem gespeicherten Modell
EPISODES    = 5                  # Wie viele Durchläufe du sehen willst

def evaluate():
    # Environment mit Fenster-Modus
    env = gym.make(ENV_NAME, render_mode='human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Agent laden
    agent = DQN(obs_dim, act_dim)
    load_model(agent, MODEL_PATH)

    for ep in range(1, EPISODES+1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()  # Fenster aktualisieren
            # Greedy-Aktion (keine Exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = agent(state_tensor).argmax(dim=1).item()

            # ENV-Schritt
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"Episode {ep} – Score: {total_reward}")
    env.close()

if __name__ == '__main__':
    evaluate()
