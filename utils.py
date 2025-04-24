import matplotlib.pyplot as plt
import torch


def plot_rewards(rewards):
    plt.plot(rewards)
    plt.title('Reward pro Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model