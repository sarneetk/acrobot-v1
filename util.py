import matplotlib.pyplot as plt

def plot_graph(episodes, rewards, algorithm):
    plt.plot(rewards, episodes)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title(algorithm + 'Plot')
    plt.show()