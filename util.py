import matplotlib.pyplot as plt

def plot_graph(episodes, rewards, algorithm,Addln_info):
    plt.plot(episodes, rewards)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title(algorithm + '_Plot')
    path='assets/'+algorithm+Addln_info+'.jpg'
    plt.savefig(path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()