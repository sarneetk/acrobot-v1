import gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from util import plot_graph
import torch
torch.manual_seed(0) # set random seed
#import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
#from torch.distributions import Categorical
from policy import Policy
#from gym.wrappers.monitoring.video_recorder import VideoRecorder


class ModelManager:
    def __init__(self,env_name='Acrobot-v1', learningRate=0.001):
        self.env = gym.make(env_name)
        self.env.seed(0)
        print('observation space:', self.env.observation_space)
        print('action space:', self.env.action_space)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy = Policy().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learningRate)

    def train(self,n_episodes=5000, max_t=1000, gamma=1.0, print_every=100):
        scores_deque = deque(maxlen=100)
        scores = []
        output = open('checkpoint.pth', mode="wb")
        rewards_per_episode = {}
        for i_episode in range(1, n_episodes+1):
            saved_log_probs = []
            rewards = []
            state = self.env.reset()
            for t in range(max_t):
                action, log_prob = self.policy.act(state)
                saved_log_probs.append(log_prob)
                state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                if done:
                    break
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))

            discounts = [gamma**i for i in range(len(rewards)+1)]
            R = sum([a*b for a,b in zip(discounts, rewards)])

            policy_loss = []
            for log_prob in saved_log_probs:
                policy_loss.append(-log_prob * R)
            policy_loss = torch.cat(policy_loss).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            if i_episode % print_every == 0:
                torch.save(self.policy.state_dict(), output)
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                rewards_per_episode[i_episode] = sum(rewards)
                # print(rewards_per_episode)

        plot_graph(rewards_per_episode.keys(), rewards_per_episode.values(), 'FFN')
        self.env.close()
        output.close()
        return scores








