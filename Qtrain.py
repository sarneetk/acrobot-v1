import gym
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
import time
import matplotlib.pyplot as plt
from util import plot_graph

from IPython import display
torch.manual_seed(0) # set random seed

class QLearning:
    def __init__(self, env_name='Acrobot-v1', learningRate=0.001):
        self.env = gym.make(env_name)
        self.env.seed(0)
        print('observation space:', self.env.observation_space)
        print('action space:', self.env.action_space)
        self.action_size = self.env.action_space.n
        # discretize the obs space into buckets
        buckets = 15
        discrete_obs_size = [buckets] * len(self.env.observation_space.low)
        self.bucket_size = (self.env.observation_space.high - self.env.observation_space.low) / discrete_obs_size

        # create q table to match all possible states with actions
        self.q_table = np.random.uniform(high=3, low=0, size=(discrete_obs_size + [self.env.action_space.n]))

        #Set hyper Parameters:
        # set hyperparameters
        self.lr = 0.1
        self.discount_rate = 0.95
        self.episodes = 5000
        self.discount_factor = 0.9

        # exploration vs exploitation factor
        self.eps = 1
        self.eps_start_decay = 1
        self.eps_end_decay = 3 * self.episodes // 4
        self.eps_decay_rate = self.eps / (self.eps_end_decay - self.eps_start_decay)

    # used to make continuous state outputs discrete
    def discretize(self,state, bucket_size, max_size=14):
        discrete_state = (state - self.env.observation_space.low) / bucket_size

        # clip to make sure it the state format never exceeds the bucket size
        discrete_state = np.clip(discrete_state.astype(np.int), None, max_size)

        return tuple(discrete_state)

    def train(self, n_episodes=5000, max_t=1000, gamma=1.0, print_every=100):
        # reset the environment
        # learning loop
        rewards_per_episode = []
        ave_reward_list= {}
        for i in range(self.episodes):
            state = self.discretize(self.env.reset(), self.bucket_size)
            done = False
            rewards = []
            total_reward=0

            # run an episode until completion
            while not done:

                if np.random.uniform(0, 1) < self.eps:
                    # get random action for exploration
                    action = self.env.action_space.sample()
                else:
                    # get action based on current q_table
                    action = np.argmax(self.q_table[state])

                # take action and recieve new state
                new_state, reward, done, _ = self.env.step(action)
                new_state = self.discretize(new_state, self.bucket_size)
                rewards.append(reward)
                total_reward+=reward
                if not done:
                    # update q table
                    self.q_table[state][action] = self.q_table[state][action] + self.lr * (
                                reward + self.discount_factor * np.max(self.q_table[new_state]) - self.q_table[state][action])
                elif done == True:
                    self.q_table[state][action] = reward

                    # update state
                state = new_state
            rewards_per_episode.append(sum(rewards))
            # slowly decrease exploration factor
            if self.eps_start_decay <= i < self.eps_end_decay:
                self.eps -= self.eps_decay_rate

            if i % 100 == 0:
                ave_reward = np.mean(rewards_per_episode)
                print("Episodes: {}/{}     Epsilon:{}  Ave. Reward:{}".format(i, self.episodes, self.eps,ave_reward))
                ave_reward_list[i]=ave_reward
                rewards_per_episode=[]

        Addl_info = '_' + str(self.episodes)+'_lr'+str(self.lr)+'_r'+str(self.discount_rate)
        plot_graph(ave_reward_list.keys(), ave_reward_list.values(), 'Q_Learning',Addl_info)

        self.env.close()

    def  play_agent(self):
        # test environment with q table
        recorder = VideoRecorder(self.env, path='assets/Acrobot_Q.mp4', enabled=True)
        state = self.discretize(self.env.reset(), self.bucket_size)
        done = False
        # img = plt.imshow(self.env.render(mode='rgb_array'))
        while not done:
            recorder.capture_frame()
            self.env.render()
            #img.set_data(self.env.render(mode='rgb_array'))
            #display.display(plt.gcf())
            #display.clear_output(wait=True)
            action = np.argmax(self.q_table[state])
            new_state, reward, done, _ = self.env.step(action)
            state = self.discretize(new_state, self.bucket_size)
            time.sleep(0.1)
        self.env.close()

