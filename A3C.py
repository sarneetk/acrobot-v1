import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from util import plot_graph

# Hyperparameters
n_train_processes = 3
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_ep = 1000
max_test_ep = 10


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v





class A3C_Learning:
    def __init__(self,env_name):

        self.global_model = ActorCritic()
        self.global_model.share_memory()
        self.env_name=env_name

    def A3C_process(self):
        processes = []
        for rank in range(n_train_processes + 1):  # + 1 for test process
            if rank == 0:
                p = mp.Process(target=self.test, args=(self.global_model,self.env_name))
            else:
                p = mp.Process(target=self.train, args=(self.global_model, rank,self.env_name))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def train(self,global_model, rank,env_name):
        local_model = ActorCritic()
        local_model.load_state_dict(global_model.state_dict())

        optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

        #env = gym.make('CartPole-v1')
        env = gym.make(env_name)
        rewards_per_episode = []
        ave_reward_list = {}
        for n_epi in range(max_train_ep):
            done = False
            s = env.reset()
            while not done:
                s_lst, a_lst, r_lst = [], [], []
                for t in range(update_interval):
                    prob = local_model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done, info = env.step(a)


                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append(r/100.0)

                    s = s_prime
                    if done:
                        break

                s_final = torch.tensor(s_prime, dtype=torch.float)
                R = 0.0 if done else local_model.v(s_final).item()
                td_target_lst = []
                for reward in r_lst[::-1]:
                    R = gamma * R + reward
                    td_target_lst.append([R])
                td_target_lst.reverse()

                s_batch, a_batch, td_target = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                    torch.tensor(td_target_lst)
                advantage = td_target - local_model.v(s_batch)

                pi = local_model.pi(s_batch, softmax_dim=1)
                pi_a = pi.gather(1, a_batch)
                loss = -torch.log(pi_a) * advantage.detach() + \
                    F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())

                optimizer.zero_grad()
                loss.mean().backward()
                for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                local_model.load_state_dict(global_model.state_dict())
            rewards_per_episode.append( sum(r_lst)*10000)
            # print(rewards_per_episode)
            if n_epi % 100 == 0:
                ave_reward = np.mean(rewards_per_episode)
                #print("Episodes: {}/{}     Epsilon:{}  Ave. Reward:{}".format(n_epi, self.episodes, self.eps, ave_reward))
                ave_reward_list[n_epi] = ave_reward
                rewards_per_episode = []

        env.close()
        print("Training process {} reached maximum episode.".format(rank))
        Addl_info='_'+str(max_train_ep)
        plot_graph(ave_reward_list.keys(), ave_reward_list.values(),
                   'A3C_Learning', Addl_info)
        #plot_graph(rewards_per_episode.keys(), rewards_per_episode.values(), 'cartpole  ')


    def test(self,global_model,env_name):
        #env = gym.make('CartPole-v1')
        env = gym.make(env_name)
        score = 0.0
        print_interval = 20
        print("Started Testing")
        recorder = VideoRecorder(env, path='assets/Acrobot_A3C.mp4', enabled=True)
        for n_epi in range(max_test_ep):
            done = False
            s = env.reset()
            while not done:
                recorder.capture_frame()
                env.render()
                prob = global_model.pi(torch.from_numpy(s).float())
                a = Categorical(prob).sample().item()

                s_prime, r, done, info = env.step(a)
                s = s_prime
                score += r

            if n_epi % print_interval == 0 and n_epi != 0:
                print("# of episode :{}, avg score : {:.1f}".format(
                    n_epi, score/print_interval))
                score = 0.0
                time.sleep(1)
        env.close()