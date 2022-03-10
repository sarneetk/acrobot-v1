import gym
import time
from policy import Policy
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

class RenderManager:
    def __init__(self,env_name):
        torch.manual_seed(0) # set random seed

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        state_dict = torch.load('checkpoint.pth')

        self.policy = Policy()
        self.policy.load_state_dict(state_dict)
        self.policy = self.policy.to(device)
        self.env = gym.make(env_name)
        self.env.seed(0)

    def play_agent(self):

        recorder = VideoRecorder(self.env, path='./video.mp4', enabled=True)
        state=self.env.reset()
        for i_episode in range(20):
            observation = self.env.reset()
            for t in range(1000):
                recorder.capture_frame()
                action, _ = self.policy.act(state)
                self.env.render()
                state, reward, done, _ = self.env.step(action)

                if done:
                    break
                time.sleep(0.1)
        self.env.close()


