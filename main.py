from train import ModelManager
from render import RenderManager
from Qtrain import QLearning
import os,sys,argparse

def FFN_Reinforcement (parameters):
    mode = parameters['mode']
    env_name = 'Acrobot-v1'
    if mode == 'train_only':
        print("TRAINING ONLY")
        #env_name = 'Acrobot-v1'
        learningRate = 0.001
        modelMgr = ModelManager(env_name, learningRate)
        episodes= parameters['n_episodes']
        modelMgr.train(n_episodes=episodes)
    elif mode == 'train_and_render':
        print(" TRAINING AND RENDERING ")
        #env_name = 'Acrobot-v1'
        learningRate = 0.001
        modelMgr = ModelManager(env_name, learningRate)
        episodes = parameters['n_episodes']
        modelMgr.train(n_episodes=episodes)
        print(" RENDERING ACROBOT-V1 ")
        rendermgr = RenderManager(env_name)
        rendermgr.play_agent()
    elif mode == 'render_only':
        print(" RENDERING ONLY ")
        if not os.path.exists('checkpoint.pth'):
            print('no-op: check points does not exist. use --mode "train_and_render"')
        else:
            rendermgr=RenderManager(env_name)
            rendermgr.play_agent()
    else:
        print('no-op: unknown mode {}'.format(mode))

def QLearn_Reinforcement(parameters):
    mode = parameters['mode']
    env_name = 'Acrobot-v1'
    if mode == 'train_and_render':
        QLearn=QLearning(env_name)
        QLearn.train()
        QLearn.play_agent()
    else:
        print('no-op: unknown mode {}'.format(mode))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True,choices={'FFN','Q'})
    parser.add_argument('--mode', required=True, choices={'train_only','train_and_render', 'render_only'})
    parser.add_argument('--n_episodes', type=int, default=10000)
    par__main = vars(parser.parse_args(sys.argv[1:]))
    model = par__main['models']
    if(model=='FFN'):
        FFN_Reinforcement(par__main)
    elif(model=='Q'):
        QLearn_Reinforcement(par__main)
    else:
        print('no-op: unknown mode {} {}'.format(model))




