from train import ModelManager
from render import RenderManager
from Qtrain import QLearning
from A2C import A2C_Learning
from A3C import A3C_Learning

import os,sys,argparse

def FFN_Reinforcement (parameters):
    mode = parameters['mode']
    env_name = 'CartPole-v1' if (parameters['env'] == 'cartpole') else 'Acrobot-v1'
    if mode == 'train_only':
        print("TRAINING ONLY")
        learningRate = 0.001
        modelMgr = ModelManager(env_name, learningRate)
        episodes= parameters['n_episodes']
        modelMgr.train(n_episodes=episodes)
    elif mode == 'train_and_render':
        print(" TRAINING AND RENDERING ")
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
            rendermgr.play_agent(env_name)
    else:
        print('no-op: unknown mode {}'.format(mode))

def QLearn_Reinforcement(parameters):
    mode = parameters['mode']
    env_name = 'CartPole-v1' if (parameters['env'] == 'cartpole') else 'Acrobot-v1'
    if mode == 'train_and_render':
        QLearn=QLearning(env_name)
        QLearn.train()
        QLearn.play_agent()
    else:
        print('no-op: unknown mode {}'.format(mode))


def A2C_Reinforcement(parameters):
    mode = parameters['mode']
    env_name = 'CartPole-v1' if (parameters['env'] == 'cartpole') else 'Acrobot-v1'
    A2C = A2C_Learning(env_name)
    render=True
    A2C.play_agent(render)

def A3C_Reinforcement(parameters):
    mode = parameters['mode']
    env_name = 'CartPole-v1' if (parameters['env'] == 'cartpole') else 'Acrobot-v1'
    # env_name = 'CartPole-v1'
    A3C = A3C_Learning(env_name)
    A3C.A3C_process()
    print("Finished A3C")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True,choices={'FFN','Q', 'A2C','A3C'})
    parser.add_argument('--env', required=True,choices={'acrobot','cartpole'})

    parser.add_argument('--mode', required=False, choices={'train_only','train_and_render', 'render_only'})
    parser.add_argument('--n_episodes', type=int, default=10000)
    par__main = vars(parser.parse_args(sys.argv[1:]))
    model = par__main['models']
    if(model=='FFN'):
        FFN_Reinforcement(par__main)
    elif(model=='Q'):
        QLearn_Reinforcement(par__main)
    elif(model=='A2C'):

        A2C_Reinforcement(par__main)
    elif (model == 'A3C'):
        A3C_Reinforcement(par__main)
    else:
        print('no-op: unknown model {} {}'.format(model))




