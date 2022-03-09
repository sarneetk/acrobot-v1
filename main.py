from train import ModelManager
from render import RenderManager
import time
import os,sys,argparse

def main (parameters):
    mode = parameters['mode']
    if mode == 'train_and_render':

        env_name = 'Acrobot-v1'
        learningRate = 0.001
        modelMgr = ModelManager(env_name, learningRate)
        episodes= parameters['n_episodes']
        modelMgr.train(n_episodes=episodes)
        print(" RENDERING ACROBOT-V1 ")
        #time.sleep(0.1)
        rendermgr = RenderManager()
        rendermgr.play_agent()
    elif mode == 'render_only':
        if not os.path.exists('checkpoint.pth'):
            print('no-op: check points does not exist. use --mode "train_and_render"')
        else:
            rendermgr=RenderManager()
            rendermgr.play_agent()
    else:
        print('no-op: unknown mode {}'.format(mode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices={'train_and_render', 'render_only'})
    parser.add_argument('--n_episodes', type=int, default=10000)
    par__main = vars(parser.parse_args(sys.argv[1:]))
    main(par__main)




