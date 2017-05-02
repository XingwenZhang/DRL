import argparse

import sys
sys.path.append('../THOR')
from THOREnv import THOREnvironment
from DQNAgent import DQNAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='running mode (default to train)')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='whether to use GPU (default to True)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='the id of the GPU to be used (default to 0)')
    parser.add_argument('--double_dqn', type=bool, default=True,
                        help='whether to use double DQN algorithm (default to True)')
    parser.add_argument('--dueling_dqn', type=bool, default=True,
                        help='whether to use dueling DQN architecture (default to True)')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='path to save the model for training (default to None)')
    parser.add_argument('--model_load_path', type=str, default=None,
                        help='path to load the model for training/testing (default to None)')
    parser.add_argument('--summary_folder', type=str, default='./logs',
                        help='folder to dump tensorflow summary for visualization in tensorboard. (default to ./logs)')
    parser.add_argument('--model_save_freq', type=int, default=10000,
                        help='dump model at every k-th iteration (default to 10000)')
    parser.add_argument('--display', type=bool, default=False,
                        help='whether to render to result. (default to False)')
    args = parser.parse_args()

    feat_mode = True
    THOREnvironment.pre_load(feat_mode=feat_mode)
    env = THOREnvironment(feat_mode=feat_mode)
    agent = DQNAgent(env)

    if args.mode == 'train':
        assert(args.model_save_path is not None)
        agent.learn(double_dqn=args.double_dqn, dueling_dqn=args.dueling_dqn,
                    model_save_frequency=args.model_save_freq, model_save_path=args.model_save_path, model_load_path = args.model_load_path,
                    use_gpu=args.use_gpu, gpu_id=args.gpu_id,
                    summary_folder=args.summary_folder)
    else:
        assert(args.model_load_path is not None)
        agent.test(dueling_dqn=args.dueling_dqn, 
                   model_load_path=args.model_load_path,
                   use_gpu=args.use_gpu, gpu_id=args.gpu_id,
                   summary_folder=args.summary_folder)
    print('finished.')
