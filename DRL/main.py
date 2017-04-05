import argparse

from DQNEnv import DQNEnvironment
from DQNAgent import DQNAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym_environment', type=str, default='SpaceInvaders-v0',
                        help='OpenAI Gym Environment to be used (default to SpaceInvaders-v0)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='running mode (default to train)')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='whether to use GPU (default to True)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='the id of the GPU to be used (default to 0)')
    parser.add_argument('--model_dump_path', type=str, default='model/DQN_model.ckpt',
                        help='path to save/load the model for training/testing (default to model/DQN_model.ckpt)')
    parser.add_argument('--model_save_freq', type=int, default=10000,
                        help='dump model at every k-th iteration (default to 10000)')
    parser.add_argument('--display', type=bool, default=False,
                        help='whether to render to result. (default to False)')
    args = parser.parse_args()

    env = DQNEnvironment(environment_name=args.gym_environment, display=args.display)
    agent = DQNAgent(env)

    if args.mode == 'train':
        agent.learn(model_save_frequency=args.model_save_freq, model_save_path=args.model_dump_path,
                    use_gpu=args.use_gpu, gpu_id=args.gpu_id)
    else:
        agent.test(model_load_path=args.model_path)

    print('finished.')
