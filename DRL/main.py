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
    parser.add_argument('--double_dqn', type=bool, default=False,
                        help='whether to use double DQN algorithm (default to False)')
    parser.add_argument('--dueling_dqn', type=bool, default=False,
                        help='whether to use dueling DQN architecture (default to False)')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='path to save the model for training (default to None)')
    parser.add_argument('--model_load_path', type=str, default=None,
                        help='path to load the model for training/testing (default to None)')
    parser.add_argument('--model_save_freq', type=int, default=10000,
                        help='dump model at every k-th iteration (default to 10000)')
    parser.add_argument('--display', type=bool, default=False,
                        help='whether to render to result. (default to False)')
    args = parser.parse_args()

    if args.mode == 'train':
        env = DQNEnvironment(environment_name=args.gym_environment, display=args.display)
        agent = DQNAgent(env)
        assert(args.model_save_path is not None)
        agent.learn(double_dqn=args.double_dqn, dueling_dqn=args.dueling_dqn,
                    model_save_frequency=args.model_save_freq, model_save_path=args.model_save_path, model_load_path = args.model_load_path,
                    use_gpu=args.use_gpu, gpu_id=args.gpu_id)
    else:
        # disable frame skipping during testing result in better performance (because the agent can take more actions)
        env = DQNEnvironment(environment_name=args.gym_environment, display=args.display, frame_skipping=False)
        agent = DQNAgent(env)
        assert(args.model_load_path is not None)
        agent.test(dueling_dqn=args.dueling_dqn, 
                   model_load_path=args.model_load_path,
                   use_gpu=args.use_gpu, gpu_id=args.gpu_id)

    print('finished.')
