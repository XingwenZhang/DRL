import argparse

from PGEnv import PGEnvironment
from PGAgent import PGAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym_environment', type=str, default='Pong-v0',
                        help='OpenAI Gym Environment to be used (default to Pong-v0)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='running mode (default to train)')
    parser.add_argument('--use_gpu', type=bool, default=False,
                        help='whether to use GPU (default to True)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='the id of the GPU to be used (default to 0)')
    parser.add_argument('--model_save_path', type=str, default='./model/PG_model.ckpt',
                        help='path to save/load the model for training/testing (default to model/PG_model.ckpt)')
    parser.add_argument('--check_point', type=int, default=None,
                        help='index of the ckeck point (default to None)')
    parser.add_argument('--model_save_freq', type=int, default=100,
                        help='dump model at every k-th iteration (default to 100)')
    parser.add_argument('--display', type=bool, default=False,
                        help='whether to render to result. (default to False)')
    args = parser.parse_args()

    if args.mode == 'train':
        env = PGEnvironment(environment_name=args.gym_environment, display=args.display)
        agent = PGAgent(env)
        assert(args.model_save_path is not None)
        agent.learn(model_save_frequency=args.model_save_freq, model_save_path=args.model_save_path, check_point = args.check_point,
                    use_gpu=args.use_gpu, gpu_id=args.gpu_id)
    else:
        # disable frame skipping during testing result in better performance (because the agent can take more actions)
        env = PGEnvironment(environment_name=args.gym_environment, display=args.display, frame_skipping=False)
        agent = PGAgent(env)
        assert(args.check_point is not None)
        agent.test(model_save_path = args.model_save_path, check_point=args.check_point,
                   use_gpu=args.use_gpu, gpu_id=args.gpu_id)

    print('finished.')
