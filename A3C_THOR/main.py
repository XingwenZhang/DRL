import argparse

import sys
sys.path.append('../THOR')
from THOREnv import THOREnvironment
from A3CAgent import A3CAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='running mode (default to train)')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='whether to use GPU (default to False)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='the id of the GPU to be used (default to 0)')
    parser.add_argument('--model_save_path', type=str, default='./models/a3c_model.ckpt',
                        help='path to save/load the model for training/testing (default to models/a3c_model.ckpt)')
    parser.add_argument('--check_point', type=int, default=None,
                        help='index of the ckeck point (default to None)')
    parser.add_argument('--model_save_freq', type=int, default=5000,
                        help='dump model at every k-th iteration (default to 5000)')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='number of threads to use in A3C algorithm (default to 4)')
    parser.add_argument('--feature_mode', type=bool, default=True,
                        help='whether or not to use extrated feature (default to True)')
    args = parser.parse_args()

    is_test_mode = (args.mode == 'test')
    THOREnvironment.pre_load(feat_mode=True, load_img_force=is_test_mode)
    env = THOREnvironment(feat_mode=True)

    if args.mode == 'train':
        agent = A3CAgent(num_threads = args.num_threads,
                         model_save_frequency=args.model_save_freq, model_save_path=args.model_save_path,
                         feature_mode = args.feature_mode)
        assert(args.model_save_path is not None)
        agent.learn(check_point = args.check_point, use_gpu=args.use_gpu, gpu_id=args.gpu_id)
    else:
        # disable frame skipping during testing result in better performance (because the agent can take more actions)
        assert(args.check_point is not None)
        agent = A3CAgent(num_threads = 1,
                         model_save_frequency=args.model_save_freq, model_save_path=args.model_save_path,
                         check_point=args.check_point, feature_mode = args.feature_mode,
                         use_gpu=args.use_gpu, gpu_id=args.gpu_id)
        agent.test(model_save_path = args.model_save_path)

    print('finished.')
