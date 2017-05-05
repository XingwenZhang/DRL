import argparse

import sys
sys.path.append('../THOR')
from THOREnv import THOREnvironment
from A3CManager import A3CManager

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
    parser.add_argument('--model_save_interval', type=int, default=50000,
                        help='dump model at every k-th iteration (default to 50000)')
    parser.add_argument('--num_agents', type=int, default=4,
                        help='number of agents to use in A3C algorithm (default to 4)')
    parser.add_argument('--feature_mode', type=bool, default=True,
                        help='whether or not to use extrated feature (default to True)')
    args = parser.parse_args()

    is_test_mode = (args.mode == 'test')
    THOREnvironment.pre_load(feat_mode=True, load_img_force=is_test_mode)
    env = THOREnvironment(feat_mode=True)

    if args.mode == 'train':
        mgr = A3CManager(num_agents = args.num_agents,
                         model_save_interval=args.model_save_interval, model_save_path=args.model_save_path, 
                         feature_mode = args.feature_mode)
        assert(args.model_save_path is not None)
        mgr.learn(use_gpu=args.use_gpu, gpu_id=args.gpu_id, check_point = args.check_point)
    else:
        # disable frame skipping during testing result in better performance (because the agent can take more actions)
        assert(args.check_point is not None)
        mgr = A3CManager(num_agents = 1,
                         model_save_intervaluency=args.model_save_interval, model_save_path=args.model_save_path,
                         feature_mode = args.feature_mode)
        mgr.test(use_gpu=args.use_gpu, gpu_id=args.gpu_id, check_point = args.check_point)

    print('finished.')
