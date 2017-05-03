import tensorflow as tf

# feature-length
feature_length = 2048  #resnet

# max iterations for training (-1 means forever)
max_iterations = 5000000

# discounted factor
discounted_factor = 0.99

# learning rate
lr = 0.00025

# epsilon greedy exploration
initial_exploration = 1
final_exploration = 0.1
final_exploration_frame = 1000000
exploration_change_rate = (final_exploration - initial_exploration)/final_exploration_frame
test_exploration = 0.1

# random start during training 
random_start = 30

# number of history frames to be fed into Q network
num_history_frames = 4

# frame_size
frame_size = (84, 84)

# batch_size
batch_size = 32

# weight initializer
weight_initializer = tf.truncated_normal

# frame-skipping interval
frame_skip_interval = 4

# experience reply memory size
replay_memory = 1000000 

# number of random actions to take (to fill replay memory) before learning starts
replay_start_size = 50000

# the frequency with which the target network is updated
target_network_update_freq = 10000

# hyper-parameters recommended by DRL course
lr = 0.0001

# clip the reward to [-1, 1]
reward_clipping = True

# prioritized experience replay setting
prioritized_experience_replay = False
prioritized_experience_replay_alpha = 0.7
prioritized_experience_replay_beta = 0.5

# debug options
fix_target_image = True
fix_target_image_idx = 59
