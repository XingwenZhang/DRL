import tensorflow as tf


# max iterations for training (-1 means forever)
max_iterations = 18000000

# discounted factor
discounted_factor = 0.99

# learning rate
lr = 0.00025

# epsilon greedy exploration
initial_exploration = 1
final_exploration = 0.1
final_exploration_frame = 1000000
exploration_change_rate = (final_exploration - initial_exploration)/final_exploration_frame
test_exploration = 0.

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

# summary folder for tensor board
summary_folder = './logs/'
