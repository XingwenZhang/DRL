import tensorflow as tf


# max iterations for training (-1 means forever)
max_iterations = 50000

# discounted factor
discounted_factor = 0.99

# learning rate
lr = 1e-3

# decay
decay = 0.99

# epsilon greedy exploration
initial_exploration = 0.5
final_exploration = 0.0
final_exploration_frame = 1000
exploration_change_rate = (final_exploration - initial_exploration)/final_exploration_frame

# number of history frames to be fed into network
num_history_frames = -1 # not used in PG

# frame_size
frame_size = 6400 # input image is already flattened 

# num_hid, number of hidden units
num_hid = 150

# max_step, max number of step
max_step = 40

# batch_size
batch_size = 2

# weight initializer
weight_initializer = tf.truncated_normal

# frame-skipping interval
frame_skip_interval = 1 # no frame skip

# experience replay memory size
replay_memory = -1 # no replay memory

# number of random actions to take (to fill replay memory) before learning starts
replay_start_size = 0 # no replay memory

# the frequency with which the target network is updated
target_network_update_freq = 10000

# summary folder for tensor board
summary_folder = './logs/'
