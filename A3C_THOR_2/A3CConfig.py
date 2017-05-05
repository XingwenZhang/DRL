import tensorflow as tf

# max iterations for training (-1 means forever)
max_iterations = 10000000

# discounted factor
discounted_factor = 0.99

# learning rate
learning_rate = 1e-5
decay_rate = 0.99
decay_step = 100000
momentum = 0.8

# decay
decay = 0.99

# epsilon greedy exploration
initial_exploration = 1.0
final_exploration = 0.1
final_exploration_frame = 200000
exploration_change_rate = (final_exploration - initial_exploration)/final_exploration_frame
test_exploration = 0.0

# max_step, max number of step
max_steps = 50

# batch_size
batch_size = 1

# weight initializer
weight_initializer = tf.truncated_normal

# number of history frame
num_history_frames = 4

# frame-skipping interval
frame_skip_interval = 4

# experience replay memory size
replay_memory = -1 # no replay memory

# number of random actions to take (to fill replay memory) before learning starts
replay_start_size = 0 # no replay memory

# summary folder for tensor board
summary_folder = './logs/'

# resnet pretrain model path
resnet_meta_graph = '../Resnet/ResNet-L50.meta'
resnet_pretrain_model = '../Resnet/ResNet-L50.ckpt'
