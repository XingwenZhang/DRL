# thor environment setting

import os
assert 'THOR_HOME' in os.environ, 'please first set env_var THOR_HOME as the absolute path of DRL/THOR'

# open display
display = False

# size of the input image to the network
net_input_width=224
net_input_height=224

THOR_HOME=os.environ['THOR_HOME']
darwin_build=THOR_HOME + '/thor_binary/thor-cmu-201703101557-OSXIntel64.app/Contents/MacOS/thor-cmu-201703101557-OSXIntel64'
linux_build=THOR_HOME + '/thor_binary/thor-cmu-201703101558-Linux64'
x_display="0.0"

# supported environments and actions (do not change the order of supported_actions)
supported_envs = ['FloorPlan224']
# supported_actions = ['MoveAhead', 'MoveBack'];
supported_actions = ['MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight']; 
# supported_actions = ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown']

# under what threshold we think two images are identical (used for collecting target images)
target_image_diff_threshold = 10

# number of randomly sampled targets per scene
targets_per_scene = 100

# directory where target images are stored
target_images_folder = THOR_HOME + '/target_images'
env_db_folder = THOR_HOME + '/env_db_a' + str(len(supported_actions))

# random actions being taken when new episode is started
random_start = 30  # TODO: check the value used in paper

# maximum number of steps before the episode terminates
episode_max_steps = 10000

# for debugging
episode_max_steps = 10000000


