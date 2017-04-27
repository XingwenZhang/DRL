"""
This script provdies an example of a random agent
"""

import THORConfig as config
from THOREnv import THOREnvironment
import random
import skimage.io

feat_mode = True
thor_env = THOREnvironment(feat_mode = feat_mode)

num_actions = thor_env.get_num_actions()
num_targets = thor_env.get_num_targets()

print('number of actions: ' + str(num_actions))
print('number of targets: ' + str(num_targets))

thor_env.reset_random()

print('cur_env_idx: ' + str(thor_env.get_env_idx()))
print('cur_env_name: ' + thor_env.get_env_name())

while True:
	action = random.randrange(0, num_actions)
	observation, action_success, reward, done = thor_env.step(action)
	print(observation)
	if reward > 0:
		print('found !')
		assert(done)
	if done:
		break
print('total_episode_reward: ' + str(thor_env.get_total_episode_reward()))

