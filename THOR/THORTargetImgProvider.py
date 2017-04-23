""" This module implements a utility class that manages the target image
"""
import os
import THORConfig as config
import skimage.io
import skimage.transform
import skimage.exposure
import numpy as np
import robosims.server
import random

class THORTargetManager:
	def __init__(self):
		pass

	def collect_target_images(self, env_name, dump_folder, suffix='.png', num_images=100):
		"""
		Find all target images of the scene and save them to dump_folder
		"""
		env = robosims.server.Controller(player_screen_width=config.screen_width,
                                         player_screen_height=config.screen_height,
                                         darwin_build=config.darwin_build,
                                         linux_build=config.linux_build,
                                         x_display=config.x_display)
		env.start()
		event = env.reset(env_name)

		target_images = np.empty((num_images, config.net_input_height, config.net_input_width, 3))
		i = 0
		while True:
			action = random.choice(config.supported_actions)
			max_steps = 10 if action.startswith('Move') else 2
			for _ in range(random.randrange(1, max_steps)):
				event = env.step(dict(action=action))
				success = event.metadata['lastActionSuccess']
				if not success:
					break

			# resize to network input size
			target_image = skimage.transform.resize(event.frame, (config.net_input_height, config.net_input_width))

			# check low contrast
			if skimage.exposure.is_low_contrast(target_image):
				print('low contrast!')
				continue

			# check duplication
			if i !=0:
				if (np.sum(np.abs(target_images[0:i] - target_image), axis=(1,2,3)) < config.target_image_diff_threshold).any():
					continue
			target_images[i] = target_image
			i+=1
			if i % 10==0:
				print(str(i) + ' target images collected.')
			if i == num_images:
				break
		del env

		# dump images
		for i in range(num_images):
			skimage.io.imsave(os.path.join(dump_folder, str(i)+suffix), target_images[i])


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_folder', default='target_images', type=str)
	parser.add_argument('--num_images_per_env', default=10000, type=int)
	args = parser.parse_args()
	if not os.path.exists(args.save_folder):
		os.mkdir(args.save_folder)
	mgr = THORTargetManager()
	for env_name in config.supported_envs:
		print('collecing target images for ' + env_name + '...')
		dump_folder = os.path.join(args.save_folder, env_name)
		if not os.path.exists(dump_folder):
			os.mkdir(dump_folder)
		mgr.collect_target_images(env_name, dump_folder, num_images=args.num_images_per_env)
	print('done.')



