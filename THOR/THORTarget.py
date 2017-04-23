""" This module implements a utility class that manages the target image
"""
import os
import shutil
import THORConfig as config
import skimage
import skimage.io
import skimage.exposure
import numpy as np
import robosims.server
import random

class THORTargetManager:
	def __init__(self, target_images_folder):
		self.target_images_folder = target_images_folder

	def get_target_image(self, env_name, target_idx):
		folder = os.path.join(self.target_images_folder, env_name)
		file_path = os.path.join(self.target_images_folder, env_name, str(target_idx) + '.png')
		return skimage.img_as_float(skimage.io.imread(file_path))

	def collect_target_images(self, env_name):
		"""
		Find all target images of the scene and save them to dump_folder
		"""
		dump_folder = os.path.join(self.target_images_folder, env_name)
		if not os.path.exists(dump_folder):
			os.mkdir(dump_folder)
		num_images = config.targets_per_scene
		env = robosims.server.Controller(player_screen_width=config.screen_width,
                                         player_screen_height=config.screen_height,
                                         darwin_build=config.darwin_build,
                                         linux_build=config.linux_build,
                                         x_display=config.x_display)
		env.start()
		event = env.reset(env_name)
		target_images = np.empty((num_images, config.screen_height, config.screen_width, 3))
		i = 0
		while True:
			action = random.choice(config.supported_actions)
			max_steps = 10 if action.startswith('Move') else 2
			for _ in range(random.randrange(1, max_steps)):
				event = env.step(dict(action=action))
				success = event.metadata['lastActionSuccess']
				target_image = skimage.img_as_float(event.frame)
				if not success:
					break
			# check low contrast
			if skimage.exposure.is_low_contrast(target_image):
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
			skimage.io.imsave(os.path.join(dump_folder, str(i) + '.png'), target_images[i])


if __name__ == '__main__':
	if os.path.exists(config.target_images_folder):
		shutil.rmtree(config.target_images_folder)
	os.mkdir(config.target_images_folder)
	mgr = THORTargetManager(config.target_images_folder)
	for env_name in config.supported_envs:
		print('collecing target images for ' + env_name + '...')
		mgr.collect_target_images(env_name)
	print('done.')



