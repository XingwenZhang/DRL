""" This module implements a utility class that manages the target image
"""
import os
import THORConfig as config
import skimage.io
import robosims.server
import random

class THORTargetManager:
	def __init__(self):
		pass

	def collect_target_images(self, env_name, dump_folder, suffix='.png'):
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
		num_objects = len(event.metadata['objects'])
		objects_found = [False] * num_objects
		objects_found_count = 0
		while (True):
			event = env.step(dict(action=random.choice(config.supported_actions)))
			objects = event.metadata['objects']
			for i in range(num_objects):
				obj = objects[i]
				if obj['visible'] and not objects_found[i]:
					objects_found[i] = True
					objects_found_count += 1
					file_path = os.path.join(dump_folder, obj['objectId']) + suffix
					skimage.io.imsave(file_path, event.frame)
					print('target image for ' + obj['objectId'] + ' saved.[{0}/{1}]'.format(objects_found_count, num_objects))
					if objects_found_count == num_objects:
						break

		# finally dump the object id list
		print('dumping object id list')
		f = open(os.path.join(dump_folder, config.object_id_list_name), 'w')
		for i in range(num_objects):
			obj = objects[i]
			f.write(obj['objectId'] + '\n')
		f.close()

		print('all target images in the environments are saved.')


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_folder', default='taget_images')
	args = parser.parse_args()
	if not os.path.exists(args.save_folder):
		os.mkdir(args.save_folder)
	mgr = THORTargetManager()
	for env_name in config.supported_envs:
		print('collecing target images for ' + env_name + '...')
		dump_folder = os.path.join(args.save_folder, env_name)
		if not os.path.exists(dump_folder):
			os.mkdir(dump_folder)
		mgr.collect_target_images(env_name, dump_folder)
	print('done.')



