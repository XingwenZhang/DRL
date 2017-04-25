import os
import shutil
import THORConfig as config
import THORTarget

if os.path.exists(config.target_images_folder):
	shutil.rmtree(config.target_images_folder)
os.mkdir(config.target_images_folder)
mgr = THORTarget.THORTargetManager(config.target_images_folder)
for env_name in config.supported_envs:
	print('collecing target images for ' + env_name + '...')
	mgr.collect_target_images(env_name)
print('done.')