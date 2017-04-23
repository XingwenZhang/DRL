save_folder='target_images'
num_images_per_env=1000

rm -rf ${save_folder}
python THORTargetImgProvider.py --save_folder ${save_folder} --num_images_per_env ${num_images_per_env}
