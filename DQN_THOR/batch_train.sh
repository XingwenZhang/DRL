session_name=$1
gpu_idx=$2
if [$session_name == '']
then
	echo 'please specify a session_name.'
	exit
fi
if [$gpu_idx == '']
then
	echo 'please specify the gpu idx.'
	exit
fi
model_directory=model_${session_name}
summary_directory=summary_${session_name}
log_file=log_${session_name}.txt
rm -rf ${model_directory}/*
rm -rf ${summary_directory}/*
mkdir -p ${model_directory}
mkdir -p ${summary_directory}
export CUDA_VISIBLE_DEVICES=${gpu_idx}
python main.py --model_save_path ${model_directory}/model.ckpt --summary_folder ${summary_directory}
