# DRL
Deep Reinforcement Learning


## THOR

+ Prepare

	```
	cd THOR
	
	# set THOR_HOME env_var
	# (or you can simply set it in your ~/.bashrc file for convenience)
	export THOR_HOME=$(pwd)
	
	# get THOR's binary build
	./get_thor_binaries.sh
	
	# build environment DB 
	python script_build_db.py
	
	# collect target images 
	python script_generate_targets.py
	```


