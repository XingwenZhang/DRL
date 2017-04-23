# DRL
Deep Reinforcement Learning

## Usage

Run `python DRL/main.py --help` for detailed usage.


## THOR

+ Prepare

```
cd THOR

# get THOR's binary build
./get_thor_binaries.sh

# collect target images 
# (by default it will sample 100 images for each environment)
./collect_target_images.sh

# set THOR_HOME env_var
# (or you can simply set it in your ~/.bashrc file for convenience)
export THOR_HOME=$(pwd)
```


