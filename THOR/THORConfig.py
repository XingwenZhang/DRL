# thor environment setting
screen_width=224
screen_height=224
darwin_build='thor_binary/thor-cmu-201703101557-OSXIntel64.app/Contents/MacOS/thor-cmu-201703101557-OSXIntel64'
linux_build='thor_binary/thor-cmu-201703101558-Linux64'
x_display="0.0"

# supported environments and actions
supported_envs = ['FloorPlan223', 'FloorPlan224', 'FloorPlan225']
supported_actions = ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown']

# THORTargetImgProvider configurations
object_id_list_name = 'obj_id_list.txt'

# random actions being taken when new episode is started
random_start = 30  # TODO: check the value used in paper

# maximum number of steps before the episode terminates
episode_max_steps = 10000


