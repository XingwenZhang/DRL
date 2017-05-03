#!/usr/bin/env python
import robosims
import json
import THORConfig as config
from THOROfflineEnv import PoseRecorder

actions = {'a': 'MoveLeft',
           'd': 'MoveRight',
           'w': 'MoveAhead',
           's': 'MoveBack',
           'j': 'RotateLeft',
           'l': 'RotateRight',
           'i': 'LookUp',
           'k': 'LookDown'}

recorder = PoseRecorder()
env = robosims.controller.ChallengeController(unity_path=config.binary_build)
env.start()
t = json.loads(open(config.target_folder).read())
env.initialize_target(t[1])
env.step(action=dict(action='LookDown'))
while True:
  action_command = raw_input()
  if action_command not in actions:
      continue
  action_name = actions[action_command]
  event = env.step(action=dict(action=action_name))

  if event.metadata['lastActionSuccess']:
    recorder.record(action_name)
  print(recorder.get_pose())
  print(event.metadata['lastActionSuccess'])
env.stop()
