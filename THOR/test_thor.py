#!/usr/bin/env python
import robosims.server

actions = {'a': 'MoveLeft',
           'd': 'MoveRight',
           'w': 'MoveAhead',
           's': 'MoveBack',
           'j': 'RotateLeft',
           'l': 'RotateRight',
           'i': 'LookUp',
           'k': 'LookDown'}

env = robosims.server.Controller(
        player_screen_width=300,
        player_screen_height=300,
        darwin_build='thor_binary/thor-cmu-201703101557-OSXIntel64.app/Contents/MacOS/thor-cmu-201703101557-OSXIntel64',
        linux_build='thor_binary/thor-cmu-201703101558-Linux64',
        x_display="0.0")

env.start()
env.reset('FloorPlan224') # FloorPlan223 and FloorPlan224 are also available

while True:
    action = raw_input()
    if action not in actions:
        continue
    event = env.step(dict(action=actions[action]))
    print(event.metadata['lastActionSuccess'])
    print(event.metadata)
env.stop()
