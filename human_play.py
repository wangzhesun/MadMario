import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame
from gym.wrappers import Monitor
from _image_view import ImageViewer
import time
from pyglet import clock

_NOP = 0

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
# env = gym_super_mario_bros.make('SuperMarioBros-3-1-v0')

# env = JoypadSpace(
#     env,
#     [['right'],
#     ['right', 'A']]
# )
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)


# env = wrap_env(env)
env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
episodes = 1000000

logger = MetricLogger(save_dir, episodes)



################################
# get the mapping of keyboard keys to actions in the environment
if hasattr(env, 'get_keys_to_action'):
    keys_to_action = env.get_keys_to_action()
elif hasattr(env.unwrapped, 'get_keys_to_action'):
    keys_to_action = env.unwrapped.get_keys_to_action()
else:
    raise ValueError('env has no get_keys_to_action method')
# create the image viewer
viewer = ImageViewer(
    env.spec.id if env.spec is not None else env.__class__.__name__,
    env.observation_space.shape[0], # height
    env.observation_space.shape[1], # width
    monitor_keyboard=True,
    relevant_keys=set(sum(map(list, keys_to_action.keys()), []))
)

# create a done flag for the environment
done = True
# prepare frame rate limiting
target_frame_duration = 1 / env.metadata['video.frames_per_second']
last_frame_time = 0
# start the main game loop
counter = 0
try:
    while True:
        current_frame_time = time.time()
        # limit frame rate
        if last_frame_time + target_frame_duration > current_frame_time:
            continue
        # save frame beginning time for next refresh
        last_frame_time = current_frame_time
        # clock tick
        clock.tick()
        # reset if the environment is done
        if done:
            if counter != 0:
                logger.log_episode()
                logger.record(
                    episode=counter,
                    epsilon=0.1,
                    step=1
                )
            counter += 1


            done = False
            state = env.reset()
            viewer.show(env.unwrapped.screen)
        # unwrap the action based on pressed relevant keys
        action = keys_to_action.get(viewer.pressed_keys, _NOP)
        next_state, reward, done, _ = env.step(action)

        logger.log_step(reward, None, None)

        viewer.show(env.unwrapped.screen)
        # pass the observation data through the callback
        # if callback is not None:
        #     callback(state, action, reward, done, next_state)
        state = next_state
        # shutdown if the escape key is pressed
        if viewer.is_escape_pressed:
            break
except KeyboardInterrupt:
    pass


viewer.close()
env.close()
