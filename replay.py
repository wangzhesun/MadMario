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


def wrap_env(env):
    '''
    This monitoring wrapper records your  outputs and saves them in
    an   mp4 file in the stated directory. If we don't change the video directory
    the videos will get stored in 'content/' directory as default.
    '''
    env = Monitor(env, './video003', force=True)
    return env

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


env = wrap_env(env)
env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

# checkpoint = Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
# checkpoint = Path('trained_mario.chkpt')
checkpoint = Path('mario_net_16.chkpt')
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
mario.exploration_rate = mario.exploration_rate_min

episodes = 1000000

logger = MetricLogger(save_dir, episodes)

for e in range(episodes):

    state = env.reset()

    while True:

        env.render()
        # env.render(mode='rgb_array')

        action = mario.act(state)

        next_state, reward, done, info = env.step(action)

        mario.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None)

        state = next_state

        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
