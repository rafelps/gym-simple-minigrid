#!/usr/bin/env python3

import argparse
import gym
from gym_simple_minigrid.window import Window


class EpisodeReward:
    def __init__(self):
        self.last = self.reward = None
        self.reset()

    def increment(self, r):
        self.last = r
        self.reward += r

    def reset(self):
        self.reward = self.last = 0


def redraw():
    img = env.render('rgb_array', tile_size=args.tile_size)
    window.show_img(img)
    window.set_caption(f'Step {env.step_count:3}.        Last reward = {er.last:2}    Accumulated reward = '
                       f'{er.reward:3}')


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    env.reset()
    er.reset()

    redraw()


def step(action):
    obs, reward, done, info = env.step(action)
    er.increment(reward)
    print(f'\tstep = {env.step_count}, accumulated reward = {er.reward}')

    redraw()
    if done:
        print(f'done! episode reward = {er.reward}')
        reset()


def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='Simple-MiniGrid-FourRooms-15x15-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="if provided, the same exact environment will always be generated",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)

args = parser.parse_args()

# env and window as global variables
env = gym.make(args.env)

window = Window(f"Manual Control for {env.name}")
window.reg_key_handler(key_handler)

er = EpisodeReward()

reset()

# Blocking event loop
window.show(block=True)
