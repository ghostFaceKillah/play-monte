import csv
import cv2
import gym
import numpy as np
import os
import pygame
import time

import utils


class DataGatheringWithReset(object):
    """
    Gathers tuples, observation, action, reward, done, emulator state

    Offers a call to get back a state that was n states ago.
    """
    def __init__(self, write_state=False):

        self.should_write_state = write_state
        self.new_trajectory()

        self.f = None

    def clear_buffers_and_counters(self):

        self.obs_t = []
        self.obs_next = []
        self.actions = []
        self.rewards = []
        self.state = [] # emulator state
        self.done = []
        self.info = []

        self.frame_id = 0
        self.score = 0

    def new_trajectory(self):
        """
        Prepare writing filepath
        Flush buffers to prepare for collection of new data.
        """

        self.clear_buffers_and_counters()
        self.traj_id = utils.get_next_traj_id()
        self.img_dir, self.state_dir, self.fname = utils.prepare_data_dir(self.traj_id)

    def save_trajectory(self):
        pass

    def write_state(self, env):
        state_path = os.path.join(self.state_dir, "{:07d}.npy".format(self.frame_id))
        state = env.env.clone_full_state()
        np.save(state_path, state)

    def write_img(self, obs_next):
        img_path = os.path.join(self.img_dir, "{:07d}.png".format(self.frame_id))
        cv2.imwrite(img_path, cv2.cvtColor(obs_next, cv2.COLOR_RGB2BGR))

    def gather_data(self, obs_t, obs_next, action, rew, done, info, env):
        self.lst_nonzro_act_t = time.time()

        self.write_img(obs_next)

        if self.should_write_state:
            self.write_state(env)

        self.score += rew
        if abs(rew) > 0.001:
            print("Reward!: {}".format(rew))

        self.frame_id += 1

        if done:
            self.save_trajectory()
            self.new_trajectory()


def get_keys(relevant_keys, pressed_keys, running):
    """
    Get all pygame events and record presses of relevant keys.
    """
    # process pygame events
    for event in pygame.event.get():
        # test events, set key states
        if event.type == pygame.KEYDOWN:
            if event.key in relevant_keys:
                pressed_keys.append(event.key)
            elif event.key == 27:
                running = False
        elif event.type == pygame.KEYUP:
            if event.key in relevant_keys:
                pressed_keys.remove(event.key)
        elif event.type == pygame.QUIT:
            running = False

    return pressed_keys, running


def default_key_to_action_mapper(env):
    """
    Get a default mapper action list : [int] -> action: int.
    And list of keys that are relevant to the given atari env.
    This mapping is given by Atari env, won't work with anything else.
    """

    if hasattr(env, 'get_keys_to_action'):
        keys_to_action = env.get_keys_to_action()
    elif hasattr(env.unwrapped, 'get_keys_to_action'):
        keys_to_action = env.unwrapped.get_keys_to_action()
    else:
        assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                      "please specify one manually"

    relevant_keys = set(sum(map(list, keys_to_action.keys()),[]))

    def mapper(pressed_keys):
        return keys_to_action[tuple(sorted(pressed_keys))]

    return mapper, relevant_keys


def play(env, fps=30, callback=None, keys_to_action_mapper=None, relevant_keys=None):

    if keys_to_action_mapper is None:
        keys_to_action_mapper, relevant_keys = default_key_to_action_mapper(env)

    pressed_keys = []  # Holds state between env steps
    running = True
    env_done = True
    obs = None

    clock = pygame.time.Clock()
    pygame.display.set_mode((1, 1))

    while running:
        if env_done:
            env_done = False
            obs = env.reset()
        else:
            action = keys_to_action_mapper(pressed_keys)
            prev_obs = obs
            obs, rew, env_done, info = env.step(action)

            if callback is not None:
                callback(prev_obs, obs, action, rew, env_done, info, env)

        pressed_keys, running = get_keys(relevant_keys, pressed_keys, running)

        env.render()
        clock.tick(fps)

    pygame.quit()


if __name__ == '__main__':
    # env_name = 'Alien'
    # env_name = 'Asteroids'
    # env_name = 'Atlantis'
    # env_name = 'BattleZone'
    # env_name = 'Gravitar'
    # env_name = 'MontezumaRevenge'
    env_name = 'Pitfall'
    # env_name = 'PrivateEye'
    # env_name = 'Qbert'
    # env_name = 'UpNDown'

    env = gym.make("{}NoFrameskip-v4".format(env_name))

    # data = DataGathering(write_state=True)

    data = DataGatheringWithReset(write_state=True)

    play(
        env,
        fps=40,
        callback=data.gather_data,
        # keys_to_action=utils.extended_keymap()
    )
