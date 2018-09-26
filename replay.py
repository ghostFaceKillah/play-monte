import cv2
import gym
import numpy as np
import os
import pandas as pd
import pygame
import time

from enum import Enum
from typing import Sequence, List, Set

import utils

from queue import PriorityQueue


class DataGatheringWithReset:
    """
    Gathers tuples, observation, action, reward, done, emulator state.

    Offers a call to get back a state that was n states ago.
    """
    def __init__(self):
        self.new_trajectory()

    def _clear_buffers_and_counters(self):
        """ Clear buffers and counters to prepare for writing of a new trajectory. """

        self.obs_next = []
        self.actions = []
        self.rewards = []
        self.state = [] # emulator state
        self.done = []
        self.info = []

    def new_trajectory(self):
        """
        Prepare for recording new trajectory.
        Prepare filepaths.
        Flush buffers to prepare for collection of new data.
        """

        self._clear_buffers_and_counters()
        self.traj_id = utils.get_next_traj_id()
        self.img_dir, self.state_dir, self.traj_csv_fname = utils.prepare_data_dir(self.traj_id)

    def _get_state_path(self, frame_id):
        state_path = os.path.join(self.state_dir, "{:07d}.npy".format(frame_id))
        return state_path

    def _write_state(self, state, frame_id):
        state_path = self._get_state_path(frame_id)
        np.save(state_path, state)

    def get_img_path(self, frame_id):
        return os.path.join(self.img_dir, "{:07d}.png".format(frame_id))

    def _write_img(self, obs_next, frame_id):
        """ Save an image to dr"""
        img_path = self.get_img_path(frame_id)
        cv2.imwrite(img_path, cv2.cvtColor(obs_next, cv2.COLOR_RGB2BGR))

    def save_trajectory(self):
        """
        Save the whole trajectory to the drive.
        0) Save the trajectory data csv to the drive
        1) Save all pictures
        2) Save all emulator states
        """

        no_frames = len(self.actions)

        df = pd.DataFrame({
            'frame': range(no_frames),
            'reward': self.rewards,
            # score
            'terminal': self.done,
            'action': self.actions,
            'lifes': [i['ale.lives'] for i in self.info]
        })
        df['score'] = df.reward.cumsum()
        df = df[['frame', 'reward', 'score', 'terminal', 'action', 'lifes']]
        df.to_csv(self.traj_csv_fname, index=False)

        for i in range(no_frames):
            self._write_img(self.obs_next[i], i)
            self._write_state(self.state[i], i)

    def _extract_state(self, env):
        """ Pull out state out of the enironment"""
        return env.env.clone_full_state()

    def _log_transition(self, obs_t, obs_next, action, rew, done, info, env):
        """ Maybe we want to log the transition in some way. Then log it here. """
        if abs(rew) > 0.001:
            print("Reward!: {}".format(rew))

    def store_transition(self, obs_t, obs_next, action, rew, done, info, env):
        state = self._extract_state(env)

        # Save data
        self.obs_next.append(obs_next)
        self.actions.append(action)
        self.rewards.append(rew)
        self.state.append(state)
        self.done.append(done)
        self.info.append(info)

        self._log_transition(obs_t, obs_next, action, rew, done, info, env)

    def rewind(self, n_back):
        """ Rewind time back by n_back steps. """
        idx = max(0, len(self.actions) - n_back)
        state = self.state[idx]

        self.obs_next = self.obs_next[:idx]
        self.actions = self.actions[:idx]
        self.rewards = self.rewards[:idx]
        self.state = self.state[:idx]
        self.done = self.done[:idx]
        self.info = self.info[:idx]

        return state


class Keyboard:
    def __init__(self):
        self.pressed_keys = set()

    def process_keyboard_state(self):
        keys_to_append = []
        keys_to_remove = []

        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                keys_to_append.append(event.key)
            if event.type == pygame.KEYUP:
                keys_to_remove.append(event.key)
            if event.type == pygame.QUIT:
                keys_to_append.append(27)

        for key in keys_to_append:
            self.pressed_keys.add(key)

        for key in keys_to_remove:
            if key in self.pressed_keys:
                self.pressed_keys.remove(key)

    def get_pressed_keys(self):
        return self.get_pressed_keys()


class DefaultKeyToActionMapper:
    def __init__(self):
        self.keys_to_action = utils.extended_keymap()
        self.relevant_keys = set(sum(map(list, self.keys_to_action.keys()), []))

    def map(self, pressed_keys: Sequence[int]) -> int:
        action_meta = tuple(sorted(key for key in pressed_keys if key in self.relevant_keys))
        if action_meta in self.keys_to_action:
            return self.keys_to_action[action_meta]
        else:
            return 0


class MetaAction(Enum):
    CLOSE = 0
    REWIND = 1
    SAVE = 2
    EPISODE_END = 3


class DefaultKeyToMetaActionMapper:
    timeout = time.time() - done_time > 5

    def map(self, pressed_keys) -> Sequence[MetaAction]:
        """
        Map to meta-action, such as "close", "rewind"
        """

        actions = []

        if pygame.K_s in pressed_keys:
            actions.append(MetaAction.SAVE)
        if pygame.K_r in pressed_keys:
            actions.append(MetaAction.REWIND)
        if pygame.K_ESCAPE in pressed_keys:
            actions.append(MetaAction.CLOSE)

        return actions


class Play:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        self.data = DataGatheringWithReset()
        self.key_to_action_mapper = DefaultKeyToActionMapper()
        self.key_to_meta_action_mapper = DefaultKeyToMetaActionMapper()
        self.keyboard = Keyboard()

        self.clock = pygame.time.Clock()

        self.obs: np.ndarray = None
        self.prev_obs: np.ndarray = None
        self.action: int = None
        self.meta_actions: Sequence[MetaAction] = []

    def done(self):
        return False

    def pre_main_loop(self):
        pygame.display.set_mode((1, 1))

        obs = env.reset()

        self.obs = obs
        self.prev_obs = obs

    def before_env_step(self):
        env.render()

        self.keyboard.process_keyboard_state()
        keys = self.keyboard.get_pressed_keys()
        self.action = self.key_to_action_mapper.map(keys)
        self.meta_actions = self.key_to_meta_action_mapper.map(keys)

    def env_step(self):
        self.prev_obs = self.obs
        obs, rew, env_done, info = env.step(self.action)

        self.obs = obs
        self.data.store_transition(self.prev_obs, obs, self.action, rew, env_done, info, env)

        if done:
            self.meta_actions.append*

    def _rewind(self):
        prev_env_state = self.data.rewind(self.fps)

        # Atari specific!!!
        env.env.restore_full_state(prev_env_state)


    def post_env_step(self):
        # Process all the actions

        for meta_action in self.meta_actions:
            if meta_action is MetaAction.CLOSE:
                self.data.save_trajectory()
            elif meta_action is MetaAction.REWIND:


        if save or timeout:
            print("Saving data...")
            data.new_trajectory()
            print("Done!")

            env_done = False
            obs = env.reset()

        if env_done:
            print("(s)ave or (r)ewind environment")
            done_time = time.time()

        if rewind and process_rewind:
            env_done = False
            process_rewind = False
            pressed_keys = set()

        if not rewind:
            process_rewind = True

        pass

        env.render()
        clock.tick(fps)

    def close(self):
        print("Saving data...")
        data.save_trajectory()
        data.new_trajectory()
        print("Done!")

        pygame.quit()


def play(env, settings):
    p = Play(env, settings)

    p.pre_main_loop()

    while not p.done():

        p.before_env_step()

        p.env_step()
        p.post_env_step()

    p.close()


if __name__ == '__main__':
    pygame.init()

    # env_name = 'Alien'
    # env_name = 'Amidar'
    # env_name = 'Asteroids'
    # env_name = 'Atlantis'
    env_name = 'BankHeist'
    # env_name = 'BattleZone'
    # env_name = 'Gravitar'
    # env_name = 'MontezumaRevenge'
    # env_name = 'MsPacman'
    # env_name = 'Pitfall'
    # env_name = 'PrivateEye'
    # env_name = 'RoadRunner'
    # env_name = 'Solaris'
    # env_name = 'Qbert'
    # env_name = 'UpNDown'
    # env_name = 'YarsRevenge'

    # Hard ones : Bank Heist, Gravitar, Ms. Pacman, Pitfall!, Solaris

    env = gym.make("{}NoFrameskip-v4".format(env_name))

    play(env, fps=60)
