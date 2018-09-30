import abc
import numpy as np
import pygame
import os
from typing import List

from action import MetaAction
from data import DataGatheringWithReset
from keyboard import AtariXorKeyboard, DefaultKeyToActionMapper, DefaultKeyToMetaActionMapper
import utils


class AbstractPlay(abc.ABC):
    """
    Used for recording expert trajectories.

    Its main goal is holding & updating state of sub-components:
    expert inputs, env and data recording in clear, independent steps.

    Example usage
    -------------
    p = HumanDataGatheringPlay(env, settings)
    p.pre_main_loop()
    while not p.done():
        p.before_env_step()
        p.env_step()
        p.post_env_step()

    """
    @abc.abstractmethod
    def done(self):
        ...

    def pre_main_loop(self):
        ...

    def before_env_step(self):
        ...

    def env_step(self):
        ...

    def post_env_step(self):
        ...

    def close(self):
        ...


class State:
    """
    Data class representing current state in the reinforcement
    learning sense, as collected from the environment.
    """
    obs: np.ndarray
    prev_obs: np.ndarray
    action: int
    meta_actions: List[MetaAction]


class HumanDataGatheringPlay:
    """
    In its current this class is pretty atari-specific,
    especially rewind-processing and keyboard processing.
    """
    def __init__(self, env, config):
        self.env = env
        self.config = config

        if 'fps' in self.config:
            self.fps = self.config['fps']
        else:
            self.fps = 60

        self.data = DataGatheringWithReset(self._get_data_base_dir())
        self.key_to_action_mapper = DefaultKeyToActionMapper()
        self.key_to_meta_action_mapper = DefaultKeyToMetaActionMapper()
        self.keyboard = AtariXorKeyboard()

        pygame.init()
        self.clock = pygame.time.Clock()

        self.state = State()
        self.closing = False

        self._handlers = {
            MetaAction.CLOSE: self.close,
            MetaAction.REWIND: self._process_rewind,
            MetaAction.SAVE: self._process_save,
            MetaAction.EPISODE_END: self._process_episode_end
        }

    def _config(self, key, default):
        if key in self.config:
            return self.config[key]
        else:
            return default

    def _get_data_base_dir(self):
        """
        Get the dir where the data is going to be recorded.
        """
        base_dir = self._config(
            'base_dir',
            os.path.join(utils.ROOT_DIR, 'data')
        )

        env_name = self.env.spec._env_name

        run_name = self._config(
            'run_name',
            'default'
        )

        final_dir = os.path.join(base_dir, env_name, run_name)

        return final_dir

    def done(self):
        return self.closing

    def pre_main_loop(self):
        pygame.display.set_mode((1, 1))

        obs = self.env.reset()

        self.state.obs = obs
        self.state.prev_obs = obs

    def before_env_step(self):

        self.keyboard.process_keyboard_state()
        keys = self.keyboard.get_pressed_keys()
        self.state.action = self.key_to_action_mapper.map(keys)
        self.state.meta_actions = self.key_to_meta_action_mapper.map(keys)

        self.env.render()

    def env_step(self):
        self.state.prev_obs = self.state.obs
        self.state.obs, rew, done, info = self.env.step(self.state.action)

        self.data.store_transition(
            self.state.prev_obs, self.state.obs, self.state.action,
            rew, done, info, self.env
        )

        if done:
            self.state.meta_actions.append(MetaAction.EPISODE_END)

    def _process_rewind(self):
        prev_env_state = self.data.rewind(self.fps)
        self.keyboard.make_rewind_key_up()
        self.env.env.restore_full_state(prev_env_state)

    def close(self):
        print("Saving data...")
        self.data.save_trajectory()
        print("Done!")
        self.env.close()
        self.closing = True

    def _process_save(self):
        self.data.save_trajectory()
        self.data.new_trajectory()

    def _process_episode_end(self):
        obs = self.env.reset()

        self.state.obs = obs
        self.state.prev_obs = obs

        self.data.save_trajectory()
        self.data.new_trajectory()

    def post_env_step(self):
        # Process all the meta actions

        for meta_action in set(self.state.meta_actions):
            self._handlers[meta_action]()

        self.clock.tick(self.fps)
