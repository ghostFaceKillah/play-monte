import abc
import numpy as np
import pygame
import os
import gym

from typing import List

from action import MetaAction
from agent import Agent, HumanAgent
from data import DataGatheringWithReset, DataGatherer
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
    def __init__(self, agent: Agent, env: gym.Env, data: DataGatherer):
        self.agent = agent
        self.env = env
        self.data = data

        self.obs = None
        self.prev_obs = None
        self.action = None
        self.meta_actions = []

    @abc.abstractmethod
    def done(self):
        """
        Decide if we are done with data gathering.
        """
        ...

    def pre_main_loop(self):
        """
        Perform initialization before the main data gathering loop.
        """
        self.obs = self.prev_obs = self.env.reset()
        self.agent.initialize()

    def before_env_step(self):
        """
        Things you need to do before the environment step:
        Pass inputs to agent, get back actions and metactions
        """
        self.action, self.meta_actions = self.agent.act(self.env, self.obs)

    def env_step(self):
        self.prev_obs = self.obs
        self.obs, rew, done, info = self.env.step(self.action)

        self.data.store_transition(self.prev_obs, self.obs, self.action, rew, done, info, self.env)

        if done:
            self.meta_actions.append(MetaAction.EPISODE_END)

    def post_env_step(self):
        """" Process the meta-actions: closing the env, etc """
        ...

    @abc.abstractmethod
    def close(self):
        ...


class HumanDataGatheringPlay(AbstractPlay):
    """
    An instance of data collector that:
    - is Atari-specific (due to way state is saved and restored)
    - Offers rolling back time when you press 'R'
    """
    def __init__(self, agent: HumanAgent, env, data: DataGatherer):
        super().__init__(agent, env, data)
        self.closing = False
        self.fps = self.agent.fps

        self._handlers = {
            MetaAction.CLOSE: self.close,
            MetaAction.REWIND: self._process_rewind,
            MetaAction.SAVE: self._process_save,
            MetaAction.EPISODE_END: self._process_episode_end
        }

    def _process_rewind(self):
        prev_env_state = self.data.rewind(self.fps)
        self.agent.rewind_key_up()
        self.env.env.restore_full_state(prev_env_state)

    def before_env_step(self):
        """
        Things you need to do before the environment step:
        Pass inputs to agent, get back actions and metactions
        """
        self.agent.process_keyboard()
        self.action, self.meta_actions = self.agent.act(self.env, self.obs)

    def _process_save(self):
        self.data.save_trajectory()
        self.data.new_trajectory()

    def _process_episode_end(self):
        obs = self.env.reset()

        self.obs = obs
        self.prev_obs = obs

        self.data.save_trajectory()
        self.data.new_trajectory()

    def done(self):
        return self.closing

    def close(self):
        print("Saving data...")
        self.data.save_trajectory()
        print("Done!")
        self.env.close()
        self.closing = True

    def post_env_step(self):
        # Process all the meta actions

        for meta_action in set(self.meta_actions):
            self._handlers[meta_action]()

        self.agent.tick_clock()

