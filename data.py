import abc
import os
from typing import List

import cv2
import numpy as np
import pandas as pd

import utils


class DataGatherer(abc.ABC):
    @abc.abstractmethod
    def store_transition(self, obs_t, obs_next, action, rew, done, info, env):
        ...

    @abc.abstractmethod
    def new_trajectory(self):
        ...

    @abc.abstractmethod
    def save_trajectory(self):
        ...


class DataGatheringWithReset(DataGatherer):
    """
    Gathers tuples, observation, action, reward, done, emulator state.

    Offers a call to get back a state that was n states ago.

    Saves the trajectory to the drive.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.new_trajectory()

        self.obs: List[...]
        self.obs_next: List[...]
        self.actions: List[...]
        self.rewards: List[...]
        self.state: List[...]
        self.done: List[...]
        self.info: List[...]

        self.traj_id: int
        self.img_dir: str
        self.state_dir: str
        self.traj_csv_fname: str

    def _clear_buffers_and_counters(self):
        """
        Clear buffers and counters to prepare for writing of a new trajectory.
        """

        self.obs_next = []
        self.actions = []
        self.rewards = []
        self.state = [] # emulator state
        self.done = []
        self.info = []

    def new_trajectory(self):
        """
        Prepare for recording of a new trajectory. Prepare filepaths.
        Flush buffers to prepare for new data.
        """

        self._clear_buffers_and_counters()
        self.traj_id = utils.get_next_traj_id(self.root_dir)
        self.img_dir, self.state_dir, self.traj_csv_fname = utils.prepare_data_dir(self.traj_id, self.root_dir)

    def _get_state_path(self, frame_id) -> str:
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
        """ Pull out the state out of the enironment"""
        return env.env.clone_full_state()

    def _log_transition(self, obs_t, obs_next, action, rew, done, info, env):
        """ Maybe log the transition. """
        if abs(rew) > 0.001:
            print("Reward!: {}".format(rew))

    def store_transition(self, obs_t, obs_next, action, rew, done, info, env):
        """ Store a transition for the further record """
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