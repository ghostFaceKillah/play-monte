import cv2
import gym
import numpy as np
import os
import pandas as pd
import pygame
import time

import utils


class DataGatheringWithReset(object):
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

    def process_transition(self, obs_t, obs_next, action, rew, done, info, env):
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


def get_keys(relevant_keys, pressed_keys, running):
    """
    Get all pygame events and record presses of relevant keys.
    """
    # process pygame events
    keys_to_append = []
    keys_to_remove = []

    for event in pygame.event.get():
        # test events, set key states
        if event.type == pygame.KEYDOWN:
            keys_to_append.append(event.key)
            if event.key == 27:
                running = False
        if event.type == pygame.KEYUP:
            keys_to_remove.append(event.key)
        if event.type == pygame.QUIT:
            running = False

    for key in keys_to_append:
        pressed_keys.add(key)

    for key in keys_to_remove:
        if key in pressed_keys:
            pressed_keys.remove(key)

    return pressed_keys, running


def default_key_to_action_mapper(env):
    """
    Get a default mapper action list : [int] -> action: int.
    And list of keys that are relevant to the given atari env.
    This mapping is given by Atari env, won't work with anything else.
    """
    keys_to_action = utils.extended_keymap()

    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    def mapper(pressed_keys):
        action_meta = tuple(sorted(key for key in pressed_keys if key in relevant_keys))
        if action_meta in keys_to_action:
            return keys_to_action[action_meta]
        else:
            return 0

    return mapper


def check_if_rewind_or_save(keys):
    rewind = False
    save = False

    for key in keys:
        if key == pygame.K_s:
            save = True
        if key == pygame.K_r:
            rewind = True

    return rewind, save


def play(env, fps=30, keys_to_action_mapper=None, relevant_keys=None):

    data = DataGatheringWithReset()

    if keys_to_action_mapper is None:
        keys_to_action_mapper = default_key_to_action_mapper(env)

    pressed_keys = set()  # Holds state between env steps
    running = True
    env_done = False
    obs = env.reset()
    done_time = 0
    save = False
    rewind = False
    process_rewind = True

    clock = pygame.time.Clock()
    pygame.display.set_mode((1, 1))

    while running:
        if env_done:
            timeout = time.time() - done_time > 5

            if save or timeout:
                print("Saving data...")
                data.save_trajectory()
                data.new_trajectory()
                print("Done!")

                env_done = False
                obs = env.reset()

        else:
            action = keys_to_action_mapper(pressed_keys)
            prev_obs = obs
            obs, rew, env_done, info = env.step(action)

            data.process_transition(prev_obs, obs, action, rew, env_done, info, env)

            # Perhaps reset data accumulators
            if env_done:
                print("(s)ave or (r)ewind environment")
                done_time = time.time()

        if rewind and process_rewind:
            prev_env_state = data.rewind(fps)
            env.env.restore_full_state(prev_env_state)
            env_done = False
            process_rewind = False
            pressed_keys = set()

        if not rewind:
            process_rewind = True

        pressed_keys, running = get_keys(relevant_keys, pressed_keys, running)
        running = True

        pygame.event.pump()
        another_pressed_keys = pygame.key.get_pressed()

        print("Pressed keys = {}".format(pressed_keys))
        print("Another pressed keys = {}".format(another_pressed_keys))
        rewind, save = check_if_rewind_or_save(pressed_keys)

        env.render()
        clock.tick(fps)

    print("Saving data...")
    data.save_trajectory()
    data.new_trajectory()
    print("Done!")

    pygame.quit()


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
