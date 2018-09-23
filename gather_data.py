import csv
import cv2
import gym
import numpy as np
import os
import time

import play
import utils


class DataGathering(object):
    """
    play_utils.play accepts a callback, here is excerpt from docs:

    callback: lambda or None
                     Callback if a callback is provided it will be executed after
                     every step. It takes the following input:
    obs_t: observation before performing action
    obs_tp1: observation after performing action
    action: action that was executed
    rew: reward that was received
    done: whether the environemnt is done or not
    info: debug info
    """
    def __init__(self, write_state=False):
        self.obs_t = []
        self.obs_next = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.info = []
        self.lst_nonzro_act_t = time.time()

        self.write_state = write_state

        self.f = None
        self.reset_logging()

    def reset_logging(self):
        if self.f is not None:
            self.f.close()

        self.traj_id = utils.get_next_traj_id()
        self.img_dir, self.state_dir, csv_name = utils.prepare_data_dir(self.traj_id)
        self.f = open(csv_name, 'wt')
        self.logger = csv.DictWriter(self.f, fieldnames=('frame', 'reward', 'score','terminal', 'action', 'lifes'))
        self.logger.writeheader()

        self.frame_id = 0
        self.score = 0

    def save_data(self, obs_t, obs_next, action, rew, done, info, env):
        if action != 0:
            self.lst_nonzro_act_t = time.time()

        # Only write data if there was any activity in last n seconds
        if time.time() - self.lst_nonzro_act_t < 5:
            img_path = os.path.join(self.img_dir, "{:07d}.png".format(self.frame_id))
            cv2.imwrite(img_path, cv2.cvtColor(obs_next, cv2.COLOR_RGB2BGR))

            if self.write_state:
                state_path = os.path.join(self.state_dir, "{:07d}.npy".format(self.frame_id))
                state = env.env.clone_full_state()
                np.save(state_path, state)

            self.score += rew
            if abs(rew) > 0.001:
                print("Reward!: {}".format(rew))
		
            self.logger.writerow({
                'frame': self.frame_id,
                'reward': rew,
                'score': self.score,
                'terminal': done,
                'action': action,
                'lifes': info['ale.lives']
            })
            self.frame_id += 1

        # NOTE: If the framework is slow then this is the cause...
        self.f.flush()

        if done:
            self.reset_logging()
            self.frame_id += 1


if __name__ == '__main__':
    # env_name = 'Alien'
    # env_name = 'Asteroids'
    # env_name = 'Atlantis'
    # env_name = 'BattleZone'
    # env_name = 'Gravitar'
    # env_name = 'MontezumaRevenge'
    env_name = 'Freeway'
    # env_name = 'Pitfall'
    # env_name = 'PrivateEye'
    # env_name = 'Qbert'
    # env_name = 'UpNDown'

    env = gym.make("{}NoFrameskip-v4".format(env_name))

    # data = DataGathering(write_state=True)

    play.play(
        env,
        zoom=2,
        fps=40,
        # callback=data.save_data,
        # keys_to_action=utils.extended_keymap()
    )

