"""
My own experiments in gathering human expert data.
"""

import numpy as np
import pandas as pd

import gym
import gym.utils.play as play_utils



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


class DataGathering(object):
    def __init__(self):
        self.obs_t = []
        self.obs_next = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.info = []

        self.idx = 0

    def save_data(self, obs_t, obs_next, action, rew, done, info):
        self.obs_t.append(obs_t)
        self.obs_next.append(obs_next)
        self.actions.append(action)
        self.rewards.append(rew)
        self.done.append(done)
        self.info.append(info)

        if self.idx % 1000 == 0:
            print("Step {}".format(self.idx))

        self.idx += 1

    def write_to_drive(self):
        print("There are {} data points".format(len(self.obs_t)))
        print("Saving data...")

        np.save('data/obs_t.npy', np.array(self.obs_t))
        np.save('data/obs_next.npy', np.array(self.obs_next))
        np.save('data/actions.npy', np.array(self.actions))
        np.save('data/rewards.npy', np.array(self.rewards))
        np.save('data/done.npy', np.array(self.done))

        print("Saved.")


if __name__ == '__main__':
    # env = gym.make("MontezumaRevengeNoFrameskip-v4")
    env = gym.make("PongNoFrameskip-v4")

    data = DataGathering()

    play_utils.play(
        env,
        zoom=4,
        fps=60,
        callback=data.save_data
    )

    data.write_to_drive()
