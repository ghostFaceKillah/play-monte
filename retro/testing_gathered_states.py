import gym
import numpy as np
import os

if __name__ == '__main__':
    env = gym.make("MontezumaRevengeNoFrameskip-v4")

    while True:
        done = False

        atari_env = env.env

        state_dirs = os.listdir('data/states')
        state_dir = np.random.choice(state_dirs)
        state_short_fname = np.random.choice(os.listdir('data/states/{}'.format(state_dir)))
        state_fname = 'data/states/{}/{}'.format(state_dir, state_short_fname)

        state = np.load(state_fname)
        env.reset()
        atari_env.restore_full_state(state)

        while not done:
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)
            env.render()


