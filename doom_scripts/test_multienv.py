import doom.env as doom_env
from doom.env.wrappers import SetResolution
import time

if __name__ == '__main__':
    env_one = SetResolution('256x160')(doom_env.DoomBasicEnv())
    env_two = SetResolution('256x160')(doom_env.DoomMyWayHomeEnv())

    env_one.reset()
    env_two.reset()

    env_one.render()
    env_two.render()

    while True:
        env_one.step(env_one.action_space.sample())
        env_one.render()
        env_two.step(env_two.action_space.sample())
        env_two.render()



