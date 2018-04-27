import doom.env.doom_health_gathering as some_env
import gym


if __name__ == '__main__':
    env = some_env.DoomHealthGatheringEnv()
    env._reset()
    for i in range(10):
        env._step(env.action_space.sample())

    print("ww")


