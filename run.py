import gym
import pygame

from play import Play

def runner(env, settings):
    p = Play(env, settings)

    p.pre_main_loop()

    while not p.done():

        p.before_env_step()
        p.env_step()
        p.post_env_step()


if __name__ == '__main__':

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

    settings = {}

    runner(env, settings)
