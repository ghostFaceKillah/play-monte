import gym

from data import DataGatheringWithReset
from play import HumanDataGatheringPlay
from agent import HumanAgent
from utils import get_data_base_dir



def runner(env, settings):

    agent = HumanAgent(settings['fps'] if 'fps' in settings else 60, env)
    base_dir = get_data_base_dir(settings, env)
    data = DataGatheringWithReset(base_dir)

    p = HumanDataGatheringPlay(agent, env, data)

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
    # env_name = 'BankHeist'
    # env_name = 'BattleZone'
    # env_name = 'Gravitar'
    env_name = 'MontezumaRevenge'
    # env_name = 'MsPacman'
    # env_name = 'Pitfall'
    # env_name = 'PrivateEye'
    # env_name = 'RoadRunner'
    # env_name = 'Solaris'
    # env_name = 'Qbert'
    # env_name = 'UpNDown'
    # env_name = 'YarsRevenge'

    # env_name = 'Venture'
    # env_name = 'Tennis'
    # env_name = 'StarGunner'
    # env_name = 'Frostbite'
    # env_name = 'Freeway'
    # env_name = 'DemonAttack'

    # Hard ones : Bank Heist, Gravitar, Ms. Pacman, Pitfall!, Solaris, Hero

    env = gym.make("{}NoFrameskip-v4".format(env_name))

    settings = {
        'base_dir': '/Users/misiu-dev/temp/atari_trajectories'
    }

    runner(env, settings)
