import gym

import gather_data
import play
import utils

if __name__ == '__main__':
    env = gym.make("MontezumaRevengeNoFrameskip-v4")

    data = gather_data.DataGathering()

    play.play(
        env,
        fps=40,
        fullscreen=True,
        callback=data.save_data,
        keys_to_action=utils.extended_keymap()
    )
