# To see the scenario description go to "../../scenarios/README.md"

from __future__ import print_function

import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import uuid
import vizdoom


def get_next_traj_id(root_data_dir='data'):
    return str(uuid.uuid1())


def get_next_traj_number(root_data_dir='data'):
    if not os.path.exists(root_data_dir):
        return 0

    relevant_files = [
        int(x) for x in os.listdir(os.path.join(root_data_dir, 'screens'))
        if x != '.DS_Store'
    ]

    if len(relevant_files) == 0:
        return 0
    else:
        return 1 + max(relevant_files)


def prepare_data_dir(traj_name, root_data_dir='data'):
    screen_dir = os.path.join(root_data_dir, 'screens', traj_name)
    states_dir = os.path.join(root_data_dir, 'states', traj_name)

    if not os.path.exists(screen_dir):
        os.makedirs(screen_dir)

    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    csv_dir = os.path.join(root_data_dir, 'trajectories')

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    csv_name = os.path.join(csv_dir, "{}.csv".format(traj_name))

    return screen_dir, states_dir, csv_name


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

        self.traj_id = get_next_traj_id()
        self.img_dir, self.state_dir, csv_name = prepare_data_dir(self.traj_id)
        self.f = open(csv_name, 'wt')
        self.logger = csv.DictWriter(self.f, fieldnames=('frame', 'reward', 'score','terminal', 'action'))
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
                'action': action
            })
            self.frame_id += 1

        # NOTE: If the framework is slow then this is the cause...
        self.f.flush()

        # if done:
        #     self.reset_logging()
        #     self.frame_id += 1


def get_configured_game():
    game = vizdoom.DoomGame()

    # Choose scenario config file you wish to watch.
    # Don't load two configs cause the second will overrite the first one.
    # Multiple config files are ok but combining these ones doesn't make much sense.

    game.load_config("scenarios/my_way_home.cfg")
    # game.load_config("scenarios/deathmatch.cfg")

    # Enables freelook in engine
    game.add_game_args("+freelook 1")

    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    # game.set_screen_resolution(vizdoom.ScreenResolution.RES_320X180)

    game.set_window_visible(True)
    game.set_mode(vizdoom.Mode.SPECTATOR)

    game.init()

    return game


def show_gray(img):
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()


def show_color(img):
    plt.imshow(img)
    plt.show()


def run_episode(game):
    data = DataGathering()
    callback = data.save_data

    game.new_episode()

    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer

        swapped_img = img.swapaxes(0, 1).swapaxes(1, 2)
        reshaped_img = swapped_img
        # reshaped_img = cv2.resize(swapped_img, (84, 84), interpolation=cv2.INTER_AREA)
        # gray_img = cv2.cvtColor(reshaped_img, cv2.COLOR_RGB2GRAY)

        game.advance_action()
        last_action = game.get_last_action()
        reward = game.get_last_reward()

        # Encode action combination as binary number
        # Should we get action meanings?
        stuff = [
            last_action[i] * mult
            for i, mult in enumerate([1, 2, 4])
        ]

        action = int(sum(stuff))

        # print("State #" + str(state.number))
        # print("Game variables: ", state.game_variables)
        print("Stuff", stuff)
        print("Action:", last_action)
        print("Numerical action:", action)
        # We do not catch all actions ...
        # print("Reward:", reward)

        callback(
            obs_t=None,
            obs_next=reshaped_img,
            action=action,
            rew=reward,
            done=game.is_episode_finished(),
            info={},
            env=game
        )


if __name__ == '__main__':
    game = get_configured_game()

    while True:
        # try:
        run_episode(game)
        # except:
        #   game.close()
