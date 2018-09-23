"""
Default envs are mapped to wasd.
Let's extend it: Make up arrow perform up action

Default monte keymap
{(): 0,
 (32,): 1,
 (32, 97): 12,
 (32, 97, 115): 17,
 (32, 97, 119): 15,
 (32, 100): 11,
 (32, 100, 115): 16,
 (32, 100, 119): 14,
 (32, 115): 13,
 (32, 119): 10,
 (97,): 4,
 (97, 115): 9,
 (97, 119): 7,
 (100,): 3,
 (100, 115): 8,
 (100, 119): 6,
 (115,): 5,
 (119,): 2
 }

NOTE: actions like '(97, 100)'  = ('left', 'right') are illegal
and are ignored by gym env
"""

import os
import pygame


def extended_keymap():
    steering_variants = [
        {
            "up": pygame.K_w,
            "down": pygame.K_s,
            "left": pygame.K_a,
            "right": pygame.K_d,
            "fire": pygame.K_SPACE
        },
        {
            "up": pygame.K_UP,
            "down": pygame.K_DOWN,
            "left": pygame.K_LEFT,
            "right": pygame.K_RIGHT,
            "fire": pygame.K_SPACE
        }
    ]


    raw_action_map = {
        (): 0,
        ('fire',): 1,
        ('fire', 'left'): 12,
        ('down', 'fire', 'left'): 17,
        ('fire', 'left', 'up'): 15,
        ('fire', 'right'): 11,
        ('down', 'fire', 'right'): 16,
        ('fire', 'right', 'up'): 14,
        ('down', 'fire'): 13,
        ('fire', 'up'): 10,
        ('left',): 4,
        ('down', 'left'): 9,
        ('left', 'up'): 7,
        ('right',): 3,
        ('down', 'right'): 8,
        ('right', 'up'): 6,
        ('down',): 5,
        ('up',): 2,
    }

    action_map = {}

    for steering in steering_variants:
        for meta_keys, action in raw_action_map.items():
            keys = tuple(sorted([
                steering[meta_key]
                for meta_key in meta_keys
            ]))
            action_map[keys] = action

    return action_map


def mkdir_p(dir):
    """ Check if directory exists and if not, make it."""
    if not os.path.exists(dir):
        os.makedirs(dir)


def prepare_data_dir(traj_no, root_data_dir='data'):
    screen_dir = os.path.join(root_data_dir, 'screens', "{:06d}".format(traj_no))
    states_dir = os.path.join(root_data_dir, 'states', "{:06d}".format(traj_no))

    if not os.path.exists(screen_dir):
        os.makedirs(screen_dir)

    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    csv_dir = os.path.join(root_data_dir, 'trajectories')

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    csv_name = os.path.join(csv_dir, "{:06d}.csv".format(traj_no))

    return screen_dir, states_dir, csv_name


def get_next_traj_id(root_data_dir='data'):
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