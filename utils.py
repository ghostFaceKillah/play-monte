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
