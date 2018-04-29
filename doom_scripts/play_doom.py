import pygame

import doom.env.doom_health_gathering as doom_health

"""
TODO: make a cleaner way to encode which actions we take,
I guess the best would be indicator variables?
"""


button_to_action = {
    pygame.K_UP: 13,
    pygame.K_RIGHT: 14,
    pygame.K_LEFT: 15
}


def key_map_helper(pressed_buttons):
    """
    list of pressed buttons ->  list of game action numbers
    """
    game_action = []

    for button in pressed_buttons:
        if button in button_to_action:
            game_action.append(button_to_action[button])

    return game_action


def play(env, fps=30):

    pressed_keys = []
    running = True
    env_done = True

    clock = pygame.time.Clock()
    pygame.display.set_mode((1, 1))

    while running:
        if env_done:
            env_done = False
            obs = env.reset()
        else:
            # map pressed key to action
            action = key_map_helper(pressed_keys)
            obs, rew, env_done, info = env.step(action)
            if abs(rew) > 0.1:
                print("Got reward = {}".format(rew))

            # if callback is not None:
            #     callback(prev_obs, obs, action, rew, env_done, info, env)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pressed_keys.append(event.key)
                if event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False

        env.render()
        clock.tick(fps)

    pygame.quit()


def main():
    env = doom_health.DoomHealthGatheringEnv()
    play(env, fps=35)
    env.close()


if __name__ == '__main__':
    main()
