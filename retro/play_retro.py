import numpy as np
import pygame
import retro


game_buttons = [
    'B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z'
]

button_map = {
    pygame.K_SPACE: 'B',
    pygame.K_q: 'B',
    pygame.K_w: 'A',
    pygame.K_BACKSPACE: 'MODE',
    pygame.K_RETURN: 'START',
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
    pygame.K_a: 'C',
    pygame.K_s: 'Y',
    pygame.K_z: 'X',
    pygame.K_x: 'Z',
}

button_to_action = {
    key: game_buttons.index(action_name)
    for key, action_name in button_map.items()
}


def key_map_helper(pressed_buttons):
    """
    Transform pressed keys into action space of the game.
    """
    action = np.zeros(len(game_buttons), dtype=np.int8)
    for button in pressed_buttons:
        if button in button_to_action:
            game_action_idx = button_to_action[button]
            action[game_action_idx] = 1
    return action


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
    env_name = 'SonicTheHedgehog2-Genesis'
    # env_name = 'Airstriker-Genesis'
    env = retro.make(game=env_name, state='EmeraldHillZone.Act1')
    # print(env.BUTTONS)
    play(env, fps=60)


if __name__ == '__main__':
    main()
