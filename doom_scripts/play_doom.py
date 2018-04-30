import pygame
import doom.env


"""
TODO: make a cleaner way to encode which actions we take,
I guess the best would be indicator variables?
"""


button_to_action = {
    pygame.K_UP: 13,
    pygame.K_RIGHT: 14,
    pygame.K_LEFT: 15,
    pygame.K_SPACE: 0
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


def play(env, fps=35):

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


def health():
    """
    Trzeba zbierać medpacki, nie ma żadnych wrogów.
    Nie ma nagrody za medpacka co dziwne.
    Jak się nie zbierze medpacka to się
    """
    import doom.env.doom_health_gathering as doom_env_maker
    env = doom_env_maker.DoomHealthGatheringEnv()
    play(env)
    # env.close()


def defend_the_center():
    """
    Trudne! rurkowiec szybko wyskakuje, nie idzie na niego wycelować ręcznie.
    """
    import doom.env.doom_defend_center as doom_env_maker
    env = doom_env_maker.DoomDefendCenterEnv()
    play(env, fps=20)


def defend_the_line():
    """
    Dość trudne ale możliwe, jest linia i po drugiej stronei spawnują się potworki.
    Rurkowce i jakieś szalone pająki co strzlają ogniam

    """
    import doom.env.doom_defend_line as doom_env_maker
    env = doom_env_maker.DoomDefendLineEnv()
    play(env, fps=5)


def my_way_home():
    """
    To jest env który mam już dosć dobrze obczajony i zebrane dane:

    Chodzi w nim o to, aby dość  do końca labiryntu.
    Ten labirynt jest zawsze taki sam.
    """
    import doom.env as doom_env_maker
    env = doom_env_maker.DoomMyWayHomeEnv()
    play(env, fps=35)


def basic():
    """
    Płaski korytarz bez możliwości obracania się w lewo i w prawo
    """
    import doom.env.doom_basic as doom_env_maker
    env = doom_env_maker.DoomBasicEnv()
    play(env, fps=35)



if __name__ == '__main__':
    # defend_the_line()
    my_way_home()
    # basic()