import gym
import pygame

from pygame.locals import VIDEORESIZE


def get_keys(relevant_keys):
    pressed_keys = []
    # process pygame events
    for event in pygame.event.get():
        # test events, set key states
        if event.type == pygame.KEYDOWN:
            if event.key in relevant_keys:
                pressed_keys.append(event.key)
            elif event.key == 27:
                running = False
        elif event.type == pygame.KEYUP:
            if event.key in relevant_keys:
                pressed_keys.remove(event.key)
        elif event.type == pygame.QUIT:
            running = False

    return pressed_keys, running


def play(env, fps=30, callback=None, keys_to_action=None):

    obs_s = env.observation_space
    assert type(obs_s) == gym.spaces.box.Box
    assert len(obs_s.shape) == 2 or (len(obs_s.shape) == 3 and obs_s.shape[2] in [1,3])

    # We need to make some kind of mapper from keys to action

    # Some environments have mapper from keys to action,
    # for example atari ones. We could potentially use it

    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                          "please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()),[]))

    pressed_keys = []
    running = True
    env_done = True
    obs = None

    clock = pygame.time.Clock()
    pygame.display.set_mode((1, 1))

    while running:
        if env_done:
            env_done = False
            obs = env.reset()
        else:
            action = keys_to_action[tuple(sorted(pressed_keys))]
            prev_obs = obs
            obs, rew, env_done, info = env.step(action)

            if callback is not None:
                callback(prev_obs, obs, action, rew, env_done, info, env)

            # except KeyError as e:
            #     print("Warning: ignoring illegal action '{}'".format(e))

        pressed_keys, running = get_keys(relevant_keys)

        env.render()
        clock.tick(fps)
    pygame.quit()
