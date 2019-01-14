import abc
import gym
import pygame

from keyboard import AtariXorKeyboard, DefaultKeyToActionMapper, DefaultKeyToMetaActionMapper


class Agent(abc.ABC):
    """
    Perhaps a human, perhaps an algo

    """
    @abc.abstractmethod
    def initialize(self):
        ...

    @abc.abstractmethod
    def act(self, obs, env: gym.Env):
        ...


class HumanAgent(Agent):
    def __init__(self, fps: int, env: gym.Env):

        self.env = env
        self.fps = fps

        self.key_to_action_mapper = DefaultKeyToActionMapper()
        self.key_to_meta_action_mapper = DefaultKeyToMetaActionMapper()
        self.keyboard = AtariXorKeyboard()

        self.action = None
        self.meta_actions = []

        self.clock = None
        pygame.init()
        self.clock = pygame.time.Clock()

        pygame.display.set_mode((1, 1))

    def initialize(self):
        pass

    def rewind_key_up(self):
        self.keyboard.make_rewind_key_up()

    def process_keyboard(self):

        self.keyboard.process_keyboard_state()
        keys = self.keyboard.get_pressed_keys()
        self.action = self.key_to_action_mapper.map(keys)
        self.meta_actions = self.key_to_meta_action_mapper.map(keys)

        self.env.render()

    def act(self, obs, env: gym.Env):
        return self.action, self.meta_actions

    def tick_clock(self):
        self.clock.tick(self.fps)

