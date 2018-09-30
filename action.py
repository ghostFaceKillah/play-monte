from enum import Enum
import pygame

REWIND_KEY = pygame.K_r
SAVE_KEY = pygame.K_s


class MetaAction(Enum):
    CLOSE = 0
    REWIND = 1
    SAVE = 2
    EPISODE_END = 3