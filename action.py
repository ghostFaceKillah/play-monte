from enum import Enum


class MetaAction(Enum):
    CLOSE = 0
    REWIND = 1
    SAVE = 2
    EPISODE_END = 3