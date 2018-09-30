from typing import Set, Sequence

import pygame

import utils
from action import MetaAction, REWIND_KEY, SAVE_KEY


class Keyboard:
    """
    Class for managing state of the keyboard.
    Essentially a wrapper around pygame.
    Processes keyboard events and provides interface
    to keyboard state.
    """
    def __init__(self):
        self.pressed_keys = set()

    def process_keyboard_state(self):
        """ Process keyboard events and store them in the state. """
        keys_to_append = []
        keys_to_remove = []

        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                keys_to_append.append(event.key)
            elif event.type == pygame.KEYUP:
                keys_to_remove.append(event.key)
            elif event.type == pygame.QUIT:
                keys_to_append.append(27)

        for key in keys_to_append:
            self.pressed_keys.add(key)

        for key in keys_to_remove:
            if key in self.pressed_keys:
                self.pressed_keys.remove(key)

    def get_pressed_keys(self):
        return self.pressed_keys


_EXCLUSIONS = {
    pygame.K_UP: [pygame.K_DOWN],
    pygame.K_DOWN: [pygame.K_UP],
    pygame.K_LEFT: [pygame.K_RIGHT],
    pygame.K_RIGHT: [pygame.K_LEFT],
}


class AtariXorKeyboard(Keyboard):
    """
    A specialized Keyboard that gets rid of 'ghosting'
    problem, where keys would get stuck in pressed state,
    especially if many keys are pressed at the same time.

    It deals with the problem by not allowing 'key pressed'
    state in two opposite direction, e.g. left key press
    overwrites an earlier right key press event,
    even when no 'right key up' event has been recorded.
    """

    def make_rewind_key_up(self):
        """ make sure rewind key is not pressed """
        if REWIND_KEY in self.pressed_keys:
            self.pressed_keys.remove(REWIND_KEY)

    def _maybe_apply_exclude(self, key: int):
        if key in _EXCLUSIONS:
            keys_to_exclude = _EXCLUSIONS[key]
            self.pressed_keys = self.pressed_keys.difference(keys_to_exclude)

    def process_keyboard_state(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.pressed_keys.add(event.key)
                self._maybe_apply_exclude(event.key)
            elif event.type == pygame.KEYUP:
                if event.key in self.pressed_keys:
                    self.pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                self.pressed_keys.add(27)


class DefaultKeyToActionMapper:
    """ Maps key pressed on keyboard to environment action. """
    def __init__(self):
        self.keys_to_action = utils.extended_keymap()
        self.relevant_keys = set(sum(map(list, self.keys_to_action.keys()), []))

    def map(self, pressed_keys: Set[int]) -> int:
        action_meta = tuple(sorted(key for key in pressed_keys if key in self.relevant_keys))
        if action_meta in self.keys_to_action:
            return self.keys_to_action[action_meta]
        else:
            return 0


class DefaultKeyToMetaActionMapper:
    def map(self, pressed_keys) -> Sequence[MetaAction]:
        """
        Maps keys pressed on keyboard to a meta-action, e.g.
        "close", "rewind".
        """
        actions = []

        if SAVE_KEY in pressed_keys:
            actions.append(MetaAction.SAVE)
        if REWIND_KEY in pressed_keys:
            actions.append(MetaAction.REWIND)
        if pygame.K_ESCAPE in pressed_keys:
            actions.append(MetaAction.CLOSE)

        return actions