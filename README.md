Playing Montezuma's Revenge for greater good
============================================

Good luck !
Steering: arrows + space for jump
Recording stops if you don't move for 5 seconds.

Dependencies
============

cv2, gym, pygame

```
pip install opencv-python
pip install gym[atari]
pip install pygame
```


Running
------
```
python gather_data.py

```
or if you prefer fullscreen
```
python gather_fullscreen.py

```

TODO:
- [ ] Rewind handler
- [ ] Per game name saver, session name, etc
- [ ] Comment all of the code



Comments
--------
- [X] How to deal with ghosting: Pressing down cancels other arrows, etc.
- [ ] Reimplement support for Vizdoom, etc. history of commits could be helpful.


