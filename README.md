A tool for collecting human expert trajectories
============================================

Currently focused on Atari.

Steering: arrows + space for jump

Dependencies
============

make pipenv / pip freeze requirements


Running
------
```
python run.py

```

```

TODO:
-----
- [X] Rewind handler
- [ ] Better requirements
- [ ] Per game name saver, session name, etc
- [ ] Comment all of the code
- [ ] Put human inputs behind a layer of abstraction, as we could potentially
 run an automatic expert.



Comments
--------
- [X] How to deal with ghosting: Pressing down cancels other arrows, etc.
- [ ] Reimplement support for Vizdoom, etc. history of commits could be helpful.
