from retro import play_retro
from retro_contest.local import make


def main():
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    play_retro.play(env, fps=15)


if __name__ == '__main__':
    main()