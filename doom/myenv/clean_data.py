"""
Clean out zero trajectories.
"""

import os

trajectories = os.listdir('data/screens')

for traj in trajectories:
    inner = os.path.join('data/screens', traj)
    l = len(os.listdir(inner))

    traj_path = os.path.join('data/trajectories/{}.csv'.format(traj))

    print(traj, l, traj_path)

    if l == 0:
        os.rmdir(inner)
        os.remove(traj_path)

    # os.rmdir(inner)
