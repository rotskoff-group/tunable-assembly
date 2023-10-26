import argparse

import numpy as np

from rigid_structure import generate_rigid_positions

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--num", type=int, default=9, help="Number of particles"
)

args = parser.parse_args()
num = args.num

wall, num = generate_rigid_positions(num, 1)
wall = np.array(wall)


def calculate_rg(pos, num):
    mean = np.mean(pos, axis=0)
    Rg = 1.0 / num * np.sum((pos - mean) ** 2)
    return Rg**0.5


rg = calculate_rg(wall, num)
np.savetxt(f"r_g_{num}.txt", np.array([rg]))
