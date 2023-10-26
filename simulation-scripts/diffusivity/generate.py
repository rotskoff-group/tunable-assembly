import argparse
import math

import gsd.hoomd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--density", type=float, default=0.4)
parser.add_argument("--N_particles", type=int, default=10000)

args = parser.parse_args()
density = args.density
N_particles = args.N_particles

# Initialize system
K = math.ceil(N_particles ** (1 / 2))
L = (N_particles / density) ** (1 / 2)
x = np.linspace(-L / 2, L / 2, K, endpoint=False)
position = np.zeros((N_particles, 3))
count = 0
for i in range(K):
    for j in range(K):
        position[count, 0] = x[i]
        position[count, 1] = x[j]
        count = count + 1
        if count == N_particles:
            break
    if count == N_particles:
        break

snapshot = gsd.hoomd.Frame()
snapshot.particles.N = N_particles
snapshot.particles.position = position[0:N_particles]
snapshot.particles.typeid = [0] * N_particles
snapshot.particles.mass = [1] * N_particles
snapshot.particles.diameter = [1] * N_particles
snapshot.configuration.box = [L, L, 0, 0, 0, 0]
snapshot.particles.types = ["A"]
with gsd.hoomd.open(name="lattice.gsd", mode="w") as f:
    f.append(snapshot)
