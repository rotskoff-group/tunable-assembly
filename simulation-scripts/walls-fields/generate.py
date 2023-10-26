import math
import argparse

import gsd.hoomd
import hoomd
import hoomd.md
import numpy as np

from rigid_structure import generate_rigid_positions
parser = argparse.ArgumentParser()
parser.add_argument("--density", type=float, default=0.4)
parser.add_argument("--separation", type=float, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num", type=int, default=9)

args = parser.parse_args()
density = args.density
separation = args.separation
seed = args.seed
num = args.num

# Set random seed in generator
rng = np.random.default_rng(seed)

# Initialize solvent particles
N_particles = 10000
K = math.ceil(N_particles ** (1 / 2))
L = (N_particles / density) ** 0.5
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

# Initialize the passive objects
# Will use the rigid feature to make them
if separation > L:
    print("Separation is greater than box size!")

wall, num = generate_rigid_positions(num, 1)

# Make first rigid object
pos_W0 = np.array([[-separation / 2, 0, 0]])

# Make second rigid object
pos_W1 = np.array([[separation / 2, 0, 0]])

for i in range(N_particles):
    dist_0_ = position[i, :] - pos_W0
    dist_1_ = position[i, :] - pos_W1
    dist_0 = np.sum(dist_0_**2) ** 0.5
    dist_1 = np.sum(dist_1_**2) ** 0.5
    if dist_0 < dist_1:
        if dist_0 < 16:
            position[i, 0] -= 15 + rng.normal(loc=-0.1, scale=0.1)
            if position[i, 1] >= 0:
                position[i, 1] += 12 + rng.normal(loc=0.1, scale=0.1)
            else:
                position[i, 1] -= 12 + rng.normal(loc=-0.1, scale=0.1)
    elif dist_1 < dist_0:
        if dist_1 < 16:
            position[i, 0] += 15 + rng.normal(loc=0.1, scale=0.1)
            if position[i, 1] >= 0:
                position[i, 1] += 12 + rng.normal(loc=0.1, scale=0.1)
            else:
                position[i, 1] -= 12 + rng.normal(loc=-0.1, scale=0.1)
    elif dist_1 == dist_0:
        if dist_1 < 16:
            position[i, 0] += rng.normal(loc=0.0, scale=0.1)
            if position[i, 1] <= 0:
                position[i, 1] -= 40 + rng.normal(loc=0.0, scale=0.1)
            else:
                position[i, 1] += 40 + rng.normal(loc=0.0, scale=0.1)

# Calculate moment of inertia
mass = 1
I_m = np.zeros((3, 3))
for r in wall:
    I_m += mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))
I_zz = I_m[2, 2]

snapshot = gsd.hoomd.Frame()
snapshot.particles.N = N_particles + 2
snapshot.particles.position = position[0:N_particles]
snapshot.particles.position = np.append(
    snapshot.particles.position, pos_W0, axis=0
)
snapshot.particles.position = np.append(
    snapshot.particles.position, pos_W1, axis=0
)
snapshot.particles.typeid = [0] * N_particles
snapshot.particles.typeid.append(1)
snapshot.particles.typeid.append(1)
snapshot.particles.mass = [1] * N_particles
snapshot.particles.mass.append(num)
snapshot.particles.mass.append(num)
snapshot.particles.diameter = [1] * (N_particles + 2)
snapshot.configuration.box = [L, L, 0, 0, 0, 0]
snapshot.particles.types = ["A", "Wall", "Wall_0"]

# Generate random orientations
orientation = np.zeros((N_particles, 4))
random_angles = rng.uniform(0, 2 * np.pi, N_particles)
orientation[:, 0] = np.cos(random_angles / 2)
orientation[:, 3] = np.sin(random_angles / 2)
snapshot.particles.moment_inertia = [[0, 0, 0]] * N_particles
snapshot.particles.moment_inertia.append([0, 0, I_zz])
snapshot.particles.moment_inertia.append([0, 0, I_zz])
snapshot.particles.orientation = orientation.tolist()
snapshot.particles.orientation.append([0, 0, 0, 1])
snapshot.particles.orientation.append([0, 0, 0, 1])
with gsd.hoomd.open(name="lattice.gsd", mode="w") as f:
    f.append(snapshot)

# Now make the rigid objects
rigid = hoomd.md.constrain.Rigid()
rigid.body["Wall"] = {
    "constituent_types": ["Wall_0" for i in range(num)],
    "positions": wall,
    "orientations": [(0.0, 0.0, 0.0, 1.0) for i in range(num)],
    "charges": [0.0 for i in range(num)],
    "diameters": [1.0 for i in range(num)],
}

cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=1)
sim.create_state_from_gsd(filename="lattice.gsd")
rigid.create_bodies(sim.state)
hoomd.write.GSD.write(state=sim.state, mode="wb", filename="lattice_rigid.gsd")
