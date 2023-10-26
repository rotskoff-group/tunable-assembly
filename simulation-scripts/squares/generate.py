import argparse

import gsd.hoomd
import hoomd
import hoomd.md
import numpy as np

from rigid_structure import generate_rigid_positions

parser = argparse.ArgumentParser()
parser.add_argument("--volume_faction_passive", type=float, default=0.2)
parser.add_argument("--volume_faction_active", type=float, default=0.2)
parser.add_argument("--size_passive", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()
volume_fraction_passive = args.volume_faction_passive
volume_fraction_active = args.volume_faction_active
size_passive = args.size_passive
seed = args.seed

# Set random seed in generator
rng = np.random.default_rng(seed)

# Initialize solute and solvent particles
# Set number of square particles, and then figure out box size
# and number of active particles based on volume fraction
N_particles_passive = 100
L = (size_passive**2 * N_particles_passive / volume_fraction_passive) ** (
    1 / 2
)
N_particles_active = int(volume_fraction_active * L**2 / (np.pi * 0.5**2))
N_particles = N_particles_active + N_particles_passive

# Place particles in a large cell that will get compressed later
L_new = 20 * L
K = int(np.ceil(N_particles**0.5))
x = np.linspace(-L_new / 2, L_new / 2, K, endpoint=False)
position = np.zeros((N_particles, 3))
count = 0
for i in range(K):
    if count == N_particles:
        break
    for j in range(K):
        if count == N_particles:
            break
        position[count, 0] = x[i]
        position[count, 1] = x[j] + (i % 2) * 0.5
        count = count + 1
wall, num = generate_rigid_positions(size_passive, 1)
# Calculate moment of inertia
mass = 1
I_m = np.zeros((3, 3))
for r in wall:
    I_m += mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))
I_zz = I_m[2, 2]

snapshot = gsd.hoomd.Frame()
snapshot.particles.N = N_particles
snapshot.particles.position = position
# Shuffle the positions to randomize particle types
rng.shuffle(snapshot.particles.position)
snapshot.particles.typeid = [0] * N_particles
for i in range(N_particles_active, N_particles):
    snapshot.particles.typeid[i] = 1
snapshot.particles.mass = [1] * N_particles
for i in range(N_particles_active, N_particles):
    snapshot.particles.mass[i] = num
snapshot.particles.diameter = [1] * N_particles
snapshot.configuration.box = [L_new, L_new, 0, 0, 0, 0]
snapshot.particles.types = ["A", "Wall", "Wall_0"]

# Generate random orientations
orientation = np.zeros((N_particles, 4))
random_angles = rng.uniform(0, 2 * np.pi, N_particles)
orientation[:, 0] = np.cos(random_angles / 2)
orientation[:, 3] = np.sin(random_angles / 2)
snapshot.particles.moment_inertia = [[0, 0, 0]] * N_particles
for i in range(N_particles_active, N_particles):
    snapshot.particles.moment_inertia[i] = [0, 0, I_zz]
snapshot.particles.orientation = orientation
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
