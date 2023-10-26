import argparse

import hoomd
import numpy as np

from rigid_structure import generate_rigid_positions

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=int, default=0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num", type=int, default=9)
parser.add_argument("--activity", type=float, default=80)
parser.add_argument("--torque", type=float, default=5)
parser.add_argument("--dt", type=float, default=0.00001)

args = parser.parse_args()
base = args.base
seed = args.seed
num = args.num
activity = args.activity
torque = args.torque
dt = args.dt

communicator = hoomd.communicator.Communicator(ranks_per_partition=1)
gpu = hoomd.device.GPU(communicator=communicator)
sim = hoomd.Simulation(device=gpu, seed=seed + communicator.partition)
sim.create_state_from_gsd(filename="restart.gsd")
wall, num = generate_rigid_positions(num, 1)

rigid = hoomd.md.constrain.Rigid()
rigid.body["Wall"] = {
    "constituent_types": ["Wall_0" for i in range(num)],
    "positions": wall,
    "orientations": [(0.0, 0.0, 0.0, 1.0) for i in range(num)],
    "charges": [0.0 for i in range(num)],
    "diameters": [1.0 for i in range(num)],
}

# Integrator
integrator = hoomd.md.Integrator(dt=dt, integrate_rotational_dof=True)
integrator.rigid = rigid
cell = hoomd.md.nlist.Cell(buffer=0.4, exclusions=["body"])

# WCA potential
lj = hoomd.md.pair.LJ(nlist=cell, default_r_cut=2 ** (1.0 / 6.0), mode="shift")
lj.params[(["A", "Wall_0"], ["A", "Wall_0"])] = dict(epsilon=40.0, sigma=1)
lj.r_cut[(["A", "Wall_0"], ["A", "Wall_0"])] = 2 ** (1.0 / 6.0)
lj.params[("Wall", ["A", "Wall_0", "Wall"])] = dict(epsilon=0.0, sigma=0)
lj.r_cut[("Wall", ["A", "Wall_0", "Wall"])] = 0
integrator.forces.append(lj)

# Bond between wall particles
bond = hoomd.md.bond.Tether()
bond.params["W-W"] = dict(k_b=4000, l_min=0.0, l_c1=0.5, l_c0=40.0, l_max=40.5)
integrator.forces.append(bond)

# Integration settings
r_g = np.genfromtxt(f"r_g_{num}.txt")
gamma_wall_b = 1
gamma_wall_r = 4.0 / 3.0 * r_g**2 * gamma_wall_b
particles = hoomd.filter.Type(("A", "Wall"))
brownian = hoomd.md.methods.Brownian(kT=1.0, filter=particles)
brownian.gamma.default = 1.0
brownian.gamma["A"] = 1.0
brownian.gamma["Wall"] = gamma_wall_b
brownian.gamma_r["A"] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
brownian.gamma_r["Wall"] = [gamma_wall_r, gamma_wall_r, gamma_wall_r]
integrator.methods.append(brownian)
sim.operations.integrator = integrator

# Active force
active = hoomd.md.force.Active(filter=hoomd.filter.Type(["A"]))
active.active_force["A"] = (activity, 0, 0)
active.active_torque["A"] = (0, 0, torque)
integrator.forces.append(active)
rotational_diffusion_updater = active.create_diffusion_updater(
    trigger=1, rotational_diffusion=3.0
)
sim.operations += rotational_diffusion_updater

# Thermodynamic properties
pc = hoomd.filter.Rigid(("center", "free"))
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=pc)
sim.operations.computes.append(thermodynamic_properties)
sim.run(0)

# GSD writers
gsd_writer = hoomd.write.GSD(
    filename="trajectory_prod.gsd",
    trigger=hoomd.trigger.Periodic(1000000),
    mode="ab",
)
gsd_writer_restart = hoomd.write.GSD(
    filename="restart.gsd",
    trigger=hoomd.trigger.Periodic(10000),
    mode="wb",
    truncate=True,
)
sim.operations.writers.append(gsd_writer)
sim.operations.writers.append(gsd_writer_restart)

# Logger
logger = hoomd.logging.Logger()
logger.add(
    thermodynamic_properties,
    quantities=[
        "kinetic_energy",
        "potential_energy",
        "pressure",
        "pressure_tensor",
    ],
)
logger.add(sim, quantities=["timestep", "tps"])

logger_pos = hoomd.logging.Logger()
logger_pos.add(sim, quantities=["timestep"])


# Measure wall force
class WallForce:
    def __init__(self, sim, N):
        self.sim = sim
        self.N = N

    @property
    def GetWallForce0(self):
        with self.sim.state.cpu_local_snapshot as snap:
            wall_id = snap.particles.rtag[self.N + 0]
            return np.array(snap.particles.net_force[wall_id])

    @property
    def GetWallForce1(self):
        with self.sim.state.cpu_local_snapshot as snap:
            wall_id = snap.particles.rtag[self.N + 1]
            return np.array(snap.particles.net_force[wall_id])


# Output position and orientations of wall
class WallPosOrien:
    def __init__(self, sim, N):
        self.sim = sim
        self.N = N

    @property
    def GetWallPos0(self):
        with self.sim.state.cpu_local_snapshot as snap:
            wall_id = snap.particles.rtag[self.N + 0]
            return np.array(snap.particles.position[wall_id])

    @property
    def GetWallPos1(self):
        with self.sim.state.cpu_local_snapshot as snap:
            wall_id = snap.particles.rtag[self.N + 1]
            return np.array(snap.particles.position[wall_id])

    @property
    def GetWallOrien0(self):
        with self.sim.state.cpu_local_snapshot as snap:
            wall_id = snap.particles.rtag[self.N + 0]
            return np.array(snap.particles.orientation[wall_id])

    @property
    def GetWallOrien1(self):
        with self.sim.state.cpu_local_snapshot as snap:
            wall_id = snap.particles.rtag[self.N + 1]
            return np.array(snap.particles.orientation[wall_id])

    @property
    def GetWallDistance(self):
        with self.sim.state.cpu_local_snapshot as snap:
            wall_id_0 = snap.particles.rtag[self.N + 0]
            wall_id_1 = snap.particles.rtag[self.N + 1]
            Lx = snap.global_box.Lx
            Ly = snap.global_box.Ly
            wall_pos_0 = np.array(snap.particles.position[wall_id_0])
            wall_pos_1 = np.array(snap.particles.position[wall_id_1])
            separation = wall_pos_1 - wall_pos_0
            separation[0] = separation[0] - Lx * np.round(separation[0] / Lx)
            separation[1] = separation[1] - Ly * np.round(separation[1] / Ly)
            distance = (separation[0] ** 2 + separation[1] ** 2) ** 0.5
            return distance


wall_force = WallForce(sim, sim.state.N_particles - (num * 2 + 2))
wall_pos_orien = WallPosOrien(sim, sim.state.N_particles - (num * 2 + 2))
logger[("WallForce", "wall_force_0")] = (
    wall_force,
    "GetWallForce0",
    "sequence",
)
logger[("WallForce", "wall_force_1")] = (
    wall_force,
    "GetWallForce1",
    "sequence",
)
logger_pos[("WallPosOrien", "wall_pos_0")] = (
    wall_pos_orien,
    "GetWallPos0",
    "sequence",
)
logger_pos[("WallPosOrien", "wall_pos_1")] = (
    wall_pos_orien,
    "GetWallPos1",
    "sequence",
)
logger_pos[("WallPosOrien", "wall_orien_0")] = (
    wall_pos_orien,
    "GetWallOrien0",
    "sequence",
)
logger_pos[("WallPosOrien", "wall_orien_1")] = (
    wall_pos_orien,
    "GetWallOrien1",
    "sequence",
)
logger_pos[("WallPosOrien", "wall_distance")] = (
    wall_pos_orien,
    "GetWallDistance",
    "scalar",
)

gsd_writer_log = hoomd.write.GSD(
    filename="log_prod.gsd",
    trigger=hoomd.trigger.Periodic(10000),
    mode="ab",
    filter=hoomd.filter.Null(),
    log=logger,
)
gsd_writer_log_pos = hoomd.write.GSD(
    filename="log_prod_pos.gsd",
    trigger=hoomd.trigger.Periodic(1000),
    mode="ab",
    filter=hoomd.filter.Null(),
    log=logger_pos,
)
sim.operations.writers.append(gsd_writer_log)
sim.operations.writers.append(gsd_writer_log_pos)

# Run for long time
walltime_limit = 60 * 29  # time in seconds
while sim.device.communicator.walltime + sim.walltime < walltime_limit:
    sim.run(10000)
hoomd.write.GSD.write(state=sim.state, mode="wb", filename="restart.gsd")
