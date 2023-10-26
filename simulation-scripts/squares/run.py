import argparse

import freud
import hoomd
import numpy as np

from rigid_structure import generate_rigid_positions

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=int, default=0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num", type=int, default=10)
parser.add_argument("--torque", type=float, default=5.0)
parser.add_argument("--activity", type=float, default=20.0)

args = parser.parse_args()
base = args.base
seed = args.seed
num = args.num
torque = args.torque
activity = args.activity

communicator = hoomd.communicator.Communicator(ranks_per_partition=1)
gpu = hoomd.device.GPU(communicator=communicator)
sim = hoomd.Simulation(device=gpu, seed=seed + communicator.partition)
sim.create_state_from_gsd(filename="restart.gsd")
wall, num2 = generate_rigid_positions(num, 1)

rigid = hoomd.md.constrain.Rigid()
rigid.body["Wall"] = {
    "constituent_types": ["Wall_0" for i in range(num2)],
    "positions": wall,
    "orientations": [(0.0, 0.0, 0.0, 1.0) for i in range(num2)],
    "charges": [0.0 for i in range(num2)],
    "diameters": [1.0 for i in range(num2)],
}

# Integrator
integrator = hoomd.md.Integrator(dt=0.000005, integrate_rotational_dof=True)
integrator.rigid = rigid
cell = hoomd.md.nlist.Cell(buffer=0.4, exclusions=["body"])

# WCA potential
lj = hoomd.md.pair.LJ(nlist=cell, default_r_cut=2 ** (1.0 / 6.0), mode="shift")
lj.params[(["A", "Wall_0"], ["A", "Wall_0"])] = dict(epsilon=40.0, sigma=1)
lj.r_cut[(["A", "Wall_0"], ["A", "Wall_0"])] = 2 ** (1.0 / 6.0)
lj.params[("Wall", ["A", "Wall_0", "Wall"])] = dict(epsilon=0.0, sigma=0)
lj.r_cut[("Wall", ["A", "Wall_0", "Wall"])] = 0
integrator.forces.append(lj)

# Integration settings
particles = hoomd.filter.Type(("A", "Wall"))
r_g = np.genfromtxt(f"r_g_{num}.txt")
gamma_wall_b = 1
gamma_wall_r = 4.0 / 3.0 * r_g**2 * gamma_wall_b
brownian = hoomd.md.methods.Brownian(kT=1.0, filter=particles)
brownian.gamma.default = 1.0
brownian.gamma["A"] = 1.0
brownian.gamma["Wall"] = gamma_wall_b
brownian.gamma_r["A"] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
brownian.gamma_r["Wall"] = [gamma_wall_r, gamma_wall_r, gamma_wall_r]
integrator.methods.append(brownian)
sim.operations.integrator = integrator

# Active force
active = hoomd.md.force.Active(
    filter=hoomd.filter.Type(["A", "Wall", "Wall_0"])
)
active.active_force["A"] = (activity, 0, 0)
active.active_force["Wall"] = (0, 0, 0)
active.active_force["Wall_0"] = (0, 0, 0)
active.active_torque["A"] = (0, 0, torque)
active.active_torque["Wall"] = (0, 0, 0)
active.active_torque["Wall_0"] = (0, 0, 0)
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
    trigger=hoomd.trigger.Periodic(200000),
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


# Measure Voronoi volumes
class VoronoiVolume:
    def __init__(self, sim):
        self.sim = sim
        self.box = sim.state.box
        self.voro = freud.locality.Voronoi()

    @property
    def GetVoronoiVolume(self):
        with self.sim.state.cpu_local_snapshot as snap:
            tags = snap.particles.typeid == 1
            positions = snap.particles.position[tags]
            self.voro.compute((self.box, positions))
            return self.voro.volumes


voronoi_volume = VoronoiVolume(sim)
logger[("VoronoiVolume", "voronoi_volume")] = (
    voronoi_volume,
    "GetVoronoiVolume",
    "sequence",
)

gsd_writer_log = hoomd.write.GSD(
    filename="log_prod.gsd",
    trigger=hoomd.trigger.Periodic(10000),
    mode="ab",
    filter=hoomd.filter.Null(),
    log=logger,
)
sim.operations.writers.append(gsd_writer_log)


walltime_limit = 60 * 29
while sim.device.communicator.walltime + sim.walltime < walltime_limit:
    sim.run(10000)
hoomd.write.GSD.write(state=sim.state, mode="wb", filename="restart.gsd")
