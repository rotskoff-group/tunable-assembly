import argparse

import hoomd
import numpy as np

from rigid_structure import generate_rigid_positions

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=int, default=0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num", type=int, default=10)

args = parser.parse_args()
base = args.base
seed = args.seed
num = args.num

communicator = hoomd.communicator.Communicator(ranks_per_partition=1)
gpu = hoomd.device.GPU(communicator=communicator)
sim = hoomd.Simulation(device=gpu, seed=seed + communicator.partition)
sim.create_state_from_gsd(filename="lattice_rigid.gsd")
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
integrator = hoomd.md.Integrator(dt=0.00001, integrate_rotational_dof=True)
integrator.rigid = rigid
cell = hoomd.md.nlist.Tree(buffer=0.4, exclusions=["body"])

# WCA potential
lj = hoomd.md.pair.LJ(nlist=cell, default_r_cut=2 ** (1.0 / 6.0), mode="shift")
lj.params[(["A", "Wall_0"], ["A", "Wall_0"])] = dict(epsilon=40.0, sigma=1)
lj.r_cut[(["A", "Wall_0"], ["A", "Wall_0"])] = 2 ** (1.0 / 6.0)
lj.params[("Wall", ["A", "Wall_0", "Wall"])] = dict(epsilon=0.0, sigma=0)
lj.r_cut[("Wall", ["A", "Wall_0", "Wall"])] = 0
integrator.forces.append(lj)

# Integration settings
# Use NVE with fixed displacement initially
particles = hoomd.filter.Type(("A", "Wall"))
displacement = hoomd.md.methods.DisplacementCapped(
    filter=particles, maximum_displacement=0.01
)
integrator.methods.append(displacement)
sim.operations.integrator = integrator

# Set temperature
sim.state.thermalize_particle_momenta(filter=particles, kT=1.0)

# Active force
active = hoomd.md.force.Active(
    filter=hoomd.filter.Type(["A", "Wall", "Wall_0"])
)
active.active_force["A"] = (1e-15, 0, 0)
active.active_force["Wall"] = (0, 0, 0)
active.active_force["Wall_0"] = (0, 0, 0)
active.active_torque["A"] = (0, 0, 0)
active.active_torque["Wall"] = (0, 0, 0)
active.active_torque["Wall_0"] = (0, 0, 0)
integrator.forces.append(active)
rotational_diffusion_updater = active.create_diffusion_updater(
    trigger=1, rotational_diffusion=100.0
)
sim.operations += rotational_diffusion_updater

# Thermodynamic properties
pc = hoomd.filter.Rigid(("center", "free"))
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=pc)
sim.operations.computes.append(thermodynamic_properties)
hoomd.write.GSD.write(
    state=sim.state, mode="wb", filename="lattice_thermalize.gsd"
)
sim.run(0)

# GSD writers
gsd_writer = hoomd.write.GSD(
    filename="trajectory_equil.gsd",
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
gsd_writer_log = hoomd.write.GSD(
    filename="log_equil.gsd",
    trigger=hoomd.trigger.Periodic(10000),
    mode="ab",
    filter=hoomd.filter.Null(),
    log=logger,
)
sim.operations.writers.append(gsd_writer_log)

walltime_limit = 9 * 60

# Equilibriate for long time
rho = sim.state.N_particles / sim.state.box.volume
initial_box = sim.state.box
final_box = hoomd.Box.from_box(initial_box)
final_rho = 20**2 * rho
final_box.volume = sim.state.N_particles / final_rho
box_resize_trigger = hoomd.trigger.Periodic(10)
ramp = hoomd.variant.Power(
    A=0, B=1, power=0.1, t_start=sim.timestep, t_ramp=2000000
)
box_resize = hoomd.update.BoxResize(
    box1=initial_box, box2=final_box, variant=ramp, trigger=box_resize_trigger
)
sim.operations.updaters.append(box_resize)

while sim.device.communicator.walltime + sim.walltime < 1.5 * walltime_limit:
    sim.run(10000)
    sim.state.thermalize_particle_momenta(filter=particles, kT=1.0)

# Now switch to a Brownian integrator
r_g = np.genfromtxt(f"r_g_{num}.txt")
gamma_wall_b = 1
gamma_wall_r = 4.0 / 3.0 * r_g**2 * gamma_wall_b
brownian = hoomd.md.methods.Brownian(kT=1.0, filter=particles)
brownian.gamma.default = 1.0
brownian.gamma["A"] = 1.0
brownian.gamma["Wall"] = gamma_wall_b
brownian.gamma_r["A"] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
brownian.gamma_r["Wall"] = [gamma_wall_r, gamma_wall_r, gamma_wall_r]
integrator.methods.remove(displacement)
integrator.methods.append(brownian)

# Equilibriate for long time
while sim.device.communicator.walltime + sim.walltime < 3 * walltime_limit:
    sim.run(10000)
hoomd.write.GSD.write(state=sim.state, mode="wb", filename="restart.gsd")
