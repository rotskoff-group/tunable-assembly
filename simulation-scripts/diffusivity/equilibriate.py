import argparse

import hoomd

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()
seed = args.seed

communicator = hoomd.communicator.Communicator(ranks_per_partition=1)
gpu = hoomd.device.GPU(communicator=communicator)
sim = hoomd.Simulation(device=gpu, seed=seed + communicator.partition)
sim.create_state_from_gsd(filename="lattice.gsd")

integrator = hoomd.md.Integrator(dt=0.00001, integrate_rotational_dof=True)
cell = hoomd.md.nlist.Cell(buffer=0.4)

# WCA potential
lj = hoomd.md.pair.LJ(nlist=cell, default_r_cut=2 ** (1.0 / 6.0), mode="shift")
lj.params[("A", "A")] = dict(epsilon=40.0, sigma=1)
integrator.forces.append(lj)

# Integration settings
# Use NVE with fixed displacement initially
particles = hoomd.filter.Type(["A"])
displacement = hoomd.md.methods.DisplacementCapped(
    filter=particles, maximum_displacement=0.05
)
integrator.methods.append(displacement)
sim.operations.integrator = integrator

# Set temperature
sim.state.thermalize_particle_momenta(filter=particles, kT=1.0)

# Active force
active = hoomd.md.force.Active(filter=hoomd.filter.Type(["A"]))
active.active_force["A"] = (1e-15, 0, 0)
active.active_torque["A"] = (0, 0, 0)
integrator.forces.append(active)
rotational_diffusion_updater = active.create_diffusion_updater(
    trigger=1, rotational_diffusion=100.0
)
sim.operations += rotational_diffusion_updater

# Thermodynamic properties
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All()
)
sim.operations.computes.append(thermodynamic_properties)
hoomd.write.GSD.write(
    state=sim.state, mode="wb", filename="lattice_thermalize.gsd"
)
sim.run(0)

# GSD
gsd_writer = hoomd.write.GSD(
    filename="trajectory_equil.gsd",
    trigger=hoomd.trigger.Periodic(200000),
    mode="ab",
    dynamic=["property", "momentum"],
)
gsd_writer_restart = hoomd.write.GSD(
    filename="restart.gsd",
    trigger=hoomd.trigger.Periodic(10000),
    mode="wb",
    truncate=True,
    dynamic=["property", "momentum"],
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
while sim.device.communicator.walltime + sim.walltime < 1.5 * walltime_limit:
    sim.run(10000)
    sim.state.thermalize_particle_momenta(filter=particles, kT=1.0)

# Now switch to a Brownian integrator
brownian = hoomd.md.methods.Brownian(kT=1.0, filter=particles)
brownian.gamma.default = 1.0
brownian.gamma["A"] = 1.0
brownian.gamma_r["A"] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
integrator.methods.remove(displacement)
integrator.methods.append(brownian)
# sim.operations.integrator = integrator

# Equilibriate for long time
while sim.device.communicator.walltime + sim.walltime < 3 * walltime_limit:
    sim.run(10000)
