import argparse

import hoomd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--activity", type=float, default=80)
parser.add_argument("--torque", type=float, default=5)

args = parser.parse_args()
seed = args.seed
activity = args.activity
torque = args.torque

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
brownian = hoomd.md.methods.Brownian(kT=1.0, filter=hoomd.filter.Type(["A"]))
brownian.gamma.default = 1.0
brownian.gamma["A"] = 1.0
brownian.gamma_r["A"] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
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
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All()
)
sim.operations.computes.append(thermodynamic_properties)
sim.run(0)

# GSD
gsd_writer = hoomd.write.GSD(
    filename="trajectory.gsd",
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


# Measure swim pressure
class SwimPressure:
    def __init__(self, sim, active_force, gamma, gamma_r, position_old=None):
        self.sim = sim
        self.active_force = active_force
        self.gamma = gamma
        self.gamma_r = gamma_r

    @property
    def GetSwimPressure(self):
        with self.sim.state.cpu_local_snapshot as snap:
            with self.active_force.cpu_local_force_arrays as active_force:
                return (
                    0.5
                    * self.gamma
                    / self.gamma_r
                    * np.sum(snap.particles.net_force * active_force.force)
                    / snap.local_box.volume
                )


swim_pressure = SwimPressure(sim, active, gamma=1.0, gamma_r=3.0)
logger[("SwimPressure", "swim_pressure")] = (
    swim_pressure,
    "GetSwimPressure",
    "scalar",
)

gsd_writer_log = hoomd.write.GSD(
    filename="log.gsd",
    trigger=hoomd.trigger.Periodic(10000),
    mode="ab",
    filter=hoomd.filter.Null(),
    log=logger,
)
sim.operations.writers.append(gsd_writer_log)

# Run for long time
walltime_limit = 60 * 29  # time in seconds
while sim.device.communicator.walltime + sim.walltime < walltime_limit:
    sim.run(10000)
