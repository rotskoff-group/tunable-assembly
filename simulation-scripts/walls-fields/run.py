import argparse

import freud
import hoomd
import numpy as np

from rigid_structure import generate_rigid_positions

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=int, default=0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num", type=int, default=9)
parser.add_argument("--activity", type=float, default=80)
parser.add_argument("--torque", type=float, default=5)

args = parser.parse_args()
base = args.base
seed = args.seed
num = args.num
activity = args.activity
torque = args.torque

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
integrator = hoomd.md.Integrator(dt=0.00001, integrate_rotational_dof=True)
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
particles = hoomd.filter.Type(("A"))
brownian = hoomd.md.methods.Brownian(kT=1.0, filter=particles)
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


def evaluate_euler_angles(orientation):
    q0 = orientation[:, 0]
    q1 = orientation[:, 1]
    q2 = orientation[:, 2]
    q3 = orientation[:, 3]
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    pitch = np.arcsin(2 * (q0 * q2 - q3 * q1))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return roll, pitch, yaw


class FieldAnalyzer(hoomd.custom.Action):
    def __init__(self, freq=10000, width=420, subregion=26, skin=2):
        self.freq = freq
        self.count_int = 0
        self.width = width
        self.skin = skin
        self.subregion = subregion
        self.box_subregion = np.array(
            [2 * subregion + 2 * skin, 2 * subregion + 2 * skin, 0, 0, 0, 0]
        )
        self.gaussian_density = freud.density.GaussianDensity(
            width=[width, width], r_max=0.5, sigma=1
        )
        self.densities = []
        self.cos_densities = []
        self.sin_densities = []
        self.flux_x = []
        self.flux_y = []

    def attach(self, simulation):
        self._state = simulation.state
        self._comm = simulation.device.communicator

    def detach(self):
        del self._state
        del self._comm

    def act(self, timestep):
        with self._state.cpu_local_snapshot as snap:
            # Filter out the particles in the subregion
            pos = snap.particles.position
            vel = snap.particles.velocity
            orien = snap.particles.orientation
            types = snap.particles.typeid
            mask = (
                (pos[:, 0] > -self.subregion)
                & (pos[:, 0] <= self.subregion)
                & (pos[:, 1] > -self.subregion)
                & (pos[:, 1] <= self.subregion)
                & (types == 0)
            )
            pos = pos[mask]
            vel = vel[mask]
            orien = orien[mask]
            # Orientation stuff
            roll, pitch, yaw = evaluate_euler_angles(orien)
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            # Now to compute fields
            nq = freud.locality.AABBQuery(box=self.box_subregion, points=pos)
            # Now compute stuff
            self.gaussian_density.compute(system=nq)
            if self.count_int == 0:
                self.densities = np.copy(self.gaussian_density.density)
            else:
                self.densities += self.gaussian_density.density
            self.gaussian_density.compute(system=nq, values=cos_yaw)
            if self.count_int == 0:
                self.cos_densities = np.copy(self.gaussian_density.density)
            else:
                self.cos_densities += self.gaussian_density.density
            self.gaussian_density.compute(system=nq, values=sin_yaw)
            if self.count_int == 0:
                self.sin_densities = np.copy(self.gaussian_density.density)
            else:
                self.sin_densities += self.gaussian_density.density
            self.gaussian_density.compute(system=nq, values=vel[:, 0])
            if self.count_int == 0:
                self.flux_x = np.copy(self.gaussian_density.density)
            else:
                self.flux_x += self.gaussian_density.density
            self.gaussian_density.compute(system=nq, values=vel[:, 1])
            if self.count_int == 0:
                self.flux_y = np.copy(self.gaussian_density.density)
            else:
                self.flux_y += self.gaussian_density.density
            self.count_int += 1

    def output(self):
        # Average densities
        self.densities = np.array(self.densities) / self.count_int
        density_mean = np.mean(self.densities)
        self.cos_densities = np.array(self.cos_densities)
        self.sin_densities = np.array(self.sin_densities)
        orientation_average = np.arctan2(
            self.sin_densities, self.cos_densities
        )
        self.flux_x = np.array(self.flux_x) / self.count_int
        self.flux_y = np.array(self.flux_y) / self.count_int
        # Output
        np.save("densities.npy", self.densities)
        np.save("density_mean.npy", density_mean)
        np.save("cos.npy", self.cos_densities)
        np.save("sin.npy", self.sin_densities)
        np.save("orientation.npy", orientation_average)
        np.save("flux_x.npy", self.flux_x)
        np.save("flux_y.npy", self.flux_y)


wall_force = WallForce(sim, sim.state.N_particles - (num * 2 + 2))
logger[("WallForce", "WallForce0")] = (
    wall_force,
    "GetWallForce0",
    "sequence",
)
logger[("WallForce", "WallForce1")] = (
    wall_force,
    "GetWallForce1",
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

field_analyzer = FieldAnalyzer(freq=2000)
field_operation = hoomd.update.CustomUpdater(
    action=field_analyzer, trigger=hoomd.trigger.Periodic(field_analyzer.freq)
)
sim.operations.updaters.append(field_operation)

# Run for long time
walltime_limit = 60 * 29  # time in seconds
while sim.device.communicator.walltime + sim.walltime < walltime_limit:
    sim.run(10000)
hoomd.write.GSD.write(state=sim.state, mode="wb", filename="restart.gsd")
field_analyzer.output()
