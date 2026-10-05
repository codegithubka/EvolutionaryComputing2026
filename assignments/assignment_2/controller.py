"""Fixed experimental setup for A2: world, body, NN controller and fitness.

Every configuration (mut / xmut / xrand / random) imports this module, so the
body, world, controller inputs, simulation length and fitness function are
identical across everything we compare. Only ``Setup`` values change the task,
and the frozen defaults below are the ones used for the final experiments.

Evaluation does not use ``mujoco.set_mjcb_control``: ``simulate`` steps the
physics itself and writes ``data.ctrl`` between steps, so there is no global
callback state and evaluations are deterministic per genotype.
"""

# Standard library
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt

# Local libraries (ARIEL)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.simulation.environments import (
    AmphitheatreTerrainWorld,
    CraterTerrainWorld,
    OlympicArena,
    RuggedTerrainWorld,
    SimpleFlatWorld,
)
from ariel.simulation.tasks.targeted_locomotion import distance_to_target

# Type aliases
type FloatArray = npt.NDArray[np.float64]

# --- REGISTRIES --- #
BODIES: dict[str, Callable[[], CoreModule]] = {
    name: getattr(john_set, name)
    for name in (
        "baby_a",
        "baby_b",
        "gecko",
        "linkin_modified",
        "snake",
        "turtle",
        "iguana",
        "spider_8",
        "spider_12",
        "spider_16",
        "centipede_3",
        "centipede_4",
        "centipede_5",
    )
}
# SimpleTiltedWorld is excluded by the brief.
WORLDS: dict[str, Callable[[], object]] = {
    "SimpleFlatWorld": SimpleFlatWorld,
    "RuggedTerrainWorld": RuggedTerrainWorld,
    "CraterTerrainWorld": CraterTerrainWorld,
    "AmphitheatreTerrainWorld": AmphitheatreTerrainWorld,
    "OlympicArena": OlympicArena,
}

HINGE_LIMIT: float = np.pi / 2  # actuator ctrlrange is [-pi/2, pi/2]
EPS: float = 1e-6


# ============================================================================ #
#  1. THE SETUP
# ============================================================================ #


@dataclass(frozen=True)
class Setup:
    """Everything that defines the task. Frozen after the pilot."""

    body: str = "spider_8"  # best learner in the body pilot after linkin_modified
    world: str = "SimpleFlatWorld"
    spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.1)
    # Off-axis and ~3.6 m away: the robot must steer as well as walk, and the
    # 2 m straight-ahead target was nearly solved within 520 pilot evaluations.
    target_xy: tuple[float, float] = (3.0, 2.0)
    sim_duration: float = 15.0  # seconds of simulated time
    control_every: int = 1  # physics steps per controller update (dt = 2 ms)
    hidden_size: int = 6
    osc_freq_hz: float = 1.0  # fixed clock input, never evolved
    dist_scale: float = 4.0  # distance input = min(dist, scale) / scale

    @property
    def target(self) -> FloatArray:
        """Target position as an (x, y, z) array for the fitness function."""
        return np.array([*self.target_xy, 0.0])


@dataclass(frozen=True)
class Compiled:
    """A compiled world + robot and the addresses the controller reads."""

    model: mj.MjModel
    root_qadr: int  # qpos address of the robot's free joint (x, y, z, qw..qz)
    hinge_qadr: npt.NDArray[np.int64]  # qpos address of each actuated hinge

    @property
    def num_inputs(self) -> int:
        """Hinge angles + target direction (2) + distance + sin + cos + bias."""
        return len(self.hinge_qadr) + 6

    @property
    def num_outputs(self) -> int:
        """One command per actuated hinge."""
        return int(self.model.nu)


@cache
def compile_setup(
    body: str,
    world: str,
    spawn_pos: tuple[float, float, float],
) -> Compiled:
    """Build and compile the world with the robot in it (once per process)."""
    mj.set_mjcb_control(None)
    env = WORLDS[world]()
    robot = BODIES[body]()
    env.spawn(robot.spec, position=list(spawn_pos), correct_collision_with_floor=True)
    model = env.spec.compile()

    free = [j for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
    assert len(free) == 1, f"expected one free joint (the robot), got {len(free)}"
    hinge_joints = model.actuator_trnid[:, 0]
    assert all(model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE for j in hinge_joints)
    assert np.allclose(model.actuator_ctrlrange, [-HINGE_LIMIT, HINGE_LIMIT])
    return Compiled(
        model=model,
        root_qadr=int(model.jnt_qposadr[free[0]]),
        hinge_qadr=model.jnt_qposadr[hinge_joints].astype(np.int64),
    )


def compiled(setup: Setup) -> Compiled:
    """Cached compiled model for ``setup``."""
    return compile_setup(setup.body, setup.world, setup.spawn_pos)


# ============================================================================ #
#  2. THE CONTROLLER
# ============================================================================ #


def genotype_length(setup: Setup) -> int:
    """Number of weights: inputs x hidden + hidden x outputs (bias is an input)."""
    c = compiled(setup)
    return c.num_inputs * setup.hidden_size + setup.hidden_size * c.num_outputs


def decode(genotype: Sequence[float], setup: Setup) -> tuple[FloatArray, FloatArray]:
    """Reshape a flat genotype into the two weight matrices ``(w1, w2)``."""
    c = compiled(setup)
    flat = np.asarray(genotype, dtype=np.float64)
    assert flat.shape == (genotype_length(setup),), flat.shape
    split = c.num_inputs * setup.hidden_size
    w1 = flat[:split].reshape(c.num_inputs, setup.hidden_size)
    w2 = flat[split:].reshape(setup.hidden_size, c.num_outputs)
    return w1, w2


def yaw_from_quat(quat: FloatArray) -> float:
    """Heading angle of a MuJoCo (w, x, y, z) quaternion around world z."""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def sensory_inputs(data: mj.MjData, c: Compiled, setup: Setup) -> FloatArray:
    """Controller input vector, every entry roughly in [-1, 1].

    - hinge angles / (pi/2)
    - unit vector to the target in the robot's heading frame (2)
    - distance to the target, clipped to ``dist_scale`` and normalised
    - sin and cos of a fixed-frequency clock (rhythm for a gait; not a CPG,
      it has no evolved parameters)
    - constant bias input 1
    """
    qpos = data.qpos
    root = c.root_qadr
    dx = setup.target_xy[0] - qpos[root]
    dy = setup.target_xy[1] - qpos[root + 1]
    dist = float(np.hypot(dx, dy))
    yaw = yaw_from_quat(qpos[root + 3 : root + 7])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    local_x = cos_y * dx + sin_y * dy
    local_y = -sin_y * dx + cos_y * dy
    phase = 2.0 * np.pi * setup.osc_freq_hz * data.time
    return np.concatenate(
        (
            qpos[c.hinge_qadr] / HINGE_LIMIT,
            (
                local_x / max(dist, EPS),
                local_y / max(dist, EPS),
                min(dist, setup.dist_scale) / setup.dist_scale,
                np.sin(phase),
                np.cos(phase),
                1.0,
            ),
        ),
    )


def nn_forward(inputs: FloatArray, w1: FloatArray, w2: FloatArray) -> FloatArray:
    """inputs -> tanh hidden -> tanh outputs, scaled to the hinge range."""
    return np.tanh(np.tanh(inputs @ w1) @ w2) * HINGE_LIMIT


# ============================================================================ #
#  3. EVALUATION
# ============================================================================ #


@dataclass(frozen=True)
class SimResult:
    """Fitness plus behaviour diagnostics (logged only, not part of fitness)."""

    fitness: float  # planar distance to the target at the end (minimised)
    start_distance: float
    final_x: float
    final_y: float
    min_z: float  # lowest core height reached (flip / collapse indicator)
    path_length: float  # planar distance travelled by the core


def simulate(genotype: Sequence[float], setup: Setup) -> SimResult:
    """Run one headless episode with DIRECT control and score it."""
    c = compiled(setup)
    model = c.model
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    w1, w2 = decode(genotype, setup)

    root = c.root_qadr
    start = data.qpos[root : root + 3].copy()
    prev_xy = start[:2].copy()
    min_z = float(start[2])
    path_length = 0.0

    n_updates = round(setup.sim_duration / (model.opt.timestep * setup.control_every))
    for _ in range(n_updates):
        actions = nn_forward(sensory_inputs(data, c, setup), w1, w2)
        assert np.all(np.isfinite(actions)), "non-finite controller output"
        data.ctrl[:] = actions
        mj.mj_step(model, data, nstep=setup.control_every)
        xy = data.qpos[root : root + 2]
        path_length += float(np.hypot(*(xy - prev_xy)))
        prev_xy = xy.copy()
        min_z = min(min_z, float(data.qpos[root + 2]))

    final = data.qpos[root : root + 3].copy()
    assert np.all(np.isfinite(final)), "simulation diverged"
    return SimResult(
        fitness=distance_to_target(final, setup.target),
        start_distance=distance_to_target(start, setup.target),
        final_x=float(final[0]),
        final_y=float(final[1]),
        min_z=min_z,
        path_length=path_length,
    )


def random_genotype(rng: np.random.Generator, setup: Setup, std: float = 0.5) -> list[float]:
    """N(0, std) weights; used by pilots and sanity checks outside ariel.ec."""
    return rng.normal(0.0, std, genotype_length(setup)).tolist()
