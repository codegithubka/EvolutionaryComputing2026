"""Replay a run's best controller in the viewer or render it to video.

Usage, from the repository root:
    uv run assignments/assignment_2/replay_best.py __data__/A2/mut/seed_00            # viewer
    uv run assignments/assignment_2/replay_best.py __data__/A2/mut/seed_00 --video    # mp4

The headless score is re-checked first, so the replay is the controller that
was actually evaluated.
"""

# Standard library
import argparse
import json
from pathlib import Path

# Third-party libraries
import mujoco as mj
from mujoco import viewer
from rich.console import Console

# Local scripts
from controller import Setup, compiled, decode, nn_forward, sensory_inputs, simulate

# Local libraries (ARIEL)
from ariel.utils.renderers import tracking_video_renderer
from ariel.utils.video_recorder import VideoRecorder

console = Console()


def load_run(run_dir: Path) -> tuple[Setup, list[float], float]:
    """Setup, best genotype and its logged fitness from a run directory."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    setup_dict = config["config"]["setup"]
    setup_dict["spawn_pos"] = tuple(setup_dict["spawn_pos"])
    setup_dict["target_xy"] = tuple(setup_dict["target_xy"])
    best = json.loads((run_dir / "best_genome.json").read_text(encoding="utf-8"))
    return Setup(**setup_dict), best["genotype"], best["fitness"]


def main() -> None:
    """Re-score the best controller, then show or record it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--video", action="store_true", help="render an mp4 instead")
    args = parser.parse_args()

    setup, genotype, logged = load_run(args.run_dir)
    rescored = simulate(genotype, setup).fitness
    console.log(f"logged fitness {logged:.4f}, re-simulated {rescored:.4f}")
    assert abs(rescored - logged) < 1e-9, "replay does not match the logged fitness"

    c = compiled(setup)
    model = c.model
    data = mj.MjData(model)
    w1, w2 = decode(genotype, setup)

    def control(m: mj.MjModel, d: mj.MjData) -> None:
        """Same DIRECT control as ``simulate``, updated every control_every steps."""
        if round(d.time / m.opt.timestep) % setup.control_every == 0:
            d.ctrl[:] = nn_forward(sensory_inputs(d, c, setup), w1, w2)

    mj.set_mjcb_control(control)
    try:
        if args.video:
            recorder = VideoRecorder(
                file_name=f"{args.run_dir.parent.name}_{args.run_dir.name}",
                output_folder=args.run_dir,
            )
            tracking_video_renderer(
                model,
                data,
                duration=setup.sim_duration,
                video_recorder=recorder,
                geom_to_track="robot1_core",
            )
        else:
            mj.mj_resetData(model, data)
            viewer.launch(model=model, data=data)
    finally:
        mj.set_mjcb_control(None)


if __name__ == "__main__":
    main()
