"""Run the experiment grid (configs x seeds) in parallel.

Usage, from the repository root:
    uv run assignments/assignment_2/run_experiments.py --configs mut xmut --seeds 0-14
    uv run assignments/assignment_2/run_experiments.py --configs xrand random --seeds 0-9
"""

# Standard library
import argparse
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

# Third-party libraries
from rich.console import Console

# Local scripts
from controller import BODIES, WORLDS, Setup
from ea_neuro import VARIATIONS, EAConfig, run_ea, run_random_search

console = Console()

# --- DEFAULTS --- #
CONFIG_NAMES: tuple[str, ...] = (*VARIATIONS, "random")
DEFAULTS = EAConfig()
DEFAULT_SEEDS: str = "0-9"
DEFAULT_OUT: Path = Path("__data__") / "A2"


# ============================================================================ #
#  Helpers
# ============================================================================ #


def parse_seeds(spec: str) -> list[int]:
    """Parse ``"0-19"``, ``"3"`` or ``"0,2,5-7"`` into a sorted seed list."""
    seeds: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.update(range(int(lo), int(hi) + 1))
        elif part:
            seeds.add(int(part))
    return sorted(seeds)


def make_config(name: str, args: argparse.Namespace) -> EAConfig:
    """Resolved config for a named configuration. mu = lambda = ``--pop``."""
    setup = replace(
        Setup(),
        body=args.body,
        world=args.world,
        sim_duration=args.duration,
    )
    return EAConfig(
        variation="mut" if name == "random" else name,
        pop_size=args.pop,
        offspring_size=args.pop,
        generations=args.gens,
        sigma=args.sigma,
        mutation_probability=args.pm,
        setup=setup,
    )


def run_one(task: tuple[str, int, argparse.Namespace]) -> tuple[str, int, float, float]:
    """Worker: run ONE (config, seed) task. Top-level so ``spawn`` can pickle it."""
    name, seed, args = task
    config = make_config(name, args)
    out_dir = Path(args.out) / name / f"seed_{seed:02d}"
    start = time.perf_counter()
    if name == "random":
        rows = run_random_search(config, seed, out_dir)
    else:
        rows = run_ea(config, seed, out_dir)
    return name, seed, rows[-1]["best_so_far"], time.perf_counter() - start


# ============================================================================ #
#  Entry point
# ============================================================================ #


def main() -> None:
    """Parse arguments and run every (config, seed) task in a process pool."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=CONFIG_NAMES,
        default=["mut", "xmut", "xrand", "random"],
    )
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="e.g. 0-14")
    parser.add_argument("--pop", type=int, default=DEFAULTS.pop_size, help="mu = lambda")
    parser.add_argument("--gens", type=int, default=DEFAULTS.generations)
    parser.add_argument("--sigma", type=float, default=DEFAULTS.sigma)
    parser.add_argument("--pm", type=float, default=DEFAULTS.mutation_probability)
    parser.add_argument("--body", choices=sorted(BODIES), default=DEFAULTS.setup.body)
    parser.add_argument("--world", choices=sorted(WORLDS), default=DEFAULTS.setup.world)
    parser.add_argument("--duration", type=float, default=DEFAULTS.setup.sim_duration)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    tasks = [(name, seed, args) for name in args.configs for seed in seeds]
    console.log(
        f"{len(tasks)} runs: configs={args.configs} seeds={seeds[0]}..{seeds[-1]} "
        f"pop={args.pop} gens={args.gens} sigma={args.sigma} pm={args.pm} "
        f"body={args.body} world={args.world} duration={args.duration} "
        f"workers={args.workers} out={args.out}",
    )

    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for name, seed, best, secs in pool.imap_unordered(run_one, tasks):
            console.log(
                f"{name:8s} seed {seed:2d}  final best {best:.4f}  ({secs:.1f}s)"
            )
    console.log(f"done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
