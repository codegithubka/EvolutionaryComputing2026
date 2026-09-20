"""Run the experiment grid (configs x seeds) in parallel.

Usage, from the repository root:
    uv run assignments/assignment_1/run_experiments.py --configs point subtree random --seeds 0-19
"""

# Standard library
import argparse
import multiprocessing as mp
import time
from pathlib import Path

# Third-party libraries
from rich.console import Console

# Local scripts
from ea_body import EAConfig, run_ea, run_random_search

console = Console()

# --- DEFAULTS --- #
CONFIG_NAMES: tuple[str, ...] = ("point", "subtree", "random")
DEFAULT_POP: int = 100
DEFAULT_GENS: int = 100
DEFAULT_SEEDS: str = "0-19"
DEFAULT_OUT: Path = Path("__data__") / "A1"


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


def make_config(name: str, pop: int, gens: int) -> EAConfig:
    """Resolved config for a named configuration. mu = lambda = ``pop``."""
    mutation = "subtree" if name == "subtree" else "point"
    return EAConfig(
        mutation=mutation,
        pop_size=pop,
        offspring_size=pop,
        generations=gens,
    )


def run_one(task: tuple[str, int, int, int, str]) -> tuple[str, int, float, float]:
    """Worker: run ONE (config, seed) task. Top-level so ``spawn`` can pickle it."""
    name, seed, pop, gens, out_root = task
    config = make_config(name, pop, gens)
    out_dir = Path(out_root) / name / f"seed_{seed:02d}"
    start = time.perf_counter()
    if name == "random":
        rows = run_random_search(config, seed, out_dir)
    else:
        rows = run_ea(config, seed, out_dir)
    return name, seed, rows[-1]["best"], time.perf_counter() - start


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
        default=list(CONFIG_NAMES),
    )
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="e.g. 0-19")
    parser.add_argument("--pop", type=int, default=DEFAULT_POP, help="mu = lambda")
    parser.add_argument("--gens", type=int, default=DEFAULT_GENS)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    tasks = [
        (name, seed, args.pop, args.gens, str(args.out))
        for name in args.configs
        for seed in seeds
    ]
    console.log(
        f"{len(tasks)} runs: configs={args.configs} seeds={seeds[0]}..{seeds[-1]} "
        f"pop={args.pop} gens={args.gens} workers={args.workers} out={args.out}",
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
