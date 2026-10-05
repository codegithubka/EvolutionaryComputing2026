"""Pilot runs that fix the free parameters before the final experiments.

Pilot seeds start at 100 and are never reused in the final runs.

Usage, from the repository root:
    uv run assignments/assignment_2/pilot.py bodies          # short mut run on every John Set body
    uv run assignments/assignment_2/pilot.py sigma --body gecko --gens 50
"""

# Standard library
import argparse
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

# Third-party libraries
import numpy as np
from rich.console import Console
from rich.table import Table

# Local scripts
from controller import BODIES, WORLDS, Setup
from ea_neuro import EAConfig, run_ea

console = Console()

PILOT_SEEDS: list[int] = [100, 101, 102]
DEFAULT_OUT: Path = Path("__data__") / "A2_pilot"


def run_one(task: tuple[str, EAConfig, int, Path]) -> tuple[str, int, float, float, float]:
    """Worker: one pilot EA run -> (label, seed, start best, final best, seconds)."""
    label, config, seed, out_dir = task
    start = time.perf_counter()
    rows = run_ea(config, seed, out_dir)
    return label, seed, rows[0]["best"], rows[-1]["best_so_far"], time.perf_counter() - start


def run_pool(tasks: list[tuple[str, EAConfig, int, Path]], workers: int) -> dict[str, list]:
    """Run the tasks in a spawn pool and group results by label."""
    results: dict[str, list] = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
        for label, seed, first, final, secs in pool.imap_unordered(run_one, tasks):
            console.log(f"{label:24s} seed {seed}  {first:.3f} -> {final:.3f}  ({secs:.0f}s)")
            results.setdefault(label, []).append((first, final, secs))
    return results


def report(results: dict[str, list], title: str) -> None:
    """Table of mean start/final best and improvement per label."""
    table = Table(title=title)
    for col in ("config", "start best", "final best (mean ± std)", "improvement", "s/run"):
        table.add_column(col)
    rows = []
    for label, vals in results.items():
        first, final, secs = (np.asarray(v) for v in zip(*vals, strict=True))
        rows.append((final.mean(), label, first.mean(), final.std(), (first - final).mean(), secs.mean()))
    for final_mean, label, first_mean, final_std, gain, secs in sorted(rows):
        table.add_row(
            label,
            f"{first_mean:.3f}",
            f"{final_mean:.3f} ± {final_std:.3f}",
            f"{gain:.3f}",
            f"{secs:.0f}",
        )
    console.print(table)


def main() -> None:
    """Parse arguments and run the chosen pilot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pilot", choices=["bodies", "sigma"])
    parser.add_argument("--body", choices=sorted(BODIES), default=Setup().body)
    parser.add_argument("--world", choices=sorted(WORLDS), default=Setup().world)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=2, help="how many pilot seeds")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    seeds = PILOT_SEEDS[: args.seeds]
    base = EAConfig(pop_size=args.pop, offspring_size=args.pop, generations=args.gens)
    tasks: list[tuple[str, EAConfig, int, Path]] = []
    if args.pilot == "bodies":
        for body in BODIES:
            config = replace(base, setup=replace(base.setup, body=body, world=args.world))
            for seed in seeds:
                tasks.append((body, config, seed, args.out / "bodies" / body / f"seed_{seed}"))
    else:
        setup = replace(base.setup, body=args.body, world=args.world)
        for variation in ("mut", "xmut"):
            for sigma in args.sigmas:
                config = replace(base, variation=variation, sigma=sigma, setup=setup)
                label = f"{variation} sigma={sigma}"
                for seed in seeds:
                    tasks.append(
                        (label, config, seed, args.out / "sigma" / f"{variation}_{sigma}" / f"seed_{seed}"),
                    )

    console.log(f"{len(tasks)} pilot runs, {base.total_evaluations} evaluations each")
    start = time.perf_counter()
    results = run_pool(tasks, args.workers)
    report(results, f"pilot: {args.pilot}")
    console.log(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
