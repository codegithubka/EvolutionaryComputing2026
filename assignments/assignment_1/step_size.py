"""Measure the effect of a single application of each tree mutation operator.

Usage, from the repository root:
    uv run assignments/assignment_1/step_size.py --n 1000 --seed 12345
"""

# Standard library
import argparse
import copy
import csv
import random
from collections.abc import Callable
from pathlib import Path

# Third-party libraries
import numpy as np
from rich.console import Console
from rich.table import Table

# Local scripts
from ea_body import EAConfig, fitness_function, load_targets, ramped_random_genome
from tree_edit_distance import tree_edit_distance

# Local libraries (ARIEL)
from ariel.ec.genotypes.tree.operators import (
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

console = Console(width=150)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
MAX_NODES: int = EAConfig().max_nodes

OPERATORS: dict[str, Callable[[TreeGenome], None]] = {
    "replace_node (P)": mutate_replace_node,
    "subtree_replacement (S)": lambda g: mutate_subtree_replacement(
        g,
        max_modules=EAConfig().subtree_max_modules,
    ),
    "shrink": mutate_shrink,
    "hoist": mutate_hoist,
}


def make_parents(kind: str, n: int, seed: int) -> list[TreeGenome]:
    """Draw ``n`` parents of the given kind with a fixed seed."""
    random.seed(seed)
    config = EAConfig()
    if kind == "ramped":
        return [ramped_random_genome(config) for _ in range(n)]
    return [random_tree(config.max_modules) for _ in range(n)]


def characterise(
    name: str,
    operator: Callable[[TreeGenome], None],
    parents: list[TreeGenome],
    parent_fitness: list[float],
    targets: list,
    seed: int,
) -> dict[str, float | int | str]:
    """Apply ``operator`` once to a deep copy of every parent and measure it."""
    random.seed(seed)
    d_nodes, d_fit, noop, grow, shrink, over = [], [], [], [], [], []
    for parent, p_fit in zip(parents, parent_fitness, strict=True):
        child = TreeGenome.from_dict(copy.deepcopy(parent.to_dict()))
        operator(child)
        p_body, c_body = parent.to_networkx(), child.to_networkx()
        n_p, n_c = p_body.number_of_nodes(), c_body.number_of_nodes()
        d_nodes.append(abs(n_c - n_p))
        d_fit.append(abs(fitness_function(c_body, targets) - p_fit))
        noop.append(tree_edit_distance(p_body, c_body) == 0.0)
        grow.append(n_c > n_p)
        shrink.append(n_c < n_p)
        over.append(n_c > MAX_NODES)
    return {
        "operator": name,
        "n": len(parents),
        "mean_abs_d_nodes": float(np.mean(d_nodes)),
        "mean_abs_d_fitness": float(np.mean(d_fit)),
        "median_abs_d_fitness": float(np.median(d_fit)),
        "noop_rate": float(np.mean(noop)),
        "growth_rate": float(np.mean(grow)),
        "shrink_rate": float(np.mean(shrink)),
        "over_cap_rate": float(np.mean(over)),
    }


def main() -> None:
    """Characterise all four operators on both parent sets."""
    parser = argparse.ArgumentParser(description="tree mutation step sizes")
    parser.add_argument("--n", type=int, default=1000, help="parents per set")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", type=Path, default=RESULTS / "step_size.csv")
    args = parser.parse_args()

    targets = load_targets()
    rows: list[dict] = []
    for kind_idx, kind in enumerate(("ramped", "full21")):
        parents = make_parents(kind, args.n, args.seed + 1000 * kind_idx)
        parent_fitness = [fitness_function(p.to_networkx(), targets) for p in parents]
        for op_idx, (name, operator) in enumerate(OPERATORS.items()):
            row = characterise(
                name,
                operator,
                parents,
                parent_fitness,
                targets,
                seed=args.seed + 1000 * kind_idx + op_idx + 1,
            )
            rows.append({"parents": kind, **row})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = Table(title=f"Mutation step sizes (n={args.n} parents, seed={args.seed})")
    for col in rows[0]:
        table.add_column(col, justify="left" if col in {"parents", "operator"} else "right")
    for row in rows:
        table.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()])
    console.print(table)
    console.log(f"saved {args.out}")


if __name__ == "__main__":
    main()
