"""(mu + lambda) EA and random-search baseline for evolving robot bodies towards the A1 target set.

The two EA variants differ only in ``EAConfig.mutation``: "point" applies
``mutate_replace_node`` and "subtree" applies ``mutate_subtree_replacement``.
Fitness is minimised.
"""

# Standard library
import copy
import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

# Third-party libraries
import networkx as nx
import numpy as np

# Local scripts
from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

# Local libraries (ARIEL)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.operators import (
    mutate_replace_node,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

# Type aliases
type MutationType = Literal["point", "subtree"]

# --- PATHS --- #
HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
TARGET_DIR: Path = HERE / "target_bodies"

# --- CSV SCHEMA --- #
NUM_TARGETS: int = 5
CSV_COLUMNS: list[str] = [
    "gen",
    "evals",
    "best",
    "mean",
    "median",
    "worst",
    "std",
    "mean_size",
    "min_size",
    "max_size",
    "unique_genotypes",
    "mut_noop_rate",
    "mut_improve_rate",
    "size_rejections",
    "size_fallbacks",
    *[f"best_d{i}" for i in range(NUM_TARGETS)],
]


# ============================================================================ #
#  1. CONFIGURATION
# ============================================================================ #


@dataclass(frozen=True)
class EAConfig:
    """All parameters of one run. P and S differ ONLY in ``mutation``."""

    mutation: MutationType = "point"
    pop_size: int = 100
    offspring_size: int = 100
    generations: int = 100
    tournament_size: int = 3
    max_modules: int = 20
    init_min_modules: int = 1
    init_max_modules: int = 20
    size_guard_attempts: int = 10
    subtree_max_modules: int = 20

    @property
    def max_nodes(self) -> int:
        """Largest allowed body in nodes (modules + core)."""
        return self.max_modules + 1

    @property
    def total_evaluations(self) -> int:
        """Evaluation budget of one run: mu + lambda * generations."""
        return self.pop_size + self.offspring_size * self.generations


# ============================================================================ #
#  2. TARGETS AND FITNESS
# ============================================================================ #


def target_paths(target_dir: Path = TARGET_DIR) -> list[Path]:
    """Return the target JSON files in sorted order (defines best_d0..d4)."""
    paths = sorted(target_dir.glob("*.json"))
    if len(paths) != NUM_TARGETS:
        msg = f"expected {NUM_TARGETS} targets in {target_dir}, got {len(paths)}"
        raise FileNotFoundError(msg)
    return paths


def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph, sorted by filename."""
    return [load_graph_from_json(p) for p in target_paths(target_dir)]


def genotype_to_graph(genotype: dict[str, Any]) -> nx.DiGraph:
    """Decode a stored genotype dict (``TreeGenome.to_dict``) into a body."""
    return TreeGenome.from_dict(genotype).to_networkx()


def fitness_function(body: nx.DiGraph, targets: list[nx.DiGraph]) -> float:
    """Mean + 1 std of the weighted TED to all targets. LOWER IS BETTER."""
    return float(mean_plus_std_tree_edit_distance(body, targets))


def genotype_size(genotype: dict[str, Any]) -> int:
    """Number of nodes (modules + core) in a stored genotype."""
    return len(genotype["nodes"])


def canonical_genotype(genotype: dict[str, Any]) -> str:
    """Order-independent string of a genotype, used to count unique ones."""
    nodes = sorted(
        (int(k), v["type"], v["rotation"]) for k, v in genotype["nodes"].items()
    )
    edges = sorted(
        (int(e["parent"]), int(e["child"]), e["face"]) for e in genotype["edges"]
    )
    return json.dumps([nodes, edges])


# ============================================================================ #
#  3. INITIALISATION AND MUTATION
# ============================================================================ #


def ramped_random_genome(config: EAConfig) -> TreeGenome:
    """Random body with ``k ~ U{init_min, init_max}`` modules (k + 1 nodes)."""
    k = random.randint(config.init_min_modules, config.init_max_modules)
    return random_tree(k)


def apply_mutation(genome: TreeGenome, config: EAConfig) -> None:
    """Apply exactly ONE mutation of the configured type, in place."""
    match config.mutation:
        case "point":
            mutate_replace_node(genome)
        case "subtree":
            mutate_subtree_replacement(
                genome,
                max_modules=config.subtree_max_modules,
            )
        case _:
            msg = f"unknown mutation {config.mutation!r}"
            raise ValueError(msg)


def mutate_with_size_guard(
    parent_genotype: dict[str, Any],
    config: EAConfig,
) -> tuple[TreeGenome, int, bool]:
    """Clone the parent and mutate it, rejecting children over the size cap."""
    rejections = 0
    for _ in range(config.size_guard_attempts):
        child = TreeGenome.from_dict(copy.deepcopy(parent_genotype))
        apply_mutation(child, config)
        if len(child.nodes) <= config.max_nodes:
            return child, rejections, False
        rejections += 1
    fallback = TreeGenome.from_dict(copy.deepcopy(parent_genotype))
    return fallback, rejections, True


# ============================================================================ #
#  4. RUN STATE AND LOGGING HELPERS
# ============================================================================ #


@dataclass
class RunState:
    """Mutable bookkeeping shared by the EAOperation steps of ONE run."""

    targets: list[nx.DiGraph]
    generation: int = 0
    evals: int = 0
    selected_parents: list[Individual] = field(default_factory=list)
    children: list[Individual] = field(default_factory=list)
    size_rejections: int = 0
    size_fallbacks: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)
    best_genotype: dict[str, Any] | None = None
    best_fitness: float | None = None


def _sort_key(position: int, ind: Individual) -> tuple[float, float, int]:
    """(fitness, id, position): lowest fitness first, ties -> lowest id."""
    ind_id = float(ind.id) if ind.id is not None else float("inf")
    return (ind.fitness, ind_id, position)


def rank_individuals(individuals: list[Individual]) -> list[Individual]:
    """Return individuals best-first (minimisation, stable id tie-break)."""
    keyed = sorted(
        enumerate(individuals),
        key=lambda pair: _sort_key(pair[0], pair[1]),
    )
    return [ind for _, ind in keyed]


def summary_row(
    gen: int,
    evals: int,
    fitnesses: list[float],
    genotypes: list[dict[str, Any]],
    best_genotype: dict[str, Any],
    targets: list[nx.DiGraph],
) -> dict[str, Any]:
    """Fitness / size / diversity statistics shared by EA and random search."""
    fit = np.asarray(fitnesses, dtype=float)
    sizes = np.asarray([genotype_size(g) for g in genotypes], dtype=float)
    best_dists = list(
        distances_to_targets(genotype_to_graph(best_genotype), targets),
    )
    row: dict[str, Any] = {
        "gen": gen,
        "evals": evals,
        "best": float(fit.min()),
        "mean": float(fit.mean()),
        "median": float(np.median(fit)),
        "worst": float(fit.max()),
        "std": float(fit.std()),
        "mean_size": float(sizes.mean()),
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "unique_genotypes": len({canonical_genotype(g) for g in genotypes}),
        "mut_noop_rate": None,
        "mut_improve_rate": None,
        "size_rejections": None,
        "size_fallbacks": None,
    }
    for i, d in enumerate(best_dists):
        row[f"best_d{i}"] = float(d)
    return row


def _format_cell(value: Any) -> str:
    """CSV cell: floats via repr (exact round-trip), None -> empty."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_generations_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the per-generation rows with the fixed column order."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(CSV_COLUMNS) + "\n")
        for row in rows:
            fh.write(",".join(_format_cell(row[c]) for c in CSV_COLUMNS) + "\n")


def _git_commit() -> dict[str, Any]:
    """Commit hash of the repository and whether the tree is dirty."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _package_versions() -> dict[str, str | None]:
    """Versions of Python and the packages this experiment depends on."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for pkg in ("ariel", "numpy", "networkx", "sqlmodel", "sqlalchemy", "mujoco"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def write_config_json(
    config: EAConfig,
    seed: int,
    algorithm: Literal["ea", "random_search"],
    out_dir: Path,
) -> None:
    """Record everything needed to reproduce this run."""
    resolved = asdict(config)
    resolved["max_nodes"] = config.max_nodes
    resolved["total_evaluations"] = config.total_evaluations
    if algorithm == "random_search":
        for unused in (
            "mutation",
            "tournament_size",
            "size_guard_attempts",
            "subtree_max_modules",
        ):
            resolved.pop(unused)
    payload = {
        "algorithm": algorithm,
        "seed": seed,
        "config": resolved,
        "targets": [
            str(p.relative_to(REPO_ROOT)) for p in target_paths()
        ],
        "fitness": "mean_plus_std_tree_edit_distance (minimised)",
        "git": _git_commit(),
        "versions": _package_versions(),
        "argv": sys.argv,
    }
    (out_dir / "config.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def write_best_genome(
    genotype: dict[str, Any],
    fitness: float,
    targets: list[nx.DiGraph],
    out_dir: Path,
) -> None:
    """Save the best body with its fitness and per-target distances."""
    body = genotype_to_graph(genotype)
    payload = {
        "fitness": fitness,
        "distances": list(distances_to_targets(body, targets)),
        "num_nodes": body.number_of_nodes(),
        "genotype": genotype,
    }
    (out_dir / "best_genome.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


# ============================================================================ #
# ============================================================================ #


@EAOperation
def parent_selection(
    population: Population,
    config: EAConfig,
    state: RunState,
) -> Population:
    """Lambda independent tournaments of size k (with replacement)."""
    state.generation += 1
    alive = sorted(population, key=lambda ind: ind.id)
    state.selected_parents = []
    for _ in range(config.offspring_size):
        contestants = [random.choice(alive) for _ in range(config.tournament_size)]
        winner = min(contestants, key=lambda ind: (ind.fitness, ind.id))
        state.selected_parents.append(winner)
    return population


@EAOperation
def reproduce(
    population: Population,
    config: EAConfig,
    state: RunState,
) -> Population:
    """One NEW child per selected parent: clone -> one mutation -> size guard."""
    state.children = []
    state.size_rejections = 0
    state.size_fallbacks = 0
    for parent in state.selected_parents:
        child_genome, rejections, fallback = mutate_with_size_guard(
            parent.genotype,
            config,
        )
        state.size_rejections += rejections
        state.size_fallbacks += int(fallback)

        parent_body = genotype_to_graph(parent.genotype)
        noop = tree_edit_distance(parent_body, child_genome.to_networkx()) == 0.0

        child = Individual()
        child.genotype = child_genome.to_dict()
        child.tags = {
            "origin": config.mutation,
            "parent_id": parent.id,
            "parent_fitness": parent.fitness,
            "noop": noop,
            "size_rejections": rejections,
            "size_fallback": fallback,
        }
        state.children.append(child)

    population.extend(state.children)
    return population


@EAOperation
def evaluate(population: Population, state: RunState) -> Population:
    """Compute fitness for every individual that still needs it."""
    for ind in population:
        if ind.requires_eval:
            ind.fitness = fitness_function(
                genotype_to_graph(ind.genotype),
                state.targets,
            )
            state.evals += 1
    return population


@EAOperation
def survivor_selection(population: Population, config: EAConfig) -> Population:
    """(mu + lambda) truncation. Returns ALL mu + lambda individuals."""
    assert not any(ind.requires_eval for ind in population), (
        "survivor_selection called with unevaluated individuals"
    )
    ranked = rank_individuals(list(population))
    for ind in ranked[config.pop_size :]:
        ind.alive = False
    return population


@EAOperation
def log_stats(population: Population, config: EAConfig, state: RunState) -> Population:
    """Append this generation's CSV row, computed over the mu alive survivors."""
    survivors = rank_individuals([ind for ind in population if ind.alive])
    assert len(survivors) == config.pop_size
    best = survivors[0]

    row = summary_row(
        gen=state.generation,
        evals=state.evals,
        fitnesses=[ind.fitness for ind in survivors],
        genotypes=[ind.genotype for ind in survivors],
        best_genotype=best.genotype,
        targets=state.targets,
    )
    n_children = len(state.children)
    row["mut_noop_rate"] = sum(c.tags["noop"] for c in state.children) / n_children
    row["mut_improve_rate"] = (
        sum(c.fitness < c.tags["parent_fitness"] for c in state.children)
        / n_children
    )
    row["size_rejections"] = state.size_rejections
    row["size_fallbacks"] = state.size_fallbacks
    state.rows.append(row)

    state.best_genotype = copy.deepcopy(best.genotype)
    state.best_fitness = best.fitness
    return population


# ============================================================================ #
#  6. RUNNERS
# ============================================================================ #


def run_ea(config: EAConfig, seed: int, out_dir: Path) -> list[dict[str, Any]]:
    """Run one (mu + lambda) EA and write all its output files."""
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config_json(config, seed, "ea", out_dir)

    state = RunState(targets=load_targets())

    initial: list[Individual] = []
    for _ in range(config.pop_size):
        genome = ramped_random_genome(config)
        ind = Individual()
        ind.genotype = genome.to_dict()
        ind.tags = {"origin": "init"}
        ind.fitness = fitness_function(genome.to_networkx(), state.targets)
        state.evals += 1
        initial.append(ind)

    ranked = rank_individuals(initial)
    row = summary_row(
        gen=0,
        evals=state.evals,
        fitnesses=[ind.fitness for ind in initial],
        genotypes=[ind.genotype for ind in initial],
        best_genotype=ranked[0].genotype,
        targets=state.targets,
    )
    row["size_rejections"] = 0
    row["size_fallbacks"] = 0
    state.rows.append(row)
    state.best_genotype = copy.deepcopy(ranked[0].genotype)
    state.best_fitness = ranked[0].fitness

    operations = [
        parent_selection(config=config, state=state),
        reproduce(config=config, state=state),
        evaluate(state=state),
        survivor_selection(config=config),
        log_stats(config=config, state=state),
    ]
    ea = EA(
        Population(initial),
        operations,
        num_steps=config.generations,
        first_generation_id=0,
        is_maximisation=False,
        quiet=True,
        db_file_path=out_dir / "database.db",
        db_handling="delete",
    )
    ea.run()

    assert state.evals == config.total_evaluations, (
        f"evaluation count {state.evals} != {config.total_evaluations}"
    )
    write_generations_csv(state.rows, out_dir / "generations.csv")
    write_best_genome(
        state.best_genotype,
        state.best_fitness,
        state.targets,
        out_dir,
    )
    return state.rows


def run_random_search(
    config: EAConfig,
    seed: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Random search with the EA's budget and the EA's initialiser."""
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config_json(config, seed, "random_search", out_dir)

    targets = load_targets()
    evals = 0
    best_fitness = float("inf")
    best_genotype: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []

    for gen in range(config.generations + 1):
        block_size = config.pop_size if gen == 0 else config.offspring_size
        fitnesses: list[float] = []
        genotypes: list[dict[str, Any]] = []
        for _ in range(block_size):
            genome = ramped_random_genome(config)
            fit = fitness_function(genome.to_networkx(), targets)
            evals += 1
            genotype = genome.to_dict()
            fitnesses.append(fit)
            genotypes.append(genotype)
            if fit < best_fitness:
                best_fitness = fit
                best_genotype = genotype

        row = summary_row(
            gen=gen,
            evals=evals,
            fitnesses=fitnesses,
            genotypes=genotypes,
            best_genotype=best_genotype,
            targets=targets,
        )
        row["best"] = best_fitness
        rows.append(row)

    assert evals == config.total_evaluations
    write_generations_csv(rows, out_dir / "generations.csv")
    write_best_genome(best_genotype, best_fitness, targets, out_dir)
    return rows
