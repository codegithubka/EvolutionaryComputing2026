"""(mu + lambda) neuroevolution EA and random-search baseline for A2.

The EA variants differ ONLY in ``EAConfig.variation``, i.e. how a child's
genotype is formed before the (identical) Gaussian mutation:

    mut    child = copy(p1)
    xmut   child = uniform_crossover(p1, p2)            p2 = second tournament winner
    xrand  child = uniform_crossover(p1, random)        "headless chicken" (Jones 1995);
                                                        the random donor is never evaluated
    xunit  child = unit_crossover(p1, p2)               whole hidden units (column of w1 +
                                                        row of w2) taken from either parent

Fitness (planar distance to the target) is minimised.
"""

# Standard library
import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

# Third-party libraries
import numpy as np

# Local scripts
from controller import Setup, compiled, decode, genotype_length, simulate

# Local libraries (ARIEL)
from ariel.ec import (
    EA,
    Crossover,
    EAOperation,
    FloatMutator,
    FloatsGenerator,
    Individual,
    Population,
    set_seed,
)

# Type aliases
type Variation = Literal["mut", "xmut", "xrand", "xunit"]
type Genotype = list[float]

VARIATIONS: tuple[str, ...] = ("mut", "xmut", "xrand", "xunit")

# --- PATHS --- #
HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent

# --- CSV SCHEMA --- #
CSV_COLUMNS: list[str] = [
    "gen",
    "evals",
    "best",
    "best_so_far",
    "mean",
    "median",
    "worst",
    "std",
    "diversity",
    "child_better_than_best_parent",
    "child_worse_than_worst_parent",
    "child_minus_p1",
    "child_survival",
    "clone_redraws",
    "best_min_z",
    "best_path_length",
    "best_final_x",
    "best_final_y",
]


# ============================================================================ #
#  1. CONFIGURATION
# ============================================================================ #


@dataclass(frozen=True)
class EAConfig:
    """All parameters of one run. EA arms differ ONLY in ``variation``."""

    variation: Variation = "mut"
    pop_size: int = 30
    offspring_size: int = 30
    # 9030 evaluations. At 150 generations no arm met the plateau criterion
    # (mean best-so-far still improved 10-12% over the last 20% of evaluations).
    generations: int = 300
    tournament_size: int = 3
    init_std: float = 0.5
    # Pilot (1530 evals, 3 seeds, mut + xmut): best worst-arm mean; the
    # joint-mean winner 0.05 was a tie (+0.015 m) and favoured xmut only.
    sigma: float = 0.2
    mutation_probability: float = 0.1
    setup: Setup = field(default_factory=Setup)

    @property
    def total_evaluations(self) -> int:
        """Evaluation budget of one run: mu + lambda * generations."""
        return self.pop_size + self.offspring_size * self.generations


# ============================================================================ #
#  2. INITIALISATION AND VARIATION
# ============================================================================ #


def random_genotype(config: EAConfig) -> Genotype:
    """N(0, init_std) weights from ariel.ec's shared RNG (seeded by set_seed)."""
    length = genotype_length(config.setup)
    return list(FloatsGenerator.normal(0.0, config.init_std, length))


def uniform_crossover(p1: Genotype, p2: Genotype) -> Genotype:
    """First child of ariel.ec's uniform crossover (each gene 50/50)."""
    child, _ = Crossover.uniform(p1, p2, swap_probability=0.5)
    return cast("Genotype", child)


def unit_crossover(p1: Genotype, p2: Genotype, setup: Setup) -> Genotype:
    """Take each hidden unit WHOLE (its w1 column and w2 row) from one parent."""
    a1, a2 = decode(p1, setup)
    b1, b2 = decode(p2, setup)
    w1, w2 = a1.copy(), a2.copy()
    for j in range(setup.hidden_size):
        if random.random() < 0.5:
            w1[:, j] = b1[:, j]
            w2[j, :] = b2[j, :]
    return np.concatenate((w1.ravel(), w2.ravel())).tolist()


def mutate(genotype: Genotype, config: EAConfig) -> tuple[Genotype, int]:
    """Gaussian mutation, redrawn until at least one gene changed.

    Simulation is deterministic, so an unchanged child would waste an
    evaluation, and how often that happens differs between variation types.
    Returns the child and the number of redraws needed.
    """
    redraws = 0
    while True:
        child = cast(
            "Genotype",
            FloatMutator.gaussian(
                genotype,
                std=config.sigma,
                mutation_probability=config.mutation_probability,
            ),
        )
        if child != genotype:
            return child, redraws
        redraws += 1


def vary(
    p1: Genotype,
    p2: Genotype | None,
    config: EAConfig,
) -> tuple[Genotype, int]:
    """Form a child with the configured variation, then mutate it."""
    match config.variation:
        case "mut":
            pre = list(p1)
        case "xmut":
            pre = uniform_crossover(p1, cast("Genotype", p2))
        case "xrand":
            pre = uniform_crossover(p1, random_genotype(config))
        case "xunit":
            pre = unit_crossover(p1, cast("Genotype", p2), config.setup)
        case _:
            msg = f"unknown variation {config.variation!r}"
            raise ValueError(msg)
    return mutate(pre, config)


def uses_second_parent(variation: Variation) -> bool:
    """Whether the child has a second SELECTED (and evaluated) parent."""
    return variation in {"xmut", "xunit"}


# ============================================================================ #
#  3. EVALUATION AND RUN STATE
# ============================================================================ #


DIAGNOSTICS: tuple[str, ...] = ("min_z", "path_length", "final_x", "final_y")


@dataclass
class RunState:
    """Mutable bookkeeping shared by the EAOperation steps of ONE run."""

    generation: int = 0
    evals: int = 0
    parents: list[tuple[Individual, Individual | None]] = field(default_factory=list)
    children: list[Individual] = field(default_factory=list)
    clone_redraws: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)
    best: dict[str, Any] | None = None


def snapshot(ind: Individual) -> dict[str, Any]:
    """Plain copy of an individual (ORM objects expire after each DB commit)."""
    return {
        "fitness": ind.fitness,
        "genotype": list(cast("Genotype", ind.genotype)),
        "diagnostics": {k: ind.tags[k] for k in DIAGNOSTICS},
    }


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


def mean_pairwise_distance(genotypes: list[Genotype]) -> float:
    """Mean L2 distance between all genotype pairs (permutation-blind)."""
    x = np.asarray(genotypes, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    sq = np.sum(x * x, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * x @ x.T, 0.0)
    return float(np.sqrt(d2)[np.triu_indices(n, k=1)].mean())


def summary_row(
    gen: int,
    evals: int,
    individuals: list[Individual],
    best_so_far: dict[str, Any],
) -> dict[str, Any]:
    """Fitness / diversity statistics shared by the EA and random search."""
    fit = np.asarray([ind.fitness for ind in individuals], dtype=float)
    row: dict[str, Any] = dict.fromkeys(CSV_COLUMNS)
    row.update(
        {
            "gen": gen,
            "evals": evals,
            "best": float(fit.min()),
            "best_so_far": best_so_far["fitness"],
            "mean": float(fit.mean()),
            "median": float(np.median(fit)),
            "worst": float(fit.max()),
            "std": float(fit.std()),
            "diversity": mean_pairwise_distance(
                [cast("Genotype", ind.genotype) for ind in individuals],
            ),
            **{f"best_{k}": best_so_far["diagnostics"][k] for k in DIAGNOSTICS},
        },
    )
    return row


def new_individual(genotype: Genotype, tags: dict[str, Any]) -> Individual:
    """An unevaluated individual with the given genotype and tags."""
    ind = Individual()
    ind.genotype = genotype
    ind.tags = tags
    return ind


def evaluate_individual(ind: Individual, config: EAConfig, state: RunState) -> None:
    """Simulate ``ind`` and merge fitness + diagnostics into its tags."""
    result = simulate(cast("Genotype", ind.genotype), config.setup)
    ind.tags = {k: getattr(result, k) for k in DIAGNOSTICS}
    ind.fitness = result.fitness
    state.evals += 1


# ============================================================================ #
#  4. OUTPUT
# ============================================================================ #


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
    for pkg in ("ariel", "numpy", "mujoco", "sqlmodel", "sqlalchemy"):
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
    resolved["total_evaluations"] = config.total_evaluations
    c = compiled(config.setup)
    resolved["genotype_length"] = genotype_length(config.setup)
    resolved["num_inputs"] = c.num_inputs
    resolved["num_outputs"] = c.num_outputs
    if algorithm == "random_search":
        for unused in ("variation", "tournament_size", "sigma", "mutation_probability"):
            resolved.pop(unused)
    payload = {
        "algorithm": algorithm,
        "seed": seed,
        "config": resolved,
        "fitness": "planar distance to target (minimised)",
        "git": _git_commit(),
        "versions": _package_versions(),
        "argv": sys.argv,
    }
    (out_dir / "config.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def write_best_genome(best: dict[str, Any], out_dir: Path) -> None:
    """Save the best controller with its fitness and diagnostics."""
    (out_dir / "best_genome.json").write_text(
        json.dumps(best, indent=2) + "\n",
        encoding="utf-8",
    )


# ============================================================================ #
#  5. THE EA PIPELINE
# ============================================================================ #


def tournament(candidates: list[Individual], k: int) -> Individual:
    """Best of ``k`` uniform draws with replacement (minimisation)."""
    contestants = [random.choice(candidates) for _ in range(k)]
    return min(contestants, key=lambda ind: (ind.fitness, ind.id))


@EAOperation
def parent_selection(
    population: Population,
    config: EAConfig,
    state: RunState,
) -> Population:
    """Lambda (p1, p2) pairs by tournament; p2 only for two-parent variations."""
    state.generation += 1
    alive = sorted(population, key=lambda ind: ind.id)
    state.parents = []
    for _ in range(config.offspring_size):
        p1 = tournament(alive, config.tournament_size)
        p2 = None
        if uses_second_parent(config.variation):
            others = [ind for ind in alive if ind.id != p1.id]
            p2 = tournament(others, config.tournament_size)
        state.parents.append((p1, p2))
    return population


@EAOperation
def reproduce(
    population: Population,
    config: EAConfig,
    state: RunState,
) -> Population:
    """One NEW child per (p1, p2) pair: variation -> mutation."""
    state.children = []
    state.clone_redraws = 0
    for p1, p2 in state.parents:
        child_genotype, redraws = vary(
            cast("Genotype", p1.genotype),
            cast("Genotype | None", p2.genotype if p2 is not None else None),
            config,
        )
        state.clone_redraws += redraws
        parent_fits = [p1.fitness] + ([p2.fitness] if p2 is not None else [])
        child = new_individual(
            child_genotype,
            {"origin": config.variation, "parent_fits": parent_fits},
        )
        state.children.append(child)
    population.extend(state.children)
    return population


@EAOperation
def evaluate(population: Population, config: EAConfig, state: RunState) -> Population:
    """Simulate every individual that still needs it."""
    for ind in population:
        if ind.requires_eval:
            evaluate_individual(ind, config, state)
    return population


@EAOperation
def survivor_selection(population: Population, config: EAConfig) -> Population:
    """(mu + lambda) truncation. Returns ALL mu + lambda individuals."""
    assert not any(
        ind.requires_eval for ind in population
    ), "survivor_selection called with unevaluated individuals"
    ranked = rank_individuals(list(population))
    for ind in ranked[config.pop_size :]:
        ind.alive = False
    return population


@EAOperation
def log_stats(population: Population, config: EAConfig, state: RunState) -> Population:
    """Append this generation's CSV row, computed over the mu alive survivors."""
    survivors = rank_individuals([ind for ind in population if ind.alive])
    assert len(survivors) == config.pop_size
    state.best = snapshot(survivors[0])

    row = summary_row(state.generation, state.evals, survivors, state.best)
    kids = state.children
    n = len(kids)
    best_parent = [min(c.tags["parent_fits"]) for c in kids]
    worst_parent = [max(c.tags["parent_fits"]) for c in kids]
    p1_fit = [c.tags["parent_fits"][0] for c in kids]
    row["child_better_than_best_parent"] = (
        sum(c.fitness < b for c, b in zip(kids, best_parent, strict=True)) / n
    )
    row["child_worse_than_worst_parent"] = (
        sum(c.fitness > w for c, w in zip(kids, worst_parent, strict=True)) / n
    )
    row["child_minus_p1"] = float(
        np.mean([c.fitness - f for c, f in zip(kids, p1_fit, strict=True)]),
    )
    row["child_survival"] = sum(c.alive for c in kids) / n
    row["clone_redraws"] = state.clone_redraws
    state.rows.append(row)
    return population


# ============================================================================ #
#  6. RUNNERS
# ============================================================================ #


def seed_everything(seed: int) -> None:
    """Seed ariel.ec's shared RNG and the stdlib RNG (the only two used)."""
    set_seed(seed)
    random.seed(seed)


def run_ea(config: EAConfig, seed: int, out_dir: Path) -> list[dict[str, Any]]:
    """Run one (mu + lambda) EA and write all its output files."""
    seed_everything(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config_json(config, seed, "ea", out_dir)

    state = RunState()
    initial: list[Individual] = []
    for _ in range(config.pop_size):
        ind = new_individual(random_genotype(config), {"origin": "init"})
        evaluate_individual(ind, config, state)
        initial.append(ind)

    state.best = snapshot(rank_individuals(initial)[0])
    state.rows.append(summary_row(0, state.evals, initial, state.best))

    operations = [
        parent_selection(config=config, state=state),
        reproduce(config=config, state=state),
        evaluate(config=config, state=state),
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

    assert (
        state.evals == config.total_evaluations
    ), f"evaluation count {state.evals} != {config.total_evaluations}"
    write_generations_csv(state.rows, out_dir / "generations.csv")
    assert state.best is not None
    write_best_genome(state.best, out_dir)
    return state.rows


def run_random_search(
    config: EAConfig,
    seed: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Random search with the EA's budget and the EA's initialiser.

    Logged in generation-sized blocks (mu, then lambda per generation) so the
    curves share the EA's evaluations axis. Seed s draws the same first mu
    genotypes as the EA's initial population.
    """
    seed_everything(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config_json(config, seed, "random_search", out_dir)

    state = RunState()
    best: dict[str, Any] | None = None
    for gen in range(config.generations + 1):
        block_size = config.pop_size if gen == 0 else config.offspring_size
        block: list[Individual] = []
        for _ in range(block_size):
            ind = new_individual(random_genotype(config), {})
            evaluate_individual(ind, config, state)
            block.append(ind)
            if best is None or ind.fitness < best["fitness"]:
                best = snapshot(ind)
        assert best is not None
        state.rows.append(summary_row(gen, state.evals, block, best))

    assert state.evals == config.total_evaluations
    assert best is not None
    write_generations_csv(state.rows, out_dir / "generations.csv")
    write_best_genome(best, out_dir)
    return state.rows
