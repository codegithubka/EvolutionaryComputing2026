# Assignment 2: Neuroevolution for targeted locomotion

**Research question:** When evolving the weights of a fixed-topology neural-network controller for targeted locomotion, does recombination improve the final distance to the target compared with mutation alone, at an equal evaluation budget? If it does, is the gain due to combining information from two parents, or only to the larger variation crossover introduces?

- **H1:** Crossover between two selected parents (`xmut`) and mutation alone (`mut`) reach a different final distance to the target.
- **H2:** `xmut` and crossover with a random individual (`xrand`, Jones's headless-chicken control) reach a different final distance.
- **H3:** Every EA variant reaches a lower final distance than random search.

| Config   | Child before mutation |
|----------|-----------------------|
| `mut`    | copy of parent 1 |
| `xmut`   | uniform crossover of two tournament winners |
| `xrand`  | uniform crossover of a tournament winner with a fresh random genotype (never evaluated) |
| `xunit`  | (optional) whole hidden units taken from either of two tournament winners |
| `random` | random search with the same budget and initialiser |

## Setup

| Parameter | Value |
|---|---|
| Body / world | `spider_8` (John Set) on `SimpleFlatWorld` |
| Task | spawn at (0, 0), target at (3, 2), 15 s simulated, planar distance to target minimised |
| Controller | inputs (hinge angles, target direction in heading frame, distance, sin/cos 1 Hz clock, bias) → 6 tanh → 8 tanh, direct control every physics step (2 ms) |
| EA | (μ+λ), μ = λ = 30, tournament k = 3, 300 generations (9,030 evaluations) |
| Variation | uniform crossover (p = 0.5), Gaussian mutation σ = 0.2, per-gene p = 0.1, every child differs from its parent |
| Seeds | 0–14 for `mut` and `xmut`, 0–9 for `xrand` and `random`; pilot seeds 100–102 |

## Files

| File | Contents |
|---|---|
| `controller.py` | Fixed setup: body, world, controller inputs, NN, `simulate()` and fitness |
| `ea_neuro.py` | (μ+λ) EA on `ariel.ec`, variation operators, random search, CSV/JSON output |
| `run_experiments.py` | Runs a grid of configurations × seeds in parallel |
| `pilot.py` | Body and σ pilots (seeds 100+, never reused) |
| `replay_best.py` | Re-checks a run's best controller, then opens the viewer or renders an mp4 |

Each run writes `config.json` (all parameters, seed, git commit, package versions), `generations.csv`, `best_genome.json` and the `ariel.ec` database to `__data__/A2/<config>/seed_XX/`.

## Reproducing

From the repository root:

```bash
uv run assignments/assignment_2/pilot.py bodies
uv run assignments/assignment_2/pilot.py sigma --pop 30 --gens 50 --seeds 3
caffeinate -i uv run assignments/assignment_2/run_experiments.py --configs mut xmut --seeds 0-14
caffeinate -i uv run assignments/assignment_2/run_experiments.py --configs xrand random --seeds 0-9
A2_DATA=__data__/A2 uv run jupyter nbconvert --to notebook --execute --inplace assignments/assignment_2/analysis.ipynb
uv run assignments/assignment_2/replay_best.py __data__/A2/mut/seed_00 --video
```

Runs are deterministic: the same seed and configuration give an identical `generations.csv`. For a given seed, every configuration starts from the same initial population.
