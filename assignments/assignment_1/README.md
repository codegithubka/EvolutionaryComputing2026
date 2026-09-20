# Assignment 1: Evolving robot bodies towards a target set

**Research question:** With an equal evaluation budget, does subtree-replacement mutation reach a different final fitness than point mutation when evolving a robot body towards a set of target bodies?

- **H1:** Subtree-replacement mutation reaches a better final fitness than point mutation, because point mutation cannot add modules to a body.
- **H2:** Point mutation, which takes smaller steps, is at least as good as subtree-replacement mutation at generation 10.
- **H3:** Both EA variants reach a better final fitness than random search.

From the repository root:

```bash
uv run assignments/assignment_1/step_size.py
uv run assignments/assignment_1/run_experiments.py --configs point subtree random --seeds 0-19
uv run jupyter nbconvert --to notebook --execute --inplace assignments/assignment_1/analysis.ipynb
```

Runs are deterministic: the same seed and configuration give an identical `generations.csv`.
