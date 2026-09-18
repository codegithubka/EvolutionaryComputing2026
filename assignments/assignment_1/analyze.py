"""Figures, statistics tables and body renders from the experiment output.

Usage, from the repository root:
    uv run assignments/assignment_1/analyze.py --data __data__/A1 --out assignments/assignment_1/results
"""

# Standard library
import argparse
import json
from itertools import combinations
from pathlib import Path

# Third-party libraries
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from scipy.stats import mannwhitneyu

# Local scripts
from ea_body import genotype_to_graph, load_targets

console = Console(width=150)

# --- CONFIG NAMES, LABELS AND COLOURS --- #
CONFIGS: tuple[str, ...] = ("point", "subtree", "random")
LABELS: dict[str, str] = {
    "point": "P (point)",
    "subtree": "S (subtree)",
    "random": "RS (random search)",
}
SHORT: dict[str, str] = {"point": "P", "subtree": "S", "random": "RS"}
COLOURS: dict[str, str] = {
    "point": "#2a78d6",
    "subtree": "#eb6834",
    "random": "#1baf7a",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
H2_GENERATION: int = 10
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]
RENDER_FOVY: float = 2.0


# ============================================================================ #
#  1. STATISTICS
# ============================================================================ #


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney A12 = P(X < Y) + 0.5 P(X = Y)."""
    x = np.asarray(x, dtype=float)[:, None]
    y = np.asarray(y, dtype=float)[None, :]
    return float(((x < y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (p_values[i], i))
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running_max
    return adjusted


def mwu(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U (scipy; exact for small untied samples)."""
    result = mannwhitneyu(x, y, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def self_test() -> None:
    """Check the statistics helpers on toy inputs with known answers."""
    assert a12([1, 2, 3], [4, 5, 6]) == 1.0
    assert a12([4, 5, 6], [1, 2, 3]) == 0.0
    assert a12([1, 2], [1, 2]) == 0.5
    assert a12([1], [1, 2]) == 0.75
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.5, 0.9]), [1.0, 1.0])
    u, p = mwu(np.arange(5), np.arange(5, 10))
    assert u == 0.0 and np.isclose(p, 2 / 252)
    console.log("[green]self-test passed[/green]: A12, Holm and MWU agree with known values")


# ============================================================================ #
#  2. LOADING
# ============================================================================ #


def load_runs(data: Path) -> dict[str, list[pd.DataFrame]]:
    """Return {config: [one generations DataFrame per seed]} for configs present."""
    runs: dict[str, list[pd.DataFrame]] = {}
    for name in CONFIGS:
        files = sorted((data / name).glob("seed_*/generations.csv"))
        if files:
            frames = []
            for f in files:
                df = pd.read_csv(f)
                df["seed_dir"] = f.parent.name
                frames.append(df)
            runs[name] = frames
    if not runs:
        msg = f"no generations.csv found under {data}"
        raise FileNotFoundError(msg)
    return runs


def per_generation(frames: list[pd.DataFrame], column: str) -> pd.DataFrame:
    """Mean and std (ddof=1) over runs of ``column`` for every generation."""
    stacked = pd.concat(frames)[["gen", column]]
    grouped = stacked.groupby("gen")[column]
    return pd.DataFrame(
        {"mean": grouped.mean(), "std": grouped.std(ddof=1).fillna(0.0)},
    )


def values_at(frames: list[pd.DataFrame], column: str, gen: int | None) -> np.ndarray:
    """``column`` of each run at generation ``gen`` (None = each run's last row)."""
    out = []
    for df in frames:
        row = df.iloc[-1] if gen is None else df[df["gen"] == gen].iloc[0]
        out.append(float(row[column]))
    return np.asarray(out)


# ============================================================================ #
#  3. FIGURES
# ============================================================================ #


def _style(ax: plt.Axes) -> None:
    """Recessive grid and axes, ink-coloured text."""
    ax.grid(visible=True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _band(ax: plt.Axes, stats: pd.DataFrame, colour: str, label: str, style: str) -> None:
    """Line at the mean with a +-1 std band."""
    x = stats.index.to_numpy()
    ax.plot(x, stats["mean"], color=colour, linewidth=2, linestyle=style, label=label)
    ax.fill_between(
        x,
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        color=colour,
        alpha=0.15,
        linewidth=0,
    )


def fig_convergence(runs: dict[str, list[pd.DataFrame]], out: Path) -> None:
    """Fig 1: best and population-mean fitness, mean +- std over runs."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    for name in ("point", "subtree"):
        if name in runs:
            n = len(runs[name])
            _band(ax, per_generation(runs[name], "best"), COLOURS[name],
                  f"{LABELS[name]} best (n={n})", "-")
            _band(ax, per_generation(runs[name], "mean"), COLOURS[name],
                  f"{LABELS[name]} population mean", "--")
    if "random" in runs:
        _band(ax, per_generation(runs["random"], "best"), COLOURS["random"],
              f"{LABELS['random']} best-so-far (n={len(runs['random'])})", ":")
    any_run = next(iter(runs.values()))[0]
    per_gen = int(any_run["evals"].iloc[1] - any_run["evals"].iloc[0]) if len(any_run) > 1 else 0
    ax.set_xlabel(f"generation (1 generation = {per_gen} evaluations, same for RS)")
    ax.set_ylabel("fitness (mean + std TED, lower is better)")
    _style(ax)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "fig1_convergence.png")
    plt.close(fig)


def fig_size(runs: dict[str, list[pd.DataFrame]], target_mean_size: float, out: Path) -> None:
    """Fig 2: mean body size per generation, P vs S, + target mean size."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    for name in ("point", "subtree"):
        if name in runs:
            _band(ax, per_generation(runs[name], "mean_size"), COLOURS[name],
                  f"{LABELS[name]} (n={len(runs[name])})", "-")
    ax.axhline(target_mean_size, color=INK_2, linestyle="--", linewidth=1.2,
               label=f"target mean size ({target_mean_size:.1f} nodes)")
    ax.set_xlabel("generation")
    ax.set_ylabel("mean body size (nodes)")
    _style(ax)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "fig2_size.png")
    plt.close(fig)


def fig_final_box(final: dict[str, np.ndarray], out: Path) -> None:
    """Fig 3: boxplot of final best fitness per config, runs overlaid."""
    names = list(final)
    fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=200)
    box = ax.boxplot(
        [final[n] for n in names],
        tick_labels=[SHORT[n] for n in names],
        widths=0.5,
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 1.5},
        showfliers=False,
    )
    for patch, name in zip(box["boxes"], names, strict=True):
        patch.set_facecolor(COLOURS[name])
        patch.set_alpha(0.35)
        patch.set_edgecolor(COLOURS[name])
    for i, name in enumerate(names, start=1):
        vals = np.sort(final[name])
        offsets = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0.0]
        ax.scatter(i + np.asarray(offsets), vals, s=12, color=COLOURS[name],
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_ylabel("final best fitness (lower is better)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig3_final_best.png")
    plt.close(fig)


# ============================================================================ #
#  4. TABLES
# ============================================================================ #


def final_best_table(final: dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean +- std (ddof=1) and median / IQR of final best fitness."""
    rows = []
    for name, vals in final.items():
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        rows.append({
            "config": SHORT[name],
            "n": len(vals),
            "mean": vals.mean(),
            "std": vals.std(ddof=1) if len(vals) > 1 else float("nan"),
            "median": med,
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "min": vals.min(),
            "max": vals.max(),
        })
    return pd.DataFrame(rows)


def tests_table(
    final: dict[str, np.ndarray],
    at_h2: dict[str, np.ndarray] | None,
) -> pd.DataFrame:
    """MWU + Holm over the three final-best pairs, plus H2 at generation 10."""
    rows = []
    pairs = [(a, b) for a, b in combinations(CONFIGS, 2) if a in final and b in final]
    for a, b in pairs:
        u, p = mwu(final[a], final[b])
        rows.append({
            "test": "final best",
            "X": SHORT[a],
            "Y": SHORT[b],
            "n_X": len(final[a]),
            "n_Y": len(final[b]),
            "U": u,
            "p": p,
            "A12_P(X<Y)": a12(final[a], final[b]),
        })
    for row, p_adj in zip(rows, holm([r["p"] for r in rows]), strict=True):
        row["p_holm"] = p_adj
    if at_h2 is not None:
        u, p = mwu(at_h2["point"], at_h2["subtree"])
        rows.append({
            "test": f"H2 best at gen {H2_GENERATION}",
            "X": "P",
            "Y": "S",
            "n_X": len(at_h2["point"]),
            "n_Y": len(at_h2["subtree"]),
            "U": u,
            "p": p,
            "A12_P(X<Y)": a12(at_h2["point"], at_h2["subtree"]),
            "p_holm": float("nan"),
        })
    return pd.DataFrame(rows)


def size_guard_table(runs: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """Total size-guard rejections and fallbacks per EA config."""
    rows = []
    for name in ("point", "subtree"):
        if name not in runs:
            continue
        rej = [int(df["size_rejections"].sum()) for df in runs[name]]
        fb = [int(df["size_fallbacks"].sum()) for df in runs[name]]
        offspring = sum(int(df["evals"].iloc[-1] - df["evals"].iloc[0]) for df in runs[name])
        rows.append({
            "config": SHORT[name],
            "runs": len(runs[name]),
            "offspring_total": offspring,
            "size_rejections_total": sum(rej),
            "size_fallbacks_total": sum(fb),
            "rejections_per_offspring": sum(rej) / offspring if offspring else float("nan"),
            "rejections_per_run_mean": float(np.mean(rej)),
            "fallbacks_per_run_mean": float(np.mean(fb)),
        })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Markdown pipe table, floats to 4 significant digits (no tabulate)."""
    def cell(v: object) -> str:
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    lines = [
        "| " + " | ".join(df.columns) + " |",
        "|" + "|".join("---" for _ in df.columns) + "|",
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save_table(df: pd.DataFrame, out: Path, stem: str) -> None:
    """Write a table as CSV (full precision) and Markdown (4 significant)."""
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(to_markdown(df) + "\n", encoding="utf-8")


# ============================================================================ #
# ============================================================================ #


def render_body(genotype: dict, save_path: Path) -> None:
    """Build the body in MuJoCo and save one still frame (offscreen)."""
    import mujoco as mj  # noqa: PLC0415

    from ariel.body_phenotypes.robogen_lite.constructor import (  # noqa: PLC0415
        construct_mjspec_from_graph,
    )
    from ariel.simulation.environments import SimpleFlatWorld  # noqa: PLC0415
    from ariel.utils.renderers import single_frame_renderer  # noqa: PLC0415

    mj.set_mjcb_control(None)

    world = SimpleFlatWorld()
    robot = construct_mjspec_from_graph(genotype_to_graph(genotype))
    world.spawn(robot.spec, position=SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    single_frame_renderer(
        model,
        data,
        width=640,
        height=640,
        cam_fovy=RENDER_FOVY,
        save=True,
        save_path=str(save_path),
    )


def render_best_bodies(data: Path, runs: dict[str, list[pd.DataFrame]], out: Path) -> None:
    """Render the best final body of each config (lowest final best, then seed)."""
    for name, frames in runs.items():
        finals = [(float(df["best"].iloc[-1]), df["seed_dir"].iloc[0]) for df in frames]
        best_fit, seed_dir = min(finals)
        payload = json.loads((data / name / seed_dir / "best_genome.json").read_text())
        save_path = out / f"best_body_{name}.png"
        try:
            render_body(payload["genotype"], save_path)
            console.log(f"rendered {name} {seed_dir} (fitness {best_fit:.4f}, "
                        f"{payload['num_nodes']} nodes) -> {save_path}")
        except Exception as exc:  # noqa: BLE001
            console.log(f"[yellow]could not render {name}: {exc}[/yellow]")


# ============================================================================ #
#  6. ENTRY POINT
# ============================================================================ #


def main() -> None:
    """Build every figure and table from the run directories."""
    parser = argparse.ArgumentParser(description="EC A1 analysis")
    parser.add_argument("--data", type=Path, default=Path("__data__") / "A1")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--no-render", action="store_true", help="skip MuJoCo renders")
    parser.add_argument("--self-test", action="store_true", help="check stats code and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    self_test()

    runs = load_runs(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frames in runs.items():
        gens = sorted({int(df["gen"].iloc[-1]) for df in frames})
        evals = sorted({int(df["evals"].iloc[-1]) for df in frames})
        console.log(f"{LABELS[name]:20s} {len(frames)} runs, last gen {gens}, final evals {evals}")
        if len(gens) != 1 or len(evals) != 1:
            msg = f"{name}: runs differ in length (last gens {gens}, final evals {evals})"
            raise ValueError(msg)
    all_evals = {int(df["evals"].iloc[-1]) for frames in runs.values() for df in frames}
    if len(all_evals) != 1:
        msg = f"configs used different evaluation budgets: {sorted(all_evals)}"
        raise ValueError(msg)

    target_sizes = [t.number_of_nodes() for t in load_targets()]
    target_mean_size = float(np.mean(target_sizes))
    console.log(f"target sizes {target_sizes} -> mean {target_mean_size:.2f} nodes")

    final = {name: values_at(frames, "best", None) for name, frames in runs.items()}
    at_h2 = None
    if "point" in runs and "subtree" in runs and all(
        (df["gen"] == H2_GENERATION).any() for n in ("point", "subtree") for df in runs[n]
    ):
        at_h2 = {n: values_at(runs[n], "best", H2_GENERATION) for n in ("point", "subtree")}
    else:
        console.log(f"[yellow]H2 skipped: generation {H2_GENERATION} not in every P/S run[/yellow]")

    fig_convergence(runs, args.out)
    fig_size(runs, target_mean_size, args.out)
    fig_final_box(final, args.out)

    tables = {
        "final_best": final_best_table(final),
        "tests": tests_table(final, at_h2),
        "size_guard": size_guard_table(runs),
    }
    for stem, df in tables.items():
        save_table(df, args.out, stem)
        console.rule(stem)
        console.print(to_markdown(df))

    if not args.no_render:
        render_best_bodies(args.data, runs, args.out)
    console.log(f"all output in {args.out}")


if __name__ == "__main__":
    main()
