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
import seaborn as sns
from matplotlib.lines import Line2D
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
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COL_W: float = 3.33
PAGE_W: float = 7.0
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


def set_style() -> None:
    """Shared seaborn theme for every figure: paper context, hairline grid, sans."""
    sns.set_theme(
        context="paper",
        style="ticks",
        palette=[COLOURS[name] for name in CONFIGS],
        font="sans-serif",
        rc={
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "lines.linewidth": 1.5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        },
    )


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    """Save a figure as PNG (preview) and PDF (vector, for the LaTeX report)."""
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}")
    plt.close(fig)


def _band(ax: plt.Axes, stats: pd.DataFrame, name: str, style: str, *,
          band: bool = True) -> None:
    """Line at the across-run mean, optionally with a +-1 sample std band."""
    x = stats.index.to_numpy()
    ax.plot(x, stats["mean"], color=COLOURS[name], linestyle=style)
    if band:
        ax.fill_between(x, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                        color=COLOURS[name], alpha=0.18, linewidth=0)


def _end_label(ax: plt.Axes, stats: pd.DataFrame, text: str) -> None:
    """Direct label just right of a line's last point, in secondary ink."""
    ax.annotate(text, xy=(stats.index[-1], stats["mean"].iloc[-1]),
                xytext=(3, 0), textcoords="offset points",
                va="center", ha="left", fontsize=7, color=INK_2)


def _shared_legend(fig: plt.Figure, names: list[str], styles: dict[str, str]) -> None:
    """One legend for all panels: colour = algorithm, line style = statistic."""
    handles = [Line2D([], [], color=COLOURS[n], linewidth=2.5) for n in names]
    labels = [LABELS[n] for n in names]
    for label, style in styles.items():
        handles.append(Line2D([], [], color=INK_2, linestyle=style))
        labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.2,
               columnspacing=1.4)


def fig_dynamics(runs: dict[str, list[pd.DataFrame]], target_mean_size: float,
                 out: Path) -> None:
    """Fig 1: (a) fitness and (b) body size per generation, mean +- std over runs."""
    fig, (ax_fit, ax_size) = plt.subplots(1, 2, figsize=(PAGE_W, 2.5), sharex=True)
    ea_names = [n for n in ("point", "subtree") if n in runs]

    for name in ea_names:
        best = per_generation(runs[name], "best")
        _band(ax_fit, per_generation(runs[name], "mean"), name, "--", band=False)
        _band(ax_fit, best, name, "-")
        _end_label(ax_fit, best, SHORT[name])
    if "random" in runs:
        best = per_generation(runs["random"], "best")
        _band(ax_fit, best, "random", "-")
        _end_label(ax_fit, best, SHORT["random"])
    ax_fit.set_title("(a) Fitness", loc="left")
    ax_fit.set_ylabel("Fitness (lower is better)")

    for name in ea_names:
        size = per_generation(runs[name], "mean_size")
        _band(ax_size, size, name, "-")
        _end_label(ax_size, size, SHORT[name])
    ax_size.axhline(target_mean_size, color=MUTED, linestyle=":", linewidth=1.2)
    ax_size.annotate(f"target mean ({target_mean_size:.1f})",
                     xy=(0, target_mean_size), xytext=(2, -3), textcoords="offset points",
                     va="top", ha="left", fontsize=7, color=INK_2)
    ax_size.set_title("(b) Mean body size", loc="left")
    ax_size.set_ylabel("Modules per body")

    any_run = next(iter(runs.values()))[0]
    per_gen = int(any_run["evals"].iloc[1] - any_run["evals"].iloc[0]) if len(any_run) > 1 else 0
    for ax in (ax_fit, ax_size):
        ax.set_xlabel(f"Generation ({per_gen} evaluations each)")
        ax.margins(x=0)
        ax.set_xlim(right=ax.get_xlim()[1] * 1.07)
    sns.despine(fig)
    _shared_legend(fig, [n for n in CONFIGS if n in runs],
                   {"best (RS: best so far)": "-", "population mean": "--"})
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    _save(fig, out, "fig1_dynamics")


def fig_final(final: dict[str, np.ndarray], out: Path) -> None:
    """Fig 2: final best fitness per config, box + every run as a point."""
    names = [n for n in CONFIGS if n in final]
    long = pd.DataFrame(
        [{"config": LABELS[n], "fitness": v} for n in names for v in final[n]],
    )
    order = [LABELS[n] for n in names]
    palette = {LABELS[n]: COLOURS[n] for n in names}
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    sns.boxplot(data=long, x="config", y="fitness", hue="config", order=order,
                hue_order=order, palette=palette, width=0.5, showfliers=False,
                boxprops={"alpha": 0.35}, linecolor=INK_2, linewidth=0.8,
                legend=False, ax=ax)
    sns.swarmplot(data=long, x="config", y="fitness", hue="config", order=order,
                  hue_order=order, palette=palette, size=3, edgecolor="white",
                  linewidth=0.4, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Final best fitness (lower is better)")
    sns.despine(fig)
    fig.tight_layout()
    _save(fig, out, "fig2_final_best")


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
#  5. RENDERING THE BEST BODIES
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

    set_style()
    fig_dynamics(runs, target_mean_size, args.out)
    fig_final(final, args.out)

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
