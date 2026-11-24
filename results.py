"""Score every hand-in under submissions/ and draw the class plot; the scoring workflow runs it after each merge.

Usage:
    uv run --with matplotlib==3.11.2 python results.py --key KEY.jsonl [--submissions submissions] [--out results]

Each folder submissions/<user>/<run-id>/ (harness/submission.py) is checked, then scored with score.py against
KEY.jsonl, the answer key the workflow decodes from a repository secret.
Writes <out>/runs.csv, one row per run with score.py's run columns followed by the run's status and failure
counts from run.json, and <out>/plot.png: accuracy against total tokens per run, and total tokens and wall time
by task set, one colour per strategy, one row of panels per model.
A folder that fails the check is reported and left out. Nothing per question is printed or written.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import score  # noqa: E402
from harness import submission  # noqa: E402

META_FIELDS = ["status", "n_answered", "threshold", "context_cap", "model_errors", "invalid_tool_calls", "tool_errors",
               "python_errors", "python_timeouts", "context_overflows"]
FIELDS = score.RUN_FIELDS + META_FIELDS
STRATEGY_COLOURS = {"single": "#0065BD", "compaction": "#E37222", "orchestrator": "#A2AD00"}  # TUM blue, orange, green
TUM_BLUE_DARKER, TUM_GRAY_20, TUM_GRAY_50, TUM_BLACK = "#003359", "#CCCCCC", "#808080", "#1A1A1A"


def hand_ins(submissions: Path) -> list[Path]:
    return sorted(p for p in Path(submissions).glob("*/*") if p.is_dir())


def score_hand_in(folder: Path, key: dict, sets_dir: Path) -> dict:
    """The run row of one checked hand-in, with the run.json fields of META_FIELDS."""
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    task = json.loads((Path(sets_dir) / f"{meta['task_set']}.json").read_text(encoding="utf-8"))
    run_id = folder.name
    _results, row, _warnings = score.score_run(folder / f"{run_id}.answers.jsonl", key, task["ids"],
                                               task.get("name", meta["task_set"]), folder / f"{run_id}.usage.jsonl",
                                               run_id, meta["strategy"], submission.owner(meta), meta["model"])
    row.update({k: meta.get(k, "") if meta.get(k) is not None else "" for k in META_FIELDS})
    return row


def score_all(submissions: Path, key_path: Path, sets_dir: Path) -> tuple[list[dict], list[str]]:
    key = {k["id"]: k for k in score.read_jsonl(Path(key_path))}
    rows, problems = [], []
    for folder in hand_ins(submissions):
        errors = submission.check_folder(folder, sets_dir)
        if errors:
            problems += errors
            continue
        rows.append(score_hand_in(folder, key, sets_dir))
    rows.sort(key=lambda r: (r["model"], r["user"], r["run_id"]))
    return rows, problems


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def set_order(name: str) -> tuple[int, int, str]:
    """Spread sets first by size (n01 ... n40), then the single-file sets."""
    digits = "".join(c for c in name.rpartition("n")[2] if c.isdigit())
    return (name.startswith("single"), int(digits or 0), name)


def set_label(name: str) -> str:
    return name.replace("single_", "").replace("_", " ") if name.startswith("single") else name


def num(row: dict, field: str) -> float:
    try:
        return float(row.get(field) or 0)
    except ValueError:
        return 0.0


def plot(rows: list[dict], path: Path) -> None:
    import logging

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.edgecolor": TUM_GRAY_20, "axes.linewidth": 0.5, "axes.labelcolor": TUM_BLACK,
        "xtick.color": TUM_BLACK, "ytick.color": TUM_BLACK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": TUM_GRAY_20, "grid.linewidth": 0.4, "axes.axisbelow": True})

    models = sorted({r["model"] for r in rows}) or ["no runs yet"]
    sets = sorted({r["task_set"] for r in rows}, key=set_order)
    strategies = [s for s in STRATEGY_COLOURS if any(r["strategy"] == s for r in rows)] or list(STRATEGY_COLOURS)
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4.2 * len(models) + 0.6), dpi=160, squeeze=False)
    for i, model in enumerate(models):
        mine = [r for r in rows if r["model"] == model]
        acc, tok, wall = axes[i]
        for strategy in strategies:
            colour = STRATEGY_COLOURS[strategy]
            runs = [r for r in mine if r["strategy"] == strategy]
            for r in runs:
                ok = r.get("status", "ok") == "ok"
                acc.scatter(max(num(r, "input_tokens") + num(r, "output_tokens"), 1), num(r, "accuracy"),
                            s=18 + 6 * num(r, "n_questions"), marker="s" if r["task_set"].startswith("single") else "o",
                            facecolor=colour if ok else "none", edgecolor=colour, linewidth=1.2, alpha=0.8)
            offset = (strategies.index(strategy) - (len(strategies) - 1) / 2) * 0.22
            for ax, field in ((tok, None), (wall, "wall_seconds")):
                xs, ys, medians = [], [], []
                for j, s in enumerate(sets):
                    vals = [num(r, "input_tokens") + num(r, "output_tokens") if field is None else num(r, field)
                            for r in runs if r["task_set"] == s]
                    xs += [j + offset] * len(vals)
                    ys += vals
                    medians.append(sorted(vals)[len(vals) // 2] if vals else float("nan"))
                ax.scatter(xs, ys, s=22, color=colour, alpha=0.75, edgecolor="none")
                spread = [j for j, s in enumerate(sets) if not s.startswith("single")]
                ax.plot([j + offset for j in spread], [medians[j] for j in spread], color=colour, linewidth=1.2)
        # Log axes need data: before the first hand-in the panels stay linear and empty.
        if mine:
            acc.set_xscale("log")
        acc.set_ylim(-0.04, 1.04)
        acc.set_xlabel("total tokens per run (input + output, log scale)")
        acc.set_ylabel("accuracy")
        acc.set_title(model, loc="left", color=TUM_BLUE_DARKER, fontsize=11, weight="semibold")
        for ax, label in ((tok, "total tokens per run (log scale)"), (wall, "wall time per run (s, log scale)")):
            if mine:
                ax.set_yscale("log")
            ax.set_xticks(range(len(sets)), [set_label(s) for s in sets])
            ax.set_xlim(-0.6, len(sets) - 0.4)
            ax.set_xlabel("task set")
            ax.set_ylabel(label)
            ax.tick_params(length=0)
        acc.tick_params(length=0)
    handles = [Line2D([], [], marker="o", linestyle="", color=STRATEGY_COLOURS[s], label=s) for s in strategies]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor="none", color=TUM_GRAY_50,
                       label="run failed or incomplete"),
                Line2D([], [], marker="s", linestyle="", color=TUM_GRAY_50, label="single-file set"),
                Line2D([], [], marker="o", linestyle="", color=TUM_GRAY_50, markersize=3, label="marker size: questions"),
                Line2D([], [], color=TUM_GRAY_50, label="median over runs")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.05 if len(models) == 1 else 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", type=Path, required=True, help="answer key, JSONL")
    ap.add_argument("--submissions", type=Path, default=ROOT / "submissions")
    ap.add_argument("--sets", type=Path, default=ROOT / "questions" / "sets")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--no-plot", action="store_true", help="write runs.csv only")
    args = ap.parse_args(argv)
    rows, problems = score_all(args.submissions, args.key, args.sets)
    for p in problems:
        print(f"skipped: {p}")
    write_csv(rows, args.out / "runs.csv")
    if not args.no_plot:
        plot(rows, args.out / "plot.png")
    users = len({r["user"] for r in rows})
    print(f"{len(rows)} runs from {users} student(s) scored into {args.out / 'runs.csv'}"
          f"{'' if args.no_plot else ' and plot.png'}; {len(problems)} problem(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
