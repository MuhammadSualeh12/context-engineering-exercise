"""Run one strategy on one task set and write everything into runs/<run-id>/.

Usage, typed on one line:
    uv run python -m harness.run --strategy {single,compaction,orchestrator} --set n05 --group g07
                                 [--submit-dir submissions] [--threshold 30000]
                                 [--context-cap 48000 | --no-context-cap]
                                 [--model deepseek/deepseek-v4.1-flash-1017] [--run-id ID] [--key KEY.jsonl]

The strategies are the single agent (single), smart compaction (compaction) and the orchestrator with one
ifc-worker subagent per IFC file (orchestrator).
The run writes <run-id>.answers.jsonl, <run-id>.usage.jsonl (docs/usage-log.md), <run-id>.trace.jsonl, run.json
and trace.html into runs/<run-id>/.
With an answer key it also writes the score rows <run-id>.run.csv and <run-id>.questions.csv from score.py.
With --submit-dir submissions it also copies the answers, the usage log and run.json to
submissions/<group>/<run-id>/, the folder a pull request hands in (harness/submission.py).
Every agent has a context cap of 48000 tokens, unless --context-cap N or --no-context-cap sets another
(harness/context_cap.py).
An agent stops when its next model call would send more than the cap.
A single agent or smart-compaction agent then ends the run with status context_overflow, a result to hand in.
An ifc-worker that reaches the cap fails only its own task, and the orchestrator carries on.
A run that fails (timeout, API error, Ctrl-C) still writes what it has; run.json gives the reason and counts
the failed model calls, malformed tool calls and Python errors.
The defaults come from harness/defaults.py.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from langgraph.errors import GraphRecursionError

from harness import submission
from harness import defaults
from harness.agents import STRATEGIES, build_agent, chat_model
from harness.recorder import Recorder, now_iso
from harness.view import write_html

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
SUBMISSIONS = ROOT / "submissions"
# A graph step is one node (model call, tool call, middleware); 500 steps is roughly 150 model calls, a cost guard.
STEP_LIMIT = 500


def task_prompt(questions: list[dict]) -> str:
    lines = [f"Answer the {len(questions)} questions below about the IFC models in `models`.", ""]
    lines += [f"[{q['id']}] {q['question']}\n" for q in questions]
    lines.append("When you are done, reply with one JSON object and nothing else, mapping every question id to its "
                 'answer in the format the question asks for, for example {"q90": 12, "q91": "B", "q92": ["Level 1", '
                 '"Level 2"], "q93": 17.2}. Use null for a question you could not answer.')
    return "\n".join(lines)


def parse_answers(text: str, ids: list[str]) -> dict:
    """The JSON object in the agent's final reply, even inside a ```json fence or after prose with braces.

    Tries a decode at every `{` and keeps the last object that parses and is not nested in an earlier one."""
    decoder, found, pos = json.JSONDecoder(), {}, 0
    while (start := text.find("{", pos)) >= 0:
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(obj, dict):
            found = obj
        pos = end
    return {qid: found[qid] for qid in ids if qid in found}


def final_text(state: dict | None) -> str:
    messages = (state or {}).get("messages", [])
    return next((m.text for m in reversed(messages) if m.type == "ai" and m.text.strip()), "")


def free_run_id(group: str, strategy: str, set_name: str, runs_dir: Path = RUNS,
                submit_dir: Path | None = SUBMISSIONS) -> str:
    """The first <group>-<strategy>-<set>-<n> taken neither in runs/ nor among the group's submissions."""
    def taken(run_id: str) -> bool:
        return (runs_dir / run_id).exists() or bool(submit_dir and (submit_dir / group / run_id).exists())

    n = 1
    while taken(f"{group}-{strategy}-{set_name}-{n}"):
        n += 1
    return f"{group}-{strategy}-{set_name}-{n}"


def failure_status(exc: Exception) -> str:
    text = f"{type(exc).__name__} {exc}".lower()
    if any(s in text for s in ("context length", "context_length", "context window")):
        return "context_overflow"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    return "error"


def run(strategy: str, set_path: Path, model: str, group: str, run_id: str, threshold: int, llm=None,
        key: Path | None = None, runs_dir: Path = RUNS, submit_dir: Path | None = None,
        context_cap: int | None = None) -> dict:
    task = json.loads(set_path.read_text(encoding="utf-8"))
    ids = task["ids"]
    out = runs_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    files = {kind: out / f"{run_id}.{kind}.jsonl" for kind in ("answers", "usage", "trace")}
    for path in files.values():
        path.write_text("", encoding="utf-8")

    recorder = Recorder(run_id, model, files["usage"], files["trace"])
    meta = {"run_id": run_id, "group": group, "strategy": strategy, "task_set": task.get("name", set_path.stem),
            "n_questions": len(ids), "model": model, "threshold": threshold if strategy == "compaction" else None,
            "context_cap": context_cap, "started": now_iso()}
    recorder.event("run_start", agent=strategy, **meta)
    agent = build_agent(strategy, llm or chat_model(model), recorder.event, threshold, context_cap)

    state, status, error, t0 = None, "ok", "", time.perf_counter()
    config = {"callbacks": [recorder], "recursion_limit": STEP_LIMIT}
    try:
        for state in agent.stream({"messages": [("user", task_prompt(task["questions"]))]}, config,
                                  stream_mode="values"):
            pass
    except GraphRecursionError:
        status, error = "step_limit", f"stopped after {STEP_LIMIT} graph steps"
    except KeyboardInterrupt:
        status, error = "interrupted", "stopped with Ctrl-C"
    except Exception as exc:  # fail soft: keep the logs and the answers found so far
        status, error = failure_status(exc), f"{type(exc).__name__}: {exc}"[:1000]

    answers = parse_answers(final_text(state), ids)
    if status == "ok" and len(answers) < len(ids):
        status = "incomplete_answers"
    with files["answers"].open("w", encoding="utf-8") as fh:
        for qid, answer in answers.items():
            fh.write(json.dumps({"id": qid, "answer": answer}, ensure_ascii=False) + "\n")

    meta.update(status=status, error=error, wall_seconds=round(time.perf_counter() - t0, 1),
                n_answered=len(answers), **recorder.totals, **recorder.failures)
    meta["cost_usd"] = round(meta["cost_usd"], 6)
    recorder.event("run_end", agent=strategy, status=status, error=error, wall_seconds=meta["wall_seconds"])
    if key and key.exists():
        row = score_run(meta, set_path, files, key, out)
        meta.update(n_correct=int(row["n_correct"]), accuracy=float(row["accuracy"]))
    (out / "run.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    write_html(out)
    if submit_dir is not None:
        meta["submission"] = submission.write(out, meta, submit_dir).as_posix()
    return meta


def score_run(meta: dict, set_path: Path, files: dict, key: Path, out: Path) -> dict:
    """Score with score.py, which writes the run row and the per-question rows; returns the run row."""
    sys.path.insert(0, str(ROOT))
    import score

    run_id, run_csv = meta["run_id"], out / f"{meta['run_id']}.run.csv"
    score.main([str(files["answers"]), str(key), "--set", str(set_path), "--usage", str(files["usage"]),
                "--csv", str(run_csv), "--questions-csv", str(out / f"{run_id}.questions.csv"),
                "--strategy", meta["strategy"], "--group", meta["group"], "--model", meta["model"],
                "--run-id", run_id])
    with run_csv.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))[-1]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=STRATEGIES, required=True)
    ap.add_argument("--set", required=True, help="task set name from questions/sets/, e.g. n05")
    ap.add_argument("--model", default=defaults.MODEL, help=f"OpenRouter model id (default: {defaults.MODEL})")
    ap.add_argument("--group", required=True, help="the group name from the lecture, e.g. g07")
    ap.add_argument("--threshold", type=int, default=defaults.THRESHOLD,
                    help="compaction threshold: context tokens at which smart compaction warns the agent "
                         f"(default: {defaults.THRESHOLD})")
    cap = ap.add_mutually_exclusive_group()
    cap.add_argument("--context-cap", type=int, default=defaults.CONTEXT_CAP,
                     help="context cap per agent in tokens; an agent whose next call would exceed it stops "
                          f"(default: {defaults.CONTEXT_CAP})")
    cap.add_argument("--no-context-cap", dest="context_cap", action="store_const", const=None,
                     help="no context cap: every agent has the model's full context window")
    ap.add_argument("--run-id", help="default: <group>-<strategy>-<set>-<n> with the first free n")
    ap.add_argument("--key", type=Path, default=ROOT / "answer-key.jsonl",
                    help="answer key for scoring; the student repo has none, GitHub scores the hand-ins")
    ap.add_argument("--submit-dir", type=Path, help="also write the hand-in folder <dir>/<group>/<run-id>/, "
                                                    "e.g. --submit-dir submissions")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = parser()
    args = ap.parse_args(argv)

    set_path = ROOT / "questions" / "sets" / f"{args.set}.json"
    if not set_path.exists():
        ap.error(f"no task set {set_path}; check the --set name, and if questions/sets/ is missing, get the "
                 "questions with git pull (README, On the lecture day)")
    if not os.environ.get("OPENROUTER_API_KEY"):
        ap.error("OPENROUTER_API_KEY is not set in this terminal window; set it as in the README, "
                 "then run `uv run python -m harness.check`")
    if not submission.NAME.match(args.group):
        ap.error("--group takes letters, digits, '-', '_' and '.' only, e.g. g07")
    submit_dir = args.submit_dir.resolve() if args.submit_dir else None
    run_id = args.run_id or free_run_id(args.group, args.strategy, args.set, RUNS, submit_dir or SUBMISSIONS)
    if (RUNS / run_id).exists() or (submit_dir and (submit_dir / args.group / run_id).exists()):
        ap.error(f"run {run_id} exists already; pick another --run-id")
    if not submission.NAME.match(run_id):
        ap.error("--run-id takes letters, digits, '-', '_' and '.' only")

    if args.context_cap is not None and args.context_cap <= 0:
        ap.error("--context-cap takes a positive number of tokens")
    meta = run(args.strategy, set_path, args.model, args.group, run_id, args.threshold, key=args.key,
               submit_dir=submit_dir, context_cap=args.context_cap)
    print(f"\n{run_id}: {meta['status']}, {meta['n_answered']}/{meta['n_questions']} answered, {meta['calls']} calls, "
          f"{meta['input_tokens']} tokens in ({meta['cached_tokens']} cached), {meta['output_tokens']} out, "
          f"{meta['cost_usd']:.4f} USD, {meta['wall_seconds']} s")
    if meta["error"]:
        print(f"error: {meta['error']}")
    if meta["status"] != "ok":
        print(f"status {meta['status']}: see How a run ends in the README")
    if "accuracy" in meta:
        print(f"score: {meta['n_correct']}/{meta['n_questions']} correct")
    else:
        print(f"no answer key at {args.key}: answers saved, scoring skipped")
    print(f"trace: uv run python -m harness.view runs/{run_id}")
    if "submission" in meta:
        print(f"hand-in: {meta['submission']} (commit it and open a pull request)")
    return 0 if meta["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
