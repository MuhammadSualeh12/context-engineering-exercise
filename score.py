"""Score a run's answers against an answer key, deterministically (no LLM judge).

Usage:
    uv run score.py ANSWERS.jsonl KEY.jsonl [--set questions/sets/n10.json] [--usage USAGE.jsonl]
                    [--csv runs.csv --strategy {single,compaction,orchestrator}] [--questions-csv questions.csv]
                    [--run-id ID] [--group NAME] [--model NAME]

ANSWERS.jsonl has one line per question: {"id": "q07", "answer": ...}.
KEY.jsonl is the answer key, supplied by the instructor or by the scoring action.
The questions scored are the ids of --set if given, else every id in the key;
a question with no valid answer line counts as missing, and missing counts as wrong.

Answer rules per type (the type is declared in each question's text):
    integer  an integer, or a string or float holding exactly one ("12", 12.0)
    float    a number, or a string holding one number followed by nothing or the question's unit ("17.2", "17.2 m");
             correct within the tolerance
    enum     one of the listed choices; case and surrounding whitespace are ignored, a number is compared as text
    set      a JSON list of strings (or a string holding one); order and duplicates are ignored,
             case too unless the key says case_sensitive
    mcq      one option letter A to D: "B", "b", "(B)", "B.", "B)", or the letter followed by the exact option text
             ("B. About 85 m"); a hedge such as "B or C" is wrong

Prints one line per question and the accuracy.
--csv appends one run row (header written if the file is new or empty), with usage totals from --usage,
a harness log in the format of docs/usage-log.md; --questions-csv appends one row per question.
Only the basename of each input file is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

NUMBER = re.compile(r"^\s*([-+]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$", re.S)
BARE_LETTER = re.compile(r"^\(?([A-Da-d])\)?\.?$")
LETTER_TEXT = re.compile(r"^\(?([A-Da-d])(?:\)|\.|:|\)\.)\s+(.+)$", re.S)
STRATEGIES = ["single", "compaction", "orchestrator"]
AGENTS = {"orchestrator", "subagent", "single"}
CATEGORIES = [1, 2, 3, 4]
RUN_FIELDS = ["run_id", "group", "model", "strategy", "task_set", "n_questions", "n_correct", "accuracy",
              *(f"cat{c}_correct" for c in CATEGORIES), "n_calls", "input_tokens", "cached_tokens", "output_tokens",
              "cost_usd", "wall_seconds", "answers_file", "usage_file"]
QUESTION_FIELDS = ["run_id", "group", "strategy", "task_set", "id", "category", "type", "status"]
USAGE_FIELDS = {"run_id": str, "agent": str, "call_index": int, "model": str, "input_tokens": int, "cached_tokens": int,
                "output_tokens": int, "cost_usd": (int, float), "latency_s": (int, float), "timestamp": str}


def to_number(value, unit: str | None = None) -> float | None:
    """The number in value; a string must hold one number followed by nothing or the unit."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        m = NUMBER.match(value.replace("\u2212", "-"))
        if m and (not m.group(2) or (unit and norm(m.group(2)) == norm(unit))):
            return float(m.group(1).replace(",", "."))
    return None


def norm(s: str, case_sensitive: bool = False) -> str:
    s = " ".join(str(s).split())
    return s if case_sensitive else s.casefold()


def mcq_letter(answer: str, options: dict | None) -> str | None:
    """The single letter an MCQ answer names, or None for a hedge, a second letter or a text that is not the option's."""
    s = answer.strip()
    m = BARE_LETTER.match(s)
    if m:
        return m.group(1).upper()
    m = LETTER_TEXT.match(s)
    if m and options:
        letter, text = m.group(1).upper(), m.group(2).strip().removesuffix(".")
        if letter in options and norm(text) == norm(str(options[letter]).removesuffix(".")):
            return letter
    return None


def is_correct(key: dict, answer) -> bool:
    """True when answer matches the key entry; None or a malformed answer is simply wrong."""
    if answer is None:
        return False
    t = key["type"]
    if t == "integer":
        x = to_number(answer)
        return x is not None and x == int(x) and int(x) == key["answer"]
    if t == "float":
        x = to_number(answer, key.get("unit"))
        return x is not None and abs(x - key["answer"]) <= key["tolerance"] + 1e-9
    if t == "enum":
        return isinstance(answer, (str, int, float)) and not isinstance(answer, bool) and norm(str(answer)) == norm(str(key["answer"]))
    if t == "set":
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                return False
        if not isinstance(answer, list) or not all(isinstance(a, (str, int)) and not isinstance(a, bool) for a in answer):
            return False
        cs = key.get("case_sensitive", False)
        return {norm(str(a), cs) for a in answer} == {norm(str(g), cs) for g in key["answer"]}
    if t == "mcq":
        return isinstance(answer, str) and mcq_letter(answer, key.get("options")) == key["answer"]
    raise ValueError(f"unknown key type {t!r} for {key.get('id')}")


def read_jsonl(path: Path, warnings: list[str] | None = None) -> list[dict]:
    """JSON objects, one per line; with a warnings list, a bad line is reported there and skipped, else it aborts."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                problem = None if isinstance(row, dict) else f"a JSON {type(row).__name__}, not an object"
            except json.JSONDecodeError as exc:
                problem = f"not valid JSON ({exc.msg})"
            if problem is None:
                rows.append(row)
            elif warnings is None:
                raise SystemExit(f"{path}:{n}: {problem}")
            else:
                warnings.append(f"{path.name}:{n}: {problem}, line ignored")
    return rows


def load_answers(path: Path) -> tuple[dict, list[str]]:
    answers, warnings = {}, []
    for row in read_jsonl(path, warnings):
        qid = row.get("id")
        if qid is None or "answer" not in row:
            warnings.append(f"line without id or answer ignored: {json.dumps(row)[:80]}")
        elif qid in answers:
            warnings.append(f"duplicate answer for {qid}: the first one counts")
        else:
            answers[qid] = row["answer"]
    return answers, warnings


def load_usage(path: Path, run_id: str | None) -> tuple[list[dict], list[str]]:
    """Valid usage lines of one run (docs/usage-log.md); an invalid line is warned about and skipped."""
    calls, warnings = [], []
    for row in read_jsonl(path, warnings):
        missing = [k for k in USAGE_FIELDS if k not in row]
        wrong = [k for k, t in USAGE_FIELDS.items() if k in row and (not isinstance(row[k], t) or isinstance(row[k], bool))]
        problem = (f"missing {', '.join(missing)}" if missing else f"wrong type for {', '.join(wrong)}" if wrong
                   else f"agent {row['agent']!r} is not one of {', '.join(sorted(AGENTS))}" if row["agent"] not in AGENTS else None)
        if problem is None:
            try:
                row["_start"] = datetime.fromisoformat(row["timestamp"]).timestamp()
            except ValueError:
                problem = f"timestamp {row['timestamp']!r} is not ISO 8601"
        if problem:
            warnings.append(f"{path.name}: usage line {json.dumps(row)[:60]} ignored: {problem}")
        else:
            calls.append(row)
    ids = sorted({c["run_id"] for c in calls})
    if run_id is not None:
        calls = [c for c in calls if c["run_id"] == run_id]
    elif len(ids) > 1:
        raise SystemExit(f"{path} holds several run_id values ({', '.join(ids)}); pick one with --run-id")
    return calls, warnings


def usage_totals(calls: list[dict]) -> dict:
    if not calls:
        return {}
    end = max(c["_start"] + c["latency_s"] for c in calls)
    return dict(n_calls=len(calls), input_tokens=sum(c["input_tokens"] for c in calls),
                cached_tokens=sum(c["cached_tokens"] for c in calls), output_tokens=sum(c["output_tokens"] for c in calls),
                cost_usd=f"{sum(c['cost_usd'] for c in calls):.6f}", wall_seconds=f"{end - min(c['_start'] for c in calls):.1f}")


def score(answers: dict, key: dict, ids: list[str]) -> list[dict]:
    return [dict(id=qid, type=key[qid]["type"], category=key[qid].get("category", ""), answered=qid in answers,
                 answer=answers.get(qid), expected=key[qid]["answer"], correct=is_correct(key[qid], answers.get(qid)))
            for qid in ids]


def score_run(answers_path: Path, key: dict, ids: list[str], set_name: str, usage_path: Path | None = None,
              run_id: str | None = None, strategy: str = "", group: str = "", model: str | None = None):
    """Score one run: (per-question results with a status each, the run row, warnings). Prints nothing."""
    answers, warnings = load_answers(answers_path)
    extra = sorted(set(answers) - set(ids))
    if extra:
        warnings.append(f"answers outside the scored questions ignored: {', '.join(extra)}")
    calls = []
    if usage_path:
        calls, usage_warnings = load_usage(usage_path, run_id)
        warnings += usage_warnings
    results = score(answers, key, ids)
    for r in results:
        r["status"] = "correct" if r["correct"] else ("missing" if not r["answered"] else "wrong")
    n, n_correct = len(results), sum(r["correct"] for r in results)
    accuracy = n_correct / n if n else 0.0
    row = dict(run_id=run_id or (calls[0]["run_id"] if calls else answers_path.stem), group=group,
               model=model or "+".join(sorted({c["model"] for c in calls})), strategy=strategy, task_set=set_name,
               n_questions=n, n_correct=n_correct, accuracy=f"{accuracy:.4f}",
               **{f"cat{c}_correct": sum(1 for r in results if r["correct"] and r["category"] == c) for c in CATEGORIES},
               **usage_totals(calls), answers_file=answers_path.name, usage_file=usage_path.name if usage_path else "")
    return results, row, warnings


def append_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", type=Path, help="run answers, JSONL")
    ap.add_argument("key", type=Path, help="answer key, JSONL")
    ap.add_argument("--set", type=Path, help="task set JSON from questions/sets/; scores only its ids")
    ap.add_argument("--usage", type=Path, help="harness usage log, JSONL (docs/usage-log.md)")
    ap.add_argument("--csv", type=Path, help="append one run row to this CSV file")
    ap.add_argument("--questions-csv", type=Path, help="append one row per scored question to this CSV file")
    ap.add_argument("--run-id", help="run id (default: the usage log's run_id, else the answers file stem)")
    ap.add_argument("--strategy", choices=STRATEGIES, help="context strategy of the run; needed with --csv")
    ap.add_argument("--group", default="", help="student group name")
    ap.add_argument("--model", help="model name (default: the usage log's models, joined by +)")
    args = ap.parse_args(argv)
    if (args.csv or args.questions_csv) and not args.strategy:
        ap.error("--csv and --questions-csv need --strategy")

    key = {k["id"]: k for k in read_jsonl(args.key)}
    if args.set:
        task = json.loads(args.set.read_text(encoding="utf-8"))
        ids, set_name = task["ids"], task.get("name", args.set.stem)
    else:
        ids, set_name = list(key), "all"
    unknown = [i for i in ids if i not in key]
    if unknown:
        raise SystemExit(f"ids missing from the answer key: {', '.join(unknown)}")

    results, row, warnings = score_run(args.answers, key, ids, set_name, args.usage, args.run_id,
                                       args.strategy or "", args.group, args.model)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    for r in results:
        print(f"{r['id']:5} {r['type']:7} {r['status']:8} answer={json.dumps(r['answer'], ensure_ascii=False)} "
              f"expected={json.dumps(r['expected'], ensure_ascii=False)}")
    n, n_correct = len(results), row["n_correct"]
    n_answered = sum(r["answered"] for r in results)
    print(f"accuracy {n_correct}/{n} = {n_correct / n if n else 0.0:.3f} ({n_answered} answered)")

    run_id = row["run_id"]
    if args.csv:
        append_csv(args.csv, RUN_FIELDS, [row])
    if args.questions_csv:
        append_csv(args.questions_csv, QUESTION_FIELDS,
                   [dict(run_id=run_id, group=args.group, strategy=args.strategy, task_set=set_name,
                         **{k: r[k] for k in ("id", "category", "type", "status")}) for r in results])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
