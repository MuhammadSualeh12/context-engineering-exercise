"""Check the hand-in folders submissions/<user>/<run-id>/ before a pull request; the pull request's check runs it too.

Usage:
    uv run python -m harness.submission [FOLDER ...]       check hand-in folders (default: all under submissions/)
    uv run python -m harness.submission --changed FILE     check the changed paths of a pull request

`harness.run --submit-dir submissions` writes the hand-in folder.
It holds exactly three files: <run-id>.answers.jsonl, <run-id>.usage.jsonl (docs/usage-log.md) and run.json,
the run's metadata.
The trace stays in runs/.
The last line reads "0 problem(s)" when every folder passes; each FAIL line names a folder and its problem.
--changed takes the output of `git diff --name-status`, or one path per line.
<user> is the student's GitHub user name, from `harness.run --user`.
A pull request may only add files inside submissions/<user>/<run-id>/, and every folder it touches must pass.
The check needs only the standard library and score.py, so the workflow runs it without installing the harness.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import score  # noqa: E402

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_BYTES = 2_000_000
# Fields of run.json the hand-in keeps; a local score (when the run had a key) and local paths stay out.
DROPPED_FIELDS = {"n_correct", "accuracy", "submission"}
REQUIRED = {"run_id": str, "user": str, "strategy": str, "task_set": str, "model": str, "status": str}


def owner(meta: dict):
    """The user a run belongs to; run.json called it group in the pilot's hand-ins, before students worked alone."""
    return meta.get("user", meta.get("group"))


def file_names(run_id: str) -> list[str]:
    return [f"{run_id}.answers.jsonl", f"{run_id}.usage.jsonl", "run.json"]


def write(run_dir: Path, meta: dict, submit_root: Path) -> Path:
    """Copy a finished run's answers and usage log and write its run.json into submit_root/<user>/<run-id>/."""
    run_id = meta["run_id"]
    dest = Path(submit_root) / meta["user"] / run_id
    dest.mkdir(parents=True, exist_ok=False)
    for name in file_names(run_id)[:2]:
        shutil.copyfile(run_dir / name, dest / name)
    kept = {k: v for k, v in meta.items() if k not in DROPPED_FIELDS}
    (dest / "run.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def check_folder(folder: Path, sets_dir: Path) -> list[str]:
    """Every problem of one hand-in folder, each as one line naming the folder; an empty list means it passes."""
    folder = Path(folder)
    where = f"{folder.parent.name}/{folder.name}"
    if not folder.is_dir():
        return [f"{where}: not a folder"]
    run_id, user = folder.name, folder.parent.name
    errors = [f"{where}: {n} is not a valid name (letters, digits, '-', '_', '.')" for n in (user, run_id)
              if not NAME.match(n)]
    expected = file_names(run_id)
    present = sorted(p.name for p in folder.iterdir())
    errors += [f"{where}: unexpected file {n} (a hand-in holds {', '.join(expected)})" for n in present if n not in expected]
    errors += [f"{where}: missing {n}" for n in expected if n not in present]
    for name in present:
        path = folder / name
        if path.is_symlink() or not path.is_file():
            errors.append(f"{where}: {name} is not a regular file")
        elif path.stat().st_size > MAX_BYTES:
            errors.append(f"{where}: {name} is larger than {MAX_BYTES // 1_000_000} MB")
    if errors:
        return errors

    meta, ids = _check_meta(folder, where, run_id, user, sets_dir, errors)
    warnings: list[str] = []
    answers = score.read_jsonl(folder / expected[0], warnings)
    seen = set()
    for row in answers:
        qid = row.get("id")
        if not isinstance(qid, str) or "answer" not in row:
            warnings.append(f"answer line without id or answer: {json.dumps(row)[:60]}")
        elif ids is not None and qid not in ids:
            warnings.append(f"answer for {qid}: {qid} is not in task set {meta.get('task_set')}")
        elif qid in seen:
            warnings.append(f"duplicate answer for {qid}")
        seen.add(qid)
    calls, usage_warnings = score.load_usage(folder / expected[1], run_id)
    warnings += usage_warnings
    lines = score.read_jsonl(folder / expected[1], [])
    other = sorted({str(r.get("run_id")) for r in lines if "run_id" in r and r.get("run_id") != run_id})
    warnings += [f"usage lines with run_id {o}, not the folder's {run_id}" for o in other]
    return errors + [f"{where}: {w}" for w in warnings]


def _check_meta(folder, where, run_id, user, sets_dir, errors):
    try:
        meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        errors.append(f"{where}: run.json is not valid JSON ({exc})")
        return {}, None
    if not isinstance(meta, dict):
        errors.append(f"{where}: run.json is not a JSON object")
        return {}, None
    meta = dict(meta, user=owner(meta))
    errors += [f"{where}: run.json lacks {k} or it is not a {t.__name__}" for k, t in REQUIRED.items()
               if not isinstance(meta.get(k), t)]
    if meta.get("run_id") != run_id:
        errors.append(f"{where}: run_id {meta.get('run_id')} in run.json is not the folder name {run_id}")
    if meta.get("user") != user:
        errors.append(f"{where}: user {meta.get('user')} in run.json is not the folder name {user}")
    if meta.get("strategy") not in score.STRATEGIES:
        errors.append(f"{where}: strategy {meta.get('strategy')!r} is not one of {', '.join(score.STRATEGIES)}")
    task_set = meta.get("task_set")
    set_path = Path(sets_dir) / f"{task_set}.json"
    if not isinstance(task_set, str) or not NAME.match(task_set) or not set_path.exists():
        errors.append(f"{where}: unknown task set {task_set!r}")
        return meta, None
    return meta, set(json.loads(set_path.read_text(encoding="utf-8"))["ids"])


def check_changed(lines: list[str]) -> tuple[list[str], list[str]]:
    """The hand-in folders a pull request adds files to, and an error for every change outside the rules."""
    folders, errors = [], []
    for line in lines:
        if not line.strip():
            continue
        status, _, path = line.rstrip("\n").partition("\t")
        if not path:
            status, path = "A", status.strip()
        parts = path.split("/")
        if not status.startswith("A"):
            errors.append(f"{path}: a pull request may only add files (git status {status})")
        elif len(parts) != 4 or parts[0] != "submissions" or not all(NAME.match(p) for p in parts[1:]):
            errors.append(f"{path}: a pull request may only add files in submissions/<user>/<run-id>/")
        elif (folder := "/".join(parts[:3])) not in folders:
            folders.append(folder)
    return folders, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", type=Path, help="hand-in folders (default: all under submissions/)")
    ap.add_argument("--changed", type=Path, help="changed paths of a pull request, from git diff --name-status")
    ap.add_argument("--root", type=Path, default=ROOT, help="repository root the changed paths are relative to")
    ap.add_argument("--sets", type=Path, default=ROOT / "questions" / "sets", help="task set folder")
    args = ap.parse_args(argv)
    errors, folders = [], list(args.folders)
    if args.changed:
        changed, errors = check_changed(args.changed.read_text(encoding="utf-8").splitlines())
        folders += [args.root / f for f in changed]
        if not changed and not errors:
            errors.append("the pull request adds no hand-in folder under submissions/<user>/<run-id>/")
    elif not folders:
        folders = sorted(p for p in (args.root / "submissions").glob("*/*") if p.is_dir())
    for folder in folders:
        errors += check_folder(folder, args.sets)
    for e in errors:
        print(f"FAIL  {e}")
    print(f"{len(folders)} hand-in folder(s) checked, {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
