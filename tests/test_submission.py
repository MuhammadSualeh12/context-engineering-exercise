"""The hand-in folder a run writes for its pull request, and the check the pull-request workflow runs on it."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fake_model  # noqa: E402
from fake_model import ScriptedModel  # noqa: E402
from harness import submission  # noqa: E402
from harness.run import free_run_id, run  # noqa: E402

SETS = ROOT / "questions" / "sets"
N05 = SETS / "n05.json"

pytestmark = pytest.mark.skipif(not N05.exists(), reason="questions/ is published on the lecture morning")


def submitted(tmp_path, strategy="single", run_id="g07-single-n05-1", **model):
    fake_model.SEEN_TOOLS.clear()
    fake_model.SEEN_MESSAGES.clear()
    meta = run(strategy, N05, "fake/model", "g07", run_id, 40_000, llm=ScriptedModel(**model),
               runs_dir=tmp_path / "runs", submit_dir=tmp_path / "submissions")
    return meta, tmp_path / "submissions" / "g07" / run_id


def test_a_run_writes_its_submission_folder(tmp_path):
    meta, folder = submitted(tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == ["g07-single-n05-1.answers.jsonl",
                                                        "g07-single-n05-1.usage.jsonl", "run.json"]
    assert (folder / "g07-single-n05-1.answers.jsonl").read_bytes() == \
        (tmp_path / "runs" / "g07-single-n05-1" / "g07-single-n05-1.answers.jsonl").read_bytes()
    saved = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert saved["run_id"] == meta["run_id"] and saved["status"] == "ok"
    assert meta["submission"] == folder.as_posix()
    assert submission.check_folder(folder, SETS) == []


def test_free_run_id_skips_ids_already_submitted(tmp_path):
    (tmp_path / "submissions" / "g07" / "g07-single-n05-1").mkdir(parents=True)
    (tmp_path / "runs").mkdir()
    assert free_run_id("g07", "single", "n05", tmp_path / "runs", tmp_path / "submissions") == "g07-single-n05-2"


def test_a_failed_run_is_still_a_valid_submission(tmp_path):
    meta, folder = submitted(tmp_path, python_calls=3, fail_after=2, fail_message="provider error")
    assert meta["status"] == "error" and submission.check_folder(folder, SETS) == []


def rewrite(path: Path, rows: list) -> None:
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows), encoding="utf-8")


@pytest.mark.parametrize("damage, expected", [
    (lambda f: (f / "notes.txt").write_text("x"), "unexpected file notes.txt"),
    (lambda f: (f / "g07-single-n05-1.usage.jsonl").unlink(), "missing g07-single-n05-1.usage.jsonl"),
    (lambda f: rewrite(f / "g07-single-n05-1.answers.jsonl", ["{not json"]), "not valid JSON"),
    (lambda f: rewrite(f / "g07-single-n05-1.answers.jsonl", [{"id": "q99", "answer": 1}]), "q99 is not in task set n05"),
    (lambda f: rewrite(f / "g07-single-n05-1.answers.jsonl", [{"id": "q50"}]), "without id or answer"),
    (lambda f: rewrite(f / "g07-single-n05-1.usage.jsonl", [{"run_id": "other"}]), "missing agent"),
    (lambda f: (f / "run.json").write_text(json.dumps({"run_id": "g07-single-n05-1", "group": "g08", "strategy": "single",
                                                        "task_set": "n05", "model": "m", "status": "ok"})),
     "group g08 in run.json is not the folder's group g07"),
    (lambda f: (f / "run.json").write_text(json.dumps({"run_id": "g07-single-n05-1", "group": "g07", "strategy": "magic",
                                                        "task_set": "n05", "model": "m", "status": "ok"})),
     "strategy 'magic'"),
    (lambda f: (f / "run.json").write_text(json.dumps({"run_id": "g07-single-n05-1", "group": "g07", "strategy": "single",
                                                        "task_set": "n99", "model": "m", "status": "ok"})),
     "unknown task set 'n99'"),
    (lambda f: (f / "g07-single-n05-1.answers.jsonl").write_text("x" * (submission.MAX_BYTES + 1)), "larger than"),
])
def test_check_folder_reports_each_problem(tmp_path, damage, expected):
    _, folder = submitted(tmp_path)
    damage(folder)
    errors = submission.check_folder(folder, SETS)
    assert any(expected in e for e in errors), errors


def test_usage_lines_of_another_run_are_rejected(tmp_path):
    _, folder = submitted(tmp_path)
    usage = folder / "g07-single-n05-1.usage.jsonl"
    rows = [json.loads(line) for line in usage.read_text(encoding="utf-8").splitlines()]
    rewrite(usage, [dict(rows[0], run_id="g07-single-n05-9")])
    assert any("run_id g07-single-n05-9" in e for e in submission.check_folder(folder, SETS))


def test_check_changed_accepts_only_new_submission_files():
    folders, errors = submission.check_changed([
        "A\tsubmissions/g07/g07-single-n05-1/run.json",
        "A\tsubmissions/g07/g07-single-n05-1/g07-single-n05-1.answers.jsonl",
        "submissions/g08/g08-orchestrator-n10-1/run.json",
    ])
    assert errors == [] and folders == ["submissions/g07/g07-single-n05-1", "submissions/g08/g08-orchestrator-n10-1"]
    _, errors = submission.check_changed(["M\tscore.py", "M\tsubmissions/g07/g07-single-n05-1/run.json",
                                          "A\tsubmissions/g07/run.json", "D\tsubmissions/g09/g09-x-1/run.json",
                                          "A\tsubmissions/../score.py/x/y"])
    assert len(errors) == 5, errors


def test_main_checks_the_changed_folders(tmp_path, capsys):
    _, folder = submitted(tmp_path)
    changed = tmp_path / "changed.txt"
    changed.write_text("".join(f"A\t{p.relative_to(tmp_path).as_posix()}\n" for p in folder.iterdir()), encoding="utf-8")
    assert submission.main(["--changed", str(changed), "--root", str(tmp_path), "--sets", str(SETS)]) == 0
    (folder / "extra.txt").write_text("x")
    assert submission.main(["--changed", str(changed), "--root", str(tmp_path), "--sets", str(SETS)]) == 1
    assert "unexpected file extra.txt" in capsys.readouterr().out
