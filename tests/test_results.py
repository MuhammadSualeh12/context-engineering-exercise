"""results.py scores every hand-in into one runs.csv and draws the class plot; offline, with a made-up key."""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fake_model  # noqa: E402
import results  # noqa: E402
import score  # noqa: E402
from fake_model import ScriptedModel  # noqa: E402
from harness.run import run  # noqa: E402

SETS = ROOT / "questions" / "sets"

pytestmark = pytest.mark.skipif(not (SETS / "n05.json").exists(), reason="questions/ is published on the lecture morning")


@pytest.fixture
def class_room(tmp_path):
    """Three hand-ins from two groups, one of them a failed run, one broken folder, and a key where every answer is 0."""
    subs = tmp_path / "submissions"
    for group, strategy, set_name, model in (("g01", "single", "n05", {}), ("g02", "orchestrator", "n05", {}),
                                             ("g02", "single", "n01", {"python_calls": 3, "fail_after": 1})):
        fake_model.SEEN_TOOLS.clear()
        fake_model.SEEN_MESSAGES.clear()
        run(strategy, SETS / f"{set_name}.json", "fake/model", group, f"{group}-{strategy}-{set_name}-1", 40_000,
            llm=ScriptedModel(**model), runs_dir=tmp_path / "runs", submit_dir=subs)
    (subs / "g03" / "g03-single-n05-1").mkdir(parents=True)
    ids = json.loads((SETS / "n05.json").read_text(encoding="utf-8"))["ids"]
    key = tmp_path / "key.jsonl"
    key.write_text("".join(json.dumps({"id": i, "type": "integer", "answer": 0, "category": 2}) + "\n"
                           for i in sorted(set(ids) | set(json.loads((SETS / "n01.json").read_text())["ids"]))),
                   encoding="utf-8")
    return tmp_path, subs, key


def test_every_valid_hand_in_becomes_one_row(class_room, capsys):
    tmp_path, subs, key = class_room
    assert results.main(["--key", str(key), "--submissions", str(subs), "--out", str(tmp_path / "out"), "--no-plot"]) == 0
    with (tmp_path / "out" / "runs.csv").open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert reader.fieldnames == score.RUN_FIELDS + results.META_FIELDS
    assert [r["run_id"] for r in rows] == ["g01-single-n05-1", "g02-orchestrator-n05-1", "g02-single-n01-1"]
    ok, orch, failed = rows
    assert ok["accuracy"] == "1.0000" and ok["status"] == "ok" and int(ok["n_calls"]) > 0
    assert failed["status"] == "error" and failed["model_errors"] == "1" and failed["accuracy"] == "0.0000"
    out = capsys.readouterr().out
    assert "skipped: g03/g03-single-n05-1: missing" in out and "3 runs from 2 group(s)" in out
    assert "expected" not in out


def test_the_context_cap_and_its_overflows_are_columns(tmp_path):
    subs = tmp_path / "submissions"
    run("single", SETS / "n05.json", "fake/model", "g01", "g01-single-n05-1", 40_000, runs_dir=tmp_path / "runs",
        llm=ScriptedModel(python_calls=4, code="print('x' * 3900)"), submit_dir=subs, context_cap=3000)
    key = tmp_path / "key.jsonl"
    key.write_text("".join(json.dumps({"id": i, "type": "integer", "answer": 0, "category": 2}) + "\n"
                           for i in json.loads((SETS / "n05.json").read_text())["ids"]), encoding="utf-8")
    rows, problems = results.score_all(subs, key, SETS)
    assert not problems and len(rows) == 1
    assert (rows[0]["status"], rows[0]["context_cap"], rows[0]["context_overflows"]) == ("context_overflow", 3000, 1)


def test_rescoring_rewrites_the_csv_instead_of_appending(class_room):
    tmp_path, subs, key = class_room
    for _ in range(2):
        results.main(["--key", str(key), "--submissions", str(subs), "--out", str(tmp_path / "out"), "--no-plot"])
    assert len(results.read_csv(tmp_path / "out" / "runs.csv")) == 3


def test_plot_is_written(class_room):
    pytest.importorskip("matplotlib")
    tmp_path, subs, key = class_room
    results.main(["--key", str(key), "--submissions", str(subs), "--out", str(tmp_path / "out")])
    assert (tmp_path / "out" / "plot.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_set_order_puts_single_file_sets_last():
    names = ["single_4351_n10", "n40", "n05", "single_4351_n05", "n01"]
    assert sorted(names, key=results.set_order) == ["n01", "n05", "n40", "single_4351_n05", "single_4351_n10"]
