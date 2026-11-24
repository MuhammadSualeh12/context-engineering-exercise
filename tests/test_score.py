import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import score  # noqa: E402

INT = dict(id="q1", type="integer", category=2, answer=10)
FLT = dict(id="q2", type="float", category=2, answer=17.1846, unit="m", tolerance=0.2)
ENUM = dict(id="q3", type="enum", category=1, answer="Autodesk Revit 2024 (DEU)", choices=["ARCHICAD-64", "Autodesk Revit 2024 (DEU)"])
SET = dict(id="q4", type="set", category=1, answer=["A101", "B101"], case_sensitive=False)
SET_CS = dict(id="q5", type="set", category=1, answer=["0dDCr5N_z4GRbZNqI0x3QH", "1aj$VJZFn2TxepZUBcKpac"], case_sensitive=True)
MCQ = dict(id="q6", type="mcq", category=3, answer="B",
           options={"A": "About 40 m", "B": "About 85 m", "C": "About 160 m", "D": "A dry riser in each stairwell"})
MCQ_D = dict(MCQ, id="q8", category=4, answer="D")
ENUM_NUM = dict(id="q7", type="enum", category=1, answer="130625", choices=["130625", "Default"])


@pytest.mark.parametrize("answer", [10, "10", 10.0, " 10 "])
def test_integer_correct(answer):
    assert score.is_correct(INT, answer)


@pytest.mark.parametrize("answer", [11, 10.5, "ten", True, [10], None, "10 or 14", "10 doors", "10 m"])
def test_integer_wrong(answer):
    assert not score.is_correct(INT, answer)


@pytest.mark.parametrize("answer", [17.1846, 17.0, 17.38, "17.2 m", "17,2", 17, "17.2m", " 17.2 ", "17.2 m "])
def test_float_within_tolerance(answer):
    assert score.is_correct(FLT, answer)


@pytest.mark.parametrize("answer", [16.9, 17.4, "about 17", "m 17.2", None, float("nan"),
                                    "17.2 mm", "17.2 ft", "17.2 or 56.4", "10 or 14", "17.2 m or 56.4 m", "17.2 m²"])
def test_float_outside_tolerance_or_malformed(answer):
    assert not score.is_correct(FLT, answer)


def test_float_unit_with_several_words():
    flush = dict(type="float", answer=6.1, unit="litres per flush", tolerance=0.05)
    assert score.is_correct(flush, "6.1 litres per flush")
    assert not score.is_correct(flush, "6.1 litres")


def test_float_tolerance_boundary_is_inclusive():
    assert score.is_correct(dict(type="float", answer=1.0, tolerance=0.01), 1.01)
    assert not score.is_correct(dict(type="float", answer=1.0, tolerance=0.01), 1.0101)


@pytest.mark.parametrize("answer", ["Autodesk Revit 2024 (DEU)", "autodesk revit 2024 (deu)", "  Autodesk  Revit 2024 (DEU) "])
def test_enum_correct(answer):
    assert score.is_correct(ENUM, answer)


@pytest.mark.parametrize("answer", ["Autodesk Revit 2024", "ARCHICAD-64", 2024, None])
def test_enum_wrong(answer):
    assert not score.is_correct(ENUM, answer)


def test_enum_number_compared_as_string():
    assert score.is_correct(ENUM_NUM, 130625)
    assert score.is_correct(ENUM_NUM, "130625")
    assert not score.is_correct(ENUM_NUM, 130626)
    assert not score.is_correct(ENUM_NUM, True)


@pytest.mark.parametrize("answer", [["A101", "B101"], ["B101", "A101"], ["a101", "B101", "A101"], '["B101", "A101"]'])
def test_set_order_case_duplicates_ignored(answer):
    assert score.is_correct(SET, answer)


@pytest.mark.parametrize("answer", [["A101"], ["A101", "B101", "C101"], "A101, B101", [1, 2], None])
def test_set_wrong(answer):
    assert not score.is_correct(SET, answer)


def test_set_case_sensitive_for_globalids():
    assert score.is_correct(SET_CS, list(reversed(SET_CS["answer"])))
    assert not score.is_correct(SET_CS, [g.lower() for g in SET_CS["answer"]])


@pytest.mark.parametrize("answer", ["B", "b", "(B)", "B.", "B)", " B ", "(b).",
                                    "B. About 85 m", "B) About 85 m", "(B) about  85 m", "B: About 85 m."])
def test_mcq_letter_accepted(answer):
    assert score.is_correct(MCQ, answer)


@pytest.mark.parametrize("answer", ["A", "Bx", "Option B", "E", "", 2, None, "BC",
                                    "B or C", "B,C", "B, C", "B/C", "b) maybe", "B maybe", "B. About 160 m",
                                    "B. About 85 m or C", "(B) or (C)", "B C", "B. About"])
def test_mcq_hedged_or_wrong(answer):
    assert not score.is_correct(MCQ, answer)


def test_mcq_option_text_may_start_with_a_letter():
    assert score.is_correct(MCQ_D, "D. A dry riser in each stairwell")
    assert not score.is_correct(MCQ_D, "D. A")


def test_mcq_text_without_options_in_key_is_wrong():
    bare = dict(id="q9", type="mcq", answer="B")
    assert score.is_correct(bare, "B")
    assert not score.is_correct(bare, "B. About 85 m")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def files(tmp_path):
    key = tmp_path / "key.jsonl"
    write_jsonl(key, [INT, FLT, ENUM, SET, MCQ])
    answers = tmp_path / "run1.jsonl"
    write_jsonl(answers, [dict(id="q1", answer=10), dict(id="q2", answer=20.0), dict(id="q6", answer="B"),
                          dict(id="q6", answer="A"), dict(id="q9", answer="X")])
    task = tmp_path / "set.json"
    task.write_text(json.dumps(dict(name="tiny", ids=["q1", "q2", "q3", "q6"])), encoding="utf-8")
    return key, answers, task


USAGE = [
    dict(run_id="g1-orch-n10", agent="orchestrator", call_index=1, model="m/big", input_tokens=1000, cached_tokens=0,
         output_tokens=50, cost_usd=0.01, latency_s=2.0, timestamp="2026-11-26T10:00:00Z"),
    dict(run_id="g1-orch-n10", agent="subagent", call_index=2, model="m/small", input_tokens=2000, cached_tokens=1500,
         output_tokens=100, cost_usd=0.002, latency_s=3.5, timestamp="2026-11-26T10:00:05Z"),
    dict(run_id="g1-orch-n10", agent="orchestrator", call_index=3, model="m/big", input_tokens=1200, cached_tokens=1000,
         output_tokens=30, cost_usd=0.005, latency_s=1.5, timestamp="2026-11-26T10:00:20.500Z"),
]


def test_end_to_end_missing_duplicates_and_run_row(files, tmp_path, capsys):
    key, answers, task = files
    out = tmp_path / "results.csv"
    assert score.main([str(answers), str(key), "--set", str(task), "--csv", str(out), "--strategy", "single", "--user", "u1",
                       "--model", "m/big"]) == 0
    printed = capsys.readouterr()
    assert "q3    enum    missing" in printed.out
    assert "q2    float   wrong" in printed.out
    assert "accuracy 2/4 = 0.500 (3 answered)" in printed.out
    assert "duplicate answer for q6" in printed.err
    assert "q9" in printed.err
    with out.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == score.RUN_FIELDS
        rows = list(reader)
    assert rows == [dict(run_id="run1", user="u1", model="m/big", strategy="single", task_set="tiny", n_questions="4", n_correct="2",
                         accuracy="0.5000", cat1_correct="0", cat2_correct="1", cat3_correct="1", cat4_correct="0",
                         n_calls="", input_tokens="", cached_tokens="", output_tokens="", cost_usd="", wall_seconds="",
                         answers_file="run1.jsonl", usage_file="")]
    score.main([str(answers), str(key), "--set", str(task), "--csv", str(out), "--strategy", "single"])
    assert len(list(csv.DictReader(out.open()))) == 2
    score.main([str(answers), str(key), "--set", str(task), "--csv", str(out), "--strategy", "single", "--group", "g2"])
    assert list(csv.DictReader(out.open()))[-1]["user"] == "g2"  # --group is a hidden alias of --user


def test_run_row_with_usage_log_and_per_question_csv(files, tmp_path):
    key, answers, task = files
    usage = tmp_path / "sub" / "usage.jsonl"
    usage.parent.mkdir()
    write_jsonl(usage, USAGE)
    out, perq = tmp_path / "runs.csv", tmp_path / "questions.csv"
    score.main([str(answers), str(key), "--set", str(task), "--usage", str(usage), "--csv", str(out),
                "--questions-csv", str(perq), "--strategy", "orchestrator", "--user", "u1"])
    [row] = list(csv.DictReader(out.open(newline="")))
    assert row["run_id"] == "g1-orch-n10"
    assert row["model"] == "m/big+m/small"
    assert (row["n_calls"], row["input_tokens"], row["cached_tokens"], row["output_tokens"]) == ("3", "4200", "2500", "180")
    assert row["cost_usd"] == "0.017000"
    assert row["wall_seconds"] == "22.0"
    assert (row["answers_file"], row["usage_file"]) == ("run1.jsonl", "usage.jsonl")
    assert str(tmp_path) not in out.read_text()
    with perq.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == score.QUESTION_FIELDS
        rows = list(reader)
    assert [(r["run_id"], r["id"], r["category"], r["status"]) for r in rows] == [
        ("g1-orch-n10", "q1", "2", "correct"), ("g1-orch-n10", "q2", "2", "wrong"),
        ("g1-orch-n10", "q3", "1", "missing"), ("g1-orch-n10", "q6", "3", "correct")]
    assert "answer" not in reader.fieldnames


def test_usage_log_bad_lines_are_skipped_with_a_warning(files, tmp_path, capsys):
    key, answers, task = files
    usage = tmp_path / "usage.jsonl"
    bad_agent = dict(USAGE[0], agent="planner")
    no_cost = {k: v for k, v in USAGE[0].items() if k != "cost_usd"}
    usage.write_text("".join(json.dumps(r) + "\n" for r in [USAGE[0], bad_agent, no_cost]) + "{oops\n", encoding="utf-8")
    out = tmp_path / "runs.csv"
    score.main([str(answers), str(key), "--set", str(task), "--usage", str(usage), "--csv", str(out), "--strategy", "single"])
    err = capsys.readouterr().err
    assert "planner" in err and "cost_usd" in err and "not valid JSON" in err
    assert list(csv.DictReader(out.open()))[0]["n_calls"] == "1"


def test_usage_log_of_a_compaction_run_counts_its_agent_old_and_new_name(files, tmp_path, capsys):
    key, answers, task = files
    usage = tmp_path / "usage.jsonl"
    # the agent is "compaction" from now on; the pilot logs of the instructor still say "single"
    usage.write_text("".join(json.dumps(dict(USAGE[0], agent=a, call_index=i)) + "\n"
                             for i, a in enumerate(["compaction", "single"], 1)), encoding="utf-8")
    out = tmp_path / "runs.csv"
    score.main([str(answers), str(key), "--set", str(task), "--usage", str(usage), "--csv", str(out),
                "--strategy", "compaction"])
    assert "is not one of" not in capsys.readouterr().err
    assert list(csv.DictReader(out.open()))[0]["n_calls"] == "2"


def test_usage_log_with_two_run_ids_needs_run_id(files, tmp_path):
    key, answers, task = files
    usage = tmp_path / "usage.jsonl"
    write_jsonl(usage, [USAGE[0], dict(USAGE[1], run_id="other")])
    with pytest.raises(SystemExit, match="run_id"):
        score.main([str(answers), str(key), "--usage", str(usage)])
    out = tmp_path / "runs.csv"
    score.main([str(answers), str(key), "--usage", str(usage), "--run-id", "other", "--csv", str(out), "--strategy", "single"])
    assert list(csv.DictReader(out.open()))[0]["n_calls"] == "1"


def test_strategy_is_a_fixed_choice_and_needed_for_csv(files, tmp_path):
    key, answers, task = files
    with pytest.raises(SystemExit):
        score.main([str(answers), str(key), "--strategy", "subagents"])
    with pytest.raises(SystemExit):
        score.main([str(answers), str(key), "--csv", str(tmp_path / "runs.csv")])


def test_bad_answer_lines_are_warned_and_count_as_missing(files, tmp_path, capsys):
    key, _answers, task = files
    answers = tmp_path / "messy.jsonl"
    answers.write_text('{"id": "q1", "answer": 10}\n{"id": "q2", "answer": 17.2\n[1, 2]\n"q3"\n{"id": "q6", "answer": "B"}\n',
                       encoding="utf-8")
    assert score.main([str(answers), str(key), "--set", str(task)]) == 0
    printed = capsys.readouterr()
    assert "messy.jsonl:2" in printed.err and "messy.jsonl:3" in printed.err and "messy.jsonl:4" in printed.err
    assert "q2    float   missing" in printed.out
    assert "accuracy 2/4 = 0.500 (2 answered)" in printed.out


def test_without_set_scores_every_key_question(files, capsys):
    key, answers, _task = files
    score.main([str(answers), str(key)])
    assert "accuracy 2/5" in capsys.readouterr().out


def test_unknown_set_id_is_an_error(files, tmp_path):
    key, answers, _ = files
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dict(name="bad", ids=["q1", "q42"])), encoding="utf-8")
    with pytest.raises(SystemExit, match="q42"):
        score.main([str(answers), str(key), "--set", str(bad)])
