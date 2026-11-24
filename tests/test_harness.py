"""Offline tests of the harness with a scripted chat model: no key, no network, no IFC model needed."""

import json
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fake_model  # noqa: E402
import score  # noqa: E402
from fake_model import ScriptedModel  # noqa: E402
from harness.compaction import DROPPED, ContextBudget, drop_tool_results  # noqa: E402
from harness.python_tool import new_namespace, run_code  # noqa: E402
from harness.run import parse_answers, run  # noqa: E402

N01 = ROOT / "questions" / "sets" / "n01.json"
N05 = ROOT / "questions" / "sets" / "n05.json"

pytestmark = pytest.mark.skipif(not N05.exists(), reason="questions/ is published on the lecture morning")


def fake_run(tmp_path, strategy, task_set=N05, threshold=40_000, context_cap=None, **model):
    fake_model.SEEN_TOOLS.clear()
    fake_model.SEEN_MESSAGES.clear()
    meta = run(strategy, task_set, "fake/model", "test", f"t-{strategy}", threshold, llm=ScriptedModel(**model),
               runs_dir=tmp_path, context_cap=context_cap)
    return meta, tmp_path / f"t-{strategy}"


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("strategy, expected", [
    ("single", [["python"]]),
    ("compaction", [["drop_old_tool_results", "python", "summarise_history"]]),
    ("orchestrator", [["python"], ["task"]]),
])
def test_each_strategy_exposes_only_its_tools(tmp_path, strategy, expected):
    fake_run(tmp_path, strategy)
    assert sorted(map(list, {tuple(t) for t in fake_model.SEEN_TOOLS})) == expected


def test_orchestrator_has_no_python_tool_and_workers_have_own_namespaces(tmp_path):
    code = "n = globals().get('n', 0) + 1\nprint('n', n)"
    _, out = fake_run(tmp_path, "orchestrator", python_calls=2, code=code)
    trace = lines(out / "t-orchestrator.trace.jsonl")
    assert not [r for r in trace if r["type"] == "tool" and r["agent"] == "orchestrator" and r["name"] != "task"]
    results = sorted(r["result"].strip() for r in trace if r["type"] == "tool" and r["name"] == "python")
    assert results == ["n 1", "n 1", "n 2", "n 2"]
    tasks = [r for r in trace if r["type"] == "tool" and r["name"] == "task"]
    assert len(tasks) == 2 and all(any(c.get("scope") == t["starts_scope"] for c in trace) for t in tasks)


@pytest.mark.parametrize("strategy", ["single", "compaction", "orchestrator"])
def test_usage_log_follows_the_documented_schema(tmp_path, strategy):
    meta, out = fake_run(tmp_path, strategy)
    rows = lines(out / f"t-{strategy}.usage.jsonl")
    assert rows and [r["call_index"] for r in rows] == list(range(1, len(rows) + 1))
    for r in rows:
        assert set(r) == set(score.USAGE_FIELDS)
        assert all(isinstance(r[k], t) for k, t in score.USAGE_FIELDS.items())
        assert r["run_id"] == "t-" + strategy and r["cost_usd"] == 0.0001 and r["cached_tokens"] > 0
    agents = {r["agent"] for r in rows}
    assert agents == ({"orchestrator", "subagent"} if strategy == "orchestrator" else {strategy})
    calls, warnings = score.load_usage(out / f"t-{strategy}.usage.jsonl", None)
    assert len(calls) == len(rows) == meta["calls"] and not warnings


@pytest.mark.parametrize("strategy", ["single", "compaction", "orchestrator"])
def test_answers_are_scored_by_score_py_and_the_run_row_written(tmp_path, strategy):
    meta, out = fake_run(tmp_path, strategy)
    task = json.loads(N05.read_text(encoding="utf-8"))
    key = tmp_path / "key.jsonl"  # every answer of the scripted model is 0
    key.write_text("".join(json.dumps({"id": i, "type": "integer", "answer": 0, "category": 2}) + "\n"
                           for i in task["ids"]), encoding="utf-8")
    answers, warnings = score.load_answers(out / f"t-{strategy}.answers.jsonl")
    assert set(answers) == set(task["ids"]) and not warnings and meta["status"] == "ok"
    run_csv = out / "run.csv"
    score.main([str(out / f"t-{strategy}.answers.jsonl"), str(key), "--set", str(N05), "--usage",
                str(out / f"t-{strategy}.usage.jsonl"), "--csv", str(run_csv), "--strategy", strategy])
    row = run_csv.read_text(encoding="utf-8").splitlines()[1].split(",")
    assert row[score.RUN_FIELDS.index("n_correct")] == str(len(task["ids"]))


@pytest.mark.parametrize("strategy", ["single", "compaction", "orchestrator"])
def test_trace_html_is_written_and_well_formed(tmp_path, strategy):
    _, out = fake_run(tmp_path, strategy)
    html = (out / "trace.html").read_text(encoding="utf-8")
    parser = HTMLParser()
    parser.feed(html)
    assert f"t-{strategy}" in html and "Timeline" in html and "cached" in html
    assert ("class='sub'" in html) == (strategy == "orchestrator")


def test_compaction_warns_past_the_threshold_and_drops_old_tool_results(tmp_path):
    _, out = fake_run(tmp_path, "compaction", task_set=N01, threshold=400, python_calls=3, code="print('x' * 2000)")
    kinds = [r["kind"] for r in lines(out / "t-compaction.trace.jsonl") if r["type"] == "event"]
    assert "warning" in kinds and "drop_old_tool_results" in kinds
    warned = [m for m in fake_model.SEEN_MESSAGES if isinstance(m[-1], HumanMessage) and "CONTEXT WARNING" in m[-1].content]
    assert warned
    # the warning is not kept in the history, and the dropped results stay dropped
    final = fake_model.SEEN_MESSAGES[-1]
    assert not [m for m in final if isinstance(m, HumanMessage) and "CONTEXT WARNING" in m.content]
    assert [m for m in final if isinstance(m, ToolMessage) and m.content == DROPPED]


def test_the_compaction_agent_is_named_compaction_in_trace_events_and_page(tmp_path):
    _, out = fake_run(tmp_path, "compaction", task_set=N01, threshold=400, python_calls=3, code="print('x' * 2000)")
    trace = lines(out / "t-compaction.trace.jsonl")
    assert {r["agent"] for r in trace} == {"compaction"}
    html = (out / "trace.html").read_text(encoding="utf-8")
    assert "compaction" in html and ">single<" not in html and " single " not in html


def test_summarise_history_replaces_the_history_with_the_summary(tmp_path):
    meta, out = fake_run(tmp_path, "compaction", task_set=N01, threshold=400, python_calls=2,
                         code="print('x' * 2000)", compact_with="summarise_history")
    assert "summarise_history" in [r["kind"] for r in lines(out / "t-compaction.trace.jsonl") if r["type"] == "event"]
    # system prompt, task, the call to summarise_history and its result: the python call and output are gone
    final = fake_model.SEEN_MESSAGES[-1]
    assert [m.type for m in final] == ["system", "human", "ai", "tool"]
    assert "SUMMARY: nothing yet" in final[-1].content and meta["status"] == "ok"


def test_context_size_counts_the_reasoning_the_provider_sees():
    # Reasoning models get their reasoning back in every call (additional_kwargs, not content): in the pilot the
    # visible text was 18k tokens of an 84k-token context and the 40k warning never fired.
    from harness.compaction import context_tokens
    from langchain_core.messages import AIMessage
    call = {"name": "python", "args": {"code": "print(1)"}, "id": "c1"}
    ai = AIMessage("", tool_calls=[call], additional_kwargs={"reasoning_content": "r" * 400},
                   usage_metadata={"input_tokens": 50_000, "output_tokens": 3_000, "total_tokens": 53_000})
    after = ToolMessage("y" * 4000, tool_call_id="c1", name="python")
    messages = [HumanMessage("task"), ai, after]
    tokens = context_tokens(messages)
    assert 53_000 + 900 <= tokens <= 53_000 + 1100  # the provider's count up to the last reply, then about 4 chars a token
    # without usage figures the estimate still counts the reasoning text
    bare = AIMessage("", tool_calls=[call], additional_kwargs={"reasoning_content": "r" * 40_000})
    assert context_tokens([HumanMessage("task"), bare]) >= 10_000


def test_compaction_warns_on_the_reported_context_size():
    events, seen = [], []
    budget = ContextBudget(threshold=40_000, log_event=lambda kind, **data: events.append(kind))
    from langchain_core.messages import AIMessage
    ai = AIMessage("", tool_calls=[{"name": "python", "args": {"code": "1"}, "id": "c1"}],
                   usage_metadata={"input_tokens": 60_000, "output_tokens": 500, "total_tokens": 60_500})

    class Request:
        messages = [HumanMessage("task"), ai, ToolMessage("ok", tool_call_id="c1", name="python")]

        def override(self, messages):
            r = Request()
            r.messages = messages
            return r

    budget.wrap_model_call(Request(), lambda r: seen.append(r.messages))
    assert events == ["warning"] and "CONTEXT WARNING" in seen[0][-1].content


def test_hard_fallback_drops_old_tool_results_without_the_agent():
    events = []
    budget = ContextBudget(threshold=100, log_event=lambda kind, **data: events.append(kind))
    messages = [HumanMessage("task", id="h")] + [
        ToolMessage("y" * 400, tool_call_id=f"c{i}", name="python", id=f"t{i}") for i in range(4)]
    update = budget.before_model({"messages": messages}, None)
    assert [m.id for m in update["messages"]] == ["t0", "t1"] and events == ["auto_compaction"]
    assert drop_tool_results(update["messages"] + messages[3:], keep_last=2) == []


@pytest.mark.parametrize("message, status", [("provider error", "error"),
                                             ("This endpoint's maximum context length is 8192 tokens", "context_overflow")])
def test_a_failing_run_still_writes_its_files(tmp_path, message, status):
    meta, out = fake_run(tmp_path, "single", python_calls=3, fail_after=2, fail_message=message)
    assert meta["status"] == status and message in meta["error"]
    assert len(lines(out / "t-single.usage.jsonl")) == 2
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["status"] == status
    assert (out / "trace.html").exists() and (out / "t-single.answers.jsonl").exists()


@pytest.mark.parametrize("message, status", [("Request timed out.", "timeout"),
                                             ("Error code: 400 - parallel tool calls are not supported", "error")])
def test_a_failed_model_call_is_classified_and_counted(tmp_path, message, status):
    meta, out = fake_run(tmp_path, "single", python_calls=3, fail_after=1, fail_message=message)
    assert meta["status"] == status and meta["model_errors"] == 1
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["model_errors"] == 1


def test_python_errors_timeouts_and_malformed_tool_calls_are_counted(tmp_path):
    meta, _ = fake_run(tmp_path, "single", python_calls=2, code="1 / 0")
    assert (meta["python_errors"], meta["python_timeouts"], meta["invalid_tool_calls"]) == (2, 0, 0)
    meta, _ = fake_run(tmp_path / "b", "single", invalid_first=True)
    assert meta["invalid_tool_calls"] == 1


def test_parse_answers_finds_the_json_object():
    ids = ["q01", "q02"]
    assert parse_answers('Done.\n```json\n{"q01": 3, "q02": "B", "q99": 1}\n```', ids) == {"q01": 3, "q02": "B"}
    assert parse_answers("no json here", ids) == {}
    assert parse_answers('{"q01": 3,', ids) == {}
    # prose with braces before the JSON object, and a dict that parses inside the answers
    text = 'I used {wall.Name} and set {"a": 1}.\n{"q01": {"x": 1}, "q02": "B"}\nThat is all {really}.'
    assert parse_answers(text, ids) == {"q01": {"x": 1}, "q02": "B"}


@pytest.mark.parametrize("strategy", ["single", "orchestrator"])
def test_a_rate_limited_call_is_retried_and_the_run_ends_ok(tmp_path, monkeypatch, strategy):
    # ChatOpenRouter's max_retries only covers 5xx in the OpenRouter SDK, so a 429 used to end the run.
    from harness import agents
    monkeypatch.setattr(agents, "RETRY_DELAY_S", 0.0, raising=False)
    meta, _ = fake_run(tmp_path, strategy, rate_limited=1)
    assert meta["status"] == "ok" and meta["model_errors"] == 1 and meta["n_answered"] == meta["n_questions"]


def test_a_rate_limited_worker_call_is_retried(tmp_path, monkeypatch):
    # A subagent spec does not inherit the main agent's middleware, so the worker needs its own retry.
    from harness import agents
    monkeypatch.setattr(agents, "RETRY_DELAY_S", 0.0, raising=False)
    meta, out = fake_run(tmp_path, "orchestrator", worker_rate_limited=1)
    tasks = [r for r in lines(out / "t-orchestrator.trace.jsonl") if r["type"] == "tool" and r["name"] == "task"]
    assert meta["status"] == "ok" and meta["model_errors"] == 1 and meta["tool_errors"] == 0
    assert len(tasks) == 2 and not any(t["failed"] for t in tasks)


def test_failing_workers_leave_the_orchestrator_its_answer(tmp_path):
    # A task call that raised was re-raised by langgraph's tool error handler and ended the whole run.
    meta, out = fake_run(tmp_path, "orchestrator", worker_fails=True)
    answers = lines(out / "t-orchestrator.answers.jsonl")
    assert meta["status"] == "ok" and len(answers) == meta["n_questions"]
    results = [m for m in fake_model.SEEN_MESSAGES[-1] if isinstance(m, ToolMessage) and m.name == "task"]
    assert len(results) == 2 and all(m.status == "error" and m.content.startswith("worker failed") for m in results)


# Four python calls of about 1000 tokens each: the context passes 3000 tokens before the final reply.
BIG = {"python_calls": 4, "code": "print('x' * 3900)"}


def test_context_cap_stops_the_single_agent_but_not_compaction(tmp_path):
    assert fake_run(tmp_path / "off", "single", **BIG)[0]["status"] == "ok"
    meta, out = fake_run(tmp_path, "single", context_cap=3000, **BIG)
    assert meta["status"] == "context_overflow" and "context window" in meta["error"]
    assert (meta["context_cap"], meta["context_overflows"], meta["model_errors"]) == (3000, 1, 0)
    events = [r for r in lines(out / "t-single.trace.jsonl") if r["type"] == "event" and r["kind"] == "context_overflow"]
    assert len(events) == 1 and events[0]["cap"] == 3000 and events[0]["tokens"] > 3000
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["status"] == "context_overflow"
    assert (out / "t-single.answers.jsonl").exists() and (out / "trace.html").exists()
    meta, _ = fake_run(tmp_path / "b", "compaction", threshold=1500, context_cap=3000, **BIG)
    assert meta["status"] == "ok" and meta["context_overflows"] == 0


def test_command_line_defaults_come_from_the_pilot():
    # deepseek, compaction at 30000 and a context cap of 48000 on by default, with a flag to turn the cap off
    from harness import check
    from harness.run import parser
    args = parser().parse_args(["--strategy", "single", "--set", "n05", "--user", "octocat"])
    assert (args.model, args.threshold, args.context_cap) == ("deepseek/deepseek-v4.1-flash-1017", 30_000, 48_000)
    assert check.DEFAULT_MODEL == args.model
    off = parser().parse_args(["--strategy", "single", "--set", "n05", "--user", "octocat", "--no-context-cap"])
    assert off.context_cap is None
    with pytest.raises(SystemExit):
        parser().parse_args(["--strategy", "single", "--set", "n05", "--user", "u", "--context-cap", "9",
                             "--no-context-cap"])


def test_user_names_the_run_and_group_still_works_as_a_hidden_alias():
    # Students work alone since 24 Nov: --user is the GitHub user name, --group stays for old commands.
    from harness.run import parser
    base = ["--strategy", "single", "--set", "n05"]
    assert parser().parse_args(base + ["--user", "octocat"]).user == "octocat"
    assert parser().parse_args(base + ["--group", "g07"]).user == "g07"
    text = parser().format_help()
    assert "--user" in text and "group" not in text.lower().replace("mutually_exclusive_group", "")


def test_run_without_a_user_stops_with_a_hint(capsys):
    from harness.run import main
    with pytest.raises(SystemExit):
        main(["--strategy", "single", "--set", "n05"])
    assert "--user" in capsys.readouterr().err


def offline_stubs(monkeypatch):
    from harness import check

    def never(*_args):
        raise AssertionError("the offline check must not touch the key or the model")

    monkeypatch.setattr(check, "check_models", lambda: "11 models (stub)")
    monkeypatch.setattr(check, "check_key", never)
    monkeypatch.setattr(check, "check_call", never)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return check


def test_offline_check_covers_everything_but_the_key_and_the_model_call(monkeypatch, capsys):
    # The Moodle mail quotes the last line, so it must read exactly "all checks passed".
    check = offline_stubs(monkeypatch)
    assert check.main(["--offline"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "all checks passed"
    passed = {l[6:].split(":")[0] for l in lines if l.startswith("ok    ")}
    assert {"git", "packages", "IFC models", "harness"} <= passed
    assert not [l for l in lines if l.startswith("FAIL")]


def test_missing_questions_are_a_note_not_a_failure(monkeypatch, capsys, tmp_path):
    # Before the lecture morning the clone has no questions/, and the offline check still passes.
    check = offline_stubs(monkeypatch)
    monkeypatch.setattr(check, "SETS_DIR", tmp_path / "questions" / "sets")
    assert check.main(["--offline"]) == 0
    out = capsys.readouterr().out
    assert "note  questions:" in out and out.splitlines()[-1] == "all checks passed"


def test_full_check_adds_the_key_and_the_model_call(monkeypatch, capsys):
    check = offline_stubs(monkeypatch)
    monkeypatch.setattr(check, "check_key", lambda: "set")
    monkeypatch.setattr(check, "check_call", lambda model: f"{model} replied 'OK'")
    assert check.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "all checks passed"
    assert any(l.startswith("ok    API key") for l in lines) and any(l.startswith("ok    model call") for l in lines)


def test_the_compaction_hard_fallback_fires_under_the_default_cap():
    from harness.defaults import CONTEXT_CAP, THRESHOLD
    budget = ContextBudget(threshold=THRESHOLD, log_event=lambda kind, **data: None)
    assert budget.hard_limit == 45_000 < CONTEXT_CAP


def test_context_cap_applies_to_each_worker_on_its_own(tmp_path):
    meta, out = fake_run(tmp_path, "orchestrator", context_cap=3000, **BIG)
    assert meta["status"] == "ok" and meta["n_answered"] == meta["n_questions"] and meta["context_overflows"] == 2
    events = [r for r in lines(out / "t-orchestrator.trace.jsonl") if r["type"] == "event" and r["kind"] == "context_overflow"]
    assert {e["agent"] for e in events} == {"ifc-worker"} and len({e["scope"] for e in events}) == 2
    results = [m for m in fake_model.SEEN_MESSAGES[-1] if isinstance(m, ToolMessage) and m.name == "task"]
    assert len(results) == 2 and all(m.content.startswith("worker failed: ContextOverflow") for m in results)


def test_python_tool_keeps_variables_and_cuts_long_output():
    ns = new_namespace()
    assert run_code("x = 41", ns).startswith("(no output")
    assert run_code("print(x + 1)", ns).strip() == "42"
    long = run_code("print('a' * 10000)", ns)
    assert len(long) < 4300 and "characters cut" in long
    assert all("\\" not in p for p in ns["models"].values())


SPANS = """import time
t0 = time.time(); time.sleep(0.3)
spans = globals().setdefault('spans', []); spans.append((t0, time.time()))
print(sorted(spans))"""


def test_parallel_python_calls_of_one_agent_run_one_after_another(tmp_path):
    # They share one namespace: run at the same time, a call that rebinds `f` frees the IFC file another call is
    # still reading, and the process segfaults (pilot runs glm-compaction-n05-1 and mimo-compaction-n05-1).
    _, out = fake_run(tmp_path, "single", task_set=N01, python_calls=3, parallel=3, code=SPANS)
    results = [r["result"] for r in lines(out / "t-single.trace.jsonl") if r["type"] == "tool" and r["name"] == "python"]
    spans = max((eval(r) for r in results), key=len)
    assert len(spans) == 3 and all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


@pytest.mark.skipif(not (ROOT / "models" / "fzk_house.ifc").exists(), reason="needs the IFC models")
def test_parallel_calls_rebinding_a_model_variable_do_not_crash(tmp_path):
    script = tmp_path / "parallel.py"
    script.write_text(f"""import sys, uuid
sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'tests')!r}]
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from fake_model import ScriptedModel
from harness.run import run
geom = '''import ifcopenshell.geom
f = ifcopenshell.open(models["fzk_house"])
s = ifcopenshell.geom.settings()
for w in f.by_type("IfcWall"):
    ifcopenshell.geom.create_shape(s, w)
print("geometry done")'''
reopen = 'f = ifcopenshell.open(models["fantasy_residential_building_1"]); print(len(f.by_type("IfcDoor")))'
class Parallel(ScriptedModel):
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if not any(isinstance(m, AIMessage) for m in messages):
            calls = [{{"name": "python", "args": {{"code": c}}, "id": uuid.uuid4().hex[:8]}} for c in (geom, reopen)]
            return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=calls))])
        return super()._generate(messages, stop, run_manager, **kw)
for k in range(3):
    meta = run("single", Path({str(N01)!r}), "fake/model", "t", f"t{{k}}", 40000, llm=Parallel(python_calls=0),
               runs_dir=Path({str(tmp_path)!r}))
print("survived", meta["status"])
""", encoding="utf-8")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "survived ok" in r.stdout, (r.returncode, r.stderr[-2000:])


@pytest.mark.skipif(not (ROOT / "models" / "fzk_house.ifc").exists(), reason="needs the IFC models")
def test_entities_outlive_rebinding_their_file_variable(tmp_path):
    # IfcOpenShell entities do not keep their file alive: once `f` is rebound, a list taken from the old `f`
    # points into freed memory and geometry on it segfaults (pilot run deepseek-compaction-n10-1).
    script = tmp_path / "rebind.py"
    script.write_text(f"""import gc, sys
sys.path.insert(0, {str(ROOT)!r})
from harness.python_tool import new_namespace, run_code
ns = new_namespace()
print(run_code('f = ifcopenshell.open(models["fzk_house"]); walls = f.by_type("IfcWall"); print(len(walls))', ns))
print(run_code('f = ifcopenshell.open(models["ac20"]); import gc; gc.collect(); print("rebound")', ns))
print(run_code('''import ifcopenshell.geom
s = ifcopenshell.geom.settings()
n = 0
for w in walls:
    n += len(ifcopenshell.geom.create_shape(s, w).geometry.verts)
print("verts", n > 0, walls[0].is_a())''', ns))
print(run_code('g = ifcopenshell.open(models["fzk_house"]); print("same file", len(g.by_type("IfcWall")) == len(walls))', ns))
""", encoding="utf-8")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.returncode, r.stdout[-1000:], r.stderr[-2000:])
    assert "verts True IfcWall" in r.stdout and "same file True" in r.stdout


def test_a_call_after_a_timeout_waits_for_the_one_still_running():
    from harness.python_tool import new_namespace, run_code
    import threading
    ns, lock = new_namespace(), threading.Lock()
    assert run_code("import time; time.sleep(0.6); x = 1", ns, timeout=0.1, lock=lock).startswith("TimeoutError")
    assert "still running" in run_code("x = 2", ns, timeout=0.1, lock=lock)
    time.sleep(0.7)
    assert run_code("print(x)", ns, timeout=1, lock=lock).strip() == "1"


def test_a_call_that_never_ends_does_not_block_the_agent(monkeypatch):
    # Pilot glm-orchestrator-single_4351_n10-1: one worker call ran on past its timeout and every later call of that
    # worker failed with "still running" for 25 minutes; the agent now gets a fresh namespace instead.
    from types import SimpleNamespace
    from harness import python_tool
    monkeypatch.setattr(python_tool, "TIMEOUT_S", 0.2)
    monkeypatch.setattr(python_tool, "new_namespace", lambda: {})
    tool = python_tool.make_python_tool()
    runtime = SimpleNamespace(config={"metadata": {"langgraph_checkpoint_ns": "tools:a|tools:c1"}})
    call = lambda code: tool.func(code=code, runtime=runtime)
    call("x = 1")
    first = call("import time; time.sleep(1.5)")
    assert first.startswith("TimeoutError") and "variables" in first
    assert call("print('x' in globals())").strip() == "False"  # fresh namespace, the call ran


def test_python_tool_reports_errors_timeouts_and_blocks_the_network():
    ns = new_namespace()
    assert "ZeroDivisionError" in run_code("1 / 0", ns)
    t0 = time.perf_counter()
    assert "TimeoutError" in run_code("import time; time.sleep(3)", ns, timeout=0.5)
    assert time.perf_counter() - t0 < 2
    code = "import socket\ns = socket.socket()\ns.connect(('127.0.0.1', 9))"
    assert "no network access" in run_code(code, ns)
