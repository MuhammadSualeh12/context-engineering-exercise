"""The trace page: durations, event lines, agent names and a layout that fits a laptop screen."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import view  # noqa: E402

T = "2026-11-24T01:22:20.000Z"


def call(index, agent="compaction", tools=("python",), latency=4.2):
    return {"type": "call", "agent": agent, "scope": "", "call_index": index, "time": T, "latency_s": latency,
            "input_tokens": 30000, "cached_tokens": 20000, "output_tokens": 300, "cost_usd": 0.001,
            "cost_reported": True, "n_messages": 9, "input": [],
            "output": {"role": "ai", "content": "", "tool_calls": [{"name": n, "args": "{}"} for n in tools]}}


def tool(name="python", latency=0.174, agent="compaction"):
    return {"type": "tool", "agent": agent, "scope": "", "starts_scope": None, "name": name, "time": T,
            "latency_s": latency, "args": "{}", "result": "ok", "failed": False}


def event(kind, agent="compaction", **data):
    return {"type": "event", "kind": kind, "agent": agent, "time": T, **data}


META = {"run_id": "cap48k-compaction-n20-1", "strategy": "compaction", "task_set": "n20", "model": "m/x",
        "status": "ok", "n_questions": 20, "n_answered": 20, "wall_seconds": 111.8}


def test_tool_durations_under_a_second_are_shown_in_milliseconds():
    html = view.render(META, [call(1), tool(latency=0.174), tool(latency=0.001), tool(latency=0.0002),
                              tool(latency=2.345)])
    assert "174 ms" in html and "1 ms" in html and "&lt;1 ms" in html and "2.3 s" in html
    assert "0.0 s" not in html


def test_events_read_as_sentences():
    records = [
        event("warning", tokens=30923, threshold=30000),
        event("drop_old_tool_results", tokens_before=32100, dropped=46, keep_last=1),
        event("summarise_history", tokens_before=33000, removed=12, summary="q1: 4"),
        event("auto_compaction", tokens_before=46000, dropped=7, hard_limit=45000),
        event("context_overflow", agent="ifc-worker", scope="", tokens=49000, cap=48000),
        event("model_error", error="TooManyRequestsResponseError()"),
    ]
    html = view.render(META, records)
    assert "Warning at 30,923 tokens" in html
    assert "Dropped 46 old tool results" in html
    assert "12 messages replaced by a summary" in html and "q1: 4" in html
    assert "Hard limit of 45,000 tokens passed at 46,000 tokens: dropped 7 old tool results" in html
    assert "Context cap of 48,000 tokens reached" in html and "ifc-worker" in html
    assert "Model call failed: TooManyRequestsResponseError()" in html
    assert not re.search(r"\w+=\d", html.split("</style>")[1]), "no raw key=value pairs in the timeline"


def test_unknown_event_kinds_still_show_their_data():
    html = view.render(META, [event("something_new", count=3)])
    assert "something_new" in html and "count" in html and "3" in html


def test_long_tool_names_are_not_cut():
    css = view.CSS
    assert "text-overflow: ellipsis" not in css
    html = view.render(META, [call(1, tools=("drop_old_tool_results",)), tool("drop_old_tool_results", 0.003)])
    assert "&rarr; drop_old_tool_results" in html and "drop_old_tool_results</span>" in html


def test_repeated_tool_calls_of_one_reply_are_counted():
    html = view.render(META, [call(1, agent="orchestrator", tools=("task",) * 11)])
    assert "&rarr; task &times; 11" in html and "task, task" not in html


def test_rows_wrap_instead_of_overflowing_a_narrow_window():
    css = view.CSS
    assert "nowrap" not in css and "flex-wrap: wrap" in css
    assert "min-width: 8em" not in css  # the token bars shrink with the window
    html = view.render(META, [call(1)])
    assert "class='scroll'" in html  # the totals table scrolls inside its own box


def test_the_agent_of_an_old_compaction_trace_is_shown_as_compaction():
    old = [call(1, agent="single"), tool(agent="single"), event("warning", agent="single", tokens=30923, threshold=30000)]
    html = view.render(META, old)
    assert ">single<" not in html and " single " not in html and "#1 compaction" in html
    # a single-agent run keeps its name
    html = view.render({**META, "strategy": "single"}, [call(1, agent="single")])
    assert "#1 single" in html
