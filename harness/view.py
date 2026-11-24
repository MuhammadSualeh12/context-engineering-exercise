"""Turn a run's trace into one self-contained trace.html: totals, then every model call and tool call in order.

Usage:
    uv run python -m harness.view runs/<run-id> [--no-open]
"""

import argparse
import json
import sys
import webbrowser
from html import escape
from pathlib import Path

CSS = """
body { font: 14px/1.4 system-ui, sans-serif; margin: 1.5em auto; max-width: 70em; padding: 0 1em; color: #222; }
h1 { font-size: 1.3em; margin-bottom: .2em; overflow-wrap: anywhere; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; margin: .8em 0; }
td, th { padding: .2em .8em; text-align: right; border-bottom: 1px solid #ddd; }
th:first-child, td:first-child { text-align: left; }
details { margin: .15em 0; }
summary { cursor: pointer; display: flex; flex-wrap: wrap; gap: .1em .8em; align-items: center; }
summary .what { flex: 0 1 15em; min-width: 9em; overflow-wrap: anywhere; }
.num { font-variant-numeric: tabular-nums; flex: 0 1 7em; min-width: 4.5em; text-align: right; color: #555; }
.tool .num { margin-left: auto; }
.bar { flex: 1 1 3em; height: .8em; background: #eee; position: relative; min-width: 3em; }
.bar span { position: absolute; top: 0; bottom: 0; left: 0; }
.bar .input { background: #7aa6d8; } .bar .cached { background: #2f5f99; }
.call > summary { font-weight: 600; }
.tool > summary { color: #555; padding-left: 1.2em; }
.sub { margin: .3em 0 .6em 1.8em; padding-left: .8em; border-left: 3px solid #c9a227; }
.event { margin: .3em 0; padding: .2em .6em; background: #fff3cd; border-left: 3px solid #e0a800; overflow-wrap: anywhere; }
.event .who { color: #555; margin-right: .4em; }
.event.error { background: #f8d7da; border-color: #c82333; }
pre { white-space: pre-wrap; word-break: break-word; background: #f6f6f6; padding: .5em; margin: .3em 0 .3em 1.2em;
      max-height: 30em; overflow: auto; font-size: 12px; }
.role { font-weight: 600; margin-left: 1.2em; }
.status-ok { color: #1e7e34; } .status-bad { color: #c82333; }
"""

JS = """
function toggleAll(open) { document.querySelectorAll('details').forEach(d => d.open = open); }
"""


def read_trace(run_dir: Path) -> tuple[dict, list[dict]]:
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    trace = next(run_dir.glob("*.trace.jsonl"), None)
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()] if trace else []
    return meta, sorted(records, key=lambda r: r["time"])


def agent_names(meta: dict, records: list[dict]) -> list[dict]:
    """Traces written before the rename call the agent of a compaction run "single"; show it as "compaction"."""
    strategy = meta.get("strategy") or next((r.get("strategy") for r in records if r.get("kind") == "run_start"), None)
    if strategy != "compaction":
        return records
    return [{**r, "agent": "compaction"} if r.get("agent") == "single" else r for r in records]


def duration(seconds: float) -> str:
    """Milliseconds under a second, so a fast python call does not read as 0.0 s."""
    if seconds < 0.0005:
        return "&lt;1 ms"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.1f} s"


def totals_by_agent(calls: list[dict]) -> dict:
    agents: dict[str, dict] = {}
    for c in calls:
        a = agents.setdefault(c["agent"], dict.fromkeys(["calls", "input_tokens", "cached_tokens", "output_tokens",
                                                         "cost_usd", "latency_s"], 0))
        a["calls"] += 1
        for k in ("input_tokens", "cached_tokens", "output_tokens", "cost_usd", "latency_s"):
            a[k] += c[k]
    return agents


def pre(text) -> str:
    return f"<pre>{escape(str(text))}</pre>"


def render_call(c: dict, max_input: int) -> str:
    width = 100 * c["input_tokens"] / max_input
    cached = 100 * c["cached_tokens"] / max_input
    cost = f"${c['cost_usd']:.5f}" if c.get("cost_reported", True) else "no cost"
    body = "".join(f"<div class='role'>{escape(m['role'])}</div>{pre(m['content'])}" for m in c["input"])
    out = c["output"]
    body += f"<div class='role'>reply</div>{pre(out['content'] or '(no text)')}"
    body += "".join(f"<div class='role'>calls {escape(t['name'])}</div>{pre(t['args'])}" for t in out.get("tool_calls", []))
    names = [t["name"] for t in out.get("tool_calls", [])]
    calls = ", ".join(escape(name) + (f" &times; {names.count(name)}" if names.count(name) > 1 else "")
                      for name in dict.fromkeys(names)) or "reply"
    return (f"<details class='call'><summary><span class='what'>#{c['call_index']} {escape(c['agent'])} "
            f"&rarr; {calls}</span>"
            f"<span class='bar' title='input tokens, cached part dark'><span class='input' style='width:{width:.1f}%'>"
            f"</span><span class='cached' style='width:{cached:.1f}%'></span></span>"
            f"<span class='num'>{c['input_tokens']:,} in</span><span class='num'>{c['cached_tokens']:,} cached</span>"
            f"<span class='num'>{c['output_tokens']:,} out</span><span class='num'>{cost}</span>"
            f"<span class='num'>{duration(c['latency_s'])}</span></summary>"
            f"<div>{c['n_messages']} messages sent; new since the agent's last reply:</div>{body}</details>")


def n(value) -> str:
    return f"{value:,}" if isinstance(value, (int, float)) else str(value)


def event_text(e: dict) -> str | None:
    """One sentence per harness event; None for a kind this page does not know."""
    g = e.get
    kind = e["kind"]
    if kind == "run_start":
        parts = [f"Run started: {g('strategy')} on {g('task_set')} ({g('n_questions')} questions) with {g('model')}"]
        if g("threshold"):
            parts.append(f"compaction threshold {n(g('threshold'))} tokens")
        parts.append(f"context cap {n(g('context_cap'))} tokens" if g("context_cap") else "no context cap")
        return ", ".join(parts)
    if kind == "run_end":
        text = f"Run ended with status {g('status')} after {g('wall_seconds')} s"
        return text + (f": {g('error')}" if g("error") else "")
    if kind == "warning":
        return f"Warning at {n(g('tokens'))} tokens, over the compaction threshold of {n(g('threshold'))}"
    if kind == "drop_old_tool_results":
        return (f"Dropped {n(g('dropped'))} old tool results at {n(g('tokens_before'))} tokens, "
                f"kept the last {n(g('keep_last'))}")
    if kind == "summarise_history":
        return f"At {n(g('tokens_before'))} tokens, {n(g('removed'))} messages replaced by a summary:"
    if kind == "auto_compaction":
        return (f"Hard limit of {n(g('hard_limit'))} tokens passed at {n(g('tokens_before'))} tokens: "
                f"dropped {n(g('dropped'))} old tool results without the agent")
    if kind == "context_overflow":
        return (f"Context cap of {n(g('cap'))} tokens reached: the next call would send {n(g('tokens'))} tokens, "
                "so the agent stops")
    if kind == "model_error":
        return f"Model call failed: {g('error')}"
    return None


def render_event(e: dict) -> str:
    text = event_text(e)
    if text is None:  # a kind added later: its name and data as they are
        details = {k: v for k, v in e.items() if k not in ("type", "kind", "agent", "time", "scope", "summary")}
        text = f"{e['kind']}: " + ", ".join(f"{k} {v}" for k, v in details.items() if v not in (None, ""))
    summary = pre(e["summary"]) if e.get("summary") else ""
    error = " error" if e["kind"] in ("model_error", "context_overflow") or e.get("status") not in (None, "ok") else ""
    who = "" if e["kind"] in ("run_start", "run_end") else f"<span class='who'>{escape(e['agent'])}</span>"
    return f"<div class='event{error}'>&#9873; {who}{escape(text)}{summary}</div>"


def render_scope(records: list[dict], scope: str, max_input: int) -> str:
    parts = []
    for r in records:
        if r.get("scope", "") != scope:
            continue
        if r["type"] == "call":
            parts.append(render_call(r, max_input))
        elif r["type"] == "event":
            parts.append(render_event(r))
        else:
            mark = "failed " if r["failed"] else ""
            parts.append(f"<details class='tool'><summary><span class='what'>&#9881; {mark}{escape(r['name'])}</span>"
                         f"<span class='num'>{duration(r['latency_s'])}</span></summary>"
                         f"<div class='role'>args</div>{pre(r['args'])}<div class='role'>result</div>{pre(r['result'])}"
                         "</details>")
            if r.get("starts_scope"):
                # the subagent's own calls, drawn under the task call that started it
                parts.append(f"<div class='sub'>{render_scope(records, r['starts_scope'], max_input)}</div>")
    return "\n".join(parts)


def render(meta: dict, records: list[dict]) -> str:
    records = agent_names(meta, records)
    calls = [r for r in records if r["type"] == "call"]
    agents = totals_by_agent(calls)
    total = {k: sum(a[k] for a in agents.values()) for k in ("calls", "input_tokens", "cached_tokens", "output_tokens",
                                                             "cost_usd")}
    rows = "".join(f"<tr><td>{escape(name)}</td><td>{a['calls']}</td><td>{a['input_tokens']:,}</td>"
                   f"<td>{a['cached_tokens']:,}</td><td>{a['output_tokens']:,}</td><td>${a['cost_usd']:.4f}</td>"
                   f"<td>{a['latency_s']:.1f} s</td></tr>" for name, a in agents.items())
    status = meta.get("status", "unknown")
    score = f", score {meta['n_correct']}/{meta['n_questions']}" if "n_correct" in meta else ""
    heading = (f"<h1>{escape(meta.get('run_id', 'run'))}</h1><div>{escape(meta.get('strategy', ''))} on "
               f"{escape(meta.get('task_set', ''))} with {escape(meta.get('model', ''))}, "
               f"<b class='status-{'ok' if status == 'ok' else 'bad'}'>{escape(status)}</b>{escape(score)}, "
               f"{meta.get('n_answered', '?')}/{meta.get('n_questions', '?')} answered, "
               f"{meta.get('wall_seconds', '?')} s wall time</div>")
    if meta.get("error"):
        heading += f"<div class='event error'>{escape(meta['error'])}</div>"
    table = (f"<div class='scroll'><table><tr><th>agent</th><th>calls</th><th>input</th><th>cached</th><th>output</th><th>cost</th>"
             f"<th>model time</th></tr>{rows}<tr><th>total</th><th>{total['calls']}</th><th>{total['input_tokens']:,}</th>"
             f"<th>{total['cached_tokens']:,}</th><th>{total['output_tokens']:,}</th><th>${total['cost_usd']:.4f}</th>"
             "<th></th></tr></table></div>")
    buttons = "<button onclick='toggleAll(true)'>expand all</button> <button onclick='toggleAll(false)'>collapse all</button>"
    max_input = max((c["input_tokens"] for c in calls), default=1) or 1
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{escape(meta.get('run_id', 'trace'))}</title>"
            f"<style>{CSS}</style><script>{JS}</script></head><body>{heading}{table}{buttons}"
            f"<h2>Timeline</h2>{render_scope(records, '', max_input)}</body></html>")


def write_html(run_dir: Path) -> Path:
    meta, records = read_trace(run_dir)
    path = run_dir / "trace.html"
    path.write_text(render(meta, records), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--no-open", action="store_true", help="only write trace.html")
    args = ap.parse_args(argv)
    path = write_html(args.run_dir)
    print(f"wrote {path}")
    if not args.no_open:
        webbrowser.open(path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
