"""Callbacks that write the usage log (docs/usage-log.md) and the trace of a run, one JSON object per line."""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

TEXT_LIMIT = 2000
USAGE_AGENT = {"single": "single", "compaction": "compaction", "orchestrator": "orchestrator"}  # any other: subagent


def short(value, limit: int = TEXT_LIMIT) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + f" ... [{len(text) - limit} characters more]"


def message_record(m) -> dict:
    record = {"role": m.type, "content": short(m.content)}
    if isinstance(m, AIMessage) and m.tool_calls:
        record["tool_calls"] = [{"name": c["name"], "args": short(c["args"])} for c in m.tool_calls]
    return record


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def scope(metadata: dict) -> str:
    """The subagent run a callback belongs to: "" for the main agent, "tools:<task id>" inside a subagent."""
    return metadata.get("langgraph_checkpoint_ns", "").rpartition("|")[0]


class Recorder(BaseCallbackHandler):
    """Passed once in the run config; LangChain hands it down to every subagent."""

    def __init__(self, run_id: str, model: str, usage_path: Path, trace_path: Path, main_agent: str = "single"):
        self.run_id, self.model = run_id, model
        self.main_agent = main_agent  # the name of the strategy's main agent: single, compaction or orchestrator
        self.usage_path, self.trace_path = usage_path, trace_path
        self.lock = threading.Lock()  # parallel subagents call back from several threads
        self.started: dict = {}
        self.call_index = 0
        self.totals = {"calls": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        # Failures kept as data in run.json: failed model calls (each 429 retried by the agent counts too), tool
        # calls with malformed arguments, tools that raised, python calls that ended in a traceback or the time
        # limit, and agents stopped by --context-cap.
        self.failures = {"model_errors": 0, "invalid_tool_calls": 0, "tool_errors": 0, "python_errors": 0,
                         "python_timeouts": 0, "context_overflows": 0}

    def _append(self, path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def event(self, kind: str, agent: str | None = None, **data) -> None:
        """A harness event for the trace: a compaction step, the run's start or end."""
        with self.lock:
            self.failures["context_overflows"] += kind == "context_overflow"  # agents stopped by --context-cap
            self._append(self.trace_path, {"type": "event", "kind": kind, "agent": agent or self.main_agent, "time": now_iso(), **data})

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        metadata = metadata or {}
        history = messages[0]
        # Only what is new since the agent's last reply: the rest of the history is already in the trace.
        last_ai = max((i for i, m in enumerate(history) if isinstance(m, AIMessage)), default=-1)
        self.started[run_id] = {"t0": time.perf_counter(), "time": now_iso(), "metadata": metadata,
                                "n_messages": len(history), "new": [message_record(m) for m in history[last_ai + 1:]]}

    def on_llm_end(self, response, *, run_id, **kwargs):
        start = self.started.pop(run_id, None)
        if start is None:
            return
        latency = time.perf_counter() - start["t0"]
        message = response.generations[0][0].message
        usage = message.usage_metadata or {}
        meta = message.response_metadata or {}
        agent = start["metadata"].get("lc_agent_name") or self.main_agent
        cached = (usage.get("input_token_details") or {}).get("cache_read") or 0
        cost = meta.get("cost")
        with self.lock:
            self.call_index += 1
            row = {"run_id": self.run_id, "agent": USAGE_AGENT.get(agent, "subagent"), "call_index": self.call_index,
                   "model": meta.get("model_name") or self.model, "input_tokens": usage.get("input_tokens", 0),
                   "cached_tokens": cached, "output_tokens": usage.get("output_tokens", 0),
                   "cost_usd": float(cost or 0.0), "latency_s": round(latency, 3), "timestamp": start["time"]}
            self._append(self.usage_path, row)
            self._append(self.trace_path, {
                "type": "call", "agent": agent, "scope": scope(start["metadata"]), "call_index": self.call_index,
                "time": start["time"], "latency_s": row["latency_s"], "input_tokens": row["input_tokens"],
                "cached_tokens": cached, "output_tokens": row["output_tokens"], "cost_usd": row["cost_usd"],
                "cost_reported": cost is not None, "n_messages": start["n_messages"], "input": start["new"],
                "output": message_record(message)})
            self.failures["invalid_tool_calls"] += len(getattr(message, "invalid_tool_calls", None) or [])
            for key, value in (("calls", 1), ("input_tokens", row["input_tokens"]), ("cached_tokens", cached),
                               ("output_tokens", row["output_tokens"]), ("cost_usd", row["cost_usd"])):
                self.totals[key] += value

    def on_llm_error(self, error, *, run_id, **kwargs):
        start = self.started.pop(run_id, None) or {"time": now_iso(), "metadata": {}}
        with self.lock:
            self.failures["model_errors"] += 1
        self.event("model_error", agent=start["metadata"].get("lc_agent_name") or self.main_agent,
                   scope=scope(start["metadata"]), error=short(repr(error), 500))

    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, inputs=None, **kwargs):
        self.started[run_id] = {"t0": time.perf_counter(), "time": now_iso(), "metadata": metadata or {},
                                "name": (serialized or {}).get("name") or kwargs.get("name"),
                                "args": inputs if inputs is not None else input_str}

    def _tool_end(self, run_id, result, failed: bool):
        start = self.started.pop(run_id, None)
        if start is None:
            return
        ns = start["metadata"].get("langgraph_checkpoint_ns", "")
        with self.lock:
            text = str(result)
            self.failures["tool_errors"] += failed
            if start["name"] == "python" and not failed:
                self.failures["python_errors"] += "Traceback (most recent call last)" in text
                self.failures["python_timeouts"] += text.startswith("TimeoutError: the code ran longer")
            self._append(self.trace_path, {
                "type": "tool", "agent": start["metadata"].get("lc_agent_name") or self.main_agent, "scope": scope(start["metadata"]),
                # a task call's own namespace is the scope of the subagent it starts
                "starts_scope": ns if start["name"] == "task" else None, "name": start["name"], "time": start["time"],
                "latency_s": round(time.perf_counter() - start["t0"], 3), "args": short(start["args"]),
                "result": short(result), "failed": failed})

    def on_tool_end(self, output, *, run_id, **kwargs):
        content = getattr(output, "content", None)
        if content is None and hasattr(output, "update"):  # a Command from a compaction tool
            content = next((m.content for m in output.update.get("messages", [])[::-1] if hasattr(m, "content")), "")
        self._tool_end(run_id, content if content is not None else str(output), failed=False)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._tool_end(run_id, repr(error), failed=True)
