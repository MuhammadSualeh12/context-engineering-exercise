"""The agent's only way to look at the models: a Python tool with IfcOpenShell and a persistent namespace."""

import io
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MAX_CHARS = 4000
TIMEOUT_S = 60

_local = threading.local()


class _ThreadStdout(io.TextIOBase):
    """Sends print() of a tool thread to that thread's buffer, so parallel subagents do not mix their output."""

    def __init__(self, real):
        self.real = real

    def write(self, s):
        buffer = getattr(_local, "buffer", None)
        return (buffer or self.real).write(s)

    def flush(self):
        self.real.flush()


_real_connect = socket.socket.connect


def _guarded_connect(self, *args, **kwargs):
    # Only tool threads are blocked, the harness's own model calls still go out; a deterrent, not a sandbox.
    if getattr(_local, "buffer", None) is not None:
        raise PermissionError("no network access from the python tool")
    return _real_connect(self, *args, **kwargs)


_real_open = None
OPEN_FILES = "__open_ifc_files__"  # namespace entry holding every file the agent opened


def _keep_alive_open(path, *args, **kwargs):
    """ifcopenshell.open inside a tool call: one file object per path and agent, kept for the agent's whole run.

    IfcOpenShell entities do not keep their file alive, so `walls = f.by_type("IfcWall")` followed by a later
    `f = ifcopenshell.open(...)` left `walls` pointing into freed memory, and geometry on it segfaulted.
    Opening the same path again returns the same object, which also saves the parse time."""
    files = getattr(_local, "files", None)
    if files is None:  # outside a tool call
        return _real_open(path, *args, **kwargs)
    key = (os.path.realpath(os.fspath(path)) if isinstance(path, (str, os.PathLike)) else repr(path),
           repr(args), repr(sorted(kwargs.items())))
    if key not in files:
        files[key] = _real_open(path, *args, **kwargs)
    return files[key]


def _patch_open() -> None:
    global _real_open
    import ifcopenshell

    if ifcopenshell.open is not _keep_alive_open:
        _real_open = ifcopenshell.open
        ifcopenshell.open = _keep_alive_open


def model_paths() -> dict[str, str]:
    """Model name (file stem) to absolute path, with forward slashes so the paths read the same on Windows."""
    return {p.stem: p.as_posix() for p in sorted(MODELS_DIR.glob("*.ifc"))}


def new_namespace() -> dict:
    import ifcopenshell
    import ifcopenshell.util.element

    _patch_open()
    return {"ifcopenshell": ifcopenshell, "models": model_paths(), OPEN_FILES: {}}


def truncate(text: str, limit: int = MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit * 2 // 3], text[-(limit // 3):]
    return (f"{head}\n... [{len(text) - len(head) - len(tail)} characters cut: the output limit is {limit} characters, "
            f"print counts or a few rows instead of whole lists] ...\n{tail}")


def run_code(code: str, namespace: dict, timeout: float = TIMEOUT_S, lock=None) -> str:
    """Exec code in namespace and return what it printed, or the error, cut to MAX_CHARS.

    Calls that share a namespace pass the same lock and run one after another: run at the same time, a call that
    rebinds `f` frees the IFC file another call is still reading, and the whole process segfaults."""
    lock = lock or threading.Lock()
    started = time.monotonic()
    if not lock.acquire(timeout=timeout):
        return (f"TimeoutError: an earlier call of this agent is still running after {timeout:.0f} s, "
                "so this one did not start; try a narrower query")
    if not isinstance(sys.stdout, _ThreadStdout):
        sys.stdout = _ThreadStdout(sys.stdout)
    socket.socket.connect = _guarded_connect
    out = io.StringIO()

    files = namespace.setdefault(OPEN_FILES, {})

    def target():
        _local.buffer = out
        _local.files = files if isinstance(files, dict) else None
        try:
            exec(compile(code, "<python>", "exec"), namespace)
        except BaseException:
            out.write(traceback.format_exc(limit=3))
        finally:
            _local.buffer = _local.files = None
            lock.release()  # by the worker, so a call that timed out keeps the namespace until it really ends

    # A thread cannot be killed: after a timeout it keeps running in the background; the python tool tells the
    # agent so and gives it a fresh namespace (make_python_tool).
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(max(0.0, timeout - (time.monotonic() - started)))
    if worker.is_alive():
        return f"TimeoutError: the code ran longer than {timeout:.0f} s; try a narrower query"
    return truncate(out.getvalue()) or "(no output; use print() to see results)"


def make_python_tool():
    """One tool object; each agent (main agent or one subagent run) gets its own namespace."""
    namespaces: dict[str, dict] = {}
    locks: dict = {}
    creating = threading.Lock()  # parallel first calls must not each build their own namespace

    @tool
    def python(code: str, runtime: ToolRuntime) -> str:
        """Run Python code and return what it prints. Variables persist between calls.
        `ifcopenshell` is imported and `models` maps each model name to its IFC file path,
        e.g. `f = ifcopenshell.open(models["4351"])`. There is no internet access.
        Output is cut to 4000 characters, so print counts or a few rows, not whole lists."""
        # The namespace is "tools:<call id>" for a tool of the main agent and "tools:<task id>|tools:<call id>"
        # inside a subagent; dropping the last part gives one key per agent run.
        key = runtime.config.get("metadata", {}).get("langgraph_checkpoint_ns", "").rpartition("|")[0]
        with creating:
            if key not in namespaces:
                namespaces[key] = new_namespace()
                locks[key] = threading.Lock()
            namespace, lock = namespaces[key], locks[key]
        result = run_code(code, namespace, timeout=TIMEOUT_S, lock=lock)
        if result.startswith("TimeoutError: the code ran longer"):
            # The call runs on in the background and holds the old namespace (and keeps its IFC files alive);
            # waiting for it blocked every later call, so the agent starts over in a fresh namespace.
            with creating:
                if namespaces.get(key) is namespace:
                    namespaces[key], locks[key] = new_namespace(), threading.Lock()
            result += ("\nThe call goes on in the background, so your variables were reset: "
                       "open the models again in the next call.")
        return result

    return python
