"""Check the setup: git, the packages, the IFC models, the agent harness, the OpenRouter key and one tiny model call.

Usage:
    uv run python -m harness.check --offline     the evening before: everything except the key and the model call
    uv run python -m harness.check               on the lecture morning, with the key set: everything
    [--model deepseek/deepseek-v4.1-flash-1017]

The model call costs a fraction of a cent.
The last line reads "all checks passed" when the setup is ready; each FAIL line names a problem.
A missing questions/ folder is only a note: the questions come on the lecture morning.
"""

import argparse
import os
import shutil
import subprocess
import sys
from importlib.metadata import version

from harness.defaults import MODEL as DEFAULT_MODEL
from harness.python_tool import MODELS_DIR, new_namespace, run_code

SETS_DIR = MODELS_DIR.parent / "questions" / "sets"


def check_git() -> str:
    if not shutil.which("git"):
        raise RuntimeError("git is not installed or not on the PATH; install it (README, Before the lecture, step 1)")
    return subprocess.run(["git", "--version"], capture_output=True, text=True, check=True).stdout.strip()


def check_packages() -> str:
    names = ["deepagents", "langchain", "langgraph", "langchain-openrouter", "ifcopenshell"]
    return ", ".join(f"{n} {version(n)}" for n in names)


def check_models() -> str:
    from fetch_models import MODELS, is_valid

    bad = [m for m, (_p, _k, size, digest) in MODELS.items() if not is_valid(MODELS_DIR / f"{m}.ifc", size, digest)]
    if bad:
        raise RuntimeError(f"missing or damaged: {', '.join(bad)}; run `uv run fetch_models.py`")
    result = run_code("print(ifcopenshell.open(models['fzk_house']).schema)", new_namespace())
    if "Error" in result:
        raise RuntimeError(result)
    return f"{len(MODELS)} models in {MODELS_DIR}, fzk_house opens as {result.strip()}"


def check_harness() -> str:
    """Build the agent of every strategy on a stand-in model, which needs neither the key nor the network."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    from harness.agents import STRATEGIES, build_agent

    for strategy in STRATEGIES:
        build_agent(strategy, GenericFakeChatModel(messages=iter([])))
    return f"the {', '.join(STRATEGIES)} agents build"


def questions_note() -> str | None:
    """None when the task sets are in the clone, else the note printed instead of a check."""
    if SETS_DIR.is_dir() and any(SETS_DIR.glob("*.json")):
        return None
    return "not in the clone yet; they come on the lecture morning (README, On the lecture day)"


def check_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in this terminal window; set it with "
                           '$env:OPENROUTER_API_KEY = "sk-or-..." in PowerShell, '
                           'export OPENROUTER_API_KEY="sk-or-..." on macOS and Linux')
    return f"OPENROUTER_API_KEY is set ({len(key)} characters)"


def check_call(model: str) -> str:
    from harness.agents import chat_model

    reply = chat_model(model).invoke("Reply with the single word OK.")
    usage = reply.usage_metadata or {}
    cost = reply.response_metadata.get("cost")
    return (f"{model} replied {reply.text.strip()[:40]!r}, {usage.get('input_tokens')} tokens in, "
            f"{usage.get('output_tokens')} out, cost {'not reported' if cost is None else f'{cost:.6f} USD'}")


# Printed under a FAIL line of the check with the same name.
HINTS = {
    "git": "install git as in step 1 of the README, close the terminal and open a new one",
    "packages": "uv run installs the packages; check the internet connection and run the check again",
    "model call": "Unauthorized or 401: the key is wrong or incomplete, set it again; TooManyRequests or 429: "
                  "rate limit, wait a minute and run the check again",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--offline", action="store_true", help="skip the key and the model call")
    args = ap.parse_args(argv)
    checks = [("git", check_git), ("packages", check_packages), ("IFC models", check_models),
              ("harness", check_harness)]
    if not args.offline:
        checks += [("API key", check_key), ("model call", lambda: check_call(args.model))]
    failed = 0
    for name, check in checks:
        try:
            print(f"ok    {name}: {check()}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            if name in HINTS:
                print(f"      {HINTS[name]}")
            if name == "API key":
                break  # the model call cannot work without it
    note = questions_note()
    if note:
        print(f"note  questions: {note}")
    else:
        print(f"ok    questions: {len(list(SETS_DIR.glob('*.json')))} task sets in {SETS_DIR}")
    if args.offline:
        print("      the key and the model call are checked on the lecture morning, without --offline")
    print("all checks passed" if not failed else f"{failed} check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
