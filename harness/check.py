"""Check the setup before the lecture: packages, OpenRouter key, IFC models, one tiny model call.

Usage:
    uv run python -m harness.check [--model deepseek/deepseek-v4.1-flash-1017] [--offline]
"""

import argparse
import os
import sys
from importlib.metadata import version

from harness.defaults import MODEL as DEFAULT_MODEL
from harness.python_tool import MODELS_DIR, new_namespace, run_code


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


def check_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in this shell")
    return f"OPENROUTER_API_KEY is set ({len(key)} characters)"


def check_call(model: str) -> str:
    from harness.agents import chat_model

    reply = chat_model(model).invoke("Reply with the single word OK.")
    usage = reply.usage_metadata or {}
    cost = reply.response_metadata.get("cost")
    return (f"{model} replied {reply.text.strip()[:40]!r}, {usage.get('input_tokens')} tokens in, "
            f"{usage.get('output_tokens')} out, cost {'not reported' if cost is None else f'{cost:.6f} USD'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--offline", action="store_true", help="skip the key and the model call")
    args = ap.parse_args(argv)
    checks = [("packages", check_packages), ("IFC models", check_models)]
    if not args.offline:
        checks += [("API key", check_key), ("model call", lambda: check_call(args.model))]
    failed = 0
    for name, check in checks:
        try:
            print(f"ok    {name}: {check()}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            if name == "API key":
                break  # the model call cannot work without it
    print("all checks passed" if not failed else f"{failed} check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
