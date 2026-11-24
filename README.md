# Context engineering exercise

Student exercise for lecture 7 at TUM on 26 November 2026.
Three agent strategies answer questions about IFC building models: a single agent, a single agent that compacts its own context, and an orchestrator that hands the questions to worker agents.

## Questions

The question sets in `questions/` are added to this repository on the morning of the lecture.
Until then the setup and its check work, but `harness.run` has no task set to run.
On the lecture day, bring the fork up to date: press Sync fork on the fork's GitHub page and run `git pull` in the clone, or run this in the clone:

```
git pull https://github.com/sylvainHellin/context-engineering-exercise.git main
```

## Setup

1. Fork this repository on GitHub and clone the fork.
2. Install uv (https://docs.astral.sh/uv/).
3. Run `uv run fetch_models.py` to download the eleven IFC models into `models/`.
4. Set the OpenRouter key handed out for the lecture in the environment variable `OPENROUTER_API_KEY`
   (`export OPENROUTER_API_KEY=...` on macOS and Linux, `$env:OPENROUTER_API_KEY = "..."` in PowerShell).
5. Check the setup before the lecture; it makes one model call of a fraction of a cent:

   ```
   uv run python -m harness.check
   ```

## Running the agent

Run one strategy (`single`, `compaction` or `orchestrator`) on one task set, with the group name from the lecture:

```
uv run python -m harness.run --strategy single --set n05 --group g07 --submit-dir submissions
```

The run uses the exercise defaults:

- The model is `deepseek/deepseek-v4.1-flash-1017`; `--model` takes another OpenRouter model id.
- The compaction strategy warns the agent once its context passes 30000 tokens (`--threshold`) and drops old tool results by itself at 1.5 times that, 45000 tokens.
- Every agent has a context window of 48000 tokens (`--context-cap`).
  A single agent that would go past it ends the run with status `context_overflow`; an orchestrator worker that would go past it fails only its own task.
- `--no-context-cap` gives every agent the model's full context window.

Everything the run produces lands in `runs/<run-id>/`, which git ignores.
Open the trace of a run, every model call and tool call with its tokens and cost, in the browser:

```
uv run python -m harness.view runs/<run-id>
```

With `--submit-dir submissions` the run also writes its hand-in to `submissions/<group>/<run-id>/`:
the answers, the usage log and `run.json`, the run's metadata, without the trace.
A failed or interrupted run hands in what it got; its failure is part of the results.

## Handing in a run

No answer key ships with the exercise: the runs are scored on GitHub after the lecturer merges them.

1. Check the hand-in folders: `uv run python -m harness.submission`.
2. Commit only the new folders under `submissions/<group>/` and push them to the fork:

   ```
   git add submissions/g07
   git commit -m "g07: single on n05"
   git push
   ```

3. Open a pull request from the fork to this repository's `main` branch.
   Its check passes when the pull request only adds hand-in folders and each folder is complete.
4. The lecturer merges the pull requests during the lecture.
   Each merge rescores every hand-in into `results/runs.csv` and `results/plot.png` on `main`.

One pull request may carry several runs; push more commits to it until it is merged, then open a new one.

## Rules

- The agent answers from the IFC files with its Python tool only, without web access.
- A pull request changes nothing outside `submissions/<group>/`; the lecturer closes any other.
