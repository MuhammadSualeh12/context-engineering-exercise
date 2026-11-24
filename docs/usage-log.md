# Usage log format

The harness writes one JSON object per line for every model call of a run, to `runs/<run_id>/<run_id>.usage.jsonl`.
`score.py --usage` reads it and adds the run's call count, token totals, cost and wall time to the run row.

## Fields

- `run_id` (string): the run's id, the same on every line of the run, for example `g07-orchestrator-n20-1`.
- `agent` (string): `single` for the agent of a single-agent run, `compaction` for the agent of a smart-compaction run; in an orchestrator run, `orchestrator` for the orchestrator and `subagent` for every ifc-worker.
  Logs written before 2026-11-24 name the agent of a smart-compaction run `single`; `score.py` still accepts them.
- `call_index` (integer): the call's position in the run, counted from 1 across all agents.
- `model` (string): the OpenRouter model id the call went to, for example `deepseek/deepseek-v4.1-flash-1017`.
- `input_tokens` (integer): prompt tokens of the call, cached tokens included.
- `cached_tokens` (integer): the part of `input_tokens` read from the provider's prompt cache, 0 when none.
- `output_tokens` (integer): completion tokens, reasoning tokens included.
- `cost_usd` (number): the call's cost in US dollars as OpenRouter reports it.
- `latency_s` (number): seconds from sending the request to receiving the full response.
- `timestamp` (string): ISO 8601 time with time zone at which the request was sent, for example `2026-11-26T10:15:03.120Z`.

## Example line

```json
{"run_id": "g07-orchestrator-n20-1", "agent": "subagent", "call_index": 4, "model": "deepseek/deepseek-v4.1-flash-1017", "input_tokens": 5120, "cached_tokens": 4096, "output_tokens": 312, "cost_usd": 0.00184, "latency_s": 3.42, "timestamp": "2026-11-26T10:15:03.120Z"}
```

## What score.py derives

- `n_calls` is the number of valid lines, and the token and cost columns are their sums.
- `wall_seconds` runs from the earliest `timestamp` to the latest `timestamp` plus that call's `latency_s`.
- `model` in the run row lists the distinct models of the log joined by `+`, unless `--model` names one.
- A line that is not valid JSON, lacks a field, has a field of the wrong type or an unknown `agent` is reported as a warning and left out of the totals.
- A log with several `run_id` values needs `--run-id` to pick one.

## Run row

`score.py --csv runs.csv` appends one row per run with the columns `run_id`, `group`, `model`, `strategy`, `task_set`, `n_questions`, `n_correct`, `accuracy`, `cat1_correct` to `cat4_correct`, `n_calls`, `input_tokens`, `cached_tokens`, `output_tokens`, `cost_usd`, `wall_seconds`, `answers_file` and `usage_file`.
`strategy` is one of `single`, `compaction` and `orchestrator`.
The file columns hold basenames only.
`--questions-csv` appends one row per scored question with `run_id`, `group`, `strategy`, `task_set`, `id`, `category`, `type` and `status` (`correct`, `wrong` or `missing`), without the answers.
