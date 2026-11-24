"""Exercise defaults, set from the instructor's pilot runs on 23 and 24 November 2026.

MODEL made no tool-call errors and was the cheapest of the three models tried.
THRESHOLD makes compaction warn from n10 upwards; HARD_FACTOR puts the automatic fallback (45000 tokens) under
CONTEXT_CAP, the context window every agent gets unless `--no-context-cap` is passed: at 48000 the single agent
passes n10 and overflows on the larger sets, where compaction and the orchestrator's workers stay far below.
"""

MODEL = "deepseek/deepseek-v4.1-flash-1017"
THRESHOLD = 30_000
HARD_FACTOR = 1.5
CONTEXT_CAP = 48_000
