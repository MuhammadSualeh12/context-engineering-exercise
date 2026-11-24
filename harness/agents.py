"""The three strategies as deepagents: one agent, one agent that compacts its context, an orchestrator with subagents."""

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, create_deep_agent, register_harness_profile
from langchain.agents.middleware import AgentMiddleware, ModelRetryMiddleware
from langchain_core.messages import ToolMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.errors import GraphBubbleUp

from harness.compaction import ContextBudget
from harness.context_cap import ContextCap
from harness.defaults import THRESHOLD
from harness.python_tool import make_python_tool

STRATEGIES = ["single", "compaction", "orchestrator"]
WORKER = "ifc-worker"

# deepagents adds file tools, a general-purpose subagent and its own summarisation by default; the exercise
# compares strategies, so the agent gets only what the strategy gives it. The profile applies to every
# ChatOpenRouter model.
register_harness_profile("openrouter", HarnessProfile(
    excluded_tools=frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep", "delete", "execute"}),
    excluded_middleware=frozenset({"SummarizationMiddleware"}),
    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)))

PYTHON_PROMPT = """You answer questions about IFC building models by running Python with IfcOpenShell in the `python` tool.
Variables persist between calls, `ifcopenshell` is imported and `models` maps each model name to its file path.
Open each model once and keep it in a variable. Print only what you need: long output is cut.
Answer from the files only; there is no internet access."""

COMPACTION_PROMPT = PYTHON_PROMPT + """
Your context is limited. When you get a CONTEXT WARNING, free context with drop_old_tool_results or
summarise_history before you continue, and keep the answers found so far in your messages."""

ORCHESTRATOR_PROMPT = f"""You coordinate {WORKER} subagents that answer questions about IFC building models.
You cannot open files yourself: only an {WORKER} can, through its Python tool.
Group the questions by IFC model and start one {WORKER} per model with the `task` tool, all in one message so they
run in parallel. Give each worker the model name and its questions verbatim, with their ids and answer formats.
Each worker replies with a short finding, one line per question id; check that every question has an answer."""

WORKER_PROMPT = PYTHON_PROMPT + """
You work on one model. When done, reply with a short finding and nothing else: one line per question,
`<id>: <answer>` in the answer format the question asks for."""


def chat_model(model: str) -> ChatOpenRouter:
    # max_retries covers 5xx and connection errors only (the OpenRouter SDK retries no 4xx); a 429 of the shared
    # key is retried by rate_limit_retry() in each agent. timeout is in milliseconds.
    return ChatOpenRouter(model=model, max_retries=3, timeout=180_000)


RETRY_DELAY_S = 2.0  # first wait after a 429, doubled each retry: 2, 4, 8, 16, 32 s


def is_rate_limit(exc: BaseException) -> bool:
    """A 429: the SDK's TooManyRequestsResponseError, or an error ChatOpenRouter found in the response body."""
    text = str(exc).lower()
    return getattr(exc, "status_code", None) == 429 or "(code: 429)" in text or "rate limit" in text


def rate_limit_retry() -> ModelRetryMiddleware:
    # on_failure="error": after the last retry the error ends the run (or the worker) and is counted, rather
    # than coming back as a reply the agent would take for its answer.
    return ModelRetryMiddleware(max_retries=5, retry_on=is_rate_limit, on_failure="error",
                                initial_delay=RETRY_DELAY_S, backoff_factor=2.0, max_delay=60.0)


class WorkerErrors(AgentMiddleware):
    """A task call that raises comes back to the orchestrator as an error result instead of ending the run.

    Without it langgraph's tool error handler re-raised the exception of one failed ifc-worker and the run lost
    the answers of every other worker."""

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "task":
            return handler(request)
        try:
            return handler(request)
        except GraphBubbleUp:  # interrupts and other graph control flow are not failures
            raise
        except Exception as exc:
            return ToolMessage(f"worker failed: {type(exc).__name__}: {exc}"[:1000]
                               + "\nStart the worker again, or answer null for its questions.",
                               tool_call_id=request.tool_call["id"], name="task", status="error")


def build_agent(strategy: str, llm, log_event=None, threshold: int = THRESHOLD, context_cap: int | None = None):
    """The agent graph of one strategy; its name tags every model call in the usage log.

    context_cap gives each agent, main agent and every ifc-worker alike, a context window of that many tokens
    (harness/context_cap.py); None leaves the model's own window."""
    log_event = log_event or (lambda kind, **data: None)

    def model_middleware(agent: str) -> list:
        # outermost first: the cap sees the request as sent, the retry wraps the model call alone
        cap = [ContextCap(context_cap, log_event, agent)] if context_cap else []
        return [*cap, rate_limit_retry()]

    if strategy == "single":
        return create_deep_agent(llm, tools=[make_python_tool()], system_prompt=PYTHON_PROMPT,
                                 middleware=model_middleware("single"), name="single")
    if strategy == "compaction":
        budget = ContextBudget(threshold, log_event)
        return create_deep_agent(llm, tools=[make_python_tool()], system_prompt=COMPACTION_PROMPT,
                                 middleware=[budget, *model_middleware("single")], name="single")
    if strategy == "orchestrator":
        # A subagent spec does not inherit the main agent's middleware, so the worker gets its own retry and cap.
        worker = {"name": WORKER, "system_prompt": WORKER_PROMPT, "tools": [make_python_tool()],
                  "middleware": model_middleware(WORKER),
                  "description": "Answers questions about one IFC model with Python and IfcOpenShell; "
                                 "give it the model name and the questions with their ids and answer formats."}
        return create_deep_agent(llm, tools=[], system_prompt=ORCHESTRATOR_PROMPT, subagents=[worker],
                                 middleware=[WorkerErrors(), *model_middleware("orchestrator")],
                                 name="orchestrator")
    raise ValueError(f"unknown strategy {strategy!r}, expected one of {', '.join(STRATEGIES)}")
