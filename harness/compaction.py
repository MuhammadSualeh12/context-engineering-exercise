"""Strategy compaction: the agent is warned when its context grows too large and manages it with two tools."""

from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langgraph.types import Command

from harness.defaults import HARD_FACTOR

DROPPED = "[tool result dropped to save context]"


def _reasoning_tokens(message) -> int:
    kwargs = getattr(message, "additional_kwargs", None) or {}
    return len(str(kwargs.get("reasoning_content") or "")) // 4


def context_tokens(messages) -> int:
    """The context the provider sees: its own count (input plus output tokens) up to the last reply that
    reports usage, then about 4 characters a token for the messages after it.

    A reasoning model gets its reasoning back in every call, in additional_kwargs rather than in the text, so
    counting the text alone missed most of the context (pilot: 18k counted of 84k sent). Without usage figures
    the estimate covers the whole conversation and adds the reasoning text."""
    for i in range(len(messages) - 1, -1, -1):
        usage = getattr(messages[i], "usage_metadata", None)
        if usage and usage.get("input_tokens"):
            rest = messages[i + 1:]
            return (usage["input_tokens"] + usage.get("output_tokens", 0)
                    + (count_tokens_approximately(rest) if rest else 0))
    return count_tokens_approximately(messages) + sum(_reasoning_tokens(m) for m in messages)


def drop_tool_results(messages, keep_last: int) -> list[ToolMessage]:
    """Replacements for all but the last keep_last python results; a message with the same id overwrites the old one."""
    results = [m for m in messages if isinstance(m, ToolMessage) and m.name == "python" and m.content != DROPPED]
    old = results[:-keep_last] if keep_last > 0 else results
    return [ToolMessage(DROPPED, tool_call_id=m.tool_call_id, name=m.name, id=m.id) for m in old]


class ContextBudget(AgentMiddleware):
    def __init__(self, threshold: int, log_event: Callable[..., None], hard_factor: float = HARD_FACTOR):
        super().__init__()
        self.threshold, self.hard_limit, self.log_event = threshold, int(threshold * hard_factor), log_event

        @tool
        def drop_old_tool_results(runtime: ToolRuntime, keep_last: int = 2) -> Command:
            """Free context: replace the output of all but the last `keep_last` python calls with a short stub.
            Note the facts you still need in your reply first, since dropped outputs are gone."""
            messages = runtime.state["messages"]
            before = context_tokens(messages)
            dropped = drop_tool_results(messages, keep_last)
            self.log_event("drop_old_tool_results", tokens_before=before, dropped=len(dropped), keep_last=keep_last)
            note = ToolMessage(f"Dropped {len(dropped)} old tool results.", tool_call_id=runtime.tool_call_id,
                               name="drop_old_tool_results")
            return Command(update={"messages": [*dropped, note]})

        @tool
        def summarise_history(summary: str, runtime: ToolRuntime) -> Command:
            """Free context: replace your whole history since the task with `summary`.
            The summary must keep every answer and fact found so far and what is left to do."""
            messages = runtime.state["messages"]
            first_task = next(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
            # Everything between the task and the AI message that called this tool goes.
            this_call = max(i for i, m in enumerate(messages) if isinstance(m, AIMessage))
            removed = messages[first_task + 1:this_call]
            self.log_event("summarise_history", tokens_before=context_tokens(messages), removed=len(removed),
                           summary=summary)
            note = ToolMessage(f"History replaced by your summary:\n{summary}", tool_call_id=runtime.tool_call_id,
                               name="summarise_history")
            return Command(update={"messages": [*(RemoveMessage(id=m.id) for m in removed), note]})

        self.tools = [drop_old_tool_results, summarise_history]

    def before_model(self, state, runtime):
        # The hard fallback, for an agent that ignored the warning: drop old tool results without asking.
        tokens = context_tokens(state["messages"])
        if tokens > self.hard_limit:
            dropped = drop_tool_results(state["messages"], keep_last=2)
            if dropped:
                self.log_event("auto_compaction", tokens_before=tokens, dropped=len(dropped), hard_limit=self.hard_limit)
                return {"messages": dropped}
        return None

    def wrap_model_call(self, request, handler):
        # The warning goes into this call only, not into the history, so it does not pile up; right after a
        # compaction the agent gets one call without it, so a task that alone is over budget cannot loop.
        tokens = context_tokens(request.messages)
        just_compacted = getattr(request.messages[-1], "name", None) in {t.name for t in self.tools}
        if tokens > self.threshold and not just_compacted:
            self.log_event("warning", tokens=tokens, threshold=self.threshold)
            warning = HumanMessage(
                f"CONTEXT WARNING: your context is about {tokens} tokens, over the budget of {self.threshold}. "
                "Free context before you continue: call drop_old_tool_results to blank tool outputs you no longer "
                "need, or summarise_history with every answer and fact found so far. "
                f"At {self.hard_limit} tokens old tool results are dropped automatically.")
            request = request.override(messages=[*request.messages, warning])
        return handler(request)
