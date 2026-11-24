"""A context window smaller than the model's: the agent whose next call would exceed it stops, as on a real overflow.

The exercise models have a window of about 1M tokens, which no task set nears (pilot: at most 172000 tokens), so
`--context-cap` gives every agent the window the lecture talks about. The cap applies per agent: the main agent
and each ifc-worker count their own context. A main agent that overflows ends the run with status
context_overflow; a worker that overflows fails its task call, and the orchestrator can start it again or answer
null for its questions.
"""

from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware

from harness.compaction import context_tokens


class ContextOverflow(Exception):
    """Raised instead of the model call; harness.run maps "context window" in the message to context_overflow."""


class ContextCap(AgentMiddleware):
    def __init__(self, cap: int, log_event: Callable[..., None], agent: str):
        super().__init__()
        self.cap, self.log_event, self.agent = cap, log_event, agent

    def wrap_model_call(self, request, handler):
        # The system message sits before every reply that reports usage, so it only adds to the estimate of a
        # first call, which has no usage figures yet.
        system = [request.system_message] if request.system_message is not None else []
        tokens = context_tokens([*system, *request.messages])
        if tokens > self.cap:
            self.log_event("context_overflow", agent=self.agent, scope=_scope(), tokens=tokens, cap=self.cap)
            raise ContextOverflow(f"context window exceeded: the next call of {self.agent} would send about "
                                  f"{tokens} tokens, over the cap of {self.cap} (--context-cap)")
        return handler(request)


def _scope() -> str:
    """"" for the main agent, "tools:<task id>" inside a subagent, as in the trace."""
    try:
        from langgraph.config import get_config

        return get_config().get("metadata", {}).get("langgraph_checkpoint_ns", "").rpartition("|")[0]
    except RuntimeError:  # outside a graph run
        return ""
