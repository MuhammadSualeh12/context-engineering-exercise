"""A scripted chat model for offline tests: it calls tools by a fixed plan and reports usage like OpenRouter."""

import re
import uuid

from pydantic import Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

SEEN_TOOLS: list[list[str]] = []  # the tool names of every model call
SEEN_MESSAGES: list[list] = []  # the messages of every model call


class RateLimited(Exception):
    """Like the OpenRouter SDK's TooManyRequestsResponseError: an HTTP error with status_code 429."""
    status_code = 429


class ScriptedModel(BaseChatModel):
    python_calls: int = 1  # python calls before the final reply
    parallel: int = 1  # python calls per reply, run by the agent in parallel
    code: str = "print(sorted(models))"
    compact_with: str = "drop_old_tool_results"  # the tool called on a CONTEXT WARNING
    fail_after: int | None = None  # raise once the conversation holds this many replies
    fail_message: str = "provider error"
    invalid_first: bool = False  # the first reply holds one tool call with malformed arguments
    rate_limited: int = 0  # the first this many calls of the run raise a 429
    worker_rate_limited: int = 0  # the first this many calls of subagents raise a 429
    worker_fails: bool = False  # every model call of a subagent (an agent with the python tool) raises
    calls: dict = Field(default_factory=dict)  # shared by the copies bind_tools makes
    bound: tuple[str, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _get_ls_params(self, **kwargs):
        # deepagents picks the harness profile by provider, so pose as OpenRouter like ChatOpenRouter does.
        return {"ls_provider": "openrouter", "ls_model_name": "fake/model", "ls_model_type": "chat"}

    def bind_tools(self, tools, **kwargs):
        names = tuple(t.name if hasattr(t, "name") else t.get("name") or t["function"]["name"] for t in tools)
        return self.model_copy(update={"bound": names})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        SEEN_TOOLS.append(sorted(self.bound))
        SEEN_MESSAGES.append(list(messages))
        task = next(m.content for m in messages if isinstance(m, HumanMessage))
        ids = sorted(set(re.findall(r"\bq\d\d\b", task)))
        made = [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
        self.calls["n"] = self.calls.get("n", 0) + 1
        if self.calls["n"] <= self.rate_limited:
            raise RateLimited("Rate limit exceeded: free-models-per-min")
        if "python" in self.bound:
            self.calls["worker"] = self.calls.get("worker", 0) + 1
            if self.calls["worker"] <= self.worker_rate_limited:
                raise RateLimited("Rate limit exceeded: free-models-per-min")
        if self.worker_fails and "python" in self.bound:
            raise RuntimeError("worker crashed")
        if self.fail_after is not None and sum(isinstance(m, AIMessage) for m in messages) >= self.fail_after:
            raise RuntimeError(self.fail_message)
        last = messages[-1]
        calls = []
        if "task" in self.bound and "task" not in made:
            calls = [("task", {"description": f"Model {name}: questions {', '.join(ids)}", "subagent_type": "ifc-worker"})
                     for name in ("fzk_house", "ac20")]
        elif isinstance(last, HumanMessage) and "CONTEXT WARNING" in last.content and self.compact_with in self.bound:
            args = {"keep_last": 1} if self.compact_with == "drop_old_tool_results" else {"summary": "SUMMARY: nothing yet"}
            calls = [(self.compact_with, args)]
        elif "python" in self.bound and made.count("python") < self.python_calls and "summarise_history" not in made:
            calls = [("python", {"code": self.code})] * self.parallel
        if self.invalid_first and not any(isinstance(m, AIMessage) for m in messages):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                "", invalid_tool_calls=[{"name": "python", "args": "{\"code\": \"print(", "id": "bad1",
                                         "error": "malformed JSON", "type": "invalid_tool_call"}],
                usage_metadata={"input_tokens": 50, "output_tokens": 7, "total_tokens": 57},
                response_metadata={"model_name": "fake/model", "cost": 0.0001}))])
        content = "" if calls else "{" + ", ".join(f'"{i}": 0' for i in ids) + "}"
        n_in = sum(len(str(m.content)) for m in messages) // 4 + 50
        message = AIMessage(
            content, tool_calls=[{"name": n, "args": a, "id": uuid.uuid4().hex[:8]} for n, a in calls],
            usage_metadata={"input_tokens": n_in, "output_tokens": 7, "total_tokens": n_in + 7,
                            "input_token_details": {"cache_read": n_in // 2}},
            response_metadata={"model_name": "fake/model", "cost": 0.0001})
        return ChatResult(generations=[ChatGeneration(message=message)])
