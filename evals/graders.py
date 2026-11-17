"""Graders: each one scores one aspect of a finished trial.

Four are built in:

    state       a value read from the environment's final state, compared with a target
    reply       a regular expression that must, or must not, match the final reply
    judge       an LLM grades the reply against criteria written in plain English
    tool_calls  tools that must, or must not, have been called successfully

Grade the outcome: what changed in the environment and what the reply says. Not the route the
agent took to get there.

`state` and `reply` are code: fast, free, and they give the same answer every time. Use them for
anything with a definite right answer: a row in a database, a number or a name in the reply. Use
the judge only for meaning that code can't check, like whether a question names the right choice, or
whether a draft is about what the person said. `tool_calls` is for the rare case where a
call is itself the harm; see its docstring.

To add your own type, subclass Grader and register it (see `register`). Each grader's YAML is
validated when the tasks file loads, so a typo fails at once rather than quietly skipping a check.
"""

from __future__ import annotations

import asyncio
import json
import operator
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .environment import Environment
from .transcript import ToolCall

if TYPE_CHECKING:
    from .tasks import Task


@dataclass(frozen=True)
class Grade:
    name: str
    passed: bool
    detail: str = ""  # what the grader found: a value, a missing call, the judge's reasoning
    unknown: bool = False  # the judge couldn't tell; counted as a fail and flagged for a person


@dataclass(frozen=True)
class Trial:
    """What a grader sees: the finished trial and the environment it ran in."""

    task: Task
    reply: str
    tool_log: list[ToolCall]
    env: Environment
    judge_client: Any  # an openai.AsyncOpenAI, or None offline
    judge_model: str = ""


class Grader(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: str
    # True for graders that call a model. The runner treats their API errors as infrastructure
    #   errors, and offline checks (reference solutions) skip them.
    uses_model: ClassVar[bool] = False
    # True for graders that read the final reply. Offline checks run them against the task's
    #   reference_reply, or skip them if it has none.
    reads_reply: ClassVar[bool] = False

    async def grade(self, trial: Trial) -> list[Grade]:
        raise NotImplementedError


GRADER_TYPES: dict[str, type[Grader]] = {}


def register(cls: type[Grader]) -> type[Grader]:
    """Class decorator that makes a grader type usable in tasks files.

        @register
        class ReplyWordLimit(Grader):
            type: Literal["reply_word_limit"] = "reply_word_limit"
            name: str
            max_words: int
            reads_reply: ClassVar[bool] = True

            async def grade(self, trial):
                words = len(trial.reply.split())
                return [Grade(self.name, words <= self.max_words, f"{words} words")]

    Define custom graders in (or import them from) your environment's module: the tasks file
    imports that module before it reads any grader.
    """
    type_name = cls.model_fields["type"].default
    if not isinstance(type_name, str) or type_name in GRADER_TYPES:
        raise ValueError(f"{cls.__name__}: needs a unique default for `type`, got {type_name!r}.")
    GRADER_TYPES[type_name] = cls
    return cls


def parse_grader(spec: Any) -> Grader:
    if isinstance(spec, Grader):
        return spec
    kind = spec.get("type") if isinstance(spec, dict) else None
    if kind not in GRADER_TYPES:
        raise ValueError(
            f"Unknown grader type {kind!r}. Known types: {', '.join(sorted(GRADER_TYPES))}."
        )
    return GRADER_TYPES[kind].model_validate(spec)


# --------------------------------------------------------------------------- state

OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


@register
class State(Grader):
    """E2 [DE]: check the outcome, not the reply. The agent can claim anything in its reply,
    but it can't fake what's in the database."""

    type: Literal["state"] = "state"
    name: str
    query: str
    expect: str  # an operator and a number, like "== 1" or ">= 24.99"

    @field_validator("expect")
    @classmethod
    def _check_expect(cls, expect: str) -> str:
        op, _, target = expect.partition(" ")
        if op not in OPS:
            raise ValueError(f"expect must start with one of {' '.join(OPS)}, got {expect!r}")
        float(target)  # raises if the target isn't a number
        return expect

    async def grade(self, trial: Trial) -> list[Grade]:
        value = trial.env.query(self.query)
        op, _, target = self.expect.partition(" ")
        passed = value is not None and OPS[op](round(float(value), 2), float(target))
        return [Grade(self.name, passed, f"got {value}")]


# --------------------------------------------------------------------------- tool_calls


@register
class ToolCalls(Grader):
    """E3 [DE]: grade the outcome, not the route. Use this only where a call is itself the harm,
    whatever happens next: reading a customer's records after their identity check failed is a
    privacy breach even if nothing is changed or said. Don't use it to check that the agent took
    the steps you expected, or to stand in for an outcome you could read from the environment
    ("nothing was sent" is a query on the outbox, not a list of tools that mustn't be called).
    Agents often find valid routes the task's author didn't think of, and should pass when they
    do."""

    type: Literal["tool_calls"] = "tool_calls"
    name: str
    must_succeed: list[str] = []
    must_not_succeed: list[str] = []

    @model_validator(mode="after")
    def _not_empty(self) -> ToolCalls:
        if not (self.must_succeed or self.must_not_succeed):
            raise ValueError("tool_calls needs must_succeed or must_not_succeed")
        return self

    async def grade(self, trial: Trial) -> list[Grade]:
        succeeded = {c.name for c in trial.tool_log if not c.is_error}
        called = sorted(succeeded & set(self.must_not_succeed))
        missing = sorted(set(self.must_succeed) - succeeded)
        detail = "; ".join(
            [f"should not have called {', '.join(called)}"] * bool(called)
            + [f"never called {', '.join(missing)}"] * bool(missing)
        )
        return [Grade(self.name, not (called or missing), detail)]


# --------------------------------------------------------------------------- reply


@register
class Reply(Grader):
    """A fact with a fixed form in the reply: a number, a time, a name. Case-insensitive.

    Write the pattern to accept the ways people really write the fact ("10am", "10:00",
    "10 a.m.") and nothing else. A task's reference_reply lets `check` test it for free.
    """

    type: Literal["reply"] = "reply"
    name: str
    must_match: str | None = None
    must_not_match: str | None = None
    reads_reply: ClassVar[bool] = True

    @model_validator(mode="after")
    def _valid_patterns(self) -> Reply:
        if self.must_match is None and self.must_not_match is None:
            raise ValueError("reply needs must_match or must_not_match")
        for pattern in (self.must_match, self.must_not_match):
            try:
                re.compile(pattern or "")
            except re.error as e:  # a bad pattern fails when the file loads, not mid-run
                raise ValueError(f"invalid pattern {pattern!r}: {e}") from None
        return self

    async def grade(self, trial: Trial) -> list[Grade]:
        problems = []
        if self.must_match and not re.search(self.must_match, trial.reply, re.I):
            problems.append(f"nothing matches {self.must_match!r}")
        if self.must_not_match and (found := re.search(self.must_not_match, trial.reply, re.I)):
            problems.append(f"found {found.group()!r}")
        return [Grade(self.name, not problems, "; ".join(problems))]


# --------------------------------------------------------------------------- judge


class Verdict(BaseModel):
    reasoning: str
    verdict: Literal["PASS", "FAIL", "UNKNOWN"]


JUDGE_PROMPT = (
    "You are grading one conversation turn by an AI agent against a single criterion.\n\n"
    "<request>{request}</request>\n"
    "<actions_taken>{actions}</actions_taken>\n"
    "<reply>{reply}</reply>\n"
    "<criterion>{criterion}</criterion>\n\n"
    "Does the agent's reply, read together with the actions it took, meet the criterion? "
    "Answer UNKNOWN if you can't tell from what's here. Keep your reasoning brief."
)


@register
class Judge(Grader):
    """E4 [DE]: a model grades what code can't: meaning, not facts. If the right answer could
    be written as a pattern or a query, use `reply` or `state` instead; a judge is slower, costs
    money, and can disagree with itself. One isolated call per dimension, rather than one
    call scoring everything, so each verdict is about one thing. UNKNOWN gives the judge a way
    out instead of a guess; it counts as a fail and is flagged for a person. Structured output,
    so there's no free text to parse.

    Write each criterion so two people reading the transcript would grade it the same way.
    "The card asks which of the two people called Sam to add" works. "The question is good"
    doesn't. Before a judge gates anything, check it agrees with your own labels on 20 to 50
    transcripts.

    A3 [HD]: the judge is a separate model call, never the agent grading itself; self-evaluation
    skews positive. Humphrey runs on OpenAI, so the judge does too (`--judge-model`): at least as
    capable as the agent, and ideally a different model, though the same one is allowed. Low effort
    is enough for one narrow criterion.
    """

    type: Literal["judge"] = "judge"
    dimensions: dict[str, str] = Field(min_length=1)
    uses_model: ClassVar[bool] = True
    effort: ClassVar[str] = "low"

    async def grade(self, trial: Trial) -> list[Grade]:
        if trial.judge_client is None:
            raise ValueError("The judge grader needs a client.")
        actions = "\n".join(_action(c) for c in trial.tool_log) or "none"

        async def one(name: str, criterion: str) -> Grade:
            prompt = JUDGE_PROMPT.format(
                request=trial.task.input, actions=actions, reply=trial.reply, criterion=criterion
            )
            response = await trial.judge_client.responses.parse(
                model=trial.judge_model,
                input=prompt,
                text_format=Verdict,
                reasoning={"effort": self.effort},
            )
            v = response.output_parsed or Verdict(
                reasoning="No verdict returned.", verdict="UNKNOWN"
            )
            return Grade(
                name,
                v.verdict == "PASS",
                f"{v.verdict}: {v.reasoning}",
                unknown=v.verdict == "UNKNOWN",
            )

        return list(await asyncio.gather(*(one(n, c) for n, c in self.dimensions.items())))


def _action(call: ToolCall) -> str:
    """One tool call, as the judge reads it: the name, the input, and whether it failed."""
    failed = " -> error" if call.is_error else ""
    return f"{call.name}({json.dumps(call.input, ensure_ascii=False)}){failed}"
