"""ask_user_question: the model asks the person and waits for the answer.

Modelled on Claude Code's AskUserQuestion: one to four questions, each
with two to four options (label plus description). The app, not the
model, always adds "Something else" (type your own) and Skip, so the
model can't box the person into its own framing. ``recommended`` is a
flag rather than a "(Recommended)" label suffix, and that option is moved
first. A skipped question must not be asked again; asking it again in the
same run is refused (OpenWorker's anti-loop rule).

Like tau's ask_user_question extension, the tool itself waits on a UI
bridge (app/ui_bridge.py). Each run gets its own instance, bound to that
run's bridge, so the skipped-questions memory is per run too.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.ui_bridge import Answer, UiBridge
from brain.messages import TextContent
from brain.tools import AgentTool, ToolCancellationToken, ToolResult, ToolUpdateCallback

NAME = "ask_user_question"
MAX_QUESTIONS = 4
MIN_OPTIONS = 2
MAX_OPTIONS = 4
MAX_HEADER = 12
MAX_LABEL = 60
# The card always offers these itself.
RESERVED_LABELS = frozenset({"other", "something else", "skip", "none of these"})

SKIP_NOTE = (
    "Don't ask this again. If it's safe, pick a sensible default and say which you chose; "
    "if the choice is consequential or hard to undo, say what you need instead."
)


class QuestionError(ValueError):
    """The call breaks the contract; the message tells the model how to fix it."""


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuestionError(f"{what} must be a non-empty string")
    return value.strip()


def validate_questions(raw: Any) -> list[dict[str, Any]]:
    """Check the call and return the questions in canonical form, with any
    recommended option moved first."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_QUESTIONS:
        raise QuestionError(f"questions must be a list of 1 to {MAX_QUESTIONS} questions")
    questions: list[dict[str, Any]] = []
    headers: set[str] = set()
    for qi, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise QuestionError(f"question {qi} must be an object")
        question = _text(item.get("question"), f"question {qi}'s question")
        header = _text(item.get("header"), f"question {qi}'s header")
        if len(header) > MAX_HEADER:
            raise QuestionError(f"question {qi}'s header must be at most {MAX_HEADER} characters")
        if header.lower() in headers:
            raise QuestionError(f"headers must be unique; '{header}' is used twice")
        headers.add(header.lower())
        raw_options = item.get("options")
        if not isinstance(raw_options, list) or not MIN_OPTIONS <= len(raw_options) <= MAX_OPTIONS:
            raise QuestionError(
                f"question {qi} needs {MIN_OPTIONS} to {MAX_OPTIONS} options "
                "(the card adds 'Something else' and Skip itself)"
            )
        options: list[dict[str, Any]] = []
        labels: set[str] = set()
        for oi, opt in enumerate(raw_options, start=1):
            if not isinstance(opt, Mapping):
                raise QuestionError(f"question {qi}, option {oi} must be an object")
            label = _text(opt.get("label"), f"question {qi}, option {oi}'s label")
            if len(label) > MAX_LABEL:
                raise QuestionError(f"option labels must be at most {MAX_LABEL} characters")
            key = label.lower().rstrip(".… ")
            if key in RESERVED_LABELS:
                raise QuestionError(
                    f"don't offer '{label}': the card always adds 'Something else' and Skip"
                )
            if key in labels:
                raise QuestionError(f"question {qi} offers '{label}' twice")
            labels.add(key)
            description = opt.get("description", "")
            options.append(
                {
                    "label": label,
                    "description": description.strip() if isinstance(description, str) else "",
                    "recommended": opt.get("recommended") is True,
                }
            )
        if sum(o["recommended"] for o in options) > 1:
            raise QuestionError(f"question {qi} recommends more than one option")
        options.sort(key=lambda o: not o["recommended"])  # stable: recommended first
        questions.append(
            {
                "question": question,
                "header": header,
                "multiSelect": item.get("multiSelect") is True,
                "options": options,
            }
        )
    return questions


def _clean_answer(question: dict[str, Any], raw: Any) -> dict[str, Any]:
    """One question's answer, keeping only labels the question offered."""
    if not isinstance(raw, Mapping) or raw.get("skipped") is True:
        return {"choices": [], "text": None, "skipped": True}
    offered = [o["label"] for o in question["options"]]
    picked = raw.get("choices") or []
    choices = [label for label in offered if isinstance(picked, list) and label in picked]
    if not question["multiSelect"]:
        choices = choices[:1]
    text = raw.get("text")
    text = text.strip() if isinstance(text, str) and text.strip() else None
    if not choices and text is None:
        return {"choices": [], "text": None, "skipped": True}
    return {"choices": choices, "text": text, "skipped": False}


def _line(question: dict[str, Any], answer: dict[str, Any]) -> str:
    head = f'- {question["header"]} ("{question["question"]}"): '
    if answer["skipped"]:
        return head + "skipped. " + SKIP_NOTE
    parts = [", ".join(answer["choices"])] if answer["choices"] else []
    if answer["text"] is not None:
        parts.append(f'in their own words: "{answer["text"]}"')
    return head + "; ".join(parts)


def format_outcome(questions: list[dict[str, Any]], outcome: Answer | None) -> ToolResult:
    """The model-facing text, with the structure kept in details for the UI."""
    details: dict[str, Any] = {"questions": questions}
    if outcome is None:
        details["status"] = "stopped"
        text = "The user didn't answer: the run stopped before they could."
        return ToolResult(content=[TextContent(text=text)], details=details)
    reply = outcome.get("reply")
    if isinstance(reply, str) and reply.strip():
        details.update(status="replied", reply=reply.strip())
        text = f'The user replied instead of answering: "{reply.strip()}"'
        return ToolResult(content=[TextContent(text=text)], details=details)
    raw_answers = outcome.get("answers") if outcome.get("skipped") is not True else None
    raw_answers = raw_answers if isinstance(raw_answers, Mapping) else {}
    answers = {q["header"]: _clean_answer(q, raw_answers.get(q["header"])) for q in questions}
    all_skipped = all(a["skipped"] for a in answers.values())
    details.update(status="skipped" if all_skipped else "answered", answers=answers)
    lines = [_line(q, answers[q["header"]]) for q in questions]
    lead = "The user skipped the questions:" if all_skipped else "The user answered:"
    return ToolResult(content=[TextContent(text="\n".join([lead, *lines]))], details=details)


PARAMETERS: dict[str, Any] = {
    "questions": {
        "type": "array",
        "minItems": 1,
        "maxItems": MAX_QUESTIONS,
        "description": f"1 to {MAX_QUESTIONS} related questions, asked together in one card.",
        "items": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The full question. Name what it's about; it may be all the user reads.",
                },
                "header": {
                    "type": "string",
                    "description": f"A short label of at most {MAX_HEADER} characters, e.g. \"Period\".",
                },
                "multiSelect": {
                    "type": "boolean",
                    "description": "True if the user may pick more than one option.",
                },
                "options": {
                    "type": "array",
                    "minItems": MIN_OPTIONS,
                    "maxItems": MAX_OPTIONS,
                    "description": (
                        f"{MIN_OPTIONS} to {MAX_OPTIONS} genuinely different choices. Don't add "
                        "'Other' or 'Skip': the card always offers them."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "A few words; the answer value."},
                            "description": {
                                "type": "string",
                                "description": "What choosing it means, or why you recommend it.",
                            },
                            "recommended": {
                                "type": "boolean",
                                "description": "Mark at most one option as your recommendation.",
                            },
                        },
                        "required": ["label", "description"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["question", "header", "options"],
            "additionalProperties": False,
        },
    }
}

GUIDELINES: tuple[str, ...] = (
    "Use ask_user_question when a request has several reasonable readings that lead to "
    "noticeably different work, when a fact only the user knows is missing, or when a "
    "preference with no sensible default shapes the result.",
    "Don't use ask_user_question for anything a tool can check, for choices with a sensible "
    "default (use it and say so), to ask permission for an action, or to ask whether to "
    "continue. Group related questions into one call, and never re-ask a skipped question.",
)


def create_ask_user_question_tool(bridge: UiBridge) -> AgentTool:
    """The tool for one run, bound to that run's bridge."""
    skipped: set[str] = set()

    async def execute(
        tool_call_id: str,
        arguments: Mapping[str, Any],
        signal: ToolCancellationToken | None = None,
        _on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        try:
            questions = validate_questions(arguments.get("questions"))
        except QuestionError as exc:
            return ToolResult.error(str(exc))
        again = [q["question"] for q in questions if q["question"].lower() in skipped]
        if again:
            return ToolResult.error(
                f"The user already skipped this, so don't ask it again: {again[0]}"
            )
        if not bridge.has_ui:
            return ToolResult.error(
                "Asking the user isn't available here. Use a sensible default and say which you chose."
            )
        if signal is not None and signal.is_cancelled():
            return format_outcome(questions, None)
        result = format_outcome(questions, await bridge.ask_questions(tool_call_id, questions))
        for header, answer in (result.details or {}).get("answers", {}).items():
            if answer["skipped"]:
                skipped.update(q["question"].lower() for q in questions if q["header"] == header)
        return result

    return AgentTool(
        name=NAME,
        label="Asking you",
        description=(
            "Ask the user one to four questions, each with two to four options, and wait for "
            "the answer. The card always lets them type their own answer or skip."
        ),
        parameters=PARAMETERS,
        execute_fn=execute,
        required=("questions",),
        prompt_snippet="Ask the user clarifying questions with options, and wait for the answer.",
        prompt_guidelines=GUIDELINES,
    )
