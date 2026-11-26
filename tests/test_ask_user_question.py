"""ask_user_question: the contract, the answers the model sees, the off
switch, and the round trip through /api/agent and the answer route."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import pytest
from httpx import AsyncClient

from app.tools.ask_user_question import (
    QuestionError,
    create_ask_user_question_tool,
    format_outcome,
    validate_questions,
)
from app.ui_bridge import NullUiBridge


def _question(**overrides: Any) -> dict[str, Any]:
    question = {
        "question": "Which months should the dashboard cover?",
        "header": "Period",
        "options": [
            {"label": "This year", "description": "Jan to Oct"},
            {"label": "Last 12 months", "description": "The CSV's range", "recommended": True},
        ],
    }
    question.update(overrides)
    return question


class _AnsweringBridge:
    """Answers every question card with a fixed answer."""

    def __init__(self, answer: dict[str, Any] | None) -> None:
        self.answer = answer
        self.asked: list[list[dict[str, Any]]] = []

    @property
    def has_ui(self) -> bool:
        return True

    async def ask_questions(self, call_id: str, questions: list[dict[str, Any]]):
        self.asked.append(questions)
        return self.answer


# --- the contract ------------------------------------------------------------


def test_recommended_option_moves_first() -> None:
    [q] = validate_questions([_question()])
    assert [o["label"] for o in q["options"]] == ["Last 12 months", "This year"]
    assert q["options"][0]["recommended"] is True
    assert q["multiSelect"] is False


@pytest.mark.parametrize(
    ("questions", "message"),
    [
        ([], "1 to 4 questions"),
        ([_question(header=f"H{i}") for i in range(5)], "1 to 4 questions"),
        ([_question(header="Far too long a header")], "at most 12"),
        ([_question(), _question()], "unique"),
        ([_question(options=[{"label": "Only one", "description": ""}])], "2 to 4 options"),
        (
            [_question(options=[{"label": "Same", "description": ""}, {"label": "same", "description": ""}])],
            "twice",
        ),
        (
            [_question(options=[{"label": "A", "description": ""}, {"label": "Other", "description": ""}])],
            "Something else",
        ),
        (
            [
                _question(
                    options=[
                        {"label": "A", "description": "", "recommended": True},
                        {"label": "B", "description": "", "recommended": True},
                    ]
                )
            ],
            "more than one",
        ),
    ],
)
def test_bad_calls_are_refused_with_a_reason(questions: list, message: str) -> None:
    with pytest.raises(QuestionError, match=message):
        validate_questions(questions)


# --- what the model gets back ------------------------------------------------


def _two_questions() -> list[dict[str, Any]]:
    return validate_questions(
        [
            _question(),
            _question(
                question="Which charts?",
                header="Charts",
                multiSelect=True,
                options=[
                    {"label": "Revenue", "description": ""},
                    {"label": "Products", "description": ""},
                    {"label": "Regions", "description": ""},
                ],
            ),
        ]
    )


def test_answers_come_back_one_line_per_question() -> None:
    result = format_outcome(
        _two_questions(),
        {
            "answers": {
                "Period": {"choices": ["Last 12 months"]},
                "Charts": {"choices": ["Regions", "Revenue"], "text": "and a table"},
            }
        },
    )
    assert result.text.splitlines() == [
        "The user answered:",
        '- Period ("Which months should the dashboard cover?"): Last 12 months',
        '- Charts ("Which charts?"): Revenue, Regions; in their own words: "and a table"',
    ]
    assert result.details["status"] == "answered"


def test_unknown_choices_are_dropped_and_single_choice_keeps_one() -> None:
    result = format_outcome(
        _two_questions(),
        {"answers": {"Period": {"choices": ["This year", "Last 12 months", "Invented"]}}},
    )
    answers = result.details["answers"]
    assert answers["Period"]["choices"] == ["Last 12 months"]  # option order, first only
    assert answers["Charts"]["skipped"] is True
    assert "skipped. Don't ask this again" in result.text


def test_skipping_the_card_skips_every_question() -> None:
    result = format_outcome(_two_questions(), {"skipped": True})
    assert result.details["status"] == "skipped"
    assert result.text.startswith("The user skipped the questions:")


def test_a_reply_in_the_composer_is_passed_on() -> None:
    result = format_outcome(_two_questions(), {"reply": "  just use everything  "})
    assert result.text == 'The user replied instead of answering: "just use everything"'
    assert result.details["status"] == "replied"


def test_no_answer_means_the_run_stopped() -> None:
    result = format_outcome(_two_questions(), None)
    assert result.details["status"] == "stopped"
    assert "didn't answer" in result.text


# --- the tool ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_a_ui_the_model_is_told_to_decide() -> None:
    tool = create_ask_user_question_tool(NullUiBridge())
    result = await tool.execute("c1", {"questions": [_question()]})
    assert result.is_error is True
    assert "sensible default" in result.text


@pytest.mark.asyncio
async def test_a_bad_call_never_reaches_the_user() -> None:
    bridge = _AnsweringBridge({"skipped": True})
    tool = create_ask_user_question_tool(bridge)
    result = await tool.execute("c1", {"questions": [_question(header="Much too long header")]})
    assert result.is_error is True
    assert bridge.asked == []


@pytest.mark.asyncio
async def test_a_skipped_question_is_not_asked_again_in_the_run() -> None:
    bridge = _AnsweringBridge({"skipped": True})
    tool = create_ask_user_question_tool(bridge)
    first = await tool.execute("c1", {"questions": [_question()]})
    assert first.details["status"] == "skipped"
    again = await tool.execute("c2", {"questions": [_question()]})
    assert again.is_error is True
    assert "don't ask it again" in again.text
    assert len(bridge.asked) == 1


# --- the off switch ----------------------------------------------------------


def test_the_setting_removes_the_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.settings import reset_settings_for_tests
    from app.tools import tools_for_run

    assert "ask_user_question" in [t.name for t in tools_for_run(NullUiBridge())]
    monkeypatch.setenv("HUMPHREY_ASK_USER_QUESTION", "0")
    reset_settings_for_tests()
    assert "ask_user_question" not in [t.name for t in tools_for_run(NullUiBridge())]


# --- the round trip ----------------------------------------------------------


def _events(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(block.strip()[6:])
        for block in body.split("\n\n")
        if block.strip().startswith("data: ")
    ]


async def _wait_for_bridge(session_id: str):
    from app import runs

    for _ in range(500):
        run = runs.get(session_id)
        if run is not None and run.bridge.waiting:
            return run.bridge
        await asyncio.sleep(0.01)
    raise AssertionError("the run never asked")


@pytest.mark.asyncio
async def test_a_question_round_trip(client: AsyncClient) -> None:
    session_id = uuid.uuid4().hex

    async def answer() -> int:
        bridge = await _wait_for_bridge(session_id)
        [request_id] = list(bridge._pending)
        response = await client.post(
            f"/api/sessions/{session_id}/ui/{request_id}",
            json={
                "answers": {
                    "Period": {"choices": ["Last 12 months"]},
                    "Charts": {"choices": ["Top products"], "text": "and a table"},
                }
            },
        )
        return response.status_code

    run = client.post(
        "/api/agent", json={"message": {"content": "ask me something"}, "session_id": session_id}
    )
    response, status = await asyncio.gather(run, answer())
    assert status == 204

    events = _events(response.text)
    [request] = [e for e in events if e["type"] == "ui_request"]
    assert request["kind"] == "question"
    assert [q["header"] for q in request["questions"]] == ["Period", "Charts"]
    assert request["questions"][0]["options"][0]["label"] == "Last 12 months"  # recommended first
    [end] = [e for e in events if e["type"] == "tool_end"]
    assert end["call_id"] == request["call_id"]
    assert end["question"]["status"] == "answered"
    assert "Top products" in end["result"]
    assert events[-1]["type"] == "agent_end"

    # Reopening the chat brings the card back with its answer.
    stored = (await client.get(f"/api/sessions/{session_id}")).json()
    [row] = [m for m in stored["messages"] if m["role"] == "question"]
    assert row["question"]["answers"]["Period"]["choices"] == ["Last 12 months"]


@pytest.mark.asyncio
async def test_a_closed_bridge_leaves_the_question_unanswered(client: AsyncClient) -> None:
    session_id = uuid.uuid4().hex

    async def stop() -> None:
        (await _wait_for_bridge(session_id)).close()

    run = client.post(
        "/api/agent", json={"message": {"content": "ask me"}, "session_id": session_id}
    )
    response, _ = await asyncio.gather(run, stop())
    [end] = [e for e in _events(response.text) if e["type"] == "tool_end"]
    assert end["question"]["status"] == "stopped"


@pytest.mark.asyncio
async def test_answering_nothing_that_waits_is_a_404(client: AsyncClient) -> None:
    response = await client.post(
        f"/api/sessions/{uuid.uuid4().hex}/ui/{uuid.uuid4().hex}", json={"skipped": True}
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_the_answer_route_takes_json_only(client: AsyncClient) -> None:
    response = await client.post(
        f"/api/sessions/{uuid.uuid4().hex}/ui/{uuid.uuid4().hex}",
        content='{"skipped": true}',
        headers={"content-type": "text/plain"},
    )
    assert response.status_code == 415
