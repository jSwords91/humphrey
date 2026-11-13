"""Compaction records on a Session (app/sessions).

Compaction used to replace the stored transcript with the summary, so a
reopened chat lost its earlier messages, and the summary (stamped later than
the messages kept after it) pulled their artefacts above it in the UI. Now
the transcript is never rewritten: each compaction is a record, and the
model's context is rebuilt from the latest one.
"""

from __future__ import annotations

import json

import pytest

from app.context_window import COMPACTION_SUMMARY_PREFIX, estimate_context_tokens
from app.sessions import Compaction, Session
from brain.messages import AssistantMessage, UserMessage


def _turns(n: int, *, start_ms: int = 1_000_000) -> list:
    messages: list = []
    for i in range(n):
        t = start_ms + i * 1000
        messages.append(UserMessage(content=f"q{i}", timestamp_ms=t))
        messages.append(AssistantMessage(content=f"a{i}", model="mock", timestamp_ms=t + 500))
    return messages


def test_model_context_is_the_transcript_until_something_is_compacted() -> None:
    session = Session(id="s", messages=_turns(3))
    assert session.model_context() == session.messages


def test_a_compaction_shortens_the_model_context_but_not_the_transcript() -> None:
    transcript = _turns(5)
    session = Session(id="s", messages=list(transcript))
    session.record_compaction("first summary", kept_from=6)  # keep q3 onwards

    assert session.messages == transcript
    context = session.model_context()
    assert context[0].text == COMPACTION_SUMMARY_PREFIX + "first summary"
    assert [m.text for m in context[1:]] == ["q3", "a3", "q4", "a4"]


def test_a_second_compaction_maps_back_onto_the_full_transcript() -> None:
    session = Session(id="s", messages=_turns(5))
    session.record_compaction("first", kept_from=6)  # context: [S1, q3, a3, q4, a4]
    session.messages.extend(_turns(2, start_ms=9_000_000))  # q0', a0', q1', a1'
    # Compact the new context, keeping from its index 3 (q4).
    session.record_compaction("second", kept_from=3)

    assert session.compactions[-1].first_kept_index == 8
    context = session.model_context()
    assert context[0].text == COMPACTION_SUMMARY_PREFIX + "second"
    assert [m.text for m in context[1:]] == ["q4", "a4", "q0", "a0", "q1", "a1"]


def test_record_compaction_rejects_a_boundary_outside_the_context() -> None:
    session = Session(id="s", messages=_turns(2))
    with pytest.raises(ValueError):
        session.record_compaction("x", kept_from=5)
    session.record_compaction("x", kept_from=2)
    with pytest.raises(ValueError):
        session.record_compaction("y", kept_from=0)  # the summary itself can't be kept alone


def test_token_usage_from_before_a_compaction_is_ignored() -> None:
    """The summary is stamped later than the kept messages, so their usage
    (which measured the old, longer context) mustn't anchor the estimate."""
    messages = _turns(3)
    messages[-1] = AssistantMessage(
        content="a2", model="mock", timestamp_ms=messages[-1].timestamp_ms, usage={"total_tokens": 99_999}
    )
    session = Session(id="s", messages=messages)
    assert estimate_context_tokens(system="", messages=tuple(session.model_context()), tools=()) >= 99_999

    session.record_compaction("summary", kept_from=2)
    estimate = estimate_context_tokens(system="", messages=tuple(session.model_context()), tools=())
    assert estimate < 1_000


def test_compactions_round_trip_through_storage() -> None:
    session = Session(id="s", messages=_turns(4))
    session.record_compaction("kept safe", kept_from=4)
    restored = Session.from_dict(json.loads(json.dumps(session.to_dict())))
    assert restored.compactions == session.compactions
    assert restored.model_context() == session.model_context()


@pytest.mark.parametrize(
    "record",
    [
        {"summary": "x", "first_kept_index": 99, "created_ms": 1},  # past the transcript
        {"summary": "x", "first_kept_index": -1, "created_ms": 1},
        {"summary": "x", "created_ms": 1},  # missing field
        {"summary": "x", "first_kept_index": "two", "created_ms": 1},
        "not a record",
    ],
)
def test_a_bad_stored_record_is_dropped_not_fatal(record: object) -> None:
    data = Session(id="s", messages=_turns(2)).to_dict()
    data["compactions"] = [record]
    restored = Session.from_dict(data)
    assert restored.compactions == []
    assert len(restored.model_context()) == 4  # the model just sees everything


def test_a_session_from_before_compaction_records_loads_unchanged() -> None:
    data = Session(id="s", messages=_turns(2)).to_dict()
    del data["compactions"]
    data["schema_version"] = 3
    restored = Session.from_dict(data)
    assert restored.compactions == []
    assert [m.text for m in restored.messages] == ["q0", "a0", "q1", "a1"]


def test_ui_shows_every_message_and_the_summary_where_it_happened() -> None:
    session = Session(id="s", messages=_turns(2))  # q0..a1, up to t=1_001_500
    session.compactions.append(Compaction(summary="sum", first_kept_index=2, created_ms=1_001_800))
    session.messages += [
        UserMessage(content="next", timestamp_ms=1_001_800),  # same millisecond
        AssistantMessage(content="reply", model="mock", timestamp_ms=1_002_500),
    ]

    rows = session.history_rows()
    assert [(r.role, r.content) for r in rows] == [
        ("user", "q0"),
        ("assistant", "a0"),
        ("user", "q1"),
        ("assistant", "a1"),
        ("context_summary", "sum"),
        ("user", "next"),
        ("assistant", "reply"),
    ]
    assert [r.ts for r in rows] == sorted(r.ts for r in rows)


def test_a_legacy_inline_summary_is_ordered_before_the_messages_it_precedes() -> None:
    """Before schema 4 the summary was stored as the first message, stamped
    at compaction time, after the messages kept behind it."""
    kept = _turns(2)
    legacy = UserMessage(content=COMPACTION_SUMMARY_PREFIX + "old summary", timestamp_ms=9_999_999)
    session = Session(id="s", messages=[legacy, *kept])

    rows = session.history_rows()
    assert rows[0].role == "context_summary"
    assert [r.ts for r in rows] == sorted(r.ts for r in rows)
    assert rows[0].ts < kept[0].timestamp_ms
