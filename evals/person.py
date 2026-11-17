"""The person: who answers Humphrey's question cards during a trial.

In production the browser answers through WebUiBridge. In a trial, a ScriptedPerson does, from
the task's `person:` block, so a question gets the same answer every time:

    person: {pick: "O'Neill"}                    # choose the option whose label matches
    person: {pick: "total", say: "The total"}    # ...or, if none matches, type this instead
    person: {say: "It's about evals"}            # always type an answer ("Something else")
    person: {skip: true}                         # skip every question

The person is the user's side of the conversation, like the task's `input`, so it belongs to
the task, and the runner makes a fresh one per trial, as agentic-evals makes a fresh Session.
A task with no `person:` skips any question it's asked: a person who didn't expect a question
has nothing to say to it.

It records every card it was shown, for the transcript.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from app.ui_bridge import Answer


class Person(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pick: str | None = None  # a regular expression, matched against option labels
    say: str | None = None  # typed as "Something else" when nothing matches `pick`
    skip: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> Person:
        if self.skip and (self.pick or self.say):
            raise ValueError("a person who skips can't also pick or say something")
        if self.pick:
            try:
                re.compile(self.pick)
            except re.error as e:
                raise ValueError(f"invalid pick pattern {self.pick!r}: {e}") from None
        return self


SKIPS = Person(skip=True)


class ScriptedPerson:
    """A UiBridge that answers from a script and records what it was asked."""

    def __init__(self, script: Person | None = None) -> None:
        self.script = script or SKIPS
        self.cards: list[dict[str, Any]] = []  # each: the questions shown, and the answer given

    @property
    def has_ui(self) -> bool:
        return True

    async def ask_questions(self, call_id: str, questions: list[dict[str, Any]]) -> Answer | None:
        answer = self.answer(questions)
        self.cards.append({"questions": questions, "answer": answer})
        return answer

    def answer(self, questions: list[dict[str, Any]]) -> Answer:
        if self.script.skip:
            return {"skipped": True}
        answers = {}
        for q in questions:
            labels = [o["label"] for o in q["options"]]
            picked = [
                lbl for lbl in labels if self.script.pick and re.search(self.script.pick, lbl, re.I)
            ]
            if picked:
                answers[q["header"]] = {"choices": picked[:1]}
            elif self.script.say:
                answers[q["header"]] = {"text": self.script.say}
            else:
                answers[q["header"]] = {"skipped": True}
        return {"answers": answers}
