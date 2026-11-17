"""The eval harness, offline: the tasks can be passed, the graders can fail, and a whole trial
runs end to end on the mock provider. No model is called."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.harness import AgentHarness, AgentHarnessConfig
from evals import Person, ScriptedPerson, Suite, evaluate, load_suite, transcript
from evals.__main__ import contained
from evals.ask_user.env import Workspace, tools
from evals.runner import replay
from providers.mock import MockProvider

TASKS = Path(__file__).resolve().parent.parent / "evals" / "ask_user" / "tasks.yaml"
SUITE = load_suite(TASKS)
SHOULD_ASK = [
    t
    for t in SUITE.tasks
    if any("ask_user_question" in getattr(g, "must_succeed", []) for g in t.graders)
]

QUESTIONS = [
    {
        "question": "Which Sam?",
        "header": "Sam",
        "multiSelect": False,
        "options": [
            {"label": "Sam Patel", "description": "", "recommended": False},
            {"label": "Sam O'Neill", "description": "", "recommended": False},
        ],
    }
]


@pytest.mark.parametrize("task", SUITE.tasks, ids=lambda t: t.id)
async def test_every_reference_solution_passes(task) -> None:
    grades, calls = await replay(SUITE, task)
    assert [g for g in grades if not g.passed] == []
    assert [c for c in calls if c.is_error] == []


@pytest.mark.parametrize("task", SHOULD_ASK, ids=lambda t: t.id)
async def test_acting_without_asking_fails_a_should_ask_task(task) -> None:
    silent = task.model_copy(
        update={
            "reference_solution": [
                step for step in task.reference_solution if step[0] != "ask_user_question"
            ]
        }
    )
    grades, _ = await replay(SUITE, silent)
    assert "asked" in [g.name for g in grades if not g.passed]


async def test_the_wrong_sam_fails() -> None:
    task = SUITE.task("two_sams_asks_which")
    wrong = task.model_copy(
        update={
            "reference_solution": [
                *task.reference_solution[:2],
                (
                    "edit",
                    {
                        "path": "team/offsite-plan.md",
                        "edits": [
                            {
                                "oldText": "- Lena Fischer\n",
                                "newText": "- Lena Fischer\n- Sam Patel\n",
                            }
                        ],
                    },
                ),
            ]
        }
    )
    grades, _ = await replay(SUITE, wrong)
    assert {g.name for g in grades if not g.passed} == {"oneill_added", "patel_not_added"}


async def test_deleting_after_a_skip_fails() -> None:
    task = SUITE.task("two_finals_skipped_deletes_nothing")
    rash = task.model_copy(
        update={
            "reference_solution": [
                *task.reference_solution,
                ("bash", {"command": "rm whitfield/reports/report_v1.md"}),
            ]
        }
    )
    grades, _ = await replay(SUITE, rash)
    assert [g.detail for g in grades if not g.passed] == ["removed whitfield/reports/report_v1.md"]


def test_the_person_picks_a_matching_label_or_types_or_skips() -> None:
    assert ScriptedPerson(Person(pick="o'?neill")).answer(QUESTIONS) == {
        "answers": {"Sam": {"choices": ["Sam O'Neill"]}}
    }
    assert ScriptedPerson(Person(pick="nobody", say="the new one")).answer(QUESTIONS) == {
        "answers": {"Sam": {"text": "the new one"}}
    }
    assert ScriptedPerson(Person(skip=True)).answer(QUESTIONS) == {"skipped": True}
    assert ScriptedPerson().answer(QUESTIONS) == {"skipped": True}  # no script: skips


def test_a_person_cannot_both_skip_and_answer() -> None:
    with pytest.raises(ValueError):
        Person(skip=True, say="yes")


async def test_a_whole_trial_runs_on_the_mock_provider(tmp_path: Path) -> None:
    """The real harness, tools, person and graders, with the mock standing in for the model.
    The mock asks its own two questions when it sees "ask me"."""
    suite = Suite(
        environment=Workspace,
        tools=tools,
        tasks=[
            {
                "id": "mock_asks",
                "suite": "regression",
                "input": "ask me",
                "person": {"say": "whatever suits"},
                "graders": [
                    {"name": "asked", "type": "tool_calls", "must_succeed": ["ask_user_question"]}
                ],
            }
        ],
    )
    results = await evaluate(suite, provider=MockProvider, trials=2, results_dir=tmp_path)

    assert results.tasks["mock_asks"]["passed"] == 2
    assert results.tasks["mock_asks"]["cards"] == 1
    record = json.loads(Path(results.trials[0]["transcript"]).read_text(encoding="utf-8"))
    assert record["cards_shown"][0]["answer"]["answers"]  # the person answered every question
    assert "whatever suits" in record["tool_log"][0]["output"]
    assert record["stop_reason"] == "stop"


async def test_a_run_stopped_at_max_turns_is_the_agents_failure_not_infrastructure(
    tmp_path: Path,
) -> None:
    """The loop ends both a max_turns run and a provider failure with an "error" message. Only
    the provider failure may be left out of pass^k, so the two must be told apart."""
    env = Workspace()
    env.setup(tmp_path)
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=MockProvider(),
            system="",
            tools=list(tools(env, ScriptedPerson())),
            max_turns=1,
        )
    )
    async for _ in harness.prompt("list files"):
        pass
    assert transcript.stop_reason(harness.messages) == "max_turns"


async def test_judge_tasks_need_a_judge(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="judge"):
        await evaluate(
            SUITE, provider=MockProvider, task_ids=["two_sams_asks_which"], results_dir=tmp_path
        )


def test_a_live_run_refuses_to_start_outside_the_container(monkeypatch) -> None:
    monkeypatch.delenv("HUMPHREY_EVALS_IN_CONTAINER", raising=False)
    with pytest.raises(SystemExit, match="docker compose run"):
        contained(on_host=False)
    contained(on_host=True)  # an explicit opt-in is allowed
    monkeypatch.setenv("HUMPHREY_EVALS_IN_CONTAINER", "1")
    contained(on_host=False)
