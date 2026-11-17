# Evals

Task-based evals for Humphrey. The first suite covers one feature: when Humphrey should ask
you a question (`ask_user_question`), and when it should just get on with it.

This follows the approach I use in [agentic-evals](https://github.com/jSwords91/agentic-evals).
See more of my methodology there: why grade outcomes and not routes, why pass^k, when a judge
earns its place. The comments here use the same decision IDs (E3, E7 and so on) as that repo's
design notes.

## Running it

```bash
uv run --group evals python -m evals check evals/ask_user/tasks.yaml       # free, no model
docker compose run --rm evals run evals/ask_user/tasks.yaml --trials 3     # live, costs money
```

`check` replays each task's reference solution through Humphrey's real tools and grades it. If
something fails there, the task or a grader is wrong, not Humphrey. It runs fine on your
machine: the reference solutions only touch a temporary copy of the workspace.

`run` calls the model, so it needs `OPENAI_API_KEY` in `.env`, and it runs in a container on
purpose. The tasks ask Humphrey to delete and tidy files. Its read, write and edit tools are
confined to the trial's folder, but its bash tool only starts there: nothing stops a confused
`rm` from reaching your own files. In the container, the worst it can damage is the container.
The code is mounted read-only, and only `results/` is written back. `run` refuses to start
outside the container unless you pass `--on-host`.

The useful options:

| Option | Default | |
| --- | --- | --- |
| `--model`, `--effort` | `OPENAI_MODEL`, `REASONING_EFFORT` | The model under test |
| `--judge-model` | `gpt-6.1-sol` | Ideally a different model from the agent's, but the same is allowed |
| `--suite` | `all` | `regression`, `capability` or `all` |
| `--task ID` | | Run only this task (repeatable) |
| `--trials` | 3 | Trials per task |
| `--gate` | | Exit 1 unless mean regression pass^k reaches this |
| `--compare FILE` | | An earlier `results/latest.json`, to see which tasks flipped |

Results go to `results/`: a table, the failed trials, and a transcript per trial in
`results/transcripts/`. Each transcript has the grades, the reply, every tool call, the
question cards Humphrey showed and how they were answered, the files it changed, and the full
messages.

## How it fits Humphrey

A trial runs Humphrey's own code: `build_harness` (its prompt, skills and turn limit) and
`tools_for_run` (its tools). Only two things are swapped:

- **The workspace.** Each trial gets a fresh copy of `ask_user/seed/`, and the tools and the
  prompt both point at it.
- **The person.** In the app, the browser answers question cards through `WebUiBridge`. In a
  trial, a `ScriptedPerson` answers from the task's `person:` block, the same way every time.

```yaml
person: {pick: "O'?Neill", say: "Sam O'Neill, from Delivery"}   # choose the matching option, else type this
person: {skip: true}                                            # skip the questions
```

A task with no `person:` skips any question it's asked.

## The ask_user suite

Every task that should ask has a partner that reads alike but is clear, so an agent that always
asks, or never does, fails half of them:

| Should ask | Shouldn't ask |
| --- | --- |
| Add Sam to the offsite (there are two Sams) | Add Priya (there's one) |
| Change the price in the proposal (two options, two prices) | Change Option A's price |
| Tidy the reports, I only need the final one (two "final" files) | Clear out the .tmp files |
| Draft an intro for my Leeds talk (nothing says what it's about) | Draft an intro for my Bristol talk (there are notes) |
| Tom's changed his email (we don't know the new one) | Tom's email is now ... |

Plus two clear requests with a sensible default (chart the sales, find the best month), and one
capability task: the person skips the "which final?" question, so nothing should be deleted.

The should-ask tasks are obvious on purpose. Borderline cases should come from real use, once
there are transcripts of Humphrey asking when it shouldn't have, or not asking when it should.

### How they're graded

- **Did it ask?** `tool_calls` on `ask_user_question`. This is the one place the evals check a
  tool call, because the card is something you see: an interruption on a clear task is the
  harm, and so is acting on a guess when only you know the answer.
- **Did your answer end up in the work?** `state` graders read the workspace:
  `files:<path regex>` counts files, and `grep:<path regex>:<regex>` counts files that contain
  a match. `unchanged` checks that files are exactly as seeded.
- **Did it ask about the right thing?** A `judge`, with the right answer in the criterion. It's
  used only where code can't decide. Check it against your own reading of some transcripts
  before you trust it.

## Adding a task

1. If it needs new files, add them to `ask_user/seed/`. Keep each task's files its own, so one
   task's graders can't be confused by another's.
2. Add it to `ask_user/tasks.yaml`, with a reference solution.
3. Run `check`, then `run --task <id> --trials 5`, and read the transcripts.

## Layout

| File | What it is |
| --- | --- |
| `environment.py` | The `Environment` base class: the world a trial runs in |
| `person.py` | `Person` and `ScriptedPerson`: who answers the question cards |
| `tasks.py` | Tasks and suites, from YAML |
| `graders.py` | `state`, `reply`, `tool_calls`, `judge`, and how to add your own |
| `metrics.py` | pass@k, pass^k and the gate |
| `transcript.py` | Reading a finished run: reply, tool calls, tokens |
| `runner.py` | `evaluate()` and `replay()` |
| `results.py`, `report.py` | The table, failures and comparisons |
| `ask_user/` | The suite: `env.py`, `seed/`, `tasks.yaml` |

`tests/test_evals.py` runs offline: every reference solution passes, acting without asking fails
the should-ask tasks, and a whole trial runs on the mock provider.
