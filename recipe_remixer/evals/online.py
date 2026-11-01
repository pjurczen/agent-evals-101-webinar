"""Set up online evaluators on the Recipe Remixer tracing project.

Online evaluators score traces as they arrive, so failures show up in
production without running an experiment. This script creates two evaluators
and attaches each to the project with a rule: a filter, a sampling rate, and
a date to backfill from.

- steps_match_judge: the LLM judge in llm/steps_match/judge.py, on every
  remix_request trace. Its prompt and model go to the prompt hub as
  recipe-remixer-steps-match.
- substitute_found: the code evaluator in code/substitute_found.py, on every
  get_substitute tool run.

The feedback keys match the offline evaluators, so the same filters work on
traces and on experiments. The LLM judge calls OpenAI from LangSmith, so the
workspace needs an OPENAI_API_KEY secret (Settings > Secrets).

Rerunning skips evaluators that already exist. Backfill only happens when a
rule is created, so to change an evaluator and rescore past runs, pass
--replace. That deletes the evaluator, its rule, and the scores it gave, then
creates it again.

Run from the repo root:

    uv run python -m recipe_remixer.evals.online --backfill-from 2026-10-29
    uv run python -m recipe_remixer.evals.online --backfill-from 2026-10-29 --replace
"""

import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts.structured import StructuredPrompt
from langsmith import Client
from langsmith.utils import LangSmithConflictError
from pydantic import BaseModel, Field

from recipe_remixer.evals.llm.judge import JUDGE_MODEL
from recipe_remixer.evals.llm.steps_match.judge import STEPS_MATCH_PROMPT

load_dotenv()

PROMPT_HANDLE = "recipe-remixer-steps-match"
SUBSTITUTE_FOUND_CODE = Path(__file__).parent / "code" / "substitute_found.py"


class StepsMatch(BaseModel):
    # Each top-level key becomes a feedback key, so this one matches the
    # offline judge's.
    steps_match_judge: bool = Field(description="True for PASS, false for FAIL")


def _steps_match_prompt() -> str:
    # Swap the closing PASS/FAIL line for the boolean the schema asks for. A
    # judge that gets no recipe should fail it, not pass it.
    question, _ = STEPS_MATCH_PROMPT.rsplit("\n\n", 1)
    return question + "\n\nIf the recipe is missing or empty, answer false. Otherwise answer true for PASS or false for FAIL."


async def _create_steps_match(client: Client):
    prompt = StructuredPrompt.from_messages_and_schema(
        [("human", _steps_match_prompt())], schema=StepsMatch.model_json_schema()
    )
    try:
        client.push_prompt(PROMPT_HANDLE, object=prompt | init_chat_model(JUDGE_MODEL))
    except LangSmithConflictError:
        pass  # The prompt hasn't changed since the last push, so use that commit.
    commit = client.pull_prompt_commit(PROMPT_HANDLE)
    return await client.evaluators.create(
        name="steps_match_judge",
        type="llm",
        llm_evaluator={
            "prompt_repo_handle": PROMPT_HANDLE,
            "commit_hash_or_tag": commit.commit_hash,
            # The run's outputs are under "output", not "outputs" as in the
            # docs' example. "outputs.recipe" reaches the judge as null.
            "variable_mapping": {"recipe": "output.recipe"},
        },
    )


async def _create_substitute_found(client: Client):
    return await client.evaluators.create(
        name="substitute_found",
        type="code",
        code_evaluator={"code": SUBSTITUTE_FOUND_CODE.read_text(), "language": "python"},
    )


async def _delete_scores(client: Client, project_id: str, since: datetime, name: str, filter: str) -> None:
    # Otherwise the old scores would sit next to the new ones on the same runs.
    runs = client.runs.query(project_ids=[project_id], min_start_time=since, filter=filter, selects=["ID"])
    run_ids = [run.id async for run in runs]
    for feedback in client.list_feedback(run_ids=run_ids, feedback_key=[name]):
        if feedback.feedback_source and feedback.feedback_source.type == "auto_eval":
            client.delete_feedback(feedback.id)


async def _find(client: Client, name: str):
    return next(iter([e async for e in client.evaluators.list(name_contains=name) if e.name == name]), None)


async def setup(
    client: Client, project_id: str, backfill_from: datetime, name: str, create, filter: str, replace: bool
) -> None:
    existing = await _find(client, name)
    if existing and not replace:
        print(f"{name} already exists, skipping it. Pass --replace to recreate it.")
        return
    if existing:
        await client.evaluators.delete(existing.id, delete_run_rules=True)
        await _delete_scores(client, project_id, backfill_from, name, filter)

    created = await create(client)
    # The SDK has no helper for rules yet, so this calls the REST API directly.
    client.request_with_retries(
        "POST",
        "/runs/rules",
        json={
            "display_name": name,
            "session_id": project_id,
            "evaluator_id": created.evaluator.id,
            "filter": filter,
            "sampling_rate": 1.0,
            "backfill_from": backfill_from.isoformat(),
        },
    )
    print(f"Created {name} on runs matching {filter}, backfilling from {backfill_from:%Y-%m-%d}.")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill-from", required=True, help="Score past runs from this date, e.g. 2026-10-29")
    parser.add_argument("--replace", action="store_true", help="Delete and recreate evaluators that already exist")
    args = parser.parse_args()

    client = Client()
    project_id = str(client.read_project(project_name=os.environ["LANGSMITH_PROJECT"]).id)
    backfill_from = datetime.fromisoformat(args.backfill_from).replace(tzinfo=timezone.utc)

    for name, create, filter in [
        ("steps_match_judge", _create_steps_match, 'eq(name, "remix_request")'),
        ("substitute_found", _create_substitute_found, 'eq(name, "get_substitute")'),
    ]:
        await setup(client, project_id, backfill_from, name, create, filter, args.replace)


if __name__ == "__main__":
    asyncio.run(main())
