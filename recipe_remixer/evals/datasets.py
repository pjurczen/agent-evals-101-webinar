"""Build the Recipe Remixer datasets from live traces in LangSmith.

Instead of a file of saved runs, this queries the tracing project with the
same filters you'd use in the UI, so it works the same way on production
traffic:

- REMIXER_DATASET_NAME: one example per remix_request trace. Inputs only,
  since the judges don't need a reference output. The diet goes in the
  example's metadata, so you can filter and split the dataset by it.
- REMIXER_REWRITE_DATASET_NAME: for the single step eval of rewrite. One draft
  per trace that failed the online steps_match_judge: the agent's new
  ingredient list with the original steps.

Run online.py first, so the traces have steps_match_judge scores.

Run from the repo root:

    uv run python -m recipe_remixer.evals.datasets --since 2026-10-29
    uv run python -m recipe_remixer.evals.datasets --since 2026-10-29 --replace
"""

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DIETS = ["vegetarian", "dairy free"]
REMIX_FILTER = 'and(eq(name, "remix_request"), eq(status, "success"))'
AGENT_FILTER = 'eq(name, "recipe_remixer")'
FAILED_STEPS_MATCH = 'and(eq(feedback_key, "steps_match_judge"), eq(feedback_score, 0))'


async def _query(client: Client, project_id: str, since: datetime, selects: list[str], **filters) -> list:
    # runs.query only looks back one day unless you pass min_start_time.
    runs = client.runs.query(project_ids=[project_id], min_start_time=since, selects=selects, **filters)
    return [run async for run in runs]


async def remix_examples(client: Client, project_id: str, since: datetime) -> list[dict]:
    runs = await _query(client, project_id, since, ["ID", "INPUTS"], filter=REMIX_FILTER)
    return [
        {
            "inputs": run.inputs,
            "metadata": {"diet": next(d for d in DIETS if d in run.inputs["message"].lower())},
            "source_run_id": run.id,
        }
        for run in runs
    ]


async def rewrite_examples(client: Client, project_id: str, since: datetime) -> list[dict]:
    agents = await _query(
        client, project_id, since, ["ID", "TRACE_ID", "OUTPUTS"], filter=AGENT_FILTER, trace_filter=FAILED_STEPS_MATCH
    )
    if not agents:
        return []

    # A root run's ID is its trace ID, so this fetches each original recipe.
    roots = await _query(client, project_id, since, ["ID", "INPUTS"], ids=[str(a.trace_id) for a in agents])
    recipes = {str(root.id): root.inputs["recipe"] for root in roots}

    examples = []
    for agent in agents:
        new = agent.outputs["structured_response"]
        draft = {"dish": new["dish"], "ingredients": new["ingredients"], "steps": recipes[str(agent.trace_id)]["steps"]}
        examples.append({"inputs": {"recipe": draft}, "source_run_id": agent.id})
    return examples


def _dedupe(examples: list[dict]) -> list[dict]:
    # The same request can show up more than once, like a retry or the same
    # recipe run twice. Keep the first of each.
    unique: dict[str, dict] = {}
    for example in examples:
        unique.setdefault(json.dumps(example["inputs"], sort_keys=True), example)
    return list(unique.values())


def save(client: Client, name: str, examples: list[dict], replace: bool) -> None:
    if client.has_dataset(dataset_name=name):
        if not replace:
            print(f"Dataset {name} already exists, skipping it. Pass --replace to rebuild it.")
            return
        # Deleting a dataset also deletes the experiments run on it.
        client.delete_dataset(dataset_name=name)

    examples = _dedupe(examples)
    dataset = client.create_dataset(name)
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Created {name} with {len(examples)} examples.")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", required=True, help="Only use traces from this date, e.g. 2026-10-29")
    parser.add_argument("--replace", action="store_true", help="Delete and rebuild datasets that already exist")
    args = parser.parse_args()

    client = Client()
    project_id = str(client.read_project(project_name=os.environ["LANGSMITH_PROJECT"]).id)
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    save(client, os.environ["REMIXER_DATASET_NAME"], await remix_examples(client, project_id, since), args.replace)

    rewrite = await rewrite_examples(client, project_id, since)
    if not rewrite:
        print("No traces have failed the online steps_match_judge yet. If you just ran online.py, wait for the backfill.")
        return
    save(client, os.environ["REMIXER_REWRITE_DATASET_NAME"], rewrite, args.replace)


if __name__ == "__main__":
    asyncio.run(main())
