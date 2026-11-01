"""Alignment: does fits_diet_judge agree with your own labels?

This tests the judge, not the agent. It runs `fits_diet_judge` over a dataset
of Recipe Remixer traces you labeled by hand, so the judge's column sits next
to yours. Rows where they differ are where the judge, or a label, is wrong.

Label the traces in an annotation queue with the feedback keys
`vegetarian_human` and `dairy_free_human`, then add them to the dataset set by
REMIXER_JUDGE_DATASET_NAME. Your labels stay on the traces you labeled, so this
script copies them onto each experiment run, the same way
dinner_bot/llm_eval/judge_experiments.py adds its human labels.

Run from the repo root: uv run python -m recipe_remixer.evals.llm.fits_diet.alignment
"""

import os

from dotenv import load_dotenv
from langsmith import Client, evaluate, get_current_run_tree

from recipe_remixer.evals.llm.fits_diet.judge import fits_diet_judge

load_dotenv()

HUMAN_KEYS = ["vegetarian_human", "dairy_free_human"]


def _human_labels(client: Client, examples: list) -> dict[str, list]:
    """Your labels, by the ID of the trace each example was added from."""
    run_ids = [e.source_run_id for e in examples if e.source_run_id]
    if not run_ids:
        return {}
    labels: dict[str, list] = {}
    for feedback in client.list_feedback(run_ids=run_ids, feedback_key=HUMAN_KEYS):
        labels.setdefault(str(feedback.run_id), []).append(feedback)
    return labels


def main() -> None:
    client = Client()
    dataset_name = os.environ["REMIXER_JUDGE_DATASET_NAME"]

    examples = list(client.list_examples(dataset_name=dataset_name))
    stored = {str(e.id): e.outputs for e in examples}
    labels = _human_labels(client, examples)

    # The target replays the output saved with each example, so the judges
    # grade the exact recipe you labeled. Two examples can share the same
    # inputs, so it looks the output up by the example ID evaluate() puts on
    # the run.
    def remix_request(inputs: dict) -> dict:
        return stored[str(get_current_run_tree().reference_example_id)]

    results = evaluate(
        remix_request,
        data=dataset_name,
        evaluators=[fits_diet_judge],
        experiment_prefix="fits-diet-alignment",
        client=client,
    )
    client.flush()

    labeled = 0
    for row in results:
        run = row["run"]
        human = labels.get(str(row["example"].source_run_id), [])
        labeled += bool(human)
        for feedback in human:
            client.create_feedback(
                run.id,
                key=feedback.key,
                score=feedback.score,
                value=feedback.value,
                comment=feedback.comment,
                session_id=results.experiment_id,
            )

    print(f"Copied your labels onto {labeled} of {len(examples)} examples.")


if __name__ == "__main__":
    main()
