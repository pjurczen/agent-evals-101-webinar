"""Single step eval: run `rewrite` on its own, without the rest of the agent.

Each example is the draft `rewrite` should get: the agent's new ingredient
list from a production trace, with the original steps. Build the dataset with
recipe_remixer.evals.datasets, which picks the traces that failed the
online steps_match_judge. The target passes each draft to `rewrite`, and
steps_match_judge checks the steps it returns.

If `rewrite` passes, the step works when it gets the swaps, so the bug is in
what the full agent passes to it.

Run from the repo root: uv run python -m recipe_remixer.evals.experiments.rewrite_step
"""

import os

from dotenv import load_dotenv
from langsmith import evaluate

from recipe_remixer.agent import Recipe, format_recipe, rewrite
from recipe_remixer.evals.llm.steps_match.judge import steps_match_judge

load_dotenv()


def target(inputs: dict) -> dict:
    draft = Recipe(**inputs["recipe"])
    steps = rewrite(draft)
    # steps_match_judge reads the whole recipe, so return it with the new steps.
    return {"recipe": format_recipe(draft.model_copy(update={"steps": steps})), "steps": steps}


def main() -> None:
    evaluate(
        target,
        data=os.environ["REMIXER_REWRITE_DATASET_NAME"],
        evaluators=[steps_match_judge],
        experiment_prefix="rewrite",
        max_concurrency=4,
    )


if __name__ == "__main__":
    main()
