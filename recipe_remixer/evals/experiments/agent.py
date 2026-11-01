"""Run Recipe Remixer over a LangSmith dataset with the judges and the tool check.

Build the dataset from live traces first with recipe_remixer.evals.datasets.

Run from the repo root:

    uv run python -m recipe_remixer.evals.experiments.agent --prefix remixer-buggy
    uv run python -m recipe_remixer.evals.experiments.agent --prefix remixer-fixed
"""

import argparse
import os

from dotenv import load_dotenv
from langsmith import evaluate

from recipe_remixer.agent import remix
from recipe_remixer.evals.code.substitute_found import substitute_found
from recipe_remixer.evals.llm.fits_diet.judge import fits_diet_judge
from recipe_remixer.evals.llm.steps_match.judge import steps_match_judge

load_dotenv()


def target(inputs: dict) -> dict:
    return remix(inputs["message"], inputs["recipe"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="remixer", help="Experiment name prefix")
    args = parser.parse_args()

    evaluate(
        target,
        data=os.environ["REMIXER_DATASET_NAME"],
        evaluators=[steps_match_judge, fits_diet_judge, substitute_found],
        experiment_prefix=args.prefix,
        max_concurrency=4,
    )


if __name__ == "__main__":
    main()
