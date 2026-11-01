"""LLM judge that catches the bug: do the steps only use listed ingredients?"""

from recipe_remixer.evals.llm.judge import grade

STEPS_MATCH_PROMPT = """You are grading recipes from an app that changes recipes to fit a diet.

Question: Do the steps only use ingredients from the ingredient list?

A recipe FAILS if any step uses an ingredient that isn't on the list.

A step can call a listed ingredient by a shorter or more general name. For
example, "the stock" for vegetable stock or "the cheese" for vegan cheese is
fine. Salt, pepper, oil, and water are always allowed.

Only judge whether the steps match the ingredients. Do not judge taste, food
safety, or writing quality.

Recipe:
{recipe}

Explain why in one sentence, then answer PASS or FAIL."""


def steps_match_judge(outputs: dict) -> dict:
    return grade(STEPS_MATCH_PROMPT, recipe=outputs["recipe"])
