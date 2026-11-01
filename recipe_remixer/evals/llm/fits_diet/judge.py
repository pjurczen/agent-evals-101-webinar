"""LLM judge: does the recipe fit the diet the user asked for?"""

from recipe_remixer.evals.llm.judge import grade

FITS_DIET_PROMPT = """You are grading recipes from an app that changes recipes to fit a diet.

Question: Is this recipe {diet}?

{rule}

A step can call a listed ingredient by a shorter name, like "the stock" for
vegetable stock or "the cheese" for vegan cheese.

Only judge whether the recipe is {diet}. Do not judge any other diet, taste,
food safety, or writing quality.

Recipe:
{recipe}

Explain why in one sentence, then answer PASS or FAIL."""

# The judge only sees the rule for the diet the user asked for. With both
# rules in one prompt, it failed dairy free recipes for having chicken.
DIET_RULES = {
    "vegetarian": """A recipe FAILS if any ingredient or step uses meat, poultry, fish, or
seafood, or something made from them, like chicken stock or fish sauce.
Eggs and dairy are vegetarian.""",
    "dairy free": """A recipe FAILS if any ingredient or step uses milk, butter, cream, cheese,
yogurt, or anything else made from milk. Meat, fish, and eggs are dairy free.
Plant based versions, like oat milk, coconut yogurt, or vegan cheese, are
dairy free.""",
}


def fits_diet_judge(inputs: dict, outputs: dict) -> dict:
    diet = next(d for d in DIET_RULES if d in inputs["message"].lower())
    return grade(FITS_DIET_PROMPT, diet=diet, rule=DIET_RULES[diet], recipe=outputs["recipe"])
