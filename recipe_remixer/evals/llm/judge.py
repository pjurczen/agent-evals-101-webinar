"""Shared setup for the LLM judges.

Each judge asks one question about a recipe and returns 1 for PASS or 0 for
FAIL, with a one sentence reason as the comment. The judge's function name is
the feedback key in LangSmith.
"""

import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai:gpt-4o-mini")
JUDGE_REASONING_EFFORT = os.getenv("JUDGE_REASONING_EFFORT", "")


# The reason comes first so the judge decides after explaining. With the
# verdict first, it failed oat milk recipes while saying they had no dairy.
class Verdict(BaseModel):
    reason: str = Field(description="One sentence explaining why")
    verdict: Literal["PASS", "FAIL"]


_model_kwargs: dict[str, Any] = {}
if JUDGE_REASONING_EFFORT:
    _model_kwargs["reasoning_effort"] = JUDGE_REASONING_EFFORT

_judge = init_chat_model(JUDGE_MODEL, **_model_kwargs).with_structured_output(Verdict)


def grade(prompt: str, **fields: str) -> dict:
    result = _judge.invoke(prompt.format(**fields))
    return {"score": int(result.verdict == "PASS"), "comment": result.reason}
