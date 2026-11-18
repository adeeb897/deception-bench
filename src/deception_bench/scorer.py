"""LLM judge (original: processor/eval_single_turn.py)."""

from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

# Exercise 2.2. Original: config.EVAL_MODEL_PARAMS, judge defaults to gpt-4o.
JUDGE_CONFIG = GenerateConfig()
JUDGE_DEFAULT_MODEL: str | None = None


def build_judge_prompt(metadata: dict, response_text: str, timestamp: str | None) -> str:
    """Exercise 2.1: the exact prompt the original judge receives."""
    raise NotImplementedError


def parse_verdict(text: str) -> dict:
    """Exercise 2.3: raw judge output -> {condition: {"thought": ..., "response": ...}}."""
    raise NotImplementedError


# The original reports two labels per response: "thought" (deceptive intent)
# and "response" (deceptive behaviour). Exercise 3.3 adds breakdowns here.
@scorer(metrics={"thought": [mean()], "response": [mean()]})
def deception_judge() -> Scorer:
    """Exercise 2.4."""

    async def score(state: TaskState, target: Target) -> Score:
        raise NotImplementedError

    return score
