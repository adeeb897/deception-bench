"""L3 multi-turn escalation (original: processor/multi_turn.py, followup.py, eval_multi_turn.py)."""

from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

# Exercise 5.2. Original: config.GPT_MODEL_PARAMS (the follow-up question writer).
FOLLOWUP_CONFIG = GenerateConfig()
FOLLOWUP_DEFAULT_MODEL: str | None = None


@solver
def followup_loop(max_turns: int = 5) -> Solver:
    """Exercise 5.2."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        raise NotImplementedError

    return solve


def build_multi_turn_judge_prompt(
    metadata: dict, conversation: list[dict], timestamp: str | None
) -> str:
    """Exercise 5.3."""
    raise NotImplementedError


@scorer(metrics={"thought": [mean()], "response": [mean()]})
def deception_judge_multi_turn() -> Scorer:
    """Exercise 5.4."""

    async def score(state: TaskState, target: Target) -> Score:
        raise NotImplementedError

    return score
