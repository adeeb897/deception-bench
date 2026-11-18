"""Target-side pieces (original: utils.generate_single_response)."""

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver

# Exercise 1.2. Original: config.SINGLE_TURN_MODEL_PARAMS.
TARGET_CONFIG = GenerateConfig()


@solver
def add_format_suffix() -> Solver:
    """Exercise 1.1: append prompts.THOUGHT_RESPONSE_SUFFIX to the user prompt."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        raise NotImplementedError

    return solve


def clean_output(text: str) -> str:
    """Exercise 1.3: the original's post-processing of every model output."""
    raise NotImplementedError
