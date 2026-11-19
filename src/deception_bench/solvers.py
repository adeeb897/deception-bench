"""Target-side pieces (original: utils.generate_single_response)."""

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver

from deception_bench import prompts

# Target sampling settings. Original: config.SINGLE_TURN_MODEL_PARAMS (same for multi-turn).
TARGET_CONFIG = GenerateConfig(max_tokens=500, temperature=0.0, top_p=0.95)


@solver
def add_format_suffix() -> Solver:
    """Append prompts.THOUGHT_RESPONSE_SUFFIX to the user prompt, with no separator."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text += prompts.THOUGHT_RESPONSE_SUFFIX
        return state

    return solve


def clean_output(text: str) -> str:
    """Clean a model reply as the original does: newlines to spaces, drop ``` fences, strip."""
    return text.replace("\n", " ").replace("```json", "").replace("```", "").strip()
