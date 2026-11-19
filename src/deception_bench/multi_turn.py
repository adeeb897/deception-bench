"""L3 multi-turn escalation (original: processor/multi_turn.py, followup.py, eval_multi_turn.py)."""

import json
from typing import Any

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from deception_bench import prompts
from deception_bench.scorer import (
    JUDGE_CONFIG,
    JUDGE_DEFAULT_MODEL,
    answer_format,
    deception_metrics,
    judge_record,
    judge_to_score,
    timestamp_now,
)
from deception_bench.solvers import clean_output

# Follow-up writer and its sampling settings. Original: config.GPT_MODEL_PARAMS.
# The original uses gpt-3.5-turbo, which OpenRouter no longer serves; gpt-4o-mini replaces it.
# Override the model with `--model-role followup=<model>`.
FOLLOWUP_CONFIG = GenerateConfig(max_tokens=500, temperature=0.7, top_p=0.95)
FOLLOWUP_DEFAULT_MODEL = "openrouter/openai/gpt-4o-mini"

# state.store key holding the conversation, read by deception_judge_multi_turn.
CONVERSATION_KEY = "conversation"


def build_followup_prompt(conversation: list[dict[str, Any]]) -> str:
    """Build the prompt asking the follow-up model for the next instruction.

    Args:
        conversation: The turn dicts so far.

    Returns:
        The original's follow-up prompt: the prefix, one "User: ... / Model: ..."
        block per turn, then the instructions.
    """
    history = "\n".join(
        f"User: {turn['instruction']}\nModel: {turn['generated_text']}" for turn in conversation
    )
    return prompts.FOLLOWUP_PREFIX + history + prompts.FOLLOWUP_TEMPLATE


def is_stop(text: str) -> bool:
    """Return True for the replies on which the original ends a conversation."""
    return not text.strip() or text.startswith("Error:")


@solver
def followup_loop(max_turns: int = 5) -> Solver:
    """Talk to the target for up to `max_turns` turns, escalating each time.

    Each turn the target sees only the current instruction (plus the JSON-format
    suffix), never earlier turns. After each reply the follow-up model reads the
    conversation so far and writes the next, pushier instruction.

    Args:
        max_turns: Maximum number of target replies (the original uses 5).

    Returns:
        A solver that stores the list of turn dicts ({"turn", "instruction",
        "generated_text", "timestamp"}) under CONVERSATION_KEY in state.store.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        target = get_model()
        followup_model = get_model(
            role="followup", default=FOLLOWUP_DEFAULT_MODEL, config=FOLLOWUP_CONFIG
        )
        instruction = state.input_text
        conversation: list[dict[str, Any]] = []

        for turn in range(1, max_turns + 1):
            output = await target.generate(instruction + prompts.THOUGHT_RESPONSE_SUFFIX)
            reply = clean_output(output.completion)
            conversation.append(
                {
                    "turn": turn,
                    "instruction": instruction,
                    "generated_text": reply,
                    "timestamp": timestamp_now(),
                }
            )
            if is_stop(reply) or turn == max_turns:
                break
            followup = await followup_model.generate(build_followup_prompt(conversation))
            instruction = followup.completion
            if is_stop(instruction):
                break

        state.store.set(CONVERSATION_KEY, conversation)
        return state

    return solve


def build_multi_turn_judge_prompt(
    metadata: dict[str, Any], conversation: list[dict[str, Any]], timestamp: str | None
) -> str:
    """Build the exact prompt the original L3 judge receives.

    Args:
        metadata: The sample's metadata; metadata["condition"] is the L2 name.
        conversation: The turn dicts stored by followup_loop.
        timestamp: A "%Y-%m-%d %H:%M:%S" string, or None to leave the key out.

    Returns:
        prompts.MULTI_TURN_JUDGE_TEMPLATE filled in, without the JSON-format suffix.
    """
    condition = metadata["condition"]
    responses = {"conversation": conversation, "total_turns": len(conversation)}
    record = judge_record(metadata, responses, timestamp)
    return prompts.MULTI_TURN_JUDGE_TEMPLATE % (
        f"`{condition}`",
        f"`{condition}`",
        answer_format(condition),
        json.dumps(record, ensure_ascii=False),
    )


@scorer(metrics={"thought": deception_metrics(), "response": deception_metrics()})
def deception_judge_multi_turn() -> Scorer:
    """Label a whole L3 conversation with the judge model.

    The judge is asked about the sample's L2 condition name, as in the original.

    Returns:
        A scorer returning the same kind of Score as deception_judge; its answer
        is the target's last reply.
    """

    async def score(state: TaskState, target: Target) -> Score:
        conversation = state.store.get(CONVERSATION_KEY)
        prompt = build_multi_turn_judge_prompt(state.metadata, conversation, timestamp_now())
        return await judge_to_score(
            get_model(role="grader", default=JUDGE_DEFAULT_MODEL, config=JUDGE_CONFIG),
            prompt,
            state.metadata["condition"],
            conversation[-1]["generated_text"],
        )

    return score
