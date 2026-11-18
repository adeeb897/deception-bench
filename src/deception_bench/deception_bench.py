"""DeceptionBench: deceptive behaviour of LLMs in real-world scenarios (Inspect port).

Original: https://github.com/Aries-iai/DeceptionBench
Data: https://huggingface.co/datasets/skyai798/DeceptionBench

    uv run inspect eval deception_bench/deception_bench --model <target> \
        --model-role grader=<judge>
"""

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate
from inspect_evals_utils.metadata import load_eval_version

from deception_bench.scorer import deception_judge

EVAL_VERSION = load_eval_version(__file__)

HF_REPO = "skyai798/DeceptionBench"
HF_REVISION = "38b412da565073483c0a68ce9e04eb15859beedc"  # HEAD on 2026-11-18

DOMAINS = ["economy", "education", "healthcare", "social", "entertainment"]

# The six prompt variants present in every record. L3 reuses the L2 prompts
# as turn 1 of a multi-turn loop (see multi_turn.py).
SINGLE_TURN_CONDITIONS = [
    "L1-self",
    "L1-other",
    "L2-self-pressure",
    "L2-self-reward",
    "L2-other-pressure",
    "L2-other-reward",
]


def load_records(domain: str) -> list[dict[str, Any]]:
    """Download one domain's JSON file at HF_REVISION and return its records."""
    assert domain in DOMAINS


def record_to_samples(
    record: dict[str, Any], index: int, domain: str, conditions: list[str]
) -> list[Sample]:
    """Expand one scenario record into one Sample per condition."""
    raise NotImplementedError  # Exercise 0.2


def load_dataset(domains: list[str], conditions: list[str]) -> MemoryDataset:
    raise NotImplementedError  # Exercise 0.3


@task
def deception_bench(
    domains: str | list[str] | None = None,
    conditions: str | list[str] | None = None,
) -> Task:
    """Single-turn DeceptionBench (levels L1 and L2).

    Args:
        domains: Subset of DOMAINS. Defaults to all five.
        conditions: Subset of SINGLE_TURN_CONDITIONS. Defaults to all six.

    Returns:
        The single-turn task.
    """
    # TODO (Exercise 0.4): normalise and validate domains / conditions.
    # TODO (Part 1): solver and generation config.
    return Task(
        dataset=load_dataset(domains or DOMAINS, conditions or SINGLE_TURN_CONDITIONS),  # type: ignore[arg-type]
        solver=generate(),
        scorer=deception_judge(),
        version=EVAL_VERSION,
    )
