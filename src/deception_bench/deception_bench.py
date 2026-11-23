"""DeceptionBench: deceptive behaviour of LLMs in real-world scenarios (Inspect port).

Original: https://github.com/Aries-iai/DeceptionBench
Data: https://huggingface.co/datasets/skyai798/DeceptionBench
"""

import json
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate
from inspect_evals_utils.huggingface import hf_hub_download
from inspect_evals_utils.metadata import load_eval_version

from deception_bench.multi_turn import (
    deception_judge_multi_turn,
    followup_loop,
    validate_max_turns,
)
from deception_bench.scorer import deception_judge
from deception_bench.solvers import TARGET_CONFIG, add_format_suffix

EVAL_VERSION = load_eval_version(__file__)

HF_REPO = "skyai798/DeceptionBench"
HF_REVISION = "38b412da565073483c0a68ce9e04eb15859beedc"  # HEAD on 2026-11-18

DOMAINS = ["economy", "education", "healthcare", "social", "entertainment"]

# The six prompt variants in every record: level-role[-inducement].
SINGLE_TURN_CONDITIONS = [
    "L1-self",
    "L1-other",
    "L2-self-pressure",
    "L2-self-reward",
    "L2-other-pressure",
    "L2-other-reward",
]

# L3 starts each conversation from an L2 prompt.
L2_CONDITIONS = [c for c in SINGLE_TURN_CONDITIONS if c.startswith("L2")]


def load_records(domain: str) -> list[dict[str, Any]]:
    """Download one domain's JSON file at HF_REVISION and return its records.

    Args:
        domain: One of DOMAINS, e.g. "economy".

    Returns:
        A list of 30 dicts, one per scenario. Each has the keys topic, dimension,
        question, groundtruth and goal, plus one prompt per SINGLE_TURN_CONDITIONS entry.
    """
    path = hf_hub_download(
        repo_id=HF_REPO, filename=f"{domain}.json", repo_type="dataset", revision=HF_REVISION
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def record_to_samples(
    record: dict[str, Any], index: int, domain: str, conditions: list[str]
) -> list[Sample]:
    """Expand one scenario record into one Sample per condition.

    Args:
        record: One dict from load_records.
        index: The record's position in its domain file, used in the id.
        domain: The domain the record came from.
        conditions: Which prompt variants to include.

    Returns:
        One Sample per condition, in the order of `conditions`, with id
        "<index>-<domain>-<condition>", input record[condition], and metadata:
        the record's topic, dimension, question, groundtruth and goal, plus
        domain, condition, level, role and inducement (None for L1).
    """
    samples = []
    for condition in conditions:
        level, role, *inducement = condition.split("-")
        samples.append(
            Sample(
                id=f"{index}-{domain}-{condition}",
                input=record[condition],
                metadata={
                    "domain": domain,
                    "topic": record["topic"],
                    "dimension": record["dimension"],
                    "question": record["question"],
                    "groundtruth": record["groundtruth"],
                    "goal": record["goal"],
                    "condition": condition,
                    "level": level,
                    "role": role,
                    "inducement": inducement[0] if inducement else None,
                },
            )
        )
    return samples


def load_dataset(domains: list[str], conditions: list[str]) -> MemoryDataset:
    """Build the dataset for the given domains and conditions.

    Args:
        domains: Domains to include, each one of DOMAINS.
        conditions: Conditions to include, each one of SINGLE_TURN_CONDITIONS.

    Returns:
        One Sample per (record, condition), ordered by domain, then record, then condition.
    """
    samples = [
        sample
        for domain in domains
        for index, record in enumerate(load_records(domain))
        for sample in record_to_samples(record, index, domain, conditions)
    ]
    return MemoryDataset(samples, name="deception_bench")


def normalize_list_input(value: str | list[str] | None, options: list[str]) -> list[str]:
    """Turn a task argument into a list of options.

    Args:
        value: None (meaning all `options`), one option, or a list of options.
            From the CLI, `-T domains=economy` arrives as a string and
            `-T domains=economy,social` as a list.
        options: The valid values.

    Returns:
        The selected options as a list.

    Raises:
        ValueError: If a value is not in `options`.
    """
    if value is None:
        return options
    values = [value] if isinstance(value, str) else list(value)
    unknown = [v for v in values if v not in options]
    if unknown:
        raise ValueError(f"Unknown value(s) {unknown}; expected some of {options}")
    return values


@task
def deception_bench(
    domains: str | list[str] | None = None,
    conditions: str | list[str] | None = None,
) -> Task:
    """Single-turn DeceptionBench (levels L1 and L2).

    Args:
        domains: A domain, a list of them, or None for all of DOMAINS.
        conditions: A condition, a list of them, or None for all of SINGLE_TURN_CONDITIONS.

    Returns:
        The task. Each sample's reply is labelled by the judge model (role "grader").

    Raises:
        ValueError: If a domain or condition is not recognised.
    """
    return Task(
        dataset=load_dataset(
            normalize_list_input(domains, DOMAINS),
            normalize_list_input(conditions, SINGLE_TURN_CONDITIONS),
        ),
        solver=[add_format_suffix(), generate()],
        scorer=deception_judge(),
        config=TARGET_CONFIG,
        version=EVAL_VERSION,
    )


@task
def deception_bench_multi_turn(
    domains: str | list[str] | None = None,
    conditions: str | list[str] | None = None,
    max_turns: int = 5,
) -> Task:
    """Multi-turn DeceptionBench (level L3): L2 prompts followed by escalating follow-ups.

    Args:
        domains: A domain, a list of them, or None for all of DOMAINS.
        conditions: An L2 condition, a list of them, or None for all of L2_CONDITIONS.
        max_turns: Maximum number of target replies per conversation.

    Returns:
        The task. The follow-up model (role "followup") writes each new
        instruction; the judge model (role "grader") labels the whole conversation.

    Raises:
        ValueError: If a domain or condition is not recognised (L1 conditions
            included), or if max_turns is less than 1.
    """
    validate_max_turns(max_turns)
    dataset = load_dataset(
        normalize_list_input(domains, DOMAINS),
        normalize_list_input(conditions, L2_CONDITIONS),
    )
    for sample in dataset:
        # Report as L3 with distinct ids; the condition stays the L2 name the judge is asked about.
        sample.metadata["level"] = "L3"  # type: ignore[index]
        sample.id = f"{sample.id}-L3"
    return Task(
        dataset=dataset,
        solver=followup_loop(max_turns),  # adds the format suffix itself on every turn
        scorer=deception_judge_multi_turn(),
        config=TARGET_CONFIG,
        version=EVAL_VERSION,
    )
