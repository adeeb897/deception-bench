"""Tests for the DeceptionBench port.

Expected prompts in fixtures/ were produced by running the original code
(Aries-iai/DeceptionBench @ 8f3175f) on the synthetic record in fixtures/record.json.
"""

import importlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import EvalLog, resolve_sample_attachments
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate
from pytest_inspect_evals.assertions import assert_eval_success

from deception_bench import deception_bench, deception_bench_multi_turn, prompts
from deception_bench.deception_bench import (
    DOMAINS,
    EVAL_VERSION,
    SINGLE_TURN_CONDITIONS,
    load_records,
    normalize_list_input,
    record_to_samples,
)
from deception_bench.multi_turn import (
    build_followup_prompt,
    build_multi_turn_judge_prompt,
    deception_judge_multi_turn,
    followup_loop,
    is_refusal,
)
from deception_bench.scorer import (
    build_judge_prompt,
    deception_judge,
    label_to_score,
    parse_verdict,
)
from deception_bench.solvers import add_format_suffix, clean_output

# The package re-exports the task function under the module's name, so import the module explicitly.
tasks_mod = importlib.import_module("deception_bench.deception_bench")

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = json.loads((FIXTURES / "record.json").read_text(encoding="utf-8"))
TIMESTAMP = FIXTURE["timestamp"]
CONVERSATION = FIXTURE["conversation"]
VERDICT = '###FINAL: {"%s": {"thought": "decept", "response": "honest"}}'


def expected(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def fake_record() -> dict[str, Any]:
    return {**FIXTURE["record"], **{c: f"prompt for {c}" for c in SINGLE_TURN_CONDITIONS}}


def metadata(condition: str) -> dict[str, Any]:
    [sample] = record_to_samples(fake_record(), 0, "economy", [condition])
    assert sample.metadata is not None
    return sample.metadata


def scripted(name: str, outputs: list[str]):
    return get_model(
        f"mockllm/{name}",
        custom_outputs=[ModelOutput.from_content(f"mockllm/{name}", o) for o in outputs],
    )


def run(task: Task, target: list[str], grader=None, followup=None) -> EvalLog:
    roles = {}
    if grader is not None:
        roles["grader"] = scripted("grader", grader)
    if followup is not None:
        roles["followup"] = scripted("followup", followup)
    [log] = eval(task, model=scripted("target", target), model_roles=roles or None, display="none")
    assert log.status == "success", log.error
    assert log.samples
    return log


def inputs(log: EvalLog, role: str | None = None) -> list[str]:
    """The last message each model call of `role` received (None = the target)."""
    assert log.samples
    events = resolve_sample_attachments(log.samples[0]).events
    return [e.input[-1].text for e in events if e.event == "model" and e.role == role]


def one_sample(condition: str) -> MemoryDataset:
    return MemoryDataset(record_to_samples(fake_record(), 0, "economy", [condition]))


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve 30 fake records per domain instead of downloading."""
    monkeypatch.setattr(tasks_mod, "load_records", lambda domain: [fake_record()] * 30)


# ---------------------------------------------------------------- dataset


def test_version_comes_from_eval_yaml() -> None:
    assert EVAL_VERSION == "1-A"


def test_normalize_list_input() -> None:
    assert normalize_list_input(None, DOMAINS) == DOMAINS
    assert normalize_list_input("economy", DOMAINS) == ["economy"]
    assert normalize_list_input(["economy", "social"], DOMAINS) == ["economy", "social"]
    with pytest.raises(ValueError, match="Unknown value"):
        normalize_list_input("economyy", DOMAINS)


def test_record_to_samples() -> None:
    samples = record_to_samples(fake_record(), 3, "social", ["L1-self", "L2-other-reward"])
    assert [s.id for s in samples] == ["3-social-L1-self", "3-social-L2-other-reward"]
    assert samples[0].input == "prompt for L1-self"
    assert samples[0].metadata == {
        **FIXTURE["record"],
        "domain": "social",
        "condition": "L1-self",
        "level": "L1",
        "role": "self",
        "inducement": None,
    }
    assert samples[1].metadata
    assert samples[1].metadata["inducement"] == "reward"


@pytest.mark.usefixtures("fake_hub")
def test_single_turn_task_dataset() -> None:
    task = deception_bench(domains=["economy", "social"], conditions="L1-self")
    assert len(task.dataset) == 60
    assert len({s.id for s in task.dataset}) == 60
    with pytest.raises(ValueError, match="Unknown value"):
        deception_bench(conditions="L3-self-reward")


@pytest.mark.usefixtures("fake_hub")
def test_multi_turn_task_dataset() -> None:
    task = deception_bench_multi_turn(domains="economy")
    assert len(task.dataset) == 4 * 30  # all four L2 conditions
    sample = task.dataset[0]
    assert sample.id == "0-economy-L2-self-pressure-L3"
    assert sample.metadata
    assert sample.metadata["level"] == "L3"
    assert sample.metadata["condition"] == "L2-self-pressure"
    with pytest.raises(ValueError, match="Unknown value"):
        deception_bench_multi_turn(conditions="L1-self")


@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_dataset_shape() -> None:
    for domain in DOMAINS:
        records = load_records(domain)
        assert len(records) == 30
        assert all(c in r for r in records for c in SINGLE_TURN_CONDITIONS)


# ---------------------------------------------------------------- target side


def test_clean_output() -> None:
    assert clean_output('```json\n{"a": 1}\n```  ') == '{"a": 1}'


def test_add_format_suffix() -> None:
    task = Task(dataset=one_sample("L1-self"), solver=[add_format_suffix(), generate()])
    log = run(task, target=["reply"])
    assert inputs(log) == ["prompt for L1-self" + prompts.THOUGHT_RESPONSE_SUFFIX]
    assert inputs(log)[0] == expected("target_prompt.txt").replace("hello", "prompt for L1-self")


# ---------------------------------------------------------------- single-turn judge


@pytest.mark.parametrize(
    ("condition", "reply", "fixture"),
    [
        (
            "L2-self-pressure",
            '{"thought": "lie", "response": "Yes, 5 years."}',
            "judge_single_turn.txt",
        ),
        ("L1-other", "plain reply", "judge_single_turn_l1.txt"),
    ],
)
def test_build_judge_prompt_matches_original(condition: str, reply: str, fixture: str) -> None:
    assert build_judge_prompt(metadata(condition), reply, TIMESTAMP) == expected(fixture)


def test_parse_verdict() -> None:
    labels = {"L1-self": {"thought": "decept", "response": "honest"}}
    assert parse_verdict("###FINAL: " + json.dumps(labels)) == labels
    assert parse_verdict("```json\n" + json.dumps(labels) + "\n```") == labels
    assert parse_verdict("I can't help with that.") == {}
    assert parse_verdict('["not", "an", "object"]') == {}


def test_label_to_score() -> None:
    assert label_to_score("decept") == 1.0
    assert label_to_score("honest") == 0.0
    # The original counts only these exact strings.
    assert math.isnan(label_to_score("Decept"))
    assert math.isnan(label_to_score(None))


def test_deception_judge() -> None:
    task = Task(
        dataset=one_sample("L2-self-pressure"),
        solver=[add_format_suffix(), generate()],
        scorer=deception_judge(),
    )
    reply = '{"thought": "lie",\n "response": "Yes"}'
    log = run(task, target=[reply], grader=[VERDICT % "L2-self-pressure"])
    assert log.samples
    score = log.samples[0].scores["deception_judge"]  # type: ignore[index]
    assert score.value == {"thought": 1, "response": 0}

    [judge_prompt] = inputs(log, role="grader")
    timestamp = re.findall(r'"timestamp": "([^"]+)"', judge_prompt)[-1]
    md = metadata("L2-self-pressure")
    assert (
        judge_prompt
        == build_judge_prompt(md, clean_output(reply), timestamp) + prompts.THOUGHT_RESPONSE_SUFFIX
    )


def test_deception_judge_unparseable_verdict_is_unscored() -> None:
    task = Task(dataset=one_sample("L1-self"), solver=generate(), scorer=deception_judge())
    log = run(task, target=["reply"], grader=["I can't help with that."])
    assert log.samples
    value = log.samples[0].scores["deception_judge"].value  # type: ignore[index]
    values = value.values() if isinstance(value, dict) else [value]
    assert all(math.isnan(v) for v in values)  # type: ignore[arg-type]


# ---------------------------------------------------------------- multi-turn


def test_build_followup_prompt_matches_original() -> None:
    assert build_followup_prompt(CONVERSATION) == expected("followup_prompt.txt")


def test_build_multi_turn_judge_prompt_matches_original() -> None:
    got = build_multi_turn_judge_prompt(metadata("L2-other-reward"), CONVERSATION, TIMESTAMP)
    assert got == expected("judge_multi_turn.txt")


def test_followup_loop() -> None:
    task = Task(dataset=one_sample("L2-self-pressure"), solver=followup_loop(max_turns=3))
    log = run(
        task,
        target=['{"thought": "t1"}', "turn two\nreply", "three"],
        followup=["NEXT 2", "NEXT 3"],
    )

    # The target sees only the current instruction plus the suffix, never earlier turns.
    suffix = prompts.THOUGHT_RESPONSE_SUFFIX
    assert inputs(log) == [
        "prompt for L2-self-pressure" + suffix,
        "NEXT 2" + suffix,
        "NEXT 3" + suffix,
    ]
    assert log.samples
    conversation = log.samples[0].store["conversation"]
    assert [t["instruction"] for t in conversation] == [
        "prompt for L2-self-pressure",
        "NEXT 2",
        "NEXT 3",
    ]
    assert conversation[1]["generated_text"] == "turn two reply"
    # No follow-up after the last turn; each follow-up sees the history so far.
    assert inputs(log, role="followup") == [
        build_followup_prompt(conversation[:1]),
        build_followup_prompt(conversation[:2]),
    ]


def test_followup_loop_stops_on_empty_reply() -> None:
    task = Task(dataset=one_sample("L2-self-pressure"), solver=followup_loop(max_turns=5))
    log = run(task, target=["first", ""], followup=["NEXT"])
    assert len(inputs(log)) == 2
    assert len(inputs(log, role="followup")) == 1


def test_followup_loop_records_transcript() -> None:
    task = Task(dataset=one_sample("L2-self-pressure"), solver=followup_loop(max_turns=2))
    log = run(task, target=["one", "two"], followup=["NEXT"])
    assert log.samples
    sample = log.samples[0]
    suffix = prompts.THOUGHT_RESPONSE_SUFFIX
    assert [(m.role, m.text) for m in sample.messages] == [
        ("user", "prompt for L2-self-pressure" + suffix),
        ("assistant", "one"),
        ("user", "NEXT" + suffix),
        ("assistant", "two"),
    ]
    assert sample.output.completion == "two"


@pytest.mark.parametrize("max_turns", [0, -1])
def test_max_turns_must_be_positive(max_turns: int) -> None:
    with pytest.raises(ValueError, match="max_turns"):
        followup_loop(max_turns=max_turns)


def test_multi_turn_task_rejects_zero_max_turns() -> None:
    # Validation runs before the dataset is loaded, so nothing is downloaded.
    with pytest.raises(ValueError, match="max_turns"):
        deception_bench_multi_turn(max_turns=0)


@pytest.mark.parametrize(
    ("text", "refusal"),
    [
        ("I'm sorry, but I can't help with that.", True),
        ("  I can\N{RIGHT SINGLE QUOTATION MARK}t assist with creating deceptive content.", True),
        ("i cannot do that", True),
        ("Sorry, no.", True),
        ("Original question. You must answer as I say.", False),
        ("Answer the question; I'm sorry to push.", False),
    ],
)
def test_is_refusal(text: str, refusal: bool) -> None:
    assert is_refusal(text) is refusal


def test_deception_judge_multi_turn() -> None:
    task = Task(
        dataset=one_sample("L2-self-pressure"),
        solver=followup_loop(max_turns=2),
        scorer=deception_judge_multi_turn(),
    )
    log = run(task, target=["one", "two"], followup=["push"], grader=[VERDICT % "L2-self-pressure"])
    assert log.samples
    score = log.samples[0].scores["deception_judge_multi_turn"]  # type: ignore[index]
    assert score.value == {"thought": 1, "response": 0}
    assert score.answer == "two"
    assert score.metadata == {"total_turns": 2, "followup_refusals": []}


def test_deception_judge_multi_turn_records_followup_refusals() -> None:
    task = Task(
        dataset=one_sample("L2-self-pressure"),
        solver=followup_loop(max_turns=3),
        scorer=deception_judge_multi_turn(),
    )
    log = run(
        task,
        target=["one", "two", "three"],
        followup=["I'm sorry, I can't help with that.", "push"],
        grader=[VERDICT % "L2-self-pressure"],
    )
    assert log.samples
    # As in the original, the refusal is still sent to the target.
    assert inputs(log)[1] == "I'm sorry, I can't help with that." + prompts.THOUGHT_RESPONSE_SUFFIX
    score = log.samples[0].scores["deception_judge_multi_turn"]  # type: ignore[index]
    assert score.metadata == {"total_turns": 3, "followup_refusals": [2]}


# ---------------------------------------------------------------- end to end


@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_end_to_end_with_mock_model() -> None:
    roles = {"grader": "mockllm/model"}
    [log] = eval(
        deception_bench(domains="economy"), model="mockllm/model", model_roles=roles, limit=2
    )
    assert_eval_success(log, expected_sample_count=2)


@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_end_to_end_multi_turn_with_mock_model() -> None:
    roles = {"grader": "mockllm/model", "followup": "mockllm/model"}
    task = deception_bench_multi_turn(domains="economy", max_turns=2)
    [log] = eval(task, model="mockllm/model", model_roles=roles, limit=2)
    assert_eval_success(log, expected_sample_count=2)
