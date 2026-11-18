import pytest
from inspect_ai import eval
from pytest_inspect_evals.assertions import assert_eval_success

from deception_bench import deception_bench
from deception_bench.deception_bench import (
    DOMAINS,
    EVAL_VERSION,
    SINGLE_TURN_CONDITIONS,
    load_records,
)

# Port the notebook's test cells here as each part is finished.


def test_version_comes_from_eval_yaml() -> None:
    assert EVAL_VERSION == "1-A"


@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_dataset_shape() -> None:
    for domain in DOMAINS:
        records = load_records(domain)
        assert len(records) == 30
        assert all(c in r for r in records for c in SINGLE_TURN_CONDITIONS)


@pytest.mark.skip(reason="TODO: enable once deception_judge is implemented")
@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_end_to_end_with_mock_model() -> None:
    [log] = eval(tasks=deception_bench(domains=["economy"]), model="mockllm/model", limit=2)
    assert_eval_success(log, expected_sample_count=2)
