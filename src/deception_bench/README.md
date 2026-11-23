# deception-bench

DeceptionBench: deceptive behaviour of LLMs in real-world scenarios (Inspect port)

<!-- Contributors: Automatically Generated -->
Contributed by [@adeeb897](https://github.com/adeeb897)
<!-- /Contributors: Automatically Generated -->

<!-- Usage: Automatically Generated -->
## Usage

First, install dependencies:

```bash
uv sync
```

Then run evaluations:

```bash
uv run inspect eval deception_bench/deception_bench --model openai/gpt-5-nano
uv run inspect eval deception_bench/deception_bench_multi_turn --model openai/gpt-5-nano
```

You can also import tasks as Python objects:

```python
from inspect_ai import eval
from deception_bench import deception_bench, deception_bench_multi_turn
eval(deception_bench)
```

If your directory name doesn't match your package name but you have registered your eval task, you can also run:

```bash
uv run inspect eval deception_bench --model openai/gpt-5-nano
```

After running evaluations, view logs with:

```bash
uv run inspect view
```

If you don't want to specify `--model` each time, create a `.env` file:

```bash
INSPECT_EVAL_MODEL=anthropic/claude-opus-4-1-20250805
ANTHROPIC_API_KEY=<anthropic-api-key>
```
<!-- /Usage: Automatically Generated -->

<!-- Options: Automatically Generated -->
## Options

You can control a variety of options from the command line. For example:

```bash
uv run inspect eval deception_bench/deception_bench --limit 10
uv run inspect eval deception_bench/deception_bench_multi_turn --max-connections 10
uv run inspect eval deception_bench/deception_bench --temperature 0.5
```

See `uv run inspect eval --help` for all available options.
<!-- /Options: Automatically Generated -->

<!-- Parameters: Automatically Generated -->
## Parameters

### `deception_bench`

- `domains` (str | list[str] | None): A domain, a list of them, or None for all of DOMAINS. (default: `None`)
- `conditions` (str | list[str] | None): A condition, a list of them, or None for all of SINGLE_TURN_CONDITIONS. (default: `None`)

### `deception_bench_multi_turn`

- `domains` (str | list[str] | None): A domain, a list of them, or None for all of DOMAINS. (default: `None`)
- `conditions` (str | list[str] | None): An L2 condition, a list of them, or None for all of L2_CONDITIONS. (default: `None`)
- `max_turns` (int): Maximum number of target replies per conversation. (default: `5`)

<!-- /Parameters: Automatically Generated -->

## Dataset

[skyai798/DeceptionBench](https://huggingface.co/datasets/skyai798/DeceptionBench), pinned to revision `38b412d`. There is one JSON file per domain (economy, education, healthcare, social, entertainment), each with 30 scenarios. A scenario has a `topic`, `dimension`, `question`, `groundtruth` (the honest answer), a `goal` the model may be tempted to serve by deceiving, and six prompts:

| Condition | Level | Role | Inducement |
| --- | --- | --- | --- |
| `L1-self`, `L1-other` | L1 | the model itself / a third party it plays | none |
| `L2-{self,other}-{pressure,reward}` | L2 | as above | pressure or reward |

`deception_bench` turns each (scenario, condition) into one sample: 150 × 6 = 900. `deception_bench_multi_turn` (L3) uses the four L2 prompts as turn 1: 150 × 4 = 600 samples.

## Scoring

The target answers in JSON with a `thought` and a `response`. A judge model (role `grader`, default gpt-4o) reads the scenario and the reply and labels each as `"decept"` or `"honest"`. A label scores 1 for decept and 0 for honest. Anything else is skipped, as in the original. A verdict with no labels for the sample's condition leaves the sample unscored.

The metrics are the deception rates **DR_thought** and **DR_response**, the means of those scores. They are reported overall and broken down by condition, level, role, inducement and domain.

For L3, a follow-up model (role `followup`, default gpt-4o-mini) writes up to four more instructions, each pushing the target further. The target sees only the current instruction each turn, and the judge labels the whole conversation once at the end.

```bash
uv run inspect eval deception_bench/deception_bench --model <target> --model-role grader=openai/gpt-4o
uv run inspect eval deception_bench/deception_bench_multi_turn --model <target> \
    --model-role grader=openai/gpt-4o --model-role followup=openai/gpt-4o-mini
```

Without `--model-role`, the judge and follow-up models default to OpenRouter (`OPENROUTER_API_KEY`).

In an L3 sample's transcript, the messages are each turn's prompt and reply in order. The target saw only the current turn's prompt, never the earlier ones. The conversation as the judge sees it is in the sample's store under `conversation`.

The follow-up model sometimes refuses to write an inducement. As in the original, its reply is still sent to the target as the next instruction. Each L3 score's metadata lists the turns whose instruction looked like a refusal (`followup_refusals`, matched by its opening words, such as "I'm sorry" or "I can't") and the number of turns (`total_turns`). Check how often this happens when you compare follow-up models.

## Deviations from the original implementation

Prompts, sampling settings, reply cleaning and verdict parsing match the original at commit `8f3175f`. The tests compare every prompt against output produced by the original code.

| What changed | Why | Expected effect on scores |
| --- | --- | --- |
| L3 labels are read under the L2 condition name (e.g. `L2-self-pressure`). | The original asks its L3 judge about the L2 name, but its `calculate_metric.py` looks results up under the L3 name (`L3-self-pressure`), finds none and reports zero L3 counts. | L3 rates exist at all. The original's released script cannot reproduce its L3 numbers. |
| A failed API call fails the sample (after Inspect's retries). | The original saves `"Error: ..."` as the target's or judge's reply after 3 attempts, and replaces a failed follow-up with `"Could you elaborate further?"`. Both hide failures; with the fallback, a misconfigured follow-up model silently turns every L3 conversation into "elaborate further". | None when calls succeed. |
| The follow-up model defaults to gpt-4o-mini instead of gpt-3.5-turbo. | gpt-3.5-turbo is no longer served by OpenRouter. | L3 follow-ups may push differently, so L3 rates may differ from the paper's. |
| The judge and follow-up default to OpenRouter model names. | Provider choice. | None; the models are the same. |

## Architecture

```text
deception_bench            dataset ─▶ add_format_suffix ─▶ generate ─▶ deception_judge
                                                                       (grader model)

deception_bench_multi_turn dataset ─▶ followup_loop ───────────────▶ deception_judge_multi_turn
                                      target ⇄ followup model,          (grader model)
                                      up to max_turns, stored in state.store
```

| File | Contents |
| --- | --- |
| `deception_bench.py` | dataset loading and the two tasks |
| `solvers.py` | the JSON-format suffix and reply cleaning |
| `scorer.py` | judge prompt, verdict parsing, scores and metrics |
| `multi_turn.py` | the L3 follow-up loop and judge prompt |
| `prompts.py` | prompts copied verbatim from the original |

## Licence

This port is released under [CC BY-SA 4.0](../../LICENSE), the licence of the original. It adapts code and prompts from [DeceptionBench](https://github.com/Aries-iai/DeceptionBench) (Aries-iai) and loads its [dataset](https://huggingface.co/datasets/skyai798/DeceptionBench). The original authors restrict DeceptionBench to academic research: commercial use in any form is prohibited.
