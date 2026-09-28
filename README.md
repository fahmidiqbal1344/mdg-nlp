# Evaluating Multi-Dimensional Generalization of Large Language Models in Temporal Extraction Tasks

This repository contains the code used for studying **multi-dimensional generalization of large language models (LLMs) in temporal extraction tasks**.

The study focuses on two foundational temporal extraction tasks:

- **TIMEX** — time expression extraction
- **EVENTX** — event expression extraction

Rather than evaluating models only on an in-domain test set, we study whether performance generalizes across multiple types of distribution shift. The experimental framework varies the **reasoning strategy**, **model family/architecture/scale**, and **generalization dimension**.

## Generalization Dimensions

We evaluate four dimensions of generalization:

- **Domain generalization** — evaluates transfer from general-domain temporal data to data with different lexical and domain characteristics, including clinical text and vocabulary-shifted data.
- **Robustness generalization** — evaluates performance under meaning-preserving adversarial perturbations.
- **Compositional generalization** — evaluates whether models can jointly extract time and event expressions after being evaluated on the individual extraction tasks.
- **Structural generalization** — evaluates generalization from shorter sentence-level inputs to longer document-level contexts.

## Reasoning Strategies

The repository supports seven reasoning strategies based on combinations of inductive, deductive, and abductive reasoning.

### D — Deductive

Uses an explicit task definition or set of instructions without demonstrations.

### I — Inductive

Uses in-context examples without an explicit task definition. Few-shot examples are sampled from the base dataset.

### A — Abductive

The model generates intermediate reasoning or hypotheses before producing the final extraction.

### D + I — Deductive + Inductive

Uses both explicit task instructions and few-shot demonstrations.

### D + A — Deductive + Abductive

Uses explicit task instructions together with intermediate reasoning.

### I + A — Inductive + Abductive

Uses few-shot demonstrations together with intermediate reasoning.

### D + I + A — Deductive + Inductive + Abductive

Combines task instructions, few-shot demonstrations, and intermediate reasoning.

## Models

Models used in the experiments are available through the MDG Hugging Face organization:

https://huggingface.co/mdg-nlp/models

The experiments study models across multiple:

- model families
- architectures, including Dense and Mixture-of-Experts (MoE)
- parameter scales

## Datasets

Datasets used in the experiments are available through the MDG Hugging Face organization:

https://huggingface.co/mdg-nlp/datasets

The datasets cover the base TIMEX and EVENTX tasks as well as the domain, robustness, compositional, and structural generalization settings.

## Research Questions

### RQ1: Transferability of Base-Task Performance

If a model configuration performs well on the base temporal extraction task, does that performance consistently transfer across different generalization dimensions?

We compare model configurations on the base task and examine whether their relative performance is preserved under domain shift, adversarial perturbation, compositional evaluation, and longer input contexts.

### RQ2: Correlation Across Generalization Dimensions

To what extent are different generalization dimensions correlated?

We study whether a model configuration that performs well under one type of distribution shift also tends to perform well under other types of shift.

### RQ3: Effects of Reasoning Strategy, Model Architecture, and Scale

How do reasoning strategy and model characteristics affect generalization?

We investigate:

- inductive, deductive, and abductive prompting strategies
- combinations of these reasoning strategies
- Dense vs. Mixture-of-Experts architectures
- model scale

## Repository Structure

```text
mdg/
├── dataset_loaders/
│   ├── temporal.py
│   └── spatial.py
│
├── evaluators/
│   ├── temporal_evals.py
│   └── spatial_evals.py
│
├── exp_configs/
│   └── *.yml
│
├── methods/
│   ├── few_shot/
│   ├── zero_shot/
│   ├── prompts/
│   └── openrouter_utils.py
│
├── metrics/
│   ├── partial_spans.py
│   ├── temporal_spans.py
│   └── classification.py
│
├── scripts/
│   └── temporal_data_creation/
│
├── datamodels.py
├── experiment_runner.py
└── registry.py
```

## Setup

Clone the repository:

```bash
git clone git@github.com:nlpatunt/mdg.git
cd mdg
```

Create and activate a Python environment. For example:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install datasets pyyaml python-dotenv requests jinja2 tqdm
```

Depending on the experiment, additional packages may be required by specific evaluation or data-processing scripts.

## OpenRouter API Key

The prompted LLM experiments use the OpenRouter API.

Set an OpenRouter API key in your environment:

```bash
export OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
```

You can also store API keys in a `.env` file:

```text
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
```

The environment variable used by an experiment is specified in the YAML configuration file through `api_key_env`.

For example:

```yaml
method:
  init_params:
    api_key_env: OPENROUTER_API_KEY
```

## Experiment Configuration

Experiments are controlled through YAML configuration files in:

```text
mdg/exp_configs/
```

A configuration specifies three main components:

1. dataset
2. method/model
3. evaluator

Example:

```yaml
dataset:
  name: timex_hf
  init_params:
    hf_location: mdg-nlp/timex-compositional-sentence
  run_params: {}

method:
  name: llama-3.1-openrouter-few-shot-compositional
  init_params:
    api_key_env: OPENROUTER_API_KEY
    model: mistralai/ministral-3b-2512

  run_params:
    reasoning_type: deductive
    sleep_ms: 500
    retries: 3
    timeout_sec: 60
    show_progress: true
    max_docs: 400
    k: 5
    example_selector: random
    seed: 42
    split: test
    per_label_k: 3
    model_output_path: runs/example/model_outputs.jsonl

evaluators:
  - name: compositional_expression_partial
    init_params:
      case_insensitive: true
      strip: true
    run_params: {}
```

## Running an Experiment

Experiments are run through `mdg.experiment_runner`.

From the directory containing the `mdg/` package, run:

```bash
python -m mdg.experiment_runner \
  --config mdg/exp_configs/mock66.yml \
  --output mdg/runs/mock66_results.json
```

The runner:

1. loads the YAML configuration
2. loads the requested dataset
3. initializes the requested model/method
4. runs inference on the test data
5. evaluates the predictions
6. saves the experiment configuration and evaluation results

The output directory is created automatically if it does not already exist.

## Running a Different Configuration

To run another experiment, point `--config` to a different YAML file:

```bash
python -m mdg.experiment_runner \
  --config mdg/exp_configs/<config_name>.yml \
  --output mdg/runs/<result_name>.json
```

For example:

```bash
python -m mdg.experiment_runner \
  --config mdg/exp_configs/mock67.yml \
  --output mdg/runs/mock67_results.json
```

## Configuration Components

### Dataset

The dataset section specifies the registered dataset loader and its initialization parameters.

```yaml
dataset:
  name: timex_hf
  init_params:
    hf_location: <huggingface-dataset>
```

### Method

The method section specifies the registered model wrapper and model-specific parameters.

```yaml
method:
  name: <registered-method-name>

  init_params:
    api_key_env: OPENROUTER_API_KEY
    model: <openrouter-model-name>
```

The reasoning strategy is selected through:

```yaml
run_params:
  reasoning_type: deductive
```

Additional experiment settings such as the number of demonstrations, random seed, retry behavior, and maximum number of documents can also be specified under `run_params`.

### Evaluator

Evaluation methods are specified under:

```yaml
evaluators:
  - name: <registered-evaluator-name>
    init_params: {}
    run_params: {}
```

Multiple evaluators can be included in a single experiment configuration.

## Output

When `--output` is provided, the experiment runner writes a JSON file containing:

```json
{
  "config_hash": "...",
  "config": {},
  "results": []
}
```

The output stores:

- a SHA-256 hash of the experiment configuration
- the complete experiment configuration
- evaluation results

This makes it easier to associate saved results with the exact configuration used to generate them.

## Evaluation

The primary evaluation metric for temporal expression and event expression extraction is an F1 score based on one-to-one span matching.

Predicted and gold expressions are matched using word-level overlap, and the optimal one-to-one assignment is computed before precision, recall, and F1 are calculated.

This allows partial span matches while preventing multiple predicted expressions from being matched to the same gold expression.

## Experimental Overview

The overall experimental pipeline can be summarized as:

```text
Model Family
    ↓
Model Architecture / Scale
    ↓
Reasoning Strategy
    ↓
Temporal Extraction Task
    ├── TIMEX
    └── EVENTX
    ↓
Generalization Dimension
    ├── Domain
    ├── Robustness
    ├── Compositional
    └── Structural
    ↓
Evaluation
```

The goal is not only to measure performance on each dataset independently, but also to study whether strong performance is preserved across different forms of generalization.

## Main Findings

The study finds that strong base-task performance is generally associated with stronger generalization, but this relationship becomes weaker under substantial distribution shifts.

Inductive prompting provides relatively consistent performance across multiple generalization dimensions. However, improvements from reasoning strategy combinations, model architecture, and model scale are not uniform across all dimensions.

These results motivate evaluating temporal extraction systems across multiple distribution shifts rather than relying only on standard in-domain performance.
