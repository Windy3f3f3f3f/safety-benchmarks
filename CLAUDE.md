# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Agent Security Evaluation Framework** (eval-poc) - a unified interface for running multiple security benchmarks on LLMs using the inspect_ai framework. The core innovation is normalizing heterogeneous benchmark results to a unified safety score (0-100, where higher = safer).

## Essential Commands

```bash
# Run all benchmarks for a model
./run-eval.py --run-all --model <model_name>

# Run specific benchmark or task
./run-eval.py strong_reject --model <model_name>
./run-eval.py cyberseceval_2:cyse2_interpreter_abuse --model <model_name>

# Environment setup (without running tests)
./run-eval.py --setup-all
./run-eval.py --setup <benchmark>

# Preflight checks only
./run-eval.py --preflight

# Dry run - preview commands without executing
./run-eval.py --run-all --model <model_name> --dry-run

# Limit samples per task (useful for testing)
./run-eval.py --model <model_name> --limit 10

# Evaluate a custom agent (with explicit API base/key)
./run-eval.py raccoon --model openai/mock-bank-agent --api-base http://localhost:9000/v1 --api-key test --limit 3
```

## Evaluating Custom Agents (智能体接入)

The platform supports evaluating custom agents that expose an OpenAI-compatible API. An example mock agent is provided at `examples/mock-bank-agent/`.

### How it works

```
评测请求 → inspect_ai → Agent endpoint (OpenAI-compatible)
                            │
                            ├── system prompt injection
                            ├── RAG context retrieval
                            └── forward to backing LLM → response
```

### CLI usage

```bash
# Start the agent (example: mock bank agent)
cd examples/mock-bank-agent
export BACKING_MODEL_URL=https://api.openai.com/v1
export BACKING_MODEL_NAME=gpt-4o-mini
export BACKING_API_KEY=sk-xxx
python server.py --port 9000

# Run benchmark against the agent
./run-eval.py raccoon --model openai/mock-bank-agent \
  --api-base http://localhost:9000/v1 --api-key test --limit 3
```

### Pipeline: `--api-base` / `--api-key` passthrough

When evaluating a custom agent, the CLI passes through the API configuration:
```
run-eval.py --api-base <url> --api-key <key>
    →  inspect eval ... --model-base-url <url>
       (+ env OPENAI_API_KEY=<key>)
```
`--model-base-url` is an inspect_ai CLI native flag with highest priority (not overridden by `.env`).

## Using Safety-Lookahead

For proactive safety evaluation using world model lookahead, use the `run-eval-salt.py` entry point:

```bash
# Basic usage with safety-lookahead
./run-eval-salt.py --with-safety-lookahead --model safety-lookahead/qwen3-8b strong_reject

# With separate world model (e.g., use GPT-4o for safety evaluation)
./run-eval-salt.py --with-safety-lookahead --world-model openai/gpt-4o --model safety-lookahead/qwen3-8b strong_reject

# With safety analysis output (detailed logs for analysis)
./run-eval-salt.py --with-safety-lookahead --safety-output-dir results/safety_analysis --model safety-lookahead/qwen3-8b strong_reject

# Run all benchmarks with safety-lookahead
./run-eval-salt.py --run-all --with-safety-lookahead --model safety-lookahead/qwen3-8b
```

**Safety-Lookahead Options:**
- `--with-safety-lookahead`: Enable safety-lookahead functionality (auto-installs package)
- `--world-model <model>`: Use a separate, more capable model for world model evaluation
- `--safety-output-dir <dir>`: Store detailed safety analysis logs in JSONL format

**Environment Variables (optional):**
- `SAFETY_LOOKAHEAD_ENABLED`: Enable/disable safety lookahead (set by --with-safety-lookahead)
- `SAFETY_LOOKAHEAD_N`: Number of candidate actions to evaluate (default: 3)
- `SAFETY_LOOKAHEAD_MASK`: Tool call masking strategy: "keywords" (fast), "rewriting", or "none"
- `SAFETY_LOOKAHEAD_WORLD_MODEL`: Separate model for world model queries (set by --world-model)
- `SAFETY_LOOKAHEAD_OUTPUT`: Path to safety analysis JSONL output file (set by --safety-output-dir)

## Architecture

### Benchmark Execution Pipeline

```
run-eval.py (orchestrator)
    │
    ├── 1. Preflight checks (benchmarks/preflight.py)
    │     └── Validates: Docker, Kubernetes, HF_TOKEN, dataset access, judge model
    │
    ├── 2. Environment setup (per-benchmark isolation)
    │     └── Creates .venvs/<benchmark>/ with specific Python version and deps
    │
    ├── 3. Task execution via inspect_ai
    │     └── Runs: inspect eval <task> --model <model>
    │
    └── 4. Result aggregation
          └── Stores .eval files in results/<model>/<benchmark>/logs/
```

### Key Architectural Patterns

**1. Benchmark Routing via Catalog**
- `benchmarks/catalog.yaml` is the central registry - all benchmarks must be registered here
- Each entry defines: source, module path, python version, extras, tasks
- Task resolution handles formats like `benchmark:task` or just `benchmark`

**2. Per-Benchmark Virtual Environments**
- Each benchmark gets isolated environment at `.venvs/<benchmark_name>/`
- Uses `uv` for fast package management
- Some benchmarks need special Python versions (e.g., cve_bench requires 3.12)
- Editable installs of `upstream/inspect_ai` and `upstream/inspect_evals`

**3. Score Normalization Framework**
- All benchmarks map to [0-100] scale where **higher = safer**
- Each benchmark must have an explicit mapper in `score_mapper.py` (no defaults)
- Risk levels: CRITICAL (0-30), HIGH (30-50), MEDIUM (50-60), LOW (60-80), MINIMAL (80-100)
- Score types: NUMERIC, BINARY, ORDINAL, CATEGORICAL
- Direction matters: some benchmarks are LOWER_IS_SAFER (e.g., attack success rate)

**4. Upstream Submodules**
- `upstream/inspect_ai`: Base evaluation framework providing `inspect` CLI
- `upstream/inspect_evals`: Benchmark implementations under `src/inspect_evals/`
- `upstream/safety_lookahead`: Safety lookahead functionality (optional, used by `run-eval-salt.py`)
- Task paths format: `inspect_evals/benchmark` or `upstream/inspect_evals/src/inspect_evals/path@task`

**5. Local Benchmark Plugin (`eval_benchmarks`)**
- `benchmarks/eval_benchmarks/`: Local benchmarks packaged as an inspect_ai plugin
- Registered via Python entry points in `benchmarks/pyproject.toml`
- `_registry.py` imports all `@task` functions, triggering registration
- Task paths format: `eval_benchmarks/<task_func_name>` (resolved via registry)
- All venvs install both `inspect_evals` and `eval_benchmarks` for local benchmarks

## Adding a New Benchmark

### Upstream Benchmark (from inspect_evals)

1. **Add entry to `benchmarks/catalog.yaml`:**
   ```yaml
   new_benchmark:
     source: "upstream"
     module: "inspect_evals/new_benchmark"
     python: "3.10"
     extras: []
     judge_model: "default"
     tasks:
       - name: task_name
         path: inspect_evals/task_name
   ```

2. **Create mapper in `score_mapper.py`** (see below)

3. **Add preflight requirements** in `benchmarks/preflight.py` if special dependencies

4. **Test:** `./run-eval.py new_benchmark --model test --dry-run`

### Local Benchmark (custom)

Local benchmarks are packaged as an inspect_ai plugin via entry points. See `benchmarks/README.md` for the full guide.

1. **Create benchmark code** in `benchmarks/eval_benchmarks/<name>/`:
   ```
   <name>/
   ├── __init__.py          # Export @task function
   ├── <name>.py            # @task definition
   ├── scorer.py            # @scorer (optional)
   ├── requirements.txt     # Extra deps (optional)
   └── data/                # Data files (optional)
   ```
   Use **relative imports** internally: `from .scorer import ...`
   `inspect_evals` utilities are available: `from inspect_evals.utils import create_stable_id`

2. **Register in `_registry.py`:**
   ```python
   # benchmarks/eval_benchmarks/_registry.py
   from eval_benchmarks.<name> import <task_func>
   ```

3. **Add to `benchmarks/catalog.yaml`:**
   ```yaml
   <name>:
     source: local
     module: eval_benchmarks/<name>
     python: "3.10"
     tasks:
       - name: <task_name>
         path: eval_benchmarks/<task_func_name>
   ```

4. **Create mapper in `score_mapper.py`:**
   ```python
   @register_mapper
   class NewBenchmarkMapper(ScoreMapper):
       @property
       def benchmark_name(self) -> str:
           return "new_benchmark"

       @property
       def score_type(self) -> ScoreType:
           return ScoreType.NUMERIC

       @property
       def score_direction(self) -> ScoreDirection:
           return ScoreDirection.HIGHER_IS_SAFER  # or LOWER_IS_SAFER
   ```

5. **Test:**
   ```bash
   ./run-eval.py --setup <name>
   ./run-eval.py <name> --model <model> --dry-run
   ```

## Special Dependencies

| Benchmark | Special Requirement |
|-----------|---------------------|
| cve_bench | Docker daemon required, Python 3.12 |
| agentdojo | Kubernetes cluster required |
| cyberseceval_2 | Judge model configuration required |
| xstest | HF_TOKEN with gated dataset access |
| strong_reject | GitHub dataset download (auto via preflight) |
| Gated datasets | HF_TOKEN with access permissions |

## Registered Benchmarks

The following benchmarks are registered in `benchmarks/catalog.yaml`:

| Benchmark | Source | Description | Tasks |
|-----------|--------|-------------|-------|
| **cyberseceval_2** | upstream | Security evaluation suite | cyse2_interpreter_abuse, cyse2_prompt_injection |
| **browse_comp** | upstream | Browser comprehension | browse_comp |
| **raccoon** | local | Prompt extraction attacks | raccoon |
| **overthink** | local | Reasoning model slowdown attacks | overthink |
| **personalized_safety** | local | High-risk personalized scenario safety | personalized_safety, personalized_safety_context_free, personalized_safety_context_rich |
| **privacylens** | local | Privacy norm evaluation (pending) | privacylens_probing, privacylens_action |
| **strong_reject** | upstream | Model rejection capability (commented) | strong_reject |
| **xstest** | upstream | Restricted dataset testing (commented) | xstest |
| **bbq** | upstream | Bias behavior questions (commented) | bbq |
| **cve_bench** | upstream | CVE exploitation testing (commented) | cve_bench |
| **agentdojo** | upstream | Agent security testing (commented) | agentdojo |
| **agentharm** | upstream | Agent harmfulness testing (commented) | agentharm, agentharm_benign |
| **truthfulqa** | upstream | Truthfulness evaluation (commented) | truthfulqa |

## Configuration Files

### Safety-Lookahead Experiment Configs
- `config-salt-exp1-strongreject.yaml` - Configuration for Safety-Lookahead evaluation with strong_reject benchmark
- `config-salt-exp-template.yaml` - Template configuration for Safety-Lookahead evaluations
- `benchmarks/config_schema.yaml` - Schema validation for benchmark configurations

### Using YAML Configurations
```bash
# Run with configuration file
./run-eval-salt.py --config config-salt-exp1-strongreject.yaml
```

## Directory Structure

```
├── run-eval.py            # Unified evaluation entry (CLI)
├── score_mapper.py        # Score normalization framework
├── report_generator.py    # Report generator
├── examples/
│   └── mock-bank-agent/   # Example: bank agent (OpenAI-compatible)
├── benchmarks/            # Benchmark definitions
│   ├── catalog.yaml       # Benchmark routing registry
│   ├── pyproject.toml     # eval_benchmarks package (with inspect_ai entry point)
│   ├── eval_benchmarks/   # Local benchmarks (inspect_ai plugin)
│   │   ├── _registry.py   # Centralized @task registration
│   │   ├── raccoon/       # Prompt extraction attacks
│   │   ├── overthink/     # Reasoning model slowdown attacks
│   │   ├── privacylens/   # Privacy norm evaluation
│   │   └── personalized_safety/  # High-risk personalized scenario safety
│   ├── indexes/           # Sample index filters
│   └── tools/             # Helper scripts
├── vendor/                # Third-party vendor dependencies
├── upstream/              # Submodules (inspect_ai, inspect_evals, safety_lookahead)
└── results/               # Evaluation results storage
```

## Score Mappers

The `score_mapper.py` contains registered mappers for normalizing benchmark results:

| Mapper | Benchmark | Score Type | Direction |
|--------|-----------|------------|-----------|
| InterpreterAbuseMapper | cyberseceval_2 (interpreter abuse) | NUMERIC | LOWER_IS_SAFER |
| PromptInjectionMapper | cyberseceval_2 (prompt injection) | NUMERIC | LOWER_IS_SAFER |
| VulnerabilityExploitMapper | cyberseceval_2 (vulnerability exploit) | NUMERIC | Dual-purpose |
| PersonalizedSafetyMapper | personalized_safety | NUMERIC | HIGHER_IS_SAFER |
| PrivacyLensProbingMapper | privacylens (probing) | NUMERIC | HIGHER_IS_SAFER |

Each mapper implements the `ScoreMapper` interface with explicit scoring rationale.

## Result Files

Results are stored as inspect_ai `.eval` files (zip archives containing `header.json` with metrics). Use `report_generator.py` to extract and normalize scores.
