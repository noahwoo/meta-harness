# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **text classification reference experiment** for Meta-Harness — a system that autonomously evolves LLM-based "memory systems" for text classification tasks. An outer loop (Claude Opus via CLI) proposes new memory system implementations; an inner loop evaluates them on labeled datasets; the cycle repeats to discover novel memory architectures.

## Commands

```bash
# Install dependencies
uv sync

# Run one evolution iteration (outer loop)
uv run python meta_harness.py --iterations 1

# Run a single memory system on a single dataset (inner loop)
PYTHONPATH=.. uv run python -m text_classification.inner_loop \
  --memory fewshot_all --dataset Symptom2Disease

# Print benchmark summary
uv run python benchmark.py --results

# Run tests
uv run python -m pytest tests/

# Lint (agents/ is excluded via pyproject.toml)
uv run ruff check .

# Validate a new agent imports correctly
uv run python -c "from text_classification.agents.<name> import *; print('OK')"
```

## Architecture

**Execution flow:** `meta_harness.py` → `benchmark.py` → `inner_loop.py`

- **meta_harness.py** — Outer evolution loop. Calls Claude CLI (via `claude_wrapper.py`) to propose 3 new memory systems per iteration, benchmarks them, updates the frontier.
- **benchmark.py** — Async sweep layer. Evaluates all dataset × memory system × seed combinations in parallel (`asyncio` + `ThreadPoolExecutor`).
- **inner_loop.py** — Single-run evaluator. Trains and evaluates one memory system on one dataset. Supports online mode (predict-then-learn) and offline mode (multi-epoch with ground truth).
- **memory_system.py** — Abstract `MemorySystem` base class. All agents implement `predict()`, `learn_from_batch()`, `get_state()`, `set_state()`.
- **llm.py** — Unified LLM client (litellm-backed) with disk caching, retry-with-backoff, parallel batching, and cost tracking.
- **claude_wrapper.py** — Subprocess wrapper around the `claude` CLI with stream-json parsing.

**Key interface — `MemorySystem`:**
```python
class MemorySystem(ABC):
    def predict(self, input: str) -> tuple[str, dict]: ...
    def learn_from_batch(self, batch_results: list[dict]) -> None: ...
    def get_state(self) -> str: ...
    def set_state(self, state: str) -> None: ...
```

Agents use `self.call_llm(prompt)` (not `self._llm` directly) and `extract_json_field(response, "final_answer")` for answer extraction.

**Data layer (`data/`):**
- `api.py` — Dataset loading entrypoints with balanced sampling and split control
- `evaluators.py` — Per-task evaluation (exact match, Jaccard, F1)
- `loaders.py` — JSONL loader + HuggingFace transfer dataset loader
- MCE datasets are vendored locally under `data/`; transfer tasks load from HuggingFace at runtime

**Agents (`agents/`):**
- Write target for generated candidates (auto-discovered by benchmark)
- Baselines: `no_memory.py`, `fewshot_all.py`
- Generated agents are NOT linted (ruff excludes `agents/`)

## Configuration

`config.yaml` is the single source of truth for: datasets, models, split sizes, inner loop settings (mode, epochs, temperature, batch size), benchmark concurrency, and registered memory systems.

## Key Conventions

- `inner_loop.py` uses package-mode imports — run with `PYTHONPATH=..` when executing from this directory
- Default model: `openrouter/openai/gpt-oss-120b` (override via `--model` and optionally `--api-base`)
- Logs go to `logs/<dataset>/<memory>/<model>/` (log.jsonl, val.json, memory.json)
- Test results go to `results/` (separate from logs, never exposed during evolution)
- Evolution state tracked in `evolution_summary.jsonl` and `frontier_val.json`
- Python ≥ 3.11 required
