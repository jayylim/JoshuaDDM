# JoshuaDDM
**Drift Diffusion Model for AI Judging**

I am trying to understand how 

### Project Structure
---
```
JoshuaDDM/
├── project_idea.md          # Research question, design notes, evaluation plan
├── notes.md                 # Misc scratch notes
├── requirements.txt         # Shared Python dependencies (currently: numpy)
├── src/
│   ├── ddm/                 # The DDM "judge" itself -- deployment-agnostic,
│   │   │                    # no LLM loading/prompting code lives here.
│   │   ├── drift.py         # DDMConfig, DriftDiffusionModel, TimeScale, etc.
│   │   ├── __init__.py      # Re-exports public API: `from src.ddm import ...`
│   │   └── tests/           # Unit tests for the DDM engine (empty for now)
│   └── model-testing/       # LLM interaction / data collection (unrelated to
│       └── runs/            # the DDM math -- kept in its own folder so the
│           ├── test_agent.py  # judge package has zero LLM dependencies)
│           ├── prompts.jsonl
│           └── outputs.jsonl
```

`src/ddm/` and `src/model-testing/` are intentionally separate: `src/ddm/`
contains only the accumulator/statistics logic (no model loading, no prompts,
no tokenizers), so it can be unit-tested and reasoned about independently of
whichever LLM/deployment eventually feeds it evidence. See
`src/ddm/drift.py` docstrings for the DDM implementation details, and
`project_idea.md` for the research design this code is meant to support.

`src/model-testing/runs/outputs.jsonl` saves records shaped like:
 ```
 {
  "ts": "2026-08-04T16:45:00+08:00",
  "mode": "preset",
  "prompt_id": "p001",
  "prompt": "Explain recursion in one paragraph.",
  "response": "...",
  "model": "qwen2.5:1.5b-instruct",
  "temperature": 0.7,
  "latency_s": 2.14
}
``` 

The agent is now entirely **interactive** (requires manual input at every step) and has **no memory** (turn 2 has no memory of turn 1)

### Plan and To-Do
---
- ~~Theoretical DDM Architecture~~
- ~~Skeleton DDM engine (`src/ddm/drift.py`)~~
- Develop Algorithm for DDM based on LLM Outputs
  - Define the concrete evidence source (probe on hidden states, classifier
    on generated text, tool-call heuristic, ...) and implement it as an
    `EvidenceSource` in `src/ddm/` (e.g. a new `src/ddm/evidence.py`)
  - Wire `src/model-testing/runs/test_agent.py` (or a new capture script) to
    emit the raw per-step signal the evidence source needs (hidden states,
    tokens, etc.) per generation step
  - Add a labeled dataset of benign/malicious trials for evaluation
  - Implement evaluation metrics (accuracy, false-positive rate, detection
    latency) comparing DDM accumulation vs. single-shot classification
  - Add unit tests under `src/ddm/tests/` for the accumulator/boundary logic
  - Fit/sweep DDM parameters (drift rate, boundary, noise) against the
    labeled dataset once real evidence scores are available
- ~~Load QWEN 1.5B locally (test agent)~~
- ~~Load QWEN 2.5B locally (judge agent)~~


## Run

```python
ollama run qwen2.5:1.5b-instruct
```
Running `test_agent.py` will ask for prompts from the prompt list
- `:list` to see all available prompts
- `:p p###` to input desired prompt

To smoke-test the DDM engine on its own (synthetic evidence, no LLM involved):

```bash
python -m src.ddm.drift
```