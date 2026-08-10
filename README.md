# JoshuaDDM
**Drift Diffusion Model for AI Judging**

I am trying to understand how 

### Project Structure
---
 `outputs` saves: 
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
- Theoretical DDM Architecture
- Develop Algorithm for DDM based on LLM Outputs
- 
- ~~Load QWEN 1.5B locally (test agent)~~
- ~~Load QWEN 2.5B locally (judge agent)~~


## Run

```python
ollama run qwen2.5:1.5b-instruct
```
Running `test_agent.py` will ask for prompts from the prompt list
- `:list` to see all available prompts
- `:p p###` to input desired prompt