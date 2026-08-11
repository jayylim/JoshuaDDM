# Project Idea

The overall research question stands as:
> Can a drift diffusion model (DDM) act as a reliable active monitoring system and meta-controller for LLMs?

## Goal

The project aims to build an evidence-based **judge** or **active monitoring system** for an LLM's behaviour, focused initially on a binary question:

- **Malicious** 
- **Benign / continue**

The judge should be usable regardless of how the LLM is deployed: ordinary chat, tool use, API-driven agents, or multi-agent systems.

The proposed mechanism is a **drift diffusion model (DDM)**. Instead of making a decision from one observation, the DDM accumulates evidence over a sequence of steps until there is enough evidence to make a binary decision.

The simplest initial research question is:

> Does sequentially accumulating evidence about an LLM's behaviour produce a more accurate or earlier maliciousness judgment than making a single-shot classification?

A white-box model such as Qwen2.5-1.5B-Instruct is useful because its internal activations can be inspected directly.

---

## Recommended Time Scale

The DDM should use **discrete decision steps**, rather than wall-clock time.

A decision step is one meaningful new piece of evidence.

For an ordinary autoregressive LLM, the simplest starting point is:

- one evidence update per generated token (at the the targetted layers)

For more complex agent systems, a later version could instead update on meaningful events such as:

- tool calls
- API requests
- retrieved information
- messages between agents
- actions that modify an external environment

The main implication is that the DDM models **how evidence develops over the behavioural trajectory**, not how many seconds the computation takes.

For experimental evaluation, one complete prompt-response interaction (with underlying computations) can be treated as a **trial**.

---

## Pieces That Need to Be Designed

### 1. Target Label

The system needs a clear definition of what counts as malicious behaviour.

Initially, every trial should have a binary ground-truth label:

- malicious
- benign

Care should be taken to distinguish genuinely malicious intent from prompts that merely contain dangerous vocabulary.

---

### 2. Evidence Signals

At each decision step, the system needs a scalar score (to be plugged into the DDM) representing evidence for maliciousness.

Possible sources include:

- internal hidden-state activations
- generated text
- proposed tool calls
- API arguments
- external actions

A simple fersion could include a classifier trained on internal activations, such as a **probe** trained to determine whether some property is detectable from an internal model representation. For example, it could output:

`P(malicious | current hidden state)`

This probability can be converted into the evidence supplied to the DDM.

---

### 3. Evidence Accumulator

The DDM maintains a running internal state representing accumulated evidence. 

Conceptually:

`new accumulator = old accumulator + new evidence`

Repeated evidence pointing toward malicious behaviour moves the accumulator toward the malicious decision boundary.

This is the core difference from a normal classifier: the decision depends on the trajectory of evidence rather than one isolated prediction.

---

### 4. Decision Boundary

The DDM needs a threshold at which accumulated evidence is considered sufficient to classify the trajectory as malicious.

A low threshold produces:

- earlier interventions
- more false positives

A high threshold produces:

- fewer false positives
- later or missed detections

How this decision boundary is determined and evaluated (literature-based, AI vs Human decision boundaries, etc.) must be explicitly established. 

This parameter is also theoretically optimisable through machine learning/testing.
---

### 5. Core DDM Parameters

A minimal implementation should eventually represent the following concepts:

- **Drift rate:** the average direction and speed at which evidence moves toward a decision.
- **Boundary:** how much accumulated evidence is required before making the decision.
- **Starting bias:** whether the system begins neutral or already biased toward one outcome.
- **Noise / variability:** uncertainty in the evidence samples.
- **Non-decision time:** fixed delay not caused by evidence accumulation, mainly relevant when comparing the model with human reaction-time data.

The first prototype does not need every parameter. It can begin with:

- one scalar evidence value per step/type of computation
- one accumulator
- one maliciousness threshold
- one scaling or drift parameter

---

## Initial Experimental Direction

The first experiment should remain deliberately simple.

1. Run Qwen2.5-1.5B-Instruct on labelled benign and malicious prompts.
2. Record internal representations during generation.
3. Train a simple probe to estimate maliciousness from those representations.
4. Produce one evidence score per generated token.
5. Accumulate those scores using a simple DDM-style process.
6. Stop or classify the trajectory when the maliciousness boundary is reached.
7. Compare this against a single-shot probe that does not accumulate evidence.

The central evaluation should ask whether accumulation improves:

- classification accuracy
- false-positive rate
- detection latency
- robustness to ambiguous examples

---

## Possible Future Lines of Investigation

### Layer-Wise Evidence

Instead of accumulating evidence across generated tokens, examine how maliciousness information develops across transformer layers.

This asks a different question:

> At what stage of the model's internal computation does the relevant information become detectable?

This is useful for understanding model mechanisms, but token-level accumulation is the simpler initial DDM experiment.

---

### Action-Level Evidence

For agentic systems, use actions rather than tokens as the main evidence steps.

Examples:

- calling an API
- reading a file
- sending a message
- executing a tool

The implication is that consequential actions may be more meaningful evidence units than individual tokens.

---

### Multi-Agent Monitoring

In a multi-agent system, evidence could be accumulated across the behaviour of several agents.

This would investigate whether apparently benign individual actions combine into a harmful system-level trajectory.

---

### Multiple Evidence Sources

A later judge could combine:

- internal activations
- generated language
- tool-use behaviour
- environmental consequences

The DDM would then act as an evidence-fusion mechanism rather than relying on one classifier.

---

### Human Behaviour Validation

Because DDMs originate from models of human decision-making, the system could eventually be compared with human judgments.

Human participants could judge the same trajectories and provide:

- malicious / benign decisions
- reaction times
- confidence ratings

The project could then test whether similar DDM parameters explain both human and machine judgments.

This would help determine whether the proposed judge is merely an engineering classifier or whether its evidence-accumulation dynamics resemble experimentally observed human decision behaviour.

---

### Mechanistic Investigation

If internal activation signals reliably predict malicious behaviour, later work can investigate whether those signals are merely correlated with the behaviour or causally involved in producing it.

This would require interventions on internal model representations and is a later-stage mechanistic interpretability question.

---

## Core Research Progression

The project can be organized around three increasingly strong questions:

1. **Is malicious behaviour detectable from the model's internal state?**
2. **Does evidence for malicious behaviour develop systematically over a trajectory?**
3. **Does a DDM-style accumulation process improve the final intervention decision?**

This progression keeps the project experimentally testable while leaving room for later work on agents, multi-agent systems, human validation, and mechanistic interpretability.
