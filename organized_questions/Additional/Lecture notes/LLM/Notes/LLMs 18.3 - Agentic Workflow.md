# LLMs 18.3 — Agentic Workflows

## Overview

This lecture covers **agentic workflows** — iterative, multi-step LLM systems that reason, act, observe, and self-correct. Covers ReAct, Self-Refine, Reflexion, Plan-and-Execute (ReWOO), HuggingGPT, and multi-agent systems.

---

## Why Agentic Workflows?

### Andrew Ng's HumanEval Benchmark Example

| Setup | HumanEval Accuracy |
|---|---|
| GPT-3.5 (zero-shot) | 48.1% |
| GPT-4 (zero-shot) | 67.0% |
| **GPT-3.5 + Agentic workflow** | **> GPT-4 zero-shot** |

**Key Insight**: A weaker model with an agentic loop can surpass a stronger model used in a single pass.

### Real-World Applications
- **SWE-bench**: Automated GitHub issue fixing
- **Cloud operations**: Automated incident response
- **Code generation**: Iterative coding with testing

---

## 1. ReAct (Reasoning + Acting)

**Paper**: ICLR 2023

### Core Idea
Combine **Chain-of-Thought reasoning** with **action execution** in an interleaved loop:

```
Thought → Action → Observation → Thought → Action → Observation → ... → Answer
```

### Comparison with Other Approaches

| Approach | Reasoning | Action | Problem |
|---|---|---|---|
| **Standard prompting** | No | No | Limited capability |
| **Chain-of-Thought (CoT)** | Yes | No | Hallucinates plausibly; no grounding |
| **Act-only** | No | Yes | Gets stuck without evidence; no strategic planning |
| **ReAct** | **Yes** | **Yes** | Best of both; grounded reasoning |

### HotpotQA Results (Multi-hop QA)

| Method | Accuracy |
|---|---|
| CoT (Self-Consistency) | Baseline |
| Act-only | Lower |
| ReAct | Competitive |
| **CoT-SC → ReAct** (fallback) | **Best** |
| **ReAct → CoT-SC** (fallback) | **Best** |

### Failure Mode Analysis

| Method | Failure Type |
|---|---|
| CoT | **Hallucinates plausibly** — generates confident but wrong reasoning |
| ReAct | **Gets stuck** — when search returns no useful results, cannot recover |

### Combination Strategies
- **CoT-SC → ReAct**: Run CoT with self-consistency first; if marginalized distribution is unclear → fall back to ReAct
- **ReAct → CoT-SC**: Run ReAct for 5-6 steps; if no answer → fall back to CoT self-consistency

### Fine-Tuning Insight
A **smaller model fine-tuned** with ReAct trajectories can surpass a 540B model using any prompting technique — no human annotation needed if tools provide ground truth.

---

## 2. SWE-Agent (Software Engineering Agent)

**Application**: ReAct for automated software engineering (GitHub issue resolution)

### Environment
- Access to repository files
- Code editor (edit specific lines)
- Code execution / testing
- GitHub issue descriptions

### Key Design Decisions
1. **Curated action space**: Limited set of well-defined commands instead of full bash
   - `open <filename>` — opens file with context window
   - `scroll_up` / `scroll_down` — navigate within file
   - `find_file` / `search_dir` — localize relevant code
   - `edit <start>:<end>` — modify specific line range
2. **Context management**: Show limited file context to keep prompt tractable

### Results
| Method | SWE-bench Resolution Rate |
|---|---|
| RAG-based | ~10% |
| **SWE-Agent (ReAct)** | **~20%** |

### Contributions
1. Use ReAct framework for software engineering
2. **Hand-engineer task-specific tools** rather than giving open-ended shell access
3. Constraining the action space improves LLM performance

---

## 3. Self-Refine (2023)

### Core Idea
**Iteratively improve** LLM output using the **same LLM** as both generator and critic.

```
Input → Generate Output → Feedback (same LLM) → Refine Output → Feedback → ... → Final Output
```

### Key Properties
- **Complementary** to any base prompting strategy (CoT, ReAct, etc.)
- Does not replace the base method — wraps around it
- **Discrimination is easier than generation**: LLMs evaluate better than they generate

### When It Works Best
- Evaluation criteria is clear and expressible in natural language
- Feedback can be specific (e.g., "improve time complexity", "add more detail about X")
- Multi-dimensional evaluation (coherence, relevance, engagement scored separately)

### Results

| Task | Self-Refine Impact |
|---|---|
| Dialogue response generation | Significant improvement |
| Code optimization | Large boost (GPT-3.5 and GPT-4) |
| Math reasoning | Minimal change (LLM already at its limit) |

### Why Self-Refine Works
1. **Discrimination < Generation** in difficulty — easier to spot errors than avoid them
2. **Focused feedback** on specific failure modes > general instructions
3. **Multi-dimensional scoring** highlights exactly where to improve
4. Maintains or improves performance — rarely hurts

---

## 4. Reflexion (2023)

### Core Idea
Extension of Self-Refine that **decouples evaluation from feedback** and uses **external tools** for evaluation.

```
Input → LLM generates output → Tool/Executor evaluates → LLM reflects on evaluation
→ Generates improved output → Tool evaluates → ... → Final Output
```

### Key Difference from Self-Refine

| Aspect | Self-Refine | Reflexion |
|---|---|---|
| Evaluator | Same LLM | External tools (test cases, heuristics, binary rewards) |
| Feedback generator | Same LLM (coupled) | LLM reflects on tool output (decoupled) |
| Memory | No memory of past failures | **Stores past failure experiences** |

### Memory of Past Failures
- When the agent fails, it stores **what went wrong and why**
- In subsequent attempts, it avoids the same failing strategies
- Critical for **recovering from plateaus** where vanilla ReAct gets stuck

### Example (HotpotQA)
1. **Trial 1**: Search for "Allo Allo" → no useful results → wrong answer
2. **Reflection**: "I searched the wrong title. I should search for individual actors instead."
3. **Trial 2**: Search for actor "Sam Kelly" → finds character name → correct answer

### Results (Coding Tasks)

| Task | Reflexion vs Baseline |
|---|---|
| HumanEval (Python) | Significant improvement |
| HumanEval (Rust) | Improvement |
| MBPP | Better in most cases |

### Can be Combined with ReAct
- ReAct handles step-by-step reasoning and action
- Reflexion adds inter-episode learning from failures

---

## 5. ReWOO (Plan-and-Execute)

### Problem with ReAct
- As iterations grow, **context window fills up** with logs
- Every thought-action-observation cycle appends to the prompt
- Long trajectories become **expensive and error-prone**

### ReWOO Approach
1. **Plan phase**: Create a complete plan upfront (identify all steps and tools needed)
2. **Execute phase**: Run each step independently (possibly in parallel)
3. **Synthesize phase**: Combine all outputs into final response

### Comparison

| Aspect | ReAct | ReWOO |
|---|---|---|
| Planning | Step-by-step (interleaved) | Upfront (all at once) |
| Context growth | Linear with steps | Bounded per step |
| Parallelism | Sequential only | Steps can run in parallel |
| Error recovery | Built-in (observe → adjust) | Harder (if step 2 fails, steps 3-4 outputs may be useless) |
| Token cost | High | Lower |

### Practical Approach
- Use **ReWOO for planning** + **ReAct within each worker** for error recovery
- Train a dedicated **planner LLM** (e.g., LLaMA-8B for planning only)

---

## 6. HuggingGPT (Task Planning with Hugging Face Models)

### Idea
- Use LLM as a **planner** that orchestrates multiple specialized models from HuggingFace
- Each model is a "tool": image generation, pose detection, captioning, TTS, etc.

### Example Task
> "Generate an image of a girl reading a book with the same pose as a boy in example.jpg, then describe the new image with your voice."

### Execution Plan (ReWOO-style)
1. **Pose Detection** model → extract boy's pose from image
2. **Pose-to-Image** model → generate girl reading with same pose
3. **Image Captioning** model → describe generated image
4. **Text-to-Speech** model → convert caption to audio

### Key Feature
- Creates a **dependency graph** among tools
- Independent steps can run in parallel
- LLM only plans; specialized models execute

---

## 7. Multi-Agent Systems

### Extension from Single Agent
- Instead of one LLM with multiple prompts, create **separate agents with distinct personas**
- Each agent has its own **system prompt, tools, and specialization**

### Examples

| Agent | Role |
|---|---|
| Developer Agent | Write code |
| Tester Agent | Write and run tests |
| Reviewer Agent | Review code quality |
| Search Agent | Gather information |
| Planner Agent | Create high-level plans |

### Benefits
- **Focused prompts**: each agent handles fewer instructions → better adherence  
- **Specialized**: each agent fine-tuned for its role
- **Communication** between agents improves iteratively

### Status (as of lecture)
- Shows significant potential
- Still a long way to go for robust multi-agent coordination
- Trending research direction

---

## Framework Comparison

| Framework | Type | Key Mechanism | Combinable? |
|---|---|---|---|
| **ReAct** | Reasoning + Acting | Thought-Action-Observation loop | — |
| **Self-Refine** | Iterative refinement | Same LLM as generator + critic | + ReAct |
| **Reflexion** | Iterative refinement + tools | External evaluator + failure memory | + ReAct |
| **ReWOO** | Plan-and-Execute | Upfront planning, parallel execution | + ReAct (per step) |
| **HuggingGPT** | Multi-model orchestration | LLM as planner, models as tools | + ReWOO |
| **Multi-Agent** | Collaborative agents | Persona-based specialization | + All |

---

## Key Takeaways

1. **Agentic workflows > single-pass**: Even weaker models with iteration beat stronger models without
2. **ReAct** combines reasoning and action but can get stuck
3. **Self-Refine** is complementary — it wraps around any base strategy
4. **Reflexion** adds memory of past failures for recovery
5. **ReWOO** reduces token cost by separating planning from execution
6. **Discrimination is easier than generation** — the core insight behind all refinement approaches
7. **Constrain the action space** for better LLM performance (SWE-Agent lesson)
8. **Multi-agent** systems decompose complex tasks but are still maturing
