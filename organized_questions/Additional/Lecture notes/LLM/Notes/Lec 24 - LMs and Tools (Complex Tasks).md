# Lecture 24: LMs and Tools — Automating Complex Tasks

## Three Research Directions (Post Mid-2024)

| Direction | Focus | Key Papers |
|-----------|-------|------------|
| High-Fidelity Data Synthesis | Better synthetic training data for tool-calling | ToolFlow, MAGNET |
| RL After SFT | Reinforcement learning to improve tool use | MAGNET (MDPO) |
| Realistic Benchmarks | Reliable agentic evaluation | τ-bench, BFCL V3 |

---

## 1. Multi-Turn vs Multi-Step Function Calling

| Aspect | Multi-Turn | Multi-Step |
|--------|-----------|------------|
| Definition | Multiple user–agent exchanges over time | Multiple tool calls within a single turn |
| Example | User asks about flights → agent clarifies → user provides dates → agent books | Agent searches flights → gets airport code → books ticket (all in one turn) |
| Complexity | Requires dialogue management | Requires planning and chaining |

---

## 2. ToolFlow (ICLR 2025)

### Core Idea
Synthesize multi-step tool-calling data via **graph-based pipeline**.

### Pipeline

1. **Tool Graph Construction**
   - Nodes = tool/API definitions
   - Edges = semantic relatedness between tools
   - Densely connected graph

2. **Plan Generation via Random Walk**
   - Sample a path of $h$ tools from the graph for a dialogue of length $h$
   - The sampled tools form the "function signature path"

3. **Multi-Agent Dialogue Synthesis**
   - Use teacher LLM to generate user utterances, tool calls, and tool responses along the sampled path

### Results
- Improved LLaMA 3.1 8B from **74% → 85%** on BFCL (procedure category)

---

## 3. MAGNET (Google, March 2025)

### Problem
Multi-turn, multi-step function calling with both SFT and DPO.

### Trajectory Notation

$$\tau = (u_1, a_1, t_1, u_2, a_2, t_2, \ldots, u_H, a_H, t_H)$$

- $u_h$ = user utterance at turn $h$
- $a_h$ = agent action (tool call or natural language)
- $t_h$ = tool response

### Data Synthesis Pipeline

#### Step 1: Function Signature Path (Sparse Graph)
- LLM assigns labels to each API
- Randomly sample 30 neighbors per node with same label
- LLM checks relatedness → **sparsely connected** graph (unlike ToolFlow's dense graph)

#### Step 2: Signature Path Enhancement
Three realistic augmentation techniques:

| Technique | Description | Example |
|-----------|-------------|---------|
| **Insert** | Add helper API for implicit argument resolution | `get_airport_symbol_by_city` inserted before `get_flights_by_airport` |
| **Merge** | Combine multiple API calls into single user turn | "Book a flight" implies search + book |
| **Split** | Split one function call across multiple turns | User gives partial info, agent asks for rest |

#### Step 3: Back Translation
Given function signature $fs_h$, generate the corresponding user utterance $u_h$.

#### Step 4: Forward Translation
Instantiate function signature with actual parameter values using:
- User utterance
- Previous tool responses

#### Step 5: Fill Agent Actions
- If function is fully instantiated → action = tool call
- If function is incomplete → action = natural language (ask for missing params)
- If final step → action = natural language response summarizing results

### Negative Trajectory Generation
1. Feed partial trajectories to the **student LLM** (smaller model)
2. Collect error statistics (wrong tool, hallucinated answer, missing arguments)
3. **Systematically inject** those errors to create negative trajectories

### MDPO Loss (Multi-Turn DPO)

Key challenge: winner and loser trajectories have **different lengths** and **different state prefixes**.

$$\mathcal{L} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{MDPO}}$$

### Results

| Model | BFCL V3 Multi-Turn |
|-------|-------------------|
| Teacher (GPT-4 level) | 20.75 |
| Student (Qwen 2.5 7B) + SFT | > 20.75 |
| Student (Qwen 2.5 7B) + SFT + MDPO | **best** |
| Student (Qwen 2.5 14B) + SFT + MDPO | 5 → 37 |

**Key Takeaway**: Even a weak teacher can produce a stronger student with good data synthesis + RL.

### Caveats
- MAGNET is **overfit for BFCL**: uses same APIs as BFCL, covers BFCL-specific error patterns
- Results may not generalize to other benchmarks

---

## 4. Realistic Evaluation: τ-bench

### Problem: Accuracy ≠ Reliability
- 95% accuracy sounds good, but **5% failure on write operations** = exploitable
- Users can try 5 different ways to hack the system
- **Read operations** (search): failure is tolerable
- **Write operations** (booking, cancellation): failure has real cost

### τ-bench Design

| Feature | Description |
|---------|-------------|
| **Executable tools** | APIs operate on real domain databases; write ops change DB state |
| **Domain rules** | Natural language policies (e.g., "no cancellation within 24 hours of departure") |
| **User simulator** | GPT-4 based; given persona + instructions; responds dynamically to agent errors |
| **Dynamic evaluation** | If agent makes mistake on turn 1, user simulator adapts |

### Domains

| Domain | Tasks | Users | Tools | Key Features |
|--------|-------|-------|-------|--------------|
| Retail | 115 | — | Read + Write | Cancel/modify orders, returns |
| Airline | 50 | 500 | Read + Write | Complex policies (membership tiers, time limits) |

### Evaluation: Pass@K vs Pass^K

| Metric | Formula | Measures |
|--------|---------|----------|
| **Pass@K** | $P(\text{at least 1 correct in } k \text{ samples from } n)$ | Feasibility (can it ever succeed?) |
| **Pass^K** (phat@K) | $P(\text{all } k \text{ runs correct})$ | **Reliability** (is it consistently correct?) |

$$\text{Pass@K}: \quad 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

### Results Insight
- GPT-4o: **60%** on Pass@1, drops to **~40%** on Pass^K
- Reliability gap exposes why agentic solutions aren't production-ready for write operations

### τ²-bench Extension
- **Both user and agent** have access to tools
- Example: Telecom troubleshooting where customer must also perform actions (toggle settings)

---

## 5. Summary of Open Problems

| Area | Status |
|------|--------|
| Synthetic data generation | Still far from realistic; benchmark-specific |
| RL for tool calling | ~1% improvement; reward design, negative trajectory sampling unsolved |
| Evaluation | Domain-specific benchmarks needed; reliability metrics (Pass^K) emerging |
| Convergence | All three directions will eventually merge into domain-specific evaluation frameworks |
