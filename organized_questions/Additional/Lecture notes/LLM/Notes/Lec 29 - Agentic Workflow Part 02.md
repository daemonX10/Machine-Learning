# Lecture 29: Agentic Workflow Part 02 — Memory Management & RL Fine-Tuning for Agents

---

## Part 1: MemGPT — Memory Management in AI Agents

### The Context Window Problem

LLMs have a **fixed context window** — once exceeded, information is lost. MemGPT addresses this by introducing an **OS-inspired virtual memory system** for LLM agents.

**Core Analogy**: Just as an OS manages virtual memory (paging between RAM and disk), MemGPT pages information between a limited LLM context and external storage.

---

### MemGPT Architecture

#### Context Window Partitions

| Component | Type | Description |
|-----------|------|-------------|
| **System Instructions** | Read-only | MemGPT persona, rules, tool descriptions |
| **Working Context** | Read/Write | Scratchpad for facts, user preferences, plans |
| **FIFO Queue** | Rolling buffer | Recent messages with **recursive summarization** |

- When the FIFO queue overflows, older messages are **summarized** and the summary replaces the raw messages
- Summaries can themselves be summarized (recursive)

#### External Storage

| Storage | Purpose | Contents |
|---------|---------|----------|
| **Recall Storage** | Message history database | Searchable past conversations |
| **Archival Storage** | Long-term knowledge | Documents, cross-session learnings |

---

### MemGPT Tool Set

| Tool | Function |
|------|----------|
| `send_message` | Communicate with user |
| `working_context_append` | Add facts to working memory |
| `working_context_replace` | Update working memory content |
| `recall_search` | Search past message history |
| `archival_memory_insert` | Store info in long-term archive |
| `archival_memory_search` | Query archival storage (paginated, top-10 per page) |

**Key Design**: The LLM decides autonomously when to use each tool — e.g., whether to answer immediately or scroll to the next page of archival search results.

---

### MemGPT Experiments & Results

#### Experiment 1: Conversational Consistency (Multi-Session Chat)

- Dataset: Multi-session chat conversations testing persona consistency and engagement
- **Metrics**:
  - **Sim-1**: Max cosine similarity between generated utterance and any persona fact
  - **Sim-3**: Average of top-3 similarities (breadth of engagement)
  - **Sim-H**: Similarity relative to human baseline

| Model | Consistency | Engagement |
|-------|------------|------------|
| GPT-3.5 baseline | ~30-40% | Low |
| GPT-4 + MemGPT | ~92-93% | High (Sim-1, Sim-3 > humans) |

#### Experiment 2: Document QA (Natural Questions + Wikipedia)

- Wikipedia pages stored in archival storage
- MemGPT scrolls through paginated results (10 docs/page) to find answers

| Setup | Accuracy |
|-------|----------|
| GPT-4 + 1 doc | ~40% |
| GPT-4 + 200 docs (stuffed context) | **Worse** than 1 doc (distraction) |
| MemGPT + GPT-4 | **Constant high** (no context length sensitivity) |

**Key Insight**: Larger context windows ≠ better performance. MemGPT's structured memory management outperforms naive context stuffing.

---

## Part 2: SweetRL — RL Fine-Tuning Small LMs for Agentic Tasks

### Motivation

Can small LLMs (1B–8B params) match proprietary models (GPT-4o) on **collaborative, multi-turn agentic tasks**?

---

### CollabBench Benchmark

A benchmark for **multi-turn human-agent collaboration** with two tasks:

#### Task 1: Backend Programming
- Agent collaborates with user to write a Python function
- User provides **underspecified** high-level description + function signature
- User simulator (Llama 3 70B) has access to the ground-truth code
- Agent (Llama 3.1 8B) must ask clarifying questions before generating code
- Evaluation: Unit test pass rate (10 tests per task)
- **10K training tasks, 1K test tasks, 15K trajectories**

#### Task 2: Frontend Web Design
- Agent generates HTML matching a reference web page
- User simulator (Qwen2-VL 72B) has access to the reference web page image
- Evaluation: Image similarity between generated and reference pages
- **10K training tasks, 500 test tasks, 6K trajectories**

| Model | Single-Turn Success | Multi-Turn (Collaborative) Success |
|-------|--------------------|------------------------------------|
| GPT-4o | ~25-30% | ~40% |
| Llama 3.1 8B (baseline) | ~15% | ~22% |

**Takeaway**: Collaboration doubles/triples performance → benchmark validates the need for multi-turn interaction.

---

### Training Pipeline: From SFT to SweetRL

#### Step 1: Rejection Fine-Tuning (RFT)
- Use **only positive trajectories** (all unit tests passed)
- Standard SFT on successful conversations
- Result: +6 points success rate improvement

#### Step 2: Multi-Turn DPO (MDPO)
- Use **both positive and negative trajectories**
- Modified DPO loss over entire trajectory (not single turn)
- Result: +6 more points (total ~34%)
- **Limitation**: Poor **credit assignment** — penalizes entire trajectory equally

#### Step 3: SweetRL (Stepwise Evaluation with Training-time Information)

**Key Innovation**: Uses **additional training-time information** (unit tests / reference images) to compute **stepwise rewards**.

##### Architecture: Actor-Critic Model

```
┌─────────────────────────────────┐
│         Critic Model            │
│  Inputs: O_t (history) +        │
│          A_t (action) +         │
│          C (hidden info:        │
│            unit tests/images)   │
│  Output: Step-wise reward       │
└──────────┬──────────────────────┘
           │ reward signal
           ▼
┌─────────────────────────────────┐
│      Policy (Actor) Model       │
│  Fine-tuned via RL (PPO/DPO)    │
│  Critic discarded at inference  │
└─────────────────────────────────┘
```

##### POMDP Formulation
- **Observable state** $O_t$: Conversation history (visible to agent)
- **Hidden state** $C$: Unit tests or reference web page (only available during training)
- **Action** $A_t$: Natural language response (question or code)
- **Advantage function**: $A(O_t, A_t, C) = Q^\pi(O_t, A_t, C) - V^\pi(O_t, C)$

##### Training Procedure
1. Train critic model using positive/negative trajectories + hidden info $C$
2. For any task, generate 16 candidate actions at each turn
3. Compute advantage for each using critic
4. Select top-50% as "chosen", bottom as "rejected"
5. Apply DPO/PPO with stepwise rewards
6. **Discard critic at inference** — only the trained policy (actor) is needed

---

### Results Comparison

| Method | Additional Info Needed | Credit Assignment | Success Rate (Backend) | Win Rate (Frontend) |
|--------|----------------------|-------------------|----------------------|---------------------|
| Rejection FT | No | N/A | ~28% | — |
| Multi-Turn DPO | No | Trajectory-level | ~34% | — |
| **SweetRL** | **Yes (train-time only)** | **Step-level** | **~40%** (= GPT-4o) | **~48.2%** (≈ GPT-4o) |

---

### Key Takeaways

1. **MemGPT** solves context overflow via OS-inspired virtual memory (paging, tiered storage)
2. **SweetRL** enables small LLMs to match GPT-4o on collaborative tasks via stepwise RL with training-time information
3. **MDPO vs SweetRL**: MDPO is more general (no extra info needed); SweetRL gives better credit assignment when training-time signals exist
4. Current success rates (~40%) are insufficient for production — RL fine-tuning for specific agent tasks is an **active research area**
5. Multi-agent RL (master + worker agents) is the next frontier
