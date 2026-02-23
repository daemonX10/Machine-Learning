# Lecture 23 — LLMs and Tools: Teaching LLMs to Use External APIs

---

## 1. Recap: The Generalization Problem

Previous approaches (TALM, PAL, Toolformer) share a critical limitation:

> **Cannot generalize to tools unseen during training.**

If an enterprise has custom APIs, the model must be retrained with annotated examples for each new tool.

**Goal of this lecture**: Teach LLMs to use **any** tool given at test time — the **function calling** paradigm.

---

## 2. Function Calling — The Application Layer Architecture

### 2.1 Overview

```
User Query
    ↓
┌─────────────────────────┐
│   Application Layer     │
│  ┌───────────────────┐  │
│  │ Tool Registration │  │  ← Developer registers available tools (JSON specs)
│  └───────────────────┘  │
│  ┌───────────────────┐  │
│  │       LLM         │  │  ← Receives query + tool descriptions in prompt
│  └───────────────────┘  │
│  ┌───────────────────┐  │
│  │  Tool Executor    │  │  ← Executes the function call, returns result
│  └───────────────────┘  │
└─────────────────────────┘
    ↓
Response to User
```

### 2.2 Tool Definition Format (JSON / OpenAPI Spec)

```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City name"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"]
  }
}
```

### 2.3 Key Insight

By including the **tool description in the prompt**, the model can generalize to tools **never seen during training** — it just needs to understand the JSON schema and fill in the arguments.

---

## 3. Gorilla — API Bench (Patil et al., 2023)

### 3.1 Dataset Construction

| Aspect | Detail |
|--------|--------|
| **Source** | Model cards from HuggingFace Hub, TorchHub, TensorFlow Hub |
| **Size** | ~1,645 API entries |
| **Data generation** | Self-Instruct: GPT-4 generates (instruction, API call) pairs |
| **Total examples** | ~16,450 |
| **Model** | LLaMA 7B fine-tuned on this data |

### 3.2 Self-Instruct Pipeline

1. Collect API documentation (model cards with function signatures, descriptions, parameters).
2. Use GPT-4 to generate user instructions that would require calling each API.
3. Use GPT-4 to generate the corresponding API call with correct arguments.
4. Pair: (instruction + API reference) → (API call).

### 3.3 Critical Design Choice: Retriever-Aware Training

At training time, the API reference document is **included in the input**:

```
Instruction: "I want to classify images of dogs and cats"
API Reference: {full documentation of relevant APIs}
→ Output: torchvision.models.resnet50(pretrained=True)
```

This teaches the model to **read and follow** tool documentation, enabling generalization to **unseen tools** at inference.

### 3.4 Inference Modes

| Mode | How APIs are provided | Performance |
|------|----------------------|-------------|
| **Oracle retriever** | Ground-truth API docs given | Highest accuracy (~90%+) |
| **BM25 retriever** | Top-K API docs from BM25 search | Moderate |
| **No retriever** | Model relies on parametric memory only | Lowest |

### 3.5 Evaluation: AST Matching

Instead of exact string match, use **Abstract Syntax Tree** comparison:

$$\text{AST accuracy} = \mathbb{1}[\text{AST}(\hat{y}) \equiv \text{AST}(y^*)]$$

- Ignores formatting differences (whitespace, argument order).
- Checks structural equivalence of function calls.
- More robust than string matching.

### 3.6 Additional Metrics

| Metric | What it Measures |
|--------|-----------------|
| **Hallucination rate** | % of generated API calls that don't exist |
| **Error rate** | % of calls with wrong arguments/types |

### 3.7 Key Results

- Gorilla (7B) performs comparably to GPT-3.5/GPT-4 when API docs are in the prompt.
- Without retriever: proprietary models hallucinate significantly.
- **Gap between oracle retriever and BM25 retriever is large** → retriever quality is critical bottleneck.

---

## 4. Berkeley Function Calling Leaderboard (BFCL)

- Created by the Gorilla authors (UC Berkeley).
- De facto standard benchmark for evaluating function calling.
- Current version: **V4**.

| Category | Description |
|----------|-------------|
| Single turn | One user message → one function call |
| Multi-turn | Conversation with multiple tool calls |
| Format sensitivity | Robustness to different JSON formats |
| Agentic solutions | Complex multi-step workflows |

---

## 5. ToolAlpaca

### 5.1 Key Difference from Gorilla

- Uses **real-world inspired** tools (not just ML model cards).
- Supports **multi-turn dialogues**.

### 5.2 Data Pipeline

1. **Source**: Public APIs GitHub repo → 1,400 app names + descriptions.
2. **API generation**: LLM generates function documentation and OpenAPI specs for each app.
3. **Dialogue generation** via three specialized prompts:

| Prompt Agent | Role |
|-------------|------|
| **User agent** | Generates user utterances |
| **Assistant agent** | Decides: chat with user or call API + generate arguments |
| **Tool executor agent** | Synthesizes tool response following OpenAPI response schema |

- Agents operate in a **deterministic flow**: User → Assistant → (if API call) Tool Executor → Assistant → User → ...
- Separate prompts prevent single-point prompt failures.

### 5.3 Model

Fine-tuned **Vicuna** model → called **ToolAlpaca**.

### 5.4 Evaluation

| Dimension | What it Measures |
|-----------|-----------------|
| **Procedure** | Correct tool selected? Correct arguments? |
| **Response** | Final response quality |
| **Overall** | Combined procedure + response |

ToolAlpaca (Vicuna) performs comparably to GPT-3.5. Works on both **simulated** and **real-world** APIs not seen during training.

---

## 6. ToolBench / ToolLLaMA

### 6.1 Key Innovation

- Uses **real executable APIs** from RapidAPI Hub (not simulated).
- Dialogues grounded on **multiple tools** per conversation.

### 6.2 Data Pipeline

| Step | Detail |
|------|--------|
| API Source | RapidAPI Hub — 53K APIs crawled, filtered to 16K functional ones |
| Tool Sampling | APIs grouped hierarchically by domain; sample cohesive tool subsets |
| Instruction Generation | Given $n$ tools → LLM generates user query requiring multiple tools |
| Dialogue Generation | Multi-agent simulation with real API execution |

### 6.3 Handling Failures: Depth-First Search (DFS)

Problem with naive sequential generation: API calls may fail → model gets stuck in retry loops → generates failed trajectories.

**DFS Solution**:
1. If a tool call fails, **backtrack** to the last successful state.
2. Try a different tool call or different arguments.
3. Continue until a successful path is found.
4. Use only the **successful trajectory** for training.

### 6.4 Custom API Retriever

- Training data: (user instruction, relevant APIs) pairs from the instruction generation phase.
- Trained a specialized retriever for tool selection (outperforms general-purpose retrievers like BM25 for this task).

---

## 7. APIGen / xLAM (Salesforce)

### 7.1 Focus

- Target: Top performance on **BFCL V1** benchmark.
- Generated **~60K** single-turn function calling examples.
- Source: RapidAPI (filtered to 3,000 high-quality APIs).

### 7.2 Diversity Engineering

| Diversity Dimension | Description |
|--------------------|-------------|
| **Question verbosity** | Crisp vs. verbose user queries with background stories |
| **Misspellings** | Deliberate typos in user input |
| **Indirect requests** | "Should I carry an umbrella?" instead of "What's the weather?" |

### 7.3 BFCL Single-Turn Categories

| Category | Description |
|----------|-------------|
| **Single** | One API call with one invocation |
| **Multiple** | Same API called multiple times with different arguments |
| **Parallel** | Multiple similar APIs — model must pick the right one |
| **Parallel Multiple** | Combination of parallel + multiple |

### 7.4 Multi-Stage Verification Pipeline

| Stage | What it Checks |
|-------|---------------|
| **Format checker** | JSON structure valid? Required keys present? Key names match API spec? |
| **Execution checker** | Does the API call actually execute successfully on RapidAPI? |
| **Semantic checker** | Given input + output (ignoring tools), does the response make sense? |

### 7.5 Generator Model Quality

| Model | Verification Pass Rate |
|-------|----------------------|
| DeepSeek 236B (MoE) | Highest |
| Mixtral 8×7B | ≈ DeepSeek 33B |
| Smaller models | Significantly lower |

### 7.6 Result

xLAM (Mixtral-based, fine-tuned) achieved accuracy close to GPT-4 on BFCL at time of release.

---

## 8. Licensing Concerns

| Dataset/Model | License Issue |
|--------------|---------------|
| Most prior work | Uses GPT-4 for data generation → non-permissive license |
| RapidAPI specs | Not permissively licensed for commercial use |
| **Open solution** | Use Apache 2.0 / MIT licensed models + open-source APIs |

---

## 9. From Function Calling to Agents

### The Gap

Simply giving tools to an LLM is **not enough** for realistic agentic solutions:

- Real business processes require **sequences of API calls** with **dependencies**.
- Example: "Open a bank account" → KYC check → account type selection → approval → creation.
- LLMs can't discover these procedures from tool descriptions alone.

### Granular Sub-tasks of Function Calling

| Sub-task | Description |
|----------|-------------|
| **Tool name detection** | Which API to call? |
| **Slot filling** | What arguments to provide? |
| **Argument validation** | Are the arguments valid? |
| **Call sequencing** | What order to call multiple APIs? |

Training on these **granular tasks** alongside full function calling improves agent performance.

---

## 10. Summary — Evolution of Tool-Use Approaches

| Paper | Year | Tools | Multi-turn | Multi-tool | Key Innovation |
|-------|------|-------|-----------|-----------|----------------|
| **Gorilla / APIBench** | 2023 | ML model cards (1.6K) | No | No | API docs in prompt; AST matching; BFCL benchmark |
| **ToolAlpaca** | 2023 | Real-world inspired (1.4K) | Yes | No | Multi-agent dialogue synthesis; separate prompt agents |
| **ToolBench / ToolLLaMA** | 2023 | Real APIs (16K from RapidAPI) | Yes | Yes | DFS for failed trajectories; custom API retriever |
| **APIGen / xLAM** | 2024 | Real APIs (3K filtered) | No | Yes (parallel) | Diversity engineering; multi-stage verification |

### Key Takeaways

1. **Tool descriptions in the prompt** are the key mechanism enabling generalization to unseen tools.
2. **Retriever quality** is the critical bottleneck — oracle retriever ≫ BM25 retriever.
3. **Synthetic data quality** matters more than quantity — multi-stage verification is essential.
4. **Multi-turn, multi-tool** dialogues are needed for realistic agent applications.
5. **Licensing** is a practical concern when building commercial systems.
