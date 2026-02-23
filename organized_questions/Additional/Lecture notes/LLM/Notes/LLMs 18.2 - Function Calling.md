# LLMs 18.2 — Function Calling

## Overview

This lecture covers **function calling** — enabling LLMs to invoke external APIs/functions with structured arguments during conversations. Covers data generation, training, evaluation, and key papers from Gorilla to xLAM.

---

## Motivation: Chatbot with Transactional Capabilities

**Example**: Chatbot for a university (IIT Delhi) that can both answer FAQs (RAG) and perform transactions (hostel booking, fee payment, course registration).

**Requirements**:
1. Determine **when** to call a function vs. respond conversationally
2. Select the **correct function** from available APIs
3. Extract the **right parameters** from user utterances
4. Handle **missing parameters** by asking follow-up questions
5. **Use API response** to generate natural language answers

---

## Function Calling Architecture

### 4-Step Pipeline

```
1. Define Tools (JSON schema) → 2. LLM outputs function name + args
→ 3. Application executes function → 4. LLM generates NL response from result
```

### Tool Definition Format (JSON Schema)

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

### Three Core Training Tasks

| Task | Description |
|---|---|
| **Tool Selection** | Given user query + available tools → pick correct tool |
| **Tool Output Utilization** | Given API response → generate natural language answer |
| **Missing Parameter Handling** | Detect missing required params → ask user for them |

---

## 1. Gorilla (UC Berkeley, 2023)

**Model**: Fine-tuned LLaMA-7B on API documentation

### API Bench Dataset

| Source | # Model Cards |
|---|---|
| HuggingFace Hub | ~900 |
| TorchHub | ~200 |
| TensorHub | ~500 |
| **Total** | **~1,645** |

### Data Generation
- **Self-Instruct**: Generate (instruction, API call) pairs from model card documentation using GPT-4
- For each API: generate user instructions that would require calling that API

### Retrieval-Augmented Training
- At training time, optionally include API documentation in the prompt
- Prevents model from **memorizing** specific API signatures
- Enables generalization when API details change

### Evaluation: AST-Based Matching
- **Problem with string matching**: functionally equivalent API calls can have different string representations
- **Solution**: Parse generated API calls into **Abstract Syntax Trees (AST)** and compare structurally
- Measure **hallucination rate** (inventing non-existent APIs) and **wrong-library errors**

### Berkeley Function Calling Leaderboard (BFCL)
- Evolved from v0 → v1 → v2 → v3
- Standard benchmark for comparing function calling models
- Tracks GPT, Claude, open-source models

### Limitations
- Low diversity of APIs (ML model hubs only)
- Single-turn only (no multi-turn dialogue)
- Cannot handle real conversational function calling

---

## 2. ToolAlpaca (Mid-2023)

**Model**: Fine-tuned Vicuna (instead of LLaMA)

### Dataset: Real-World Inspired APIs
- Scraped **tool names + descriptions** from the web
- **Synthesized** everything else (detailed specs, function names, parameters)
- ~400 real-world inspired APIs spanning many categories

### Key Improvement: Multi-Turn Dialogues
- Supports **question generation** (asking for missing params)
- Supports **response generation** (NL answers from API results)
- Simulates end-to-end conversational function calling

### Example Flow
```
User: "What are the holidays in Japan next year?"
Agent: "Could you specify which year?" (missing parameter)
User: "2024"
Agent: → calls get_holidays(country="Japan", year=2024)
Agent: "Here are the holidays in Japan for 2024: ..."
```

### Evaluation on Unseen Tools
- Tested on ~10-12 real-world APIs not seen during training
- GPT-3.5 still ahead, but ToolAlpaca-7B/13B closing the gap

---

## 3. ToolLLM / ToolBench (Mid-2023)

**Key Innovation**: Used **real REST APIs** from Rapid API Hub (~16,000 APIs)

### Dataset Creation Challenges
- Real APIs have: empty descriptions, meaningless parameter names (`var1`, `var2`), broken endpoints
- Single-path dialogue generation failed due to API quality issues

### Depth-First Tree Search (DFTSS) for Data Generation
1. Generate first user instruction from subset of APIs
2. Generate agent response + tool calls
3. If API execution **fails** → backtrack and try different path
4. At each node, use ChatGPT to decide: continue / backtrack / terminate
5. Find at least one successful trajectory per example

### Byproduct: API Retriever
- Trained a specialized retriever for API selection
- Better than generic retrievers (e.g., BERT-based) for finding relevant APIs from large catalogs

### Multi-Tool Support
- Explored **chaining multiple APIs** within a single conversation
- Some queries require calling 2-3 different tools sequentially

### Results
- ToolLLaMA (fine-tuned LLaMA-7B) + their retriever competitive with Gorilla on API Bench
- With Oracle retriever: Gorilla slightly better (overfit to specific APIs)
- Key insight: generalizes to out-of-domain APIs

### Limitations
- Dialogue quality still poor due to messy real-world APIs
- Model may succeed on metrics but conversations look unnatural

---

## 4. xLAM / APIGen (Salesforce, 2024)

**Key Innovation**: High-quality data generation with permissive licensing (Apache) — all data and models commercially usable.

### Data Pipeline

```
16K Rapid APIs → Filter to 3K clean APIs → Seed QA → Generate data
→ 3-Stage Verification → Fine-tune models
```

### Three-Stage Verification

| Stage | Method | Purpose |
|---|---|---|
| **Format Check** | Rule-based JSON validation | Reject malformed outputs, wrong enums, type mismatches |
| **Execution Check** | Actually call the API | Reject calls that fail or return errors |
| **Semantic Check** | LLM-as-judge | Reject incoherent/unnatural dialogues |

### Data Generation Analysis

| Generator Model | 40K samples generated | Usable after filters |
|---|---|---|
| DeepSeek-V2 Chat | 40K | ~33K (best quality) |
| Mixtral-7B | 40K | Lower (formatting issues) |
| 33B model | 40K | Lower (formatting issues) |

**Key Finding**: DeepSeek-V2 Chat produced the highest quality synthetic data with minimal semantic failures.

### Models Trained

| Model | Size | Architecture |
|---|---|---|
| xLAM-1B | 1B | — |
| xLAM-7B | 7B | Based on DeepSeek Coder |
| xLAM-8x22B | 8×22B | MoE (Mixture of Experts) |

### Results
- xLAM-8x22B reached **top positions** on Berkeley Function Calling Leaderboard
- Even xLAM-1B and xLAM-7B outperformed many larger models
- Follows LIMA principle: small number of high-quality examples > large number of moderate quality

### Data Diversity
- Single-tool calls
- Multi-tool calls (chaining)
- **Parallel tool calls** (multiple APIs from same category — fine-grained selection)

---

## 5. IBM Granite Function Calling (2024)

### Motivation
- Previous models required LLM to handle **all** of function calling end-to-end
- Enterprise chatbots need **granular control**: use LLM for specific subtasks only

### Approach
- Repurposed existing **task-oriented dialogue datasets** (human-created, from early 2000s)
- Synthesized additional data using open models (Mixtral, Flan-XXL)
- Created **granular tasks**: tool selection only, parameter extraction only, response generation only

### Decoupled Function Calling Tasks

| Task | Description |
|---|---|
| Intent/Tool Selection | Which API to call? |
| Slot/Parameter Filling | What parameter values? |
| Response Generation | NL response from API output |
| Full Pipeline | End-to-end function calling |

### Results
- Granite-20B: competitive on BFCL leaderboard at time of release
- Better for **application developers** who need modular control

---

## Evolution of Function Calling Datasets

| Dataset | Year | # APIs | Source | Dialogue | License |
|---|---|---|---|---|---|
| API Bench | 2023 | ~1.6K | ML model hubs | Single-turn | Academic |
| ToolAlpaca | 2023 | ~400 | Web-inspired (synthetic specs) | Multi-turn | Academic |
| ToolBench | 2023 | ~16K | Rapid API (real) | Multi-turn | Academic |
| **APIGen/xLAM** | **2024** | **3K clean** | **Rapid API (filtered)** | **Multi-turn** | **Apache (commercial)** |
| Granite data | 2024 | Mixed | TOD datasets + synthetic | Multi-turn | Apache |

---

## Key Takeaways

1. **AST-based evaluation** > string matching for function calling accuracy
2. **Data quality >> data quantity** — APIGen's filtered 3K APIs beat ToolBench's noisy 16K
3. **Retrieval-augmented function calling**: combine API retriever + function calling model
4. **Licensing matters**: GPT-4 synthesized data cannot be used commercially
5. **Granular task decomposition** enables better enterprise integration
6. Modern open-source models (xLAM-7B) approach GPT-4 on function calling benchmarks
