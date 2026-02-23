# Lecture 30: Agentic Workflow Part 03 — Agent Protocols (MCP, A2A, agents.json)

---

## Overview: The Need for Protocols

As the agentic ecosystem grows (LangGraph, CrewAI, AutoGen, OpenAI Agent Kit, etc.), **interoperability** becomes critical. Protocols standardize how agents connect to tools, data, and other agents.

### Protocol Development Stages

| Stage | Description |
|-------|-------------|
| **Concept** | Idea proposed |
| **Drafting** | Specification being written |
| **Landing** | Published, early adopters testing |
| **Factual Standard** | Widely adopted, de facto standard |

### Two Protocol Categories

| Category | Purpose | Example |
|----------|---------|---------|
| **Context-oriented** | Connect agents to tools/data/resources | MCP |
| **Inter-agent communication** | Agent-to-agent interaction | A2A, agents.json |

---

## 1. MCP — Model Context Protocol (Anthropic, Nov 2024)

### Core Concept  
**Analogy**: MCP is the "USB" for AI applications — a universal connector between AI apps and external tools/data.

### Architecture: Four Actors

```
┌──────────┐
│   User   │
└────┬─────┘
     │
┌────▼─────────────────────────────┐
│            Host                   │
│  (AI Application, e.g. Claude)   │
│                                  │
│  ┌──────────┐  ┌──────────┐     │
│  │ MCP      │  │ MCP      │     │
│  │ Client 1 │  │ Client 2 │     │
│  └────┬─────┘  └────┬─────┘     │
└───────┼──────────────┼───────────┘
        │              │
   ┌────▼─────┐  ┌─────▼────┐
   │MCP Server│  │MCP Server│
   │(GitHub)  │  │(Calendar)│
   └──────────┘  └──────────┘
```

| Actor | Role |
|-------|------|
| **User** | End user interacting with the AI app |
| **Host** | The AI application (e.g., Claude Desktop, VS Code) |
| **MCP Client** | One per server, lives inside the host; manages connection |
| **MCP Server** | Connected to external resources; exposes tools/data |

### Transport Protocols

| Transport | Use Case |
|-----------|----------|
| **stdio** | Local server (same machine) |
| **HTTP + SSE** | Remote server (JSON-RPC 2.0) |

### Lifecycle

1. **Initialize**: Client ↔ Server handshake (capability negotiation)
2. **List tools**: `tools/list` → Server returns available tools
3. **Bind to LLM**: Tools are added to the LLM's context
4. **Tool calls**: LLM requests `tools/call` with tool name + arguments → Server executes and returns results

---

### MCP's Three Primitives

#### 1. Tools (LLM-controlled)
- **Discovery**: `tools/list` returns all available tools with schemas
- **Invocation**: `tools/call` with tool name + arguments
- **Safety**: Protocol supports **human-in-the-loop approval** before execution
- **Consumer**: The LLM itself decides when/how to call tools

#### 2. Resources (Application-controlled)
- Read-only data exposed by the server (calendar entries, documents, databases)
- **Discovery**: `resources/list` → returns available resources
- **Access**: `resources/read` → returns contents
- **Templated Resources**: For large collections (e.g., email), use URI templates with parameters

  ```
  Template: activities/{city}/{category}
  Example:  activities/Barcelona/museums → list of museums
  ```
- **Autocomplete**: Protocol supports partial completion for resource URIs
- **Security**: Read-only, but still needs access control (e.g., bank accounts, grades)
- **Consumer**: The AI application (not the LLM directly)

#### 3. Prompts (User-controlled)
- Pre-defined prompt templates exposed by the server
- Accessed via slash commands (e.g., `/code-review`)
- User fills in required fields; prompt is auto-constructed
- **Benefits**: 
  - No need for users to write full prompts
  - Server maintainers can update prompt templates without client changes
  - Consistent, well-tested prompts for common tasks

---

### MCP Advantages

#### Privacy & Security
- **Credentials never touch the LLM**: MCP client handles authentication locally

  ```json
  {
    "mcpServers": {
      "github": {
        "command": "npx -y @modelcontextprotocol/server-github",
        "env": {
          "GITHUB_PERSONAL_ACCESS_TOKEN": "<token>"
        }
      }
    }
  }
  ```
- Token stays in local environment → never sent to cloud LLM
- **Decoupling**: LLM provides task-relevant parameters; MCP client injects credentials separately

#### Dynamic Updates (No Downtime)
- MCP client registers for **change notifications**
- If server updates tools/resources/prompts → server notifies client
- Client terminates connection, reconnects, gets updated specs
- **No manual re-download** of API specs, no code changes needed

---

## 2. A2A — Agent-to-Agent Protocol (Google, 2025)

### Three Actors

| Actor | Role |
|-------|------|
| **User** | Interacts with the client agent only |
| **Client Agent** | User-facing agent; delegates subtasks |
| **Remote Agent (A2A Server)** | Specialized agents for specific tasks |

### Topology
- Not limited to star (hub-and-spoke)
- Supports **mesh**, **chain**, and arbitrary topologies
- Agent A → Agent B → Agent C (chained delegation)

### Agent Card (Discovery Mechanism)
- Analogous to HuggingFace **model cards** but for agents
- Contains:
  - Agent name, description, capabilities
  - Supported communication protocols
  - Trust/authentication requirements
- **Purpose**: Discovery + capability assessment (can this agent handle my subtask?)

### Key Concepts

| Term | Definition |
|------|------------|
| **Task** | A subtask delegated to a remote agent |
| **Message** | Back-and-forth communication between agents |
| **Artifact** | Files, images, or structured outputs exchanged |

### MCP vs A2A: When to Use Which

| Scenario | Use MCP | Use A2A |
|----------|---------|---------|
| Single agent + external tools | ✅ | ❌ |
| Multi-agent collaboration | ❌ | ✅ |
| Tool reuse across apps | ✅ | ❌ |
| Agent reuse across systems | ❌ | ✅ |
| Fine-grained tool orchestration | ✅ | ❌ |
| High-level task delegation | ❌ | ✅ |

**Can be combined**: A travel planner uses A2A for agent delegation (flight agent, hotel agent) while each agent internally uses MCP for tool access.

### A2A Limitations (Current)
- **Discovery**: No marketplace/catalog yet (unlike MCP which has searchable registries)
- Agents must be hardcoded in config
- Protocol still in early adoption (first draft stage)

---

## 3. agents.json (Google, Early 2025 — Abandoned)

### Problem Addressed
Existing APIs (Gmail, GitHub) have 50-200+ functions designed for **developers**, not LLMs:
- Descriptions lack LLM-friendly detail
- Parameter names are cryptic
- Multi-step workflows require implicit knowledge (e.g., "reply to email" needs: search threads → get thread ID → get base64 RFC822 content → call reply API)

### Solution: Flows & Links

| Concept | Purpose |
|---------|---------|
| **Flow** | Named multi-step workflow (e.g., "reply_to_contact") |
| **Link** | Maps output of one API call to input of the next |

```
Flow: "reply_to_contact"
  Step 1: search_threads(sender=contact) → thread_id
  Step 2: list_messages(thread_id) → message_id, base64_content  
  Step 3: reply(message_id, base64_content, body=response)
```

### Why It Failed
- People found **workarounds**: rewriting API descriptions, wrapping APIs in Python/Lambda
- MCP prompts can partially solve the same problem via metadata fields
- Too domain-specific; lacked a general-purpose vision
- Abandoned around April 2025 when MCP gained dominance

---

## Summary Comparison

| Protocol | Type | Status | Scope |
|----------|------|--------|-------|
| **MCP** | Context-oriented | ✅ Factual standard | Tools, resources, prompts for agents |
| **A2A** | Inter-agent | 🟡 First draft, testing | Agent-to-agent communication |
| **agents.json** | Context-oriented | ❌ Abandoned | API workflow abstraction |

### Future Outlook
- MCP is the **clear winner** in context protocols
- A2A will likely win the inter-agent protocol race
- Possible **merger** of MCP + A2A into a unified protocol
- Discovery and marketplace infrastructure still evolving
