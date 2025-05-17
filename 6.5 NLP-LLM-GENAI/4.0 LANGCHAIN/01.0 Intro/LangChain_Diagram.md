# LangChain Diagrams

This document contains visual representations of LangChain's architecture, components, and workflows. These diagrams help visualize how LangChain works and how its various components interact.

## Core Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       LangChain Framework                        │
├─────────────┬─────────────┬────────────────┬───────────────────┤
│             │             │                │                   │
│   Models    │   Prompts   │   Retrievers   │      Agents       │
│             │             │                │                   │
├─────────────┼─────────────┼────────────────┼───────────────────┤
│             │             │                │                   │
│   Memory    │   Chains    │  Embeddings    │      Tools        │
│             │             │                │                   │
└─────────────┴─────────────┴────────────────┴───────────────────┘
```

## LangChain Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Applications                             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│                            Agents                                │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│                            Chains                                │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│           Indexes             │            Memory                │
└───────────────────────────────┼─────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│                           Prompts                                │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│                            Models                                │
└─────────────────────────────────────────────────────────────────┘
```

## LLM Chain Flow Diagram

```
┌───────────┐     ┌──────────────┐     ┌─────────┐     ┌──────────┐
│           │     │              │     │         │     │          │
│   Input   ├────►│    Prompt    ├────►│   LLM   ├────►│  Output  │
│           │     │   Template   │     │         │     │          │
└───────────┘     └──────────────┘     └─────────┘     └──────────┘
```

## Retrieval-Augmented Generation (RAG) Flow

```
                  ┌───────────────────┐
                  │                   │
                  │     Document      │
                  │      Store        │
                  │                   │
                  └─────────┬─────────┘
                            │
                            ▼
┌───────────┐     ┌─────────────────┐     ┌───────────────┐
│           │     │                 │     │               │
│   Query   ├────►│    Retriever    ├────►│  Relevant     │
│           │     │                 │     │  Documents    │
└───────────┘     └─────────────────┘     └───────┬───────┘
                                                  │
                                                  ▼
┌───────────────────────────┐     ┌─────────┐     ┌───────────────┐
│                           │     │         │     │               │
│  Context-Enriched Prompt  │◄───┤  Prompt  │◄────┤  User Query   │
│                           │     │         │     │               │
└──────────────┬────────────┘     └─────────┘     └───────────────┘
               │
               ▼
      ┌─────────────────┐     ┌────────────────┐
      │                 │     │                │
      │      LLM        ├────►│     Response   │
      │                 │     │                │
      └─────────────────┘     └────────────────┘
```

## Agent System Architecture

```
                 ┌───────────────────────────┐
                 │                           │
                 │         Agent             │
                 │                           │
                 └─────────┬─────────────────┘
                           │
              ┌────────────┴───────────┐
              │                        │
┌─────────────▼─────┐         ┌────────▼───────────┐
│                   │         │                    │
│   Reasoning       │         │     Tool Usage     │
│   (LLM)           │         │                    │
│                   │         │                    │
└─────────┬─────────┘         └──┬────────────────┘
          │                      │
          │                      │
┌─────────▼─────────┐   ┌────────▼───────────┐
│                   │   │                    │
│  Action Planning  │   │  Tool Execution    │
│                   │   │                    │
└─────────┬─────────┘   └──┬─────────────────┘
          │                │
          └────────────────┘
                 │
      ┌──────────▼──────────┐
      │                     │
      │      Response       │
      │                     │
      └─────────────────────┘
```

## Memory Systems in LangChain

```
┌──────────────────────────────────────────────────────────────────┐
│                         Memory Types                              │
├──────────────┬──────────────┬───────────────┬───────────────────┤
│              │              │               │                   │
│ Conversation │   Vector     │   Entity      │   Summary         │
│   Buffer     │   Store      │   Memory      │   Memory          │
│              │              │               │                   │
└──────────────┴──────────────┴───────────────┴───────────────────┘
                           │
┌──────────────────────────┴───────────────────────────────────────┐
│                        Memory Backends                            │
├──────────────┬──────────────┬───────────────┬───────────────────┤
│              │              │               │                   │
│  In-Memory   │    Files     │   Databases   │  Cloud Storage    │
│              │              │               │                   │
└──────────────┴──────────────┴───────────────┴───────────────────┘
```

## Document Processing Pipeline

```
┌───────────────┐     ┌────────────────┐     ┌─────────────────┐
│               │     │                │     │                 │
│   Document    ├────►│  Text Splitter  ├────►│   Chunked Text  │
│   Loader      │     │                │     │                 │
└───────────────┘     └────────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────┐
                                            │                 │
                                            │   Embedding     │
                                            │   Model         │
                                            │                 │
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │                 │
                                            │  Vector Store   │
                                            │                 │
                                            └─────────────────┘
```

## LangChain Expression Language (LCEL) Flow

```
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│            │   │            │   │            │   │            │
│ Component1 ├──►│ Component2 ├──►│ Component3 ├──►│ Component4 │
│            │   │            │   │            │   │            │
└────────────┘   └────────────┘   └────────────┘   └────────────┘

                        │
                        ▼

┌───────────────────────────────────────────────────────────────┐
│                                                               │
│  chain = Component1 | Component2 | Component3 | Component4    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Typical QA System Architecture with LangChain

```
 User Question
      │
      ▼
┌────────────────┐
│                │
│  Preprocessor  │
│                │
└────────┬───────┘
         │
         ▼
┌────────────────┐    ┌─────────────────┐    ┌────────────────┐
│                │    │                 │    │                │
│   Retriever    │◄───┤   Vector Store  │◄───┤  Document DB   │
│                │    │                 │    │                │
└────────┬───────┘    └─────────────────┘    └────────────────┘
         │
         ▼
┌────────────────┐
│                │
│  Context       │
│  Generation    │
│                │
└────────┬───────┘
         │
         ▼
┌────────────────┐
│                │
│  Prompt        │
│  Construction  │
│                │
└────────┬───────┘
         │
         ▼
┌────────────────┐
│                │
│  LLM           │
│                │
└────────┬───────┘
         │
         ▼
┌────────────────┐
│                │
│  Response      │
│  Generation    │
│                │
└────────┬───────┘
         │
         ▼
     Answer
```

## Multi-Agent System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                        Orchestrator Agent                       │
│                                                                 │
└───┬───────────────────────┬───────────────────────┬────────────┘
    │                       │                       │
    ▼                       ▼                       ▼
┌─────────────┐      ┌─────────────┐         ┌─────────────┐
│             │      │             │         │             │
│ Research    │      │ Planning    │         │ Execution   │
│ Agent       │      │ Agent       │         │ Agent       │
│             │      │             │         │             │
└──┬──────────┘      └──┬──────────┘         └──┬──────────┘
   │                    │                       │
   ▼                    ▼                       ▼
┌─────────────┐      ┌─────────────┐         ┌─────────────┐
│             │      │             │         │             │
│ Knowledge   │      │ Task        │         │ Tool        │
│ Base        │      │ Queue       │         │ Set         │
│             │      │             │         │             │
└─────────────┘      └─────────────┘         └─────────────┘
```

## Retrieval Methods Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                      Retrieval Methods                           │
├────────────────┬──────────────────┬────────────────────────────┤
│                │                  │                            │
│ Similarity     │ Hybrid           │ Self-Query                 │
│ Search         │ Search           │ Retrieval                  │
│                │                  │                            │
├────────────────┴──────────────────┴────────────────────────────┤
│                                                                │
│                   Performance Characteristics                   │
├────────────────┬──────────────────┬────────────────────────────┤
│                │                  │                            │
│ Semantic       │ Keyword +        │ Natural Language           │
│ Matching       │ Semantic         │ Query Understanding        │
│                │                  │                            │
└────────────────┴──────────────────┴────────────────────────────┘
```

## Integration Points Diagram

```
                  ┌───────────────────────┐
                  │                       │
                  │    LangChain Core     │
                  │                       │
                  └───┬───────────────┬───┘
                      │               │
        ┌─────────────┘               └──────────────┐
        │                                            │
┌───────▼──────────┐                      ┌──────────▼─────────┐
│                  │                      │                    │
│  Model           │                      │  Storage           │
│  Providers       │                      │  Providers         │
│                  │                      │                    │
└──────────────────┘                      └────────────────────┘
   │                                            │
   ▼                                            ▼
┌──────────────────┐                      ┌────────────────────┐
│ OpenAI           │                      │ Pinecone           │
│ Anthropic        │                      │ Chroma             │
│ Hugging Face     │                      │ FAISS              │
│ Google           │                      │ Weaviate           │
│ Azure            │                      │ Qdrant             │
│ Cohere           │                      │ MongoDB            │
└──────────────────┘                      └────────────────────┘
                                        
                    ┌───────────────┐
                    │               │
                    │  Document     │
                    │  Loaders      │
                    │               │
                    └───────────────┘
                             │
                             ▼
                    ┌───────────────┐
                    │ PDF           │
                    │ HTML          │
                    │ JSON          │
                    │ CSV           │
                    │ Markdown      │
                    │ Notion        │
                    │ Google Drive  │
                    └───────────────┘
```

## LangChain Tool Ecosystem

```
┌─────────────────────────────────────────────────────────────────┐
│                          Tool Types                              │
├─────────────┬─────────────┬────────────────┬───────────────────┤
│             │             │                │                   │
│  Search     │   REPL      │   Structured   │     API           │
│  Tools      │   Tools     │   Output       │     Tools         │
│             │             │   Tools        │                   │
├─────────────┼─────────────┼────────────────┼───────────────────┤
│             │             │                │                   │
│  Human      │   Shell     │   Database     │     File          │
│  Tools      │   Tools     │   Tools        │     Tools         │
│             │             │                │                   │
└─────────────┴─────────────┴────────────────┴───────────────────┘
```

## Chain Types and Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Chain Types                             │
├─────────────┬─────────────┬────────────────┬───────────────────┤
│             │             │                │                   │
│  LLM        │  Sequential │   Router       │     QA            │
│  Chain      │   Chain     │   Chain        │     Chain         │
│             │             │                │                   │
├─────────────┼─────────────┼────────────────┼───────────────────┤
│             │             │                │                   │
│  Retrieval  │  MapReduce  │   Refine       │     Transform     │
│  Chain      │   Chain     │   Chain        │     Chain         │
│             │             │                │                   │
└─────────────┴─────────────┴────────────────┴───────────────────┘
```

## RAG System Extended Architecture

```
Offline Processing:
┌───────────────┐     ┌────────────────┐     ┌─────────────────┐
│               │     │                │     │                 │
│   Document    ├────►│  Text Splitter  ├────►│   Text Chunks   │
│   Sources     │     │                │     │                 │
└───────────────┘     └────────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────┐
                                            │                 │
                                            │   Metadata      │
                                            │   Enrichment    │
                                            │                 │
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │                 │
                                            │  Embedding      │
                                            │  Generation     │
                                            │                 │
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │                 │
                                            │  Vector Store   │
                                            │  Indexing       │
                                            │                 │
                                            └─────────────────┘

Runtime Query Processing:
┌───────────────┐     ┌────────────────┐     ┌─────────────────┐
│               │     │                │     │                 │
│   User        ├────►│  Query         ├────►│   Query         │
│   Query       │     │  Understanding │     │   Embedding     │
└───────────────┘     └────────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────┐
                                            │                 │
                                            │   Vector        │
                                            │   Search        │
                                            │                 │
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │                 │
                                            │  Re-ranking     │
                                            │                 │
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │                 │
                                            │  Context        │
                                            │  Construction   │
                                            │                 │
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │                 │
                                            │  LLM            │
                                            │  Generation     │
                                            │                 │
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │                 │
                                            │  Response       │
                                            │                 │
                                            └─────────────────┘
```

## Advanced LangChain Components Interaction

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│                       LangChain Application                     │
│                                                                │
└────────────────────────────┬───────────────────────────────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
    ┌────────────▼──────────┐   ┌────────▼───────────┐
    │                       │   │                    │
    │      Chains           │   │     Agents         │
    │                       │   │                    │
    └────────────┬──────────┘   └────────┬───────────┘
                 │                       │
                 │                       │
    ┌────────────▼──────────┐   ┌────────▼───────────┐
    │                       │   │                    │
    │      Retrievers       │   │     Tools          │
    │                       │   │                    │
    └────────────┬──────────┘   └────────┬───────────┘
                 │                       │
                 │                       │
    ┌────────────▼──────────┐   ┌────────▼───────────┐
    │                       │   │                    │
    │      Memory           │   │     Embeddings     │
    │                       │   │                    │
    └────────────┬──────────┘   └────────┬───────────┘
                 │                       │
                 └───────────┬───────────┘
                             │
                  ┌──────────▼─────────┐
                  │                    │
                  │      Models        │
                  │                    │
                  └────────────────────┘
```

## Event-Based Monitoring with LangSmith

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                          LangChain App                          │
│                                                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Emits Events
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                          LangSmith                              │
│                                                                 │
├─────────────┬─────────────┬────────────────┬───────────────────┤
│             │             │                │                   │
│  Traces     │  Feedback   │   Datasets     │     Evaluation    │
│             │             │                │                   │
└─────────────┴─────────────┴────────────────┴───────────────────┘
```

## Different Types of Chains

### LLMChain
```
Input → Prompt Template → LLM → Output
```

### Sequential Chain
```
Input → Chain1 → Output1 → Chain2 → Output2 → ... → ChainN → Final Output

```

### Router Chain
```
                    ┌─────► Chain1 ─────┐
                    │                   │
Input → Routing LLM ├─────► Chain2 ─────┤ → Output
                    │                   │
                    └─────► Chain3 ─────┘
```

### Retrieval QA Chain
```
Question ──────────────────────┐
                               │
                               ▼
Document DB → Retriever → Retrieved Docs → Context Construction
                                               │
                                               ▼
                                       Prompt Template
                                               │
                                               ▼
                                             LLM
                                               │
                                               ▼
                                            Answer
```

## Memory Types and Their Use Cases
```

┌─────────────────────────────────────────────────────────────────┐
│                        Memory Types                              │
├───────────────┬───────────────────────┬──────────────────────────┤
│ Type          │ Description           │ Use Cases                │
├───────────────┼───────────────────────┼──────────────────────────┤
│ Buffer Memory │ Stores raw history    │ Simple chatbots          │
├───────────────┼───────────────────────┼──────────────────────────┤
│ Summary       │ Keeps compressed      │ Long conversations       │
│ Memory        │ history               │                          │
├───────────────┼───────────────────────┼──────────────────────────┤
│ Vector Store  │ Stores embeddings     │ Semantic search in       │
│ Memory        │ of conversations      │ conversation history     │
├───────────────┼───────────────────────┼──────────────────────────┤
│ Entity Memory │ Tracks information    │ Personalized             │
│               │ about entities        │ interactions             │
├───────────────┼───────────────────────┼──────────────────────────┤
│ Conversation  │ Manages messages      │ Multi-turn               │
│ Token Buffer  │ within token limits   │ conversations            │
└───────────────┴───────────────────────┴──────────────────────────┘
```

## Core Abstractions in LangChain

```
┌─────────────────────────────────────────────────────────────────┐
│                     Core LangChain Abstractions                  │
├─────────────┬─────────────┬────────────────┬───────────────────┤
│             │             │                │                   │
│  Model      │  Prompt     │   Index        │     Chain         │
│             │             │                │                   │
├─────────────┼─────────────┼────────────────┼───────────────────┤
│             │             │                │                   │
│  Memory     │  Retriever  │   Embedding    │     Agent         │
│             │             │                │                   │
└─────────────┴─────────────┴────────────────┴───────────────────┘
```

## LangChain Class Hierarchy and Functions

Below are comprehensive diagrams showing the most important classes and their functions in LangChain.

### Model Classes Hierarchy

```mermaid
classDiagram
    class BaseLanguageModel {
        +invoke(prompt)
        +generate(prompts)
        +predict(text)
        +predict_messages(messages)
        +with_structured_output(schema)
    }
    
    class LLM {
        +_call(prompt)
        +_generate(prompts)
        +stream(prompt)
        +get_num_tokens(text)
    }
    
    class ChatModel {
        +_generate_messages(messages)
        +_stream_messages(messages)
        +converse(messages)
    }
    
    class BaseChatModel {
        +_call(messages)
        +_generate(messages)
        +stream(messages)
    }
    
    BaseLanguageModel <|-- LLM
    BaseLanguageModel <|-- BaseChatModel
    BaseChatModel <|-- ChatModel
    
    class OpenAI {
        +model_name
        +temperature
        +max_tokens
        +model_kwargs
    }
    
    class ChatOpenAI {
        +model_name
        +temperature
        +max_tokens
        +model_kwargs
    }
    
    class HuggingFacePipeline {
        +model
        +tokenizer
        +pipeline
    }
    
    LLM <|-- OpenAI
    LLM <|-- HuggingFacePipeline
    ChatModel <|-- ChatOpenAI
    
    class Embeddings {
        +embed_documents(texts)
        +embed_query(text)
    }
    
    class OpenAIEmbeddings {
        +model
        +embedding_dim
    }
    
    Embeddings <|-- OpenAIEmbeddings
```

### Prompt Classes and Functions

```mermaid
classDiagram
    class BasePromptTemplate {
        +input_variables
        +format(values)
        +format_prompt(values)
        +partial(values)
        +save(file_path)
    }
    
    class StringPromptTemplate {
        +template
        +template_format
        +validate_template()
    }
    
    class PromptTemplate {
        +template
        +from_template(template)
        +from_file(file_path, template_format)
    }
    
    class ChatPromptTemplate {
        +messages
        +from_messages(messages)
        +format_messages(values)
    }
    
    class MessagePromptTemplate {
        +prompt
        +role
        +format(values)
        +format_messages(values)
    }
    
    class HumanMessagePromptTemplate {
        +from_template(template)
    }
    
    class AIMessagePromptTemplate {
        +from_template(template)
    }
    
    class SystemMessagePromptTemplate {
        +from_template(template)
    }
    
    BasePromptTemplate <|-- StringPromptTemplate
    StringPromptTemplate <|-- PromptTemplate
    BasePromptTemplate <|-- ChatPromptTemplate
    BasePromptTemplate <|-- MessagePromptTemplate
    MessagePromptTemplate <|-- HumanMessagePromptTemplate
    MessagePromptTemplate <|-- AIMessagePromptTemplate
    MessagePromptTemplate <|-- SystemMessagePromptTemplate
```

### Chain Classes Hierarchy

```mermaid
classDiagram
    class Runnable {
        +invoke(input)
        +batch(inputs)
        +stream(input)
        +pipe(destination)
    }
    
    class Chain {
        +run(inputs)
        +apply(inputs)
        +from_llm(llm, prompt)
    }
    
    class LLMChain {
        +llm
        +prompt
        +output_parser
        +predict(values)
        +predict_and_parse(values)
    }
    
    class SequentialChain {
        +chains
        +input_variables
        +output_variables
    }
    
    class SimpleSequentialChain {
        +chains
    }
    
    class RouterChain {
        +routes
        +default_route
        +route(inputs)
    }
    
    class ConversationChain {
        +llm
        +memory
        +prompt
    }
    
    class RetrievalQA {
        +retriever
        +combine_documents_chain
    }
    
    Runnable <|-- Chain
    Chain <|-- LLMChain
    Chain <|-- SequentialChain
    Chain <|-- RouterChain
    Chain <|-- ConversationChain
    Chain <|-- RetrievalQA
    SequentialChain <|-- SimpleSequentialChain
```

### Memory Classes

```mermaid
classDiagram
    class BaseMemory {
        +load_memory_variables(inputs)
        +save_context(inputs, outputs)
        +clear()
    }
    
    class ConversationBufferMemory {
        +chat_memory
        +return_messages
        +add_message(message)
        +clear()
    }
    
    class ConversationBufferWindowMemory {
        +k
        +add_message(message)
    }
    
    class ConversationSummaryMemory {
        +llm
        +summarize(messages)
    }
    
    class VectorStoreRetrieverMemory {
        +retriever
        +search(query)
    }
    
    BaseMemory <|-- ConversationBufferMemory
    ConversationBufferMemory <|-- ConversationBufferWindowMemory
    BaseMemory <|-- ConversationSummaryMemory
    BaseMemory <|-- VectorStoreRetrieverMemory
```

### Document Loading and Processing

```mermaid
classDiagram
    class BaseLoader {
        +load()
        +lazy_load()
    }
    
    class TextLoader {
        +file_path
        +encoding
        +read_file()
    }
    
    class PDFLoader {
        +file_path
        +extract_text()
    }
    
    class WebBaseLoader {
        +web_path
        +scrape_content()
    }
    
    class TextSplitter {
        +split_text(text)
        +split_documents(documents)
        +create_documents(texts)
    }
    
    class RecursiveCharacterTextSplitter {
        +chunk_size
        +chunk_overlap
        +separators
    }
    
    class TokenTextSplitter {
        +chunk_size
        +chunk_overlap
        +tokenizer
    }
    
    BaseLoader <|-- TextLoader
    BaseLoader <|-- PDFLoader
    BaseLoader <|-- WebBaseLoader
    
    TextSplitter <|-- RecursiveCharacterTextSplitter
    TextSplitter <|-- TokenTextSplitter
```

### Vector Stores and Retrievers

```mermaid
classDiagram
    class VectorStore {
        +add_texts(texts)
        +add_documents(documents)
        +similarity_search(query)
        +similarity_search_by_vector(embedding)
        +as_retriever()
    }
    
    class Chroma {
        +collection
        +embedding_function
        +persist()
    }
    
    class FAISS {
        +index
        +embedding_function
        +save_local(folder_path)
        +load_local(folder_path)
    }
    
    class Pinecone {
        +index
        +embedding_function
        +namespace
    }
    
    class BaseRetriever {
        +get_relevant_documents(query)
        +invoke(query)
    }
    
    class VectorStoreRetriever {
        +vectorstore
        +search_type
        +search_kwargs
    }
    
    class ContextualCompressionRetriever {
        +base_retriever
        +base_compressor
    }
    
    VectorStore <|-- Chroma
    VectorStore <|-- FAISS
    VectorStore <|-- Pinecone
    
    BaseRetriever <|-- VectorStoreRetriever
    BaseRetriever <|-- ContextualCompressionRetriever
```

### Agents and Tools

```mermaid
classDiagram
    class BaseTool {
        +name
        +description
        +_run(input)
        +run(input)
    }
    
    class Tool {
        +func
        +coroutine
    }
    
    class StructuredTool {
        +args_schema
        +validate_args(args)
    }
    
    class WikipediaQueryRun {
        +api_wrapper
        +top_k_results
    }
    
    class SerpAPIWrapper {
        +api_key
        +search(query)
    }
    
    class BaseAgent {
        +llm_chain
        +allowed_tools
        +parse_reaction(response)
        +plan(inputs)
    }
    
    class Agent {
        +create_prompt(tools)
        +get_default_output_parser()
    }
    
    class ZeroShotAgent {
        +prefix
        +suffix
        +format_instructions
    }
    
    class ReActAgent {
        +observation_prefix
        +observation_suffix
        +llm_prefix
    }
    
    class AgentExecutor {
        +agent
        +tools
        +max_iterations
        +run(inputs)
        +take_next_step()
    }
    
    BaseTool <|-- Tool
    BaseTool <|-- StructuredTool
    Tool <|-- WikipediaQueryRun
    Tool <|-- SerpAPIWrapper
    
    BaseAgent <|-- Agent
    Agent <|-- ZeroShotAgent
    Agent <|-- ReActAgent
```

### LangChain Expression Language (LCEL) Components

```mermaid
classDiagram
    class Runnable {
        +invoke(input)
        +batch(inputs)
        +stream(input)
        +pipe(destination)
    }
    
    class RunnablePassthrough {
        +assign(**kwargs)
        +bind(values)
    }
    
    class RunnableParallel {
        +assign(**kwargs)
        +bind(values)
    }
    
    class RunnableSequence {
        +steps
        +first
        +last
    }
    
    class RunnableBranch {
        +branches
        +default
    }
    
    class RunnableLambda {
        +func
    }
    
    Runnable <|-- RunnablePassthrough
    Runnable <|-- RunnableParallel
    Runnable <|-- RunnableSequence
    Runnable <|-- RunnableBranch
    Runnable <|-- RunnableLambda
```

### Common Function Usage Patterns

```mermaid
flowchart TD
    A[User Input] --> B[Preprocess Input]
    B --> C{Task Type}
    
    C -->|Generation| D[Create Prompt Template]
    D --> E[Create LLM]
    E --> F[Create Chain]
    F --> G[Run Chain]
    G --> H[Process Output]
    
    C -->|Retrieval| I[Load Documents]
    I --> J[Split Text]
    J --> K[Create Embeddings]
    K --> L[Store in Vector DB]
    L --> M[Create Retriever]
    M --> N[Create Chain]
    N --> O[Run Chain]
    O --> P[Process Output]
    
    C -->|Conversational| Q[Create Memory]
    Q --> R[Create Prompt Template]
    R --> S[Create LLM]
    S --> T[Create Chain]
    T --> U[Run Chain]
    U --> V[Update Memory]
    V --> W[Process Output]
    
    C -->|Agent| X[Define Tools]
    X --> Y[Create Agent]
    Y --> Z[Create Agent Executor]
    Z --> AA[Run Agent]
    AA --> AB[Process Output]
```

## Common LangChain Imports and Their Use Cases

This section provides a reference guide for the most frequently used LangChain imports and their typical use cases. Understanding these import patterns will help you quickly implement various LangChain functionalities.

### Models Imports

```mermaid
graph TD
    A[Models] --> B[LLMs]
    A --> C[Chat Models]
    A --> D[Embeddings]
    
    B --> B1[from langchain.llms import OpenAI, HuggingFacePipeline]
    C --> C1[from langchain.chat_models import ChatOpenAI, ChatAnthropic]
    D --> D1[from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings]
```

| Import Statement | Purpose | Example Use |
|-----------------|---------|-------------|
| `from langchain.llms import OpenAI` | Import text-completion LLM models | `llm = OpenAI(temperature=0.7)` |
| `from langchain.chat_models import ChatOpenAI` | Import conversational models | `chat = ChatOpenAI(model="gpt-4")` |
| `from langchain.embeddings import OpenAIEmbeddings` | Import embedding models | `embeddings = OpenAIEmbeddings()` |
| `from langchain_openai import OpenAI, ChatOpenAI` | Direct provider imports (newer syntax) | `model = ChatOpenAI()` |
| `from langchain_anthropic import ChatAnthropic` | Import Anthropic models | `model = ChatAnthropic(model="claude-3-sonnet-20240229")` |
| `from langchain_google_vertexai import ChatVertexAI` | Import Google's Vertex AI models | `model = ChatVertexAI()` |
| `from langchain_huggingface import HuggingFaceEndpoint` | Import HuggingFace endpoints | `model = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.1")` |

### Prompts Imports

```mermaid
graph TD
    A[Prompts] --> B[Templates]
    A --> C[Message Templates]
    
    B --> B1[from langchain.prompts import PromptTemplate]
    B --> B2[from langchain.prompts import FewShotPromptTemplate]
    C --> C1[from langchain.prompts import ChatPromptTemplate]
    C --> C2[from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate]
```

| Import Statement | Purpose | Example Use |
|-----------------|---------|-------------|
| `from langchain.prompts import PromptTemplate` | Create reusable text prompts with variables | `template = PromptTemplate.from_template("Answer about {topic}")` |
| `from langchain.prompts import ChatPromptTemplate` | Create multi-message chat prompts | `prompt = ChatPromptTemplate.from_messages([("system", "You are a helper"), ("human", "{query}")])` |
| `from langchain.prompts import FewShotPromptTemplate` | Create prompts with examples | `few_shot = FewShotPromptTemplate(examples=examples, example_prompt=example_prompt, prefix=prefix, suffix=suffix)` |
| `from langchain.prompts.chat import SystemMessagePromptTemplate` | Create system message templates | `system_message = SystemMessagePromptTemplate.from_template("You are a {role}")` |
| `from langchain.prompts.chat import HumanMessagePromptTemplate` | Create human message templates | `human_message = HumanMessagePromptTemplate.from_template("{input}")` |
| `from langchain.prompts.chat import AIMessagePromptTemplate` | Create AI message templates | `ai_message = AIMessagePromptTemplate.from_template("{response}")` |

### Chains Imports

```mermaid
graph TD
    A[Chains] --> B[Basic Chains]
    A --> C[Specialized Chains]
    
    B --> B1[from langchain.chains import LLMChain]
    B --> B2[from langchain.chains import SequentialChain, SimpleSequentialChain]
    C --> C1[from langchain.chains import RetrievalQA]
    C --> C2[from langchain.chains import ConversationChain]
    C --> C3[from langchain.chains.router import MultiPromptChain]
```

| Import Statement | Purpose | Example Use |
|-----------------|---------|-------------|
| `from langchain.chains import LLMChain` | Basic chain connecting prompt to LLM | `chain = LLMChain(llm=llm, prompt=prompt)` |
| `from langchain.chains import SequentialChain` | Chain multiple chains together | `chain = SequentialChain(chains=[chain1, chain2], input_variables=["query"])` |
| `from langchain.chains import SimpleSequentialChain` | Simpler sequential chain with single inputs/outputs | `chain = SimpleSequentialChain(chains=[chain1, chain2])` |
| `from langchain.chains import RetrievalQA` | Chain for question answering over documents | `qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)` |
| `from langchain.chains import ConversationChain` | Chain with conversation memory | `conversation = ConversationChain(llm=llm, memory=memory)` |
| `from langchain.chains.router import MultiPromptChain` | Route inputs to different chains | `router_chain = MultiPromptChain(router_chain=router, destination_chains=destination_chains)` |
| `from langchain.chains.summarize import load_summarize_chain` | Chain for text summarization | `summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")` |

### Memory Imports

```mermaid
graph TD
    A[Memory] --> B[Basic Memory]
    A --> C[Advanced Memory]
    
    B --> B1[from langchain.memory import ConversationBufferMemory]
    B --> B2[from langchain.memory import ConversationBufferWindowMemory]
    C --> C1[from langchain.memory import ConversationSummaryMemory]
    C --> C2[from langchain.memory import VectorStoreRetrieverMemory]
```

| Import Statement | Purpose | Example Use |
|-----------------|---------|-------------|
| `from langchain.memory import ConversationBufferMemory` | Store full conversation history | `memory = ConversationBufferMemory()` |
| `from langchain.memory import ConversationBufferWindowMemory` | Store limited conversation turns | `memory = ConversationBufferWindowMemory(k=5)` |
| `from langchain.memory import ConversationSummaryMemory` | Summarize conversation history | `memory = ConversationSummaryMemory(llm=llm)` |
| `from langchain.memory import ConversationTokenBufferMemory` | Limit memory by token count | `memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=2000)` |
| `from langchain.memory import VectorStoreRetrieverMemory` | Store memories in vector store | `memory = VectorStoreRetrieverMemory(retriever=retriever)` |

### Document Processing Imports

```mermaid
graph TD
    A[Document Processing] --> B[Loaders]
    A --> C[Splitters]
    A --> D[Transformers]
    
    B --> B1[from langchain.document_loaders import TextLoader, PDFLoader]
    B --> B2[from langchain.document_loaders import WebBaseLoader]
    C --> C1[from langchain.text_splitter import RecursiveCharacterTextSplitter]
    C --> C2[from langchain.text_splitter import TokenTextSplitter]
    D --> D1[from langchain.document_transformers import HTML2TextTransformer]
```

| Import Statement | Purpose | Example Use |
|-----------------|---------|-------------|
| `from langchain.document_loaders import TextLoader` | Load text files | `loader = TextLoader("document.txt")` |
| `from langchain.document_loaders import PDFLoader` | Load PDF files | `loader = PDFLoader("document.pdf")` |
| `from langchain.document_loaders import WebBaseLoader` | Load web pages | `loader = WebBaseLoader("https://example.com")` |
| `from langchain.document_loaders import DirectoryLoader` | Load files from directory | `loader = DirectoryLoader("./data/", glob="**/*.pdf", loader_cls=PDFLoader)` |
| `from langchain.document_loaders.csv_loader import CSVLoader` | Load CSV files | `loader = CSVLoader("data.csv")` |
| `from langchain.text_splitter import RecursiveCharacterTextSplitter` | Split text by character with awareness of structure | `splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` |
| `from langchain.text_splitter import TokenTextSplitter` | Split text based on token count | `splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=20)` |

### Vector Stores and Retrievers Imports

```mermaid
graph TD
    A[Vector Stores & Retrievers] --> B[Vector Stores]
    A --> C[Retrievers]
    
    B --> B1[from langchain.vectorstores import Chroma, FAISS]
    B --> B2[from langchain.vectorstores import Pinecone]
    C --> C1[from langchain.retrievers import ContextualCompressionRetriever]
    C --> C2[from langchain.retrievers.multi_query import MultiQueryRetriever]
```

| Import Statement | Purpose | Example Use |
|-----------------|---------|-------------|
| `from langchain.vectorstores import Chroma` | In-memory or persistent vector store | `db = Chroma.from_documents(docs, embedding)` |
| `from langchain.vectorstores import FAISS` | Facebook AI Similarity Search | `db = FAISS.from_documents(docs, embedding)` |
| `from langchain.vectorstores import Pinecone` | Pinecone vector database | `db = Pinecone.from_documents(docs, embedding, index_name="my-index")` |
| `from langchain.vectorstores import Milvus` | Milvus vector database | `db = Milvus.from_documents(docs, embedding)` |
| `from langchain.vectorstores import Qdrant` | Qdrant vector database | `db = Qdrant.from_documents(docs, embedding)` |
| `from langchain.retrievers import ContextualCompressionRetriever` | Compress retriever results | `retriever = ContextualCompressionRetriever(base_retriever=vectorstore.as_retriever(), base_compressor=compressor)` |
| `from langchain.retrievers.multi_query import MultiQueryRetriever` | Generate multiple queries | `retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)` |

### Agents and Tools Imports

```mermaid
graph TD
    A[Agents & Tools] --> B[Agents]
    A --> C[Tools]
    A --> D[Toolkits]
    
    B --> B1[from langchain.agents import initialize_agent, AgentType]
    B --> B2[from langchain.agents import create_react_agent, create_openai_functions_agent]
    C --> C1[from langchain.tools import BaseTool, StructuredTool, Tool]
    C --> C2[from langchain.tools import WikipediaQueryRun, SerpAPIWrapper]
    D --> D1[from langchain.agents.agent_toolkits import create_sql_agent]
```

| Import Statement | Purpose | Example Use |
|-----------------|---------|-------------|
| `from langchain.agents import initialize_agent, AgentType` | Create agent with predefined types | `agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)` |
| `from langchain.agents import create_react_agent` | Create ReAct agent | `agent = create_react_agent(llm, tools, prompt)` |
| `from langchain.agents import create_openai_functions_agent` | Create OpenAI Functions agent | `agent = create_openai_functions_agent(llm, tools, prompt)` |
| `from langchain.agents import AgentExecutor` | Execute agent actions | `agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)` |
| `from langchain.tools import Tool` | Define tools for agents | `tool = Tool(name="Search", func=search_func, description="Search for information")` |
| `from langchain.tools.python.tool import PythonREPLTool` | Python execution tool | `python_tool = PythonREPLTool()` |
| `from langchain.tools import WikipediaQueryRun` | Search Wikipedia | `wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())` |
| `from langchain.agents.agent_toolkits import create_sql_agent` | Create SQL database agent | `agent = create_sql_agent(llm=llm, toolkit=sql_toolkit, verbose=True)` |

### Output Parsing Imports

```mermaid
graph TD
    A[Output Parsing] --> B[Basic Parsers]
    A --> C[Structured Parsers]
    
    B --> B1[from langchain.output_parsers import StrOutputParser]
    C --> C1[from langchain.output_parsers import PydanticOutputParser]
    C --> C2[from langchain.output_parsers import ResponseSchema, StructuredOutputParser]
```

| Import Statement | Purpose | Example Use |
|-----------------|---------|-------------|
| `from langchain.output_parsers import StrOutputParser` | Parse output as string | `parser = StrOutputParser()` |
| `from langchain.output_parsers import PydanticOutputParser` | Parse output into Pydantic model | `parser = PydanticOutputParser(pydantic_object=MyModel)` |
| `from langchain.output_parsers import ResponseSchema, StructuredOutputParser` | Parse output into structured format | `parser = StructuredOutputParser.from_response_schemas([ResponseSchema(name="answer")])` |
| `from langchain.output_parsers import CommaSeparatedListOutputParser` | Parse comma-separated list | `parser = CommaSeparatedListOutputParser()` |
| `from langchain.output_parsers.json import SimpleJsonOutputParser` | Parse JSON output | `parser = SimpleJsonOutputParser()` |

### LCEL (LangChain Expression Language) Imports

```mermaid
graph TD
    A[LCEL Components] --> B[Core Runnables]
    A --> C[Utility Runnables]
    
    B --> B1[from langchain.schema.runnable import RunnablePassthrough]
    B --> B2[from langchain.schema.runnable import RunnableSequence]
    C --> C1[from langchain.schema.runnable import RunnableParallel]
    C --> C2[from langchain.schema.runnable import RunnableBranch, RunnableLambda]
```

| Import Statement | Purpose | Example Use |
|-----------------|---------|-------------|
| `from langchain.schema.runnable import RunnablePassthrough` | Pass input unchanged | `chain = {"context": retriever, "question": RunnablePassthrough()} \| prompt \| model` |
| `from langchain.schema.runnable import RunnableSequence` | Explicit sequence of steps | `chain = RunnableSequence([retriever, prompt, model, parser])` |
| `from langchain.schema.runnable import RunnableParallel` | Process inputs in parallel | `chain = RunnableParallel({"summary": summarize_chain, "entities": entity_chain})` |
| `from langchain.schema.runnable import RunnableBranch` | Select branch based on condition | `chain = RunnableBranch((lambda x: condition, chain_a), chain_b)` |
| `from langchain.schema.runnable import RunnableLambda` | Apply custom function | `chain = RunnableLambda(lambda x: process_data(x))` |

### Core Framework Imports

```mermaid
graph TD
    A[Core Framework] --> B[Basic Components]
    A --> C[Callbacks]
    
    B --> B1[from langchain.core.messages import HumanMessage, AIMessage, SystemMessage]
    B --> B2[from langchain.core.documents import Document]
    C --> C1[from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler]
    C --> C2[from langchain.callbacks.manager import CallbackManager]
```

| Import Statement | Purpose | Example Use |
|-----------------|---------|-------------|
| `from langchain.core.messages import HumanMessage, AIMessage, SystemMessage` | Message objects for chat | `messages = [SystemMessage(content="You are a helpful AI"), HumanMessage(content="Hello")]` |
| `from langchain.core.documents import Document` | Document objects | `doc = Document(page_content="text", metadata={"source": "file.txt"})` |
| `from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler` | Stream model output | `model = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])` |
| `from langchain.callbacks.manager import CallbackManager` | Manage multiple callbacks | `callback_manager = CallbackManager([handler1, handler2])` |
| `from langchain.schema import BaseOutputParser` | Base class for custom parsers | `class CustomParser(BaseOutputParser): ...` |

### Integration Imports

```mermaid
graph TD
    A[Integrations] --> B[LangSmith]
    A --> C[Other Integrations]
    
    B --> B1[from langchain.smith import RunEvalConfig]
    B --> B2[from langsmith import Client]
    C --> C1[from langchain_community import various_imports]
    C --> C2[from langchain_openai import various_imports]
```

| Import Statement | Purpose | Example Use |
|-----------------|---------|-------------|
| `from langchain.smith import RunEvalConfig` | Configure evaluation runs | `eval_config = RunEvalConfig(evaluators=["qa"])` |
| `from langsmith import Client` | LangSmith client | `client = Client()` |
| `from langchain_community.chat_models import ChatOllama` | Community models | `model = ChatOllama(model="llama2")` |
| `from langchain_community.vectorstores import Qdrant` | Community vector stores | `db = Qdrant.from_documents(docs, embedding)` |
| `from langchain_openai import ChatOpenAI` | OpenAI integration | `model = ChatOpenAI()` |
| `from langchain_experimental.agents import create_pandas_dataframe_agent` | Experimental features | `agent = create_pandas_dataframe_agent(llm, df, verbose=True)` |

### Migration Note

Recent versions of LangChain have reorganized imports. The pattern is now:

- `langchain`: Core abstractions and interfaces
- `langchain_community`: Community-contributed components
- `langchain_{provider}`: Provider-specific implementations (e.g., `langchain_openai`)

For example:
- Old: `from langchain.chat_models import ChatOpenAI`
- New: `from langchain_openai import ChatOpenAI`

Many imports still work with the old pattern but are being gradually migrated to the new structure.
