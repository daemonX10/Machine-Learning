# LangChain Models: In-Depth Guide

## Table of Contents
- [LangChain Models: In-Depth Guide](#langchain-models-in-depth-guide)
  - [Table of Contents](#table-of-contents)
  - [Introduction to Models in LangChain](#introduction-to-models-in-langchain)
  - [Types of Models](#types-of-models)
    - [Language Models](#language-models)
      - [LLMs vs Chat Models](#llms-vs-chat-models)
    - [Embedding Models](#embedding-models)
  - [Working with Language Models](#working-with-language-models)
    - [Closed-Source Models](#closed-source-models)
      - [OpenAI Models](#openai-models)
      - [Anthropic Models](#anthropic-models)
      - [Google Models](#google-models)
    - [Open-Source Models](#open-source-models)
      - [Using Hugging Face Inference API](#using-hugging-face-inference-api)
      - [Running Models Locally](#running-models-locally)
  - [Working with Embedding Models](#working-with-embedding-models)
    - [OpenAI Embeddings](#openai-embeddings)
    - [Open-Source Embeddings](#open-source-embeddings)
  - [Building a Document Similarity Application](#building-a-document-similarity-application)
  - [Conclusion](#conclusion)

## Introduction to Models in LangChain

The Model component in LangChain is a crucial part of the framework designed to facilitate interactions with various language models and embedding models. In simple terms, while there are many different AI models from various companies in the world, they all behave differently when you try to interact with them through code. LangChain's Model component provides a common interface that allows you to connect easily with any type of AI model.

```
┌───────────────────────────────────────────────────┐
│               LangChain Model Component            │
├───────────────────┬───────────────────────────────┤
│  Language Models  │       Embedding Models         │
├───────────────────┼───────────────────────────────┤
│ - Text in,        │ - Text in,                    │
│   Text out        │   Vector (numbers) out        │
│ - Used for:       │ - Used for:                   │
│   * Text gen      │   * Semantic search           │
│   * QA            │   * Document similarity       │
│   * Chat          │   * Clustering                │
│   * Code gen      │   * RAG applications          │
└───────────────────┴───────────────────────────────┘
```

There are two main types of models in LangChain:

1. **Language Models**: These models take text input, process it, and return text output. For example, if you ask "What is the capital of India?", they understand the text, process it, and return "New Delhi".

2. **Embedding Models**: These models also take text input but instead of returning text, they return a series of numbers (vectors). These vectors represent the semantic meaning of the text, which is useful for tasks like semantic search.

## Types of Models

### Language Models

Language models process text input and generate text output. Within language models, LangChain further categorizes them into two types:

#### LLMs vs Chat Models

**LLMs (Language Learning Models):**
- General-purpose models for NLP applications
- Can be used for text generation, summarization, code generation, etc.
- Take a string (plain text) as input and return a string as output
- Support for LLMs in LangChain is gradually being phased out in favor of Chat models

**Chat Models:**
- Specialized language models for conversation tasks
- Ideal for building chatbots, virtual assistants, etc.
- Take a sequence of messages as input and return chat messages as output
- Support multi-turn conversation with memory of previous interactions
- Can understand roles (system, user, assistant)

Here's a comparison table showing the key differences:

| Aspect | LLMs | Chat Models |
|--------|------|------------|
| Purpose | Free-form text generation | Multi-turn conversation |
| Training | General text (books, articles, websites) | General text + Fine-tuned on conversation data |
| Memory | No concept of memory | Supports conversation history |
| Role Awareness | Cannot assign roles | Can assign roles (e.g., "You are a knowledgeable doctor") |
| Example Models | GPT-3, LLaMA, Falcon | GPT-3.5/4, Claude, Gemini, Llama-2 Chat |
| Best Use Cases | Text generation, summarization, translation, code generation | Conversational AI, chatbots, virtual assistants, customer support, AI tutors |

Currently, the AI industry is shifting towards Chat models, and most modern AI applications are built using Chat models rather than LLMs. LangChain now recommends using Chat models for new projects.

### Embedding Models

Embedding models convert text into vector representations (embeddings). These vectors capture the semantic meaning of the text and are particularly useful for:

- Semantic search
- Document similarity comparison
- Clustering similar texts
- Building RAG (Retrieval-Augmented Generation) applications

Embeddings are vectors (typically containing hundreds or thousands of dimensions) that represent the meaning of text in a way that computers can understand and compare.

## Working with Language Models

Let's start by exploring how to work with both types of language models in LangChain.

### Closed-Source Models

Closed-source models are proprietary models accessed through APIs, requiring payment for usage. We'll look at three major providers:

#### OpenAI Models

To use OpenAI models, you first need to create an account on the OpenAI platform and obtain an API key.

```python
# Example code for using OpenAI LLM
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load API key from .env file

# Create an LLM instance
llm = OpenAI(model="gpt-3.5-turbo-instruct")  # Using GPT-3.5 Turbo

# Invoke the model with a prompt
result = llm.invoke("What is the capital of India?")
print(result)  # Output: New Delhi
```

For Chat models:

```python
# Example code for using OpenAI Chat models
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Create a Chat model instance
model = ChatOpenAI(model="gpt-4")

# Invoke the model with a prompt
result = model.invoke("What is the capital of India?")
print(result.content)  # Notice we access the content attribute
```

You can also adjust parameters like:

- **temperature**: Controls randomness in model outputs (0.0-2.0)
  - 0-0.3: Deterministic outputs (good for factual tasks, code)
  - 0.5-0.7: Balanced creativity (good for Q&A, explanations)
  - 0.9-1.2: More creative outputs (good for writing, storytelling)
  - 1.5+: Very random/diverse outputs (brainstorming)

- **max_tokens**: Limits the length of the output

#### Anthropic Models

Anthropic provides Claude models, which are known for their strong performance and safety features.

```python
# Example code for using Anthropic's Claude
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

# Create a Claude chat model instance
model = ChatAnthropic(model="claude-3-opus-20240229")

# Invoke the model
result = model.invoke("What is the capital of India?")
print(result.content)
```

#### Google Models

Google provides Gemini (formerly PaLM) models through their API.

```python
# Example code for using Google's Gemini models
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Create a Gemini chat model instance
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Invoke the model
result = model.invoke("What is the capital of India?")
print(result.content)
```

### Open-Source Models

Open-source models are freely available AI models that can be downloaded, modified, fine-tuned, and deployed without restrictions. Unlike closed-source models such as GPT, Claude, and Gemini, open-source models offer full control and customization.

Key advantages of open-source models:

| Aspect | Open-Source Models | Closed-Source Models |
|--------|-------------------|----------------------|
| Cost | Free to use locally | Pay-per-token API usage |
| Control | Full control to modify and fine-tune | Limited to provider's infrastructure |
| Data Privacy | Data stays on your machine | Data sent to provider's servers |
| Customization | Can fine-tune on your data | Limited fine-tuning options |
| Deployment | Deploy on your own servers/cloud | Dependent on provider |

Popular open-source language models include:
- LLaMA
- Mistral
- Falcon
- Bloom

Most open-source models can be found on Hugging Face, which is the largest repository of open-source LLMs.

#### Using Hugging Face Inference API

You can use Hugging Face's inference API to access open-source models without downloading them:

```python
# Example for using Hugging Face API with LangChain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Create model using HF endpoint
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation"
    )
)

# Invoke the model
result = llm.invoke("What is the capital of India?")
print(result.content)
```

#### Running Models Locally

For complete control, you can download and run models locally:

```python
# Example of running a model locally
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

# Optional: For storage location control
os.environ["HF_HOME"] = "D:/huggingface_cache"  

# Create model using local pipeline
llm = ChatHuggingFace(
    llm=HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={"temperature": 0.7, "max_new_tokens": 100}
    )
)

# Invoke the model
result = llm.invoke("What is the capital of India?")
print(result.content)
```

**Disadvantages of local model inference:**
1. Requires solid hardware (especially for larger models)
2. Complicated setup process
3. Less refined responses compared to closed-source models
4. Limited multimodal abilities (mostly text-only)

## Working with Embedding Models

Embedding models convert text into vectors that capture meaning. These vectors are crucial for semantic search and RAG applications.

### OpenAI Embeddings

```python
# Example code for OpenAI embeddings (single query)
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Create embedding model instance
embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

# Generate embeddings for a query
query_embedding = embedding.embed_query("Delhi is the capital of India")
print(str(query_embedding))  # Vector with 32 dimensions
```

For multiple documents:

```python
# Example for embedding multiple documents
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

document_embeddings = embedding.embed_documents(documents)
# Returns a list of vectors, one for each document
```

### Open-Source Embeddings

You can also use open-source embedding models from Hugging Face:

```python
# Using Hugging Face embedding models
from langchain_huggingface import HuggingFaceEmbeddings

# Create embedding model instance
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Generate embeddings
text = "Delhi is the capital of India"
text_embedding = embedding.embed_query(text)
```

The cost of embedding generation is typically much lower than that of language model generation (about $0.0001 per million tokens for OpenAI embeddings).

## Building a Document Similarity Application

Let's create a simple document similarity application that uses embeddings to find the most relevant document for a given query:

```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# Create embedding model
embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

# Sample documents (about cricketers)
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive approach and consistent performance. He has been the former captain of the Indian cricket team.",
    "Jasprit Bumrah is an Indian fast bowler known for his unconventional bowling action and ability to bowl yorkers. He is considered one of the best fast bowlers in the world.",
    "Kane Williamson is a New Zealand cricketer known for his calm demeanor and technical batting skills. He has been captaining the New Zealand cricket team.",
    "Steve Smith is an Australian cricketer known for his unorthodox batting style. He has been one of the best Test batsmen in the world.",
    "Ben Stokes is an English all-rounder known for his match-winning performances with both bat and ball."
]

# Generate embeddings for all documents
document_embeddings = embedding.embed_documents(documents)

# User query
query = "Tell me about Virat Kohli"

# Generate embedding for the query
query_embedding = embedding.embed_query(query)

# Calculate similarity scores
# Reshape query embedding to be compatible with cosine_similarity
scores = cosine_similarity([query_embedding], document_embeddings)[0]

# Find the most similar document
# Create pairs of index and similarity score
indexed_scores = list(enumerate(scores))
# Sort by similarity score (descending)
sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
# Get index and score of the top result
index, score = sorted_scores[0]

# Print the result
print(f"Query: {query}")
print(f"Most similar document: {documents[index]}")
print(f"Similarity score: {score}")
```

This application:
1. Creates embeddings for a set of documents
2. Creates an embedding for the user query
3. Computes the similarity between the query and each document
4. Returns the most similar document

In a real-world RAG application, you would store these embeddings in a vector database for efficiency, but this example demonstrates the fundamental concept.

## Conclusion

In this guide, we've explored:

1. The two main types of models in LangChain: Language Models and Embedding Models
2. The difference between LLMs and Chat Models (with Chat Models being more modern and preferred)
3. How to work with closed-source models from OpenAI, Anthropic, and Google
4. How to use open-source models via Hugging Face's API or locally
5. How to generate embeddings using both proprietary and open-source models
6. How to build a simple document similarity application using embeddings

Understanding how to work with these different models in LangChain provides the foundation for building more complex applications like chatbots, RAG systems, and other AI-powered tools.

As LangChain continues to evolve, the focus is shifting more towards Chat models and away from LLMs, so it's recommended to use Chat models for new projects whenever possible.