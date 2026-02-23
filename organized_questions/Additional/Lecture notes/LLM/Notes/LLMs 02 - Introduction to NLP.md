# Lecture 02 — Introduction to Natural Language Processing

---

## 1. What is NLP?

**Natural Language Processing (NLP)** — a field at the intersection of **Computer Science**, **AI**, and **Computational Linguistics** concerned with interactions between computers and human languages.

**Natural language** — a language that evolved naturally (without predefined grammar). Grammar was formalized *after* the language emerged. Contrast with artificial/programming languages where grammar is defined *first*.

---

## 2. Why NLP is Challenging

### 2.1 Language Diversity
- **6,000+** official languages worldwide
- **19,000+** languages/dialects in India alone
- Enormous dialectal variation even within one language

### 2.2 Ambiguity — The Core Challenge

| Type | Example | Issue |
|------|---------|-------|
| **Lexical** | "I saw a girl with a telescope" | Who has the telescope? |
| **Syntactic** | "Mary had a little lamb" | Pet lamb or ate lamb? |
| **Semantic** | "Virat Kohli was on fire last night" | Literal vs. figurative meaning |
| **Pragmatic** | "Do you know what time it is?" | Asking time vs. implying lateness |
| **Punctuation** | "Let's eat Grandma" vs. "Let's eat, Grandma" | Comma changes meaning entirely |

### 2.3 The Buffalo Sentence

> "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo."

This is a **valid English sentence** using three meanings of "buffalo":
1. **Buffalo** (noun) — the animal
2. **Buffalo** (proper noun) — city in New York
3. **buffalo** (verb) — to bully/intimidate

Meaning: *Buffalo-city bison whom other Buffalo-city bison bully, themselves bully Buffalo-city bison.*

### 2.4 Other Challenges

| Challenge | Example |
|-----------|---------|
| **Non-standard English** | "LOL", "imo", "tbh" |
| **Segmentation** | "New York-New Haven Railway" — space-based tokenization fails |
| **Multiword expressions** | "hot dog", phrasal verbs (kick the bucket) |
| **Neologisms** | "unfriend", "retweet" |
| **World knowledge** | Parser tagged "Michelle Obama" as wife of Jeb Bush |
| **Tricky named entities** | "Let It Be" — one entity (song) vs. three tokens |

---

## 3. NLP Layers (Bottom to Top — Increasing Complexity)

| Layer | Deals With | Key Task |
|-------|-----------|----------|
| **Morphology** | Word structure (prefix, suffix, root, stem) | Stemming, Lemmatization |
| **Syntax** | Sentence structure (grammar) | POS tagging, Chunking, Parsing |
| **Semantics** | Meaning of words & sentences | WSD, SRL, distributional similarity |
| **Pragmatics** | Context-dependent meaning | Dialogue, intent understanding |
| **Discourse** | Cross-sentence / conversation context | Coreference resolution, coherence |

---

## 4. Morphology

Study of **word formation** — how words are composed from smaller meaningful units (**morphemes**).

- "independently" → **in** + **depend** + **ent** + **ly**
- Tools: Finite-state automata, morphological rules

### Stemming vs. Lemmatization

| Feature | Stemming | Lemmatization |
|---------|----------|---------------|
| Output | May not be a dictionary word | Always a dictionary word |
| Speed | Fast (rule-based) | Slower (needs dictionary) |
| Consistency | Same stem for related forms | Correct root form |
| Example | "running" → "run" / "runn" | "running" → "run" |

> **Why stemming?** It guarantees that two surface words from the same lemma always map to the same stem — useful for grouping.

### Morphological Richness

| Language | Richness | Example |
|----------|----------|---------|
| English, Chinese | **Morphologically poor** | Verbs barely change |
| Hindi, Hungarian | **Morphologically rich** | Verbs encode gender, number, tense |

---

## 5. Syntax

### 5.1 POS Tagging

Assigning grammatical categories to each word.

| Tag Set | Number of Tags |
|---------|---------------|
| Penn Treebank (PTB) | 45 tags (PRP, VBD, DT, NN, ...) |
| C7 tag set | 146 tags |

**Ambiguity in POS**: "He went to the **park** in a car" (NN) vs. "They went to **park** the car" (VB)

### 5.2 Chunking

Grouping tokens into **phrases** (noun phrase, verb phrase, prepositional phrase).

Example: `[NP Mumbai green lights women icons] [PP on] [NP traffic signals] [VP earns] [NP global praise]`

### 5.3 Parsing

Building a **parse tree** (constituency tree) from a sentence to reveal grammatical structure.

- Resolves structural ambiguity (e.g., telescope example — different tree structures yield different meanings)

---

## 6. Semantics

Three approaches to understanding word meaning:

### 6.1 Decompositional Semantics
Break entities into **features**:

| Entity | Human | Female | Adult |
|--------|-------|--------|-------|
| Boy | ✓ | ✗ | ✗ |
| Girl | ✓ | ✓ | ✗ |
| Man | ✓ | ✗ | ✓ |
| Woman | ✓ | ✓ | ✓ |

### 6.2 Ontological Semantics

Use structured knowledge bases like **WordNet**:
- Entities connected through semantic relations
- Relations: **synonym**, **antonym**, **hypernym**, **hyponym**, **meronym**, **holonym**
- Example: "car" **is-a** "motor vehicle" **is-a** "wheeled vehicle"
- Problem: Manual creation is expensive, hard to keep updated

### 6.3 Distributional Semantics ⭐

> *"You shall know a word by the company it keeps."* — J.R. Firth (1957)

- Build a **co-occurrence matrix**: rows = words, columns = words, entries = co-occurrence count within a sliding window
- Each row = a **vector representation** of a word
- Similarity between words = **cosine similarity** of their vectors

| | pet | owner | run | bark |
|-----|-----|-------|-----|------|
| dog | 3 | 5 | 4 | 0 |
| cat | 4 | 5 | 3 | 0 |
| car | 0 | 1 | 0 | 0 |

→ dog and cat have similar vectors → **semantically similar**

This is the foundation of all modern word embedding methods (Word2Vec, GloVe, etc.)

---

## 7. Pragmatics & Discourse

### Pragmatics
> *"The negotiation of meaning between speaker and listener."*

- Meaning depends on **context**, **tone**, **gesture**
- "Can you pass the water bottle?" → Expected: pass it. Not: "Yes I can."

### Discourse
- Understanding meaning across **multiple sentences/turns** in a dialogue
- Models must track context over conversations (ChatGPT session context)

---

## 8. Core NLP Tasks

| Task | Description |
|------|-------------|
| **Semantic Role Labeling (SRL)** | Assign roles (agent, patient, source, destination) to words |
| **Textual Entailment** | Does sentence A entail sentence B? (positive / negative / neutral) |
| **Coreference Resolution** | Resolving "he", "it", "this" to their referents |
| **Information Extraction** | Extract entities, relations, events from text |
| **Named Entity Recognition (NER)** | Tag words as Person, Location, Organization, Time, etc. |
| **Word Sense Disambiguation (WSD)** | "bank" → financial institution or river bank? |
| **Sentiment Analysis** | Classify text as positive / negative / neutral |
| **Emotion Detection** | Classify into 6 emotions (happy, sad, angry, etc.) |
| **Machine Translation** | Translate between languages |
| **Text Summarization** | Extractive, abstractive, or aspect-based |
| **Question Answering** | Factoid, list, or descriptive questions |
| **Dialogue Systems** | Chatbots, conversational AI |

### Summarization Types

| Type | Method |
|------|--------|
| **Extractive** | Select important sentences, copy-paste |
| **Abstractive** | Understand meaning, rephrase/generate new text |
| **Aspect-based** | Summarize only w.r.t. a specific aspect (e.g., battery life) |

---

## 9. NLP Trinity

Three axes of NLP research:

| Axis | Examples |
|------|----------|
| **Languages** | Hindi, English, French, Marathi, ... |
| **Tasks** | POS tagging, parsing, SRL, NER, ... |
| **Algorithms** | HMM, CRF, MEMM, DNN, LLMs, ... |

NLP research explores combinations across these three dimensions.

---

## 10. Key Takeaways

- NLP is fundamentally challenging due to **ambiguity** at every linguistic level
- Understanding progresses from **surface form** (morphology, syntax) to **deep meaning** (semantics, pragmatics)
- **Distributional semantics** is the foundational principle behind modern NLP
- LLMs must handle all these layers to truly "understand" language
