# Types of RNN Architectures

## Overview

RNN architectures are classified based on the **nature of input and output** — whether they are sequential (many) or non-sequential (one).

| Type | Input | Output | Example |
|------|-------|--------|---------|
| **Many-to-One** | Sequence | Single value | Sentiment Analysis |
| **One-to-Many** | Single value | Sequence | Image Captioning |
| **Many-to-Many** | Sequence | Sequence | Machine Translation |
| **One-to-One** | Single value | Single value | Image Classification (not truly RNN) |

> **Technically, only 3 types are true RNN architectures** — One-to-One has no recurrence and is just a standard ANN/CNN.

---

## 1. Many-to-One RNN

### Concept

- **Input:** Sequential data (e.g., words in a sentence)
- **Output:** A single scalar/class label
- The network processes the full input sequence and produces output only at the **last time step**

### Applications

| Application | Input | Output |
|-------------|-------|--------|
| **Sentiment Analysis** | Text (sequence of words) | Positive/Negative (1 or 0) |
| **Rating Prediction** | Movie review text | Star rating (1–5) |

### Architecture (Unfolded)

```
x₁ → [RNN] → o₁ ──→ [RNN] → o₂ ──→ ... ──→ [RNN] → oₜ → ŷ
         ↑              ↑                        ↑
        h₀             h₁                      hₜ₋₁
```

- Input is fed **word by word** at each time step
- Hidden state is passed forward at every step
- **Only the last time step's output** is used for final prediction ($\hat{y}$)

---

## 2. One-to-Many RNN

### Concept

- **Input:** Non-sequential data (e.g., a single image or number)
- **Output:** Sequential data (e.g., a sequence of words)
- A single input is provided once, and the network generates output over multiple time steps

### Applications

| Application | Input | Output |
|-------------|-------|--------|
| **Image Captioning** | Image (non-sequential) | Text description ("A man is playing cricket") |
| **Music Generation** | A seed note/value | Sequence of musical notes |

### Architecture (Unfolded)

```
x → [RNN] → o₁ → [RNN] → o₂ → [RNN] → o₃ → ...
                    ↑              ↑
                   y₁             y₂
```

- Input is provided **only once** at the first time step
- Each time step produces an output
- Output from each step feeds back as input to the next step

---

## 3. Many-to-Many RNN

### Concept

- **Input:** Sequential data
- **Output:** Sequential data
- Also called **Sequence-to-Sequence (Seq2Seq)** models

### Two Sub-Types

| Sub-Type | Condition | Example |
|----------|-----------|---------|
| **Same Length** | $\|input\| = \|output\|$ | POS Tagging, Named Entity Recognition |
| **Variable Length** | $\|input\| \neq \|output\|$ | Machine Translation |

---

### 3a. Same-Length Many-to-Many

- Input and output sequences have the **same number of elements**
- Each input at time step $t$ produces an output at the same time step $t$

#### Applications

| Application | Description |
|-------------|-------------|
| **POS Tagging** | For every word → predict its part of speech (noun, verb, etc.) |
| **Named Entity Recognition (NER)** | For every word → determine if it's an entity or not |

#### Architecture

```
x₁ → [RNN] → y₁
        ↓
x₂ → [RNN] → y₂
        ↓
x₃ → [RNN] → y₃
```

- Every time step receives an input **and** produces an output
- Output at each step also feeds back as hidden state to the next step

---

### 3b. Variable-Length Many-to-Many (Encoder-Decoder)

- Input and output sequences can have **different lengths**
- The architecture is split into two parts:
  - **Encoder:** Processes the entire input sequence (no output produced)
  - **Decoder:** Generates the output sequence after encoding is complete

#### Application: Machine Translation

| Language | Sentence | Word Count |
|----------|----------|------------|
| English (input) | "Hey my name is Nitish" | 5 |
| Hindi (output) | "Mera naam Nitish hai" | 4 |

> No guarantee that input and output will have the same number of words across languages.

#### Architecture

```
Encoder                          Decoder
x₁ → [RNN] → x₂ → [RNN] → x₃ → [RNN] → x₄ → [RNN] ──→ [RNN] → y₁ → [RNN] → y₂ → ...
```

- **Encoder phase:** All inputs are fed sequentially; no output is produced
- **Decoder phase:** Once the last input is processed, output generation begins
- The encoder's final hidden state serves as the **context vector** for the decoder

#### Why Wait for Full Input?

Translation requires understanding the **full sentence** — grammar, context, and meaning — before producing output. Word-by-word translation would lose grammatical structure and context.

---

## 4. One-to-One (Not Truly RNN)

- **Input:** Non-sequential (e.g., image)
- **Output:** Non-sequential (e.g., class label)
- **No recurrence** — single time step, no feedback loop
- This is just a standard **ANN or CNN**

| Example | Input | Output |
|---------|-------|--------|
| Image Classification | Image | Cat/Dog (1 or 0) |

---

## Summary

| Architecture | Input → Output | Recurrence | Key Application |
|-------------|---------------|------------|-----------------|
| Many-to-One | Sequence → Scalar | Yes | Sentiment Analysis |
| One-to-Many | Scalar → Sequence | Yes | Image Captioning |
| Many-to-Many (Same) | Sequence → Sequence (same length) | Yes | POS Tagging, NER |
| Many-to-Many (Variable) | Sequence → Sequence (different length) | Yes (Encoder-Decoder) | Machine Translation |
| One-to-One | Scalar → Scalar | No | Image Classification (ANN/CNN) |
