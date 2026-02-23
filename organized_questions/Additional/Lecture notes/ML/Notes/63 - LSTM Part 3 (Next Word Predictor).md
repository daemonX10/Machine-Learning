# LSTM Part 3 — Next Word Predictor (Practical Project)

## Overview

- **Task:** Build a **Next Word Predictor** using LSTM in Keras/TensorFlow
- **Approach:** Convert text generation into a **supervised multi-class classification** problem
- **Dataset:** A small FAQ document (for demonstration purposes)
- **Real-world examples:** SwiftKey keyboard, Gmail Smart Compose, code completion tools

---

## What is a Next Word Predictor?

A system that, given a sequence of words, predicts the **most probable next word**.

> Essentially, you're building a **text generator** — give it some text, it generates the next word, then the next, and so on (like ChatGPT, but simpler).

---

## Strategy: Text Generation as Supervised Learning

### Core Idea

Convert the text generation task into a **supervised learning** problem:
- **Input:** A sequence of words
- **Output:** The next word

### Building the Dataset from Text

Given a sentence: *"Hi my name is Nitish"*

| Input Sequence | Output (Next Word) |
|---------------|-------------------|
| Hi | my |
| Hi my | name |
| Hi my name | is |
| Hi my name is | Nitish |

For a second sentence: *"I live in Gurgaon"*

| Input Sequence | Output |
|---------------|--------|
| I | live |
| I live | in |
| I live in | Gurgaon |

Repeat for **every sentence** in the corpus → creates a large training dataset.

---

## Step-by-Step Implementation

### Step 1: Tokenization — Convert Words to Numbers

Models don't understand English — they need **numbers**.

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])

# View word → index mapping
tokenizer.word_index
# {'the': 1, 'program': 13, 'about': 93, ...}
```

Each unique word gets a **unique integer index** (starting from 1, not 0).

### Step 2: Convert Sentences to Sequences of Numbers

```python
for sentence in text_data.split('\n'):
    token_sequence = tokenizer.texts_to_sequences([sentence])[0]
```

**Example:** "about the program" → `[93, 1, 13]`

### Step 3: Create Input-Output Pairs (N-grams)

For each tokenized sentence, create progressively longer subsequences:

```python
input_sequences = []
for sentence in text_data.split('\n'):
    token_sentence = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_sentence)):
        n_gram = token_sentence[:i+1]
        input_sequences.append(n_gram)
```

**Example:** Sentence `[93, 1, 13]` generates:
- `[93, 1]` (input: 93, output: 1)
- `[93, 1, 13]` (input: [93, 1], output: 13)

### Step 4: Zero Padding

Different sequences have different lengths → pad them to **equal length** (required for batch training):

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_length = max(len(x) for x in input_sequences)  # e.g., 57
padded = pad_sequences(input_sequences, maxlen=max_length, padding='pre')
```

**Pre-padding** adds zeros at the **beginning**:
```
[0, 0, 0, ..., 93, 1]      # padded 2-word sequence
[0, 0, 0, ..., 93, 1, 13]  # padded 3-word sequence
```

### Step 5: Split into X (Input) and y (Output)

```python
X = padded[:, :-1]  # all columns except last
y = padded[:, -1]   # last column only
```

### Step 6: One-Hot Encode the Output (Multi-Class Classification)

> **Why classification, not regression?**
> Output is **discrete** — each number maps to a specific word. A model predicting 2.7 is meaningless (no word at index 2.7).

```python
from tensorflow.keras.utils import to_categorical

total_words = len(tokenizer.word_index) + 1  # +1 because indices start at 1
y = to_categorical(y, num_classes=total_words)
```

- `y` shape becomes: `(num_samples, total_words)` — e.g., `(863, 283)`
- Each row is a **one-hot vector** with a 1 at the position of the target word

---

## Model Architecture

Three layers: **Embedding → LSTM → Dense (Softmax)**

### Architecture Diagram

```
Input (56 integers)
       ↓
┌─────────────┐
│  Embedding   │  → Each of 56 words → 100-dim dense vector
│  (283, 100)  │     Output shape: (56, 100)
└──────┬──────┘
       ↓
┌─────────────┐
│    LSTM      │  → 150 hidden units
│   (150)      │     Processes 56 time steps, outputs h_56
└──────┬──────┘     Output shape: (150,)
       ↓
┌─────────────┐
│Dense (283)   │  → One node per word, softmax activation
│  Softmax     │     Output shape: (283,) — probabilities
└──────┬──────┘
       ↓
  Predicted Next Word = argmax(output)
```

### Layer Details

| Layer | Purpose | Output Shape |
|-------|---------|-------------|
| **Embedding(283, 100, input_length=56)** | Converts each token to a 100-D dense vector | (56, 100) |
| **LSTM(150)** | Processes sequence across 56 time steps | (150,) |
| **Dense(283, activation='softmax')** | Outputs probability for each word | (283,) |

### Code

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_length - 1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()
```

### Parameter Breakdown

| Layer | Parameters |
|-------|-----------|
| Embedding | $283 \times 100 = 28{,}300$ |
| LSTM(150) | $4 \times [150 \times (150 + 100) + 150] = 150{,}600$ |
| Dense(283) | $150 \times 283 + 283 = 42{,}733$ |

---

## Training

```python
model.fit(X, y, epochs=100)
# After 100 epochs: training accuracy ≈ 94%
```

> Training accuracy of ~94% is expected on small data. Validation accuracy would be lower (overfitting on small corpus).

---

## Prediction Pipeline

### Predict a Single Next Word

```python
import numpy as np

text = "mail"

# Step 1: Tokenize
token_text = tokenizer.texts_to_sequences([text])[0]

# Step 2: Pad
padded_token = pad_sequences([token_text], maxlen=max_length - 1, padding='pre')

# Step 3: Predict
prediction = model.predict(padded_token)
position = np.argmax(prediction)

# Step 4: Map index → word
for word, index in tokenizer.word_index.items():
    if index == position:
        print(word)  # Output: "us"
```

### Generate Multiple Words (Auto-regressive)

```python
text = "mail"
for i in range(5):
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_token = pad_sequences([token_text], maxlen=max_length - 1, padding='pre')
    prediction = model.predict(padded_token)
    position = np.argmax(prediction)
    
    for word, index in tokenizer.word_index.items():
        if index == position:
            text = text + " " + word
            break

    print(text)
```

**Example output:**
```
mail us
mail us you
mail us you make
mail us you make the
mail us you make the payment
```

Each predicted word is **appended** to the input, and the model predicts the next — this is **auto-regressive generation**.

---

## How to Improve the Model

| Strategy | Details |
|----------|---------|
| **More data** | Use larger datasets (jokes, quotes, TV scripts from Kaggle) |
| **Hyperparameter tuning** | Adjust LSTM units, optimizer, learning rate, epochs |
| **Advanced architectures** | Stacked LSTMs, Bidirectional LSTMs, Transformers (GPT, BERT) |
| **Validation set** | Split data to monitor overfitting via validation accuracy |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Task type** | Multi-class classification (not regression) |
| **Key conversion** | Text generation → Supervised learning via n-gram input/output pairs |
| **Padding** | Pre-padding with zeros to equalize sequence lengths |
| **Output encoding** | One-hot encoding of target word |
| **Architecture** | Embedding → LSTM → Dense (softmax) |
| **Loss function** | Categorical cross-entropy |
| **Generation method** | Auto-regressive: predict next word, append, repeat |
| **Limitation** | Small data = poor generalization; works well on training data |
