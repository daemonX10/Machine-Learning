# RNN Sentiment Analysis — Keras Implementation

## Overview

Two approaches to build a sentiment analysis model using SimpleRNN in Keras:

| Approach | Text Representation | Result Quality |
|----------|---------------------|----------------|
| **Integer Encoding + Padding** | Sparse, integer sequences | Lower accuracy |
| **Embedding Layer** | Dense, learned vectors | Higher accuracy (~80%+) |

---

## Text Preprocessing Pipeline

### Step 1: Tokenization

Convert raw text into word-level tokens and build a vocabulary with integer indices.

```python
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(oov_token='<nothing>')  # handles unseen words
tokenizer.fit_on_texts(documents)

# View vocabulary
print(tokenizer.word_index)    # {'india': 1, 'jeetega': 2, ...}
print(tokenizer.word_counts)   # frequency of each word
```

| Attribute | Purpose |
|-----------|---------|
| `word_index` | Maps each unique word → integer index |
| `word_counts` | Frequency count of each word |
| `oov_token` | Token used for out-of-vocabulary words during prediction |

### Step 2: Text to Sequences (Integer Encoding)

Replace each word with its integer index:

```python
sequences = tokenizer.texts_to_sequences(documents)
# "go india" → [10, 2]
```

### Step 3: Padding

Make all sequences the same length by adding zeros:

```python
from keras.utils import pad_sequences

padded = pad_sequences(sequences, padding='post')  # zeros at end
# or padding='pre' for zeros at beginning
```

- **`padding='post'`**: Zeros added **after** the sequence
- **`padding='pre'`**: Zeros added **before** the sequence
- **`maxlen`**: Truncate/pad all sequences to this fixed length

---

## Approach 1: Integer Encoding + SimpleRNN

### Dataset: IMDB Reviews (Keras built-in)

```python
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data()
# x_train.shape = (25000,) — 25K reviews, already integer-encoded
# x_test.shape  = (25000,)
```

> The IMDB dataset is **pre-tokenized and integer-encoded** — no manual tokenization needed.

### Padding the Reviews

```python
from keras.utils import pad_sequences

x_train = pad_sequences(x_train, maxlen=50, padding='post')
x_test  = pad_sequences(x_test,  maxlen=50, padding='post')
# Each review now has exactly 50 integers
# Shape: (25000, 50)
```

### Model Architecture

```python
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

model = Sequential()
model.add(SimpleRNN(32, input_shape=(50, 1), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```

#### Architecture Diagram

```
Input (50 time steps × 1 feature)
         ↓
   [SimpleRNN — 32 units]
         ↓
   [Dense — 1 unit, sigmoid]
         ↓
     Output (0 or 1)
```

#### Parameter Count

| Layer | Parameters | Calculation |
|-------|-----------|-------------|
| SimpleRNN(32) | $32 \times 1 + 32 \times 32 + 32 = 1088$ | $W_{in}(32) + W_h(1024) + b(32)$ |
| Dense(1) | $32 + 1 = 33$ | weights(32) + bias(1) |
| **Total** | **1121** | |

### `return_sequences` Parameter

| Value | Behavior | Use Case |
|-------|----------|----------|
| `False` (default) | Only the **last time step's** output is returned | Sentiment Analysis (Many-to-One) |
| `True` | Output at **every time step** is returned | NER, Machine Translation (Many-to-Many) |

---

## Approach 2: Embedding Layer + SimpleRNN

### Why Use Embeddings?

| Problem with Integer Encoding | Embedding Solution |
|-------------------------------|-------------------|
| **Sparse representation** — after padding, most values are zeros | **Dense representation** — all values are non-zero |
| **High dimensionality** — vector size = vocabulary size | **Low dimensionality** — configurable (e.g., 2, 32, 128) |
| **No semantic meaning** — numbers are arbitrary indices | **Captures semantic meaning** — similar words have similar vectors |

### How the Embedding Layer Works

The embedding layer is a **trainable lookup table**:

1. Input: integer-encoded word index
2. Output: dense vector of specified dimension
3. Internally, it's a weight matrix of shape `(vocab_size, embedding_dim)`
4. Weights are **randomly initialized** and **learned during training** via backpropagation

```
Vocabulary size = V, Embedding dimension = d

Embedding Matrix: V × d
Input word index i → Row i of the matrix → d-dimensional vector
```

#### Embedding Layer Parameters

```python
from keras.layers import Embedding

Embedding(input_dim, output_dim, input_length)
```

| Parameter | Meaning |
|-----------|---------|
| `input_dim` | Vocabulary size (number of unique words) |
| `output_dim` | Embedding dimension (size of dense vector per word) |
| `input_length` | Length of input sequences (number of time steps) |

### Model Architecture with Embedding

```python
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=2, input_length=50))
model.add(SimpleRNN(32, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```

#### Architecture Diagram

```
Input (50 integers)
         ↓
   [Embedding — 10000 × 2]  →  Output: (50, 2)
         ↓
   [SimpleRNN — 32 units]
         ↓
   [Dense — 1 unit, sigmoid]
         ↓
     Output (0 or 1)
```

#### Parameter Count

| Layer | Parameters | Calculation |
|-------|-----------|-------------|
| Embedding | $10000 \times 2 = 20000$ | vocab_size × embedding_dim |
| SimpleRNN(32) | $(2 \times 32) + (32 \times 32) + 32 = 1120$ | $W_{in} + W_h + b$ |
| Dense(1) | $32 + 1 = 33$ | weights + bias |
| **Total** | **21153** | |

### Results Comparison

| Approach | Training Accuracy | Notes |
|----------|-------------------|-------|
| Integer Encoding | Low (~55-60%) | Sparse, no semantic info |
| Embedding Layer | ~80%+ | Dense, semantic meaning captured |

---

## Embedding: Custom vs Pre-trained

| Option | Description | When to Use |
|--------|-------------|-------------|
| **Train with model** | Embedding layer learns during training | Dataset-specific, generally better for domain tasks |
| **Pre-trained (Word2Vec, GloVe)** | Use embeddings trained on large corpora | Small dataset, need general semantic knowledge |

> Training embeddings **with your model** tends to give better results because the vectors are **customized to your specific dataset** and capture its semantic nuances.

---

## Summary

| Aspect | Approach 1 (Integer) | Approach 2 (Embedding) |
|--------|---------------------|------------------------|
| Representation | Sparse integers | Dense vectors |
| Semantic meaning | None | Captured |
| Dimensionality | High (vocab size) | Low (configurable) |
| Accuracy | Lower | Higher |
| Extra layer | None | Embedding layer |
