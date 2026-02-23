# Cross Attention

## Overview

- **Definition:** A mechanism in transformer architectures that computes attention **between two different sequences** (e.g., input and output), unlike self-attention which operates within a single sequence
- **Also called:** Encoder-Decoder Attention
- **Purpose:** Allows the decoder to focus on relevant parts of the input sequence when generating each token of the output sequence

---

## Why Cross Attention Is Needed

When generating the next output word in a decoder, the prediction depends on **two things**:

| Factor | Source | Mechanism |
|--------|--------|-----------|
| What has been generated so far (decoder history) | Decoder's own previous outputs | **Self-Attention** |
| What the input sentence contains | Encoder's processed representations | **Cross Attention** |

> Self-attention captures intra-sequence relationships. Cross attention captures **inter-sequence** relationships.

---

## Self-Attention vs Cross Attention

### Input Difference

| Aspect | Self-Attention | Cross Attention |
|--------|---------------|-----------------|
| **Number of input sequences** | 1 (single sequence) | 2 (input + output sequences) |
| **Example** | "We are friends" → embeddings of all 3 words | "We are friends" (encoder) + "हम दोस्त हैं" (decoder) |

### Processing Difference — How Q, K, V Are Computed

**Self-Attention:**

All three vectors come from the **same sequence**:

$$Q = X \cdot W_Q, \quad K = X \cdot W_K, \quad V = X \cdot W_V$$

Where $X$ is the embedding matrix of the single input sequence.

**Cross Attention:**

Vectors come from **two different sequences**:

$$Q = X_{\text{decoder}} \cdot W_Q$$

$$K = X_{\text{encoder}} \cdot W_K, \quad V = X_{\text{encoder}} \cdot W_V$$

| Vector | Source |
|--------|--------|
| **Query (Q)** | Output sequence (decoder side, e.g., Hindi words) |
| **Key (K)** | Input sequence (encoder side, e.g., English words) |
| **Value (V)** | Input sequence (encoder side, e.g., English words) |

> This is the **biggest difference**: Q comes from the decoder, K and V come from the encoder.

### Output Difference

| Aspect | Self-Attention | Cross Attention |
|--------|---------------|-----------------|
| **Output count** | One contextual embedding per token in the input sequence | One contextual embedding per token in the **output** sequence |
| **Output represents** | Each word's representation enriched by context from the same sequence | Each output word's representation enriched by context from the **input** sequence |

---

## Step-by-Step Computation

**Example:** English → Hindi translation  
Input (encoder): "We are friends"  
Output (decoder): "हम दोस्त हैं"

### Step 1: Compute Q, K, V

For each **Hindi** word (decoder side), compute a **query vector**:

$$q_{\text{हम}} = e_{\text{हम}} \cdot W_Q, \quad q_{\text{दोस्त}} = e_{\text{दोस्त}} \cdot W_Q, \quad q_{\text{हैं}} = e_{\text{हैं}} \cdot W_Q$$

For each **English** word (encoder side), compute **key and value vectors**:

$$k_{\text{we}} = e_{\text{we}} \cdot W_K, \quad v_{\text{we}} = e_{\text{we}} \cdot W_V$$

$$k_{\text{are}} = e_{\text{are}} \cdot W_K, \quad v_{\text{are}} = e_{\text{are}} \cdot W_V$$

$$k_{\text{friends}} = e_{\text{friends}} \cdot W_K, \quad v_{\text{friends}} = e_{\text{friends}} \cdot W_V$$

### Step 2: Compute Attention Scores

Dot product of each query with all keys:

$$\text{score}(q_i, k_j) = q_i \cdot k_j^T$$

This produces a matrix of shape (output_tokens × input_tokens):

|  | We | Are | Friends |
|---|---|---|---|
| **हम** | high | low | low |
| **दोस्त** | low | low | high |
| **हैं** | low | high | low |

### Step 3: Scale and Softmax

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Step 4: Weighted Sum with Value Vectors

Each attention weight multiplied by the corresponding value vector, then summed:

$$\text{CE}_{\text{हम}} = w_1 \cdot v_{\text{we}} + w_2 \cdot v_{\text{are}} + w_3 \cdot v_{\text{friends}}$$

> The output contextual embeddings capture how each output token relates to the input tokens.

---

## Connection to Bahdanau / Luong Attention

Cross attention in transformers is conceptually the **same idea** as Bahdanau and Luong attention from RNN-based encoder-decoder models:

| Aspect | Bahdanau/Luong Attention | Transformer Cross Attention |
|--------|-------------------------|----------------------------|
| **Goal** | Find which encoder hidden state is most relevant to current decoder step | Find which encoder representations are most relevant to current decoder token |
| **Formula** | $c_i = \sum_j \alpha_{ij} h_j$ | $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$ |
| **Similarity computation** | Neural network (Bahdanau) or dot product (Luong) | Scaled dot product |
| **Architecture** | RNN/LSTM-based | Transformer-based (no recurrence) |

From the original paper ("Attention Is All You Need"):

> *"In encoder-decoder attention, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanism in sequence-to-sequence models."*

---

## Applications of Cross Attention

Cross attention is used wherever **two different sequences or modalities** need to interact:

| Application | Input Modality | Output Modality |
|-------------|---------------|-----------------|
| Machine Translation | Source language text | Target language text |
| Question Answering | Question text | Answer text |
| Image Captioning | Image features | Text caption |
| Text-to-Image Generation | Text prompt | Image |
| Text-to-Speech | Text | Audio waveform |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it does** | Computes attention between two different sequences |
| **Q source** | Decoder (output sequence) |
| **K, V source** | Encoder (input sequence) |
| **Key formula** | $\text{softmax}\left(\frac{Q_{\text{dec}} \cdot K_{\text{enc}}^T}{\sqrt{d_k}}\right) V_{\text{enc}}$ |
| **Conceptual ancestor** | Bahdanau/Luong attention |
| **Output** | One contextual embedding per output sequence token |
| **Use cases** | Translation, QA, multimodal tasks (image captioning, text-to-image, TTS) |
