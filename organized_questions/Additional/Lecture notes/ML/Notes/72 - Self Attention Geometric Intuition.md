# Self Attention — Geometric Intuition

## Recap: Self-Attention Flow

For a sentence like "money bank" with 2D embeddings:

### Step-by-Step Process

1. **Get embeddings:** $e_{\text{money}}$, $e_{\text{bank}}$
2. **Linear transform** with three weight matrices $W_q$, $W_k$, $W_v$:
   - $q_i = W_q \cdot e_i$, $\quad k_i = W_k \cdot e_i$, $\quad v_i = W_v \cdot e_i$
3. **Dot product** of query with all keys → similarity scores
4. **Scale** by $\frac{1}{\sqrt{d_k}}$ → scaled scores
5. **Softmax** → normalized attention weights
6. **Weighted sum** of value vectors → contextual embedding

---

## Geometric View: Step by Step

### Step 1: Embedding Vectors in Space

Word embeddings are **vectors in high-dimensional space**. For visualization, consider 2D:

- $e_{\text{money}} = [2, 6]$ — a vector pointing mostly in the second dimension
- $e_{\text{bank}} = [7, 3]$ — a vector pointing mostly in the first dimension

In a Word2Vec visualization:
- Semantically similar words **cluster together**
- Different meaning → far apart in the space

### Step 2: Linear Transformation (Q, K, V Generation)

When you multiply a vector by a matrix, you perform a **linear transformation**:
- The vector **moves** to a new position in space (rotation + scaling)
- Direction and magnitude both change

From $e_{\text{money}}$:

$$q_{\text{money}} = W_q \cdot e_{\text{money}} \quad \text{(vector transforms to new position)}$$

$$k_{\text{money}} = W_k \cdot e_{\text{money}} \quad \text{(different transform → different position)}$$

$$v_{\text{money}} = W_v \cdot e_{\text{money}} \quad \text{(yet another position)}$$

> **One embedding vector → three different vectors** in potentially different regions of the space, each optimized for its role (querying, being queried, contributing value).

Same process for $e_{\text{bank}}$ → $q_{\text{bank}}$, $k_{\text{bank}}$, $v_{\text{bank}}$.

**Result:** From 2 embedding vectors, we now have 6 vectors in the space.

### Step 3: Dot Product as Angular Similarity

The dot product between query and key vectors:

$$s_{ij} = q_i^T \cdot k_j = \|q_i\| \cdot \|k_j\| \cdot \cos(\theta)$$

**Geometric interpretation:**
- **Small angle** between vectors → $\cos(\theta)$ is large → **high similarity** → high dot product
- **Large angle** → $\cos(\theta)$ is small/negative → **low similarity** → low dot product

**Example** (computing $y_{\text{bank}}$):

| Dot Product | Vectors | Angle | Score |
|---|---|---|---|
| $q_{\text{bank}}^T \cdot k_{\text{money}}$ | Large angular distance | Low | ~10 |
| $q_{\text{bank}}^T \cdot k_{\text{bank}}$ | Small angular distance | High | ~32 |

### Step 4: Scaling by $\frac{1}{\sqrt{d_k}}$

$$s'_{ij} = \frac{s_{ij}}{\sqrt{d_k}}$$

Geometrically, this is just **rescaling** the similarity scores — it prevents very large values from causing softmax saturation (where one weight dominates completely).

**Example** ($d_k = 2$):

$$s'_{21} = \frac{10}{\sqrt{2}} \approx 7.09 \qquad s'_{22} = \frac{32}{\sqrt{2}} \approx 22.6$$

### Step 5: Softmax → Attention Weights

$$w_{21} \approx 0.2 \qquad w_{22} \approx 0.8$$

These weights determine **how much each word contributes** to the final contextual embedding.

### Step 6: Scalar × Vector → Scaling

$$0.2 \cdot v_{\text{money}} \quad \text{and} \quad 0.8 \cdot v_{\text{bank}}$$

**Geometrically:** Each value vector is **scaled down** — its direction stays the same but its magnitude is reduced proportionally to the attention weight.

- $v_{\text{money}}$ is scaled to 20% of its original length
- $v_{\text{bank}}$ is scaled to 80% of its original length

### Step 7: Vector Addition → Contextual Embedding

$$y_{\text{bank}} = 0.2 \cdot v_{\text{money}} + 0.8 \cdot v_{\text{bank}}$$

**Geometrically:** Apply the **parallelogram law of vector addition**:

1. Place both scaled vectors at the origin
2. Complete the parallelogram
3. The **diagonal** from the origin = resultant vector = $y_{\text{bank}}$

> The resulting contextual vector $y_{\text{bank}}$ lies **between** the two scaled value vectors, pulled more toward whichever had the higher weight.

---

## The Key Geometric Insight: Self-Attention as Gravity

### Before Self-Attention

```
e_money ●                        ● e_bank
        (far apart — different static embeddings)
```

### After Self-Attention

```
e_money ●       ● y_bank
        (y_bank has moved CLOSER to e_money!)
```

The contextual embedding $y_{\text{bank}}$ is **pulled toward** the context word "money" compared to the original $e_{\text{bank}}$.

> **Self-attention acts like gravity** — context words pull the target word's embedding toward themselves.

### Context-Dependent Behavior

| Sentence | Context Word | Where $y_{\text{bank}}$ Moves |
|----------|-------------|------------------------------|
| **money** bank | money | Toward the "finance" region |
| **river** bank | river | Toward the "geography" region |

**Same word, different context → different position in embedding space.**

This is the geometric proof that self-attention creates genuinely **context-aware** representations.

---

## Visual Summary

```
Original embeddings:
    e_money ●                              ● e_bank
                    (static, fixed)

After Self-Attention on "money bank":
    e_money ●           ● y_bank
                (pulled toward money's region)

After Self-Attention on "river bank":
    e_river ●           ● y_bank
                (pulled toward river's region)
```

### Full Pipeline (Geometric)

$$\underbrace{e_i}_{\text{point in space}} \xrightarrow{W_q, W_k, W_v} \underbrace{q_i, k_i, v_i}_{\text{3 transformed points}} \xrightarrow{\text{dot product}} \underbrace{s_{ij}}_{\text{angular similarity}} \xrightarrow{\text{softmax}} \underbrace{w_{ij}}_{\text{pull strength}} \xrightarrow{\text{weighted sum}} \underbrace{y_i}_{\text{new position}}$$

---

## Summary

| Concept | Geometric Interpretation |
|---------|------------------------|
| **Embedding vector** | A point/arrow in high-dimensional space |
| **Linear transform** ($W_q, W_k, W_v$) | Rotation + scaling → moves vector to new location |
| **Dot product** ($q^T k$) | Measures angular closeness between vectors |
| **Scaling** ($\div \sqrt{d_k}$) | Shrinks scores to prevent softmax saturation |
| **Softmax** | Converts scores to pull-strength proportions (sum = 1) |
| **Scalar × vector** | Shrinks value vector proportionally |
| **Vector addition** | Combines all contributions (parallelogram law) |
| **Final $y_i$** | New vector position — pulled toward relevant context words |
| **Self-attention overall** | Acts like **gravity** — context words attract the target word's embedding |
