# Lecture 23 — Dropout (Theory)

## 1. Overfitting in Neural Networks

Neural networks are **prone to overfitting** because:
- Complex architectures (multiple layers × multiple neurons per layer)
- Fully connected layers create **massive number of connections**
- The network has enough capacity to memorize **every small pattern** in training data
- Result: excellent training accuracy, poor test accuracy

### Possible Solutions to Overfitting

| # | Solution | Description |
|---|---|---|
| 1 | **Add more data** | More diverse data → better generalization |
| 2 | **Reduce network complexity** | Fewer layers, fewer neurons per layer |
| 3 | **Early stopping** | Stop training when validation loss starts increasing |
| 4 | **Regularization** (L1/L2) | Penalize large weights |
| 5 | **Dropout** | Randomly disable neurons during training |

---

## 2. What Is Dropout?

> **Dropout** randomly **turns off** (drops) a fraction of neurons in each layer during each training iteration.

- Proposed by **Hinton & Srivastava**
- Proven to improve accuracy by **~2%** (which is significant at high accuracy levels, e.g., 95% → 97%)

### How It Works

1. Set a **dropout rate** $p$ for each layer (e.g., $p = 0.5$)
2. For **every training iteration** (every batch):
   - Randomly select $p$ fraction of neurons to **deactivate**
   - Deactivated neurons lose all connections — effectively removed from the network
   - Train the **remaining** (thinner) network on that batch
3. Next batch → **different random set** of neurons dropped → different sub-network

```
Epoch 1, Batch 1: Drop neurons {2, 4} → Train sub-network A
Epoch 1, Batch 2: Drop neurons {1, 3} → Train sub-network B  
Epoch 1, Batch 3: Drop neurons {3, 5} → Train sub-network C
...
```

Each batch effectively trains a **different architecture** → ensemble of sub-networks.

---

## 3. Why Dropout Works (Two Intuitions)

### Intuition 1: Prevents Co-adaptation

Without dropout, a neuron can become **overly dependent** on one specific input neuron (one weight becomes very large, others very small).

With dropout:
- The "favorite" input neuron **may not be available** in the next batch
- The neuron is **forced to distribute attention** across all inputs
- Weights become more **balanced** → captures **general patterns** instead of noise
- Ignores minor fluctuations, focuses on **overall trends**

*Company analogy:* If 50% of employees randomly don't come to work each day, everyone must be capable of doing multiple tasks. No one can depend solely on one colleague → the company becomes more **resilient and generalized**.

### Intuition 2: Ensemble Learning (Random Forest Analogy)

| Random Forest | Dropout |
|---|---|
| Train **multiple decision trees** on random subsets of features | Train **multiple sub-networks** by randomly dropping neurons |
| Each tree sees different columns | Each batch trains a different architecture |
| Final prediction = **majority vote** across all trees | Final prediction = **average** across all sub-networks |
| Reduces overfitting via ensemble | Reduces overfitting via ensemble |

With $n$ total neurons (input + hidden), there are $2^n$ possible sub-networks. For 10 neurons → **1024** unique architectures.

> Dropout creates an **implicit ensemble** of exponentially many networks, all sharing weights.

---

## 4. Dropout at Test Time

**Dropout is applied only during training. At test time, ALL neurons are active.**

### Weight Scaling at Test Time

Since a neuron was only present $(1-p)$ fraction of the time during training, its learned weight is calibrated for a thinner network. At test time (full network), weights must be scaled down:

$$w_{\text{test}} = w_{\text{train}} \times (1 - p)$$

| Training | Testing |
|---|---|
| Dropout rate $p = 0.25$ | All neurons active |
| Neuron present 75% of the time | Weight scaled: $w \times 0.75$ |
| Random sub-networks | Full network with scaled weights |

> **In practice:** Keras/TensorFlow handles this automatically — you don't need to manually scale weights.

---

## 5. Key Properties

| Property | Detail |
|---|---|
| **When applied** | Only during training (not at test/inference time) |
| **Where applied** | Input layer and hidden layers (never on output layer) |
| **Dropout rate $p$** | Fraction of neurons to drop (typically 0.2–0.5) |
| **Effect on convergence** | Training becomes **slower** (more epochs needed) |
| **Effect on loss function** | Loss function shape changes each batch (different architecture) → gradient calculation becomes harder |
| **Net benefit** | ~2% accuracy improvement; significant overfitting reduction |

---

## 6. Summary

1. Dropout **randomly deactivates** neurons during each training batch
2. It works by **(a)** preventing co-adaptation of neurons and **(b)** creating an implicit ensemble of sub-networks
3. Similar in spirit to **Random Forest** — many weak models → one strong model
4. At test time, all neurons are active but weights are scaled by $(1-p)$
5. Dropout is applied **only during training**; Keras handles test-time scaling automatically
6. Typical improvement: **~2% accuracy gain** + significant overfitting reduction
