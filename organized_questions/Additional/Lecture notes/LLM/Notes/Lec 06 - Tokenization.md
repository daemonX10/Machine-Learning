# Lecture 06 — Tokenization (BPE, WordPiece, Unigram LM)

## 1. Why Subword Tokenization?

| Strategy | Pros | Cons |
|----------|------|------|
| **Word-level** | Intuitive | OOV (out-of-vocabulary) problem; huge vocabulary needed |
| **Character-level** | No OOV; tiny vocabulary | Too granular; loses word-level semantics; very long sequences |
| **Subword** | Handles OOV via morphological decomposition; compact vocabulary | Slightly complex tokenization logic |

**Subword insight**: Morphological patterns transfer. If the model knows:
- `run` + `ing` → running
- then `catch` + `ing` → catching (can generalize to unseen words)

---

## 2. Tokenizer Components

| Component | Phase | Description |
|-----------|-------|-------------|
| **Token Learner** | Training | Builds the vocabulary from a corpus |
| **Token Segmenter** | Inference | Splits new text into tokens using the learned vocabulary |

---

## 3. Byte Pair Encoding (BPE)

> **Used by**: GPT, GPT-2, RoBERTa, DeBERTa

### 3.1 Token Learner (Training)

**Algorithm**:

```
1. Pre-tokenize: clean corpus, split into words, compute word frequencies
2. Initialize base vocabulary = all unique characters in corpus
3. Repeat until desired vocab size:
   a. Count frequency of all adjacent character pairs across corpus
   b. Merge the most frequent pair into a new token
   c. Add new token to vocabulary
   d. Record merge rule in ordered rule book
   e. Update corpus with merged token
```

**Example**:

```
Corpus (with frequencies):
  "low"     ×5    → l o w
  "lowest"  ×2    → l o w e s t
  "newer"   ×6    → n e w e r
  "wider"   ×3    → w i d e r
  "new"     ×2    → n e w

Base vocab: {d, e, i, l, n, o, r, s, t, w}

Iteration 1: most frequent pair = (e, r) → count = 6+3 = 9
  Merge: "er" → vocab adds "er", rule book: [(e,r)→er]
  
Iteration 2: most frequent pair = (e, w) → count = 6+2 = 8
  Merge: "ew" → vocab adds "ew", rule book: [(e,r)→er, (e,w)→ew]
  
... continue until desired vocabulary size
```

### 3.2 Token Segmenter (Inference / Test Time)

```
1. Split word into individual characters
2. Apply merge rules IN ORDER (same order as learned)
3. Tokens not in vocabulary → [UNK]
```

**Example** (with rules `[(e,r)→er, (n,ew)→new]`):

```
Input: "newer" → n e w e r
Apply rule 1 (e,r→er): n e w er
Apply rule 2 (n,ew): doesn't match (no "ew" yet — "e" was consumed by "er")
Result: [n, e, w, er]
```

> **Order of merge rules matters** — different order → different tokenization.

### 3.3 Byte-Level BPE (GPT-2, RoBERTa)

- Base vocabulary = **256 byte** values (instead of Unicode characters)
- Can represent **any** text (including emojis, non-Latin scripts) without `[UNK]`
- Never produces unknown tokens

---

## 4. WordPiece

> **Used by**: BERT, MobileBERT, Funnel Transformer, DeBERTa

### 4.1 Key Differences from BPE

| Aspect | BPE | WordPiece |
|--------|-----|-----------|
| **Merge criterion** | Most **frequent** pair | Highest **likelihood** (PMI score) |
| **Subword marker** | None | `##` prefix for non-initial subwords |
| **Rule storage** | Ordered merge rule book | Only final vocabulary (no merge rules) |
| **Test-time strategy** | Apply merge rules in order | **Greedy longest-match-first** from vocabulary |
| **OOV handling** | Character-level fallback | **Entire word** → `[UNK]` |

### 4.2 Merge Score (PMI-based)

$$\text{score}(t_1, t_2) = \frac{\text{freq}(t_1 t_2)}{\text{freq}(t_1) \times \text{freq}(t_2)}$$

> This is **pointwise mutual information** — it prefers merging pairs that co-occur more than expected by chance.

**Why not just frequency?**
- High-frequency individual tokens appearing together ≠ meaningful merge
- PMI normalizes: if "t" and "h" are individually very common, "th" needs to be disproportionately frequent to justify merging

### 4.3 The `##` Prefix Notation

Non-initial subwords are prefixed with `##`:

```
"unbelievable" → ["un", "##believ", "##able"]
"BERT"         → ["B", "##E", "##R", "##T"]
```

- `##` indicates this token is a **continuation** (not a word start)
- Helps in de-tokenization (word boundary reconstruction)

### 4.4 Token Segmenter (Test Time)

**Greedy longest-match-first** (no merge rules needed):

```
Word: "unbelievable"
  Check: "unbelievable" in vocab? No
  Check: "unbelievabl" in vocab? No
  ...
  Check: "un" in vocab? Yes → take "un"
  Remaining: "believable"
  Check: "##believable" in vocab? No
  ...
  Check: "##believ" in vocab? Yes → take "##believ"
  Remaining: "able"
  Check: "##able" in vocab? Yes → take "##able"
  
Result: ["un", "##believ", "##able"]
```

### 4.5 OOV Handling

- If at any point during segmentation, no match is found for a remaining substring → the **entire original word** becomes `[UNK]`
- This is stricter than BPE (which falls back to individual characters)

---

## 5. Unigram Language Model (SentencePiece)

> **Used by**: ALBERT, T5, XLNet, mBART

### 5.1 Fundamental Difference

| Aspect | BPE / WordPiece | Unigram LM |
|--------|-----------------|------------|
| Direction | Start small → **grow** vocabulary by merging | Start large → **shrink** vocabulary by removing |
| Determinism | Deterministic (one split per word) | **Probabilistic** (multiple valid splits, context-dependent) |
| Approach | Greedy / frequency-based | **Expectation-Maximization** |

### 5.2 Unigram Assumption

Each word's probability is the product of its subword probabilities (independence assumption):

$$P(\text{word}) = \prod_{i=1}^{n} P(\text{subword}_i)$$

### 5.3 Algorithm

```
1. Initialize large vocabulary:
   - All possible substrings up to max length, OR
   - Run BPE/WordPiece with high vocab limit
   - Compute initial probabilities: P(token) = freq(token) / total_freq

2. E-step: For each word, enumerate ALL possible segmentations
   - Compute P(segmentation) = ∏ P(subword_i) for each
   - Use probabilities to weight contributions

3. M-step: Re-estimate token probabilities from weighted counts

4. For each token, compute: "If I remove this token, 
   how much does the total log-likelihood decrease?"
   - Remove tokens with LEAST impact (most redundant)
   - Keep a fraction (e.g., remove bottom 20%)

5. Repeat steps 2-4 until desired vocabulary size
```

### 5.4 Log-Likelihood Objective

$$\mathcal{L} = \sum_{\text{word } w} \log P(w) = \sum_w \log \left( \sum_{\text{segmentation } s} \prod_{t \in s} P(t) \right)$$

- In practice, approximate with **Viterbi** (best segmentation) instead of summing over all segmentations

### 5.5 Token Removal Criterion

For each token $t$ in vocabulary:
1. Temporarily remove $t$
2. Recompute probabilities (denominator changes since total count decreases)
3. Recompute total log-likelihood of corpus
4. $\Delta \mathcal{L}_t = \mathcal{L}_{\text{without } t} - \mathcal{L}_{\text{with } t}$
5. Remove the token with the **smallest** $|\Delta \mathcal{L}_t|$ (least missed)

### 5.6 Probabilistic Segmentation

Unlike BPE/WordPiece, the same word can have **different tokenizations**:

```
"unigram" could be tokenized as:
  ["uni", "gram"]     with P = 0.4
  ["un", "i", "gram"] with P = 0.35
  ["unigram"]         with P = 0.25
```

At inference, **Viterbi algorithm** selects the highest-probability segmentation.

---

## 6. Comparison Table

| Feature | BPE | WordPiece | Unigram LM |
|---------|-----|-----------|------------|
| **Models** | GPT, GPT-2, RoBERTa | BERT, MobileBERT | ALBERT, T5, XLNet |
| **Direction** | Bottom-up (merge) | Bottom-up (merge) | Top-down (prune) |
| **Merge/Prune criterion** | Pair frequency | PMI (likelihood) | Log-likelihood impact |
| **Deterministic?** | Yes | Yes | No (probabilistic) |
| **Stores merge rules?** | Yes | No | No |
| **Test-time method** | Apply rules in order | Greedy longest-match | Viterbi (max probability) |
| **OOV behavior** | Char-level fallback | Entire word → `[UNK]` | Very rare (large initial vocab) |
| **Subword marker** | None (or Ġ prefix for GPT-2) | `##` prefix | `▁` (underscore for word start) |
