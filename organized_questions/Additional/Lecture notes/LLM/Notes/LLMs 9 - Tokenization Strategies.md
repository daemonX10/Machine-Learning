# LLMs 9 — Tokenization Strategies

## 1. Why Tokenization Matters

### The Input Pipeline Problem
Every neural model (Transformer, RNN, Word2Vec) takes a **sequence of tokens** as input. How we define "token" fundamentally affects model performance.

### Word-Level vs. Character-Level

| Approach | Vocabulary Size | Problem |
|----------|----------------|---------|
| **Word-level** | Huge (exponential combinations) | Cannot handle unknown/OOV words |
| **Character-level** | Tiny (26 + symbols) | Loses semantic meaning, very long sequences |

### Subword Tokenization — The Middle Ground
- Split words into **meaningful sub-units**: "catching" → "catch" + "ing"
- Shared morphemes link related words: "catching" and "hugging" both have "ing"
- Handles OOV words by decomposing into known subwords
- Used in all modern LLMs (BERT, GPT, T5, LLaMA)

---

## 2. Two-Module Framework

All tokenization strategies have two components:

| Module | When | Purpose |
|--------|------|---------|
| **Token Learner** | Training time | Learn vocabulary + merge rules from corpus |
| **Token Segmenter** | Test/inference time | Segment unseen words into known subword tokens |

---

## 3. Byte Pair Encoding (BPE)

**Origin:** Text compression (1994). Adapted for NMT by Sennrich et al. (2016).

**Used in:** GPT, GPT-2, RoBERTa, BART

### Algorithm Overview

#### Step 1: Pre-tokenization
- Clean corpus (remove punctuation, normalize case, etc.)
- Split sentences into words using spaces/delimiters

#### Step 2: Initialize Base Vocabulary
- Extract all **unique characters** from words
- Base vocab = set of unique characters (e.g., `{a, b, c, g, s, t}`)

#### Step 3: Iterative Merging
Repeat until vocabulary reaches target size:
1. Count **frequency of all adjacent character pairs** (bigrams)
2. Merge the **most frequent pair** into a new token
3. Add new token to vocabulary (keep original characters too)
4. Record the **merge rule**
5. Replace all occurrences of the pair in the corpus

### Worked Example

**Corpus:** cat(10), bat(5), bag(12), tag(4), cats(5)

**Base vocabulary:** `{a, b, c, g, s, t}`

| Iteration | Most Frequent Pair | New Token | New Rule |
|-----------|-------------------|-----------|----------|
| 1 | `a, t` (freq=20) | `at` | `a + t → at` |
| 2 | `a, g` (freq=16) | `ag` | `a + g → ag` |
| 3 | `c, at` (freq=15) | `cat` | `c + at → cat` |

**Vocabulary after 3 iterations:** `{a, b, c, g, s, t, at, ag, cat}`

> Original characters are **never removed** from the vocabulary.

### Test-Time Segmentation (Token Segmenter)

1. Split unknown word into individual characters
2. Apply learned merge rules **in the exact order they were learned**
3. Merge characters according to rules

**Example:** Segment "bags" → `b, a, g, s`
- Rule 1 (`a+t→at`): no `a,t` pair → skip
- Rule 2 (`a+g→ag`): found! → `b, ag, s`
- Rule 3 (`c+at→cat`): no match → stop
- **Result:** `[b, ag, s]`

### Handling Unknown Characters

If a character doesn't exist in the vocabulary:
- Replace with special `[UNK]` token
- Example: "mat" → `m` not in vocab → `[UNK], at`

### GPT-2/RoBERTa Variant
- Use **byte-level** BPE instead of character-level
- Base vocabulary = 256 bytes → handles **any** Unicode character

---

## 4. WordPiece Tokenization

**Used in:** BERT, MobileBERT, Funnel Transformer, ELECTRA

**Key difference from BPE:** Merge criterion is **likelihood score**, not raw frequency.

### Score Function

$$\text{score}(c, d) = \frac{\text{freq}(cd)}{\text{freq}(c) \times \text{freq}(d)}$$

This is a form of **Pointwise Mutual Information (PMI)**.

### Why Likelihood > Frequency?

| Scenario | Pair freq | Individual freqs | Score |
|----------|----------|-------------------|-------|
| High pair freq, high individual freqs | 100 | 50 × 50 | 0.04 (penalized!) |
| Medium pair freq, low individual freqs | 20 | 5 × 5 | 0.80 (prioritized!) |

→ Merges pairs that are **disproportionately common together** relative to their individual frequencies.

### Special Prefix: `##`
Distinguishes **word-initial** vs. **word-internal** characters:
- "hello" → `h, ##e, ##l, ##l, ##o`
- "token" → `t, ##o, ##k, ##e, ##n`

**Vocabulary treats `s` and `##s` as different tokens.**

### Merging Process
Same iterative loop as BPE, but:
1. Compute **score** (not frequency) for all pairs
2. Merge highest-scoring pair
3. Merged token preserves `##` prefix rules:
   - `s + ##u → su` (initial + internal = initial)
   - `##u + ##n → ##un` (internal + internal = internal)

### Worked Example

**Corpus:** sunflower(1), sunflowers(2), flower(3), flowers(1)

**Initial segmentation:** `s, ##u, ##n, ##f, ##l, ##o, ##w, ##e, ##r, ##s, f, ...`

| Iteration | Highest Score Pair | New Token |
|-----------|-------------------|-----------|
| 1 | `s, ##u` | `su` |
| 2 | `##e, ##r` | `##er` |
| 3 | `##e, ##d` | `##ed` |

### Test-Time Segmentation — Longest Match First

**No merge rules stored** — only the vocabulary.

Algorithm: **Greedy longest-match from left to right**

1. From the start of the word, find the **longest subword** present in the vocabulary
2. Split there
3. Repeat on the remaining portion

**Example:** Segment "fused" with vocab containing `{f, ##u, ##s, ##e, ##d, ##ed}`
- Start: longest match for "fused" → `f` (since "fu" not in vocab)
- Remaining "##used": longest match → `##u`
- Remaining "##sed": longest match → `##s`
- Remaining "##ed": longest match → `##ed` ✓
- **Result:** `[f, ##u, ##s, ##ed]`

### Handling Unknown Words

**Critical difference from BPE:** If **any** subword is not in the vocabulary, the **entire word** becomes `[UNK]`.

| Method | Unknown sub-token handling |
|--------|--------------------------|
| **BPE** | Only the unknown character → `[UNK]`, rest preserved |
| **WordPiece** | **Whole word** → `[UNK]` |

---

## 5. Unigram Language Model Tokenization

**Used in:** SentencePiece (T5, XLNet, ALBERT, InternVL)

### Key Differences from BPE/WordPiece

| Aspect | BPE / WordPiece | Unigram LM |
|--------|----------------|------------|
| **Direction** | Start small vocab → grow via merging | Start large vocab → **shrink by pruning** |
| **Segmentation** | Single deterministic split per word | **Multiple candidate splits** per word |
| **Algorithm** | Greedy frequency/score | **Expectation-Maximization (EM)** |

### Unigram Language Model Assumption

For a segmentation $S = [s_1, s_2, \dots, s_k]$:

$$P(S) = \prod_{i=1}^{k} P(s_i)$$

All subword tokens are **independent** (unigram assumption).

### Algorithm

#### Step 1: Initialize Large Base Vocabulary
- Include all characters + all frequent substrings
- Can also run BPE first to generate initial vocabulary

#### Step 2: For Each Word, Find Best Segmentation

For word "run" with vocab `{r, u, n, ru, un, run}`:

| Segmentation | Probability |
|-------------|-------------|
| `r, u, n` | $P(r) \times P(u) \times P(n)$ |
| `ru, n` | $P(ru) \times P(n)$ |
| `r, un` | $P(r) \times P(un)$ |

Best segmentation = highest probability split.

> **Viterbi algorithm** is used to efficiently find the best split (avoids exhaustive enumeration).

#### Step 3: Compute Corpus Log-Loss

$$\mathcal{L} = -\sum_{\text{word } w} \text{freq}(w) \times \log P(\text{best\_split}(w))$$

#### Step 4: Prune Vocabulary (EM — Maximization Step)

For each vocabulary token, compute: *what happens to $\mathcal{L}$ if we remove this token?*

- If removal **doesn't change** $\mathcal{L}$ → token is **redundant** → safe to remove
- If removal **increases** $\mathcal{L}$ significantly → token is **important** → keep it
- Select the token whose removal has **least impact** on loss

**In practice:** Remove **10–20% of tokens** per iteration.

#### Step 5: Repeat
- Update unigram probabilities with modified vocabulary
- Recompute best splits for all words
- Recompute loss
- Prune again
- Stop when vocabulary reaches target size

### Worked Example

**Corpus:** run(3), bug(5), fun(13), sun(10)

**Initial vocab:** `{r, u, n, ru, un, b, g, bu, ug, f, fu, s, su, run, bug, fun, sun}`

**Unigram probabilities:** $P(r) = 3/155$, $P(u) = 31/155$, $P(ru) = 10/155$, etc.

| Step | Remove Token | Loss Before | Loss After | Decision |
|------|-------------|-------------|------------|----------|
| 1 | `un` | 66.102 | 66.102 | ✅ Remove (no impact) |
| 2 | `bu` | 61.155 | 61.155 | ✅ Remove (no impact) |
| 3 | `ug` | 61.155 | 61.155 | ✅ Remove (no impact) |
| 4 | `ru` | 61.155 | 63.013 | ❌ Keep (loss increases) |

---

## 6. Comparison of Tokenization Methods

| Feature | BPE | WordPiece | Unigram LM |
|---------|-----|-----------|------------|
| **Direction** | Bottom-up (grow vocab) | Bottom-up (grow vocab) | Top-down (shrink vocab) |
| **Merge criterion** | Frequency | Likelihood (PMI) | — |
| **Prune criterion** | — | — | Minimal loss impact |
| **Segmentation** | Apply rules in order | Longest match from vocab | Best probability split (Viterbi) |
| **Stores rules?** | Yes | No (vocab only) | No (vocab + probabilities) |
| **Multiple segmentations?** | No (deterministic) | No (deterministic) | Yes (probabilistic) |
| **Unknown handling** | Per-character `[UNK]` | Whole-word `[UNK]` | Probabilistic fallback |
| **Used in** | GPT, GPT-2, RoBERTa | BERT, ELECTRA | T5, XLNet, ALBERT |

---

## 7. Tokenization in Practice

| Model | Tokenizer | Vocab Size |
|-------|-----------|------------|
| GPT-2 | Byte-level BPE | 50,257 |
| GPT-4 | Byte-level BPE (cl100k) | ~100,000 |
| BERT | WordPiece | 30,522 |
| T5 | SentencePiece (Unigram) | 32,000 |
| LLaMA | SentencePiece (BPE) | 32,000 |
| LLaMA 3 | tiktoken (BPE) | 128,256 |

---

## 8. Key Takeaways

1. **Subword tokenization** is the standard — balances vocabulary size and coverage
2. **BPE** is simplest: merge most frequent pairs iteratively
3. **WordPiece** improves on BPE with likelihood-based merging (PMI score)
4. **Unigram LM** takes the opposite approach: start big, prune based on loss impact
5. All three handle **OOV words** by decomposing into known subwords
6. Tokenization is critical for **multilingual** and **low-resource** language support
7. `[UNK]` handling differs: BPE replaces unknown characters, WordPiece replaces entire words
