# NLP Fundamentals (Tokenization, NER, POS Tagging) - Theory Questions

## Core Questions

## Question 1
**What is tokenization and why is it the fundamental first step in NLP pipelines?**

**Answer:**

**Definition:**
Tokenization is the process of breaking raw text into smaller meaningful units called tokens (words, subwords, or characters). It is the fundamental first step because all downstream NLP models operate on numerical representations of tokens, not raw text strings.

**Core Concepts:**
- Token: A discrete unit of text (word, subword, character, or symbol)
- Vocabulary: The set of all unique tokens the model recognizes
- Token ID: Integer mapping of each token for model input
- Whitespace tokenization: Splitting by spaces (simplest approach)
- Subword tokenization: Breaking words into smaller frequent units (BPE, WordPiece)

**Why Fundamental:**
- Converts unstructured text → structured numerical input
- Defines the vocabulary size and model's understanding capacity
- Affects OOV handling, sequence length, and memory usage
- Poor tokenization = poor model performance downstream

**Practical Relevance:**
- Every NLP pipeline starts with tokenization (classification, NER, translation, QA)
- Choice of tokenizer affects model accuracy, inference speed, and memory
- Pre-trained models require exact same tokenizer used during training

**Python Code Example:**
```python
# Pipeline: Raw text -> Tokens -> Token IDs -> Model input

from transformers import AutoTokenizer

# Step 1: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Step 2: Tokenize text
text = "Machine learning is amazing!"
tokens = tokenizer.tokenize(text)
# Output: ['machine', 'learning', 'is', 'amazing', '!']

# Step 3: Convert to IDs
token_ids = tokenizer.encode(text)
# Output: [101, 3698, 4083, 2003, 6429, 999, 102]
# 101=[CLS], 102=[SEP]

# Step 4: Full encoding for model
encoded = tokenizer(text, return_tensors="pt")
# Output: {'input_ids': tensor, 'attention_mask': tensor}
```

**Interview Tips:**
- Always mention that tokenization choice depends on task and language
- Know difference between word-level, subword, and character-level
- BERT uses WordPiece, GPT uses BPE - know this distinction

---

## Question 2
**Explain the difference between stemming and lemmatization with examples.**

**Answer:**

**Definition:**
Stemming reduces words to their root form by chopping off suffixes using crude heuristic rules. Lemmatization reduces words to their dictionary base form (lemma) using vocabulary and morphological analysis. Lemmatization is linguistically accurate; stemming is faster but less precise.

**Core Concepts:**
| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| Approach | Rule-based suffix stripping | Dictionary + morphological analysis |
| Output | May not be valid word | Always valid dictionary word |
| Speed | Faster | Slower |
| Accuracy | Lower | Higher |
| POS needed | No | Often yes |

**Examples:**
| Word | Stemming (Porter) | Lemmatization |
|------|-------------------|---------------|
| running | runn | run |
| studies | studi | study |
| better | better | good |
| caring | car | care |
| wolves | wolv | wolf |

**When to Use:**
- **Stemming:** Information retrieval, search engines, when speed matters
- **Lemmatization:** Sentiment analysis, chatbots, when meaning matters

**Python Code Example:**
```python
# Pipeline: Word -> Apply Stemmer/Lemmatizer -> Normalized form

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')

# Initialize
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "studies", "better", "wolves", "caring"]

# Stemming
stemmed = [stemmer.stem(w) for w in words]
# Output: ['run', 'studi', 'better', 'wolv', 'care']

# Lemmatization (default: noun)
lemmatized = [lemmatizer.lemmatize(w) for w in words]
# Output: ['running', 'study', 'better', 'wolf', 'caring']

# Lemmatization with POS tag (verb)
lemmatized_verb = [lemmatizer.lemmatize(w, pos='v') for w in words]
# Output: ['run', 'study', 'better', 'wolf', 'care']
```

**Interview Tips:**
- Porter Stemmer is most common; Snowball is improved version
- Lemmatization needs POS tag for accuracy ("better" → "good" only if adjective)
- Modern deep learning models often skip both - subword tokenizers handle morphology

---

## Question 3
**What are BPE, WordPiece, and SentencePiece? How do subword tokenization algorithms work?**

**Answer:**

**Definition:**
Subword tokenization algorithms split text into units between characters and words, learning frequent subword patterns from data. BPE (Byte Pair Encoding), WordPiece, and SentencePiece are the three dominant algorithms that balance vocabulary size with representation power.

**Core Concepts:**

| Algorithm | Used By | Approach |
|-----------|---------|----------|
| BPE | GPT, RoBERTa | Merge most frequent adjacent pairs iteratively |
| WordPiece | BERT, DistilBERT | Merge pairs that maximize likelihood |
| SentencePiece | T5, XLNet, mBERT | Language-agnostic, treats text as raw stream |

**How BPE Works (Algorithm):**
1. Start with character-level vocabulary + end-of-word symbol
2. Count frequency of all adjacent symbol pairs
3. Merge the most frequent pair into new symbol
4. Repeat until desired vocabulary size reached

**Example:**
```
Corpus: "low lower lowest"
Initial: ['l', 'o', 'w', '</w>', 'l', 'o', 'w', 'e', 'r', '</w>', ...]
Step 1: Merge 'l'+'o' → 'lo' (most frequent)
Step 2: Merge 'lo'+'w' → 'low'
Step 3: Merge 'low'+'e' → 'lowe'
Final vocab: ['low</w>', 'low', 'er</w>', 'est</w>', ...]
```

**Key Differences:**
- **BPE:** Frequency-based merging
- **WordPiece:** Uses likelihood (probability) instead of frequency
- **SentencePiece:** No pre-tokenization, works directly on raw text (good for CJK)

**Mathematical Formulation (WordPiece):**
$$\text{Score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \times \text{freq}(y)}$$
Merge pair with highest score (maximizes likelihood gain).

**Python Code Example:**
```python
# Pipeline: Train BPE -> Tokenize new text

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Step 1: Initialize BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Step 2: Train on corpus
trainer = BpeTrainer(vocab_size=1000, special_tokens=["[UNK]", "[PAD]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Step 3: Tokenize
output = tokenizer.encode("unhappiness")
print(output.tokens)
# Output: ['un', 'happi', 'ness'] (subwords)
```

**Interview Tips:**
- BPE is greedy (frequency), WordPiece is probabilistic
- SentencePiece is a library that can implement BPE or Unigram
- Subword tokenizers solve OOV by breaking unknown words into known subwords

---

## Question 4
**What is Named Entity Recognition (NER) and what are the common entity types?**

**Answer:**

**Definition:**
Named Entity Recognition (NER) is a sequence labeling task that identifies and classifies named entities in text into predefined categories like person, organization, location, and date. It extracts structured information from unstructured text for downstream applications like knowledge graphs and information extraction.

**Core Concepts:**
- **Entity:** A real-world object that can be denoted with a proper name
- **Sequence Labeling:** Assigning a label to each token in a sequence
- **Entity Span:** Start and end positions of entity in text
- **Entity Linking:** Connecting extracted entities to knowledge base entries

**Common Entity Types (Standard):**

| Type | Tag | Examples |
|------|-----|----------|
| Person | PER | "Elon Musk", "Albert Einstein" |
| Organization | ORG | "Google", "United Nations" |
| Location | LOC | "Paris", "Mount Everest" |
| Geo-Political | GPE | "France", "New York City" |
| Date/Time | DATE/TIME | "January 2024", "3:00 PM" |
| Money | MONEY | "$500", "10 million euros" |
| Percentage | PERCENT | "25%", "fifty percent" |
| Miscellaneous | MISC | "Nobel Prize", "World Cup" |

**Domain-Specific Types:**
- **Medical:** Disease, Drug, Gene, Symptom
- **Legal:** Court, Law, Judge, Case
- **Finance:** Ticker, Company, Currency

**Python Code Example:**
```python
# Pipeline: Text -> NER Model -> Entity spans with labels

import spacy

# Step 1: Load pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# Step 2: Process text
text = "Apple Inc. was founded by Steve Jobs in California."
doc = nlp(text)

# Step 3: Extract entities
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")
    
# Output:
# Apple Inc. -> ORG
# Steve Jobs -> PERSON
# California -> GPE
```

**Interview Tips:**
- NER is foundation for knowledge extraction and QA systems
- Nested NER handles entities within entities ("Bank of America" → ORG contains LOC)
- Fine-grained NER may have 100+ entity types

---

## Question 5
**Explain the BIO/IOB tagging scheme used in NER and sequence labeling tasks.**

**Answer:**

**Definition:**
BIO (Beginning-Inside-Outside) tagging scheme is a labeling format where each token gets a tag indicating whether it begins an entity (B), continues inside an entity (I), or is outside any entity (O). It enables unambiguous representation of entity boundaries in sequence labeling.

**Core Concepts:**
- **B-TAG:** Beginning of an entity of type TAG
- **I-TAG:** Inside/continuation of entity of type TAG
- **O:** Outside any entity (not part of named entity)

**Why BIO is Needed:**
Without BIO, consecutive entities of same type would merge incorrectly.

**Example:**
```
Text:    "John  Smith  works  at  New   York   Times"
BIO:      B-PER I-PER  O      O   B-ORG I-ORG  I-ORG

Text:    "John  met   Mary  in  Paris"
BIO:      B-PER O     B-PER O   B-LOC
```

**Tagging Variants:**

| Scheme | Tags | Use Case |
|--------|------|----------|
| BIO | B, I, O | Standard, most common |
| BIOES | B, I, O, E, S | E=End, S=Single-token entity |
| BILOU | B, I, L, O, U | L=Last, U=Unit (single) |

**BIOES Example:**
```
Text:    "New   York   Times"
BIOES:    B-ORG I-ORG  E-ORG

Text:    "IBM"
BIOES:    S-ORG
```

**Why BIOES > BIO:**
- More explicit boundary information
- Helps model learn entity boundaries better
- Slightly improves NER accuracy

**Python Code Example:**
```python
# Pipeline: Tokens + Labels -> BIO format -> Model training

# Sample data in BIO format
sentence = ["John", "Smith", "works", "at", "Apple"]
bio_tags = ["B-PER", "I-PER", "O", "O", "B-ORG"]

# Convert to training format
training_data = list(zip(sentence, bio_tags))
# [('John', 'B-PER'), ('Smith', 'I-PER'), ('works', 'O'), 
#  ('at', 'O'), ('Apple', 'B-ORG')]

# Extract entities from BIO tags
def extract_entities(tokens, tags):
    entities = []
    current_entity = []
    current_type = None
    
    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if current_entity:
                entities.append((' '.join(current_entity), current_type))
            current_entity = [token]
            current_type = tag[2:]
        elif tag.startswith('I-') and current_type:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append((' '.join(current_entity), current_type))
            current_entity = []
            current_type = None
    
    if current_entity:
        entities.append((' '.join(current_entity), current_type))
    
    return entities

print(extract_entities(sentence, bio_tags))
# Output: [('John Smith', 'PER'), ('Apple', 'ORG')]
```

**Interview Tips:**
- Always use B- prefix even for single-token entities (consistency)
- BIOES/BILOU often gives 1-2% F1 improvement over BIO
- During prediction, enforce valid transitions (I-PER cannot follow B-ORG)

---

## Question 6
**What is Part-of-Speech (POS) tagging? Why is it important for downstream NLP tasks?**

**Answer:**

**Definition:**
Part-of-Speech (POS) tagging assigns grammatical categories (noun, verb, adjective, etc.) to each word in a sentence based on its definition and context. It provides syntactic information essential for parsing, NER, sentiment analysis, and understanding sentence structure.

**Core Concepts:**
- **POS Tag:** Grammatical category of a word
- **Context-dependent:** Same word can have different POS ("run" as verb vs noun)
- **Tag Set:** Predefined set of tags (Penn Treebank has 45 tags)

**Common POS Tags (Penn Treebank):**

| Tag | Description | Example |
|-----|-------------|---------|
| NN | Noun, singular | "dog", "city" |
| NNS | Noun, plural | "dogs", "cities" |
| VB | Verb, base form | "run", "eat" |
| VBD | Verb, past tense | "ran", "ate" |
| VBG | Verb, gerund | "running", "eating" |
| JJ | Adjective | "happy", "fast" |
| RB | Adverb | "quickly", "very" |
| DT | Determiner | "the", "a" |
| IN | Preposition | "in", "on", "at" |
| PRP | Personal pronoun | "he", "she", "it" |

**Why Important for Downstream Tasks:**

| Task | How POS Helps |
|------|---------------|
| NER | Entities are typically nouns (NN, NNP) |
| Sentiment | Adjectives carry sentiment weight |
| Parsing | Determines phrase structure |
| Lemmatization | Need POS for correct lemma |
| Information Extraction | Filter by grammatical role |

**Python Code Example:**
```python
# Pipeline: Text -> Tokenize -> POS Tagger -> Tagged tokens

import nltk
nltk.download('averaged_perceptron_tagger')

# Method 1: NLTK
text = "The quick brown fox jumps over the lazy dog"
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), 
#  ('fox', 'NN'), ('jumps', 'VBZ'), ...]

# Method 2: spaCy (more accurate)
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

for token in doc:
    print(f"{token.text}: {token.pos_} ({token.tag_})")
# The: DET (DT)
# quick: ADJ (JJ)
# fox: NOUN (NN)
# jumps: VERB (VBZ)
```

**Interview Tips:**
- Universal POS tags (17 tags) vs Penn Treebank (45 tags) - know the difference
- Accuracy: ~97% for English with neural models
- Ambiguity example: "book" can be NN (noun) or VB (verb)

---

## Question 7
**How do CRF (Conditional Random Fields) work for sequence labeling tasks like NER and POS tagging?**

**Answer:**

**Definition:**
Conditional Random Fields (CRF) are discriminative probabilistic models that predict a sequence of labels given an input sequence, modeling the conditional probability P(Y|X) while considering dependencies between adjacent labels. They excel at sequence labeling by capturing transition patterns between tags.

**Core Concepts:**
- **Discriminative model:** Models P(Y|X) directly, not P(X,Y)
- **Global normalization:** Normalizes over entire sequence, not per-token
- **Feature functions:** Capture relationships between input and labels
- **Transition matrix:** Learns valid label transitions (B-PER → I-PER is valid)

**Mathematical Formulation:**

$$P(Y|X) = \frac{1}{Z(X)} \exp\left(\sum_{t=1}^{T} \sum_{k} \lambda_k f_k(y_{t-1}, y_t, X, t)\right)$$

Where:
- $Z(X)$ = partition function (normalization)
- $f_k$ = feature functions
- $\lambda_k$ = learned weights
- $y_t$ = label at position $t$

**Why CRF > Softmax for Sequence Labeling:**

| Approach | Issue |
|----------|-------|
| Per-token Softmax | Ignores label dependencies; can predict "I-PER" after "B-ORG" |
| CRF Layer | Learns transition scores; enforces valid sequences |

**CRF Components:**
1. **Emission scores:** Score for each label at each position (from LSTM/BERT)
2. **Transition scores:** Learned matrix A where A[i,j] = score of transitioning from label i to j
3. **Viterbi decoding:** Find best label sequence efficiently

**Algorithm Steps (Training):**
1. Compute emission scores from neural network
2. Apply transition matrix for all label pairs
3. Use forward algorithm to compute Z(X) (partition function)
4. Compute negative log-likelihood loss
5. Backpropagate and update weights

**Algorithm Steps (Inference - Viterbi):**
1. Initialize scores for first position
2. For each position: score = emission + max(previous_score + transition)
3. Backtrack to find best path

**Python Code Example:**
```python
# Pipeline: Tokens -> BERT embeddings -> CRF layer -> Predicted tags

import torch
from torchcrf import CRF

# Step 1: Define model with CRF
class NERModel(torch.nn.Module):
    def __init__(self, vocab_size, tagset_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = torch.nn.Linear(hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)
    
    def forward(self, x, tags=None, mask=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:  # Training
            loss = -self.crf(emissions, tags, mask=mask)
            return loss
        else:  # Inference
            return self.crf.decode(emissions, mask=mask)

# Step 2: Training
model = NERModel(vocab_size=1000, tagset_size=9, embed_dim=100, hidden_dim=128)
# Loss = negative log-likelihood from CRF

# Step 3: Inference (Viterbi decoding)
# predictions = model(input_ids)  # Returns best tag sequence
```

**Interview Tips:**
- CRF adds ~1-2% F1 improvement for NER
- Modern BERT+CRF is standard for production NER
- Transition matrix size = num_tags × num_tags
- Viterbi is O(T × K²) where T=sequence length, K=num tags

---

## Question 8
**What is the difference between rule-based, statistical, and neural approaches to NER?**

**Answer:**

**Definition:**
NER approaches have evolved from handcrafted rules to statistical models to neural networks. Rule-based uses pattern matching, statistical uses features with ML classifiers like CRF, and neural uses deep learning to automatically learn representations and patterns.

**Core Comparison:**

| Aspect | Rule-Based | Statistical | Neural |
|--------|-----------|-------------|--------|
| Approach | Regex, gazetteers, patterns | Feature engineering + CRF/HMM | End-to-end deep learning |
| Features | Handcrafted | Manual feature extraction | Automatic representation |
| Training Data | None needed | Moderate | Large amounts |
| Adaptability | Low (needs rule rewrite) | Moderate | High (fine-tuning) |
| Accuracy | Domain-specific high | Good | State-of-the-art |
| Interpretability | High | Moderate | Low |

**Rule-Based Approach:**
```
Pattern: [A-Z][a-z]+ (Inc|Corp|Ltd)  → ORG
Gazetteer: {"New York", "London", "Paris"} → LOC
Context: "Dr. [Name]" → PERSON
```
- Pros: No training data, interpretable, precise for known patterns
- Cons: Doesn't generalize, maintenance nightmare

**Statistical Approach (CRF/HMM):**
- Features: Capitalization, word shape, prefix/suffix, POS tag, gazetteer lookup
- Model: CRF learns feature weights and transition probabilities
- Pros: Generalizes better, principled probability framework
- Cons: Requires feature engineering expertise

**Neural Approach (LSTM/Transformer):**
- Architecture: BERT → Linear → CRF
- Learns: Character patterns, contextual embeddings, entity boundaries
- Pros: State-of-the-art, handles unseen entities, transfer learning
- Cons: Needs data, compute-intensive, less interpretable

**Python Code Example:**
```python
# Comparing three approaches

# 1. Rule-based (simple regex)
import re
def rule_based_ner(text):
    entities = []
    # Pattern for organizations
    orgs = re.findall(r'[A-Z][a-z]+ (Inc|Corp|Ltd)\.?', text)
    # Pattern for dates
    dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b', text)
    return {"ORG": orgs, "DATE": dates}

# 2. Statistical (sklearn-crfsuite)
# Feature extraction + CRF training (conceptual)
def extract_features(word, pos):
    return {
        'word': word.lower(),
        'is_capitalized': word[0].isupper(),
        'is_all_caps': word.isupper(),
        'pos': pos,
        'prefix_2': word[:2],
        'suffix_2': word[-2:]
    }

# 3. Neural (transformers)
from transformers import pipeline
ner = pipeline("ner", model="dslim/bert-base-NER")
result = ner("Apple Inc was founded by Steve Jobs")
# [{'entity': 'B-ORG', 'word': 'Apple'}, ...]
```

**When to Use Each:**
- **Rule-based:** Quick prototype, specific domain, no training data
- **Statistical:** Moderate data, need interpretability, limited compute
- **Neural:** Best accuracy needed, sufficient data and compute

**Interview Tips:**
- Hybrid approaches combine all three (rules for post-processing neural output)
- Statistical CRF is still used as final layer on top of BERT
- Gazetteers remain useful even in neural systems

---

## Question 9
**How do modern transformers like BERT handle tokenization differently from traditional methods?**

**Answer:**

**Definition:**
BERT uses WordPiece subword tokenization which splits words into frequent subword units, adds special tokens ([CLS], [SEP], [PAD]), and maintains a fixed vocabulary. Unlike traditional word-level tokenization, it handles OOV words by breaking them into known subwords, enabling open vocabulary modeling.

**Core Differences:**

| Aspect | Traditional (Word-level) | BERT (WordPiece) |
|--------|-------------------------|------------------|
| OOV Handling | [UNK] token | Splits into subwords |
| Vocabulary | Very large or limited | Fixed size (~30K) |
| Morphology | Lost | Partially captured |
| Special Tokens | None | [CLS], [SEP], [PAD], [MASK] |
| Subword prefix | N/A | ## for continuation |

**BERT Tokenization Process:**
1. Add special tokens: [CLS] at start, [SEP] at end
2. Apply WordPiece algorithm to split text
3. Convert tokens to IDs using vocabulary
4. Generate attention mask (1 for real tokens, 0 for padding)
5. Generate token type IDs (for sentence A vs B)

**Example:**
```
Input: "unhappiness is contagious"

Word-level:  ["unhappiness", "is", "contagious"]  # 3 tokens
WordPiece:   ["un", "##hap", "##pi", "##ness", "is", "con", "##ta", "##gious"]

Full BERT:   ["[CLS]", "un", "##hap", "##pi", "##ness", "is", 
              "con", "##ta", "##gious", "[SEP]"]
```

**Special Tokens Purpose:**

| Token | Purpose |
|-------|---------|
| [CLS] | Classification token; pooled for sentence-level tasks |
| [SEP] | Separator between sentences |
| [PAD] | Padding for batch processing |
| [MASK] | Masked token for pre-training |
| [UNK] | Unknown (rarely used due to subwords) |

**Python Code Example:**
```python
# Pipeline: Text -> BERT Tokenizer -> Model-ready tensors

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "unhappiness is contagious"

# Step 1: Basic tokenization
tokens = tokenizer.tokenize(text)
print(tokens)
# ['un', '##happ', '##iness', 'is', 'con', '##tag', '##ious']

# Step 2: Full encoding with special tokens
encoded = tokenizer(text, return_tensors='pt', padding=True)
print(encoded)
# {'input_ids': tensor([[101, 4895, 18223, 11231, 2003, 9530, 16259, 6313, 102]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]]),
#  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]])}

# Step 3: Decode back
decoded = tokenizer.decode(encoded['input_ids'][0])
# '[CLS] unhappiness is contagious [SEP]'

# For NER: Track subword alignment
tokens_with_offsets = tokenizer(text, return_offsets_mapping=True)
# Offset mapping helps align subword predictions to original words
```

**Key Implications for NER:**
- Subword tokens must be realigned to original words
- First subword gets the label; rest get special label or are ignored
- Handle ## prefix tokens during post-processing

**Interview Tips:**
- BERT vocab size = 30,522 (base) with WordPiece
- GPT uses BPE (no ## prefix, uses Ġ for space)
- [CLS] embedding is used for classification, not for token-level tasks

---

## Question 10
**What is the out-of-vocabulary (OOV) problem and how do subword tokenizers solve it?**

**Answer:**

**Definition:**
The Out-of-Vocabulary (OOV) problem occurs when a model encounters words not present in its training vocabulary, typically mapped to a useless [UNK] token. Subword tokenizers solve this by breaking unknown words into smaller known subword units, ensuring any word can be represented.

**Core Concepts:**
- **Closed vocabulary:** Fixed word set; unseen words → [UNK]
- **Open vocabulary:** Can represent any word via subword composition
- **[UNK] token:** Placeholder for unknown words; loses all semantic information
- **Subword composition:** Build unknown words from known pieces

**The OOV Problem:**
```
Vocabulary: ["happy", "sad", "the", ...]
New word: "unhappiness"
Word-level: "unhappiness" → [UNK]  ❌ Information lost!

Subword vocab: ["un", "happy", "ness", ...]  
Subword: "unhappiness" → ["un", "happy", "ness"]  ✓ Meaning preserved!
```

**Why OOV is Problematic:**
- Rare words, typos, new terms all become [UNK]
- [UNK] has no useful semantics
- Critical for names, technical terms, multilingual text
- High OOV rate = poor model performance

**How Subword Tokenizers Solve OOV:**

| Scenario | Word-Level | Subword (BPE) |
|----------|-----------|---------------|
| Rare word: "cryptocurrency" | [UNK] | ["crypt", "o", "currency"] |
| Typo: "hapyy" | [UNK] | ["hap", "y", "y"] |
| New name: "ChatGPT" | [UNK] | ["Chat", "G", "PT"] |
| Morphology: "running" | [UNK] or separate | ["run", "ning"] |

**Trade-off:**
- Smaller vocab + longer sequences vs Larger vocab + shorter sequences
- Subwords balance this: ~30K vocab handles most text efficiently

**Python Code Example:**
```python
# Pipeline: Demonstrate OOV handling

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example words (some rare/unusual)
words = ["happy", "unhappiness", "cryptocurrency", "xyzabc123"]

for word in words:
    tokens = tokenizer.tokenize(word)
    print(f"{word} -> {tokens}")

# Output:
# happy -> ['happy']
# unhappiness -> ['un', '##happ', '##iness']
# cryptocurrency -> ['cry', '##pt', '##oc', '##ur', '##ren', '##cy']
# xyzabc123 -> ['x', '##yz', '##ab', '##c', '##12', '##3']

# Even gibberish gets tokenized (no [UNK])!

# Check UNK usage
unk_id = tokenizer.unk_token_id
encoded = tokenizer.encode("xyzabc123 unhappiness")
print(f"Contains UNK: {unk_id in encoded}")  # False
```

**Vocabulary Size Comparison:**

| Tokenizer | Vocabulary Size | OOV Rate |
|-----------|----------------|----------|
| Word-level (top 50K) | 50,000 | ~5-10% |
| BERT WordPiece | 30,522 | ~0% |
| GPT-2 BPE | 50,257 | ~0% |

**Interview Tips:**
- Subword tokenizers virtually eliminate OOV
- Character-level is extreme subword (always 0% OOV but very long sequences)
- OOV was major problem in word2vec/GloVe era
- FastText partially solved OOV with character n-grams

---

## Interview Questions

## Question 11
**How do you choose between word-level, subword, and character-level tokenization for different NLP tasks?**

**Answer:**

**Definition:**
The choice depends on vocabulary size constraints, language characteristics, and task requirements. Subword is the default for most modern NLP; character-level for morphologically rich languages or noisy text; word-level for simple tasks with clean, limited vocabulary.

**Decision Framework:**

| Factor | Word-Level | Subword | Character-Level |
|--------|-----------|---------|-----------------|
| Vocabulary size | Very large | Moderate (~30K) | Tiny (~100) |
| Sequence length | Short | Moderate | Very long |
| OOV handling | Poor | Excellent | Perfect |
| Training speed | Fast | Moderate | Slow |
| Morphology capture | None | Partial | Full |

**Choose Word-Level When:**
- Clean, well-defined vocabulary (e.g., sentiment on product reviews)
- Pre-trained word embeddings available (Word2Vec, GloVe)
- Simple classification with limited vocabulary
- Legacy systems or resource constraints

**Choose Subword When:**
- Using transformer models (BERT, GPT) - this is the default
- Multilingual or cross-lingual tasks
- Need to handle rare words, typos, new terms
- Most modern NLP applications

**Choose Character-Level When:**
- Extremely noisy text (OCR, social media with typos)
- Morphologically rich languages (Turkish, Finnish)
- Text with no clear word boundaries
- Tasks requiring character patterns (spelling correction)

**Task-Specific Recommendations:**

| Task | Recommended | Reason |
|------|-------------|--------|
| Text Classification | Subword | Balance of vocab and semantics |
| NER | Subword + alignment | Handles rare entity names |
| Machine Translation | Subword (BPE) | Cross-lingual, handles morphology |
| Spelling Correction | Character | Need character-level patterns |
| Social Media Analysis | Subword or Char | Handles hashtags, typos, slang |
| Code Generation | Byte-level BPE | Handles any character/token |

**Python Code Example:**
```python
# Comparison of tokenization approaches

text = "I loveeee #MachineLearning!!! 🚀"

# 1. Word-level (NLTK)
import nltk
word_tokens = nltk.word_tokenize(text)
# ['I', 'loveeee', '#', 'MachineLearning', '!', '!', '!', '🚀']

# 2. Subword (BERT)
from transformers import BertTokenizer
bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
subword_tokens = bert_tok.tokenize(text)
# ['i', 'love', '##ee', '##e', '#', 'machine', '##lea', '##rning', '!', ...]

# 3. Character-level
char_tokens = list(text)
# ['I', ' ', 'l', 'o', 'v', 'e', 'e', 'e', 'e', ...]

print(f"Word: {len(word_tokens)} tokens")      # ~8
print(f"Subword: {len(subword_tokens)} tokens") # ~12
print(f"Char: {len(char_tokens)} tokens")       # ~35
```

**Interview Tips:**
- Default answer: "Subword tokenization with modern transformers"
- Know trade-off: vocab size ↔ sequence length
- Character-level is rarely used standalone; often combined with CNN/RNN

---

## Question 12
**What are the trade-offs between stemming and lemmatization for information retrieval?**

**Answer:**

**Definition:**
Stemming offers faster processing and higher recall by aggressively matching word variants, while lemmatization provides higher precision with linguistically correct forms. For information retrieval, stemming is often preferred for recall-oriented systems; lemmatization for precision-critical applications.

**Trade-off Comparison:**

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| Speed | Faster (rule-based) | Slower (dictionary lookup) |
| Recall | Higher (aggressive matching) | Lower (exact forms) |
| Precision | Lower (over-stemming) | Higher (correct forms) |
| Storage | Smaller index | Larger index |
| Errors | Over/under-stemming | Few errors |

**Information Retrieval Impact:**

**Stemming Benefits:**
- Query "running" matches documents with "run", "runs", "runner"
- Reduces index size (fewer unique terms)
- Increases recall: finds more relevant documents

**Stemming Problems (Over-stemming):**
```
"universal" → "univers"
"university" → "univers"
"universe" → "univers"
# All conflated incorrectly!
```

**Lemmatization Benefits:**
- "better" → "good" (correct normalization)
- No false conflation
- Better for semantic search

**When to Choose:**

| Use Case | Choice | Reason |
|----------|--------|--------|
| Web search | Stemming | Recall matters, speed critical |
| Legal document search | Lemmatization | Precision critical |
| Question answering | Lemmatization | Semantic accuracy needed |
| Log analysis | Stemming | Fast, recall-oriented |
| Modern neural IR | Neither | Embeddings handle variations |

**Python Code Example:**
```python
# IR comparison: Stemming vs Lemmatization

from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Build simple inverted index
documents = [
    "The runner is running fast",
    "She runs every morning",
    "Run for better health"
]

def build_index(docs, normalizer):
    index = {}
    for doc_id, doc in enumerate(docs):
        for word in doc.lower().split():
            normalized = normalizer(word)
            if normalized not in index:
                index[normalized] = set()
            index[normalized].add(doc_id)
    return index

# Compare indexes
stem_index = build_index(documents, stemmer.stem)
lemma_index = build_index(documents, lambda w: lemmatizer.lemmatize(w, pos='v'))

# Search for "running"
query = "running"
stem_query = stemmer.stem(query)  # "run"
lemma_query = lemmatizer.lemmatize(query, pos='v')  # "run"

print(f"Stemming index entries: {len(stem_index)}")
print(f"Lemma index entries: {len(lemma_index)}")
# Stemming typically produces fewer unique entries
```

**Interview Tips:**
- Modern neural retrievers (DPR, ColBERT) often skip both - embeddings handle morphology
- Hybrid approach: lemmatize for indexing, stem for query expansion
- Language-specific: stemming works well for English, less so for morphologically rich languages

---

## Question 13
**How do you handle tokenization for languages without clear word boundaries (Chinese, Japanese, Thai)?**

**Answer:**

**Definition:**
Languages like Chinese, Japanese, and Thai don't use spaces between words, requiring specialized segmentation approaches. Solutions include dictionary-based methods, statistical models (HMM, CRF), or character/subword tokenization that bypasses explicit word segmentation.

**Core Challenge:**
```
English: "I love machine learning"  → Spaces define words
Chinese: "我喜欢机器学习"           → No spaces; "机器学习" = "machine learning"
Japanese: "私は機械学習が好きです"    → Mixed scripts, no spaces
Thai: "ฉันรักการเรียนรู้เครื่อง"      → No spaces
```

**Approaches:**

| Method | Description | Example Tools |
|--------|-------------|---------------|
| Dictionary-based | Maximum matching with word dictionary | Jieba (Chinese) |
| Statistical (CRF/HMM) | Learn segmentation from annotated data | Stanford Segmenter |
| Character-level | Each character is a token | Simple but loses word semantics |
| Subword (SentencePiece) | Learn segments from raw text | mBERT, XLM-R |

**Chinese Word Segmentation Example:**
```
Input: "我喜欢机器学习"
Segmented: "我 / 喜欢 / 机器 / 学习"
English: "I / like / machine / learning"
```

**Algorithm: Maximum Matching (Forward):**
1. Start from beginning of string
2. Find longest word in dictionary that matches
3. Segment that word, move forward
4. Repeat until end

**Ambiguity Problem:**
```
"研究生命起源" can be:
- "研究 / 生命 / 起源" (study origin of life)
- "研究生 / 命 / 起源" (graduate student's fate origin) ❌
```
Requires context and statistical methods to resolve.

**Python Code Example:**
```python
# Pipeline: Raw text -> Segmenter -> Tokenized text

# Method 1: Jieba for Chinese
import jieba

chinese_text = "我喜欢机器学习和深度学习"

# Default segmentation
tokens = jieba.lcut(chinese_text)
print(tokens)
# ['我', '喜欢', '机器', '学习', '和', '深度', '学习']

# Search mode (finer granularity for search engines)
tokens_search = jieba.lcut_for_search(chinese_text)
# ['我', '喜欢', '机器', '学习', '和', '深度', '学习']

# Method 2: SentencePiece (language-agnostic)
# Works directly on raw text, learns subwords without pre-segmentation
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
tokens = tokenizer.tokenize(chinese_text)
print(tokens)
# ['▁我', '喜欢', '机器', '学习', '和', '深度', '学习']

# Method 3: Character-level (simple baseline)
char_tokens = list(chinese_text)
# ['我', '喜', '欢', '机', '器', '学', '习', '和', '深', '度', '学', '习']
```

**Modern Solution:**
- Use SentencePiece-based tokenizers (mBERT, XLM-R)
- They learn subwords directly from raw text
- No need for language-specific segmenters
- Works for 100+ languages

**Interview Tips:**
- Jieba is standard for Chinese NLP
- MeCab is standard for Japanese
- SentencePiece is language-agnostic solution used by multilingual models
- Character-level works but loses semantic grouping

---

## Question 14
**What techniques work best for tokenization of social media text with informal language, hashtags, and emojis?**

**Answer:**

**Definition:**
Social media text requires specialized tokenization to handle hashtags, mentions, emojis, slang, elongation ("sooooo"), and non-standard spelling. Techniques include specialized pre-processing, social media-trained tokenizers, and emoji/hashtag-aware segmentation.

**Challenges in Social Media Text:**
```
Tweet: "OMG I loveeeee #MachineLearning 🔥🔥🔥 @elonmusk is amazingggg!!!"

Issues: 
- Elongation: "loveeeee", "amazingggg"
- Hashtag: #MachineLearning (compound word)
- Mentions: @elonmusk
- Emojis: 🔥🔥🔥
- Slang: OMG
- Punctuation: !!! (emphasis)
```

**Techniques:**

| Technique | Purpose | Example |
|-----------|---------|---------|
| Elongation normalization | "loveeeee" → "love" | Regex: reduce repeated chars |
| Hashtag segmentation | #MachineLearning → "Machine Learning" | WordNinja, CamelCase split |
| Emoji handling | Keep, remove, or convert to text | emoji library |
| Mention handling | @user → [USER] token | Anonymization |
| Slang expansion | "OMG" → "Oh My God" | Slang dictionary |

**Pre-processing Pipeline:**
1. Normalize URLs → [URL]
2. Normalize mentions → [USER]
3. Segment hashtags
4. Normalize elongation
5. Handle emojis (keep or convert to description)
6. Apply tokenizer

**Python Code Example:**
```python
# Pipeline: Raw tweet -> Preprocess -> Tokenize

import re
import emoji

def preprocess_tweet(text):
    # Step 1: Replace URLs
    text = re.sub(r'http\S+|www\S+', '[URL]', text)
    
    # Step 2: Replace mentions
    text = re.sub(r'@\w+', '[USER]', text)
    
    # Step 3: Normalize elongation (keep max 2 repeated chars)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # "loveeeee" → "lovee"
    
    # Step 4: Segment hashtags
    def segment_hashtag(match):
        tag = match.group(1)
        # CamelCase split
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
        return words
    text = re.sub(r'#(\w+)', segment_hashtag, text)
    
    # Step 5: Convert emojis to text (optional)
    text = emoji.demojize(text, delimiters=(" ", " "))
    # 🔥 → " fire "
    
    return text

# Test
tweet = "OMG I loveeeee #MachineLearning 🔥🔥 @elonmusk https://t.co/xyz"
cleaned = preprocess_tweet(tweet)
print(cleaned)
# "OMG I lovee Machine Learning fire fire [USER] [URL]"

# Now apply standard tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
tokens = tokenizer.tokenize(cleaned)
print(tokens)
```

**Specialized Tokenizers:**
- **TweetTokenizer (NLTK):** Basic social media handling
- **BERTweet:** BERT pre-trained on 850M tweets
- **RoBERTa-Twitter:** Twitter-specific language model

**Interview Tips:**
- Use pre-trained social media models (BERTweet) when possible
- Hashtag segmentation is critical for semantic understanding
- Emojis carry sentiment - don't blindly remove them
- Normalize but don't over-clean (lose information)

---

## Question 15
**How do you handle NER for nested or overlapping entities?**

**Answer:**

**Definition:**
Nested NER occurs when entities contain other entities (e.g., "Bank of America" contains location "America"). Traditional BIO tagging can only assign one label per token. Solutions include span-based models, multi-layer tagging, or hypergraph representations.

**The Problem:**
```
Text: "Bank of America headquarters in New York City"

Flat NER:   [Bank of America]_ORG ... [New York City]_LOC

Nested NER: [Bank of [America]_LOC]_ORG ... [[New York]_LOC City]_LOC
            └── ORG ──────────────┘     └── LOC ─────────────────┘
                  └── LOC ─┘                    └── LOC ──┘
```

**Why Standard BIO Fails:**
- Each token gets exactly one tag
- Cannot represent "America" as both part of ORG and as LOC simultaneously

**Solutions:**

| Approach | Description | Complexity |
|----------|-------------|------------|
| Span-based | Enumerate all spans, classify each | O(n²) spans |
| Layered/Stacked | Multiple BIO layers for nesting levels | Multiple passes |
| Hypergraph | Represent all possible nestings | Complex inference |
| Biaffine | Score all (start, end, type) tuples | O(n²) |

**Span-Based Approach (Most Common):**
1. Generate all possible spans up to max length
2. Encode each span using [start_token; end_token; span_representation]
3. Classify each span into entity type or "none"
4. Handle overlaps in post-processing

**Algorithm (Span Enumeration):**
```
Text: "Bank of America" (3 tokens)
Spans: 
  (0,0): "Bank"
  (0,1): "Bank of"  
  (0,2): "Bank of America"  ← ORG
  (1,1): "of"
  (1,2): "of America"
  (2,2): "America"          ← LOC (nested!)
```

**Python Code Example:**
```python
# Pipeline: Text -> Generate spans -> Classify spans -> Extract nested entities

import torch
import torch.nn as nn

class SpanBasedNER(nn.Module):
    def __init__(self, encoder, hidden_dim, num_labels, max_span_len=10):
        super().__init__()
        self.encoder = encoder  # BERT or similar
        self.max_span_len = max_span_len
        self.span_classifier = nn.Linear(hidden_dim * 2, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Step 1: Encode tokens
        outputs = self.encoder(input_ids, attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # Step 2: Generate span representations
        batch_size, seq_len, hidden_dim = hidden.shape
        spans = []
        
        for start in range(seq_len):
            for end in range(start, min(start + self.max_span_len, seq_len)):
                # Span representation: concat of start and end tokens
                span_repr = torch.cat([hidden[:, start], hidden[:, end]], dim=-1)
                spans.append((start, end, span_repr))
        
        # Step 3: Classify each span
        span_reprs = torch.stack([s[2] for s in spans], dim=1)
        logits = self.span_classifier(span_reprs)
        
        return logits, spans

# Inference: Extract nested entities
def extract_nested_entities(spans, predictions, tokens, id2label):
    entities = []
    for (start, end, _), pred in zip(spans, predictions):
        label = id2label[pred]
        if label != "O":  # Not "none"
            entity_text = " ".join(tokens[start:end+1])
            entities.append({
                "text": entity_text,
                "start": start,
                "end": end,
                "label": label
            })
    return entities  # Can have overlapping entities!
```

**Post-Processing for Nested Entities:**
- Keep all predicted entities (unlike flat NER where you'd choose one)
- May need to filter low-confidence overlapping predictions
- Build entity hierarchy if needed

**Interview Tips:**
- Span-based is current state-of-the-art for nested NER
- ACE, GENIA datasets have nested annotations for benchmarking
- Computational cost: O(n²) spans vs O(n) for flat NER
- In production, limit max span length for efficiency

---

## Question 16
**What strategies help with NER in low-resource languages with limited training data?**

**Answer:**

**Definition:**
Low-resource NER faces scarcity of labeled data. Strategies include cross-lingual transfer from high-resource languages, data augmentation, multilingual pre-training, active learning, and distant supervision using knowledge bases.

**Core Strategies:**

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Cross-lingual transfer | Train on English, apply to target language | Multilingual model available |
| Data augmentation | Generate synthetic training data | Some labeled data exists |
| Distant supervision | Auto-label using knowledge bases | KB coverage for target entities |
| Active learning | Strategically select samples to annotate | Limited annotation budget |
| Zero-shot transfer | Use multilingual embeddings directly | No target language data |

**1. Cross-Lingual Transfer:**
```
Train: English NER data (abundant)
Model: mBERT or XLM-R (multilingual)
Apply: Target low-resource language

Why it works: Shared multilingual representation space
```

**2. Data Augmentation Techniques:**
- **Mention replacement:** Replace entities with other entities of same type
- **Back-translation:** Translate to another language and back
- **Synonym replacement:** Replace words with synonyms
- **Entity swapping:** Swap entity positions in sentence

**3. Distant Supervision:**
```
Knowledge Base: Wikidata, DBpedia
Process:
1. Get entity list from KB (e.g., all person names)
2. Match entities in unlabeled text
3. Auto-generate noisy labels
4. Train with noise-aware methods
```

**Python Code Example:**
```python
# Strategy 1: Cross-lingual transfer with XLM-R

from transformers import AutoModelForTokenClassification, AutoTokenizer

# Train on English, apply to Hindi/Swahili/etc.
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fine-tune on English NER data
# ... training code ...

# Apply directly to target language (zero-shot)
hindi_text = "नरेंद्र मोदी भारत के प्रधानमंत्री हैं"
# Model transfers knowledge across languages

# Strategy 2: Simple data augmentation
import random

def augment_ner_data(sentence, entities, entity_dict):
    """Replace entities with other entities of same type"""
    augmented = sentence
    for entity in entities:
        if entity['type'] in entity_dict:
            replacement = random.choice(entity_dict[entity['type']])
            augmented = augmented.replace(entity['text'], replacement)
    return augmented

entity_dict = {
    'PER': ['John Smith', 'Maria Garcia', 'Chen Wei'],
    'ORG': ['Google', 'Microsoft', 'Amazon'],
    'LOC': ['Paris', 'Tokyo', 'Mumbai']
}

original = "Barack Obama visited New York"
entities = [{'text': 'Barack Obama', 'type': 'PER'}, 
            {'text': 'New York', 'type': 'LOC'}]

augmented = augment_ner_data(original, entities, entity_dict)
# "John Smith visited Mumbai"
```

**Interview Tips:**
- mBERT/XLM-R are go-to for cross-lingual transfer
- Distant supervision is noisy - use noise-robust training
- 100-500 annotated sentences can bootstrap reasonable NER with transfer learning
- Active learning can reduce annotation needs by 50-80%

---

## Question 17
**How do you implement domain adaptation for NER models (e.g., from news to biomedical text)?**

**Answer:**

**Definition:**
Domain adaptation bridges the gap between source domain (e.g., news with abundant data) and target domain (e.g., biomedical with limited data). Techniques include continued pre-training on target domain text, gradual fine-tuning, and domain-specific feature integration.

**Core Challenge:**
```
News NER:      "Apple Inc. announced a new product"
               [Apple Inc.]_ORG

Biomedical:    "Apple extract inhibits tumor growth"
               Same word "Apple" but different context (not ORG!)
               [tumor]_DISEASE, [growth]_PROCESS
```

**Domain Shift Types:**
- **Vocabulary shift:** New terms (gene names, drug names)
- **Entity type shift:** Different entity categories
- **Distribution shift:** Different writing styles, sentence structures

**Domain Adaptation Strategies:**

| Strategy | Description | Data Required |
|----------|-------------|---------------|
| Continued pre-training | Pre-train LM on target domain text | Unlabeled target text |
| Fine-tuning | Fine-tune on small target labeled data | Labeled target data |
| Multi-task learning | Joint training on source + target | Both datasets |
| Feature augmentation | Add domain-specific features | Domain gazetteers |

**Step-by-Step Approach:**
1. **Continued Pre-training:** Pre-train BERT on unlabeled biomedical text (PubMed)
2. **Domain-specific vocabulary:** Add domain terms to tokenizer
3. **Fine-tune on target NER:** Use limited biomedical NER data
4. **Add gazetteers:** Include drug/disease dictionaries as features

**Python Code Example:**
```python
# Pipeline: Base model -> Domain pre-train -> Fine-tune NER

from transformers import (
    AutoModelForMaskedLM, 
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer, 
    TrainingArguments
)

# Step 1: Continued pre-training on domain text
def domain_pretrain(base_model, domain_corpus):
    """
    Continue MLM pre-training on biomedical text
    """
    model = AutoModelForMaskedLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Train on domain corpus with MLM objective
    # ... training loop ...
    
    model.save_pretrained("domain-adapted-bert")
    return model

# Step 2: Fine-tune for NER on target domain
def finetune_ner(domain_model_path, ner_dataset, num_labels):
    model = AutoModelForTokenClassification.from_pretrained(
        domain_model_path,
        num_labels=num_labels
    )
    
    training_args = TrainingArguments(
        output_dir="./ner-model",
        num_train_epochs=5,
        learning_rate=2e-5,
        # Lower learning rate for domain adaptation
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ner_dataset
    )
    trainer.train()
    return model

# Step 3: Use domain-specific pre-trained models
# BioBERT, PubMedBERT, ClinicalBERT, SciBERT
from transformers import AutoModel

biomedical_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
# Already pre-trained on PubMed + PMC
```

**Pre-trained Domain Models:**
| Domain | Model | Pre-training Data |
|--------|-------|-------------------|
| Biomedical | BioBERT, PubMedBERT | PubMed abstracts |
| Clinical | ClinicalBERT | MIMIC clinical notes |
| Scientific | SciBERT | Semantic Scholar papers |
| Legal | LegalBERT | Legal documents |
| Finance | FinBERT | Financial news |

**Interview Tips:**
- Use existing domain-specific BERT when available
- Continued pre-training helps even with 10-50MB of domain text
- Gradual unfreezing: fine-tune top layers first, then all layers
- Domain gazetteers significantly boost NER on specialized entities

---

## Question 18
**What approaches work best for NER in noisy text like social media or OCR output?**

**Answer:**

**Definition:**
Noisy text NER requires robustness to spelling errors, missing punctuation, informal language, and OCR artifacts. Approaches include noise-aware training, character-level representations, spelling correction pre-processing, and models trained on noisy data.

**Types of Noise:**

| Source | Noise Type | Example |
|--------|-----------|---------|
| Social media | Informal spelling | "u", "2morrow", "gonna" |
| OCR | Character confusion | "rn" → "m", "0" → "O" |
| ASR | Phonetic errors | "there" vs "their" |
| Typos | Missing/swapped chars | "Gogle" instead of "Google" |

**Strategies:**

| Approach | Description | Effectiveness |
|----------|-------------|---------------|
| Noise injection training | Add synthetic noise during training | High |
| Character embeddings | Capture sub-word patterns | Medium-High |
| Spelling correction | Pre-process to correct errors | Medium |
| Robust tokenization | Handle noisy tokens gracefully | Medium |
| Pre-trained on noisy data | BERTweet, etc. | High |

**1. Noise Injection Training:**
```python
def inject_noise(text, noise_prob=0.1):
    """Add synthetic noise for robust training"""
    noisy = []
    for char in text:
        if random.random() < noise_prob:
            noise_type = random.choice(['delete', 'swap', 'insert', 'replace'])
            if noise_type == 'delete':
                continue
            elif noise_type == 'swap' and len(noisy) > 0:
                noisy[-1], char = char, noisy[-1]
            elif noise_type == 'insert':
                noisy.append(random.choice('abcdefghijklmnopqrstuvwxyz'))
            elif noise_type == 'replace':
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
        noisy.append(char)
    return ''.join(noisy)
```

**2. Character-Level Representations:**
- CNN or LSTM over characters captures morphology
- Robust to OOV and spelling variations
- Combine with word embeddings

**Python Code Example:**
```python
# Pipeline: Noisy text -> Normalize -> Character+Word features -> NER

import re
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Step 1: Noise normalization
def normalize_noisy_text(text):
    # Normalize repeated characters
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # "sooooo" -> "soo"
    
    # Common OCR fixes
    ocr_fixes = {'0': 'O', '1': 'l', '|': 'l', 'rn': 'm'}
    for error, fix in ocr_fixes.items():
        text = text.replace(error, fix)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

# Step 2: Use noise-robust model
# BERTweet is trained on noisy Twitter data
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForTokenClassification.from_pretrained(
    "vinai/bertweet-base",
    num_labels=9
)

# Step 3: Character-level CNN (conceptual)
import torch.nn as nn

class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim, num_filters):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim)
        self.conv = nn.Conv1d(char_embed_dim, num_filters, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, char_ids):
        # char_ids: (batch, word_len)
        embedded = self.char_embed(char_ids)  # (batch, word_len, embed)
        embedded = embedded.permute(0, 2, 1)   # (batch, embed, word_len)
        conv_out = self.conv(embedded)
        pooled = self.pool(conv_out).squeeze(-1)
        return pooled  # Character-level word representation
```

**Practical Pipeline:**
1. Light normalization (don't over-correct)
2. Use character-level features
3. Train with noise injection
4. Use pre-trained models on similar noisy data

**Interview Tips:**
- Don't over-normalize - may lose information
- Character CNNs help with spelling variations
- BERTweet/RoBERTa-Twitter for social media
- OCR post-correction is a separate research area

---

## Question 19
**How do you handle entity disambiguation and linking NER results to knowledge bases?**

**Answer:**

**Definition:**
Entity Linking (EL) connects extracted entity mentions to unique entries in a knowledge base (Wikipedia, Wikidata). It resolves ambiguity when the same name refers to different entities (e.g., "Apple" → fruit or company). The process involves candidate generation, candidate ranking, and NIL prediction.

**The Problem:**
```
"Michael Jordan was a great basketball player."
"Michael Jordan is a famous ML professor."

Both "Michael Jordan" mentions → different KB entries!
- Basketball player: Q41421 (Wikidata)
- ML Professor: Q3308285 (Wikidata)
```

**Entity Linking Pipeline:**

| Step | Description | Output |
|------|-------------|--------|
| 1. Mention Detection | Find entity spans (NER) | "Michael Jordan" |
| 2. Candidate Generation | Get potential KB matches | [Q41421, Q3308285, ...] |
| 3. Candidate Ranking | Score candidates using context | Q41421: 0.95, Q3308285: 0.03 |
| 4. NIL Prediction | Detect if entity not in KB | "new_entity" or linked ID |

**Candidate Generation Methods:**
- **Alias table:** Pre-built mapping of surface forms to candidates
- **Fuzzy matching:** Handle typos, variations
- **Prior probability:** P(entity | mention) from Wikipedia anchor text

**Candidate Ranking Features:**
- Textual similarity (mention vs entity description)
- Context compatibility (surrounding text vs entity context)
- Coherence (other entities in same document)
- Prior probability (popularity)

**Python Code Example:**
```python
# Pipeline: Mention -> Generate candidates -> Rank -> Link

# Method 1: Using spaCy EntityLinker
import spacy

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entity_linker", after="ner")

text = "Apple was founded by Steve Jobs"
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text} -> {ent.kb_id_}")
# Apple -> Q312 (Apple Inc.)
# Steve Jobs -> Q19837 (Steve Jobs)

# Method 2: Custom Entity Linker
from sentence_transformers import SentenceTransformer, util

class SimpleEntityLinker:
    def __init__(self, kb_descriptions):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.kb = kb_descriptions  # {entity_id: description}
        self.kb_embeddings = self.model.encode(list(kb_descriptions.values()))
        self.kb_ids = list(kb_descriptions.keys())
    
    def link(self, mention, context, top_k=5):
        # Step 1: Generate candidates (simplified: use all KB)
        # In practice: use alias table for efficient candidate generation
        
        # Step 2: Create query from mention + context
        query = f"{mention}: {context}"
        query_emb = self.model.encode(query)
        
        # Step 3: Rank candidates by similarity
        scores = util.cos_sim(query_emb, self.kb_embeddings)[0]
        top_indices = scores.argsort(descending=True)[:top_k]
        
        results = [(self.kb_ids[i], float(scores[i])) for i in top_indices]
        return results

# Usage
kb = {
    "Q41421": "Michael Jordan, American basketball player for Chicago Bulls",
    "Q3308285": "Michael I. Jordan, American computer scientist, machine learning",
    "Q312": "Apple Inc., American technology company founded by Steve Jobs"
}

linker = SimpleEntityLinker(kb)
results = linker.link("Michael Jordan", "great basketball player who won 6 championships")
# [('Q41421', 0.85), ('Q3308285', 0.32), ...]
```

**NIL Prediction:**
When entity is not in knowledge base:
- Set threshold on ranking score
- Train binary classifier for in-KB vs out-of-KB
- Cluster NIL mentions for new entity discovery

**Interview Tips:**
- Entity linking is harder than NER - same mention, different entities
- Wikipedia is most common KB; Wikidata for structured data
- Prior probability (popularity) is a strong baseline feature
- Modern approach: bi-encoder for candidate generation, cross-encoder for ranking

---

## Question 20
**What are the challenges of POS tagging for morphologically rich languages?**

**Answer:**

**Definition:**
Morphologically rich languages (Turkish, Finnish, Hungarian, Arabic) have complex word structures with many inflections, cases, and affixes. This creates massive vocabulary, high OOV rates, and ambiguity that simple POS taggers struggle with.

**Core Challenges:**

| Challenge | Description | Example (Turkish) |
|-----------|-------------|-------------------|
| Vocabulary explosion | One lemma → many forms | "ev" (house) → evler, evde, evden, evlerin, ... |
| OOV rate | Unseen word forms | Up to 50% OOV in test data |
| Ambiguity | Same suffix, different meanings | "-ler" = plural OR 3rd person |
| Agglutination | Multiple morphemes per word | "evlerinden" = house+PLURAL+from+their |
| Long dependencies | Affixes interact with each other | Root properties affect suffix choices |

**Example - Turkish Agglutination:**
```
Word: "Avrupalılaştıramadıklarımızdanmışsınızcasına"
Meaning: "As if you were one of those we could not Europeanize"
One word = entire sentence!
```

**Solutions:**

| Solution | Description | Effectiveness |
|----------|-------------|---------------|
| Subword tokenization | BPE/WordPiece handles morphemes | High |
| Morphological analysis | Explicit morpheme segmentation | High |
| Character-level models | Learn morphological patterns | Medium-High |
| Joint morphology+POS | Predict both together | High |
| Universal Dependencies | Cross-lingual annotation standards | Enables transfer |

**Python Code Example:**
```python
# Pipeline: Morphologically rich text -> Subword/Morphological analysis -> POS

# Approach 1: Subword tokenization (BPE handles morphology implicitly)
from transformers import AutoTokenizer

# Turkish BERT uses subword tokenization
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

turkish_word = "evlerinden"  # from their houses
tokens = tokenizer.tokenize(turkish_word)
print(tokens)
# ['ev', '##ler', '##in', '##den'] - morphemes separated!

# Approach 2: Morphological analyzer (language-specific)
# For Turkish: Zemberek, TRMorph
# Conceptual example:
def morphological_analysis(word):
    """
    Returns morpheme breakdown
    Example output for 'evlerinden':
    {
        'root': 'ev',
        'morphemes': ['ev', 'ler', 'i', 'n', 'den'],
        'tags': ['NOUN', 'PLURAL', 'POSS.3SG', 'ABL'],
        'pos': 'NOUN'
    }
    """
    pass

# Approach 3: Character-level CNN for morphology
import torch.nn as nn

class MorphAwareTagger(nn.Module):
    def __init__(self, char_vocab, word_vocab, num_tags):
        super().__init__()
        # Character-level representation
        self.char_embed = nn.Embedding(char_vocab, 50)
        self.char_cnn = nn.Conv1d(50, 100, kernel_size=3)
        
        # Word-level (subword) representation
        self.word_embed = nn.Embedding(word_vocab, 300)
        
        # Combine and classify
        self.lstm = nn.LSTM(400, 200, bidirectional=True)
        self.classifier = nn.Linear(400, num_tags)
    
    def forward(self, word_ids, char_ids):
        # Get character-level features (captures morphology)
        char_emb = self.char_embed(char_ids)
        char_features = self.char_cnn(char_emb.permute(0, 2, 1))
        char_repr = char_features.max(dim=-1)[0]
        
        # Combine with word embeddings
        word_emb = self.word_embed(word_ids)
        combined = torch.cat([word_emb, char_repr], dim=-1)
        
        # Sequence labeling
        lstm_out, _ = self.lstm(combined)
        logits = self.classifier(lstm_out)
        return logits
```

**Best Practices:**
1. Use multilingual models (mBERT, XLM-R) pre-trained on target language
2. Character-level features are essential
3. Consider joint morphological analysis + POS tagging
4. Universal Dependencies provides consistent cross-lingual tagset

**Interview Tips:**
- Subword tokenization is key breakthrough for morphologically rich languages
- Finnish/Hungarian can have 10,000+ forms per lemma
- Turkish BERT, Arabic BERT exist for major morphologically rich languages
- POS accuracy typically 5-10% lower than English for these languages

---

## Question 21
**How do you handle ambiguous words that can have multiple POS tags depending on context?**

**Answer:**

**Definition:**
POS ambiguity occurs when the same word can function as different parts of speech (e.g., "book" as noun or verb). Resolution requires contextual understanding using surrounding words, syntactic patterns, and sequential models that consider the entire sentence.

**Common Ambiguous Words:**

| Word | POS 1 | POS 2 | Example |
|------|-------|-------|---------|
| book | NN (noun) | VB (verb) | "Read the book" vs "Book a flight" |
| run | VB (verb) | NN (noun) | "Run fast" vs "A morning run" |
| that | DT (determiner) | IN (conjunction) | "That book" vs "I know that..." |
| light | JJ (adjective) | NN (noun) | "Light weight" vs "Turn on the light" |

**Resolution Approaches:**

| Method | How It Resolves Ambiguity |
|--------|---------------------------|
| N-gram context | Look at surrounding word POS patterns |
| HMM | Transition probabilities between tags |
| CRF | Global sequence optimization |
| BiLSTM | Bidirectional context encoding |
| Transformer | Full sentence attention |

**Contextual Clues:**
```
"I will book a table" 
- "will" + [WORD] → [WORD] is likely VB (verb after modal)
- "a" + [WORD] → Unlikely (article before verb is rare)
- Sequence context: VB wins

"I read the book"
- "the" + [WORD] → [WORD] is likely NN (article before noun)
- NN wins
```

**Python Code Example:**
```python
# Pipeline: Sentence -> Context-aware tagger -> Resolved POS

import spacy

nlp = spacy.load("en_core_web_sm")

# Same word, different contexts
sentences = [
    "I need to book a flight",      # book = VERB
    "I read an interesting book",   # book = NOUN
    "The light is too bright",      # light = NOUN
    "This bag is very light"        # light = ADJ
]

for sent in sentences:
    doc = nlp(sent)
    for token in doc:
        if token.text.lower() in ['book', 'light']:
            print(f"'{sent}' -> {token.text}: {token.pos_}")

# Output:
# 'I need to book a flight' -> book: VERB
# 'I read an interesting book' -> book: NOUN
# 'The light is too bright' -> light: NOUN
# 'This bag is very light' -> light: ADJ

# How context helps (simplified demonstration)
def get_context_features(tokens, idx):
    """Extract context features for disambiguation"""
    features = {
        'word': tokens[idx].lower(),
        'prev_word': tokens[idx-1].lower() if idx > 0 else '<START>',
        'next_word': tokens[idx+1].lower() if idx < len(tokens)-1 else '<END>',
        'prev_pos': None,  # From previous prediction
    }
    
    # Pattern-based hints
    if features['prev_word'] in ['the', 'a', 'an']:
        features['likely_noun'] = True
    if features['prev_word'] in ['will', 'can', 'should', 'to']:
        features['likely_verb'] = True
        
    return features
```

**Why Transformers Excel:**
- Self-attention sees entire sentence at once
- Can capture long-range dependencies
- Pre-training learns grammatical patterns
- Contextual embeddings differ for same word in different contexts

**Interview Tips:**
- ~11% of English words are ambiguous in POS
- Contextual embeddings (BERT) give different vectors for same word in different contexts
- HMM/CRF use transition probabilities: P(VB|MD) > P(NN|MD)
- Accuracy for ambiguous words is typically lower than overall accuracy

---

## Question 22
**What is the role of context window in POS tagging and how do neural models capture it?**

**Answer:**

**Definition:**
Context window refers to the surrounding words considered when predicting a token's POS tag. Larger context enables better disambiguation. Neural models capture context through recurrent connections (LSTM), convolutional filters (CNN), or self-attention (Transformer).

**Context Window Importance:**
```
"The old man the boats"
Without context: "man" → likely NOUN (most common)
With context: "old man" → "man" is VERB (old people operate the boats)

Full context needed to resolve this garden-path sentence.
```

**Context Window by Model Type:**

| Model | Context Window | Mechanism |
|-------|---------------|-----------|
| N-gram | Fixed (2-5 words) | Look at adjacent n-grams |
| HMM | Limited (Markov) | Previous state only |
| CRF | Fixed feature window | Handcrafted features |
| LSTM | Entire sequence | Hidden state propagation |
| BiLSTM | Entire sequence (both directions) | Forward + backward states |
| Transformer | Entire sequence | Self-attention over all tokens |

**How Neural Models Capture Context:**

**1. BiLSTM:**
```
Forward:  →  h₁ → h₂ → h₃ → h₄  (past context)
Backward: ←  h₁ ← h₂ ← h₃ ← h₄  (future context)
Combined: [h_forward; h_backward] for each position
```

**2. Transformer Self-Attention:**
```
Every token attends to every other token
Attention(Q, K, V) = softmax(QK^T / √d) V

Token "man" attends to:
- "old" (helps determine subject/verb)
- "boats" (object → "man" is verb)
```

**Python Code Example:**
```python
# Pipeline: Demonstrate context capture in different models

import torch
import torch.nn as nn

# 1. Fixed Window (Traditional)
class WindowTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, window_size, num_tags):
        super().__init__()
        self.window_size = window_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Context = current word + window on each side
        self.fc = nn.Linear(embed_dim * (2 * window_size + 1), num_tags)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embed(x)  # (batch, seq_len, embed)
        # Create windows (simplified - would need padding)
        # Each position sees [-window, ..., 0, ..., +window]
        pass

# 2. BiLSTM (Full sequence context)
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
    
    def forward(self, x):
        embedded = self.embed(x)
        lstm_out, _ = self.lstm(embedded)
        # lstm_out contains context from entire sequence
        logits = self.fc(lstm_out)
        return logits

# 3. Transformer (Self-attention)
# Each position attends to all other positions
class TransformerTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_tags):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_tags)
    
    def forward(self, x):
        embedded = self.embed(x).permute(1, 0, 2)  # (seq, batch, embed)
        # Self-attention: every token sees every other token
        attended, _ = self.attention(embedded, embedded, embedded)
        logits = self.fc(attended.permute(1, 0, 2))
        return logits
```

**Context Window Trade-offs:**

| Aspect | Small Window | Large Window |
|--------|--------------|--------------|
| Speed | Faster | Slower |
| Memory | Less | More |
| Long dependencies | Misses | Captures |
| Local patterns | Good | Good |

**Interview Tips:**
- Transformers have unlimited context (within sequence length)
- BiLSTM captures context but may forget distant information
- BERT uses 512 token context window (sufficient for most POS tasks)
- Context is why neural POS taggers achieve 97%+ accuracy

---

## Question 23
**How do you evaluate NER systems? Explain precision, recall, F1, and entity-level vs token-level metrics.**

**Answer:**

**Definition:**
NER evaluation uses precision (correct predictions / all predictions), recall (correct predictions / all ground truth), and F1 (harmonic mean). Entity-level evaluation requires exact span and type match; token-level evaluates each token independently.

**Core Metrics:**

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}$$

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}$$

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Entity-Level vs Token-Level:**

| Metric Type | What Counts as Correct | Strictness |
|-------------|------------------------|------------|
| Entity-level (Exact) | Exact span + correct type | Strict |
| Entity-level (Partial) | Overlap + correct type | Medium |
| Token-level | Each token independently | Lenient |

**Example:**
```
Ground Truth: [New York City]_LOC
Prediction:   [New York]_LOC [City]_LOC

Entity-level (Exact): 
  - 0 TP (spans don't match exactly)
  - 2 FP (predicted entities)
  - 1 FN (missed ground truth)

Token-level:
  - 3 TP (New, York, City all tagged LOC)
  - 0 FP, 0 FN
```

**Evaluation Variants:**

| Variant | Requirement |
|---------|-------------|
| Strict | Exact boundary + exact type |
| Type-only | Correct type (any overlap) |
| Boundary-only | Exact boundary (any type) |
| Partial | Some overlap + correct type |

**Python Code Example:**
```python
# Pipeline: Predictions + Ground Truth -> Compute metrics

from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import classification_report as token_report

# Entity-level evaluation using seqeval
y_true = [['O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'I-LOC']]
y_pred = [['O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O']]

# Entity-level F1 (seqeval handles BIO correctly)
print(classification_report(y_true, y_pred))
# Note: "New York City" with last token wrong = entity missed

# Token-level evaluation (each token independently)
y_true_flat = [tag for seq in y_true for tag in seq]
y_pred_flat = [tag for seq in y_pred for tag in seq]
print(token_report(y_true_flat, y_pred_flat))

# Manual entity extraction for exact match
def extract_entities(tags):
    entities = []
    current = None
    start = None
    
    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            if current:
                entities.append((start, i, current))
            current = tag[2:]
            start = i
        elif tag.startswith('I-') and current:
            continue
        else:
            if current:
                entities.append((start, i, current))
            current = None
    
    if current:
        entities.append((start, len(tags), current))
    
    return set(entities)

# Compute entity-level metrics manually
true_entities = extract_entities(y_true[0])
pred_entities = extract_entities(y_pred[0])

tp = len(true_entities & pred_entities)
fp = len(pred_entities - true_entities)
fn = len(true_entities - pred_entities)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Entity P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}")
```

**Micro vs Macro Averaging:**
- **Micro:** Aggregate TP, FP, FN across all classes, then compute
- **Macro:** Compute per-class, then average (treats rare classes equally)

**Interview Tips:**
- Always report entity-level F1 for NER (industry standard)
- CoNLL-2003 uses entity-level exact match
- seqeval library is standard for NER evaluation
- Token-level inflates scores (partial entities count as correct)

---

## Question 24
**What is few-shot NER and how can you identify new entity types with minimal examples?**

**Answer:**

**Definition:**
Few-shot NER identifies entities of new types using only a handful of examples (1-10), without retraining the model from scratch. Approaches include prompt-based methods, prototype networks, and in-context learning with large language models.

**Core Challenge:**
```
Traditional NER: Train on 10,000+ examples per entity type
Few-shot NER: New entity type "PRODUCT" with only 5 examples

Examples provided:
- "iPhone 15" -> PRODUCT
- "Tesla Model 3" -> PRODUCT  
- "MacBook Pro" -> PRODUCT

Now identify: "The Galaxy S24 has great features"
                    ↓
             [Galaxy S24]_PRODUCT
```

**Approaches:**

| Method | Description | LLM Required |
|--------|-------------|--------------|
| Prototype Networks | Learn entity type embeddings | No |
| Prompt-based | Frame as text generation/QA | Yes |
| In-context Learning | Provide examples in prompt | Yes (GPT-3+) |
| Meta-learning | Learn to learn from few examples | No |

**1. Prototype Networks:**
- Compute average embedding for each entity type from examples
- Classify new mentions by nearest prototype
- Works with small encoder models

**2. In-Context Learning (LLMs):**
```
Prompt:
"Identify PRODUCT entities in text. Examples:
'I bought an iPhone 15' -> 'iPhone 15' is PRODUCT
'Tesla Model 3 is fast' -> 'Tesla Model 3' is PRODUCT

Now identify PRODUCT in: 'The Galaxy S24 has great features'"

Output: "Galaxy S24"
```

**Python Code Example:**
```python
# Approach 1: Prototype-based Few-shot NER

import torch
from transformers import AutoTokenizer, AutoModel

class PrototypeFewShotNER:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.prototypes = {}  # entity_type -> embedding
    
    def get_span_embedding(self, text, start, end):
        """Get embedding for entity span"""
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        # Average token embeddings in span
        span_emb = outputs.last_hidden_state[0, start:end].mean(dim=0)
        return span_emb
    
    def add_examples(self, entity_type, examples):
        """
        examples: list of (text, entity_text) tuples
        """
        embeddings = []
        for text, entity_text in examples:
            start = text.find(entity_text)
            end = start + len(entity_text)
            emb = self.get_span_embedding(text, start, end)
            embeddings.append(emb)
        
        # Prototype = average of example embeddings
        self.prototypes[entity_type] = torch.stack(embeddings).mean(dim=0)
    
    def predict(self, text, candidate_spans):
        """Classify candidate spans"""
        results = []
        for start, end in candidate_spans:
            span_emb = self.get_span_embedding(text, start, end)
            
            # Find nearest prototype
            best_type = None
            best_sim = -1
            for etype, proto in self.prototypes.items():
                sim = torch.cosine_similarity(span_emb, proto, dim=0)
                if sim > best_sim:
                    best_sim = sim
                    best_type = etype
            
            results.append((start, end, best_type, best_sim.item()))
        return results

# Usage
ner = PrototypeFewShotNER()

# Few-shot examples for new entity type
ner.add_examples('PRODUCT', [
    ("I bought an iPhone 15", "iPhone 15"),
    ("Tesla Model 3 is fast", "Tesla Model 3"),
    ("Love my MacBook Pro", "MacBook Pro")
])

# Predict on new text
# candidates = get_candidate_spans(text)  # From NP chunker or sliding window

# Approach 2: LLM In-Context Learning
from openai import OpenAI

def few_shot_ner_llm(examples, text, entity_type):
    prompt = f"Identify {entity_type} entities in text.\n\nExamples:\n"
    for ex_text, entity in examples:
        prompt += f"Text: '{ex_text}' -> {entity_type}: '{entity}'\n"
    prompt += f"\nNow identify {entity_type} in: '{text}'"
    
    # Call LLM API
    # response = client.completions.create(...)
    return prompt  # Would return actual entities from LLM
```

**Interview Tips:**
- Few-shot NER is hot research area
- GPT-3/4 with in-context learning is surprisingly effective
- Prototype networks work without large LLMs
- SetFit and sentence-transformers enable efficient few-shot learning

---

## Question 25
**How do you handle tokenization for domain-specific texts like legal, medical, or code?**

**Answer:**

**Definition:**
Domain-specific tokenization requires handling specialized vocabulary (medical terms, legal jargon, code syntax) that general tokenizers fragment poorly. Solutions include domain-specific pre-trained tokenizers, vocabulary expansion, and custom tokenization rules.

**Domain Challenges:**

| Domain | Challenge | Example |
|--------|-----------|---------|
| Medical | Complex terms, abbreviations | "COVID-19", "mg/dL", "t.i.d." |
| Legal | Latin phrases, citations | "habeas corpus", "42 U.S.C. § 1983" |
| Code | Syntax, identifiers, operators | "getUserById()", "!=", "//comment" |
| Scientific | Chemical formulas, equations | "H₂SO₄", "E=mc²" |

**General Tokenizer Problems:**
```
Medical: "acetaminophen" → ['ace', '##tam', '##ino', '##phen']  # 4 tokens
Code: "getUserById" → ['get', '##User', '##By', '##Id']  # Loses semantics
Legal: "§1983" → ['§', '##19', '##83']  # Splits legal reference
```

**Solutions:**

| Solution | Description | When to Use |
|----------|-------------|-------------|
| Domain pre-trained model | Use BioBERT, LegalBERT, CodeBERT | Best option if available |
| Vocabulary extension | Add domain terms to tokenizer | Limited new terms |
| Custom pre-tokenization | Domain-aware splitting rules | Complex syntax (code) |
| Train custom tokenizer | Train BPE on domain corpus | Full control needed |

**Python Code Example:**
```python
# Pipeline: Domain text -> Domain-aware tokenization

# Solution 1: Use domain-specific models
from transformers import AutoTokenizer

# Medical
bio_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
tokens = bio_tokenizer.tokenize("acetaminophen dosage 500mg")
# Better handling of medical terms

# Code
code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
tokens = code_tokenizer.tokenize("def getUserById(user_id):")
# Understands code syntax

# Legal
legal_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Solution 2: Add tokens to vocabulary
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Add domain-specific tokens
new_tokens = ['covid-19', 'mrna', 'acetaminophen', '§', 'getUserById']
num_added = tokenizer.add_tokens(new_tokens)
print(f"Added {num_added} tokens")

# Now these tokenize as single tokens
tokens = tokenizer.tokenize("covid-19 vaccine")
# ['covid-19', 'vaccine'] instead of ['co', '##vid', '-', '19', ...]

# Remember to resize model embeddings!
# model.resize_token_embeddings(len(tokenizer))

# Solution 3: Custom pre-tokenization for code
import re

def code_pretokenize(code):
    """Split code into meaningful units before subword tokenization"""
    # Split camelCase: getUserById -> get User By Id
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    # Split snake_case: user_id -> user id
    code = re.sub(r'_', ' ', code)
    # Keep operators together
    code = re.sub(r'([!=<>]=)', r' \1 ', code)
    return code

code = "getUserById(userId) != null"
pretokenized = code_pretokenize(code)
# "get User By Id(user Id) != null"

# Solution 4: Train custom BPE tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_domain_tokenizer(corpus_file, vocab_size=30000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    tokenizer.train([corpus_file], trainer)
    return tokenizer

# Train on medical corpus for medical-specific BPE
```

**Best Practices:**
1. First check if domain-specific model exists (BioBERT, CodeBERT, etc.)
2. If fine-tuning, add frequent domain terms to vocabulary
3. For code: use dedicated code models (CodeBERT, GraphCodeBERT)
4. Never split important domain identifiers

**Interview Tips:**
- Domain-specific models are pre-trained on domain corpora (PubMed, legal docs, GitHub)
- Adding tokens requires resizing model embeddings and additional training
- Code tokenization is unique - needs to handle syntax meaningfully
- Medical: BioBERT, PubMedBERT; Legal: LegalBERT; Code: CodeBERT

---

## Question 26
**What is the impact of tokenization choices on model vocabulary size and training efficiency?**

**Answer:**

**Definition:**
Tokenization directly determines vocabulary size, sequence length, and memory usage. Larger vocabularies mean more embedding parameters but shorter sequences; smaller vocabularies mean longer sequences but fewer parameters. This trade-off affects training speed, memory, and model capacity.

**Core Trade-offs:**

| Tokenization | Vocab Size | Sequence Length | Parameters | Memory |
|--------------|------------|-----------------|------------|--------|
| Character-level | ~100 | Very long | Few embeddings | High (long sequences) |
| Subword (BPE) | ~30K-50K | Moderate | Moderate | Balanced |
| Word-level | 100K+ | Short | Many embeddings | High (large vocab) |

**Impact Analysis:**

**1. Vocabulary Size Impact:**
```
Embedding layer size = vocab_size × embedding_dim

Word-level:  100,000 × 768 = 76.8M parameters (embeddings alone!)
Subword:     30,000 × 768 = 23M parameters
Character:   256 × 768 = 0.2M parameters
```

**2. Sequence Length Impact:**
```
Text: "internationalization"

Word-level:    ["internationalization"]           → 1 token
Subword (BPE): ["international", "ization"]       → 2 tokens
Character:     ['i','n','t','e','r',...,'n']      → 20 tokens

Transformer complexity: O(n²) where n = sequence length
Longer sequences = slower training + more memory
```

**3. Training Efficiency Equation:**

$$\text{Compute} \propto \text{Batch Size} \times \text{Seq Length}^2 \times \text{Vocab Size}$$

**Python Code Example:**
```python
# Pipeline: Compare tokenization choices and their impacts

from transformers import AutoTokenizer
import time

text = "The internationalization of artificial intelligence research continues."

tokenizers = {
    'bert': 'bert-base-uncased',
    'gpt2': 'gpt2',
    'char': None  # Manual character-level
}

# Compare token counts
print("Tokenization comparison:")
print("-" * 50)

# BERT (WordPiece ~30K vocab)
bert_tok = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_tokens = bert_tok.tokenize(text)
print(f"BERT WordPiece: {len(bert_tokens)} tokens")
print(f"  Vocab size: {bert_tok.vocab_size}")
print(f"  Tokens: {bert_tokens[:5]}...")

# GPT-2 (BPE ~50K vocab)
gpt_tok = AutoTokenizer.from_pretrained('gpt2')
gpt_tokens = gpt_tok.tokenize(text)
print(f"GPT-2 BPE: {len(gpt_tokens)} tokens")
print(f"  Vocab size: {gpt_tok.vocab_size}")

# Character-level
char_tokens = list(text)
print(f"Character: {len(char_tokens)} tokens")
print(f"  Vocab size: ~256")

# Memory/compute estimation
def estimate_memory(seq_len, vocab_size, hidden_dim=768, batch_size=32):
    """Rough memory estimate for transformer forward pass"""
    # Embedding lookup
    embed_params = vocab_size * hidden_dim
    
    # Attention matrices (simplified)
    attention_mem = batch_size * seq_len * seq_len * 4  # float32
    
    # Sequence processing
    hidden_mem = batch_size * seq_len * hidden_dim * 4
    
    total_mb = (embed_params + attention_mem + hidden_mem) / (1024 * 1024)
    return total_mb

print("\nMemory estimates (batch=32):")
print(f"BERT (seq={len(bert_tokens)}): {estimate_memory(len(bert_tokens), 30522):.1f} MB")
print(f"Char (seq={len(char_tokens)}): {estimate_memory(len(char_tokens), 256):.1f} MB")
```

**Practical Guidelines:**

| Scenario | Recommendation |
|----------|----------------|
| General NLP | Subword (30K-50K vocab) |
| Multilingual | Larger vocab (100K+) |
| Memory constrained | Smaller vocab, accept longer sequences |
| Speed critical | Balanced vocab (30K) |
| Rare vocabulary domain | Smaller subword vocab |

**Interview Tips:**
- BERT: 30K vocab, GPT-2: 50K vocab, XLM-R: 250K vocab (multilingual)
- Doubling vocab doubles embedding parameters
- Sequence length has O(n²) impact on attention computation
- Subword is the sweet spot - that's why everyone uses it

---

## Question 27
**How do multilingual models like mBERT handle tokenization across different scripts and languages?**

**Answer:**

**Definition:**
Multilingual BERT (mBERT) uses a shared WordPiece vocabulary trained on 104 languages, enabling cross-lingual transfer. It handles multiple scripts (Latin, Cyrillic, Chinese, Arabic) by including script-specific characters and subwords in a single unified vocabulary.

**Core Mechanism:**

| Aspect | How mBERT Handles It |
|--------|---------------------|
| Vocabulary | Shared 110K WordPiece tokens across all languages |
| Scripts | Includes characters from all scripts in vocab |
| Training | Masked LM on Wikipedia for each language |
| Representation | Shared embedding space across languages |

**Vocabulary Composition:**
```
mBERT Vocabulary (~110K tokens):
- English subwords: "the", "ing", "tion", ...
- Chinese characters: "的", "是", "我", ...
- Arabic subwords: "ال", "من", ...
- Cyrillic: "и", "не", "это", ...
- Shared subwords: Numbers, punctuation, common patterns
```

**Cross-lingual Transfer Intuition:**
```
English: "I love cats"
French:  "J'aime les chats"
German:  "Ich liebe Katzen"

Shared structure → Similar representations → Transfer possible
```

**Vocabulary Distribution Challenge:**
```
High-resource languages (English) → More vocabulary slots
Low-resource languages → Fewer slots, more character-level splitting

"internationalization" (English) → 2-3 tokens
Same length word in low-resource language → 5-10 tokens
```

**Python Code Example:**
```python
# Pipeline: See how mBERT tokenizes different languages

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

texts = {
    'English': "Machine learning is powerful",
    'French': "L'apprentissage automatique est puissant",
    'Chinese': "机器学习很强大",
    'Arabic': "التعلم الآلي قوي",
    'Hindi': "मशीन लर्निंग शक्तिशाली है",
    'Russian': "Машинное обучение мощное"
}

print("mBERT Tokenization across languages:")
print("-" * 60)

for lang, text in texts.items():
    tokens = tokenizer.tokenize(text)
    print(f"{lang}: {text}")
    print(f"  Tokens ({len(tokens)}): {tokens}")
    print()

# Output shows:
# - English: Fewer subwords (well-represented in vocab)
# - Chinese: Character-level + some common phrases
# - Low-resource: More fragmentation

# Vocabulary analysis
print(f"Total vocabulary size: {tokenizer.vocab_size}")

# Check language-specific tokens
sample_tokens = ['machine', '机器', 'обучение', 'التعلم']
for tok in sample_tokens:
    if tok in tokenizer.vocab:
        print(f"'{tok}' is in vocabulary")
    else:
        subtokens = tokenizer.tokenize(tok)
        print(f"'{tok}' splits to: {subtokens}")
```

**XLM-RoBERTa Improvements:**
- 250K vocabulary (larger than mBERT's 110K)
- Trained on more languages (100+)
- Better coverage for low-resource languages
- Uses SentencePiece (language-agnostic)

**Challenges:**
1. **Curse of multilinguality:** Single model capacity shared across languages
2. **Vocabulary imbalance:** High-resource languages dominate
3. **Script mixing:** Code-switching text is challenging

**Interview Tips:**
- mBERT vocab: 110K, XLM-R vocab: 250K
- Shared vocabulary enables zero-shot cross-lingual transfer
- Low-resource languages get fragmented tokenization
- For best results on single language: use language-specific model

---

## Question 28
**What techniques help preserve named entities during tokenization (e.g., preventing "New York" from splitting)?**

**Answer:**

**Definition:**
Entity-preserving tokenization ensures multi-word entities stay together or are properly tracked during subword tokenization. Techniques include pre-tokenization with NER, adding entities to vocabulary, and maintaining token-to-entity alignment for downstream processing.

**The Problem:**
```
Standard tokenization:
"New York Stock Exchange" → ["New", "York", "Stock", "Exchange"]
                           or ["New", "York", "Stock", "Ex", "##change"]

Issue: Entity boundaries lost; "New York" split from "Stock Exchange"
```

**Solutions:**

| Technique | Description | Trade-off |
|-----------|-------------|-----------|
| Add to vocabulary | Add entity as single token | Increases vocab size |
| Pre-tokenization | Run NER first, mark entities | Two-pass processing |
| Entity markers | Wrap entities with special tokens | Extra tokens in sequence |
| Alignment tracking | Map subwords back to entities | Post-processing overhead |

**Approach 1: Add Entities to Vocabulary**
```python
# Add multi-word entities as single tokens
tokenizer.add_tokens(["New York", "Machine Learning", "United States"])
```

**Approach 2: Entity Markers**
```
Original: "I visited New York last week"
Marked:   "I visited [ENT]New York[/ENT] last week"
```

**Approach 3: Subword-to-Entity Alignment**
```
Text: "New York"
Subwords: ["New", "York"]
Alignment: [(0, 'B-LOC'), (1, 'I-LOC')]

During inference: Aggregate subword predictions
```

**Python Code Example:**
```python
# Pipeline: Text with entities -> Preserve entities -> Tokenize

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Approach 1: Add entities to vocabulary
entities = ["new york", "machine learning", "united states", "world war ii"]
tokenizer.add_tokens(entities)
print(f"Vocab size after: {len(tokenizer)}")

text = "I studied machine learning in new york"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['i', 'studied', 'machine learning', 'in', 'new york']
# Entities preserved as single tokens!

# Approach 2: Subword alignment for NER
def align_labels_with_subwords(text, entities, tokenizer):
    """
    Align entity labels with subword tokens
    entities: list of (start_char, end_char, label)
    """
    encoding = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']
    
    # Initialize labels
    labels = ['O'] * len(tokens)
    
    for start, end, label in entities:
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start is None:
                continue
            # Check if token overlaps with entity
            if tok_start >= start and tok_end <= end:
                if tok_start == start:
                    labels[idx] = f'B-{label}'
                else:
                    labels[idx] = f'I-{label}'
    
    return list(zip(tokens, labels))

text = "Steve Jobs founded Apple in California"
entities = [(0, 10, 'PER'), (19, 24, 'ORG'), (28, 38, 'LOC')]

aligned = align_labels_with_subwords(text, entities, tokenizer)
for token, label in aligned:
    if label != 'O':
        print(f"{token}: {label}")

# Approach 3: Entity marker tokens
def add_entity_markers(text, entities):
    """Add markers around entities"""
    # Sort by position (reverse to preserve indices)
    sorted_ents = sorted(entities, key=lambda x: x[0], reverse=True)
    
    for start, end, label in sorted_ents:
        text = text[:end] + f" [/{label}]" + text[end:]
        text = text[:start] + f"[{label}] " + text[start:]
    
    return text

marked = add_entity_markers("Apple was founded in California", 
                            [(0, 5, 'ORG'), (21, 31, 'LOC')])
print(marked)
# "[ORG] Apple [/ORG] was founded in [LOC] California [/LOC]"
```

**Best Practice for NER:**
1. Don't modify vocabulary (pretrained weights won't exist for new tokens)
2. Use offset mapping for alignment
3. Take first subword's prediction for entity label
4. Or use special handling (max pooling, voting) for multi-subword entities

**Interview Tips:**
- Adding tokens requires retraining/finetuning embeddings
- Offset mapping is standard way to handle alignment
- First-subword labeling is common convention for NER
- Entity markers help models learn entity boundaries

---

## Question 29
**How do you implement real-time NER for streaming text applications?**

**Answer:**

**Definition:**
Real-time NER processes text streams with low latency requirements (milliseconds). Implementation requires model optimization (quantization, distillation), efficient batching, caching, and handling partial/incomplete sentences from continuous streams.

**Core Challenges:**

| Challenge | Description |
|-----------|-------------|
| Latency | Must respond in milliseconds |
| Throughput | Handle high volume of requests |
| Context | Entities may span message boundaries |
| Partial text | Incomplete sentences in streams |
| Resource usage | Limited CPU/memory budget |

**Optimization Strategies:**

| Strategy | Latency Reduction | Trade-off |
|----------|-------------------|-----------|
| Model distillation | 3-5x faster | Slight accuracy drop |
| Quantization (INT8) | 2-4x faster | Minimal accuracy drop |
| ONNX Runtime | 1.5-2x faster | Export complexity |
| Smaller model | 5-10x faster | Accuracy drop |
| Batching | Better throughput | Higher latency |
| Caching | Near-instant for repeats | Memory usage |

**Architecture for Streaming:**
```
Input Stream → Buffer → Sentence Detector → NER Model → Output Stream
                ↓
         Context Window (for cross-boundary entities)
```

**Python Code Example:**
```python
# Pipeline: Text stream -> Buffer -> Batch NER -> Emit entities

import time
from collections import deque
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import torch

# Step 1: Load optimized model
class RealtimeNER:
    def __init__(self, model_name="dslim/bert-base-NER", device="cpu"):
        # Use smaller/distilled model for speed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.eval()
        
        # Optimization: Use ONNX or TorchScript
        # self.model = torch.jit.script(self.model)
        
        self.device = device
        self.model.to(device)
        
        # Buffer for streaming
        self.buffer = deque(maxlen=5)  # Keep last 5 messages for context
        
        # Cache for repeated queries
        self.cache = {}
    
    @torch.no_grad()
    def predict(self, text):
        # Check cache first
        if text in self.cache:
            return self.cache[text]
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Decode
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        labels = [self.model.config.id2label[p.item()] for p in predictions[0]]
        
        # Extract entities
        entities = self._extract_entities(tokens, labels)
        
        # Cache result
        self.cache[text] = entities
        
        return entities
    
    def _extract_entities(self, tokens, labels):
        entities = []
        current_entity = []
        current_label = None
        
        for token, label in zip(tokens, labels):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            if label.startswith('B-'):
                if current_entity:
                    entities.append((' '.join(current_entity), current_label))
                current_entity = [token.replace('##', '')]
                current_label = label[2:]
            elif label.startswith('I-') and current_label:
                current_entity.append(token.replace('##', ''))
            else:
                if current_entity:
                    entities.append((' '.join(current_entity), current_label))
                current_entity = []
                current_label = None
        
        if current_entity:
            entities.append((' '.join(current_entity), current_label))
        
        return entities
    
    def process_stream(self, text_generator):
        """Process streaming text"""
        for text in text_generator:
            start = time.time()
            entities = self.predict(text)
            latency = (time.time() - start) * 1000
            yield {
                'text': text,
                'entities': entities,
                'latency_ms': latency
            }

# Step 2: Simulate streaming
def simulate_stream():
    messages = [
        "Breaking: Elon Musk announces new Tesla factory",
        "Apple Inc reports record Q4 earnings",
        "President Biden visits Berlin tomorrow"
    ]
    for msg in messages:
        yield msg
        time.sleep(0.1)  # Simulate stream delay

# Step 3: Run real-time NER
ner = RealtimeNER()

for result in ner.process_stream(simulate_stream()):
    print(f"Text: {result['text']}")
    print(f"Entities: {result['entities']}")
    print(f"Latency: {result['latency_ms']:.1f}ms\n")
```

**Production Optimizations:**
1. Use DistilBERT (40% smaller, 60% faster)
2. ONNX Runtime for inference
3. Batch multiple messages when possible
4. GPU with TensorRT for high throughput
5. Async processing with message queues

**Interview Tips:**
- DistilBERT-NER is good for real-time (60% faster than BERT)
- Target latency: <50ms for real-time applications
- Consider spaCy for simpler/faster NER if BERT accuracy not needed
- Caching helps with repeated/similar text patterns

---

## Question 30
**What is cross-lingual transfer learning for NER? How does it work?**

**Answer:**

**Definition:**
Cross-lingual transfer trains NER on one language (usually high-resource like English) and applies it to another language (low-resource) without target language training data. It works through multilingual representations that map semantically similar words across languages to nearby embeddings.

**Core Mechanism:**
```
Training: English NER data
         "Apple Inc. is in California"
         [Apple Inc.]_ORG    [California]_LOC

Transfer: German (zero-shot)
         "Apple Inc. ist in Kalifornien"
         [Apple Inc.]_ORG    [Kalifornien]_LOC

Why it works: Multilingual embeddings place "California" and "Kalifornien" 
             in similar representation space.
```

**Transfer Approaches:**

| Approach | Description | Requirements |
|----------|-------------|--------------|
| Zero-shot | Train on source, apply directly to target | Multilingual model |
| Translate-train | Translate source training data | Translation system |
| Translate-test | Translate target to source at inference | Translation system |
| Few-shot | Add small target language data | Limited target labels |

**How Multilingual Representations Enable Transfer:**

$$\text{embed}(\text{"king"}_{\text{EN}}) \approx \text{embed}(\text{"roi"}_{\text{FR}}) \approx \text{embed}(\text{"König"}_{\text{DE}})$$

Semantically similar words across languages have similar embeddings.

**Python Code Example:**
```python
# Pipeline: Train on English -> Apply to other languages (zero-shot)

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
import torch

# Step 1: Load multilingual model
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    num_labels=9  # BIO tags for 4 entity types + O
)

# Step 2: Fine-tune on English NER (CoNLL-2003)
# Assume english_dataset is prepared
# trainer = Trainer(model=model, train_dataset=english_dataset, ...)
# trainer.train()

# Step 3: Zero-shot inference on other languages
def predict_ner(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.pop('offset_mapping')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    id2label = model.config.id2label
    results = []
    for token, pred, offset in zip(tokens, predictions, offset_mapping[0]):
        if offset[0] == offset[1]:  # Skip special tokens
            continue
        results.append((token, id2label[pred.item()]))
    
    return results

# Test on multiple languages (zero-shot transfer)
test_sentences = {
    'English': "Barack Obama visited Paris last week",
    'German': "Angela Merkel besuchte Berlin gestern",
    'French': "Emmanuel Macron est à Londres",
    'Spanish': "El presidente visitó Madrid ayer",
    'Chinese': "习近平访问了北京"
}

print("Zero-shot Cross-lingual NER:")
print("-" * 50)
for lang, text in test_sentences.items():
    result = predict_ner(text, model, tokenizer)
    entities = [(t, l) for t, l in result if l != 'O']
    print(f"{lang}: {text}")
    print(f"  Entities: {entities}\n")

# Step 4: Translate-train approach (alternative)
def translate_train_data(source_data, source_lang, target_lang, translator):
    """
    Translate training data to target language
    Note: Entity alignment is challenging
    """
    translated_data = []
    for text, entities in source_data:
        # Translate text
        translated_text = translator.translate(text, source_lang, target_lang)
        # Align entities (complex - may use word alignment)
        # translated_entities = align_entities(...)
        # translated_data.append((translated_text, translated_entities))
    return translated_data
```

**Performance (Typical F1 Scores):**

| Target Language | Zero-shot mBERT | Zero-shot XLM-R |
|-----------------|-----------------|-----------------|
| German | 65-70% | 70-75% |
| Spanish | 70-75% | 75-80% |
| Dutch | 70-75% | 75-80% |
| Chinese | 50-55% | 55-65% |

**Best Practices:**
1. Use XLM-RoBERTa over mBERT for better cross-lingual transfer
2. Few-shot (10-50 examples) significantly boosts performance
3. Related languages transfer better (English → German > English → Chinese)
4. Translate-train often outperforms zero-shot if good translation available

**Interview Tips:**
- XLM-RoBERTa is current state-of-the-art for cross-lingual NER
- Zero-shot works surprisingly well for related languages
- Transfer degrades for distant language pairs (different scripts)
- Shared vocabulary tokens (proper nouns, numbers) transfer best

---

## Question 31
**How do you handle NER for entities that change over time (new celebrities, companies, products)?**

**Answer:**

**Definition:**
Temporal entity drift occurs when new entities emerge and existing ones become obsolete. Solutions include periodic retraining, gazetteer updates, continual learning, and leveraging external knowledge bases that are regularly updated.

**The Problem:**
```
Model trained in 2020:
- Knows: "Tesla", "iPhone 11", "Trump"
- Doesn't know: "ChatGPT", "iPhone 15", "Threads app"

Real-world: ~1000s of new notable entities emerge monthly
```

**Strategies:**

| Strategy | Description | Effort |
|----------|-------------|--------|
| Periodic retraining | Retrain on updated data periodically | High |
| Gazetteer updates | Add new entities to lookup lists | Low |
| Continual learning | Update model incrementally | Medium |
| Knowledge base linking | Link to updated KB (Wikidata) | Medium |
| LLM zero-shot | Use LLMs with recent knowledge | Low |

**Approach 1: Gazetteer-Enhanced NER**
```
Static model + Dynamic gazetteers

Gazetteer (updated weekly):
- NEW_PRODUCTS: ["iPhone 15", "ChatGPT", "Vision Pro"]
- NEW_COMPANIES: ["OpenAI", "Anthropic", "Threads"]
- NEW_PERSONS: [trending celebrities]

Combine: Model predictions + Gazetteer matches
```

**Approach 2: Continual Learning**
- Collect new entity examples over time
- Fine-tune model incrementally
- Use elastic weight consolidation to prevent forgetting

**Python Code Example:**
```python
# Pipeline: Updatable NER with dynamic gazetteers

import re
from datetime import datetime

class TemporalNER:
    def __init__(self, base_model):
        self.model = base_model  # Pre-trained NER model
        self.gazetteers = {
            'PRODUCT': set(),
            'ORG': set(),
            'PER': set()
        }
        self.last_updated = None
    
    def update_gazetteers(self, new_entities):
        """
        Update gazetteers with new entities
        Called periodically (daily/weekly)
        """
        for entity_type, entities in new_entities.items():
            self.gazetteers[entity_type].update(entities)
        self.last_updated = datetime.now()
    
    def predict(self, text):
        # Step 1: Get model predictions
        model_entities = self.model.predict(text)
        
        # Step 2: Add gazetteer matches
        gazetteer_entities = self._match_gazetteers(text)
        
        # Step 3: Merge (gazetteer overrides for new entities)
        merged = self._merge_entities(model_entities, gazetteer_entities)
        
        return merged
    
    def _match_gazetteers(self, text):
        """Find entities from gazetteers in text"""
        entities = []
        text_lower = text.lower()
        
        for entity_type, entity_set in self.gazetteers.items():
            for entity in entity_set:
                # Case-insensitive search
                pattern = re.escape(entity.lower())
                for match in re.finditer(pattern, text_lower):
                    entities.append({
                        'text': text[match.start():match.end()],
                        'start': match.start(),
                        'end': match.end(),
                        'type': entity_type,
                        'source': 'gazetteer'
                    })
        
        return entities
    
    def _merge_entities(self, model_ents, gaz_ents):
        """Merge model and gazetteer predictions"""
        # Prefer gazetteer for overlapping spans
        all_entities = gaz_ents.copy()
        
        for ment in model_ents:
            overlap = False
            for gent in gaz_ents:
                if self._spans_overlap(ment, gent):
                    overlap = True
                    break
            if not overlap:
                all_entities.append(ment)
        
        return all_entities
    
    def _spans_overlap(self, e1, e2):
        return not (e1['end'] <= e2['start'] or e2['end'] <= e1['start'])

# Usage
ner = TemporalNER(base_model=None)  # Replace with actual model

# Weekly update with new entities
ner.update_gazetteers({
    'PRODUCT': ['iPhone 15', 'ChatGPT', 'Claude', 'Gemini'],
    'ORG': ['OpenAI', 'Anthropic', 'Mistral AI'],
    'PER': ['Sam Altman', 'Dario Amodei']
})

# Now recognizes new entities even if model was trained in 2020
```

**Production Best Practices:**
1. Maintain curated entity lists by domain
2. Subscribe to news/trend APIs for automatic entity discovery
3. Use Wikidata SPARQL queries to get new entities
4. Quarterly model retraining with accumulated data
5. A/B test updated models before deployment

**Interview Tips:**
- Gazetteers are simple but effective for entity drift
- Continual learning is research area - catastrophic forgetting is a challenge
- LLMs (GPT-4) have more recent knowledge but may hallucinate
- Hybrid: use model for common entities, gazetteers for trending/new ones

---

## Question 32
**What are gazetteer features in NER and when should you use them?**

**Answer:**

**Definition:**
Gazetteers are curated lists of known entities (names, locations, organizations) used as lookup features for NER. They provide high-precision recognition for listed entities and are especially valuable for domain-specific or rare entities that models struggle with.

**Core Concept:**
```
Gazetteers = Dictionary lookup for entities

PERSON gazetteer: {"Barack Obama", "Elon Musk", "Taylor Swift", ...}
LOCATION gazetteer: {"New York", "Paris", "Tokyo", ...}
COMPANY gazetteer: {"Apple Inc", "Google", "Microsoft", ...}

Text: "Elon Musk visited New York"
Lookup: "Elon Musk" ∈ PERSON, "New York" ∈ LOCATION
Result: High-confidence entity detection
```

**Gazetteer Features for ML:**

| Feature | Description |
|---------|-------------|
| in_person_gaz | Is word/phrase in person list? |
| in_org_gaz | Is word/phrase in organization list? |
| in_loc_gaz | Is word/phrase in location list? |
| gaz_prefix | Does phrase start with gazetteer entry? |
| gaz_suffix | Does phrase end with gazetteer entry? |

**When to Use Gazetteers:**

| Scenario | Gazetteer Value |
|----------|-----------------|
| Domain-specific entities | Very High (medical terms, legal terms) |
| Rare/unusual names | High (foreign names, brand names) |
| High-precision required | High (no false positives acceptable) |
| Well-curated lists exist | High |
| General newswire NER | Medium (model usually sufficient) |
| Noisy/informal text | Medium-Low (variations not in list) |

**Python Code Example:**
```python
# Pipeline: Text -> Gazetteer lookup -> Feature extraction -> NER

class GazetteerFeatures:
    def __init__(self):
        self.gazetteers = {
            'PER': self._load_gazetteer('persons.txt'),
            'ORG': self._load_gazetteer('organizations.txt'),
            'LOC': self._load_gazetteer('locations.txt'),
            'DRUG': self._load_gazetteer('drug_names.txt')
        }
        # Build prefix tree for efficient matching
        self.max_ngram = 5
    
    def _load_gazetteer(self, filepath):
        """Load gazetteer from file"""
        # In practice: load from file
        sample = {
            'persons.txt': {'elon musk', 'barack obama', 'taylor swift'},
            'organizations.txt': {'apple inc', 'google', 'microsoft', 'openai'},
            'locations.txt': {'new york', 'san francisco', 'paris', 'london'},
            'drug_names.txt': {'aspirin', 'ibuprofen', 'acetaminophen'}
        }
        return sample.get(filepath, set())
    
    def extract_features(self, tokens):
        """
        Extract gazetteer features for each token
        """
        features = []
        n = len(tokens)
        
        for i in range(n):
            token_features = {
                'token': tokens[i],
                'in_PER_gaz': False,
                'in_ORG_gaz': False,
                'in_LOC_gaz': False,
                'in_DRUG_gaz': False,
                'gaz_B': None,  # Beginning of gazetteer match
                'gaz_I': None   # Inside gazetteer match
            }
            
            # Check n-grams starting at this position
            for length in range(1, min(self.max_ngram + 1, n - i + 1)):
                ngram = ' '.join(tokens[i:i+length]).lower()
                
                for gaz_type, gaz_set in self.gazetteers.items():
                    if ngram in gaz_set:
                        key = f'in_{gaz_type}_gaz'
                        token_features[key] = True
                        if length == 1:
                            token_features['gaz_B'] = gaz_type
                        else:
                            # Mark all tokens in match
                            token_features['gaz_B'] = gaz_type
            
            features.append(token_features)
        
        return features

# Usage
gaz = GazetteerFeatures()

text = "Elon Musk announced that OpenAI released a new model"
tokens = text.lower().split()

features = gaz.extract_features(tokens)

for f in features:
    if any(f[k] for k in ['in_PER_gaz', 'in_ORG_gaz', 'in_LOC_gaz']):
        print(f"{f['token']}: {f}")

# Combining with neural NER
def combine_gazetteer_with_model(text, model_preds, gaz_features):
    """
    Combine model predictions with gazetteer matches
    """
    final_preds = []
    for pred, gaz in zip(model_preds, gaz_features):
        # If gazetteer match with high confidence
        if gaz['gaz_B'] and pred == 'O':
            final_preds.append(f'B-{gaz["gaz_B"]}')
        else:
            final_preds.append(pred)
    return final_preds
```

**Best Practices:**
1. Keep gazetteers clean and curated
2. Handle case-insensitivity and normalization
3. Use prefix trees (trie) for efficient multi-word matching
4. Combine gazetteer with model (not replace)
5. Update gazetteers regularly for temporal entities

**Interview Tips:**
- Gazetteers boost precision significantly for known entities
- They don't generalize - unseen entities won't be matched
- Common sources: Wikipedia, Wikidata, domain databases
- Neural + gazetteer hybrid is common in production NER

---

## Question 33
**How do you balance precision vs recall in NER for different business use cases?**

**Answer:**

**Definition:**
Precision measures accuracy of predictions (how many detected entities are correct), while recall measures coverage (how many actual entities were found). The optimal balance depends on business cost of false positives vs false negatives.

**Core Trade-off:**

$$\text{Precision} = \frac{TP}{TP + FP}$$ (Of what I found, how much is correct?)

$$\text{Recall} = \frac{TP}{TP + FN}$$ (Of what exists, how much did I find?)

**Business Case Decision Matrix:**

| Use Case | Priority | Why |
|----------|----------|-----|
| Medical diagnosis | High Recall | Missing a disease mention is dangerous |
| Legal contract review | High Recall | Missing a clause could be costly |
| Customer-facing chatbot | High Precision | Wrong answers hurt trust |
| Spam/fraud detection | High Precision | False positives block legitimate users |
| Search/retrieval | High Recall | Users want comprehensive results |
| Data anonymization (PII) | High Recall | Missing PII is a privacy violation |
| Knowledge graph population | Balanced | Both quality and coverage matter |

**How to Adjust Balance:**

| Method | Effect |
|--------|--------|
| Confidence threshold | ↑ threshold = ↑ precision, ↓ recall |
| Training data balance | More examples = better recall for that class |
| Post-processing rules | Filter predictions = ↑ precision |
| Model ensemble | Multiple models = ↑ recall |
| Loss function weighting | Weight FN higher = ↑ recall |

**Python Code Example:**
```python
# Pipeline: Adjust NER threshold based on business requirements

import numpy as np
from sklearn.metrics import precision_recall_curve

class BusinessTunedNER:
    def __init__(self, model, business_case='balanced'):
        self.model = model
        self.business_case = business_case
        self.thresholds = self._set_thresholds()
    
    def _set_thresholds(self):
        """Set confidence thresholds based on business case"""
        if self.business_case == 'high_precision':
            # Higher threshold = fewer but more accurate predictions
            return {'PER': 0.9, 'ORG': 0.85, 'LOC': 0.85}
        elif self.business_case == 'high_recall':
            # Lower threshold = more predictions, some may be wrong
            return {'PER': 0.3, 'ORG': 0.3, 'LOC': 0.3}
        else:  # balanced
            return {'PER': 0.5, 'ORG': 0.5, 'LOC': 0.5}
    
    def predict(self, text):
        # Get model predictions with confidence scores
        raw_predictions = self.model.predict_with_confidence(text)
        
        # Filter based on thresholds
        filtered = []
        for entity in raw_predictions:
            threshold = self.thresholds.get(entity['type'], 0.5)
            if entity['confidence'] >= threshold:
                filtered.append(entity)
        
        return filtered

# Finding optimal threshold using precision-recall curve
def find_optimal_threshold(y_true, y_scores, target='f1'):
    """
    Find threshold that optimizes target metric
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    if target == 'f1':
        # Optimize F1
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
    elif target == 'precision_at_recall':
        # Maximize precision while maintaining recall >= 0.9
        valid = recalls >= 0.9
        if valid.any():
            best_idx = np.argmax(precisions[valid])
        else:
            best_idx = 0
    elif target == 'recall_at_precision':
        # Maximize recall while maintaining precision >= 0.9
        valid = precisions >= 0.9
        if valid.any():
            best_idx = np.argmax(recalls[valid])
        else:
            best_idx = 0
    
    return thresholds[best_idx]

# Business case examples
def configure_for_business(use_case):
    """
    Configure NER based on business requirements
    """
    configs = {
        'pii_detection': {
            'threshold': 0.3,  # Low threshold - catch all PII
            'post_filter': False,
            'priority': 'recall',
            'acceptable_precision': 0.7
        },
        'customer_chatbot': {
            'threshold': 0.85,  # High threshold - avoid wrong answers
            'post_filter': True,
            'priority': 'precision',
            'acceptable_recall': 0.6
        },
        'medical_ner': {
            'threshold': 0.4,
            'post_filter': False,
            'priority': 'recall',
            'secondary_review': True  # Human review for low-confidence
        }
    }
    return configs.get(use_case, configs['customer_chatbot'])
```

**Scenario-Based Decision:**
```
Scenario: Building NER for medical records to identify drug mentions

Analysis:
- False Negative: Miss drug interaction → patient harm (HIGH COST)
- False Positive: Flag non-drug as drug → extra review (LOW COST)

Decision: Optimize for HIGH RECALL
- Use lower confidence threshold
- Accept more false positives
- Add human review for flagged entities
```

**Interview Tips:**
- Always ask about business context before recommending precision/recall balance
- PII detection: recall is critical (privacy violations are expensive)
- User-facing: precision matters (bad UX from wrong predictions)
- Use F2 score (weights recall) or F0.5 (weights precision) as alternatives to F1

---

## Question 34
**What is the Universal Dependencies project and how does it standardize POS tagging across languages?**

**Answer:**

**Definition:**
Universal Dependencies (UD) is a framework for consistent grammatical annotation across 100+ languages. It provides standardized POS tags, morphological features, and dependency relations, enabling cross-lingual NLP research and multilingual model development.

**Core Components:**

| Component | Description | Example |
|-----------|-------------|---------|
| UPOS | Universal POS tags (17 tags) | NOUN, VERB, ADJ, ADV |
| XPOS | Language-specific POS | NN, VB, JJ (Penn Treebank) |
| Features | Morphological features | Number=Plur, Tense=Past |
| DEPREL | Dependency relations | nsubj, dobj, amod |

**Universal POS Tags (17 Categories):**

| Tag | Description | Example |
|-----|-------------|---------|
| NOUN | Noun | cat, freedom |
| VERB | Verb | run, think |
| ADJ | Adjective | big, old |
| ADV | Adverb | quickly, very |
| PROPN | Proper noun | Paris, John |
| PRON | Pronoun | he, she, it |
| DET | Determiner | the, a, this |
| ADP | Adposition | in, on, of |
| NUM | Numeral | one, 2024 |
| CONJ | Conjunction | and, or |
| AUX | Auxiliary | is, was, will |
| PUNCT | Punctuation | . , ! |
| PART | Particle | not, 's |
| INTJ | Interjection | oh, wow |
| SYM | Symbol | $, % |
| X | Other | foreign words |
| SCONJ | Subordinating conj. | if, because |

**Why Universal Dependencies Matters:**

| Benefit | Description |
|---------|-------------|
| Cross-lingual consistency | Same annotation for same concept across languages |
| Model transfer | Train on one language, apply to another |
| Linguistic research | Compare syntax across languages |
| Tool interoperability | Same format for all UD tools |

**Comparison: Penn Treebank vs Universal Dependencies**
```
Penn Treebank (45 tags, English-specific):
NN, NNS, NNP, NNPS, VB, VBD, VBG, VBN, VBP, VBZ, ...

Universal Dependencies (17 tags, language-agnostic):
NOUN, VERB, ADJ, ADV, PROPN, ...

Mapping:
NN, NNS → NOUN
NNP, NNPS → PROPN
VB, VBD, VBG, VBN, VBP, VBZ → VERB
```

**Python Code Example:**
```python
# Pipeline: Text -> UD POS tagging -> Universal tags

import spacy

# Load model with UD tagset
nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

print("Token | UPOS | XPOS | Features")
print("-" * 50)
for token in doc:
    print(f"{token.text:10} | {token.pos_:6} | {token.tag_:5} | {token.morph}")

# Output:
# The        | DET    | DT    | Definite=Def|PronType=Art
# quick      | ADJ    | JJ    | Degree=Pos
# brown      | ADJ    | JJ    | Degree=Pos
# fox        | NOUN   | NN    | Number=Sing
# jumps      | VERB   | VBZ   | Number=Sing|Person=3|Tense=Pres|VerbForm=Fin
# over       | ADP    | IN    | 
# the        | DET    | DT    | Definite=Def|PronType=Art
# lazy       | ADJ    | JJ    | Degree=Pos
# dog        | NOUN   | NN    | Number=Sing
# .          | PUNCT  | .     | PunctType=Peri

# Cross-lingual UD tagging
from stanza import Pipeline

# German with UD
de_nlp = Pipeline('de', processors='tokenize,pos')
de_doc = de_nlp("Der schnelle braune Fuchs springt.")

for sent in de_doc.sentences:
    for word in sent.words:
        print(f"{word.text:15} | {word.upos:6} | {word.xpos}")

# German uses same UPOS tags as English!
# Der            | DET    | ART
# schnelle       | ADJ    | ADJA
# braune         | ADJ    | ADJA
# Fuchs          | NOUN   | NN
# springt        | VERB   | VVFIN
```

**UD Treebanks Available:**
- 200+ treebanks for 100+ languages
- Consistent format (CoNLL-U)
- Enables training multilingual models

**Interview Tips:**
- UD has 17 universal POS tags vs 45 in Penn Treebank
- spaCy, Stanza, UDPipe all support UD format
- UPOS is universal, XPOS is language-specific (for fine-grained analysis)
- UD enables cross-lingual transfer for syntactic tasks

---

## Question 35
**How do you handle POS tagging for code-mixed or transliterated text?**

**Answer:**

**Definition:**
Code-mixed text alternates between two or more languages in a single utterance. Transliterated text writes one language in another's script (e.g., Hindi in Latin). Both challenge POS taggers trained on monolingual, standard text.

**Examples:**

| Type | Example |
|------|---------|
| Code-mixing | "I went to the market aur maine sabzi li" (English + Hindi) |
| Transliteration | "Main bahut khush hoon" (Hindi in Latin script) |
| Mixed both | "That movie was bahut accha" (English + Romanized Hindi) |

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| Language identification | Which language is each word? |
| Script mismatch | Standard model expects native script |
| No standard spelling | "bahut" vs "bohot" vs "bhot" |
| Grammar mixing | Which language's rules apply? |
| Limited training data | Code-mixed datasets are scarce |

**Approaches:**

| Approach | Description |
|----------|-------------|
| Language ID first | Identify language per word, route to appropriate tagger |
| Joint model | Train single model on code-mixed data |
| Transliteration norm | Normalize to standard script before tagging |
| Multilingual model | Use mBERT/XLM-R that handles multiple languages |

**Python Code Example:**
```python
# Pipeline: Code-mixed text -> Language ID -> Unified POS tagging

import re

class CodeMixedPOSTagger:
    def __init__(self, en_tagger, hi_tagger, transliterator=None):
        self.en_tagger = en_tagger
        self.hi_tagger = hi_tagger
        self.transliterator = transliterator
        
        # Simple language detection patterns
        self.hi_romanized_words = {
            'aur', 'hai', 'hoon', 'hain', 'kya', 'nahi', 'bahut', 
            'accha', 'maine', 'mera', 'sabzi', 'khush'
        }
    
    def detect_language(self, word):
        """Simple per-word language detection"""
        word_lower = word.lower()
        
        # Check if Devanagari script
        if re.match(r'[\u0900-\u097F]+', word):
            return 'hi'
        
        # Check romanized Hindi vocabulary
        if word_lower in self.hi_romanized_words:
            return 'hi-roman'
        
        # Check if English word
        if re.match(r'^[a-zA-Z]+$', word):
            return 'en'
        
        return 'unknown'
    
    def tag(self, text):
        """Tag code-mixed text"""
        words = text.split()
        results = []
        
        for word in words:
            lang = self.detect_language(word)
            
            if lang == 'en':
                # Use English tagger
                pos = self._get_english_pos(word)
            elif lang == 'hi':
                # Use Hindi tagger
                pos = self._get_hindi_pos(word)
            elif lang == 'hi-roman':
                # Transliterate then tag
                if self.transliterator:
                    hi_word = self.transliterator.romanized_to_devanagari(word)
                    pos = self._get_hindi_pos(hi_word)
                else:
                    pos = 'UNK'
            else:
                pos = 'X'  # Unknown
            
            results.append({
                'word': word,
                'lang': lang,
                'pos': pos
            })
        
        return results
    
    def _get_english_pos(self, word):
        # Simplified - would use actual tagger
        return 'EN-POS'
    
    def _get_hindi_pos(self, word):
        # Simplified - would use actual tagger
        return 'HI-POS'

# Using multilingual model (better approach)
from transformers import AutoTokenizer, AutoModelForTokenClassification

def multilingual_pos_tagger(text):
    """
    Use mBERT for unified code-mixed POS tagging
    """
    model_name = "QCRI/bert-base-multilingual-cased-pos-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    
    # Decode predictions
    predictions = outputs.logits.argmax(dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    return list(zip(tokens, predictions[0].tolist()))

# Handling transliteration
def normalize_transliterated(text):
    """
    Normalize common transliteration variations
    """
    normalizations = {
        'bohot': 'bahut',
        'bhot': 'bahut',
        'acha': 'accha',
        'achha': 'accha',
        'hai': 'hai',
        'he': 'hai',
        'h': 'hai'
    }
    
    words = text.split()
    normalized = [normalizations.get(w.lower(), w) for w in words]
    return ' '.join(normalized)
```

**Best Practices:**
1. Use multilingual models (mBERT, XLM-R) as base
2. Fine-tune on code-mixed datasets if available
3. Normalize transliterations before processing
4. Build per-word language ID as first step
5. Use Universal Dependencies tags for consistency

**Interview Tips:**
- Code-mixing is common in multilingual societies (India, Singapore)
- GLUECoS benchmark has code-mixed datasets
- mBERT handles code-mixing reasonably well out-of-box
- Spelling normalization is crucial for transliterated text

---

## Question 36
**What role does dependency parsing play after POS tagging?**

**Answer:**

**Definition:**
Dependency parsing analyzes grammatical structure by identifying relationships between words, creating a tree where each word depends on another. POS tags inform parsing by constraining which dependency relations are valid (e.g., verbs govern subjects, nouns take adjective modifiers).

**Core Concept:**
```
Sentence: "The quick fox jumps over the fence"

Dependency Tree:
        jumps (ROOT/VERB)
       /     \
     fox      over
    / \         \
  The quick    fence
                 |
                the

Relations:
- "fox" is subject (nsubj) of "jumps"
- "quick" is adjective modifier (amod) of "fox"  
- "fence" is object of preposition (pobj)
```

**How POS Helps Parsing:**

| POS Tag | Likely Dependency Role |
|---------|------------------------|
| VERB | ROOT, head of clause |
| NOUN | Subject, object, complement |
| ADJ | Modifier of noun (amod) |
| ADV | Modifier of verb (advmod) |
| DET | Determiner of noun (det) |
| ADP | Introduces prepositional phrase |

**Common Dependency Relations:**

| Relation | Description | Example |
|----------|-------------|---------|
| nsubj | Nominal subject | "**John** runs" |
| dobj | Direct object | "I saw **him**" |
| amod | Adjectival modifier | "**big** dog" |
| det | Determiner | "**the** book" |
| prep | Prepositional modifier | "ran **to** store" |
| pobj | Object of preposition | "to **store**" |
| ROOT | Root of sentence | Usually main verb |

**Parsing Algorithms:**
1. **Transition-based:** Stack-based, builds tree left-to-right
2. **Graph-based:** Scores all possible edges, finds best tree

**Python Code Example:**
```python
# Pipeline: Text -> POS tagging -> Dependency Parsing -> Tree structure

import spacy

nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

# Display dependency structure
print("Token | POS | Head | Dependency")
print("-" * 50)
for token in doc:
    print(f"{token.text:10} | {token.pos_:5} | {token.head.text:10} | {token.dep_}")

# Output:
# The        | DET   | fox        | det
# quick      | ADJ   | fox        | amod
# brown      | ADJ   | fox        | amod
# fox        | NOUN  | jumps      | nsubj
# jumps      | VERB  | jumps      | ROOT
# over       | ADP   | jumps      | prep
# the        | DET   | dog        | det
# lazy       | ADJ   | dog        | amod
# dog        | NOUN  | over       | pobj
# .          | PUNCT | jumps      | punct

# Practical applications
def extract_subject_verb_object(doc):
    """Extract SVO triples from parsed sentence"""
    triples = []
    
    for token in doc:
        if token.dep_ == 'ROOT':
            verb = token
            subject = None
            obj = None
            
            for child in token.children:
                if child.dep_ in ['nsubj', 'nsubjpass']:
                    subject = child
                elif child.dep_ in ['dobj', 'pobj']:
                    obj = child
            
            if subject and obj:
                triples.append((subject.text, verb.text, obj.text))
    
    return triples

# Extract relationships
doc = nlp("John loves Mary")
svo = extract_subject_verb_object(doc)
print(svo)  # [('John', 'loves', 'Mary')]

# Use for information extraction
doc = nlp("Apple acquired Beats for $3 billion")
for token in doc:
    if token.dep_ == 'dobj':
        print(f"Acquisition target: {token.text}")
```

**Downstream Applications:**
- Information extraction (who did what to whom)
- Relation extraction (extract facts)
- Semantic role labeling
- Question generation
- Machine translation

**Interview Tips:**
- POS tagging is prerequisite for parsing - provides word categories
- Two main paradigms: transition-based (fast) vs graph-based (accurate)
- Universal Dependencies standardizes both POS and dependency relations
- spaCy, Stanza provide integrated POS+parsing pipelines

---

## Question 37
**How do transformer-based models perform joint NER and relation extraction?**

**Answer:**

**Definition:**
Joint NER and relation extraction identifies entities and their relationships simultaneously, rather than as separate pipeline stages. Transformers enable this by producing rich contextual representations that can be used for both entity detection and pairwise relation classification in a single forward pass.

**Pipeline vs Joint Approach:**

| Aspect | Pipeline | Joint |
|--------|----------|-------|
| Architecture | NER → RE (separate models) | Single model, shared encoder |
| Error propagation | NER errors cascade to RE | End-to-end training reduces this |
| Efficiency | Two forward passes | One forward pass |
| Interaction | RE can't help NER | Tasks inform each other |

**Joint Model Architecture:**
```
Input: "Steve Jobs founded Apple in California"

BERT Encoder
     ↓
[Shared Representations]
    ↙         ↘
NER Head      Relation Head
   ↓              ↓
Entities      Relations

Output:
- Entities: [Steve Jobs]_PER, [Apple]_ORG, [California]_LOC
- Relations: (Steve Jobs, founded, Apple), (Apple, located_in, California)
```

**Common Approaches:**

| Approach | Description |
|----------|-------------|
| Table filling | Predict entity tags + relation for all token pairs |
| Span-based | Enumerate spans, classify entities and relations |
| Set prediction | Predict set of (subject, relation, object) tuples |
| Sequence labeling | Special tags encode relations |

**Python Code Example:**
```python
# Pipeline: Text -> BERT -> Joint NER + RE

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class JointNERRE(nn.Module):
    def __init__(self, model_name, num_entity_labels, num_relation_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # NER head (token classification)
        self.ner_classifier = nn.Linear(hidden_size, num_entity_labels)
        
        # Relation head (entity pair classification)
        # Input: concat of two entity representations
        self.relation_classifier = nn.Linear(hidden_size * 2, num_relation_labels)
    
    def forward(self, input_ids, attention_mask, entity_spans=None):
        # Step 1: Encode with BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Step 2: NER predictions (per token)
        ner_logits = self.ner_classifier(sequence_output)
        
        # Step 3: Relation predictions (for entity pairs)
        relation_logits = None
        if entity_spans is not None:
            # Get entity representations (average pool over span)
            entity_reprs = []
            for start, end in entity_spans:
                entity_repr = sequence_output[:, start:end, :].mean(dim=1)
                entity_reprs.append(entity_repr)
            
            # Classify all entity pairs
            relation_logits = []
            for i, repr_i in enumerate(entity_reprs):
                for j, repr_j in enumerate(entity_reprs):
                    if i != j:
                        pair_repr = torch.cat([repr_i, repr_j], dim=-1)
                        rel_logit = self.relation_classifier(pair_repr)
                        relation_logits.append((i, j, rel_logit))
        
        return ner_logits, relation_logits

# Training loop (simplified)
def train_joint_model(model, dataloader, optimizer):
    model.train()
    
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ner_labels = batch['ner_labels']
        relation_labels = batch['relation_labels']
        entity_spans = batch['entity_spans']
        
        # Forward pass
        ner_logits, relation_logits = model(
            input_ids, attention_mask, entity_spans
        )
        
        # Compute joint loss
        ner_loss = nn.CrossEntropyLoss()(
            ner_logits.view(-1, ner_logits.size(-1)),
            ner_labels.view(-1)
        )
        
        rel_loss = 0
        if relation_logits:
            for i, j, logit in relation_logits:
                rel_loss += nn.CrossEntropyLoss()(
                    logit, 
                    relation_labels[i, j]
                )
        
        # Joint loss (weighted)
        loss = ner_loss + 0.5 * rel_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Inference
def extract_entities_and_relations(text, model, tokenizer):
    # Step 1: Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    
    # Step 2: Get NER predictions
    ner_logits, _ = model(inputs['input_ids'], inputs['attention_mask'])
    ner_preds = ner_logits.argmax(dim=-1)
    
    # Step 3: Extract entity spans from NER
    entity_spans = extract_spans_from_bio(ner_preds)
    
    # Step 4: Get relation predictions for entity pairs
    _, relation_logits = model(
        inputs['input_ids'], 
        inputs['attention_mask'],
        entity_spans
    )
    
    return entity_spans, relation_logits
```

**State-of-the-Art Models:**
- **SpERT:** Span-based entity and relation transformer
- **PL-Marker:** Packing levitated markers for entity/relation
- **TPLinker:** Token pair linking for joint extraction

**Interview Tips:**
- Joint models outperform pipelines by 2-5% F1
- Key insight: relation extraction can help NER (and vice versa)
- Computational cost: O(n²) for all entity pairs
- Negative relations (no relation) dominate - handle class imbalance

---

## Question 38
**What is byte-level BPE and why do models like GPT use it?**

**Answer:**

**Definition:**
Byte-level BPE operates on raw bytes (256 possible values) rather than Unicode characters, enabling the tokenizer to handle any text input including unknown characters, emojis, and any script without [UNK] tokens. GPT-2/3/4 use it for truly open-vocabulary language modeling.

**Standard BPE vs Byte-Level BPE:**

| Aspect | Standard BPE | Byte-level BPE |
|--------|--------------|----------------|
| Base units | Unicode characters | Raw bytes (0-255) |
| [UNK] tokens | Yes, for unknown chars | Never needed |
| Vocabulary | Language-specific | Universal |
| Emoji/symbols | May fail | Always works |
| Base vocab | 1000s of characters | 256 bytes |

**How Byte-Level BPE Works:**
```
Text: "Hello 👋"

Standard: ['Hello', '👋'] or ['Hello', '[UNK]'] if emoji unknown

Byte-level:
1. Convert to bytes: [72, 101, 108, 108, 111, 32, 240, 159, 145, 139]
                     H   e    l    l    o   space  👋 in UTF-8 bytes
2. Apply BPE on bytes: Merge frequent byte pairs
3. Result: ['Hello', ' ', 'Ġ', bytes for 👋...]
```

**Why GPT Uses Byte-Level:**
1. **No UNK tokens:** Can encode literally any input
2. **Multilingual:** Works for all languages without preprocessing
3. **Robust:** Handles typos, code, unusual characters
4. **Simple:** No language-specific tokenization rules

**Byte-Level Encoding Details:**
- UTF-8 encodes characters as 1-4 bytes
- ASCII characters = 1 byte (unchanged)
- Emojis = 4 bytes
- Non-ASCII characters = 2-4 bytes

**Python Code Example:**
```python
# Pipeline: Understand byte-level tokenization

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Regular text
text = "Hello World!"
tokens = tokenizer.tokenize(text)
print(f"Regular: {tokens}")
# ['Hello', 'ĠWorld', '!']
# Note: Ġ represents space (byte 32)

# Text with emojis
text_emoji = "I love 🤖 robots!"
tokens_emoji = tokenizer.tokenize(text_emoji)
print(f"With emoji: {tokens_emoji}")
# Works without [UNK]!

# Non-Latin script (works without special handling)
chinese = "你好世界"
tokens_cn = tokenizer.tokenize(chinese)
print(f"Chinese: {tokens_cn}")
# Each character becomes byte sequences

# Why it works: UTF-8 encoding
def show_byte_encoding(text):
    """Show how text is encoded as bytes"""
    utf8_bytes = text.encode('utf-8')
    print(f"Text: {text}")
    print(f"UTF-8 bytes: {list(utf8_bytes)}")
    print(f"Byte count: {len(utf8_bytes)}")

show_byte_encoding("A")      # [65] - 1 byte
show_byte_encoding("é")      # [195, 169] - 2 bytes
show_byte_encoding("你")     # [228, 189, 160] - 3 bytes
show_byte_encoding("🤖")     # [240, 159, 164, 150] - 4 bytes

# Vocabulary mapping in GPT-2
# GPT-2 maps bytes 0-255 to printable characters to avoid issues
# Byte 32 (space) -> 'Ġ'
# Byte 10 (newline) -> 'Ċ'
print(f"Space byte mapping: {tokenizer.byte_encoder[ord(' ')]}")
# 'Ġ'

# Compare vocab sizes
print(f"GPT-2 vocab size: {tokenizer.vocab_size}")  # 50,257
```

**Trade-offs:**

| Advantage | Disadvantage |
|-----------|--------------|
| Universal - any input works | Multi-byte chars = multiple tokens |
| No preprocessing needed | Chinese/emoji uses more tokens |
| Robust to noise/typos | Slight efficiency loss for non-ASCII |
| Simple implementation | Less interpretable tokens |

**Models Using Byte-Level BPE:**
- GPT-2, GPT-3, GPT-4
- RoBERTa
- BLOOM

**Interview Tips:**
- GPT-2 uses byte-level BPE with 50,257 tokens
- The "Ġ" character you see is the space byte (encoded for visibility)
- Byte-level is why GPT can handle code, emojis, any language
- Trade-off: Unicode characters may become multiple tokens (less efficient)

---

## Question 39
**How do you debug tokenization issues affecting downstream model performance?**

**Answer:**

**Definition:**
Tokenization bugs can silently degrade model performance. Debugging involves inspecting token sequences, checking alignment with labels, verifying special token handling, and comparing tokenization between training and inference.

**Common Tokenization Issues:**

| Issue | Symptom | Cause |
|-------|---------|-------|
| Train/inference mismatch | Accuracy drop in production | Different tokenizer versions |
| Label misalignment | Poor NER performance | Subwords not aligned with labels |
| Truncation | Information loss | Sequence too long |
| Special token issues | Weird predictions at boundaries | [CLS], [SEP] handling |
| Vocabulary mismatch | High loss, poor output | Wrong tokenizer for model |

**Debugging Checklist:**

1. **Inspect Tokenization Output**
2. **Check Token-Label Alignment**
3. **Verify Special Tokens**
4. **Compare Train vs Inference**
5. **Check for Truncation**
6. **Validate Vocabulary Match**

**Python Code Example:**
```python
# Pipeline: Debug tokenization step by step

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def debug_tokenization(text, tokenizer):
    """
    Comprehensive tokenization debugging
    """
    print("=" * 60)
    print(f"Input text: {text}")
    print("=" * 60)
    
    # 1. Basic tokenization
    tokens = tokenizer.tokenize(text)
    print(f"\n1. Tokens ({len(tokens)}): {tokens}")
    
    # 2. Full encoding with special tokens
    encoded = tokenizer(text, return_tensors='pt')
    all_tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    print(f"\n2. With special tokens: {all_tokens}")
    
    # 3. Token IDs
    print(f"\n3. Token IDs: {encoded['input_ids'][0].tolist()}")
    
    # 4. Check for UNK tokens
    unk_id = tokenizer.unk_token_id
    ids = encoded['input_ids'][0].tolist()
    if unk_id in ids:
        print(f"\n⚠️  WARNING: UNK token found at positions: {[i for i, x in enumerate(ids) if x == unk_id]}")
    else:
        print("\n✓ No UNK tokens")
    
    # 5. Offset mapping (for label alignment)
    encoded_with_offsets = tokenizer(text, return_offsets_mapping=True)
    print(f"\n5. Offset mapping: {encoded_with_offsets['offset_mapping']}")
    
    # 6. Check for truncation
    max_length = tokenizer.model_max_length
    if len(tokens) > max_length - 2:  # -2 for [CLS] and [SEP]
        print(f"\n⚠️  WARNING: Text will be truncated! Tokens: {len(tokens)}, Max: {max_length - 2}")
    
    # 7. Decode back
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print(f"\n7. Decoded: {decoded}")
    
    # 8. Check round-trip
    if decoded.replace('[CLS]', '').replace('[SEP]', '').strip() != text.lower():
        print("\n⚠️  Round-trip mismatch!")

# Debug example
debug_tokenization("The café serves açaí bowls", tokenizer)

# Check NER label alignment
def check_ner_alignment(text, entities, tokenizer):
    """
    Verify NER labels align with subword tokens
    entities: list of (start_char, end_char, label)
    """
    encoding = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']
    
    print("Token | Offset | Entity")
    print("-" * 40)
    
    for token, (start, end) in zip(tokens, offsets):
        entity_label = 'O'
        for ent_start, ent_end, label in entities:
            if start >= ent_start and end <= ent_end:
                if start == ent_start:
                    entity_label = f'B-{label}'
                else:
                    entity_label = f'I-{label}'
        print(f"{token:15} | ({start:2}, {end:2}) | {entity_label}")

text = "John Smith works at Apple"
entities = [(0, 10, 'PER'), (20, 25, 'ORG')]
check_ner_alignment(text, entities, tokenizer)

# Verify tokenizer-model compatibility
def verify_tokenizer_model_match(tokenizer, model):
    """
    Verify tokenizer vocabulary matches model embeddings
    """
    vocab_size = len(tokenizer)
    embed_size = model.get_input_embeddings().num_embeddings
    
    if vocab_size != embed_size:
        print(f"⚠️  MISMATCH: Tokenizer vocab={vocab_size}, Model embeddings={embed_size}")
        return False
    print(f"✓ Tokenizer and model vocabulary sizes match: {vocab_size}")
    return True
```

**Common Fixes:**

| Issue | Fix |
|-------|-----|
| Version mismatch | Pin tokenizer version in requirements |
| Label alignment | Use offset_mapping, ignore [CLS]/[SEP] |
| Truncation | Chunk long documents, use stride |
| Special tokens | Mask them during loss computation |
| OOV issues | Use subword tokenizer or add tokens |

**Interview Tips:**
- Always save tokenizer with model to ensure consistency
- Offset mapping is essential for token-level tasks like NER
- Check for UNK tokens - high UNK rate indicates vocabulary issues
- Test with edge cases: emojis, non-ASCII, very long text

---

## Question 40
**What are the memory and latency considerations for different tokenization approaches in production?**

**Answer:**

**Definition:**
Production tokenization must balance speed, memory footprint, and accuracy. Considerations include vocabulary loading time, tokenization throughput, batching efficiency, and memory usage for large vocabularies.

**Performance Comparison:**

| Approach | Speed | Memory | Vocab Load Time |
|----------|-------|--------|-----------------|
| Whitespace split | Fastest | Minimal | None |
| Regex-based | Fast | Low | Low |
| WordPiece/BPE | Moderate | Moderate | Moderate |
| SentencePiece | Moderate | Low | Low |
| Full tokenizer pipeline | Slower | Higher | Higher |

**Key Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| Latency | Time to tokenize single request | <5ms |
| Throughput | Requests per second | 1000+ |
| Memory | RAM for tokenizer | <100MB |
| Cold start | Time to load tokenizer | <1s |

**Memory Considerations:**

```
Vocabulary memory = vocab_size × avg_token_length × 2 (Unicode)

BERT (30K vocab):   ~1MB for vocabulary
GPT-2 (50K vocab):  ~2MB for vocabulary
Multilingual (250K): ~10MB for vocabulary

Plus:
- Trie/hash table for lookup: 2-3x vocabulary size
- Merge rules (BPE): ~vocab_size entries
- Special token mappings: Minimal
```

**Python Code Example:**
```python
# Pipeline: Benchmark tokenization performance

import time
import sys
from transformers import AutoTokenizer

def benchmark_tokenizer(tokenizer_name, texts, num_iterations=100):
    """
    Benchmark tokenization speed and memory
    """
    # Cold start time
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    load_time = time.time() - start
    
    # Memory estimate
    vocab_size = len(tokenizer)
    
    # Tokenization speed
    start = time.time()
    for _ in range(num_iterations):
        for text in texts:
            _ = tokenizer(text, return_tensors='pt')
    total_time = time.time() - start
    
    avg_latency = (total_time / (num_iterations * len(texts))) * 1000  # ms
    throughput = (num_iterations * len(texts)) / total_time
    
    return {
        'load_time_s': load_time,
        'vocab_size': vocab_size,
        'avg_latency_ms': avg_latency,
        'throughput_per_s': throughput
    }

# Sample texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require careful tokenization.",
    "Natural language processing is a fascinating field."
]

# Benchmark different tokenizers
results = {}
for name in ['bert-base-uncased', 'gpt2', 'xlm-roberta-base']:
    results[name] = benchmark_tokenizer(name, texts)
    print(f"\n{name}:")
    for k, v in results[name].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

# Production optimizations
class OptimizedTokenizer:
    def __init__(self, tokenizer_name, use_fast=True, max_length=128):
        # Use fast tokenizer (Rust-based)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            use_fast=use_fast
        )
        self.max_length = max_length
        
        # Pre-compute common patterns
        self._cache = {}
        self._cache_size = 10000
    
    def tokenize(self, text):
        # Check cache first
        if text in self._cache:
            return self._cache[text]
        
        # Tokenize
        result = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        
        # Cache if space available
        if len(self._cache) < self._cache_size:
            self._cache[text] = result
        
        return result
    
    def batch_tokenize(self, texts):
        """Batch tokenization is more efficient"""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

# Compare fast vs slow tokenizer
def compare_fast_slow(tokenizer_name, text):
    # Slow (Python-based)
    slow_tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    
    # Fast (Rust-based)
    fast_tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    # Benchmark
    iterations = 1000
    
    start = time.time()
    for _ in range(iterations):
        slow_tok.tokenize(text)
    slow_time = time.time() - start
    
    start = time.time()
    for _ in range(iterations):
        fast_tok.tokenize(text)
    fast_time = time.time() - start
    
    print(f"Slow tokenizer: {slow_time:.3f}s")
    print(f"Fast tokenizer: {fast_time:.3f}s")
    print(f"Speedup: {slow_time/fast_time:.1f}x")
```

**Production Best Practices:**

| Practice | Benefit |
|----------|---------|
| Use fast tokenizers | 3-10x speedup |
| Batch tokenization | Better throughput |
| Cache common inputs | Avoid redundant work |
| Limit max_length | Reduce memory/compute |
| Pre-load tokenizer | Avoid cold start |
| Use ONNX export | Faster inference |

**Interview Tips:**
- HuggingFace fast tokenizers are Rust-based, 3-10x faster
- Batching amortizes overhead, always batch when possible
- Vocabulary size impacts embedding memory linearly
- For real-time: cache tokenizer instance, use fast tokenizers

---

## Question 41
**How do you handle NER confidence scoring and when to abstain from predictions?**

**Answer:**

**Definition:**
NER confidence scoring quantifies prediction certainty, enabling systems to abstain from low-confidence predictions or flag them for human review. This is crucial for high-stakes applications where wrong predictions are costly.

**Why Confidence Matters:**

| Scenario | Low Confidence Action |
|----------|----------------------|
| Medical NER | Flag for doctor review |
| Legal NER | Abstain, request human verification |
| Chatbot | Ask user for clarification |
| Data pipeline | Log for later review |

**Confidence Scoring Methods:**

| Method | Description | Quality |
|--------|-------------|---------|
| Softmax probability | max(softmax(logits)) | Poor calibration |
| Entropy | Higher entropy = less confident | Better |
| CRF score | Sequence-level probability | Good for BIO |
| MC Dropout | Run multiple passes, measure variance | Best, but slow |
| Temperature scaling | Post-hoc calibration | Improves softmax |

**Mathematical Formulation:**

**Softmax Confidence:**
$$\text{confidence} = \max_i \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Entropy (Uncertainty):**
$$H = -\sum_i p_i \log(p_i)$$

Low entropy = high confidence, High entropy = uncertain

**Python Code Example:**
```python
# Pipeline: NER predictions -> Confidence scores -> Abstention decision

import torch
import torch.nn.functional as F
import numpy as np

class ConfidenceNER:
    def __init__(self, model, tokenizer, abstain_threshold=0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.abstain_threshold = abstain_threshold
    
    def predict_with_confidence(self, text):
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt')
        
        # Get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Predictions and confidence
        max_probs, predictions = probs.max(dim=-1)
        
        # Entropy-based uncertainty
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        results = []
        for i, (token, pred, conf, ent) in enumerate(zip(
            tokens, predictions[0], max_probs[0], entropy[0]
        )):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            label = self.model.config.id2label[pred.item()]
            
            # Abstention decision
            should_abstain = conf.item() < self.abstain_threshold
            
            results.append({
                'token': token,
                'label': label if not should_abstain else 'ABSTAIN',
                'confidence': conf.item(),
                'entropy': ent.item(),
                'abstain': should_abstain
            })
        
        return results
    
    def mc_dropout_confidence(self, text, n_samples=10):
        """
        Monte Carlo Dropout for better uncertainty estimation
        """
        self.model.train()  # Enable dropout
        
        inputs = self.tokenizer(text, return_tensors='pt')
        
        all_probs = []
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                all_probs.append(probs)
        
        # Stack and compute statistics
        all_probs = torch.stack(all_probs)
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        
        # Confidence = mean, Uncertainty = std
        max_probs, predictions = mean_probs.max(dim=-1)
        max_std = std_probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
        
        self.model.eval()
        
        return max_probs, max_std  # Low std = confident

# Calibration with temperature scaling
class TemperatureScaler:
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def calibrate(self, logits, labels):
        """
        Find optimal temperature using validation set
        """
        # Grid search for best temperature
        best_temp = 1.0
        best_ece = float('inf')
        
        for temp in np.arange(0.5, 3.0, 0.1):
            scaled_probs = F.softmax(logits / temp, dim=-1)
            ece = self._compute_ece(scaled_probs, labels)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
        
        self.temperature = best_temp
        return best_temp
    
    def _compute_ece(self, probs, labels, n_bins=10):
        """Expected Calibration Error"""
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels)
        
        ece = 0.0
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                avg_conf = confidences[in_bin].mean()
                avg_acc = accuracies[in_bin].float().mean()
                ece += prop_in_bin * abs(avg_conf - avg_acc)
        
        return ece
```

**Abstention Strategies:**

| Strategy | Description |
|----------|-------------|
| Threshold-based | Abstain if confidence < threshold |
| Entropy-based | Abstain if entropy > threshold |
| Multi-class | Abstain if top-2 predictions are close |
| Selective prediction | Guarantee precision at cost of coverage |

**Interview Tips:**
- Softmax probabilities are often overconfident - use calibration
- MC Dropout gives better uncertainty but is slower (multiple passes)
- Temperature scaling is simple post-hoc calibration
- For critical applications, always include abstention mechanism

---

## Question 42
**What is span-based NER and how does it differ from token classification approaches?**

**Answer:**

**Definition:**
Span-based NER classifies text spans (contiguous sequences of tokens) rather than individual tokens. Instead of BIO tagging each token, it enumerates candidate spans and classifies each as an entity type or "none". This naturally handles nested entities and avoids BIO decoding issues.

**Comparison:**

| Aspect | Token Classification (BIO) | Span-based |
|--------|---------------------------|------------|
| Unit | Individual tokens | Contiguous spans |
| Output | Sequence of BIO tags | Set of (span, type) |
| Nested entities | Cannot handle | Handles naturally |
| Decoding | BIO → entities | Direct span output |
| Complexity | O(n) predictions | O(n²) candidate spans |

**Token Classification Approach:**
```
Input:  "Steve Jobs founded Apple"
Tags:   B-PER I-PER  O       B-ORG
Decode: [Steve Jobs]_PER, [Apple]_ORG
```

**Span-based Approach:**
```
Input: "Steve Jobs founded Apple"

Candidate spans (up to length 3):
(0,0): "Steve"        → classify
(0,1): "Steve Jobs"   → PER ✓
(0,2): "Steve Jobs founded" → None
(1,1): "Jobs"         → classify
(1,2): "Jobs founded" → None
(2,2): "founded"      → None
(3,3): "Apple"        → ORG ✓

Output: {(0,1): PER, (3,3): ORG}
```

**Why Span-based Works Better for Nested NER:**
```
Text: "Bank of America"

Nested entities:
- [Bank of America]_ORG
- [America]_LOC (inside ORG)

Token BIO: Each token gets ONE tag → Can't represent both
Span-based: Multiple spans can be entities → Both detected
```

**Python Code Example:**
```python
# Pipeline: Text -> Enumerate spans -> Classify -> Extract entities

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SpanNER(nn.Module):
    def __init__(self, model_name, num_labels, max_span_len=8):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.max_span_len = max_span_len
        
        # Span representation: start + end + width embedding
        self.width_embedding = nn.Embedding(max_span_len, 50)
        
        # Classifier: 2*hidden + width_embed -> num_labels
        self.classifier = nn.Linear(hidden_size * 2 + 50, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Step 1: Encode tokens
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        batch_size, seq_len, hidden_size = hidden.shape
        
        # Step 2: Enumerate all spans
        span_reprs = []
        span_indices = []
        
        for start in range(seq_len):
            for end in range(start, min(start + self.max_span_len, seq_len)):
                # Span representation
                start_repr = hidden[:, start, :]
                end_repr = hidden[:, end, :]
                width = end - start
                width_repr = self.width_embedding(
                    torch.tensor([width]).to(hidden.device)
                ).expand(batch_size, -1)
                
                span_repr = torch.cat([start_repr, end_repr, width_repr], dim=-1)
                span_reprs.append(span_repr)
                span_indices.append((start, end))
        
        # Stack all span representations
        span_reprs = torch.stack(span_reprs, dim=1)  # (batch, num_spans, repr_dim)
        
        # Step 3: Classify each span
        logits = self.classifier(span_reprs)  # (batch, num_spans, num_labels)
        
        return logits, span_indices

def predict_entities(model, tokenizer, text, id2label, threshold=0.5):
    """Extract entities from span predictions"""
    inputs = tokenizer(text, return_tensors='pt', return_offsets_mapping=True)
    offset_mapping = inputs.pop('offset_mapping')
    
    with torch.no_grad():
        logits, span_indices = model(**inputs)
        probs = torch.softmax(logits, dim=-1)
    
    entities = []
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    for idx, (start, end) in enumerate(span_indices):
        # Get prediction (argmax)
        pred_label_id = probs[0, idx].argmax().item()
        pred_prob = probs[0, idx, pred_label_id].item()
        
        label = id2label[pred_label_id]
        
        # Filter: not "None" and above threshold
        if label != 'O' and pred_prob > threshold:
            span_text = tokenizer.decode(inputs['input_ids'][0, start:end+1])
            entities.append({
                'text': span_text,
                'start': start,
                'end': end,
                'label': label,
                'confidence': pred_prob
            })
    
    # Handle overlapping predictions (keep highest confidence)
    entities = remove_overlaps(entities)
    
    return entities

def remove_overlaps(entities):
    """Keep only highest confidence for overlapping spans"""
    entities = sorted(entities, key=lambda x: -x['confidence'])
    
    kept = []
    for ent in entities:
        overlaps = False
        for kept_ent in kept:
            if (ent['start'] <= kept_ent['end'] and 
                ent['end'] >= kept_ent['start']):
                overlaps = True
                break
        if not overlaps:
            kept.append(ent)
    
    return kept
```

**Span-based Advantages:**
1. Natural handling of nested/overlapping entities
2. No BIO decoding errors
3. Span length features can help
4. Easier to add span-level features

**Span-based Disadvantages:**
1. O(n²) complexity (vs O(n) for BIO)
2. Most spans are negative (class imbalance)
3. Needs max span length hyperparameter

**Interview Tips:**
- SpERT and SpanBERT are popular span-based NER models
- O(n²) is acceptable for short texts; chunk long documents
- Span-based is preferred for nested NER (ACE, GENIA datasets)
- Use span width embeddings - they help!

---

## Question 43
**How do you implement active learning for efficient NER annotation?**

**Answer:**

**Definition:**
Active learning iteratively selects the most informative unlabeled samples for human annotation, reducing labeling costs by 50-80% while maintaining model quality. For NER, it selects sentences where the model is most uncertain or would learn the most from.

**Core Concept:**
```
Traditional: Randomly annotate 1000 sentences
Active Learning: Strategically select 300 most valuable sentences

Both achieve similar F1, but active learning uses 70% less annotation!
```

**Active Learning Loop:**
1. Train initial model on small labeled set
2. Predict on large unlabeled pool
3. Score samples by uncertainty/informativeness
4. Select top-K most informative samples
5. Send to human for annotation
6. Add to training set, retrain
7. Repeat until budget exhausted or performance satisfied

**Uncertainty Sampling Strategies:**

| Strategy | Description | For NER |
|----------|-------------|---------|
| Least Confidence | Lowest max probability | Min(max(P(tag))) per token |
| Margin Sampling | Smallest gap between top-2 | Top1 - Top2 probability |
| Entropy | Highest entropy | Sum of token entropies |
| Sequence Entropy | Sentence-level uncertainty | CRF sequence probability |
| Query-by-Committee | Disagreement between models | Ensemble variance |

**Python Code Example:**
```python
# Pipeline: Train -> Score unlabeled -> Select -> Annotate -> Repeat

import numpy as np
import torch
import torch.nn.functional as F

class ActiveLearningNER:
    def __init__(self, model, tokenizer, strategy='entropy'):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
    
    def compute_uncertainty(self, texts):
        """
        Compute uncertainty scores for each text
        """
        scores = []
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt')
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)[0]  # (seq_len, num_labels)
                
                if self.strategy == 'least_confidence':
                    # Lowest maximum probability
                    max_probs = probs.max(dim=-1).values
                    score = 1 - max_probs.mean().item()
                
                elif self.strategy == 'margin':
                    # Smallest margin between top-2
                    sorted_probs = probs.sort(dim=-1, descending=True).values
                    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
                    score = 1 - margins.mean().item()
                
                elif self.strategy == 'entropy':
                    # Token-level entropy averaged
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    score = entropy.mean().item()
                
                elif self.strategy == 'total_token_entropy':
                    # Sum of entropies (longer = higher, more entities = higher)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    score = entropy.sum().item()
                
                scores.append(score)
        
        return np.array(scores)
    
    def select_samples(self, unlabeled_texts, n_samples):
        """
        Select most informative samples for annotation
        """
        scores = self.compute_uncertainty(unlabeled_texts)
        
        # Select top-n highest uncertainty
        top_indices = np.argsort(scores)[-n_samples:]
        
        selected = [unlabeled_texts[i] for i in top_indices]
        selected_scores = scores[top_indices]
        
        return selected, selected_scores, top_indices

def active_learning_loop(
    model, 
    tokenizer,
    labeled_data,
    unlabeled_data,
    annotation_budget,
    samples_per_round=50
):
    """
    Full active learning loop
    """
    history = []
    
    while annotation_budget > 0:
        # Step 1: Train on current labeled data
        train_model(model, labeled_data)
        
        # Step 2: Evaluate current performance
        f1 = evaluate_model(model, test_data)
        history.append({
            'labeled_size': len(labeled_data),
            'f1': f1
        })
        print(f"Labeled: {len(labeled_data)}, F1: {f1:.3f}")
        
        # Step 3: Select samples for annotation
        al = ActiveLearningNER(model, tokenizer, strategy='entropy')
        samples_to_annotate, _, indices = al.select_samples(
            unlabeled_data, 
            min(samples_per_round, annotation_budget)
        )
        
        # Step 4: Get annotations (simulated or human)
        # In practice: send to annotation platform
        annotations = get_human_annotations(samples_to_annotate)
        
        # Step 5: Add to labeled, remove from unlabeled
        labeled_data.extend(zip(samples_to_annotate, annotations))
        unlabeled_data = [u for i, u in enumerate(unlabeled_data) 
                         if i not in indices]
        
        annotation_budget -= len(samples_to_annotate)
    
    return history

# Diversity sampling (avoid redundant samples)
def diversity_sampling(embeddings, selected_indices, n_samples):
    """
    Select diverse samples using k-means or max-min distance
    """
    from sklearn.cluster import KMeans
    
    # Cluster unlabeled embeddings
    kmeans = KMeans(n_clusters=n_samples)
    kmeans.fit(embeddings)
    
    # Select sample closest to each cluster center
    selected = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(embeddings - center, axis=1)
        closest = np.argmin(distances)
        selected.append(closest)
    
    return selected
```

**Hybrid Strategy (Uncertainty + Diversity):**
1. Filter top-K uncertain samples
2. Among K, select diverse subset using clustering
3. Best of both worlds

**Interview Tips:**
- Active learning can reduce annotation needs by 50-80%
- Entropy is simple and effective baseline
- Combine uncertainty + diversity for best results
- Start with 50-100 random samples, then switch to active learning

---

## Question 44
**What preprocessing steps are essential before tokenization (normalization, cleaning)?**

**Answer:**

**Definition:**
Preprocessing prepares raw text for tokenization by standardizing formats, removing noise, and normalizing variations. Essential steps include Unicode normalization, case handling, whitespace normalization, and domain-specific cleaning.

**Preprocessing Pipeline:**
```
Raw Text → Unicode Normalize → Clean → Normalize → Tokenize
```

**Essential Steps:**

| Step | Purpose | Example |
|------|---------|---------|
| Unicode normalization | Standardize character encoding | é (composed) → é (decomposed) |
| Whitespace normalization | Consistent spacing | Multiple spaces → single space |
| Case normalization | Optional lowercasing | "Apple" → "apple" |
| Punctuation handling | Standardize quotes, dashes | "smart quotes" → "straight quotes" |
| Number normalization | Optional | "123" → [NUM] |
| URL/email handling | Replace or remove | http://... → [URL] |
| HTML/XML stripping | Remove tags | \<p>text\</p> → text |

**Unicode Normalization Forms:**

| Form | Description | Use Case |
|------|-------------|----------|
| NFC | Composed (single codepoint) | Most common |
| NFD | Decomposed (base + combining) | Text search |
| NFKC | Compatibility composed | Aggressive normalization |
| NFKD | Compatibility decomposed | Aggressive normalization |

**Python Code Example:**
```python
# Pipeline: Raw text -> Full preprocessing -> Clean text

import re
import unicodedata
import html

class TextPreprocessor:
    def __init__(self, 
                 lowercase=False,
                 normalize_unicode=True,
                 remove_urls=True,
                 remove_emails=True,
                 normalize_whitespace=True,
                 strip_html=True,
                 normalize_quotes=True):
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.strip_html = strip_html
        self.normalize_quotes = normalize_quotes
    
    def preprocess(self, text):
        # Step 1: Handle None/empty
        if not text:
            return ""
        
        # Step 2: Decode HTML entities
        text = html.unescape(text)  # &amp; → &
        
        # Step 3: Strip HTML tags
        if self.strip_html:
            text = re.sub(r'<[^>]+>', ' ', text)
        
        # Step 4: Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Step 5: Replace URLs
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
        # Step 6: Replace emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
        
        # Step 7: Normalize quotes and dashes
        if self.normalize_quotes:
            # Smart quotes → straight quotes
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
            # Em/en dash → hyphen
            text = text.replace('—', '-').replace('–', '-')
        
        # Step 8: Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Step 9: Lowercase (optional)
        if self.lowercase:
            text = text.lower()
        
        return text

# Usage
preprocessor = TextPreprocessor(lowercase=False)

samples = [
    'Check out https://example.com for more info!',
    'Contact us at info@company.com',
    '<p>Hello <b>World</b></p>',
    '"Smart   quotes"   and — dashes',
    'Café and naïve have accents',
]

for sample in samples:
    cleaned = preprocessor.preprocess(sample)
    print(f"Original: {sample}")
    print(f"Cleaned:  {cleaned}\n")

# Domain-specific preprocessing
def preprocess_medical_text(text):
    """Medical text preprocessing"""
    preprocessor = TextPreprocessor()
    text = preprocessor.preprocess(text)
    
    # Expand common abbreviations
    abbreviations = {
        'pt': 'patient',
        'dx': 'diagnosis',
        'hx': 'history',
        'rx': 'prescription',
        'yo': 'year old'
    }
    for abbr, full in abbreviations.items():
        text = re.sub(rf'\b{abbr}\b', full, text, flags=re.IGNORECASE)
    
    return text

def preprocess_social_media(text):
    """Social media preprocessing"""
    preprocessor = TextPreprocessor()
    text = preprocessor.preprocess(text)
    
    # Normalize elongation
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Keep hashtags but segment them
    def segment_hashtag(m):
        tag = m.group(1)
        return '#' + re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
    text = re.sub(r'#(\w+)', segment_hashtag, text)
    
    return text
```

**When NOT to Preprocess:**

| Scenario | Reason |
|----------|--------|
| Using BERT-like models | They handle raw text well |
| Case matters (NER) | "apple" (fruit) vs "Apple" (company) |
| Sentiment analysis | Punctuation carries sentiment (!!!) |
| Code analysis | Syntax matters |

**Interview Tips:**
- Modern transformers need less preprocessing than traditional methods
- Always do Unicode normalization (NFKC is safe default)
- Don't lowercase for NER - case is a strong feature
- Domain-specific preprocessing can significantly help (medical, legal, social)

---

## Question 45
**How do you handle special tokens ([CLS], [SEP], [PAD]) in transformer tokenizers?**

**Answer:**

**Definition:**
Special tokens are reserved tokens with specific meanings: [CLS] for classification pooling, [SEP] for sentence separation, [PAD] for batch padding, and [MASK] for masked language modeling. Proper handling is crucial for correct model behavior.

**Special Tokens Overview:**

| Token | BERT ID | Purpose | When Added |
|-------|---------|---------|------------|
| [CLS] | 101 | Classification; start of sequence | Automatically |
| [SEP] | 102 | Separator; end of sequence/between sentences | Automatically |
| [PAD] | 0 | Padding for batch uniformity | During batching |
| [MASK] | 103 | Masked token for pre-training | MLM training |
| [UNK] | 100 | Unknown token | Rare in subword tokenizers |

**Token Sequence Structure:**
```
Single sentence:  [CLS] token1 token2 ... tokenN [SEP] [PAD] [PAD]
Two sentences:    [CLS] sent1_tokens [SEP] sent2_tokens [SEP] [PAD]
```

**Attention Mask:**
```
Tokens:         [CLS] Hello World [SEP] [PAD] [PAD]
Attention mask:   1     1    1     1     0     0
                 ↑ Real tokens     ↑ Padding (ignored)
```

**Token Type IDs (Segment IDs):**
```
[CLS] Sentence A [SEP] Sentence B [SEP]
  0       0        0       1        1
  └── Segment 0 ──┘  └── Segment 1 ─┘
```

**Python Code Example:**
```python
# Pipeline: Understanding and handling special tokens

from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 1. Inspect special tokens
print("Special tokens:")
print(f"  [CLS]: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
print(f"  [SEP]: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
print(f"  [PAD]: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"  [MASK]: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
print(f"  [UNK]: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")

# 2. Single sentence encoding
text = "Hello World"
encoded = tokenizer(text, return_tensors='pt')

tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
print(f"\nSingle sentence: {tokens}")
# ['[CLS]', 'hello', 'world', '[SEP]']

# 3. Sentence pair encoding
sent1 = "What is AI?"
sent2 = "AI stands for Artificial Intelligence"
encoded_pair = tokenizer(sent1, sent2, return_tensors='pt')

tokens_pair = tokenizer.convert_ids_to_tokens(encoded_pair['input_ids'][0])
print(f"\nSentence pair: {tokens_pair}")
# ['[CLS]', 'what', 'is', 'ai', '?', '[SEP]', 'ai', 'stands', 'for', ...]

print(f"Token type IDs: {encoded_pair['token_type_ids'][0].tolist()}")
# [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# 4. Batch with padding
texts = ["Short text", "This is a longer piece of text", "Hi"]
batch = tokenizer(texts, padding=True, return_tensors='pt')

print(f"\nBatch shape: {batch['input_ids'].shape}")
for i, text in enumerate(texts):
    tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
    mask = batch['attention_mask'][i].tolist()
    print(f"  {text}: {tokens}")
    print(f"  Attention mask: {mask}")

# 5. Handling special tokens in NER
def get_ner_labels_with_special_tokens(tokens, bio_labels, tokenizer):
    """
    Add special token labels for NER
    """
    # Special tokens get -100 (ignored in loss)
    full_labels = [-100]  # [CLS]
    full_labels.extend([label2id[l] for l in bio_labels])
    full_labels.append(-100)  # [SEP]
    
    return full_labels

# 6. Extracting [CLS] for classification
def get_cls_embedding(model, input_ids, attention_mask):
    """
    Get [CLS] token embedding for classification
    """
    outputs = model(input_ids, attention_mask=attention_mask)
    last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
    
    # [CLS] is always at position 0
    cls_embedding = last_hidden[:, 0, :]  # (batch, hidden)
    
    return cls_embedding

# 7. Masking special tokens in attention for some tasks
def create_special_token_mask(input_ids, tokenizer):
    """
    Create mask that's 0 for special tokens, 1 for real tokens
    """
    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id
    }
    
    mask = torch.ones_like(input_ids)
    for special_id in special_ids:
        mask[input_ids == special_id] = 0
    
    return mask
```

**Key Points for Different Tasks:**

| Task | [CLS] Usage | [SEP] Usage | Padding |
|------|-------------|-------------|---------|
| Classification | Pool for prediction | End marker | Mask with attention_mask |
| NER | Ignore (label=-100) | Ignore | Ignore |
| QA | Not used directly | Separates question/context | Mask |
| MLM | Not masked | Not masked | Mask |

**Interview Tips:**
- [CLS] embedding is used for sentence-level classification
- Always use attention_mask to handle [PAD] tokens
- NER: label special tokens with -100 (PyTorch ignores in loss)
- Token type IDs distinguish sentence A from sentence B
- [SEP] appears between sentences AND at the end

---

## Question 46
**What is the difference between greedy and optimal tokenization algorithms?**

**Answer:**

**Definition:**
Greedy tokenization picks the locally best token at each step (longest match), while optimal tokenization finds the globally best segmentation considering all possible splits. BPE uses greedy merging; Unigram LM uses optimal (Viterbi) decoding.

**Comparison:**

| Aspect | Greedy | Optimal |
|--------|--------|---------|
| Approach | Longest match at each step | Best overall segmentation |
| Speed | Faster O(n) | Slower O(n²) |
| Consistency | Deterministic | Deterministic |
| Quality | May miss better splits | Globally optimal |
| Examples | BPE, WordPiece | Unigram LM |

**Greedy Example (BPE):**
```
Word: "lowest"
Vocabulary: ["low", "er", "est", "low", "e", "s", "t"]

Greedy (left-to-right, longest match):
Step 1: "low" matches → ["low"]
Step 2: "est" matches → ["low", "est"]
Result: ["low", "est"]

Alternative (not chosen by greedy):
["lo", "we", "st"] - greedy doesn't explore this
```

**Optimal Example (Unigram):**
```
Word: "lowest"
Vocabulary with probabilities:
  P("low") = 0.1, P("est") = 0.05
  P("lo") = 0.08, P("west") = 0.02
  P("l") = 0.3, P("o") = 0.2, ...

Optimal: Find segmentation that maximizes:
  ∏ P(token) for all tokens

Uses Viterbi algorithm to find global optimum:
  P("low" + "est") = 0.1 × 0.05 = 0.005
  P("lo" + "west") = 0.08 × 0.02 = 0.0016
  
Best: ["low", "est"] (happens to match greedy here)
```

**Viterbi Algorithm for Optimal Tokenization:**
```
1. Create lattice of all possible tokens at each position
2. Compute cumulative probability for each path
3. Backtrack to find highest probability segmentation
```

**Python Code Example:**
```python
# Pipeline: Compare greedy vs optimal tokenization

import math

class GreedyTokenizer:
    def __init__(self, vocab):
        self.vocab = set(vocab)
        self.max_word_len = max(len(w) for w in vocab)
    
    def tokenize(self, text):
        """Greedy left-to-right longest match"""
        tokens = []
        i = 0
        
        while i < len(text):
            # Find longest matching token
            match = None
            for length in range(min(self.max_word_len, len(text) - i), 0, -1):
                candidate = text[i:i+length]
                if candidate in self.vocab:
                    match = candidate
                    break
            
            if match:
                tokens.append(match)
                i += len(match)
            else:
                # Character fallback
                tokens.append(text[i])
                i += 1
        
        return tokens

class UnigramTokenizer:
    def __init__(self, vocab_with_probs):
        self.vocab = vocab_with_probs  # {token: log_prob}
        self.max_word_len = max(len(w) for w in vocab_with_probs.keys())
    
    def tokenize(self, text):
        """Optimal tokenization using Viterbi"""
        n = len(text)
        
        # best_score[i] = best log prob to tokenize text[0:i]
        best_score = [-float('inf')] * (n + 1)
        best_score[0] = 0
        
        # backpointer[i] = position where best tokenization to i starts last token
        backpointer = [0] * (n + 1)
        
        # Dynamic programming
        for i in range(1, n + 1):
            for length in range(1, min(self.max_word_len + 1, i + 1)):
                start = i - length
                token = text[start:i]
                
                if token in self.vocab:
                    score = best_score[start] + self.vocab[token]
                    if score > best_score[i]:
                        best_score[i] = score
                        backpointer[i] = start
        
        # Backtrack to get tokens
        tokens = []
        i = n
        while i > 0:
            start = backpointer[i]
            tokens.append(text[start:i])
            i = start
        
        tokens.reverse()
        return tokens

# Compare
vocab = ['low', 'lower', 'lowest', 'est', 'er', 'l', 'o', 'w', 'e', 's', 't']
greedy = GreedyTokenizer(vocab)

vocab_probs = {
    'low': math.log(0.1), 'lower': math.log(0.05), 'lowest': math.log(0.03),
    'est': math.log(0.08), 'er': math.log(0.1),
    'l': math.log(0.2), 'o': math.log(0.15), 'w': math.log(0.1),
    'e': math.log(0.2), 's': math.log(0.1), 't': math.log(0.1)
}
optimal = UnigramTokenizer(vocab_probs)

word = "lowest"
print(f"Greedy:  {greedy.tokenize(word)}")
print(f"Optimal: {optimal.tokenize(word)}")
```

**Real-World Tokenizers:**

| Tokenizer | Algorithm | Used By |
|-----------|-----------|---------|
| BPE | Greedy merge | GPT-2, RoBERTa |
| WordPiece | Greedy (likelihood-based) | BERT |
| Unigram | Optimal (Viterbi) | T5, XLNet, ALBERT |
| SentencePiece | Either BPE or Unigram | Many models |

**Interview Tips:**
- BPE/WordPiece are greedy - fast but may not be optimal
- Unigram uses Viterbi - optimal but slightly slower
- In practice, difference in downstream performance is minimal
- Unigram allows multiple valid segmentations (subword regularization)

---

## Question 47
**How do you implement privacy-preserving NER for sensitive documents (PII detection)?**

**Answer:**

**Definition:**
Privacy-preserving NER identifies and protects Personally Identifiable Information (PII) like names, SSNs, credit cards, and addresses. It requires high recall (missing PII is costly), pattern matching for structured data, and secure handling of detected entities.

**PII Entity Types:**

| Type | Examples | Detection Method |
|------|----------|------------------|
| Names | John Smith | NER model |
| SSN | 123-45-6789 | Regex pattern |
| Credit Card | 4111-1111-1111-1111 | Regex + Luhn check |
| Phone | (555) 123-4567 | Regex pattern |
| Email | john@example.com | Regex pattern |
| Address | 123 Main St, City, ST 12345 | NER + patterns |
| DOB | 01/15/1990 | Date pattern + context |
| Medical IDs | MRN: 12345 | Domain patterns |

**PII Detection Pipeline:**
```
Document → Pattern Matching → NER Model → Merge Results → Redact/Mask
              ↓                   ↓
         Structured PII      Unstructured PII
         (SSN, CC, Phone)    (Names, Addresses)
```

**Python Code Example:**
```python
# Pipeline: Document -> Detect PII -> Mask/Redact

import re
from transformers import pipeline

class PIIDetector:
    def __init__(self, ner_model="dslim/bert-base-NER"):
        self.ner_pipeline = pipeline("ner", model=ner_model, grouped_entities=True)
        
        # Regex patterns for structured PII
        self.patterns = {
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
            'CREDIT_CARD': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'PHONE': r'\b(?:\+1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b',
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'IP_ADDRESS': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'DATE_OF_BIRTH': r'\b(?:DOB|Date of Birth)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        }
    
    def detect_pii(self, text):
        """Detect all PII in text"""
        pii_entities = []
        
        # 1. Pattern-based detection (structured PII)
        for pii_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pii_entities.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': pii_type,
                    'source': 'regex',
                    'confidence': 1.0  # Pattern matches are certain
                })
        
        # 2. NER-based detection (names, organizations, locations)
        ner_results = self.ner_pipeline(text)
        for entity in ner_results:
            if entity['entity_group'] in ['PER', 'PERSON']:
                pii_entities.append({
                    'text': entity['word'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'type': 'PERSON_NAME',
                    'source': 'ner',
                    'confidence': entity['score']
                })
        
        # 3. Additional validation
        pii_entities = self._validate_entities(pii_entities)
        
        return pii_entities
    
    def _validate_entities(self, entities):
        """Validate detected PII (e.g., Luhn check for credit cards)"""
        validated = []
        for entity in entities:
            if entity['type'] == 'CREDIT_CARD':
                # Luhn algorithm validation
                if self._luhn_check(entity['text']):
                    validated.append(entity)
            else:
                validated.append(entity)
        return validated
    
    def _luhn_check(self, card_number):
        """Validate credit card number using Luhn algorithm"""
        digits = [int(d) for d in card_number if d.isdigit()]
        if len(digits) < 13:
            return False
        
        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return checksum % 10 == 0
    
    def redact(self, text, entities, method='mask'):
        """Redact PII from text"""
        # Sort by position (reverse to maintain indices)
        entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        redacted = text
        for entity in entities:
            if method == 'mask':
                replacement = f"[{entity['type']}]"
            elif method == 'hash':
                import hashlib
                h = hashlib.sha256(entity['text'].encode()).hexdigest()[:8]
                replacement = f"[{entity['type']}:{h}]"
            elif method == 'generalize':
                replacement = self._generalize(entity)
            
            redacted = redacted[:entity['start']] + replacement + redacted[entity['end']:]
        
        return redacted
    
    def _generalize(self, entity):
        """Generalize PII instead of removing"""
        if entity['type'] == 'PERSON_NAME':
            return "[PERSON]"
        elif entity['type'] == 'EMAIL':
            # Keep domain, mask username
            parts = entity['text'].split('@')
            return f"***@{parts[1]}"
        elif entity['type'] == 'PHONE':
            return "***-***-" + entity['text'][-4:]
        return f"[{entity['type']}]"

# Usage
detector = PIIDetector()

text = """
Dear John Smith,
Your SSN 123-45-6789 has been verified.
Please contact us at john.smith@email.com or (555) 123-4567.
Credit card ending in 4111-1111-1111-1111 was charged.
"""

pii = detector.detect_pii(text)
print("Detected PII:")
for p in pii:
    print(f"  {p['type']}: {p['text']} (confidence: {p['confidence']:.2f})")

redacted = detector.redact(text, pii, method='mask')
print(f"\nRedacted:\n{redacted}")
```

**Privacy Considerations:**

| Consideration | Implementation |
|---------------|----------------|
| High recall | Lower confidence thresholds |
| Audit logging | Log detections (not values) |
| Secure storage | Don't store raw PII |
| Re-identification | Consider context attacks |
| Differential privacy | Add noise to aggregates |

**Interview Tips:**
- PII detection requires HIGH RECALL - missing PII is a privacy violation
- Combine regex (structured) + NER (unstructured) for best coverage
- Credit card validation with Luhn algorithm reduces false positives
- HIPAA, GDPR, CCPA have specific PII requirements
- Tools: Microsoft Presidio, AWS Comprehend PII, spaCy with custom NER

---

## Question 48
**What are the best practices for fine-tuning pre-trained models on custom NER datasets?**

**Answer:**

**Definition:**
Fine-tuning adapts pre-trained models (BERT, RoBERTa) to custom NER tasks. Best practices include proper data formatting, learning rate selection, handling class imbalance, using appropriate loss functions, and preventing overfitting.

**Fine-tuning Pipeline:**
```
Custom NER Data → Format to BIO → Align with subwords → 
Configure training → Fine-tune → Evaluate → Deploy
```

**Best Practices Checklist:**

| Category | Practice |
|----------|----------|
| Data | Convert to BIO/BIOES format |
| Data | Align labels with subwords |
| Data | Use train/val/test split |
| Learning Rate | Use 2e-5 to 5e-5 (BERT default) |
| Epochs | 3-5 epochs (rarely more) |
| Batch Size | 16-32 (increase with gradient accumulation) |
| Scheduler | Linear warmup + decay |
| Loss | CrossEntropy with class weights for imbalance |
| Regularization | Dropout (already in BERT) |
| Evaluation | Entity-level F1 (not token-level) |

**Python Code Example:**
```python
# Pipeline: Custom NER data -> Fine-tune BERT -> Evaluate

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

# Step 1: Prepare data in correct format
def prepare_ner_data(examples, tokenizer, label2id):
    """
    Tokenize and align labels with subwords
    """
    tokenized = tokenizer(
        examples['tokens'],
        is_split_into_words=True,
        truncation=True,
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of word gets the label
                label_ids.append(label2id[label[word_idx]])
            else:
                # Subsequent tokens of same word
                # Option 1: Same label (common)
                # Option 2: -100 (ignore)
                label_ids.append(label2id[label[word_idx]])
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized['labels'] = labels
    return tokenized

# Step 2: Define metrics
def compute_metrics(eval_preds, id2label):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)
    
    # Convert to label strings (ignore -100)
    true_labels = []
    pred_labels = []
    
    for pred, label in zip(predictions, labels):
        true_seq = []
        pred_seq = []
        for p, l in zip(pred, label):
            if l != -100:
                true_seq.append(id2label[l])
                pred_seq.append(id2label[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)
    
    return {
        'precision': precision_score(true_labels, pred_labels),
        'recall': recall_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels)
    }

# Step 3: Configure training
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Label mappings
labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# Training arguments (best practices)
training_args = TrainingArguments(
    output_dir="./ner-model",
    
    # Learning rate: 2e-5 to 5e-5 for BERT
    learning_rate=3e-5,
    
    # Epochs: 3-5 is usually enough
    num_train_epochs=4,
    
    # Batch size (adjust based on GPU memory)
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    
    # Gradient accumulation for larger effective batch
    gradient_accumulation_steps=2,
    
    # Learning rate schedule
    warmup_ratio=0.1,  # 10% warmup
    lr_scheduler_type="linear",
    
    # Regularization
    weight_decay=0.01,
    
    # Evaluation
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    
    # Logging
    logging_steps=100,
    
    # Mixed precision (faster training)
    fp16=True,
)

# Data collator handles padding
data_collator = DataCollatorForTokenClassification(tokenizer)

# Step 4: Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda p: compute_metrics(p, id2label)
)

trainer.train()

# Step 5: Evaluate on test set
results = trainer.evaluate(test_dataset)
print(f"Test F1: {results['eval_f1']:.4f}")
```

**Common Mistakes to Avoid:**

| Mistake | Fix |
|---------|-----|
| Wrong label alignment | Use word_ids() for subword alignment |
| Too high learning rate | Use 2e-5 to 5e-5 |
| Too many epochs | 3-5 epochs, use early stopping |
| Token-level evaluation | Use seqeval for entity-level F1 |
| Ignoring class imbalance | Weight loss or oversample rare entities |
| Not freezing layers | For small data, freeze lower layers |

**Interview Tips:**
- Always align labels with subwords using word_ids()
- seqeval library for proper entity-level evaluation
- Learning rate 3e-5 is safe default for BERT fine-tuning
- Add CRF layer for 1-2% F1 improvement
- Use fp16 for faster training on modern GPUs

---

## Question 49
**How do you handle NER for hierarchical or multi-level entity types?**

**Answer:**

**Definition:**
Hierarchical NER handles entities with type hierarchies (e.g., PERSON → ATHLETE → FOOTBALLER). Multi-level tagging assigns both coarse and fine-grained types. Approaches include hierarchical classification, multi-task learning, and type embeddings.

**Hierarchy Example:**
```
ENTITY
├── PERSON
│   ├── POLITICIAN
│   ├── ATHLETE
│   │   ├── FOOTBALLER
│   │   └── TENNIS_PLAYER
│   └── ARTIST
├── ORGANIZATION
│   ├── COMPANY
│   ├── GOVERNMENT
│   └── SPORTS_TEAM
└── LOCATION
    ├── CITY
    ├── COUNTRY
    └── FACILITY
```

**Why Hierarchical NER:**
- More informative predictions (not just "PERSON" but "FOOTBALLER")
- Consistency constraints (FOOTBALLER implies ATHLETE implies PERSON)
- Better generalization (share knowledge across related types)
- Business needs (fine-grained vs coarse-grained requirements)

**Approaches:**

| Approach | Description |
|----------|-------------|
| Flat multi-class | All leaf types as separate classes |
| Hierarchical loss | Penalize less for hierarchically related errors |
| Multi-task | Predict at each level separately |
| Type embeddings | Encode hierarchy in type vectors |
| Constraint decoding | Enforce valid parent-child relations |

**Python Code Example:**
```python
# Pipeline: Text -> Hierarchical NER -> Multi-level entity types

import torch
import torch.nn as nn
from transformers import BertModel

class HierarchicalNER(nn.Module):
    def __init__(self, model_name, type_hierarchy):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Store hierarchy
        self.hierarchy = type_hierarchy
        self.levels = self._get_levels()
        
        # Separate classifier for each level
        self.level_classifiers = nn.ModuleDict()
        for level, types in self.levels.items():
            self.level_classifiers[level] = nn.Linear(hidden_size, len(types))
    
    def _get_levels(self):
        """Extract types at each hierarchy level"""
        return {
            'level_0': ['O', 'PERSON', 'ORGANIZATION', 'LOCATION'],
            'level_1': ['O', 'POLITICIAN', 'ATHLETE', 'ARTIST', 
                       'COMPANY', 'GOVERNMENT', 'CITY', 'COUNTRY'],
            'level_2': ['O', 'FOOTBALLER', 'TENNIS_PLAYER', 
                       'SPORTS_TEAM', 'FACILITY']
        }
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Predict at each level
        level_logits = {}
        for level, classifier in self.level_classifiers.items():
            level_logits[level] = classifier(sequence_output)
        
        return level_logits

def hierarchical_loss(level_logits, level_labels, hierarchy, alpha=0.5):
    """
    Loss that considers hierarchy:
    - Full penalty for wrong prediction
    - Reduced penalty if prediction is ancestor/descendant of true type
    """
    total_loss = 0
    ce_loss = nn.CrossEntropyLoss()
    
    for level, logits in level_logits.items():
        labels = level_labels[level]
        
        # Standard CE loss at each level
        loss = ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss
    
    return total_loss

# Simpler approach: Multi-label with type embeddings
class TypeEmbeddingNER(nn.Module):
    def __init__(self, model_name, num_types, hierarchy_adj):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Type embeddings that capture hierarchy
        self.type_embeddings = nn.Embedding(num_types, 128)
        
        # Initialize type embeddings based on hierarchy
        self._init_type_embeddings(hierarchy_adj)
        
        # Bilinear scoring
        self.scorer = nn.Bilinear(hidden_size, 128, 1)
    
    def _init_type_embeddings(self, adj):
        """Initialize similar embeddings for related types"""
        # Graph-based initialization
        pass
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        token_reprs = outputs.last_hidden_state  # (batch, seq, hidden)
        
        # Score each token against each type
        batch_size, seq_len, _ = token_reprs.shape
        num_types = self.type_embeddings.num_embeddings
        
        all_type_embs = self.type_embeddings.weight  # (num_types, type_dim)
        
        # Compute scores
        scores = []
        for t in range(num_types):
            type_emb = all_type_embs[t].unsqueeze(0).unsqueeze(0)
            type_emb = type_emb.expand(batch_size, seq_len, -1)
            score = self.scorer(token_reprs, type_emb).squeeze(-1)
            scores.append(score)
        
        scores = torch.stack(scores, dim=-1)  # (batch, seq, num_types)
        
        return scores

# Inference with hierarchy constraints
def constrained_decode(predictions, hierarchy):
    """
    Ensure predictions are consistent with hierarchy
    e.g., if FOOTBALLER predicted, parent types should also be true
    """
    # Get parent types
    parent_map = {
        'FOOTBALLER': ['ATHLETE', 'PERSON'],
        'TENNIS_PLAYER': ['ATHLETE', 'PERSON'],
        'ATHLETE': ['PERSON'],
        'COMPANY': ['ORGANIZATION'],
        # ... etc
    }
    
    # Propagate to ancestors
    consistent = predictions.copy()
    for pred in predictions:
        if pred in parent_map:
            for parent in parent_map[pred]:
                if parent not in consistent:
                    consistent.add(parent)
    
    return consistent
```

**Evaluation for Hierarchical NER:**
- Exact match: Full type path must match
- Partial match: Credit for correct ancestors
- Level-wise F1: Evaluate each level separately

**Interview Tips:**
- Flat classification ignores hierarchy - wastes structure
- Multi-task learning at each level is common approach
- Type embeddings can encode semantic similarity
- Constraint decoding ensures consistency at inference
- Datasets: OntoNotes (18 types), FIGER (113 types), TypeNet (~1000 types)

---

## Question 50
**What metrics and techniques help evaluate tokenization quality?**

**Answer:**

**Definition:**
Tokenization quality affects downstream model performance. Evaluation metrics include vocabulary coverage, fertility (tokens per word), token boundary quality, and downstream task performance. Good tokenization balances vocabulary size, sequence length, and semantic preservation.

**Key Metrics:**

| Metric | Description | Good Value |
|--------|-------------|------------|
| Vocabulary coverage | % words not going to [UNK] | >99% |
| Fertility | Avg tokens per word | 1.2-1.5 |
| Sequence length ratio | Output length / input words | ~1.2-1.5 |
| Subword quality | Meaningful subword splits | Subjective |
| OOV rate | % unknown tokens | <1% |
| Compression ratio | Bits per character | Lower = better |

**Fertility Definition:**
$$\text{Fertility} = \frac{\text{Number of tokens}}{\text{Number of words}}$$

High fertility = more fragmentation = longer sequences

**Python Code Example:**
```python
# Pipeline: Evaluate tokenization quality

from transformers import AutoTokenizer
from collections import Counter
import numpy as np

class TokenizationEvaluator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def evaluate(self, texts):
        """Comprehensive tokenization evaluation"""
        results = {
            'fertility': [],
            'unk_rate': [],
            'seq_length_ratio': [],
            'vocab_coverage': [],
            'subword_counts': Counter()
        }
        
        unk_id = self.tokenizer.unk_token_id
        
        for text in texts:
            words = text.split()
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Fertility: tokens per word
            fertility = len(tokens) / max(len(words), 1)
            results['fertility'].append(fertility)
            
            # UNK rate
            if unk_id:
                unk_count = token_ids.count(unk_id)
                unk_rate = unk_count / max(len(token_ids), 1)
                results['unk_rate'].append(unk_rate)
            
            # Sequence length ratio
            ratio = len(tokens) / max(len(words), 1)
            results['seq_length_ratio'].append(ratio)
            
            # Count subword patterns
            for token in tokens:
                if token.startswith('##') or token.startswith('Ġ'):
                    results['subword_counts']['continuation'] += 1
                else:
                    results['subword_counts']['word_start'] += 1
        
        # Aggregate
        summary = {
            'avg_fertility': np.mean(results['fertility']),
            'std_fertility': np.std(results['fertility']),
            'avg_unk_rate': np.mean(results['unk_rate']) if results['unk_rate'] else 0,
            'avg_seq_ratio': np.mean(results['seq_length_ratio']),
            'subword_ratio': results['subword_counts']['continuation'] / 
                           max(sum(results['subword_counts'].values()), 1)
        }
        
        return summary
    
    def compare_tokenizers(self, texts, tokenizer_names):
        """Compare multiple tokenizers"""
        comparison = {}
        
        for name in tokenizer_names:
            tok = AutoTokenizer.from_pretrained(name)
            evaluator = TokenizationEvaluator(tok)
            comparison[name] = evaluator.evaluate(texts)
        
        return comparison

def evaluate_subword_quality(tokenizer, words):
    """
    Check if tokenization produces meaningful subwords
    """
    quality_checks = {
        'morphological': 0,  # Splits at morpheme boundaries
        'arbitrary': 0       # Random-looking splits
    }
    
    # Known good splits
    good_patterns = ['ing', 'tion', 'ness', 'able', 'ment', 'er', 'ed', 'ly']
    
    for word in words:
        tokens = tokenizer.tokenize(word)
        
        if len(tokens) > 1:
            # Check if splits are meaningful
            for token in tokens[1:]:  # Skip first token
                clean = token.replace('##', '').replace('Ġ', '')
                if clean in good_patterns:
                    quality_checks['morphological'] += 1
                else:
                    quality_checks['arbitrary'] += 1
    
    total = sum(quality_checks.values())
    if total > 0:
        morphological_ratio = quality_checks['morphological'] / total
        return {'morphological_ratio': morphological_ratio}
    return {'morphological_ratio': None}

# Usage
tokenizers = ['bert-base-uncased', 'gpt2', 'xlm-roberta-base']

sample_texts = [
    "The internationalization of artificial intelligence continues.",
    "Unprecedented breakthroughs in biotechnology research.",
    "Cryptocurrency markets experienced unprecedented volatility."
]

for tok_name in tokenizers:
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    evaluator = TokenizationEvaluator(tokenizer)
    
    results = evaluator.evaluate(sample_texts)
    print(f"\n{tok_name}:")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Avg fertility: {results['avg_fertility']:.2f}")
    print(f"  UNK rate: {results['avg_unk_rate']:.4f}")
    print(f"  Subword ratio: {results['subword_ratio']:.2f}")

# Check downstream impact
def evaluate_downstream_impact(tokenizer1, tokenizer2, model, test_data):
    """
    Compare tokenization impact on downstream task
    """
    # Tokenize with each tokenizer
    # Train/evaluate model
    # Compare performance
    pass
```

**What Makes Good Tokenization:**

| Good Sign | Bad Sign |
|-----------|----------|
| Low fertility (1.2-1.5) | High fertility (>2.0) |
| Near-zero UNK rate | High UNK rate |
| Morphological splits | Arbitrary splits |
| Consistent behavior | Inconsistent |
| Works cross-lingually | Language-specific issues |

**Interview Tips:**
- Fertility is key metric - lower is better (fewer tokens = faster inference)
- Zero UNK rate with subword tokenizers
- For multilingual, check fertility across languages
- Best evaluation: downstream task performance
- Common issue: tokenizer-model mismatch (different vocab)

---
