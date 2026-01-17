# Text Understanding (Classification, Sentiment, Topic Modeling) - Theory Questions

## Core Questions

## Question 1
**What is text classification and what are common approaches (Naive Bayes, SVM, Neural Networks)?**

**Answer:**

**Definition:**
Text classification assigns predefined categories to text documents based on their content. Common approaches range from probabilistic (Naive Bayes), to discriminative (SVM), to neural networks (CNN, RNN, Transformers), with modern transformers achieving state-of-the-art performance.

**Core Approaches:**

| Approach | Mechanism | Strengths |
|----------|-----------|-----------|
| Naive Bayes | P(class\|text) using Bayes theorem | Fast, works with small data |
| SVM | Find optimal hyperplane | Good with high-dim features |
| CNN | Convolutional filters on text | Captures local patterns |
| LSTM | Sequential processing | Handles variable length |
| Transformers | Self-attention | State-of-the-art |

**Naive Bayes:**
$$P(c|d) = \frac{P(d|c) \cdot P(c)}{P(d)} \propto P(c) \prod_{w \in d} P(w|c)$$

- Assumes word independence (naive)
- Fast training and inference
- Works well for spam detection, sentiment

**SVM:**
- Maps text to high-dimensional feature space (TF-IDF)
- Finds maximum margin hyperplane
- Kernel trick for non-linear boundaries
- Strong baseline for many tasks

**Neural Networks:**
- CNN: Sliding window captures n-gram patterns
- LSTM: Captures long-range dependencies
- BERT: Contextual embeddings, fine-tune for any task

**Python Code Example:**
```python
# Pipeline: Text -> Vectorize -> Classify

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from transformers import pipeline

# Sample data
texts = ["Great product, love it!", "Terrible quality, waste of money",
         "Amazing experience", "Worst purchase ever"]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative

# Approach 1: Naive Bayes
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
nb_pipeline.fit(texts, labels)

# Approach 2: SVM
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear'))
])
svm_pipeline.fit(texts, labels)

# Approach 3: Transformer (BERT)
classifier = pipeline("text-classification", 
                      model="distilbert-base-uncased-finetuned-sst-2-english")

# Predict
test_text = "This is wonderful!"
print(f"Naive Bayes: {nb_pipeline.predict([test_text])}")
print(f"SVM: {svm_pipeline.predict([test_text])}")
print(f"BERT: {classifier(test_text)}")
```

**When to Use Each:**

| Scenario | Best Choice |
|----------|-------------|
| Small data, fast needed | Naive Bayes |
| Medium data, interpretable | SVM |
| Large data, best accuracy | Transformers |
| Limited compute | Naive Bayes/SVM |

**Interview Tips:**
- Naive Bayes is surprisingly competitive baseline
- SVM with TF-IDF was SOTA before deep learning
- BERT fine-tuning is current standard
- Always compare against simple baselines

---

## Question 2
**Explain the difference between multi-class and multi-label text classification.**

**Answer:**

**Definition:**
Multi-class classification assigns exactly one class from multiple options (mutually exclusive). Multi-label classification assigns zero, one, or multiple classes simultaneously (non-exclusive). Multi-label requires different loss functions and evaluation metrics.

**Key Differences:**

| Aspect | Multi-class | Multi-label |
|--------|-------------|-------------|
| Classes per sample | Exactly 1 | 0, 1, or many |
| Mutual exclusivity | Yes | No |
| Output activation | Softmax | Sigmoid (per class) |
| Loss function | Cross-entropy | Binary cross-entropy per label |
| Output sum | = 1.0 | Any value |

**Examples:**

| Task | Type | Classes |
|------|------|---------|
| Sentiment (pos/neg/neu) | Multi-class | {positive, negative, neutral} |
| Topic tagging | Multi-label | {sports, politics, tech, ...} |
| Language detection | Multi-class | {english, french, spanish} |
| Movie genres | Multi-label | {action, comedy, romance, ...} |

**Mathematical Formulation:**

**Multi-class (Softmax):**
$$P(y=k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad \sum_k P(y=k) = 1$$

**Multi-label (Sigmoid per class):**
$$P(y_k=1) = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}, \quad \text{independent}$$

**Python Code Example:**
```python
# Pipeline: Multi-class vs Multi-label classification

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# Multi-class classifier
class MultiClassClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(cls_output)
        return logits  # Apply softmax at loss computation
    
    def predict(self, logits):
        probs = F.softmax(logits, dim=-1)
        return probs.argmax(dim=-1)  # Single class

# Multi-label classifier
class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        return logits  # Apply sigmoid per label
    
    def predict(self, logits, threshold=0.5):
        probs = torch.sigmoid(logits)
        return (probs > threshold).int()  # Multiple 1s possible

# Training differences
def train_multiclass(model, batch):
    logits = model(batch['input_ids'], batch['attention_mask'])
    # CrossEntropyLoss expects class indices
    loss = F.cross_entropy(logits, batch['labels'])
    return loss

def train_multilabel(model, batch):
    logits = model(batch['input_ids'], batch['attention_mask'])
    # BCEWithLogitsLoss expects multi-hot labels
    loss = F.binary_cross_entropy_with_logits(logits, batch['labels'].float())
    return loss

# Evaluation differences
from sklearn.metrics import accuracy_score, f1_score

# Multi-class: accuracy, macro-F1
y_true_mc = [0, 1, 2, 1, 0]
y_pred_mc = [0, 1, 1, 1, 0]
print(f"Multi-class Accuracy: {accuracy_score(y_true_mc, y_pred_mc)}")

# Multi-label: sample-based, micro-F1, macro-F1
y_true_ml = [[1,0,1], [0,1,0], [1,1,1]]  # Multi-hot
y_pred_ml = [[1,0,0], [0,1,0], [1,1,0]]
print(f"Multi-label Micro-F1: {f1_score(y_true_ml, y_pred_ml, average='micro')}")
print(f"Multi-label Macro-F1: {f1_score(y_true_ml, y_pred_ml, average='macro')}")
```

**Evaluation Metrics:**

| Multi-class | Multi-label |
|-------------|-------------|
| Accuracy | Subset accuracy (exact match) |
| Macro/Micro F1 | Micro/Macro F1 |
| Confusion matrix | Per-label precision/recall |

**Interview Tips:**
- Key difference: softmax (sum=1) vs sigmoid (independent)
- Multi-label uses BCEWithLogitsLoss (binary cross-entropy)
- Threshold selection is crucial for multi-label
- Hamming loss measures per-label accuracy for multi-label

---

## Question 3
**What is sentiment analysis? Explain the difference between lexicon-based and ML-based approaches.**

**Answer:**

**Definition:**
Sentiment analysis determines the emotional tone (positive, negative, neutral) of text. Lexicon-based methods use predefined sentiment word lists with scores, while ML-based methods learn patterns from labeled data. ML approaches are more accurate but require training data.

**Core Comparison:**

| Aspect | Lexicon-Based | ML-Based |
|--------|---------------|----------|
| Training data | Not needed | Required |
| Domain adaptation | Manual (new words) | Retrain on domain data |
| Interpretability | High (see which words) | Lower (model decision) |
| Context handling | Limited | Better (especially deep learning) |
| Negation handling | Rule-based | Learned |
| Setup effort | Low | Higher |

**Lexicon-Based Approach:**
```
Lexicon: {"good": +1, "great": +2, "bad": -1, "terrible": -2, ...}

Sentence: "The movie was great but the ending was terrible"
Score: great(+2) + terrible(-2) = 0 (neutral)

Issue: Doesn't understand "great but terrible" nuance
```

**ML-Based Approach:**
- Learns patterns from labeled examples
- Captures context, sarcasm, negation better
- Naive Bayes → SVM → LSTM → BERT (evolution)

**Python Code Example:**
```python
# Pipeline: Compare lexicon vs ML sentiment

# Approach 1: Lexicon-based (VADER)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

vader = SentimentIntensityAnalyzer()

def lexicon_sentiment(text):
    scores = vader.polarity_scores(text)
    # Returns: {'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.6}
    compound = scores['compound']
    
    if compound >= 0.05:
        return 'positive', compound
    elif compound <= -0.05:
        return 'negative', compound
    else:
        return 'neutral', compound

# Approach 2: ML-based (BERT)
from transformers import pipeline

ml_sentiment = pipeline("sentiment-analysis", 
                        model="distilbert-base-uncased-finetuned-sst-2-english")

def ml_sentiment_fn(text):
    result = ml_sentiment(text)[0]
    return result['label'].lower(), result['score']

# Compare on challenging examples
test_cases = [
    "This is a great movie!",                    # Clear positive
    "This is not a good movie.",                 # Negation
    "The movie was so bad it was good.",         # Sarcasm
    "I expected more from this director.",       # Implicit negative
    "The acting was great but plot terrible.",   # Mixed
]

print("Text | Lexicon | ML-Based")
print("-" * 60)
for text in test_cases:
    lex_result = lexicon_sentiment(text)
    ml_result = ml_sentiment_fn(text)
    print(f"{text[:35]:35} | {lex_result[0]:8} | {ml_result[0]}")

# Custom lexicon example
class SimpleLexiconSentiment:
    def __init__(self):
        self.lexicon = {
            'good': 1, 'great': 2, 'excellent': 2, 'amazing': 2,
            'bad': -1, 'terrible': -2, 'awful': -2, 'horrible': -2,
            'not': -1  # Negation modifier
        }
        self.negation_words = {'not', "n't", 'never', 'no'}
    
    def analyze(self, text):
        words = text.lower().split()
        score = 0
        negation = False
        
        for word in words:
            # Check for negation
            if word in self.negation_words:
                negation = True
                continue
            
            if word in self.lexicon:
                word_score = self.lexicon[word]
                if negation:
                    word_score *= -1
                    negation = False
                score += word_score
        
        return score, 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'
```

**When to Use Each:**

| Scenario | Recommendation |
|----------|----------------|
| Quick prototype | Lexicon (VADER) |
| No labeled data | Lexicon |
| Domain-specific accuracy | ML (fine-tune on domain) |
| Social media | VADER (handles emojis, slang) |
| Production quality | ML (BERT-based) |

**Interview Tips:**
- VADER is best off-the-shelf lexicon (handles social media)
- ML beats lexicon on most benchmarks
- Lexicon is interpretable - you know which words contributed
- Both struggle with sarcasm and implicit sentiment

---

## Question 4
**What is aspect-based sentiment analysis (ABSA) and how does it differ from document-level sentiment?**

**Answer:**

**Definition:**
Aspect-Based Sentiment Analysis (ABSA) identifies sentiments toward specific aspects/features within text, not the overall document sentiment. For "The food was great but service was terrible," ABSA extracts: food→positive, service→negative, while document-level might return neutral/mixed.

**Key Differences:**

| Aspect | Document-Level | ABSA |
|--------|---------------|------|
| Granularity | Whole document | Specific aspects |
| Output | Single sentiment | Multiple (aspect, sentiment) pairs |
| Use case | General opinion | Detailed feature analysis |
| Complexity | Simpler | More complex |

**ABSA Components:**
1. **Aspect Extraction:** Identify aspects mentioned (food, service, price)
2. **Sentiment Classification:** Determine sentiment toward each aspect
3. **Aspect Categorization:** Map aspects to predefined categories

**Example:**
```
Review: "The laptop has amazing battery life but the keyboard is uncomfortable."

Document-level: Mixed/Positive

ABSA output:
- Aspect: "battery life" → Positive
- Aspect: "keyboard" → Negative
```

**Python Code Example:**
```python
# Pipeline: Review -> Extract aspects -> Classify sentiment per aspect

from transformers import pipeline
import spacy

# Simple ABSA implementation
class SimpleABSA:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment = pipeline("sentiment-analysis")
        
        # Predefined aspect keywords (domain-specific)
        self.aspect_keywords = {
            'food': ['food', 'dish', 'meal', 'taste', 'flavor'],
            'service': ['service', 'staff', 'waiter', 'server'],
            'price': ['price', 'cost', 'value', 'expensive', 'cheap'],
            'ambiance': ['ambiance', 'atmosphere', 'decor', 'music']
        }
    
    def extract_aspects(self, text):
        """Extract aspect terms from text"""
        doc = self.nlp(text.lower())
        aspects = []
        
        # Look for nouns that match aspect keywords
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                for category, keywords in self.aspect_keywords.items():
                    if token.text in keywords:
                        # Get surrounding context
                        start = max(0, token.i - 5)
                        end = min(len(doc), token.i + 5)
                        context = doc[start:end].text
                        aspects.append({
                            'term': token.text,
                            'category': category,
                            'context': context
                        })
        
        return aspects
    
    def analyze(self, text):
        """Full ABSA pipeline"""
        aspects = self.extract_aspects(text)
        
        results = []
        for aspect in aspects:
            # Get sentiment for aspect context
            sentiment = self.sentiment(aspect['context'])[0]
            results.append({
                'aspect': aspect['term'],
                'category': aspect['category'],
                'sentiment': sentiment['label'],
                'confidence': sentiment['score']
            })
        
        return results

# Usage
absa = SimpleABSA()
review = "The food was delicious but the service was slow and the price too high."
results = absa.analyze(review)

for r in results:
    print(f"{r['aspect']} ({r['category']}): {r['sentiment']}")

# Output:
# food (food): POSITIVE
# service (service): NEGATIVE
# price (price): NEGATIVE

# Advanced: Using specialized ABSA models
# pyabsa library for aspect-based sentiment
# from pyabsa import AspectTermExtraction, AspectSentimentClassification
```

**ABSA Subtasks:**

| Task | Description |
|------|-------------|
| Aspect Term Extraction (ATE) | Find aspect mentions in text |
| Aspect Category Detection (ACD) | Classify aspects into categories |
| Aspect Sentiment Classification (ASC) | Sentiment per aspect |
| End-to-End ABSA | All of above jointly |

**Interview Tips:**
- ABSA is crucial for product/service analytics
- Joint models outperform pipeline approaches
- SemEval datasets are standard benchmarks
- Business use: understand what customers like/dislike specifically

---

## Question 5
**Explain LDA (Latent Dirichlet Allocation) and how it discovers topics in documents.**

**Answer:**

**Definition:**
LDA is a probabilistic generative model that discovers latent topics in a document collection. It assumes each document is a mixture of topics, and each topic is a distribution over words. LDA learns these distributions to uncover thematic structure.

**Core Intuition:**
```
Document = Mix of Topics
Topic = Distribution over Words

Example:
Document about "AI in healthcare"
- 40% AI topic (words: neural, learning, model)
- 35% Healthcare topic (words: patient, diagnosis, treatment)
- 25% General topic (words: system, data, research)
```

**Generative Story (How LDA thinks documents are created):**
1. For each topic k: draw word distribution φ_k ~ Dirichlet(β)
2. For each document d: draw topic distribution θ_d ~ Dirichlet(α)
3. For each word in document d:
   - Choose topic z ~ Multinomial(θ_d)
   - Choose word w ~ Multinomial(φ_z)

**Mathematical Formulation:**

$$P(w|d) = \sum_{k=1}^{K} P(w|z=k) \cdot P(z=k|d)$$

Where:
- $P(z=k|d) = \theta_{dk}$ (topic proportion in document)
- $P(w|z=k) = \phi_{kw}$ (word probability in topic)

**Dirichlet Prior:**
$$\theta_d \sim \text{Dir}(\alpha), \quad \phi_k \sim \text{Dir}(\beta)$$

- α controls document-topic sparsity (low α = fewer topics per doc)
- β controls topic-word sparsity (low β = fewer words per topic)

**Python Code Example:**
```python
# Pipeline: Documents -> Preprocess -> LDA -> Topics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Sample documents
documents = [
    "Machine learning algorithms improve with data",
    "Neural networks are used for deep learning",
    "The stock market experienced volatility today",
    "Investors are concerned about economic growth",
    "Natural language processing understands text",
    "Financial markets react to interest rates"
]

# Step 1: Create document-term matrix
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(documents)

# Step 2: Fit LDA
n_topics = 2
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=100
)
lda.fit(doc_term_matrix)

# Step 3: Display topics
feature_names = vectorizer.get_feature_names_out()

def display_topics(model, feature_names, n_words=5):
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-n_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Topic {topic_idx}: {', '.join(top_words)}")

display_topics(lda, feature_names)
# Topic 0: learning, neural, machine, networks, deep (AI topic)
# Topic 1: market, financial, stock, investors, economic (Finance topic)

# Step 4: Get topic distribution for new document
new_doc = ["Deep learning models for stock prediction"]
new_vec = vectorizer.transform(new_doc)
topic_dist = lda.transform(new_vec)
print(f"New doc topics: {topic_dist[0]}")
# [0.55, 0.45] - Mix of both topics

# Using Gensim (more features)
from gensim import corpora
from gensim.models import LdaModel

# Preprocessing
texts = [[word for word in doc.lower().split() if len(word) > 3] 
         for doc in documents]

# Create dictionary and corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train LDA
lda_gensim = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=2,
    passes=10
)

# Print topics
for idx, topic in lda_gensim.print_topics():
    print(f"Topic {idx}: {topic}")
```

**LDA Hyperparameters:**

| Parameter | Effect |
|-----------|--------|
| K (num_topics) | More topics = finer granularity |
| α (alpha) | Lower = fewer topics per document |
| β (beta) | Lower = more distinct topics |
| iterations | More = better convergence |

**Interview Tips:**
- LDA is unsupervised - no labeled data needed
- Choosing K (number of topics) is crucial - use coherence score
- Preprocessing matters: remove stopwords, lemmatize
- Modern alternative: BERTopic (neural topic modeling)

---

## Question 6
**What is TF-IDF and why is it fundamental for text classification and retrieval?**

**Answer:**

**Definition:**
TF-IDF (Term Frequency-Inverse Document Frequency) weights words by their importance: high frequency in a document (TF) but rare across the corpus (IDF). It identifies discriminative words, making it fundamental for classification, search, and information retrieval.

**Mathematical Formulation:**

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

**Term Frequency (TF):**
$$\text{TF}(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total words in } d}$$

**Inverse Document Frequency (IDF):**
$$\text{IDF}(t) = \log\frac{N}{\text{docs containing } t}$$

Where N = total documents in corpus.

**Intuition:**
```
Word "the":     High TF, Low IDF → Low TF-IDF (common everywhere)
Word "neural":  High TF in ML docs, High IDF → High TF-IDF (discriminative)
```

**Why TF-IDF Works:**
- Downweights common words (the, is, and)
- Upweights distinctive words (algorithm, diagnosis)
- Creates sparse, meaningful feature vectors
- No training required - just statistics

**Python Code Example:**
```python
# Pipeline: Documents -> TF-IDF vectors -> Use for classification/retrieval

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
    "Machine learning uses algorithms to learn from data",
    "Deep learning is a subset of machine learning",
    "The stock market closed higher today",
    "Investors watch market trends carefully"
]

# Step 1: Fit TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Step 2: Examine vocabulary and weights
feature_names = vectorizer.get_feature_names_out()
print(f"Vocabulary size: {len(feature_names)}")

# Show TF-IDF for first document
doc_0_tfidf = tfidf_matrix[0].toarray()[0]
word_scores = [(word, score) for word, score in zip(feature_names, doc_0_tfidf) if score > 0]
word_scores.sort(key=lambda x: -x[1])
print(f"Top words in doc 0: {word_scores[:5]}")

# Step 3: Document similarity (retrieval)
query = "machine learning algorithms"
query_vec = vectorizer.transform([query])

similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
ranked_docs = np.argsort(similarities)[::-1]

print("\nSearch results for:", query)
for idx in ranked_docs:
    print(f"  {similarities[idx]:.3f}: {documents[idx][:50]}...")

# Step 4: Use for classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Classification pipeline
classifier = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train and predict
texts = ["Great product", "Terrible quality", "Love it", "Waste of money"]
labels = [1, 0, 1, 0]
classifier.fit(texts, labels)

new_text = ["Amazing purchase"]
prediction = classifier.predict(new_text)
print(f"\nPrediction for '{new_text[0]}': {prediction[0]}")

# TF-IDF variants
from sklearn.feature_extraction.text import TfidfVectorizer

# Sublinear TF (log scaling)
tfidf_sublinear = TfidfVectorizer(sublinear_tf=True)
# TF = 1 + log(count) instead of raw count

# N-grams
tfidf_ngram = TfidfVectorizer(ngram_range=(1, 2))
# Includes unigrams and bigrams

# With preprocessing
tfidf_full = TfidfVectorizer(
    max_features=5000,
    min_df=2,          # Ignore rare words
    max_df=0.95,       # Ignore very common words
    stop_words='english',
    ngram_range=(1, 2)
)
```

**TF-IDF Variants:**

| Variant | Formula | Use |
|---------|---------|-----|
| Sublinear TF | 1 + log(tf) | Reduces impact of very high frequency |
| Smooth IDF | log((N+1)/(df+1)) + 1 | Avoids zero division |
| BM25 | Saturation + length normalization | Information retrieval |

**Interview Tips:**
- TF-IDF is unsupervised - no training labels needed
- Still a strong baseline for many tasks
- BM25 is TF-IDF variant used in Elasticsearch, search engines
- Modern: combine TF-IDF with neural embeddings for hybrid search

---

## Question 7
**How do word embeddings (Word2Vec, GloVe, FastText) improve text classification?**

**Answer:**

**Definition:**
Word embeddings map words to dense, low-dimensional vectors where semantically similar words have similar representations. They improve text classification by capturing semantic relationships, handling synonyms, and providing pre-trained features that generalize better than one-hot encoding.

**Key Algorithms:**

| Algorithm | Approach | Key Feature |
|-----------|----------|-------------|
| Word2Vec | Predict word from context (CBOW) or context from word (Skip-gram) | Local context windows |
| GloVe | Factorize word co-occurrence matrix | Global statistics |
| FastText | Skip-gram with character n-grams | Handles OOV words |

**Why Embeddings Help:**

| Problem | One-Hot | Embeddings |
|---------|---------|------------|
| Dimensionality | Vocabulary size (huge) | 100-300 dims (small) |
| Semantics | No similarity info | Similar words → close vectors |
| Synonyms | Completely different | Nearly identical vectors |
| OOV words | Zero vector | FastText handles via subwords |

**Mathematical Intuition:**

Word2Vec (Skip-gram objective):
$$\max \sum_{(w,c) \in D} \log P(c|w) = \log \frac{\exp(v_c \cdot v_w)}{\sum_{c'} \exp(v_{c'} \cdot v_w)}$$

Result: $v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$

**Python Code Example:**
```python
# Pipeline: Load embeddings -> Aggregate for document -> Classify

import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression

# Load pre-trained embeddings
# word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors.bin', binary=True)

# Simulated embeddings for demo
class SimpleEmbeddings:
    def __init__(self, dim=100):
        self.dim = dim
        self.vocab = {}
    
    def __getitem__(self, word):
        if word not in self.vocab:
            np.random.seed(hash(word) % 2**32)
            self.vocab[word] = np.random.randn(self.dim)
        return self.vocab[word]
    
    def __contains__(self, word):
        return True

embeddings = SimpleEmbeddings(dim=100)

# Method 1: Average word embeddings
def text_to_embedding(text, embeddings, dim=100):
    words = text.lower().split()
    vectors = [embeddings[w] for w in words if w in embeddings]
    
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(dim)

# Create document embeddings
texts = ["great movie loved it", "terrible film waste of time",
         "amazing performance", "boring and slow"]
labels = [1, 0, 1, 0]

X = np.array([text_to_embedding(t, embeddings) for t in texts])
y = np.array(labels)

# Train classifier on embeddings
clf = LogisticRegression()
clf.fit(X, y)

# Predict
new_text = "wonderful experience"
new_vec = text_to_embedding(new_text, embeddings).reshape(1, -1)
print(f"Prediction: {clf.predict(new_vec)}")

# Method 2: Using Gensim
from gensim.models import Word2Vec

# Train custom Word2Vec
sentences = [text.lower().split() for text in texts]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, epochs=100)

# Get word vector
print(f"Vector for 'great': {w2v_model.wv['great'][:5]}...")

# Method 3: FastText (handles OOV)
from gensim.models import FastText

ft_model = FastText(sentences, vector_size=100, window=5, min_count=1)

# FastText can handle OOV words!
oov_word = "movieeee"  # Misspelling
print(f"FastText handles OOV: {ft_model.wv[oov_word][:5]}...")
```

**Embedding Aggregation Methods:**

| Method | Description | Quality |
|--------|-------------|---------|
| Average | Mean of word vectors | Simple, works well |
| Weighted avg | TF-IDF weighted mean | Better for long docs |
| Max pooling | Element-wise max | Captures salient features |
| Concatenate | Combine multiple methods | Most information |

**Interview Tips:**
- Embeddings capture semantics (similarity, analogies)
- FastText is best for noisy text (handles typos)
- Pre-trained embeddings (GloVe, Word2Vec) are plug-and-play
- Modern: Use BERT embeddings (contextual) instead

---

## Question 8
**What is the bag-of-words model and what are its limitations?**

**Answer:**

**Definition:**
Bag-of-Words (BoW) represents text as a vector of word counts, ignoring word order and grammar. Each document becomes a sparse vector where each dimension corresponds to a vocabulary word. It's simple but loses sequential and semantic information.

**How BoW Works:**
```
Vocabulary: [apple, banana, eat, I, love]

Doc 1: "I love apple"     → [1, 0, 0, 1, 1]
Doc 2: "I eat banana"     → [0, 1, 1, 1, 0]
Doc 3: "apple apple love" → [2, 0, 0, 0, 1]
```

**Limitations:**

| Limitation | Example |
|------------|---------|
| No word order | "dog bites man" = "man bites dog" |
| No semantics | "happy" and "joyful" are unrelated |
| High dimensionality | Vocabulary size vectors |
| Sparse vectors | Most entries are zero |
| No context | "bank" (river) = "bank" (financial) |
| Ignores grammar | Loses syntactic structure |

**Python Code Example:**
```python
# Pipeline: Text -> BoW -> Classification

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Sample documents
docs = [
    "I love machine learning",
    "machine learning is great",
    "I hate bad algorithms",
    "bad code is frustrating"
]
labels = [1, 1, 0, 0]  # positive, negative

# Create BoW representation
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(docs)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW matrix shape:", bow_matrix.shape)
print("Doc 0:", bow_matrix[0].toarray())

# Limitation 1: No word order
doc1 = "dog bites man"
doc2 = "man bites dog"

vec1 = vectorizer.fit_transform([doc1, doc2])
print(f"\n'{doc1}' == '{doc2}': {(vec1[0].toarray() == vec1[1].toarray()).all()}")
# True - same representation for different meanings!

# Limitation 2: No semantics
from sklearn.metrics.pairwise import cosine_similarity

vocab_docs = ["happy joyful", "happy sad", "car automobile"]
v = CountVectorizer()
vecs = v.fit_transform(vocab_docs)

print("\nSemantic similarity (BoW):")
print(f"'happy joyful' vs 'happy sad': {cosine_similarity(vecs[0], vecs[1])[0,0]:.2f}")
# High similarity despite opposite meanings!

# Limitation 3: High dimensionality
large_corpus = ["doc " + str(i) + " has unique word" + str(i) for i in range(1000)]
large_bow = CountVectorizer().fit_transform(large_corpus)
print(f"\nLarge corpus: {large_bow.shape[1]} dimensions (mostly zeros)")

# Improvements over basic BoW
# 1. N-grams (capture some word order)
bigram_vec = CountVectorizer(ngram_range=(1, 2))
bigram_matrix = bigram_vec.fit_transform(["dog bites man", "man bites dog"])
print(f"\nWith bigrams, same docs: {(bigram_matrix[0].toarray() == bigram_matrix[1].toarray()).all()}")
# False - bigrams capture some order!

# 2. TF-IDF (weight by importance)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(docs)
# Downweights common words, upweights distinctive ones
```

**BoW vs Modern Approaches:**

| Aspect | BoW | Embeddings | Transformers |
|--------|-----|------------|--------------|
| Word order | ✗ | ✗ (static) | ✓ |
| Semantics | ✗ | ✓ | ✓ |
| Context | ✗ | ✗ | ✓ |
| Sparsity | High | Dense | Dense |
| Speed | Fast | Fast | Slow |

**When BoW is Still Useful:**
- Quick baseline
- Interpretability needed
- Limited compute
- Simple classification with distinctive keywords

**Interview Tips:**
- BoW is simplest text representation - always mention limitations
- N-grams partially address word order
- TF-IDF improves BoW by weighting
- Modern NLP uses embeddings/transformers instead

---

## Question 9
**How do CNNs and RNNs differ in their approach to text classification?**

**Answer:**

**Definition:**
CNNs apply convolutional filters to capture local n-gram patterns in parallel, while RNNs process text sequentially to capture long-range dependencies. CNNs are faster and good for phrase-level patterns; RNNs better model sequential structure and long-term context.

**Core Differences:**

| Aspect | CNN | RNN (LSTM/GRU) |
|--------|-----|----------------|
| Processing | Parallel (all positions) | Sequential (left-to-right) |
| Context | Local (filter window) | Global (hidden state) |
| Speed | Faster | Slower (sequential) |
| Long dependencies | Limited by filter size | Better (but vanishing gradient) |
| Position invariance | Yes (same filter everywhere) | No (order matters) |

**CNN for Text:**
```
Input: "I love this movie" → Embeddings (4×d)

Apply filters (size 2, 3, 4):
- Size 2: ["I love", "love this", "this movie"]
- Size 3: ["I love this", "love this movie"]
- Size 4: ["I love this movie"]

Max-pool each → Concatenate → Classify
```

**RNN for Text:**
```
Input: "I love this movie"

h0 → [I] → h1 → [love] → h2 → [this] → h3 → [movie] → h4

Final h4 captures full sequence → Classify
```

**Python Code Example:**
```python
# Pipeline: Text -> Embeddings -> CNN/RNN -> Classification

import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN for Text Classification
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, 
                 filter_sizes=[2, 3, 4], num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Multiple filter sizes for different n-gram patterns
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) 
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed)
        embedded = embedded.permute(0, 2, 1)  # (batch, embed, seq_len)
        
        # Apply each filter and max-pool
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch, filters, seq-fs+1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch, filters, 1)
            conv_outputs.append(pooled.squeeze(2))
        
        # Concatenate all filter outputs
        concat = torch.cat(conv_outputs, dim=1)  # (batch, filters * num_sizes)
        
        dropped = self.dropout(concat)
        logits = self.fc(dropped)
        return logits

# RNN (LSTM) for Text Classification
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 num_layers=2, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.5
        )
        
        direction_mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_mult, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed)
        
        # LSTM processes sequentially
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch, seq_len, hidden * directions)
        
        # Use last hidden state (or mean pool)
        # For bidirectional: concatenate forward and backward
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        dropped = self.dropout(hidden)
        logits = self.fc(dropped)
        return logits

# Compare architectures
vocab_size, embed_dim, num_classes = 10000, 100, 2
batch_size, seq_len = 32, 50

cnn = TextCNN(vocab_size, embed_dim, num_classes)
rnn = TextRNN(vocab_size, embed_dim, 128, num_classes)

x = torch.randint(0, vocab_size, (batch_size, seq_len))

cnn_out = cnn(x)
rnn_out = rnn(x)

print(f"CNN output: {cnn_out.shape}")  # (32, 2)
print(f"RNN output: {rnn_out.shape}")  # (32, 2)
```

**When to Use:**

| Use Case | Best Choice |
|----------|-------------|
| Short texts (tweets) | CNN |
| Long documents | RNN/LSTM |
| Speed critical | CNN |
| Order matters strongly | RNN |
| Phrase detection | CNN |

**Interview Tips:**
- CNN is faster due to parallelization
- LSTM better for long dependencies but slower
- BiLSTM (bidirectional) captures both directions
- Modern transformers largely replaced both for text classification

---

## Question 10
**How do transformer models like BERT revolutionize text classification tasks?**

**Answer:**

**Definition:**
BERT (Bidirectional Encoder Representations from Transformers) revolutionized text classification through pre-training on massive unlabeled text, bidirectional context understanding, and transfer learning. Fine-tuning BERT on small labeled datasets achieves state-of-the-art results with minimal task-specific architecture.

**Why BERT is Revolutionary:**

| Before BERT | After BERT |
|-------------|------------|
| Train from scratch | Pre-trained, just fine-tune |
| Task-specific architecture | Same model for all tasks |
| Need large labeled data | Works with small labeled data |
| Unidirectional context | Full bidirectional context |
| Static word embeddings | Contextual embeddings |

**BERT Key Innovations:**
1. **Pre-training:** Masked LM + Next Sentence Prediction on huge corpus
2. **Bidirectional:** Sees full context (left and right)
3. **Transfer Learning:** Pre-train once, fine-tune for any task
4. **[CLS] Token:** Special token for classification tasks

**How BERT Classification Works:**
```
Input: [CLS] This movie is great [SEP]
         ↓
      BERT Encoder (12 layers)
         ↓
     [CLS] representation (768-dim)
         ↓
     Linear layer → softmax
         ↓
     Class probabilities
```

**Python Code Example:**
```python
# Pipeline: Text -> BERT tokenizer -> BERT -> Fine-tune classifier

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# Step 1: Load pre-trained BERT
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 2: Prepare data
texts = ["I love this movie!", "This film is terrible."]
labels = [1, 0]

# Tokenize
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
encodings['labels'] = torch.tensor(labels)

# Step 3: Fine-tune
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,  # Lower LR for fine-tuning
    warmup_steps=100,
    weight_decay=0.01,
)

# Create dataset
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings['labels'])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

dataset = TextDataset(encodings)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# Step 4: Inference
model.eval()
with torch.no_grad():
    test_text = "Amazing performance by the actors!"
    inputs = tokenizer(test_text, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = probs.argmax().item()
    print(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")

# Simpler: Using pipeline
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I absolutely loved this film!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

**BERT Variants for Classification:**

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| BERT-base | 110M params | Moderate | High |
| DistilBERT | 66M params | 1.6x faster | ~97% of BERT |
| RoBERTa | 125M params | Same | Higher |
| ALBERT | 12M params | 18x smaller | Similar |

**Fine-tuning Best Practices:**
- Learning rate: 2e-5 to 5e-5
- Epochs: 2-4 (rarely more)
- Batch size: 16-32
- Warmup: 10% of steps
- Use [CLS] token output

**Interview Tips:**
- BERT's bidirectionality is key - sees full context
- Fine-tuning = small dataset can achieve SOTA
- [CLS] token is specially trained for classification
- Alternatives: RoBERTa (no NSP), DistilBERT (faster), DeBERTa (current SOTA)

---

## Interview Questions

## Question 11
**How do you handle extremely imbalanced datasets in text classification?**

**Answer:**

**Definition:**
Class imbalance occurs when some classes have far more examples than others (e.g., 95% negative, 5% positive). This causes models to favor the majority class. Solutions include resampling, class weighting, threshold adjustment, and specialized loss functions.

**Strategies:**

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Oversampling (SMOTE) | Duplicate/synthesize minority samples | Small minority class |
| Undersampling | Remove majority samples | Large dataset |
| Class weights | Weight loss by inverse frequency | Most common |
| Threshold tuning | Adjust classification threshold | At inference |
| Focal loss | Down-weight easy examples | Extreme imbalance |

**Python Code Example:**
```python
# Pipeline: Handle imbalanced classification

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn

# Example: 95% negative, 5% positive
labels = [0]*950 + [1]*50  # Highly imbalanced

# Method 1: Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
print(f"Class weights: {class_weights}")  # [0.53, 10.0]

# Use in loss function
weights = torch.tensor(class_weights, dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=weights)

# Method 2: Oversampling with WeightedRandomSampler
from torch.utils.data import WeightedRandomSampler, DataLoader

sample_weights = [10.0 if l == 1 else 1.0 for l in labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
# Now minority class sampled more frequently

# Method 3: Focal Loss (for extreme imbalance)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Method 4: Threshold adjustment at inference
def predict_with_threshold(probs, threshold=0.3):
    """Lower threshold for minority class"""
    return (probs[:, 1] > threshold).int()  # Positive if prob > 0.3 (not 0.5)
```

**Interview Tips:**
- Class weights are simplest and often sufficient
- Focal loss is effective for extreme imbalance
- Always evaluate with F1, not accuracy (accuracy misleading)
- Consider business cost: is missing positive worse than false positive?

---

## Question 12
**What techniques work best for multi-label classification with label dependencies?**

**Answer:**

**Definition:**
Label dependencies exist when the presence of one label affects the probability of another (e.g., "action" and "thriller" often co-occur in movies). Techniques include classifier chains, label embeddings, and graph neural networks to model these correlations.

**Approaches:**

| Technique | Description |
|-----------|-------------|
| Binary Relevance | Independent classifier per label (ignores dependencies) |
| Classifier Chains | Sequential prediction, each uses previous labels |
| Label Powerset | Treat each label combination as unique class |
| Label Embeddings | Learn label representations capturing correlations |
| Graph Neural Networks | Model label co-occurrence as graph |

**Python Code Example:**
```python
# Classifier Chain: Each label uses previous predictions
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

base_clf = LogisticRegression()
chain = ClassifierChain(base_clf, order='random')
# Labels predicted in sequence, each sees previous predictions

# Label embedding approach (conceptual)
class LabelAwareClassifier(nn.Module):
    def __init__(self, text_dim, num_labels, label_embed_dim):
        super().__init__()
        self.label_embeddings = nn.Embedding(num_labels, label_embed_dim)
        self.classifier = nn.Linear(text_dim + label_embed_dim, 1)
    
    def forward(self, text_repr):
        # Score each label considering text and label embedding
        all_labels = torch.arange(num_labels)
        label_embs = self.label_embeddings(all_labels)
        # Combine with text and score
        scores = []
        for i in range(num_labels):
            combined = torch.cat([text_repr, label_embs[i]], dim=-1)
            scores.append(self.classifier(combined))
        return torch.cat(scores, dim=-1)
```

**Interview Tips:**
- Binary relevance is baseline - ignores all dependencies
- Classifier chains are simple and effective
- Label co-occurrence matrix can guide modeling

---

## Question 13
**How do you implement domain adaptation when training data differs from target domain?**

**Answer:**

**Definition:**
Domain adaptation bridges the gap between source domain (training data) and target domain (deployment). Techniques include continued pre-training on target domain text, gradual fine-tuning, and domain-adversarial training.

**Strategies:**

| Strategy | Description |
|----------|-------------|
| Continued pre-training | MLM on target domain unlabeled text |
| Gradual unfreezing | Fine-tune top layers first, then all |
| Domain-adversarial | Learn domain-invariant features |
| Data augmentation | Back-translate to increase diversity |
| Instance weighting | Weight source samples by target similarity |

**Python Code Example:**
```python
# Step 1: Continued pre-training on target domain
from transformers import AutoModelForMaskedLM, Trainer

# Load base model
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

# Continue MLM training on target domain (unlabeled medical/legal text)
# trainer.train() on domain corpus

# Step 2: Then fine-tune for classification
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('./domain-adapted-model')
# Fine-tune on labeled data
```

**Interview Tips:**
- Use domain-specific BERT if available (BioBERT, LegalBERT)
- Continued pre-training on 10-50MB domain text helps significantly
- Gradual unfreezing prevents catastrophic forgetting

---

## Question 14
**What strategies help with classifying very long documents that exceed model context limits?**

**Answer:**

**Definition:**
BERT's 512 token limit is problematic for long documents. Strategies include chunking with aggregation, hierarchical models, sparse attention transformers, and selecting relevant sections.

**Strategies:**

| Strategy | Description |
|----------|-------------|
| Chunking + pooling | Split into chunks, aggregate predictions |
| Sliding window | Overlapping chunks, combine scores |
| Hierarchical | Sentence-level → Document-level |
| Longformer/BigBird | Sparse attention for long sequences |
| Key section selection | Classify intro/conclusion only |

**Python Code Example:**
```python
# Chunking strategy
def classify_long_doc(text, model, tokenizer, max_len=512, stride=256):
    # Tokenize full document
    tokens = tokenizer.tokenize(text)
    
    # Create overlapping chunks
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+max_len-2]  # Space for [CLS], [SEP]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
        if i + max_len >= len(tokens):
            break
    
    # Classify each chunk
    chunk_probs = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        chunk_probs.append(probs)
    
    # Aggregate: mean pooling
    final_probs = torch.stack(chunk_probs).mean(dim=0)
    return final_probs.argmax().item()

# Alternative: Use Longformer (4096 tokens)
from transformers import LongformerForSequenceClassification
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096')
```

**Interview Tips:**
- Longformer/BigBird handle up to 4096-16K tokens natively
- Chunking is simple and works well for most cases
- First/last chunk often most informative (intro/conclusion)

---

## Question 15
**How do you build effective text classifiers with limited labeled data (few-shot learning)?**

**Answer:**

**Definition:**
Few-shot learning builds classifiers with only a handful of examples per class (1-10). Techniques include prompt-based methods with LLMs, contrastive learning (SetFit), and prototype networks.

**Approaches:**

| Technique | Examples Needed | Method |
|-----------|-----------------|--------|
| In-context learning (GPT) | 1-5 | Examples in prompt |
| SetFit | 8-16 | Contrastive fine-tuning |
| Prompt tuning | 10-50 | Learn soft prompts |
| Prototypical networks | 5-20 | Class prototypes |

**Python Code Example:**
```python
# Method 1: GPT-3/4 in-context learning
def few_shot_classify(text, examples):
    prompt = "Classify sentiment as positive or negative.\n\n"
    for ex_text, ex_label in examples:
        prompt += f"Text: {ex_text}\nSentiment: {ex_label}\n\n"
    prompt += f"Text: {text}\nSentiment:"
    # Call GPT API
    return gpt_response(prompt)

# Method 2: SetFit (efficient few-shot)
from setfit import SetFitModel, SetFitTrainer

# Only 8 examples per class!
train_texts = ["Great product!", "Terrible quality", ...]
train_labels = [1, 0, ...]

model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
trainer = SetFitTrainer(model=model, train_texts=train_texts, train_labels=train_labels)
trainer.train()

# Predict
model.predict(["This is amazing!"])
```

**Interview Tips:**
- SetFit is current best for few-shot without LLM API costs
- GPT-4 with few examples often beats fine-tuned models
- Contrastive learning helps with representation quality
- 8-16 diverse examples per class is often sufficient

---

## Question 16
**What approaches work best for multilingual and cross-lingual text classification?**

**Answer:**

**Definition:**
Multilingual classification handles multiple languages with one model. Cross-lingual classification trains on one language (usually English) and applies to others (zero-shot transfer). Key enabler: multilingual pre-trained models like mBERT and XLM-R.

**Approaches:**

| Approach | Description |
|----------|-------------|
| Multilingual model | mBERT, XLM-R trained on 100+ languages |
| Zero-shot transfer | Train on English, apply to other languages |
| Translate-train | Translate training data to target languages |
| Translate-test | Translate test data to English |

**Python Code Example:**
```python
# Zero-shot cross-lingual with XLM-RoBERTa
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Train on English
model_name = "xlm-roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fine-tune on English sentiment data
# trainer.train(english_data)

# Apply directly to other languages (zero-shot)
french_text = "Ce film est excellent!"
spanish_text = "Esta película es terrible."

for text in [french_text, spanish_text]:
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    pred = outputs.logits.argmax().item()
    print(f"{text} -> {'Positive' if pred == 1 else 'Negative'}")
```

**Interview Tips:**
- XLM-R outperforms mBERT for cross-lingual tasks
- Zero-shot transfer works surprisingly well for related languages
- Translate-train often better if translation quality is good
- Performance degrades for distant language pairs

---

## Question 17
**How do you handle sentiment analysis for texts with sarcasm and irony?**

**Answer:**

**Definition:**
Sarcasm/irony involve saying the opposite of what's meant ("Great, another rainy day!"). Standard sentiment models fail because surface-level positive words mask negative intent. Detection requires context, world knowledge, and sometimes speaker history.

**Challenges:**

| Challenge | Example |
|-----------|--------|
| Surface vs intent mismatch | "Oh wonderful, flat tire" (negative) |
| Context dependency | Needs prior knowledge |
| Missing multimodal cues | No tone/expression in text |
| Dataset scarcity | Hard to annotate reliably |

**Approaches:**

| Approach | Description |
|----------|-------------|
| Context modeling | Include parent post, thread context |
| Sentiment contrast | Detect positive words + negative context |
| Hashtag signals | #sarcasm, #not as training labels |
| User history | Model user's typical sarcasm patterns |
| Multi-task learning | Joint sarcasm detection + sentiment |

**Python Code Example:**
```python
# Two-stage pipeline: Detect sarcasm first, then adjust sentiment
from transformers import pipeline

# Stage 1: Sarcasm detection
sarcasm_model = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")

# Stage 2: Standard sentiment
sentiment_model = pipeline("sentiment-analysis")

def sarcasm_aware_sentiment(text):
    # Check for sarcasm
    sarcasm_result = sarcasm_model(text)[0]
    is_sarcastic = sarcasm_result['label'] == 'irony' and sarcasm_result['score'] > 0.7
    
    # Get base sentiment
    sentiment = sentiment_model(text)[0]
    
    if is_sarcastic:
        # Flip sentiment if sarcastic
        if sentiment['label'] == 'POSITIVE':
            return {'label': 'NEGATIVE', 'note': 'sarcasm detected'}
        else:
            return {'label': 'POSITIVE', 'note': 'sarcasm detected'}
    
    return sentiment

# Example
print(sarcasm_aware_sentiment("Oh great, another Monday"))
```

**Interview Tips:**
- Context is crucial - parent tweet, thread, user history
- Hashtags like #sarcasm often used for training data
- Still an unsolved problem - state-of-art ~75% accuracy
- Multi-task learning (sarcasm + sentiment) helps

---

## Question 18
**What techniques help detect and handle mixed/conflicting sentiments in reviews?**

**Answer:**

**Definition:**
Mixed sentiment occurs when text contains both positive and negative opinions ("Great battery life but terrible screen"). Techniques include sentence-level analysis, aspect-based sentiment, and multi-label classification to capture nuanced opinions.

**Detection Approaches:**

| Approach | Description |
|----------|-------------|
| Sentence-level analysis | Classify each sentence separately |
| Aspect-based sentiment | Extract aspects, sentiment per aspect |
| Sentiment span detection | Identify positive/negative spans |
| Multi-label output | Predict both positive AND negative |

**Python Code Example:**
```python
# Approach 1: Sentence-level sentiment aggregation
from transformers import pipeline
import nltk
nltk.download('punkt')

sentiment_model = pipeline("sentiment-analysis")

def analyze_mixed_sentiment(text):
    sentences = nltk.sent_tokenize(text)
    results = {'positive': [], 'negative': [], 'neutral': []}
    
    for sent in sentences:
        pred = sentiment_model(sent)[0]
        label = pred['label'].lower()
        results[label].append({'text': sent, 'score': pred['score']})
    
    # Summarize
    summary = {
        'overall': 'mixed' if (results['positive'] and results['negative']) else 'uniform',
        'positive_aspects': len(results['positive']),
        'negative_aspects': len(results['negative']),
        'details': results
    }
    return summary

# Example
review = "The camera quality is amazing. However, battery drains quickly. Customer service was helpful."
print(analyze_mixed_sentiment(review))
# Output: mixed sentiment with 2 positive, 1 negative

# Approach 2: Aspect-based for structured output
# Uses models like pyabsa or aspect extraction
aspects = {
    'camera': 'positive',
    'battery': 'negative', 
    'service': 'positive'
}
```

**Representation Options:**
- Sentiment ratio: 70% positive, 30% negative
- Per-sentence labels with confidence
- Aspect-sentiment pairs
- Star rating + detailed breakdown

**Interview Tips:**
- Overall sentiment can be misleading for mixed reviews
- Aspect-based provides actionable insights
- E-commerce applications need mixed sentiment handling
- Consider sentiment intensity, not just polarity

---

## Question 19
**How do you implement fine-grained emotion detection beyond positive/negative sentiment?**

**Answer:**

**Definition:**
Fine-grained emotion detection identifies specific emotions (joy, anger, fear, sadness, surprise, disgust) rather than just polarity. It's more informative for customer feedback, mental health analysis, and nuanced understanding.

**Emotion Taxonomies:**

| Taxonomy | Emotions |
|----------|----------|
| Ekman (6 basic) | Joy, Anger, Fear, Sadness, Surprise, Disgust |
| Plutchik (8+) | Above + Trust, Anticipation + combinations |
| GoEmotions (27) | Fine-grained: admiration, amusement, annoyance, etc. |

**Key Differences from Sentiment:**

| Aspect | Sentiment | Emotion |
|--------|-----------|--------|
| Output | 2-3 classes | 6-28 classes |
| Granularity | Coarse | Fine-grained |
| Multi-label | Rare | Common (text can express multiple) |

**Python Code Example:**
```python
from transformers import pipeline

# Pre-trained emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None  # Return all emotions with scores
)

text = "I can't believe they cancelled my flight!"
result = emotion_classifier(text)
print(result)
# [{'label': 'anger', 'score': 0.78}, {'label': 'disgust', 'score': 0.12}, ...]

# GoEmotions for fine-grained (27 emotions)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "monologg/bert-base-cased-goemotions-original"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_emotions(text, threshold=0.3):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze()  # Multi-label
    
    # Get emotions above threshold
    emotions = model.config.id2label
    detected = [(emotions[i], float(probs[i])) 
                for i in range(len(probs)) if probs[i] > threshold]
    return sorted(detected, key=lambda x: x[1], reverse=True)

print(predict_emotions("This is absolutely wonderful!"))
```

**Interview Tips:**
- Use sigmoid (multi-label), not softmax - text can express multiple emotions
- GoEmotions dataset (Google) is current standard
- Emotion intensity is also measurable (how angry?)
- Cultural context affects emotion expression

---

## Question 20
**What challenges arise in cross-cultural sentiment analysis and how do you address them?**

**Answer:**

**Definition:**
Cross-cultural sentiment analysis must handle variations in how emotions are expressed across cultures, languages, and regions. Same words may carry different sentiment weights, and expression styles vary significantly.

**Key Challenges:**

| Challenge | Example |
|-----------|--------|
| Direct vs indirect expression | Western cultures more direct; Asian cultures more indirect |
| Polarity differences | "Not bad" = positive in some cultures, neutral in others |
| Formality levels | Formal language may mask true sentiment |
| Cultural references | Idioms, slang don't translate |
| Annotation bias | Annotators bring cultural bias |

**Solutions:**

| Approach | Description |
|----------|-------------|
| Multilingual models | XLM-R, mBERT for language transfer |
| Culture-specific fine-tuning | Train on target culture data |
| Diverse annotation | Annotators from target culture |
| Cross-cultural datasets | Balanced representation |
| Calibration | Adjust scores per region |

**Python Code Example:**
```python
# Using XLM-R for multilingual sentiment
from transformers import pipeline

# Multilingual model handles 100+ languages
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

# Same concept, different expressions
texts = [
    "This is amazing!",           # English, direct
    "Es ist nicht schlecht",      # German "not bad" = positive
    "まあまあですね",               # Japanese "so-so" = often positive
]

for text in texts:
    print(f"{text}: {sentiment_model(text)}")

# Culture-aware calibration
def culture_calibrated_sentiment(text, culture):
    base_score = sentiment_model(text)[0]['score']
    
    # Calibration factors (example)
    calibration = {
        'east_asian': {'neutral_shift': 0.15},  # Shift neutral toward positive
        'western': {'neutral_shift': 0.0},
        'middle_eastern': {'intensity_scale': 1.2}  # Expressions more intense
    }
    
    if culture in calibration:
        base_score += calibration[culture].get('neutral_shift', 0)
    
    return min(max(base_score, 0), 1)  # Clip to [0,1]
```

**Interview Tips:**
- Never assume US-trained model works globally
- "Not bad" polarity varies significantly by culture
- Expression intensity differs - Middle East > East Asia typically
- Local annotators essential for quality data
- Test on target culture before deployment

---

## Question 21
**How do you choose the optimal number of topics in LDA? Explain coherence scores.**

**Answer:**

**Definition:**
Choosing K (number of topics) is crucial for LDA quality. Coherence scores measure how semantically similar the top words in each topic are - higher coherence means more interpretable topics. Common metrics: c_v, u_mass, c_npmi.

**Approaches to Choose K:**

| Method | Description |
|--------|-------------|
| Coherence score | Plot coherence vs K, pick peak |
| Perplexity | Lower = better fit (but not interpretability) |
| Elbow method | Find diminishing returns point |
| Domain knowledge | Expected number of themes |
| Human evaluation | Gold standard but expensive |

**Coherence Metrics:**

| Metric | Range | Description |
|--------|-------|-------------|
| c_v | 0-1 | Sliding window + NPMI, best correlation with humans |
| u_mass | -14 to 14 | Uses document co-occurrence |
| c_npmi | -1 to 1 | Normalized PMI |
| c_uci | -∞ to +∞ | Pointwise mutual information |

**Coherence Intuition:**
A topic with words ["python", "code", "programming", "function"] has high coherence.
A topic with words ["python", "banana", "election", "guitar"] has low coherence.

**Python Code Example:**
```python
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

# Grid search for optimal K
coherence_scores = []
k_range = range(5, 50, 5)

for num_topics in k_range:
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10
    )
    
    # Calculate c_v coherence (best for human correlation)
    coherence_model = CoherenceModel(
        model=lda, 
        texts=texts, 
        dictionary=dictionary, 
        coherence='c_v'
    )
    score = coherence_model.get_coherence()
    coherence_scores.append(score)
    print(f"K={num_topics}, Coherence={score:.4f}")

# Plot and find peak
plt.plot(k_range, coherence_scores)
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.title('Optimal Topic Number Selection')

best_k = k_range[coherence_scores.index(max(coherence_scores))]
print(f"Optimal K: {best_k}")
```

**Interview Tips:**
- c_v coherence correlates best with human judgment
- Perplexity can improve while interpretability worsens
- Typical K: 10-50 for most corpora
- Multiple runs needed (LDA is stochastic)
- Elbow point often better than absolute maximum

---

## Question 22
**What preprocessing steps are critical before applying topic modeling?**

**Answer:**

**Definition:**
Topic modeling preprocessing significantly impacts results. Critical steps include tokenization, lowercasing, stopword removal, lemmatization, removing rare/frequent terms, and handling special characters. Poor preprocessing leads to noisy, uninterpretable topics.

**Critical Steps:**

| Step | Purpose | Impact if Skipped |
|------|---------|-------------------|
| Lowercase | Normalize case | "Python" vs "python" separate |
| Stopword removal | Remove common words | Topics dominated by "the", "is" |
| Lemmatization | Reduce to base form | "running", "runs" separate |
| Remove rare words | Filter noise | Typos become topic words |
| Remove frequent words | Domain stopwords | Domain-common words dominate |
| Punctuation removal | Clean text | Noise in vocabulary |
| Bigrams/trigrams | Capture phrases | "machine learning" split |

**Python Code Example:**
```python
import spacy
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary

nlp = spacy.load('en_core_web_sm')

def preprocess_for_lda(texts):
    processed = []
    
    for text in texts:
        doc = nlp(text.lower())
        
        # Lemmatize, remove stops/punct/short words
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop 
            and not token.is_punct 
            and len(token.text) > 2
            and token.is_alpha
        ]
        processed.append(tokens)
    
    # Build bigrams (e.g., "machine_learning")
    bigram_model = Phrases(processed, min_count=5)
    bigram_phraser = Phraser(bigram_model)
    processed = [bigram_phraser[doc] for doc in processed]
    
    return processed

# Create dictionary with filtering
texts = preprocess_for_lda(raw_documents)
dictionary = Dictionary(texts)

# Filter extremes
dictionary.filter_extremes(
    no_below=5,    # Appear in at least 5 docs
    no_above=0.5,  # Appear in at most 50% of docs
    keep_n=10000   # Keep top 10k words
)

# Create corpus
corpus = [dictionary.doc2bow(text) for text in texts]
```

**Interview Tips:**
- Stopword removal is most impactful step
- Domain-specific stopwords often needed (e.g., "patient" in medical)
- Bigrams capture multi-word concepts
- Filter extremes: too rare = noise, too common = uninformative
- Lemmatization > stemming for interpretability

---

## Question 23
**How do you handle topic modeling for short texts like tweets?**

**Answer:**

**Definition:**
Short texts (tweets, titles, queries) lack sufficient word co-occurrence for standard LDA. Solutions include document pooling, biterm models, neural topic models, and embedding-based approaches like BERTopic.

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| Sparse word co-occurrence | Too few words per doc for patterns |
| High vocabulary variance | Many unique words across docs |
| Context missing | Limited context for inference |

**Solutions:**

| Approach | Description |
|----------|-------------|
| Document pooling | Aggregate by user/hashtag/time |
| Biterm Topic Model | Model word pairs, not documents |
| BERTopic | Embedding clusters + topic extraction |
| Top2Vec | Similar embedding-based approach |
| Pre-trained LDA | Use external corpus, apply to short texts |

**Python Code Example:**
```python
# Approach 1: BERTopic (best for short texts)
from bertopic import BERTopic

tweets = [
    "Just got vaccinated!",
    "New iPhone looks amazing",
    "Stock market crash today",
    # ... more tweets
]

# BERTopic uses embeddings, works great on short texts
topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",  # Sentence embeddings
    min_topic_size=10,
    verbose=True
)
topics, probs = topic_model.fit_transform(tweets)

# View topics
topic_model.get_topic_info()
topic_model.visualize_topics()

# Approach 2: Document pooling
from collections import defaultdict

def pool_by_hashtag(tweets_with_hashtags):
    pooled = defaultdict(list)
    for tweet, hashtag in tweets_with_hashtags:
        pooled[hashtag].append(tweet)
    
    # Combine into pseudo-documents
    return [' '.join(texts) for texts in pooled.values()]

# Now apply standard LDA to pooled documents
pooled_docs = pool_by_hashtag(tweet_data)
```

**Interview Tips:**
- BERTopic is current best for short texts
- Pooling by metadata (user, time, hashtag) creates longer docs
- Biterm model specifically designed for short texts
- Standard LDA fails on short texts - don't use directly

---

## Question 24
**What is dynamic topic modeling and when should you use it?**

**Answer:**

**Definition:**
Dynamic Topic Modeling (DTM) tracks how topics evolve over time. Unlike static LDA, DTM assumes topic distributions change across time slices (years, months). Use for analyzing trends, tracking discourse evolution, or understanding shifting narratives.

**Use Cases:**

| Use Case | Example |
|----------|--------|
| News trends | How "AI" coverage changed 2010-2024 |
| Scientific discourse | Evolution of research topics |
| Social movements | How language around issues evolves |
| Brand perception | Sentiment topic shifts over time |

**How It Works:**
DTM chains LDA models across time slices. Topics at time $t$ evolve from topics at time $t-1$:
$$\beta_{t} = \beta_{t-1} + \epsilon$$

where $\beta$ is topic-word distribution and $\epsilon$ is drift.

**Python Code Example:**
```python
# Dynamic Topic Modeling with gensim
from gensim.models import LdaSeqModel
from gensim.corpora import Dictionary

# Organize documents by time
time_slices = [100, 150, 200, 180]  # Docs per year (2020, 2021, 2022, 2023)

# Create corpus (all docs in temporal order)
dictionary = Dictionary(all_texts)
corpus = [dictionary.doc2bow(text) for text in all_texts]

# Train dynamic model
dtm = LdaSeqModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,
    time_slice=time_slices,
    passes=10
)

# View topic evolution
for topic_id in range(10):
    print(f"\nTopic {topic_id} evolution:")
    for time_idx in range(len(time_slices)):
        words = dtm.print_topic(topic_id, time=time_idx)
        print(f"  Time {time_idx}: {words}")

# Alternative: BERTopic with timestamps
from bertopic import BERTopic

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

# Visualize topics over time
topics_over_time = topic_model.topics_over_time(docs, timestamps)
topic_model.visualize_topics_over_time(topics_over_time)
```

**Interview Tips:**
- Use when temporal analysis is the goal
- Requires sufficient docs per time slice
- BERTopic's `topics_over_time` is simpler alternative
- Computational cost higher than static LDA
- Useful for trend analysis, not just current state

---

## Question 25
**How do you explain and interpret topic model results to non-technical stakeholders?**

**Answer:**

**Definition:**
Explaining topic models requires translating statistical outputs into business-relevant insights. Focus on: topic labels (not just word lists), representative documents, visualizations, and actionable interpretations rather than technical details.

**Communication Strategies:**

| Strategy | Description |
|----------|-------------|
| Human labels | Name topics ("Customer Complaints" not "Topic 3") |
| Representative docs | Show example documents for each topic |
| Word clouds | Visual, intuitive representation |
| Topic distribution | "40% of feedback is about shipping" |
| Trend charts | How topics change over time |

**Python Code Example:**
```python
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Interactive visualization with pyLDAvis
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'topic_visualization.html')

# 2. Word clouds per topic
def topic_wordcloud(lda_model, topic_id):
    words = dict(lda_model.show_topic(topic_id, topn=50))
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(words)
    plt.imshow(wc)
    plt.title(f"Topic {topic_id}")
    plt.axis('off')
    plt.show()

# 3. Topic naming helper
def suggest_topic_name(lda_model, topic_id, topn=5):
    top_words = [word for word, prob in lda_model.show_topic(topic_id, topn)]
    print(f"Topic {topic_id}: {', '.join(top_words)}")
    print("Suggested name: _____________")
    return top_words

# 4. Find representative documents
def get_representative_docs(lda_model, corpus, texts, topic_id, n=3):
    doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]
    
    # Find docs with highest probability for this topic
    scored = []
    for i, doc_topic in enumerate(doc_topics):
        topic_prob = dict(doc_topic).get(topic_id, 0)
        scored.append((i, topic_prob))
    
    top_docs = sorted(scored, key=lambda x: x[1], reverse=True)[:n]
    return [(texts[i], prob) for i, prob in top_docs]

# 5. Summary statistics
def topic_summary(lda_model, corpus):
    topic_counts = [0] * lda_model.num_topics
    for doc in corpus:
        dominant = max(lda_model.get_document_topics(doc), key=lambda x: x[1])
        topic_counts[dominant[0]] += 1
    
    for i, count in enumerate(topic_counts):
        pct = count / len(corpus) * 100
        print(f"Topic {i}: {count} docs ({pct:.1f}%)")
```

**Interview Tips:**
- Always name topics with human-readable labels
- pyLDAvis provides excellent interactive exploration
- Show "this is a typical customer complaint about shipping"
- Avoid jargon: "Dirichlet prior" = "assumption about how focused documents are"
- Focus on insights, not methodology

---

## Question 26
**What is the difference between extractive and neural topic models (BERTopic, Top2Vec)?**

**Answer:**

**Definition:**
Extractive topic models (LDA) use word co-occurrence statistics. Neural topic models (BERTopic, Top2Vec) use pre-trained embeddings to capture semantic meaning, then cluster documents, and extract topic words from clusters.

**Comparison:**

| Aspect | LDA (Extractive) | BERTopic/Top2Vec (Neural) |
|--------|-----------------|---------------------------|
| Word representation | Bag-of-words | Dense embeddings |
| Semantic understanding | No | Yes |
| Short texts | Poor | Good |
| Speed | Fast | Slower |
| Topic quality | Good with tuning | Often better out-of-box |
| Interpretability | High | High |

**How Neural Topic Models Work:**

```
1. Document → Embedding (BERT/Sentence-BERT)
2. Dimensionality reduction (UMAP)
3. Clustering (HDBSCAN)
4. Topic extraction (c-TF-IDF)
```

**Python Code Example:**
```python
# BERTopic pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Custom embedding model (optional)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create model with custom settings
topic_model = BERTopic(
    embedding_model=embedding_model,
    min_topic_size=10,         # Minimum docs per topic
    nr_topics="auto",          # Automatic topic count
    verbose=True
)

# Fit
topics, probs = topic_model.fit_transform(documents)

# Explore results
topic_model.get_topic_info()  # Summary
topic_model.get_topic(0)       # Words for topic 0
topic_model.visualize_topics() # Interactive viz

# Top2Vec alternative
from top2vec import Top2Vec

model = Top2Vec(
    documents, 
    embedding_model='universal-sentence-encoder',
    speed="learn"  # Options: fast-learn, learn, deep-learn
)

# Get topics
topic_words, word_scores, topic_nums = model.get_topics(10)

# Compare LDA vs BERTopic on same data
from gensim.models import LdaModel

# LDA needs preprocessing, dictionary, corpus
# BERTopic works directly on raw text
```

**When to Use Each:**

| Scenario | Best Choice |
|----------|-------------|
| Long documents, large corpus | LDA |
| Short texts (tweets, queries) | BERTopic/Top2Vec |
| No preprocessing time | BERTopic |
| Interpretable baselines | LDA |
| Semantic similarity needed | Neural |

**Interview Tips:**
- BERTopic is current state-of-art for most use cases
- Neural models capture "king-queen" semantic relationships
- LDA still preferred when interpretability paramount
- BERTopic works out-of-box, LDA needs tuning

---

## Question 27
**How do you make text classifiers explainable (LIME, SHAP, attention visualization)?**

**Answer:**

**Definition:**
Explainability reveals WHY a model predicted a certain class. LIME creates local linear approximations, SHAP uses game-theoretic attribution, and attention visualization highlights model focus. Essential for high-stakes domains (medical, legal, finance).

**Techniques:**

| Method | Type | Approach |
|--------|------|----------|
| LIME | Model-agnostic | Perturb words, fit local linear model |
| SHAP | Model-agnostic | Shapley values for word contribution |
| Attention | Model-specific | Visualize attention weights |
| Integrated Gradients | Gradient-based | Accumulate gradients along path |

**Python Code Example:**
```python
# LIME for text explanation
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['negative', 'positive'])

def predict_proba(texts):
    # Your model's prediction function
    return model.predict_proba(vectorizer.transform(texts))

exp = explainer.explain_instance(
    "This movie was absolutely terrible and boring",
    predict_proba,
    num_features=10
)
exp.show_in_notebook()
# Output: "terrible": -0.45, "boring": -0.32 (push toward negative)

# SHAP for transformers
import shap
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
explainer = shap.Explainer(classifier)

shap_values = explainer(["The product quality is excellent but shipping was slow"])
shap.plots.text(shap_values)

# Attention visualization
def visualize_attention(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs, output_attentions=True)
    
    # Get last layer attention
    attention = outputs.attentions[-1].squeeze()  # [heads, seq, seq]
    avg_attention = attention.mean(dim=0)  # Average over heads
    
    # CLS token attention to all words
    cls_attention = avg_attention[0]  # [seq_len]
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Display
    for token, weight in zip(tokens, cls_attention.tolist()):
        print(f"{token}: {weight:.3f}")
```

**Important Caveats:**
- Attention weights ≠ explanation (debated in research)
- SHAP is mathematically grounded (Shapley values)
- LIME explanations can be unstable
- Multiple methods together give confidence

**Interview Tips:**
- LIME/SHAP are model-agnostic, work with any classifier
- Attention visualization popular but contested
- Counterfactual explanations: "Would be positive if 'terrible' removed"
- Regulators increasingly require explainability

---

## Question 28
**What is adversarial robustness in text classification and how do you improve it?**

**Answer:**

**Definition:**
Adversarial robustness measures model stability against intentionally crafted inputs. Adversarial examples use typos, synonyms, or paraphrases to fool classifiers while preserving meaning. Defense involves training on adversarial examples and input sanitization.

**Attack Types:**

| Level | Attack | Example |
|-------|--------|--------|
| Character | Typos, homoglyphs | "great" → "gr8", "greаt" |
| Word | Synonyms | "excellent" → "outstanding" |
| Sentence | Paraphrase | "I love it" → "It's something I love" |
| Universal | Trigger phrases | Specific tokens flip any prediction |

**Defense Strategies:**

| Defense | Description |
|---------|-------------|
| Adversarial training | Include adversarial examples in training |
| Data augmentation | Synonym replacement, back-translation |
| Spell correction | Normalize inputs before classification |
| Ensemble | Multiple models vote |
| Certified robustness | Provable bounds on perturbations |

**Python Code Example:**
```python
# Generate adversarial examples with TextAttack
from textattack import Augmenter
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.pre_transformation import RepeatModification

# Create synonym-based augmenter
augmenter = Augmenter(
    transformation=WordSwapEmbedding(max_candidates=50),
    constraints=[RepeatModification()],
    pct_words_to_swap=0.2
)

# Generate adversarial variants for training
original = "This product is excellent and works perfectly"
adversarial_samples = augmenter.augment(original)
# ["This product is outstanding and works perfectly",
#  "This product is superb and functions perfectly", ...]

# Defense: Adversarial training
adversarial_training_data = []
for text, label in training_data:
    adversarial_training_data.append((text, label))  # Original
    for aug_text in augmenter.augment(text):
        adversarial_training_data.append((aug_text, label))  # Adversarial

# Defense: Spell correction preprocessing
from spellchecker import SpellChecker

def sanitize_input(text):
    spell = SpellChecker()
    words = text.split()
    corrected = [spell.correction(word) or word for word in words]
    return ' '.join(corrected)

# Apply before classification
clean_text = sanitize_input("This pr0duct is terribl3")
prediction = model.predict(clean_text)
```

**Interview Tips:**
- TextAttack library provides comprehensive attack/defense tools
- Character-level models (CharCNN) more robust to typos
- Adversarial training is most effective general defense
- No perfect defense - it's an arms race
- Real-world attacks: spam filters, content moderation evasion

---

## Question 29
**How do you implement hierarchical text classification with parent-child category relationships?**

**Answer:**

**Definition:**
Hierarchical classification handles taxonomies where categories have parent-child relationships (e.g., Electronics → Phones → Smartphones). Approaches include flat classification with constraints, cascading classifiers, and hierarchy-aware neural models.

**Example Hierarchy:**
```
Products
├── Electronics
│   ├── Phones
│   └── Laptops
└── Clothing
    ├── Shirts
    └── Pants
```

**Approaches:**

| Approach | Description |
|----------|-------------|
| Flat | Predict leaf nodes, infer parents |
| Cascading | Classifier at each level |
| Local per node | Binary classifier per node |
| Global | One model, hierarchy constraints |
| Hierarchy-aware | Embed hierarchy in model |

**Python Code Example:**
```python
# Approach 1: Cascading classifiers
class HierarchicalClassifier:
    def __init__(self):
        self.level1_clf = LogisticRegression()  # Root categories
        self.level2_clfs = {}  # Child classifiers per parent
    
    def fit(self, texts, labels_l1, labels_l2):
        # Train level 1
        vectors = vectorizer.fit_transform(texts)
        self.level1_clf.fit(vectors, labels_l1)
        
        # Train level 2 classifiers per parent
        for parent in set(labels_l1):
            mask = [l == parent for l in labels_l1]
            child_texts = [t for t, m in zip(texts, mask) if m]
            child_labels = [l for l, m in zip(labels_l2, mask) if m]
            
            self.level2_clfs[parent] = LogisticRegression()
            self.level2_clfs[parent].fit(
                vectorizer.transform(child_texts), child_labels
            )
    
    def predict(self, text):
        vec = vectorizer.transform([text])
        parent = self.level1_clf.predict(vec)[0]
        child = self.level2_clfs[parent].predict(vec)[0]
        return parent, child

# Approach 2: Flat with hierarchy constraints
class HierarchyConstrainedClassifier:
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy  # {child: parent}
        self.clf = None
    
    def predict(self, text):
        probs = self.clf.predict_proba([text])[0]
        
        # Enforce hierarchy: child prob <= parent prob
        for child, parent in self.hierarchy.items():
            child_idx = self.label_to_idx[child]
            parent_idx = self.label_to_idx[parent]
            probs[child_idx] = min(probs[child_idx], probs[parent_idx])
        
        return self.labels[probs.argmax()]
```

**Interview Tips:**
- Cascading approach is intuitive but error propagates
- Flat + constraints balances simplicity and structure
- Hierarchy-aware losses penalize distant errors more
- For deep hierarchies, neural approaches work best
- Consider mandatory path consistency (leaf → root)

---

## Question 30
**What are the key evaluation metrics for text classification (accuracy, F1, macro/micro averaging)?**

**Answer:**

**Definition:**
Evaluation metrics measure classifier performance. Accuracy works for balanced data. F1 balances precision and recall. Macro-averaging treats classes equally; micro-averaging weights by frequency. Choice depends on class distribution and business needs.

**Core Metrics:**

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| Precision | TP/(TP+FP) | Minimize false positives |
| Recall | TP/(TP+FN) | Minimize false negatives |
| F1 | 2·(P·R)/(P+R) | Balance P and R |

**Macro vs Micro Averaging:**

| Averaging | Calculation | Effect |
|-----------|-------------|--------|
| Macro | Mean of per-class metrics | Equal weight to all classes |
| Micro | Global TP/FP/FN counts | Weighted by class frequency |
| Weighted | Per-class weighted by support | Balance between macro/micro |

**Example:**
- 1000 samples: 900 class A, 100 class B
- Model: 100% on A, 0% on B
- Micro-F1: 90% (dominated by majority)
- Macro-F1: 50% (average of 100% and 0%)

**Python Code Example:**
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
y_pred = [0, 0, 1, 1, 1, 0, 2, 2, 2]

# Basic metrics
print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")

# Precision/Recall/F1 with different averaging
print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro'):.3f}")
print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro'):.3f}")
print(f"F1 (macro): {f1_score(y_true, y_pred, average='macro'):.3f}")
print(f"F1 (micro): {f1_score(y_true, y_pred, average='micro'):.3f}")
print(f"F1 (weighted): {f1_score(y_true, y_pred, average='weighted'):.3f}")

# Full report
print(classification_report(y_true, y_pred, target_names=['class_0', 'class_1', 'class_2']))

# Confusion matrix
print(confusion_matrix(y_true, y_pred))
```

**When to Use Each:**

| Scenario | Metric |
|----------|--------|
| Balanced classes | Accuracy or Micro-F1 |
| Imbalanced classes | Macro-F1 |
| Minority class important | Macro-F1 or class-specific F1 |
| Multi-label | Per-label + averaged |

**Interview Tips:**
- Always check class distribution before choosing metric
- Macro-F1 for imbalanced data
- Accuracy misleading when one class dominates
- Business context determines precision vs recall tradeoff

---

## Question 31
**How do you handle concept drift when text patterns change over time?**

**Answer:**

**Definition:**
Concept drift occurs when the relationship between input text and target labels changes over time. Examples: new slang, evolving topics, changing user behavior. Models trained on old data become stale and need updating.

**Types of Drift:**

| Type | Description | Example |
|------|-------------|--------|
| Sudden | Abrupt change | Policy change, new product launch |
| Gradual | Slow evolution | Language evolution, trend shifts |
| Recurring | Seasonal patterns | Holiday sentiment |
| Incremental | Small continuous changes | Slang evolution |

**Detection and Handling:**

| Strategy | Description |
|----------|-------------|
| Monitor metrics | Track performance on recent data |
| Statistical tests | Detect distribution shifts |
| Sliding window | Train on recent data only |
| Online learning | Continuous model updates |
| Periodic retraining | Scheduled model refresh |

**Python Code Example:**
```python
from scipy import stats
import numpy as np

# Drift detection using performance monitoring
class DriftDetector:
    def __init__(self, window_size=100, threshold=0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.accuracies = []
    
    def add_prediction(self, y_true, y_pred):
        correct = int(y_true == y_pred)
        self.accuracies.append(correct)
    
    def detect_drift(self):
        if len(self.accuracies) < 2 * self.window_size:
            return False
        
        old_window = self.accuracies[-2*self.window_size:-self.window_size]
        new_window = self.accuracies[-self.window_size:]
        
        old_acc = np.mean(old_window)
        new_acc = np.mean(new_window)
        
        if old_acc - new_acc > self.threshold:
            return True  # Drift detected
        return False

# Continuous retraining strategy
class AdaptiveClassifier:
    def __init__(self, base_model, retrain_interval=1000):
        self.model = base_model
        self.buffer = []  # Store recent labeled examples
        self.retrain_interval = retrain_interval
    
    def predict(self, text):
        return self.model.predict([text])[0]
    
    def add_feedback(self, text, true_label):
        self.buffer.append((text, true_label))
        
        if len(self.buffer) >= self.retrain_interval:
            self.retrain()
    
    def retrain(self):
        texts, labels = zip(*self.buffer)
        self.model.fit(texts, labels)
        self.buffer = []  # Clear buffer
        print("Model retrained on recent data")
```

**Interview Tips:**
- Monitor production metrics continuously
- Keep holdout from different time periods for testing
- Sliding window balances recency and data volume
- COVID-19 is classic example of sudden drift
- New slang/emojis cause gradual drift

---

## Question 32
**What is zero-shot text classification and how do models like BART/GPT enable it?**

**Answer:**

**Definition:**
Zero-shot classification classifies text into categories without any training examples for those categories. It uses models pre-trained on natural language inference (NLI) or large language models that understand category descriptions through prompting.

**How It Works:**

| Approach | Mechanism |
|----------|----------|
| NLI-based | Treat classification as entailment: "Is this text about {category}?" |
| Prompt-based | Ask LLM directly: "Classify this as: sports/politics/tech" |
| Embedding similarity | Compare text embedding to label embeddings |

**Python Code Example:**
```python
# Approach 1: NLI-based (BART-MNLI, DeBERTa)
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text = "Apple announced the new iPhone with better camera and battery"
candidate_labels = ["technology", "sports", "politics", "entertainment"]

result = classifier(text, candidate_labels)
print(result)
# {'labels': ['technology', 'entertainment', 'politics', 'sports'],
#  'scores': [0.92, 0.04, 0.02, 0.01]}

# Approach 2: GPT prompt-based
def gpt_zero_shot(text, labels):
    prompt = f"""Classify the following text into one of these categories: {', '.join(labels)}
    
Text: {text}

Category:"""
    # Call GPT API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Approach 3: Embedding similarity
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

text_emb = model.encode(text)
label_embs = model.encode(candidate_labels)

scores = util.cos_sim(text_emb, label_embs)
predicted = candidate_labels[scores.argmax()]
```

**Interview Tips:**
- BART-MNLI is go-to for zero-shot without LLM costs
- GPT-4 often outperforms but expensive
- Label phrasing matters: "about technology" vs "technology"
- Works best when labels are descriptive
- Not truly "zero" - models learned from NLI/pre-training

---

## Question 33
**How do you implement real-time sentiment analysis for streaming social media data?**

**Answer:**

**Definition:**
Real-time sentiment analysis processes continuous streams of social media data with low latency. Requires stream processing (Kafka, Flink), efficient models, and monitoring dashboards. Key challenges: throughput, latency, and handling bursty traffic.

**Architecture:**

| Component | Technology |
|-----------|------------|
| Data ingestion | Twitter/Reddit API, webhooks |
| Message queue | Kafka, RabbitMQ |
| Stream processing | Apache Flink, Spark Streaming |
| Model serving | FastAPI, TensorFlow Serving |
| Storage | Elasticsearch, TimescaleDB |
| Visualization | Grafana, custom dashboards |

**Python Code Example:**
```python
# Kafka consumer with real-time sentiment
from kafka import KafkaConsumer
from transformers import pipeline
import json

# Lightweight model for speed
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

consumer = KafkaConsumer(
    'social-media-stream',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

def process_stream():
    batch = []
    batch_size = 32  # Batch for efficiency
    
    for message in consumer:
        tweet = message.value
        batch.append(tweet['text'])
        
        if len(batch) >= batch_size:
            # Batch prediction for throughput
            results = sentiment_model(batch)
            
            for text, result in zip(batch, results):
                # Store/emit result
                emit_result({
                    'text': text,
                    'sentiment': result['label'],
                    'score': result['score'],
                    'timestamp': time.time()
                })
            
            batch = []

# FastAPI endpoint for model serving
from fastapi import FastAPI

app = FastAPI()

@app.post("/sentiment")
async def analyze_sentiment(text: str):
    result = sentiment_model(text)[0]
    return {"sentiment": result["label"], "confidence": result["score"]}
```

**Optimization Tips:**
- Use DistilBERT (2x faster than BERT)
- Batch predictions for throughput
- GPU inference for high volume
- Cache frequent phrases
- Use ONNX for faster inference

**Interview Tips:**
- Latency vs throughput tradeoff
- Smaller models (DistilBERT) for real-time
- Batch processing increases throughput significantly
- Monitor for drift as topics change

---

## Question 34
**What techniques reduce bias in sentiment analysis across demographic groups?**

**Answer:**

**Definition:**
Bias in sentiment analysis means different performance or sentiment scores across demographic groups (race, gender, age). Techniques include balanced data collection, debiasing embeddings, fairness constraints, and adversarial training.

**Types of Bias:**

| Bias Type | Example |
|-----------|--------|
| Label bias | "Angry" sentiment associated with certain names |
| Selection bias | Training data from one demographic |
| Embedding bias | Word2Vec associations (man:doctor, woman:nurse) |
| Representation bias | Underrepresented groups in training |

**Mitigation Techniques:**

| Technique | Description |
|-----------|-------------|
| Balanced datasets | Equal representation across groups |
| Counterfactual augmentation | Swap demographic markers, keep label |
| Debiased embeddings | Remove demographic dimensions |
| Fairness constraints | Equalize performance across groups |
| Adversarial debiasing | Train to be unpredictive of demographics |

**Python Code Example:**
```python
# Counterfactual data augmentation
def augment_counterfactual(text, label):
    """Swap demographic markers to reduce bias"""
    name_swaps = {
        'John': 'Lakisha', 'Michael': 'Jamal',
        'Emily': 'Aisha', 'Sarah': 'Latoya'
    }
    
    augmented = [text]  # Original
    for old_name, new_name in name_swaps.items():
        if old_name in text:
            augmented.append(text.replace(old_name, new_name))
    
    return [(t, label) for t in augmented]  # Same label for all

# Fairness evaluation
def evaluate_fairness(model, test_data, demographic_groups):
    """Check performance across demographic groups"""
    results = {}
    
    for group_name, group_data in demographic_groups.items():
        preds = model.predict([t for t, _ in group_data])
        labels = [l for _, l in group_data]
        
        results[group_name] = {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='macro')
        }
    
    # Check for disparities
    accuracies = [r['accuracy'] for r in results.values()]
    disparity = max(accuracies) - min(accuracies)
    print(f"Performance disparity: {disparity:.3f}")
    
    return results

# Adversarial debiasing (conceptual)
class DebiasingClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.sentiment_head = nn.Linear(768, 2)
        self.demographic_head = nn.Linear(768, num_demographics)  # Adversarial
    
    def forward(self, input_ids):
        features = self.encoder(input_ids).pooler_output
        sentiment = self.sentiment_head(features)
        demographic = self.demographic_head(GradientReversal(features))  # Reverse gradient
        return sentiment, demographic
```

**Interview Tips:**
- Test on demographic-balanced benchmarks
- Counterfactual augmentation is simple and effective
- Don't collect demographic data unnecessarily
- Bias can emerge from training data, embeddings, or model
- Equity (equal outcomes) vs equality (equal treatment) debate

---

## Question 35
**How do you handle sentiment analysis for domain-specific language (finance, healthcare)?**

**Answer:**

**Definition:**
Domain-specific sentiment requires understanding specialized vocabulary and context. "Bullish" is positive in finance, "aggressive" treatment may be positive in oncology. Solutions include domain-specific models, lexicons, and continued pre-training.

**Domain Challenges:**

| Domain | Challenge Example |
|--------|------------------|
| Finance | "Bearish outlook" = negative, "aggressive growth" = positive |
| Healthcare | "Aggressive treatment" = positive, "negative test" = positive |
| Legal | Technical jargon, neutral tone hides sentiment |
| Tech | "Disruptive" = positive, abbreviations |

**Approaches:**

| Approach | Description |
|----------|-------------|
| Domain-specific models | FinBERT, BioBERT, LegalBERT |
| Domain lexicons | Custom positive/negative word lists |
| Continued pre-training | MLM on domain corpus |
| Fine-tuning on domain data | Labeled domain-specific examples |

**Python Code Example:**
```python
# Approach 1: Use domain-specific pre-trained model
from transformers import pipeline

# Finance sentiment
finance_sentiment = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"  # Finance-specific
)

finance_text = "The company reported bearish quarterly results with declining revenues"
result = finance_sentiment(finance_text)
print(result)  # Correctly identifies as negative

# Approach 2: Domain-specific lexicon
finance_lexicon = {
    'positive': ['bullish', 'rally', 'surge', 'profit', 'growth', 'upgrade'],
    'negative': ['bearish', 'crash', 'decline', 'loss', 'downgrade', 'default']
}

def lexicon_sentiment(text, lexicon):
    words = text.lower().split()
    pos = sum(1 for w in words if w in lexicon['positive'])
    neg = sum(1 for w in words if w in lexicon['negative'])
    
    if pos > neg: return 'positive'
    elif neg > pos: return 'negative'
    return 'neutral'

# Approach 3: Continued pre-training for custom domain
from transformers import AutoModelForMaskedLM, Trainer

# Step 1: Continue MLM on domain corpus
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
# trainer.train(domain_mlm_dataset)
# model.save_pretrained('domain-adapted-bert')

# Step 2: Fine-tune for sentiment on domain data
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('domain-adapted-bert')
```

**Interview Tips:**
- FinBERT, BioBERT available off-the-shelf
- 10-50MB domain text for continued pre-training
- Lexicons fast for rule-based baseline
- Domain experts needed for labeling
- General models often fail on domain-specific text

---

## Question 36
**What is contrastive learning and how does it improve text representations?**

**Answer:**

**Definition:**
Contrastive learning learns representations by pulling similar examples together and pushing dissimilar ones apart in embedding space. For text, it creates better sentence embeddings without labeled data. Key methods: SimCLR for text, SimCSE, and SBERT.

**How It Works:**

| Step | Description |
|------|-------------|
| Positive pairs | Same text with different augmentations |
| Negative pairs | Different texts |
| Objective | Maximize similarity for positives, minimize for negatives |

**Contrastive Loss (InfoNCE):**
$$\mathcal{L} = -\log \frac{\exp(sim(z_i, z_j)/\tau)}{\sum_{k=1}^{N} \exp(sim(z_i, z_k)/\tau)}$$

where $z_i, z_j$ are positive pair embeddings, $\tau$ is temperature.

**Python Code Example:**
```python
# SimCSE: Simple contrastive learning for sentences
from sentence_transformers import SentenceTransformer

# SimCSE uses dropout as augmentation - same text, different dropout = positive pair
model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased')

# High quality sentence embeddings
sentences = [
    "The weather is beautiful today",
    "It's a lovely sunny day",
    "I need to buy groceries"
]

embeddings = model.encode(sentences)

# Similar sentences closer in embedding space
from sentence_transformers import util
scores = util.cos_sim(embeddings, embeddings)
print(scores)
# sentences 0 and 1 will have high similarity, 2 will be different

# Contrastive training loop (conceptual)
class ContrastiveModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.temperature = 0.05
    
    def forward(self, texts1, texts2):
        # texts1 and texts2 are positive pairs (same text, different augmentation)
        z1 = self.encoder(texts1)  # [batch, dim]
        z2 = self.encoder(texts2)  # [batch, dim]
        
        # Similarity matrix
        sim = torch.mm(z1, z2.T) / self.temperature
        
        # Labels: diagonal is positive pairs
        labels = torch.arange(len(texts1))
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim, labels)
        return loss
```

**Interview Tips:**
- SimCSE uses dropout for augmentation (no explicit augmentation needed)
- Contrastive learning reduces need for labeled data
- Temperature controls how hard negatives are weighted
- Used in Sentence-BERT, OpenAI embeddings
- Results in better clustering and semantic search

---

## Question 37
**How do you build intent classification systems for chatbots and virtual assistants?**

**Answer:**

**Definition:**
Intent classification identifies user's goal from their message (e.g., "book_flight", "check_balance", "get_weather"). It's the first step in dialog systems, routing queries to appropriate handlers. Challenges include similar intents, out-of-scope detection, and slot filling.

**Architecture:**

| Component | Purpose |
|-----------|--------|
| Intent classifier | Identify user goal |
| Slot extractor | Extract parameters (date, location) |
| OOD detector | Identify out-of-scope queries |
| Confidence threshold | Fallback when uncertain |

**Example Intents:**
```
User: "I want to fly to Paris next Friday"
Intent: book_flight
Slots: {destination: "Paris", date: "next Friday"}
```

**Python Code Example:**
```python
# Simple intent classifier with BERT
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class IntentClassifier:
    def __init__(self, model_path, intents):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.intents = intents
        self.threshold = 0.7  # Confidence threshold
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        
        top_prob, top_idx = probs.max(dim=0)
        intent = self.intents[top_idx.item()]
        
        if top_prob < self.threshold:
            return {'intent': 'fallback', 'confidence': float(top_prob)}
        
        return {'intent': intent, 'confidence': float(top_prob)}

# Using Rasa NLU (production framework)
# nlu.yml format:
# - intent: greet
#   examples: |
#     - hello
#     - hi
#     - hey there

# Joint intent + slot extraction with transformers
from transformers import pipeline

# Use token classification for slots
ner_model = pipeline("ner", model="dslim/bert-base-NER")

def extract_intent_and_slots(text):
    intent = intent_classifier.predict(text)
    entities = ner_model(text)  # Extract entities as slots
    
    return {
        'intent': intent['intent'],
        'confidence': intent['confidence'],
        'slots': entities
    }
```

**Interview Tips:**
- Rasa, Dialogflow, Amazon Lex are production frameworks
- Handle "I don't know" with OOD detection
- Few-shot learning with SetFit for new intents
- Slot filling is often joint task with intent
- Confidence calibration crucial for fallback

---

## Question 38
**What is knowledge distillation and how do you compress large classification models?**

**Answer:**

**Definition:**
Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model. Student learns to mimic teacher's soft outputs (probabilities), not just hard labels. Result: smaller, faster model with minimal accuracy loss.

**How It Works:**

| Step | Description |
|------|-------------|
| 1. Train teacher | Large model (BERT-large) on task |
| 2. Generate soft labels | Teacher's probability distributions |
| 3. Train student | Small model on soft labels + hard labels |

**Loss Function:**
$$\mathcal{L} = \alpha \cdot \mathcal{L}_{CE}(y, p_s) + (1-\alpha) \cdot \mathcal{L}_{KD}(p_t, p_s)$$

where $p_t$ = teacher probs, $p_s$ = student probs, temperature T softens distributions.

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft label loss (knowledge distillation)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# Training loop
teacher = BertForSequenceClassification.from_pretrained('bert-large-uncased-finetuned')
student = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

teacher.eval()  # Teacher in eval mode
loss_fn = DistillationLoss(temperature=4.0, alpha=0.5)

for batch in dataloader:
    with torch.no_grad():
        teacher_logits = teacher(**batch).logits
    
    student_logits = student(**batch).logits
    loss = loss_fn(student_logits, teacher_logits, batch['labels'])
    loss.backward()
    optimizer.step()

# Pre-distilled models available
from transformers import DistilBertModel
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# 40% smaller, 60% faster, 97% accuracy
```

**Interview Tips:**
- DistilBERT is pre-distilled, ready to use
- Temperature softens probability distribution
- Soft labels contain more information than hard labels
- Typical: 2-6x speedup with <3% accuracy drop
- Can distill ensembles into single model

---

## Question 39
**How do you handle out-of-scope/out-of-domain detection in classification systems?**

**Answer:**

**Definition:**
Out-of-scope (OOS) detection identifies inputs that don't belong to any known class. Critical for production systems to avoid false confident predictions. Techniques: confidence thresholding, dedicated OOS class, and outlier detection.

**Why It Matters:**
- Intent: "What's the capital of France?" → not banking intent
- Sentiment: Technical documentation → not applicable
- Classification: Inputs model wasn't trained for

**Approaches:**

| Approach | Description |
|----------|-------------|
| Confidence threshold | Reject if max probability < threshold |
| OOS class | Train with OOS examples |
| Entropy | High entropy = uncertain |
| Embedding distance | Far from training clusters = OOS |
| Energy-based | Low energy for in-distribution |

**Python Code Example:**
```python
import torch
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# Approach 1: Confidence thresholding
def predict_with_oos(model, text, threshold=0.7):
    logits = model(text)
    probs = torch.softmax(logits, dim=-1)
    max_prob = probs.max().item()
    
    if max_prob < threshold:
        return {'label': 'out_of_scope', 'confidence': max_prob}
    
    return {'label': classes[probs.argmax()], 'confidence': max_prob}

# Approach 2: Entropy-based
def entropy_oos_detector(model, text, entropy_threshold=1.5):
    logits = model(text)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
    
    if entropy > entropy_threshold:
        return True  # Out of scope (high uncertainty)
    return False

# Approach 3: Embedding distance
class EmbeddingOOSDetector:
    def __init__(self, encoder, train_texts):
        self.encoder = encoder
        # Store training embeddings
        self.train_embeddings = encoder.encode(train_texts)
        self.lof = LocalOutlierFactor(novelty=True)
        self.lof.fit(self.train_embeddings)
    
    def is_oos(self, text):
        embedding = self.encoder.encode([text])
        # -1 = outlier, 1 = inlier
        return self.lof.predict(embedding)[0] == -1

# Approach 4: Train explicit OOS class
# Add OOS examples to training data
training_data = [
    ("transfer money", "transfer_intent"),
    ("check balance", "balance_intent"),
    ("what's the weather", "out_of_scope"),  # OOS example
    ("tell me a joke", "out_of_scope"),
]
```

**Interview Tips:**
- Confidence alone is unreliable (models overconfident)
- Combine multiple signals: confidence + entropy + distance
- Collect OOS examples from production logs
- Calibration (Platt scaling) helps confidence thresholding
- Fallback strategy: "I don't understand, please rephrase"

---

## Question 40
**What is calibration in text classification and why does it matter for production systems?**

**Answer:**

**Definition:**
Calibration ensures model confidence matches actual accuracy. A well-calibrated model predicting 80% confidence should be correct 80% of the time. Neural networks are often overconfident. Calibration enables reliable thresholding and decision-making.

**Why It Matters:**
- Overconfident: predicts 95% confidence but only 70% accurate
- Underconfident: predicts 60% confidence but 90% accurate
- Production: need reliable "I'm not sure" signals

**Measuring Calibration:**

| Metric | Description |
|--------|-------------|
| Expected Calibration Error (ECE) | Avg gap between confidence and accuracy |
| Reliability diagram | Plot confidence vs accuracy |
| Brier score | Mean squared error of probabilities |

**Calibration Methods:**

| Method | Description |
|--------|-------------|
| Temperature scaling | Divide logits by learned T |
| Platt scaling | Logistic regression on logits |
| Isotonic regression | Non-parametric calibration |
| Label smoothing | Soft labels during training |

**Python Code Example:**
```python
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Measure calibration with reliability diagram
def plot_calibration(y_true, y_prob, n_bins=10):
    fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    plt.plot(mean_predicted, fraction_pos, 's-', label='Model')
    plt.plot([0, 1], [0, 1], '--', label='Perfect calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.legend()
    plt.title('Reliability Diagram')
    plt.show()

# Temperature scaling (most common for neural nets)
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature

def calibrate_temperature(model, val_loader):
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01)
    nll_loss = nn.CrossEntropyLoss()
    
    # Collect validation logits
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(**batch).logits
            logits_list.append(logits)
            labels_list.append(batch['labels'])
    
    all_logits = torch.cat(logits_list)
    all_labels = torch.cat(labels_list)
    
    # Optimize temperature
    def eval_temp():
        optimizer.zero_grad()
        loss = nll_loss(scaler(all_logits), all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_temp)
    print(f"Optimal temperature: {scaler.temperature.item():.3f}")
    return scaler
```

**Interview Tips:**
- Modern neural nets are typically overconfident
- Temperature scaling is simple and effective
- Calibrate on held-out validation set
- Critical for: medical diagnosis, autonomous systems
- Label smoothing during training also helps

---

## Question 41
**How do you combine topic modeling with classification for better feature engineering?**

**Answer:**

**Definition:**
Topic distributions from LDA/BERTopic can be used as additional features for classification. They capture document-level themes that complement word-level features. Particularly useful when training data is limited or topics are domain-relevant.

**Integration Strategies:**

| Strategy | Description |
|----------|-------------|
| Topic features | Append topic distribution to feature vector |
| Topic-weighted embeddings | Weight words by topic relevance |
| Multi-task learning | Joint topic + classification objective |
| Topic as regularization | Encourage topic-coherent representations |

**Python Code Example:**
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models import LdaModel
from scipy.sparse import hstack

# Step 1: Train topic model
lda = LdaModel(corpus, num_topics=20, id2word=dictionary)

# Step 2: Get topic distributions for each document
def get_topic_features(texts, lda, dictionary):
    topic_features = []
    for text in texts:
        bow = dictionary.doc2bow(text.split())
        topic_dist = lda.get_document_topics(bow, minimum_probability=0)
        features = [prob for _, prob in sorted(topic_dist)]
        topic_features.append(features)
    return np.array(topic_features)

# Step 3: Combine with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_features = vectorizer.fit_transform(texts)
topic_features = get_topic_features(texts, lda, dictionary)

# Concatenate features
combined_features = hstack([tfidf_features, topic_features])

# Train classifier
clf = LogisticRegression()
clf.fit(combined_features, labels)

# With neural networks: concatenate embeddings
class TopicAwareClassifier(nn.Module):
    def __init__(self, bert_dim, num_topics, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.topic_proj = nn.Linear(num_topics, 64)
        self.classifier = nn.Linear(bert_dim + 64, num_classes)
    
    def forward(self, input_ids, topic_dist):
        bert_out = self.bert(input_ids).pooler_output  # [batch, 768]
        topic_out = self.topic_proj(topic_dist)         # [batch, 64]
        combined = torch.cat([bert_out, topic_out], dim=-1)
        return self.classifier(combined)
```

**When This Helps:**
- Limited training data - topics provide extra signal
- Document-level features matter (e.g., categorizing reports)
- Topics align with classification categories
- Interpretability needed - topics are human-readable

**Interview Tips:**
- Topic features are especially useful for traditional ML
- BERT already captures topic-like information
- Topics add interpretability to predictions
- Multi-task learning can jointly optimize both objectives

---

## Question 42
**What are the differences between supervised, semi-supervised, and unsupervised topic models?**

**Answer:**

**Definition:**
Unsupervised topic models (LDA) discover topics from text alone. Supervised models (sLDA) use labels to guide topic discovery. Semi-supervised use partial labels. Choice depends on available labels and whether topics should align with known categories.

**Comparison:**

| Type | Labels | Use Case |
|------|--------|----------|
| Unsupervised (LDA) | None | Discover unknown themes |
| Supervised (sLDA) | Full | Topics that predict labels |
| Semi-supervised | Partial | Guide with some seed words/labels |
| Seeded | Keywords | Steer topics toward concepts |

**When to Use Each:**

| Scenario | Best Choice |
|----------|-------------|
| Exploratory analysis | Unsupervised |
| Topics must match categories | Supervised |
| Domain keywords known | Seeded/Semi-supervised |
| Predictive modeling | Supervised (sLDA) |

**Python Code Example:**
```python
# 1. Unsupervised LDA
from gensim.models import LdaModel

unsupervised_lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10
)
# Topics emerge purely from data

# 2. Seeded/Guided topic model
from guidedlda import GuidedLDA

seed_topics = {
    0: ['sports', 'game', 'team', 'player'],  # Sports topic
    1: ['politics', 'election', 'vote', 'government'],  # Politics
    2: ['technology', 'software', 'computer', 'app']  # Tech
}

guided_model = GuidedLDA(n_topics=10, random_state=42)
guided_model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)

# 3. Supervised LDA (sLDA) - conceptual
# Topics optimized to predict document labels
# Available in: slda package, lda2vec

# 4. BERTopic with supervision
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

# Semi-supervised: provide some labels
partial_labels = [0, 0, 1, -1, -1, 2, ...]  # -1 = unknown

topic_model = BERTopic()
topics, _ = topic_model.fit_transform(docs, y=partial_labels)
# Topics influenced by provided labels
```

**Interview Tips:**
- LDA is purely unsupervised - may not match business categories
- Seeded LDA steers toward known concepts
- sLDA learns topics predictive of response variable
- BERTopic supports semi-supervised via partial labels
- Semi-supervised useful when some labeled data available

---

## Question 43
**How do you handle spam/fake review detection using classification and sentiment analysis?**

**Answer:**

**Definition:**
Fake review detection identifies fraudulent reviews using text patterns, behavioral signals, and anomaly detection. Features include writing style, sentiment extremity, timing patterns, and reviewer behavior. Combines classification with domain knowledge.

**Types of Fake Reviews:**

| Type | Characteristics |
|------|----------------|
| Promotional spam | Excessive positive, product keywords |
| Negative attacks | Competitor sabotage |
| Paid reviews | Templated, generic |
| Review farms | Similar patterns across reviewers |

**Features for Detection:**

| Category | Features |
|----------|----------|
| Text | Sentiment extremity, length, readability |
| Linguistic | First-person pronouns, specificity |
| Behavioral | Posting frequency, rating deviation |
| Temporal | Burst patterns, review timing |
| Meta | Verified purchase, reviewer history |

**Python Code Example:**
```python
import numpy as np
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier

def extract_fake_review_features(review_text, review_meta):
    """Extract features for fake review detection"""
    blob = TextBlob(review_text)
    
    features = {
        # Text features
        'length': len(review_text),
        'word_count': len(review_text.split()),
        'avg_word_length': np.mean([len(w) for w in review_text.split()]),
        
        # Sentiment features
        'sentiment': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'sentiment_extreme': abs(blob.sentiment.polarity) > 0.8,
        
        # Linguistic features
        'exclamation_count': review_text.count('!'),
        'caps_ratio': sum(1 for c in review_text if c.isupper()) / len(review_text),
        'first_person': sum(1 for w in review_text.lower().split() if w in ['i', 'my', 'me']),
        
        # Meta features (from review_meta)
        'is_verified': review_meta.get('verified_purchase', False),
        'reviewer_review_count': review_meta.get('review_count', 0),
        'rating_deviation': abs(review_meta.get('rating', 3) - 3),
    }
    
    return features

# Train classifier
X = [extract_fake_review_features(r['text'], r['meta']) for r in reviews]
y = [r['is_fake'] for r in reviews]

clf = RandomForestClassifier(n_estimators=100)
clf.fit(pd.DataFrame(X), y)

# Neural approach with BERT
class FakeReviewDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.meta_proj = nn.Linear(10, 64)  # Meta features
        self.classifier = nn.Linear(768 + 64, 2)
    
    def forward(self, input_ids, meta_features):
        text_emb = self.bert(input_ids).pooler_output
        meta_emb = self.meta_proj(meta_features)
        combined = torch.cat([text_emb, meta_emb], dim=-1)
        return self.classifier(combined)
```

**Interview Tips:**
- Text alone insufficient - behavioral signals crucial
- Verified purchase is strong signal
- Look for burst patterns (many reviews in short time)
- Fake reviews often extreme sentiment
- Cross-product reviewer behavior helps

---

## Question 44
**What is document embedding and how do Doc2Vec, USE, and sentence-BERT work?**

**Answer:**

**Definition:**
Document embeddings map variable-length text to fixed-size vectors capturing semantic meaning. Unlike word embeddings, they represent entire documents/sentences. Key methods: Doc2Vec (extension of Word2Vec), USE (Universal Sentence Encoder), and SBERT (Sentence-BERT).

**Comparison:**

| Method | Approach | Strengths |
|--------|----------|----------|
| Doc2Vec | Paragraph vectors learned with words | Fast, no pre-training needed |
| USE | Transformer encoder | Multilingual, zero-shot |
| SBERT | BERT + siamese/triplet training | Best quality, semantic similarity |

**How They Work:**

**Doc2Vec:** Adds document ID as input alongside words, learns vector for each document.

**USE:** Transformer or DAN encoder trained on various NLU tasks.

**SBERT:** Fine-tunes BERT with siamese networks on NLI/STS data.

**Python Code Example:**
```python
# 1. Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [
    TaggedDocument(words=['machine', 'learning', 'is', 'great'], tags=['doc_1']),
    TaggedDocument(words=['deep', 'learning', 'neural', 'networks'], tags=['doc_2']),
]

model = Doc2Vec(documents, vector_size=100, epochs=20)

# Get document vector
vec = model.infer_vector(['machine', 'learning', 'models'])

# 2. Universal Sentence Encoder
import tensorflow_hub as hub

use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

sentences = ["This is a sentence", "Another sentence here"]
embeddings = use_model(sentences)  # [2, 512]

# 3. Sentence-BERT (best for semantic similarity)
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Machine learning is a subset of AI",
    "AI includes machine learning techniques",
    "I love pizza"
]
embeddings = sbert_model.encode(sentences)

# Compute similarity
from sentence_transformers import util
similarities = util.cos_sim(embeddings, embeddings)
print(similarities)
# First two sentences will have high similarity, third will be different

# Use for downstream tasks
from sklearn.linear_model import LogisticRegression

train_embeddings = sbert_model.encode(train_texts)
clf = LogisticRegression()
clf.fit(train_embeddings, train_labels)
```

**Interview Tips:**
- SBERT is current best for semantic similarity
- Doc2Vec still useful for large corpora with limited compute
- USE good for multilingual, quick deployment
- All produce fixed-size vectors for any text length
- Quality: SBERT > USE > Doc2Vec typically

---

## Question 45
**How do you implement online learning for classification models that update continuously?**

**Answer:**

**Definition:**
Online learning updates models incrementally as new data arrives, without retraining from scratch. Essential for streaming data, concept drift adaptation, and when storage is limited. Techniques: partial_fit, incremental gradient descent, and reservoir sampling.

**Use Cases:**
- Real-time sentiment monitoring
- Spam filters adapting to new patterns
- Personalization from user feedback
- Concept drift handling

**Approaches:**

| Approach | Description |
|----------|-------------|
| Partial fit | Update with mini-batches |
| SGD online | Stochastic gradient updates |
| Reservoir sampling | Maintain representative sample |
| Sliding window | Train on recent data only |

**Python Code Example:**
```python
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

# HashingVectorizer for online learning (stateless)
vectorizer = HashingVectorizer(n_features=10000)

# SGDClassifier supports partial_fit
clf = SGDClassifier(loss='log_loss')  # Logistic regression

classes = np.array([0, 1])  # Must specify classes upfront

def process_stream(stream):
    batch_size = 100
    batch_texts, batch_labels = [], []
    
    for text, label in stream:
        batch_texts.append(text)
        batch_labels.append(label)
        
        if len(batch_texts) >= batch_size:
            X_batch = vectorizer.transform(batch_texts)
            clf.partial_fit(X_batch, batch_labels, classes=classes)
            
            batch_texts, batch_labels = [], []
            print(f"Updated model with {batch_size} samples")

# For neural networks: continuous fine-tuning
class OnlineNeuralClassifier:
    def __init__(self, model, learning_rate=1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def update(self, texts, labels):
        self.model.train()
        
        # Low learning rate for stability
        inputs = tokenizer(texts, return_tensors='pt', padding=True)
        labels = torch.tensor(labels)
        
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()

# Elastic weight consolidation (prevent forgetting)
# Regularize to stay close to previous weights
```

**Interview Tips:**
- Use HashingVectorizer (stateless) for online vectorization
- SGDClassifier.partial_fit() for sklearn online learning
- Low learning rate prevents catastrophic forgetting
- Consider elastic weight consolidation for neural nets
- Validate periodically on held-out data

---

## Question 46
**What role does attention mechanism play in understanding which words drive classification?**

**Answer:**

**Definition:**
Attention weights show how much the model "focuses" on each token when making predictions. In classification, attention can indicate which words influenced the decision. However, attention as explanation is debated - weights don't always reflect true importance.

**Attention in Classification:**

| Layer | Role |
|-------|------|
| Self-attention | Token interactions, context building |
| CLS attention | Which tokens inform [CLS] representation |
| Classification head | Final prediction from [CLS] |

**Visualization:**

**Python Code Example:**
```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased-finetuned-sst-2-english')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def visualize_attention(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs, output_attentions=True)
    
    # Get last layer attention
    attention = outputs.attentions[-1].squeeze()  # [heads, seq, seq]
    
    # Average over heads, get CLS row (what CLS attends to)
    cls_attention = attention.mean(dim=0)[0]  # [seq_len]
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Visualize
    plt.figure(figsize=(10, 3))
    plt.bar(range(len(tokens)), cls_attention.detach().numpy())
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.title(f"Prediction: {outputs.logits.argmax().item()}")
    plt.ylabel('Attention weight')
    plt.show()
    
    return list(zip(tokens, cls_attention.tolist()))

# Example
result = visualize_attention("This movie was absolutely terrible and boring")
# High attention on "terrible", "boring"

# Attention rollout (more accurate than raw attention)
def attention_rollout(attentions):
    """Compute attention flow through layers"""
    rollout = torch.eye(attentions[0].size(-1))
    
    for attention in attentions:
        attention_heads_fused = attention.mean(dim=1).squeeze()
        attention_heads_fused = attention_heads_fused + torch.eye(attention_heads_fused.size(-1))
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(attention_heads_fused, rollout)
    
    return rollout
```

**Important Caveat:**
Research shows attention weights don't always correlate with feature importance. Use SHAP/LIME for more faithful explanations.

**Interview Tips:**
- Attention provides interpretability but isn't perfect explanation
- CLS token attention shows what model "looked at"
- Attention rollout tracks information flow through layers
- SHAP/LIME are more faithful explanation methods
- Different heads learn different patterns

---

## Question 47
**How do you benchmark and compare different text classification approaches fairly?**

**Answer:**

**Definition:**
Fair benchmarking requires consistent data splits, proper cross-validation, statistical significance testing, and reporting multiple metrics. Avoid cherry-picking results or unfair comparisons (different data, hyperparameter effort).

**Best Practices:**

| Practice | Description |
|----------|-------------|
| Fixed random seeds | Reproducibility |
| Same train/val/test split | Fair comparison |
| Cross-validation | Reduce variance |
| Multiple metrics | Accuracy, F1, AUC |
| Statistical tests | Significance of differences |
| Hyperparameter budget | Equal tuning effort |

**Python Code Example:**
```python
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats

def benchmark_models(models, X, y, n_folds=5, seed=42):
    """Fair comparison of multiple models"""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = {}
    
    for name, model in models.items():
        # Cross-validation scores
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
    
    return results

def statistical_comparison(results, model1, model2):
    """Paired t-test for significance"""
    scores1 = results[model1]['scores']
    scores2 = results[model2]['scores']
    
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    print(f"{model1} vs {model2}: t={t_stat:.3f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        winner = model1 if scores1.mean() > scores2.mean() else model2
        print(f"Significant difference (p<0.05). {winner} is better.")
    else:
        print("No significant difference.")

# Run benchmark
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(),
    'BERT': bert_classifier  # Custom wrapper
}

results = benchmark_models(models, X_tfidf, y)
statistical_comparison(results, 'SVM', 'BERT')

# Report table
def generate_results_table(results):
    print("\n| Model | F1 (macro) | Std |")
    print("|-------|------------|-----|")
    for name, res in sorted(results.items(), key=lambda x: -x[1]['mean']):
        print(f"| {name} | {res['mean']:.4f} | {res['std']:.4f} |")
```

**Interview Tips:**
- Always report std and sample size
- McNemar's test for classifier comparison on same test set
- Report training time and inference speed
- Consider data leakage in preprocessing
- State computational resources used

---

## Question 48
**What is prompt engineering for text classification with large language models?**

**Answer:**

**Definition:**
Prompt engineering crafts input prompts that guide LLMs to perform classification without fine-tuning. Techniques include zero-shot prompts, few-shot examples, chain-of-thought reasoning, and structured output formats. Critical for using GPT-4, Claude, etc. for classification.

**Prompt Patterns:**

| Pattern | Description |
|---------|-------------|
| Zero-shot | Direct instruction, no examples |
| Few-shot | Include labeled examples |
| Chain-of-thought | "Let's think step by step" |
| Output format | "Respond with only: positive/negative" |

**Python Code Example:**
```python
import openai

# 1. Zero-shot classification
def zero_shot_classify(text, labels):
    prompt = f"""Classify the following text into one of these categories: {', '.join(labels)}

Text: {text}

Respond with only the category name, nothing else."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # Deterministic
    )
    return response.choices[0].message.content.strip()

# 2. Few-shot classification
def few_shot_classify(text, examples, labels):
    prompt = f"Classify text as {' or '.join(labels)}.\n\nExamples:\n"
    
    for ex_text, ex_label in examples:
        prompt += f"Text: {ex_text}\nCategory: {ex_label}\n\n"
    
    prompt += f"Text: {text}\nCategory:"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# 3. Chain-of-thought for complex classification
def cot_classify(text):
    prompt = f"""Analyze the sentiment of this text step by step:

Text: "{text}"

Step 1: Identify key sentiment words
Step 2: Consider context and tone
Step 3: Determine overall sentiment

Final answer (positive/negative/neutral):"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 4. Structured output (JSON)
def structured_classify(text):
    prompt = f"""Classify this review and extract aspects.
Respond in JSON format only.

Text: "{text}"

Format:
{{
    "sentiment": "positive/negative/neutral",
    "confidence": 0.0-1.0,
    "aspects": ["aspect1", "aspect2"]
}}"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

**Interview Tips:**
- Temperature=0 for consistent classification
- Few-shot examples should be diverse and representative
- Clear output format instructions reduce parsing errors
- Chain-of-thought improves complex reasoning
- Cost: GPT-4 expensive at scale vs fine-tuned small models

---

## Question 49
**How do you handle privacy-preserving text classification for sensitive content?**

**Answer:**

**Definition:**
Privacy-preserving NLP protects sensitive information (PII, medical records, financial data) during classification. Techniques include on-device processing, differential privacy, federated learning, and data anonymization.

**Privacy Concerns:**

| Concern | Example |
|---------|--------|
| Training data leakage | Model memorizes SSNs |
| Inference privacy | Cloud sees sensitive text |
| PII exposure | Names, addresses in predictions |
| Model extraction | Adversary steals model |

**Techniques:**

| Technique | Description |
|-----------|-------------|
| On-device inference | No data leaves device |
| Differential privacy | Add noise to gradients |
| Federated learning | Train on distributed data |
| Data anonymization | Remove/mask PII before processing |
| Secure enclaves | Hardware-protected computation |

**Python Code Example:**
```python
# 1. PII removal before classification
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def anonymize_text(text):
    doc = nlp(text)
    
    # Replace named entities
    anonymized = text
    for ent in reversed(doc.ents):  # Reverse to maintain positions
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EMAIL']:
            anonymized = anonymized[:ent.start_char] + f"[{ent.label_}]" + anonymized[ent.end_char:]
    
    # Regex patterns for structured data
    anonymized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', anonymized)  # SSN
    anonymized = re.sub(r'\b\d{16}\b', '[CREDIT_CARD]', anonymized)  # Credit card
    anonymized = re.sub(r'\S+@\S+', '[EMAIL]', anonymized)  # Email
    
    return anonymized

# 2. Differential privacy training
from opacus import PrivacyEngine

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = torch.optim.Adam(model.parameters())

# Wrap with privacy engine
privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=1.0,  # Privacy-accuracy tradeoff
    max_grad_norm=1.0
)

# Training with DP guarantees
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Check privacy budget
epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"Privacy budget: ε={epsilon:.2f}")

# 3. On-device inference with ONNX
import onnxruntime as ort

# Export to ONNX for edge deployment
torch.onnx.export(model, dummy_input, "model.onnx")

# Run on device (no cloud)
session = ort.InferenceSession("model.onnx")
result = session.run(None, {"input_ids": input_ids})
```

**Interview Tips:**
- Always anonymize PII before cloud processing
- Differential privacy has accuracy cost
- On-device inference best for full privacy
- GDPR/HIPAA compliance often requires these techniques
- Consider threat model: who are you protecting against?

---

## Question 50
**What are best practices for deploying text classification models in production (latency, throughput, monitoring)?**

**Answer:**

**Definition:**
Production deployment requires optimizing for latency, throughput, reliability, and observability. Key practices: model optimization (quantization, distillation), proper serving infrastructure, monitoring, and graceful degradation.

**Key Considerations:**

| Aspect | Optimization |
|--------|-------------|
| Latency | Smaller models, quantization, batching |
| Throughput | GPU inference, horizontal scaling |
| Reliability | Health checks, fallbacks, retries |
| Monitoring | Drift detection, performance tracking |

**Optimization Techniques:**

| Technique | Speedup | Accuracy Impact |
|-----------|---------|----------------|
| Distillation | 2-6x | 1-3% drop |
| Quantization (INT8) | 2-4x | <1% drop |
| ONNX Runtime | 2-3x | None |
| TensorRT | 3-5x | None |
| Batching | 2-10x throughput | None |

**Python Code Example:**
```python
# 1. Model optimization with ONNX
from transformers import AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

# Convert to ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    export=True
)
# 2-3x faster inference

# 2. Quantization
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# 2-4x faster, smaller model size

# 3. FastAPI serving with batching
from fastapi import FastAPI
import asyncio

app = FastAPI()
batch_queue = asyncio.Queue()

async def batch_processor():
    while True:
        batch = []
        while len(batch) < 32:  # Max batch size
            try:
                item = await asyncio.wait_for(batch_queue.get(), timeout=0.01)
                batch.append(item)
            except asyncio.TimeoutError:
                break
        
        if batch:
            texts, futures = zip(*batch)
            results = model.predict(list(texts))  # Batch inference
            for future, result in zip(futures, results):
                future.set_result(result)

@app.post("/predict")
async def predict(text: str):
    future = asyncio.get_event_loop().create_future()
    await batch_queue.put((text, future))
    return await future

# 4. Monitoring
import prometheus_client

latency_histogram = prometheus_client.Histogram(
    'prediction_latency_seconds', 'Prediction latency'
)
confidence_histogram = prometheus_client.Histogram(
    'prediction_confidence', 'Model confidence'
)

def monitored_predict(text):
    start = time.time()
    result = model.predict(text)
    
    latency_histogram.observe(time.time() - start)
    confidence_histogram.observe(result['confidence'])
    
    return result
```

**Production Checklist:**
- [ ] Model versioning (MLflow, DVC)
- [ ] A/B testing infrastructure
- [ ] Fallback to simpler model if main fails
- [ ] Input validation and sanitization
- [ ] Rate limiting
- [ ] Logging and alerting
- [ ] Drift monitoring dashboard

**Interview Tips:**
- DistilBERT is often best latency/accuracy tradeoff
- Batching crucial for GPU utilization
- ONNX Runtime is quick win for optimization
- Monitor both model metrics and system metrics
- Plan for model updates without downtime

---
