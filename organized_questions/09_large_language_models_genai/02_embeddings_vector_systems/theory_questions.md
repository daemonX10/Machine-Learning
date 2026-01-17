# Embeddings & Vector Systems - Theory Questions

## Embedding Fundamentals

### Question 1
**How do you choose between Word2Vec, GloVe, FastText, and BERT embeddings for your NLP task?**

**Answer:**

**Definition:**
Choose based on: **task requirements** (static vs contextual), **computational budget**, **vocabulary coverage**, and **accuracy needs**. BERT provides contextual embeddings for highest accuracy; FastText handles OOV well; Word2Vec/GloVe are efficient for simpler tasks or constrained environments.

**Decision Framework:**

| Requirement | Best Choice | Reason |
|-------------|-------------|--------|
| **Highest accuracy** | BERT/Sentence-BERT | Contextualized representations |
| **Real-time inference** | Word2Vec/GloVe | Fast lookup |
| **OOV handling** | FastText | Subword embeddings |
| **Limited compute** | Word2Vec/GloVe | No model inference |
| **Morphological languages** | FastText | Captures word structure |
| **Domain-specific** | Fine-tuned BERT | Transfer learning |

**Embedding Comparison:**

| Aspect | Word2Vec | GloVe | FastText | BERT |
|--------|----------|-------|----------|------|
| **Type** | Static | Static | Static | Contextual |
| **Training** | Predictive | Co-occurrence | Predictive+subword | Masked LM |
| **OOV** | ❌ | ❌ | ✓ (subwords) | ✓ (tokenizer) |
| **Speed** | Fast | Fast | Fast | Slow |
| **Memory** | Low | Low | Medium | High |
| **Quality** | Good | Good | Better | Best |
| **Dimension** | 100-300 | 100-300 | 100-300 | 768-1024 |

**Python Code Example:**
```python
import numpy as np
from typing import List, Dict

class EmbeddingSelector:
    """Help select appropriate embedding for task"""
    
    def __init__(self):
        self.requirements = {}
    
    def analyze_requirements(self, 
                            need_contextual: bool,
                            latency_ms: float,
                            memory_mb: float,
                            oov_percentage: float,
                            accuracy_priority: str) -> str:
        """Recommend embedding based on requirements"""
        
        recommendations = []
        
        # Contextual requirement
        if need_contextual:
            recommendations.append(('BERT', 5))
        else:
            recommendations.append(('Word2Vec', 3))
            recommendations.append(('GloVe', 3))
        
        # Latency constraint
        if latency_ms < 10:
            recommendations.append(('Word2Vec', 4))
            recommendations.append(('GloVe', 4))
            recommendations.append(('FastText', 3))
        elif latency_ms < 100:
            recommendations.append(('FastText', 3))
            recommendations.append(('DistilBERT', 2))
        else:
            recommendations.append(('BERT', 3))
        
        # OOV handling
        if oov_percentage > 10:
            recommendations.append(('FastText', 5))
            recommendations.append(('BERT', 3))
        
        # Aggregate scores
        scores = {}
        for model, score in recommendations:
            scores[model] = scores.get(model, 0) + score
        
        return max(scores.keys(), key=lambda x: scores[x])

# Loading different embeddings
def load_word2vec(path: str):
    from gensim.models import KeyedVectors
    return KeyedVectors.load_word2vec_format(path, binary=True)

def load_glove(path: str, dim: int = 300):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def load_fasttext(path: str):
    import fasttext
    return fasttext.load_model(path)

def load_bert(model_name: str = 'bert-base-uncased'):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Unified embedding interface
class UnifiedEmbedder:
    def __init__(self, embedding_type: str, model_path: str = None):
        self.embedding_type = embedding_type
        self._load_model(model_path)
    
    def _load_model(self, path):
        if self.embedding_type == 'bert':
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')
        elif self.embedding_type == 'word2vec':
            from gensim.models import KeyedVectors
            self.model = KeyedVectors.load_word2vec_format(path, binary=True)
        # ... other loaders
    
    def embed(self, text: str) -> np.ndarray:
        if self.embedding_type == 'bert':
            return self._bert_embed(text)
        elif self.embedding_type == 'word2vec':
            return self._static_embed(text)
    
    def _bert_embed(self, text: str) -> np.ndarray:
        import torch
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling
        return outputs.last_hidden_state.mean(dim=1).numpy()[0]
    
    def _static_embed(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vectors = [self.model[w] for w in words if w in self.model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(300)
```

**Interview Tips:**
- BERT for accuracy-critical tasks; static embeddings for speed
- FastText is best for morphologically rich languages (German, Turkish)
- Consider Sentence-BERT for sentence-level similarity tasks
- Production often uses static embeddings + BERT fallback for important queries

---

### Question 2
**What are the computational trade-offs between static embeddings (Word2Vec) versus contextualized embeddings (BERT)?**

**Answer:**

**Definition:**
Static embeddings (Word2Vec, GloVe) use **precomputed lookup tables** (O(1) per word), while contextualized embeddings (BERT) require **model inference** for each input (O(n·d²) per sentence). Trade-off: static is 100-1000x faster but loses context; BERT captures polysemy and context at high compute cost.

**Computational Comparison:**

| Aspect | Static (Word2Vec) | Contextualized (BERT) |
|--------|-------------------|----------------------|
| **Lookup time** | O(1) | O(n·L·d²) |
| **Memory (vocab 100K)** | ~120MB | ~440MB (model) |
| **Latency per sentence** | <1ms | 10-100ms |
| **Batch throughput** | ~100K/sec | ~100-1000/sec |
| **GPU required** | No | Recommended |
| **Handles polysemy** | No | Yes |

**Cost Analysis:**

| Metric | Word2Vec | BERT-base | BERT-large |
|--------|----------|-----------|------------|
| **Parameters** | 0 (lookup) | 110M | 340M |
| **FLOPS/token** | ~0 | ~22B | ~84B |
| **Memory/batch** | O(batch·dim) | O(batch·seq·dim) | O(batch·seq·dim) |
| **GPU memory** | 0 | 1-2GB | 4-8GB |

**Python Code Example:**
```python
import numpy as np
import time
from typing import List

class EmbeddingBenchmark:
    """Compare embedding computational costs"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_static(self, embeddings: dict, texts: List[str], 
                         num_runs: int = 100) -> dict:
        """Benchmark static embedding lookup"""
        
        def embed_text(text):
            words = text.lower().split()
            vectors = [embeddings[w] for w in words if w in embeddings]
            return np.mean(vectors, axis=0) if vectors else np.zeros(300)
        
        # Warmup
        for text in texts[:10]:
            embed_text(text)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(num_runs):
            for text in texts:
                embed_text(text)
        elapsed = time.perf_counter() - start
        
        return {
            'total_time': elapsed,
            'per_text_ms': (elapsed / (num_runs * len(texts))) * 1000,
            'throughput': (num_runs * len(texts)) / elapsed
        }
    
    def benchmark_bert(self, model, tokenizer, texts: List[str],
                       num_runs: int = 10) -> dict:
        """Benchmark BERT embedding extraction"""
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Warmup
        inputs = tokenizer(texts[:5], return_tensors='pt', 
                          padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_runs):
            inputs = tokenizer(texts, return_tensors='pt',
                              padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        return {
            'total_time': elapsed,
            'per_text_ms': (elapsed / (num_runs * len(texts))) * 1000,
            'throughput': (num_runs * len(texts)) / elapsed,
            'device': str(device)
        }
    
    def compare(self, static_results: dict, bert_results: dict) -> dict:
        """Compare static vs BERT performance"""
        speedup = bert_results['per_text_ms'] / static_results['per_text_ms']
        
        return {
            'static_latency_ms': static_results['per_text_ms'],
            'bert_latency_ms': bert_results['per_text_ms'],
            'speedup_factor': speedup,
            'recommendation': 'static' if speedup > 10 else 'bert'
        }

# Memory estimation
def estimate_memory(embedding_type: str, vocab_size: int = 100000,
                    dim: int = 300, batch_size: int = 32,
                    seq_len: int = 128) -> dict:
    """Estimate memory requirements"""
    
    if embedding_type == 'static':
        # Just the embedding matrix
        embedding_memory = vocab_size * dim * 4  # float32
        inference_memory = batch_size * dim * 4
        
        return {
            'embedding_mb': embedding_memory / 1e6,
            'inference_mb': inference_memory / 1e6,
            'total_mb': (embedding_memory + inference_memory) / 1e6
        }
    
    elif embedding_type == 'bert-base':
        # Model parameters + activations
        params = 110_000_000
        model_memory = params * 4  # float32
        
        # Activations for inference
        hidden_size = 768
        num_layers = 12
        activations = batch_size * seq_len * hidden_size * num_layers * 4
        
        # Attention matrices
        attention = batch_size * 12 * seq_len * seq_len * num_layers * 4
        
        return {
            'model_mb': model_memory / 1e6,
            'activations_mb': activations / 1e6,
            'attention_mb': attention / 1e6,
            'total_mb': (model_memory + activations + attention) / 1e6
        }
```

**Interview Tips:**
- Static embeddings: best for high-throughput, low-latency systems
- BERT: needed when context matters (polysemy, sentiment)
- Hybrid: pre-filter with static, re-rank with BERT
- DistilBERT offers 60% speedup with minimal quality loss

---

### Question 3
**How do you handle out-of-vocabulary (OOV) words with different embedding approaches?**

**Answer:**

**Definition:**
OOV handling strategies vary by embedding type: **FastText** uses subword n-grams (built-in), **BERT** uses WordPiece/BPE tokenization (splits unknown words), static embeddings require fallbacks like **random initialization**, **zero vectors**, or **character-level models**. Goal: maintain semantic meaning for unseen words.

**OOV Handling by Embedding Type:**

| Embedding | OOV Strategy | Quality |
|-----------|--------------|---------|
| **Word2Vec** | None (crashes or zero) | Poor |
| **GloVe** | None (crashes or zero) | Poor |
| **FastText** | Subword n-grams | Good |
| **BERT** | WordPiece tokenization | Excellent |
| **Custom** | Character CNN/RNN | Variable |

**Strategies for Static Embeddings:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Zero vector** | Return zeros | Simple fallback |
| **Random init** | Random vector | May hurt similarity |
| **Similar word** | Edit distance match | Typos |
| **Morphological** | Prefix/suffix matching | Related words |
| **Character model** | CharCNN/CharRNN | Any OOV |
| **Subword hashing** | Hash subwords to buckets | Scalable |

**Python Code Example:**
```python
import numpy as np
from collections import defaultdict
import re
from typing import Optional

class OOVHandler:
    """Handle out-of-vocabulary words for static embeddings"""
    
    def __init__(self, embeddings: dict, dim: int = 300):
        self.embeddings = embeddings
        self.dim = dim
        self.vocab = set(embeddings.keys())
        
        # Precompute prefix/suffix indices
        self._build_affix_index()
    
    def _build_affix_index(self, prefix_len: int = 3, suffix_len: int = 3):
        """Build index for morphological matching"""
        self.prefix_index = defaultdict(list)
        self.suffix_index = defaultdict(list)
        
        for word in self.vocab:
            if len(word) >= prefix_len:
                self.prefix_index[word[:prefix_len]].append(word)
            if len(word) >= suffix_len:
                self.suffix_index[word[-suffix_len:]].append(word)
    
    def get_embedding(self, word: str, strategy: str = 'subword') -> np.ndarray:
        """Get embedding with OOV handling"""
        
        word_lower = word.lower()
        
        # Known word
        if word_lower in self.embeddings:
            return self.embeddings[word_lower]
        
        # Apply OOV strategy
        if strategy == 'zero':
            return np.zeros(self.dim)
        
        elif strategy == 'random':
            np.random.seed(hash(word) % (2**32))
            return np.random.randn(self.dim) * 0.1
        
        elif strategy == 'similar':
            return self._find_similar_word(word_lower)
        
        elif strategy == 'morphological':
            return self._morphological_embedding(word_lower)
        
        elif strategy == 'subword':
            return self._subword_embedding(word_lower)
        
        return np.zeros(self.dim)
    
    def _find_similar_word(self, word: str) -> np.ndarray:
        """Find embedding of similar word (edit distance)"""
        
        # Simple approach: check words with same prefix
        if len(word) >= 3:
            prefix = word[:3]
            candidates = self.prefix_index.get(prefix, [])
            
            if candidates:
                # Return embedding of shortest candidate
                best = min(candidates, key=len)
                return self.embeddings[best]
        
        return np.zeros(self.dim)
    
    def _morphological_embedding(self, word: str) -> np.ndarray:
        """Combine prefix and suffix word embeddings"""
        
        vectors = []
        
        # Check prefix matches
        if len(word) >= 3:
            prefix = word[:3]
            prefix_words = self.prefix_index.get(prefix, [])
            if prefix_words:
                vectors.append(self.embeddings[prefix_words[0]])
        
        # Check suffix matches
        if len(word) >= 3:
            suffix = word[-3:]
            suffix_words = self.suffix_index.get(suffix, [])
            if suffix_words:
                vectors.append(self.embeddings[suffix_words[0]])
        
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.dim)
    
    def _subword_embedding(self, word: str, n: int = 3) -> np.ndarray:
        """Average embeddings of character n-grams (FastText-style)"""
        
        # Generate n-grams
        padded = f"<{word}>"
        ngrams = [padded[i:i+n] for i in range(len(padded) - n + 1)]
        
        # Hash n-grams to vocabulary words (simple approach)
        vectors = []
        for ngram in ngrams:
            # Check if ngram appears in any vocab word
            for vocab_word in list(self.vocab)[:1000]:  # Limit for speed
                if ngram in vocab_word:
                    vectors.append(self.embeddings[vocab_word])
                    break
        
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.dim)

# FastText (built-in OOV handling)
def fasttext_oov_demo():
    """FastText handles OOV via subword n-grams"""
    import fasttext
    
    # FastText automatically handles OOV
    model = fasttext.load_model('wiki.en.bin')
    
    # Works for any word, even made-up ones
    known = model.get_word_vector('computer')
    unknown = model.get_word_vector('computering')  # Made-up word
    misspelled = model.get_word_vector('computr')   # Typo
    
    return known, unknown, misspelled

# BERT (tokenizer handles OOV)
def bert_oov_demo():
    """BERT uses WordPiece for OOV"""
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Known word
    known = tokenizer.tokenize('computer')  # ['computer']
    
    # Unknown word gets split
    unknown = tokenizer.tokenize('electronegativity')  
    # ['electron', '##ega', '##tivity']
    
    # Complete gibberish
    gibberish = tokenizer.tokenize('xyzqwerty')
    # ['xyz', '##q', '##wer', '##ty']
    
    return known, unknown, gibberish
```

**Interview Tips:**
- FastText is best for production OOV handling in static embeddings
- BERT's subword tokenization rarely has true OOV issues
- For custom solutions, character-level models are most robust
- Track OOV rate in production — high rates indicate vocabulary drift

---

### Question 4
**When would you use subword embeddings (FastText) versus word-level embeddings for morphologically rich languages?**

**Answer:**

**Definition:**
Use **subword embeddings (FastText)** for morphologically rich languages (German, Turkish, Finnish, Arabic) where words have many inflected forms. Subword captures shared morphemes, handles OOV from affixes, and reduces vocabulary size. Word-level embeddings fail when vocabulary explodes due to morphological variation.

**When to Use Subword:**

| Scenario | Subword (FastText) | Word-level |
|----------|-------------------|------------|
| **Agglutinative languages** | ✓ Best | ✗ Poor |
| **Compound words** | ✓ (German) | ✗ |
| **Rich inflection** | ✓ | ✗ |
| **Limited training data** | ✓ Better generalization | ✗ |
| **Typo tolerance** | ✓ Partial matching | ✗ |
| **Fixed vocabulary** | Either | ✓ Simpler |

**Morphological Complexity by Language:**

| Language | Type | Unique Forms | Subword Benefit |
|----------|------|--------------|-----------------|
| **Turkish** | Agglutinative | Very High | Critical |
| **Finnish** | Agglutinative | Very High | Critical |
| **German** | Fusional + Compounds | High | Strong |
| **Arabic** | Root-based | High | Strong |
| **Russian** | Fusional | Medium | Moderate |
| **English** | Analytic | Low | Mild |
| **Chinese** | Isolating | Low | Word-level OK |

**Python Code Example:**
```python
import numpy as np
from collections import Counter

class MorphologicalAnalyzer:
    """Analyze vocabulary for morphological complexity"""
    
    def __init__(self, corpus_words: list):
        self.words = corpus_words
        self.word_freq = Counter(corpus_words)
    
    def morphological_richness(self) -> dict:
        """Calculate metrics indicating morphological complexity"""
        
        unique_words = len(set(self.words))
        total_words = len(self.words)
        
        # Type-token ratio (higher = more unique forms)
        ttr = unique_words / total_words
        
        # Average word length (longer = likely agglutinative)
        avg_length = np.mean([len(w) for w in self.words])
        
        # Hapax legomena ratio (words appearing once)
        hapax = sum(1 for w, c in self.word_freq.items() if c == 1)
        hapax_ratio = hapax / unique_words
        
        return {
            'type_token_ratio': ttr,
            'avg_word_length': avg_length,
            'hapax_ratio': hapax_ratio,
            'unique_words': unique_words,
            'recommend_subword': ttr > 0.3 or avg_length > 8
        }
    
    def estimate_vocabulary_explosion(self, sample_sizes: list) -> dict:
        """Estimate how vocabulary grows with corpus size"""
        
        results = {}
        for size in sample_sizes:
            sample = self.words[:size]
            unique = len(set(sample))
            results[size] = {
                'unique_words': unique,
                'growth_rate': unique / size
            }
        
        # If vocabulary keeps growing linearly, need subword
        return results

# FastText for morphological languages
def train_fasttext_for_morphological(corpus_file: str, 
                                      language: str) -> dict:
    """Configure FastText for morphologically rich languages"""
    import fasttext
    
    # Language-specific configurations
    configs = {
        'turkish': {
            'minn': 2,       # Min n-gram
            'maxn': 6,       # Max n-gram (longer for agglutinative)
            'dim': 300,
            'ws': 5          # Context window
        },
        'german': {
            'minn': 3,
            'maxn': 6,
            'dim': 300,
            'ws': 5
        },
        'english': {
            'minn': 3,
            'maxn': 5,
            'dim': 300,
            'ws': 5
        }
    }
    
    config = configs.get(language, configs['english'])
    
    model = fasttext.train_unsupervised(
        corpus_file,
        model='skipgram',
        **config
    )
    
    return model

# Demonstrating subword advantage
def subword_advantage_demo():
    """Show how subword handles morphological variants"""
    
    # German compound words
    german_examples = [
        'Handschuh',           # glove (hand + shoe)
        'Handschuhe',          # gloves
        'Lederhandschuh',      # leather glove
        'Lederhandschuhe',     # leather gloves
    ]
    
    # Turkish agglutination
    turkish_examples = [
        'ev',                   # house
        'evler',                # houses
        'evlerimiz',            # our houses
        'evlerimizde',          # in our houses
        'evlerimizdekiler',     # those in our houses
    ]
    
    # With FastText, all variants share subword components
    # With word-level, each is a separate embedding (or OOV)
    
    return {
        'german_unique_forms': len(set(german_examples)),
        'turkish_unique_forms': len(set(turkish_examples)),
        'subword_coverage': 'All forms represented via shared n-grams',
        'word_level_coverage': 'Each form needs separate training data'
    }

# Comparing approaches
def compare_approaches(vocab_size: int, corpus_size: int) -> dict:
    """Compare storage and coverage"""
    
    word_level = {
        'storage': vocab_size * 300 * 4,  # bytes
        'oov_rate': max(0, (vocab_size - 100000) / vocab_size),
        'training_data_needed': vocab_size * 5  # rough estimate
    }
    
    subword = {
        'storage': 50000 * 300 * 4 + 2_000_000 * 4,  # base + n-gram hashes
        'oov_rate': 0.01,  # Near-zero
        'training_data_needed': corpus_size  # Works with existing data
    }
    
    return {'word_level': word_level, 'subword': subword}
```

**Interview Tips:**
- Always use FastText/subword for Turkish, Finnish, Hungarian, Arabic
- German benefits greatly from subword due to compound words
- English is borderline — subword helps with typos and rare words
- BPE (used in BERT) is another subword approach optimized for compression

---

### Question 5
**How do you implement domain adaptation when pre-trained embeddings don't match your specific use case?**

**Answer:**

**Definition:**
Domain adaptation adjusts pre-trained embeddings to a target domain via: **continued pre-training** on domain corpus, **fine-tuning** on domain task, **retrofitting** with domain knowledge graphs, or **projection** to align domain-specific vectors. Goal: maintain general knowledge while capturing domain-specific semantics.

**Adaptation Strategies:**

| Strategy | Approach | Best For |
|----------|----------|----------|
| **Continued pre-training** | Train on domain corpus (MLM) | Large domain data |
| **Fine-tuning** | Train on labeled domain task | Supervised domain data |
| **Retrofitting** | Inject domain synonyms/relations | Domain ontologies |
| **Linear projection** | Learn A·x + b mapping | Small domain data |
| **Adapter modules** | Add domain-specific layers | Multiple domains |

**When to Use Each:**

| Scenario | Recommended Approach |
|----------|---------------------|
| **Medical NLP** | BioBERT (continued pre-training) |
| **Legal documents** | Fine-tune on legal classification |
| **Company jargon** | Retrofit with internal glossary |
| **Limited labeled data** | Linear projection + few examples |
| **Multi-domain system** | Adapter modules per domain |

**Python Code Example:**
```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class EmbeddingDomainAdapter:
    """Adapt pre-trained embeddings to target domain"""
    
    def __init__(self, pretrained_embeddings: Dict[str, np.ndarray], dim: int = 300):
        self.pretrained = pretrained_embeddings
        self.dim = dim
        self.adapted = pretrained_embeddings.copy()
    
    def continued_pretraining(self, domain_corpus: List[str], epochs: int = 5):
        """Continue training on domain corpus"""
        from gensim.models import Word2Vec
        
        # Tokenize corpus
        tokenized = [text.lower().split() for text in domain_corpus]
        
        # Initialize with pre-trained vectors
        model = Word2Vec(vector_size=self.dim, min_count=1)
        model.build_vocab(tokenized)
        
        # Load pre-trained weights where available
        for word in model.wv.index_to_key:
            if word in self.pretrained:
                model.wv[word] = self.pretrained[word]
        
        # Continue training
        model.train(tokenized, total_examples=len(tokenized), epochs=epochs)
        
        # Update adapted embeddings
        self.adapted = {word: model.wv[word] for word in model.wv.index_to_key}
        return self.adapted
    
    def retrofit(self, synonym_pairs: List[Tuple[str, str]], 
                 iterations: int = 10, alpha: float = 1.0):
        """Retrofit embeddings using domain relationships"""
        
        # Build adjacency from synonyms
        adjacency = {}
        for w1, w2 in synonym_pairs:
            if w1 not in adjacency:
                adjacency[w1] = []
            if w2 not in adjacency:
                adjacency[w2] = []
            adjacency[w1].append(w2)
            adjacency[w2].append(w1)
        
        # Retrofitting iterations
        new_embeddings = {w: v.copy() for w, v in self.pretrained.items()}
        
        for _ in range(iterations):
            for word in adjacency:
                if word not in self.pretrained:
                    continue
                
                neighbors = [n for n in adjacency[word] if n in new_embeddings]
                if not neighbors:
                    continue
                
                # Update: balance original embedding and neighbor average
                original = self.pretrained[word]
                neighbor_avg = np.mean([new_embeddings[n] for n in neighbors], axis=0)
                
                new_embeddings[word] = (original + alpha * neighbor_avg) / (1 + alpha)
        
        self.adapted = new_embeddings
        return self.adapted
    
    def linear_projection(self, source_words: List[str], 
                         target_vectors: np.ndarray):
        """Learn linear mapping from source to target domain"""
        
        # Get source vectors
        source_vectors = np.array([self.pretrained[w] for w in source_words])
        
        # Learn projection: W * source ≈ target
        W, _, _, _ = np.linalg.lstsq(source_vectors, target_vectors, rcond=None)
        
        # Apply to all embeddings
        self.adapted = {
            word: vec @ W for word, vec in self.pretrained.items()
        }
        return self.adapted

# BERT domain adaptation
class BERTDomainAdapter:
    """Adapt BERT to domain via continued pretraining"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        from transformers import (BertForMaskedLM, BertTokenizer,
                                   DataCollatorForLanguageModeling,
                                   TrainingArguments, Trainer)
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
    
    def continued_pretraining(self, domain_texts: List[str], 
                               output_dir: str, epochs: int = 3):
        """Continue MLM training on domain corpus"""
        from transformers import (DataCollatorForLanguageModeling,
                                   TrainingArguments, Trainer)
        from datasets import Dataset
        
        # Prepare dataset
        dataset = Dataset.from_dict({'text': domain_texts})
        
        def tokenize(examples):
            return self.tokenizer(examples['text'], 
                                 truncation=True, 
                                 max_length=512)
        
        tokenized = dataset.map(tokenize, batched=True)
        
        # MLM data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=0.15
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            save_steps=1000,
            learning_rate=2e-5
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator
        )
        
        trainer.train()
        return self.model

# Adapter-based domain adaptation
class DomainAdapter(nn.Module):
    """Lightweight adapter for domain-specific tuning"""
    
    def __init__(self, input_dim: int, adapter_dim: int = 64):
        super().__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, input_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Residual adapter
        adapter_out = self.up_project(self.activation(self.down_project(x)))
        return x + adapter_out
```

**Interview Tips:**
- Continued pre-training is most effective but requires compute
- Retrofitting works well with domain ontologies (medical, legal)
- Adapters are efficient for multi-domain systems
- Always evaluate on domain-specific benchmarks, not just general tasks

---

### Question 6
**When should you train custom embeddings versus using pre-trained models?**

**Answer:**

**Definition:**
Train custom embeddings when: **domain vocabulary differs significantly** (medical, legal, code), **data is proprietary** (internal documents), or **language is low-resource**. Use pre-trained when domain overlaps with training corpus, data is limited, or compute is constrained. Generally, pre-trained + fine-tuning outperforms training from scratch.

**Decision Matrix:**

| Factor | Train Custom | Use Pre-trained |
|--------|--------------|-----------------|
| **Domain match** | Poor overlap | Good overlap |
| **Data volume** | >10M tokens | <10M tokens |
| **Vocabulary** | Specialized | General |
| **Compute budget** | Available | Limited |
| **Update frequency** | Static domain | Evolving |
| **Quality bar** | Maximum | Good enough |

**Use Cases Comparison:**

| Scenario | Recommendation | Reason |
|----------|----------------|--------|
| **General chatbot** | Pre-trained | Standard vocabulary |
| **Medical NER** | Custom or BioBERT | Medical terms |
| **Code search** | Custom (CodeBERT) | Programming syntax |
| **Internal docs** | Custom | Proprietary jargon |
| **Low-resource lang** | Custom + transfer | Limited pre-trained |
| **Quick prototype** | Pre-trained | Fast iteration |

**Python Code Example:**
```python
import numpy as np
from typing import List, Dict

class EmbeddingDecisionHelper:
    """Help decide between custom vs pre-trained embeddings"""
    
    def __init__(self, pretrained_vocab: set):
        self.pretrained_vocab = pretrained_vocab
    
    def analyze_domain_corpus(self, domain_texts: List[str]) -> Dict:
        """Analyze domain corpus for custom embedding decision"""
        
        # Extract domain vocabulary
        domain_words = set()
        for text in domain_texts:
            domain_words.update(text.lower().split())
        
        # Calculate overlap
        overlap = domain_words.intersection(self.pretrained_vocab)
        coverage = len(overlap) / len(domain_words)
        
        # Unique domain terms
        domain_specific = domain_words - self.pretrained_vocab
        
        return {
            'domain_vocab_size': len(domain_words),
            'pretrained_coverage': coverage,
            'oov_words': len(domain_specific),
            'oov_ratio': len(domain_specific) / len(domain_words),
            'sample_oov': list(domain_specific)[:20],
            'recommendation': 'custom' if coverage < 0.7 else 'pretrained'
        }
    
    def estimate_training_requirements(self, corpus_size: int) -> Dict:
        """Estimate resources for custom training"""
        
        # Rule of thumb: need ~100x occurrences per word for good embedding
        estimated_vocab = int(corpus_size ** 0.5)  # Heaps' law approximation
        min_corpus_size = estimated_vocab * 100
        
        # Training time estimation (very rough)
        training_hours = corpus_size / 10_000_000  # ~1 hour per 10M tokens
        
        return {
            'estimated_vocab': estimated_vocab,
            'min_recommended_corpus': min_corpus_size,
            'sufficient_data': corpus_size >= min_corpus_size,
            'estimated_training_hours': training_hours,
            'recommended_epochs': 5 if corpus_size > 10_000_000 else 10
        }

# Training custom Word2Vec
def train_custom_word2vec(corpus: List[List[str]], 
                          dim: int = 300,
                          window: int = 5,
                          min_count: int = 5) -> dict:
    """Train custom Word2Vec embeddings"""
    from gensim.models import Word2Vec
    
    model = Word2Vec(
        sentences=corpus,
        vector_size=dim,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=10
    )
    
    return {word: model.wv[word] for word in model.wv.index_to_key}

# Custom BERT pre-training
def train_custom_bert(corpus_file: str, output_dir: str):
    """Train BERT from scratch (requires significant compute)"""
    from transformers import (
        BertConfig, BertForMaskedLM, BertTokenizer,
        LineByLineTextDataset, DataCollatorForLanguageModeling,
        TrainingArguments, Trainer
    )
    
    # Train custom tokenizer first
    from tokenizers import BertWordPieceTokenizer
    
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(files=[corpus_file], vocab_size=30000)
    tokenizer.save_model(output_dir)
    
    # Initialize model
    config = BertConfig(
        vocab_size=30000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )
    model = BertForMaskedLM(config)
    
    # This requires significant compute (days on multiple GPUs)
    # Usually better to start from pre-trained and continue training
    
    return model

# Hybrid approach: pre-trained + domain-specific
class HybridEmbedding:
    """Combine pre-trained with custom domain embeddings"""
    
    def __init__(self, pretrained: Dict, custom: Dict, alpha: float = 0.5):
        self.pretrained = pretrained
        self.custom = custom
        self.alpha = alpha
    
    def get_embedding(self, word: str) -> np.ndarray:
        """Get blended embedding"""
        word = word.lower()
        
        has_pretrained = word in self.pretrained
        has_custom = word in self.custom
        
        if has_pretrained and has_custom:
            # Blend both
            return (self.alpha * self.pretrained[word] + 
                    (1 - self.alpha) * self.custom[word])
        elif has_custom:
            return self.custom[word]
        elif has_pretrained:
            return self.pretrained[word]
        else:
            return np.zeros(300)
```

**Interview Tips:**
- Pre-trained + fine-tuning almost always beats training from scratch
- Need >10M tokens for reasonable custom embeddings
- Domain-specific pre-trained models (BioBERT, CodeBERT) are best of both worlds
- For internal jargon, consider retrofitting pre-trained with company vocabulary

---

### Question 7
**What techniques help optimize BERT embedding extraction for real-time inference requirements?**

**Answer:**

**Definition:**
Optimize BERT inference via: **model distillation** (DistilBERT: 60% faster), **quantization** (INT8: 2-4x speedup), **ONNX Runtime** (optimized inference), **batch processing**, **layer pruning** (use fewer layers), and **caching** (store frequent embeddings). Target: <50ms latency for production real-time systems.

**Optimization Techniques:**

| Technique | Speedup | Quality Loss | Effort |
|-----------|---------|--------------|--------|
| **DistilBERT** | 1.6x | ~3% | Low |
| **TinyBERT** | 3-4x | ~5% | Low |
| **INT8 quantization** | 2-4x | <1% | Medium |
| **ONNX Runtime** | 1.5-2x | 0% | Low |
| **Layer pruning (6L)** | 2x | ~2% | Low |
| **Caching** | 10-100x | 0% | Medium |
| **GPU batching** | 5-10x | 0% | Low |

**Optimization Pipeline:**
```
Original BERT (440MB, 100ms)
    → Distillation (DistilBERT: 265MB, 60ms)
    → Quantization (INT8: 100MB, 30ms)
    → ONNX conversion (optimized ops: 25ms)
    → Caching layer (cache hits: <1ms)
```

**Python Code Example:**
```python
import torch
import numpy as np
from typing import List, Dict
import time

class OptimizedBERTEmbedder:
    """Optimized BERT for real-time embedding extraction"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased',
                 use_onnx: bool = True,
                 use_cache: bool = True,
                 cache_size: int = 10000):
        
        self.use_onnx = use_onnx
        self.use_cache = use_cache
        
        if use_onnx:
            self._setup_onnx(model_name)
        else:
            self._setup_pytorch(model_name)
        
        if use_cache:
            self.cache = {}
            self.cache_size = cache_size
    
    def _setup_pytorch(self, model_name: str):
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def _setup_onnx(self, model_name: str):
        """Setup ONNX Runtime for faster inference"""
        from transformers import AutoTokenizer
        import onnxruntime as ort
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Export or load ONNX model
        onnx_path = f"{model_name.replace('/', '_')}.onnx"
        
        try:
            self.session = ort.InferenceSession(onnx_path)
        except:
            # Export model to ONNX
            self._export_to_onnx(model_name, onnx_path)
            self.session = ort.InferenceSession(onnx_path)
    
    def _export_to_onnx(self, model_name: str, output_path: str):
        """Export PyTorch model to ONNX"""
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        dummy_input = self.tokenizer(
            "dummy text",
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True
        )
        
        torch.onnx.export(
            model,
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'attention_mask': {0: 'batch', 1: 'sequence'}
            },
            opset_version=12
        )
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Get embeddings with optimization"""
        
        # Check cache
        if self.use_cache:
            cached, uncached_texts, uncached_indices = self._check_cache(texts)
            if not uncached_texts:
                return cached
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Compute embeddings
        if self.use_onnx:
            embeddings = self._embed_onnx(uncached_texts)
        else:
            embeddings = self._embed_pytorch(uncached_texts)
        
        # Update cache
        if self.use_cache:
            self._update_cache(uncached_texts, embeddings)
            return self._merge_cached(cached, embeddings, uncached_indices, len(texts))
        
        return embeddings
    
    def _embed_pytorch(self, texts: List[str]) -> np.ndarray:
        """PyTorch inference"""
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
    
    def _embed_onnx(self, texts: List[str]) -> np.ndarray:
        """ONNX Runtime inference"""
        inputs = self.tokenizer(
            texts,
            return_tensors='np',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        outputs = self.session.run(
            None,
            {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
        )
        
        # Mean pooling
        return outputs[0].mean(axis=1)
    
    def _check_cache(self, texts):
        """Check which texts are cached"""
        cached = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self.cache:
                cached[i] = self.cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        return cached, uncached_texts, uncached_indices
    
    def _update_cache(self, texts, embeddings):
        """Update cache with LRU eviction"""
        for text, emb in zip(texts, embeddings):
            if len(self.cache) >= self.cache_size:
                # Evict oldest
                oldest = next(iter(self.cache))
                del self.cache[oldest]
            self.cache[text] = emb

# Quantization for CPU inference
def quantize_for_cpu(model_name: str):
    """Dynamic quantization for CPU deployment"""
    from transformers import AutoModel
    
    model = AutoModel.from_pretrained(model_name)
    
    # Dynamic quantization (weights only)
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    return quantized

# Benchmark utility
def benchmark_embedder(embedder, texts: List[str], num_runs: int = 100):
    """Benchmark embedding extraction"""
    # Warmup
    _ = embedder.embed(texts[:5])
    
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = embedder.embed(texts)
    elapsed = time.perf_counter() - start
    
    return {
        'total_time': elapsed,
        'per_batch_ms': (elapsed / num_runs) * 1000,
        'per_text_ms': (elapsed / (num_runs * len(texts))) * 1000,
        'throughput': (num_runs * len(texts)) / elapsed
    }
```

**Interview Tips:**
- DistilBERT + ONNX + caching can achieve <10ms latency
- Quantization works best on CPU; use FP16 for GPU
- Cache hit rates of 60-80% common for production search
- Consider async batching to improve GPU utilization

---

### Question 8
**How do you implement effective dimensionality reduction (PCA, UMAP) for high-dimensional embeddings without losing semantic information?**

**Answer:**

**Definition:**
Reduce embedding dimensions via **PCA** (linear, fast, preserves variance), **UMAP** (non-linear, preserves local structure), or **learned projections** (task-optimized). Balance: reduce from 768→256 dims with <5% quality loss for storage/speed gains. Critical: evaluate on downstream task, not just reconstruction error.

**Method Comparison:**

| Method | Type | Speed | Quality | Use Case |
|--------|------|-------|---------|----------|
| **PCA** | Linear | Fast | Good | General compression |
| **UMAP** | Non-linear | Slow | Excellent | Visualization |
| **t-SNE** | Non-linear | Very slow | Good | Visualization only |
| **Autoencoder** | Learned | Medium | Best | Task-specific |
| **Random projection** | Linear | Very fast | Moderate | Approximate |

**Dimension Reduction Guidelines:**

| Original Dim | Target Dim | Expected Quality Loss |
|--------------|------------|----------------------|
| 768 (BERT) | 512 | <1% |
| 768 (BERT) | 256 | 2-3% |
| 768 (BERT) | 128 | 5-8% |
| 768 (BERT) | 64 | 10-15% |
| 300 (Word2Vec) | 100 | 3-5% |

**Python Code Example:**
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from typing import Tuple

class EmbeddingReducer:
    """Dimensionality reduction for embeddings"""
    
    def __init__(self, target_dim: int, method: str = 'pca'):
        self.target_dim = target_dim
        self.method = method
        self.reducer = None
        self.scaler = None
    
    def fit(self, embeddings: np.ndarray):
        """Fit reducer on embeddings"""
        
        # Standardize for better reduction
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(embeddings)
        
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.target_dim)
            self.reducer.fit(scaled)
            
            # Report variance explained
            var_explained = sum(self.reducer.explained_variance_ratio_)
            print(f"PCA: {var_explained:.2%} variance explained with {self.target_dim} dims")
        
        elif self.method == 'umap':
            self.reducer = umap.UMAP(
                n_components=self.target_dim,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine'
            )
            self.reducer.fit(scaled)
        
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to lower dimension"""
        scaled = self.scaler.transform(embeddings)
        return self.reducer.transform(scaled)
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(embeddings)
        return self.transform(embeddings)

class WhiteningPCA:
    """PCA with whitening for better cosine similarity preservation"""
    
    def __init__(self, target_dim: int):
        self.target_dim = target_dim
        self.mean = None
        self.components = None
        self.singular_values = None
    
    def fit(self, embeddings: np.ndarray):
        """Fit whitening PCA"""
        self.mean = embeddings.mean(axis=0)
        centered = embeddings - self.mean
        
        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        self.components = Vt[:self.target_dim]
        self.singular_values = S[:self.target_dim]
        
        return self
    
    def transform(self, embeddings: np.ndarray, whiten: bool = True) -> np.ndarray:
        """Transform with optional whitening"""
        centered = embeddings - self.mean
        projected = centered @ self.components.T
        
        if whiten:
            # Scale by inverse singular values
            projected = projected / (self.singular_values + 1e-8)
        
        return projected

# Learned dimensionality reduction (autoencoder)
import torch
import torch.nn as nn

class EmbeddingAutoencoder(nn.Module):
    """Autoencoder for learned dimensionality reduction"""
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

def train_autoencoder(embeddings: np.ndarray, latent_dim: int,
                      epochs: int = 100) -> EmbeddingAutoencoder:
    """Train autoencoder for dimensionality reduction"""
    
    input_dim = embeddings.shape[1]
    model = EmbeddingAutoencoder(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Convert to tensor
    X = torch.FloatTensor(embeddings)
    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            
            optimizer.zero_grad()
            x_recon, z = model(x)
            
            # Reconstruction loss + regularization
            loss = nn.MSELoss()(x_recon, x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    model.eval()
    return model

# Evaluation of reduction quality
def evaluate_reduction(original: np.ndarray, reduced: np.ndarray,
                       pairs: list) -> dict:
    """Evaluate quality of dimensionality reduction"""
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Compute similarities before and after
    original_sims = []
    reduced_sims = []
    
    for i, j in pairs:
        orig_sim = cosine_similarity([original[i]], [original[j]])[0][0]
        red_sim = cosine_similarity([reduced[i]], [reduced[j]])[0][0]
        original_sims.append(orig_sim)
        reduced_sims.append(red_sim)
    
    correlation = np.corrcoef(original_sims, reduced_sims)[0, 1]
    
    return {
        'correlation': correlation,
        'mean_absolute_error': np.mean(np.abs(np.array(original_sims) - np.array(reduced_sims))),
        'quality_preserved': correlation > 0.95
    }
```

**Interview Tips:**
- PCA is usually sufficient for production; UMAP for visualization
- Always evaluate on downstream task (search accuracy), not reconstruction
- Whitening helps preserve cosine similarity after reduction
- 256 dimensions is often a sweet spot for BERT embeddings

---

## Embedding Operations & Similarity

### Question 9
**Explain the different similarity metrics (cosine, euclidean, dot product) and when to use each.**

**Answer:**

**Definition:**
**Cosine similarity** measures angle between vectors (normalized, ignores magnitude). **Euclidean distance** measures geometric distance (sensitive to magnitude). **Dot product** combines angle and magnitude. Choice depends on: whether magnitude is meaningful, whether embeddings are normalized, and computational constraints.

**Metric Comparison:**

| Metric | Formula | Range | Normalized? | Use Case |
|--------|---------|-------|-------------|----------|
| **Cosine** | $\frac{a \cdot b}{\|a\| \|b\|}$ | [-1, 1] | Yes | Text similarity |
| **Euclidean** | $\sqrt{\sum(a_i - b_i)^2}$ | [0, ∞) | No | Clustering |
| **Dot Product** | $\sum a_i \cdot b_i$ | (-∞, ∞) | No | Learned embeddings |
| **Manhattan** | $\sum|a_i - b_i|$ | [0, ∞) | No | High-dim sparse |

**When to Use Each:**

| Scenario | Best Metric | Reason |
|----------|-------------|--------|
| **Semantic search** | Cosine | Direction matters, not magnitude |
| **Sentence-BERT** | Cosine | Training objective uses cosine |
| **Recommendation** | Dot product | Magnitude encodes preference strength |
| **Two-tower models** | Dot product | Efficiently computed |
| **K-means clustering** | Euclidean | Cluster centroids meaningful |
| **Normalized embeddings** | Any equivalent | Cosine = dot product = Euclidean ordering |

**Mathematical Relationships:**
- If vectors are L2-normalized: $\cos(a,b) = a \cdot b$
- $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2(a \cdot b)$
- For normalized vectors: $\|a-b\|^2 = 2(1 - \cos(a,b))$

**Python Code Example:**
```python
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMetrics:
    """Comprehensive similarity metric implementations"""
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity: angle between vectors"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot / (norm_a * norm_b + 1e-8)
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance: geometric distance"""
        return np.sqrt(np.sum((a - b) ** 2))
    
    @staticmethod
    def dot_product(a: np.ndarray, b: np.ndarray) -> float:
        """Dot product: combines direction and magnitude"""
        return np.dot(a, b)
    
    @staticmethod
    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Manhattan (L1) distance"""
        return np.sum(np.abs(a - b))
    
    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """L2 normalize vector"""
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)

# Metric selection helper
def select_metric(embedding_type: str, task: str, normalized: bool) -> str:
    """Recommend similarity metric"""
    
    recommendations = {
        ('bert', 'search', True): 'dot_product',  # Equiv to cosine when normalized
        ('bert', 'search', False): 'cosine',
        ('sentence-bert', 'search', True): 'cosine',
        ('word2vec', 'similarity', False): 'cosine',
        ('recommendation', 'ranking', False): 'dot_product',
        ('clustering', 'kmeans', False): 'euclidean'
    }
    
    key = (embedding_type, task, normalized)
    return recommendations.get(key, 'cosine')

# Batch similarity computation
def batch_similarity(queries: np.ndarray, candidates: np.ndarray, 
                     metric: str = 'cosine') -> np.ndarray:
    """Efficient batch similarity computation"""
    
    if metric == 'cosine':
        # Normalize and dot product
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        candidates_norm = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
        return queries_norm @ candidates_norm.T
    
    elif metric == 'dot_product':
        return queries @ candidates.T
    
    elif metric == 'euclidean':
        # Efficient L2 distance using (a-b)^2 = a^2 + b^2 - 2ab
        q_sq = np.sum(queries ** 2, axis=1, keepdims=True)
        c_sq = np.sum(candidates ** 2, axis=1, keepdims=True).T
        cross = 2 * queries @ candidates.T
        distances = np.sqrt(np.maximum(q_sq + c_sq - cross, 0))
        return -distances  # Negate for similarity ordering

# Demonstration of metric equivalence
def demonstrate_equivalence():
    """Show when metrics are equivalent"""
    
    # Random vectors
    a = np.random.randn(768)
    b = np.random.randn(768)
    
    # Original metrics
    cos_orig = SimilarityMetrics.cosine_similarity(a, b)
    dot_orig = SimilarityMetrics.dot_product(a, b)
    euc_orig = SimilarityMetrics.euclidean_distance(a, b)
    
    # Normalize
    a_norm = SimilarityMetrics.normalize(a)
    b_norm = SimilarityMetrics.normalize(b)
    
    # Normalized metrics
    cos_norm = SimilarityMetrics.cosine_similarity(a_norm, b_norm)
    dot_norm = SimilarityMetrics.dot_product(a_norm, b_norm)
    euc_norm = SimilarityMetrics.euclidean_distance(a_norm, b_norm)
    
    print(f"Original - Cosine: {cos_orig:.4f}, Dot: {dot_orig:.4f}, Euclidean: {euc_orig:.4f}")
    print(f"Normalized - Cosine: {cos_norm:.4f}, Dot: {dot_norm:.4f}, Euclidean: {euc_norm:.4f}")
    print(f"Relationship: Cosine = Dot when normalized: {np.isclose(cos_norm, dot_norm)}")
    print(f"Euclidean = sqrt(2(1-cos)): {np.isclose(euc_norm, np.sqrt(2*(1-cos_norm)))}")
```

**Interview Tips:**
- Cosine is the default for text embeddings (magnitude is often noise)
- Always normalize embeddings before storing for consistent similarity
- Dot product is faster (no normalization) when embeddings are pre-normalized
- Inner product (dot) is preferred in FAISS for efficiency

---

### Question 10
**When should you use averaged word embeddings versus more sophisticated aggregation (attention pooling) for document representation?**

**Answer:**

**Definition:**
**Average pooling** sums word embeddings and divides by count — simple, fast, but loses word importance. **Attention pooling** learns to weight important words higher. Use averaging for quick baselines and efficiency; use attention pooling when accuracy is critical and importance varies across words.

**Aggregation Methods:**

| Method | Complexity | Quality | Use Case |
|--------|------------|---------|----------|
| **Mean pooling** | O(n) | Baseline | Fast similarity |
| **Max pooling** | O(n) | Similar | Feature detection |
| **TF-IDF weighted** | O(n) | Better | Important words |
| **Attention pooling** | O(n²) | Best | Quality-critical |
| **CLS token** | O(1) | Good | BERT classification |

**When to Use Each:**

| Scenario | Recommended | Why |
|----------|-------------|-----|
| **Real-time search** | Mean pooling | Speed priority |
| **Document classification** | Attention pooling | Accuracy matters |
| **Semantic similarity** | Mean or CLS | Sentence-BERT uses mean |
| **Short texts (<20 words)** | Mean pooling | All words matter |
| **Long documents** | Attention pooling | Need focus |
| **Topic detection** | TF-IDF weighted | Rare words important |

**Python Code Example:**
```python
import numpy as np
import torch
import torch.nn as nn
from typing import List

class DocumentEmbedder:
    """Multiple aggregation strategies for document embeddings"""
    
    def __init__(self, word_embeddings: dict, dim: int = 300):
        self.embeddings = word_embeddings
        self.dim = dim
    
    def mean_pooling(self, words: List[str]) -> np.ndarray:
        """Simple mean of word embeddings"""
        vectors = [self.embeddings[w] for w in words if w in self.embeddings]
        if not vectors:
            return np.zeros(self.dim)
        return np.mean(vectors, axis=0)
    
    def max_pooling(self, words: List[str]) -> np.ndarray:
        """Element-wise max pooling"""
        vectors = [self.embeddings[w] for w in words if w in self.embeddings]
        if not vectors:
            return np.zeros(self.dim)
        return np.max(vectors, axis=0)
    
    def tfidf_weighted(self, words: List[str], tfidf_weights: dict) -> np.ndarray:
        """TF-IDF weighted average"""
        weighted_sum = np.zeros(self.dim)
        weight_total = 0.0
        
        for word in words:
            if word in self.embeddings:
                weight = tfidf_weights.get(word, 1.0)
                weighted_sum += weight * self.embeddings[word]
                weight_total += weight
        
        if weight_total == 0:
            return np.zeros(self.dim)
        return weighted_sum / weight_total
    
    def sif_embedding(self, words: List[str], word_freq: dict, 
                      a: float = 1e-3) -> np.ndarray:
        """Smooth Inverse Frequency (SIF) embedding"""
        # SIF: weight = a / (a + p(w)) where p(w) is word probability
        total_words = sum(word_freq.values())
        
        weighted_sum = np.zeros(self.dim)
        weight_total = 0.0
        
        for word in words:
            if word in self.embeddings:
                p_w = word_freq.get(word, 1) / total_words
                weight = a / (a + p_w)
                weighted_sum += weight * self.embeddings[word]
                weight_total += weight
        
        if weight_total == 0:
            return np.zeros(self.dim)
        return weighted_sum / weight_total

# Attention pooling with learnable weights
class AttentionPooling(nn.Module):
    """Learnable attention-based pooling"""
    
    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            embeddings: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) - 1 for valid, 0 for padding
        Returns:
            pooled: (batch, embed_dim)
        """
        # Compute attention scores
        scores = self.attention(embeddings).squeeze(-1)  # (batch, seq_len)
        
        # Mask padding tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax attention weights
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)
        
        return pooled, weights

# Self-attention pooling (more sophisticated)
class SelfAttentionPooling(nn.Module):
    """Self-attention based document representation"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Query vector for pooling
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            embeddings: (batch, seq_len, embed_dim)
        Returns:
            pooled: (batch, embed_dim)
        """
        batch_size = embeddings.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        
        # Attend to all word embeddings
        pooled, weights = self.attention(
            query, embeddings, embeddings,
            key_padding_mask=(mask == 0) if mask is not None else None
        )
        
        return pooled.squeeze(1), weights

# BERT-style pooling options
class BERTPooler:
    """Different pooling strategies for BERT outputs"""
    
    @staticmethod
    def cls_pooling(outputs, attention_mask):
        """Use [CLS] token representation"""
        return outputs.last_hidden_state[:, 0, :]
    
    @staticmethod
    def mean_pooling(outputs, attention_mask):
        """Mean of all token embeddings (Sentence-BERT style)"""
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    @staticmethod
    def max_pooling(outputs, attention_mask):
        """Max pooling over sequence"""
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, dim=1)[0]

# Comparison benchmark
def compare_pooling_methods(texts: List[str], embeddings: dict, labels: List[int]):
    """Compare pooling methods on classification task"""
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    embedder = DocumentEmbedder(embeddings)
    
    results = {}
    for method_name, method in [
        ('mean', embedder.mean_pooling),
        ('max', embedder.max_pooling)
    ]:
        X = np.array([method(text.split()) for text in texts])
        scores = cross_val_score(LogisticRegression(), X, labels, cv=5)
        results[method_name] = scores.mean()
    
    return results
```

**Interview Tips:**
- Sentence-BERT uses mean pooling — proven effective for similarity
- CLS token works well for classification but not similarity
- SIF (Smooth Inverse Frequency) is a strong baseline for sentence embeddings
- Attention pooling requires training data but gives best results

---

### Question 11
**How do you handle the context length limitations when extracting BERT embeddings from long documents?**

**Answer:**

**Definition:**
Handle BERT's 512-token limit via: **chunking** (split into overlapping windows), **hierarchical encoding** (encode chunks → aggregate), **Longformer/BigBird** (sparse attention for 4K+ tokens), or **sentence-level pooling** (embed sentences → aggregate). Choose based on document structure and downstream task requirements.

**Strategies for Long Documents:**

| Strategy | Max Length | Quality | Complexity |
|----------|------------|---------|------------|
| **Truncation** | 512 | Poor | Low |
| **Chunking + Mean** | Unlimited | Moderate | Low |
| **Chunking + Attention** | Unlimited | Good | Medium |
| **Hierarchical** | Unlimited | Good | Medium |
| **Longformer** | 4096+ | Best | Low |
| **Sentence pooling** | Unlimited | Good | Medium |

**Chunking Approaches:**

| Type | Description | Use Case |
|------|-------------|----------|
| **Non-overlapping** | Split every 512 tokens | Speed priority |
| **Overlapping** | 50% overlap | Context preservation |
| **Sentence-aware** | Split at sentence boundaries | Coherent chunks |
| **Paragraph-aware** | Split at paragraphs | Document structure |

**Python Code Example:**
```python
import torch
import numpy as np
from typing import List, Tuple
from transformers import AutoModel, AutoTokenizer

class LongDocumentEmbedder:
    """Handle long documents exceeding BERT's token limit"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 max_length: int = 512,
                 stride: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.stride = stride
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.max_length - 2, len(tokens))  # -2 for [CLS], [SEP]
            chunk_tokens = [self.tokenizer.cls_token_id] + tokens[start:end] + [self.tokenizer.sep_token_id]
            chunks.append(chunk_tokens)
            
            if end >= len(tokens):
                break
            start += self.stride
        
        return chunks
    
    def embed_chunks(self, chunk_tokens: List[List[int]]) -> List[np.ndarray]:
        """Embed each chunk"""
        embeddings = []
        
        for tokens in chunk_tokens:
            # Pad to max_length
            attention_mask = [1] * len(tokens)
            padding_length = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            
            inputs = {
                'input_ids': torch.tensor([tokens]).to(self.device),
                'attention_mask': torch.tensor([attention_mask]).to(self.device)
            }
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling over non-padding tokens
            hidden = outputs.last_hidden_state[0]
            mask = inputs['attention_mask'][0].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(0) / mask.sum()
            
            embeddings.append(pooled.cpu().numpy())
        
        return embeddings
    
    def embed_document(self, text: str, aggregation: str = 'mean') -> np.ndarray:
        """Get single embedding for long document"""
        chunks = self.chunk_text(text)
        chunk_embeddings = self.embed_chunks(chunks)
        
        if aggregation == 'mean':
            return np.mean(chunk_embeddings, axis=0)
        elif aggregation == 'max':
            return np.max(chunk_embeddings, axis=0)
        elif aggregation == 'first':
            return chunk_embeddings[0]
        elif aggregation == 'weighted':
            # Weight by position (earlier chunks often more important)
            weights = np.exp(-np.arange(len(chunk_embeddings)) * 0.1)
            weights /= weights.sum()
            return np.average(chunk_embeddings, axis=0, weights=weights)

# Hierarchical encoding
class HierarchicalEncoder:
    """Two-level encoding: sentences → document"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def split_sentences(self, text: str) -> List[str]:
        """Split document into sentences"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def encode_hierarchical(self, text: str, 
                           aggregation: str = 'mean') -> np.ndarray:
        """Encode via sentence-level pooling"""
        sentences = self.split_sentences(text)
        
        if not sentences:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        # Encode all sentences
        sentence_embeddings = self.model.encode(sentences)
        
        if aggregation == 'mean':
            return np.mean(sentence_embeddings, axis=0)
        elif aggregation == 'max':
            return np.max(sentence_embeddings, axis=0)
        elif aggregation == 'weighted_position':
            # Later sentences often contain conclusions
            weights = np.linspace(0.5, 1.5, len(sentences))
            weights /= weights.sum()
            return np.average(sentence_embeddings, axis=0, weights=weights)

# Longformer for truly long documents
class LongformerEmbedder:
    """Use Longformer for documents up to 4096 tokens"""
    
    def __init__(self, model_name: str = 'allenai/longformer-base-4096'):
        from transformers import LongformerModel, LongformerTokenizer
        
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name)
        self.model.eval()
    
    def embed(self, text: str) -> np.ndarray:
        """Embed long document directly"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=4096,
            padding=True
        )
        
        # Set global attention on [CLS] token
        global_attention_mask = torch.zeros_like(inputs['input_ids'])
        global_attention_mask[:, 0] = 1
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                global_attention_mask=global_attention_mask
            )
        
        # Use [CLS] token embedding
        return outputs.last_hidden_state[:, 0, :].numpy()[0]
```

**Interview Tips:**
- Overlapping chunks preserve context better but increase compute
- Sentence-level hierarchical is often best for diverse documents
- Longformer/BigBird are preferred when available and document < 4K tokens
- Weight chunks by importance (first/last often most relevant)

---

### Question 12
**What strategies help reduce bias in word embeddings for fair and inclusive NLP applications?**

**Answer:**

**Definition:**
Reduce embedding bias via: **debiasing projections** (remove gender/race subspace), **counterfactual data augmentation** (balance training data), **adversarial training** (learn bias-invariant representations), or **post-hoc calibration**. Goal: maintain semantic utility while reducing stereotypical associations.

**Bias Types in Embeddings:**

| Bias Type | Example | Impact |
|-----------|---------|--------|
| **Gender** | "programmer" → male | Hiring bias |
| **Racial** | Name associations | Discrimination |
| **Religious** | Stereotype association | Unfair treatment |
| **Age** | Negative stereotypes | Employment bias |

**Debiasing Approaches:**

| Method | Approach | When to Use |
|--------|----------|-------------|
| **Hard debiasing** | Project out bias direction | Post-training fix |
| **Soft debiasing** | Partial projection | Balance utility/fairness |
| **Counterfactual** | Augment training data | Pre-training |
| **Adversarial** | Train to hide protected attr | Training time |
| **Re-training** | Curated balanced corpus | Maximum control |

**Python Code Example:**
```python
import numpy as np
from typing import List, Tuple, Dict
from sklearn.decomposition import PCA

class EmbeddingDebiaser:
    """Methods to reduce bias in word embeddings"""
    
    def __init__(self, embeddings: Dict[str, np.ndarray]):
        self.embeddings = embeddings
        self.bias_direction = None
    
    def identify_bias_direction(self, 
                                 word_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Identify bias direction from definitional pairs
        E.g., [("man", "woman"), ("he", "she"), ("king", "queen")]
        """
        differences = []
        
        for w1, w2 in word_pairs:
            if w1 in self.embeddings and w2 in self.embeddings:
                diff = self.embeddings[w1] - self.embeddings[w2]
                differences.append(diff)
        
        if not differences:
            raise ValueError("No valid word pairs found")
        
        # PCA to find principal direction
        pca = PCA(n_components=1)
        pca.fit(differences)
        self.bias_direction = pca.components_[0]
        
        return self.bias_direction
    
    def hard_debias(self, words_to_debias: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Hard debiasing: project out bias direction completely
        (Bolukbasi et al., 2016)
        """
        if self.bias_direction is None:
            raise ValueError("Call identify_bias_direction first")
        
        debiased = {}
        words = words_to_debias or self.embeddings.keys()
        
        for word in words:
            if word not in self.embeddings:
                continue
            
            vec = self.embeddings[word]
            # Project out bias direction
            projection = np.dot(vec, self.bias_direction) * self.bias_direction
            debiased[word] = vec - projection
        
        return debiased
    
    def soft_debias(self, words_to_debias: List[str], 
                    lambda_param: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Soft debiasing: partial projection (retain some structure)
        """
        if self.bias_direction is None:
            raise ValueError("Call identify_bias_direction first")
        
        debiased = {}
        
        for word in words_to_debias:
            if word not in self.embeddings:
                continue
            
            vec = self.embeddings[word]
            projection = np.dot(vec, self.bias_direction) * self.bias_direction
            # Partial removal
            debiased[word] = vec - lambda_param * projection
        
        return debiased
    
    def measure_bias(self, word: str, 
                     attribute_words_a: List[str],
                     attribute_words_b: List[str]) -> float:
        """
        Measure bias of a word toward two attribute sets
        WEAT-style metric
        """
        if word not in self.embeddings:
            return 0.0
        
        word_vec = self.embeddings[word]
        
        # Average similarity to each attribute set
        sim_a = np.mean([
            self._cosine(word_vec, self.embeddings.get(w, np.zeros_like(word_vec)))
            for w in attribute_words_a if w in self.embeddings
        ])
        
        sim_b = np.mean([
            self._cosine(word_vec, self.embeddings.get(w, np.zeros_like(word_vec)))
            for w in attribute_words_b if w in self.embeddings
        ])
        
        return sim_a - sim_b  # Positive = biased toward A
    
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# Counterfactual Data Augmentation
class CounterfactualAugmenter:
    """Generate counterfactual training data to reduce bias"""
    
    def __init__(self):
        self.swap_pairs = {
            'he': 'she', 'she': 'he',
            'him': 'her', 'her': 'him',
            'his': 'her', 'man': 'woman',
            'woman': 'man', 'male': 'female',
            'female': 'male', 'boy': 'girl',
            'girl': 'boy'
        }
    
    def augment(self, text: str) -> str:
        """Create counterfactual version of text"""
        words = text.split()
        augmented = []
        
        for word in words:
            lower = word.lower()
            if lower in self.swap_pairs:
                # Preserve capitalization
                swap = self.swap_pairs[lower]
                if word[0].isupper():
                    swap = swap.capitalize()
                augmented.append(swap)
            else:
                augmented.append(word)
        
        return ' '.join(augmented)
    
    def create_balanced_dataset(self, texts: List[str]) -> List[str]:
        """Create balanced dataset with counterfactuals"""
        balanced = []
        for text in texts:
            balanced.append(text)
            counterfactual = self.augment(text)
            if counterfactual != text:
                balanced.append(counterfactual)
        return balanced

# WEAT (Word Embedding Association Test)
def weat_score(embeddings: Dict[str, np.ndarray],
               target_x: List[str], target_y: List[str],
               attribute_a: List[str], attribute_b: List[str]) -> float:
    """
    WEAT: Measures differential association
    Caliskan et al., 2017
    """
    def mean_sim(word, attributes):
        if word not in embeddings:
            return 0
        sims = []
        for attr in attributes:
            if attr in embeddings:
                sim = np.dot(embeddings[word], embeddings[attr]) / (
                    np.linalg.norm(embeddings[word]) * np.linalg.norm(embeddings[attr]) + 1e-8
                )
                sims.append(sim)
        return np.mean(sims) if sims else 0
    
    def association(word):
        return mean_sim(word, attribute_a) - mean_sim(word, attribute_b)
    
    # Differential association
    x_assoc = [association(w) for w in target_x if w in embeddings]
    y_assoc = [association(w) for w in target_y if w in embeddings]
    
    effect_size = (np.mean(x_assoc) - np.mean(y_assoc)) / np.std(x_assoc + y_assoc)
    
    return effect_size
```

**Interview Tips:**
- Hard debiasing can hurt some downstream tasks — always evaluate
- Bias is often relearned during fine-tuning — debias at multiple stages
- WEAT is standard for measuring bias; include in evaluation pipeline
- Counterfactual augmentation is often more effective than post-hoc debiasing

---

### Question 13
**How do you implement effective cross-lingual embedding alignment techniques?**

**Answer:**

**Definition:**
Cross-lingual alignment maps embeddings from different languages into a **shared vector space** where similar concepts are close. Methods: **supervised** (parallel dictionary), **unsupervised** (adversarial + refinement), or **joint training** (multilingual models like mBERT). Enables zero-shot cross-lingual transfer.

**Alignment Approaches:**

| Method | Supervision | Quality | Use Case |
|--------|-------------|---------|----------|
| **Procrustes** | Dictionary | Good | Seed dictionary available |
| **Adversarial** | None | Moderate | No parallel data |
| **MUSE** | Optional | Good | General purpose |
| **mBERT/XLM-R** | None (joint) | Best | Full solution |
| **VecMap** | Dictionary | Good | Easy to use |

**Procrustes Alignment:**
Given source embeddings $X$ and target embeddings $Z$ with dictionary pairs:
$$W^* = \arg\min_W \|WX - Z\|_F^2 \text{ s.t. } W^TW = I$$

Solution: $W = VU^T$ from SVD of $Z^TX = U\Sigma V^T$

**Python Code Example:**
```python
import numpy as np
from typing import Dict, List, Tuple
from scipy.linalg import orthogonal_procrustes

class CrossLingualAligner:
    """Align embeddings across languages"""
    
    def __init__(self, source_embeddings: Dict[str, np.ndarray],
                 target_embeddings: Dict[str, np.ndarray]):
        self.source = source_embeddings
        self.target = target_embeddings
        self.W = None  # Alignment matrix
    
    def procrustes_alignment(self, 
                             dictionary: List[Tuple[str, str]]) -> np.ndarray:
        """
        Supervised alignment using bilingual dictionary
        (Conneau et al., 2017)
        """
        # Build matrices from dictionary pairs
        X = []  # Source vectors
        Z = []  # Target vectors
        
        for src_word, tgt_word in dictionary:
            if src_word in self.source and tgt_word in self.target:
                X.append(self.source[src_word])
                Z.append(self.target[tgt_word])
        
        X = np.array(X)
        Z = np.array(Z)
        
        # Normalize
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        
        # Orthogonal Procrustes
        self.W, _ = orthogonal_procrustes(X, Z)
        
        return self.W
    
    def align_embedding(self, source_vec: np.ndarray) -> np.ndarray:
        """Transform source embedding to target space"""
        if self.W is None:
            raise ValueError("Call procrustes_alignment first")
        return source_vec @ self.W
    
    def translate_word(self, source_word: str, k: int = 5) -> List[str]:
        """Find nearest neighbors in target language"""
        if source_word not in self.source:
            return []
        
        # Align source word
        aligned = self.align_embedding(self.source[source_word])
        
        # Find nearest in target
        similarities = {}
        for word, vec in self.target.items():
            sim = np.dot(aligned, vec) / (
                np.linalg.norm(aligned) * np.linalg.norm(vec) + 1e-8
            )
            similarities[word] = sim
        
        # Top-k
        return sorted(similarities.keys(), 
                     key=lambda x: similarities[x], 
                     reverse=True)[:k]

# Unsupervised alignment (adversarial)
import torch
import torch.nn as nn

class AdversarialAligner:
    """Unsupervised cross-lingual alignment via adversarial training"""
    
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        self.discriminator = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def train(self, source_embs: np.ndarray, target_embs: np.ndarray,
              epochs: int = 5, batch_size: int = 32):
        """
        Train alignment via adversarial game:
        - W tries to make aligned source look like target
        - Discriminator tries to distinguish source from target
        """
        X = torch.FloatTensor(source_embs)
        Z = torch.FloatTensor(target_embs)
        
        opt_W = torch.optim.Adam(self.W.parameters(), lr=1e-4)
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            # Train discriminator
            for _ in range(5):
                idx_x = np.random.choice(len(X), batch_size)
                idx_z = np.random.choice(len(Z), batch_size)
                
                aligned = self.W(X[idx_x])
                target = Z[idx_z]
                
                # D should predict 0 for aligned source, 1 for target
                pred_aligned = self.discriminator(aligned.detach())
                pred_target = self.discriminator(target)
                
                loss_D = -torch.log(1 - pred_aligned + 1e-8).mean() - \
                         torch.log(pred_target + 1e-8).mean()
                
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
            
            # Train W to fool discriminator
            idx_x = np.random.choice(len(X), batch_size)
            aligned = self.W(X[idx_x])
            pred = self.discriminator(aligned)
            
            loss_W = -torch.log(pred + 1e-8).mean()  # Fool D
            
            opt_W.zero_grad()
            loss_W.backward()
            opt_W.step()
            
            # Orthogonalize W (Procrustes refinement)
            with torch.no_grad():
                U, _, V = torch.svd(self.W.weight.data)
                self.W.weight.data = U @ V.T
        
        return self.W.weight.data.numpy()

# CSLS (Cross-domain Similarity Local Scaling) for better retrieval
def csls_similarity(source_vec: np.ndarray, 
                    target_embeddings: Dict[str, np.ndarray],
                    k: int = 10) -> Dict[str, float]:
    """
    CSLS: Reduces hubness problem in cross-lingual retrieval
    """
    # Compute all similarities
    sims = {}
    for word, vec in target_embeddings.items():
        sims[word] = np.dot(source_vec, vec) / (
            np.linalg.norm(source_vec) * np.linalg.norm(vec) + 1e-8
        )
    
    # Mean similarity of target words to their k nearest source neighbors
    # (In practice, precompute this for all target words)
    
    # CSLS score: 2*cos(s,t) - r_T(t) - r_S(s)
    # Simplified version here
    return sims

# Evaluation
def evaluate_alignment(aligned_source: Dict[str, np.ndarray],
                       target: Dict[str, np.ndarray],
                       test_dict: List[Tuple[str, str]]) -> Dict:
    """Evaluate word translation accuracy"""
    
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    for src, tgt in test_dict:
        if src not in aligned_source or tgt not in target:
            continue
        
        total += 1
        src_aligned = aligned_source[src]
        
        # Find nearest neighbors
        sims = [(w, np.dot(src_aligned, v) / (np.linalg.norm(src_aligned) * np.linalg.norm(v) + 1e-8))
                for w, v in target.items()]
        sims.sort(key=lambda x: x[1], reverse=True)
        
        top_5 = [w for w, _ in sims[:5]]
        
        if top_5[0] == tgt:
            correct_1 += 1
        if tgt in top_5:
            correct_5 += 1
    
    return {
        'P@1': correct_1 / total if total > 0 else 0,
        'P@5': correct_5 / total if total > 0 else 0,
        'total_evaluated': total
    }
```

**Interview Tips:**
- Procrustes is simple and effective with ~5K dictionary pairs
- mBERT/XLM-RoBERTa provide aligned spaces out-of-the-box
- CSLS retrieval metric handles hubness (some words are NN of many)
- Distant language pairs (EN-ZH, EN-AR) are harder to align

---

## Embedding Storage & Retrieval

### Question 14
**How do you implement efficient embedding storage and retrieval systems for large-scale applications?**

**Answer:**

**Definition:**
Efficient embedding storage uses: **vector databases** (FAISS, Milvus, Pinecone) with **ANN indices** (HNSW, IVF), **quantization** for compression, **sharding** for scale, and **caching** for frequent queries. Balance: recall@k vs latency vs storage cost. Target: sub-100ms retrieval at billion-scale.

**Storage Architecture:**

| Component | Purpose | Options |
|-----------|---------|---------|
| **Index** | Fast similarity search | HNSW, IVF, LSH |
| **Storage** | Persist vectors | Disk, memory-mapped |
| **Compression** | Reduce size | PQ, SQ, binary |
| **Sharding** | Horizontal scale | Hash, range |
| **Cache** | Reduce latency | Redis, in-memory |

**Scaling Considerations:**

| Scale | Vectors | Approach | Latency |
|-------|---------|----------|---------|
| **Small** | <100K | FAISS in-memory | <5ms |
| **Medium** | 100K-10M | Single node + IVF | <20ms |
| **Large** | 10M-100M | Sharded + PQ | <50ms |
| **Massive** | >100M | Distributed cluster | <100ms |

**Python Code Example:**
```python
import numpy as np
import faiss
from typing import List, Tuple, Dict
import pickle
import os

class EmbeddingStore:
    """Production-ready embedding storage and retrieval"""
    
    def __init__(self, dim: int, index_type: str = 'hnsw',
                 use_gpu: bool = False):
        self.dim = dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.index = None
        self.id_map = {}  # External ID -> Internal ID
        self.reverse_map = {}  # Internal ID -> External ID
        self.metadata = {}  # External ID -> metadata
    
    def build_index(self, embeddings: np.ndarray, ids: List[str] = None):
        """Build index from embeddings"""
        n = len(embeddings)
        
        if ids is None:
            ids = [str(i) for i in range(n)]
        
        # Create ID mappings
        for i, ext_id in enumerate(ids):
            self.id_map[ext_id] = i
            self.reverse_map[i] = ext_id
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build index based on type
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatIP(self.dim)
        
        elif self.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(self.dim, 32)  # 32 neighbors
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64
        
        elif self.index_type == 'ivf':
            nlist = min(int(np.sqrt(n)), 4096)
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self.index.train(embeddings)
            self.index.nprobe = min(nlist // 4, 64)
        
        elif self.index_type == 'ivf_pq':
            nlist = min(int(np.sqrt(n)), 4096)
            m = 32  # Number of subquantizers
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, 8)
            self.index.train(embeddings)
        
        # Add vectors
        self.index.add(embeddings)
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors"""
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Normalize query
        faiss.normalize_L2(query)
        
        # Search
        distances, indices = self.index.search(query, k)
        
        # Map back to external IDs
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # -1 indicates not found
                ext_id = self.reverse_map.get(idx, str(idx))
                results.append((ext_id, float(dist)))
        
        return results
    
    def batch_search(self, queries: np.ndarray, k: int = 10) -> List[List[Tuple[str, float]]]:
        """Batch search for efficiency"""
        faiss.normalize_L2(queries)
        distances, indices = self.index.search(queries, k)
        
        all_results = []
        for idx_row, dist_row in zip(indices, distances):
            results = []
            for idx, dist in zip(idx_row, dist_row):
                if idx >= 0:
                    ext_id = self.reverse_map.get(idx, str(idx))
                    results.append((ext_id, float(dist)))
            all_results.append(results)
        
        return all_results
    
    def add(self, embedding: np.ndarray, ext_id: str, metadata: Dict = None):
        """Add single embedding"""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        faiss.normalize_L2(embedding)
        
        internal_id = self.index.ntotal
        self.id_map[ext_id] = internal_id
        self.reverse_map[internal_id] = ext_id
        
        if metadata:
            self.metadata[ext_id] = metadata
        
        self.index.add(embedding)
    
    def save(self, path: str):
        """Save index and mappings"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, os.path.join(path, 'index.faiss'))
        else:
            faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        
        # Save mappings
        with open(os.path.join(path, 'mappings.pkl'), 'wb') as f:
            pickle.dump({
                'id_map': self.id_map,
                'reverse_map': self.reverse_map,
                'metadata': self.metadata,
                'dim': self.dim,
                'index_type': self.index_type
            }, f)
    
    def load(self, path: str):
        """Load index and mappings"""
        self.index = faiss.read_index(os.path.join(path, 'index.faiss'))
        
        with open(os.path.join(path, 'mappings.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.id_map = data['id_map']
            self.reverse_map = data['reverse_map']
            self.metadata = data['metadata']
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )

# Sharded store for massive scale
class ShardedEmbeddingStore:
    """Distribute embeddings across multiple shards"""
    
    def __init__(self, num_shards: int, dim: int):
        self.num_shards = num_shards
        self.shards = [EmbeddingStore(dim) for _ in range(num_shards)]
    
    def _get_shard(self, ext_id: str) -> int:
        """Hash-based shard assignment"""
        return hash(ext_id) % self.num_shards
    
    def add(self, embedding: np.ndarray, ext_id: str, metadata: Dict = None):
        """Add to appropriate shard"""
        shard_id = self._get_shard(ext_id)
        self.shards[shard_id].add(embedding, ext_id, metadata)
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search all shards and merge results"""
        all_results = []
        
        for shard in self.shards:
            results = shard.search(query, k)
            all_results.extend(results)
        
        # Sort by score and take top-k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]
```

**Interview Tips:**
- HNSW is best for high recall; IVF+PQ for memory efficiency
- Always normalize embeddings for cosine similarity with inner product index
- Consider GPU for query-heavy workloads; CPU for memory-heavy
- Sharding strategy affects load balancing — hash by ID or cluster by embedding

---

### Question 15
**What techniques help optimize memory usage when working with large embedding matrices?**

**Answer:**

**Definition:**
Optimize embedding memory via: **quantization** (FP16/INT8 reduce 2-4x), **dimensionality reduction** (PCA), **sparse representations**, **memory mapping** (load on demand), **product quantization** (32x compression), and **pruning** (remove infrequent embeddings). Goal: fit billion-scale embeddings in available memory.

**Memory Reduction Techniques:**

| Technique | Compression | Quality Loss | Complexity |
|-----------|-------------|--------------|------------|
| **FP32 → FP16** | 2x | ~0% | Low |
| **FP32 → INT8** | 4x | 1-2% | Medium |
| **PQ (m=32)** | 32x | 5-10% | Medium |
| **Binary** | 32x | 10-20% | Low |
| **Dim reduction** | 2-4x | 2-5% | Low |
| **Pruning** | Variable | 0% (on kept) | Low |

**Memory Calculation:**
```
Full: N vectors × D dimensions × 4 bytes = 4ND bytes

1B vectors, 768-dim FP32: 1B × 768 × 4 = 3TB
1B vectors, 768-dim FP16: 1B × 768 × 2 = 1.5TB  
1B vectors, 768-dim INT8: 1B × 768 × 1 = 768GB
1B vectors, 768-dim PQ32: 1B × 32 × 1 = 32GB
```

**Python Code Example:**
```python
import numpy as np
import faiss
from typing import Optional
import mmap
import os

class MemoryEfficientEmbeddings:
    """Memory-optimized embedding storage"""
    
    def __init__(self, dim: int, dtype: str = 'float32'):
        self.dim = dim
        self.dtype = dtype
        self.dtype_map = {
            'float32': np.float32,
            'float16': np.float16,
            'int8': np.int8
        }
    
    def quantize_to_float16(self, embeddings: np.ndarray) -> np.ndarray:
        """Convert to FP16 (50% memory reduction)"""
        return embeddings.astype(np.float16)
    
    def quantize_to_int8(self, embeddings: np.ndarray) -> tuple:
        """
        Symmetric INT8 quantization (75% memory reduction)
        Returns quantized values and scale factors
        """
        # Compute scale per vector
        max_vals = np.abs(embeddings).max(axis=1, keepdims=True)
        scales = max_vals / 127.0
        
        # Quantize
        quantized = np.round(embeddings / (scales + 1e-8)).astype(np.int8)
        
        return quantized, scales.astype(np.float32)
    
    def dequantize_int8(self, quantized: np.ndarray, 
                        scales: np.ndarray) -> np.ndarray:
        """Dequantize INT8 back to float"""
        return quantized.astype(np.float32) * scales
    
    def product_quantization(self, embeddings: np.ndarray, 
                             m: int = 32, nbits: int = 8) -> tuple:
        """
        Product Quantization for extreme compression
        Splits vector into m subvectors, quantizes each to nbits
        """
        n, d = embeddings.shape
        
        # Train PQ
        pq = faiss.ProductQuantizer(d, m, nbits)
        pq.train(embeddings.astype(np.float32))
        
        # Encode
        codes = pq.compute_codes(embeddings.astype(np.float32))
        
        return codes, pq
    
    def binary_quantization(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Binary quantization (32x compression)
        Sign of each dimension → 1 bit
        """
        # Convert to binary: positive → 1, negative → 0
        binary = (embeddings > 0).astype(np.uint8)
        
        # Pack bits
        packed = np.packbits(binary, axis=1)
        
        return packed
    
    def hamming_distance(self, query_packed: np.ndarray, 
                         db_packed: np.ndarray) -> np.ndarray:
        """Compute Hamming distance for binary vectors"""
        # XOR and popcount
        xor = np.bitwise_xor(query_packed, db_packed)
        return np.unpackbits(xor, axis=1).sum(axis=1)

class MemoryMappedEmbeddings:
    """Memory-mapped embeddings for out-of-core processing"""
    
    def __init__(self, filepath: str, dim: int, dtype: np.dtype = np.float32):
        self.filepath = filepath
        self.dim = dim
        self.dtype = dtype
        self.mmap = None
    
    def create(self, embeddings: np.ndarray):
        """Create memory-mapped file"""
        fp = np.memmap(
            self.filepath,
            dtype=self.dtype,
            mode='w+',
            shape=embeddings.shape
        )
        fp[:] = embeddings[:]
        fp.flush()
        del fp
    
    def load(self, n_vectors: Optional[int] = None):
        """Load memory-mapped file"""
        # Get file size to determine number of vectors
        file_size = os.path.getsize(self.filepath)
        bytes_per_vector = self.dim * np.dtype(self.dtype).itemsize
        total_vectors = file_size // bytes_per_vector
        
        n = n_vectors or total_vectors
        
        self.mmap = np.memmap(
            self.filepath,
            dtype=self.dtype,
            mode='r',
            shape=(n, self.dim)
        )
        return self.mmap
    
    def get(self, indices: np.ndarray) -> np.ndarray:
        """Get specific embeddings by index"""
        return np.array(self.mmap[indices])
    
    def search_batch(self, query: np.ndarray, batch_size: int = 10000,
                     k: int = 10) -> list:
        """Search large mmap in batches"""
        n_total = len(self.mmap)
        
        # Normalize query
        query = query / (np.linalg.norm(query) + 1e-8)
        
        all_scores = []
        all_indices = []
        
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch = np.array(self.mmap[start:end])
            
            # Normalize batch
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            batch_norm = batch / (norms + 1e-8)
            
            # Compute similarities
            scores = batch_norm @ query
            
            # Get top-k for this batch
            top_k = min(k, len(scores))
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            
            for idx in top_indices:
                all_scores.append(scores[idx])
                all_indices.append(start + idx)
        
        # Global top-k
        combined = list(zip(all_indices, all_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined[:k]

# Sparse embeddings for efficiency
class SparseEmbeddings:
    """Use sparse representations when applicable"""
    
    def __init__(self, dim: int, sparsity: float = 0.9):
        self.dim = dim
        self.sparsity = sparsity
    
    def sparsify(self, embeddings: np.ndarray, top_k: int = None) -> list:
        """Keep only top-k dimensions per vector"""
        if top_k is None:
            top_k = int(self.dim * (1 - self.sparsity))
        
        sparse_vectors = []
        for vec in embeddings:
            # Get top-k indices
            top_indices = np.argpartition(np.abs(vec), -top_k)[-top_k:]
            
            # Create sparse representation
            sparse = [(int(idx), float(vec[idx])) for idx in top_indices]
            sparse_vectors.append(sparse)
        
        return sparse_vectors
    
    def sparse_dot_product(self, sparse1: list, sparse2: list) -> float:
        """Compute dot product of two sparse vectors"""
        dict1 = dict(sparse1)
        
        result = 0.0
        for idx, val in sparse2:
            if idx in dict1:
                result += dict1[idx] * val
        
        return result
```

**Interview Tips:**
- FP16 is almost always safe; INT8 needs calibration
- PQ is best for read-heavy workloads (slower to add)
- Memory mapping enables larger-than-RAM datasets
- Combine techniques: PQ on disk, cache hot vectors in memory

---

### Question 16
**How do you implement embedding compression techniques for mobile and edge deployments?**

**Answer:**

**Definition:**
Edge deployment compression uses: **aggressive quantization** (INT4/binary), **knowledge distillation** (smaller models), **pruning** (remove dimensions), **vocabulary reduction**, and **on-device caching**. Target: <50MB model size, <100ms latency on mobile CPU, minimal quality degradation.

**Mobile Constraints:**

| Constraint | Typical Limit | Impact |
|------------|---------------|--------|
| **Model size** | 10-100MB | Limits vocabulary |
| **RAM** | 2-4GB (shared) | Can't load full embeddings |
| **Compute** | 2-4 TOPS | Limits model complexity |
| **Latency** | <100ms | Requires optimization |
| **Battery** | Limited | Favor smaller models |

**Compression Strategies for Mobile:**

| Technique | Compression | Quality Loss | Mobile-Friendly |
|-----------|-------------|--------------|-----------------|
| **INT8** | 4x | ~1% | ✓ Very |
| **INT4** | 8x | 3-5% | ✓ Good |
| **Binary** | 32x | 10-20% | ✓ Best perf |
| **Distillation** | 4-10x | 2-5% | ✓ Good |
| **Vocab pruning** | Variable | 0% (kept) | ✓ Very |

**Python Code Example:**
```python
import numpy as np
from typing import Dict, List, Tuple
import struct

class MobileEmbeddingCompressor:
    """Compress embeddings for mobile/edge deployment"""
    
    def __init__(self, embeddings: Dict[str, np.ndarray], dim: int = 300):
        self.embeddings = embeddings
        self.dim = dim
    
    def int8_quantization(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """INT8 quantization with single global scale"""
        # Find global scale
        all_values = np.concatenate([v for v in self.embeddings.values()])
        global_scale = np.abs(all_values).max() / 127.0
        
        quantized = {}
        for word, vec in self.embeddings.items():
            quantized[word] = np.round(vec / global_scale).astype(np.int8)
        
        return quantized, global_scale
    
    def int4_quantization(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """INT4 quantization (pack two values per byte)"""
        all_values = np.concatenate([v for v in self.embeddings.values()])
        global_scale = np.abs(all_values).max() / 7.0  # 4-bit signed: -8 to 7
        
        quantized = {}
        for word, vec in self.embeddings.items():
            q = np.clip(np.round(vec / global_scale), -8, 7).astype(np.int8)
            # Pack two values per byte
            packed = []
            for i in range(0, len(q), 2):
                high = (q[i] + 8) << 4
                low = (q[i+1] + 8) if i+1 < len(q) else 0
                packed.append(high | low)
            quantized[word] = np.array(packed, dtype=np.uint8)
        
        return quantized, global_scale
    
    def binary_quantization(self) -> Dict[str, np.ndarray]:
        """Binary quantization (1 bit per dimension)"""
        binary = {}
        for word, vec in self.embeddings.items():
            # Sign → binary
            bits = (vec > 0).astype(np.uint8)
            packed = np.packbits(bits)
            binary[word] = packed
        
        return binary
    
    def vocabulary_pruning(self, word_freq: Dict[str, int], 
                           top_k: int = 50000) -> Dict[str, np.ndarray]:
        """Keep only most frequent words"""
        sorted_words = sorted(word_freq.keys(), 
                             key=lambda w: word_freq[w], 
                             reverse=True)[:top_k]
        
        pruned = {w: self.embeddings[w] for w in sorted_words 
                  if w in self.embeddings}
        
        return pruned
    
    def dimension_reduction(self, target_dim: int = 128) -> Dict[str, np.ndarray]:
        """PCA-based dimension reduction"""
        from sklearn.decomposition import PCA
        
        words = list(self.embeddings.keys())
        vectors = np.array([self.embeddings[w] for w in words])
        
        pca = PCA(n_components=target_dim)
        reduced = pca.fit_transform(vectors)
        
        return {w: reduced[i] for i, w in enumerate(words)}, pca

class MobileEmbeddingExporter:
    """Export embeddings for mobile platforms"""
    
    def __init__(self, embeddings: Dict[str, np.ndarray], 
                 quantized: bool = True):
        self.embeddings = embeddings
        self.quantized = quantized
    
    def export_flatbuffer(self, output_path: str):
        """Export to FlatBuffer format (efficient for mobile)"""
        # Simplified - actual implementation uses FlatBuffers library
        words = list(self.embeddings.keys())
        vectors = np.array([self.embeddings[w] for w in words])
        
        with open(output_path, 'wb') as f:
            # Write header
            f.write(struct.pack('II', len(words), vectors.shape[1]))
            
            # Write vocabulary
            for word in words:
                encoded = word.encode('utf-8')
                f.write(struct.pack('H', len(encoded)))
                f.write(encoded)
            
            # Write vectors
            if self.quantized:
                f.write(vectors.astype(np.int8).tobytes())
            else:
                f.write(vectors.astype(np.float32).tobytes())
    
    def export_tflite_compatible(self, output_path: str):
        """Export format compatible with TFLite"""
        # Convert to TFLite lookup table
        words = list(self.embeddings.keys())
        vectors = np.array([self.embeddings[w] for w in words])
        
        # Save as numpy for conversion
        np.savez(output_path, 
                 vocabulary=np.array(words),
                 embeddings=vectors.astype(np.float16))

class MobileInference:
    """Lightweight inference for mobile"""
    
    def __init__(self, embeddings_path: str, quantized: bool = True):
        self.embeddings, self.scale = self._load(embeddings_path, quantized)
    
    def _load(self, path: str, quantized: bool):
        """Load compressed embeddings"""
        data = np.load(path, allow_pickle=True)
        embeddings = dict(zip(data['vocabulary'], data['embeddings']))
        scale = data.get('scale', 1.0)
        return embeddings, scale
    
    def get_embedding(self, word: str) -> np.ndarray:
        """Get embedding with dequantization"""
        if word not in self.embeddings:
            return None
        
        vec = self.embeddings[word]
        return vec.astype(np.float32) * self.scale
    
    def similarity(self, word1: str, word2: str) -> float:
        """Compute similarity (optimized for quantized)"""
        v1 = self.embeddings.get(word1)
        v2 = self.embeddings.get(word2)
        
        if v1 is None or v2 is None:
            return 0.0
        
        # Integer dot product (faster on mobile)
        if v1.dtype == np.int8:
            dot = np.dot(v1.astype(np.int32), v2.astype(np.int32))
            norm1 = np.sqrt(np.dot(v1.astype(np.int32), v1.astype(np.int32)))
            norm2 = np.sqrt(np.dot(v2.astype(np.int32), v2.astype(np.int32)))
            return dot / (norm1 * norm2 + 1e-8)
        else:
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
```

**Interview Tips:**
- INT8 is safe for most tasks; INT4 needs evaluation
- Vocabulary pruning to 50K words often sufficient
- Binary embeddings work well for candidate retrieval, re-rank with full
- Consider on-device model for privacy-sensitive applications

---

### Question 17
**How do you optimize embedding-based retrieval systems for sub-second response times?**

**Answer:**

**Definition:**
Achieve sub-second retrieval via: **ANN indices** (HNSW for speed, IVF for memory), **caching** (query and result caching), **pre-filtering** (metadata filters), **batching** (group concurrent queries), **hardware optimization** (GPU, SIMD), and **index partitioning**. Target: P99 < 100ms at scale.

**Latency Budget Breakdown:**

| Component | Target Latency | Optimization |
|-----------|----------------|--------------|
| **Query embedding** | 10-50ms | Distilled model, ONNX |
| **Vector search** | 5-20ms | HNSW, GPU index |
| **Post-processing** | 5-10ms | Efficient filtering |
| **Network** | 10-30ms | CDN, edge deployment |
| **Total** | <100ms | End-to-end |

**Optimization Strategies:**

| Strategy | Latency Impact | Implementation |
|----------|----------------|----------------|
| **HNSW index** | 5-10ms search | ef_search tuning |
| **Query caching** | <1ms (hit) | Redis/Memcached |
| **Result caching** | <1ms (hit) | LRU cache |
| **Pre-filtering** | 2-5x speedup | Metadata index |
| **GPU acceleration** | 2-5x speedup | FAISS GPU |
| **Batching** | Higher throughput | Async collection |

**Python Code Example:**
```python
import numpy as np
import faiss
import time
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import hashlib
import asyncio
from collections import defaultdict

class OptimizedRetrievalSystem:
    """Sub-second retrieval with multiple optimizations"""
    
    def __init__(self, dim: int, use_gpu: bool = True):
        self.dim = dim
        self.use_gpu = use_gpu
        self.index = None
        self.query_cache = {}
        self.result_cache = {}
        self.metadata_index = {}
        
    def build_optimized_index(self, embeddings: np.ndarray, 
                               metadata: List[Dict] = None):
        """Build HNSW index optimized for speed"""
        n = len(embeddings)
        
        # Normalize for inner product (cosine)
        faiss.normalize_L2(embeddings)
        
        # HNSW with tuned parameters
        self.index = faiss.IndexHNSWFlat(self.dim, 32)
        self.index.hnsw.efConstruction = 200  # Build quality
        self.index.hnsw.efSearch = 64  # Search quality/speed tradeoff
        
        self.index.add(embeddings)
        
        # Move to GPU
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Build metadata index for pre-filtering
        if metadata:
            for i, meta in enumerate(metadata):
                for key, value in meta.items():
                    if key not in self.metadata_index:
                        self.metadata_index[key] = defaultdict(list)
                    self.metadata_index[key][value].append(i)
    
    def _get_query_hash(self, query: np.ndarray) -> str:
        """Hash query for caching"""
        return hashlib.md5(query.tobytes()).hexdigest()
    
    def search_with_cache(self, query: np.ndarray, k: int = 10,
                          filters: Dict = None) -> List[Tuple[int, float]]:
        """Search with query and result caching"""
        
        # Check cache
        cache_key = (self._get_query_hash(query), k, str(filters))
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]
        
        # Pre-filter by metadata
        candidate_ids = None
        if filters:
            candidate_ids = self._apply_filters(filters)
            if len(candidate_ids) == 0:
                return []
        
        # Search
        query = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        if candidate_ids is not None:
            # Search only within candidates
            results = self._search_subset(query, candidate_ids, k)
        else:
            distances, indices = self.index.search(query, k)
            results = list(zip(indices[0], distances[0]))
        
        # Cache result
        if len(self.result_cache) < 10000:  # Limit cache size
            self.result_cache[cache_key] = results
        
        return results
    
    def _apply_filters(self, filters: Dict) -> set:
        """Apply metadata filters to get candidate IDs"""
        candidate_sets = []
        
        for key, value in filters.items():
            if key in self.metadata_index and value in self.metadata_index[key]:
                candidate_sets.append(set(self.metadata_index[key][value]))
        
        if not candidate_sets:
            return set()
        
        # Intersection of all filters
        return set.intersection(*candidate_sets)
    
    def _search_subset(self, query: np.ndarray, 
                       candidate_ids: set, k: int) -> List[Tuple[int, float]]:
        """Search within subset of vectors"""
        # For small subsets, brute force
        if len(candidate_ids) < 1000:
            # This is a simplified version
            # Real implementation would use IDSelector
            pass
        
        # For larger subsets, use IDSelectorArray
        ids = np.array(list(candidate_ids), dtype=np.int64)
        selector = faiss.IDSelectorArray(len(ids), faiss.swig_ptr(ids))
        
        # Create filtered index
        params = faiss.SearchParametersHNSW()
        params.sel = selector
        
        distances, indices = self.index.search(query, k, params=params)
        return list(zip(indices[0], distances[0]))

class BatchedRetrieval:
    """Batch queries for higher throughput"""
    
    def __init__(self, retrieval_system, batch_size: int = 32,
                 max_wait_ms: float = 10.0):
        self.system = retrieval_system
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_queries = []
        self.results = {}
    
    async def search_async(self, query: np.ndarray, k: int = 10) -> List:
        """Async search with batching"""
        query_id = id(query)
        future = asyncio.Future()
        
        self.pending_queries.append({
            'id': query_id,
            'query': query,
            'k': k,
            'future': future
        })
        
        # Trigger batch if full
        if len(self.pending_queries) >= self.batch_size:
            await self._process_batch()
        else:
            # Wait for batch or timeout
            await asyncio.sleep(self.max_wait_ms / 1000)
            if self.pending_queries:
                await self._process_batch()
        
        return await future
    
    async def _process_batch(self):
        """Process accumulated queries as batch"""
        if not self.pending_queries:
            return
        
        # Collect batch
        batch = self.pending_queries[:self.batch_size]
        self.pending_queries = self.pending_queries[self.batch_size:]
        
        # Stack queries
        queries = np.vstack([q['query'] for q in batch])
        max_k = max(q['k'] for q in batch)
        
        # Batch search
        faiss.normalize_L2(queries)
        distances, indices = self.system.index.search(queries, max_k)
        
        # Distribute results
        for i, q in enumerate(batch):
            k = q['k']
            results = list(zip(indices[i][:k], distances[i][:k]))
            q['future'].set_result(results)

# Performance monitoring
class RetrievalMonitor:
    """Monitor retrieval latency"""
    
    def __init__(self):
        self.latencies = []
    
    def timed_search(self, system, query: np.ndarray, k: int = 10):
        """Search with timing"""
        start = time.perf_counter()
        results = system.search_with_cache(query, k)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        self.latencies.append(elapsed)
        return results, elapsed
    
    def get_stats(self) -> Dict:
        """Get latency statistics"""
        if not self.latencies:
            return {}
        
        return {
            'p50': np.percentile(self.latencies, 50),
            'p95': np.percentile(self.latencies, 95),
            'p99': np.percentile(self.latencies, 99),
            'mean': np.mean(self.latencies),
            'count': len(self.latencies)
        }
```

**Interview Tips:**
- HNSW ef_search of 64-128 balances speed and recall
- Cache hit rates of 60-80% are common for production search
- Pre-filtering before vector search is critical for large collections
- Batch queries when throughput matters more than latency

---

### Question 18
**What are the best practices for handling embedding drift in production systems over time?**
**Answer:** _To be filled_

---

## Vector Databases - Selection & Architecture

### Question 19
**How do you choose between FAISS, ChromaDB, Pinecone, Weaviate, and Milvus based on your requirements?**
**Answer:** _To be filled_

---

### Question 20
**What are the trade-offs between self-hosted (FAISS/ChromaDB) versus managed solutions (Pinecone)?**
**Answer:** _To be filled_

---

### Question 21
**How do you optimize index selection (IVF, HNSW, LSH, PQ) in FAISS for different query patterns?**
**Answer:** _To be filled_

---

### Question 22
**Explain HNSW (Hierarchical Navigable Small World) graphs and when they outperform other indices.**
**Answer:** _To be filled_

---

### Question 23
**When should you use approximate nearest neighbor (ANN) search versus exact search?**
**Answer:** _To be filled_

---

### Question 24
**How do you implement hybrid search combining vector similarity with traditional keyword (BM25) search?**
**Answer:** _To be filled_

---

## Vector Database Operations

### Question 25
**How do you handle incremental updates and real-time indexing in production vector databases?**
**Answer:** _To be filled_

---

### Question 26
**What strategies help optimize memory usage and disk I/O for large vector datasets?**
**Answer:** _To be filled_

---

### Question 27
**How do you implement vector compression and Product Quantization (PQ) to balance storage and accuracy?**
**Answer:** _To be filled_

---

### Question 28
**How do you handle data partitioning and sharding strategies for large-scale vector collections?**
**Answer:** _To be filled_

---

### Question 29
**How do you handle multi-tenancy and data isolation requirements in vector database deployments?**
**Answer:** _To be filled_

---

### Question 30
**What techniques help ensure vector database performance consistency under varying load?**
**Answer:** _To be filled_

---

## Production & Scaling

### Question 31
**How do you implement effective backup and disaster recovery for vector databases?**
**Answer:** _To be filled_

---

### Question 32
**How do you implement load balancing and horizontal scaling for vector database clusters?**
**Answer:** _To be filled_

---

### Question 33
**What are the best practices for monitoring and alerting in vector database production environments?**
**Answer:** _To be filled_

---

### Question 34
**How do you implement effective caching strategies to improve vector search response times?**
**Answer:** _To be filled_

---

### Question 35
**How do you optimize batch operations versus real-time queries in vector databases?**
**Answer:** _To be filled_

---

### Question 36
**What techniques help handle schema evolution and vector format changes in production?**
**Answer:** _To be filled_

---

## Evaluation & Benchmarking

### Question 37
**How do you evaluate embedding quality using both intrinsic (analogy tasks) and extrinsic (downstream tasks) methods?**
**Answer:** _To be filled_

---

### Question 38
**How do you handle the evaluation and benchmarking challenges when comparing vector database performance?**
**Answer:** _To be filled_

---

### Question 39
**What metrics (recall@k, QPS, latency percentiles) are most important for vector search evaluation?**
**Answer:** _To be filled_

---

## Integration & Applications

### Question 40
**How do you optimize vector database integration with RAG (Retrieval-Augmented Generation) systems?**
**Answer:** _To be filled_

---

### Question 41
**What are the considerations for implementing embeddings in recommendation systems at scale?**
**Answer:** _To be filled_

---

### Question 42
**How do you optimize vector database performance for semantic search applications?**
**Answer:** _To be filled_

---

### Question 43
**How do you handle the interpretability challenges when using dense embeddings in explainable AI?**
**Answer:** _To be filled_

---

### Question 44
**When would you implement multi-vector storage versus single embedding per document?**
**Answer:** _To be filled_

---

### Question 45
**How do you implement embedding visualization techniques (t-SNE, UMAP) for model debugging?**
**Answer:** _To be filled_

---

## Security & Compliance

### Question 46
**How do you implement effective security measures and access control for vector database deployments?**
**Answer:** _To be filled_

---

### Question 47
**What are the considerations for using embeddings in privacy-preserving applications?**
**Answer:** _To be filled_

---

### Question 48
**How do you implement data governance and compliance measures for vector database systems?**
**Answer:** _To be filled_

---

## Advanced Topics

### Question 49
**How do you implement version control and reproducibility for embedding-based systems?**
**Answer:** _To be filled_

---

### Question 50
**When would you implement federated search across multiple vector databases versus centralized storage?**
**Answer:** _To be filled_

---

### Question 51
**How do you optimize embeddings for multimodal use cases (image-text, code similarity)?**
**Answer:** _To be filled_

---

### Question 52
**What strategies help future-proof vector database architectures for evolving AI requirements?**
**Answer:** _To be filled_

---

### Question 53
**How do you handle temporal aspects of embeddings when working with evolving vocabularies?**
**Answer:** _To be filled_

---

### Question 54
**What techniques help optimize embedding training for domain-specific corpora with limited data?**
**Answer:** _To be filled_

---

### Question 55
**How do you implement effective embedding caching strategies to reduce computational overhead?**
**Answer:** _To be filled_

---
