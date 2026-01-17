# Text Generation (Seq2Seq, Translation, QA, Summarization) - Theory Questions

## Core Questions

## Question 1
**What is the encoder-decoder architecture and how does it enable sequence-to-sequence learning?**

**Answer:**

**Definition:**
Encoder-decoder is a neural architecture where an encoder processes input sequence into a fixed context representation, and a decoder generates output sequence from that context. Enables variable-length input to variable-length output mapping for translation, summarization, etc.

**Architecture:**

```
Input: "Hello world" → [Encoder] → Context Vector → [Decoder] → "Bonjour monde"
```

| Component | Role |
|-----------|------|
| Encoder | Process input, create context representation |
| Context | Fixed-size summary of input |
| Decoder | Generate output token by token |

**How It Works:**
1. Encoder (LSTM/Transformer) reads input tokens
2. Final hidden state becomes context vector
3. Decoder uses context to generate output one token at a time
4. Each decoder step conditions on previous output + context

**Python Code Example:**
```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell  # Context

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg):
        hidden, cell = self.encoder(src)
        outputs = []
        for t in range(trg.size(1)):
            inp = trg[:, t:t+1]
            output, hidden, cell = self.decoder(inp, hidden, cell)
            outputs.append(output)
        return torch.cat(outputs, dim=1)
```

**Interview Tips:**
- Bottleneck: fixed-size context loses information for long sequences
- Attention mechanism solves this bottleneck
- Modern: Transformers are encoder-decoder without recurrence
- BART, T5, mBART are pre-trained encoder-decoder models

---

## Question 2
**Explain the attention mechanism in seq2seq models. Why was it a breakthrough?**

**Answer:**

**Definition:**
Attention allows the decoder to dynamically focus on relevant parts of the input at each generation step, instead of relying on a single fixed context vector. It computes weighted sum of encoder states, where weights depend on decoder's current state.

**Why It's a Breakthrough:**
- Solves information bottleneck of fixed context
- Handles long sequences without degradation
- Provides interpretable alignment (which input → which output)
- Foundation for Transformer architecture

**How It Works:**

$$\alpha_{t,i} = \frac{\exp(score(s_t, h_i))}{\sum_j \exp(score(s_t, h_j))}$$
$$c_t = \sum_i \alpha_{t,i} h_i$$

where $s_t$ = decoder state, $h_i$ = encoder states, $c_t$ = context for step t.

**Score Functions:**

| Type | Formula |
|------|--------|
| Dot | $s_t^T h_i$ |
| General | $s_t^T W h_i$ |
| Concat (Bahdanau) | $v^T \tanh(W[s_t; h_i])$ |

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch, hidden]
        # encoder_outputs: [batch, seq_len, hidden]
        
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concat decoder state with each encoder output
        energy = torch.tanh(self.attn(torch.cat([decoder_hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        
        # Softmax to get weights
        weights = F.softmax(attention, dim=1)
        
        # Weighted sum of encoder outputs
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, weights

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.attention = Attention(hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.embedding(x)
        context, weights = self.attention(hidden.squeeze(0), encoder_outputs)
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell, weights
```

**Interview Tips:**
- Bahdanau (2014) introduced attention for NMT
- Attention weights are interpretable - show alignment
- "Attention is All You Need" removed recurrence entirely
- Cross-attention: decoder attending to encoder (seq2seq)
- Self-attention: sequence attending to itself (Transformer)

---

## Question 3
**What is the Transformer architecture? Explain self-attention and multi-head attention.**

**Answer:**

**Definition:**
Transformer is a neural architecture using only attention mechanisms (no recurrence or convolution). Self-attention lets each token attend to all other tokens. Multi-head attention runs multiple attention functions in parallel to capture different relationship types.

**Key Components:**

| Component | Purpose |
|-----------|--------|
| Self-attention | Token-to-token relationships |
| Multi-head attention | Multiple attention patterns |
| Feed-forward | Position-wise transformations |
| Layer norm | Stabilize training |
| Positional encoding | Inject position information |

**Self-Attention (Scaled Dot-Product):**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- Q (Query): what to look for
- K (Key): what's available
- V (Value): what to retrieve
- Scale by $\sqrt{d_k}$ to prevent large dot products

**Multi-Head Attention:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O$$
$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Each head can learn different relationship patterns.

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)  # [batch, seq, d_model]
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)
```

**Interview Tips:**
- Transformer enables parallelization (no sequential dependency)
- 8 heads typical for base models, 16 for large
- BERT = encoder-only, GPT = decoder-only, T5/BART = encoder-decoder
- Positional encoding needed since no inherent position info

---

## Question 4
**What is beam search decoding and how does it compare to greedy and sampling methods?**

**Answer:**

**Definition:**
Beam search maintains top-k candidates (beams) at each step, exploring multiple paths simultaneously. Greedy picks only the best token at each step. Sampling introduces randomness for diversity. Choice affects quality vs diversity vs speed.

**Comparison:**

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| Greedy | Pick argmax at each step | Fast | Misses global optimum |
| Beam Search | Keep top-k candidates | Better quality | Repetitive, slow |
| Sampling | Sample from distribution | Diverse | Less coherent |
| Top-k Sampling | Sample from top-k tokens | Balanced | Needs tuning |
| Nucleus (Top-p) | Sample from top cumulative prob | Adaptive diversity | Needs tuning |

**Python Code Example:**
```python
import torch
import torch.nn.functional as F

def greedy_decode(model, src, max_len, sos_token, eos_token):
    """Greedy: always pick highest probability token"""
    encoder_out = model.encode(src)
    decoder_input = torch.tensor([[sos_token]])
    
    for _ in range(max_len):
        logits = model.decode(decoder_input, encoder_out)
        next_token = logits[:, -1, :].argmax(dim=-1)
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == eos_token:
            break
    
    return decoder_input

def beam_search(model, src, beam_size=5, max_len=50):
    """Beam search: maintain top-k candidates"""
    encoder_out = model.encode(src)
    
    # Initialize with SOS token
    beams = [(torch.tensor([[sos_token]]), 0.0)]  # (sequence, log_prob)
    
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_token:
                new_beams.append((seq, score))
                continue
            
            logits = model.decode(seq, encoder_out)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            top_probs, top_indices = log_probs.topk(beam_size)
            
            for prob, idx in zip(top_probs[0], top_indices[0]):
                new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                new_beams.append((new_seq, score + prob.item()))
        
        # Keep top beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
    
    return beams[0][0]  # Return best sequence

def sample_decode(model, src, temperature=1.0, top_p=0.9):
    """Nucleus (top-p) sampling"""
    encoder_out = model.encode(src)
    decoder_input = torch.tensor([[sos_token]])
    
    for _ in range(max_len):
        logits = model.decode(decoder_input, encoder_out)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        
        # Nucleus sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum()
        
        # Sample
        idx = torch.multinomial(sorted_probs, 1)
        next_token = sorted_indices[0, idx]
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
    
    return decoder_input
```

**Interview Tips:**
- Beam search for translation (quality matters)
- Sampling for creative generation (diversity matters)
- Temperature controls randomness (higher = more random)
- Beam size 4-6 typical for translation
- Length penalty prevents short outputs

---

## Question 5
**Explain BLEU score and its limitations for evaluating machine translation.**

**Answer:**

**Definition:**
BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between generated translation and reference translations. Scores 0-1, where 1 is perfect match. Includes brevity penalty. Standard metric for machine translation evaluation.

**Formula:**

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where:
- $p_n$ = modified n-gram precision
- $BP$ = brevity penalty (penalizes short outputs)
- $w_n$ = weights (typically uniform 1/4 for n=1..4)

**How It Works:**

| Component | Purpose |
|-----------|--------|
| N-gram precision | Count matching n-grams |
| Modified precision | Clip counts to reference max |
| Brevity penalty | Penalize short translations |
| Geometric mean | Combine 1-4 gram precisions |

**Limitations:**

| Limitation | Description |
|------------|-------------|
| Ignores meaning | Synonyms get no credit |
| Order insensitive | Within n-gram only |
| Requires exact match | "car" vs "automobile" = 0 |
| Multiple references needed | Single reference insufficient |
| Doesn't correlate well | At sentence level |

**Python Code Example:**
```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from sacrebleu import corpus_bleu as sacre_bleu

# Simple example
reference = [['the', 'cat', 'sat', 'on', 'the', 'mat']]
candidate = ['the', 'cat', 'is', 'on', 'the', 'mat']

# Sentence-level BLEU
score = sentence_bleu(reference, candidate)
print(f"Sentence BLEU: {score:.4f}")

# Corpus-level BLEU (more reliable)
references = [[['the', 'quick', 'brown', 'fox']],
              [['jumps', 'over', 'the', 'lazy', 'dog']]]
hypotheses = [['the', 'fast', 'brown', 'fox'],
              ['leaps', 'over', 'the', 'lazy', 'dog']]

corpus_score = corpus_bleu(references, hypotheses)
print(f"Corpus BLEU: {corpus_score:.4f}")

# SacreBLEU (standardized)
refs = [["The quick brown fox jumps over the lazy dog."]]
hyps = ["The fast brown fox leaps over the lazy dog."]
sacre_score = sacre_bleu(hyps, refs)
print(f"SacreBLEU: {sacre_score.score:.2f}")
```

**Alternatives:**

| Metric | Advantage |
|--------|----------|
| chrF | Character-level, better for morphological languages |
| METEOR | Includes synonyms, stems |
| BERTScore | Semantic similarity via embeddings |
| COMET | Learned metric, best correlation |

**Interview Tips:**
- Always use corpus-level BLEU, not sentence-level
- SacreBLEU for reproducible scores
- COMET correlates better with human judgment
- BLEU of 30+ is decent, 40+ is strong
- Different tokenization = different BLEU scores

---

## Question 6
**What is the difference between extractive and abstractive summarization?**

**Answer:**

**Definition:**
Extractive summarization selects and combines existing sentences from source document. Abstractive summarization generates new sentences, potentially paraphrasing and synthesizing information. Abstractive is more flexible but harder and prone to hallucination.

**Comparison:**

| Aspect | Extractive | Abstractive |
|--------|------------|-------------|
| Output | Original sentences | New sentences |
| Approach | Selection/ranking | Generation |
| Faithfulness | High (copies source) | Risk of hallucination |
| Compression | Limited | High |
| Fluency | May be choppy | More natural |
| Models | BERT + classifier | BART, T5, Pegasus |

**Python Code Example:**
```python
# Extractive: TextRank-based
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def extractive_summarize(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return ' '.join([str(sentence) for sentence in summary])

# Extractive with BERT: sentence scoring
class BertExtractiveSummarizer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def summarize(self, text, ratio=0.3):
        sentences = sent_tokenize(text)
        embeddings = self.model.encode(sentences)
        
        # Score by similarity to document centroid
        centroid = embeddings.mean(axis=0)
        scores = [np.dot(emb, centroid) for emb in embeddings]
        
        # Select top sentences
        n_select = max(1, int(len(sentences) * ratio))
        top_indices = np.argsort(scores)[-n_select:]
        top_indices = sorted(top_indices)  # Maintain order
        
        return ' '.join([sentences[i] for i in top_indices])

# Abstractive with transformers
from transformers import pipeline

abstractive_summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"  # or "t5-base", "google/pegasus-xsum"
)

text = "Long article text here..."
summary = abstractive_summarizer(text, max_length=130, min_length=30)
print(summary[0]['summary_text'])
```

**When to Use Each:**

| Scenario | Best Choice |
|----------|-------------|
| Legal/medical (accuracy critical) | Extractive |
| News headlines | Abstractive |
| Multi-document | Abstractive |
| Limited compute | Extractive |

**Interview Tips:**
- Extractive is safer (no hallucination)
- Abstractive produces more natural summaries
- Pegasus is state-of-art for abstractive
- Hybrid: extract then abstract
- Always check for factual consistency in abstractive

---

## Question 7
**How does reading comprehension QA (SQuAD-style) differ from open-domain QA?**

**Answer:**

**Definition:**
Reading comprehension QA extracts answers from a given passage (context provided). Open-domain QA must first find relevant passages from a large corpus, then extract answers. Open-domain is much harder, requiring retrieval + reading.

**Comparison:**

| Aspect | Reading Comprehension | Open-Domain |
|--------|----------------------|-------------|
| Context | Given | Must retrieve |
| Corpus | Single passage | Millions of documents |
| Pipeline | Reader only | Retriever + Reader |
| Example | SQuAD | Natural Questions, TriviaQA |
| Complexity | Easier | Much harder |

**Architecture:**

```
Reading Comprehension:
Question + Passage → [Reader Model] → Answer span

Open-Domain:
Question → [Retriever] → Top-k passages → [Reader] → Answer
```

**Python Code Example:**
```python
# Reading Comprehension (SQuAD-style)
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

context = "The Eiffel Tower is located in Paris, France. It was built in 1889."
question = "When was the Eiffel Tower built?"

result = qa_pipeline(question=question, context=context)
print(result)  # {'answer': '1889', 'start': 62, 'end': 66, 'score': 0.98}

# Open-Domain QA with Retriever + Reader
from transformers import DPRQuestionEncoder, DPRContextEncoder
import faiss

class OpenDomainQA:
    def __init__(self):
        # Retriever: encode questions and passages
        self.q_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.reader = pipeline("question-answering")
        self.index = None  # FAISS index of passage embeddings
        self.passages = []  # Corpus
    
    def build_index(self, passages):
        self.passages = passages
        # Encode all passages (in practice, use passage encoder)
        embeddings = self.encode_passages(passages)
        self.index = faiss.IndexFlatIP(768)  # Inner product
        self.index.add(embeddings)
    
    def answer(self, question, top_k=5):
        # Retrieve relevant passages
        q_emb = self.encode_question(question)
        scores, indices = self.index.search(q_emb, top_k)
        
        # Read from each passage
        answers = []
        for idx in indices[0]:
            passage = self.passages[idx]
            result = self.reader(question=question, context=passage)
            answers.append(result)
        
        # Return best answer
        return max(answers, key=lambda x: x['score'])
```

**Interview Tips:**
- SQuAD: answer always in passage, extractive
- Open-domain: may need to say "I don't know"
- DPR (Dense Passage Retrieval) for neural retrieval
- BM25 still strong baseline for retrieval
- RAG combines retriever with generator

---

## Question 8
**What is retrieval-augmented generation (RAG) and why is it important for QA?**

**Answer:**

**Definition:**
RAG combines retrieval (finding relevant documents) with generation (synthesizing answers). Instead of relying only on model's parameters, it retrieves external knowledge at inference time. Reduces hallucination, enables knowledge updates without retraining.

**Why It's Important:**

| Benefit | Description |
|---------|-------------|
| Factual grounding | Answers based on retrieved docs |
| Knowledge updates | Update docs, not model |
| Reduces hallucination | Can cite sources |
| Scalable | Knowledge in index, not parameters |
| Interpretable | Show retrieved context |

**RAG Architecture:**

```
Query → [Retriever] → Top-k Documents → [Generator] → Answer
              ↑                              ↑
         Document Index              LLM (GPT, BART, T5)
```

**Python Code Example:**
```python
# RAG with LangChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Build vector store from documents
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(documents, embeddings)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",  # Combine all docs
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
)

# Query
answer = qa_chain.run("What is the capital of France?")

# Manual RAG implementation
from sentence_transformers import SentenceTransformer
import numpy as np

class SimpleRAG:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = None  # Your LLM
        self.docs = []
        self.doc_embeddings = None
    
    def index(self, documents):
        self.docs = documents
        self.doc_embeddings = self.encoder.encode(documents)
    
    def retrieve(self, query, k=3):
        query_emb = self.encoder.encode([query])
        scores = np.dot(self.doc_embeddings, query_emb.T).flatten()
        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [self.docs[i] for i in top_k_idx]
    
    def answer(self, question):
        contexts = self.retrieve(question)
        
        prompt = f"""Answer based on the following contexts:

{chr(10).join(contexts)}

Question: {question}
Answer:"""
        
        return self.llm.generate(prompt)
```

**Interview Tips:**
- RAG reduces hallucination by grounding in retrieved docs
- Vector databases: FAISS, Pinecone, Weaviate, Chroma
- Chunk documents appropriately (not too long/short)
- Retriever quality crucial - bad retrieval = bad answers
- Can combine with web search for real-time info

---

## Question 9
**Explain the teacher forcing technique in training seq2seq models.**

**Answer:**

**Definition:**
Teacher forcing feeds ground truth tokens as decoder input during training, instead of model's own predictions. Speeds up training convergence but creates mismatch with inference (exposure bias) where model uses its own predictions.

**How It Works:**

| Mode | Decoder Input |
|------|---------------|
| Teacher forcing | Ground truth previous token |
| Autoregressive | Model's previous prediction |

**Training:**
```
Target: "I love NLP"

With teacher forcing:
  Step 1: Input=<SOS>, Target="I"
  Step 2: Input="I" (ground truth), Target="love"
  Step 3: Input="love" (ground truth), Target="NLP"

Without:
  Step 1: Input=<SOS>, Predict="I"
  Step 2: Input="I" (predicted), Predict="love"
  Step 3: Input="love" (predicted), Predict="NLP"
```

**Python Code Example:**
```python
import torch
import random

def train_step(model, src, trg, teacher_forcing_ratio=0.5):
    """
    Train with scheduled teacher forcing
    """
    encoder_outputs, hidden = model.encoder(src)
    
    # First input is SOS token
    decoder_input = trg[:, 0:1]
    outputs = []
    
    for t in range(1, trg.size(1)):
        output, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
        outputs.append(output)
        
        # Teacher forcing: use ground truth or prediction
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        if use_teacher_forcing:
            decoder_input = trg[:, t:t+1]  # Ground truth
        else:
            decoder_input = output.argmax(dim=-1)  # Prediction
    
    outputs = torch.cat(outputs, dim=1)
    loss = criterion(outputs.view(-1, vocab_size), trg[:, 1:].view(-1))
    return loss

# Scheduled sampling: decrease teacher forcing over time
class ScheduledSampling:
    def __init__(self, initial_ratio=1.0, decay=0.99, min_ratio=0.1):
        self.ratio = initial_ratio
        self.decay = decay
        self.min_ratio = min_ratio
    
    def step(self):
        self.ratio = max(self.min_ratio, self.ratio * self.decay)
    
    def get_ratio(self):
        return self.ratio

# Usage
scheduler = ScheduledSampling()
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(model, src, trg, scheduler.get_ratio())
    scheduler.step()  # Decrease teacher forcing ratio
```

**Exposure Bias Problem:**
- Training: always sees correct previous tokens
- Inference: sees its own (possibly wrong) predictions
- Error compounds - wrong token leads to more wrong tokens

**Solutions:**
- Scheduled sampling: gradually reduce teacher forcing
- Curriculum learning: start easy, increase difficulty
- Reinforcement learning: train with own outputs

**Interview Tips:**
- Teacher forcing speeds training significantly
- Without it, early training is very slow
- Exposure bias is a real problem in practice
- Scheduled sampling is common compromise
- Modern LLMs still use teacher forcing (masked LM objectives)

---

## Question 10
**What are the key evaluation metrics for summarization (ROUGE, BERTScore, factual consistency)?**

**Answer:**

**Definition:**
ROUGE measures n-gram overlap with reference summaries. BERTScore uses semantic similarity via embeddings. Factual consistency metrics check if generated summary is factually accurate relative to source. Each captures different quality aspects.

**Key Metrics:**

| Metric | Measures | How |
|--------|----------|-----|
| ROUGE-1 | Unigram overlap | Recall/Precision of words |
| ROUGE-2 | Bigram overlap | Captures phrase similarity |
| ROUGE-L | Longest common subsequence | Word order |
| BERTScore | Semantic similarity | Embedding cosine similarity |
| Factual consistency | Correctness | NLI-based or QA-based |

**ROUGE Formula:**

$$\text{ROUGE-N Recall} = \frac{\text{Matching n-grams}}{\text{Total n-grams in reference}}$$

$$\text{ROUGE-N Precision} = \frac{\text{Matching n-grams}}{\text{Total n-grams in summary}}$$

$$\text{ROUGE-N F1} = \frac{2 \cdot P \cdot R}{P + R}$$

**Python Code Example:**
```python
# ROUGE score
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

reference = "The cat sat on the mat."
prediction = "The cat is sitting on the mat."

scores = scorer.score(reference, prediction)
print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

# BERTScore - semantic similarity
from bert_score import score

references = ["The cat sat on the mat."]
candidates = ["A feline rested on the rug."]  # Same meaning, different words

P, R, F1 = score(candidates, references, lang="en")
print(f"BERTScore F1: {F1.mean():.4f}")  # High score despite different words

# Factual Consistency with NLI
from transformers import pipeline

nli_model = pipeline("text-classification", model="roberta-large-mnli")

def check_factual_consistency(source, summary):
    """Check if summary is entailed by source"""
    result = nli_model(f"{source} [SEP] {summary}")
    # Labels: entailment, neutral, contradiction
    return result[0]['label'] == 'ENTAILMENT'

# QA-based factual consistency (QuestEval)
def qa_factual_check(source, summary):
    """
    1. Generate questions from summary
    2. Answer questions using source
    3. Answer questions using summary
    4. Compare answers
    """
    # If answers match, summary is factually consistent
    pass
```

**When to Use Each:**

| Metric | Best For |
|--------|----------|
| ROUGE | Quick evaluation, comparing systems |
| BERTScore | Paraphrase-heavy summaries |
| Factual consistency | Abstractive summarization |

**Interview Tips:**
- ROUGE-2 most commonly reported
- BERTScore captures paraphrasing that ROUGE misses
- Factual consistency is crucial for abstractive
- Human evaluation still gold standard
- Report multiple metrics, not just one

---

## Interview Questions

## Question 11
**How do you handle sequence generation for tasks with extreme input/output length disparities?**

**Answer:**

**Definition:**
Length disparity means input is much longer/shorter than output. Examples: long document → short summary (compression), keyword → paragraph (expansion). Challenges include information selection, context preservation, and generation control.

**Strategies by Type:**

| Disparity | Challenge | Strategies |
|-----------|-----------|------------|
| Long → Short | Information selection | Hierarchical, chunking |
| Short → Long | Content expansion | Prompting, retrieval |
| Variable | Flexibility | Length conditioning |

**Python Code Example:**
```python
# Long input → Short output (Summarization)
def hierarchical_summarize(long_text, model, max_chunk=1024, final_length=100):
    """Hierarchical approach for very long documents"""
    # Split into chunks
    words = long_text.split()
    chunks = [' '.join(words[i:i+max_chunk]) for i in range(0, len(words), max_chunk)]
    
    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        summary = model.generate(chunk, max_length=200)
        chunk_summaries.append(summary)
    
    # Combine and summarize again
    combined = ' '.join(chunk_summaries)
    final_summary = model.generate(combined, max_length=final_length)
    return final_summary

# Alternatively: Longformer for long inputs
from transformers import LongformerForConditionalGeneration

model = LongformerForConditionalGeneration.from_pretrained('allenai/led-base-16384')
# Handles up to 16K tokens

# Short input → Long output (Expansion)
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

def expand_with_retrieval(keywords, retrieved_context):
    prompt = f"""Based on these keywords: {keywords}

Context: {retrieved_context}

Write a detailed paragraph:"""
    return generator(prompt, max_length=500)

# Length conditioning
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Control output length via special tokens or max_length
def generate_with_length_control(text, target_length):
    # Some models support length control tokens
    input_text = f"summarize in {target_length} words: {text}"
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=target_length * 2)
    return tokenizer.decode(outputs[0])
```

**Interview Tips:**
- LED, Longformer handle long inputs (16K+ tokens)
- Hierarchical approaches for very long documents
- Length control tokens in some models
- Short → long often needs retrieval for content
- Consider sliding window with overlap for context

---

## Question 12
**What techniques prevent repetition and improve diversity in text generation?**

**Answer:**

**Definition:**
Repetition is a common failure mode where models repeat phrases or sentences. Diversity techniques encourage varied outputs. Solutions include n-gram blocking, penalty terms, and sampling strategies.

**Techniques:**

| Technique | Description |
|-----------|-------------|
| N-gram blocking | Prevent repeating n-grams |
| Repetition penalty | Reduce probability of repeated tokens |
| Top-k sampling | Sample from top-k tokens |
| Nucleus (top-p) | Sample from top cumulative probability |
| Temperature | Control randomness |
| Diverse beam search | Encourage different beams |

**Python Code Example:**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors='pt')

# Without repetition handling (may repeat)
output_repetitive = model.generate(
    **inputs,
    max_length=100,
    do_sample=False  # Greedy
)

# With repetition penalty
output_no_repeat = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    repetition_penalty=1.2,  # Penalize repeated tokens
    no_repeat_ngram_size=3,  # Block repeating 3-grams
    top_p=0.9,               # Nucleus sampling
    temperature=0.8          # Some randomness
)

# Diverse beam search
output_diverse = model.generate(
    **inputs,
    max_length=100,
    num_beams=5,
    num_beam_groups=5,
    diversity_penalty=0.5,  # Encourage different beams
    num_return_sequences=3  # Return multiple diverse outputs
)

# Custom n-gram blocking
def generate_with_ngram_block(model, input_ids, max_len, n=3):
    generated = input_ids.clone()
    seen_ngrams = set()
    
    for _ in range(max_len):
        outputs = model(generated)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Block n-grams we've seen
        if generated.size(1) >= n - 1:
            last_ngram = tuple(generated[0, -(n-1):].tolist())
            for token_id in range(next_token_logits.size(-1)):
                potential_ngram = last_ngram + (token_id,)
                if potential_ngram in seen_ngrams:
                    next_token_logits[0, token_id] = float('-inf')
        
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        
        # Record new n-gram
        if generated.size(1) >= n:
            new_ngram = tuple(generated[0, -n:].tolist())
            seen_ngrams.add(new_ngram)
    
    return generated
```

**Interview Tips:**
- `no_repeat_ngram_size=3` is simple and effective
- `repetition_penalty=1.1-1.3` works well
- Top-p sampling better than top-k generally
- Temperature: 0.7-0.9 for balanced diversity
- Beam search tends toward repetition

---

## Question 13
**How do you implement controllable text generation (style, length, tone)?**

**Answer:**

**Definition:**
Controllable generation steers model output toward desired attributes (formal/casual tone, specific length, positive/negative sentiment). Approaches include control codes, prompting, fine-tuning, and steering during decoding.

**Control Dimensions:**

| Attribute | Example |
|-----------|--------|
| Style | Formal, casual, poetic |
| Tone | Positive, negative, neutral |
| Length | Short, medium, long |
| Topic | Sports, technology, politics |
| Persona | Expert, friendly, professional |

**Approaches:**

| Method | Description |
|--------|-------------|
| Control codes | Prepend tokens like "<FORMAL>" |
| Prompting | Describe desired style in prompt |
| CTRL model | Trained with control prefixes |
| PPLM | Perturb hidden states toward attribute |
| Fine-tuning | Train on style-specific data |

**Python Code Example:**
```python
# Approach 1: Prompt-based control (simplest)
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2-medium')

# Control via prompt
prompts = {
    'formal': "Write a formal business email: Dear Sir or Madam,",
    'casual': "Write a casual message: Hey! What's up,",
    'positive': "Write a positive review: This product is",
    'negative': "Write a negative review: I'm disappointed because"
}

for style, prompt in prompts.items():
    output = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']
    print(f"{style}: {output[:100]}...")

# Approach 2: Control codes (CTRL model style)
# During training, prepend control token
# At inference, use desired control token

class ControlledGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.control_tokens = {
            'formal': '<FORMAL>',
            'casual': '<CASUAL>',
            'short': '<SHORT>',
            'long': '<LONG>'
        }
    
    def generate(self, text, controls):
        # Prepend control tokens
        control_str = ' '.join([self.control_tokens[c] for c in controls])
        input_text = f"{control_str} {text}"
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0])

# Approach 3: Length control via max_length and length penalty
output = model.generate(
    input_ids,
    max_length=50,          # Short output
    min_length=30,          # Ensure minimum
    length_penalty=1.5,     # Encourage brevity
    early_stopping=True
)

# Approach 4: Classifier-guided decoding
def guided_decoding(model, tokenizer, prompt, attribute_classifier, target_class):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    
    for _ in range(50):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        
        # Score each possible continuation
        best_score = float('-inf')
        best_token = 0
        
        for token_id in logits.topk(50).indices[0]:
            candidate = torch.cat([input_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            text = tokenizer.decode(candidate[0])
            
            # Check attribute score
            attr_score = attribute_classifier(text)[target_class]
            combined_score = logits[0, token_id] + attr_score * 2
            
            if combined_score > best_score:
                best_score = combined_score
                best_token = token_id
        
        input_ids = torch.cat([input_ids, best_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    return tokenizer.decode(input_ids[0])
```

**Interview Tips:**
- Prompting is simplest for modern LLMs
- Control codes require training/fine-tuning
- Length penalty in beam search for brevity
- PPLM for post-hoc attribute control
- InstructGPT/ChatGPT follow style instructions well

---

## Question 14
**What are the challenges of machine translation for low-resource language pairs?**

**Answer:**

**Definition:**
Low-resource translation involves language pairs with little parallel training data (e.g., English-Swahili, French-Yoruba). Challenges include limited data, domain mismatch, and lack of evaluation resources. Solutions involve transfer learning, multilingual models, and data augmentation.

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| Limited parallel data | Few translated sentence pairs |
| No pre-trained models | Language-specific models missing |
| Evaluation | Few test sets, no native speakers |
| Domain mismatch | Available data is religious/news |
| Morphological complexity | Many low-resource languages are morphologically rich |

**Solutions:**

| Solution | Approach |
|----------|----------|
| Multilingual models | M2M-100, mBART, NLLB |
| Transfer learning | Pre-train on high-resource, adapt |
| Back-translation | Generate synthetic parallel data |
| Cross-lingual transfer | Train on related language |
| Unsupervised NMT | Use monolingual data only |

**Python Code Example:**
```python
# Use multilingual model (works for 100+ languages)
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer

# NLLB (No Language Left Behind) - 200 languages
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

def translate_low_resource(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Example: English to Yoruba (low-resource)
translation = translate_low_resource(
    "Hello, how are you?",
    src_lang="eng_Latn",
    tgt_lang="yor_Latn"  # Yoruba
)

# Back-translation for data augmentation
def back_translate(texts, src_lang, pivot_lang):
    """
    Augment low-resource data via back-translation
    src -> pivot -> src creates paraphrased version
    """
    augmented = []
    for text in texts:
        # Forward translation
        pivot_text = translate(text, src_lang, pivot_lang)
        # Back translation
        back_text = translate(pivot_text, pivot_lang, src_lang)
        augmented.append(back_text)
    return augmented

# Cross-lingual transfer: train on related high-resource pair
# E.g., Train on Spanish-Portuguese, adapt to Portuguese-Galician
```

**Interview Tips:**
- NLLB from Meta supports 200 languages
- Multilingual models enable zero-shot for unseen pairs
- Back-translation is crucial data augmentation
- Related language transfer helps (Hindi → Urdu)
- Quality degrades but is better than nothing

---

## Question 15
**How do you handle domain adaptation in neural machine translation?**

**Answer:**

**Definition:**
Domain adaptation adjusts a general translation model for specific domains (medical, legal, technical). General models trained on news fail on domain-specific terminology. Techniques include fine-tuning, domain tags, and terminology injection.

**Domain Challenges:**

| Challenge | Example |
|-----------|--------|
| Terminology | "Stat" = immediately (medical) |
| Style | Legal formality, technical precision |
| Coverage | Rare terms not in training data |
| Consistency | Same term should translate consistently |

**Approaches:**

| Approach | Description |
|----------|-------------|
| Fine-tuning | Continue training on domain data |
| Mixed training | Combine general + domain data |
| Domain tags | "<medical>" prefix |
| Terminology injection | Force glossary terms |
| Retrieval-augmented | Retrieve similar translations |

**Python Code Example:**
```python
# Approach 1: Fine-tune on domain data
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments

# Start with general model
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')

# Fine-tune on medical parallel data
training_args = TrainingArguments(
    output_dir='./medical-mt',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,  # Lower LR to preserve general knowledge
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=medical_dataset  # Domain-specific pairs
)
trainer.train()

# Approach 2: Terminology-constrained decoding
def translate_with_glossary(text, model, tokenizer, glossary):
    """Force glossary terms in translation"""
    # Replace terms before translation
    for src_term, tgt_term in glossary.items():
        text = text.replace(src_term, f"[TERM]{tgt_term}[/TERM]")
    
    # Translate
    inputs = tokenizer(text, return_tensors='pt')
    output = model.generate(**inputs)
    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Restore glossary terms
    # (In practice, use constrained beam search)
    return translation

glossary = {
    'myocardial infarction': 'Herzinfarkt',
    'blood pressure': 'Blutdruck'
}

# Approach 3: Domain tags
def domain_translate(text, domain, model, tokenizer):
    """Use domain tag prefix"""
    tagged_input = f"<{domain}> {text}"
    inputs = tokenizer(tagged_input, return_tensors='pt')
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example
medical_translation = domain_translate(
    "The patient presents with acute chest pain.",
    domain="medical",
    model=domain_tagged_model,
    tokenizer=tokenizer
)
```

**Interview Tips:**
- Fine-tuning is most common approach
- Use low learning rate to avoid catastrophic forgetting
- Domain data is often limited - augment with back-translation
- Terminology consistency critical for professional translation
- Consider training domain classifier to route requests

---

## Question 16
**What techniques work best for real-time/simultaneous machine translation?**

**Answer:**

**Definition:**
Simultaneous translation translates while speaker is still talking, with minimal delay. Unlike offline translation, it must decide when to commit to translation before seeing full sentence. Key challenge: wait-k policy balancing latency vs quality.

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| Incomplete input | Must translate before sentence ends |
| Word order | Source/target may have different order |
| Latency requirement | Real-time constraint |
| Revision | May need to correct earlier outputs |

**Approaches:**

| Approach | Description |
|----------|-------------|
| Wait-k | Wait for k source tokens, then translate |
| Adaptive wait | Learn when to read vs write |
| Incremental | Output partial translations |
| Re-translation | Revise previous output |

**Python Code Example:**
```python
import torch

class WaitKTranslator:
    """
    Wait-k policy: wait for k source tokens before translating
    """
    def __init__(self, model, tokenizer, k=3):
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
    
    def translate_stream(self, source_stream):
        """Translate from streaming source"""
        source_buffer = []
        translations = []
        
        for token in source_stream:
            source_buffer.append(token)
            
            # Wait-k policy: start translating after k tokens
            if len(source_buffer) >= self.k:
                # Translate current buffer
                partial_src = ' '.join(source_buffer)
                inputs = self.tokenizer(partial_src, return_tensors='pt')
                
                # Generate one target token
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    prefix_allowed_tokens_fn=self.get_prefix_fn(translations)
                )
                
                new_tokens = outputs[0, len(translations):]
                translations.extend(new_tokens.tolist())
                
                yield self.tokenizer.decode(new_tokens)
        
        # Finish remaining
        # ...

# Adaptive policy with reinforcement learning
class AdaptiveSimultaneousNMT:
    """
    Learn when to read source vs write target
    Actions: READ (get next source), WRITE (emit target)
    """
    def __init__(self, translator, policy):
        self.translator = translator
        self.policy = policy  # RL agent
    
    def translate(self, source_tokens):
        source_idx = 0
        outputs = []
        state = self.get_initial_state()
        
        while source_idx < len(source_tokens) or self.can_write(state):
            action = self.policy.predict(state)
            
            if action == 'READ' and source_idx < len(source_tokens):
                state = self.update_state(state, source_tokens[source_idx])
                source_idx += 1
            elif action == 'WRITE':
                next_token = self.translator.generate_next(state)
                outputs.append(next_token)
                state = self.update_state_output(state, next_token)
        
        return outputs

# Quality-latency tradeoff
# Higher k = better quality, higher latency
# Lower k = faster, lower quality
```

**Interview Tips:**
- Wait-k is simple and effective baseline
- k=3-5 typical tradeoff
- Adaptive policies learned via RL
- German-English harder than French-English (word order)
- Evaluation: BLEU + latency metrics

---

## Question 17
**How do you preserve named entities and technical terms during translation?**

**Answer:**

**Definition:**
Named entities (names, companies) and technical terms often shouldn't be translated or need specific translations. NMT may corrupt them. Solutions include copying mechanisms, terminology constraints, and placeholder approaches.

**Challenges:**

| Issue | Example |
|-------|--------|
| Name corruption | "Michael Jordan" → "Michel Jourdain" |
| Technical terms | "API" mistranslated |
| Brand names | "iPhone" should stay "iPhone" |
| Inconsistency | Same entity translated differently |

**Solutions:**

| Solution | Description |
|----------|-------------|
| Placeholder | Replace entities with tokens, restore after |
| Copy mechanism | Model can copy source tokens |
| Constrained decoding | Force specific translations |
| Terminology injection | Pre/post-process with glossary |
| Fine-tuning | Train with entity-aware data |

**Python Code Example:**
```python
import spacy
import re

nlp = spacy.load('en_core_web_sm')

def translate_with_entity_preservation(text, translator, entity_dict=None):
    """
    Preserve named entities during translation
    1. Extract entities
    2. Replace with placeholders
    3. Translate
    4. Restore entities
    """
    doc = nlp(text)
    
    # Extract entities and create placeholders
    entity_map = {}
    modified_text = text
    
    for i, ent in enumerate(reversed(doc.ents)):  # Reverse to maintain positions
        placeholder = f"__ENTITY_{i}__"
        entity_map[placeholder] = ent.text
        modified_text = modified_text[:ent.start_char] + placeholder + modified_text[ent.end_char:]
    
    # Translate text with placeholders
    translated = translator(modified_text)
    
    # Restore entities
    for placeholder, entity in entity_map.items():
        # Use glossary if available
        if entity_dict and entity in entity_dict:
            translated_entity = entity_dict[entity]
        else:
            translated_entity = entity  # Keep original
        translated = translated.replace(placeholder, translated_entity)
    
    return translated

# Example usage
glossary = {
    'Machine Learning': 'Machine Learning',  # Keep as-is
    'TensorFlow': 'TensorFlow',
    'New York': 'Nueva York'  # Specific translation
}

text = "Michael works at Google in New York using TensorFlow."
translated = translate_with_entity_preservation(
    text, 
    translator=my_translator,
    entity_dict=glossary
)

# Constrained beam search (force specific outputs)
from transformers import MarianMTModel

def translate_with_constraints(text, model, tokenizer, forced_terms):
    """
    Force specific term translations
    """
    inputs = tokenizer(text, return_tensors='pt')
    
    # Define allowed token sequences
    force_words_ids = [
        tokenizer(term, add_special_tokens=False).input_ids
        for term in forced_terms.values()
    ]
    
    outputs = model.generate(
        **inputs,
        force_words_ids=force_words_ids  # Require these terms
    )
    return tokenizer.decode(outputs[0])
```

**Interview Tips:**
- Placeholder approach is simple and effective
- Copy mechanism built into some seq2seq models
- Enterprise systems use terminology databases
- Test with important entities before deployment
- Case sensitivity matters for detection

---

## Question 18
**What is back-translation and how does it help with data augmentation for NMT?**

**Answer:**

**Definition:**
Back-translation generates synthetic parallel data by translating monolingual target text back to source. Provides diverse source expressions for same target, augmenting training data. Critical for low-resource scenarios.

**How It Works:**

```
Step 1: Have monolingual target data (German sentences)
Step 2: Train reverse model (German → English)
Step 3: Back-translate to get synthetic source (English)
Step 4: Use (synthetic English, real German) for training
```

**Benefits:**

| Benefit | Description |
|---------|-------------|
| More data | Leverage abundant monolingual text |
| Diversity | Same target, varied source expressions |
| Domain adaptation | Use domain-specific monolingual data |
| Low-resource | Critical when parallel data scarce |

**Python Code Example:**
```python
from transformers import MarianMTModel, MarianTokenizer

# Step 1: Train/load reverse translation model
reverse_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en')
reverse_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')

def back_translate_batch(target_texts):
    """Translate target language back to source"""
    inputs = reverse_tokenizer(target_texts, return_tensors='pt', padding=True)
    outputs = reverse_model.generate(**inputs)
    synthetic_source = [reverse_tokenizer.decode(out, skip_special_tokens=True) 
                       for out in outputs]
    return synthetic_source

# Step 2: Generate synthetic parallel data
monolingual_german = [
    "Das Wetter ist heute schön.",
    "Ich arbeite gerne mit Python.",
    # ... millions of German sentences
]

synthetic_english = back_translate_batch(monolingual_german)

# Step 3: Create augmented dataset
augmented_data = []
for src, tgt in zip(synthetic_english, monolingual_german):
    augmented_data.append({'source': src, 'target': tgt})

# Combine with real parallel data
full_training_data = real_parallel_data + augmented_data

# Step 4: Train forward model on augmented data
from transformers import Trainer, TrainingArguments

forward_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')

# Training with mixed data
trainer = Trainer(
    model=forward_model,
    train_dataset=full_training_data,
    # Tag synthetic data with lower weight optionally
)
trainer.train()

# Sampling strategies for back-translation
def back_translate_with_sampling(texts, model, tokenizer, num_samples=3):
    """Generate multiple diverse back-translations per text"""
    all_synthetic = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt')
        
        # Sample multiple translations
        outputs = model.generate(
            **inputs,
            num_return_sequences=num_samples,
            do_sample=True,
            temperature=0.9,
            top_p=0.95
        )
        
        translations = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        all_synthetic.append(translations)
    
    return all_synthetic
```

**Interview Tips:**
- Back-translation is standard practice in NMT
- Sampling > beam search for diversity
- Tag synthetic data optionally (helps model distinguish)
- Iterate: train → back-translate → train
- Noisy back-translation can help (add noise to synthetic source)

---

## Question 19
**How do multilingual translation models (mBART, M2M-100) work?**

**Answer:**

**Definition:**
Multilingual translation models translate between many language pairs with a single model. mBART pre-trains via denoising autoencoder on multiple languages. M2M-100 directly trains on 100 languages without English pivoting. Enable zero-shot translation for unseen pairs.

**Key Models:**

| Model | Languages | Approach |
|-------|-----------|----------|
| mBART | 25-50 | Denoising pre-training + fine-tune |
| M2M-100 | 100 | Direct many-to-many training |
| NLLB | 200 | Focus on low-resource languages |
| Google's | 100+ | Multilingual with language tags |

**How They Work:**

**mBART:**
1. Pre-train: denoise sentences in 25 languages (mask + shuffle)
2. Fine-tune: on parallel data for specific pairs
3. Enables zero-shot for unseen pairs via shared representations

**M2M-100:**
1. Train directly on many-to-many parallel data
2. No English-centric approach (direct fr→de without en)
3. Uses language tokens to specify direction

**Python Code Example:**
```python
# M2M-100 translation
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')

def translate_any_pair(text, src_lang, tgt_lang):
    """Translate between any of 100 language pairs"""
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors='pt')
    
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# French to German (no English pivot)
print(translate_any_pair("Bonjour le monde", "fr", "de"))
# Chinese to Spanish
print(translate_any_pair("你好世界", "zh", "es"))

# mBART for pre-training + fine-tuning
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

mbart = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
mbart_tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')

def mbart_translate(text, src_lang, tgt_lang):
    mbart_tokenizer.src_lang = src_lang
    inputs = mbart_tokenizer(text, return_tensors='pt')
    
    generated = mbart.generate(
        **inputs,
        forced_bos_token_id=mbart_tokenizer.lang_code_to_id[tgt_lang]
    )
    return mbart_tokenizer.decode(generated[0], skip_special_tokens=True)

# Language codes differ by model
# M2M-100: "fr", "de", "zh"
# mBART-50: "fr_XX", "de_DE", "zh_CN"

# NLLB for low-resource
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

nllb_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')
nllb_tokenizer = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')

# 200 languages including many low-resource
# Language codes: "eng_Latn", "fra_Latn", "zho_Hans", "yor_Latn"
```

**Interview Tips:**
- M2M-100 avoids English-centric bias
- mBART enables zero-shot via shared representations
- NLLB best for low-resource languages
- Larger models = better quality but slower
- Language code format varies by model

---

## Question 20
**What approaches handle idiomatic expressions and cultural references in translation?**

**Answer:**

**Definition:**
Idioms and cultural references don't translate literally. "It's raining cats and dogs" means heavy rain, not animals. Approaches include idiom databases, paraphrase before translation, and models trained on idiom-aware data.

**Challenges:**

| Type | Example |
|------|--------|
| Idioms | "Break a leg" ≠ Break your leg |
| Metaphors | "Heart of gold" = very kind |
| Cultural refs | "Super Bowl" - meaningless in some cultures |
| Humor | Puns don't translate |
| Sayings | "Every cloud has a silver lining" |

**Approaches:**

| Approach | Description |
|----------|-------------|
| Idiom dictionary | Map to target equivalent |
| Paraphrase first | Convert idiom to literal meaning |
| Culture adaptation | Replace with local equivalent |
| Literal + note | Translate literally, add explanation |
| LLM prompting | Explain meaning, then translate |

**Python Code Example:**
```python
# Approach 1: Idiom dictionary lookup
idiom_dict = {
    "en": {
        "break a leg": {
            "meaning": "good luck",
            "de": "Hals- und Beinbruch",
            "fr": "merde",
            "es": "mucha mierda"
        },
        "it's raining cats and dogs": {
            "meaning": "heavy rain",
            "de": "Es regnet in Strömen",
            "fr": "Il pleut des cordes"
        }
    }
}

def translate_with_idioms(text, src_lang, tgt_lang, translator):
    """Check for idioms before translation"""
    text_lower = text.lower()
    
    for idiom, translations in idiom_dict.get(src_lang, {}).items():
        if idiom in text_lower:
            if tgt_lang in translations:
                # Replace with target idiom equivalent
                text = text.replace(idiom, f"[IDIOM]{translations[tgt_lang]}[/IDIOM]")
    
    # Translate rest normally
    translated = translator(text)
    
    # Clean up markers
    translated = translated.replace("[IDIOM]", "").replace("[/IDIOM]", "")
    return translated

# Approach 2: LLM-based paraphrase + translate
def translate_idiom_aware(text, src_lang, tgt_lang, llm):
    # Step 1: Identify and paraphrase idioms
    paraphrase_prompt = f"""Identify any idioms in this text and replace them with their literal meaning:

Text: {text}

Paraphrased:"""
    
    paraphrased = llm.generate(paraphrase_prompt)
    
    # Step 2: Translate the paraphrased text
    translate_prompt = f"""Translate this {src_lang} text to {tgt_lang}:

{paraphrased}

Translation:"""
    
    return llm.generate(translate_prompt)

# Approach 3: Cultural adaptation
def adapt_cultural_reference(text, source_culture, target_culture, llm):
    prompt = f"""Adapt this text from {source_culture} culture for a {target_culture} audience.
Replace cultural references with locally appropriate equivalents while preserving the meaning.

Original: {text}

Adapted:"""
    return llm.generate(prompt)

# Example
text = "He hit it out of the park with that presentation."
# Baseball metaphor - needs adaptation for non-US audiences
adapted = adapt_cultural_reference(text, "American", "British")
# Could become: "He absolutely nailed that presentation."
```

**Interview Tips:**
- Idiom dictionaries exist but aren't complete
- LLMs handle idioms better than older NMT
- Cultural adaptation > literal translation
- Some things genuinely untranslatable
- Professional translators use localization (adaptation)

---

## Question 21
**How do you design QA systems that handle multi-hop reasoning across documents?**
**Answer:** _To be filled_

---

## Question 22
**What techniques help QA systems know when they don't know the answer?**
**Answer:** _To be filled_

---

## Question 23
**How do you implement conversational QA with context tracking across turns?**
**Answer:** _To be filled_

---

## Question 24
**What is the role of knowledge graphs in question answering systems?**
**Answer:** _To be filled_

---

## Question 25
**How do you handle questions requiring numerical reasoning or table understanding?**
**Answer:** _To be filled_

---

## Question 26
**What is dense passage retrieval (DPR) and how does it improve open-domain QA?**
**Answer:** _To be filled_

---

## Question 27
**How do you balance faithfulness vs fluency in abstractive summarization?**
**Answer:** _To be filled_

---

## Question 28
**What is the hallucination problem in summarization and how do you mitigate it?**
**Answer:** _To be filled_

---

## Question 29
**How do you implement query-focused summarization for specific information needs?**
**Answer:** _To be filled_

---

## Question 30
**What techniques work for multi-document summarization with conflicting information?**
**Answer:** _To be filled_

---

## Question 31
**How do you control summary length while maintaining coherence and coverage?**
**Answer:** _To be filled_

---

## Question 32
**What is copy mechanism in seq2seq models and when is it useful?**
**Answer:** _To be filled_

---

## Question 33
**How do you implement knowledge distillation for compressing large generation models?**
**Answer:** _To be filled_

---

## Question 34
**What is nucleus (top-p) sampling and how does it improve generation quality?**
**Answer:** _To be filled_

---

## Question 35
**How do you evaluate factual consistency in generated text?**
**Answer:** _To be filled_

---

## Question 36
**What is curriculum learning and how does it help train seq2seq models?**
**Answer:** _To be filled_

---

## Question 37
**How do you handle code generation and code summarization tasks?**
**Answer:** _To be filled_

---

## Question 38
**What techniques improve translation quality for morphologically rich languages?**
**Answer:** _To be filled_

---

## Question 39
**How do you implement hybrid extractive-abstractive summarization?**
**Answer:** _To be filled_

---

## Question 40
**What is cross-lingual summarization and what challenges does it present?**
**Answer:** _To be filled_

---

## Question 41
**How do you handle long document summarization that exceeds context limits?**
**Answer:** _To be filled_

---

## Question 42
**What is the exposure bias problem in seq2seq training and how do you address it?**
**Answer:** _To be filled_

---

## Question 43
**How do you implement real-time QA with low latency requirements?**
**Answer:** _To be filled_

---

## Question 44
**What role do pre-trained language models (T5, BART, GPT) play in text generation tasks?**
**Answer:** _To be filled_

---

## Question 45
**How do you fine-tune large language models for specific generation tasks efficiently (LoRA, adapters)?**
**Answer:** _To be filled_

---

## Question 46
**What is RLHF (Reinforcement Learning from Human Feedback) and how does it improve generation?**
**Answer:** _To be filled_

---

## Question 47
**How do you handle bias and fairness issues in text generation systems?**
**Answer:** _To be filled_

---

## Question 48
**What are the key differences between autoregressive and non-autoregressive generation?**
**Answer:** _To be filled_

---

## Question 49
**How do you build production-ready generation systems (caching, batching, optimization)?**
**Answer:** _To be filled_

---

## Question 50
**What is prompt engineering for generation tasks and how do in-context learning approaches work?**
**Answer:** _To be filled_

---
