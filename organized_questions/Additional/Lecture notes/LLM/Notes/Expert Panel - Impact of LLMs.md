# Expert Panel: Impact of LLMs — Concluding Discussion

## Panelists

| Name | Affiliation | Focus Area |
|------|------------|------------|
| **Tanmoy** (Moderator) | IIT Delhi | NLP |
| **Yatin** | IBM Research | Conversational AI, domain-specific fine-tuning |
| **Gourav Pandey** | IBM Research | RL + search techniques for LLMs |
| **Souris Das Gupta** | DA-IICT | Personalization in LLMs, energy-efficient models |
| **Anonto** | PhD Student, IIT Delhi | Interpretability, mechanistic understanding |

---

## 1. Deployment Challenges in Industry

### Scale Challenges
- Small/medium companies **cannot develop or host** LLMs independently
- LLMs are increasingly served as **LLM-as-a-Service** (cloud-based)
- On-premise deployment requires specialized expertise (vLLM and similar tools help but aren't sufficient)
- **Privacy concern**: Client data goes to external servers — trust issues

### Hallucination — The Biggest Problem
- Currently all LLM deployments require a **human verifier** at the end
- LLMs don't work autonomously in adversarial settings
- Adversarial users can exploit deployed chatbots
- Key challenge: How to implement effective **guardrails**?

---

## 2. Data Challenges

### Data Scarcity
- **Synthetic data generation** is the primary solution:
  - Started with CoT data generation
  - Now uses search + verification for higher quality
  - Approaches like STAR (Self-Taught Reasoner) address this
- Domain-specific synthetic data (healthcare, finance):
  - Feed domain documents → generate Q&A pairs → train model
  - Works for document QA and summarization tasks
- Still requires human verification of synthetic data quality

### Copyright, Privacy & Representation
- **Diversity**: Ensure data represents different groups and viewpoints
- **Copyright**: Legal teams must verify data is not copyrighted
- **Federated Learning**: Train without sharing data — each party trains on their local data
- More careful sourcing from diverse data origins needed

---

## 3. Industry-Academia Collaboration

### Current State
- Academic research **cannot progress independently** anymore — resource requirements are too high
- Industry provides: compute resources + real-world problems
- Academia provides: independent thinking + fundamental research + talent pipeline
- Programs like IBM's AIH facilitate collaboration

### Publication Dilemma
- Some companies **restrict publication** of high-value research (keep as trade secrets)
- Only allow publishing lower-value work (datasets, benchmarks)
- This is a **recent phenomenon** driven by the LLM gold rush
- Counterargument: **First-mover advantage** — deploy first, then publish
- This secrecy **slows down scientific progress** overall

### Resource Sharing
- Need state-level or university-level **computing consortiums**
- Government agencies should maintain shared data centers
- Industry collaboration is often easier than waiting for government
- Hardware obsolescence is a concern (H100 already outdating → H200)
- Industry better placed to handle obsolescence through revenue generation

---

## 4. Teaching NLP in the LLM Era

### Challenge
- How to cover both classical NLP and modern LLM-based NLP in limited semesters?
- Some universities have scrapped classical NLP courses entirely

### Recommended Approach (Souris)
1. First 3 weeks: Ground students in **classical NLP fundamentals** (linguistics, morphology, syntax)
2. Simultaneously show how classical concepts serve as **inductive biases** in neural models
3. Teach correspondence between neural models and **probabilistic graphical models** (CRF, HMM)
4. Give linguistic interpretations of modern architectures
5. Progress to Transformers, LLMs, etc.

---

## 5. PhD in India vs. Abroad

### Key Points
- Research **quality** is comparable when normalized by number of students/labs
- **Volume** of research is naturally lower (fewer labs, fewer students)
- India's position is **improving**: Big tech research labs in India, better industry-academic ties
- Historical bias toward going abroad is outdating
- Students at IIT Delhi already publish 2-3 papers during undergrad — quality is there

### Advice
> "At least come work with us for a few months and then make an informed decision" — Anonto

---

## 6. Working with Resource Constraints

### Strategies
- **Collaborate with industry** — IBM, Google, Adobe, Microsoft provide compute
- Start with **toy models** and scale up (interpretability research can start on Colab)
- Convert models to **JAX** and leverage Google's TPU resources
- Resource constraints drive **creative solutions**: efficient code, novel optimization
- Focus on **model design** and **optimization** when exploration-heavy research isn't feasible

### Positive Framing
> "India sent a spacecraft to the Moon on less budget than a Hollywood movie" — constraint breeds innovation

---

## 7. Alternative Paradigms Beyond Scaling

### Current Reality
- We're running out of **text data** (Llama 3.1: 15T tokens, Granite: ~12T tokens)
- Synthetic data will fill the void for future emergent properties
- Search + verification → higher quality synthetic data (à la o1)

### Promising Directions
- **KAN (Kolmogorov-Arnold Networks)**: Learnable activation functions, fewer parameters needed
- **Neuro-symbolic techniques**: Combining logic-driven methods with neural approaches
- **Model distillation**: Distilling large model expertise into specialized smaller models
- **Specialized models**: Instead of one model for everything, domain-specific smaller models may suffice
- **Multimodal data**: Images, videos, audio — new data modalities beyond text

### Technology Evolution View
> Technologies go through natural evolution — scale up → saturate → find alternatives → compress → specialize

---

## 8. Human-AI Collaboration

### Current State
- AI as **assistant** is already mainstream (WhatsApp + Llama 3.2, coding assistants, email drafting)
- Non-technical users already benefit (doctors creating questionnaires via WhatsApp AI)
- Code LLMs are widely used by developers

### Future
- **Autonomous AI agents**: Not there yet — hallucination and reliability remain blockers
- Human-AI collaboration is the future, but deployment requires solving trust and safety first

---

## 9. Advice for Students & Job Seekers

### What Industry Looks For

| Skill | Details |
|-------|---------|
| **Fundamentals** | Linear algebra, probability, statistics, ML basics |
| **Implementation** | Write Transformer layer from scratch in PyTorch (no HuggingFace) |
| **Backpropagation** | Manually compute gradients, write gradient functions |
| **Transfer** | Apply theoretical knowledge to new, unseen problems |
| **Data Analysis** | Statistics, pandas, data exploration skills |

### Key Messages
- **Don't skip fundamentals** — they are the ceiling on how far you can go
- Using LLMs as plug-and-play will "hit a wall" — understanding internals is essential
- To build better small models, you must **understand why large models work**
- Nobody is an "expert" in LLMs — the field moves too fast
