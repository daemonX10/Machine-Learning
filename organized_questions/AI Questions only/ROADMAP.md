# 🗺️ Machine Learning & Data Science Roadmap — Zero to Advanced

> **67 Topics · 3,946 Questions · 567 Coding Challenges**
> Study these in order. Each stage builds on the previous one.
> Estimated timeline: 6–8 months (2–3 topics per week)

---

## How to Use This Roadmap

- **Stages** go from foundational → intermediate → advanced → specialized
- **Within each stage**, topics are grouped — study them together as they reinforce each other
- ✅ Check off topics as you complete them
- 🔢 Numbers in parentheses = total questions available for that topic
- 💡 "Why now" explains the dependency and reasoning
- 🏷️ Tags: `math` `ml-core` `dl` `tool` `role` `nlp` `cv` `rl` `ops`

---

## Stage 1 — Math & Statistics Foundations (Start Here)

> 🎯 **Goal:** Build the mathematical foundation that ALL of ML is built on. You cannot skip this.
> ⏱️ **Time:** Week 1–3

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 1 | [Statistics](Statistics.md) | 75 | `math` | Mean, variance, distributions, hypothesis testing — the language of data |
| 2 | [Probability](Probability.md) | 45 | `math` | Bayes' theorem, conditional probability — foundation of every ML model |
| 3 | [Linear Algebra](Linear%20Algebra.md) | 70 | `math` | Vectors, matrices, eigenvalues — neural networks ARE linear algebra |

**📌 Checkpoint:** You can explain Bayes' theorem, compute a matrix multiplication, interpret a covariance matrix, and run a hypothesis test.

---

## Stage 2 — Python & Data Tools

> 🎯 **Goal:** Master the tools you'll use every day. Code fluency before algorithms.
> ⏱️ **Time:** Week 4–6

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 4 | [PythonMl](PythonMl.md) | 100 | `tool` | Python is the lingua franca of ML — mastering it is non-negotiable |
| 5 | [NumPy](NumPy.md) | 70 | `tool` | Numerical computing — arrays, broadcasting, vectorization |
| 6 | [Pandas](Pandas.md) | 45 | `tool` | Data manipulation — DataFrames, groupby, merge, pivot |
| 7 | [SQL in ML](SQL%20in%20ML.md) | 55 | `tool` | Data lives in databases — querying, aggregations, joins for ML pipelines |

**📌 Checkpoint:** You can load a CSV, clean data, do EDA in Pandas, write SQL queries, and implement matrix operations in NumPy.

---

## Stage 3 — Data Fundamentals & Feature Engineering

> 🎯 **Goal:** Understand the full data pipeline from raw data to model-ready features.
> ⏱️ **Time:** Week 7–9

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 8 | [Data Processing](Data%20Processing.md) | 100 | `ml-core` | ETL, data cleaning, handling missing values, outliers, normalization |
| 9 | [Feature Engineering](Feature%20Engineering.md) | 50 | `ml-core` | Feature selection, extraction, encoding — THE most impactful skill for model quality |
| 10 | [Data Mining](Data%20Mining.md) | 60 | `ml-core` | Knowledge discovery patterns — association rules, anomaly detection basics |
| 11 | [Bias & Variance](Bias%20%26%20Variance.md) | 45 | `ml-core` | The fundamental tradeoff — underfitting vs overfitting, regularization |
| 12 | [Cost Function](Cost%20Function.md) | 43 | `ml-core` | MSE, cross-entropy, loss landscapes — what models actually optimize |

**📌 Checkpoint:** You can preprocess a messy dataset, engineer meaningful features, explain the bias-variance tradeoff, and choose an appropriate loss function.

---

## Stage 4 — Core Supervised Learning

> 🎯 **Goal:** Master the classical ML algorithms that form 70% of interview questions.
> ⏱️ **Time:** Week 10–14

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 13 | [Supervised Learning](Supervised%20Learning.md) | 70 | `ml-core` | Overview — classification vs regression, training/test split, cross-validation |
| 14 | [Linear Regression](Linear%20Regression.md) | 70 | `ml-core` | The simplest model — master this before anything else |
| 15 | [Logistic Regression](Logistic%20Regression.md) | 50 | `ml-core` | Classification counterpart of linear regression; sigmoid, decision boundary |
| 16 | [Decision Trees](Decision%20Trees.md) | 60 | `ml-core` | Intuitive, interpretable; foundation for ensembles (Random Forest, XGBoost) |
| 17 | [Naive Bayes](Naive%20Bayes.md) | 45 | `ml-core` | Probabilistic classifier — fast, good baseline; applies Bayes' theorem from Stage 1 |
| 18 | [K-Nearest Neighbors](K-Nearest%20Neighbors.md) | 45 | `ml-core` | Instance-based learning; simple but powerful; teaches distance metrics |
| 19 | [SVM](SVM.md) | 70 | `ml-core` | Margin-based classification; kernel trick; strong theoretical foundation |
| 20 | [Classification Algorithms](Classification%20Algorithms.md) | 52 | `ml-core` | Comprehensive comparison — ties together everything above |
| 21 | [Model Evaluation](Model%20Evaluation.md) | 55 | `ml-core` | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix — how to measure success |

**📌 Checkpoint:** Given a classification/regression task, you can choose the right algorithm, train it, tune hyperparameters, and evaluate it properly.

---

## Stage 5 — Ensemble Methods & Gradient Optimization

> 🎯 **Goal:** Master the models that win competitions and power production systems.
> ⏱️ **Time:** Week 15–17

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 22 | [Ensemble Learning](Ensemble%20Learning.md) | 70 | `ml-core` | Bagging, boosting, stacking — why combining models beats single models |
| 23 | [Random Forest](Random%20Forest.md) | 50 | `ml-core` | Bagging + decision trees; requires Stage 4 knowledge of trees |
| 24 | [Gradient Descent](Gradient%20Descent.md) | 50 | `ml-core` | The optimization engine behind EVERY model — SGD, Adam, learning rate |
| 25 | [XGBoost](XGBoost.md) | 36 | `ml-core` | Gradient boosted trees — the go-to for tabular data; competition winner |
| 26 | [LightGBM](LightGBM.md) | 45 | `ml-core` | Faster alternative to XGBoost; histogram-based; handles large datasets |
| 27 | [Optimization](Optimization.md) | 50 | `math` | Convex optimization, constraints, Lagrangians — deeper math behind gradient descent |

**📌 Checkpoint:** You can explain bagging vs boosting, tune XGBoost hyperparameters, and derive the gradient descent update rule.

---

## Stage 6 — Unsupervised Learning & Dimensionality

> 🎯 **Goal:** Learn to find structure in unlabeled data.
> ⏱️ **Time:** Week 18–20

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 28 | [Unsupervised Learning](Unsupervised%20Learning.md) | 55 | `ml-core` | Overview — clustering, dimensionality reduction, density estimation |
| 29 | [K-Means Clustering](K-Means%20Clustering.md) | 50 | `ml-core` | Most common clustering algorithm; simple and effective |
| 30 | [Cluster Analysis](Cluster%20Analysis.md) | 50 | `ml-core` | Hierarchical, DBSCAN, spectral — beyond K-Means |
| 31 | [PCA](PCA.md) | 48 | `ml-core` | Principal Component Analysis — requires linear algebra from Stage 1 |
| 32 | [Dimensionality Reduction](Dimensionality%20Reduction.md) | 50 | `ml-core` | t-SNE, UMAP, autoencoders — visualize and compress high-dimensional data |
| 33 | [Curse of Dimensionality](Curse%20of%20Dimensionality.md) | 40 | `ml-core` | Why high dimensions break things — motivates PCA and feature selection |
| 34 | [Anomaly Detection](Anomaly%20Detection.md) | 50 | `ml-core` | Isolation Forest, LOF, statistical methods — critical for fraud/security |

**📌 Checkpoint:** You can cluster customer data, reduce features with PCA, explain t-SNE visualizations, and detect outliers.

---

## Stage 7 — Neural Networks & Deep Learning Foundations

> 🎯 **Goal:** Transition from classical ML to deep learning.
> ⏱️ **Time:** Week 21–24

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 35 | [Neural Networks](Neural%20Networks.md) | 95 | `dl` | Perceptrons, backpropagation, activation functions — the foundation of all DL |
| 36 | [Deep Learning](Deep%20Learning.md) | 80 | `dl` | Multi-layer networks, regularization (dropout, batch norm), architectures |
| 37 | [CNN](CNN.md) | 50 | `dl` | Convolutional Neural Networks — for images, spatial data |
| 38 | [RNN](RNN.md) | 47 | `dl` | Recurrent Neural Networks — for sequences, time series, text |
| 39 | [Autoencoders](Autoencoders.md) | 50 | `dl` | Unsupervised DL; compression, denoising, generative foundations |
| 40 | [Transfer Learning](Transfer%20Learning.md) | 38 | `dl` | Use pretrained models — fine-tuning, domain adaptation; practical DL skill |

**📌 Checkpoint:** You can build a CNN for image classification, an RNN for sequence prediction, explain backpropagation mathematically, and fine-tune a pretrained model.

---

## Stage 8 — DL Frameworks (Hands-On)

> 🎯 **Goal:** Master the actual tools used to build and train models.
> ⏱️ **Time:** Week 25–27

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 41 | [TensorFlow](TensorFlow.md) | 70 | `tool` | Google's framework — tf.keras, eager execution, TF Serving |
| 42 | [PyTorch](PyTorch.md) | 50 | `tool` | Meta's framework — dynamic graphs, research-friendly, Lightning |
| 43 | [Keras](Keras.md) | 70 | `tool` | High-level API — rapid prototyping, model building, callbacks |
| 44 | [Scikit-Learn](Scikit-Learn.md) | 50 | `tool` | Classical ML library — pipelines, preprocessing, model selection |

**📌 Checkpoint:** You can implement end-to-end ML pipelines in scikit-learn, train deep learning models in both TensorFlow and PyTorch, and deploy a trained model.

---

## Stage 9 — NLP & Computer Vision

> 🎯 **Goal:** Master the two biggest application domains of deep learning.
> ⏱️ **Time:** Week 28–30

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 45 | [NLP](NLP.md) | 50 | `nlp` | Tokenization, embeddings, attention, transformers — requires RNN, DL |
| 46 | [Computer Vision](Computer%20Vision.md) | 54 | `cv` | Object detection, segmentation, image generation — requires CNN, DL |
| 47 | [LLMs](LLMs.md) | 63 | `nlp` | Large Language Models — GPT, BERT, fine-tuning, prompt engineering |
| 48 | [ChatGPT](ChatGPT.md) | 53 | `nlp` | RLHF, instruction tuning, practical use of LLMs |

**📌 Checkpoint:** You can build a text classifier, explain the Transformer architecture, fine-tune BERT, and discuss GPT's training process.

---

## Stage 10 — Generative Models & Reinforcement Learning

> 🎯 **Goal:** Master advanced DL paradigms — creating new data and learning from rewards.
> ⏱️ **Time:** Week 31–33

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 49 | [GANs](GANs.md) | 52 | `dl` | Generative Adversarial Networks — image synthesis, data augmentation |
| 50 | [Reinforcement Learning](Reinforcement%20Learning.md) | 70 | `rl` | Agent-environment interaction; rewards, policies, value functions |
| 51 | [Q-Learning](Q-Learning.md) | 44 | `rl` | Model-free RL algorithm; DQN, experience replay |
| 52 | [Genetic Algorithms](Genetic%20Algorithms.md) | 67 | `ml-core` | Evolutionary optimization — selection, crossover, mutation |
| 53 | [Recommendation Systems](Recommendation%20Systems.md) | 50 | `ml-core` | Collaborative filtering, content-based, hybrid — applies everything |

**📌 Checkpoint:** You can train a GAN, implement Q-learning, explain policy gradient methods, and build a recommendation engine.

---

## Stage 11 — Time Series & Specialized ML

> 🎯 **Goal:** Handle temporal data and specialized prediction tasks.
> ⏱️ **Time:** Week 34–35

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 54 | [Time Series](Time%20Series.md) | 50 | `ml-core` | ARIMA, seasonality, stationarity, LSTM for time data |
| 55 | [Explainable AI](Explainable%20AI.md) | 35 | `ml-core` | SHAP, LIME, feature importance — making models interpretable |
| 56 | [ML Design Patterns](ML%20Design%20Patterns.md) | 70 | `ml-core` | Reusable ML solutions — feature store, model versioning, A/B testing |

**📌 Checkpoint:** You can forecast stock prices with ARIMA/LSTM, explain any model's predictions with SHAP, and design an ML system end-to-end.

---

## Stage 12 — MLOps & Production

> 🎯 **Goal:** Deploy, monitor, and maintain ML models in production.
> ⏱️ **Time:** Week 36–37

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 57 | [MLOps](MLOps.md) | 50 | `ops` | CI/CD for ML, model registry, experiment tracking, monitoring |
| 58 | [LLMOps](LLMOps.md) | 50 | `ops` | LLM-specific operations — fine-tuning pipelines, serving, cost optimization |

**📌 Checkpoint:** You can set up an ML pipeline with experiment tracking, deploy a model with CI/CD, and monitor model drift in production.

---

## Stage 13 — Big Data Tools

> 🎯 **Goal:** Handle datasets too large for a single machine.
> ⏱️ **Time:** Week 38–39

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 59 | [Apache Spark](Apache%20Spark.md) | 55 | `tool` | Distributed computing — RDDs, DataFrames, MLlib |
| 60 | [Hadoop](Hadoop.md) | 50 | `tool` | HDFS, MapReduce — the foundation of big data (still relevant in enterprise) |

**📌 Checkpoint:** You can process terabytes of data with Spark, run distributed ML training, and explain the Hadoop ecosystem.

---

## Stage 14 — Alternative Languages & Tools

> 🎯 **Goal:** Broaden your toolkit — these are role-specific. Study based on job requirements.
> ⏱️ **Time:** As needed

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 61 | [R](R.md) | 60 | `tool` | Statistical computing — ggplot2, dplyr; popular in academia/biostatistics |
| 62 | [MATLAB](MATLAB.md) | 70 | `tool` | Matrix computations — signal processing, control systems, academia |
| 63 | [Julia](Julia.md) | 65 | `tool` | High-performance scientific computing — growing in ML research |
| 64 | [Scala](Scala.md) | 70 | `tool` | JVM language for Spark — functional programming, data engineering |

**📌 Checkpoint:** Pick 1–2 based on your target role. You should be able to implement basic ML workflows in the chosen language.

---

## Stage 15 — Role-Specific Preparation

> 🎯 **Goal:** Prepare for specific job titles — study the one matching your target role.
> ⏱️ **Time:** As needed (1–2 weeks per role)

| # | Topic | Qs | Tag | Why Now |
|---|-------|-----|-----|---------|
| 65 | [Data Analyst](Data%20Analyst.md) | 99 | `role` | SQL, Excel, dashboards, statistical analysis, business communication |
| 66 | [Data Engineer](Data%20Engineer.md) | 100 | `role` | Pipelines, ETL, data warehousing, Spark, Airflow |
| 67 | [Data Scientist](Data%20Scientist.md) | 100 | `role` | End-to-end ML, experimentation, statistics, communication |

**📌 Checkpoint:** You can answer role-specific behavioral and technical questions for your target position.

---

## 📊 Summary — Study Order at a Glance

```
STAGE 1  (Math)           → Statistics → Probability → Linear Algebra
     ↓
STAGE 2  (Tools)          → Python → NumPy → Pandas → SQL
     ↓
STAGE 3  (Data)           → Data Processing → Feature Eng → Data Mining → Bias/Variance → Cost Function
     ↓
STAGE 4  (Supervised)     → Supervised → Linear/Logistic Reg → Trees → NB → KNN → SVM → Classification → Evaluation
     ↓
STAGE 5  (Ensembles)      → Ensemble → Random Forest → Gradient Descent → XGBoost → LightGBM → Optimization
     ↓
STAGE 6  (Unsupervised)   → Unsupervised → K-Means → Cluster Analysis → PCA → Dim Reduction → Curse of Dim → Anomaly
     ↓
STAGE 7  (Deep Learning)  → Neural Networks → Deep Learning → CNN → RNN → Autoencoders → Transfer Learning
     ↓
STAGE 8  (Frameworks)     → TensorFlow → PyTorch → Keras → Scikit-Learn
     ↓
STAGE 9  (NLP & CV)       → NLP → Computer Vision → LLMs → ChatGPT
     ↓
STAGE 10 (Generative/RL)  → GANs → RL → Q-Learning → Genetic Algorithms → Recommendation Systems
     ↓
STAGE 11 (Specialized)    → Time Series → Explainable AI → ML Design Patterns
     ↓
STAGE 12 (MLOps)          → MLOps → LLMOps
     ↓
STAGE 13 (Big Data)       → Apache Spark → Hadoop
     ↓
STAGE 14 (Languages)      → R / MATLAB / Julia / Scala (pick based on role)
     ↓
STAGE 15 (Roles)          → Data Analyst / Data Engineer / Data Scientist
```

---

## 🎯 Grouping by Interview Priority

### 🔴 Must-Know (Top 15 — covers 80% of ML interviews)
1. Statistics & Probability
2. Linear Regression & Logistic Regression
3. Decision Trees
4. Random Forest & XGBoost
5. Neural Networks
6. Deep Learning (CNN, RNN)
7. Model Evaluation
8. Bias & Variance
9. Feature Engineering
10. Supervised Learning
11. Unsupervised Learning (K-Means, PCA)
12. Gradient Descent
13. NLP & LLMs
14. Python + NumPy + Pandas
15. Scikit-Learn

### 🟡 Important (Next 10 — frequently asked)
16. SVM
17. Ensemble Learning
18. Dimensionality Reduction
19. Data Processing
20. Transfer Learning
21. TensorFlow / PyTorch
22. Computer Vision
23. Recommendation Systems
24. Time Series
25. SQL in ML

### 🟢 Good to Know (Next 10 — shows depth)
26. Anomaly Detection
27. GANs
28. Reinforcement Learning
29. Autoencoders
30. MLOps / LLMOps
31. Explainable AI
32. ML Design Patterns
33. Cost Function & Optimization
34. Naive Bayes & KNN
35. ChatGPT

### ⚪ Niche (Remaining — role-specific)
36–67. Big Data tools, alternative languages, genetic algorithms, role-specific prep

---

## 🔄 Prerequisite Map

```
Statistics + Probability + Linear Algebra
        │
        ├── Python + NumPy + Pandas + SQL
        │       │
        │       ├── Data Processing → Feature Engineering
        │       │       │
        │       │       └── Bias & Variance + Cost Function
        │       │               │
        │       │               ├── Linear Regression → Logistic Regression
        │       │               ├── Decision Trees → Random Forest → XGBoost/LightGBM
        │       │               ├── KNN, Naive Bayes, SVM
        │       │               ├── Ensemble Learning
        │       │               └── Model Evaluation
        │       │
        │       ├── K-Means → Cluster Analysis
        │       ├── PCA → Dimensionality Reduction
        │       └── Anomaly Detection
        │
        └── Gradient Descent + Optimization
                │
                └── Neural Networks → Deep Learning
                        │
                        ├── CNN ──── Computer Vision
                        ├── RNN ──── NLP → LLMs → ChatGPT
                        ├── Autoencoders → GANs
                        ├── Transfer Learning
                        └── Reinforcement Learning → Q-Learning

Scikit-Learn ← (classical ML tool for Stages 4-6)
TensorFlow / PyTorch / Keras ← (deep learning tool for Stages 7+)
Apache Spark / Hadoop ← (big data for production scale)
MLOps / LLMOps ← (deployment & monitoring)
```

---

## 🏃 Quick Paths by Goal

### "I have an ML interview in 2 weeks"
Stages 1 → 3 → 4 → 5 (focus on top 15 must-know topics)

### "I'm transitioning from software engineering to ML"
Stages 1 → 2 → 3 → 4 → 7 → 8

### "I want to be a Data Scientist"
Stages 1 → 2 → 3 → 4 → 5 → 6 → 11 → Stage 15 (Data Scientist)

### "I want to be a Data Engineer"
Stages 2 → 3 → 13 → Stage 15 (Data Engineer)

### "I want to work in NLP / LLMs"
Stages 1 → 2 → 4 → 7 → 8 → 9 → 12

### "I want to work in Computer Vision"
Stages 1 → 2 → 4 → 7 → 8 → Stage 9 (CV focus)

### "I want to do ML research"
All stages in order, with extra focus on Stages 1, 5, 7, 10
