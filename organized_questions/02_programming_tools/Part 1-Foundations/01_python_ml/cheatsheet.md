# 🔥 PYTHON ML — LAST-MOMENT REVISION CHEAT SHEET

> **100 questions distilled into one scannable page. Formulas → Tips → Mnemonics → Code.**

---

## 1. PYTHON CORE QUICK HITS

### Python 2 vs 3 (5 Key Diffs)

| Feature | Python 2 | Python 3 |
|---------|----------|----------|
| Print | `print "hi"` (statement) | `print("hi")` (function) |
| Division | `5/2 = 2` (floor) | `5/2 = 2.5` (true) |
| Strings | ASCII default | Unicode default |
| Range | `range()` → list | `range()` → generator |
| Exception | `except E, e:` | `except E as e:` |

> **Tip**: Python 3 is the **only** choice for ML. All modern libs require it.

### Memory Management

```
Reference Counting (primary, ~95% of cleanup)
  → ref count hits 0 → immediately freed

Cyclic Garbage Collector (secondary)
  → Gen 0 (new, frequent) → Gen 1 (survived) → Gen 2 (long-lived, rare)
```

### PEP 8 Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Functions/vars | `snake_case` | `train_model()` |
| Classes | `PascalCase` | `RandomForest` |
| Constants | `UPPER_CASE` | `LEARNING_RATE` |
| Indent | 4 spaces | — |
| Max line | 79 chars | — |

**Tools**: `black` (formatter), `flake8` (linter), `isort` (imports)

### Scope — LEGB Rule 🧠

> **L**ocal → **E**nclosing → **G**lobal → **B**uilt-in

- **Read** global: works without keyword
- **Modify** global: needs `global` keyword
- **Modify** enclosing: needs `nonlocal` keyword
- **Best practice**: Avoid globals — use classes or pass as args

### Data Structures at a Glance

| Structure | Mutable | Ordered | Dupes | Syntax |
|-----------|---------|---------|-------|--------|
| **List** | ✅ | ✅ | ✅ | `[1,2,3]` |
| **Tuple** | ❌ | ✅ | ✅ | `(1,2,3)` |
| **Set** | ✅ | ❌ | ❌ | `{1,2,3}` |
| **Dict** | ✅ | ✅ (3.7+) | keys ❌ | `{k:v}` |

- **Dict** = hash table, **O(1)** lookup. Keys must be **immutable + hashable**
- **Set** = O(1) membership test (`x in my_set`)

### List Comp vs Generator

```python
[x**2 for x in range(1_000_000)]  # List: ~8 MB in memory
(x**2 for x in range(1_000_000))  # Generator: ~120 bytes ← USE FOR BIG DATA
```

### Function Args

```python
def f(required, *args, default=10, **kwargs):
#     ↑ must      ↑ tuple    ↑ keyword    ↑ dict
```

### Decorators

```python
@decorator        # Equivalent to:
def func(): ...   # func = decorator(func)
```

Common: `@timing`, `@lru_cache`, `@validator`, `@staticmethod`

### `__slots__` — 3x Memory Savings

```python
class Point:
    __slots__ = ('x', 'y')  # No __dict__ → ~100 bytes vs ~300 bytes per instance
```

Use for millions of lightweight objects (tree nodes, streaming data points).

### Context Managers

```python
with open('model.pkl', 'rb') as f:    # __enter__ + __exit__ = guaranteed cleanup
    model = pickle.load(f)             # Even if exception occurs
```

Use for: files, DB connections, GPU memory, timing, temp seeds.

---

## 2. ML LIBRARIES CHEAT TABLE

| Library | One-Liner Purpose |
|---------|------------------|
| **NumPy** | N-dim arrays, vectorized math, 10-100x faster than lists |
| **Pandas** | DataFrames, data cleaning, EDA backbone |
| **Matplotlib** | Low-level plotting, full customization |
| **Seaborn** | High-level statistical plots, beautiful defaults |
| **Scikit-learn** | Classical ML: preprocess → train → evaluate → tune |
| **TensorFlow/Keras** | Deep learning (Google), `tf.keras` for easy model building |
| **PyTorch** | Deep learning (Meta), research-friendly, dynamic graphs |
| **XGBoost/LightGBM** | Gradient boosting, Kaggle-winning algorithms |
| **spaCy / HuggingFace** | NLP: tokenization, NER, transformers |
| **OpenCV** | Computer vision, image processing |
| **SciPy** | Advanced scientific: optimization, stats tests, sparse matrices |

### Task → Library Quick Map

```
Data Loading/Cleaning  →  Pandas
Numerical Operations   →  NumPy
Visualization          →  Matplotlib + Seaborn
Classical ML           →  Scikit-learn
Deep Learning          →  TensorFlow / PyTorch
NLP                    →  spaCy, HuggingFace Transformers
Computer Vision        →  OpenCV, torchvision
Stat Tests / Optim     →  SciPy
```

### NumPy vs SciPy

| | NumPy | SciPy |
|-|-------|-------|
| Core | `ndarray`, basic linalg | Builds ON NumPy |
| Stats | `mean`, `std`, `random` | 100+ distributions, hypothesis tests |
| Optimization | ❌ | `scipy.optimize.minimize` |
| Sparse | ❌ | `scipy.sparse` (CSR, CSC) |
| Signal | ❌ | `scipy.signal` (FFT, filters) |

### Sklearn API Pattern (Universal)

```python
model = Algorithm(**hyperparams)
model.fit(X_train, y_train)        # Learn
model.predict(X_test)              # Predict
model.score(X_test, y_test)        # Evaluate
```

---

## 3. 📐 THE FORMULA SHEET

### Loss / Error Metrics

| Formula | Name |
|---------|------|
| $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | Mean Squared Error |
| $\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | Mean Absolute Error |
| $\text{RMSE} = \sqrt{\text{MSE}}$ | Root Mean Squared Error |
| $R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | R-Squared |

### Gradient Descent

$$w = w - \alpha \cdot \nabla L$$

- **Gradient (MSE)**: $\nabla L = -\frac{2}{n}X^T(y - \hat{y})$
- **Normal Equation** (closed-form): $w = (X^TX)^{-1}X^Ty$
- Use `np.linalg.pinv` (pseudo-inverse) to handle singular matrices

| Variant | Samples/Update | Speed | Convergence |
|---------|---------------|-------|-------------|
| **Batch GD** | All N | Slow | Smooth |
| **SGD** | 1 | Fast | Noisy |
| **Mini-batch** | B (32-256) | Best | Balanced ← **most common** |

### Feature Scaling

| Method | Formula | Range | When |
|--------|---------|-------|------|
| **Z-score** | $z = \frac{x - \mu}{\sigma}$ | ~[-3, 3] | Normal dist, SVM, LogReg |
| **Min-Max** | $\frac{x - x_{min}}{x_{max} - x_{min}}$ | [0, 1] | Neural nets, bounded data |
| **Robust** | $\frac{x - \text{median}}{IQR}$ | varies | Outlier-heavy data |

### Decision Trees

| | Formula |
|-|---------|
| **Gini** | $\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2$ &nbsp;&nbsp; (binary: $2p(1-p)$) |
| **Entropy** | $H = -\sum_{i=1}^{C} p_i \log_2 p_i$ |
| **Info Gain** | $\text{Gain} = \text{Gini}_{parent} - \frac{n_L}{n}\text{Gini}_L - \frac{n_R}{n}\text{Gini}_R$ |

### Classification Metrics

```
                Predicted +    Predicted -
Actual +    │     TP       │      FN      │
Actual -    │     FP       │      TN      │
```

| Metric | Formula | Remember As |
|--------|---------|-------------|
| **Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ | Overall correctness |
| **Precision** | $\frac{TP}{TP+FP}$ | "Of predicted +, how many correct?" |
| **Recall** | $\frac{TP}{TP+FN}$ | "Of actual +, how many found?" |
| **F1** | $2\frac{P \times R}{P + R}$ | Harmonic mean of P & R |
| **Specificity** | $\frac{TN}{TN+FP}$ | True negative rate |
| **F-beta** | $(1+\beta^2)\frac{P \cdot R}{\beta^2 P + R}$ | β<1→favor P, β>1→favor R |

### Logistic Regression

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots$$

- **Odds Ratio**: $e^{\beta_i}$ → 1-unit increase in $x_i$ multiplies odds by $e^{\beta_i}$
- **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$

### Bias-Variance Tradeoff

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

### K-Means Objective

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

### Self-Attention (Transformer Core)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### GAN Loss

$$\min_G \max_D \;\; \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$$

### Regularization Penalties

| Type | Added to Loss | Effect |
|------|--------------|--------|
| **L1 (Lasso)** | $+ \lambda\sum\|w_i\|$ | Drives weights to **exactly 0** → feature selection |
| **L2 (Ridge)** | $+ \lambda\sum w_i^2$ | Shrinks weights **toward 0** → handles multicollinearity |
| **Elastic Net** | $+ \lambda_1\sum\|w_i\| + \lambda_2\sum w_i^2$ | Best of both |

### Perceptron Update Rule

$$w \mathrel{+}= \eta \cdot (y_{true} - y_{pred}) \cdot x$$

> Only converges for **linearly separable** data.

### Outlier Detection — IQR Method

$$\text{Valid range} = [Q_1 - 1.5 \times IQR, \;\; Q_3 + 1.5 \times IQR]$$

where $IQR = Q_3 - Q_1$

---

## 4. ALGORITHM QUICK REFERENCE CARDS

### Linear Regression

| | |
|-|-|
| **What** | Predict continuous target via linear combination |
| **Methods** | Normal Equation $O(n^3)$ or Gradient Descent $O(ndi)$ |
| **When** | Linear relationships, interpretability needed |
| **Key HP** | `fit_intercept`, regularization α |

### Logistic Regression

| | |
|-|-|
| **What** | Binary/multi-class classification via sigmoid |
| **When** | Baseline classifier, interpretable coefficients |
| **Key HP** | `C` (inverse regularization), `penalty` (l1/l2) |
| **Tip** | Standardize features before interpreting coefficients |

### Decision Tree

| | |
|-|-|
| **What** | Recursive splits on features using Gini/Entropy |
| **Pros** | Interpretable, no scaling needed, handles non-linear |
| **Cons** | Overfits easily, unstable (small data change → diff tree) |
| **Key HP** | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| **Criteria** | Gini (faster) vs Entropy (more balanced) |

### Random Forest

| | |
|-|-|
| **What** | Ensemble of decision trees (bagging) |
| **Reduces** | **Variance** (averaging decorrelated trees) |
| **Key HP** | `n_estimators`, `max_depth`, `max_features` |
| **Tip** | Set `n_jobs=-1` for parallel training |

### K-Means Clustering

| | |
|-|-|
| **What** | Partition data into K spherical clusters |
| **Steps** | Init centroids → Assign → Update → Repeat |
| **Complexity** | $O(n \times K \times i \times d)$ |
| **Choose K** | Elbow method or Silhouette score |
| **Weakness** | Sensitive to init (use K-Means++), assumes spherical clusters |

### SVM (Support Vector Machine)

| | |
|-|-|
| **What** | Find max-margin hyperplane between classes |
| **Kernels** | linear, rbf, poly |
| **Key HP** | `C` (regularization), `gamma` (rbf width), `kernel` |
| **Needs** | Feature scaling |

### KNN (K-Nearest Neighbors)

| | |
|-|-|
| **What** | Classify by majority vote of K nearest neighbors |
| **Complexity** | $O(nd)$ per prediction (lazy learner) |
| **Key HP** | `n_neighbors`, `metric` (euclidean, manhattan) |
| **Needs** | Feature scaling (distance-based) |

### Naive Bayes

| | |
|-|-|
| **What** | Probabilistic classifier assuming feature independence |
| **When** | Text classification (spam), baseline |
| **Variants** | Gaussian, Multinomial (text), Bernoulli |
| **Doesn't need** | Feature scaling |

### PCA (Principal Component Analysis)

| | |
|-|-|
| **What** | Unsupervised linear dimensionality reduction |
| **How** | Project onto axes of max variance (eigenvectors) |
| **When** | High-dim data, viz, noise reduction, multicollinearity |
| **Tip** | Scale first! Use 95% cumulative variance threshold |

### Neural Network

| | |
|-|-|
| **What** | Layered network: input → hidden(s) → output |
| **Training** | Forward prop → Loss → **Backprop** → Weight update |
| **Activations** | Sigmoid, ReLU, Softmax (output) |
| **Optimizer** | **Adam** = default (adaptive lr + momentum) |

### Ensemble Methods Summary

| Method | How | Reduces | Example |
|--------|-----|---------|---------|
| **Bagging** | Parallel trees on random subsets | **Variance** | Random Forest |
| **Boosting** | Sequential, fix previous errors | **Bias** | XGBoost, GradientBoosting |
| **Stacking** | Meta-model combines base models | **Both** | VotingClassifier |

> 🧠 **Mnemonic**: **B**agging→**V**ariance, **B**oosting→**B**ias

### Reinforcement Learning — Q-Learning

| Component | Description |
|-----------|-------------|
| **Agent** | Learner |
| **State (s)** | Current situation |
| **Action (a)** | Decision |
| **Reward (r)** | Feedback signal |
| **Policy (π)** | State → Action mapping |
| **Q(s,a)** | Expected reward for action a in state s |

Update: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

### GAN Architecture

```
Noise (z) → [Generator G] → Fake Data → [Discriminator D] → Real/Fake?
Real Data →→→→→→→→→→→→→→→→→→→→→→→→←
```

| Variant | Improvement |
|---------|------------|
| DCGAN | Conv layers, stable training |
| WGAN | Wasserstein distance, better convergence |
| CycleGAN | Unpaired image-to-image |
| StyleGAN | Fine-grained control |

### Transfer Learning Strategies

| Strategy | When | Data Size |
|----------|------|-----------|
| **Feature extraction** (freeze all) | Similar domain | Small |
| **Fine-tune top layers** | Slightly different | Medium |
| **Full fine-tune** | Very different | Large |

---

## 5. DATA PREPROCESSING PIPELINE

### The Checklist ✅

```
1. INSPECT:  df.shape, df.info(), df.describe(), df.isnull().sum()
2. CLEAN:    Drop dupes → Handle missing → Fix types → Remove outliers
3. ENCODE:   Categorical → Numeric
4. SCALE:    Standardize/Normalize numeric features
5. SPLIT:    Train / Validation / Test  (stratify=y for classification!)
6. BALANCE:  Handle class imbalance (SMOTE, class_weight)
```

> **60-80% of a data scientist's time** is spent on data cleaning!

### Missing Data Decision Tree

```
Missing %?
  < 5%           → Drop rows
  5-40%          → Impute:
                     Numeric?     → Median (robust to outliers)
                     Categorical? → Mode
                     Complex?     → KNNImputer / IterativeImputer (MICE)
  > 60%          → Drop column
  Missingness informative? → Add indicator column: df['col_missing'] = df['col'].isnull()
```

> **Always investigate WHY** data is missing (MCAR, MAR, MNAR) before choosing strategy.

### Encoding Decision Tree

```
Natural order exists? → YES → Ordinal Encoding (low=0, med=1, high=2)
                      → NO  → How many unique values?
                                ≤ 10   → One-Hot Encoding (pd.get_dummies, drop_first=True)
                                10-50  → Binary / Target Encoding
                                > 50   → Target Encoding / Embeddings (DL)
```

> **Dummy variable trap**: Always use `drop_first=True` to avoid multicollinearity.

### Which Algorithms Need Scaling?

| NEEDS Scaling ⚠️ | DOESN'T Need Scaling ✅ |
|-------------------|------------------------|
| SVM | Decision Tree |
| KNN | Random Forest |
| PCA | XGBoost / LightGBM |
| Neural Networks | Naive Bayes |
| Logistic Regression | |
| Linear Regression (for regularization) | |
| K-Means | |

### ⚡ Golden Rule

> **Fit scaler on TRAIN only → Transform BOTH train and test.**
> Violating this = **DATA LEAKAGE** (inflated test scores).

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit + transform
X_test = scaler.transform(X_test)        # transform ONLY
```

---

## 6. MODEL EVALUATION CHEAT CARD

### Confusion Matrix

```
                  Predicted +    Predicted -
Actual +     │      TP        │      FN       │
Actual -     │      FP        │      TN       │
```

### When to Use Which Metric 🎯

| Scenario | Prioritize | Why |
|----------|-----------|-----|
| **Spam detection** | **Precision** | Don't mark real emails as spam |
| **Cancer screening** | **Recall** | Don't miss actual cancer |
| **Fraud detection** | **Recall** | Must catch all frauds |
| **Search engines** | **Precision** | Top results must be relevant |
| **Balanced data** | **Accuracy** | Equal cost for all errors |
| **Imbalanced data** | **F1 / AUC-PR** | Accuracy is misleading |

### ROC-AUC Interpretation

| AUC | Quality |
|-----|---------|
| 1.0 | Perfect |
| 0.9-1.0 | Excellent |
| 0.8-0.9 | Good |
| 0.7-0.8 | Fair |
| 0.5 | Random guess |
| < 0.5 | Worse than random (check labels) |

> For **highly imbalanced** data → use **PR-AUC** instead of ROC-AUC.

### Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MSE** | $\frac{1}{n}\sum(y-\hat{y})^2$ | Penalizes large errors heavily |
| **MAE** | $\frac{1}{n}\sum\|y-\hat{y}\|$ | Robust to outliers |
| **RMSE** | $\sqrt{MSE}$ | Same unit as target |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | % variance explained (1=perfect, 0=mean) |

### Clustering Metrics

| Metric | Range | Meaning |
|--------|-------|---------|
| **Silhouette** | [-1, 1] | Higher = better separated clusters |
| **Inertia (WCSS)** | 0 to ∞ | Lower = tighter clusters (use elbow) |
| **Davies-Bouldin** | 0 to ∞ | Lower = better |
| **Calinski-Harabasz** | 0 to ∞ | Higher = better |

### Learning Curve Interpretation

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Both scores **low** | **Underfitting** (high bias) | More features, complex model |
| Train **high**, val **low** | **Overfitting** (high variance) | More data, regularize, simplify |
| Both **converge high** | **Good fit** ✅ | Keep it! |
| Gap stays large | Fundamental overfitting | Simplify model |

---

## 7. CROSS-VALIDATION & TUNING

### CV Types

| Method | Best For | Key Detail |
|--------|----------|-----------|
| **K-Fold** (k=5,10) | General purpose | Each fold is val once |
| **Stratified K-Fold** | Imbalanced classification | Preserves class proportions |
| **LOO** | Very small datasets | k = n (expensive) |
| **Time Series Split** | Temporal data | No future leakage |
| **Group K-Fold** | Grouped data (patients) | Groups don't span folds |
| **Nested CV** | Unbiased HP selection | Outer=eval, inner=tune |
| **Repeated K-Fold** | More robust estimates | Multiple random splits |

### CV Fits in the Workflow

```
Data → [Train+Val | Test (untouched)]
            ↓
       K-Fold CV on Train+Val → Select best model/HP
            ↓
       Final eval on Test set (ONE TIME ONLY)
```

### Hyperparameter Tuning Methods

| Method | Strategy | Speed | Best For |
|--------|----------|-------|----------|
| **Grid Search** | Try ALL combos ($n^k$) | Slow | Small search space |
| **Random Search** | Sample random combos | Fast | Large space (preferred) |
| **Bayesian (Optuna)** | Guided by past results | Smartest | Production optimization |
| **Halving Grid** | Progressive elimination | Medium | Large candidates |

> **Tip**: Use Random Search first to narrow range, then Grid Search to fine-tune.

### Common Hyperparams Per Algorithm

| Algorithm | Key Hyperparameters |
|-----------|-------------------|
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split`, `max_features` |
| **XGBoost** | `learning_rate`, `max_depth`, `n_estimators`, `subsample` |
| **SVM** | `C`, `kernel`, `gamma` |
| **KNN** | `n_neighbors`, `metric`, `weights` |
| **Neural Net** | `learning_rate`, `batch_size`, `epochs`, `layers`, `dropout` |
| **Logistic Reg** | `C`, `penalty` (l1/l2), `solver` |

### Split Ratios

| Splits | Ratio | When |
|--------|-------|------|
| Train / Test | 80/20 | Quick experiments |
| Train / Val / Test | 60/20/20 | Full pipeline |
| With CV | 80/20 + 5-fold CV on 80% | Best practice |

---

## 8. OVERFITTING vs UNDERFITTING FIX GUIDE

### Diagnosis

| Signal | Problem |
|--------|---------|
| Train ✅ high, Test ❌ low | **Overfitting** (high variance) |
| Train ❌ low, Test ❌ low | **Underfitting** (high bias) |
| Train ✅ high, Test ✅ high | **Good fit** ✅ |

### 🧠 Analogy

> **Bias** = consistently shooting **left of target** (systematic error)
> **Variance** = shots **scattered everywhere** (inconsistency)

### Overfitting Fixes (Reduce Variance)

| Fix | How |
|-----|-----|
| **More data** | Harder to memorize |
| **Regularization** | L1/L2 penalty on weights |
| **Dropout** | Randomly disable neurons (DL) |
| **Early stopping** | Stop when val loss plateaus |
| **Simpler model** | Fewer layers/trees/features |
| **Ensemble (Bagging)** | Average multiple models |
| **Data augmentation** | Synthetic training samples |
| **Feature selection** | Remove noisy features |
| **Pruning** | `max_depth`, `min_samples_leaf` |

### Underfitting Fixes (Reduce Bias)

| Fix | How |
|-----|-----|
| **More features** | Feature engineering |
| **Complex model** | More layers, higher degree |
| **Less regularization** | Lower α |
| **Train longer** | More epochs |
| **Ensemble (Boosting)** | Fix previous errors sequentially |

### Regularization Quick Ref

| Type | Penalty | Effect | Use When |
|------|---------|--------|----------|
| **L1 (Lasso)** | $\lambda\sum\|w_i\|$ | Feature selection (zeros out weights) | Many irrelevant features |
| **L2 (Ridge)** | $\lambda\sum w_i^2$ | Weight shrinkage | Multicollinearity |
| **Elastic Net** | L1 + L2 | Both | Correlated + irrelevant features |

> **Large α** → strong penalty → simpler model → more bias, less variance
> **Small α** → weak penalty → complex model → less bias, more variance
> **Always scale features before regularization!**

---

## 9. DEEP LEARNING ESSENTIALS

### Neural Network Training Loop

```
Forward Propagation        →  Compute predictions
    ↓
Loss Calculation           →  MSE, Cross-Entropy
    ↓
Backpropagation            →  Compute gradients (chain rule)
    ↓
Weight Update              →  w = w - lr × gradient
    ↓
Repeat for N epochs
```

### Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **Sigmoid** | $\frac{1}{1+e^{-z}}$ | Binary output (0-1) |
| **ReLU** | $\max(0, z)$ | Hidden layers (default) |
| **Softmax** | $\frac{e^{z_i}}{\sum e^{z_j}}$ | Multi-class output |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | Hidden layers (-1 to 1) |

### Transfer Learning

```python
# TensorFlow/Keras
base = ResNet50(weights='imagenet', include_top=False)
base.trainable = False                    # Freeze
x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation='relu')(x)
out = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=out)

# PyTorch
model = models.resnet50(pretrained=True)
for p in model.parameters(): p.requires_grad = False  # Freeze
model.fc = nn.Linear(2048, num_classes)                # Replace head
```

| Strategy | Freeze | LR | Data Size |
|----------|--------|-----|-----------|
| Feature extraction | All layers | Normal | Small |
| Fine-tune top | All except last N | **Very low** (1e-5) | Medium |
| Full fine-tune | Nothing | Very low | Large |

### Key DL Frameworks

| | TensorFlow/Keras | PyTorch |
|-|-----------------|---------|
| Style | Declarative (`Sequential`) | Imperative (`nn.Module`) |
| Best for | Production, deployment | Research, flexibility |
| Serving | TF Serving, TFLite | TorchServe, ONNX |

---

## 10. NLP QUICK REFERENCE

### Evolution of NLP

```
Rule-based → Bag-of-Words → TF-IDF → Word2Vec (2013) → Attention (2015)
    → Transformers (2017) → BERT/GPT (2018-19) → LLMs (2022+)
```

### Spam Detection Pipeline

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', MultinomialNB(alpha=0.1))
])
# ~97% accuracy
```

### Sentiment Analysis Comparison

| Method | Accuracy | Speed | Data Needed |
|--------|----------|-------|-------------|
| TF-IDF + LogReg | ~88% | Very fast | Medium |
| LSTM/GRU | ~90% | Medium | Large |
| **BERT fine-tuned** | ~94% | Slow train | Small-Medium |
| GPT zero-shot | ~90% | Fast (no train) | None |

### HuggingFace One-Liners

```python
from transformers import pipeline
classifier = pipeline('sentiment-analysis')       # Sentiment
ner = pipeline('ner', grouped_entities=True)       # Named entities
qa = pipeline('question-answering')                # QA
gen = pipeline('text-generation', model='gpt2')    # Text gen
```

---

## 11. DEPLOYMENT & PRODUCTION

### Docker — Reproducibility

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model/ ./model/
COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### FastAPI — Model Serving Pattern

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

app = FastAPI()
model = joblib.load('model.pkl')

class Request(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(req: Request):
    X = np.array(req.features).reshape(1, -1)
    return {"prediction": int(model.predict(X)[0]),
            "probability": float(model.predict_proba(X).max())}

@app.get("/health")
def health(): return {"status": "healthy"}
```

### MLflow — Experiment Tracking

```python
import mlflow
with mlflow.start_run(run_name="v2_tuned"):
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model", registered_model_name="MyModel")
```

### Microservices Architecture

```
[Client] → [API Gateway] → [Data Ingestion] → [Feature Engine] → [Model Server] → [Monitor]
```

### Scaling Strategies

| Strategy | Tool | When |
|----------|------|------|
| Parallel sklearn | `n_jobs=-1` | Multi-core |
| Large DataFrames | **Dask** | Larger than memory |
| Distributed ML | **Ray** | Cluster computing |
| Fast inference | **ONNX Runtime** | Production serving |
| Cache predictions | **Redis** | Frequent repeat queries |
| Orchestration | **Kubernetes** | Auto-scaling containers |

### Error Handling Checklist

- **Input validation**: Pydantic models with validators
- **Fallback model**: Load backup if primary fails
- **Health checks**: `/health` endpoint
- **Structured logging**: JSON format, never expose stack traces
- **Circuit breakers**: For dependent services
- **Timeouts**: On all external calls

---

## 12. 🧠 MNEMONICS & INTERVIEW POWER TIPS

### Must-Know Mnemonics

| Mnemonic | Meaning |
|----------|---------|
| **LEGB** | Scope: Local → Enclosing → Global → Built-in |
| **GIGO** | Garbage In = Garbage Out (data quality!) |
| **Bagging→Variance, Boosting→Bias** | What ensemble methods reduce |
| **Fit on train, transform both** | Prevent data leakage |

### Quick-Fire Interview Answers

| Question | One-Line Answer |
|----------|----------------|
| Precision vs Recall? | Precision = "of predicted +, correct?", Recall = "of actual +, found?" |
| Bias vs Variance? | Bias = systematic error, Variance = sensitivity to training data |
| Overfitting fix? | More data, regularization, simpler model, dropout, early stopping |
| Why scale? | Distance/gradient-based algos are sensitive to magnitude |
| Why pipelines? | Prevent data leakage, cleaner code, easier deployment |
| L1 vs L2? | L1 = feature selection (zeros), L2 = shrinkage (small but non-zero) |
| Accuracy paradox? | 99% accuracy meaningless if 99% of data is one class |
| Cold start? | Recommender problem: no history for new users/items |
| GIL? | Python can't do true multi-threading for CPU; use multiprocessing |
| Feature scaling for trees? | NOT needed — trees split on thresholds, not distances |

### Key Quotes

> "It's not who has the best algorithm that wins. It's who has the most data." — **Andrew Ng**

> "Data cleaning is **60-80%** of a data scientist's time."

> "Always start with a **simple baseline**, then iterate."

### Top Interview Tips by Topic

| Topic | Power Tip |
|-------|----------|
| **Data cleaning** | Mention systematic approach: inspect → missing → dupes → types → outliers |
| **Feature encoding** | `drop_first=True` to avoid dummy variable trap |
| **Model evaluation** | "What is the business cost of different error types?" |
| **Cross-validation** | "I use Stratified K-Fold because the target is imbalanced" |
| **Regularization** | "Features must be scaled before regularization" |
| **Deployment** | Mention Docker + FastAPI + MLflow as production stack |
| **Scaling** | Mention GIL → use multiprocessing for CPU tasks |
| **Explainability** | SHAP for global + local, LIME for model-agnostic local |
| **Profiling** | "I profile before optimizing — bottleneck is usually I/O, not training" |
| **Testing ML** | Smoke test (overfit tiny data) + regression test (accuracy threshold) |

---

## 13. IMBALANCED DATA STRATEGIES

| Strategy | Level | Code/Tool |
|----------|-------|----------|
| **SMOTE** | Data | `SMOTE().fit_resample(X, y)` |
| **Random undersample** | Data | `RandomUnderSampler()` |
| **Random oversample** | Data | `RandomOverSampler()` |
| **Class weights** | Algorithm | `class_weight='balanced'` |
| **Threshold tuning** | Post-hoc | Adjust from 0.5 based on PR curve |
| **Ensemble** | Algorithm | `BalancedRandomForest` |
| **Anomaly detection** | Algorithm | Isolation Forest (minority = anomaly) |
| **Collect more data** | Data | Best if possible |

> **Never use raw accuracy** for imbalanced data. Use **ROC-AUC**, **PR-AUC**, or **F1**.

---

## 14. FEATURE SELECTION DECISION FLOW

### Methods Table

| Category | Method | How | Speed |
|----------|--------|-----|-------|
| **Filter** | Variance Threshold | Drop near-zero variance | Fast |
| **Filter** | Correlation | Remove corr > 0.95 | Fast |
| **Filter** | Chi² / ANOVA / MI | Statistical test | Fast |
| **Wrapper** | RFE | Iteratively remove weakest | Slow |
| **Wrapper** | Forward/Backward | Greedily add/remove | Slow |
| **Embedded** | Lasso (L1) | Coefficients → 0 | Medium |
| **Embedded** | Tree importance | `feature_importances_` | Medium |

### Decision Flow

```
Start → Remove zero-variance → Remove corr > 0.95
  → Dataset size?
      Small?         → RFE (thorough, slow)
      Large?         → Tree importance (fast)
      Linear model?  → Lasso L1 (automatic)
  → Use MULTIPLE methods, take intersection for robustness
```

---

## 15. QUICK CODE SNIPPETS

### Pandas EDA (4 Lines)

```python
df.shape                 # (rows, cols)
df.info()                # dtypes, non-null counts
df.describe()            # statistical summary
df.isnull().sum()        # missing values per column
```

### Train-Test Split (Stratified)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

### Full Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1))
])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

### Classification Report

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### Save / Load Model

```python
import joblib
joblib.dump(model, 'model.pkl')           # Save
model = joblib.load('model.pkl')          # Load
```

### IQR Outlier Removal

```python
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_clean = df[(df['col'] >= Q1 - 1.5*IQR) & (df['col'] <= Q3 + 1.5*IQR)]
```

### Correlation Heatmap

```python
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

### SHAP Explainability

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

---

## 16. RECOMMENDATION SYSTEMS

| Type | Method | Use Case |
|------|--------|----------|
| **Collaborative Filtering** | User-user / Item-item similarity | "Users like you also liked…" |
| **Content-Based** | Item feature similarity (TF-IDF + cosine) | "Because you watched action movies…" |
| **Matrix Factorization** | SVD / NMF on rating matrix | Sparse rating matrices |
| **Hybrid** | Combines collaborative + content | Netflix, Spotify |

> **Cold Start Problem**: New users/items have no interaction history → use content-based fallback.

---

## 17. COMPLETE ML PROJECT CHECKLIST

```
 1. Define problem (classification / regression / clustering?)
 2. Collect & load data (Pandas)
 3. EDA (shape, describe, viz, correlations)
 4. Clean (missing, dupes, types, outliers)
 5. Feature engineering (domain knowledge, interactions)
 6. Encode categoricals
 7. Scale numerics
 8. Split data (stratify for classification!)
 9. Handle imbalance (if needed)
10. Train baseline model (LogReg or DummyClassifier)
11. Try multiple algorithms (RF, XGB, SVM)
12. Cross-validate
13. Hyperparameter tune (RandomSearch → GridSearch)
14. Evaluate on test set (ONCE)
15. Explain model (SHAP, feature importance)
16. Save model (joblib / pickle / MLflow)
17. Deploy (FastAPI + Docker)
18. Monitor (data drift, accuracy, latency)
```

---

> **Last tip**: When in doubt during an interview, say:
> *"I would start with a simple baseline, evaluate with cross-validation, and iterate based on the error analysis."*
> This shows structured thinking that interviewers love. 🎯
