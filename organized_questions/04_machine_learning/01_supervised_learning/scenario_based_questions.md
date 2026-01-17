# Supervised Learning - Scenario-Based Questions

## Question 1: Explain the bias-variance tradeoff and its significance in supervised learning.

### Definition
The bias-variance tradeoff describes the fundamental tension between two sources of error that affect a model's ability to generalize to unseen data.

### Core Concepts

| Component | What It Means | Effect |
|-----------|---------------|--------|
| **Bias** | Error from oversimplified assumptions | Model misses relevant patterns (underfitting) |
| **Variance** | Error from sensitivity to training data fluctuations | Model learns noise as patterns (overfitting) |
| **Total Error** | Bias² + Variance + Irreducible Noise | What we're trying to minimize |

### Mathematical Formulation

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

Where:
- **Bias** = $E[\hat{f}(x)] - f(x)$ (difference between average prediction and true value)
- **Variance** = $E[(\hat{f}(x) - E[\hat{f}(x)])^2]$ (how predictions vary across different training sets)

### The Tradeoff

| Model Complexity | Bias | Variance | Result |
|-----------------|------|----------|--------|
| Too Simple (e.g., Linear) | High | Low | Underfitting - misses patterns |
| Too Complex (e.g., Deep Tree) | Low | High | Overfitting - learns noise |
| Optimal | Balanced | Balanced | Best generalization |

### Practical Significance

1. **Model Selection**: Guides choice between simple vs complex models
2. **Hyperparameter Tuning**: Regularization strength controls this tradeoff
3. **Training Strategy**: More data reduces variance; better features reduce bias

### How to Manage

- **High Bias?** → Add features, use complex model, reduce regularization
- **High Variance?** → Get more data, simplify model, add regularization, use ensemble methods

---

## Question 2: Discuss the significance of backpropagation in neural networks.

### Definition
Backpropagation is the algorithm used to compute gradients of the loss function with respect to all weights in a neural network, enabling gradient descent optimization.

### Why It Matters

1. **Enables Training**: Without backprop, we couldn't train multi-layer networks efficiently
2. **Computational Efficiency**: Uses chain rule to compute all gradients in one backward pass
3. **Foundation of Deep Learning**: Made modern neural networks possible

### Core Mechanism

**Forward Pass:**
```
Input → Hidden Layers → Output → Loss
```

**Backward Pass:**
```
Loss → Gradients flow backward → Update all weights
```

### Mathematical Essence (Chain Rule)

For a weight $w$ in layer $l$:

$$\frac{\partial L}{\partial w^{(l)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial a^{(L-1)}} \cdots \frac{\partial a^{(l+1)}}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial w^{(l)}}$$

### Key Steps

1. **Forward Pass**: Compute activations layer by layer, store intermediate values
2. **Compute Output Loss**: Calculate loss at final layer
3. **Backward Pass**: Propagate error gradients from output to input layer
4. **Weight Update**: $w_{new} = w_{old} - \eta \cdot \nabla_w L$

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Vanishing Gradients | Sigmoid/tanh squash gradients | Use ReLU, skip connections |
| Exploding Gradients | Gradients multiply and grow | Gradient clipping, batch norm |

---

## Question 3: Explain the importance of hyperparameter tuning.

### Definition
Hyperparameter tuning is the process of finding optimal values for model parameters that are set before training (not learned from data).

### Why It's Critical

1. **Performance**: Right hyperparameters can dramatically improve accuracy
2. **Generalization**: Prevents overfitting/underfitting
3. **Efficiency**: Wrong settings waste compute time

### Types of Hyperparameters

| Model Type | Common Hyperparameters |
|------------|----------------------|
| Neural Networks | Learning rate, batch size, layers, neurons, dropout |
| Decision Trees | max_depth, min_samples_split, min_samples_leaf |
| SVM | C (regularization), kernel type, gamma |
| Random Forest | n_estimators, max_features, max_depth |

### Tuning Methods

| Method | How It Works | Pros/Cons |
|--------|--------------|-----------|
| **Grid Search** | Try all combinations | Thorough but expensive |
| **Random Search** | Sample random combinations | More efficient, good coverage |
| **Bayesian Optimization** | Use past results to guide search | Smart, efficient, complex setup |

### Python Example

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# Random Search (more efficient)
from scipy.stats import randint
param_dist = {'max_depth': randint(3, 10), 'n_estimators': randint(50, 300)}
random_search = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=20, cv=5)
random_search.fit(X_train, y_train)
```

### Best Practices

1. Always use cross-validation during tuning
2. Start with coarse search, then refine
3. Focus on most impactful hyperparameters first
4. Keep a held-out test set for final evaluation

---

## Question 4: Discuss the role of learning rate in neural network convergence.

### Definition
The learning rate (η) controls the step size when updating weights during gradient descent. It determines how much to adjust weights based on the computed gradient.

### Weight Update Rule

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$

### Impact on Training

| Learning Rate | Behavior | Outcome |
|--------------|----------|---------|
| **Too High** | Large steps, overshoots minimum | Divergence, loss explodes |
| **Too Low** | Tiny steps, slow progress | Very slow convergence, may get stuck |
| **Optimal** | Balanced steps | Fast, stable convergence |

### Visual Intuition

```
Too High:    ↗↘↗↘ (oscillating, never converges)
Too Low:     →→→→→→→→→... (takes forever)
Just Right:  →→→→ ✓ (smooth descent to minimum)
```

### Learning Rate Schedules

| Schedule | Description | Use Case |
|----------|-------------|----------|
| **Constant** | Same rate throughout | Simple, works for small problems |
| **Step Decay** | Reduce by factor every N epochs | Common, easy to implement |
| **Exponential Decay** | $\eta_t = \eta_0 \cdot e^{-kt}$ | Smooth reduction |
| **Cosine Annealing** | Follows cosine curve | Popular in deep learning |
| **Warmup** | Start low, increase, then decay | Stabilizes training initially |

### Adaptive Methods

| Optimizer | Learning Rate Behavior |
|-----------|----------------------|
| **Adam** | Per-parameter adaptive rates |
| **RMSprop** | Adapts based on recent gradient magnitudes |
| **AdaGrad** | Decreases rate for frequent features |

### Practical Tips

1. **Start with 0.001** for Adam, **0.01** for SGD
2. Use learning rate finder: increase rate until loss explodes
3. Reduce rate if validation loss plateaus
4. Use adaptive optimizers (Adam) for most cases

---

## Question 5: How would you approach a stock price prediction problem using supervised learning?

### Problem Framing

**Key Decision**: Predict exact price (regression) OR direction (classification)?

→ **Recommendation**: Classification (up/down) is more stable and actionable

### Approach

#### Step 1: Define Target Variable

```python
# Binary: Will price go up tomorrow?
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
```

#### Step 2: Feature Engineering

| Feature Type | Examples |
|-------------|----------|
| **Technical Indicators** | SMA, EMA, RSI, MACD, Bollinger Bands |
| **Lagged Returns** | Returns from past k days |
| **Volatility** | Rolling standard deviation |
| **Volume Features** | Volume changes, volume moving averages |
| **Market Data** | Index performance, VIX |
| **Sentiment** | News/social media sentiment scores |

#### Step 3: Model Selection

| Model | Why |
|-------|-----|
| **XGBoost/LightGBM** | Best for tabular data, handles noise well |
| **LSTM** | Captures temporal patterns in sequences |
| **Ensemble** | Combine multiple models for robustness |

#### Step 4: Validation Strategy

**Critical**: Use time-series cross-validation (walk-forward)

```python
# Never use random splits for time series!
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # Train on past, test on future
    pass
```

### Critical Caveats

1. **Non-Stationarity**: Price data changes over time → work with returns, not prices
2. **Efficient Market Hypothesis**: Markets incorporate information quickly
3. **Overfitting Risk**: Very high due to noise → rigorous validation essential
4. **Lookahead Bias**: Never use future information in features

---

## Question 6: Discuss the application of supervised learning in credit scoring.

### Problem Definition

- **Task**: Binary classification - will borrower default?
- **Target**: 1 = default, 0 = paid back
- **Challenge**: Highly imbalanced (defaults are rare)

### Pipeline

#### 1. Feature Categories

| Category | Features |
|----------|----------|
| **Application Data** | Income, employment, loan amount, purpose |
| **Credit History** | Payment history, credit length, open accounts |
| **Debt Metrics** | Debt-to-income ratio, credit utilization |

#### 2. Model Selection

| Model | Pros | Cons |
|-------|------|------|
| **Logistic Regression** | Interpretable, regulatory compliant | Lower accuracy |
| **XGBoost** | High accuracy | Black-box |
| **Hybrid** | Best of both worlds | Complex setup |

**Industry Practice**: Use XGBoost for predictions + SHAP for explanations

#### 3. Training Considerations

```python
# Handle imbalance with class weights
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')
# OR for XGBoost
model = XGBClassifier(scale_pos_weight=ratio_negative/ratio_positive)
```

#### 4. Evaluation Metrics

- **NOT Accuracy** (misleading for imbalanced data)
- **AUC-ROC**: Overall discriminative power
- **Precision-Recall**: Focus on minority class performance
- **Cost-sensitive metrics**: Account for business cost of FP vs FN

#### 5. Fairness Requirements

- Must not discriminate on protected attributes (race, gender)
- Audit model for disparate impact
- Ensure explainability for regulatory compliance

---

## Question 7: Discuss how decision trees are pruned.

### Why Prune?

An unpruned tree overfits by memorizing training noise. Pruning removes unnecessary branches to improve generalization.

### Two Types of Pruning

#### 1. Pre-Pruning (Early Stopping)

**How**: Stop tree growth before full depth

| Parameter | Effect |
|-----------|--------|
| `max_depth` | Limits tree depth |
| `min_samples_split` | Minimum samples needed to split a node |
| `min_samples_leaf` | Minimum samples in leaf nodes |
| `min_impurity_decrease` | Minimum impurity reduction for split |

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)
```

**Pros**: Computationally efficient
**Cons**: May stop too early (greedy)

#### 2. Post-Pruning (Cost-Complexity Pruning)

**How**: Grow full tree, then prune back

**Process**:
1. Build complete tree
2. Calculate cost-complexity: $R_\alpha(T) = R(T) + \alpha |T|$
   - $R(T)$ = misclassification cost
   - $|T|$ = number of leaves
   - $\alpha$ = complexity penalty
3. Prune subtrees where removing improves cross-validation score

```python
# Find optimal alpha
path = tree.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

# Cross-validate to find best alpha
best_tree = DecisionTreeClassifier(ccp_alpha=best_alpha)
```

**Pros**: Considers full tree structure, often better results
**Cons**: More computationally expensive

### Practical Recommendation

Use pre-pruning via hyperparameter tuning with cross-validation - simpler and usually sufficient.

---

## Question 8: How would you handle textual data in a supervised learning problem?

### Step 1: Text Preprocessing

```python
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def preprocess(text):
    text = text.lower()                           # Lowercase
    text = re.sub(r'[^\w\s]', '', text)          # Remove punctuation
    tokens = text.split()                         # Tokenize
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)
```

### Step 2: Vectorization Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| **CountVectorizer** | Word counts | Simple baseline |
| **TF-IDF** | Weighted by importance | Strong baseline for most tasks |
| **Word2Vec/GloVe** | Dense word embeddings | Need semantic similarity |
| **BERT** | Contextual embeddings | State-of-the-art performance |

### Approach Comparison

#### Traditional (Fast, Simple)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(texts)
model = LogisticRegression()
model.fit(X_tfidf, y)
```

#### Deep Learning (Best Performance)

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# Fine-tune on your labeled data
```

### My Strategy

1. **Always start with TF-IDF + Logistic Regression** as baseline
2. If not sufficient, move to **fine-tuned BERT** for state-of-the-art results
3. BERT understands context ("not good" vs "good") while TF-IDF doesn't

---

## Question 9: Discuss the role of transfer learning in supervised models.

### Definition
Transfer learning reuses a model trained on a large general dataset as a starting point for a specific task with limited data.

### Why It Matters

| Benefit | Explanation |
|---------|-------------|
| **Overcome Data Scarcity** | Build accurate models with limited labeled data |
| **Faster Training** | Pre-trained weights converge quickly |
| **Better Performance** | Leverages knowledge from massive datasets |
| **Regularization** | Pre-trained features prevent overfitting |

### Two Main Approaches

#### 1. Feature Extraction
- Freeze pre-trained layers
- Only train new classification head
- Use when: very limited data

#### 2. Fine-Tuning
- Unfreeze some/all pre-trained layers
- Train end-to-end with low learning rate
- Use when: more data, need task-specific adaptation

### Examples

| Domain | Pre-trained Model | Trained On | Fine-tune For |
|--------|------------------|------------|---------------|
| **Vision** | ResNet, EfficientNet | ImageNet (1.2M images) | Medical imaging, defect detection |
| **NLP** | BERT, GPT | Web text corpus | Sentiment, Q&A, classification |

### Code Example (Vision)

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained model without top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

### Key Point
Transfer learning is now the **default approach** in deep learning, not an optimization.

---

## Question 10: How would you design a supervised learning model for predicting customer churn?

### Problem Setup

- **Task**: Binary classification - will customer churn?
- **Target**: 1 = churned, 0 = retained
- **Challenge**: Imbalanced (churners are minority)

### Design Pipeline

#### Step 1: Define Churn

Be specific: "Customer who cancels subscription within 30 days"

#### Step 2: Feature Engineering

| Category | Features |
|----------|----------|
| **Demographics** | Age, location, tenure |
| **Usage Behavior** | Login frequency, features used, time spent |
| **Engagement Trends** | Usage this week vs last month (decline?) |
| **Support** | Tickets raised, satisfaction scores |
| **Billing** | Plan type, payment failures |

**Key**: Create time-windowed trend features
```python
df['usage_decline'] = df['usage_last_7d'] / df['usage_last_30d']
df['days_since_last_login'] = (today - df['last_login']).days
```

#### Step 3: Model Selection

| Model | Why |
|-------|-----|
| **Logistic Regression** | Interpretable - shows churn drivers |
| **XGBoost/LightGBM** | Best accuracy for tabular data |

#### Step 4: Handle Imbalance

```python
from imblearn.over_sampling import SMOTE
# OR
model = XGBClassifier(scale_pos_weight=10)  # If 10:1 ratio
```

#### Step 5: Evaluation

- **NOT Accuracy**
- **Precision**: Of flagged customers, how many actually churn?
- **Recall**: Of actual churners, how many did we catch?
- **AUPRC**: Best single metric for imbalanced problems
- **Lift Chart**: "Top 10% flagged captures 60% of churners"

#### Step 6: Deployment

- Run weekly on all customers
- Output: churn probability score
- Action: Target high-risk customers with retention campaigns

---

## Question 11: Discuss your strategy for developing a sentiment analysis model with supervised learning.

### Two-Phase Strategy

#### Phase 1: Strong Baseline (TF-IDF + Classical ML)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

**Why start here?**
- Fast to implement
- Often surprisingly effective
- Provides benchmark to beat

#### Phase 2: State-of-the-Art (Fine-tuned BERT)

```python
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer, 
    TrainingArguments
)

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=3  # positive, negative, neutral
)

# Tokenize data
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

# Fine-tune with low learning rate
training_args = TrainingArguments(
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16
)
```

### Why BERT Is Better

| Aspect | TF-IDF | BERT |
|--------|--------|------|
| "not good" vs "good" | Same features | Different meanings |
| Word order | Ignored | Understood |
| Context | None | Full sentence context |

### Evaluation Metrics

- **Accuracy** (if balanced classes)
- **F1-Score** (per class and macro)
- **Confusion Matrix** (identify which sentiments are confused)

---

## Question 12: Propose a supervised learning approach for fraud detection in transactions.

### Challenges

1. **Extreme Imbalance**: Fraud < 0.1% of transactions
2. **High Stakes**: FN (missed fraud) is very costly
3. **Evolving Patterns**: Fraudsters adapt quickly

### Approach

#### Step 1: Feature Engineering

| Feature Type | Examples |
|-------------|----------|
| **Transaction** | Amount, time, merchant category |
| **User Behavior** | Transaction count in 24h, avg amount |
| **Anomaly Signals** | Amount deviation from user average, new merchant, unusual location |

```python
# Example features
df['txn_count_24h'] = df.groupby('user_id')['amount'].transform(
    lambda x: x.rolling('24h').count()
)
df['amount_vs_avg'] = df['amount'] / df['user_avg_amount']
df['is_new_merchant'] = ~df['merchant_id'].isin(user_past_merchants)
```

#### Step 2: Model Selection

**Best Choice**: XGBoost or LightGBM
- Handles tabular data well
- Built-in class imbalance handling
- Robust to outliers

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    is_unbalance=True,  # Handle imbalance
    n_estimators=500,
    learning_rate=0.05
)
```

#### Step 3: Validation Strategy

**Critical**: Time-based split only!

```python
# Train on past, test on future
train = df[df['date'] < cutoff_date]
test = df[df['date'] >= cutoff_date]
```

#### Step 4: Evaluation

| Metric | Why |
|--------|-----|
| **AUPRC** | Best for imbalanced; focuses on fraud class |
| **Precision @ Recall** | "What precision if we catch 80% of fraud?" |
| **Cost-based** | Weight FN more heavily than FP |

```python
from sklearn.metrics import precision_recall_curve, auc

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
auprc = auc(recall, precision)
```

#### Step 5: Threshold Selection

Choose threshold based on business tradeoff:
- Lower threshold → More fraud caught, more false alarms
- Higher threshold → Fewer false alarms, some fraud missed

#### Step 6: Deployment

1. **Real-time scoring**: Score each transaction as it occurs
2. **Action**: Block, flag for review, or request 2FA based on score
3. **Monitoring**: Retrain regularly as fraud patterns evolve

### Summary Pipeline

```
Transaction → Feature Generation → Model Score → Threshold → Action
                                                    ↓
                                         Monitor & Retrain
```
