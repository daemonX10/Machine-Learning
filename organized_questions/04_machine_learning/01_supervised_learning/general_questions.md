# Supervised Learning Interview Questions - General Questions

---

## Question 1: Distinguish between Supervised and Unsupervised Learning

**Definition:**  
**Supervised learning** learns from labeled data (input-output pairs) to predict outcomes. **Unsupervised learning** discovers hidden patterns in unlabeled data without predefined targets.

**Key Differences:**

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|---------------------|----------------------|
| **Input Data** | Labeled (X, y) | Unlabeled (X only) |
| **Goal** | Predict known target | Discover patterns |
| **Feedback** | Direct (compare to true label) | None |
| **Tasks** | Classification, Regression | Clustering, Dimensionality Reduction |
| **Examples** | Spam detection, Price prediction | Customer segmentation, PCA |

**Intuition:**
- Supervised: Student learning with flashcards (question + answer)
- Unsupervised: Student sorting photos into groups without instructions

**Python Code Example:**
```python
# Supervised: Classification
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Uses labels
y_pred = model.predict(X_test)

# Unsupervised: Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)  # No labels needed
clusters = kmeans.labels_
```

---

## Question 2: Methods to Prevent Overfitting

**Definition:**  
Overfitting occurs when a model learns training data too well, including noise, leading to poor generalization. Prevention involves reducing model complexity, adding regularization, or increasing effective data size.

**Key Methods:**

| Method | How It Helps |
|--------|--------------|
| **Get More Data** | Clearer signal, harder to memorize |
| **Data Augmentation** | Artificially increase training set |
| **Simplify Model** | Reduce layers/neurons, limit tree depth |
| **Regularization (L1/L2)** | Penalize large weights |
| **Dropout** | Randomly disable neurons during training |
| **Early Stopping** | Stop when validation loss increases |
| **Cross-Validation** | Better estimate of generalization |
| **Ensemble Methods** | Average out individual model errors |

**Python Code Example:**
```python
from sklearn.linear_model import Ridge
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# L2 Regularization
ridge = Ridge(alpha=1.0)

# Dropout in Neural Network
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # 50% dropout

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X, y, validation_split=0.2, callbacks=[early_stop])
```

**Interview Tip:** Start with simplest methods (regularization, early stopping) before complex ones.

---

## Question 3: Handling Categorical Variables

**Definition:**  
Categorical variables must be converted to numerical format for most ML models. The encoding method depends on whether the variable is **nominal** (no order) or **ordinal** (has order).

**Encoding Methods:**

| Variable Type | Method | Description |
|---------------|--------|-------------|
| **Nominal** (no order) | One-Hot Encoding | Create binary column per category |
| **Ordinal** (has order) | Label Encoding | Assign integers based on rank |
| **High Cardinality** | Target Encoding | Replace with mean of target |
| **High Cardinality** | Feature Hashing | Hash to fixed-size vector |

**Python Code Example:**
```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

# One-Hot Encoding (nominal)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = ohe.fit_transform(df[['color']])  # color: red, blue, green

# Label Encoding (ordinal)
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])  # HS=0, BS=1, MS=2

# Pandas get_dummies (quick one-hot)
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)
```

**Interview Tip:** Never use Label Encoding for nominal variables - model assumes false ordering.

---

## Question 4: Preventing Overfitting in Neural Networks

**Definition:**  
Neural networks are especially prone to overfitting due to their high capacity. Key techniques include Dropout, weight regularization, data augmentation, early stopping, and batch normalization.

**Techniques:**

| Technique | How It Works |
|-----------|--------------|
| **Dropout** | Randomly set neurons to 0 during training |
| **L2 Regularization (Weight Decay)** | Add penalty for large weights |
| **Data Augmentation** | Create modified training samples |
| **Early Stopping** | Stop when validation loss plateaus |
| **Batch Normalization** | Normalize layer inputs (slight regularization) |
| **Reduce Architecture** | Fewer layers/neurons |
| **Transfer Learning** | Start from pretrained weights |

**Python Code Example (Keras):**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # L2
    BatchNormalization(),
    Dropout(0.5),  # 50% dropout
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X, y, validation_split=0.2, callbacks=[early_stop], epochs=100)
```

---

## Question 5: Accuracy and Why It's Not Always the Best Metric

**Definition:**  
Accuracy = (Correct Predictions) / (Total Predictions). It's misleading for **imbalanced datasets** because a model predicting only the majority class achieves high accuracy while being useless.

**Formula:**
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**The Problem:**

| Dataset | Class Distribution | Naive Model Strategy | Accuracy |
|---------|-------------------|---------------------|----------|
| Fraud Detection | 99% legit, 1% fraud | Always predict "legit" | 99% |
| Disease Diagnosis | 95% healthy, 5% sick | Always predict "healthy" | 95% |

**Better Metrics for Imbalanced Data:**
- **Precision:** Of predicted positive, how many correct?
- **Recall:** Of actual positive, how many found?
- **F1 Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Discrimination ability across thresholds
- **AUPRC:** Precision-Recall curve (best for severe imbalance)

**Interview Tip:** Always ask about class distribution before choosing metrics.

---

## Question 6: Compare RMSE and MAE

**Definition:**  
Both are regression error metrics. **MAE** (Mean Absolute Error) treats all errors equally. **RMSE** (Root Mean Squared Error) penalizes large errors more heavily due to squaring.

**Formulas:**
$$MAE = \frac{1}{N}\sum|y_i - \hat{y}_i|$$
$$RMSE = \sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$$

**Comparison:**

| Aspect | MAE | RMSE |
|--------|-----|------|
| Error Penalty | Linear | Quadratic |
| Outlier Sensitivity | Robust | Sensitive |
| Interpretation | "Average error magnitude" | "Std dev of errors" |
| Differentiability | Not at zero | Everywhere |

**When to Use:**
- **RMSE:** Large errors are especially bad (critical systems)
- **MAE:** Outliers present, want robust metric

**Python Code Example:**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
```

---

## Question 7: When to Use MAPE

**Definition:**  
MAPE (Mean Absolute Percentage Error) measures average percentage error. Use it when you need **relative error** that's easy to explain to stakeholders or when comparing forecasts across different scales.

**Formula:**
$$MAPE = \frac{100\%}{N}\sum\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

**When to Use:**
- Business stakeholders need intuitive interpretation ("5% error")
- Comparing forecasts across different scales (sales of different products)

**When to AVOID:**
- Actual values can be **zero** (division by zero)
- Data has values near zero (inflates percentage)
- **Asymmetric penalty:** Under-predictions penalized less than over-predictions

**Alternative:** sMAPE (Symmetric MAPE) uses average of actual and predicted in denominator.

**Python Code Example:**
```python
import numpy as np

def mape(y_true, y_pred):
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

print(f"MAPE: {mape(y_true, y_pred):.2f}%")
```

---

## Question 8: Supervised Learning in NLP

**Definition:**  
Supervised learning powers most NLP tasks by training models to map text inputs to labeled outputs. Modern NLP uses transfer learning: pretrain on unlabeled text, then fine-tune on task-specific labeled data.

**Key Supervised NLP Tasks:**

| Task | Type | Example |
|------|------|---------|
| **Text Classification** | Classification | Sentiment analysis, spam detection |
| **Named Entity Recognition** | Sequence labeling | Extract persons, organizations |
| **Machine Translation** | Seq2Seq | English to French |
| **Question Answering** | Extraction | Find answer in context |
| **POS Tagging** | Sequence labeling | Noun, verb, adjective tags |

**Modern Approach (Transfer Learning):**
1. **Pretrain:** Large model on massive text (BERT, GPT) - learns language structure
2. **Fine-tune:** Adapt to specific task with small labeled dataset

**Python Code Example:**
```python
from transformers import pipeline

# Sentiment Analysis (pretrained + fine-tuned)
classifier = pipeline("sentiment-analysis")
result = classifier("This movie was amazing!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.99}]

# Named Entity Recognition
ner = pipeline("ner")
entities = ner("Apple was founded by Steve Jobs in California")
```

---

## Question 9: Handling Imbalanced Datasets

**Definition:**  
Imbalanced datasets have skewed class distributions. Solutions include appropriate metrics, resampling techniques, and algorithm-level modifications like class weights.

**Three-Pronged Strategy:**

**1. Choose Right Metrics:**
- F1-Score, Precision-Recall AUC (not accuracy)

**2. Data-Level Techniques:**

| Technique | Method |
|-----------|--------|
| **Oversampling** | Duplicate/SMOTE minority class |
| **Undersampling** | Remove majority class samples |

**3. Algorithm-Level Techniques:**

| Technique | Method |
|-----------|--------|
| **Class Weights** | Penalize minority misclassification more |
| **Threshold Adjustment** | Lower decision threshold |

**Python Code Example:**
```python
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Class weights
rf = RandomForestClassifier(class_weight='balanced')  # Auto-adjust
rf.fit(X_train, y_train)

# Or manual weights
rf = RandomForestClassifier(class_weight={0: 1, 1: 10})  # 10x penalty for class 1
```

**Interview Tip:** Never resample test set - it must reflect real distribution.

---

## Question 10: When is Dimensionality Reduction Useful?

**Definition:**  
Dimensionality reduction reduces the number of features to combat curse of dimensionality, reduce overfitting, speed up training, handle multicollinearity, and enable visualization.

**When to Use:**
- Too many features relative to samples
- Features are highly correlated
- Need to visualize high-dimensional data
- Training is too slow

**Methods:**

| Category | Technique | Description |
|----------|-----------|-------------|
| **Feature Selection** | Filter (correlation) | Rank features statistically |
| **Feature Selection** | Wrapper (RFE) | Search feature subsets |
| **Feature Selection** | Embedded (Lasso) | Built into model |
| **Feature Extraction** | PCA | Linear, maximize variance |
| **Feature Extraction** | t-SNE, UMAP | Non-linear, for visualization |
| **Feature Extraction** | Autoencoder | Neural network compression |

**Python Code Example:**
```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# PCA - feature extraction
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X)

# SelectKBest - feature selection
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

---

## Question 11: Framing RL as Supervised Learning

**Definition:**  
Reinforcement Learning can be framed as supervised learning through **Imitation Learning** (predict expert actions from states) or **Value Function Fitting** (predict Q-values using regression).

**Method 1: Imitation Learning (Behavioral Cloning)**
- Input (X): States observed by expert
- Label (y): Actions expert took
- Problem: Classification (discrete actions) or Regression (continuous actions)

**Method 2: Q-Learning as Regression**
- Input (X): State-action pairs
- Target (y): Calculated target Q-value = r + gamma * max Q(s', a')
- Problem: Regression

**Python Code Example (Behavioral Cloning):**
```python
# Expert demonstrations: (state, action) pairs
states = expert_states      # observations
actions = expert_actions    # what expert did

# Train supervised classifier to mimic expert
from sklearn.ensemble import RandomForestClassifier
policy = RandomForestClassifier()
policy.fit(states, actions)

# Agent uses learned policy
predicted_action = policy.predict(new_state)
```

**Limitation:** Imitation learning can't exceed expert performance.

---

## Question 12: Role of Attention Mechanisms

**Definition:**  
Attention mechanisms allow neural networks to dynamically focus on relevant parts of input when generating output. They compute weighted combinations of input elements, where weights indicate importance for the current task.

**The Problem Before Attention:**
- Entire input compressed to fixed-size vector (bottleneck)
- Lost information for long sequences

**How Attention Works:**
1. **Query:** Current decoder state asks "what's relevant?"
2. **Keys:** Each input element provides a key
3. **Scores:** Compute similarity (Query x Keys)
4. **Weights:** Softmax to get attention distribution
5. **Context:** Weighted sum of Values

**Mathematical Formulation:**
$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Benefits:**
- Handles long sequences
- Interpretable (visualize attention weights)
- Foundation of Transformers (BERT, GPT)

**Self-Attention:** Query, Key, Value all come from same sequence - each element attends to all others.

**Interview Tip:** Transformers replaced RNNs/LSTMs by using only attention (no recurrence).

---
