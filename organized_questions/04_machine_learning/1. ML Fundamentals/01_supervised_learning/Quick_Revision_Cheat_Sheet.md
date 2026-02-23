# 🚀 Supervised Learning - Quick Revision Cheat Sheet

## ⚡ 30-Second Review

### 🎯 Core Concepts (Memorize!)
```
Supervised Learning = Learning with labeled data (X, y)
Two Types: Classification (categories) + Regression (numbers)
Goal: Learn function f: X → y that generalizes well
```

### 🔥 Essential Formulas

#### Linear Regression
```
Hypothesis: ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
Cost: MSE = (1/2m) × Σ(ŷᵢ - yᵢ)²
Update: w = w - α × (1/m) × X^T × (ŷ - y)
```

#### Logistic Regression  
```
Sigmoid: σ(z) = 1/(1 + e^(-z))
Hypothesis: ŷ = σ(w^T × X + b)
Cost: Cross-entropy = -(1/m) × Σ[y×log(ŷ) + (1-y)×log(1-ŷ)]
```

#### Decision Trees
```
Gini Impurity: 1 - Σ(pᵢ)²
Info Gain: Gini_parent - Σ(weight_child × Gini_child)
Split: Choose feature + threshold with max info gain
```

### 📊 Evaluation Metrics

#### Classification
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)  # When model says +, how often right?
Recall = TP / (TP + FN)     # How many actual + did we catch?
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

#### Regression
```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|           # Mean Absolute Error
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²          # Mean Squared Error  
RMSE = √MSE                         # Root Mean Squared Error
R² = 1 - (SS_res / SS_tot)          # Coefficient of Determination
```

---

## 🧠 Algorithm Quick Facts

| Algorithm | Type | Pros | Cons | When to Use |
|-----------|------|------|------|-------------|
| **Linear Regression** | Regression | Simple, Fast, Interpretable | Linear assumptions | Linear relationships |
| **Logistic Regression** | Classification | Probabilistic, Fast | Linear boundary | Binary/Multi-class |
| **Decision Trees** | Both | Interpretable, No preprocessing | Overfitting | Rule extraction |
| **Random Forest** | Both | Robust, Feature importance | Less interpretable | General purpose |
| **SVM** | Both | High dimensions | Slow on big data | Text, High-dim |
| **Neural Networks** | Both | Universal approximator | Black box, Data hungry | Complex patterns |

---

## 🔥 Interview Code Templates (Memorize These!)

### Standard ML Pipeline
```python
# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale features (if needed)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Predict & evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Score: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Confusion Matrix Analysis
```python
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

---

## 🎯 Key Problems & Solutions

### Overfitting (High Variance)
**Symptoms**: Perfect training, poor test performance
**Solutions**: 
- More data
- Regularization (L1/L2)
- Cross-validation
- Early stopping
- Feature selection
- Ensemble methods

### Underfitting (High Bias)  
**Symptoms**: Poor training & test performance
**Solutions**:
- More complex model
- More features
- Longer training
- Remove regularization

### Imbalanced Data
**Solutions**:
- Stratified sampling
- SMOTE/oversampling
- Class weights
- Different metrics (F1, AUC)

---

## 💡 Interview Success Tips

### 🔥 What to Say:
1. **"Let me understand the problem first..."**
   - Classification or regression?
   - How much data?
   - What's the business goal?

2. **"I'll start with a simple baseline..."**
   - Majority class for classification
   - Mean prediction for regression

3. **"Let me check the data quality..."**
   - Missing values?
   - Outliers?
   - Feature distributions?

4. **"I'll use appropriate evaluation..."**
   - Accuracy for balanced classification
   - F1/AUC for imbalanced data
   - RMSE/MAE for regression

### ⚠️ What NOT to Do:
- ❌ Use test data for hyperparameter tuning
- ❌ Forget to scale features for distance-based algorithms
- ❌ Use accuracy for imbalanced datasets
- ❌ Ignore data leakage in time series
- ❌ Skip exploratory data analysis

---

## 🚀 Quick Mental Checklist

**Before Any ML Problem:**
- [ ] Understand problem type (classification/regression)
- [ ] Check data quality and distribution
- [ ] Choose appropriate baseline
- [ ] Select relevant features
- [ ] Pick suitable algorithm
- [ ] Use proper evaluation metrics
- [ ] Validate with cross-validation
- [ ] Consider business constraints

**Common Interview Questions:**
- [ ] "How do you handle overfitting?"
- [ ] "Explain bias-variance tradeoff"
- [ ] "When would you use logistic regression vs SVM?"
- [ ] "How do you evaluate a model?"
- [ ] "What's the difference between L1 and L2 regularization?"

---

## 🎯 Last-Minute Cramming

### Algorithms in 1 Line Each:
- **Linear Regression**: Fits best line through data points
- **Logistic Regression**: Uses sigmoid to output probabilities  
- **Decision Trees**: Asks yes/no questions to split data
- **Random Forest**: Combines many decision trees
- **SVM**: Finds optimal boundary between classes
- **Neural Networks**: Stacks layers of weighted connections

### Key Hyperparameters:
- **Learning Rate**: How big steps to take (0.01, 0.1, 0.001)
- **Regularization**: Prevents overfitting (α = 0.01, 0.1, 1.0)
- **Max Depth**: Tree complexity (3, 5, 10)
- **C Parameter**: SVM regularization (0.1, 1.0, 10)

---

**🎉 You got this! Practice the code templates and understand the concepts. Good luck with your interview! 🍀**