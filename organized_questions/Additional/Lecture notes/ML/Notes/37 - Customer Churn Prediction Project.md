# Customer Churn Prediction using ANN (Keras & TensorFlow)

## Problem Statement

- **Type**: Binary Classification
- **Dataset**: Bank customer churn prediction (credit card customers)
- **Goal**: Predict whether a customer will leave the bank ($1$) or stay ($0$)
- **Dataset Shape**: 10,000 rows × 14 columns

---

## Dataset Overview

| Column | Description | Type |
|---|---|---|
| RowNumber | Row index | Drop |
| CustomerId | Unique ID | Drop |
| Surname | Customer name | Drop |
| CreditScore | Credit score | Numerical |
| Geography | Country (France, Germany, Spain) | Categorical |
| Gender | Male / Female | Categorical |
| Age | Customer age | Numerical |
| Tenure | Years with the bank | Numerical |
| Balance | Account balance | Numerical (float) |
| NumOfProducts | Products used (credit card, debit, etc.) | Numerical |
| HasCrCard | Has credit card (0/1) | Numerical |
| IsActiveMember | Active member (0/1) | Numerical |
| EstimatedSalary | Estimated salary | Numerical (float) |
| **Exited** | **Target: 1 = churned, 0 = stayed** | **Binary** |

- **Class Distribution**: ~8000 stayed, ~2000 churned → **imbalanced dataset**
- No missing values, no duplicate rows

---

## Data Preprocessing

### 1. Drop Irrelevant Columns

```python
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
```

### 2. One-Hot Encoding (Categorical Features)

```python
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
```

- `Geography` (3 categories) → 2 columns: `Geography_Germany`, `Geography_Spain` (France = 0,0)
- `Gender` (2 categories) → 1 column: `Gender_Male` (Female = 0)

### 3. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Exited'])
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### 4. Feature Scaling (StandardScaler)

> **Key Insight**: Always scale inputs when training neural networks — large value differences cause slow weight convergence.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Model Building (Keras Sequential API)

### Initial Architecture

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, activation='sigmoid', input_dim=11))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))                  # Output layer
```

**Architecture**: 11 inputs → Dense(3, sigmoid) → Dense(1, sigmoid)

**Trainable Parameters**:
- Hidden: $(11 \times 3) + 3 = 36$
- Output: $(3 \times 1) + 1 = 4$
- **Total: 40**

### Compile & Train

```python
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X_train_scaled, y_train, epochs=10)
```

| Component | Choice |
|---|---|
| Loss Function | Binary Crossentropy (log loss) |
| Optimizer | Adam |

### Prediction & Thresholding

```python
y_prob = model.predict(X_test_scaled)  # Outputs probabilities (0 to 1)
y_pred = (y_prob > 0.5).astype(int)    # Apply threshold

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))   # ~81.45%
```

> Sigmoid outputs a probability → apply threshold (default 0.5) to get class label. Optimal threshold found via ROC-AUC curve.

---

## Model Improvement

### Improved Architecture

```python
model = Sequential()
model.add(Dense(11, activation='relu', input_dim=11))   # Hidden layer 1
model.add(Dense(11, activation='relu'))                   # Hidden layer 2
model.add(Dense(1, activation='sigmoid'))                 # Output layer
```

**Changes Made**:

| Change | Before | After |
|---|---|---|
| Activation (hidden) | Sigmoid | **ReLU** |
| Nodes per hidden layer | 3 | **11** |
| Number of hidden layers | 1 | **2** |
| Epochs | 10 | **100** |

**Trainable Parameters (improved)**:
- Layer 1: $(11 \times 11) + 11 = 132$
- Layer 2: $(11 \times 11) + 11 = 132$
- Output: $(11 \times 1) + 1 = 12$
- **Total: 276**

### Training with Validation

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)
```

- **Improved Accuracy**: ~85.75%

---

## Monitoring Training with Plots

```python
import matplotlib.pyplot as plt

# Loss Plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()

# Accuracy Plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
```

### Interpreting the Plots

- Loss should **decrease** over epochs for both training and validation
- Accuracy should **increase** over epochs
- **Overfitting indicator**: Training accuracy keeps increasing but validation accuracy stagnates or drops — gap between curves shows overfitting level

---

## Key Takeaways

| Concept | Detail |
|---|---|
| Binary classification output | 1 node with sigmoid activation |
| Loss function | Binary crossentropy |
| Always scale inputs | StandardScaler before training |
| ReLU for hidden layers | Generally better than sigmoid |
| Validation split | Monitor overfitting during training |
| Overfitting solutions | Regularization, dropout, early stopping |
