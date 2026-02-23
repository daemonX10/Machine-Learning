# Graduate Admission Prediction using ANN (Regression)

## Problem Statement

- **Type**: Regression
- **Dataset**: Graduate admission prediction (university applications)
- **Goal**: Predict the chance of admission (continuous value between 0 and 1)
- **Dataset Shape**: 500 rows × 8 columns

---

## Dataset Overview

| Column | Description | Type | Range |
|---|---|---|---|
| Serial No. | Row index | Drop | — |
| GRE Score | Graduate Record Examination | Integer | 0–340 |
| TOEFL Score | Test of English as a Foreign Language | Integer | 0–120 |
| University Rating | University prestige rating | Integer | 1–5 |
| SOP | Statement of Purpose strength | Float | 1.0–5.0 |
| LOR | Letter of Recommendation strength | Float | 1.0–5.0 |
| CGPA | Undergraduate CGPA | Float | continuous |
| Research | Research experience (0/1) | Integer | 0 or 1 |
| **Chance of Admit** | **Target (probability)** | **Float** | **0–1** |

- No missing values, no duplicate rows
- This is a **regression** problem (output is a continuous number, not a class)

---

## Data Preprocessing

### 1. Drop Serial Number

```python
df.drop(columns=['Serial No.'], inplace=True)
```

### 2. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]   # All columns except last
y = df.iloc[:, -1]     # Last column (Chance of Admit)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

### 3. MinMaxScaler (not StandardScaler)

> **Why MinMaxScaler?** Use MinMaxScaler when you know the upper and lower bounds of your features (GRE: 0–340, TOEFL: 0–120, etc.). Use StandardScaler when the distribution is roughly Gaussian.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Model Building

### Initial Architecture

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(7, activation='relu', input_dim=7))    # Hidden layer
model.add(Dense(1, activation='linear'))                 # Output layer
```

**Architecture**: 7 inputs → Dense(7, relu) → Dense(1, **linear**)

> **Critical Rule**: For regression problems, the output layer activation must be **linear** (not sigmoid/softmax).

**Trainable Parameters**:
- Hidden: $(7 \times 7) + 7 = 56$
- Output: $(7 \times 1) + 1 = 8$
- **Total: 64**

### Compile & Train

```python
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train_scaled, y_train, epochs=10, validation_split=0.2)
```

| Component | Choice |
|---|---|
| Loss Function | **Mean Squared Error** (MSE) — standard for regression |
| Optimizer | Adam |
| Metrics | Not specified initially |

### Evaluation

```python
from sklearn.metrics import r2_score
y_pred = model.predict(X_test_scaled)
print(r2_score(y_test, y_pred))
```

Initial $R^2$ score was poor with only 10 epochs and simple architecture.

---

## Model Improvement

### Improved Architecture

```python
model = Sequential()
model.add(Dense(7, activation='relu', input_dim=7))    # Hidden layer 1
model.add(Dense(7, activation='relu'))                   # Hidden layer 2
model.add(Dense(1, activation='linear'))                 # Output layer
```

**Changes Made**:

| Change | Before | After |
|---|---|---|
| Hidden layers | 1 | **2** |
| Epochs | 10 | **100** |

- **Improved $R^2$ Score**: ~0.76

---

## Training Monitoring

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
```

- Loss decreases rapidly initially, then gradually stabilizes
- No significant overfitting observed — training could continue for better results

---

## Key Differences: Regression vs Classification in ANN

| Aspect | Classification | Regression |
|---|---|---|
| Output activation | Sigmoid / Softmax | **Linear** |
| Loss function | Crossentropy | **Mean Squared Error** |
| Output nodes | 1 (binary) or N (multi-class) | **1** (single value) |
| Evaluation metric | Accuracy, F1 | **$R^2$, MSE, MAE** |
| Output interpretation | Class label / probability | Continuous value |

---

## Key Takeaways

- Regression ANN: output layer uses **linear activation** with **1 node**
- Loss function for regression: **MSE** (Mean Squared Error)
- **MinMaxScaler** preferred when feature bounds are known
- More hidden layers + more epochs → better convergence
- Always plot training vs validation loss to detect overfitting
