# Feature Engineering Interview Questions - Scenario-Based Questions

## Question 1: You are using a Random Forest model and need to identify the most important features. How would you use the model's feature importance scores to guide your feature selection?

### Answer

**Approach:**

Random Forest provides built-in feature importance through two methods:
1. **Mean Decrease in Impurity (MDI)** - Gini/entropy importance
2. **Mean Decrease in Accuracy (MDA)** - Permutation importance

**Step-by-Step Implementation:**

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Method 1: Built-in Feature Importance (MDI)
mdi_importance = pd.DataFrame({
    'feature': X.columns,
    'importance_mdi': rf.feature_importances_
}).sort_values('importance_mdi', ascending=False)

print("Top 10 Features (MDI):")
print(mdi_importance.head(10))

# Method 2: Permutation Importance (MDA) - More reliable
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'feature': X.columns,
    'importance_perm': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance_perm', ascending=False)

print("\nTop 10 Features (Permutation):")
print(perm_df.head(10))
```

**Feature Selection Strategy:**

```python
def select_features_by_importance(X, y, threshold_pct=0.90, min_features=5):
    """
    Select features based on cumulative importance.
    
    Args:
        threshold_pct: Keep features until this % of importance is covered
        min_features: Minimum features to keep
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Calculate cumulative importance
    importance_df['cumulative'] = importance_df['importance'].cumsum()
    importance_df['cumulative_pct'] = importance_df['cumulative'] / importance_df['importance'].sum()
    
    # Select features
    mask = importance_df['cumulative_pct'] <= threshold_pct
    selected = importance_df[mask]['feature'].tolist()
    
    # Ensure minimum features
    if len(selected) < min_features:
        selected = importance_df.head(min_features)['feature'].tolist()
    
    return selected, importance_df

selected_features, importance_df = select_features_by_importance(X, y, threshold_pct=0.90)
print(f"Selected {len(selected_features)} features covering 90% importance")
```

**Visualization:**

```python
def plot_feature_importance(importance_df, top_n=15):
    """Visualize feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot
    top_features = importance_df.head(top_n)
    axes[0].barh(top_features['feature'], top_features['importance'])
    axes[0].set_xlabel('Importance')
    axes[0].set_title(f'Top {top_n} Feature Importances')
    axes[0].invert_yaxis()
    
    # Cumulative plot
    axes[1].plot(range(len(importance_df)), importance_df['cumulative_pct'])
    axes[1].axhline(y=0.90, color='r', linestyle='--', label='90% threshold')
    axes[1].set_xlabel('Number of Features')
    axes[1].set_ylabel('Cumulative Importance')
    axes[1].set_title('Cumulative Feature Importance')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
```

**Best Practices:**
1. **Use Permutation Importance** for final selection (less biased)
2. **Cross-validate** the selection process
3. **Compare** MDI and Permutation results
4. **Consider domain knowledge** - important features should make sense
5. **Check for multicollinearity** - correlated features split importance

---

## Question 2: Explain how ICA can be used for feature extraction. Give a practical example.

### Answer

**ICA (Independent Component Analysis):**

ICA separates a multivariate signal into independent, non-Gaussian source signals. Unlike PCA which maximizes variance, ICA maximizes statistical independence.

**Key Differences from PCA:**

| Aspect | PCA | ICA |
|--------|-----|-----|
| **Goal** | Maximize variance | Maximize independence |
| **Assumption** | Orthogonal components | Independent components |
| **Distribution** | Gaussian assumed | Non-Gaussian required |
| **Use Case** | Dimensionality reduction | Signal separation |

**Practical Example: Blind Source Separation**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA

# Create mixed signals (simulating real-world scenario)
np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# Original source signals
s1 = np.sin(2 * time)  # Sinusoidal
s2 = np.sign(np.sin(3 * time))  # Square wave
s3 = (time % 1) - 0.5  # Sawtooth

sources = np.c_[s1, s2, s3]

# Mix signals (unknown mixing matrix in real scenario)
mixing_matrix = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])
mixed_signals = np.dot(sources, mixing_matrix.T)

# Apply ICA
ica = FastICA(n_components=3, random_state=42, max_iter=500)
recovered_signals = ica.fit_transform(mixed_signals)

# Compare with PCA
pca = PCA(n_components=3)
pca_signals = pca.fit_transform(mixed_signals)

# Visualization
fig, axes = plt.subplots(4, 3, figsize=(15, 12))

titles = ['Original Sources', 'Mixed Signals', 'ICA Recovered']
for i, (signals, title) in enumerate([(sources, titles[0]), 
                                       (mixed_signals, titles[1]),
                                       (recovered_signals, titles[2])]):
    for j in range(3):
        axes[j, i].plot(time, signals[:, j])
        axes[j, i].set_title(f'{title} - Component {j+1}')

# PCA comparison
for j in range(3):
    axes[3, j].plot(time, pca_signals[:, j], alpha=0.7)
    axes[3, j].set_title(f'PCA - Component {j+1}')

plt.tight_layout()
plt.show()

print("ICA successfully separated mixed signals into original sources!")
```

**Feature Extraction Use Case: EEG Signal Processing**

```python
from sklearn.decomposition import FastICA

def extract_ica_features(data, n_components=10):
    """
    Extract ICA features from multi-channel signal data.
    
    Args:
        data: Shape (n_samples, n_channels)
        n_components: Number of independent components
        
    Returns:
        Independent components as features
    """
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    ica_features = ica.fit_transform(data)
    
    # Get mixing matrix for interpretation
    mixing_matrix = ica.mixing_
    
    return ica_features, ica, mixing_matrix


# Example with sensor data
np.random.seed(42)
n_samples = 1000
n_channels = 20

# Simulated multi-channel sensor data
sensor_data = np.random.randn(n_samples, n_channels)

# Extract ICA features
ica_features, ica_model, mixing = extract_ica_features(sensor_data, n_components=5)

print(f"Original shape: {sensor_data.shape}")
print(f"ICA features shape: {ica_features.shape}")
print(f"Mixing matrix shape: {mixing.shape}")
```

**When to Use ICA:**
- Separating mixed signals (audio, EEG, financial)
- When sources are statistically independent
- When non-Gaussianity is important
- Cocktail party problem (separating voices)

**Limitations:**
- Cannot determine scale of components
- Order of components is arbitrary
- Requires enough samples
- Assumes linear mixing

---

## Question 3: How would you balance the trade-off between overfitting and underfitting while engineering features?

### Answer

**The Feature Engineering Bias-Variance Trade-off:**

| Problem | Cause | Symptoms |
|---------|-------|----------|
| **Underfitting** | Too few/simple features | High training error, high test error |
| **Overfitting** | Too many/complex features | Low training error, high test error |

**Strategy Framework:**

```
1. Start Simple → Add Complexity → Regularize → Validate
```

**Step-by-Step Approach:**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def evaluate_feature_complexity(X, y, max_degree=5):
    """
    Evaluate model performance across feature complexity levels.
    """
    results = []
    
    for degree in range(1, max_degree + 1):
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])
        
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
        
        pipeline.fit(X, y)
        n_features = pipeline.named_steps['poly'].n_output_features_
        
        results.append({
            'degree': degree,
            'n_features': n_features,
            'cv_rmse_mean': np.sqrt(-scores.mean()),
            'cv_rmse_std': np.sqrt(-scores).std()
        })
    
    return pd.DataFrame(results)
```

**Techniques to Prevent Overfitting:**

```python
# 1. Feature Selection with Regularization
from sklearn.linear_model import LassoCV

def regularized_feature_selection(X, y, alpha_range=None):
    """Use Lasso for automatic feature selection."""
    if alpha_range is None:
        alpha_range = np.logspace(-4, 1, 50)
    
    lasso_cv = LassoCV(alphas=alpha_range, cv=5, random_state=42)
    lasso_cv.fit(X, y)
    
    # Get selected features (non-zero coefficients)
    selected_mask = lasso_cv.coef_ != 0
    selected_features = X.columns[selected_mask].tolist()
    
    print(f"Optimal alpha: {lasso_cv.alpha_:.4f}")
    print(f"Selected {len(selected_features)} of {X.shape[1]} features")
    
    return selected_features, lasso_cv

# 2. Learning Curves for Diagnosis
def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    """Plot learning curve to diagnose bias-variance."""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    test_rmse = np.sqrt(-test_scores.mean(axis=1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_rmse, 'o-', label='Training RMSE')
    plt.plot(train_sizes, test_rmse, 'o-', label='Validation RMSE')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Diagnosis
    gap = test_rmse[-1] - train_rmse[-1]
    if gap > 0.1 * test_rmse[-1]:
        print("Diagnosis: HIGH VARIANCE (Overfitting) - Reduce features or add regularization")
    elif train_rmse[-1] > 0.5:  # Adjust threshold based on problem
        print("Diagnosis: HIGH BIAS (Underfitting) - Add more features")
    else:
        print("Diagnosis: Good balance")
```

**Balanced Feature Engineering Pipeline:**

```python
def balanced_feature_engineering(X, y, target_variance_explained=0.95):
    """
    Balanced approach to feature engineering.
    """
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectFromModel
    
    # Step 1: Generate rich features
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_poly = poly.fit_transform(X)
    print(f"Step 1: Generated {X_poly.shape[1]} polynomial features")
    
    # Step 2: Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    # Step 3: Apply regularized selection
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, y)
    selector = SelectFromModel(lasso, prefit=True)
    X_selected = selector.transform(X_scaled)
    print(f"Step 2: Selected {X_selected.shape[1]} features via Lasso")
    
    # Step 4: Optional PCA for remaining multicollinearity
    pca = PCA(n_components=target_variance_explained)
    X_final = pca.fit_transform(X_selected)
    print(f"Step 3: PCA reduced to {X_final.shape[1]} components")
    
    return X_final
```

**Best Practices Summary:**

| Stage | Action | Purpose |
|-------|--------|---------|
| **Initial** | Start with domain-driven features | Avoid random complexity |
| **Expansion** | Add polynomial/interaction features | Increase expressiveness |
| **Selection** | Apply Lasso/RF importance | Remove noise features |
| **Validation** | Use cross-validation | Detect overfitting early |
| **Regularization** | Tune regularization strength | Control complexity |
| **Monitoring** | Plot learning curves | Diagnose bias-variance |

---

## Question 4: You're building a predictive maintenance model. What types of features would you engineer from sensor data?

### Answer

**Predictive Maintenance Feature Engineering Strategy:**

**1. Time-Domain Statistical Features:**

```python
import numpy as np
import pandas as pd
from scipy import stats

def extract_statistical_features(signal, window_name=""):
    """
    Extract statistical features from a sensor signal.
    """
    features = {
        f'{window_name}mean': np.mean(signal),
        f'{window_name}std': np.std(signal),
        f'{window_name}var': np.var(signal),
        f'{window_name}min': np.min(signal),
        f'{window_name}max': np.max(signal),
        f'{window_name}range': np.max(signal) - np.min(signal),
        f'{window_name}median': np.median(signal),
        f'{window_name}skewness': stats.skew(signal),
        f'{window_name}kurtosis': stats.kurtosis(signal),
        f'{window_name}rms': np.sqrt(np.mean(signal**2)),
        f'{window_name}peak_to_peak': np.ptp(signal),
        f'{window_name}crest_factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
        f'{window_name}percentile_25': np.percentile(signal, 25),
        f'{window_name}percentile_75': np.percentile(signal, 75),
        f'{window_name}iqr': np.percentile(signal, 75) - np.percentile(signal, 25),
    }
    return features
```

**2. Frequency-Domain Features:**

```python
from scipy.fft import fft
from scipy.signal import welch

def extract_frequency_features(signal, sampling_rate, window_name=""):
    """
    Extract frequency domain features using FFT.
    """
    # FFT
    n = len(signal)
    fft_vals = np.abs(fft(signal))[:n//2]
    freqs = np.fft.fftfreq(n, 1/sampling_rate)[:n//2]
    
    # Power spectral density
    f_psd, psd = welch(signal, fs=sampling_rate)
    
    features = {
        f'{window_name}dominant_freq': freqs[np.argmax(fft_vals)],
        f'{window_name}spectral_centroid': np.sum(freqs * fft_vals) / np.sum(fft_vals),
        f'{window_name}spectral_spread': np.sqrt(np.sum(((freqs - np.sum(freqs * fft_vals) / np.sum(fft_vals))**2) * fft_vals) / np.sum(fft_vals)),
        f'{window_name}spectral_entropy': stats.entropy(psd + 1e-10),
        f'{window_name}total_power': np.sum(psd),
        f'{window_name}peak_power': np.max(psd),
        f'{window_name}power_ratio_low': np.sum(psd[f_psd < 50]) / np.sum(psd),
        f'{window_name}power_ratio_high': np.sum(psd[f_psd >= 50]) / np.sum(psd),
    }
    return features
```

**3. Rolling Window Features (Trend Detection):**

```python
def extract_rolling_features(df, sensor_cols, windows=[60, 300, 3600]):
    """
    Extract rolling window features for trend detection.
    
    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor column names
        windows: Window sizes in seconds/samples
    """
    for col in sensor_cols:
        for window in windows:
            # Rolling statistics
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
            df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
            df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
            
            # Deviation from rolling mean (anomaly indicator)
            df[f'{col}_deviation_{window}'] = df[col] - df[f'{col}_rolling_mean_{window}']
            
            # Rate of change
            df[f'{col}_rate_of_change_{window}'] = df[col].diff(window) / window
            
            # Expanding features (cumulative)
            df[f'{col}_expanding_mean'] = df[col].expanding().mean()
            df[f'{col}_expanding_std'] = df[col].expanding().std()
    
    return df
```

**4. Health Indicators:**

```python
def create_health_indicators(df, sensor_cols, baseline_period=1000):
    """
    Create health indicators comparing current state to baseline.
    """
    # Establish baseline (healthy state)
    baseline_stats = {}
    for col in sensor_cols:
        baseline_stats[col] = {
            'mean': df[col].iloc[:baseline_period].mean(),
            'std': df[col].iloc[:baseline_period].std()
        }
    
    # Calculate health indicators
    for col in sensor_cols:
        # Mahalanobis-like distance from baseline
        df[f'{col}_health_score'] = np.abs(
            (df[col] - baseline_stats[col]['mean']) / 
            (baseline_stats[col]['std'] + 1e-10)
        )
        
        # Cumulative degradation
        df[f'{col}_cumulative_deviation'] = (
            np.abs(df[col] - baseline_stats[col]['mean']).cumsum()
        )
        
        # Threshold exceedance count
        threshold = baseline_stats[col]['mean'] + 3 * baseline_stats[col]['std']
        df[f'{col}_exceedance_count'] = (df[col] > threshold).cumsum()
    
    return df
```

**5. Cross-Sensor Features:**

```python
def create_cross_sensor_features(df, sensor_pairs):
    """
    Create features from relationships between sensors.
    
    Args:
        sensor_pairs: List of tuples [(sensor1, sensor2), ...]
    """
    for sensor1, sensor2 in sensor_pairs:
        # Ratio features
        df[f'{sensor1}_{sensor2}_ratio'] = df[sensor1] / (df[sensor2] + 1e-10)
        
        # Difference features
        df[f'{sensor1}_{sensor2}_diff'] = df[sensor1] - df[sensor2]
        
        # Correlation (rolling)
        df[f'{sensor1}_{sensor2}_corr_60'] = df[sensor1].rolling(60).corr(df[sensor2])
        
        # Product (interaction)
        df[f'{sensor1}_{sensor2}_product'] = df[sensor1] * df[sensor2]
    
    return df
```

**Complete Feature Engineering Pipeline:**

```python
def predictive_maintenance_features(df, sensor_cols, sampling_rate=100):
    """
    Complete feature engineering pipeline for predictive maintenance.
    """
    features_list = []
    
    # Process each time window
    window_size = 1000  # samples per window
    
    for i in range(0, len(df) - window_size, window_size // 2):  # 50% overlap
        window_features = {'window_start': i}
        
        for col in sensor_cols:
            signal = df[col].iloc[i:i+window_size].values
            
            # Statistical features
            window_features.update(extract_statistical_features(signal, f'{col}_'))
            
            # Frequency features
            window_features.update(extract_frequency_features(signal, sampling_rate, f'{col}_'))
        
        features_list.append(window_features)
    
    return pd.DataFrame(features_list)

# Example usage
# features_df = predictive_maintenance_features(sensor_df, ['vibration', 'temperature', 'pressure'])
```

**Summary of Feature Categories:**

| Category | Examples | Purpose |
|----------|----------|---------|
| **Statistical** | Mean, std, RMS, kurtosis | Capture signal distribution |
| **Frequency** | Dominant freq, spectral entropy | Detect vibration patterns |
| **Rolling** | Moving averages, trends | Capture degradation trends |
| **Health** | Deviation from baseline | Quantify degradation |
| **Cross-sensor** | Ratios, correlations | Capture system-wide patterns |

---

## Question 5: You're implementing a real-time anomaly detection system. What feature engineering strategies would you employ to detect anomalies in streaming data?

### Answer

**Real-Time Feature Engineering Challenges:**
- Limited memory (can't store all history)
- Low latency requirements
- Concept drift handling
- Efficient computation

**Strategy 1: Online Statistical Features**

```python
import numpy as np
from collections import deque

class OnlineStatistics:
    """
    Compute statistics incrementally for streaming data.
    Uses Welford's algorithm for numerical stability.
    """
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.n = 0
        self.mean = 0
        self.M2 = 0  # Sum of squared differences
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update(self, value):
        """Update statistics with new value."""
        self.window.append(value)
        self.n += 1
        
        # Welford's online algorithm for mean and variance
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
        
        # Update min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        # For windowed statistics
        if len(self.window) == self.window_size:
            old_value = self.window[0]
            # Adjust for removed value (approximate)
            self._adjust_for_removal(old_value)
    
    def _adjust_for_removal(self, old_value):
        """Approximate adjustment when removing old value from window."""
        if self.n > 1:
            self.n -= 1
            delta = old_value - self.mean
            self.mean -= delta / self.n
            delta2 = old_value - self.mean
            self.M2 -= delta * delta2
    
    def get_features(self):
        """Return current statistical features."""
        variance = self.M2 / self.n if self.n > 1 else 0
        window_list = list(self.window)
        
        return {
            'mean': self.mean,
            'std': np.sqrt(variance),
            'min': self.min_val,
            'max': self.max_val,
            'range': self.max_val - self.min_val,
            'window_mean': np.mean(window_list),
            'window_std': np.std(window_list),
            'z_score': (window_list[-1] - self.mean) / (np.sqrt(variance) + 1e-10)
        }


# Usage
stats = OnlineStatistics(window_size=100)
for value in streaming_data:
    stats.update(value)
    features = stats.get_features()
    # Use features for anomaly detection
```

**Strategy 2: Exponentially Weighted Moving Statistics**

```python
class ExponentialMovingStats:
    """
    Exponentially weighted moving average and variance.
    More recent values have higher weight.
    """
    
    def __init__(self, alpha=0.1):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = more weight to recent values.
        """
        self.alpha = alpha
        self.ewma = None
        self.ewmvar = None
        self.initialized = False
    
    def update(self, value):
        """Update with new value."""
        if not self.initialized:
            self.ewma = value
            self.ewmvar = 0
            self.initialized = True
        else:
            diff = value - self.ewma
            incr = self.alpha * diff
            self.ewma += incr
            self.ewmvar = (1 - self.alpha) * (self.ewmvar + self.alpha * diff * diff)
        
        return self.get_features(value)
    
    def get_features(self, current_value):
        """Get current features."""
        ewm_std = np.sqrt(self.ewmvar) if self.ewmvar > 0 else 1e-10
        
        return {
            'ewma': self.ewma,
            'ewm_std': ewm_std,
            'ewm_z_score': (current_value - self.ewma) / ewm_std,
            'deviation': current_value - self.ewma,
            'deviation_ratio': (current_value - self.ewma) / (self.ewma + 1e-10)
        }


# Usage for multiple sensors
class MultiSensorOnlineFeatures:
    def __init__(self, sensor_names, alpha=0.1, window_size=100):
        self.sensors = {name: {
            'ewm': ExponentialMovingStats(alpha),
            'online': OnlineStatistics(window_size)
        } for name in sensor_names}
    
    def update(self, sensor_data):
        """
        Update with new readings from all sensors.
        
        Args:
            sensor_data: dict {sensor_name: value}
        """
        features = {}
        for name, value in sensor_data.items():
            # EWM features
            ewm_features = self.sensors[name]['ewm'].update(value)
            features.update({f'{name}_{k}': v for k, v in ewm_features.items()})
            
            # Window features
            self.sensors[name]['online'].update(value)
            online_features = self.sensors[name]['online'].get_features()
            features.update({f'{name}_win_{k}': v for k, v in online_features.items()})
        
        return features
```

**Strategy 3: Change Point Detection Features**

```python
class ChangePointDetector:
    """
    Detect sudden changes in streaming data.
    """
    
    def __init__(self, short_window=10, long_window=100, threshold=3):
        self.short_window = deque(maxlen=short_window)
        self.long_window = deque(maxlen=long_window)
        self.threshold = threshold
    
    def update(self, value):
        """Update and compute change detection features."""
        self.short_window.append(value)
        self.long_window.append(value)
        
        if len(self.short_window) < 10 or len(self.long_window) < 50:
            return {'change_score': 0, 'is_change_point': False}
        
        short_mean = np.mean(self.short_window)
        long_mean = np.mean(self.long_window)
        long_std = np.std(self.long_window)
        
        # CUSUM-like score
        change_score = abs(short_mean - long_mean) / (long_std + 1e-10)
        
        return {
            'short_mean': short_mean,
            'long_mean': long_mean,
            'change_score': change_score,
            'is_change_point': change_score > self.threshold
        }
```

**Strategy 4: Lag Features for Temporal Patterns**

```python
class LagFeatureGenerator:
    """
    Generate lag features for time-dependent patterns.
    """
    
    def __init__(self, lags=[1, 5, 10, 30, 60]):
        self.lags = lags
        self.max_lag = max(lags)
        self.buffer = deque(maxlen=self.max_lag + 1)
    
    def update(self, value):
        """Update buffer and generate lag features."""
        self.buffer.append(value)
        
        features = {'current': value}
        
        if len(self.buffer) > max(self.lags):
            buffer_list = list(self.buffer)
            for lag in self.lags:
                lag_value = buffer_list[-lag - 1]
                features[f'lag_{lag}'] = lag_value
                features[f'diff_{lag}'] = value - lag_value
                features[f'pct_change_{lag}'] = (value - lag_value) / (abs(lag_value) + 1e-10)
        
        return features
```

**Complete Real-Time Anomaly Detection Pipeline:**

```python
class RealTimeAnomalyFeatures:
    """
    Complete feature engineering pipeline for real-time anomaly detection.
    """
    
    def __init__(self, sensor_names, config=None):
        self.config = config or {
            'ewm_alpha': 0.1,
            'window_size': 100,
            'lags': [1, 5, 10, 30],
            'change_threshold': 3
        }
        
        self.multi_sensor = MultiSensorOnlineFeatures(
            sensor_names, 
            alpha=self.config['ewm_alpha'],
            window_size=self.config['window_size']
        )
        
        self.change_detectors = {
            name: ChangePointDetector(threshold=self.config['change_threshold'])
            for name in sensor_names
        }
        
        self.lag_generators = {
            name: LagFeatureGenerator(lags=self.config['lags'])
            for name in sensor_names
        }
    
    def process(self, sensor_data, timestamp=None):
        """
        Process new sensor readings and generate features.
        
        Args:
            sensor_data: dict {sensor_name: value}
            timestamp: optional timestamp
            
        Returns:
            dict of features for anomaly detection
        """
        features = {}
        
        # Add timestamp features if provided
        if timestamp:
            features.update(self._timestamp_features(timestamp))
        
        # Statistical features
        features.update(self.multi_sensor.update(sensor_data))
        
        # Change detection features
        for name, value in sensor_data.items():
            change_features = self.change_detectors[name].update(value)
            features.update({f'{name}_{k}': v for k, v in change_features.items()})
            
            # Lag features
            lag_features = self.lag_generators[name].update(value)
            features.update({f'{name}_{k}': v for k, v in lag_features.items()})
        
        # Cross-sensor features
        features.update(self._cross_sensor_features(sensor_data))
        
        return features
    
    def _timestamp_features(self, timestamp):
        """Extract time-based features."""
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'minute_of_day': timestamp.hour * 60 + timestamp.minute
        }
    
    def _cross_sensor_features(self, sensor_data):
        """Compute cross-sensor features."""
        features = {}
        sensors = list(sensor_data.keys())
        
        for i, s1 in enumerate(sensors):
            for s2 in sensors[i+1:]:
                features[f'{s1}_{s2}_ratio'] = sensor_data[s1] / (sensor_data[s2] + 1e-10)
                features[f'{s1}_{s2}_diff'] = sensor_data[s1] - sensor_data[s2]
        
        return features


# Usage example
feature_engine = RealTimeAnomalyFeatures(['temperature', 'pressure', 'vibration'])

for reading in data_stream:
    features = feature_engine.process(reading['sensors'], reading['timestamp'])
    # Feed features to anomaly detection model
```

---

## Question 6: You're working on a sentiment analysis project for social media. What features would you engineer from the text data to improve the model's performance?

### Answer

**Comprehensive NLP Feature Engineering for Sentiment Analysis:**

**1. Basic Text Features:**

```python
import pandas as pd
import numpy as np
import re
from collections import Counter

def extract_basic_text_features(text):
    """
    Extract basic statistical features from text.
    """
    # Clean text
    text_lower = text.lower()
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    features = {
        # Length features
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'avg_sentence_length': len(words) / max(len(sentences), 1),
        
        # Case features
        'uppercase_count': sum(1 for c in text if c.isupper()),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'all_caps_word_count': sum(1 for w in words if w.isupper() and len(w) > 1),
        
        # Punctuation features
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'period_count': text.count('.'),
        'punctuation_ratio': sum(1 for c in text if c in '!?.,;:') / max(len(text), 1),
        
        # Special characters
        'hashtag_count': len(re.findall(r'#\w+', text)),
        'mention_count': len(re.findall(r'@\w+', text)),
        'url_count': len(re.findall(r'http\S+|www\S+', text)),
        'emoji_count': len(re.findall(r'[\U0001F600-\U0001F64F]', text)),
        
        # Repeated characters
        'repeated_chars': len(re.findall(r'(.)\1{2,}', text)),  # e.g., "sooo"
    }
    
    return features
```

**2. Sentiment Lexicon Features:**

```python
# Positive and negative word lists (subset examples)
POSITIVE_WORDS = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                  'love', 'happy', 'joy', 'best', 'awesome', 'perfect', 'beautiful'}
NEGATIVE_WORDS = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 
                  'sad', 'angry', 'poor', 'disappointing', 'ugly', 'boring'}
INTENSIFIERS = {'very', 'really', 'extremely', 'absolutely', 'totally', 'completely'}
NEGATIONS = {'not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing'}

def extract_lexicon_features(text):
    """
    Extract features based on sentiment lexicons.
    """
    words = text.lower().split()
    
    positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
    negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    intensifier_count = sum(1 for w in words if w in INTENSIFIERS)
    negation_count = sum(1 for w in words if w in NEGATIONS)
    
    features = {
        'positive_word_count': positive_count,
        'negative_word_count': negative_count,
        'sentiment_word_ratio': (positive_count - negative_count) / max(len(words), 1),
        'positive_ratio': positive_count / max(len(words), 1),
        'negative_ratio': negative_count / max(len(words), 1),
        'intensifier_count': intensifier_count,
        'negation_count': negation_count,
        'has_negation': 1 if negation_count > 0 else 0,
    }
    
    return features


# Using VADER (Valence Aware Dictionary)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def extract_vader_features(text):
    """
    Extract VADER sentiment scores.
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    return {
        'vader_positive': scores['pos'],
        'vader_negative': scores['neg'],
        'vader_neutral': scores['neu'],
        'vader_compound': scores['compound'],
    }
```

**3. N-gram and TF-IDF Features:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def create_ngram_features(texts, max_features=5000, ngram_range=(1, 2)):
    """
    Create TF-IDF features from n-grams.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        min_df=5,
        max_df=0.95
    )
    
    tfidf_matrix = tfidf.fit_transform(texts)
    
    return tfidf_matrix, tfidf


# Character n-grams (robust to typos)
def create_char_ngram_features(texts, max_features=3000, ngram_range=(2, 5)):
    """
    Create character n-gram features.
    """
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        max_features=max_features,
        ngram_range=ngram_range
    )
    
    return char_vectorizer.fit_transform(texts), char_vectorizer
```

**4. Word Embedding Features:**

```python
import numpy as np

def get_word_embeddings(text, word2vec_model):
    """
    Get average word embedding for text.
    """
    words = text.lower().split()
    word_vectors = []
    
    for word in words:
        if word in word2vec_model:
            word_vectors.append(word2vec_model[word])
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)


def extract_embedding_features(texts, word2vec_model):
    """
    Create embedding-based features for all texts.
    """
    embeddings = np.array([get_word_embeddings(text, word2vec_model) for text in texts])
    return embeddings


# Using pre-trained embeddings (example with sentence-transformers)
from sentence_transformers import SentenceTransformer

def get_sentence_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """
    Get sentence embeddings using transformer models.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings
```

**5. Part-of-Speech and Syntactic Features:**

```python
import spacy
nlp = spacy.load('en_core_web_sm')

def extract_pos_features(text):
    """
    Extract part-of-speech features.
    """
    doc = nlp(text)
    
    # POS counts
    pos_counts = Counter([token.pos_ for token in doc])
    
    features = {
        'noun_count': pos_counts.get('NOUN', 0),
        'verb_count': pos_counts.get('VERB', 0),
        'adj_count': pos_counts.get('ADJ', 0),
        'adv_count': pos_counts.get('ADV', 0),
        'pron_count': pos_counts.get('PRON', 0),
        
        # Ratios
        'noun_ratio': pos_counts.get('NOUN', 0) / max(len(doc), 1),
        'adj_ratio': pos_counts.get('ADJ', 0) / max(len(doc), 1),
        
        # Entity counts
        'entity_count': len(doc.ents),
        'person_entity_count': sum(1 for ent in doc.ents if ent.label_ == 'PERSON'),
        'org_entity_count': sum(1 for ent in doc.ents if ent.label_ == 'ORG'),
    }
    
    return features
```

**6. Social Media Specific Features:**

```python
def extract_social_media_features(text):
    """
    Extract features specific to social media text.
    """
    # Emoji sentiment (simplified mapping)
    positive_emojis = ['😀', '😊', '😃', '❤️', '👍', '🎉', '😍', '🙂']
    negative_emojis = ['😢', '😡', '😠', '👎', '💔', '😞', '😤', '🙁']
    
    pos_emoji_count = sum(text.count(e) for e in positive_emojis)
    neg_emoji_count = sum(text.count(e) for e in negative_emojis)
    
    features = {
        'positive_emoji_count': pos_emoji_count,
        'negative_emoji_count': neg_emoji_count,
        'emoji_sentiment': (pos_emoji_count - neg_emoji_count) / max(pos_emoji_count + neg_emoji_count, 1),
        
        # Slang and abbreviations
        'lol_count': text.lower().count('lol'),
        'omg_count': text.lower().count('omg'),
        
        # Elongated words (emphasis)
        'elongated_word_count': len(re.findall(r'\b\w*(.)\1{2,}\w*\b', text)),
        
        # All caps (shouting)
        'caps_word_ratio': sum(1 for w in text.split() if w.isupper()) / max(len(text.split()), 1),
    }
    
    return features
```

**Complete Feature Engineering Pipeline:**

```python
class SentimentFeatureEngineering:
    """
    Complete feature engineering pipeline for sentiment analysis.
    """
    
    def __init__(self, use_embeddings=True):
        self.use_embeddings = use_embeddings
        self.tfidf = None
        self.vader = SentimentIntensityAnalyzer()
    
    def fit(self, texts):
        """Fit feature extractors on training data."""
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        self.tfidf.fit(texts)
        return self
    
    def transform(self, texts):
        """Extract all features from texts."""
        features_list = []
        
        for text in texts:
            features = {}
            
            # Basic features
            features.update(extract_basic_text_features(text))
            
            # Lexicon features
            features.update(extract_lexicon_features(text))
            
            # VADER features
            features.update(extract_vader_features(text))
            
            # Social media features
            features.update(extract_social_media_features(text))
            
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Add TF-IDF features
        if self.tfidf:
            tfidf_features = self.tfidf.transform(texts).toarray()
            tfidf_df = pd.DataFrame(
                tfidf_features, 
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
            features_df = pd.concat([features_df, tfidf_df], axis=1)
        
        return features_df
    
    def fit_transform(self, texts):
        """Fit and transform."""
        return self.fit(texts).transform(texts)


# Usage
fe = SentimentFeatureEngineering()
X_train_features = fe.fit_transform(train_texts)
X_test_features = fe.transform(test_texts)
```

**Feature Summary:**

| Category | Features | Purpose |
|----------|----------|---------|
| **Basic** | Length, case, punctuation | Capture writing style |
| **Lexicon** | Positive/negative word counts | Direct sentiment indicators |
| **VADER** | Compound score | Pre-trained sentiment |
| **N-gram** | TF-IDF vectors | Capture phrases and context |
| **Embedding** | Word2Vec, BERT | Semantic meaning |
| **POS** | Adjective/adverb counts | Grammatical patterns |
| **Social** | Emojis, hashtags | Platform-specific signals |

---
