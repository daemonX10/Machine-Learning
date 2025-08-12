# Xgboost Interview Questions - Coding Questions

## Question 1

**What isearly stoppinginXGBoostand how can it be implemented?**

**Answer:**

Early stopping is a regularization technique that stops training when the validation performance stops improving, preventing overfitting and saving computational time.

**Concept:**
- Monitor validation metric during training
- Stop when performance doesn't improve for specified rounds
- Helps find optimal number of estimators automatically

**Implementation Examples:**

**Method 1: Using sklearn-style XGBoost with early stopping**
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import numpy as np

# Load and prepare data
data = load_boston()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val**Code a Python script that demonstrates how to useXGBoost'sbuilt-infeature importanceto rank features.**

**Answer:**

Here's a comprehensive Python script demonstrating XGBoost's built-in feature importance methods:

```python
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class XGBoostFeatureImportanceAnalyzer:
    """
    Comprehensive XGBoost Feature Importance Analysis
    
    This class provides multiple methods to analyze and rank features
    using XGBoost's built-in importance metrics.
    """
    
    def __init__(self, task_type='classification'):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        task_type : str, default='classification'
            Type of ML task: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.model = None
        self.feature_names = None
        self.importance_scores = {}
        
    def prepare_data(self, dataset_name='breast_cancer'):
        """
        Load and prepare sample datasets
        
        Parameters:
        -----------
        dataset_name : str, default='breast_cancer'
            Name of dataset: 'breast_cancer', 'diabetes'
        
        Returns:
        --------
        tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        if dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            self.task_type = 'classification'
        elif dataset_name == 'diabetes':
            data = load_diabetes()
            self.task_type = 'regression'
        else:
            raise ValueError("Supported datasets: 'breast_cancer', 'diabetes'")
        
        X, y = data.data, data.target
        self.feature_names = data.feature_names
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Dataset: {dataset_name}")
        print(f"Task type: {self.task_type}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, self.feature_names
    
    def train_model(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train XGBoost model with optimal parameters
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data (optional, for evaluation)
        """
        if self.task_type == 'classification':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                importance_type='gain'  # Default importance metric
            )
        else:  # regression
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                importance_type='gain'
            )
        
        # Train the model
        if X_test is not None and y_test is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Evaluate model performance
        if X_test is not None and y_test is not None:
            predictions = self.model.predict(X_test)
            if self.task_type == 'classification':
                score = accuracy_score(y_test, predictions)
                print(f"Model Accuracy: {score:.4f}")
            else:
                score = mean_squared_error(y_test, predictions, squared=False)
                print(f"Model RMSE: {score:.4f}")
        
        print("Model training completed!")
    
    def extract_all_importance_types(self):
        """
        Extract all available importance types from trained XGBoost model
        
        Returns:
        --------
        dict: Dictionary containing all importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        
        for imp_type in importance_types:
            try:
                # Get importance scores
                scores = self.model.get_booster().get_score(importance_type=imp_type)
                
                # Convert to DataFrame format
                if self.feature_names is not None:
                    # Map feature indices to names
                    feature_scores = []
                    for i, feature_name in enumerate(self.feature_names):
                        feature_key = f'f{i}'
                        score = scores.get(feature_key, 0.0)
                        feature_scores.append((feature_name, score))
                    
                    # Sort by importance score
                    feature_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    self.importance_scores[imp_type] = feature_scores
                else:
                    # Use feature indices
                    sorted_scores = sorted(scores.items(), 
                                         key=lambda x: x[1], reverse=True)
                    self.importance_scores[imp_type] = sorted_scores
                
                print(f"✓ Extracted {imp_type} importance scores")
            except Exception as e:
                print(f"✗ Failed to extract {imp_type}: {e}")
        
        return self.importance_scores
    
    def create_importance_dataframe(self):
        """
        Create a comprehensive DataFrame with all importance metrics
        
        Returns:
        --------
        pd.DataFrame: DataFrame with features and their importance scores
        """
        if not self.importance_scores:
            self.extract_all_importance_types()
        
        # Initialize DataFrame with feature names
        df_data = {'feature': self.feature_names}
        
        # Add importance scores for each type
        for imp_type, scores in self.importance_scores.items():
            # Create mapping from feature name to score
            score_dict = {feature: score for feature, score in scores}
            
            # Map scores to features (handle missing features with 0)
            df_data[f'{imp_type}_importance'] = [
                score_dict.get(feature, 0.0) for feature in self.feature_names
            ]
            
            # Add ranking
            ranks = pd.Series(df_data[f'{imp_type}_importance']).rank(
                method='dense', ascending=False
            )
            df_data[f'{imp_type}_rank'] = ranks.astype(int)
        
        importance_df = pd.DataFrame(df_data)
        
        # Calculate average rank across all importance types
        rank_columns = [col for col in importance_df.columns if col.endswith('_rank')]
        importance_df['average_rank'] = importance_df[rank_columns].mean(axis=1)
        
        # Sort by average rank
        importance_df = importance_df.sort_values('average_rank')
        
        return importance_df
    
    def plot_feature_importance(self, top_n=15, figsize=(15, 10)):
        """
        Create comprehensive visualizations of feature importance
        
        Parameters:
        -----------
        top_n : int, default=15
            Number of top features to display
        figsize : tuple, default=(15, 10)
            Figure size for plots
        """
        if not self.importance_scores:
            self.extract_all_importance_types()
        
        importance_df = self.create_importance_dataframe()
        
        # Select top N features
        top_features = importance_df.head(top_n)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()
        
        importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        
        for i, imp_type in enumerate(importance_types):
            if i < len(axes):
                col_name = f'{imp_type}_importance'
                
                # Sort features by current importance type
                sorted_features = top_features.sort_values(
                    col_name, ascending=True
                ).tail(top_n)
                
                # Create horizontal bar plot
                y_pos = np.arange(len(sorted_features))
                axes[i].barh(y_pos, sorted_features[col_name], 
                           alpha=0.8, color=plt.cm.viridis(i/len(importance_types)))
                axes[i].set_yticks(y_pos)
                axes[i].set_yticklabels(sorted_features['feature'], fontsize=8)
                axes[i].set_xlabel('Importance Score')
                axes[i].set_title(f'{imp_type.title()} Importance')
                axes[i].grid(axis='x', alpha=0.3)
        
        # Use the last subplot for average ranking
        if len(axes) > len(importance_types):
            sorted_avg = top_features.sort_values('average_rank')
            y_pos = np.arange(len(sorted_avg))
            axes[-1].barh(y_pos, 1/sorted_avg['average_rank'], 
                         alpha=0.8, color='orange')
            axes[-1].set_yticks(y_pos)
            axes[-1].set_yticklabels(sorted_avg['feature'], fontsize=8)
            axes[-1].set_xlabel('Inverse Average Rank')
            axes[-1].set_title('Average Ranking Across All Metrics')
            axes[-1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'XGBoost Feature Importance Analysis - Top {top_n} Features', 
                    y=1.02, fontsize=16)
        plt.show()
    
    def print_importance_ranking(self, top_n=10):
        """
        Print detailed feature importance rankings
        
        Parameters:
        -----------
        top_n : int, default=10
            Number of top features to display
        """
        importance_df = self.create_importance_dataframe()
        
        print(f"\n{'='*60}")
        print(f"TOP {top_n} MOST IMPORTANT FEATURES")
        print(f"{'='*60}")
        
        for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
            print(f"\n{i+1:2d}. {row['feature']}")
            print(f"    Average Rank: {row['average_rank']:.2f}")
            
            # Print individual importance scores
            for imp_type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
                score = row[f'{imp_type}_importance']
                rank = row[f'{imp_type}_rank']
                print(f"    {imp_type:12s}: {score:8.4f} (rank: {rank:2d})")
    
    def compare_importance_methods(self):
        """
        Compare different importance methods and their correlations
        """
        importance_df = self.create_importance_dataframe()
        
        # Select importance score columns
        importance_cols = [col for col in importance_df.columns 
                          if col.endswith('_importance')]
        
        # Calculate correlation matrix
        correlation_matrix = importance_df[importance_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   center=0, square=True)
        plt.title('Correlation Between Different Importance Metrics')
        plt.tight_layout()
        plt.show()
        
        print("\nCorrelation Analysis:")
        print("="*40)
        for i in range(len(importance_cols)):
            for j in range(i+1, len(importance_cols)):
                corr = correlation_matrix.iloc[i, j]
                method1 = importance_cols[i].replace('_importance', '')
                method2 = importance_cols[j].replace('_importance', '')
                print(f"{method1:12s} vs {method2:12s}: {corr:.3f}")
        
        return correlation_matrix
    
    def save_results(self, filename='xgboost_feature_importance.csv'):
        """
        Save feature importance results to CSV file
        
        Parameters:
        -----------
        filename : str, default='xgboost_feature_importance.csv'
            Output filename
        """
        importance_df = self.create_importance_dataframe()
        importance_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def demonstrate_feature_importance():
    """
    Complete demonstration of XGBoost feature importance analysis
    """
    print("=== XGBoost Feature Importance Analysis Demo ===\n")
    
    # Initialize analyzer
    analyzer = XGBoostFeatureImportanceAnalyzer()
    
    # Demo 1: Classification task
    print("DEMO 1: Classification Task (Breast Cancer Dataset)")
    print("-" * 50)
    
    X_train, X_test, y_train, y_test, feature_names = analyzer.prepare_data('breast_cancer')
    analyzer.train_model(X_train, y_train, X_test, y_test)
    
    # Extract and analyze importance
    analyzer.extract_all_importance_types()
    analyzer.print_importance_ranking(top_n=10)
    analyzer.plot_feature_importance(top_n=15)
    
    # Compare methods
    correlation_matrix = analyzer.compare_importance_methods()
    
    # Save results
    analyzer.save_results('breast_cancer_feature_importance.csv')
    
    print("\n" + "="*60 + "\n")
    
    # Demo 2: Regression task
    print("DEMO 2: Regression Task (Diabetes Dataset)")
    print("-" * 50)
    
    analyzer_reg = XGBoostFeatureImportanceAnalyzer()
    X_train, X_test, y_train, y_test, feature_names = analyzer_reg.prepare_data('diabetes')
    analyzer_reg.train_model(X_train, y_train, X_test, y_test)
    
    analyzer_reg.extract_all_importance_types()
    analyzer_reg.print_importance_ranking(top_n=8)  # Fewer features in diabetes dataset
    analyzer_reg.plot_feature_importance(top_n=8)
    
    analyzer_reg.save_results('diabetes_feature_importance.csv')

# Quick usage example for custom data
def analyze_custom_data(X, y, feature_names=None, task_type='classification'):
    """
    Quick function to analyze feature importance for custom data
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target variable
    feature_names : list, optional
        Names of features
    task_type : str, default='classification'
        Type of ML task
    
    Returns:
    --------
    pd.DataFrame: Feature importance results
    """
    # Initialize analyzer
    analyzer = XGBoostFeatureImportanceAnalyzer(task_type=task_type)
    
    # Set feature names
    if feature_names is not None:
        analyzer.feature_names = feature_names
    else:
        analyzer.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model and analyze
    analyzer.train_model(X_train, y_train, X_test, y_test)
    analyzer.extract_all_importance_types()
    
    # Get results
    importance_df = analyzer.create_importance_dataframe()
    
    print("Top 10 Most Important Features:")
    analyzer.print_importance_ranking(top_n=10)
    
    return importance_df

# Run the demonstration
if __name__ == "__main__":
    demonstrate_feature_importance()
    
    print("\n" + "="*60)
    print("Feature Importance Analysis Complete!")
    print("="*60)
```

**Key Features of this Implementation:**

1. **Multiple Importance Metrics:**
   - **Weight:** Number of times feature is used for splits
   - **Gain:** Average gain across all splits using the feature
   - **Cover:** Average coverage across all splits using the feature
   - **Total Gain:** Total gain when feature is used
   - **Total Cover:** Total coverage when feature is used

2. **Comprehensive Analysis:**
   - Ranking comparison across all metrics
   - Correlation analysis between methods
   - Visual comparisons and plots
   - Statistical summaries

3. **Practical Usage:**
   - Works with both classification and regression
   - Handles custom datasets
   - Exports results to CSV
   - Provides interpretable rankings

4. **Visualization:**
   - Multiple bar plots for different importance types
   - Correlation heatmaps
   - Ranking comparisons

**Understanding the Importance Types:**
- **Weight/Frequency:** How often a feature is used
- **Gain:** How much the feature improves the model
- **Cover:** How many samples are affected by the feature
- Use **gain** for overall feature contribution assessment
- Use **weight** for feature selection frequency
- Combine multiple metrics for robust feature ranking_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Method 1: Using fit with eval_set
model = xgb.XGBRegressor(
    n_estimators=1000,  # Large number, early stopping will find optimal
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

# Fit with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='rmse',
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    verbose=True  # Print progress
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score}")

# Predict using best iteration
y_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {test_rmse:.4f}")
```

**Method 2: Using native XGBoost API with early stopping**
```python
# Prepare data in XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Evaluation list for monitoring
evallist = [(dtrain, 'train'), (dval, 'eval')]

# Train with early stopping
model_native = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,  # Maximum rounds
    evals=evallist,
    early_stopping_rounds=50,
    verbose_eval=50  # Print every 50 rounds
)

print(f"Best iteration: {model_native.best_iteration}")
print(f"Best score: {model_native.best_score}")

# Predict
y_pred_native = model_native.predict(dtest)
test_rmse_native = np.sqrt(mean_squared_error(y_test, y_pred_native))
print(f"Test RMSE (native): {test_rmse_native:.4f}")
```

**Method 3: Custom early stopping with callback functions**
```python
# Custom early stopping callback
def custom_early_stopping(stopping_rounds, maximize=False, verbose=True):
    """Custom early stopping callback with more control"""
    
    def callback(env):
        # Get current iteration and evaluation results
        iteration = env.iteration
        eval_result = env.evaluation_result_list
        
        if len(eval_result) == 0:
            return
        
        # Get validation score (assuming last metric is validation)
        current_score = eval_result[-1][1]
        
        # Initialize tracking variables if first iteration
        if not hasattr(callback, 'best_score'):
            callback.best_score = current_score
            callback.best_iteration = iteration
            callback.stopping_count = 0
        
        # Check if current score is better
        is_better = (current_score > callback.best_score if maximize 
                    else current_score < callback.best_score)
        
        if is_better:
            callback.best_score = current_score
            callback.best_iteration = iteration
            callback.stopping_count = 0
            if verbose:
                print(f"[{iteration}] New best score: {current_score:.6f}")
        else:
            callback.stopping_count += 1
            if verbose and callback.stopping_count % 10 == 0:
                print(f"[{iteration}] No improvement for {callback.stopping_count} rounds")
        
        # Stop if no improvement for specified rounds
        if callback.stopping_count >= stopping_rounds:
            if verbose:
                print(f"Early stopping at iteration {iteration}")
                print(f"Best iteration: {callback.best_iteration}")
                print(f"Best score: {callback.best_score:.6f}")
            raise xgb.core.EarlyStopException(callback.best_iteration)
    
    return callback

# Use custom early stopping
try:
    model_custom = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evallist,
        callbacks=[custom_early_stopping(50, maximize=False, verbose=True)],
        verbose_eval=False  # Disable default verbose to use custom
    )
except xgb.core.EarlyStopException as e:
    print(f"Training stopped early at iteration: {e.best_iteration}")
```

**Method 4: Early stopping with different metrics**
```python
# Early stopping with multiple metrics
def early_stopping_multiple_metrics():
    """Demonstrate early stopping with multiple evaluation metrics"""
    
    # Parameters with multiple evaluation metrics
    params_multi = {
        'objective': 'reg:squarederror',
        'eval_metric': ['rmse', 'mae'],  # Multiple metrics
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    # Train with early stopping on first metric (rmse)
    model_multi = xgb.train(
        params_multi,
        dtrain,
        num_boost_round=1000,
        evals=evallist,
        early_stopping_rounds=30,
        verbose_eval=50
    )
    
    return model_multi

# Run example
model_multi_metrics = early_stopping_multiple_metrics()
```

**Method 5: Early stopping in cross-validation**
```python
# Early stopping with cross-validation
def early_stopping_cv():
    """Early stopping using cross-validation"""
    
    # Cross-validation with early stopping
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        nfold=5,
        early_stopping_rounds=50,
        metrics='rmse',
        seed=42,
        verbose_eval=50
    )
    
    print(f"Best iteration (CV): {len(cv_results)}")
    print(f"Best CV score: {cv_results['test-rmse-mean'].iloc[-1]:.6f}")
    
    return cv_results

# Run CV with early stopping
cv_results = early_stopping_cv()
```

**Best Practices for Early Stopping:**

1. **Validation Set Size:** Use 10-20% of training data for validation
2. **Stopping Rounds:** Start with 10-50 rounds, adjust based on dataset size
3. **Metric Choice:** Use same metric as final evaluation when possible
4. **Monitor Multiple Metrics:** Track both training and validation metrics
5. **Save Best Model:** Always use the best iteration, not the last one

**Common Parameters:**
- `early_stopping_rounds`: Number of rounds without improvement before stopping
- `eval_set`: Validation data for monitoring
- `eval_metric`: Metric to monitor for early stopping
- `verbose`: Whether to print progress information

---

## Question 2

**Write a Python code to load adataset, create anXGBoost model, and fit it to the data.**

**Answer:**

Here's a comprehensive example showing how to load datasets, create XGBoost models, and fit them for both classification and regression tasks:

**Example 1: Regression Task with Boston Housing Dataset**
```python
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ============= REGRESSION EXAMPLE =============
print("=== XGBoost Regression Example ===")

# Method 1: Load dataset from sklearn
def load_and_prepare_regression_data():
    """Load and prepare regression dataset"""
    # Load Boston housing dataset
    boston = load_boston()
    X, y = boston.data, boston.target
    feature_names = boston.feature_names
    
    # Create DataFrame for better handling
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(feature_names)}")
    print(f"Target statistics:\n{df['target'].describe()}")
    
    return X, y, feature_names

# Load data
X, y, feature_names = load_and_prepare_regression_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Create and train XGBoost regressor
print("\n--- Training XGBoost Regressor ---")

# Method A: Using XGBRegressor (sklearn-style)
xgb_regressor = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Fit the model
xgb_regressor.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='rmse',
    early_stopping_rounds=20,
    verbose=False
)

# Make predictions
y_pred_train = xgb_regressor.predict(X_train)
y_pred_val = xgb_regressor.predict(X_val)
y_pred_test = xgb_regressor.predict(X_test)

# Evaluate performance
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# Method B: Using native XGBoost API
print("\n--- Using Native XGBoost API ---")

# Prepare data in DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

# Set parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Train model
evallist = [(dtrain, 'train'), (dval, 'eval')]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=evallist,
    early_stopping_rounds=20,
    verbose_eval=False
)

# Predictions with native API
y_pred_native = bst.predict(dtest)
test_rmse_native = np.sqrt(mean_squared_error(y_test, y_pred_native))
print(f"Test RMSE (native API): {test_rmse_native:.4f}")
```

**Example 2: Classification Task with Iris Dataset**
```python
# ============= CLASSIFICATION EXAMPLE =============
print("\n=== XGBoost Classification Example ===")

def load_and_prepare_classification_data():
    """Load and prepare classification dataset"""
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_names, target_names

# Load classification data
X_clf, y_clf, feature_names_clf, target_names = load_and_prepare_classification_data()

# Split the data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

print(f"Training set: {X_train_clf.shape}")
print(f"Test set: {X_test_clf.shape}")

# Create and train XGBoost classifier
print("\n--- Training XGBoost Classifier ---")

xgb_classifier = xgb.XGBClassifier(
    objective='multi:softprob',
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# Fit the classifier
xgb_classifier.fit(X_train_clf, y_train_clf)

# Make predictions
y_pred_clf = xgb_classifier.predict(X_test_clf)
y_pred_proba = xgb_classifier.predict_proba(X_test_clf)

# Evaluate classification performance
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_clf, y_pred_clf, target_names=target_names))

# Show predictions with probabilities
print("\nSample Predictions:")
for i in range(min(5, len(y_test_clf))):
    true_class = target_names[y_test_clf[i]]
    pred_class = target_names[y_pred_clf[i]]
    confidence = y_pred_proba[i].max()
    print(f"True: {true_class}, Predicted: {pred_class}, Confidence: {confidence:.3f}")
```

**Example 3: Loading Custom Dataset from CSV**
```python
# ============= CUSTOM DATASET EXAMPLE =============
print("\n=== Loading Custom Dataset Example ===")

def create_sample_dataset():
    """Create a sample dataset and save as CSV"""
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to CSV
    df.to_csv('sample_dataset.csv', index=False)
    print("Sample dataset created and saved as 'sample_dataset.csv'")
    
    return df

def load_custom_dataset(file_path):
    """Load dataset from CSV file"""
    try:
        # Load CSV file
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Separate features and target
        # Assuming last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = df.columns[:-1].tolist()
        
        # Handle categorical variables if any
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            print(f"Categorical columns found: {list(categorical_columns)}")
            # Use LabelEncoder for categorical variables
            le = LabelEncoder()
            for col in categorical_columns:
                if col != df.columns[-1]:  # Skip target column
                    col_idx = df.columns.get_loc(col)
                    X[:, col_idx] = le.fit_transform(X[:, col_idx])
        
        return X, y, feature_names
        
    except FileNotFoundError:
        print(f"File {file_path} not found. Creating sample dataset...")
        df = create_sample_dataset()
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = df.columns[:-1].tolist()
        return X, y, feature_names

# Load custom dataset
X_custom, y_custom, feature_names_custom = load_custom_dataset('sample_dataset.csv')

# Split and train
X_train_custom, X_test_custom, y_train_custom, y_test_custom = train_test_split(
    X_custom, y_custom, test_size=0.2, random_state=42
)

# Train XGBoost model
print("\n--- Training on Custom Dataset ---")
xgb_custom = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

xgb_custom.fit(X_train_custom, y_train_custom)

# Evaluate
y_pred_custom = xgb_custom.predict(X_test_custom)
accuracy_custom = accuracy_score(y_test_custom, y_pred_custom)
print(f"Custom Dataset Accuracy: {accuracy_custom:.4f}")
```

**Example 4: Complete Pipeline with Preprocessing**
```python
# ============= COMPLETE PIPELINE EXAMPLE =============
print("\n=== Complete Pipeline with Preprocessing ===")

def complete_xgboost_pipeline():
    """Complete XGBoost pipeline with preprocessing"""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    
    # Create dataset with missing values and mixed types
    np.random.seed(42)
    n_samples = 1000
    
    # Generate mixed dataset
    X_numeric = np.random.randn(n_samples, 5)
    X_categorical = np.random.choice(['A', 'B', 'C'], size=(n_samples, 2))
    
    # Introduce missing values
    missing_mask = np.random.random((n_samples, 5)) < 0.1
    X_numeric[missing_mask] = np.nan
    
    # Create target variable
    y = (X_numeric[:, 0] + X_numeric[:, 1] + 
         (X_categorical[:, 0] == 'A').astype(int) > 0).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(X_numeric, columns=[f'num_{i}' for i in range(5)])
    df['cat_1'] = X_categorical[:, 0]
    df['cat_2'] = X_categorical[:, 1]
    df['target'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Preprocessing
    # Separate numeric and categorical columns
    numeric_columns = [col for col in df.columns if col.startswith('num_')]
    categorical_columns = [col for col in df.columns if col.startswith('cat_')]
    
    # Handle missing values in numeric columns
    imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Fit with evaluation
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Pipeline Accuracy: {accuracy:.4f}")
    print(f"Feature importance:")
    for feature, importance in zip(X.columns, model.feature_importances_):
        print(f"  {feature}: {importance:.4f}")
    
    return model, X_test, y_test

# Run complete pipeline
model, X_test_pipeline, y_test_pipeline = complete_xgboost_pipeline()
```

**Key Points for Loading and Training XGBoost:**

1. **Data Loading:** Use pandas for CSV files, sklearn for built-in datasets
2. **Data Splitting:** Always split into train/validation/test sets
3. **Model Creation:** Choose appropriate objective function
4. **Training:** Use early stopping and evaluation sets
5. **Evaluation:** Use appropriate metrics for your problem type
6. **Preprocessing:** Handle missing values and categorical variables appropriately

---

## Question 3

**Implement a Python function that usescross-validationto optimize thehyperparametersof anXGBoost model.**

**Answer:**

Here's a comprehensive implementation for XGBoost hyperparameter optimization using cross-validation:

```python
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, 
    cross_val_score, StratifiedKFold
)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def optimize_xgboost_hyperparameters(X, y, method='grid', 
                                   cv_folds=5, scoring='accuracy',
                                   n_iter=100, test_size=0.2, 
                                   random_state=42):
    """
    Comprehensive XGBoost hyperparameter optimization using cross-validation
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training features
    y : array-like, shape (n_samples,)
        Training targets
    method : str, default='grid'
        Optimization method: 'grid', 'random', or 'bayesian'
    cv_folds : int, default=5
        Number of cross-validation folds
    scoring : str, default='accuracy'
        Scoring metric for optimization
    n_iter : int, default=100
        Number of iterations for RandomizedSearchCV
    test_size : float, default=0.2
        Proportion of data for final testing
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    dict: Contains best model, parameters, and performance metrics
    """
    
    # 1. Data splitting
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Cross-validation folds: {cv_folds}")
    
    # 2. Define parameter grids
    if method == 'grid':
        # Comprehensive grid search parameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.5],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
        
        # Initialize GridSearchCV
        search = GridSearchCV(
            estimator=xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=random_state,
                n_jobs=-1
            ),
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
    elif method == 'random':
        # Random search parameters with distributions
        param_distributions = {
            'n_estimators': np.arange(50, 500, 50),
            'max_depth': np.arange(3, 10),
            'learning_rate': np.logspace(-3, 0, 20),
            'subsample': np.arange(0.6, 1.0, 0.1),
            'colsample_bytree': np.arange(0.6, 1.0, 0.1),
            'gamma': np.logspace(-3, 2, 20),
            'reg_alpha': np.logspace(-3, 2, 20),
            'reg_lambda': np.logspace(-3, 2, 20)
        }
        
        # Initialize RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=random_state,
                n_jobs=-1
            ),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=random_state,
            return_train_score=True
        )
    
    # 3. Perform hyperparameter optimization
    print(f"Starting {method} search optimization...")
    search.fit(X_train, y_train)
    
    # 4. Extract best model and parameters
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_score = search.best_score_
    
    print(f"\nBest cross-validation {scoring}: {best_cv_score:.4f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # 5. Final model evaluation on test set
    test_predictions = best_model.predict(X_test)
    test_score = accuracy_score(y_test, test_predictions)
    
    print(f"\nTest set accuracy: {test_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions))
    
    # 6. Analyze hyperparameter importance
    results_df = pd.DataFrame(search.cv_results_)
    
    # Plot parameter performance analysis
    plot_hyperparameter_analysis(results_df, best_params, method)
    
    # 7. Cross-validation stability analysis
    cv_scores = cross_val_score(
        best_model, X_train, y_train, 
        cv=cv_folds, scoring=scoring
    )
    
    print(f"\nCross-validation stability:")
    print(f"  Mean: {cv_scores.mean():.4f}")
    print(f"  Std:  {cv_scores.std():.4f}")
    print(f"  Min:  {cv_scores.min():.4f}")
    print(f"  Max:  {cv_scores.max():.4f}")
    
    return {
        'best_model': best_model,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'test_score': test_score,
        'cv_results': results_df,
        'search_object': search,
        'cv_stability': {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
    }

def plot_hyperparameter_analysis(results_df, best_params, method):
    """Plot hyperparameter analysis results"""
    
    # Extract key parameters for visualization
    key_params = ['max_depth', 'learning_rate', 'n_estimators', 'subsample']
    available_params = [p for p in key_params if f'param_{p}' in results_df.columns]
    
    if len(available_params) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, param in enumerate(available_params[:4]):
            if i < len(axes):
                param_col = f'param_{param}'
                
                # Group by parameter value and calculate mean score
                param_analysis = results_df.groupby(param_col).agg({
                    'mean_test_score': ['mean', 'std', 'count']
                }).round(4)
                
                param_analysis.columns = ['mean_score', 'std_score', 'count']
                param_analysis = param_analysis.reset_index()
                
                # Plot
                axes[i].errorbar(
                    param_analysis[param_col], 
                    param_analysis['mean_score'],
                    yerr=param_analysis['std_score'],
                    marker='o', capsize=5
                )
                axes[i].axvline(
                    best_params[param], 
                    color='red', linestyle='--', 
                    label=f'Best: {best_params[param]}'
                )
                axes[i].set_xlabel(param)
                axes[i].set_ylabel('CV Score')
                axes[i].set_title(f'{param} vs CV Score')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Hyperparameter Analysis - {method.title()} Search', 
                    y=1.02, fontsize=16)
        plt.show()

def advanced_xgboost_optimization(X, y, use_early_stopping=True):
    """
    Advanced XGBoost optimization with early stopping and custom evaluation
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Multi-stage optimization
    print("Stage 1: Coarse grid search for major parameters...")
    
    # Stage 1: Major parameters
    major_params = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    
    stage1_search = GridSearchCV(
        xgb.XGBClassifier(random_state=42),
        major_params, cv=3, scoring='accuracy', n_jobs=-1
    )
    stage1_search.fit(X_train, y_train)
    
    print("Stage 2: Fine-tuning around best parameters...")
    
    # Stage 2: Fine-tuning
    best_major = stage1_search.best_params_
    fine_params = {
        'n_estimators': [best_major['n_estimators']],
        'max_depth': [max(3, best_major['max_depth']-1), 
                     best_major['max_depth'],
                     best_major['max_depth']+1],
        'learning_rate': [best_major['learning_rate']],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }
    
    stage2_search = GridSearchCV(
        xgb.XGBClassifier(random_state=42),
        fine_params, cv=5, scoring='accuracy', n_jobs=-1
    )
    stage2_search.fit(X_train, y_train)
    
    # Final model with early stopping
    if use_early_stopping:
        final_model = xgb.XGBClassifier(**stage2_search.best_params_, 
                                      n_estimators=1000, random_state=42)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        print(f"Early stopping at iteration: {final_model.best_iteration}")
    else:
        final_model = stage2_search.best_estimator_
    
    # Evaluate final model
    test_score = final_model.score(X_test, y_test)
    print(f"Final test accuracy: {test_score:.4f}")
    
    return {
        'final_model': final_model,
        'stage1_best': stage1_search.best_params_,
        'stage2_best': stage2_search.best_params_,
        'test_accuracy': test_score
    }

# Example usage and demonstration
if __name__ == "__main__":
    # Load sample dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print("=== XGBoost Hyperparameter Optimization Demo ===\n")
    
    # Method 1: Grid Search
    print("1. Grid Search Optimization:")
    grid_results = optimize_xgboost_hyperparameters(
        X, y, method='grid', cv_folds=3, n_iter=50
    )
    
    print("\n" + "="*50 + "\n")
    
    # Method 2: Random Search
    print("2. Random Search Optimization:")
    random_results = optimize_xgboost_hyperparameters(
        X, y, method='random', cv_folds=5, n_iter=100
    )
    
    print("\n" + "="*50 + "\n")
    
    # Method 3: Advanced Multi-stage Optimization
    print("3. Advanced Multi-stage Optimization:")
    advanced_results = advanced_xgboost_optimization(X, y)
    
    print("\n" + "="*50 + "\n")
    
    # Compare results
    print("=== COMPARISON OF METHODS ===")
    print(f"Grid Search Test Accuracy:     {grid_results['test_score']:.4f}")
    print(f"Random Search Test Accuracy:   {random_results['test_score']:.4f}")
    print(f"Advanced Method Test Accuracy: {advanced_results['test_accuracy']:.4f}")
```

**Key Features of this Implementation:**

1. **Multiple Optimization Methods:** Grid search, random search, and advanced multi-stage optimization
2. **Cross-Validation Integration:** Proper CV setup with stratification
3. **Comprehensive Parameter Grids:** Covers all major XGBoost hyperparameters
4. **Performance Analysis:** Visualization and stability analysis
5. **Early Stopping Support:** Prevents overfitting in final models
6. **Robust Evaluation:** Separate test set for unbiased evaluation

**Usage Tips:**
- Start with random search for large parameter spaces
- Use grid search for fine-tuning around promising regions
- Always validate on separate test set
- Monitor cross-validation stability
- Consider computational resources when choosing method

---
# Temporary answers for Questions 4 and 5

## Question 4 Answer:

**Code a Python script that demonstrates how to useXGBoost'sbuilt-infeature importanceto rank features.**

**Answer:**

Here's a comprehensive Python script that demonstrates XGBoost's built-in feature importance methods to rank features:

```python
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class XGBoostFeatureImportanceAnalyzer:
    """
    Comprehensive XGBoost Feature Importance Analyzer
    
    This class demonstrates all built-in feature importance methods in XGBoost
    and provides visualization and ranking capabilities.
    """
    
    def __init__(self, model_params=None):
        """Initialize the analyzer with optional model parameters"""
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
        self.model_params = model_params or self.default_params
        self.model = None
        self.feature_names = None
        self.importance_scores = {}
    
    def extract_all_importance_types(self):
        """Extract all types of feature importance from trained XGBoost model"""
        
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print("Extracting feature importance scores...")
        
        # XGBoost provides several types of feature importance
        importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        
        for imp_type in importance_types:
            try:
                # Get importance scores from booster
                scores = self.model.get_booster().get_score(importance_type=imp_type)
                
                if self.feature_names is not None:
                    # Map feature indices to names
                    feature_scores = []
                    for i, feature_name in enumerate(self.feature_names):
                        feature_key = f'f{i}'
                        score = scores.get(feature_key, 0.0)
                        feature_scores.append((feature_name, score))
                    
                    # Sort by importance score
                    feature_scores.sort(key=lambda x: x[1], reverse=True)
                    self.importance_scores[imp_type] = feature_scores
                else:
                    # Use feature indices
                    sorted_scores = sorted(scores.items(), 
                                         key=lambda x: x[1], reverse=True)
                    self.importance_scores[imp_type] = sorted_scores
                
                print(f"✓ Extracted {imp_type} importance scores")
            except Exception as e:
                print(f"✗ Failed to extract {imp_type}: {e}")
        
        # Also get sklearn-style feature importance (uses 'gain' by default)
        try:
            sklearn_importance = self.model.feature_importances_
            if self.feature_names is not None:
                sklearn_scores = list(zip(self.feature_names, sklearn_importance))
                sklearn_scores.sort(key=lambda x: x[1], reverse=True)
                self.importance_scores['sklearn_default'] = sklearn_scores
            print("✓ Extracted sklearn-style feature importance")
        except Exception as e:
            print(f"✗ Failed to extract sklearn importance: {e}")
    
    def print_importance_rankings(self, top_n=10):
        """Print feature importance rankings for all types"""
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE RANKINGS")
        print("="*60)
        
        for imp_type, scores in self.importance_scores.items():
            print(f"\n{imp_type.upper()} IMPORTANCE:")
            print("-" * 40)
            
            for i, (feature, score) in enumerate(scores[:top_n]):
                print(f"{i+1:2d}. {feature:<20} : {score:8.4f}")
    
    def visualize_feature_importance(self, importance_types=None, top_n=15):
        """Create comprehensive visualizations of feature importance"""
        
        if importance_types is None:
            importance_types = list(self.importance_scores.keys())
        
        # Filter to available importance types
        available_types = [t for t in importance_types if t in self.importance_scores]
        
        if not available_types:
            print("No importance scores available for visualization")
            return
        
        # Calculate subplot dimensions
        n_plots = len(available_types)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes = axes.flatten()
        
        for i, imp_type in enumerate(available_types):
            scores = self.importance_scores[imp_type][:top_n]
            features, values = zip(*scores)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            axes[i].barh(y_pos, values, alpha=0.7)
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(features, fontsize=8)
            axes[i].set_xlabel('Importance Score')
            axes[i].set_title(f'{imp_type.upper()} Importance')
            axes[i].grid(True, alpha=0.3)
            
            # Invert y-axis to show highest importance at top
            axes[i].invert_yaxis()
        
        # Hide unused subplots
        for i in range(len(available_types), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

def demonstrate_feature_importance():
    """Main demonstration function"""
    
    print("XGBoost Feature Importance Demonstration")
    print("=" * 45)
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize analyzer
    analyzer = XGBoostFeatureImportanceAnalyzer()
    analyzer.feature_names = feature_names
    
    # Train model
    print("\nTraining XGBoost model...")
    analyzer.model = xgb.XGBClassifier(**analyzer.model_params)
    analyzer.model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = analyzer.model.predict(X_train)
    test_pred = analyzer.model.predict(X_test)
    
    print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    
    # Extract all importance types
    analyzer.extract_all_importance_types()
    
    # Print rankings
    analyzer.print_importance_rankings(top_n=10)
    
    # Create visualizations
    print("\nCreating importance visualizations...")
    analyzer.visualize_feature_importance(top_n=12)
    
    return analyzer

if __name__ == "__main__":
    # Run the main demonstration
    analyzer = demonstrate_feature_importance()
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE TYPES EXPLANATION")
    print("="*60)
    
    explanations = {
        'weight': 'Number of times a feature is used to split data across all trees',
        'gain': 'Average gain of splits which use the feature (most commonly used)',
        'cover': 'Average coverage of splits which use the feature',
        'total_gain': 'Total gain of splits which use the feature',
        'total_cover': 'Total coverage of splits which use the feature'
    }
    
    for imp_type, explanation in explanations.items():
        print(f"{imp_type.upper():<12}: {explanation}")
    
    print("\nRecommendations:")
    print("- Use 'gain' for most applications (default in sklearn)")
    print("- Use 'weight' to understand feature usage frequency")
    print("- Use 'cover' to understand feature's impact on samples")
    print("- Compare multiple methods for robust feature selection")
```

**Key Features:**
1. **All XGBoost Importance Types**: Weight, gain, cover, total_gain, total_cover
2. **Multiple Access Methods**: get_score(), feature_importances_, plot_importance()
3. **Comprehensive Visualization**: Bar plots showing different importance metrics
4. **Feature Ranking**: Clear rankings for all importance types

---

## Question 5 Answer:

**Implement anXGBoost modelon a givendatasetand useSHAP valuesto interpret the model's predictions.**

**Answer:**

Here's a comprehensive implementation using XGBoost with SHAP (SHapley Additive exPlanations) for model interpretation:

```python
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')

class XGBoostSHAPInterpreter:
    """
    Comprehensive XGBoost Model with SHAP Interpretation
    
    This class demonstrates how to train XGBoost models and use SHAP
    for both global and local interpretability.
    """
    
    def __init__(self, task_type='classification', model_params=None):
        """Initialize the interpreter"""
        self.task_type = task_type
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Default parameters based on task type
        if task_type == 'classification':
            self.default_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42
            }
        else:  # regression
            self.default_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42
            }
        
        self.model_params = model_params or self.default_params
    
    def load_dataset(self, dataset_name='synthetic'):
        """Load and prepare dataset"""
        
        print(f"Loading {dataset_name} dataset...")
        
        if dataset_name == 'synthetic':
            if self.task_type == 'classification':
                X, y = make_classification(
                    n_samples=1000,
                    n_features=15,
                    n_informative=10,
                    n_redundant=3,
                    n_clusters_per_class=1,
                    random_state=42
                )
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            else:
                # Create synthetic regression data
                from sklearn.datasets import make_regression
                X, y = make_regression(
                    n_samples=1000,
                    n_features=15,
                    n_informative=10,
                    noise=0.1,
                    random_state=42
                )
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                
        elif dataset_name == 'wine':
            data = load_wine()
            X, y = data.data, data.target
            feature_names = data.feature_names.tolist()
            # Convert to binary classification
            y = (y == 0).astype(int)
            
        elif dataset_name == 'boston':
            # Using a housing dataset for regression
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            X, y = data.data, data.target
            feature_names = data.feature_names.tolist()
            self.task_type = 'regression'  # Override task type
            
        else:
            raise ValueError("Unsupported dataset")
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        self.feature_names = feature_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {len(feature_names)}")
        if self.task_type == 'classification':
            print(f"Target distribution: {np.bincount(y)}")
        else:
            print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        return df
    
    def prepare_data(self, df, test_size=0.2):
        """Prepare data for training"""
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, 
            stratify=y if self.task_type == 'classification' else None
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train XGBoost model"""
        
        print("Training XGBoost model...")
        
        if self.task_type == 'classification':
            self.model = xgb.XGBClassifier(**self.model_params)
        else:
            self.model = xgb.XGBRegressor(**self.model_params)
        
        # Train the model
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate model
        if self.task_type == 'classification':
            train_pred = self.model.predict(self.X_train)
            test_pred = self.model.predict(self.X_test)
            
            train_acc = accuracy_score(self.y_train, train_pred)
            test_acc = accuracy_score(self.y_test, test_pred)
            
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
        else:
            train_pred = self.model.predict(self.X_train)
            test_pred = self.model.predict(self.X_test)
            
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            test_r2 = r2_score(self.y_test, test_pred)
            
            print(f"Training RMSE: {train_rmse:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
            print(f"Test R²: {test_r2:.4f}")
        
        print("✓ Model training completed")
        return self.model
    
    def initialize_shap_explainer(self, explainer_type='tree'):
        """Initialize SHAP explainer"""
        
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print(f"Initializing SHAP {explainer_type} explainer...")
        
        if explainer_type == 'tree':
            # TreeExplainer is most efficient for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'kernel':
            # KernelExplainer is model-agnostic but slower
            background = shap.sample(self.X_train, 100)  # Use 100 samples as background
            self.explainer = shap.KernelExplainer(self.model.predict, background)
        elif explainer_type == 'linear':
            # LinearExplainer for linear models (not applicable for XGBoost)
            raise ValueError("LinearExplainer not suitable for XGBoost")
        else:
            raise ValueError("Unsupported explainer type")
        
        print("✓ SHAP explainer initialized")
        return self.explainer
    
    def calculate_shap_values(self, data=None, max_samples=500):
        """Calculate SHAP values"""
        
        if self.explainer is None:
            raise ValueError("SHAP explainer must be initialized first")
        
        # Use test data by default, but limit samples for efficiency
        if data is None:
            data = self.X_test
        
        # Limit number of samples for performance
        if len(data) > max_samples:
            data = data.sample(n=max_samples, random_state=42)
        
        print(f"Calculating SHAP values for {len(data)} samples...")
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(data)
        
        # For classification, shap_values might be a list
        if isinstance(self.shap_values, list):
            if self.task_type == 'classification' and len(self.shap_values) == 2:
                # Binary classification - use positive class
                self.shap_values = self.shap_values[1]
        
        print("✓ SHAP values calculated")
        return self.shap_values
    
    def plot_shap_summary(self, plot_type='dot'):
        """Create SHAP summary plots"""
        
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first")
        
        print(f"Creating SHAP summary plot ({plot_type})...")
        
        # Use the data that was used for SHAP calculation
        data = self.X_test.iloc[:len(self.shap_values)]
        
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'dot':
            shap.summary_plot(
                self.shap_values, 
                data, 
                feature_names=self.feature_names,
                show=False
            )
        elif plot_type == 'bar':
            shap.summary_plot(
                self.shap_values, 
                data, 
                feature_names=self.feature_names,
                plot_type='bar',
                show=False
            )
        elif plot_type == 'violin':
            shap.summary_plot(
                self.shap_values, 
                data, 
                feature_names=self.feature_names,
                plot_type='violin',
                show=False
            )
        
        plt.title(f'SHAP Summary Plot ({plot_type.title()})')
        plt.tight_layout()
        plt.show()
    
    def plot_shap_waterfall(self, sample_idx=0):
        """Create SHAP waterfall plot for individual prediction"""
        
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first")
        
        print(f"Creating SHAP waterfall plot for sample {sample_idx}...")
        
        # Use the data that was used for SHAP calculation
        data = self.X_test.iloc[:len(self.shap_values)]
        
        plt.figure(figsize=(10, 8))
        
        # Create waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=data.iloc[sample_idx].values,
                feature_names=self.feature_names
            ),
            show=False
        )
        
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}')
        plt.tight_layout()
        plt.show()
    
    def plot_shap_force(self, sample_idx=0):
        """Create SHAP force plot for individual prediction"""
        
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first")
        
        print(f"Creating SHAP force plot for sample {sample_idx}...")
        
        # Use the data that was used for SHAP calculation
        data = self.X_test.iloc[:len(self.shap_values)]
        
        # Create force plot
        shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[sample_idx],
            data.iloc[sample_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.title(f'SHAP Force Plot - Sample {sample_idx}')
        plt.tight_layout()
        plt.show()
    
    def plot_shap_dependence(self, feature_idx=0, interaction_feature='auto'):
        """Create SHAP dependence plot"""
        
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first")
        
        feature_name = self.feature_names[feature_idx]
        print(f"Creating SHAP dependence plot for {feature_name}...")
        
        # Use the data that was used for SHAP calculation
        data = self.X_test.iloc[:len(self.shap_values)]
        
        plt.figure(figsize=(10, 6))
        
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            data,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.title(f'SHAP Dependence Plot - {feature_name}')
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_interactions(self, top_n=5):
        """Analyze feature interactions using SHAP"""
        
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first")
        
        print("Analyzing feature interactions...")
        
        # Calculate mean absolute SHAP values for feature importance
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features (by mean |SHAP|):")
        print("-" * 50)
        for i, row in feature_importance.head(top_n).iterrows():
            print(f"{row['feature']:<20}: {row['importance']:.4f}")
        
        # Plot dependence plots for top features
        top_features = feature_importance.head(top_n)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        data = self.X_test.iloc[:len(self.shap_values)]
        
        for i, (_, row) in enumerate(top_features.iterrows()):
            if i >= 6:  # Limit to 6 plots
                break
                
            feature_name = row['feature']
            feature_idx = self.feature_names.index(feature_name)
            
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                data,
                feature_names=self.feature_names,
                ax=axes[i],
                show=False
            )
            axes[i].set_title(f'{feature_name}')
        
        # Hide unused subplots
        for i in range(len(top_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('SHAP Dependence Plots - Top Features')
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def generate_interpretation_report(self):
        """Generate a comprehensive interpretation report"""
        
        print("\n" + "="*60)
        print("XGBOOST MODEL INTERPRETATION REPORT")
        print("="*60)
        
        if self.model is None:
            print("Error: Model not trained")
            return
        
        # Model performance
        print("\n1. MODEL PERFORMANCE:")
        print("-" * 25)
        
        if self.task_type == 'classification':
            test_pred = self.model.predict(self.X_test)
            test_acc = accuracy_score(self.y_test, test_pred)
            print(f"Test Accuracy: {test_acc:.4f}")
        else:
            test_pred = self.model.predict(self.X_test)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            test_r2 = r2_score(self.y_test, test_pred)
            print(f"Test RMSE: {test_rmse:.4f}")
            print(f"Test R²: {test_r2:.4f}")
        
        # XGBoost feature importance
        print("\n2. XGBOOST FEATURE IMPORTANCE:")
        print("-" * 35)
        
        xgb_importance = self.model.feature_importances_
        xgb_features = pd.DataFrame({
            'feature': self.feature_names,
            'xgb_importance': xgb_importance
        }).sort_values('xgb_importance', ascending=False)
        
        print("Top 10 features (XGBoost gain):")
        for i, row in xgb_features.head(10).iterrows():
            print(f"  {row['feature']:<20}: {row['xgb_importance']:.4f}")
        
        # SHAP analysis
        if self.shap_values is not None:
            print("\n3. SHAP ANALYSIS:")
            print("-" * 18)
            
            # Global feature importance
            mean_shap = np.abs(self.shap_values).mean(axis=0)
            shap_features = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': mean_shap
            }).sort_values('shap_importance', ascending=False)
            
            print("Top 10 features (mean |SHAP|):")
            for i, row in shap_features.head(10).iterrows():
                print(f"  {row['feature']:<20}: {row['shap_importance']:.4f}")
            
            # Feature correlation analysis
            print("\n4. FEATURE IMPORTANCE CORRELATION:")
            print("-" * 38)
            
            merged_importance = pd.merge(xgb_features, shap_features, on='feature')
            correlation = merged_importance['xgb_importance'].corr(
                merged_importance['shap_importance']
            )
            print(f"XGBoost vs SHAP importance correlation: {correlation:.4f}")
            
        else:
            print("\n3. SHAP ANALYSIS: Not available (SHAP values not calculated)")

def demonstrate_xgboost_shap():
    """Main demonstration function"""
    
    print("XGBoost with SHAP Interpretation Demonstration")
    print("=" * 50)
    
    # Test both classification and regression
    tasks = [
        ('classification', 'synthetic'),
        ('regression', 'boston')
    ]
    
    for task_type, dataset_name in tasks:
        print(f"\n{'='*70}")
        print(f"DEMONSTRATING {task_type.upper()} TASK WITH {dataset_name.upper()} DATASET")
        print('='*70)
        
        try:
            # Initialize interpreter
            interpreter = XGBoostSHAPInterpreter(task_type=task_type)
            
            # Load and prepare data
            df = interpreter.load_dataset(dataset_name)
            X_train, X_test, y_train, y_test = interpreter.prepare_data(df)
            
            # Train model
            model = interpreter.train_model()
            
            # Initialize SHAP explainer
            explainer = interpreter.initialize_shap_explainer('tree')
            
            # Calculate SHAP values
            shap_values = interpreter.calculate_shap_values(max_samples=200)
            
            # Create visualizations
            print("\nCreating SHAP visualizations...")
            
            # Summary plots
            interpreter.plot_shap_summary('dot')
            interpreter.plot_shap_summary('bar')
            
            # Individual prediction explanations
            interpreter.plot_shap_waterfall(sample_idx=0)
            interpreter.plot_shap_force(sample_idx=0)
            
            # Feature analysis
            feature_importance = interpreter.analyze_feature_interactions(top_n=5)
            
            # Generate report
            interpreter.generate_interpretation_report()
            
        except Exception as e:
            print(f"Error in {task_type} demonstration: {e}")
            continue

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_xgboost_shap()
    
    print("\n" + "="*60)
    print("SHAP INTERPRETATION METHODS SUMMARY")
    print("="*60)
    
    methods = {
        'Summary Plot (Dot)': 'Shows feature importance and impact distribution',
        'Summary Plot (Bar)': 'Shows average feature importance',
        'Waterfall Plot': 'Shows individual prediction breakdown',
        'Force Plot': 'Interactive individual prediction explanation',
        'Dependence Plot': 'Shows feature effects and interactions',
        'Partial Dependence': 'Shows marginal effect of features'
    }
    
    for method, description in methods.items():
        print(f"{method:<20}: {description}")
    
    print("\nKey Benefits of SHAP:")
    print("- Model-agnostic explanations")
    print("- Both global and local interpretability")
    print("- Mathematically grounded in game theory")
    print("- Consistent and efficient explanations")
    print("- Rich visualization ecosystem")
```

**Key Features:**
1. **Complete SHAP Integration**: TreeExplainer, KernelExplainer support
2. **Multiple Visualization Types**: Summary, waterfall, force, dependence plots
3. **Both Tasks**: Classification and regression examples
4. **Feature Interaction Analysis**: Identifies important features and interactions
5. **Comprehensive Reporting**: Combines XGBoost and SHAP insights

**SHAP Explanation Types:**
- **Global**: Overall feature importance across all predictions
- **Local**: Individual prediction explanations
- **Interactions**: How features interact with each other
