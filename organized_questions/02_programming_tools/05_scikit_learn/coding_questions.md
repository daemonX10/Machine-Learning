# Scikit Learn Interview Questions - Coding Questions

## Question 1

**Describe the k-means clustering process as implemented in Scikit-Learn.**

### Theory
K-means clustering is an unsupervised learning algorithm that partitions data into k clusters by minimizing the within-cluster sum of squares (WCSS). Scikit-Learn implements the Lloyd's algorithm with k-means++ initialization for optimal performance.

### Code Example
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, 
                       random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, 
                n_init=10, max_iter=300)
y_pred = kmeans.fit_predict(X_scaled)

# Get cluster centers
centers = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"Cluster centers:\n{centers}")
print(f"Inertia (WCSS): {inertia:.2f}")

# Visualize results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
plt.title('True Clusters')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200)
plt.title('K-means Clustering')
plt.show()
```

### Explanation
1. **Initialization**: K-means++ selects initial centroids to speed up convergence
2. **Assignment**: Each point is assigned to the nearest centroid using Euclidean distance
3. **Update**: Centroids are recalculated as the mean of assigned points
4. **Iteration**: Steps 2-3 repeat until convergence or max_iter is reached
5. **Output**: Final cluster assignments and centroids

### Use Cases
- Customer segmentation in marketing
- Image compression and segmentation
- Data preprocessing for other algorithms
- Market research and demographic analysis

### Best Practices
- Always standardize data before clustering
- Use the elbow method to determine optimal k
- Set random_state for reproducible results
- Consider multiple initializations (n_init parameter)

### Pitfalls
- Assumes spherical clusters of similar size
- Sensitive to outliers and initialization
- Requires pre-specification of k
- Poor performance on non-globular clusters

### Optimization
```python
# Elbow method for optimal k
from sklearn.metrics import silhouette_score

def find_optimal_k(X, max_k=10):
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    return inertias, silhouette_scores

# Find optimal k
inertias, sil_scores = find_optimal_k(X_scaled)
optimal_k = np.argmax(sil_scores) + 2
print(f"Optimal k based on silhouette score: {optimal_k}")
```

### Debugging
- Check data scaling and outliers if clusters seem unbalanced
- Increase max_iter if algorithm doesn't converge
- Use different initialization methods if results are unstable
- Validate cluster quality using silhouette analysis

---

## Question 2

**How does Scikit-Learn implement logistic regression differently from linear regression?**

### Theory
Logistic regression uses the logistic (sigmoid) function to model probabilities for classification, while linear regression models continuous outcomes. Scikit-Learn implements logistic regression using different solvers and regularization techniques optimized for classification tasks.

### Code Example
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler

# Generate classification data for logistic regression
X_class, y_class = make_classification(n_samples=1000, n_features=2, 
                                      n_redundant=0, n_informative=2,
                                      random_state=42, n_clusters_per_class=1)

# Generate regression data for linear regression
X_reg, y_reg = make_regression(n_samples=1000, n_features=1, noise=10, 
                              random_state=42)

# Split datasets
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# Standardize features
scaler_class = StandardScaler()
X_class_train_scaled = scaler_class.fit_transform(X_class_train)
X_class_test_scaled = scaler_class.transform(X_class_test)

scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

# Logistic Regression
log_reg = LogisticRegression(solver='liblinear', random_state=42)
log_reg.fit(X_class_train_scaled, y_class_train)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_reg_train_scaled, y_reg_train)

# Predictions
y_class_pred = log_reg.predict(X_class_test_scaled)
y_class_proba = log_reg.predict_proba(X_class_test_scaled)

y_reg_pred = lin_reg.predict(X_reg_test_scaled)

# Evaluation
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_class_test, y_class_pred):.3f}")
print(f"Coefficients: {log_reg.coef_[0]}")
print(f"Intercept: {log_reg.intercept_[0]:.3f}")
print("\nClassification Report:")
print(classification_report(y_class_test, y_class_pred))

print("\nLinear Regression Results:")
print(f"MSE: {mean_squared_error(y_reg_test, y_reg_pred):.3f}")
print(f"Coefficients: {lin_reg.coef_}")
print(f"Intercept: {lin_reg.intercept_:.3f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Logistic regression plot
axes[0].scatter(X_class_test[:, 0], X_class_test[:, 1], c=y_class_test, 
               cmap='viridis', alpha=0.6, label='True')
axes[0].scatter(X_class_test[:, 0], X_class_test[:, 1], c=y_class_pred, 
               cmap='viridis', marker='x', s=50, label='Predicted')
axes[0].set_title('Logistic Regression Classification')
axes[0].legend()

# Linear regression plot
axes[1].scatter(X_reg_test, y_reg_test, alpha=0.6, label='True')
axes[1].plot(X_reg_test, y_reg_pred, color='red', label='Predicted')
axes[1].set_title('Linear Regression')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Explanation
**Key Differences:**

1. **Output Function**: 
   - Linear: f(x) = wx + b (continuous)
   - Logistic: f(x) = 1/(1 + e^-(wx + b)) (probability)

2. **Cost Function**:
   - Linear: Mean Squared Error
   - Logistic: Log-likelihood (cross-entropy)

3. **Optimization**:
   - Linear: Closed-form solution or gradient descent
   - Logistic: Iterative methods (Newton-Raphson, LBFGS, etc.)

4. **Solvers in Scikit-Learn**:
   - `liblinear`: Good for small datasets
   - `lbfgs`: Default for small datasets
   - `sag`/`saga`: For large datasets
   - `newton-cg`: For multinomial problems

### Use Cases
**Logistic Regression:**
- Binary/multiclass classification
- Probability estimation
- Medical diagnosis
- Marketing response prediction

**Linear Regression:**
- Price prediction
- Sales forecasting
- Risk assessment
- Continuous outcome modeling

### Best Practices
- Always scale features for logistic regression
- Choose appropriate solver based on dataset size
- Use regularization (L1/L2) to prevent overfitting
- Check class balance for classification problems

### Pitfalls
- Logistic regression assumes linear relationship between features and log-odds
- Sensitive to outliers (though less than linear regression)
- Can struggle with complex non-linear relationships
- Multicollinearity affects coefficient interpretation

### Debugging
```python
# Check convergence
log_reg_verbose = LogisticRegression(solver='lbfgs', max_iter=1000, 
                                   verbose=1, random_state=42)
log_reg_verbose.fit(X_class_train_scaled, y_class_train)

# Check for class imbalance
from collections import Counter
print("Class distribution:", Counter(y_class_train))

# Feature importance (coefficient magnitude)
feature_importance = np.abs(log_reg.coef_[0])
print("Feature importance:", feature_importance)
```

### Optimization
```python
# Regularized logistic regression with different penalties
from sklearn.linear_model import LogisticRegressionCV

# Cross-validated regularization
log_reg_cv = LogisticRegressionCV(cv=5, penalty='elasticnet', 
                                 solver='saga', l1_ratios=[0.1, 0.5, 0.9],
                                 random_state=42, max_iter=1000)
log_reg_cv.fit(X_class_train_scaled, y_class_train)

print(f"Best C: {log_reg_cv.C_[0]:.3f}")
print(f"Best l1_ratio: {log_reg_cv.l1_ratio_[0]:.3f}")
```

---

## Question 3

**Write a Python script using Scikit-Learn to train and evaluate a logistic regression model.**

### Theory
A complete machine learning pipeline includes data loading, preprocessing, model training, evaluation, and interpretation. This example demonstrates best practices for building a robust logistic regression classifier with proper validation and metrics.

### Code Example
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class LogisticRegressionPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None
        self.model = None
        self.scaler = None
        
    def load_and_prepare_data(self):
        """Load and prepare the breast cancer dataset"""
        # Load dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        
        # Create DataFrame for easier handling
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        print("Dataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Target distribution:\n{df['target'].value_counts()}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        return X, y, feature_names
    
    def create_pipeline(self):
        """Create preprocessing and modeling pipeline"""
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(
                solver='liblinear',
                random_state=self.random_state,
                max_iter=1000
            ))
        ])
        return self.pipeline
    
    def train_and_evaluate(self, X, y):
        """Train model and perform comprehensive evaluation"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, 
            stratify=y
        )
        
        # Train pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Store trained components
        self.model = self.pipeline.named_steps['logistic']
        self.scaler = self.pipeline.named_steps['scaler']
        
        return X_train, X_test, y_train, y_test, y_pred, y_pred_proba
    
    def evaluate_model(self, y_test, y_pred, y_pred_proba):
        """Comprehensive model evaluation"""
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("Model Performance Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def cross_validation(self, X, y):
        """Perform cross-validation"""
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5, 
                                   scoring='roc_auc')
        
        print(f"\nCross-Validation ROC-AUC Scores:")
        print(f"Scores: {cv_scores}")
        print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def plot_results(self, X_test, y_test, y_pred, y_pred_proba, feature_names):
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        
        # Feature Importance (Coefficients)
        coef = self.model.coef_[0]
        indices = np.argsort(np.abs(coef))[-10:]  # Top 10 features
        axes[1,0].barh(range(len(indices)), coef[indices])
        axes[1,0].set_yticks(range(len(indices)))
        axes[1,0].set_yticklabels([feature_names[i] for i in indices])
        axes[1,0].set_title('Top 10 Feature Coefficients')
        axes[1,0].set_xlabel('Coefficient Value')
        
        # Prediction Probability Distribution
        axes[1,1].hist(y_pred_proba[y_test == 0], alpha=0.5, label='Benign', bins=20)
        axes[1,1].hist(y_pred_proba[y_test == 1], alpha=0.5, label='Malignant', bins=20)
        axes[1,1].set_xlabel('Prediction Probability')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Prediction Probability Distribution')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def learning_curve_analysis(self, X, y):
        """Plot learning curves to assess model performance"""
        train_sizes, train_scores, val_scores = learning_curve(
            self.pipeline, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc'
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', 
                label='Training Score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', 
                label='Validation Score')
        plt.fill_between(train_sizes, 
                        np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                        alpha=0.1)
        plt.fill_between(train_sizes,
                        np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                        alpha=0.1)
        plt.xlabel('Training Set Size')
        plt.ylabel('ROC-AUC Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    """Main execution function"""
    # Initialize pipeline
    lr_pipeline = LogisticRegressionPipeline(random_state=42)
    
    # Load and prepare data
    X, y, feature_names = lr_pipeline.load_and_prepare_data()
    
    # Create pipeline
    pipeline = lr_pipeline.create_pipeline()
    
    # Train and evaluate
    X_train, X_test, y_train, y_test, y_pred, y_pred_proba = lr_pipeline.train_and_evaluate(X, y)
    
    # Evaluate model
    metrics = lr_pipeline.evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Cross-validation
    cv_scores = lr_pipeline.cross_validation(X, y)
    
    # Visualizations
    lr_pipeline.plot_results(X_test, y_test, y_pred, y_pred_proba, feature_names)
    
    # Learning curves
    lr_pipeline.learning_curve_analysis(X, y)
    
    return lr_pipeline, metrics

# Execute the pipeline
if __name__ == "__main__":
    pipeline, results = main()
```

### Explanation
1. **Data Loading**: Uses breast cancer dataset with proper exploration
2. **Preprocessing**: StandardScaler for feature normalization
3. **Pipeline Creation**: Combines preprocessing and modeling
4. **Training**: Stratified split to maintain class balance
5. **Evaluation**: Comprehensive metrics including ROC-AUC, precision, recall
6. **Visualization**: Confusion matrix, ROC curve, feature importance
7. **Cross-Validation**: 5-fold CV for robust performance estimation

### Use Cases
- Medical diagnosis and healthcare applications
- Binary classification problems
- Interpretable machine learning models
- Baseline model for comparison

### Best Practices
- Use stratified splitting for imbalanced datasets
- Implement proper cross-validation
- Create reusable pipeline classes
- Include comprehensive evaluation metrics
- Visualize results for better understanding

### Pitfalls
- Not scaling features can lead to poor convergence
- Ignoring class imbalance affects model performance
- Overfitting on small datasets
- Not validating assumptions of logistic regression

### Debugging
```python
# Check for convergence issues
pipeline_debug = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(
        solver='lbfgs', max_iter=2000, verbose=1
    ))
])

# Monitor training progress
print("Fitting with verbose output...")
pipeline_debug.fit(X_train, y_train)
```

### Optimization
```python
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning
param_grid = {
    'logistic__C': [0.01, 0.1, 1, 10, 100],
    'logistic__penalty': ['l1', 'l2', 'elasticnet'],
    'logistic__solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, 
    scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
```

---

## Question 4

**Create a Python function that uses Scikit-Learn to perform a k-fold cross-validation on a dataset.**

### Theory
K-fold cross-validation divides the dataset into k equally-sized folds, using k-1 folds for training and 1 fold for validation, repeating this process k times. This provides a more robust estimate of model performance by using all data for both training and validation.

### Code Example
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (cross_val_score, cross_validate, 
                                   StratifiedKFold, KFold, LeaveOneOut)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, make_scorer)
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class CrossValidationAnalyzer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
    
    def basic_cv_function(self, model, X, y, cv=5, scoring='accuracy'):
        """
        Basic k-fold cross-validation function
        
        Parameters:
        -----------
        model : sklearn estimator
            The machine learning model to evaluate
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        cv : int, default=5
            Number of folds
        scoring : str, default='accuracy'
            Scoring metric
            
        Returns:
        --------
        dict : Cross-validation results
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
        
        print(f"Cross-Validation Results ({scoring}):")
        print(f"Scores: {scores}")
        print(f"Mean: {results['mean']:.4f} (+/- {results['std'] * 2:.4f})")
        print(f"Range: [{results['min']:.4f}, {results['max']:.4f}]")
        
        return results
    
    def comprehensive_cv(self, models, X, y, cv=5, scoring_metrics=None):
        """
        Comprehensive cross-validation with multiple models and metrics
        
        Parameters:
        -----------
        models : dict
            Dictionary of model_name: model_instance
        X : array-like
            Training data
        y : array-like
            Target values
        cv : int or cross-validation generator
            Cross-validation strategy
        scoring_metrics : list
            List of scoring metrics to evaluate
            
        Returns:
        --------
        pandas.DataFrame : Results for all models and metrics
        """
        if scoring_metrics is None:
            # Determine if classification or regression
            unique_values = np.unique(y)
            if len(unique_values) <= 10 and all(isinstance(x, (int, np.integer)) for x in unique_values):
                scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            else:
                scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        results = []
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Perform cross-validation with multiple metrics
            cv_results = cross_validate(
                model, X, y, cv=cv, 
                scoring=scoring_metrics,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Store results for each metric
            for metric in scoring_metrics:
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                results.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Test_Mean': test_scores.mean(),
                    'Test_Std': test_scores.std(),
                    'Train_Mean': train_scores.mean(),
                    'Train_Std': train_scores.std(),
                    'Overfitting': train_scores.mean() - test_scores.mean()
                })
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def stratified_cv_classification(self, model, X, y, n_splits=5):
        """
        Stratified k-fold cross-validation for classification
        Maintains class distribution in each fold
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                             random_state=self.random_state)
        
        # Multiple scoring metrics for classification
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_macro',
            'recall': 'recall_macro',
            'f1': 'f1_macro',
            'roc_auc': 'roc_auc_ovr_weighted'
        }
        
        cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)
        
        print("Stratified Cross-Validation Results:")
        print("-" * 40)
        for metric in scoring.keys():
            scores = cv_results[f'test_{metric}']
            print(f"{metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def nested_cv(self, model, param_grid, X, y, inner_cv=3, outer_cv=5):
        """
        Nested cross-validation for unbiased performance estimation
        """
        from sklearn.model_selection import GridSearchCV
        
        outer_scores = []
        outer_fold = KFold(n_splits=outer_cv, shuffle=True, 
                          random_state=self.random_state)
        
        for train_idx, test_idx in outer_fold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner loop: hyperparameter tuning
            grid_search = GridSearchCV(
                model, param_grid, cv=inner_cv, 
                scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Outer loop: performance estimation
            best_model = grid_search.best_estimator_
            score = best_model.score(X_test, y_test)
            outer_scores.append(score)
        
        print(f"Nested CV Results:")
        print(f"Scores: {outer_scores}")
        print(f"Mean: {np.mean(outer_scores):.4f} (+/- {np.std(outer_scores) * 2:.4f})")
        
        return outer_scores
    
    def time_series_cv(self, model, X, y, n_splits=5):
        """
        Time series cross-validation with expanding window
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        print(f"Time Series CV Results:")
        print(f"Scores: {scores}")
        print(f"Mean: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
        
        return scores
    
    def learning_curve_cv(self, model, X, y, cv=5):
        """
        Learning curve with cross-validation
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv, n_jobs=-1
        )
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, np.mean(train_scores, axis=1), 'o-', 
                label='Training Score')
        plt.plot(train_sizes_abs, np.mean(val_scores, axis=1), 'o-', 
                label='Validation Score')
        
        # Add error bands
        plt.fill_between(train_sizes_abs,
                        np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                        alpha=0.1)
        plt.fill_between(train_sizes_abs,
                        np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                        alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curves with Cross-Validation')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return train_sizes_abs, train_scores, val_scores

def demonstration():
    """Demonstrate different cross-validation techniques"""
    
    # Initialize analyzer
    cv_analyzer = CrossValidationAnalyzer(random_state=42)
    
    # Load datasets
    print("Loading datasets...")
    breast_cancer = load_breast_cancer()
    X_class, y_class = breast_cancer.data, breast_cancer.target
    
    diabetes = load_diabetes()
    X_reg, y_reg = diabetes.data, diabetes.target
    
    # Classification models
    classification_models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42))
        ]),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(random_state=42))
        ])
    }
    
    # Regression models
    regression_models = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', LinearRegression())
        ]),
        'Random Forest Reg': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', SVR())
        ])
    }
    
    print("\n" + "="*50)
    print("CLASSIFICATION CROSS-VALIDATION")
    print("="*50)
    
    # Basic CV for classification
    print("\n1. Basic Cross-Validation:")
    cv_analyzer.basic_cv_function(
        classification_models['Logistic Regression'], 
        X_class, y_class, cv=5, scoring='accuracy'
    )
    
    # Comprehensive CV for classification
    print("\n2. Comprehensive Cross-Validation:")
    class_results = cv_analyzer.comprehensive_cv(
        classification_models, X_class, y_class, cv=5
    )
    print("\nResults Summary:")
    print(class_results.pivot(index='Model', columns='Metric', values='Test_Mean'))
    
    # Stratified CV
    print("\n3. Stratified Cross-Validation:")
    cv_analyzer.stratified_cv_classification(
        classification_models['Logistic Regression'],
        X_class, y_class, n_splits=5
    )
    
    print("\n" + "="*50)
    print("REGRESSION CROSS-VALIDATION")
    print("="*50)
    
    # Basic CV for regression
    print("\n1. Basic Cross-Validation:")
    cv_analyzer.basic_cv_function(
        regression_models['Linear Regression'], 
        X_reg, y_reg, cv=5, scoring='r2'
    )
    
    # Comprehensive CV for regression
    print("\n2. Comprehensive Cross-Validation:")
    reg_results = cv_analyzer.comprehensive_cv(
        regression_models, X_reg, y_reg, cv=5
    )
    print("\nResults Summary:")
    print(reg_results.pivot(index='Model', columns='Metric', values='Test_Mean'))
    
    # Learning curves
    print("\n3. Learning Curves:")
    cv_analyzer.learning_curve_cv(
        regression_models['Linear Regression'],
        X_reg, y_reg, cv=5
    )
    
    return cv_analyzer, class_results, reg_results

# Run demonstration
if __name__ == "__main__":
    analyzer, class_df, reg_df = demonstration()
```

### Explanation
1. **Basic CV**: Simple k-fold validation with single metric
2. **Comprehensive CV**: Multiple models and metrics comparison
3. **Stratified CV**: Maintains class distribution for classification
4. **Nested CV**: Unbiased performance estimation with hyperparameter tuning
5. **Time Series CV**: Temporal data validation with expanding window
6. **Learning Curves**: Performance vs training set size analysis

### Use Cases
- Model selection and comparison
- Performance estimation and validation
- Hyperparameter tuning validation
- Detecting overfitting and underfitting
- Robust model evaluation

### Best Practices
- Use stratified CV for classification with imbalanced classes
- Choose appropriate CV strategy based on data characteristics
- Use nested CV for unbiased hyperparameter tuning
- Consider computational cost vs. validation quality trade-off
- Always set random_state for reproducibility

### Pitfalls
- Data leakage between folds (e.g., temporal dependencies)
- Not accounting for class imbalance in classification
- Using too few folds (high variance) or too many folds (high bias)
- Ignoring computational complexity with large datasets

### Debugging
```python
# Check fold sizes and class distribution
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_class, y_class)):
    print(f"Fold {fold+1}:")
    print(f"  Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    print(f"  Train class dist: {np.bincount(y_class[train_idx])}")
    print(f"  Val class dist: {np.bincount(y_class[val_idx])}")
```

### Optimization
```python
# Parallel processing for faster CV
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y, cv=5, 
    scoring='accuracy', 
    n_jobs=-1  # Use all available cores
)

# Custom scoring function
def custom_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

custom_scorer = make_scorer(custom_f1_score)
scores = cross_val_score(model, X, y, cv=5, scoring=custom_scorer)
```

---

## Question 5

**Implementfeature extraction from textusingScikit-Learn'sCountVectorizerorTfidfVectorizer.**

**Answer:** 

### Text Feature Extraction with CountVectorizer and TfidfVectorizer

Text feature extraction transforms unstructured text into numerical features that machine learning models can process. Scikit-learn provides two primary tools: CountVectorizer for simple word counts and TfidfVectorizer for weighted importance scoring.

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Sample text data for demonstration
documents = [
    "Machine learning is fascinating and powerful",
    "Deep learning requires large datasets",
    "Natural language processing transforms text",
    "Computer vision analyzes images effectively",
    "Data science combines statistics and programming",
    "Machine learning algorithms learn from data",
    "Text mining extracts insights from documents",
    "Artificial intelligence automates complex tasks"
]

labels = [0, 1, 1, 1, 0, 0, 1, 1]  # 0: general ML, 1: specialized ML

print("=== TEXT FEATURE EXTRACTION DEMONSTRATION ===\n")

# 1. CountVectorizer - Simple Word Counts
print("1. COUNTVECTORIZER ANALYSIS")
print("-" * 40)

# Basic CountVectorizer
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)

print(f"Vocabulary size: {len(count_vectorizer.vocabulary_)}")
print(f"Feature matrix shape: {count_matrix.shape}")
print(f"Matrix sparsity: {1 - count_matrix.nnz / np.prod(count_matrix.shape):.2%}")

# Display vocabulary
vocab = count_vectorizer.get_feature_names_out()
print(f"\nVocabulary (first 10 terms): {vocab[:10]}")

# Show count matrix for first document
first_doc_counts = count_matrix[0].toarray().flatten()
word_counts = [(word, count) for word, count in zip(vocab, first_doc_counts) if count > 0]
print(f"\nFirst document word counts: {word_counts}")

# 2. Advanced CountVectorizer with preprocessing
print("\n2. ADVANCED COUNTVECTORIZER")
print("-" * 40)

advanced_count = CountVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),  # Include bigrams
    min_df=2,           # Ignore terms appearing in fewer than 2 documents
    max_df=0.8,         # Ignore terms appearing in more than 80% of documents
    max_features=100    # Keep only top 100 features
)

advanced_count_matrix = advanced_count.fit_transform(documents)
print(f"Advanced vocabulary size: {len(advanced_count.vocabulary_)}")
print(f"Feature matrix shape: {advanced_count_matrix.shape}")

# Show n-grams
advanced_vocab = advanced_count.get_feature_names_out()
bigrams = [term for term in advanced_vocab if ' ' in term]
print(f"Bigrams found: {bigrams}")

# 3. TfidfVectorizer - Weighted Importance
print("\n3. TFIDFVECTORIZER ANALYSIS")
print("-" * 40)

tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.9
)

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
print(f"Feature matrix shape: {tfidf_matrix.shape}")

# Compare TF-IDF scores for first document
tfidf_vocab = tfidf_vectorizer.get_feature_names_out()
first_doc_tfidf = tfidf_matrix[0].toarray().flatten()
word_tfidf = [(word, score) for word, score in zip(tfidf_vocab, first_doc_tfidf) if score > 0]
word_tfidf.sort(key=lambda x: x[1], reverse=True)
print(f"\nFirst document TF-IDF scores (top 5): {word_tfidf[:5]}")

# 4. Custom Text Preprocessing
print("\n4. CUSTOM TEXT PREPROCESSING")
print("-" * 40)

def custom_preprocessor(text):
    """Custom text preprocessing function"""
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def custom_tokenizer(text):
    """Custom tokenization function"""
    # Simple word tokenization
    return text.split()

custom_vectorizer = TfidfVectorizer(
    preprocessor=custom_preprocessor,
    tokenizer=custom_tokenizer,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=1
)

custom_matrix = custom_vectorizer.fit_transform(documents)
print(f"Custom vectorizer vocabulary size: {len(custom_vectorizer.vocabulary_)}")

# 5. Feature Extraction Pipeline for Classification
print("\n5. CLASSIFICATION PIPELINE")
print("-" * 40)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    documents, labels, test_size=0.3, random_state=42, stratify=labels
)

# Create pipelines with different vectorizers
pipelines = {
    'CountVectorizer + NB': Pipeline([
        ('vectorizer', CountVectorizer(stop_words='english')),
        ('classifier', MultinomialNB())
    ]),
    'TfidfVectorizer + NB': Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english')),
        ('classifier', MultinomialNB())
    ]),
    'TfidfVectorizer + LogReg': Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42))
    ])
}

# Train and evaluate pipelines
results = {}
for name, pipeline in pipelines.items():
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Store results
    results[name] = {
        'predictions': y_pred,
        'pipeline': pipeline
    }
    
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))

# 6. Feature Importance Analysis
print("\n6. FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

# Get feature names and coefficients from logistic regression
logreg_pipeline = results['TfidfVectorizer + LogReg']['pipeline']
vectorizer = logreg_pipeline.named_steps['vectorizer']
classifier = logreg_pipeline.named_steps['classifier']

feature_names = vectorizer.get_feature_names_out()
coefficients = classifier.coef_[0]

# Get top positive and negative features
feature_importance = list(zip(feature_names, coefficients))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

print("Top 10 most important features:")
for feature, coef in feature_importance[:10]:
    print(f"  {feature}: {coef:.4f}")

# 7. Vocabulary Analysis
print("\n7. VOCABULARY ANALYSIS")
print("-" * 40)

# Analyze vocabulary distribution
doc_freq = np.array(tfidf_matrix.sum(axis=0)).flatten()
vocab_df = pd.DataFrame({
    'term': tfidf_vocab,
    'document_frequency': doc_freq
})
vocab_df = vocab_df.sort_values('document_frequency', ascending=False)

print("Top 10 terms by document frequency:")
print(vocab_df.head(10))

# 8. Demonstration of Different Parameter Effects
print("\n8. PARAMETER COMPARISON")
print("-" * 40)

vectorizer_configs = [
    ('Basic', {'stop_words': None, 'ngram_range': (1, 1)}),
    ('Stop Words', {'stop_words': 'english', 'ngram_range': (1, 1)}),
    ('Bigrams', {'stop_words': 'english', 'ngram_range': (1, 2)}),
    ('Min DF=2', {'stop_words': 'english', 'ngram_range': (1, 2), 'min_df': 2})
]

for name, params in vectorizer_configs:
    vec = TfidfVectorizer(**params)
    matrix = vec.fit_transform(documents)
    print(f"{name}: vocabulary size = {len(vec.vocabulary_)}, shape = {matrix.shape}")

# 9. Practical Implementation Tips
print("\n9. IMPLEMENTATION BEST PRACTICES")
print("-" * 40)

class TextFeatureExtractor:
    """Production-ready text feature extraction class"""
    
    def __init__(self, vectorizer_type='tfidf', **kwargs):
        """
        Initialize text feature extractor
        
        Parameters:
        - vectorizer_type: 'count' or 'tfidf'
        - kwargs: parameters for vectorizer
        """
        default_params = {
            'lowercase': True,
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.8,
            'max_features': 1000
        }
        default_params.update(kwargs)
        
        if vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(**default_params)
        else:
            self.vectorizer = TfidfVectorizer(**default_params)
        
        self.is_fitted = False
    
    def fit(self, documents):
        """Fit vectorizer to documents"""
        self.vectorizer.fit(documents)
        self.is_fitted = True
        return self
    
    def transform(self, documents):
        """Transform documents to feature matrix"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        return self.vectorizer.transform(documents)
    
    def fit_transform(self, documents):
        """Fit and transform in one step"""
        return self.vectorizer.fit_transform(documents)
    
    def get_feature_names(self):
        """Get feature names"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        return self.vectorizer.get_feature_names_out()
    
    def get_vocabulary_size(self):
        """Get vocabulary size"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        return len(self.vectorizer.vocabulary_)

# Demonstrate the class
extractor = TextFeatureExtractor(vectorizer_type='tfidf', max_features=50)
feature_matrix = extractor.fit_transform(documents)

print(f"Custom extractor vocabulary size: {extractor.get_vocabulary_size()}")
print(f"Feature matrix shape: {feature_matrix.shape}")

print("\n=== SUMMARY ===")
print("1. CountVectorizer: Simple word frequency counts")
print("2. TfidfVectorizer: Weighted importance using TF-IDF")
print("3. Both support extensive preprocessing options")
print("4. Key parameters: stop_words, ngram_range, min_df, max_df")
print("5. Essential for text classification and NLP pipelines")
```

### Key Concepts

**CountVectorizer:**
- Converts text to numerical features using word counts  
- Creates sparse matrix with document-term frequencies
- Good baseline for simple text classification
- Parameters: `stop_words`, `ngram_range`, `min_df`, `max_df`

**TfidfVectorizer:**
- Applies TF-IDF (Term Frequency-Inverse Document Frequency) weighting
- Reduces impact of common words, emphasizes distinctive terms
- Generally better performance than simple counts
- Same parameters as CountVectorizer plus TF-IDF specific options

**Best Practices:**
1. **Text Preprocessing**: Remove noise, normalize case, handle special characters
2. **Stop Words**: Remove common words that don't carry semantic meaning
3. **N-grams**: Include word combinations for better context
4. **Feature Selection**: Use `min_df`/`max_df` to filter extreme frequencies
5. **Pipeline Integration**: Combine with scikit-learn pipelines for end-to-end workflows

**Use Cases:**
- Document classification
- Sentiment analysis  
- Topic modeling preparation
- Information retrieval
- Text similarity computation

**Common Pitfalls:**
- Not handling out-of-vocabulary words in production
- Ignoring sparsity implications for memory usage
- Over-fitting with too many features
- Not considering domain-specific preprocessing needs

---
## Question 6
## Question 6

**Normalize a given dataset usingScikit-Learn's preprocessingmodule, then train and test aNaive Bayes classifier.**

**Answer:** 

### Dataset Normalization and Naive Bayes Classification

Data normalization is crucial for many machine learning algorithms. While Naive Bayes classifiers are generally robust to different scales, normalization can still improve performance, especially when features have vastly different ranges.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    Normalizer, QuantileTransformer, PowerTransformer
)
from sklearn.naive_bayes import (
    GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
)
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=== DATASET NORMALIZATION AND NAIVE BAYES CLASSIFICATION ===\n")

# 1. Load and Prepare Dataset
print("1. DATASET PREPARATION")
print("-" * 40)

# Load wine dataset
wine_data = load_wine()
X_wine, y_wine = wine_data.data, wine_data.target

# Create a synthetic dataset for comparison
X_synthetic, y_synthetic = make_classification(
    n_samples=1000, n_features=10, n_informative=8, 
    n_redundant=2, n_clusters_per_class=1, random_state=42
)

# Use wine dataset for main demonstration
X, y = X_wine, y_wine
feature_names = wine_data.feature_names

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")

# Display basic statistics
df = pd.DataFrame(X, columns=feature_names)
print(f"\nFeature statistics (first 5 features):")
print(df.iloc[:, :5].describe())

# Check for different scales
print(f"\nFeature ranges:")
for i, name in enumerate(feature_names[:8]):  # Show first 8 features
    print(f"  {name}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}] (std: {X[:, i].std():.2f})")

# 2. Split the data
print(f"\n2. DATA SPLITTING")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# 3. Different Normalization Techniques
print(f"\n3. NORMALIZATION TECHNIQUES COMPARISON")
print("-" * 40)

# Define different scalers
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'Normalizer': Normalizer(),
    'QuantileTransformer': QuantileTransformer(random_state=42),
    'PowerTransformer': PowerTransformer(random_state=42)
}

# Store normalized datasets
normalized_data = {}

for name, scaler in scalers.items():
    # Fit on training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    normalized_data[name] = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'scaler': scaler
    }
    
    # Show transformation effects
    print(f"\n{name}:")
    print(f"  Original range: [{X_train[:, 0].min():.2f}, {X_train[:, 0].max():.2f}]")
    print(f"  Scaled range: [{X_train_scaled[:, 0].min():.2f}, {X_train_scaled[:, 0].max():.2f}]")
    print(f"  Scaled std: {X_train_scaled[:, 0].std():.4f}")

# 4. Naive Bayes Variants
print(f"\n4. NAIVE BAYES CLASSIFIER VARIANTS")
print("-" * 40)

# Define different Naive Bayes classifiers
nb_classifiers = {
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(alpha=1.0),  # Requires non-negative features
    'BernoulliNB': BernoulliNB(alpha=1.0),
    'ComplementNB': ComplementNB(alpha=1.0)
}

# 5. Comprehensive Evaluation
print(f"\n5. COMPREHENSIVE EVALUATION")
print("-" * 40)

results = {}

# Test without normalization first
print("WITHOUT NORMALIZATION:")
for nb_name, nb_classifier in nb_classifiers.items():
    try:
        # Train
        nb_classifier.fit(X_train, y_train)
        
        # Predict
        y_pred = nb_classifier.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        results[f'No_Scaling_{nb_name}'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }
        
        print(f"  {nb_name}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
        
    except Exception as e:
        print(f"  {nb_name}: Failed - {str(e)}")
        results[f'No_Scaling_{nb_name}'] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

# Test with different normalization techniques
for scaler_name, data in normalized_data.items():
    print(f"\nWITH {scaler_name.upper()}:")
    
    X_train_scaled = data['X_train']
    X_test_scaled = data['X_test']
    
    # Handle negative values for MultinomialNB and ComplementNB
    if scaler_name in ['StandardScaler', 'RobustScaler', 'PowerTransformer']:
        # Skip Multinomial and Complement NB for scalers that can produce negative values
        nb_subset = {'GaussianNB': nb_classifiers['GaussianNB'], 
                    'BernoulliNB': nb_classifiers['BernoulliNB']}
    else:
        nb_subset = nb_classifiers
    
    for nb_name, nb_classifier in nb_subset.items():
        try:
            # Create fresh instance
            if nb_name == 'GaussianNB':
                classifier = GaussianNB()
            elif nb_name == 'MultinomialNB':
                classifier = MultinomialNB(alpha=1.0)
            elif nb_name == 'BernoulliNB':
                classifier = BernoulliNB(alpha=1.0)
            else:
                classifier = ComplementNB(alpha=1.0)
            
            # Train
            classifier.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = classifier.predict(X_test_scaled)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            results[f'{scaler_name}_{nb_name}'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred,
                'classifier': classifier,
                'scaler': data['scaler']
            }
            
            print(f"  {nb_name}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
            
        except Exception as e:
            print(f"  {nb_name}: Failed - {str(e)}")
            results[f'{scaler_name}_{nb_name}'] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

# 6. Find Best Combination
print(f"\n6. BEST PERFORMING COMBINATIONS")
print("-" * 40)

# Sort results by accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

print("Top 10 combinations by accuracy:")
for i, (combo, metrics) in enumerate(sorted_results[:10]):
    scaler_name = combo.split('_')[0] if '_' in combo else 'No_Scaling'
    nb_name = '_'.join(combo.split('_')[1:]) if '_' in combo else combo
    
    print(f"{i+1:2d}. {scaler_name:<15} + {nb_name:<12}: "
          f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

# 7. Detailed Analysis of Best Model
best_combo, best_metrics = sorted_results[0]
print(f"\n7. DETAILED ANALYSIS OF BEST MODEL")
print("-" * 40)
print(f"Best combination: {best_combo}")

if 'classifier' in best_metrics:
    best_classifier = best_metrics['classifier']
    best_scaler = best_metrics['scaler']
    best_predictions = best_metrics['predictions']
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, best_predictions, 
                              target_names=wine_data.target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_predictions)
    print(f"\nConfusion Matrix:")
    print(cm)

# 8. Cross-Validation Analysis
print(f"\n8. CROSS-VALIDATION ANALYSIS")
print("-" * 40)

# Test top 3 combinations with cross-validation
top_combinations = sorted_results[:3]

for combo, _ in top_combinations:
    scaler_name = combo.split('_')[0] if '_' in combo else 'No_Scaling'
    nb_name = '_'.join(combo.split('_')[1:]) if '_' in combo else combo
    
    if scaler_name == 'No_Scaling':
        # No scaling pipeline
        if nb_name == 'GaussianNB':
            pipeline = Pipeline([('classifier', GaussianNB())])
        elif nb_name == 'MultinomialNB':
            pipeline = Pipeline([('classifier', MultinomialNB(alpha=1.0))])
        elif nb_name == 'BernoulliNB':
            pipeline = Pipeline([('classifier', BernoulliNB(alpha=1.0))])
        else:
            pipeline = Pipeline([('classifier', ComplementNB(alpha=1.0))])
    else:
        # With scaling pipeline
        scaler = scalers[scaler_name]
        if nb_name == 'GaussianNB':
            classifier = GaussianNB()
        elif nb_name == 'MultinomialNB':
            classifier = MultinomialNB(alpha=1.0)
        elif nb_name == 'BernoulliNB':
            classifier = BernoulliNB(alpha=1.0)
        else:
            classifier = ComplementNB(alpha=1.0)
        
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', classifier)
        ])
    
    try:
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        
        print(f"{combo}:")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  CV Scores: {cv_scores}")
        
    except Exception as e:
        print(f"{combo}: CV Failed - {str(e)}")

# 9. Practical Implementation Example
print(f"\n9. PRACTICAL IMPLEMENTATION EXAMPLE")
print("-" * 40)

class NormalizedNaiveBayes:
    """Production-ready normalized Naive Bayes classifier"""
    
    def __init__(self, scaler_type='standard', nb_type='gaussian', **kwargs):
        """
        Initialize normalized Naive Bayes classifier
        
        Parameters:
        - scaler_type: 'standard', 'minmax', 'robust', 'quantile', 'power'
        - nb_type: 'gaussian', 'multinomial', 'bernoulli', 'complement'
        - kwargs: additional parameters for scaler and classifier
        """
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'quantile':
            self.scaler = QuantileTransformer(random_state=42)
        elif scaler_type == 'power':
            self.scaler = PowerTransformer(random_state=42)
        else:
            self.scaler = None
        
        # Initialize Naive Bayes classifier
        if nb_type == 'gaussian':
            self.classifier = GaussianNB()
        elif nb_type == 'multinomial':
            self.classifier = MultinomialNB(alpha=kwargs.get('alpha', 1.0))
        elif nb_type == 'bernoulli':
            self.classifier = BernoulliNB(alpha=kwargs.get('alpha', 1.0))
        elif nb_type == 'complement':
            self.classifier = ComplementNB(alpha=kwargs.get('alpha', 1.0))
        
        self.scaler_type = scaler_type
        self.nb_type = nb_type
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the normalized Naive Bayes classifier"""
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.classifier.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.classifier.predict_proba(X_scaled)
    
    def score(self, X, y):
        """Return accuracy score"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# Demonstrate the custom class
custom_nb = NormalizedNaiveBayes(scaler_type='standard', nb_type='gaussian')
custom_nb.fit(X_train, y_train)

custom_predictions = custom_nb.predict(X_test)
custom_accuracy = accuracy_score(y_test, custom_predictions)

print(f"Custom NormalizedNaiveBayes accuracy: {custom_accuracy:.4f}")
print(f"Custom model predictions (first 10): {custom_predictions[:10]}")
print(f"Actual labels (first 10): {y_test[:10]}")

# 10. Feature Impact Analysis
print(f"\n10. FEATURE IMPACT ANALYSIS")
print("-" * 40)

# Analyze feature importance using the best model
if 'classifier' in best_metrics and hasattr(best_metrics['classifier'], 'theta_'):
    # For Gaussian NB, we can analyze feature means per class
    best_classifier = best_metrics['classifier']
    
    print("Feature means per class (using best model):")
    for class_idx, class_name in enumerate(wine_data.target_names):
        print(f"\nClass {class_idx} ({class_name}):")
        class_means = best_classifier.theta_[class_idx]
        
        # Show top 5 features with highest means
        feature_importance = list(zip(feature_names, class_means))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for feat_name, mean_val in feature_importance[:5]:
            print(f"  {feat_name}: {mean_val:.4f}")

print(f"\n=== SUMMARY ===")
print("1. Data normalization can significantly impact Naive Bayes performance")
print("2. GaussianNB works well with normalized continuous features")
print("3. MultinomialNB requires non-negative features")
print("4. Different scalers work better for different data distributions")
print("5. Cross-validation provides robust performance estimates")
print("6. Pipeline approach ensures proper train/test separation")
```

### Key Concepts

**Normalization Techniques:**
- **StandardScaler**: Zero mean, unit variance (z-score normalization)
- **MinMaxScaler**: Scale to [0,1] range  
- **RobustScaler**: Uses median and IQR, robust to outliers
- **QuantileTransformer**: Maps to uniform distribution
- **PowerTransformer**: Makes data more Gaussian-like

**Naive Bayes Variants:**
- **GaussianNB**: Assumes continuous features follow normal distribution
- **MultinomialNB**: For discrete counts (requires non-negative features)  
- **BernoulliNB**: For binary/boolean features
- **ComplementNB**: Complement version of MultinomialNB

**Best Practices:**
1. **Proper Data Splitting**: Always fit scaler on training data only
2. **Pipeline Usage**: Ensures consistent preprocessing
3. **Feature Requirements**: Check if classifier requires non-negative features
4. **Cross-Validation**: Use for robust performance estimation
5. **Scaler Selection**: Choose based on data distribution and outliers

**Use Cases:**
- Text classification (MultinomialNB with TF-IDF)
- Medical diagnosis (GaussianNB with normalized lab values)
- Spam detection (BernoulliNB with binary features)
- Sentiment analysis (ComplementNB for imbalanced classes)

**Common Pitfalls:**
- Fitting scaler on entire dataset instead of training set only
- Using MultinomialNB with negative features
- Not considering feature distributions when choosing scalers
- Ignoring class imbalance in evaluation metrics

---
## Question 7

**Demonstrate how to use Scikit-Learnâ€™s Pipeline to combine preprocessing and model training steps.**

### Theory
Scikit-Learn's Pipeline allows you to chain multiple data transformation steps with a final estimator into a single object. This ensures proper data flow, prevents data leakage, and makes hyperparameter tuning more convenient by treating the entire workflow as a single estimator.

### Code Example
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load sample dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

print("=== SCIKIT-LEARN PIPELINE DEMONSTRATION ===\n")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset shape: {X.shape}")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# 1. Basic Pipeline Example
print("\n1. BASIC PIPELINE")
print("-" * 40)

# Create a basic pipeline
basic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Train the pipeline
basic_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = basic_pipeline.predict(X_test)
accuracy = basic_pipeline.score(X_test, y_test)

print(f"Basic Pipeline Accuracy: {accuracy:.4f}")
print(f"Pipeline steps: {basic_pipeline.steps}")

# 2. Complex Pipeline with Feature Selection
print("\n2. COMPLEX PIPELINE WITH FEATURE SELECTION")
print("-" * 40)

complex_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('polynomial_features', PolynomialFeatures(degree=2, interaction_only=True)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Train and evaluate
complex_pipeline.fit(X_train, y_train)
complex_accuracy = complex_pipeline.score(X_test, y_test)

print(f"Complex Pipeline Accuracy: {complex_accuracy:.4f}")

# Get selected features
selected_features_mask = complex_pipeline.named_steps['feature_selection'].get_support()
selected_features = [feature_names[i] for i, selected in enumerate(selected_features_mask) if selected]
print(f"Selected features: {selected_features[:5]}...")  # Show first 5

# 3. Pipeline with Grid Search
print("\n3. PIPELINE WITH HYPERPARAMETER TUNING")
print("-" * 40)

# Define pipeline for grid search
grid_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=42))
])

# Define parameter grid
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['linear', 'rbf', 'poly'],
    'classifier__gamma': ['scale', 'auto', 0.001, 0.01]
}

# Perform grid search
grid_search = GridSearchCV(
    grid_pipeline, param_grid, cv=5, 
    scoring='accuracy', n_jobs=-1, verbose=1
)

print("Performing grid search...")
grid_search.fit(X_train, y_train)

# Best results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"Test set accuracy: {grid_search.score(X_test, y_test):.4f}")

# 4. Multiple Pipeline Comparison
print("\n4. COMPARING MULTIPLE PIPELINES")
print("-" * 40)

pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(random_state=42))
    ])
}

# Evaluate each pipeline
results = {}
for name, pipeline in pipelines.items():
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
    # Train and test
    pipeline.fit(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_score': test_score
    }
    
    print(f"{name}:")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Test Score: {test_score:.4f}")

# 5. Custom Pipeline Components
print("\n5. CUSTOM PIPELINE COMPONENTS")
print("-" * 40)

from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    """Custom transformer to remove outliers using IQR method"""
    
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds = None
        self.upper_bounds = None
    
    def fit(self, X, y=None):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        self.lower_bounds = Q1 - self.factor * IQR
        self.upper_bounds = Q3 + self.factor * IQR
        
        return self
    
    def transform(self, X):
        # Create mask for outliers
        mask = np.all(
            (X >= self.lower_bounds) & (X <= self.upper_bounds), 
            axis=1
        )
        return X[mask]

class FeatureLogger(BaseEstimator, TransformerMixin):
    """Custom transformer that logs feature statistics"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print(f"Feature shape: {X.shape}")
        print(f"Feature range: [{X.min():.2f}, {X.max():.2f}]")
        return X

# Custom pipeline with custom components
custom_pipeline = Pipeline([
    ('logger1', FeatureLogger()),
    ('scaler', StandardScaler()),
    ('logger2', FeatureLogger()),
    ('feature_selection', SelectKBest(f_classif, k=15)),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])

print("Training custom pipeline with logging:")
custom_pipeline.fit(X_train, y_train)
custom_accuracy = custom_pipeline.score(X_test, y_test)
print(f"Custom Pipeline Accuracy: {custom_accuracy:.4f}")

# 6. Pipeline with Column Transformer
print("\n6. COLUMN TRANSFORMER PIPELINE")
print("-" * 40)

# Create mixed dataset for demonstration
from sklearn.datasets import make_classification

# Generate mixed data (some features need different preprocessing)
X_mixed, y_mixed = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, random_state=42
)

# Split features into different groups (simulate different feature types)
numeric_features_1 = list(range(0, 10))  # Features 0-9
numeric_features_2 = list(range(10, 20))  # Features 10-19

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num1', StandardScaler(), numeric_features_1),
        ('num2', MinMaxScaler(), numeric_features_2)
    ],
    remainder='passthrough'  # Keep remaining features unchanged
)

# Pipeline with column transformer
column_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Split mixed data
X_mixed_train, X_mixed_test, y_mixed_train, y_mixed_test = train_test_split(
    X_mixed, y_mixed, test_size=0.2, random_state=42, stratify=y_mixed
)

# Train and evaluate
column_pipeline.fit(X_mixed_train, y_mixed_train)
column_accuracy = column_pipeline.score(X_mixed_test, y_mixed_test)
print(f"Column Transformer Pipeline Accuracy: {column_accuracy:.4f}")

# 7. Pipeline Inspection and Debugging
print("\n7. PIPELINE INSPECTION")
print("-" * 40)

# Access pipeline components
best_pipeline = pipelines['Random Forest']
best_pipeline.fit(X_train, y_train)

print("Pipeline components:")
for i, (name, step) in enumerate(best_pipeline.steps):
    print(f"  Step {i+1}: {name} -> {type(step).__name__}")

# Access intermediate results
scaler = best_pipeline.named_steps['scaler']
classifier = best_pipeline.named_steps['classifier']

# Transform data up to scaler
X_scaled = scaler.transform(X_test)
print(f"Scaled data shape: {X_scaled.shape}")
print(f"Scaled data range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

# Get feature importance from random forest
if hasattr(classifier, 'feature_importances_'):
    feature_importance = classifier.feature_importances_
    top_features = np.argsort(feature_importance)[-10:]
    
    print("\nTop 10 most important features:")
    for i, idx in enumerate(reversed(top_features)):
        print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")

# 8. Pipeline Persistence
print("\n8. PIPELINE PERSISTENCE")
print("-" * 40)

import joblib
import pickle
from datetime import datetime

# Save pipeline using joblib (recommended)
pipeline_filename = f'best_pipeline_{datetime.now().strftime("%Y%m%d_%H%M")}.joblib'
joblib.dump(best_pipeline, pipeline_filename)
print(f"Pipeline saved as: {pipeline_filename}")

# Load pipeline
loaded_pipeline = joblib.load(pipeline_filename)
loaded_accuracy = loaded_pipeline.score(X_test, y_test)
print(f"Loaded pipeline accuracy: {loaded_accuracy:.4f}")

# Verify it's the same
print(f"Original == Loaded: {accuracy == loaded_accuracy}")

# Alternative: pickle (less efficient for numpy arrays)
with open('pipeline_pickle.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)
print("Pipeline also saved using pickle")

print("\n=== PIPELINE SUMMARY ===")
print("Key Benefits:")
print("1. Prevents data leakage by applying transformations consistently")
print("2. Simplifies hyperparameter tuning across entire workflow")
print("3. Makes code more maintainable and reproducible")
print("4. Enables easy model deployment and persistence")
print("5. Supports complex preprocessing workflows")
```

### Explanation
**Pipeline Components:**
1. **Transformers**: Objects that modify data (e.g., StandardScaler, SelectKBest)
2. **Estimators**: Final predictive models (e.g., LogisticRegression, SVC)
3. **Steps**: Named tuple pairs of (name, transformer/estimator)

**Key Methods:**
- `fit()`: Trains all transformers and final estimator
- `transform()`: Applies transformations (not available if final step is estimator)
- `predict()`: Makes predictions using the trained pipeline
- `fit_transform()`: Fits and transforms in one step

### Use Cases
- **Data Preprocessing**: Standardization, normalization, feature selection
- **Feature Engineering**: Polynomial features, dimensionality reduction
- **Model Training**: End-to-end ML workflows
- **Hyperparameter Tuning**: Grid search across entire pipeline
- **Production Deployment**: Consistent data processing

### Best Practices
1. **Naming**: Use descriptive step names for easy debugging
2. **Data Leakage Prevention**: Fit transformers only on training data
3. **Custom Transformers**: Inherit from BaseEstimator and TransformerMixin
4. **Persistence**: Use joblib for better numpy array handling
5. **Inspection**: Access intermediate steps for debugging

### Pitfalls
- **Memory Usage**: Large datasets may require careful memory management
- **Serialization**: Some custom components may not pickle properly
- **Version Compatibility**: Ensure scikit-learn versions match when loading
- **Data Leakage**: Never fit preprocessing on full dataset before splitting

### Debugging
```python
# Debug pipeline steps
for name, step in pipeline.steps:
    print(f"Step: {name}")
    if hasattr(step, 'get_params'):
        print(f"  Parameters: {step.get_params()}")

# Check data flow
X_intermediate = pipeline[:-1].transform(X_test)  # All steps except last
print(f"Intermediate shape: {X_intermediate.shape}")
```

### Optimization
```python
# Memory-efficient pipeline
from sklearn.pipeline import make_pipeline

# Use make_pipeline for automatic naming
memory_pipeline = make_pipeline(
    StandardScaler(),
    SelectKBest(k=10),
    LogisticRegression(),
    memory='cache_dir'  # Cache intermediate results
)

# Parallel processing in grid search
param_grid = {'logisticregression__C': [0.1, 1, 10]}
grid_search = GridSearchCV(
    memory_pipeline, param_grid, 
    cv=5, n_jobs=-1  # Parallel processing
)
```

---

## Question 8

**Write a Python function that uses Scikit-Learnâ€™s RandomForestClassifier and performs a grid search to find the best hyperparameters.**

### Theory
RandomForestClassifier is an ensemble method that combines multiple decision trees to reduce overfitting and improve generalization. Grid search systematically tests different hyperparameter combinations using cross-validation to find the optimal configuration for a given dataset.

### Code Example
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    cross_val_score, validation_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
import time
import warnings
warnings.filterwarnings('ignore')

def random_forest_grid_search(X, y, test_size=0.2, cv_folds=5, 
                             scoring='accuracy', search_type='grid',
                             n_jobs=-1, verbose=True):
    """
    Comprehensive RandomForestClassifier with hyperparameter optimization
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target values
    test_size : float, default=0.2
        Proportion of dataset for testing
    cv_folds : int, default=5
        Number of cross-validation folds
    scoring : str, default='accuracy'
        Scoring metric for optimization
    search_type : str, default='grid'
        'grid' for GridSearchCV, 'random' for RandomizedSearchCV
    n_jobs : int, default=-1
        Number of parallel jobs
    verbose : bool, default=True
        Whether to print progress and results
        
    Returns:
    --------
    dict : Dictionary containing results and best model
    """
    
    if verbose:
        print("=== RANDOM FOREST GRID SEARCH OPTIMIZATION ===\n")
        print(f"Dataset shape: {X.shape}")
        print(f"Classes: {len(np.unique(y))}")
        print(f"Class distribution: {np.bincount(y)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    if verbose:
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    # Define comprehensive parameter grid
    if search_type == 'grid':
        param_grid = {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [None, 10, 20, 30],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__max_features': ['sqrt', 'log2', None],
            'rf__bootstrap': [True, False]
        }
        
        search = GridSearchCV(
            pipeline, param_grid, cv=cv_folds,
            scoring=scoring, n_jobs=n_jobs, verbose=1 if verbose else 0
        )
        
    else:  # Random search
        param_distributions = {
            'rf__n_estimators': randint(50, 300),
            'rf__max_depth': [None] + list(range(10, 51, 10)),
            'rf__min_samples_split': randint(2, 21),
            'rf__min_samples_leaf': randint(1, 11),
            'rf__max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'rf__bootstrap': [True, False]
        }
        
        search = RandomizedSearchCV(
            pipeline, param_distributions, n_iter=100,
            cv=cv_folds, scoring=scoring, n_jobs=n_jobs,
            random_state=42, verbose=1 if verbose else 0
        )
    
    # Perform search
    if verbose:
        print(f"\nPerforming {search_type} search with {cv_folds}-fold CV...")
        start_time = time.time()
    
    search.fit(X_train, y_train)
    
    if verbose:
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.2f} seconds")
    
    # Get best model and evaluate
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_score = search.best_score_
    
    # Test set evaluation
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    if len(np.unique(y)) == 2:  # Binary classification
        test_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:  # Multi-class
        test_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    # Results dictionary
    results = {
        'best_model': best_model,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'predictions': y_pred,
        'prediction_probabilities': y_pred_proba,
        'search_results': search.cv_results_,
        'X_test': X_test,
        'y_test': y_test
    }
    
    if verbose:
        print(f"\n=== RESULTS ===")
        print(f"Best parameters: {best_params}")
        print(f"Best CV score ({scoring}): {best_cv_score:.4f}")
        print(f"\nTest Set Performance:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")
        print(f"  ROC-AUC: {test_auc:.4f}")
    
    return results

def analyze_grid_search_results(results, feature_names=None, class_names=None):
    """
    Comprehensive analysis of grid search results
    """
    print("\n=== DETAILED ANALYSIS ===")
    
    best_model = results['best_model']
    y_test = results['y_test']
    y_pred = results['predictions']
    y_pred_proba = results['prediction_probabilities']
    
    # Classification report
    print("\nClassification Report:")
    if class_names:
        print(classification_report(y_test, y_pred, target_names=class_names))
    else:
        print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Feature importance
    rf_model = best_model.named_steps['rf']
    feature_importance = rf_model.feature_importances_
    
    if feature_names is not None:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(importance_df.head(10))
    else:
        top_indices = np.argsort(feature_importance)[-10:]
        print(f"\nTop 10 Feature Importances:")
        for i, idx in enumerate(reversed(top_indices)):
            print(f"  Feature {idx}: {feature_importance[idx]:.4f}")
    
    return {
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def visualize_results(results, feature_names=None):
    """
    Create comprehensive visualizations of results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    y_test = results['y_test']
    y_pred = results['predictions']
    y_pred_proba = results['prediction_probabilities']
    best_model = results['best_model']
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # 2. ROC Curve (for binary classification)
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {results["test_auc"]:.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
    else:
        # For multi-class, show prediction confidence distribution
        max_proba = np.max(y_pred_proba, axis=1)
        axes[0,1].hist(max_proba, bins=30, alpha=0.7)
        axes[0,1].set_xlabel('Prediction Confidence')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Prediction Confidence Distribution')
    
    # 3. Feature Importance
    rf_model = best_model.named_steps['rf']
    feature_importance = rf_model.feature_importances_
    
    if feature_names is not None and len(feature_names) <= 20:
        # Show all features if 20 or fewer
        indices = np.argsort(feature_importance)
        axes[1,0].barh(range(len(indices)), feature_importance[indices])
        axes[1,0].set_yticks(range(len(indices)))
        axes[1,0].set_yticklabels([feature_names[i] for i in indices])
    else:
        # Show top 10 features
        top_indices = np.argsort(feature_importance)[-10:]
        axes[1,0].barh(range(len(top_indices)), feature_importance[top_indices])
        axes[1,0].set_yticks(range(len(top_indices)))
        if feature_names is not None:
            axes[1,0].set_yticklabels([feature_names[i] for i in top_indices])
        else:
            axes[1,0].set_yticklabels([f'Feature {i}' for i in top_indices])
    
    axes[1,0].set_title('Feature Importance')
    axes[1,0].set_xlabel('Importance')
    
    # 4. Cross-validation scores distribution
    cv_results = results['search_results']
    cv_scores = cv_results['mean_test_score']
    
    axes[1,1].hist(cv_scores, bins=30, alpha=0.7)
    axes[1,1].axvline(results['best_cv_score'], color='red', linestyle='--', 
                     label=f'Best Score: {results["best_cv_score"]:.4f}')
    axes[1,1].set_xlabel('Cross-Validation Score')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('CV Score Distribution')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()

def parameter_sensitivity_analysis(X, y, param_name, param_values, 
                                 other_params=None, cv_folds=5):
    """
    Analyze sensitivity to a specific parameter
    """
    if other_params is None:
        other_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'random_state': 42
        }
    
    train_scores = []
    val_scores = []
    
    for param_value in param_values:
        # Update parameter
        params = other_params.copy()
        params[param_name] = param_value
        
        # Create model
        rf = RandomForestClassifier(**params)
        
        # Get validation curve
        train_score, val_score = validation_curve(
            rf, X, y, param_name=param_name, param_range=[param_value],
            cv=cv_folds, scoring='accuracy', n_jobs=-1
        )
        
        train_scores.append(train_score.mean())
        val_scores.append(val_score.mean())
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, train_scores, 'o-', label='Training Score')
    plt.plot(param_values, val_scores, 'o-', label='Validation Score')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy Score')
    plt.title(f'Parameter Sensitivity: {param_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return train_scores, val_scores

# Demonstration
if __name__ == "__main__":
    # Load sample dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = data.target_names
    
    print("=== RANDOM FOREST HYPERPARAMETER OPTIMIZATION DEMO ===\n")
    
    # 1. Grid Search
    print("1. GRID SEARCH")
    print("-" * 40)
    
    grid_results = random_forest_grid_search(
        X, y, search_type='grid', scoring='f1_weighted'
    )
    
    # 2. Detailed Analysis
    analysis = analyze_grid_search_results(
        grid_results, feature_names, class_names
    )
    
    # 3. Visualizations
    print("\n3. VISUALIZATIONS")
    print("-" * 40)
    visualize_results(grid_results, feature_names)
    
    # 4. Random Search Comparison
    print("\n4. RANDOM SEARCH COMPARISON")
    print("-" * 40)
    
    random_results = random_forest_grid_search(
        X, y, search_type='random', scoring='f1_weighted'
    )
    
    print(f"\nComparison:")
    print(f"Grid Search Best Score: {grid_results['best_cv_score']:.4f}")
    print(f"Random Search Best Score: {random_results['best_cv_score']:.4f}")
    
    # 5. Parameter Sensitivity
    print("\n5. PARAMETER SENSITIVITY ANALYSIS")
    print("-" * 40)
    
    # Analyze n_estimators sensitivity
    n_estimators_values = [10, 50, 100, 200, 300, 500]
    train_scores, val_scores = parameter_sensitivity_analysis(
        X, y, 'n_estimators', n_estimators_values
    )
```

### Explanation
**Grid Search Process:**
1. **Parameter Grid Definition**: Specify ranges for each hyperparameter
2. **Cross-Validation**: Each combination tested using k-fold CV
3. **Best Parameter Selection**: Combination with highest CV score chosen
4. **Final Evaluation**: Best model tested on held-out test set

**Key Parameters:**
- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of trees (None = unlimited)
- `min_samples_split`: Minimum samples required to split node
- `min_samples_leaf`: Minimum samples required at leaf node
- `max_features`: Number of features considered for best split

### Use Cases
- **Classification Tasks**: Multi-class and binary classification
- **Feature Selection**: Built-in feature importance ranking
- **Robust Predictions**: Ensemble reduces overfitting
- **Missing Value Handling**: Can handle missing values naturally
- **Large Datasets**: Efficient parallel processing

### Best Practices
1. **Stratified CV**: Maintain class distribution in cross-validation
2. **Random Search First**: Use random search for initial exploration
3. **Progressive Refinement**: Start broad, then narrow parameter ranges
4. **Computational Budget**: Balance search depth with available time
5. **Feature Scaling**: Not required for tree-based methods

### Pitfalls
- **Overfitting**: Too many trees or deep trees can overfit
- **Computational Cost**: Grid search can be very expensive
- **Parameter Interaction**: Some parameters interact in complex ways
- **Class Imbalance**: May need to adjust class weights
- **Memory Usage**: Large forests require significant memory

### Debugging
```python
# Check for overfitting
best_rf = results['best_model'].named_steps['rf']
print(f"Train accuracy: {best_rf.score(X_train, y_train):.4f}")
print(f"Test accuracy: {results['test_accuracy']:.4f}")

# Feature importance analysis
feature_importance = best_rf.feature_importances_
low_importance = np.sum(feature_importance < 0.01)
print(f"Features with <1% importance: {low_importance}")
```

### Optimization
```python
# Memory and time optimization
from sklearn.model_selection import RandomizedSearchCV

# Use random search for large parameter spaces
param_dist = {
    'rf__n_estimators': randint(100, 1000),
    'rf__max_depth': randint(10, 100),
    'rf__min_samples_split': uniform(0.01, 0.2)
}

random_search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=50,  # Limit iterations
    cv=3,  # Reduce CV folds for speed
    n_jobs=-1, random_state=42
)

# Early stopping for large datasets
rf_early = RandomForestClassifier(
    n_estimators=1000,
    warm_start=True,  # Allow incremental training
    oob_score=True,   # Out-of-bag error estimation
    random_state=42
)
```

---

## Question 9

**Use Scikit-Learn to visualize the decision boundary of a SVM with a non-linear kernel.**

### Theory
Support Vector Machines (SVM) create decision boundaries by finding optimal hyperplanes that maximize the margin between classes. Non-linear kernels (RBF, polynomial) transform data into higher-dimensional spaces where linear separation becomes possible. Visualizing these boundaries helps understand how SVMs handle complex, non-linearly separable data.

### Code Example
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

def create_mesh_grid(X, h=0.01):
    """Create a mesh grid for plotting decision boundaries"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_decision_boundary(X, y, classifier, title, ax=None, 
                          support_vectors=True, mesh_step=0.01):
    """
    Plot SVM decision boundary with support vectors
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        2D feature data
    y : array-like, shape (n_samples,)
        Target labels
    classifier : trained SVM classifier
        Fitted SVM model
    title : str
        Plot title
    ax : matplotlib axis, optional
        Axis to plot on
    support_vectors : bool, default=True
        Whether to highlight support vectors
    mesh_step : float, default=0.01
        Step size for mesh grid
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mesh grid
    xx, yy = create_mesh_grid(X, mesh_step)
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get predictions for mesh points
    Z = classifier.decision_function(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    # Plot decision boundary and margins
    ax.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='solid')
    ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, colors='black', linestyles='dashed')
    
    # Plot data points
    unique_labels = np.unique(y)
    colors = ['red', 'green', 'blue']
    for i, label in enumerate(unique_labels):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                  s=50, alpha=0.7, label=f'Class {label}')
    
    # Highlight support vectors
    if support_vectors and hasattr(classifier, 'support_vectors_'):
        ax.scatter(classifier.support_vectors_[:, 0], 
                  classifier.support_vectors_[:, 1],
                  s=100, facecolors='none', edgecolors='black', 
                  linewidths=2, label='Support Vectors')
    
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

def comprehensive_svm_visualization():
    """Comprehensive SVM visualization with different kernels and datasets"""
    
    print("=== SVM DECISION BOUNDARY VISUALIZATION ===\n")
    
    # 1. Create different types of datasets
    datasets = {}
    
    # Linear separable data
    X_linear, y_linear = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, random_state=42
    )
    datasets['Linear'] = (X_linear, y_linear)
    
    # Circular data (non-linear)
    X_circles, y_circles = make_circles(
        n_samples=200, noise=0.1, factor=0.5, random_state=42
    )
    datasets['Circles'] = (X_circles, y_circles)
    
    # Moon-shaped data (non-linear)
    X_moons, y_moons = make_moons(
        n_samples=200, noise=0.15, random_state=42
    )
    datasets['Moons'] = (X_moons, y_moons)
    
    # Complex non-linear data
    np.random.seed(42)
    X_complex = np.random.randn(200, 2)
    y_complex = ((X_complex[:, 0] ** 2 + X_complex[:, 1] ** 2) > 1).astype(int)
    datasets['Complex'] = (X_complex, y_complex)
    
    # 2. Different SVM kernels
    kernels = {
        'Linear': SVC(kernel='linear', C=1.0),
        'RBF (γ=1)': SVC(kernel='rbf', C=1.0, gamma=1.0),
        'RBF (γ=0.1)': SVC(kernel='rbf', C=1.0, gamma=0.1),
        'Polynomial (d=2)': SVC(kernel='poly', degree=2, C=1.0),
        'Polynomial (d=3)': SVC(kernel='poly', degree=3, C=1.0),
        'Sigmoid': SVC(kernel='sigmoid', C=1.0)
    }
    
    # 3. Visualize each dataset with different kernels
    for dataset_name, (X, y) in datasets.items():
        print(f"\nDataset: {dataset_name}")
        print(f"Shape: {X.shape}, Classes: {np.unique(y)}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create subplot for each kernel
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, (kernel_name, svm) in enumerate(kernels.items()):
            try:
                # Train SVM
                svm.fit(X_scaled, y)
                
                # Calculate accuracy
                accuracy = svm.score(X_scaled, y)
                
                # Plot decision boundary
                title = f'{kernel_name}\nAccuracy: {accuracy:.3f}'
                plot_decision_boundary(X_scaled, y, svm, title, axes[i])
                
                print(f"  {kernel_name}: Accuracy = {accuracy:.3f}, "
                      f"Support Vectors = {len(svm.support_vectors_)}")
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                           transform=axes[i].transAxes, ha='center')
                axes[i].set_title(f'{kernel_name}\nError')
        
        plt.suptitle(f'SVM Decision Boundaries - {dataset_name} Dataset', 
                     fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

def parameter_effect_visualization():
    """Visualize the effect of different SVM parameters"""
    
    print("\n=== PARAMETER EFFECT VISUALIZATION ===\n")
    
    # Create non-linear dataset
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.4, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Effect of C parameter (regularization)
    print("1. Effect of C parameter (Regularization)")
    C_values = [0.1, 1, 10, 100]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, C in enumerate(C_values):
        svm = SVC(kernel='rbf', C=C, gamma='scale')
        svm.fit(X_scaled, y)
        
        accuracy = svm.score(X_scaled, y)
        n_support = len(svm.support_vectors_)
        
        title = f'C = {C}\nAcc: {accuracy:.3f}, SV: {n_support}'
        plot_decision_boundary(X_scaled, y, svm, title, axes[i])
    
    plt.suptitle('Effect of C Parameter on RBF SVM', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 2. Effect of gamma parameter (RBF kernel width)
    print("\n2. Effect of gamma parameter (RBF kernel width)")
    gamma_values = [0.01, 0.1, 1, 10]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, gamma in enumerate(gamma_values):
        svm = SVC(kernel='rbf', C=1.0, gamma=gamma)
        svm.fit(X_scaled, y)
        
        accuracy = svm.score(X_scaled, y)
        n_support = len(svm.support_vectors_)
        
        title = f'γ = {gamma}\nAcc: {accuracy:.3f}, SV: {n_support}'
        plot_decision_boundary(X_scaled, y, svm, title, axes[i])
    
    plt.suptitle('Effect of Gamma Parameter on RBF SVM', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 3. Effect of polynomial degree
    print("\n3. Effect of polynomial degree")
    degrees = [1, 2, 3, 4]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, degree in enumerate(degrees):
        svm = SVC(kernel='poly', degree=degree, C=1.0, coef0=1)
        svm.fit(X_scaled, y)
        
        accuracy = svm.score(X_scaled, y)
        n_support = len(svm.support_vectors_)
        
        title = f'Degree = {degree}\nAcc: {accuracy:.3f}, SV: {n_support}'
        plot_decision_boundary(X_scaled, y, svm, title, axes[i])
    
    plt.suptitle('Effect of Polynomial Degree on SVM', fontsize=16)
    plt.tight_layout()
    plt.show()

def multiclass_svm_visualization():
    """Visualize multiclass SVM decision boundaries"""
    
    print("\n=== MULTICLASS SVM VISUALIZATION ===\n")
    
    # Create multiclass dataset
    X, y = make_classification(
        n_samples=300, n_features=2, n_redundant=0, n_informative=2,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Different multiclass strategies
    strategies = ['ovr', 'ovo']  # one-vs-rest, one-vs-one
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, strategy in enumerate(strategies):
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape=strategy)
        svm.fit(X_scaled, y)
        
        accuracy = svm.score(X_scaled, y)
        n_support = len(svm.support_vectors_)
        
        # Create custom plotting function for multiclass
        xx, yy = create_mesh_grid(X_scaled, 0.02)
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision regions
        axes[i].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Set3)
        
        # Plot data points
        scatter = axes[i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, 
                                 cmap=plt.cm.Set1, s=50, alpha=0.8)
        
        # Highlight support vectors
        axes[i].scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                       s=100, facecolors='none', edgecolors='black', linewidths=2)
        
        axes[i].set_title(f'{strategy.upper()} Strategy\nAcc: {accuracy:.3f}, SV: {n_support}')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Multiclass SVM Decision Boundaries', fontsize=16)
    plt.tight_layout()
    plt.show()

def interactive_svm_demo():
    """Interactive demonstration of SVM parameters"""
    
    print("\n=== INTERACTIVE SVM PARAMETER ANALYSIS ===\n")
    
    # Create dataset
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Parameter combinations to test
    param_combinations = [
        {'kernel': 'rbf', 'C': 0.1, 'gamma': 0.1},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 1.0},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 0.01},
        {'kernel': 'poly', 'C': 1.0, 'degree': 2},
        {'kernel': 'poly', 'C': 1.0, 'degree': 3},
        {'kernel': 'sigmoid', 'C': 1.0, 'gamma': 'scale'}
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    results = []
    
    for i, params in enumerate(param_combinations):
        # Create and train SVM
        svm = SVC(**params)
        svm.fit(X_scaled, y)
        
        # Evaluate
        accuracy = svm.score(X_scaled, y)
        n_support = len(svm.support_vectors_)
        
        # Store results
        results.append({
            'params': params,
            'accuracy': accuracy,
            'n_support_vectors': n_support
        })
        
        # Plot
        param_str = ', '.join([f'{k}={v}' for k, v in params.items()])
        title = f'{param_str}\nAcc: {accuracy:.3f}, SV: {n_support}'
        plot_decision_boundary(X_scaled, y, svm, title, axes[i])
    
    plt.suptitle('SVM Parameter Combinations Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("Parameter Analysis Summary:")
    print("-" * 60)
    for i, result in enumerate(results):
        params = result['params']
        print(f"{i+1}. {params}")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   Support Vectors: {result['n_support_vectors']}")
        print()

def advanced_visualization_techniques():
    """Advanced SVM visualization techniques"""
    
    print("\n=== ADVANCED VISUALIZATION TECHNIQUES ===\n")
    
    # Create dataset
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train SVM
    svm = SVC(kernel='rbf', C=1.0, gamma=1.0)
    svm.fit(X_scaled, y)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Decision function values
    xx, yy = create_mesh_grid(X_scaled, 0.02)
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    im1 = axes[0,0].contourf(xx, yy, Z, levels=50, cmap=plt.cm.RdBu, alpha=0.8)
    axes[0,0].contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    axes[0,0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.RdBu, s=50)
    axes[0,0].scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                     s=100, facecolors='none', edgecolors='black', linewidths=2)
    axes[0,0].set_title('Decision Function Values')
    plt.colorbar(im1, ax=axes[0,0])
    
    # 2. Prediction probabilities (with probability=True)
    svm_prob = SVC(kernel='rbf', C=1.0, gamma=1.0, probability=True)
    svm_prob.fit(X_scaled, y)
    
    Z_prob = svm_prob.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_prob = Z_prob.reshape(xx.shape)
    
    im2 = axes[0,1].contourf(xx, yy, Z_prob, levels=50, cmap=plt.cm.RdBu, alpha=0.8)
    axes[0,1].contour(xx, yy, Z_prob, levels=[0.5], colors='black', linewidths=2)
    axes[0,1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.RdBu, s=50)
    axes[0,1].set_title('Prediction Probabilities')
    plt.colorbar(im2, ax=axes[0,1])
    
    # 3. Support vector analysis
    sv_indices = svm.support_
    sv_classes = y[sv_indices]
    
    axes[1,0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, alpha=0.3, s=20)
    for class_label in np.unique(y):
        class_sv = svm.support_vectors_[sv_classes == class_label]
        axes[1,0].scatter(class_sv[:, 0], class_sv[:, 1], 
                         s=100, label=f'Class {class_label} SV')
    axes[1,0].set_title('Support Vector Analysis')
    axes[1,0].legend()
    
    # 4. Margin visualization
    axes[1,1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.RdBu, s=50)
    axes[1,1].contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'red'],
                     linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
    axes[1,1].scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                     s=100, facecolors='none', edgecolors='black', linewidths=2)
    axes[1,1].set_title('Decision Boundary and Margins')
    
    plt.suptitle('Advanced SVM Visualization Techniques', fontsize=16)
    plt.tight_layout()
    plt.show()

# Main demonstration
if __name__ == "__main__":
    # Run comprehensive demonstrations
    comprehensive_svm_visualization()
    parameter_effect_visualization()
    multiclass_svm_visualization()
    interactive_svm_demo()
    advanced_visualization_techniques()
    
    print("\n=== SUMMARY ===")
    print("Key Insights from SVM Decision Boundary Visualization:")
    print("1. Linear kernels create straight decision boundaries")
    print("2. RBF kernels can create complex, curved boundaries")
    print("3. Higher C values lead to more complex boundaries (lower bias, higher variance)")
    print("4. Higher gamma values create more localized decision boundaries")
    print("5. Polynomial kernels create boundaries based on polynomial curves")
    print("6. Support vectors are the critical points that define the boundary")
    print("7. The margin represents the confidence region around the decision boundary")
```

### Explanation
**Decision Boundary Components:**
1. **Support Vectors**: Critical points that define the decision boundary
2. **Decision Function**: Continuous values indicating distance from boundary
3. **Margins**: Region between decision boundary and support vectors
4. **Kernel Transformation**: Maps data to higher-dimensional space

**Visualization Elements:**
- **Contour Plots**: Show decision regions and boundaries
- **Color Mapping**: Represents decision function values
- **Support Vector Highlighting**: Shows influential training points
- **Margin Lines**: Illustrate the separation margin

### Use Cases
- **Model Understanding**: Visualize how SVM makes decisions
- **Parameter Tuning**: See effects of C, gamma, and kernel parameters
- **Feature Engineering**: Understand which features drive decisions
- **Model Debugging**: Identify overfitting or underfitting patterns
- **Educational Purposes**: Explain SVM concepts visually

### Best Practices
1. **2D Visualization**: Use dimensionality reduction for higher dimensions
2. **Feature Scaling**: Always standardize features before SVM training
3. **Mesh Resolution**: Balance detail with computational efficiency
4. **Color Schemes**: Use intuitive color mappings for clarity
5. **Support Vector Emphasis**: Highlight these critical points

### Pitfalls
- **High-Dimensional Data**: Can't directly visualize >2D decision boundaries
- **Large Datasets**: Mesh grid computation becomes expensive
- **Memory Constraints**: Fine mesh grids require significant memory
- **Misleading 2D Projections**: May not represent true high-D relationships
- **Kernel Parameter Sensitivity**: Small changes can dramatically alter boundaries

### Debugging
```python
# Check support vector distribution
print(f"Support vectors per class: {np.bincount(y[svm.support_])}")
print(f"Total support vectors: {len(svm.support_vectors_)}")
print(f"Percentage of data as SV: {len(svm.support_vectors_) / len(X) * 100:.1f}%")

# Analyze decision function statistics
decision_values = svm.decision_function(X_scaled)
print(f"Decision function range: [{decision_values.min():.2f}, {decision_values.max():.2f}]")
```

### Optimization
```python
# Memory-efficient mesh grid
def efficient_mesh_grid(X, max_points=10000):
    """Create memory-efficient mesh grid"""
    range_x = X[:, 0].max() - X[:, 0].min()
    range_y = X[:, 1].max() - X[:, 1].min()
    
    # Calculate step size based on desired number of points
    total_range = range_x * range_y
    step_size = np.sqrt(total_range / max_points)
    
    return create_mesh_grid(X, step_size)

# Fast prediction for visualization
from sklearn.base import clone

def fast_boundary_plot(X, y, svm_model, resolution=100):
    """Fast decision boundary plotting"""
    # Create coarse mesh for speed
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    return xx, yy, Z
```

---

## Question 10

**Implement dimensionality reduction using PCA with Scikit-Learn and visualize the result.**

### Theory
Principal Component Analysis (PCA) is a dimensionality reduction technique that finds the directions (principal components) along which data varies the most. It projects high-dimensional data onto lower-dimensional subspace while preserving maximum variance, making it ideal for visualization, noise reduction, and feature extraction.

### Code Example
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

def comprehensive_pca_analysis(X, y, feature_names=None, target_names=None, 
                              dataset_name="Dataset"):
    """
    Comprehensive PCA analysis with multiple visualizations
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    y : array-like, shape (n_samples,)
        Target labels
    feature_names : list, optional
        Names of features
    target_names : list, optional
        Names of target classes
    dataset_name : str
        Name of the dataset for titles
    """
    
    print(f"=== PCA ANALYSIS: {dataset_name} ===\n")
    print(f"Original data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Data standardized: mean = {X_scaled.mean():.2e}, std = {X_scaled.std():.2f}")
    
    # 1. Determine optimal number of components
    print(f"\n1. EXPLAINED VARIANCE ANALYSIS")
    print("-" * 40)
    
    # Fit PCA with all components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Calculate cumulative explained variance
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find components needed for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
    
    print(f"Components for 95% variance: {n_components_95}")
    print(f"Components for 99% variance: {n_components_99}")
    print(f"Total components: {len(explained_variance_ratio)}")
    
    # 2. Visualization of explained variance
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual explained variance
    axes[0].bar(range(1, min(21, len(explained_variance_ratio) + 1)), 
               explained_variance_ratio[:20])
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Individual Explained Variance by Component')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative explained variance
    axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    axes[1].axhline(y=0.99, color='g', linestyle='--', label='99% threshold')
    axes[1].axvline(x=n_components_95, color='r', linestyle=':', alpha=0.7)
    axes[1].axvline(x=n_components_99, color='g', linestyle=':', alpha=0.7)
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'PCA Variance Analysis - {dataset_name}')
    plt.tight_layout()
    plt.show()
    
    # 3. 2D and 3D visualizations
    print(f"\n2. DIMENSIONALITY REDUCTION VISUALIZATIONS")
    print("-" * 40)
    
    # 2D PCA
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    # 3D PCA
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    
    print(f"2D PCA explained variance: {pca_2d.explained_variance_ratio_.sum():.4f}")
    print(f"3D PCA explained variance: {pca_3d.explained_variance_ratio_.sum():.4f}")
    
    # Create visualizations
    fig = plt.figure(figsize=(18, 6))
    
    # Original data (first 2 features)
    ax1 = fig.add_subplot(131)
    if len(np.unique(y)) <= 10:  # Reasonable number of classes for coloring
        scatter = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, 
                            cmap='tab10', alpha=0.7, s=50)
        if target_names:
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=plt.cm.tab10(i/len(target_names)), 
                                        markersize=8, label=target_names[i])
                             for i in range(len(target_names))]
            ax1.legend(handles=legend_elements, loc='best')
    else:
        ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.7, s=50)
    
    ax1.set_xlabel('Original Feature 1' if not feature_names else feature_names[0])
    ax1.set_ylabel('Original Feature 2' if not feature_names else feature_names[1])
    ax1.set_title('Original Data (First 2 Features)')
    ax1.grid(True, alpha=0.3)
    
    # 2D PCA
    ax2 = fig.add_subplot(132)
    if len(np.unique(y)) <= 10:
        scatter = ax2.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, 
                            cmap='tab10', alpha=0.7, s=50)
        if target_names:
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=plt.cm.tab10(i/len(target_names)), 
                                        markersize=8, label=target_names[i])
                             for i in range(len(target_names))]
            ax2.legend(handles=legend_elements, loc='best')
    else:
        ax2.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.7, s=50)
    
    ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.3f})')
    ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.3f})')
    ax2.set_title('2D PCA Projection')
    ax2.grid(True, alpha=0.3)
    
    # 3D PCA
    ax3 = fig.add_subplot(133, projection='3d')
    if len(np.unique(y)) <= 10:
        scatter = ax3.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                            c=y, cmap='tab10', alpha=0.7, s=50)
    else:
        ax3.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], alpha=0.7, s=50)
    
    ax3.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.3f})')
    ax3.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.3f})')
    ax3.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.3f})')
    ax3.set_title('3D PCA Projection')
    
    plt.suptitle(f'PCA Visualizations - {dataset_name}')
    plt.tight_layout()
    plt.show()
    
    # 4. Component analysis
    print(f"\n3. PRINCIPAL COMPONENT ANALYSIS")
    print("-" * 40)
    
    # Feature contributions to first few components
    components_df = pd.DataFrame(
        pca_3d.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=feature_names if feature_names else [f'Feature_{i}' for i in range(X.shape[1])]
    )
    
    print("Top feature contributions to first 3 components:")
    for pc in ['PC1', 'PC2', 'PC3']:
        print(f"\n{pc}:")
        pc_contributions = components_df[pc].abs().sort_values(ascending=False)
        print(pc_contributions.head())
    
    # Visualize component loadings
    if feature_names and len(feature_names) <= 20:  # Only if manageable number of features
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
            pc_loadings = components_df[pc].sort_values()
            axes[i].barh(range(len(pc_loadings)), pc_loadings.values)
            axes[i].set_yticks(range(len(pc_loadings)))
            axes[i].set_yticklabels(pc_loadings.index, fontsize=8)
            axes[i].set_xlabel('Loading')
            axes[i].set_title(f'{pc} Loadings')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Principal Component Loadings')
        plt.tight_layout()
        plt.show()
    
    return {
        'pca_2d': pca_2d,
        'pca_3d': pca_3d,
        'X_pca_2d': X_pca_2d,
        'X_pca_3d': X_pca_3d,
        'explained_variance_2d': pca_2d.explained_variance_ratio_.sum(),
        'explained_variance_3d': pca_3d.explained_variance_ratio_.sum(),
        'n_components_95': n_components_95,
        'n_components_99': n_components_99
    }

def pca_for_machine_learning(X, y):
    """
    Demonstrate PCA as preprocessing step for machine learning
    """
    print(f"\n=== PCA FOR MACHINE LEARNING ===")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different numbers of components
    component_counts = [2, 5, 10, min(20, X.shape[1]), X.shape[1]]
    results = []
    
    for n_components in component_counts:
        if n_components > X.shape[1]:
            continue
            
        if n_components == X.shape[1]:
            # No PCA (original features)
            X_train_pca = X_train_scaled
            X_test_pca = X_test_scaled
            explained_var = 1.0
        else:
            # Apply PCA
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            explained_var = pca.explained_variance_ratio_.sum()
        
        # Train classifier
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_pca, y_train)
        
        # Evaluate
        train_accuracy = clf.score(X_train_pca, y_train)
        test_accuracy = clf.score(X_test_pca, y_test)
        
        results.append({
            'n_components': n_components,
            'explained_variance': explained_var,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting': train_accuracy - test_accuracy
        })
        
        print(f"Components: {n_components:2d}, "
              f"Variance: {explained_var:.3f}, "
              f"Train Acc: {train_accuracy:.3f}, "
              f"Test Acc: {test_accuracy:.3f}")
    
    # Visualize results
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy vs components
    axes[0].plot(results_df['n_components'], results_df['train_accuracy'], 
                'o-', label='Training Accuracy')
    axes[0].plot(results_df['n_components'], results_df['test_accuracy'], 
                'o-', label='Test Accuracy')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Number of PCA Components')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Explained variance vs accuracy
    axes[1].scatter(results_df['explained_variance'], results_df['test_accuracy'], 
                   s=100, alpha=0.7)
    for i, row in results_df.iterrows():
        axes[1].annotate(f"{row['n_components']}", 
                        (row['explained_variance'], row['test_accuracy']),
                        xytext=(5, 5), textcoords='offset points')
    axes[1].set_xlabel('Explained Variance Ratio')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Test Accuracy vs Explained Variance')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# Main demonstration
def main_demonstration():
    """Main demonstration with multiple datasets"""
    
    print("=== PCA COMPREHENSIVE DEMONSTRATION ===\n")
    
    # 1. Iris Dataset (classic example)
    print("Loading Iris dataset...")
    iris = load_iris()
    iris_results = comprehensive_pca_analysis(
        iris.data, iris.target, 
        iris.feature_names, iris.target_names, 
        "Iris Dataset"
    )
    
    # Machine learning with PCA
    pca_ml_results = pca_for_machine_learning(iris.data, iris.target)
    
    # 2. Breast Cancer Dataset (higher dimensional)
    print("\n" + "="*50)
    print("Loading Breast Cancer dataset...")
    cancer = load_breast_cancer()
    cancer_results = comprehensive_pca_analysis(
        cancer.data, cancer.target,
        cancer.feature_names, cancer.target_names,
        "Breast Cancer Dataset"
    )
    
    print(f"\n=== SUMMARY ===")
    print("PCA Applications Demonstrated:")
    print("1. Dimensionality reduction for visualization")
    print("2. Feature extraction for machine learning")
    print("3. Data compression and noise reduction")
    print("4. Principal component interpretation")

if __name__ == "__main__":
    main_demonstration()
```

### Explanation
**PCA Workflow:**
1. **Standardization**: Scale features to unit variance
2. **Covariance Matrix**: Compute feature covariances
3. **Eigendecomposition**: Find eigenvectors (components) and eigenvalues
4. **Component Selection**: Choose components explaining desired variance
5. **Transformation**: Project data onto selected components

**Key Concepts:**
- **Principal Components**: Orthogonal directions of maximum variance
- **Explained Variance**: Proportion of total variance captured
- **Loadings**: Feature contributions to each component
- **Scree Plot**: Visualization of explained variance by component

### Use Cases
- **Visualization**: Reduce high-D data to 2D/3D for plotting
- **Feature Extraction**: Create new features from original ones
- **Noise Reduction**: Remove low-variance components
- **Data Compression**: Reduce storage requirements
- **Preprocessing**: Improve ML model performance

### Best Practices
1. **Standardization**: Always scale features before PCA
2. **Variance Threshold**: Choose components explaining 95%+ variance
3. **Interpretation**: Analyze loadings to understand components
4. **Cross-Validation**: Validate component selection on test data
5. **Domain Knowledge**: Consider interpretability of components

### Pitfalls
- **Information Loss**: Discarding components loses some information
- **Linear Assumptions**: PCA finds only linear relationships
- **Scaling Sensitivity**: Results depend heavily on feature scaling
- **Interpretability**: Components may not have clear meaning
- **Outlier Sensitivity**: Extreme values can skew components

### Debugging
```python
# Check for scaling issues
print(f"Feature means: {X.mean(axis=0)[:5]}")
print(f"Feature stds: {X.std(axis=0)[:5]}")

# Verify component orthogonality
components = pca.components_
orthogonality = np.abs(np.dot(components, components.T) 
                      - np.eye(len(components)))
print(f"Max off-diagonal: {orthogonality.max():.2e}")

# Check reconstruction quality
X_reconstructed = pca.inverse_transform(pca.transform(X_scaled))
mse = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"Reconstruction MSE: {mse:.6f}")
```

### Optimization
```python
# Memory-efficient PCA for large datasets
from sklearn.decomposition import IncrementalPCA

# Process data in batches
incremental_pca = IncrementalPCA(n_components=50, batch_size=1000)
for batch in np.array_split(X_large, n_batches):
    incremental_pca.partial_fit(batch)

X_transformed = incremental_pca.transform(X_large)

# Fast approximation for very large datasets
from sklearn.decomposition import TruncatedSVD

# Faster than PCA for sparse matrices
svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X_sparse)
```

---

## Question 11

**Create a clustering analysis on a dataset using Scikit-Learnâ€™s DBSCAN method.**

### Theory
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points in high-density areas and marks points in low-density areas as outliers. Unlike K-means, DBSCAN doesn't require specifying the number of clusters beforehand and can find clusters of arbitrary shapes.

### Code Example
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def generate_test_datasets():
    """Generate various datasets to test DBSCAN performance"""
    
    datasets = {}
    
    # 1. Gaussian blobs (easy case)
    X_blobs, y_blobs = make_blobs(
        n_samples=300, centers=4, cluster_std=0.6, 
        random_state=42, center_box=(-10, 10)
    )
    datasets['Gaussian Blobs'] = (X_blobs, y_blobs)
    
    # 2. Circles (non-linear clusters)
    X_circles, y_circles = make_circles(
        n_samples=300, noise=0.1, factor=0.6, random_state=42
    )
    datasets['Concentric Circles'] = (X_circles, y_circles)
    
    # 3. Moons (crescent shapes)
    X_moons, y_moons = make_moons(
        n_samples=300, noise=0.15, random_state=42
    )
    datasets['Moons'] = (X_moons, y_moons)
    
    # 4. Complex shapes with noise
    np.random.seed(42)
    # Create three different density regions
    cluster1 = np.random.normal([2, 2], 0.5, (100, 2))
    cluster2 = np.random.normal([-2, -2], 0.3, (80, 2))
    cluster3 = np.random.normal([2, -2], 0.7, (120, 2))
    noise = np.random.uniform(-4, 4, (50, 2))
    
    X_complex = np.vstack([cluster1, cluster2, cluster3, noise])
    y_complex = np.hstack([
        np.zeros(100), np.ones(80), np.full(120, 2), np.full(50, -1)
    ])  # -1 for noise points
    
    datasets['Complex with Noise'] = (X_complex, y_complex)
    
    return datasets

def find_optimal_dbscan_parameters(X, k_neighbors=4, plot=True):
    """
    Find optimal DBSCAN parameters using k-distance graph
    
    Parameters:
    -----------
    X : array-like
        Input data
    k_neighbors : int
        Number of neighbors for k-distance calculation
    plot : bool
        Whether to plot the k-distance graph
    """
    
    print(f"=== OPTIMAL PARAMETER SELECTION ===")
    print(f"Dataset shape: {X.shape}")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate k-distances
    neighbors = NearestNeighbors(n_neighbors=k_neighbors)
    neighbors.fit(X_scaled)
    distances, indices = neighbors.kneighbors(X_scaled)
    
    # Sort distances to k-th nearest neighbor
    k_distances = distances[:, k_neighbors-1]
    k_distances_sorted = np.sort(k_distances)[::-1]
    
    if plot:
        plt.figure(figsize=(12, 5))
        
        # K-distance plot
        plt.subplot(1, 2, 1)
        plt.plot(range(len(k_distances_sorted)), k_distances_sorted, 'b-')
        plt.xlabel('Points (sorted by distance)')
        plt.ylabel(f'{k_neighbors}-th Nearest Neighbor Distance')
        plt.title('K-Distance Graph for Epsilon Selection')
        plt.grid(True, alpha=0.3)
        
        # Add suggested epsilon (elbow point approximation)
        # Find the point with maximum curvature
        diff1 = np.diff(k_distances_sorted)
        diff2 = np.diff(diff1)
        if len(diff2) > 10:
            elbow_idx = np.argmax(diff2[:len(diff2)//2]) + 1
            suggested_eps = k_distances_sorted[elbow_idx]
            plt.axhline(y=suggested_eps, color='r', linestyle='--', 
                       label=f'Suggested ε = {suggested_eps:.3f}')
            plt.legend()
        
        # Distribution of distances
        plt.subplot(1, 2, 2)
        plt.hist(k_distances, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel(f'{k_neighbors}-th Nearest Neighbor Distance')
        plt.ylabel('Frequency')
        plt.title('Distribution of K-Distances')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Statistical suggestions
    suggested_eps = np.percentile(k_distances, 90)  # 90th percentile
    median_eps = np.median(k_distances)
    mean_eps = np.mean(k_distances)
    
    print(f"K-distance statistics (k={k_neighbors}):")
    print(f"  Mean: {mean_eps:.4f}")
    print(f"  Median: {median_eps:.4f}")
    print(f"  90th percentile: {suggested_eps:.4f}")
    print(f"  Suggested epsilon range: [{median_eps:.4f}, {suggested_eps:.4f}]")
    print(f"  Suggested min_samples: {k_neighbors}")
    
    return {
        'suggested_eps': suggested_eps,
        'median_eps': median_eps,
        'mean_eps': mean_eps,
        'suggested_min_samples': k_neighbors,
        'k_distances': k_distances_sorted
    }

def comprehensive_dbscan_analysis(X, y_true=None, dataset_name="Dataset"):
    """
    Comprehensive DBSCAN analysis with parameter tuning and evaluation
    """
    
    print(f"\n=== DBSCAN ANALYSIS: {dataset_name} ===")
    print("-" * 50)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal parameters
    param_info = find_optimal_dbscan_parameters(X_scaled, plot=True)
    
    # Test different parameter combinations
    eps_values = np.linspace(param_info['median_eps'], 
                            param_info['suggested_eps'], 5)
    min_samples_values = [3, 5, 7, 10]
    
    results = []
    
    print(f"\nTesting parameter combinations:")
    print(f"Epsilon values: {eps_values}")
    print(f"Min_samples values: {min_samples_values}")
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            # Fit DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred = dbscan.fit_predict(X_scaled)
            
            # Calculate metrics
            n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
            n_noise = list(y_pred).count(-1)
            noise_ratio = n_noise / len(y_pred)
            
            # Silhouette score (only if we have clusters)
            if n_clusters > 1 and n_noise < len(y_pred):
                # Remove noise points for silhouette calculation
                mask = y_pred != -1
                if np.sum(mask) > 1 and len(np.unique(y_pred[mask])) > 1:
                    silhouette = silhouette_score(X_scaled[mask], y_pred[mask])
                else:
                    silhouette = -1
            else:
                silhouette = -1
            
            # Adjusted Rand Index (if ground truth available)
            if y_true is not None:
                ari = adjusted_rand_score(y_true, y_pred)
            else:
                ari = None
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': noise_ratio,
                'silhouette_score': silhouette,
                'ari': ari,
                'labels': y_pred.copy()
            })
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Find best parameters based on different criteria
    valid_results = results_df[results_df['silhouette_score'] > -1]
    
    if len(valid_results) > 0:
        # Best by silhouette score
        best_silhouette = valid_results.loc[valid_results['silhouette_score'].idxmax()]
        
        # Best by ARI (if available)
        if y_true is not None:
            best_ari = valid_results.loc[valid_results['ari'].idxmax()]
        else:
            best_ari = best_silhouette
        
        print(f"\nBest parameters by silhouette score:")
        print(f"  eps={best_silhouette['eps']:.4f}, min_samples={best_silhouette['min_samples']}")
        print(f"  Clusters: {best_silhouette['n_clusters']}, Noise: {best_silhouette['n_noise']}")
        print(f"  Silhouette: {best_silhouette['silhouette_score']:.4f}")
        
        if y_true is not None:
            print(f"\nBest parameters by ARI:")
            print(f"  eps={best_ari['eps']:.4f}, min_samples={best_ari['min_samples']}")
            print(f"  ARI: {best_ari['ari']:.4f}")
    
    # Visualize parameter sensitivity
    visualize_parameter_sensitivity(results_df, dataset_name)
    
    # Detailed analysis with best parameters
    if len(valid_results) > 0:
        best_params = best_silhouette
        detailed_dbscan_analysis(X_scaled, X, best_params, y_true, dataset_name)
    
    return results_df

def visualize_parameter_sensitivity(results_df, dataset_name):
    """Visualize how DBSCAN parameters affect clustering results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Number of clusters vs parameters
    pivot_clusters = results_df.pivot_table(
        values='n_clusters', index='eps', columns='min_samples', aggfunc='mean'
    )
    
    sns.heatmap(pivot_clusters, annot=True, fmt='.0f', cmap='viridis', ax=axes[0,0])
    axes[0,0].set_title('Number of Clusters')
    axes[0,0].set_xlabel('Min Samples')
    axes[0,0].set_ylabel('Epsilon')
    
    # 2. Noise ratio vs parameters
    pivot_noise = results_df.pivot_table(
        values='noise_ratio', index='eps', columns='min_samples', aggfunc='mean'
    )
    
    sns.heatmap(pivot_noise, annot=True, fmt='.2f', cmap='Reds', ax=axes[0,1])
    axes[0,1].set_title('Noise Ratio')
    axes[0,1].set_xlabel('Min Samples')
    axes[0,1].set_ylabel('Epsilon')
    
    # 3. Silhouette score vs parameters
    valid_results = results_df[results_df['silhouette_score'] > -1]
    if len(valid_results) > 0:
        pivot_silhouette = valid_results.pivot_table(
            values='silhouette_score', index='eps', columns='min_samples', aggfunc='mean'
        )
        
        sns.heatmap(pivot_silhouette, annot=True, fmt='.3f', cmap='Blues', ax=axes[1,0])
        axes[1,0].set_title('Silhouette Score')
        axes[1,0].set_xlabel('Min Samples')
        axes[1,0].set_ylabel('Epsilon')
    
    # 4. Parameter trends
    axes[1,1].scatter(results_df['eps'], results_df['n_clusters'], 
                     c=results_df['min_samples'], cmap='viridis', alpha=0.7)
    axes[1,1].set_xlabel('Epsilon')
    axes[1,1].set_ylabel('Number of Clusters')
    axes[1,1].set_title('Clusters vs Epsilon (colored by min_samples)')
    colorbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
    colorbar.set_label('Min Samples')
    
    plt.suptitle(f'DBSCAN Parameter Sensitivity - {dataset_name}')
    plt.tight_layout()
    plt.show()

def detailed_dbscan_analysis(X_scaled, X_original, best_params, y_true, dataset_name):
    """Detailed analysis of DBSCAN results with best parameters"""
    
    print(f"\n=== DETAILED ANALYSIS WITH BEST PARAMETERS ===")
    
    # Apply DBSCAN with best parameters
    dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
    y_pred = dbscan.fit_predict(X_scaled)
    
    # Analyze results
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = list(y_pred).count(-1)
    cluster_sizes = Counter(y_pred)
    
    print(f"Final Results:")
    print(f"  Epsilon: {best_params['eps']:.4f}")
    print(f"  Min samples: {best_params['min_samples']}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    print(f"  Noise ratio: {n_noise/len(y_pred):.3f}")
    
    print(f"\nCluster sizes:")
    for cluster_id, size in sorted(cluster_sizes.items()):
        if cluster_id == -1:
            print(f"  Noise: {size} points")
        else:
            print(f"  Cluster {cluster_id}: {size} points")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original data
    if y_true is not None:
        scatter = axes[0].scatter(X_original[:, 0], X_original[:, 1], 
                                c=y_true, cmap='tab10', alpha=0.7, s=50)
        axes[0].set_title('Ground Truth')
    else:
        axes[0].scatter(X_original[:, 0], X_original[:, 1], 
                       c='blue', alpha=0.7, s=50)
        axes[0].set_title('Original Data')
    
    # DBSCAN results
    scatter = axes[1].scatter(X_original[:, 0], X_original[:, 1], 
                            c=y_pred, cmap='tab10', alpha=0.7, s=50)
    
    # Highlight noise points
    noise_mask = y_pred == -1
    if np.any(noise_mask):
        axes[1].scatter(X_original[noise_mask, 0], X_original[noise_mask, 1], 
                       c='red', marker='x', s=100, alpha=0.8, label='Noise')
        axes[1].legend()
    
    axes[1].set_title(f'DBSCAN Results\n(ε={best_params["eps"]:.3f}, min_samples={best_params["min_samples"]})')
    
    # Core points analysis
    core_samples_mask = np.zeros_like(y_pred, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    
    # Plot core vs boundary points
    axes[2].scatter(X_original[~core_samples_mask, 0], X_original[~core_samples_mask, 1],
                   c=y_pred[~core_samples_mask], cmap='tab10', alpha=0.3, s=30, 
                   label='Boundary')
    axes[2].scatter(X_original[core_samples_mask, 0], X_original[core_samples_mask, 1],
                   c=y_pred[core_samples_mask], cmap='tab10', alpha=0.8, s=50, 
                   label='Core')
    
    if np.any(noise_mask):
        axes[2].scatter(X_original[noise_mask, 0], X_original[noise_mask, 1], 
                       c='red', marker='x', s=100, alpha=0.8, label='Noise')
    
    axes[2].set_title('Core vs Boundary Points')
    axes[2].legend()
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    plt.suptitle(f'DBSCAN Analysis - {dataset_name}')
    plt.tight_layout()
    plt.show()
    
    # Performance metrics
    if y_true is not None:
        ari = adjusted_rand_score(y_true, y_pred)
        print(f"\nPerformance Metrics:")
        print(f"  Adjusted Rand Index: {ari:.4f}")
    
    if n_clusters > 1 and n_noise < len(y_pred):
        mask = y_pred != -1
        if np.sum(mask) > 1:
            silhouette = silhouette_score(X_scaled[mask], y_pred[mask])
            print(f"  Silhouette Score: {silhouette:.4f}")

def compare_clustering_algorithms(X, y_true=None, dataset_name="Dataset"):
    """Compare DBSCAN with other clustering algorithms"""
    
    print(f"\n=== CLUSTERING ALGORITHM COMPARISON ===")
    
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine number of clusters for comparison
    if y_true is not None:
        n_clusters_true = len(np.unique(y_true))
    else:
        n_clusters_true = 3  # Default assumption
    
    # Define algorithms
    algorithms = {
        'DBSCAN': DBSCAN(eps=0.3, min_samples=5),
        'K-Means': KMeans(n_clusters=n_clusters_true, random_state=42),
        'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters_true),
        'Gaussian Mixture': GaussianMixture(n_components=n_clusters_true, random_state=42)
    }
    
    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (name, algorithm) in enumerate(algorithms.items()):
        # Fit algorithm
        if name == 'Gaussian Mixture':
            y_pred = algorithm.fit_predict(X_scaled)
        else:
            y_pred = algorithm.fit_predict(X_scaled)
        
        # Calculate metrics
        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        n_noise = list(y_pred).count(-1) if -1 in y_pred else 0
        
        # Silhouette score
        if n_clusters > 1:
            if n_noise > 0:
                mask = y_pred != -1
                if np.sum(mask) > 1 and len(np.unique(y_pred[mask])) > 1:
                    silhouette = silhouette_score(X_scaled[mask], y_pred[mask])
                else:
                    silhouette = -1
            else:
                silhouette = silhouette_score(X_scaled, y_pred)
        else:
            silhouette = -1
        
        # ARI
        if y_true is not None:
            ari = adjusted_rand_score(y_true, y_pred)
        else:
            ari = None
        
        results[name] = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': silhouette,
            'ari': ari,
            'labels': y_pred
        }
        
        # Plot results
        scatter = axes[i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10', alpha=0.7, s=50)
        
        if n_noise > 0:
            noise_mask = y_pred == -1
            axes[i].scatter(X[noise_mask, 0], X[noise_mask, 1], 
                           c='red', marker='x', s=100, alpha=0.8)
        
        title = f'{name}\nClusters: {n_clusters}'
        if silhouette > -1:
            title += f', Silhouette: {silhouette:.3f}'
        if ari is not None:
            title += f', ARI: {ari:.3f}'
        
        axes[i].set_title(title)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
    
    plt.suptitle(f'Clustering Algorithm Comparison - {dataset_name}')
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print(f"\nComparison Results:")
    print(f"{'Algorithm':<15} {'Clusters':<8} {'Noise':<6} {'Silhouette':<12} {'ARI':<8}")
    print("-" * 60)
    
    for name, metrics in results.items():
        sil_str = f"{metrics['silhouette']:.4f}" if metrics['silhouette'] > -1 else "N/A"
        ari_str = f"{metrics['ari']:.4f}" if metrics['ari'] is not None else "N/A"
        print(f"{name:<15} {metrics['n_clusters']:<8} {metrics['n_noise']:<6} {sil_str:<12} {ari_str:<8}")
    
    return results

def main_demonstration():
    """Main demonstration of DBSCAN clustering"""
    
    print("=== COMPREHENSIVE DBSCAN CLUSTERING DEMONSTRATION ===\n")
    
    # Generate test datasets
    datasets = generate_test_datasets()
    
    all_results = {}
    
    # Analyze each dataset
    for dataset_name, (X, y_true) in datasets.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING: {dataset_name}")
        print(f"{'='*60}")
        
        # Comprehensive analysis
        results = comprehensive_dbscan_analysis(X, y_true, dataset_name)
        all_results[dataset_name] = results
        
        # Compare with other algorithms
        comparison = compare_clustering_algorithms(X, y_true, dataset_name)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("Key DBSCAN Insights:")
    print("1. DBSCAN excels at finding clusters of arbitrary shapes")
    print("2. It automatically identifies noise/outliers")
    print("3. No need to specify number of clusters beforehand")
    print("4. Performance depends heavily on eps and min_samples parameters")
    print("5. Works best with clusters of similar density")
    print("6. Use k-distance graph to select optimal epsilon value")

if __name__ == "__main__":
    main_demonstration()
```

### Explanation
**DBSCAN Algorithm Steps:**
1. **Core Points**: Points with at least `min_samples` neighbors within `eps` distance
2. **Border Points**: Non-core points within `eps` distance of a core point
3. **Noise Points**: Points that are neither core nor border points
4. **Cluster Formation**: Core points and their neighbors form clusters
5. **Density Connection**: Points are density-connected if there's a path of core points

**Key Parameters:**
- `eps` (epsilon): Maximum distance for neighborhood definition
- `min_samples`: Minimum points required to form a dense region

### Use Cases
- **Outlier Detection**: Automatically identifies noise points
- **Arbitrary Shapes**: Handles non-spherical clusters unlike K-means
- **Unknown Cluster Count**: No need to specify number of clusters
- **Geospatial Analysis**: Crime hotspots, urban planning
- **Image Processing**: Object detection and segmentation

### Best Practices
1. **Data Scaling**: Standardize features when scales differ significantly
2. **Parameter Selection**: Use k-distance graph for epsilon selection
3. **Domain Knowledge**: Consider expected cluster density
4. **Evaluation**: Use silhouette score and visual inspection
5. **Preprocessing**: Remove irrelevant features that add noise

### Pitfalls
- **Parameter Sensitivity**: Small changes in eps can drastically alter results
- **Varying Densities**: Struggles with clusters of different densities
- **High Dimensions**: Curse of dimensionality affects distance calculations
- **Large Datasets**: Computational complexity can be prohibitive
- **Memory Usage**: Distance calculations require significant memory

### Debugging
```python
# Analyze core samples
core_mask = np.zeros_like(labels, dtype=bool)
core_mask[dbscan.core_sample_indices_] = True

print(f"Core samples: {np.sum(core_mask)}")
print(f"Border samples: {np.sum((labels != -1) & ~core_mask)}")
print(f"Noise samples: {np.sum(labels == -1)}")

# Check neighborhood sizes
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(radius=eps).fit(X)
distances, indices = nbrs.radius_neighbors(X)
neighborhood_sizes = [len(idx) for idx in indices]
print(f"Average neighborhood size: {np.mean(neighborhood_sizes):.2f}")
```

### Optimization
```python
# Memory-efficient DBSCAN for large datasets
from sklearn.cluster import DBSCAN

# Use algorithm='ball_tree' or 'kd_tree' for better performance
dbscan_optimized = DBSCAN(
    eps=0.5, 
    min_samples=5,
    algorithm='ball_tree',  # or 'kd_tree'
    leaf_size=30,
    n_jobs=-1  # Use all cores
)

# For very large datasets, consider using approximate methods
from sklearn.cluster import OPTICS  # Similar to DBSCAN but more robust

optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
clustering = optics.fit_predict(X)
```

---

## Question 12

**How do you save a trained Scikit-Learn model to disk and load it back for later use?**

### Theory
Model persistence in Scikit-Learn involves serializing trained models to disk for later use in production, sharing, or continued analysis. The primary methods are using Python's pickle module or joblib (recommended for sklearn models due to better performance with NumPy arrays). Proper model persistence ensures reproducibility and enables deployment workflows.

### Code Example
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelPersistenceManager:
    """
    Comprehensive model persistence manager with metadata tracking
    """
    
    def __init__(self, base_directory="saved_models"):
        """
        Initialize model persistence manager
        
        Parameters:
        -----------
        base_directory : str
            Base directory for saving models
        """
        self.base_directory = base_directory
        self.ensure_directory_exists()
    
    def ensure_directory_exists(self):
        """Create base directory if it doesn't exist"""
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)
            print(f"Created directory: {self.base_directory}")
    
    def save_model_with_metadata(self, model, model_name, X_train=None, y_train=None, 
                                X_test=None, y_test=None, additional_info=None):
        """
        Save model with comprehensive metadata
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained model to save
        model_name : str
            Name for the model
        X_train, y_train : array-like, optional
            Training data for metadata
        X_test, y_test : array-like, optional
            Test data for performance metrics
        additional_info : dict, optional
            Additional information to save
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}"
        
        # Create model-specific directory
        model_dir = os.path.join(self.base_directory, model_filename)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model using joblib (recommended for sklearn)
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)
        
        # Also save using pickle for compatibility
        pickle_path = os.path.join(model_dir, "model.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Create metadata
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'timestamp': timestamp,
            'sklearn_version': joblib.__version__,
            'model_parameters': model.get_params() if hasattr(model, 'get_params') else None,
            'feature_count': X_train.shape[1] if X_train is not None else None,
            'training_samples': X_train.shape[0] if X_train is not None else None,
            'file_paths': {
                'model_joblib': model_path,
                'model_pickle': pickle_path
            }
        }
        
        # Add performance metrics if test data provided
        if X_test is not None and y_test is not None:
            try:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                metadata['performance'] = {
                    'test_accuracy': accuracy,
                    'test_samples': len(y_test),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
            except Exception as e:
                metadata['performance_error'] = str(e)
        
        # Add feature information if available
        if hasattr(model, 'feature_names_in_'):
            metadata['feature_names'] = model.feature_names_in_.tolist()
        
        # Add pipeline information if it's a pipeline
        if hasattr(model, 'steps'):
            metadata['pipeline_steps'] = [step[0] for step in model.steps]
        
        # Add additional info
        if additional_info:
            metadata['additional_info'] = additional_info
        
        # Save metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Model saved successfully:")
        print(f"  Directory: {model_dir}")
        print(f"  Model (joblib): {model_path}")
        print(f"  Model (pickle): {pickle_path}")
        print(f"  Metadata: {metadata_path}")
        
        return {
            'model_dir': model_dir,
            'model_path': model_path,
            'metadata_path': metadata_path,
            'metadata': metadata
        }
    
    def load_model_with_metadata(self, model_path_or_dir):
        """
        Load model with metadata
        
        Parameters:
        -----------
        model_path_or_dir : str
            Path to model file or directory containing model
        
        Returns:
        --------
        dict : Dictionary containing model and metadata
        """
        
        if os.path.isdir(model_path_or_dir):
            # Directory provided, look for standard files
            model_dir = model_path_or_dir
            joblib_path = os.path.join(model_dir, "model.joblib")
            pickle_path = os.path.join(model_dir, "model.pkl")
            metadata_path = os.path.join(model_dir, "metadata.json")
            
            # Try joblib first
            if os.path.exists(joblib_path):
                model_path = joblib_path
                load_method = 'joblib'
            elif os.path.exists(pickle_path):
                model_path = pickle_path
                load_method = 'pickle'
            else:
                raise FileNotFoundError("No model file found in directory")
        else:
            # File path provided
            model_path = model_path_or_dir
            model_dir = os.path.dirname(model_path)
            metadata_path = os.path.join(model_dir, "metadata.json")
            
            if model_path.endswith('.joblib'):
                load_method = 'joblib'
            else:
                load_method = 'pickle'
        
        # Load model
        try:
            if load_method == 'joblib':
                model = joblib.load(model_path)
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            print(f"Model loaded successfully from: {model_path}")
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
        
        # Load metadata if available
        metadata = None
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Metadata loaded from: {metadata_path}")
            except Exception as e:
                print(f"Warning: Could not load metadata: {str(e)}")
        
        return {
            'model': model,
            'metadata': metadata,
            'model_path': model_path,
            'load_method': load_method
        }
    
    def list_saved_models(self):
        """List all saved models in the base directory"""
        
        if not os.path.exists(self.base_directory):
            print("No saved models directory found.")
            return []
        
        models = []
        for item in os.listdir(self.base_directory):
            item_path = os.path.join(self.base_directory, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        models.append({
                            'directory': item,
                            'path': item_path,
                            'metadata': metadata
                        })
                    except Exception as e:
                        print(f"Error reading metadata for {item}: {str(e)}")
        
        return models
    
    def compare_models(self, model_paths):
        """Compare multiple saved models"""
        
        comparison_data = []
        
        for path in model_paths:
            try:
                loaded = self.load_model_with_metadata(path)
                metadata = loaded['metadata']
                
                if metadata:
                    comparison_data.append({
                        'model_name': metadata.get('model_name', 'Unknown'),
                        'model_type': metadata.get('model_type', 'Unknown'),
                        'timestamp': metadata.get('timestamp', 'Unknown'),
                        'test_accuracy': metadata.get('performance', {}).get('test_accuracy', 'N/A'),
                        'training_samples': metadata.get('training_samples', 'N/A'),
                        'feature_count': metadata.get('feature_count', 'N/A'),
                        'path': path
                    })
            except Exception as e:
                print(f"Error loading model from {path}: {str(e)}")
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print("\nModel Comparison:")
            print(df.to_string(index=False))
            return df
        else:
            print("No models to compare.")
            return None

def demonstrate_basic_persistence():
    """Demonstrate basic model saving and loading"""
    
    print("=== BASIC MODEL PERSISTENCE DEMONSTRATION ===\n")
    
    # Load and prepare data
    data = load_iris()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test model performance
    accuracy = model.score(X_test, y_test)
    print(f"Original model accuracy: {accuracy:.4f}")
    
    # Method 1: Using joblib (recommended for sklearn)
    print("\n1. SAVING WITH JOBLIB")
    joblib_filename = 'iris_model.joblib'
    joblib.dump(model, joblib_filename)
    print(f"Model saved to: {joblib_filename}")
    
    # Load and test
    loaded_model_joblib = joblib.load(joblib_filename)
    loaded_accuracy_joblib = loaded_model_joblib.score(X_test, y_test)
    print(f"Loaded model accuracy (joblib): {loaded_accuracy_joblib:.4f}")
    print(f"Accuracy match: {accuracy == loaded_accuracy_joblib}")
    
    # Method 2: Using pickle
    print("\n2. SAVING WITH PICKLE")
    pickle_filename = 'iris_model.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {pickle_filename}")
    
    # Load and test
    with open(pickle_filename, 'rb') as f:
        loaded_model_pickle = pickle.load(f)
    
    loaded_accuracy_pickle = loaded_model_pickle.score(X_test, y_test)
    print(f"Loaded model accuracy (pickle): {loaded_accuracy_pickle:.4f}")
    print(f"Accuracy match: {accuracy == loaded_accuracy_pickle}")
    
    # File size comparison
    joblib_size = os.path.getsize(joblib_filename)
    pickle_size = os.path.getsize(pickle_filename)
    
    print(f"\nFile size comparison:")
    print(f"  Joblib: {joblib_size} bytes")
    print(f"  Pickle: {pickle_size} bytes")
    print(f"  Joblib is {joblib_size/pickle_size:.2f}x the size of pickle")
    
    # Clean up
    os.remove(joblib_filename)
    os.remove(pickle_filename)

def demonstrate_pipeline_persistence():
    """Demonstrate saving and loading of sklearn pipelines"""
    
    print("\n=== PIPELINE PERSISTENCE DEMONSTRATION ===\n")
    
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    original_accuracy = pipeline.score(X_test, y_test)
    
    print(f"Original pipeline accuracy: {original_accuracy:.4f}")
    print(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")
    
    # Save pipeline
    pipeline_filename = 'cancer_pipeline.joblib'
    joblib.dump(pipeline, pipeline_filename)
    print(f"Pipeline saved to: {pipeline_filename}")
    
    # Load and test pipeline
    loaded_pipeline = joblib.load(pipeline_filename)
    loaded_accuracy = loaded_pipeline.score(X_test, y_test)
    
    print(f"Loaded pipeline accuracy: {loaded_accuracy:.4f}")
    print(f"Accuracy match: {original_accuracy == loaded_accuracy}")
    
    # Test individual pipeline components
    print(f"\nPipeline component verification:")
    
    # Scaler parameters
    original_scaler = pipeline.named_steps['scaler']
    loaded_scaler = loaded_pipeline.named_steps['scaler']
    
    print(f"Scaler means match: {np.allclose(original_scaler.mean_, loaded_scaler.mean_)}")
    print(f"Scaler scales match: {np.allclose(original_scaler.scale_, loaded_scaler.scale_)}")
    
    # Classifier parameters
    original_clf = pipeline.named_steps['classifier']
    loaded_clf = loaded_pipeline.named_steps['classifier']
    
    print(f"Classifier coefs match: {np.allclose(original_clf.coef_, loaded_clf.coef_)}")
    
    # Clean up
    os.remove(pipeline_filename)

def demonstrate_grid_search_persistence():
    """Demonstrate saving and loading of GridSearchCV results"""
    
    print("\n=== GRID SEARCH PERSISTENCE DEMONSTRATION ===\n")
    
    # Load data
    data = load_iris()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and perform grid search
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20]
    }
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    print("Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    test_accuracy = grid_search.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save entire grid search object
    grid_filename = 'iris_grid_search.joblib'
    joblib.dump(grid_search, grid_filename)
    print(f"Grid search saved to: {grid_filename}")
    
    # Load and test
    loaded_grid_search = joblib.load(grid_filename)
    loaded_test_accuracy = loaded_grid_search.score(X_test, y_test)
    
    print(f"Loaded test accuracy: {loaded_test_accuracy:.4f}")
    print(f"Best params match: {grid_search.best_params_ == loaded_grid_search.best_params_}")
    
    # Access CV results
    print(f"\nCV results available: {hasattr(loaded_grid_search, 'cv_results_')}")
    if hasattr(loaded_grid_search, 'cv_results_'):
        cv_results_df = pd.DataFrame(loaded_grid_search.cv_results_)
        print(f"CV results shape: {cv_results_df.shape}")
    
    # Clean up
    os.remove(grid_filename)

def demonstrate_comprehensive_persistence():
    """Demonstrate comprehensive model persistence with metadata"""
    
    print("\n=== COMPREHENSIVE PERSISTENCE DEMONSTRATION ===\n")
    
    # Initialize persistence manager
    manager = ModelPersistenceManager("demo_models")
    
    # Load and prepare data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train multiple models
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
    }
    
    saved_models = []
    
    for model_name, model in models.items():
        print(f"\nTraining and saving {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Save with metadata
        additional_info = {
            'dataset': 'breast_cancer',
            'training_date': datetime.now().isoformat(),
            'notes': f'Baseline {model_name} model for breast cancer classification'
        }
        
        save_info = manager.save_model_with_metadata(
            model=model,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            additional_info=additional_info
        )
        
        saved_models.append(save_info['model_dir'])
    
    # List all saved models
    print(f"\n=== SAVED MODELS LISTING ===")
    saved_model_list = manager.list_saved_models()
    
    for i, model_info in enumerate(saved_model_list):
        metadata = model_info['metadata']
        print(f"\nModel {i+1}:")
        print(f"  Name: {metadata.get('model_name', 'Unknown')}")
        print(f"  Type: {metadata.get('model_type', 'Unknown')}")
        print(f"  Timestamp: {metadata.get('timestamp', 'Unknown')}")
        print(f"  Test Accuracy: {metadata.get('performance', {}).get('test_accuracy', 'N/A')}")
    
    # Compare models
    print(f"\n=== MODEL COMPARISON ===")
    manager.compare_models([model['path'] for model in saved_model_list])
    
    # Load and test a model
    print(f"\n=== MODEL LOADING TEST ===")
    if saved_model_list:
        test_model_path = saved_model_list[0]['path']
        loaded_info = manager.load_model_with_metadata(test_model_path)
        
        model = loaded_info['model']
        metadata = loaded_info['metadata']
        
        print(f"Loaded model type: {type(model).__name__}")
        print(f"Original accuracy: {metadata['performance']['test_accuracy']:.4f}")
        
        # Test loaded model
        new_accuracy = model.score(X_test, y_test)
        print(f"New test accuracy: {new_accuracy:.4f}")
        print(f"Accuracy match: {abs(metadata['performance']['test_accuracy'] - new_accuracy) < 1e-10}")
    
    # Clean up demo directory
    import shutil
    if os.path.exists("demo_models"):
        shutil.rmtree("demo_models")
        print(f"\nCleaned up demo directory")

def production_deployment_example():
    """Example of production model deployment workflow"""
    
    print("\n=== PRODUCTION DEPLOYMENT EXAMPLE ===\n")
    
    # Simulate model training and saving in development
    print("1. DEVELOPMENT PHASE")
    print("-" * 30)
    
    # Train model
    data = load_iris()
    X, y = data.data, data.target
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    model.fit(X, y)
    training_accuracy = model.score(X, y)
    print(f"Training accuracy: {training_accuracy:.4f}")
    
    # Save for production
    production_model_path = 'production_iris_model.joblib'
    
    # Save model with version info
    model_info = {
        'model': model,
        'version': '1.0.0',
        'feature_names': data.feature_names.tolist(),
        'target_names': data.target_names.tolist(),
        'training_accuracy': training_accuracy,
        'sklearn_version': joblib.__version__
    }
    
    joblib.dump(model_info, production_model_path)
    print(f"Production model saved to: {production_model_path}")
    
    # Simulate production loading and inference
    print(f"\n2. PRODUCTION PHASE")
    print("-" * 30)
    
    # Load model in production
    loaded_info = joblib.load(production_model_path)
    production_model = loaded_info['model']
    
    print(f"Loaded model version: {loaded_info['version']}")
    print(f"Expected features: {loaded_info['feature_names']}")
    
    # Simulate new data for prediction
    new_samples = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.2, 2.9, 4.3, 1.3],  # Versicolor
        [7.3, 2.9, 6.3, 1.8]   # Virginica
    ])
    
    # Make predictions
    predictions = production_model.predict(new_samples)
    probabilities = production_model.predict_proba(new_samples)
    
    print(f"\nPredictions for new samples:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        pred_name = loaded_info['target_names'][pred]
        max_prob = prob.max()
        print(f"  Sample {i+1}: {pred_name} (confidence: {max_prob:.3f})")
    
    # Model validation
    validation_accuracy = production_model.score(X, y)
    print(f"\nValidation accuracy: {validation_accuracy:.4f}")
    print(f"Accuracy preserved: {abs(training_accuracy - validation_accuracy) < 1e-10}")
    
    # Clean up
    os.remove(production_model_path)

# Main demonstration
def main_demonstration():
    """Run all persistence demonstrations"""
    
    print("=== COMPREHENSIVE SCIKIT-LEARN MODEL PERSISTENCE GUIDE ===\n")
    
    # Basic persistence
    demonstrate_basic_persistence()
    
    # Pipeline persistence
    demonstrate_pipeline_persistence()
    
    # Grid search persistence
    demonstrate_grid_search_persistence()
    
    # Comprehensive persistence with metadata
    demonstrate_comprehensive_persistence()
    
    # Production deployment example
    production_deployment_example()
    
    print(f"\n{'='*60}")
    print("SUMMARY: MODEL PERSISTENCE BEST PRACTICES")
    print(f"{'='*60}")
    print("1. Use joblib for sklearn models (better performance with NumPy)")
    print("2. Include metadata (version, parameters, performance)")
    print("3. Save preprocessing pipelines together with models")
    print("4. Version your models for production deployment")
    print("5. Validate loaded models against original performance")
    print("6. Consider file size and loading speed trade-offs")
    print("7. Use proper directory structure for model organization")
    print("8. Include feature names and model requirements")

if __name__ == "__main__":
    main_demonstration()
```

### Explanation
**Model Persistence Methods:**
1. **joblib**: Recommended for sklearn models, efficient with NumPy arrays
2. **pickle**: Standard Python serialization, universal but slower for large arrays
3. **Custom formats**: JSON for metadata, specialized formats for specific needs

**Key Components to Save:**
- **Trained Model**: The actual estimator with fitted parameters
- **Preprocessing**: Scalers, encoders, feature selectors
- **Metadata**: Version, parameters, performance metrics, feature names
- **Pipeline**: Complete transformation and modeling workflow

### Use Cases
- **Production Deployment**: Save trained models for serving predictions
- **Model Sharing**: Transfer models between team members or environments
- **Experimentation**: Save and compare different model configurations
- **Backup and Recovery**: Preserve trained models for disaster recovery
- **Version Control**: Track model evolution over time

### Best Practices
1. **Use joblib for sklearn**: Better performance and smaller files
2. **Save Complete Pipelines**: Include all preprocessing steps
3. **Include Metadata**: Version, performance, feature information
4. **Validate After Loading**: Ensure model works correctly after loading
5. **Version Control**: Track model versions and changes
6. **Security**: Be cautious loading models from untrusted sources

### Pitfalls
- **Version Compatibility**: sklearn version differences can cause issues
- **Missing Dependencies**: Ensure all required packages are available
- **File Corruption**: Large files may become corrupted during transfer
- **Security Risks**: Pickle can execute arbitrary code when loading
- **Memory Usage**: Large models may consume significant memory

### Debugging
```python
# Check model integrity after loading
def validate_loaded_model(original_model, loaded_model, X_test, y_test):
    """Validate that loaded model matches original"""
    
    # Compare predictions
    orig_pred = original_model.predict(X_test)
    load_pred = loaded_model.predict(X_test)
    
    predictions_match = np.array_equal(orig_pred, load_pred)
    print(f"Predictions match: {predictions_match}")
    
    # Compare parameters
    if hasattr(original_model, 'get_params'):
        orig_params = original_model.get_params()
        load_params = loaded_model.get_params()
        params_match = orig_params == load_params
        print(f"Parameters match: {params_match}")
    
    return predictions_match

# Check file integrity
import hashlib

def get_file_hash(filename):
    """Get file hash for integrity checking"""
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
```

### Optimization
```python
# Compress models for storage
import joblib

# Save with compression
joblib.dump(model, 'model_compressed.joblib', compress=3)

# Memory-mapped loading for large models
model = joblib.load('large_model.joblib', mmap_mode='r')

# Partial loading for distributed systems
from joblib import load, dump

# Save model components separately
dump(model.named_steps['scaler'], 'scaler.joblib')
dump(model.named_steps['classifier'], 'classifier.joblib')

# Load components as needed
scaler = load('scaler.joblib')
classifier = load('classifier.joblib')
```

---

## Question 13

**How can you implement custom transformers in Scikit-Learn?**

### Theory
Custom transformers in Scikit-Learn allow you to integrate domain-specific data preprocessing steps into sklearn pipelines. By inheriting from BaseEstimator and TransformerMixin, custom transformers become compatible with Pipeline, GridSearchCV, and other sklearn tools. This enables reproducible, reusable data transformations that can be combined with standard ML workflows.

### Code Example
```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_boston, load_iris
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# BASIC CUSTOM TRANSFORMERS
# ============================================================================

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for log transformation of features
    """
    
    def __init__(self, features=None, base='e'):
        """
        Initialize log transformer
        
        Parameters:
        -----------
        features : list, optional
            List of feature indices or names to transform. If None, transform all features.
        base : str or float, default='e'
            Logarithm base ('e' for natural log, 10 for log10, 2 for log2, or any float)
        """
        self.features = features
        self.base = base
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no actual fitting needed for log transform)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, optional
            Target values (ignored)
        
        Returns:
        --------
        self : object
        """
        # Convert to pandas DataFrame if needed for feature selection
        if hasattr(X, 'columns') and self.features is not None:
            if isinstance(self.features[0], str):
                # Feature names provided
                self.feature_indices_ = [X.columns.get_loc(name) for name in self.features]
            else:
                # Feature indices provided
                self.feature_indices_ = self.features
        elif self.features is not None:
            # Numpy array with feature indices
            self.feature_indices_ = self.features
        else:
            # Transform all features
            self.feature_indices_ = list(range(X.shape[1]))
        
        return self
    
    def transform(self, X):
        """
        Apply log transformation
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data to transform
        
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_features)
            Transformed data
        """
        # Check if fitted
        if not hasattr(self, 'feature_indices_'):
            raise ValueError("Transformer must be fitted before transform")
        
        # Convert to numpy array for consistent handling
        X_array = np.array(X, copy=True, dtype=float)
        
        # Apply log transformation to selected features
        for idx in self.feature_indices_:
            # Add small constant to avoid log(0)
            X_array[:, idx] = X_array[:, idx] + 1e-8
            
            if self.base == 'e':
                X_array[:, idx] = np.log(X_array[:, idx])
            elif self.base == 10:
                X_array[:, idx] = np.log10(X_array[:, idx])
            elif self.base == 2:
                X_array[:, idx] = np.log2(X_array[:, idx])
            else:
                X_array[:, idx] = np.log(X_array[:, idx]) / np.log(self.base)
        
        return X_array

class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove outliers using IQR or Z-score method
    """
    
    def __init__(self, method='iqr', threshold=1.5, features=None):
        """
        Initialize outlier remover
        
        Parameters:
        -----------
        method : str, default='iqr'
            Method to use ('iqr' or 'zscore')
        threshold : float, default=1.5
            Threshold for outlier detection (1.5 for IQR, 3.0 for Z-score typically)
        features : list, optional
            Features to consider for outlier detection. If None, use all features.
        """
        self.method = method
        self.threshold = threshold
        self.features = features
    
    def fit(self, X, y=None):
        """
        Fit the outlier detector
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, optional
            Target values (ignored)
        
        Returns:
        --------
        self : object
        """
        X_array = np.array(X)
        
        # Determine features to use
        if self.features is not None:
            self.feature_indices_ = self.features
        else:
            self.feature_indices_ = list(range(X_array.shape[1]))
        
        # Calculate statistics for outlier detection
        if self.method == 'iqr':
            self.q1_ = np.percentile(X_array[:, self.feature_indices_], 25, axis=0)
            self.q3_ = np.percentile(X_array[:, self.feature_indices_], 75, axis=0)
            self.iqr_ = self.q3_ - self.q1_
            self.lower_bound_ = self.q1_ - self.threshold * self.iqr_
            self.upper_bound_ = self.q3_ + self.threshold * self.iqr_
        
        elif self.method == 'zscore':
            self.mean_ = np.mean(X_array[:, self.feature_indices_], axis=0)
            self.std_ = np.std(X_array[:, self.feature_indices_], axis=0)
        
        return self
    
    def transform(self, X):
        """
        Remove outliers from the data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        
        Returns:
        --------
        X_clean : array-like, shape (n_clean_samples, n_features)
            Data with outliers removed
        """
        X_array = np.array(X)
        
        if self.method == 'iqr':
            # Create mask for non-outliers
            mask = np.ones(X_array.shape[0], dtype=bool)
            for i, feature_idx in enumerate(self.feature_indices_):
                feature_data = X_array[:, feature_idx]
                feature_mask = (feature_data >= self.lower_bound_[i]) & \
                              (feature_data <= self.upper_bound_[i])
                mask = mask & feature_mask
        
        elif self.method == 'zscore':
            # Create mask for non-outliers using Z-score
            mask = np.ones(X_array.shape[0], dtype=bool)
            for i, feature_idx in enumerate(self.feature_indices_):
                feature_data = X_array[:, feature_idx]
                z_scores = np.abs((feature_data - self.mean_[i]) / self.std_[i])
                feature_mask = z_scores <= self.threshold
                mask = mask & feature_mask
        
        return X_array[mask]

class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create new features from existing ones
    """
    
    def __init__(self, feature_functions=None, feature_names=None):
        """
        Initialize feature creator
        
        Parameters:
        -----------
        feature_functions : list of functions
            List of functions that take a row of data and return a new feature value
        feature_names : list of str, optional
            Names for the new features
        """
        self.feature_functions = feature_functions or []
        self.feature_names = feature_names or [f'feature_{i}' for i in range(len(self.feature_functions))]
    
    def fit(self, X, y=None):
        """
        Fit the feature creator (no fitting needed)
        """
        return self
    
    def transform(self, X):
        """
        Create new features and append to existing data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        
        Returns:
        --------
        X_extended : array-like, shape (n_samples, n_features + n_new_features)
            Data with new features appended
        """
        X_array = np.array(X)
        new_features = []
        
        for func in self.feature_functions:
            # Apply function to each row
            new_feature = np.array([func(row) for row in X_array])
            new_features.append(new_feature.reshape(-1, 1))
        
        if new_features:
            # Concatenate original features with new features
            X_extended = np.concatenate([X_array] + new_features, axis=1)
        else:
            X_extended = X_array
        
        return X_extended

# ============================================================================
# ADVANCED CUSTOM TRANSFORMERS
# ============================================================================

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to select specific columns from a pandas DataFrame
    """
    
    def __init__(self, attribute_names):
        """
        Initialize DataFrame selector
        
        Parameters:
        -----------
        attribute_names : list
            List of column names to select
        """
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names]

class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Custom missing value imputer with multiple strategies
    """
    
    def __init__(self, strategy='mean', fill_value=None, features=None):
        """
        Initialize missing value imputer
        
        Parameters:
        -----------
        strategy : str, default='mean'
            Strategy to use ('mean', 'median', 'mode', 'constant', 'forward_fill', 'backward_fill')
        fill_value : scalar, default=None
            Value to use for 'constant' strategy
        features : list, optional
            Features to impute. If None, impute all features.
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.features = features
    
    def fit(self, X, y=None):
        """
        Fit the imputer
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            if self.features is not None:
                selected_features = self.features
            else:
                selected_features = self.feature_names_
        else:
            X = pd.DataFrame(X)
            selected_features = X.columns.tolist() if self.features is None else self.features
        
        self.impute_values_ = {}
        
        for feature in selected_features:
            if feature in X.columns:
                if self.strategy == 'mean':
                    self.impute_values_[feature] = X[feature].mean()
                elif self.strategy == 'median':
                    self.impute_values_[feature] = X[feature].median()
                elif self.strategy == 'mode':
                    self.impute_values_[feature] = X[feature].mode().iloc[0] if not X[feature].mode().empty else 0
                elif self.strategy == 'constant':
                    self.impute_values_[feature] = self.fill_value
        
        return self
    
    def transform(self, X):
        """
        Apply imputation
        """
        if isinstance(X, pd.DataFrame):
            X_imputed = X.copy()
        else:
            X_imputed = pd.DataFrame(X).copy()
        
        for feature, value in self.impute_values_.items():
            if feature in X_imputed.columns:
                if self.strategy in ['mean', 'median', 'mode', 'constant']:
                    X_imputed[feature].fillna(value, inplace=True)
                elif self.strategy == 'forward_fill':
                    X_imputed[feature].fillna(method='ffill', inplace=True)
                elif self.strategy == 'backward_fill':
                    X_imputed[feature].fillna(method='bfill', inplace=True)
        
        return X_imputed.values if not isinstance(X, pd.DataFrame) else X_imputed

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Custom categorical encoder with multiple encoding strategies
    """
    
    def __init__(self, strategy='onehot', handle_unknown='error', categories=None):
        """
        Initialize categorical encoder
        
        Parameters:
        -----------
        strategy : str, default='onehot'
            Encoding strategy ('onehot', 'label', 'target', 'binary')
        handle_unknown : str, default='error'
            How to handle unknown categories ('error', 'ignore')
        categories : list, optional
            Categories to use for each feature
        """
        self.strategy = strategy
        self.handle_unknown = handle_unknown
        self.categories = categories
    
    def fit(self, X, y=None):
        """
        Fit the encoder
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        self.feature_categories_ = {}
        self.feature_mappings_ = {}
        
        for column in X.columns:
            if X[column].dtype == 'object' or X[column].dtype.name == 'category':
                unique_values = X[column].unique()
                self.feature_categories_[column] = unique_values
                
                if self.strategy == 'label':
                    self.feature_mappings_[column] = {val: i for i, val in enumerate(unique_values)}
                elif self.strategy == 'target' and y is not None:
                    # Target encoding: replace category with mean target value
                    target_means = X.groupby(column)[y].mean() if hasattr(X, 'groupby') else {}
                    self.feature_mappings_[column] = target_means.to_dict()
        
        return self
    
    def transform(self, X):
        """
        Apply encoding
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_encoded = X.copy()
        
        for column in self.feature_categories_:
            if column in X_encoded.columns:
                if self.strategy == 'onehot':
                    # One-hot encoding
                    dummies = pd.get_dummies(X_encoded[column], prefix=column)
                    X_encoded = pd.concat([X_encoded.drop(column, axis=1), dummies], axis=1)
                
                elif self.strategy == 'label':
                    # Label encoding
                    mapping = self.feature_mappings_[column]
                    if self.handle_unknown == 'ignore':
                        X_encoded[column] = X_encoded[column].map(mapping).fillna(-1)
                    else:
                        X_encoded[column] = X_encoded[column].map(mapping)
                
                elif self.strategy == 'target':
                    # Target encoding
                    mapping = self.feature_mappings_[column]
                    if self.handle_unknown == 'ignore':
                        overall_mean = np.mean(list(mapping.values()))
                        X_encoded[column] = X_encoded[column].map(mapping).fillna(overall_mean)
                    else:
                        X_encoded[column] = X_encoded[column].map(mapping)
        
        return X_encoded

# ============================================================================
# DOMAIN-SPECIFIC TRANSFORMERS
# ============================================================================

class TextLengthTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for text data - extracts text length features
    """
    
    def __init__(self, text_columns=None, features=['length', 'word_count', 'char_count']):
        """
        Initialize text length transformer
        
        Parameters:
        -----------
        text_columns : list, optional
            Columns containing text data. If None, detect automatically.
        features : list, default=['length', 'word_count', 'char_count']
            Text features to extract
        """
        self.text_columns = text_columns
        self.features = features
    
    def fit(self, X, y=None):
        """
        Fit the transformer
        """
        if isinstance(X, pd.DataFrame):
            if self.text_columns is None:
                # Auto-detect text columns
                self.text_columns_ = [col for col in X.columns if X[col].dtype == 'object']
            else:
                self.text_columns_ = self.text_columns
        else:
            self.text_columns_ = self.text_columns or [0]  # Assume first column is text
        
        return self
    
    def transform(self, X):
        """
        Extract text features
        """
        if isinstance(X, pd.DataFrame):
            X_features = X.copy()
        else:
            X_features = pd.DataFrame(X)
        
        for column in self.text_columns_:
            if column in X_features.columns:
                text_series = X_features[column].astype(str)
                
                if 'length' in self.features:
                    X_features[f'{column}_length'] = text_series.str.len()
                
                if 'word_count' in self.features:
                    X_features[f'{column}_word_count'] = text_series.str.split().str.len()
                
                if 'char_count' in self.features:
                    X_features[f'{column}_char_count'] = text_series.str.replace(' ', '').str.len()
                
                # Remove original text column
                X_features = X_features.drop(column, axis=1)
        
        return X_features

# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_basic_transformers():
    """Demonstrate basic custom transformers"""
    
    print("=== BASIC CUSTOM TRANSFORMERS DEMONSTRATION ===\n")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.exponential(2, (100, 4))  # Exponential data for log transform
    X[:, 1] = X[:, 1] * 100  # Make one feature have different scale
    
    # Add some outliers
    X[0, 0] = 50  # Outlier
    X[1, 1] = 1000  # Outlier
    
    print(f"Original data shape: {X.shape}")
    print(f"Original data statistics:")
    print(f"  Mean: {X.mean(axis=0)}")
    print(f"  Std: {X.std(axis=0)}")
    print(f"  Min: {X.min(axis=0)}")
    print(f"  Max: {X.max(axis=0)}")
    
    # 1. Log Transformer
    print(f"\n1. LOG TRANSFORMER")
    print("-" * 30)
    
    log_transformer = LogTransformer(features=[0, 2], base='e')
    X_log = log_transformer.fit_transform(X)
    
    print(f"Log transformed data statistics (features 0, 2):")
    print(f"  Mean: {X_log.mean(axis=0)}")
    print(f"  Std: {X_log.std(axis=0)}")
    
    # 2. Outlier Remover
    print(f"\n2. OUTLIER REMOVER")
    print("-" * 30)
    
    outlier_remover = OutlierRemover(method='iqr', threshold=1.5)
    X_clean = outlier_remover.fit_transform(X)
    
    print(f"Original samples: {X.shape[0]}")
    print(f"After outlier removal: {X_clean.shape[0]}")
    print(f"Removed {X.shape[0] - X_clean.shape[0]} outliers")
    
    # 3. Feature Creator
    print(f"\n3. FEATURE CREATOR")
    print("-" * 30)
    
    # Define custom feature functions
    feature_functions = [
        lambda row: np.sum(row),  # Sum of all features
        lambda row: np.prod(row),  # Product of all features
        lambda row: np.max(row) - np.min(row),  # Range
        lambda row: row[0] / (row[1] + 1e-8)  # Ratio of first two features
    ]
    
    feature_names = ['sum_features', 'product_features', 'range_features', 'ratio_01']
    
    feature_creator = FeatureCreator(
        feature_functions=feature_functions, 
        feature_names=feature_names
    )
    
    X_extended = feature_creator.fit_transform(X)
    
    print(f"Original features: {X.shape[1]}")
    print(f"After feature creation: {X_extended.shape[1]}")
    print(f"New feature statistics:")
    for i, name in enumerate(feature_names):
        feature_idx = X.shape[1] + i
        print(f"  {name}: mean={X_extended[:, feature_idx].mean():.4f}, "
              f"std={X_extended[:, feature_idx].std():.4f}")

def demonstrate_advanced_transformers():
    """Demonstrate advanced custom transformers with pandas DataFrames"""
    
    print(f"\n=== ADVANCED CUSTOM TRANSFORMERS DEMONSTRATION ===\n")
    
    # Create sample DataFrame with mixed data types
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'numeric1': np.random.normal(100, 15, n_samples),
        'numeric2': np.random.exponential(2, n_samples),
        'category1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'category2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'text': [f"This is sample text number {i} with varying lengths." for i in range(n_samples)]
    }
    
    # Add missing values
    missing_indices = np.random.choice(n_samples, 20, replace=False)
    for idx in missing_indices[:10]:
        data['numeric1'][idx] = np.nan
    for idx in missing_indices[10:]:
        data['category1'][idx] = np.nan
    
    df = pd.DataFrame(data)
    y = np.random.choice([0, 1], n_samples)  # Binary target for target encoding
    
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Data types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # 1. Missing Value Imputer
    print(f"\n1. MISSING VALUE IMPUTER")
    print("-" * 30)
    
    imputer = MissingValueImputer(strategy='mean', features=['numeric1'])
    df_imputed = imputer.fit_transform(df)
    
    print(f"Missing values after imputation:")
    if isinstance(df_imputed, pd.DataFrame):
        print(df_imputed.isnull().sum())
    else:
        print("Converted to numpy array")
    
    # 2. Categorical Encoder
    print(f"\n2. CATEGORICAL ENCODER")
    print("-" * 30)
    
    # Label encoding
    label_encoder = CategoricalEncoder(strategy='label')
    df_label_encoded = label_encoder.fit_transform(df.copy())
    
    print(f"Label encoded shape: {df_label_encoded.shape}")
    print(f"Unique values in category1 after label encoding: {df_label_encoded['category1'].unique()}")
    
    # One-hot encoding
    onehot_encoder = CategoricalEncoder(strategy='onehot')
    df_onehot = onehot_encoder.fit_transform(df.copy())
    
    print(f"One-hot encoded shape: {df_onehot.shape}")
    print(f"New columns from one-hot encoding: {[col for col in df_onehot.columns if 'category' in col]}")
    
    # 3. Text Length Transformer
    print(f"\n3. TEXT LENGTH TRANSFORMER")
    print("-" * 30)
    
    text_transformer = TextLengthTransformer(
        text_columns=['text'], 
        features=['length', 'word_count', 'char_count']
    )
    
    df_text_features = text_transformer.fit_transform(df.copy())
    
    print(f"DataFrame shape after text transformation: {df_text_features.shape}")
    print(f"New text features:")
    text_feature_cols = [col for col in df_text_features.columns if 'text_' in col]
    for col in text_feature_cols:
        print(f"  {col}: mean={df_text_features[col].mean():.2f}, std={df_text_features[col].std():.2f}")

def demonstrate_pipeline_integration():
    """Demonstrate integration of custom transformers with sklearn pipelines"""
    
    print(f"\n=== PIPELINE INTEGRATION DEMONSTRATION ===\n")
    
    # Load iris dataset for classification
    data = load_iris()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create pipeline with custom transformer
    custom_pipeline = Pipeline([
        ('log_transform', LogTransformer(features=[0, 1], base='e')),
        ('feature_creator', FeatureCreator(
            feature_functions=[
                lambda row: row[0] * row[1],  # Interaction feature
                lambda row: row[2] + row[3],  # Sum feature
            ],
            feature_names=['sepal_interaction', 'petal_sum']
        )),
        ('outlier_removal', OutlierRemover(method='iqr', threshold=2.0)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train pipeline
    print(f"\nTraining custom pipeline...")
    custom_pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = custom_pipeline.score(X_train, y_train)
    test_accuracy = custom_pipeline.score(X_test, y_test)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Compare with standard pipeline
    standard_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    standard_pipeline.fit(X_train, y_train)
    standard_test_accuracy = standard_pipeline.score(X_test, y_test)
    
    print(f"\nComparison:")
    print(f"Custom pipeline accuracy: {test_accuracy:.4f}")
    print(f"Standard pipeline accuracy: {standard_test_accuracy:.4f}")
    print(f"Improvement: {test_accuracy - standard_test_accuracy:.4f}")
    
    # Demonstrate with GridSearchCV
    print(f"\n=== GRID SEARCH WITH CUSTOM TRANSFORMERS ===")
    
    # Create pipeline for grid search
    grid_pipeline = Pipeline([
        ('log_transform', LogTransformer()),
        ('outlier_removal', OutlierRemover()),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'log_transform__base': ['e', 10, 2],
        'outlier_removal__method': ['iqr', 'zscore'],
        'outlier_removal__threshold': [1.5, 2.0, 2.5],
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10]
    }
    
    # Perform grid search
    print(f"Performing grid search with custom transformers...")
    grid_search = GridSearchCV(
        grid_pipeline, param_grid, cv=3, 
        scoring='accuracy', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy with best params: {grid_search.score(X_test, y_test):.4f}")

def demonstrate_transformer_validation():
    """Demonstrate proper validation and error handling for custom transformers"""
    
    print(f"\n=== TRANSFORMER VALIDATION DEMONSTRATION ===\n")
    
    # Create test data
    np.random.seed(42)
    X_valid = np.random.normal(0, 1, (50, 3))
    X_invalid = np.array([['a', 'b', 'c'], ['d', 'e', 'f']])  # Invalid data
    
    # 1. Test LogTransformer validation
    print(f"1. TESTING LOG TRANSFORMER VALIDATION")
    print("-" * 40)
    
    log_transformer = LogTransformer(features=[0, 1])
    
    try:
        # Should work fine
        log_transformer.fit(X_valid)
        X_transformed = log_transformer.transform(X_valid)
        print(f"✓ Valid data transformed successfully: {X_transformed.shape}")
    except Exception as e:
        print(f"✗ Error with valid data: {e}")
    
    try:
        # Should handle negative values by adding small constant
        X_negative = np.array([[-1, -2, 3], [4, -5, 6]])
        log_transformer.fit(X_negative)
        X_neg_transformed = log_transformer.transform(X_negative)
        print(f"✓ Negative data handled: {X_neg_transformed.shape}")
    except Exception as e:
        print(f"✗ Error with negative data: {e}")
    
    try:
        # Should fail when transform called before fit
        new_transformer = LogTransformer()
        new_transformer.transform(X_valid)
        print(f"✗ Should have failed - transform called before fit")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # 2. Test OutlierRemover with different data types
    print(f"\n2. TESTING OUTLIER REMOVER VALIDATION")
    print("-" * 40)
    
    outlier_remover = OutlierRemover(method='iqr')
    
    # Test with normal data
    outlier_remover.fit(X_valid)
    X_clean = outlier_remover.transform(X_valid)
    print(f"✓ Outlier removal completed: {X_valid.shape[0]} -> {X_clean.shape[0]} samples")
    
    # Test with all-same values (should handle gracefully)
    X_constant = np.ones((10, 3))
    outlier_remover.fit(X_constant)
    X_constant_clean = outlier_remover.transform(X_constant)
    print(f"✓ Constant data handled: {X_constant.shape[0]} -> {X_constant_clean.shape[0]} samples")
    
    # 3. Test parameter validation
    print(f"\n3. TESTING PARAMETER VALIDATION")
    print("-" * 40)
    
    # Test invalid parameters
    try:
        invalid_transformer = LogTransformer(base='invalid')
        invalid_transformer.fit(X_valid)
        invalid_transformer.transform(X_valid)
        print(f"✗ Should have failed with invalid base")
    except Exception as e:
        print(f"✓ Invalid parameter caught: {type(e).__name__}")
    
    try:
        invalid_outlier = OutlierRemover(method='invalid_method')
        invalid_outlier.fit(X_valid)
        print(f"✗ Should have failed with invalid method")
    except Exception as e:
        print(f"✓ Invalid method parameter handled")

# Main demonstration function
def main_demonstration():
    """Run all custom transformer demonstrations"""
    
    print("=== COMPREHENSIVE CUSTOM TRANSFORMERS GUIDE ===\n")
    
    # Basic transformers
    demonstrate_basic_transformers()
    
    # Advanced transformers
    demonstrate_advanced_transformers()
    
    # Pipeline integration
    demonstrate_pipeline_integration()
    
    # Validation and error handling
    demonstrate_transformer_validation()
    
    print(f"\n{'='*60}")
    print("SUMMARY: CUSTOM TRANSFORMER BEST PRACTICES")
    print(f"{'='*60}")
    print("1. Inherit from BaseEstimator and TransformerMixin")
    print("2. Implement fit() and transform() methods")
    print("3. Store fitted parameters with trailing underscore")
    print("4. Handle edge cases and invalid inputs gracefully")
    print("5. Make transformers stateless (no side effects)")
    print("6. Support both numpy arrays and pandas DataFrames")
    print("7. Include proper parameter validation")
    print("8. Write comprehensive tests for your transformers")
    print("9. Document expected input/output formats")
    print("10. Ensure compatibility with sklearn pipelines and tools")

if __name__ == "__main__":
    main_demonstration()
```

### Explanation
**Custom Transformer Requirements:**
1. **Inheritance**: Must inherit from `BaseEstimator` and `TransformerMixin`
2. **fit() method**: Learn parameters from training data, return self
3. **transform() method**: Apply transformation to data, return transformed data
4. **Stateless**: No side effects, consistent results with same input
5. **Parameter Storage**: Fitted parameters end with underscore (e.g., `mean_`)

**Key Methods:**
- `fit(X, y=None)`: Learn transformation parameters from training data
- `transform(X)`: Apply learned transformation to new data  
- `fit_transform(X, y=None)`: Convenience method combining fit and transform
- `get_params()`: Return transformer parameters (inherited from BaseEstimator)
- `set_params(**params)`: Set transformer parameters (inherited from BaseEstimator)

### Use Cases
- **Domain-Specific Preprocessing**: Custom data cleaning for specific domains
- **Feature Engineering**: Create new features based on domain knowledge
- **Data Quality**: Custom outlier detection and missing value handling
- **Text Processing**: Extract custom features from text data
- **Pipeline Integration**: Seamless integration with sklearn workflows

### Best Practices
1. **Validation**: Check input format and parameters in fit()
2. **Error Handling**: Provide clear error messages for invalid inputs
3. **Documentation**: Document expected input/output formats
4. **Testing**: Write unit tests for all transformer functionality
5. **Consistency**: Ensure transform() works on unseen data
6. **Memory Efficiency**: Avoid copying data unnecessarily

### Pitfalls
- **Data Leakage**: Don't use target information in fit() for unsupervised transforms
- **State Management**: Don't modify input data in-place
- **Parameter Validation**: Validate parameters to prevent runtime errors
- **Shape Consistency**: Ensure output shape is predictable
- **Thread Safety**: Avoid using mutable class variables

### Debugging
```python
# Test transformer step by step
def debug_transformer(transformer, X, y=None):
    """Debug custom transformer"""
    print(f"Input shape: {X.shape}")
    print(f"Input dtype: {X.dtype}")
    
    # Test fit
    try:
        transformer.fit(X, y)
        print("✓ Fit successful")
    except Exception as e:
        print(f"✗ Fit failed: {e}")
        return
    
    # Check fitted attributes
    fitted_attrs = [attr for attr in dir(transformer) if attr.endswith('_')]
    print(f"Fitted attributes: {fitted_attrs}")
    
    # Test transform
    try:
        X_transformed = transformer.transform(X)
        print(f"✓ Transform successful: {X_transformed.shape}")
        return X_transformed
    except Exception as e:
        print(f"✗ Transform failed: {e}")

# Validate transformer compatibility
def validate_sklearn_compatibility(transformer, X, y=None):
    """Validate sklearn compatibility"""
    from sklearn.utils.estimator_checks import check_estimator
    
    try:
        check_estimator(transformer)
        print("✓ Transformer passes sklearn compatibility tests")
    except Exception as e:
        print(f"✗ Compatibility issues: {e}")
```

### Optimization
```python
# Vectorized operations for better performance
class OptimizedTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X):
        # Use numpy vectorized operations instead of loops
        return np.log(X + 1e-8)  # Faster than list comprehension
    
# Memory-efficient transformations
class MemoryEfficientTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X):
        # Modify in-place when possible
        X_copy = X.copy()  # Only copy when necessary
        X_copy[:, 0] = np.log(X_copy[:, 0] + 1e-8)
        return X_copy

# Parallel processing for expensive operations
from joblib import Parallel, delayed

class ParallelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
    
    def transform(self, X):
        # Use joblib for parallel processing
        def process_row(row):
            return expensive_operation(row)
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_row)(row) for row in X
        )
        return np.array(results)
```

---

