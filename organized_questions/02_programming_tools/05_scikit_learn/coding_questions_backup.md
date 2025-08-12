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

**Implement feature extraction from text using Scikit-Learn’s CountVectorizer or TfidfVectorizer.**

**Answer:** Text feature extraction transforms textual data into numerical vectors using CountVectorizer for token frequencies and TfidfVectorizer for weighted importance scores, enabling machine learning algorithms to process text data.

### Theory:
- **CountVectorizer**: Creates a matrix where each column represents a unique word and values are word frequencies
- **TfidfVectorizer**: Uses Term Frequency-Inverse Document Frequency to weight words by importance
- **Feature Engineering**: Includes n-grams, stop word removal, and vocabulary limiting for better performance
- **Text Preprocessing**: Handles tokenization, normalization, and cleaning automatically

### Code Example:
```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Sample text data
texts = [
    "I love this movie! It's absolutely fantastic and amazing.",
    "This film is terrible. I hate it completely.",
    "The movie was okay, nothing special but watchable.",
    "Brilliant acting and great storyline. Highly recommended!",
    "Boring and predictable. Not worth watching.",
    "Great cinematography and excellent direction.",
    "Poor script and bad acting. Very disappointing.",
    "One of the best movies I've ever seen!",
    "Average movie with some good moments.",
    "Awful film with no redeeming qualities."
]

labels = ['positive', 'negative', 'neutral', 'positive', 'negative',
         'positive', 'negative', 'positive', 'neutral', 'negative']

# 1. CountVectorizer Demonstration
print("=== CountVectorizer ===")
count_vec = CountVectorizer()
X_count = count_vec.fit_transform(texts)

print(f"Vocabulary size: {len(count_vec.vocabulary_)}")
print(f"Feature matrix shape: {X_count.shape}")

# Show vocabulary sample
vocab_sample = list(count_vec.vocabulary_.keys())[:10]
print(f"Sample vocabulary: {vocab_sample}")

# Advanced CountVectorizer
count_vec_advanced = CountVectorizer(
    max_features=50,
    ngram_range=(1, 2),
    stop_words='english'
)
X_count_advanced = count_vec_advanced.fit_transform(texts)
print(f"Advanced CountVectorizer vocabulary: {len(count_vec_advanced.vocabulary_)}")

# 2. TfidfVectorizer Demonstration
print("\n=== TfidfVectorizer ===")
tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(texts)

print(f"Vocabulary size: {len(tfidf_vec.vocabulary_)}")
print(f"Feature matrix shape: {X_tfidf.shape}")

# Show TF-IDF scores for first document
feature_names = tfidf_vec.get_feature_names_out()
first_doc_tfidf = X_tfidf[0].toarray().flatten()

print(f"First document TF-IDF scores (top 5):")
top_indices = first_doc_tfidf.argsort()[-5:][::-1]
for idx in top_indices:
    if first_doc_tfidf[idx] > 0:
        print(f"  '{feature_names[idx]}': {first_doc_tfidf[idx]:.4f}")

# Advanced TfidfVectorizer
tfidf_vec_advanced = TfidfVectorizer(
    max_features=50,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True
)
X_tfidf_advanced = tfidf_vec_advanced.fit_transform(texts)
print(f"Advanced TfidfVectorizer vocabulary: {len(tfidf_vec_advanced.vocabulary_)}")

# 3. Performance Comparison
print("\n=== Performance Comparison ===")
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

vectorizer_configs = {
    'CountVectorizer': CountVectorizer(stop_words='english'),
    'TfidfVectorizer': TfidfVectorizer(stop_words='english'),
    'TfidfVectorizer_Ngrams': TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 2)
    )
}

for name, vectorizer in vectorizer_configs.items():
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', MultinomialNB())
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{name}: {accuracy:.3f}")

# 4. Complete Pipeline Example
print("\n=== Complete Pipeline ===")
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)
    )),
    ('classifier', MultinomialNB())
])

text_pipeline.fit(X_train, y_train)
train_accuracy = text_pipeline.score(X_train, y_train)
test_accuracy = text_pipeline.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")

# 5. Test on New Text
new_texts = [
    "This is an amazing and fantastic movie!",
    "Terrible film, completely boring.",
    "The movie was decent, nothing special."
]

predictions = text_pipeline.predict(new_texts)

print("\n=== New Text Predictions ===")
for text, pred in zip(new_texts, predictions):
    print(f"Text: '{text}'")
    print(f"Prediction: {pred}")

# 6. Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Vocabulary size comparison
ax1 = axes[0]
methods = ['Count\nBasic', 'Count\nAdvanced', 'TF-IDF\nBasic', 'TF-IDF\nAdvanced']
vocab_sizes = [
    len(count_vec.vocabulary_),
    len(count_vec_advanced.vocabulary_),
    len(tfidf_vec.vocabulary_),
    len(tfidf_vec_advanced.vocabulary_)
]

ax1.bar(methods, vocab_sizes, color=['skyblue', 'lightblue', 'lightcoral', 'coral'])
ax1.set_title('Vocabulary Size Comparison')
ax1.set_ylabel('Number of Features')

# Label distribution
ax2 = axes[1]
label_counts = pd.Series(labels).value_counts()
colors = ['#2ecc71', '#e74c3c', '#f39c12']
ax2.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', colors=colors)
ax2.set_title('Label Distribution')

plt.tight_layout()
plt.show()

print("\n✅ Text feature extraction demonstration complete")
```

### Explanation:
1. **CountVectorizer**: Creates a matrix of token counts for frequency-based analysis
2. **TfidfVectorizer**: Weights words by importance using TF-IDF scoring
3. **Feature Engineering**: Includes n-grams, stop word removal, and vocabulary limiting
4. **Pipeline Integration**: Combines text processing with machine learning models
5. **Performance Comparison**: Evaluates different vectorization approaches

### Use Cases:
- **Sentiment Analysis**: Classify text sentiment using TF-IDF features
- **Document Classification**: Categorize documents by topic or type
- **Spam Detection**: Identify spam emails using text features
- **Content Recommendation**: Find similar documents based on text similarity

### Best Practices:
- **Preprocessing**: Remove noise and normalize text before vectorization
- **Feature Selection**: Use min_df and max_df to filter rare and common words
- **N-grams**: Include bigrams/trigrams to capture phrase-level information
- **Pipeline Design**: Combine preprocessing, vectorization, and modeling

### Common Pitfalls:
- **Vocabulary Explosion**: Uncontrolled vocabulary can lead to memory issues
- **Overfitting**: High-dimensional sparse features can cause overfitting
- **Memory Usage**: Large text corpora require efficient vectorization strategies

### Debugging:
```python
def debug_vectorization():
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    X = vectorizer.fit_transform(["Sample text for debugging"])
    
    print("Vocabulary:", list(vectorizer.vocabulary_.keys()))
    print("Feature matrix shape:", X.shape)
    print("Non-zero elements:", X.nnz)
```

### Optimization:
- **Memory Efficiency**: Use HashingVectorizer for large datasets
- **Speed**: Set max_features to limit vocabulary size
- **Accuracy**: Experiment with different n-gram ranges
- **Parallel Processing**: Leverage n_jobs parameter for faster fitting

---

## Question 6

**Normalize a given dataset usingScikit-Learn’s preprocessingmodule, then train and test aNaive Bayes classifier.**

**Answer:** Data normalization using sklearn's preprocessing module standardizes features to improve model performance, particularly for algorithms sensitive to feature scales like Naive Bayes classifiers.

### Theory:
- **StandardScaler**: Standardizes features by removing mean and scaling to unit variance (z-score normalization)
- **MinMaxScaler**: Scales features to a fixed range, typically [0, 1]
- **RobustScaler**: Uses median and interquartile range, less sensitive to outliers
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem with strong independence assumptions

### Code Example:
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create sample dataset
print("=== Dataset Creation ===")
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_clusters_per_class=1,
    random_state=42
)

# Add some noise and scale variations
X[:, 0] *= 100  # Scale first feature
X[:, 1] += 50   # Shift second feature
X[:, 2] *= 0.01 # Scale third feature very small

# Create DataFrame for better visualization
feature_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Original dataset statistics:")
print(df.describe())

# 2. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Different Preprocessing Methods
print("\n=== Preprocessing Comparison ===")

preprocessors = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'No_Scaling': None
}

results = {}

for name, scaler in preprocessors.items():
    print(f"\n--- {name} ---")
    
    if scaler is not None:
        # Fit scaler on training data and transform both sets
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Scaled training data stats:")
        print(f"Mean: {np.mean(X_train_scaled, axis=0)}")
        print(f"Std: {np.std(X_train_scaled, axis=0)}")
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Train Naive Bayes classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = nb_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation score
    if scaler is not None:
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', GaussianNB())
        ])
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    else:
        cv_scores = cross_val_score(nb_classifier, X_train, y_train, cv=5)
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# 4. Comprehensive Pipeline Example
print("\n=== Complete Pipeline Example ===")

class DataNormalizationPipeline:
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type
        self.scaler = None
        self.classifier = None
        self.pipeline = None
        
    def create_pipeline(self):
        """Create preprocessing and classification pipeline"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaler type")
        
        self.classifier = GaussianNB()
        
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', self.classifier)
        ])
        
        return self.pipeline
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train pipeline and evaluate performance"""
        if self.pipeline is None:
            self.create_pipeline()
        
        # Fit pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': report
        }
    
    def get_feature_statistics(self, X):
        """Get statistics of transformed features"""
        if self.scaler is None:
            raise ValueError("Pipeline not fitted yet")
        
        X_transformed = self.scaler.transform(X)
        
        return {
            'mean': np.mean(X_transformed, axis=0),
            'std': np.std(X_transformed, axis=0),
            'min': np.min(X_transformed, axis=0),
            'max': np.max(X_transformed, axis=0)
        }

# Test the pipeline
pipeline_demo = DataNormalizationPipeline('standard')
results_demo = pipeline_demo.train_and_evaluate(X_train, X_test, y_train, y_test)

print(f"Pipeline Accuracy: {results_demo['accuracy']:.3f}")
print("\nClassification Report:")
print(results_demo['classification_report'])

# 5. Real-world Example with Iris Dataset
print("\n=== Real-world Example: Iris Dataset ===")
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Split data
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

# Create and evaluate pipeline
iris_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GaussianNB())
])

iris_pipeline.fit(X_train_iris, y_train_iris)
iris_accuracy = iris_pipeline.score(X_test_iris, y_test_iris)
iris_pred = iris_pipeline.predict(X_test_iris)

print(f"Iris Classification Accuracy: {iris_accuracy:.3f}")

# 6. Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Original data distribution
ax1 = axes[0, 0]
ax1.boxplot(X[:, :4], labels=feature_names)
ax1.set_title('Original Data Distribution')
ax1.set_ylabel('Feature Values')
ax1.tick_params(axis='x', rotation=45)

# Normalized data distribution
scaler_viz = StandardScaler()
X_normalized = scaler_viz.fit_transform(X)
ax2 = axes[0, 1]
ax2.boxplot(X_normalized[:, :4], labels=feature_names)
ax2.set_title('Standardized Data Distribution')
ax2.set_ylabel('Standardized Values')
ax2.tick_params(axis='x', rotation=45)

# Performance comparison
ax3 = axes[1, 0]
methods = list(results.keys())
accuracies = [results[method]['accuracy'] for method in methods]
colors = ['skyblue', 'lightcoral', 'lightgreen', 'wheat']

bars = ax3.bar(methods, accuracies, color=colors)
ax3.set_title('Preprocessing Methods Comparison')
ax3.set_ylabel('Test Accuracy')
ax3.set_ylim(0, 1)
ax3.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')

# Confusion matrix for best method
best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_scaler = preprocessors[best_method]

if best_scaler is not None:
    X_test_best = best_scaler.fit_transform(X_train).copy()
    X_test_best = best_scaler.transform(X_test)
else:
    X_test_best = X_test

nb_best = GaussianNB()
nb_best.fit(X_train_scaled if best_scaler else X_train, y_train)
y_pred_best = nb_best.predict(X_test_best)

ax4 = axes[1, 1]
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_title(f'Confusion Matrix ({best_method})')
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')

plt.tight_layout()
plt.show()

print("\n✅ Data normalization and Naive Bayes classification complete")
```

### Explanation:
1. **Data Normalization**: Standardizes features to improve model performance
2. **Preprocessing Comparison**: Tests different scaling methods (Standard, MinMax, Robust)
3. **Pipeline Integration**: Combines preprocessing and classification in one workflow
4. **Performance Evaluation**: Compares accuracy across different normalization techniques
5. **Real-world Application**: Demonstrates on Iris dataset with proper validation

### Use Cases:
- **Feature Scaling**: Prepare data for scale-sensitive algorithms
- **Text Classification**: Normalize TF-IDF features for Naive Bayes
- **Medical Diagnosis**: Normalize clinical measurements for classification
- **Spam Detection**: Preprocess email features before classification

### Best Practices:
- **Fit on Training Only**: Always fit scalers on training data only
- **Transform Consistently**: Apply same transformation to train/test/validation sets
- **Pipeline Usage**: Use pipelines to prevent data leakage
- **Cross-validation**: Properly validate preprocessing steps

### Common Pitfalls:
- **Data Leakage**: Fitting scaler on entire dataset including test data
- **Inconsistent Scaling**: Different preprocessing for train/test sets
- **Wrong Scaler Choice**: Using inappropriate scaler for data distribution

### Debugging:
```python
def debug_preprocessing():
    # Check for data leakage
    scaler = StandardScaler()
    scaler.fit(X_train)  # Only fit on training data
    
    print("Training data mean:", np.mean(scaler.transform(X_train), axis=0))
    print("Test data mean:", np.mean(scaler.transform(X_test), axis=0))
```

### Optimization:
- **Scaler Selection**: Choose appropriate scaler based on data distribution
- **Memory Efficiency**: Use sparse matrices for large datasets
- **Parallel Processing**: Leverage n_jobs for cross-validation
- **Feature Selection**: Combine with feature selection techniques

---

## Question 7

**Demonstrate how to use Scikit-Learn’s Pipeline to combine preprocessing and model training steps.**

**Answer:** Scikit-Learn's Pipeline combines multiple processing steps into a single estimator, ensuring consistent data transformations and preventing data leakage during cross-validation and model evaluation.

### Theory:
- **Pipeline**: Sequential application of transformers followed by a final estimator
- **Data Leakage Prevention**: Ensures preprocessing is fit only on training data
- **Cross-validation Safety**: Proper handling of preprocessing in CV folds
- **Code Organization**: Clean, readable, and maintainable ML workflows

### Code Example:
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Basic Pipeline Example
print("=== Basic Pipeline Example ===")

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create basic pipeline
basic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Fit and evaluate
basic_pipeline.fit(X_train, y_train)
basic_accuracy = basic_pipeline.score(X_test, y_test)

print(f"Basic Pipeline Accuracy: {basic_accuracy:.3f}")

# 2. Advanced Pipeline with Feature Selection
print("\n=== Advanced Pipeline with Feature Selection ===")

advanced_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=15)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

advanced_pipeline.fit(X_train, y_train)
advanced_accuracy = advanced_pipeline.score(X_test, y_test)

print(f"Advanced Pipeline Accuracy: {advanced_accuracy:.3f}")

# Show selected features
selected_features = advanced_pipeline.named_steps['feature_selection'].get_support()
print(f"Selected {np.sum(selected_features)} features out of {len(selected_features)}")

# 3. Complex Pipeline with Multiple Preprocessing Steps
print("\n=== Complex Pipeline Example ===")

complex_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('feature_selection', SelectKBest(f_classif, k=50)),
    ('classifier', SVC(kernel='rbf', random_state=42))
])

complex_pipeline.fit(X_train, y_train)
complex_accuracy = complex_pipeline.score(X_test, y_test)

print(f"Complex Pipeline Accuracy: {complex_accuracy:.3f}")

# 4. Pipeline with GridSearchCV
print("\n=== Pipeline with Hyperparameter Tuning ===")

# Define pipeline
tuning_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

# Grid search
grid_search = GridSearchCV(
    tuning_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_accuracy = grid_search.score(X_test, y_test)

print(f"Best Pipeline Accuracy: {best_accuracy:.3f}")
print(f"Best Parameters: {grid_search.best_params_}")

# 5. Pipeline Comparison Class
class PipelineComparison:
    def __init__(self):
        self.pipelines = {}
        self.results = {}
    
    def add_pipeline(self, name, pipeline):
        """Add a pipeline to compare"""
        self.pipelines[name] = pipeline
    
    def compare_pipelines(self, X_train, X_test, y_train, y_test, cv=5):
        """Compare all pipelines"""
        for name, pipeline in self.pipelines.items():
            print(f"\n--- Evaluating {name} ---")
            
            # Fit pipeline
            pipeline.fit(X_train, y_train)
            
            # Test accuracy
            test_accuracy = pipeline.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv)
            
            # Store results
            self.results[name] = {
                'test_accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'pipeline': pipeline
            }
            
            print(f"Test Accuracy: {test_accuracy:.3f}")
            print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    def get_best_pipeline(self):
        """Get the best performing pipeline"""
        best_name = max(self.results.keys(), 
                       key=lambda x: self.results[x]['test_accuracy'])
        return best_name, self.results[best_name]['pipeline']
    
    def plot_comparison(self):
        """Plot pipeline comparison"""
        names = list(self.results.keys())
        test_scores = [self.results[name]['test_accuracy'] for name in names]
        cv_scores = [self.results[name]['cv_mean'] for name in names]
        cv_stds = [self.results[name]['cv_std'] for name in names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Test accuracy comparison
        bars1 = ax1.bar(names, test_scores, color='skyblue', alpha=0.7)
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars1, test_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Cross-validation scores with error bars
        bars2 = ax2.bar(names, cv_scores, yerr=cv_stds, 
                       color='lightcoral', alpha=0.7, capsize=5)
        ax2.set_title('Cross-Validation Scores')
        ax2.set_ylabel('CV Accuracy')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars2, cv_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# 6. Comprehensive Pipeline Comparison
print("\n=== Comprehensive Pipeline Comparison ===")

# Create comparison object
comparison = PipelineComparison()

# Add different pipelines
comparison.add_pipeline('Basic_LR', Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
]))

comparison.add_pipeline('RF_with_Selection', Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=20)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
]))

comparison.add_pipeline('SVM_Complex', Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('feature_selection', SelectKBest(f_classif, k=30)),
    ('classifier', SVC(kernel='rbf', random_state=42))
]))

comparison.add_pipeline('RF_RFE', Pipeline([
    ('scaler', StandardScaler()),
    ('rfe', RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=15)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
]))

# Compare pipelines
comparison.compare_pipelines(X_train, X_test, y_train, y_test)

# Get best pipeline
best_name, best_pipeline = comparison.get_best_pipeline()
print(f"\nBest Pipeline: {best_name}")

# Plot comparison
comparison.plot_comparison()

# 7. Real-world Pipeline with Mixed Data Types
print("\n=== Real-world Pipeline Example ===")

# Create mixed dataset
np.random.seed(42)
n_samples = 1000

# Numerical features
numerical_features = np.random.randn(n_samples, 3)

# Categorical features (encoded as integers)
categorical_features = np.random.randint(0, 5, size=(n_samples, 2))

# Combine features
X_mixed = np.column_stack([numerical_features, categorical_features])
y_mixed = (numerical_features[:, 0] + categorical_features[:, 0] > 0).astype(int)

# Column names
feature_names = ['num_1', 'num_2', 'num_3', 'cat_1', 'cat_2']
numerical_indices = [0, 1, 2]
categorical_indices = [3, 4]

# Create preprocessing for different column types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_indices),
        ('cat', MinMaxScaler(), categorical_indices)  # Treating as numerical for simplicity
    ]
)

# Complete pipeline
mixed_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split and evaluate
X_train_mixed, X_test_mixed, y_train_mixed, y_test_mixed = train_test_split(
    X_mixed, y_mixed, test_size=0.3, random_state=42
)

mixed_pipeline.fit(X_train_mixed, y_train_mixed)
mixed_accuracy = mixed_pipeline.score(X_test_mixed, y_test_mixed)

print(f"Mixed Data Pipeline Accuracy: {mixed_accuracy:.3f}")

# 8. Pipeline Inspection and Analysis
print("\n=== Pipeline Inspection ===")

# Get pipeline steps
print("Pipeline Steps:")
for i, (name, transformer) in enumerate(best_pipeline.named_steps.items()):
    print(f"  {i+1}. {name}: {transformer}")

# Access specific step
if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
    feature_importance = best_pipeline.named_steps['classifier'].feature_importances_
    print(f"\nTop 5 Feature Importances:")
    
    # If feature selection was applied, get selected feature indices
    if 'feature_selection' in best_pipeline.named_steps:
        selected_features = best_pipeline.named_steps['feature_selection'].get_support()
        selected_indices = np.where(selected_features)[0]
        top_features = np.argsort(feature_importance)[-5:][::-1]
        
        for i, idx in enumerate(top_features):
            original_idx = selected_indices[idx] if len(selected_indices) > idx else idx
            print(f"  {i+1}. Feature {original_idx}: {feature_importance[idx]:.3f}")

print("\n✅ Pipeline demonstration complete")
```

### Explanation:
1. **Basic Pipeline**: Combines preprocessing (scaling) with classification
2. **Feature Selection**: Integrates feature selection into the pipeline
3. **Complex Processing**: Multiple preprocessing steps in sequence
4. **Hyperparameter Tuning**: GridSearchCV with pipeline parameters
5. **Pipeline Comparison**: Systematic evaluation of different pipelines

### Use Cases:
- **Data Science Workflows**: End-to-end ML pipeline development
- **Model Deployment**: Consistent preprocessing in production
- **Cross-validation**: Proper handling of preprocessing steps
- **Feature Engineering**: Complex transformation sequences

### Best Practices:
- **Consistent Preprocessing**: Same transformations for train/test/validation
- **Parameter Naming**: Use double underscores for nested parameters
- **Pipeline Caching**: Use memory parameter for expensive transformations
- **Step Inspection**: Access individual pipeline components for analysis

### Common Pitfalls:
- **Data Leakage**: Fitting transformers on entire dataset
- **Parameter Naming**: Incorrect parameter names in grid search
- **Step Order**: Wrong sequence of preprocessing steps
- **Memory Issues**: Large intermediate representations

### Debugging:
```python
def debug_pipeline(pipeline, X_sample):
    # Inspect each step
    X_transformed = X_sample.copy()
    
    for name, transformer in pipeline.named_steps.items():
        if hasattr(transformer, 'transform'):
            X_transformed = transformer.transform(X_transformed)
            print(f"After {name}: shape={X_transformed.shape}")
        else:
            # Final estimator
            prediction = transformer.predict(X_transformed)
            print(f"Final prediction: {prediction}")
```

### Optimization:
- **Memory Usage**: Use memory parameter for caching expensive steps
- **Parallel Processing**: Leverage n_jobs in grid search
- **Feature Selection**: Reduce dimensionality early in pipeline
- **Model Selection**: Compare different pipeline configurations systematically

---

## Question 8

**Write a Python function that uses Scikit-Learn’s RandomForestClassifier and performs a grid search to find the best hyperparameters.**

**Answer:** A comprehensive grid search function for RandomForestClassifier that optimizes hyperparameters using cross-validation to find the best performing model configuration.

### Theory:
- **RandomForestClassifier**: Ensemble method combining multiple decision trees
- **Grid Search**: Exhaustive search over specified parameter values
- **Cross-validation**: Robust model evaluation to prevent overfitting
- **Hyperparameter Tuning**: Systematic optimization of model parameters

### Code Example:
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                   RandomizedSearchCV, cross_val_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import randint, uniform

def optimize_random_forest(X, y, test_size=0.3, cv_folds=5, 
                          search_method='grid', n_iter=100, 
                          random_state=42, verbose=True):
    """
    Comprehensive RandomForestClassifier optimization function
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target vector
    test_size : float, default=0.3
        Proportion of dataset for testing
    cv_folds : int, default=5
        Number of cross-validation folds
    search_method : str, default='grid'
        Search method: 'grid' or 'random'
    n_iter : int, default=100
        Number of iterations for random search
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Print progress information
    
    Returns:
    --------
    dict : Dictionary containing results and best model
    """
    
    if verbose:
        print("=== RandomForest Hyperparameter Optimization ===")
        print(f"Dataset shape: {X.shape}")
        print(f"Search method: {search_method}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Define parameter grids
    if search_method == 'grid':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Create grid search
        rf = RandomForestClassifier(random_state=random_state)
        search = GridSearchCV(
            rf, param_grid, cv=cv_folds, 
            scoring='accuracy', n_jobs=-1, verbose=1 if verbose else 0
        )
        
    else:  # random search
        param_distributions = {
            'n_estimators': randint(50, 500),
            'max_depth': [None] + list(range(10, 101, 10)),
            'min_samples_split': randint(2, 21),
            'min_samples_leaf': randint(1, 11),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Create random search
        rf = RandomForestClassifier(random_state=random_state)
        search = RandomizedSearchCV(
            rf, param_distributions, n_iter=n_iter, cv=cv_folds,
            scoring='accuracy', n_jobs=-1, random_state=random_state,
            verbose=1 if verbose else 0
        )
    
    # Perform search
    start_time = time.time()
    search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Get best model
    best_model = search.best_estimator_
    
    # Evaluate on test set
    test_accuracy = best_model.score(X_test, y_test)
    y_pred = best_model.predict(X_test)
    
    # Cross-validation score with best parameters
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds)
    
    # Results dictionary
    results = {
        'best_model': best_model,
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_,
        'test_accuracy': test_accuracy,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'search_time': search_time,
        'predictions': y_pred,
        'y_test': y_test,
        'search_object': search
    }
    
    if verbose:
        print(f"\nSearch completed in {search_time:.2f} seconds")
        print(f"Best CV Score: {search.best_score_:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"\nBest Parameters:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
    
    return results

# 1. Basic Usage Example
print("=== Basic Usage Example ===")

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Optimize RandomForest
results_basic = optimize_random_forest(X, y, search_method='grid')

# Print classification report
print("\nClassification Report:")
print(classification_report(results_basic['y_test'], results_basic['predictions']))

# 2. Compare Grid Search vs Random Search
print("\n=== Grid Search vs Random Search Comparison ===")

# Grid search
print("\n--- Grid Search ---")
results_grid = optimize_random_forest(X, y, search_method='grid', verbose=False)

# Random search
print("\n--- Random Search ---")
results_random = optimize_random_forest(X, y, search_method='random', 
                                      n_iter=50, verbose=False)

# Compare results
comparison_data = {
    'Method': ['Grid Search', 'Random Search'],
    'Best CV Score': [results_grid['best_cv_score'], results_random['best_cv_score']],
    'Test Accuracy': [results_grid['test_accuracy'], results_random['test_accuracy']],
    'Search Time': [results_grid['search_time'], results_random['search_time']]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nComparison:")
print(comparison_df)

# 3. Advanced RandomForest Optimization Class
class RandomForestOptimizer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results_history = []
        self.best_model = None
        self.best_score = 0
    
    def quick_search(self, X, y, cv=5):
        """Quick parameter search with limited grid"""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
        
        return self._perform_search(X, y, param_grid, cv, 'quick')
    
    def comprehensive_search(self, X, y, cv=5):
        """Comprehensive parameter search"""
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        return self._perform_search(X, y, param_grid, cv, 'comprehensive')
    
    def random_search(self, X, y, n_iter=100, cv=5):
        """Random parameter search"""
        param_distributions = {
            'n_estimators': randint(50, 500),
            'max_depth': [None] + list(range(5, 51, 5)),
            'min_samples_split': randint(2, 21),
            'min_samples_leaf': randint(1, 11),
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'bootstrap': [True, False]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state)
        search = RandomizedSearchCV(
            rf, param_distributions, n_iter=n_iter, cv=cv,
            scoring='accuracy', n_jobs=-1, random_state=self.random_state
        )
        
        return self._perform_random_search(X, y, search, 'random')
    
    def _perform_search(self, X, y, param_grid, cv, search_type):
        """Internal method for grid search"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        rf = RandomForestClassifier(random_state=self.random_state)
        search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        
        start_time = time.time()
        search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        test_accuracy = search.best_estimator_.score(X_test, y_test)
        
        results = {
            'search_type': search_type,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'test_accuracy': test_accuracy,
            'search_time': search_time,
            'model': search.best_estimator_
        }
        
        self.results_history.append(results)
        
        if test_accuracy > self.best_score:
            self.best_score = test_accuracy
            self.best_model = search.best_estimator_
        
        return results
    
    def _perform_random_search(self, X, y, search, search_type):
        """Internal method for random search"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        start_time = time.time()
        search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        test_accuracy = search.best_estimator_.score(X_test, y_test)
        
        results = {
            'search_type': search_type,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'test_accuracy': test_accuracy,
            'search_time': search_time,
            'model': search.best_estimator_
        }
        
        self.results_history.append(results)
        
        if test_accuracy > self.best_score:
            self.best_score = test_accuracy
            self.best_model = search.best_estimator_
        
        return results
    
    def compare_searches(self):
        """Compare all performed searches"""
        if not self.results_history:
            print("No searches performed yet!")
            return
        
        df = pd.DataFrame(self.results_history)
        
        print("Search Results Comparison:")
        print(df[['search_type', 'best_cv_score', 'test_accuracy', 'search_time']].round(4))
        
        return df
    
    def plot_results(self):
        """Plot comparison of search results"""
        if not self.results_history:
            print("No searches performed yet!")
            return
        
        df = pd.DataFrame(self.results_history)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # CV Scores
        ax1 = axes[0]
        ax1.bar(df['search_type'], df['best_cv_score'], color='skyblue', alpha=0.7)
        ax1.set_title('Best CV Scores')
        ax1.set_ylabel('CV Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Test Accuracy
        ax2 = axes[1]
        ax2.bar(df['search_type'], df['test_accuracy'], color='lightcoral', alpha=0.7)
        ax2.set_title('Test Accuracy')
        ax2.set_ylabel('Test Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        
        # Search Time
        ax3 = axes[2]
        ax3.bar(df['search_type'], df['search_time'], color='lightgreen', alpha=0.7)
        ax3.set_title('Search Time')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# 4. Using the Advanced Optimizer
print("\n=== Advanced Optimizer Example ===")

# Create larger, more complex dataset
X_complex, y_complex = make_classification(
    n_samples=2000, n_features=20, n_informative=15,
    n_redundant=5, n_clusters_per_class=2, random_state=42
)

optimizer = RandomForestOptimizer()

# Perform different searches
print("Performing quick search...")
quick_results = optimizer.quick_search(X_complex, y_complex)

print("\nPerforming random search...")
random_results = optimizer.random_search(X_complex, y_complex, n_iter=50)

# Compare results
optimizer.compare_searches()
optimizer.plot_results()

# 5. Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

best_model_complex = optimizer.best_model
feature_importance = best_model_complex.feature_importances_

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_indices = np.arange(len(feature_importance))
plt.bar(feature_indices, feature_importance, alpha=0.7)
plt.title('Feature Importance (Best RandomForest Model)')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()

# Top 10 features
top_features = np.argsort(feature_importance)[-10:][::-1]
print("Top 10 Features:")
for i, idx in enumerate(top_features):
    print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")

print("\n✅ RandomForest hyperparameter optimization complete")
```

### Explanation:
1. **Comprehensive Function**: Complete hyperparameter optimization with grid/random search
2. **Performance Comparison**: Grid search vs random search evaluation
3. **Advanced Class**: Object-oriented approach with multiple search strategies
4. **Feature Analysis**: Importance ranking and visualization
5. **Time Efficiency**: Search time comparison and optimization

### Use Cases:
- **Model Selection**: Find optimal RandomForest configuration
- **Performance Tuning**: Maximize accuracy through parameter optimization
- **Feature Analysis**: Identify most important predictive features
- **Time Constraints**: Balance search thoroughness with time requirements

### Best Practices:
- **Cross-validation**: Use proper CV to avoid overfitting during search
- **Parameter Ranges**: Choose sensible ranges based on dataset size
- **Search Strategy**: Use random search for large parameter spaces
- **Time Management**: Balance search comprehensiveness with computational cost

### Common Pitfalls:
- **Overfitting**: Tuning too many parameters on small datasets
- **Computational Cost**: Exhaustive grid search on large parameter spaces
- **Data Leakage**: Including test data in hyperparameter optimization
- **Parameter Interactions**: Ignoring parameter dependencies

### Debugging:
```python
def debug_rf_params(model, X_sample, y_sample):
    # Check model configuration
    print("Model Parameters:")
    for param, value in model.get_params().items():
        print(f"  {param}: {value}")
    
    # Check feature importance
    if hasattr(model, 'feature_importances_'):
        print(f"Feature importances shape: {model.feature_importances_.shape}")
        print(f"Top importance: {np.max(model.feature_importances_):.4f}")
```

### Optimization:
- **Parallel Processing**: Use n_jobs=-1 for faster search
- **Early Stopping**: Implement custom scoring with early stopping
- **Memory Efficiency**: Use smaller CV folds for large datasets
- **Progressive Search**: Start with coarse grid, then refine around best parameters

---

## Question 9

**Use Scikit-Learn to visualize the decision boundary of a SVM with a non-linear kernel.**

**Answer:** Visualizing SVM decision boundaries with non-linear kernels reveals how kernel functions map data to higher-dimensional spaces where complex patterns become linearly separable.

### Theory:
- **Non-linear Kernels**: Transform data into higher dimensions for complex boundary creation
- **RBF Kernel**: Radial Basis Function creates circular/elliptical decision boundaries
- **Polynomial Kernel**: Creates polynomial-shaped decision boundaries
- **Decision Boundary**: The hyperplane that separates different classes in feature space

### Code Example:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model, title="Decision Boundary", 
                          resolution=500, padding=0.1):
    """
    Plot decision boundary for 2D datasets
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Feature matrix (must be 2D for visualization)
    y : array-like, shape (n_samples,)
        Target vector
    model : fitted estimator
        Trained model with predict method
    title : str
        Plot title
    resolution : int
        Resolution of the decision boundary mesh
    padding : float
        Padding around data points
    """
    
    # Create color maps
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    cmap_light = ListedColormap(colors[:len(np.unique(y))])
    cmap_bold = ListedColormap(['#FF4444', '#2E8B8B', '#2E5B8B', '#6E9B6E', '#DEBA37'])
    
    # Create a mesh to plot the decision boundary
    h = (X[:, 0].max() - X[:, 0].min()) / resolution
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)
    
    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, 
                         edgecolors='black', s=100, alpha=0.9)
    
    # Highlight support vectors if available
    if hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=200, linewidth=2, facecolors='none', edgecolors='red',
                   label='Support Vectors')
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_svm_kernels(X, y, kernels=['linear', 'rbf', 'poly'], 
                       test_size=0.3, random_state=42):
    """
    Compare different SVM kernels and visualize their decision boundaries
    """
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)
    
    results = {}
    
    # Create subplots
    n_kernels = len(kernels)
    fig, axes = plt.subplots(1, n_kernels, figsize=(5*n_kernels, 4))
    if n_kernels == 1:
        axes = [axes]
    
    for i, kernel in enumerate(kernels):
        print(f"\n--- {kernel.upper()} Kernel ---")
        
        # Create and train SVM
        if kernel == 'poly':
            svm = SVC(kernel=kernel, degree=3, random_state=random_state)
        else:
            svm = SVC(kernel=kernel, random_state=random_state)
        
        svm.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results[kernel] = {
            'model': svm,
            'accuracy': accuracy,
            'n_support_vectors': len(svm.support_vectors_)
        }
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Support Vectors: {len(svm.support_vectors_)}")
        
        # Plot decision boundary
        ax = axes[i]
        
        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.RdYlBu)
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, 
                           cmap=plt.cm.RdYlBu, edgecolors='black', s=50)
        
        # Highlight support vectors
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                  s=100, linewidth=2, facecolors='none', edgecolors='red')
        
        ax.set_title(f'{kernel.upper()} Kernel\nAccuracy: {accuracy:.3f}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

# 1. Basic Non-linear SVM Example
print("=== Basic Non-linear SVM Example ===")

# Create non-linear dataset
X_circles, y_circles = make_circles(n_samples=300, noise=0.1, 
                                   factor=0.3, random_state=42)

# Train RBF SVM
svm_rbf = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_rbf.fit(X_circles, y_circles)

# Visualize decision boundary
plot_decision_boundary(X_circles, y_circles, svm_rbf, 
                      title="RBF SVM Decision Boundary (Circles Dataset)")

print(f"RBF SVM Accuracy: {svm_rbf.score(X_circles, y_circles):.3f}")
print(f"Support Vectors: {len(svm_rbf.support_vectors_)}")

# 2. Compare Different Datasets and Kernels
print("\n=== Kernel Comparison on Different Datasets ===")

# Dataset 1: Circles
print("\nDataset 1: Concentric Circles")
results_circles = compare_svm_kernels(X_circles, y_circles)

# Dataset 2: Moons
print("\nDataset 2: Two Moons")
X_moons, y_moons = make_moons(n_samples=300, noise=0.15, random_state=42)
results_moons = compare_svm_kernels(X_moons, y_moons)

# Dataset 3: Complex classification
print("\nDataset 3: Complex Pattern")
X_complex, y_complex = make_classification(
    n_samples=300, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=2, random_state=42
)
results_complex = compare_svm_kernels(X_complex, y_complex)

# 3. Advanced SVM Visualization Class
class SVMVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.models = {}
        self.datasets = {}
    
    def add_dataset(self, name, X, y):
        """Add a dataset for visualization"""
        self.datasets[name] = {'X': X, 'y': y}
    
    def train_and_visualize(self, dataset_name, kernels=['rbf', 'poly', 'sigmoid'],
                           kernel_params=None):
        """Train multiple SVM models and visualize"""
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        X = self.datasets[dataset_name]['X']
        y = self.datasets[dataset_name]['y']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_kernels = len(kernels)
        fig, axes = plt.subplots(2, n_kernels, figsize=self.figsize)
        if n_kernels == 1:
            axes = axes.reshape(-1, 1)
        
        for i, kernel in enumerate(kernels):
            # Set kernel parameters
            params = {'kernel': kernel, 'random_state': 42}
            if kernel_params and kernel in kernel_params:
                params.update(kernel_params[kernel])
            
            # Train model
            model = SVC(**params)
            model.fit(X_scaled, y)
            
            # Store model
            model_key = f"{dataset_name}_{kernel}"
            self.models[model_key] = {
                'model': model,
                'scaler': scaler,
                'accuracy': model.score(X_scaled, y)
            }
            
            # Plot original data
            ax1 = axes[0, i]
            scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                                edgecolors='black', s=50)
            ax1.set_title(f'{kernel.upper()} - Original Data')
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
            ax1.grid(True, alpha=0.3)
            
            # Plot decision boundary on scaled data
            ax2 = axes[1, i]
            h = 0.02
            x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
            y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax2.contourf(xx, yy, Z, alpha=0.6, cmap='viridis')
            scatter = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, 
                                cmap='viridis', edgecolors='black', s=50)
            
            # Highlight support vectors
            ax2.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                       s=100, linewidth=2, facecolors='none', edgecolors='red',
                       label='Support Vectors')
            
            ax2.set_title(f'{kernel.upper()} - Decision Boundary\n'
                         f'Accuracy: {model.score(X_scaled, y):.3f}, '
                         f'SVs: {len(model.support_vectors_)}')
            ax2.set_xlabel('Scaled Feature 1')
            ax2.set_ylabel('Scaled Feature 2')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_performance(self):
        """Compare performance of all trained models"""
        if not self.models:
            print("No models trained yet!")
            return
        
        performance_data = []
        for model_key, model_info in self.models.items():
            dataset, kernel = model_key.split('_', 1)
            performance_data.append({
                'Dataset': dataset,
                'Kernel': kernel,
                'Accuracy': model_info['accuracy'],
                'Support_Vectors': len(model_info['model'].support_vectors_)
            })
        
        df = pd.DataFrame(performance_data)
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        pivot_acc = df.pivot(index='Dataset', columns='Kernel', values='Accuracy')
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='viridis', ax=ax1)
        ax1.set_title('Accuracy Comparison')
        
        # Support vectors comparison
        pivot_sv = df.pivot(index='Dataset', columns='Kernel', values='Support_Vectors')
        sns.heatmap(pivot_sv, annot=True, fmt='d', cmap='plasma', ax=ax2)
        ax2.set_title('Support Vectors Count')
        
        plt.tight_layout()
        plt.show()
        
        return df

# 4. Using the Advanced Visualizer
print("\n=== Advanced SVM Visualizer ===")

visualizer = SVMVisualizer(figsize=(15, 10))

# Add datasets
visualizer.add_dataset('circles', X_circles, y_circles)
visualizer.add_dataset('moons', X_moons, y_moons)

# Define custom kernel parameters
kernel_params = {
    'rbf': {'gamma': 'scale', 'C': 1.0},
    'poly': {'degree': 3, 'gamma': 'scale', 'C': 1.0},
    'sigmoid': {'gamma': 'scale', 'C': 1.0}
}

# Visualize circles dataset
visualizer.train_and_visualize('circles', kernels=['rbf', 'poly', 'sigmoid'],
                              kernel_params=kernel_params)

# Visualize moons dataset
visualizer.train_and_visualize('moons', kernels=['rbf', 'poly', 'sigmoid'],
                              kernel_params=kernel_params)

# Compare performance
performance_df = visualizer.compare_performance()
print("\nPerformance Summary:")
print(performance_df)

# 5. Interactive Parameter Analysis
print("\n=== Parameter Analysis ===")

def analyze_rbf_parameters(X, y, C_values=[0.1, 1, 10], 
                          gamma_values=[0.01, 0.1, 1]):
    """Analyze effect of C and gamma parameters on RBF kernel"""
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_c = len(C_values)
    n_gamma = len(gamma_values)
    
    fig, axes = plt.subplots(n_c, n_gamma, figsize=(4*n_gamma, 4*n_c))
    
    for i, C in enumerate(C_values):
        for j, gamma in enumerate(gamma_values):
            # Train SVM
            svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
            svm.fit(X_scaled, y)
            
            # Plot decision boundary
            ax = axes[i, j] if n_c > 1 else axes[j]
            
            h = 0.02
            x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
            y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.6, cmap='RdYlBu')
            ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, 
                      cmap='RdYlBu', edgecolors='black', s=50)
            ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                      s=100, linewidth=2, facecolors='none', edgecolors='red')
            
            accuracy = svm.score(X_scaled, y)
            ax.set_title(f'C={C}, γ={gamma}\nAcc: {accuracy:.3f}, '
                        f'SVs: {len(svm.support_vectors_)}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()

# Analyze RBF parameters on circles dataset
analyze_rbf_parameters(X_circles, y_circles)

print("\n✅ SVM decision boundary visualization complete")
```

### Explanation:
1. **Decision Boundary Visualization**: Shows how non-linear kernels separate complex patterns
2. **Kernel Comparison**: Compares RBF, polynomial, and sigmoid kernels
3. **Support Vector Highlighting**: Identifies the critical data points
4. **Parameter Analysis**: Studies effect of C and gamma on decision boundaries
5. **Advanced Visualizer**: Comprehensive tool for multiple datasets and kernels

### Use Cases:
- **Pattern Recognition**: Visualize complex classification boundaries
- **Kernel Selection**: Choose appropriate kernel for data patterns
- **Parameter Tuning**: Understand hyperparameter effects visually
- **Educational Purposes**: Demonstrate SVM concepts and kernel functions

### Best Practices:
- **Feature Scaling**: Always scale features before SVM training
- **Kernel Selection**: Choose kernel based on data structure and complexity
- **Parameter Tuning**: Use grid search for optimal C and gamma values
- **Visualization Resolution**: Balance detail with computational efficiency

### Common Pitfalls:
- **Overfitting**: Too complex kernels on simple data
- **Underfitting**: Linear kernels on highly non-linear data
- **Scale Sensitivity**: Not scaling features properly
- **Parameter Selection**: Not tuning hyperparameters appropriately

### Debugging:
```python
def debug_svm_boundary(model, X, y):
    print(f"Kernel: {model.kernel}")
    print(f"Support vectors: {len(model.support_vectors_)}")
    print(f"Classes: {model.classes_}")
    print(f"Accuracy: {model.score(X, y):.3f}")
```

### Optimization:
- **Kernel Cache**: Use cache_size parameter for large datasets
- **Probability Estimates**: Enable probability=True for confidence scores
- **Memory Efficiency**: Consider LinearSVC for large linear problems
- **Parallel Processing**: Not available for SVM, consider ensemble methods

---

## Question 10

**Implement dimensionality reduction using PCA with Scikit-Learn and visualize the result.**

**Answer:** Principal Component Analysis (PCA) reduces dataset dimensionality by projecting data onto principal components that capture maximum variance, enabling visualization of high-dimensional data and improving computational efficiency.

### Theory:
- **Principal Components**: Orthogonal vectors that capture maximum variance in data
- **Dimensionality Reduction**: Reduces feature space while preserving most information
- **Variance Explained**: Measures how much information each component retains
- **Data Visualization**: Enables 2D/3D visualization of high-dimensional datasets

### Code Example:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_digits, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class PCAAnalyzer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pca = None
        self.scaler = None
        self.explained_variance_ratio = None
        self.components = None
    
    def basic_pca_example(self, X, y, n_components=2):
        """
        Basic PCA demonstration with visualization
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target vector
        n_components : int, default=2
            Number of components to keep
            
        Returns:
        --------
        X_pca : array-like
            Transformed data
        """
        
        # Standardize the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Store results
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.components = self.pca.components_
        
        print(f"Original shape: {X.shape}")
        print(f"Transformed shape: {X_pca.shape}")
        print(f"Explained variance ratio: {self.explained_variance_ratio}")
        print(f"Total variance explained: {np.sum(self.explained_variance_ratio):.3f}")
        
        return X_pca
    
    def visualize_2d_pca(self, X_pca, y, title="PCA Visualization"):
        """Visualize 2D PCA results"""
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with different colors for each class
        unique_classes = np.unique(y)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
        
        for i, class_label in enumerate(unique_classes):
            mask = y == class_label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=f'Class {class_label}', 
                       alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({self.explained_variance_ratio[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({self.explained_variance_ratio[1]:.2%} variance)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def visualize_3d_pca(self, X, y, title="3D PCA Visualization"):
        """Visualize 3D PCA results"""
        # Apply PCA with 3 components
        X_scaled = self.scaler.fit_transform(X)
        pca_3d = PCA(n_components=3, random_state=self.random_state)
        X_pca_3d = pca_3d.fit_transform(X_scaled)
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        unique_classes = np.unique(y)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
        
        for i, class_label in enumerate(unique_classes):
            mask = y == class_label
            ax.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
                      c=[colors[i]], label=f'Class {class_label}', 
                      alpha=0.7, s=50)
        
        ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
        ax.set_title(title)
        ax.legend()
        plt.show()
        
        return X_pca_3d, pca_3d
    
    def explained_variance_analysis(self, X, max_components=None):
        """Analyze explained variance for different numbers of components"""
        if max_components is None:
            max_components = min(X.shape[0], X.shape[1])
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA with all components
        pca_full = PCA(random_state=self.random_state)
        pca_full.fit(X_scaled)
        
        # Calculate cumulative explained variance
        explained_variance = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Plot explained variance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Individual explained variance
        components = range(1, len(explained_variance) + 1)
        ax1.bar(components, explained_variance, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Individual Explained Variance by Component')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative explained variance
        ax2.plot(components, cumulative_variance, 'o-', color='red', linewidth=2)
        ax2.axhline(y=0.95, color='green', linestyle='--', label='95% threshold')
        ax2.axhline(y=0.90, color='orange', linestyle='--', label='90% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal number of components
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        
        print(f"Components for 90% variance: {n_components_90}")
        print(f"Components for 95% variance: {n_components_95}")
        
        return explained_variance, cumulative_variance
    
    def pca_feature_importance(self, X, feature_names=None):
        """Analyze feature importance in principal components"""
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Create heatmap of component loadings
        components_df = pd.DataFrame(
            self.components.T,
            columns=[f'PC{i+1}' for i in range(self.components.shape[0])],
            index=feature_names
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(components_df, annot=True, cmap='RdBu_r', center=0,
                   fmt='.2f', square=True)
        plt.title('PCA Component Loadings')
        plt.xlabel('Principal Components')
        plt.ylabel('Original Features')
        plt.tight_layout()
        plt.show()
        
        # Show top contributing features for each component
        for i, component in enumerate(self.components):
            print(f"\nPrincipal Component {i+1}:")
            feature_importance = np.abs(component)
            top_features = np.argsort(feature_importance)[-5:][::-1]
            
            for j, feature_idx in enumerate(top_features):
                print(f"  {j+1}. {feature_names[feature_idx]}: {component[feature_idx]:.3f}")
        
        return components_df

# 1. Basic PCA Example with Iris Dataset
print("=== Basic PCA Example: Iris Dataset ===")

# Load Iris dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
feature_names_iris = iris.feature_names

# Initialize PCA analyzer
pca_analyzer = PCAAnalyzer()

# Apply 2D PCA
X_iris_pca = pca_analyzer.basic_pca_example(X_iris, y_iris, n_components=2)

# Visualize 2D results
pca_analyzer.visualize_2d_pca(X_iris_pca, y_iris, "Iris Dataset - PCA Visualization")

# Analyze explained variance
explained_var, cumulative_var = pca_analyzer.explained_variance_analysis(X_iris)

# Feature importance analysis
components_df = pca_analyzer.pca_feature_importance(X_iris, feature_names_iris)

# 2. 3D PCA Visualization
print("\n=== 3D PCA Visualization ===")
X_iris_3d, pca_3d = pca_analyzer.visualize_3d_pca(X_iris, y_iris, "Iris Dataset - 3D PCA")

# 3. High-Dimensional Dataset Example
print("\n=== High-Dimensional Dataset: Digits ===")

# Load digits dataset (8x8 images = 64 features)
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

print(f"Digits dataset shape: {X_digits.shape}")

# Apply PCA for visualization
pca_digits = PCAAnalyzer()
X_digits_pca = pca_digits.basic_pca_example(X_digits, y_digits, n_components=2)

# Visualize
pca_digits.visualize_2d_pca(X_digits_pca, y_digits, "Digits Dataset - PCA Visualization")

# Explained variance analysis
explained_digits, cumulative_digits = pca_digits.explained_variance_analysis(X_digits)

# 4. PCA for Dimensionality Reduction in Machine Learning
print("\n=== PCA for Machine Learning Pipeline ===")

# Create high-dimensional synthetic dataset
X_synthetic, y_synthetic = make_classification(
    n_samples=1000, n_features=50, n_informative=20,
    n_redundant=10, n_clusters_per_class=2, random_state=42
)

print(f"Synthetic dataset shape: {X_synthetic.shape}")

# Compare performance with and without PCA
def compare_with_without_pca(X, y, n_components=20):
    """Compare model performance with and without PCA"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Without PCA
    pipeline_no_pca = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline_no_pca.fit(X_train, y_train)
    y_pred_no_pca = pipeline_no_pca.predict(X_test)
    accuracy_no_pca = accuracy_score(y_test, y_pred_no_pca)
    
    # With PCA
    pipeline_pca = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline_pca.fit(X_train, y_train)
    y_pred_pca = pipeline_pca.predict(X_test)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    
    # Dimensionality reduction ratio
    reduction_ratio = (X.shape[1] - n_components) / X.shape[1]
    
    print(f"Original features: {X.shape[1]}")
    print(f"PCA features: {n_components}")
    print(f"Dimensionality reduction: {reduction_ratio:.1%}")
    print(f"Accuracy without PCA: {accuracy_no_pca:.3f}")
    print(f"Accuracy with PCA: {accuracy_pca:.3f}")
    print(f"Performance difference: {accuracy_pca - accuracy_no_pca:.3f}")
    
    # Get explained variance from PCA step
    pca_step = pipeline_pca.named_steps['pca']
    total_variance = np.sum(pca_step.explained_variance_ratio_)
    print(f"Variance retained: {total_variance:.1%}")
    
    return {
        'accuracy_no_pca': accuracy_no_pca,
        'accuracy_pca': accuracy_pca,
        'variance_retained': total_variance,
        'reduction_ratio': reduction_ratio
    }

# Compare performance
results = compare_with_without_pca(X_synthetic, y_synthetic, n_components=20)

# 5. Advanced PCA Analysis
print("\n=== Advanced PCA Analysis ===")

class AdvancedPCAAnalyzer:
    def __init__(self):
        self.results = {}
    
    def optimal_components_analysis(self, X, variance_thresholds=[0.90, 0.95, 0.99]):
        """Find optimal number of components for different variance thresholds"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA()
        pca.fit(X_scaled)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        results = {}
        for threshold in variance_thresholds:
            n_components = np.argmax(cumulative_variance >= threshold) + 1
            results[f'{threshold:.0%}'] = n_components
        
        print("Optimal number of components for variance thresholds:")
        for threshold, n_comp in results.items():
            print(f"  {threshold}: {n_comp} components")
        
        return results
    
    def reconstruction_error_analysis(self, X, n_components_list=[5, 10, 20, 30]):
        """Analyze reconstruction error for different numbers of components"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        reconstruction_errors = []
        
        for n_comp in n_components_list:
            # Apply PCA
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X_scaled)
            
            # Reconstruct data
            X_reconstructed = pca.inverse_transform(X_pca)
            
            # Calculate reconstruction error (MSE)
            mse = np.mean((X_scaled - X_reconstructed) ** 2)
            reconstruction_errors.append(mse)
        
        # Plot reconstruction error
        plt.figure(figsize=(10, 6))
        plt.plot(n_components_list, reconstruction_errors, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Number of Components')
        plt.ylabel('Reconstruction Error (MSE)')
        plt.title('PCA Reconstruction Error vs Number of Components')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print("Reconstruction errors:")
        for n_comp, error in zip(n_components_list, reconstruction_errors):
            print(f"  {n_comp} components: {error:.4f}")
        
        return reconstruction_errors

# Advanced analysis
advanced_analyzer = AdvancedPCAAnalyzer()

# Optimal components analysis
optimal_components = advanced_analyzer.optimal_components_analysis(X_synthetic)

# Reconstruction error analysis
reconstruction_errors = advanced_analyzer.reconstruction_error_analysis(X_synthetic)

# 6. Image Visualization Example
print("\n=== Image PCA Example ===")

# Visualize some original digits and their PCA reconstructions
def visualize_digit_reconstruction(X, pca, n_samples=5):
    """Visualize original vs PCA-reconstructed digit images"""
    
    # Transform and reconstruct
    X_pca = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(12, 6))
    
    for i in range(n_samples):
        # Original image
        axes[0, i].imshow(X[i].reshape(8, 8), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed image
        axes[1, i].imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'PCA Reconstruction ({pca.n_components} components)')
    plt.tight_layout()
    plt.show()

# Apply PCA to digits with different numbers of components
scaler_digits = StandardScaler()
X_digits_scaled = scaler_digits.fit_transform(X_digits[:100])  # Use subset for visualization

for n_comp in [10, 20, 30]:
    pca_viz = PCA(n_components=n_comp)
    pca_viz.fit(X_digits_scaled)
    visualize_digit_reconstruction(X_digits_scaled, pca_viz, n_samples=5)

print("\n✅ PCA dimensionality reduction and visualization complete")
```

### Explanation:
1. **Data Standardization**: Scales features to ensure equal contribution to PCA
2. **Component Extraction**: Finds orthogonal vectors capturing maximum variance
3. **Dimensionality Reduction**: Projects high-dimensional data to lower dimensions
4. **Variance Analysis**: Determines optimal number of components to retain
5. **Visualization**: Creates 2D/3D plots for high-dimensional data understanding

### Use Cases:
- **Data Visualization**: Visualize high-dimensional datasets in 2D/3D space
- **Feature Reduction**: Reduce computational complexity while preserving information
- **Noise Reduction**: Remove less important components that may contain noise
- **Data Compression**: Store data more efficiently with minimal information loss

### Best Practices:
- **Standardization**: Always standardize features before applying PCA
- **Variance Threshold**: Choose components that retain 90-95% of variance
- **Cross-validation**: Validate PCA benefits in downstream tasks
- **Interpretability**: Analyze component loadings to understand feature contributions

### Common Pitfalls:
- **No Standardization**: Unscaled features can dominate principal components
- **Over-reduction**: Removing too many components loses important information
- **Linear Assumption**: PCA only captures linear relationships in data
- **Interpretability Loss**: Principal components may not have clear real-world meaning

### Debugging:
```python
def debug_pca(pca, X_original, X_transformed):
    print(f"Original shape: {X_original.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.3f}")
    print(f"Number of components: {pca.n_components_}")
```

### Optimization:
- **Incremental PCA**: Use for large datasets that don't fit in memory
- **Sparse PCA**: When few components explain most variance
- **Kernel PCA**: For non-linear dimensionality reduction
- **Randomized PCA**: Faster computation for large matrices with many features

---

## Question 11

**Create a clustering analysis on a dataset using Scikit-Learn’s DBSCAN method.**

**Answer:**

### Theory
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed while marking points in low-density regions as outliers. Unlike K-means, DBSCAN doesn't require specifying the number of clusters beforehand and can identify clusters of arbitrary shapes.

Key parameters:
- `eps`: Maximum distance between two samples for them to be considered neighbors
- `min_samples`: Minimum number of samples in a neighborhood for a point to be core point
- `metric`: Distance metric to use (euclidean, manhattan, etc.)

### Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Simple DBSCAN demonstration
def simple_dbscan_demo():
    """Simple DBSCAN demonstration with synthetic data."""
    # Generate sample data
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, 
                      random_state=0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Original data
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # DBSCAN results
    plt.subplot(1, 2, 2)
    unique_labels = set(cluster_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'black'
            marker = 'x'
        else:
            marker = 'o'
        
        class_member_mask = (cluster_labels == k)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, s=50)
    
    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f'Estimated number of clusters: {n_clusters}')
    print(f'Estimated number of noise points: {n_noise}')
    
    return cluster_labels

# Advanced DBSCAN with parameter optimization
class DBSCANAnalyzer:
    """DBSCAN clustering analysis with parameter optimization."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_params = None
        self.best_score = -1
    
    def find_optimal_eps(self, X, min_samples=5):
        """Find optimal eps using k-distance graph."""
        from sklearn.neighbors import NearestNeighbors
        
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(X)
        
        distances, indices = neighbors.kneighbors(X)
        distances = np.sort(distances[:, min_samples-1], axis=0)
        
        # Simple elbow detection
        optimal_eps = np.percentile(distances, 90)
        
        return optimal_eps
    
    def optimize_parameters(self, X):
        """Optimize DBSCAN parameters."""
        X_scaled = self.scaler.fit_transform(X)
        
        # Get suggested eps
        suggested_eps = self.find_optimal_eps(X_scaled)
        
        best_score = -1
        best_params = None
        
        # Test different parameter combinations
        eps_values = np.linspace(suggested_eps * 0.5, suggested_eps * 1.5, 5)
        min_samples_values = [3, 5, 10]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_scaled)
                
                # Check if we have valid clusters
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters > 1:
                    # Calculate silhouette score for non-noise points
                    mask = labels != -1
                    if np.sum(mask) > min_samples:
                        score = silhouette_score(X_scaled[mask], labels[mask])
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'eps': eps, 'min_samples': min_samples}
        
        self.best_params = best_params
        self.best_score = best_score
        
        return best_params, best_score
    
    def fit_predict(self, X):
        """Fit DBSCAN with optimized parameters."""
        # Optimize parameters if not done
        if self.best_params is None:
            self.optimize_parameters(X)
        
        X_scaled = self.scaler.fit_transform(X)
        dbscan = DBSCAN(**self.best_params)
        labels = dbscan.fit_predict(X_scaled)
        
        return labels
    
    def visualize(self, X, labels, title="DBSCAN Clustering"):
        """Visualize clustering results."""
        plt.figure(figsize=(10, 8))
        
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = 'black'
                marker = 'x'
                label = 'Noise'
            else:
                marker = 'o'
                label = f'Cluster {k}'
            
            class_member_mask = (labels == k)
            xy = X[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker,
                       s=50, alpha=0.7)
        
        plt.title(f'{title}\nClusters: {len(set(labels)) - (1 if -1 in labels else 0)}, '
                 f'Noise points: {list(labels).count(-1)}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
        plt.show()

# Real-world example
def iris_dbscan_example():
    """DBSCAN example with Iris dataset."""
    from sklearn.datasets import load_iris
    
    # Load data
    iris = load_iris()
    X = iris.data[:, [0, 2]]  # Use sepal length and petal length
    
    # Apply DBSCAN analysis
    analyzer = DBSCANAnalyzer()
    
    # Optimize parameters
    best_params, best_score = analyzer.optimize_parameters(X)
    print(f"Best parameters: {best_params}")
    print(f"Best silhouette score: {best_score:.4f}")
    
    # Fit and predict
    labels = analyzer.fit_predict(X)
    
    # Visualize results
    analyzer.visualize(X, labels, "DBSCAN - Iris Dataset")
    
    # Print cluster statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    return analyzer, labels

# Run demonstrations
if __name__ == "__main__":
    print("=== SIMPLE DBSCAN DEMONSTRATION ===")
    simple_labels = simple_dbscan_demo()
    
    print("\n=== IRIS DATASET EXAMPLE ===")
    analyzer, iris_labels = iris_dbscan_example()
```

### Explanation

1. **DBSCAN Algorithm**: Density-based clustering that finds arbitrary-shaped clusters
2. **Parameter Optimization**: Automatic tuning of eps and min_samples parameters
3. **Noise Detection**: Identifies outliers as noise points (labeled as -1)
4. **Visualization**: Clear plots showing clusters and noise points
5. **Real-world Application**: Demonstrates usage on Iris dataset

### Use Cases

1. **Anomaly Detection**: Identifying outliers using noise points
2. **Customer Segmentation**: Grouping without predefined cluster count
3. **Image Segmentation**: Clustering pixels based on similarity
4. **Geographic Analysis**: Finding density-based regions
5. **Market Research**: Discovering natural groupings

### Best Practices

1. **Data Preprocessing**: Always scale features before applying DBSCAN
2. **Parameter Selection**: Use k-distance graph for eps selection
3. **Noise Handling**: Consider noise points as valuable outlier information
4. **Validation**: Use silhouette score for parameter optimization
5. **Visualization**: Always visualize results to understand clustering

### Common Pitfalls

1. **Inappropriate eps**: Too small creates many clusters; too large merges clusters
2. **High Dimensionality**: DBSCAN struggles with high-dimensional data
3. **Varying Densities**: May not work well with clusters of different densities
4. **Feature Scaling**: Forgetting to scale can bias distance calculations
5. **Parameter Sensitivity**: Small changes can dramatically affect results

### Debugging Tips

```python
# Parameter sensitivity analysis
def analyze_parameter_sensitivity(X, eps_range, min_samples_range):
    """Analyze how parameters affect clustering results."""
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            })
    
    return pd.DataFrame(results)
```

### Optimization Techniques

1. **Efficient Neighbors**: Use Ball Tree or KD Tree for faster neighbor searches
2. **Approximate Methods**: Consider OPTICS for varying density clusters
3. **Parallel Processing**: Use n_jobs parameter for parallel computation
4. **Memory Optimization**: Process large datasets in chunks
5. **Feature Engineering**: Create meaningful distance-based features

---

## Question 12

**How do you save a trained Scikit-Learn model to disk and load it back for later use?**

**Answer:**

### Theory
Model persistence is crucial for machine learning applications, allowing you to save trained models and reuse them without retraining. Scikit-Learn provides several methods for model serialization: `joblib` (recommended), `pickle`, and manual state saving. The `joblib` library is optimized for NumPy arrays and is more efficient for scikit-learn models.

Key considerations:
- **joblib**: Faster for large NumPy arrays, recommended for scikit-learn
- **pickle**: Standard Python serialization, compatible but slower
- **Version compatibility**: Models saved with different scikit-learn versions may not load
- **Security**: Only load models from trusted sources

### Code Example

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelPersistenceManager:
    """Comprehensive model persistence and management system."""
    
    def __init__(self, base_path="models"):
        self.base_path = base_path
        self.models_registry = {}
        
        # Create models directory if it doesn't exist
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        # Load existing registry
        self.registry_path = os.path.join(base_path, "models_registry.json")
        self.load_registry()
    
    def save_model_joblib(self, model, model_name, metadata=None):
        """Save model using joblib (recommended for scikit-learn)."""
        model_path = os.path.join(self.base_path, f"{model_name}_joblib.pkl")
        
        # Save the model
        joblib.dump(model, model_path)
        
        # Save metadata
        model_info = {
            'name': model_name,
            'path': model_path,
            'method': 'joblib',
            'created_at': datetime.now().isoformat(),
            'sklearn_version': self._get_sklearn_version(),
            'model_type': type(model).__name__,
            'metadata': metadata or {}
        }
        
        self.models_registry[model_name] = model_info
        self.save_registry()
        
        print(f"Model '{model_name}' saved successfully using joblib at: {model_path}")
        return model_path
    
    def load_model_joblib(self, model_name):
        """Load model using joblib."""
        if model_name not in self.models_registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_info = self.models_registry[model_name]
        model_path = model_info['path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        
        print(f"Model '{model_name}' loaded successfully from: {model_path}")
        print(f"Model type: {model_info['model_type']}")
        print(f"Created at: {model_info['created_at']}")
        
        return model, model_info
    
    def save_model_pickle(self, model, model_name, metadata=None):
        """Save model using pickle."""
        model_path = os.path.join(self.base_path, f"{model_name}_pickle.pkl")
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        model_info = {
            'name': model_name,
            'path': model_path,
            'method': 'pickle',
            'created_at': datetime.now().isoformat(),
            'sklearn_version': self._get_sklearn_version(),
            'model_type': type(model).__name__,
            'metadata': metadata or {}
        }
        
        self.models_registry[model_name] = model_info
        self.save_registry()
        
        print(f"Model '{model_name}' saved successfully using pickle at: {model_path}")
        return model_path
    
    def load_model_pickle(self, model_name):
        """Load model using pickle."""
        if model_name not in self.models_registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_info = self.models_registry[model_name]
        model_path = model_info['path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model '{model_name}' loaded successfully from: {model_path}")
        print(f"Model type: {model_info['model_type']}")
        print(f"Created at: {model_info['created_at']}")
        
        return model, model_info
    
    def save_pipeline(self, pipeline, pipeline_name, metadata=None):
        """Save complete pipeline including preprocessors."""
        pipeline_path = os.path.join(self.base_path, f"{pipeline_name}_pipeline.pkl")
        
        # Save the pipeline using joblib
        joblib.dump(pipeline, pipeline_path)
        
        # Save pipeline metadata
        pipeline_info = {
            'name': pipeline_name,
            'path': pipeline_path,
            'method': 'joblib_pipeline',
            'created_at': datetime.now().isoformat(),
            'sklearn_version': self._get_sklearn_version(),
            'pipeline_steps': [step[0] for step in pipeline.steps],
            'metadata': metadata or {}
        }
        
        self.models_registry[pipeline_name] = pipeline_info
        self.save_registry()
        
        print(f"Pipeline '{pipeline_name}' saved successfully at: {pipeline_path}")
        return pipeline_path
    
    def load_pipeline(self, pipeline_name):
        """Load complete pipeline."""
        if pipeline_name not in self.models_registry:
            raise ValueError(f"Pipeline '{pipeline_name}' not found in registry")
        
        pipeline_info = self.models_registry[pipeline_name]
        pipeline_path = pipeline_info['path']
        
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
        
        pipeline = joblib.load(pipeline_path)
        
        print(f"Pipeline '{pipeline_name}' loaded successfully from: {pipeline_path}")
        print(f"Pipeline steps: {pipeline_info['pipeline_steps']}")
        
        return pipeline, pipeline_info
    
    def list_models(self):
        """List all saved models."""
        if not self.models_registry:
            print("No models found in registry.")
            return
        
        print("Saved Models:")
        print("-" * 80)
        for name, info in self.models_registry.items():
            print(f"Name: {name}")
            print(f"Type: {info.get('model_type', 'Unknown')}")
            print(f"Method: {info['method']}")
            print(f"Created: {info['created_at']}")
            print(f"Path: {info['path']}")
            print("-" * 80)
    
    def delete_model(self, model_name):
        """Delete a saved model."""
        if model_name not in self.models_registry:
            print(f"Model '{model_name}' not found in registry.")
            return
        
        model_info = self.models_registry[model_name]
        model_path = model_info['path']
        
        # Delete the file
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Model file deleted: {model_path}")
        
        # Remove from registry
        del self.models_registry[model_name]
        self.save_registry()
        
        print(f"Model '{model_name}' removed from registry.")
    
    def save_registry(self):
        """Save the models registry to JSON file."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.models_registry, f, indent=2)
    
    def load_registry(self):
        """Load the models registry from JSON file."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.models_registry = json.load(f)
    
    def _get_sklearn_version(self):
        """Get current scikit-learn version."""
        try:
            import sklearn
            return sklearn.__version__
        except:
            return "Unknown"

# Comprehensive demonstration
def comprehensive_model_persistence_demo():
    """Comprehensive demonstration of model persistence."""
    
    # Initialize persistence manager
    manager = ModelPersistenceManager("demo_models")
    
    print("=== COMPREHENSIVE MODEL PERSISTENCE DEMONSTRATION ===\n")
    
    # Load sample data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 1. Simple Model Persistence
    print("1. SIMPLE MODEL PERSISTENCE")
    print("-" * 40)
    
    # Train a simple model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Test accuracy before saving
    initial_accuracy = rf_model.score(X_test, y_test)
    print(f"Initial model accuracy: {initial_accuracy:.4f}")
    
    # Save model using joblib
    model_metadata = {
        'dataset': 'iris',
        'features': iris.feature_names.tolist(),
        'target_names': iris.target_names.tolist(),
        'accuracy': initial_accuracy,
        'n_samples_train': len(X_train)
    }
    
    manager.save_model_joblib(rf_model, "iris_random_forest", model_metadata)
    
    # Load model and test
    loaded_model, model_info = manager.load_model_joblib("iris_random_forest")
    loaded_accuracy = loaded_model.score(X_test, y_test)
    
    print(f"Loaded model accuracy: {loaded_accuracy:.4f}")
    print(f"Accuracy match: {initial_accuracy == loaded_accuracy}")
    print()
    
    # 2. Pipeline Persistence
    print("2. PIPELINE PERSISTENCE")
    print("-" * 40)
    
    # Create a pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Train pipeline
    pipeline.fit(X_train, y_train)
    initial_pipeline_accuracy = pipeline.score(X_test, y_test)
    print(f"Initial pipeline accuracy: {initial_pipeline_accuracy:.4f}")
    
    # Save pipeline
    pipeline_metadata = {
        'dataset': 'iris',
        'preprocessing': 'StandardScaler',
        'classifier': 'LogisticRegression',
        'accuracy': initial_pipeline_accuracy
    }
    
    manager.save_pipeline(pipeline, "iris_logistic_pipeline", pipeline_metadata)
    
    # Load pipeline and test
    loaded_pipeline, pipeline_info = manager.load_pipeline("iris_logistic_pipeline")
    loaded_pipeline_accuracy = loaded_pipeline.score(X_test, y_test)
    
    print(f"Loaded pipeline accuracy: {loaded_pipeline_accuracy:.4f}")
    print(f"Accuracy match: {initial_pipeline_accuracy == loaded_pipeline_accuracy}")
    print()
    
    # 3. Compare joblib vs pickle
    print("3. JOBLIB VS PICKLE COMPARISON")
    print("-" * 40)
    
    # Create a larger model for comparison
    X_large, y_large = make_classification(
        n_samples=10000, n_features=100, n_classes=3, random_state=42
    )
    X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(
        X_large, y_large, test_size=0.3, random_state=42
    )
    
    large_model = RandomForestClassifier(n_estimators=100, random_state=42)
    large_model.fit(X_train_large, y_train_large)
    
    # Time joblib save/load
    import time
    
    start_time = time.time()
    manager.save_model_joblib(large_model, "large_model_joblib")
    joblib_save_time = time.time() - start_time
    
    start_time = time.time()
    loaded_joblib_model, _ = manager.load_model_joblib("large_model_joblib")
    joblib_load_time = time.time() - start_time
    
    # Time pickle save/load
    start_time = time.time()
    manager.save_model_pickle(large_model, "large_model_pickle")
    pickle_save_time = time.time() - start_time
    
    start_time = time.time()
    loaded_pickle_model, _ = manager.load_model_pickle("large_model_pickle")
    pickle_load_time = time.time() - start_time
    
    print(f"Joblib - Save time: {joblib_save_time:.4f}s, Load time: {joblib_load_time:.4f}s")
    print(f"Pickle - Save time: {pickle_save_time:.4f}s, Load time: {pickle_load_time:.4f}s")
    print()
    
    # 4. List all models
    print("4. MODEL REGISTRY")
    print("-" * 40)
    manager.list_models()
    
    return manager

# Simple usage examples
def simple_examples():
    """Simple examples of model persistence."""
    print("\n=== SIMPLE USAGE EXAMPLES ===\n")
    
    # Load sample data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    print("1. BASIC JOBLIB USAGE")
    print("-" * 25)
    
    # Save with joblib
    joblib.dump(model, 'my_model.pkl')
    print("Model saved with joblib.dump()")
    
    # Load with joblib
    loaded_model = joblib.load('my_model.pkl')
    print("Model loaded with joblib.load()")
    
    # Test loaded model
    accuracy = loaded_model.score(X_test, y_test)
    print(f"Loaded model accuracy: {accuracy:.4f}")
    
    print("\n2. BASIC PICKLE USAGE")
    print("-" * 25)
    
    # Save with pickle
    with open('my_model_pickle.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved with pickle.dump()")
    
    # Load with pickle
    with open('my_model_pickle.pkl', 'rb') as f:
        loaded_pickle_model = pickle.load(f)
    print("Model loaded with pickle.load()")
    
    # Test loaded model
    pickle_accuracy = loaded_pickle_model.score(X_test, y_test)
    print(f"Loaded pickle model accuracy: {pickle_accuracy:.4f}")
    
    # Clean up
    os.remove('my_model.pkl')
    os.remove('my_model_pickle.pkl')
    print("\nTemporary files cleaned up.")

# Run demonstrations
if __name__ == "__main__":
    # Simple examples
    simple_examples()
    
    # Comprehensive demo
    manager = comprehensive_model_persistence_demo()
    
    # Clean up demo models (optional)
    # import shutil
    # shutil.rmtree("demo_models")
    # print("\nDemo models directory cleaned up.")
```

### Explanation

1. **ModelPersistenceManager Class**: Comprehensive system for saving/loading models with metadata tracking
2. **Multiple Methods**: Supports both joblib and pickle serialization methods
3. **Pipeline Support**: Can save complete pipelines including preprocessors
4. **Registry System**: Maintains JSON registry of all saved models with metadata
5. **Performance Comparison**: Demonstrates speed differences between joblib and pickle

### Use Cases

1. **Production Deployment**: Save trained models for web service deployment
2. **Model Versioning**: Keep track of different model versions and their performance
3. **Batch Processing**: Save models for later batch prediction tasks
4. **Experimentation**: Store experimental models for comparison
5. **Backup and Recovery**: Create backups of important trained models

### Best Practices

1. **Use joblib for scikit-learn**: It's optimized for NumPy arrays and faster
2. **Version Tracking**: Always track scikit-learn version used to train models
3. **Metadata Storage**: Store important information about model performance and data
4. **Security**: Only load models from trusted sources
5. **File Organization**: Use clear naming conventions and organized directory structure

### Common Pitfalls

1. **Version Incompatibility**: Models may not load with different scikit-learn versions
2. **Large File Sizes**: Complex models can create very large files
3. **Missing Dependencies**: Ensure all required libraries are available when loading
4. **Path Issues**: Use absolute paths or proper relative path handling
5. **Memory Usage**: Large models consume significant memory when loaded

### Debugging Tips

```python
# Check model compatibility
def check_model_compatibility(model_path):
    """Check if a saved model can be loaded."""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully: {type(model).__name__}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Model size checker
def get_model_size(model_path):
    """Get the size of a saved model file."""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        return size_mb
    else:
        print("Model file not found")
        return 0
```

### Optimization Techniques

1. **Compression**: Use `compress` parameter in joblib for smaller files
2. **Partial Loading**: Load only necessary components of complex pipelines
3. **Model Pruning**: Remove unnecessary parameters before saving
4. **Batch Operations**: Save/load multiple models efficiently
5. **Cloud Storage**: Integrate with cloud storage for scalable model management

---

## Question 13

**How can you implement custom transformers in Scikit-Learn?**

**Answer:**

### Theory
Custom transformers in Scikit-Learn allow you to create reusable preprocessing components that integrate seamlessly with pipelines. By inheriting from `BaseEstimator` and `TransformerMixin`, you can implement the `fit()` and `transform()` methods to create transformers that follow scikit-learn conventions and can be used in cross-validation and grid search.

Key concepts:
- **BaseEstimator**: Provides `get_params()` and `set_params()` methods
- **TransformerMixin**: Provides `fit_transform()` method automatically
- **fit()**: Learn parameters from training data
- **transform()**: Apply transformation to data
- **inverse_transform()**: Optional method to reverse transformation

### Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 1. Simple Feature Selector
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to select specific features by index or name."""
    
    def __init__(self, feature_indices=None, feature_names=None):
        self.feature_indices = feature_indices
        self.feature_names = feature_names
        self.selected_indices_ = None
    
    def fit(self, X, y=None):
        """Learn which features to select."""
        if self.feature_names is not None:
            if hasattr(X, 'columns'):
                # DataFrame input
                self.selected_indices_ = [X.columns.get_loc(name) 
                                        for name in self.feature_names 
                                        if name in X.columns]
            else:
                raise ValueError("feature_names provided but X is not a DataFrame")
        elif self.feature_indices is not None:
            self.selected_indices_ = self.feature_indices
        else:
            # If nothing specified, select all features
            self.selected_indices_ = list(range(X.shape[1]))
        
        return self
    
    def transform(self, X):
        """Select the specified features."""
        if self.selected_indices_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        
        if hasattr(X, 'iloc'):
            # DataFrame input
            return X.iloc[:, self.selected_indices_].values
        else:
            # NumPy array input
            return X[:, self.selected_indices_]

# 2. Outlier Remover
class OutlierRemover(BaseEstimator, TransformerMixin):
    """Custom transformer to remove outliers using IQR method."""
    
    def __init__(self, factor=1.5, method='iqr'):
        self.factor = factor
        self.method = method
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        """Learn outlier bounds from training data."""
        X = np.array(X)
        
        if self.method == 'iqr':
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            iqr = q3 - q1
            
            self.lower_bounds_ = q1 - (self.factor * iqr)
            self.upper_bounds_ = q3 + (self.factor * iqr)
        
        elif self.method == 'zscore':
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            
            self.lower_bounds_ = mean - (self.factor * std)
            self.upper_bounds_ = mean + (self.factor * std)
        
        return self
    
    def transform(self, X):
        """Remove outliers based on learned bounds."""
        X = np.array(X)
        
        # Create mask for non-outlier rows
        mask = np.all((X >= self.lower_bounds_) & (X <= self.upper_bounds_), axis=1)
        
        return X[mask]

# 3. Log Transformer
class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for log transformation with handling of negative values."""
    
    def __init__(self, offset=1, columns=None):
        self.offset = offset
        self.columns = columns
        self.min_values_ = None
    
    def fit(self, X, y=None):
        """Learn minimum values for offset calculation."""
        X = np.array(X)
        
        if self.columns is not None:
            self.min_values_ = np.min(X[:, self.columns], axis=0)
        else:
            self.min_values_ = np.min(X, axis=0)
        
        return self
    
    def transform(self, X):
        """Apply log transformation."""
        X = np.array(X).copy()
        
        if self.columns is not None:
            # Apply to specific columns
            for i, col in enumerate(self.columns):
                # Ensure all values are positive
                offset_value = max(self.offset, abs(self.min_values_[i]) + 1)
                X[:, col] = np.log(X[:, col] + offset_value)
        else:
            # Apply to all columns
            for i in range(X.shape[1]):
                offset_value = max(self.offset, abs(self.min_values_[i]) + 1)
                X[:, i] = np.log(X[:, i] + offset_value)
        
        return X
    
    def inverse_transform(self, X):
        """Reverse the log transformation."""
        X = np.array(X).copy()
        
        if self.columns is not None:
            for i, col in enumerate(self.columns):
                offset_value = max(self.offset, abs(self.min_values_[i]) + 1)
                X[:, col] = np.exp(X[:, col]) - offset_value
        else:
            for i in range(X.shape[1]):
                offset_value = max(self.offset, abs(self.min_values_[i]) + 1)
                X[:, i] = np.exp(X[:, i]) - offset_value
        
        return X

# 4. Polynomial Feature Generator
class PolynomialFeatures(BaseEstimator, TransformerMixin):
    """Custom polynomial feature generator."""
    
    def __init__(self, degree=2, include_bias=False, interaction_only=False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.n_features_ = None
    
    def fit(self, X, y=None):
        """Learn the number of input features."""
        X = np.array(X)
        self.n_features_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Generate polynomial features."""
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Start with original features
        features = [X]
        
        # Add bias term if requested
        if self.include_bias:
            bias = np.ones((n_samples, 1))
            features.insert(0, bias)
        
        # Generate polynomial features
        for degree in range(2, self.degree + 1):
            if self.interaction_only:
                # Only interaction terms
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                        features.append(interaction)
            else:
                # All polynomial terms
                for i in range(n_features):
                    poly_feature = (X[:, i] ** degree).reshape(-1, 1)
                    features.append(poly_feature)
        
        return np.hstack(features)

# 5. Custom Scaler
class RobustScaler(BaseEstimator, TransformerMixin):
    """Custom robust scaler using median and IQR."""
    
    def __init__(self, quantile_range=(25.0, 75.0)):
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None
    
    def fit(self, X, y=None):
        """Learn median and IQR from training data."""
        X = np.array(X)
        
        self.center_ = np.median(X, axis=0)
        
        q1 = np.percentile(X, self.quantile_range[0], axis=0)
        q3 = np.percentile(X, self.quantile_range[1], axis=0)
        self.scale_ = q3 - q1
        
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        
        return self
    
    def transform(self, X):
        """Apply robust scaling."""
        X = np.array(X)
        return (X - self.center_) / self.scale_
    
    def inverse_transform(self, X):
        """Reverse the scaling."""
        X = np.array(X)
        return (X * self.scale_) + self.center_

# 6. DataFrame Transformer
class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer that works with pandas DataFrames."""
    
    def __init__(self, func, columns=None):
        self.func = func
        self.columns = columns
        self.fitted_columns_ = None
    
    def fit(self, X, y=None):
        """Store column information."""
        if hasattr(X, 'columns'):
            if self.columns is not None:
                self.fitted_columns_ = self.columns
            else:
                self.fitted_columns_ = X.columns.tolist()
        
        return self
    
    def transform(self, X):
        """Apply function to specified columns."""
        if hasattr(X, 'copy'):
            X_transformed = X.copy()
        else:
            X_transformed = pd.DataFrame(X)
        
        if self.fitted_columns_ is not None:
            for col in self.fitted_columns_:
                if col in X_transformed.columns:
                    X_transformed[col] = self.func(X_transformed[col])
        
        return X_transformed

# Comprehensive demonstration
def comprehensive_custom_transformers_demo():
    """Comprehensive demonstration of custom transformers."""
    
    print("=== COMPREHENSIVE CUSTOM TRANSFORMERS DEMONSTRATION ===\n")
    
    # Create sample data
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                              n_redundant=2, n_clusters_per_class=1, random_state=42)
    
    # Add some outliers
    outlier_indices = np.random.choice(len(X), size=50, replace=False)
    X[outlier_indices] += np.random.normal(0, 5, X[outlier_indices].shape)
    
    # Convert to DataFrame for demonstration
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)
    
    print("1. FEATURE SELECTOR TRANSFORMER")
    print("-" * 40)
    
    # Test feature selector
    selector = FeatureSelector(feature_names=['feature_0', 'feature_1', 'feature_2'])
    X_selected = selector.fit_transform(X_train)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Selected indices: {selector.selected_indices_}")
    print()
    
    print("2. OUTLIER REMOVER TRANSFORMER")
    print("-" * 40)
    
    # Test outlier remover
    outlier_remover = OutlierRemover(factor=2.0, method='iqr')
    X_no_outliers = outlier_remover.fit_transform(X_train.values)
    
    print(f"Original samples: {X_train.shape[0]}")
    print(f"After outlier removal: {X_no_outliers.shape[0]}")
    print(f"Outliers removed: {X_train.shape[0] - X_no_outliers.shape[0]}")
    print()
    
    print("3. LOG TRANSFORMER")
    print("-" * 40)
    
    # Create positive data for log transform
    X_positive = np.abs(X_train.values) + 1
    log_transformer = LogTransformer(columns=[0, 1])
    X_log = log_transformer.fit_transform(X_positive)
    X_recovered = log_transformer.inverse_transform(X_log)
    
    print(f"Original data shape: {X_positive.shape}")
    print(f"Log transformed shape: {X_log.shape}")
    print(f"Inverse transform error: {np.mean(np.abs(X_positive - X_recovered)):.6f}")
    print()
    
    print("4. POLYNOMIAL FEATURES TRANSFORMER")
    print("-" * 40)
    
    # Test polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_train.iloc[:, :3].values)  # Use first 3 features
    
    print(f"Original features: {3}")
    print(f"Polynomial features: {X_poly.shape[1]}")
    print()
    
    print("5. CUSTOM ROBUST SCALER")
    print("-" * 40)
    
    # Test custom robust scaler
    custom_scaler = RobustScaler()
    X_scaled = custom_scaler.fit_transform(X_train.values)
    X_recovered_scale = custom_scaler.inverse_transform(X_scaled)
    
    print(f"Scaling error: {np.mean(np.abs(X_train.values - X_recovered_scale)):.6f}")
    print(f"Scaled data mean: {np.mean(X_scaled, axis=0)[:3]}")
    print(f"Scaled data std: {np.std(X_scaled, axis=0)[:3]}")
    print()
    
    print("6. COMPLETE PIPELINE WITH CUSTOM TRANSFORMERS")
    print("-" * 40)
    
    # Create pipeline with custom transformers
    pipeline = Pipeline([
        ('feature_selector', FeatureSelector(feature_indices=[0, 1, 2, 3, 4])),
        ('outlier_remover', OutlierRemover(factor=2.0)),
        ('scaler', RobustScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Note: OutlierRemover changes number of samples, so we'll modify the pipeline
    # Let's create a version without outlier removal for proper pipeline usage
    pipeline_safe = Pipeline([
        ('feature_selector', FeatureSelector(feature_indices=[0, 1, 2, 3, 4])),
        ('scaler', RobustScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit and evaluate pipeline
    pipeline_safe.fit(X_train, y_train)
    y_pred = pipeline_safe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Pipeline accuracy: {accuracy:.4f}")
    
    # Test with GridSearchCV
    param_grid = {
        'feature_selector__feature_indices': [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6]],
        'classifier__n_estimators': [50, 100]
    }
    
    grid_search = GridSearchCV(pipeline_safe, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return pipeline_safe, grid_search

# Advanced custom transformer example
class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering transformer."""
    
    def __init__(self, create_interactions=True, create_ratios=True, 
                 create_aggregates=True, n_quantiles=5):
        self.create_interactions = create_interactions
        self.create_ratios = create_ratios
        self.create_aggregates = create_aggregates
        self.n_quantiles = n_quantiles
        self.feature_names_ = None
        self.quantile_boundaries_ = None
    
    def fit(self, X, y=None):
        """Learn feature engineering parameters."""
        X = np.array(X)
        
        if self.create_quantiles:
            self.quantile_boundaries_ = []
            for i in range(X.shape[1]):
                boundaries = np.percentile(X[:, i], 
                                         np.linspace(0, 100, self.n_quantiles + 1))
                self.quantile_boundaries_.append(boundaries)
        
        return self
    
    def transform(self, X):
        """Apply advanced feature engineering."""
        X = np.array(X)
        features = [X]
        
        # Create interaction features
        if self.create_interactions:
            n_features = X.shape[1]
            for i in range(min(5, n_features)):  # Limit to avoid explosion
                for j in range(i + 1, min(5, n_features)):
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                    features.append(interaction)
        
        # Create ratio features
        if self.create_ratios:
            n_features = X.shape[1]
            for i in range(min(3, n_features)):
                for j in range(i + 1, min(3, n_features)):
                    # Avoid division by zero
                    denominator = X[:, j].copy()
                    denominator[denominator == 0] = 1e-8
                    ratio = (X[:, i] / denominator).reshape(-1, 1)
                    features.append(ratio)
        
        # Create aggregate features
        if self.create_aggregates:
            # Sum, mean, std across features
            feature_sum = np.sum(X, axis=1).reshape(-1, 1)
            feature_mean = np.mean(X, axis=1).reshape(-1, 1)
            feature_std = np.std(X, axis=1).reshape(-1, 1)
            
            features.extend([feature_sum, feature_mean, feature_std])
        
        return np.hstack(features)

# Run comprehensive demonstration
if __name__ == "__main__":
    pipeline, grid_search = comprehensive_custom_transformers_demo()
    
    print("\n=== ADVANCED TRANSFORMER EXAMPLE ===")
    
    # Test advanced transformer
    X_sample = np.random.randn(100, 5)
    advanced_engineer = AdvancedFeatureEngineer()
    X_engineered = advanced_engineer.fit_transform(X_sample)
    
    print(f"Original features: {X_sample.shape[1]}")
    print(f"Engineered features: {X_engineered.shape[1]}")
```

### Explanation

1. **Multiple Custom Transformers**: Various transformers for different preprocessing tasks
2. **BaseEstimator & TransformerMixin**: Proper inheritance for scikit-learn compatibility
3. **Pipeline Integration**: All transformers work seamlessly in pipelines
4. **GridSearchCV Compatible**: Can be used in hyperparameter optimization
5. **Advanced Feature Engineering**: Complex transformer with multiple operations

### Use Cases

1. **Domain-Specific Preprocessing**: Create transformers for specific data types
2. **Feature Engineering**: Automated feature creation and selection
3. **Data Cleaning**: Custom outlier detection and handling
4. **Pipeline Standardization**: Reusable preprocessing components
5. **Experimental Preprocessing**: Test new preprocessing ideas

### Best Practices

1. **Follow Conventions**: Always inherit from BaseEstimator and TransformerMixin
2. **Implement fit/transform**: Separate learning and application phases
3. **Handle Edge Cases**: Deal with missing data, zeros, and invalid inputs
4. **Parameter Validation**: Validate constructor parameters
5. **Documentation**: Provide clear docstrings and examples

### Common Pitfalls

1. **State Leakage**: Don't use test data information in fit()
2. **Shape Changes**: Be careful with transformers that change sample count
3. **Parameter Naming**: Use trailing underscore for fitted parameters
4. **Memory Efficiency**: Consider memory usage for large datasets
5. **Inverse Transform**: Implement when reversibility is needed

### Debugging Tips

```python
# Test transformer compatibility
def test_transformer(transformer, X_sample):
    """Test if transformer follows scikit-learn conventions."""
    try:
        # Test fit and transform
        transformer.fit(X_sample)
        X_transformed = transformer.transform(X_sample)
        
        # Test fit_transform
        X_fit_transform = transformer.fit_transform(X_sample)
        
        # Check consistency
        if np.allclose(X_transformed, X_fit_transform):
            print("✓ Transformer passes basic tests")
        else:
            print("✗ Inconsistency between transform and fit_transform")
            
        return True
    except Exception as e:
        print(f"✗ Transformer failed: {str(e)}")
        return False
```

### Optimization Techniques

1. **Vectorization**: Use NumPy operations instead of loops
2. **Caching**: Cache expensive computations in fit()
3. **Sparse Matrices**: Support sparse data when appropriate
4. **Parallel Processing**: Use joblib for independent operations
5. **Memory Mapping**: Use memory-mapped arrays for large datasets

---

