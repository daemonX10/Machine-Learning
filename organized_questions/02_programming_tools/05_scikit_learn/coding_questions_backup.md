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

