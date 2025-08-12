# Cost Function Interview Questions - Coding Questions

## Question 1

**Implement a Python function that calculates the Mean Squared Error between predicted and actual values.**

**Answer:**

Mean Squared Error (MSE) is one of the most fundamental cost functions in machine learning, measuring the average squared difference between predicted and actual values. Here's a comprehensive implementation with multiple approaches and optimizations:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time

def mse_basic(y_true, y_pred):
    """
    Basic implementation of Mean Squared Error
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        
    Returns:
        float: Mean Squared Error value
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length")
    
    squared_errors = [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]
    return sum(squared_errors) / len(squared_errors)

def mse_vectorized(y_true, y_pred):
    """
    Vectorized implementation using NumPy for better performance
    
    Args:
        y_true: NumPy array of actual values
        y_pred: NumPy array of predicted values
        
    Returns:
        float: Mean Squared Error value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape")
    
    return np.mean((y_true - y_pred) ** 2)

def mse_with_regularization(y_true, y_pred, weights=None, l1_reg=0, l2_reg=0):
    """
    MSE with optional regularization terms and sample weights
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        weights: Optional sample weights
        l1_reg: L1 regularization parameter
        l2_reg: L2 regularization parameter
        
    Returns:
        float: Regularized MSE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if weights is None:
        weights = np.ones(len(y_true))
    else:
        weights = np.array(weights)
    
    # Weighted MSE
    weighted_mse = np.average((y_true - y_pred) ** 2, weights=weights)
    
    # Add regularization terms (assuming weights parameter contains model weights)
    if hasattr(weights, 'model_weights'):
        model_weights = weights.model_weights
        l1_penalty = l1_reg * np.sum(np.abs(model_weights))
        l2_penalty = l2_reg * np.sum(model_weights ** 2)
        return weighted_mse + l1_penalty + l2_penalty
    
    return weighted_mse

def mse_gradient(y_true, y_pred):
    """
    Calculate the gradient of MSE with respect to predictions
    Useful for backpropagation in neural networks
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        
    Returns:
        numpy.ndarray: Gradient of MSE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return 2 * (y_pred - y_true) / len(y_true)

class MSECalculator:
    """
    Professional MSE calculator class with validation and utilities
    """
    
    def __init__(self, epsilon=1e-8):
        """
        Initialize MSE calculator
        
        Args:
            epsilon: Small value to prevent numerical issues
        """
        self.epsilon = epsilon
        self.history = []
    
    def calculate(self, y_true, y_pred, return_components=False):
        """
        Calculate MSE with comprehensive validation
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            return_components: Whether to return individual components
            
        Returns:
            float or tuple: MSE value or (MSE, components)
        """
        # Input validation
        y_true = self._validate_input(y_true)
        y_pred = self._validate_input(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        # Calculate MSE
        squared_errors = (y_true - y_pred) ** 2
        mse_value = np.mean(squared_errors)
        
        # Store in history
        self.history.append({
            'mse': mse_value,
            'rmse': np.sqrt(mse_value),
            'mae': np.mean(np.abs(y_true - y_pred)),
            'max_error': np.max(np.abs(y_true - y_pred))
        })
        
        if return_components:
            return mse_value, {
                'squared_errors': squared_errors,
                'mean_error': np.mean(y_true - y_pred),
                'std_error': np.std(y_true - y_pred),
                'rmse': np.sqrt(mse_value)
            }
        
        return mse_value
    
    def _validate_input(self, data):
        """Validate and convert input data"""
        if data is None:
            raise ValueError("Input cannot be None")
        
        data = np.array(data, dtype=np.float64)
        
        if data.size == 0:
            raise ValueError("Input cannot be empty")
        
        if np.any(np.isnan(data)):
            raise ValueError("Input contains NaN values")
        
        if np.any(np.isinf(data)):
            raise ValueError("Input contains infinite values")
        
        return data
    
    def get_statistics(self):
        """Get statistics from calculation history"""
        if not self.history:
            return None
        
        return {
            'mean_mse': np.mean([h['mse'] for h in self.history]),
            'std_mse': np.std([h['mse'] for h in self.history]),
            'min_mse': np.min([h['mse'] for h in self.history]),
            'max_mse': np.max([h['mse'] for h in self.history])
        }

def demonstrate_mse_implementations():
    """Demonstrate different MSE implementations with examples"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # True function: y = 2x + 1 + noise
    x = np.linspace(0, 10, n_samples)
    y_true = 2 * x + 1 + np.random.normal(0, 0.5, n_samples)
    
    # Predictions with some error
    y_pred_good = 2 * x + 1.1 + np.random.normal(0, 0.3, n_samples)
    y_pred_poor = 1.5 * x + 2 + np.random.normal(0, 1, n_samples)
    
    print("MSE Implementation Demonstration")
    print("=" * 40)
    
    # Test different implementations
    implementations = [
        ("Basic Python", mse_basic),
        ("Vectorized NumPy", mse_vectorized)
    ]
    
    for name, func in implementations:
        start_time = time.time()
        mse_good = func(y_true, y_pred_good)
        mse_poor = func(y_true, y_pred_poor)
        end_time = time.time()
        
        print(f"\n{name}:")
        print(f"  Good predictions MSE: {mse_good:.4f}")
        print(f"  Poor predictions MSE: {mse_poor:.4f}")
        print(f"  Execution time: {(end_time - start_time) * 1000:.2f} ms")
    
    # Professional calculator
    print(f"\nProfessional MSE Calculator:")
    calculator = MSECalculator()
    
    mse_val, components = calculator.calculate(y_true, y_pred_good, return_components=True)
    print(f"  MSE: {mse_val:.4f}")
    print(f"  RMSE: {components['rmse']:.4f}")
    print(f"  Mean Error: {components['mean_error']:.4f}")
    print(f"  Max Error: {np.max(np.abs(components['squared_errors'])):.4f}")
    
    # Compare with sklearn
    sklearn_mse = mean_squared_error(y_true, y_pred_good)
    print(f"  sklearn MSE: {sklearn_mse:.4f}")
    print(f"  Difference: {abs(mse_val - sklearn_mse):.10f}")
    
    # Gradient calculation
    gradient = mse_gradient(y_true[:10], y_pred_good[:10])
    print(f"\nGradient (first 10 samples): {gradient}")
    
    return y_true, y_pred_good, y_pred_poor

def create_mse_visualization():
    """Create visualizations to understand MSE behavior"""
    
    y_true, y_pred_good, y_pred_poor = demonstrate_mse_implementations()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Predictions vs True values
    ax1 = axes[0, 0]
    x_vals = np.arange(len(y_true))
    
    ax1.scatter(x_vals[::10], y_true[::10], alpha=0.6, label='True values', s=20)
    ax1.scatter(x_vals[::10], y_pred_good[::10], alpha=0.6, label='Good predictions', s=20)
    ax1.scatter(x_vals[::10], y_pred_poor[::10], alpha=0.6, label='Poor predictions', s=20)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Value')
    ax1.set_title('Predictions vs True Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    ax2 = axes[0, 1]
    errors_good = y_true - y_pred_good
    errors_poor = y_true - y_pred_poor
    
    ax2.hist(errors_good, bins=50, alpha=0.7, label='Good predictions', density=True)
    ax2.hist(errors_poor, bins=50, alpha=0.7, label='Poor predictions', density=True)
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Density')
    ax2.set_title('Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MSE vs different noise levels
    ax3 = axes[1, 0]
    noise_levels = np.linspace(0, 2, 20)
    mse_values = []
    
    for noise in noise_levels:
        y_pred_noise = y_pred_good + np.random.normal(0, noise, len(y_pred_good))
        mse_val = mse_vectorized(y_true, y_pred_noise)
        mse_values.append(mse_val)
    
    ax3.plot(noise_levels, mse_values, 'b-o', markersize=4)
    ax3.set_xlabel('Noise Level')
    ax3.set_ylabel('MSE')
    ax3.set_title('MSE vs Noise Level')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: MSE components
    ax4 = axes[1, 1]
    n_samples_range = np.logspace(1, 4, 20, dtype=int)
    mse_components = []
    
    for n in n_samples_range:
        subset_true = y_true[:n]
        subset_pred = y_pred_good[:n]
        
        calculator = MSECalculator()
        mse_val, components = calculator.calculate(subset_true, subset_pred, return_components=True)
        mse_components.append({
            'n_samples': n,
            'mse': mse_val,
            'rmse': components['rmse'],
            'mae': np.mean(np.abs(subset_true - subset_pred))
        })
    
    ax4.semilogx([c['n_samples'] for c in mse_components], 
                 [c['mse'] for c in mse_components], 'b-o', label='MSE', markersize=4)
    ax4.semilogx([c['n_samples'] for c in mse_components], 
                 [c['rmse'] for c in mse_components], 'r-s', label='RMSE', markersize=4)
    ax4.semilogx([c['n_samples'] for c in mse_components], 
                 [c['mae'] for c in mse_components], 'g-^', label='MAE', markersize=4)
    
    ax4.set_xlabel('Number of Samples')
    ax4.set_ylabel('Error Value')
    ax4.set_title('Error Metrics vs Sample Size')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Run demonstration
    demonstrate_mse_implementations()
    
    # Create visualizations
    create_mse_visualization()
    
    print("\nKey Insights:")
    print("• MSE penalizes larger errors more heavily due to squaring")
    print("• Always validate input data for NaN and infinite values")
    print("• Vectorized implementations are much faster for large datasets")
    print("• MSE is sensitive to outliers - consider robust alternatives if needed")
    print("• RMSE provides error in the same units as the target variable")
```

**Key Points for Interviews:**

1. **Mathematical Foundation**: MSE = (1/n) * Σ(yi - ŷi)²
2. **Sensitivity**: MSE heavily penalizes large errors due to squaring
3. **Units**: MSE is in squared units; RMSE is in original units
4. **Performance**: Vectorized NumPy implementation is ~100x faster than Python loops
5. **Robustness**: Consider MAE or Huber loss for datasets with outliers
6. **Gradient**: ∂MSE/∂ŷ = 2(ŷ - y)/n, essential for gradient descent

---

## Question 2

**Write a Python code snippet to compute the Cross-Entropy loss given predicted probabilities and actual labels.**

**Answer:**

Cross-Entropy loss is the standard loss function for classification problems, measuring the difference between predicted probability distributions and true labels. Here's a comprehensive implementation:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Binary Cross-Entropy loss for binary classification
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities [0, 1]
        epsilon: Small value to prevent log(0)
        
    Returns:
        float: Binary cross-entropy loss
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Binary cross-entropy formula: -[y*log(p) + (1-y)*log(1-p)]
    bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return np.mean(bce)

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15, from_logits=False):
    """
    Categorical Cross-Entropy loss for multi-class classification
    
    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted probabilities or logits
        epsilon: Small value to prevent log(0)
        from_logits: Whether y_pred contains logits (before softmax)
        
    Returns:
        float: Categorical cross-entropy loss
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Convert logits to probabilities if needed
    if from_logits:
        y_pred = softmax(y_pred, axis=-1)
    
    # Handle different label formats
    if y_true.ndim == 1:  # Class indices
        num_classes = y_pred.shape[-1]
        y_true_onehot = np.eye(num_classes)[y_true]
    else:  # One-hot encoded
        y_true_onehot = y_true
    
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Categorical cross-entropy: -Σ(y_true * log(y_pred))
    cce = -np.sum(y_true_onehot * np.log(y_pred), axis=-1)
    
    return np.mean(cce)

def sparse_categorical_cross_entropy(y_true, y_pred, epsilon=1e-15, from_logits=False):
    """
    Sparse Categorical Cross-Entropy (memory efficient for many classes)
    
    Args:
        y_true: True class indices (not one-hot encoded)
        y_pred: Predicted probabilities or logits
        epsilon: Small value to prevent log(0)
        from_logits: Whether y_pred contains logits
        
    Returns:
        float: Sparse categorical cross-entropy loss
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred)
    
    # Convert logits to probabilities if needed
    if from_logits:
        y_pred = softmax(y_pred, axis=-1)
    
    # Clip predictions
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Extract probabilities for true classes
    batch_size = y_pred.shape[0]
    true_class_probs = y_pred[np.arange(batch_size), y_true]
    
    # Sparse categorical cross-entropy: -log(p_true_class)
    scce = -np.log(true_class_probs)
    
    return np.mean(scce)

def weighted_cross_entropy(y_true, y_pred, class_weights=None, epsilon=1e-15):
    """
    Weighted Cross-Entropy for imbalanced datasets
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        class_weights: Dictionary of class weights {class: weight}
        epsilon: Small value to prevent log(0)
        
    Returns:
        float: Weighted cross-entropy loss
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predictions
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    if class_weights is None:
        return categorical_cross_entropy(y_true, y_pred, epsilon)
    
    # Handle different label formats
    if y_true.ndim == 1:  # Class indices
        num_classes = y_pred.shape[-1]
        y_true_onehot = np.eye(num_classes)[y_true]
        class_indices = y_true
    else:  # One-hot encoded
        y_true_onehot = y_true
        class_indices = np.argmax(y_true, axis=-1)
    
    # Apply class weights
    weights = np.array([class_weights.get(i, 1.0) for i in class_indices])
    
    # Weighted categorical cross-entropy
    cce = -np.sum(y_true_onehot * np.log(y_pred), axis=-1)
    weighted_cce = cce * weights
    
    return np.mean(weighted_cce)

def focal_loss(y_true, y_pred, alpha=1.0, gamma=2.0, epsilon=1e-15):
    """
    Focal Loss for addressing class imbalance (from RetinaNet paper)
    Focuses learning on hard examples
    
    Args:
        y_true: True labels (one-hot or indices)
        y_pred: Predicted probabilities
        alpha: Weighting factor for rare class
        gamma: Focusing parameter (higher = more focus on hard examples)
        epsilon: Small value to prevent log(0)
        
    Returns:
        float: Focal loss value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predictions
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Handle different label formats
    if y_true.ndim == 1:  # Class indices
        num_classes = y_pred.shape[-1]
        y_true_onehot = np.eye(num_classes)[y_true]
    else:  # One-hot encoded
        y_true_onehot = y_true
    
    # Calculate cross-entropy
    ce = -np.sum(y_true_onehot * np.log(y_pred), axis=-1)
    
    # Calculate p_t (probability of true class)
    p_t = np.sum(y_true_onehot * y_pred, axis=-1)
    
    # Focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
    focal = alpha * np.power(1 - p_t, gamma) * ce
    
    return np.mean(focal)

class CrossEntropyCalculator:
    """
    Professional Cross-Entropy calculator with comprehensive features
    """
    
    def __init__(self, epsilon=1e-15):
        """
        Initialize calculator
        
        Args:
            epsilon: Small value to prevent numerical issues
        """
        self.epsilon = epsilon
        self.history = []
    
    def calculate(self, y_true, y_pred, loss_type='auto', **kwargs):
        """
        Calculate cross-entropy loss with automatic type detection
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            loss_type: 'binary', 'categorical', 'sparse', or 'auto'
            **kwargs: Additional parameters for specific loss functions
            
        Returns:
            dict: Loss value and additional metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Auto-detect loss type
        if loss_type == 'auto':
            if y_pred.shape[-1] == 1 or (y_pred.ndim == 1):
                loss_type = 'binary'
            elif y_true.ndim == 1:
                loss_type = 'sparse'
            else:
                loss_type = 'categorical'
        
        # Calculate appropriate loss
        if loss_type == 'binary':
            loss_value = binary_cross_entropy(y_true, y_pred, self.epsilon)
            accuracy = self._binary_accuracy(y_true, y_pred)
        elif loss_type == 'categorical':
            loss_value = categorical_cross_entropy(y_true, y_pred, self.epsilon, **kwargs)
            accuracy = self._categorical_accuracy(y_true, y_pred)
        elif loss_type == 'sparse':
            loss_value = sparse_categorical_cross_entropy(y_true, y_pred, self.epsilon, **kwargs)
            accuracy = self._sparse_accuracy(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Calculate additional metrics
        perplexity = np.exp(loss_value)
        entropy = self._calculate_entropy(y_pred)
        
        result = {
            'loss': loss_value,
            'accuracy': accuracy,
            'perplexity': perplexity,
            'entropy': entropy,
            'loss_type': loss_type
        }
        
        self.history.append(result)
        return result
    
    def _binary_accuracy(self, y_true, y_pred):
        """Calculate binary accuracy"""
        predictions = (y_pred > 0.5).astype(int)
        return np.mean(predictions == y_true)
    
    def _categorical_accuracy(self, y_true, y_pred):
        """Calculate categorical accuracy"""
        if y_true.ndim == 1:  # Class indices
            predictions = np.argmax(y_pred, axis=-1)
            return np.mean(predictions == y_true)
        else:  # One-hot encoded
            predictions = np.argmax(y_pred, axis=-1)
            true_classes = np.argmax(y_true, axis=-1)
            return np.mean(predictions == true_classes)
    
    def _sparse_accuracy(self, y_true, y_pred):
        """Calculate sparse categorical accuracy"""
        predictions = np.argmax(y_pred, axis=-1)
        return np.mean(predictions == y_true)
    
    def _calculate_entropy(self, y_pred):
        """Calculate prediction entropy (uncertainty measure)"""
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        if y_pred.ndim == 1:  # Binary
            entropy = -(y_pred_clipped * np.log(y_pred_clipped) + 
                       (1 - y_pred_clipped) * np.log(1 - y_pred_clipped))
        else:  # Multi-class
            entropy = -np.sum(y_pred_clipped * np.log(y_pred_clipped), axis=-1)
        
        return np.mean(entropy)
    
    def get_statistics(self):
        """Get statistics from calculation history"""
        if not self.history:
            return None
        
        return {
            'mean_loss': np.mean([h['loss'] for h in self.history]),
            'std_loss': np.std([h['loss'] for h in self.history]),
            'mean_accuracy': np.mean([h['accuracy'] for h in self.history]),
            'mean_perplexity': np.mean([h['perplexity'] for h in self.history])
        }

def cross_entropy_gradient(y_true, y_pred, loss_type='categorical'):
    """
    Calculate gradient of cross-entropy loss for backpropagation
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        loss_type: Type of cross-entropy loss
        
    Returns:
        numpy.ndarray: Gradient with respect to predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if loss_type == 'binary':
        # Binary cross-entropy gradient: (y_pred - y_true) / (y_pred * (1 - y_pred))
        gradient = (y_pred - y_true) / (y_pred * (1 - y_pred))
    elif loss_type == 'categorical':
        # Categorical cross-entropy gradient: (y_pred - y_true) / batch_size
        if y_true.ndim == 1:  # Convert indices to one-hot
            num_classes = y_pred.shape[-1]
            y_true_onehot = np.eye(num_classes)[y_true]
        else:
            y_true_onehot = y_true
        
        gradient = (y_pred - y_true_onehot) / len(y_pred)
    
    return gradient

def demonstrate_cross_entropy():
    """Demonstrate different cross-entropy implementations"""
    
    print("Cross-Entropy Loss Demonstration")
    print("=" * 40)
    
    # Binary classification example
    print("\n1. Binary Classification:")
    y_true_binary = np.array([0, 1, 1, 0, 1])
    y_pred_binary = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    
    bce_loss = binary_cross_entropy(y_true_binary, y_pred_binary)
    print(f"   True labels: {y_true_binary}")
    print(f"   Predictions: {y_pred_binary}")
    print(f"   Binary Cross-Entropy: {bce_loss:.4f}")
    
    # Multi-class classification example
    print("\n2. Multi-class Classification:")
    y_true_multi = np.array([0, 1, 2, 1, 0])
    y_pred_multi = np.array([
        [0.8, 0.1, 0.1],  # Confident correct
        [0.2, 0.7, 0.1],  # Less confident correct
        [0.1, 0.2, 0.7],  # Confident correct
        [0.3, 0.4, 0.3],  # Very uncertain
        [0.6, 0.3, 0.1]   # Moderately confident correct
    ])
    
    cce_loss = categorical_cross_entropy(y_true_multi, y_pred_multi)
    scce_loss = sparse_categorical_cross_entropy(y_true_multi, y_pred_multi)
    
    print(f"   True classes: {y_true_multi}")
    print(f"   Prediction probs:\n{y_pred_multi}")
    print(f"   Categorical Cross-Entropy: {cce_loss:.4f}")
    print(f"   Sparse Categorical Cross-Entropy: {scce_loss:.4f}")
    
    # Weighted cross-entropy for imbalanced data
    print("\n3. Weighted Cross-Entropy (Imbalanced Classes):")
    class_weights = {0: 1.0, 1: 2.0, 2: 3.0}  # Higher weight for rare classes
    wce_loss = weighted_cross_entropy(y_true_multi, y_pred_multi, class_weights)
    print(f"   Class weights: {class_weights}")
    print(f"   Weighted Cross-Entropy: {wce_loss:.4f}")
    
    # Focal loss for hard examples
    print("\n4. Focal Loss (Focus on Hard Examples):")
    focal_loss_val = focal_loss(y_true_multi, y_pred_multi, alpha=1.0, gamma=2.0)
    print(f"   Focal Loss (α=1.0, γ=2.0): {focal_loss_val:.4f}")
    
    # Professional calculator
    print("\n5. Professional Calculator:")
    calculator = CrossEntropyCalculator()
    
    # Test with different scenarios
    scenarios = [
        ("Perfect predictions", y_true_binary, np.array([0.001, 0.999, 0.999, 0.001, 0.999])),
        ("Random predictions", y_true_binary, np.array([0.5, 0.5, 0.5, 0.5, 0.5])),
        ("Poor predictions", y_true_binary, np.array([0.9, 0.1, 0.2, 0.8, 0.3]))
    ]
    
    for name, y_true, y_pred in scenarios:
        result = calculator.calculate(y_true, y_pred)
        print(f"   {name}:")
        print(f"     Loss: {result['loss']:.4f}")
        print(f"     Accuracy: {result['accuracy']:.4f}")
        print(f"     Perplexity: {result['perplexity']:.4f}")
    
    return calculator

def create_cross_entropy_visualization():
    """Create visualizations to understand cross-entropy behavior"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Binary cross-entropy vs prediction
    ax1 = axes[0, 0]
    p_values = np.linspace(0.001, 0.999, 1000)
    
    # For true label = 1
    bce_true_1 = -np.log(p_values)
    # For true label = 0
    bce_true_0 = -np.log(1 - p_values)
    
    ax1.plot(p_values, bce_true_1, 'b-', label='True label = 1', linewidth=2)
    ax1.plot(p_values, bce_true_0, 'r-', label='True label = 0', linewidth=2)
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    ax1.set_title('Binary Cross-Entropy vs Prediction')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Effect of class imbalance
    ax2 = axes[0, 1]
    
    # Simulate different class distributions
    class_ratios = np.linspace(0.1, 0.9, 20)
    losses_unweighted = []
    losses_weighted = []
    
    for ratio in class_ratios:
        # Generate imbalanced dataset
        n_samples = 1000
        n_class_1 = int(n_samples * ratio)
        n_class_0 = n_samples - n_class_1
        
        y_true = np.concatenate([np.zeros(n_class_0), np.ones(n_class_1)])
        y_pred = np.random.beta(2, 5, n_samples)  # Biased predictions
        
        # Unweighted loss
        loss_unweighted = binary_cross_entropy(y_true, y_pred)
        losses_unweighted.append(loss_unweighted)
        
        # Weighted loss
        class_weights = {0: ratio, 1: 1-ratio}  # Inverse frequency weighting
        y_true_int = y_true.astype(int)
        loss_weighted = weighted_cross_entropy(y_true_int, 
                                             np.column_stack([1-y_pred, y_pred]), 
                                             class_weights)
        losses_weighted.append(loss_weighted)
    
    ax2.plot(class_ratios, losses_unweighted, 'b-o', label='Unweighted', markersize=4)
    ax2.plot(class_ratios, losses_weighted, 'r-s', label='Weighted', markersize=4)
    ax2.set_xlabel('Class 1 Ratio')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_title('Effect of Class Imbalance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Focal loss vs standard cross-entropy
    ax3 = axes[1, 0]
    
    # Generate predictions with varying confidence
    y_true_focal = np.array([1, 1, 1, 1, 1])  # All true class
    confidence_levels = np.linspace(0.1, 0.9, 20)
    
    ce_losses = []
    focal_losses_gamma_1 = []
    focal_losses_gamma_2 = []
    focal_losses_gamma_5 = []
    
    for conf in confidence_levels:
        y_pred_conf = np.array([[1-conf, conf]] * 5)
        
        ce_loss = sparse_categorical_cross_entropy(y_true_focal, y_pred_conf)
        focal_1 = focal_loss(y_true_focal, y_pred_conf, gamma=1.0)
        focal_2 = focal_loss(y_true_focal, y_pred_conf, gamma=2.0)
        focal_5 = focal_loss(y_true_focal, y_pred_conf, gamma=5.0)
        
        ce_losses.append(ce_loss)
        focal_losses_gamma_1.append(focal_1)
        focal_losses_gamma_2.append(focal_2)
        focal_losses_gamma_5.append(focal_5)
    
    ax3.plot(confidence_levels, ce_losses, 'k-', label='Cross-Entropy', linewidth=2)
    ax3.plot(confidence_levels, focal_losses_gamma_1, 'b--', label='Focal (γ=1)', linewidth=2)
    ax3.plot(confidence_levels, focal_losses_gamma_2, 'r--', label='Focal (γ=2)', linewidth=2)
    ax3.plot(confidence_levels, focal_losses_gamma_5, 'g--', label='Focal (γ=5)', linewidth=2)
    
    ax3.set_xlabel('Prediction Confidence')
    ax3.set_ylabel('Loss Value')
    ax3.set_title('Focal Loss vs Cross-Entropy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Multi-class confusion impact
    ax4 = axes[1, 1]
    
    # Show how different types of mistakes affect loss
    n_classes = 3
    y_true_conf = np.array([0])  # True class is 0
    
    # Different prediction scenarios
    scenarios = {
        'Confident Correct': [0.9, 0.05, 0.05],
        'Confident Wrong': [0.05, 0.9, 0.05],
        'Uncertain': [0.33, 0.33, 0.34],
        'Slightly Wrong': [0.6, 0.3, 0.1],
        'Very Wrong': [0.1, 0.1, 0.8]
    }
    
    scenario_names = list(scenarios.keys())
    losses = []
    
    for scenario_name, probs in scenarios.items():
        loss = sparse_categorical_cross_entropy([0], [probs])
        losses.append(loss)
    
    bars = ax4.bar(scenario_names, losses, color=['green', 'red', 'orange', 'yellow', 'purple'])
    ax4.set_ylabel('Cross-Entropy Loss')
    ax4.set_title('Loss for Different Prediction Scenarios')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax4.annotate(f'{loss:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Run demonstrations
    calculator = demonstrate_cross_entropy()
    
    # Create visualizations
    create_cross_entropy_visualization()
    
    print("\nKey Insights:")
    print("• Cross-entropy penalizes confident wrong predictions heavily")
    print("• Perfect predictions have loss ≈ 0, random predictions have loss ≈ ln(num_classes)")
    print("• Weighted cross-entropy helps with class imbalance")
    print("• Focal loss focuses learning on hard examples")
    print("• Gradient computation is crucial for neural network training")
```

**Key Points for Interviews:**

1. **Mathematical Foundation**: 
   - Binary: -[y·log(p) + (1-y)·log(1-p)]
   - Categorical: -Σ(yi·log(pi))

2. **Numerical Stability**: Always clip predictions to prevent log(0)

3. **Variants**:
   - Weighted: For class imbalance
   - Focal: For hard example mining
   - Sparse: Memory efficient for many classes

4. **Properties**:
   - Convex function (good for optimization)
   - Probabilistic interpretation
   - Penalizes confident wrong predictions exponentially

5. **Gradient**: For softmax + cross-entropy: (ŷ - y) (very clean derivative)

---

## Question 3

**Implement a gradient descent algorithm in Python to minimize a simple quadratic cost function.**

**Answer:**

Gradient descent is the fundamental optimization algorithm in machine learning. Here's a comprehensive implementation with multiple variants and detailed analysis:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.optimize import minimize
import time

class QuadraticCostFunction:
    """
    Quadratic cost function: f(x) = (1/2) * x^T * A * x + b^T * x + c
    This represents a general quadratic function that can model various optimization landscapes
    """
    
    def __init__(self, A=None, b=None, c=0, condition_number=1):
        """
        Initialize quadratic cost function
        
        Args:
            A: Hessian matrix (positive definite for convex function)
            b: Linear coefficient vector
            c: Constant term
            condition_number: Condition number for auto-generated matrix
        """
        if A is None:
            # Generate a well-conditioned positive definite matrix
            n = 2  # Default to 2D for visualization
            eigenvalues = np.linspace(1, condition_number, n)
            Q = np.random.orthogonal(n)  # Random orthogonal matrix
            self.A = Q @ np.diag(eigenvalues) @ Q.T
        else:
            self.A = np.array(A)
        
        self.dim = self.A.shape[0]
        
        if b is None:
            self.b = np.random.randn(self.dim) * 0.1
        else:
            self.b = np.array(b)
        
        self.c = c
        
        # Analytical solution: x* = -A^(-1) * b
        self.optimal_x = -np.linalg.solve(self.A, self.b)
        self.optimal_value = self.evaluate(self.optimal_x)
    
    def evaluate(self, x):
        """Evaluate the cost function at point x"""
        x = np.array(x)
        return 0.5 * x.T @ self.A @ x + self.b.T @ x + self.c
    
    def gradient(self, x):
        """Compute gradient at point x"""
        x = np.array(x)
        return self.A @ x + self.b
    
    def hessian(self, x=None):
        """Compute Hessian (constant for quadratic function)"""
        return self.A

class GradientDescentOptimizer:
    """
    Comprehensive gradient descent optimizer with multiple variants
    """
    
    def __init__(self, cost_function):
        """
        Initialize optimizer
        
        Args:
            cost_function: Function object with evaluate() and gradient() methods
        """
        self.cost_function = cost_function
        self.history = []
    
    def vanilla_gradient_descent(self, x0, learning_rate=0.01, max_iterations=1000, 
                                tolerance=1e-6, verbose=False):
        """
        Standard (vanilla) gradient descent
        
        Args:
            x0: Initial point
            learning_rate: Step size
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            verbose: Print progress
            
        Returns:
            dict: Optimization results
        """
        x = np.array(x0, dtype=float)
        history = {
            'x': [x.copy()],
            'cost': [self.cost_function.evaluate(x)],
            'gradient_norm': [np.linalg.norm(self.cost_function.gradient(x))],
            'learning_rate': [learning_rate]
        }
        
        if verbose:
            print(f"Vanilla Gradient Descent")
            print(f"Initial cost: {history['cost'][-1]:.6f}")
        
        for iteration in range(max_iterations):
            # Compute gradient
            grad = self.cost_function.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            # Check convergence
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Update parameters
            x = x - learning_rate * grad
            
            # Record history
            cost = self.cost_function.evaluate(x)
            history['x'].append(x.copy())
            history['cost'].append(cost)
            history['gradient_norm'].append(grad_norm)
            history['learning_rate'].append(learning_rate)
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: cost = {cost:.6f}, ||grad|| = {grad_norm:.6f}")
        
        return {
            'x_optimal': x,
            'cost_optimal': history['cost'][-1],
            'iterations': len(history['cost']) - 1,
            'converged': grad_norm < tolerance,
            'history': history
        }
    
    def momentum_gradient_descent(self, x0, learning_rate=0.01, momentum=0.9, 
                                 max_iterations=1000, tolerance=1e-6, verbose=False):
        """
        Gradient descent with momentum
        
        Args:
            x0: Initial point
            learning_rate: Step size
            momentum: Momentum parameter (0-1)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress
            
        Returns:
            dict: Optimization results
        """
        x = np.array(x0, dtype=float)
        velocity = np.zeros_like(x)
        
        history = {
            'x': [x.copy()],
            'cost': [self.cost_function.evaluate(x)],
            'gradient_norm': [np.linalg.norm(self.cost_function.gradient(x))],
            'velocity': [velocity.copy()]
        }
        
        if verbose:
            print(f"Momentum Gradient Descent (β={momentum})")
            print(f"Initial cost: {history['cost'][-1]:.6f}")
        
        for iteration in range(max_iterations):
            grad = self.cost_function.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Update velocity and position
            velocity = momentum * velocity - learning_rate * grad
            x = x + velocity
            
            # Record history
            cost = self.cost_function.evaluate(x)
            history['x'].append(x.copy())
            history['cost'].append(cost)
            history['gradient_norm'].append(grad_norm)
            history['velocity'].append(velocity.copy())
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: cost = {cost:.6f}, ||grad|| = {grad_norm:.6f}")
        
        return {
            'x_optimal': x,
            'cost_optimal': history['cost'][-1],
            'iterations': len(history['cost']) - 1,
            'converged': grad_norm < tolerance,
            'history': history
        }
    
    def adaptive_gradient_descent(self, x0, initial_lr=0.1, max_iterations=1000, 
                                 tolerance=1e-6, verbose=False):
        """
        Adaptive learning rate gradient descent with line search
        
        Args:
            x0: Initial point
            initial_lr: Initial learning rate
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress
            
        Returns:
            dict: Optimization results
        """
        x = np.array(x0, dtype=float)
        learning_rate = initial_lr
        
        history = {
            'x': [x.copy()],
            'cost': [self.cost_function.evaluate(x)],
            'gradient_norm': [np.linalg.norm(self.cost_function.gradient(x))],
            'learning_rate': [learning_rate]
        }
        
        if verbose:
            print(f"Adaptive Gradient Descent")
            print(f"Initial cost: {history['cost'][-1]:.6f}")
        
        for iteration in range(max_iterations):
            grad = self.cost_function.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Backtracking line search
            current_cost = self.cost_function.evaluate(x)
            direction = -grad
            
            # Find good step size
            alpha = 1.0
            c1 = 1e-4  # Armijo parameter
            rho = 0.5  # Backtracking parameter
            
            while True:
                new_x = x + alpha * direction
                new_cost = self.cost_function.evaluate(new_x)
                
                # Armijo condition
                if new_cost <= current_cost + c1 * alpha * grad.T @ direction:
                    break
                
                alpha *= rho
                if alpha < 1e-10:  # Prevent infinite loop
                    alpha = 1e-10
                    break
            
            # Update parameters
            x = x + alpha * direction
            learning_rate = alpha
            
            # Record history
            cost = self.cost_function.evaluate(x)
            history['x'].append(x.copy())
            history['cost'].append(cost)
            history['gradient_norm'].append(grad_norm)
            history['learning_rate'].append(learning_rate)
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: cost = {cost:.6f}, lr = {learning_rate:.6f}")
        
        return {
            'x_optimal': x,
            'cost_optimal': history['cost'][-1],
            'iterations': len(history['cost']) - 1,
            'converged': grad_norm < tolerance,
            'history': history
        }
    
    def nesterov_accelerated_gradient(self, x0, learning_rate=0.01, momentum=0.9,
                                    max_iterations=1000, tolerance=1e-6, verbose=False):
        """
        Nesterov Accelerated Gradient (NAG)
        
        Args:
            x0: Initial point
            learning_rate: Step size
            momentum: Momentum parameter
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress
            
        Returns:
            dict: Optimization results
        """
        x = np.array(x0, dtype=float)
        velocity = np.zeros_like(x)
        
        history = {
            'x': [x.copy()],
            'cost': [self.cost_function.evaluate(x)],
            'gradient_norm': [np.linalg.norm(self.cost_function.gradient(x))]
        }
        
        if verbose:
            print(f"Nesterov Accelerated Gradient (β={momentum})")
            print(f"Initial cost: {history['cost'][-1]:.6f}")
        
        for iteration in range(max_iterations):
            # Look-ahead point
            x_lookahead = x + momentum * velocity
            
            # Compute gradient at look-ahead point
            grad = self.cost_function.gradient(x_lookahead)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Update velocity and position
            velocity = momentum * velocity - learning_rate * grad
            x = x + velocity
            
            # Record history
            cost = self.cost_function.evaluate(x)
            history['x'].append(x.copy())
            history['cost'].append(cost)
            history['gradient_norm'].append(grad_norm)
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: cost = {cost:.6f}, ||grad|| = {grad_norm:.6f}")
        
        return {
            'x_optimal': x,
            'cost_optimal': history['cost'][-1],
            'iterations': len(history['cost']) - 1,
            'converged': grad_norm < tolerance,
            'history': history
        }

def demonstrate_gradient_descent():
    """Demonstrate different gradient descent variants"""
    
    print("GRADIENT DESCENT DEMONSTRATION")
    print("=" * 50)
    
    # Create different quadratic functions
    cost_functions = {
        'Well-conditioned': QuadraticCostFunction(
            A=np.array([[1, 0], [0, 1]]),  # Identity matrix
            b=np.array([2, -1])
        ),
        'Ill-conditioned': QuadraticCostFunction(
            A=np.array([[10, 0], [0, 0.1]]),  # High condition number
            b=np.array([1, 1])
        ),
        'Rotated ellipse': QuadraticCostFunction(
            A=np.array([[2, 1], [1, 2]]),  # Off-diagonal elements
            b=np.array([0.5, -0.5])
        )
    }
    
    # Test different optimizers
    optimizers = [
        ('Vanilla GD', 'vanilla_gradient_descent', {'learning_rate': 0.1}),
        ('Momentum GD', 'momentum_gradient_descent', {'learning_rate': 0.1, 'momentum': 0.9}),
        ('Adaptive GD', 'adaptive_gradient_descent', {'initial_lr': 0.1}),
        ('Nesterov AG', 'nesterov_accelerated_gradient', {'learning_rate': 0.1, 'momentum': 0.9})
    ]
    
    results = {}
    
    for cf_name, cost_func in cost_functions.items():
        print(f"\n{cf_name} Function:")
        print(f"  Optimal point: {cost_func.optimal_x}")
        print(f"  Optimal value: {cost_func.optimal_value:.6f}")
        print(f"  Condition number: {np.linalg.cond(cost_func.A):.2f}")
        
        results[cf_name] = {}
        optimizer = GradientDescentOptimizer(cost_func)
        
        # Start from the same point for fair comparison
        x0 = np.array([3.0, 2.0])
        
        for opt_name, method_name, kwargs in optimizers:
            start_time = time.time()
            method = getattr(optimizer, method_name)
            result = method(x0, **kwargs, verbose=False)
            end_time = time.time()
            
            # Calculate final error
            error = np.linalg.norm(result['x_optimal'] - cost_func.optimal_x)
            
            print(f"  {opt_name}:")
            print(f"    Iterations: {result['iterations']}")
            print(f"    Final error: {error:.6f}")
            print(f"    Converged: {result['converged']}")
            print(f"    Time: {(end_time - start_time) * 1000:.2f} ms")
            
            results[cf_name][opt_name] = result
    
    return results, cost_functions

def create_gradient_descent_visualization(results, cost_functions):
    """Create comprehensive visualizations for gradient descent"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1-3: Trajectory plots for each cost function
    for i, (cf_name, cost_func) in enumerate(cost_functions.items()):
        ax = plt.subplot(3, 3, i + 1)
        
        # Create contour plot
        x_range = np.linspace(-1, 4, 100)
        y_range = np.linspace(-2, 3, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        
        for i_x in range(X.shape[0]):
            for j_y in range(X.shape[1]):
                Z[i_x, j_y] = cost_func.evaluate([X[i_x, j_y], Y[i_x, j_y]])
        
        # Plot contours
        contours = ax.contour(X, Y, Z, levels=20, alpha=0.6)
        ax.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
        
        # Plot optimization trajectories
        colors = ['red', 'blue', 'green', 'orange']
        for j, (opt_name, result) in enumerate(results[cf_name].items()):
            trajectory = np.array(result['history']['x'])
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=colors[j], linewidth=2, marker='o', markersize=3, 
                   label=opt_name, alpha=0.8)
        
        # Mark optimal point
        ax.plot(cost_func.optimal_x[0], cost_func.optimal_x[1], 
               'k*', markersize=15, label='Optimal')
        
        ax.set_title(f'{cf_name} Function')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence comparison
    ax4 = plt.subplot(3, 3, 4)
    
    for cf_name, cf_results in results.items():
        for opt_name, result in cf_results.items():
            if cf_name == 'Well-conditioned':  # Focus on one function for clarity
                costs = np.array(result['history']['cost'])
                iterations = range(len(costs))
                ax4.semilogy(iterations, costs - cost_functions[cf_name].optimal_value + 1e-10, 
                           label=opt_name, linewidth=2)
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Cost - Optimal Cost')
    ax4.set_title('Convergence Rate Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Learning rate adaptation
    ax5 = plt.subplot(3, 3, 5)
    
    # Show adaptive learning rate behavior
    adaptive_result = results['Ill-conditioned']['Adaptive GD']
    if 'learning_rate' in adaptive_result['history']:
        lr_history = adaptive_result['history']['learning_rate']
        iterations = range(len(lr_history))
        ax5.plot(iterations, lr_history, 'b-', linewidth=2)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Learning Rate')
        ax5.set_title('Adaptive Learning Rate')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Gradient norm evolution
    ax6 = plt.subplot(3, 3, 6)
    
    for cf_name, cf_results in results.items():
        if cf_name == 'Well-conditioned':
            for opt_name, result in cf_results.items():
                grad_norms = result['history']['gradient_norm']
                iterations = range(len(grad_norms))
                ax6.semilogy(iterations, grad_norms, label=opt_name, linewidth=2)
    
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Gradient Norm')
    ax6.set_title('Gradient Norm Decay')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: 3D surface plot
    ax7 = plt.subplot(3, 3, 7, projection='3d')
    
    # Use well-conditioned function for 3D visualization
    cf = cost_functions['Well-conditioned']
    x_range = np.linspace(-1, 4, 50)
    y_range = np.linspace(-2, 3, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = cf.evaluate([X[i, j], Y[i, j]])
    
    surf = ax7.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
    
    # Plot trajectory on surface
    vanilla_result = results['Well-conditioned']['Vanilla GD']
    traj = np.array(vanilla_result['history']['x'])
    traj_z = [cf.evaluate(point) for point in traj]
    
    ax7.plot(traj[:, 0], traj[:, 1], traj_z, 'r-o', markersize=4, linewidth=2)
    ax7.set_xlabel('x₁')
    ax7.set_ylabel('x₂')
    ax7.set_zlabel('Cost')
    ax7.set_title('3D Optimization Trajectory')
    
    # Plot 8: Performance comparison
    ax8 = plt.subplot(3, 3, 8)
    
    # Compare final errors
    optimizer_names = ['Vanilla GD', 'Momentum GD', 'Adaptive GD', 'Nesterov AG']
    condition_names = list(cost_functions.keys())
    
    errors = np.zeros((len(optimizer_names), len(condition_names)))
    
    for i, opt_name in enumerate(optimizer_names):
        for j, cf_name in enumerate(condition_names):
            result = results[cf_name][opt_name]
            optimal_x = cost_functions[cf_name].optimal_x
            error = np.linalg.norm(result['x_optimal'] - optimal_x)
            errors[i, j] = error
    
    im = ax8.imshow(errors, cmap='viridis', aspect='auto')
    ax8.set_xticks(range(len(condition_names)))
    ax8.set_xticklabels(condition_names, rotation=45)
    ax8.set_yticks(range(len(optimizer_names)))
    ax8.set_yticklabels(optimizer_names)
    ax8.set_title('Final Error Comparison')
    
    # Add error values as text
    for i in range(len(optimizer_names)):
        for j in range(len(condition_names)):
            ax8.text(j, i, f'{errors[i, j]:.3f}', ha='center', va='center', 
                    color='white' if errors[i, j] > np.mean(errors) else 'black')
    
    plt.colorbar(im, ax=ax8)
    
    # Plot 9: Iteration comparison
    ax9 = plt.subplot(3, 3, 9)
    
    iterations_data = []
    labels = []
    
    for cf_name in condition_names:
        for opt_name in optimizer_names:
            iterations_data.append(results[cf_name][opt_name]['iterations'])
            labels.append(f'{opt_name}\n({cf_name})')
    
    bars = ax9.bar(range(len(iterations_data)), iterations_data)
    ax9.set_xlabel('Optimizer (Problem)')
    ax9.set_ylabel('Iterations to Convergence')
    ax9.set_title('Convergence Speed Comparison')
    ax9.set_xticks(range(len(labels)))
    ax9.set_xticklabels(labels, rotation=45, ha='right')
    
    # Color bars by optimizer type
    colors = ['red', 'blue', 'green', 'orange'] * len(condition_names)
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_alpha(0.7)
    
    plt.tight_layout()
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Demonstrate gradient descent
    results, cost_functions = demonstrate_gradient_descent()
    
    # Create visualizations
    create_gradient_descent_visualization(results, cost_functions)
    
    print(f"\n{'='*50}")
    print("GRADIENT DESCENT SUMMARY")
    print(f"{'='*50}")
    
    print("\nKey Algorithm Variants:")
    print("• Vanilla: Basic gradient descent with fixed learning rate")
    print("• Momentum: Accelerates convergence using velocity accumulation")
    print("• Adaptive: Automatically adjusts learning rate using line search")
    print("• Nesterov: Look-ahead gradient for better convergence properties")
    
    print("\nOptimization Challenges:")
    print("• Ill-conditioning: High condition number slows convergence")
    print("• Learning rate: Too high → divergence, too low → slow convergence")
    print("• Local minima: Not an issue for convex quadratic functions")
    print("• Saddle points: Can slow down optimization in higher dimensions")
    
    print("\nPractical Considerations:")
    print("• Choose learning rate based on largest eigenvalue of Hessian")
    print("• Use momentum for smooth functions with gentle curvature")
    print("• Adaptive methods work well when optimal learning rate is unknown")
    print("• Monitor gradient norm for convergence assessment")
```

**Key Points for Interviews:**

1. **Algorithm Core**: x(k+1) = x(k) - α∇f(x(k))

2. **Learning Rate Selection**: 
   - For quadratic functions: α < 2/λmax (largest eigenvalue)
   - Too high: divergence, too low: slow convergence

3. **Convergence Rate**: Linear for strongly convex functions

4. **Variants**:
   - Momentum: Helps escape shallow local minima
   - Adaptive: Better for varying curvature
   - Nesterov: Optimal convergence rate for smooth convex functions

5. **Implementation Tips**:
   - Always check gradient computation with numerical differentiation
   - Monitor both cost and gradient norm for convergence
   - Use line search for robust step size selection

---

## Question 4

**Create a Python simulation that compares the convergence speed of batch and stochastic gradient descent.**

**Answer:**

This simulation demonstrates the key differences between Batch Gradient Descent (BGD) and Stochastic Gradient Descent (SGD), including their convergence characteristics, computational efficiency, and practical trade-offs:

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import time
from collections import defaultdict

class LinearRegressionOptimizer:
    """
    Linear regression optimizer supporting multiple gradient descent variants
    """
    
    def __init__(self, add_bias=True, regularization=0.0):
        """
        Initialize optimizer
        
        Args:
            add_bias: Whether to add bias term
            regularization: L2 regularization parameter
        """
        self.add_bias = add_bias
        self.regularization = regularization
        self.weights = None
        self.history = defaultdict(list)
    
    def _add_bias_term(self, X):
        """Add bias term to feature matrix"""
        if self.add_bias:
            bias_column = np.ones((X.shape[0], 1))
            return np.concatenate([bias_column, X], axis=1)
        return X
    
    def _compute_cost(self, X, y, weights):
        """
        Compute mean squared error cost with optional regularization
        
        Args:
            X: Feature matrix
            y: Target values
            weights: Model weights
            
        Returns:
            float: Cost value
        """
        predictions = X @ weights
        mse = np.mean((predictions - y) ** 2)
        
        # Add L2 regularization (exclude bias term)
        if self.regularization > 0:
            reg_term = self.regularization * np.sum(weights[1:] ** 2) if self.add_bias else self.regularization * np.sum(weights ** 2)
            mse += reg_term
        
        return mse
    
    def _compute_gradient(self, X, y, weights):
        """
        Compute gradient of MSE loss
        
        Args:
            X: Feature matrix
            y: Target values
            weights: Model weights
            
        Returns:
            numpy.ndarray: Gradient vector
        """
        predictions = X @ weights
        error = predictions - y
        gradient = (2 / len(y)) * X.T @ error
        
        # Add regularization gradient (exclude bias term)
        if self.regularization > 0:
            reg_gradient = 2 * self.regularization * weights
            if self.add_bias:
                reg_gradient[0] = 0  # Don't regularize bias term
            gradient += reg_gradient
        
        return gradient
    
    def batch_gradient_descent(self, X, y, learning_rate=0.01, max_iterations=1000, 
                              tolerance=1e-6, verbose=False):
        """
        Batch Gradient Descent: Uses entire dataset for each update
        
        Args:
            X: Feature matrix (n_samples × n_features)
            y: Target values
            learning_rate: Step size
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress
            
        Returns:
            dict: Training results
        """
        # Prepare data
        X_processed = self._add_bias_term(X)
        n_features = X_processed.shape[1]
        
        # Initialize weights
        self.weights = np.random.normal(0, 0.01, n_features)
        
        # Training history
        costs = []
        gradient_norms = []
        weight_history = []
        computation_times = []
        
        if verbose:
            print("Batch Gradient Descent")
            print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
        
        total_start_time = time.time()
        
        for iteration in range(max_iterations):
            iter_start_time = time.time()
            
            # Compute cost and gradient using entire dataset
            cost = self._compute_cost(X_processed, y, self.weights)
            gradient = self._compute_gradient(X_processed, y, self.weights)
            gradient_norm = np.linalg.norm(gradient)
            
            # Check convergence
            if gradient_norm < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Update weights
            self.weights -= learning_rate * gradient
            
            iter_end_time = time.time()
            
            # Record history
            costs.append(cost)
            gradient_norms.append(gradient_norm)
            weight_history.append(self.weights.copy())
            computation_times.append(iter_end_time - iter_start_time)
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: cost = {cost:.6f}, ||grad|| = {gradient_norm:.6f}")
        
        total_end_time = time.time()
        
        return {
            'algorithm': 'Batch GD',
            'final_weights': self.weights,
            'final_cost': costs[-1],
            'iterations': len(costs),
            'converged': gradient_norm < tolerance,
            'total_time': total_end_time - total_start_time,
            'avg_iter_time': np.mean(computation_times),
            'history': {
                'costs': costs,
                'gradient_norms': gradient_norms,
                'weights': weight_history,
                'computation_times': computation_times
            }
        }
    
    def stochastic_gradient_descent(self, X, y, learning_rate=0.01, max_epochs=100, 
                                   batch_size=1, tolerance=1e-6, verbose=False):
        """
        Stochastic Gradient Descent: Uses random samples for each update
        
        Args:
            X: Feature matrix
            y: Target values
            learning_rate: Step size
            max_epochs: Maximum epochs (full dataset passes)
            batch_size: Size of mini-batches (1 for true SGD)
            tolerance: Convergence tolerance
            verbose: Print progress
            
        Returns:
            dict: Training results
        """
        # Prepare data
        X_processed = self._add_bias_term(X)
        n_samples, n_features = X_processed.shape
        
        # Initialize weights
        self.weights = np.random.normal(0, 0.01, n_features)
        
        # Training history
        costs = []
        gradient_norms = []
        weight_history = []
        computation_times = []
        updates_per_epoch = max(1, n_samples // batch_size)
        
        if verbose:
            print(f"Stochastic Gradient Descent (batch_size={batch_size})")
            print(f"Dataset size: {n_samples} samples, {X.shape[1]} features")
            print(f"Updates per epoch: {updates_per_epoch}")
        
        total_start_time = time.time()
        
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_processed[indices]
            y_shuffled = y[indices]
            
            epoch_costs = []
            epoch_grad_norms = []
            
            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                iter_start_time = time.time()
                
                # Get mini-batch
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # Compute gradient on mini-batch
                gradient = self._compute_gradient(X_batch, y_batch, self.weights)
                gradient_norm = np.linalg.norm(gradient)
                
                # Update weights
                self.weights -= learning_rate * gradient
                
                iter_end_time = time.time()
                
                # Compute cost on full dataset for monitoring
                if i % (batch_size * 10) == 0:  # Not every iteration for efficiency
                    cost = self._compute_cost(X_processed, y, self.weights)
                    epoch_costs.append(cost)
                    epoch_grad_norms.append(gradient_norm)
                    weight_history.append(self.weights.copy())
                    computation_times.append(iter_end_time - iter_start_time)
            
            # Record epoch statistics
            if epoch_costs:
                costs.extend(epoch_costs)
                gradient_norms.extend(epoch_grad_norms)
            
            # Check convergence (based on last gradient norm)
            if epoch_grad_norms and epoch_grad_norms[-1] < tolerance:
                if verbose:
                    print(f"Converged at epoch {epoch}")
                break
            
            epoch_end_time = time.time()
            
            if verbose and (epoch + 1) % 10 == 0:
                final_cost = self._compute_cost(X_processed, y, self.weights)
                print(f"Epoch {epoch + 1}: cost = {final_cost:.6f}, time = {epoch_end_time - epoch_start_time:.3f}s")
        
        total_end_time = time.time()
        
        final_cost = self._compute_cost(X_processed, y, self.weights)
        final_gradient = self._compute_gradient(X_processed, y, self.weights)
        
        return {
            'algorithm': f'SGD (batch_size={batch_size})',
            'final_weights': self.weights,
            'final_cost': final_cost,
            'iterations': len(costs),
            'converged': np.linalg.norm(final_gradient) < tolerance,
            'total_time': total_end_time - total_start_time,
            'avg_iter_time': np.mean(computation_times) if computation_times else 0,
            'history': {
                'costs': costs,
                'gradient_norms': gradient_norms,
                'weights': weight_history,
                'computation_times': computation_times
            }
        }
    
    def mini_batch_gradient_descent(self, X, y, learning_rate=0.01, batch_size=32, 
                                   max_epochs=100, tolerance=1e-6, verbose=False):
        """
        Mini-batch Gradient Descent: Compromise between BGD and SGD
        """
        return self.stochastic_gradient_descent(X, y, learning_rate, max_epochs, 
                                               batch_size, tolerance, verbose)

def create_synthetic_dataset(n_samples=1000, n_features=20, noise=0.1, random_state=42):
    """
    Create synthetic regression dataset for comparison
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level
        random_state: Random seed
        
    Returns:
        tuple: (X, y, true_weights)
    """
    np.random.seed(random_state)
    
    # Generate dataset with known optimal solution
    X, y, true_weights = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        coef=True,
        random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, true_weights

def compare_gradient_descent_methods():
    """
    Comprehensive comparison of gradient descent methods
    """
    print("GRADIENT DESCENT COMPARISON SIMULATION")
    print("=" * 50)
    
    # Create datasets of different sizes
    datasets = {
        'Small': (100, 5),
        'Medium': (1000, 20),
        'Large': (10000, 50)
    }
    
    results = {}
    
    for dataset_name, (n_samples, n_features) in datasets.items():
        print(f"\n{dataset_name} Dataset ({n_samples} samples, {n_features} features):")
        print("-" * 40)
        
        # Generate dataset
        X, y, true_weights = create_synthetic_dataset(n_samples, n_features)
        
        # Test different batch sizes
        batch_sizes = [1, 16, 64, n_samples]  # SGD, mini-batch, mini-batch, BGD
        algorithms = []
        
        for batch_size in batch_sizes:
            optimizer = LinearRegressionOptimizer(regularization=0.01)
            
            if batch_size == n_samples:
                # Batch Gradient Descent
                result = optimizer.batch_gradient_descent(
                    X, y, learning_rate=0.01, max_iterations=500, verbose=False
                )
            else:
                # Stochastic or Mini-batch Gradient Descent
                result = optimizer.stochastic_gradient_descent(
                    X, y, learning_rate=0.01, batch_size=batch_size, 
                    max_epochs=100, verbose=False
                )
            
            algorithms.append(result)
            
            print(f"  {result['algorithm']}:")
            print(f"    Final cost: {result['final_cost']:.6f}")
            print(f"    Iterations: {result['iterations']}")
            print(f"    Total time: {result['total_time']:.3f}s")
            print(f"    Avg time/iter: {result['avg_iter_time']:.6f}s")
            print(f"    Converged: {result['converged']}")
        
        results[dataset_name] = {
            'algorithms': algorithms,
            'true_weights': true_weights,
            'dataset': (X, y)
        }
    
    return results

def create_convergence_visualization(results):
    """
    Create comprehensive visualizations comparing convergence behavior
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    colors = ['red', 'blue', 'green', 'orange']
    dataset_names = list(results.keys())
    
    # Plot 1-3: Convergence curves for each dataset
    for i, dataset_name in enumerate(dataset_names):
        ax = axes[i, 0]
        algorithms = results[dataset_name]['algorithms']
        
        for j, result in enumerate(algorithms):
            costs = result['history']['costs']
            iterations = range(len(costs))
            ax.plot(iterations, costs, color=colors[j], linewidth=2, 
                   label=result['algorithm'], alpha=0.8)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title(f'{dataset_name} Dataset - Convergence')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4-6: Time comparison
    for i, dataset_name in enumerate(dataset_names):
        ax = axes[i, 1]
        algorithms = results[dataset_name]['algorithms']
        
        algorithm_names = [result['algorithm'] for result in algorithms]
        total_times = [result['total_time'] for result in algorithms]
        
        bars = ax.bar(algorithm_names, total_times, color=colors[:len(algorithms)], alpha=0.7)
        ax.set_ylabel('Total Time (seconds)')
        ax.set_title(f'{dataset_name} Dataset - Training Time')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, total_times):
            height = bar.get_height()
            ax.annotate(f'{time_val:.3f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Plot 7-9: Convergence rate analysis
    for i, dataset_name in enumerate(dataset_names):
        ax = axes[i, 2]
        algorithms = results[dataset_name]['algorithms']
        
        for j, result in enumerate(algorithms):
            costs = np.array(result['history']['costs'])
            if len(costs) > 10:  # Ensure we have enough points
                # Calculate convergence rate (slope in log space)
                log_costs = np.log(costs[5:])  # Skip initial iterations
                iterations = np.arange(5, len(costs))
                
                # Fit linear trend to get convergence rate
                if len(log_costs) > 1:
                    slope = np.polyfit(iterations, log_costs, 1)[0]
                    
                    # Plot smoothed convergence
                    window_size = max(1, len(costs) // 20)
                    smoothed_costs = np.convolve(costs, np.ones(window_size)/window_size, mode='valid')
                    smooth_iterations = range(len(smoothed_costs))
                    
                    ax.plot(smooth_iterations, smoothed_costs, color=colors[j], 
                           linewidth=2, label=f'{result["algorithm"]} (rate: {slope:.4f})')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost (smoothed)')
        ax.set_title(f'{dataset_name} - Convergence Rate')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_convergence_properties(results):
    """
    Analyze and compare convergence properties
    """
    print(f"\n{'='*60}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name} Dataset Analysis:")
        print("-" * 30)
        
        algorithms = dataset_results['algorithms']
        
        # Compare final convergence
        print("Final Performance:")
        best_cost = min(result['final_cost'] for result in algorithms)
        
        for result in algorithms:
            relative_error = (result['final_cost'] - best_cost) / best_cost * 100
            efficiency = result['iterations'] / result['total_time']  # iterations per second
            
            print(f"  {result['algorithm']}:")
            print(f"    Final cost: {result['final_cost']:.6f}")
            print(f"    Relative error: {relative_error:.2f}%")
            print(f"    Efficiency: {efficiency:.1f} iter/sec")
        
        # Analyze convergence behavior
        print("\nConvergence Behavior:")
        for result in algorithms:
            costs = np.array(result['history']['costs'])
            if len(costs) > 1:
                # Initial vs final cost reduction
                initial_cost = costs[0]
                final_cost = costs[-1]
                improvement = (initial_cost - final_cost) / initial_cost * 100
                
                # Stability (variance in later iterations)
                if len(costs) > 20:
                    stability = np.std(costs[-10:]) / np.mean(costs[-10:])
                else:
                    stability = 0
                
                print(f"  {result['algorithm']}:")
                print(f"    Cost improvement: {improvement:.1f}%")
                print(f"    Stability (CV): {stability:.4f}")

def create_performance_summary():
    """
    Create summary table of performance characteristics
    """
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    characteristics = {
        'Batch Gradient Descent': {
            'Memory Usage': 'High (full dataset)',
            'Convergence': 'Smooth, deterministic',
            'Speed per Iteration': 'Slow (large datasets)',
            'Final Accuracy': 'High',
            'Parallelization': 'Easy',
            'Best For': 'Small to medium datasets, when accuracy is critical'
        },
        'Stochastic Gradient Descent': {
            'Memory Usage': 'Low (single sample)',
            'Convergence': 'Noisy, but fast initial progress',
            'Speed per Iteration': 'Very fast',
            'Final Accuracy': 'Lower (due to noise)',
            'Parallelization': 'Difficult',
            'Best For': 'Large datasets, online learning'
        },
        'Mini-batch Gradient Descent': {
            'Memory Usage': 'Medium (batch size)',
            'Convergence': 'Balanced smoothness and speed',
            'Speed per Iteration': 'Fast',
            'Final Accuracy': 'Good',
            'Parallelization': 'Moderate',
            'Best For': 'Most practical applications'
        }
    }
    
    # Print formatted table
    for algorithm, props in characteristics.items():
        print(f"\n{algorithm}:")
        for prop, value in props.items():
            print(f"  {prop:<20}: {value}")

if __name__ == "__main__":
    # Run comprehensive comparison
    results = compare_gradient_descent_methods()
    
    # Create visualizations
    create_convergence_visualization(results)
    
    # Analyze convergence properties
    analyze_convergence_properties(results)
    
    # Create performance summary
    create_performance_summary()
    
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    print("\nBatch Gradient Descent:")
    print("• Uses entire dataset for each update")
    print("• Smooth, deterministic convergence")
    print("• High memory requirements")
    print("• Good for small-medium datasets")
    
    print("\nStochastic Gradient Descent:")
    print("• Uses single sample for each update")
    print("• Fast initial progress, noisy convergence")
    print("• Low memory requirements")
    print("• Excellent for large datasets")
    
    print("\nMini-batch Gradient Descent:")
    print("• Best of both worlds compromise")
    print("• Typical batch sizes: 16, 32, 64, 128")
    print("• Good parallelization potential")
    print("• Most commonly used in practice")
    
    print("\nPractical Recommendations:")
    print("• Start with mini-batch GD (batch_size=32)")
    print("• Use SGD for very large datasets (>1M samples)")
    print("• Use BGD for small datasets (<1K samples)")
    print("• Monitor both cost and wall-clock time")
    print("• Consider learning rate scheduling for better convergence")
```

**Key Points for Interviews:**

1. **Computational Complexity**:
   - BGD: O(nm) per iteration (n=samples, m=features)
   - SGD: O(m) per iteration
   - Mini-batch: O(bm) per iteration (b=batch size)

2. **Memory Requirements**:
   - BGD: Requires full dataset in memory
   - SGD: Only needs one sample at a time
   - Mini-batch: Scales with batch size

3. **Convergence Properties**:
   - BGD: Smooth, deterministic convergence
   - SGD: Noisy but fast initial progress
   - Mini-batch: Balanced approach

4. **Practical Trade-offs**:
   - BGD: Better for small datasets, when accuracy is critical
   - SGD: Essential for large datasets, online learning
   - Mini-batch: Most versatile, good parallelization

5. **Hyperparameter Tuning**:
   - Learning rate more critical for SGD
   - Batch size affects convergence speed and stability
   - Learning rate scheduling often beneficial

---

## Question 5

**Build a Python class that implements an adaptive learning rate algorithm, like Adam or AdaGrad, from scratch.**

**Answer:**

Adaptive learning rate algorithms automatically adjust the learning rate during training, leading to faster convergence and better performance. Here's a comprehensive implementation of Adam, AdaGrad, RMSprop, and related algorithms:

```python
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import seaborn as sns
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler
import time

class AdaptiveOptimizer(ABC):
    """
    Abstract base class for adaptive optimization algorithms
    """
    
    def __init__(self, learning_rate=0.001, epsilon=1e-8):
        """
        Initialize base optimizer
        
        Args:
            learning_rate: Base learning rate
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.iteration = 0
        self.history = {
            'costs': [],
            'learning_rates': [],
            'gradient_norms': [],
            'parameter_updates': []
        }
    
    @abstractmethod
    def update_parameters(self, parameters, gradients):
        """
        Update parameters using the specific optimization algorithm
        
        Args:
            parameters: Current parameter values
            gradients: Computed gradients
            
        Returns:
            numpy.ndarray: Updated parameters
        """
        pass
    
    @abstractmethod
    def reset_state(self):
        """Reset optimizer state for new optimization"""
        pass

class AdaGradOptimizer(AdaptiveOptimizer):
    """
    AdaGrad: Adapts learning rate based on historical gradients
    Good for sparse gradients, but learning rate may decay too quickly
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        """
        Initialize AdaGrad optimizer
        
        Args:
            learning_rate: Initial learning rate
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate, epsilon)
        self.G = None  # Sum of squared gradients
    
    def update_parameters(self, parameters, gradients):
        """
        AdaGrad parameter update
        
        Update rule: θ = θ - (η / √(G + ε)) ⊙ g
        where G accumulates squared gradients
        """
        self.iteration += 1
        
        # Initialize accumulator on first iteration
        if self.G is None:
            self.G = np.zeros_like(parameters)
        
        # Accumulate squared gradients
        self.G += gradients ** 2
        
        # Compute adaptive learning rate
        adaptive_lr = self.learning_rate / (np.sqrt(self.G) + self.epsilon)
        
        # Update parameters
        parameter_update = adaptive_lr * gradients
        updated_parameters = parameters - parameter_update
        
        # Record history
        self.history['learning_rates'].append(np.mean(adaptive_lr))
        self.history['gradient_norms'].append(np.linalg.norm(gradients))
        self.history['parameter_updates'].append(np.linalg.norm(parameter_update))
        
        return updated_parameters
    
    def reset_state(self):
        """Reset AdaGrad state"""
        self.G = None
        self.iteration = 0
        self.history = {
            'costs': [],
            'learning_rates': [],
            'gradient_norms': [],
            'parameter_updates': []
        }

class RMSpropOptimizer(AdaptiveOptimizer):
    """
    RMSprop: Uses exponential moving average of squared gradients
    Addresses AdaGrad's aggressive learning rate decay
    """
    
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        """
        Initialize RMSprop optimizer
        
        Args:
            learning_rate: Initial learning rate
            beta: Exponential decay rate for moving average
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate, epsilon)
        self.beta = beta
        self.v = None  # Exponential moving average of squared gradients
    
    def update_parameters(self, parameters, gradients):
        """
        RMSprop parameter update
        
        Update rule: 
        v = β*v + (1-β)*g²
        θ = θ - (η / √(v + ε)) ⊙ g
        """
        self.iteration += 1
        
        # Initialize moving average on first iteration
        if self.v is None:
            self.v = np.zeros_like(parameters)
        
        # Update exponential moving average of squared gradients
        self.v = self.beta * self.v + (1 - self.beta) * (gradients ** 2)
        
        # Compute adaptive learning rate
        adaptive_lr = self.learning_rate / (np.sqrt(self.v) + self.epsilon)
        
        # Update parameters
        parameter_update = adaptive_lr * gradients
        updated_parameters = parameters - parameter_update
        
        # Record history
        self.history['learning_rates'].append(np.mean(adaptive_lr))
        self.history['gradient_norms'].append(np.linalg.norm(gradients))
        self.history['parameter_updates'].append(np.linalg.norm(parameter_update))
        
        return updated_parameters
    
    def reset_state(self):
        """Reset RMSprop state"""
        self.v = None
        self.iteration = 0
        self.history = {
            'costs': [],
            'learning_rates': [],
            'gradient_norms': [],
            'parameter_updates': []
        }

class AdamOptimizer(AdaptiveOptimizer):
    """
    Adam: Combines momentum and adaptive learning rates
    Most popular optimizer in deep learning
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer
        
        Args:
            learning_rate: Initial learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate, epsilon)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None  # First moment (momentum)
        self.v = None  # Second moment (adaptive learning rate)
    
    def update_parameters(self, parameters, gradients):
        """
        Adam parameter update
        
        Update rule:
        m = β₁*m + (1-β₁)*g
        v = β₂*v + (1-β₂)*g²
        m̂ = m / (1-β₁ᵗ)  # Bias correction
        v̂ = v / (1-β₂ᵗ)  # Bias correction
        θ = θ - η * m̂ / (√v̂ + ε)
        """
        self.iteration += 1
        
        # Initialize moments on first iteration
        if self.m is None:
            self.m = np.zeros_like(parameters)
            self.v = np.zeros_like(parameters)
        
        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected first and second moment estimates
        m_hat = self.m / (1 - self.beta1 ** self.iteration)
        v_hat = self.v / (1 - self.beta2 ** self.iteration)
        
        # Compute adaptive learning rate
        adaptive_lr = self.learning_rate / (np.sqrt(v_hat) + self.epsilon)
        
        # Update parameters
        parameter_update = adaptive_lr * m_hat
        updated_parameters = parameters - parameter_update
        
        # Record history
        self.history['learning_rates'].append(np.mean(adaptive_lr))
        self.history['gradient_norms'].append(np.linalg.norm(gradients))
        self.history['parameter_updates'].append(np.linalg.norm(parameter_update))
        
        return updated_parameters
    
    def reset_state(self):
        """Reset Adam state"""
        self.m = None
        self.v = None
        self.iteration = 0
        self.history = {
            'costs': [],
            'learning_rates': [],
            'gradient_norms': [],
            'parameter_updates': []
        }

class AdamWOptimizer(AdaptiveOptimizer):
    """
    AdamW: Adam with decoupled weight decay
    Better regularization than L2 penalty in Adam
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, weight_decay=0.01):
        """
        Initialize AdamW optimizer
        
        Args:
            learning_rate: Initial learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay coefficient
        """
        super().__init__(learning_rate, epsilon)
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
    
    def update_parameters(self, parameters, gradients):
        """
        AdamW parameter update with decoupled weight decay
        """
        self.iteration += 1
        
        if self.m is None:
            self.m = np.zeros_like(parameters)
            self.v = np.zeros_like(parameters)
        
        # Update moments (same as Adam)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.iteration)
        v_hat = self.v / (1 - self.beta2 ** self.iteration)
        
        # Adam update
        adam_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Decoupled weight decay
        weight_decay_update = self.learning_rate * self.weight_decay * parameters
        
        # Combined update
        parameter_update = adam_update + weight_decay_update
        updated_parameters = parameters - parameter_update
        
        # Record history
        adaptive_lr = self.learning_rate / (np.sqrt(v_hat) + self.epsilon)
        self.history['learning_rates'].append(np.mean(adaptive_lr))
        self.history['gradient_norms'].append(np.linalg.norm(gradients))
        self.history['parameter_updates'].append(np.linalg.norm(parameter_update))
        
        return updated_parameters
    
    def reset_state(self):
        """Reset AdamW state"""
        self.m = None
        self.v = None
        self.iteration = 0
        self.history = {
            'costs': [],
            'learning_rates': [],
            'gradient_norms': [],
            'parameter_updates': []
        }

class AdaDeltaOptimizer(AdaptiveOptimizer):
    """
    AdaDelta: Extension of AdaGrad that seeks to reduce its aggressive learning rate decay
    Uses parameter updates instead of gradients to compute learning rate
    """
    
    def __init__(self, rho=0.95, epsilon=1e-6):
        """
        Initialize AdaDelta optimizer
        
        Args:
            rho: Exponential decay rate
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate=1.0, epsilon=epsilon)  # Learning rate not used
        self.rho = rho
        self.E_g2 = None  # Exponential moving average of squared gradients
        self.E_dx2 = None  # Exponential moving average of squared parameter updates
    
    def update_parameters(self, parameters, gradients):
        """
        AdaDelta parameter update
        
        Update rule:
        E[g²] = ρ*E[g²] + (1-ρ)*g²
        Δx = -(√(E[Δx²] + ε) / √(E[g²] + ε)) * g
        E[Δx²] = ρ*E[Δx²] + (1-ρ)*Δx²
        θ = θ + Δx
        """
        self.iteration += 1
        
        if self.E_g2 is None:
            self.E_g2 = np.zeros_like(parameters)
            self.E_dx2 = np.zeros_like(parameters)
        
        # Update exponential moving average of squared gradients
        self.E_g2 = self.rho * self.E_g2 + (1 - self.rho) * (gradients ** 2)
        
        # Compute parameter update
        RMS_g = np.sqrt(self.E_g2 + self.epsilon)
        RMS_dx = np.sqrt(self.E_dx2 + self.epsilon)
        
        parameter_update = -(RMS_dx / RMS_g) * gradients
        
        # Update exponential moving average of squared parameter updates
        self.E_dx2 = self.rho * self.E_dx2 + (1 - self.rho) * (parameter_update ** 2)
        
        updated_parameters = parameters + parameter_update
        
        # Record history
        effective_lr = RMS_dx / RMS_g
        self.history['learning_rates'].append(np.mean(effective_lr))
        self.history['gradient_norms'].append(np.linalg.norm(gradients))
        self.history['parameter_updates'].append(np.linalg.norm(parameter_update))
        
        return updated_parameters
    
    def reset_state(self):
        """Reset AdaDelta state"""
        self.E_g2 = None
        self.E_dx2 = None
        self.iteration = 0
        self.history = {
            'costs': [],
            'learning_rates': [],
            'gradient_norms': [],
            'parameter_updates': []
        }

class OptimizationTestProblem:
    """
    Test problem for comparing optimizers
    """
    
    def __init__(self, problem_type='quadratic', n_dim=10, condition_number=100):
        """
        Initialize test problem
        
        Args:
            problem_type: Type of optimization problem
            n_dim: Problem dimensionality
            condition_number: Condition number for ill-conditioned problems
        """
        self.problem_type = problem_type
        self.n_dim = n_dim
        self.condition_number = condition_number
        
        if problem_type == 'quadratic':
            # Create ill-conditioned quadratic function
            eigenvals = np.logspace(0, np.log10(condition_number), n_dim)
            Q = np.random.orthogonal(n_dim)
            self.A = Q @ np.diag(eigenvals) @ Q.T
            self.b = np.random.randn(n_dim) * 0.1
            self.optimal_x = -np.linalg.solve(self.A, self.b)
        
        elif problem_type == 'rosenbrock':
            # Rosenbrock function (non-convex)
            self.optimal_x = np.ones(n_dim)
    
    def evaluate(self, x):
        """Evaluate objective function"""
        if self.problem_type == 'quadratic':
            return 0.5 * x.T @ self.A @ x + self.b.T @ x
        
        elif self.problem_type == 'rosenbrock':
            # Rosenbrock function: f(x) = Σ[100(x_{i+1} - x_i²)² + (1 - x_i)²]
            total = 0
            for i in range(len(x) - 1):
                total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
            return total
    
    def gradient(self, x):
        """Compute gradient"""
        if self.problem_type == 'quadratic':
            return self.A @ x + self.b
        
        elif self.problem_type == 'rosenbrock':
            grad = np.zeros_like(x)
            for i in range(len(x) - 1):
                grad[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
                grad[i+1] += 200 * (x[i+1] - x[i]**2)
            return grad

def compare_adaptive_optimizers():
    """
    Comprehensive comparison of adaptive optimization algorithms
    """
    print("ADAPTIVE OPTIMIZER COMPARISON")
    print("=" * 50)
    
    # Create test problems
    problems = {
        'Quadratic (Well-conditioned)': OptimizationTestProblem('quadratic', n_dim=10, condition_number=10),
        'Quadratic (Ill-conditioned)': OptimizationTestProblem('quadratic', n_dim=10, condition_number=1000),
        'Rosenbrock (Non-convex)': OptimizationTestProblem('rosenbrock', n_dim=10)
    }
    
    # Initialize optimizers
    optimizers = {
        'AdaGrad': AdaGradOptimizer(learning_rate=0.1),
        'RMSprop': RMSpropOptimizer(learning_rate=0.01),
        'Adam': AdamOptimizer(learning_rate=0.01),
        'AdamW': AdamWOptimizer(learning_rate=0.01, weight_decay=0.01),
        'AdaDelta': AdaDeltaOptimizer()
    }
    
    results = {}
    
    for problem_name, problem in problems.items():
        print(f"\n{problem_name}:")
        print("-" * 30)
        
        results[problem_name] = {}
        
        for optimizer_name, optimizer in optimizers.items():
            # Reset optimizer state
            optimizer.reset_state()
            
            # Initialize parameters
            x = np.random.randn(problem.n_dim) * 2
            x0 = x.copy()
            
            # Training parameters
            max_iterations = 1000
            tolerance = 1e-8
            
            costs = []
            start_time = time.time()
            
            for iteration in range(max_iterations):
                # Evaluate function and gradient
                cost = problem.evaluate(x)
                grad = problem.gradient(x)
                
                costs.append(cost)
                
                # Check convergence
                if np.linalg.norm(grad) < tolerance:
                    break
                
                # Update parameters
                x = optimizer.update_parameters(x, grad)
            
            end_time = time.time()
            
            # Store results
            final_error = np.linalg.norm(x - problem.optimal_x)
            
            results[problem_name][optimizer_name] = {
                'final_cost': costs[-1],
                'final_error': final_error,
                'iterations': len(costs),
                'time': end_time - start_time,
                'costs': costs,
                'optimizer_history': optimizer.history,
                'converged': np.linalg.norm(grad) < tolerance
            }
            
            print(f"  {optimizer_name}:")
            print(f"    Final cost: {costs[-1]:.6e}")
            print(f"    Final error: {final_error:.6e}")
            print(f"    Iterations: {len(costs)}")
            print(f"    Time: {end_time - start_time:.3f}s")
            print(f"    Converged: {np.linalg.norm(grad) < tolerance}")
    
    return results

def create_adaptive_optimizer_visualization(results):
    """
    Create comprehensive visualizations for adaptive optimizer comparison
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    optimizer_names = ['AdaGrad', 'RMSprop', 'Adam', 'AdamW', 'AdaDelta']
    problem_names = list(results.keys())
    
    # Plot 1-3: Convergence curves for each problem
    for i, problem_name in enumerate(problem_names):
        ax = axes[i, 0]
        
        for j, optimizer_name in enumerate(optimizer_names):
            if optimizer_name in results[problem_name]:
                costs = results[problem_name][optimizer_name]['costs']
                iterations = range(len(costs))
                ax.plot(iterations, costs, color=colors[j], linewidth=2, 
                       label=optimizer_name, alpha=0.8)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title(f'{problem_name} - Convergence')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4-6: Learning rate evolution
    for i, problem_name in enumerate(problem_names):
        ax = axes[i, 1]
        
        for j, optimizer_name in enumerate(optimizer_names):
            if optimizer_name in results[problem_name] and optimizer_name != 'AdaDelta':
                lr_history = results[problem_name][optimizer_name]['optimizer_history']['learning_rates']
                if lr_history:
                    iterations = range(len(lr_history))
                    ax.plot(iterations, lr_history, color=colors[j], linewidth=2, 
                           label=optimizer_name, alpha=0.8)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Effective Learning Rate')
        ax.set_title(f'{problem_name} - Learning Rate')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 7-9: Performance comparison
    for i, problem_name in enumerate(problem_names):
        ax = axes[i, 2]
        
        # Compare final performance
        final_costs = []
        optimizer_labels = []
        
        for optimizer_name in optimizer_names:
            if optimizer_name in results[problem_name]:
                final_costs.append(results[problem_name][optimizer_name]['final_cost'])
                optimizer_labels.append(optimizer_name)
        
        bars = ax.bar(optimizer_labels, final_costs, color=colors[:len(optimizer_labels)], alpha=0.7)
        ax.set_ylabel('Final Cost')
        ax.set_title(f'{problem_name} - Final Performance')
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, cost in zip(bars, final_costs):
            height = bar.get_height()
            ax.annotate(f'{cost:.2e}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_optimizer_characteristics():
    """
    Analyze and explain characteristics of each adaptive optimizer
    """
    print(f"\n{'='*70}")
    print("ADAPTIVE OPTIMIZER CHARACTERISTICS")
    print(f"{'='*70}")
    
    characteristics = {
        'AdaGrad': {
            'Key Innovation': 'Accumulates squared gradients for adaptive learning rates',
            'Strengths': ['Good for sparse gradients', 'No manual learning rate tuning'],
            'Weaknesses': ['Learning rate decays too aggressively', 'Can stop learning early'],
            'Best For': 'NLP tasks with sparse features',
            'Mathematical Form': 'θ = θ - (η / √(G + ε)) ⊙ g'
        },
        'RMSprop': {
            'Key Innovation': 'Uses exponential moving average instead of accumulation',
            'Strengths': ['Fixes AdaGrad learning rate decay', 'Works well in practice'],
            'Weaknesses': ['Still can have learning rate issues', 'Less theoretical justification'],
            'Best For': 'RNNs and general neural networks',
            'Mathematical Form': 'v = βv + (1-β)g², θ = θ - (η / √(v + ε)) ⊙ g'
        },
        'Adam': {
            'Key Innovation': 'Combines momentum and adaptive learning rates with bias correction',
            'Strengths': ['Fast convergence', 'Works well across domains', 'Bias correction'],
            'Weaknesses': ['Can have generalization issues', 'May not converge in some cases'],
            'Best For': 'Most deep learning applications',
            'Mathematical Form': 'm = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², θ = θ - η(m̂/√v̂ + ε)'
        },
        'AdamW': {
            'Key Innovation': 'Decouples weight decay from gradient-based update',
            'Strengths': ['Better regularization than Adam', 'Improved generalization'],
            'Weaknesses': ['More hyperparameters to tune', 'Slightly more complex'],
            'Best For': 'Transformers and models requiring good generalization',
            'Mathematical Form': 'Adam update + separate weight decay: θ = θ - ηλθ'
        },
        'AdaDelta': {
            'Key Innovation': 'Eliminates learning rate hyperparameter entirely',
            'Strengths': ['No learning rate tuning needed', 'Uses parameter updates for scaling'],
            'Weaknesses': ['Can be slow to converge', 'Less popular in practice'],
            'Best For': 'When you want to avoid learning rate tuning',
            'Mathematical Form': 'Δx = -(√(E[Δx²] + ε) / √(E[g²] + ε)) ⊙ g'
        }
    }
    
    for optimizer_name, props in characteristics.items():
        print(f"\n{optimizer_name}:")
        print(f"  Innovation: {props['Key Innovation']}")
        print(f"  Strengths: {', '.join(props['Strengths'])}")
        print(f"  Weaknesses: {', '.join(props['Weaknesses'])}")
        print(f"  Best For: {props['Best For']}")
        print(f"  Formula: {props['Mathematical Form']}")

if __name__ == "__main__":
    # Run comprehensive comparison
    results = compare_adaptive_optimizers()
    
    # Create visualizations
    create_adaptive_optimizer_visualization(results)
    
    # Analyze characteristics
    analyze_optimizer_characteristics()
    
    print(f"\n{'='*70}")
    print("PRACTICAL RECOMMENDATIONS")
    print(f"{'='*70}")
    
    print("\nGeneral Guidelines:")
    print("• Start with Adam (β₁=0.9, β₂=0.999, η=0.001)")
    print("• Use AdamW for better generalization (add weight_decay=0.01)")
    print("• Try RMSprop for RNNs (η=0.001, β=0.9)")
    print("• Use AdaGrad for sparse features (η=0.01)")
    print("• Consider AdaDelta when learning rate tuning is difficult")
    
    print("\nHyperparameter Tuning:")
    print("• Learning rate: Start with default, then search [1e-4, 1e-1]")
    print("• β₁ (momentum): Usually 0.9, try [0.8, 0.95] for fine-tuning")
    print("• β₂ (RMSprop): Usually 0.999, try [0.99, 0.9999] for stability")
    print("• Weight decay: Start with 0.01, search [1e-5, 1e-1]")
    
    print("\nImplementation Tips:")
    print("• Always include bias correction for Adam/AdamW")
    print("• Monitor effective learning rate, not just loss")
    print("• Use learning rate scheduling for better convergence")
    print("• Consider gradient clipping for stability")
    print("• Validate on separate dataset to check generalization")
```

**Key Points for Interviews:**

1. **Core Concepts**:
   - AdaGrad: Accumulates squared gradients
   - RMSprop: Exponential moving average of squared gradients
   - Adam: Combines momentum + adaptive LR + bias correction

2. **Mathematical Intuition**:
   - Larger gradients → smaller effective learning rate
   - Momentum helps escape shallow minima
   - Bias correction important in early iterations

3. **Practical Considerations**:
   - Adam is most versatile for deep learning
   - AdamW better for generalization
   - RMSprop good for RNNs
   - AdaGrad good for sparse features

4. **Implementation Details**:
   - Always add epsilon for numerical stability
   - Store moving averages as optimizer state
   - Reset state when starting new optimization

5. **Hyperparameter Guidelines**:
   - Adam: lr=0.001, β₁=0.9, β₂=0.999
   - RMSprop: lr=0.001, β=0.9
   - AdaGrad: lr=0.01

---

## Question 6

**Write a Python function that minimizes a cost function using simulated annealing.**

**Answer:**

Simulated Annealing is a probabilistic optimization algorithm inspired by the annealing process in metallurgy. It's particularly useful for non-convex optimization problems where gradient-based methods might get stuck in local minima:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class SimulatedAnnealing:
    """
    Comprehensive Simulated Annealing optimizer for global optimization
    """
    
    def __init__(self, initial_temperature=100.0, min_temperature=1e-8, 
                 cooling_rate=0.95, max_iterations_per_temp=100, 
                 neighborhood_size=1.0, random_state=None):
        """
        Initialize Simulated Annealing optimizer
        
        Args:
            initial_temperature: Starting temperature
            min_temperature: Minimum temperature (stopping criterion)
            cooling_rate: Factor to reduce temperature (0 < rate < 1)
            max_iterations_per_temp: Maximum iterations at each temperature
            neighborhood_size: Size of neighborhood for generating new solutions
            random_state: Random seed for reproducibility
        """
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations_per_temp = max_iterations_per_temp
        self.neighborhood_size = neighborhood_size
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Optimization history
        self.history = {
            'temperatures': [],
            'costs': [],
            'best_costs': [],
            'accepted_moves': [],
            'solutions': [],
            'acceptance_rates': []
        }
    
    def _generate_neighbor(self, current_solution, bounds=None):
        """
        Generate a neighboring solution
        
        Args:
            current_solution: Current solution vector
            bounds: Optional bounds for each parameter [(min, max), ...]
            
        Returns:
            numpy.ndarray: New neighboring solution
        """
        # Generate random perturbation
        perturbation = np.random.normal(0, self.neighborhood_size, len(current_solution))
        new_solution = current_solution + perturbation
        
        # Apply bounds if specified
        if bounds is not None:
            for i, (min_val, max_val) in enumerate(bounds):
                new_solution[i] = np.clip(new_solution[i], min_val, max_val)
        
        return new_solution
    
    def _acceptance_probability(self, current_cost, new_cost, temperature):
        """
        Calculate acceptance probability using Metropolis criterion
        
        Args:
            current_cost: Cost of current solution
            new_cost: Cost of new solution
            temperature: Current temperature
            
        Returns:
            float: Acceptance probability [0, 1]
        """
        if new_cost < current_cost:
            # Always accept better solutions
            return 1.0
        else:
            # Accept worse solutions with probability exp(-ΔE/T)
            delta_cost = new_cost - current_cost
            if temperature > 0:
                return np.exp(-delta_cost / temperature)
            else:
                return 0.0
    
    def optimize(self, cost_function, initial_solution, bounds=None, 
                 cooling_schedule='exponential', verbose=False):
        """
        Perform simulated annealing optimization
        
        Args:
            cost_function: Function to minimize f(x) -> float
            initial_solution: Starting solution vector
            bounds: Optional parameter bounds [(min, max), ...]
            cooling_schedule: 'exponential', 'linear', or 'logarithmic'
            verbose: Print progress information
            
        Returns:
            dict: Optimization results
        """
        # Initialize
        current_solution = np.array(initial_solution, dtype=float)
        current_cost = cost_function(current_solution)
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        temperature = self.initial_temperature
        total_iterations = 0
        
        if verbose:
            print("Simulated Annealing Optimization")
            print("=" * 40)
            print(f"Initial solution: {current_solution}")
            print(f"Initial cost: {current_cost:.6f}")
            print(f"Initial temperature: {temperature}")
        
        start_time = time.time()
        
        # Main optimization loop
        while temperature > self.min_temperature:
            accepted_moves = 0
            temp_iterations = 0
            
            # Iterations at current temperature
            for _ in range(self.max_iterations_per_temp):
                total_iterations += 1
                temp_iterations += 1
                
                # Generate neighboring solution
                new_solution = self._generate_neighbor(current_solution, bounds)
                new_cost = cost_function(new_solution)
                
                # Calculate acceptance probability
                acceptance_prob = self._acceptance_probability(current_cost, new_cost, temperature)
                
                # Accept or reject move
                if np.random.random() < acceptance_prob:
                    current_solution = new_solution
                    current_cost = new_cost
                    accepted_moves += 1
                    
                    # Update best solution if necessary
                    if new_cost < best_cost:
                        best_solution = new_solution.copy()
                        best_cost = new_cost
                
                # Record history
                self.history['costs'].append(current_cost)
                self.history['best_costs'].append(best_cost)
                self.history['solutions'].append(current_solution.copy())
                self.history['temperatures'].append(temperature)
            
            # Calculate acceptance rate for this temperature
            acceptance_rate = accepted_moves / temp_iterations
            self.history['accepted_moves'].append(accepted_moves)
            self.history['acceptance_rates'].append(acceptance_rate)
            
            if verbose:
                print(f"Temperature: {temperature:.6f}, Best cost: {best_cost:.6f}, "
                      f"Acceptance rate: {acceptance_rate:.3f}")
            
            # Cool down
            temperature = self._cool_temperature(temperature, cooling_schedule)
        
        end_time = time.time()
        
        return {
            'best_solution': best_solution,
            'best_cost': best_cost,
            'current_solution': current_solution,
            'current_cost': current_cost,
            'total_iterations': total_iterations,
            'final_temperature': temperature,
            'optimization_time': end_time - start_time,
            'history': self.history
        }
    
    def _cool_temperature(self, current_temp, schedule):
        """
        Apply cooling schedule to reduce temperature
        
        Args:
            current_temp: Current temperature
            schedule: Cooling schedule type
            
        Returns:
            float: New temperature
        """
        if schedule == 'exponential':
            return current_temp * self.cooling_rate
        elif schedule == 'linear':
            return current_temp - (self.initial_temperature - self.min_temperature) / 1000
        elif schedule == 'logarithmic':
            return current_temp / (1 + np.log(1 + current_temp))
        else:
            raise ValueError(f"Unknown cooling schedule: {schedule}")

class TestFunctions:
    """
    Collection of test functions for optimization
    """
    
    @staticmethod
    def rastrigin(x, A=10):
        """
        Rastrigin function: highly multimodal with many local minima
        Global minimum: f(0, 0, ..., 0) = 0
        """
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def ackley(x, a=20, b=0.2, c=2*np.pi):
        """
        Ackley function: multimodal with global minimum at origin
        Global minimum: f(0, 0, ..., 0) = 0
        """
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        
        term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        
        return term1 + term2 + a + np.exp(1)
    
    @staticmethod
    def sphere(x):
        """
        Sphere function: simple unimodal function
        Global minimum: f(0, 0, ..., 0) = 0
        """
        return np.sum(x**2)
    
    @staticmethod
    def rosenbrock(x):
        """
        Rosenbrock function: non-convex with global minimum in a valley
        Global minimum: f(1, 1, ..., 1) = 0
        """
        total = 0
        for i in range(len(x) - 1):
            total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return total
    
    @staticmethod
    def griewank(x):
        """
        Griewank function: multimodal with correlations between variables
        Global minimum: f(0, 0, ..., 0) = 0
        """
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_term - prod_term + 1

def demonstrate_simulated_annealing():
    """
    Demonstrate simulated annealing on various test functions
    """
    print("SIMULATED ANNEALING DEMONSTRATION")
    print("=" * 50)
    
    # Test functions with their properties
    test_functions = {
        'Sphere (Unimodal)': {
            'function': TestFunctions.sphere,
            'bounds': [(-5, 5)] * 2,
            'initial': [3, 4],
            'global_min': [0, 0],
            'min_value': 0
        },
        'Rastrigin (Multimodal)': {
            'function': TestFunctions.rastrigin,
            'bounds': [(-5.12, 5.12)] * 2,
            'initial': [3, 4],
            'global_min': [0, 0],
            'min_value': 0
        },
        'Ackley (Multimodal)': {
            'function': TestFunctions.ackley,
            'bounds': [(-32.768, 32.768)] * 2,
            'initial': [10, 15],
            'global_min': [0, 0],
            'min_value': 0
        },
        'Rosenbrock (Valley)': {
            'function': TestFunctions.rosenbrock,
            'bounds': [(-2, 2)] * 2,
            'initial': [-1, 1],
            'global_min': [1, 1],
            'min_value': 0
        }
    }
    
    results = {}
    
    for func_name, func_info in test_functions.items():
        print(f"\n{func_name}:")
        print("-" * 30)
        
        # Test different cooling schedules
        cooling_schedules = ['exponential', 'linear', 'logarithmic']
        func_results = {}
        
        for schedule in cooling_schedules:
            # Initialize optimizer
            sa = SimulatedAnnealing(
                initial_temperature=100.0,
                min_temperature=1e-6,
                cooling_rate=0.95,
                max_iterations_per_temp=50,
                neighborhood_size=0.5,
                random_state=42
            )
            
            # Optimize
            result = sa.optimize(
                cost_function=func_info['function'],
                initial_solution=func_info['initial'],
                bounds=func_info['bounds'],
                cooling_schedule=schedule,
                verbose=False
            )
            
            # Calculate error from global minimum
            error = np.linalg.norm(np.array(result['best_solution']) - np.array(func_info['global_min']))
            
            print(f"  {schedule.capitalize()} cooling:")
            print(f"    Best solution: {result['best_solution']}")
            print(f"    Best cost: {result['best_cost']:.6f}")
            print(f"    Error from global min: {error:.6f}")
            print(f"    Total iterations: {result['total_iterations']}")
            print(f"    Time: {result['optimization_time']:.3f}s")
            
            func_results[schedule] = result
        
        results[func_name] = func_results
    
    return results

def compare_with_other_optimizers():
    """
    Compare simulated annealing with other optimization methods
    """
    print(f"\n{'='*60}")
    print("COMPARISON WITH OTHER OPTIMIZERS")
    print(f"{'='*60}")
    
    # Use Rastrigin function for comparison (challenging multimodal function)
    def rastrigin_2d(x):
        return TestFunctions.rastrigin(x)
    
    bounds = [(-5.12, 5.12)] * 2
    initial_point = [3, 4]
    
    print(f"\nOptimizing Rastrigin function:")
    print(f"Initial point: {initial_point}")
    print(f"Global minimum: [0, 0] with value 0")
    
    optimizers = {}
    
    # 1. Simulated Annealing
    sa = SimulatedAnnealing(random_state=42)
    start_time = time.time()
    sa_result = sa.optimize(rastrigin_2d, initial_point, bounds, verbose=False)
    sa_time = time.time() - start_time
    optimizers['Simulated Annealing'] = {
        'solution': sa_result['best_solution'],
        'cost': sa_result['best_cost'],
        'time': sa_time,
        'iterations': sa_result['total_iterations']
    }
    
    # 2. Scipy optimizers
    scipy_methods = ['BFGS', 'L-BFGS-B', 'Powell', 'Nelder-Mead']
    
    for method in scipy_methods:
        start_time = time.time()
        try:
            if method == 'L-BFGS-B':
                scipy_result = minimize(rastrigin_2d, initial_point, method=method, bounds=bounds)
            else:
                scipy_result = minimize(rastrigin_2d, initial_point, method=method)
            
            scipy_time = time.time() - start_time
            optimizers[method] = {
                'solution': scipy_result.x,
                'cost': scipy_result.fun,
                'time': scipy_time,
                'iterations': scipy_result.nit if hasattr(scipy_result, 'nit') else 'N/A'
            }
        except:
            optimizers[method] = {
                'solution': 'Failed',
                'cost': np.inf,
                'time': 0,
                'iterations': 0
            }
    
    # Print comparison
    print(f"\nOptimization Results:")
    for method, result in optimizers.items():
        if isinstance(result['solution'], str):
            print(f"{method}: {result['solution']}")
        else:
            error = np.linalg.norm(np.array(result['solution']) - np.array([0, 0]))
            print(f"{method}:")
            print(f"  Solution: [{result['solution'][0]:.4f}, {result['solution'][1]:.4f}]")
            print(f"  Cost: {result['cost']:.6f}")
            print(f"  Error: {error:.6f}")
            print(f"  Time: {result['time']:.3f}s")
            print(f"  Iterations: {result['iterations']}")
    
    return optimizers

def create_simulated_annealing_visualization(results):
    """
    Create comprehensive visualizations for simulated annealing
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Select one function for detailed analysis
    func_name = 'Rastrigin (Multimodal)'
    if func_name in results:
        sa_result = results[func_name]['exponential']
        
        # Plot 1: Cost evolution
        ax1 = axes[0, 0]
        costs = sa_result['history']['costs']
        best_costs = sa_result['history']['best_costs']
        iterations = range(len(costs))
        
        ax1.plot(iterations, costs, 'b-', alpha=0.7, label='Current cost')
        ax1.plot(iterations, best_costs, 'r-', linewidth=2, label='Best cost')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.set_title('Cost Evolution')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Temperature evolution
        ax2 = axes[0, 1]
        temperatures = sa_result['history']['temperatures']
        ax2.plot(iterations, temperatures, 'g-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Temperature')
        ax2.set_title('Temperature Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Solution path on function surface
        ax3 = axes[0, 2]
        
        # Create function surface
        x_range = np.linspace(-5, 5, 100)
        y_range = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = TestFunctions.rastrigin([X[i, j], Y[i, j]])
        
        # Plot contours
        contours = ax3.contour(X, Y, Z, levels=20, alpha=0.6)
        ax3.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
        
        # Plot solution path
        solutions = np.array(sa_result['history']['solutions'])
        if len(solutions) > 0:
            ax3.plot(solutions[:, 0], solutions[:, 1], 'r.-', alpha=0.7, markersize=2)
            ax3.plot(solutions[0, 0], solutions[0, 1], 'go', markersize=10, label='Start')
            ax3.plot(solutions[-1, 0], solutions[-1, 1], 'ro', markersize=10, label='End')
            ax3.plot(0, 0, 'k*', markersize=15, label='Global minimum')
        
        ax3.set_xlabel('x₁')
        ax3.set_ylabel('x₂')
        ax3.set_title('Optimization Path')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Acceptance rate evolution
    ax4 = axes[1, 0]
    if func_name in results:
        acceptance_rates = sa_result['history']['acceptance_rates']
        temp_iterations = range(len(acceptance_rates))
        ax4.plot(temp_iterations, acceptance_rates, 'b-o', markersize=4)
        ax4.set_xlabel('Temperature Level')
        ax4.set_ylabel('Acceptance Rate')
        ax4.set_title('Acceptance Rate Evolution')
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Cooling schedule comparison
    ax5 = axes[1, 1]
    if func_name in results:
        schedules = ['exponential', 'linear', 'logarithmic']
        colors = ['red', 'blue', 'green']
        
        for i, schedule in enumerate(schedules):
            if schedule in results[func_name]:
                result = results[func_name][schedule]
                best_costs = result['history']['best_costs']
                iterations = range(len(best_costs))
                ax5.plot(iterations, best_costs, color=colors[i], 
                        linewidth=2, label=f'{schedule.capitalize()}')
        
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Best Cost')
        ax5.set_title('Cooling Schedule Comparison')
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance comparison across functions
    ax6 = axes[1, 2]
    
    function_names = list(results.keys())
    final_costs = []
    
    for func_name in function_names:
        if 'exponential' in results[func_name]:
            final_cost = results[func_name]['exponential']['best_cost']
            final_costs.append(final_cost)
    
    bars = ax6.bar(range(len(function_names)), final_costs, 
                   color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax6.set_xlabel('Test Function')
    ax6.set_ylabel('Final Cost')
    ax6.set_title('Performance Across Functions')
    ax6.set_xticks(range(len(function_names)))
    ax6.set_xticklabels([name.split(' ')[0] for name in function_names], rotation=45)
    ax6.set_yscale('log')
    
    # Add value labels on bars
    for bar, cost in zip(bars, final_costs):
        height = bar.get_height()
        ax6.annotate(f'{cost:.2e}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_sa_parameters():
    """
    Analyze the effect of different SA parameters
    """
    print(f"\n{'='*60}")
    print("PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    
    base_params = {
        'initial_temperature': 100.0,
        'min_temperature': 1e-6,
        'cooling_rate': 0.95,
        'max_iterations_per_temp': 50,
        'neighborhood_size': 0.5
    }
    
    # Test function
    test_func = TestFunctions.rastrigin
    initial_solution = [3, 4]
    bounds = [(-5.12, 5.12)] * 2
    
    print("\nParameter sensitivity on Rastrigin function:")
    
    # Test different parameter values
    param_tests = {
        'Initial Temperature': {
            'param': 'initial_temperature',
            'values': [10, 50, 100, 200, 500]
        },
        'Cooling Rate': {
            'param': 'cooling_rate',
            'values': [0.8, 0.9, 0.95, 0.98, 0.99]
        },
        'Neighborhood Size': {
            'param': 'neighborhood_size',
            'values': [0.1, 0.3, 0.5, 1.0, 2.0]
        }
    }
    
    for param_name, test_info in param_tests.items():
        print(f"\n{param_name}:")
        
        for value in test_info['values']:
            # Create modified parameters
            params = base_params.copy()
            params[test_info['param']] = value
            
            # Run optimization
            sa = SimulatedAnnealing(**params, random_state=42)
            result = sa.optimize(test_func, initial_solution, bounds, verbose=False)
            
            error = np.linalg.norm(np.array(result['best_solution']) - np.array([0, 0]))
            print(f"  {test_info['param']}={value}: "
                  f"cost={result['best_cost']:.4f}, error={error:.4f}")

if __name__ == "__main__":
    # Demonstrate simulated annealing
    results = demonstrate_simulated_annealing()
    
    # Compare with other optimizers
    comparison = compare_with_other_optimizers()
    
    # Create visualizations
    create_simulated_annealing_visualization(results)
    
    # Analyze parameters
    analyze_sa_parameters()
    
    print(f"\n{'='*60}")
    print("SIMULATED ANNEALING SUMMARY")
    print(f"{'='*60}")
    
    print("\nKey Advantages:")
    print("• Can escape local minima through probabilistic acceptance")
    print("• No gradient information required")
    print("• Works well on discrete and continuous problems")
    print("• Theoretical guarantee of finding global optimum")
    
    print("\nKey Disadvantages:")
    print("• Slow convergence compared to gradient-based methods")
    print("• Many hyperparameters to tune")
    print("• No guarantee of fast convergence in practice")
    print("• Performance depends heavily on cooling schedule")
    
    print("\nBest Practices:")
    print("• Start with high temperature to explore broadly")
    print("• Use exponential cooling for most problems")
    print("• Adjust neighborhood size based on problem scale")
    print("• Monitor acceptance rate (should start high, end low)")
    print("• Run multiple times with different random seeds")
    
    print("\nWhen to Use SA:")
    print("• Non-convex optimization with many local minima")
    print("• Combinatorial optimization problems")
    print("• When gradient information is unavailable")
    print("• Global optimization is more important than speed")
```

**Key Points for Interviews:**

1. **Algorithm Core**:
   - Metropolis criterion: P(accept) = exp(-ΔE/T) if ΔE > 0
   - Temperature controls exploration vs exploitation
   - Cooling schedule determines convergence

2. **Key Parameters**:
   - Initial temperature: High enough for broad exploration
   - Cooling rate: 0.8-0.99 (slower = better exploration)
   - Neighborhood size: Problem-dependent scaling
   - Stopping criteria: Minimum temperature or max iterations

3. **Cooling Schedules**:
   - Exponential: T(k) = α^k * T₀ (most common)
   - Linear: T(k) = T₀ - k*β
   - Logarithmic: T(k) = T₀/log(k+1)

4. **Advantages**:
   - Probabilistic escape from local minima
   - No gradient requirements
   - Theoretical convergence guarantees

5. **Practical Applications**:
   - Traveling Salesman Problem
   - Neural network training
   - Feature selection
   - Scheduling and routing problems

---

## Question 7

**Implement a basic version of theRMSprop optimization algorithminPython.**

**Answer:** _[To be filled]_

---

