# Python Ml Interview Questions - Scenario_Based Questions

## Question 1

**Discuss the difference between a list, a tuple, and a set in Python.**

### Theory
Python provides three fundamental data structures for collections: lists, tuples, and sets. Each has distinct characteristics regarding mutability, ordering, uniqueness, and performance, making them suitable for different use cases in machine learning applications.

### Answer

```python
import numpy as np
import pandas as pd
import time
from collections import Counter
import matplotlib.pyplot as plt
import sys

# Comprehensive demonstration of List, Tuple, and Set differences
print("=== Python Data Structures: List vs Tuple vs Set ===\n")

# 1. BASIC DEFINITIONS AND CREATION
print("1. BASIC DEFINITIONS AND CREATION")
print("-" * 50)

# Lists - Mutable, ordered collections
my_list = [1, 2, 3, 2, 4]
print(f"List: {my_list}")
print(f"List type: {type(my_list)}")

# Tuples - Immutable, ordered collections  
my_tuple = (1, 2, 3, 2, 4)
print(f"Tuple: {my_tuple}")
print(f"Tuple type: {type(my_tuple)}")

# Sets - Mutable, unordered collections with unique elements
my_set = {1, 2, 3, 2, 4}
print(f"Set: {my_set}")
print(f"Set type: {type(my_set)}")

# 2. MUTABILITY COMPARISON
print("\n2. MUTABILITY COMPARISON")
print("-" * 50)

# Lists are mutable
original_list = [1, 2, 3]
original_list.append(4)
original_list[0] = 10
print(f"Modified list: {original_list}")

# Tuples are immutable
original_tuple = (1, 2, 3)
try:
    original_tuple[0] = 10  # This will raise an error
except TypeError as e:
    print(f"Tuple modification error: {e}")

# Sets are mutable but elements must be immutable and unique
original_set = {1, 2, 3}
original_set.add(4)
original_set.discard(1)
print(f"Modified set: {original_set}")

# 3. ORDERING AND INDEXING
print("\n3. ORDERING AND INDEXING")
print("-" * 50)

data_list = ['a', 'b', 'c', 'd']
data_tuple = ('a', 'b', 'c', 'd')
data_set = {'a', 'b', 'c', 'd'}

# Lists and tuples maintain order and support indexing
print(f"List[1]: {data_list[1]}")
print(f"Tuple[1]: {data_tuple[1]}")
print(f"List slice [1:3]: {data_list[1:3]}")
print(f"Tuple slice [1:3]: {data_tuple[1:3]}")

# Sets don't maintain order and don't support indexing
try:
    print(f"Set[1]: {data_set[1]}")  # This will raise an error
except TypeError as e:
    print(f"Set indexing error: {e}")

print(f"Set iteration: {[item for item in data_set]}")

# 4. UNIQUENESS
print("\n4. UNIQUENESS")
print("-" * 50)

duplicate_data = [1, 2, 2, 3, 3, 3, 4]
list_with_duplicates = duplicate_data.copy()
tuple_with_duplicates = tuple(duplicate_data)
set_unique = set(duplicate_data)

print(f"List with duplicates: {list_with_duplicates}")
print(f"Tuple with duplicates: {tuple_with_duplicates}")
print(f"Set (automatically unique): {set_unique}")

# Remove duplicates from list while preserving order
def remove_duplicates_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

unique_list = remove_duplicates_preserve_order(duplicate_data)
print(f"List with duplicates removed (order preserved): {unique_list}")

# 5. PERFORMANCE COMPARISON
print("\n5. PERFORMANCE COMPARISON")
print("-" * 50)

def performance_test():
    """Compare performance of different operations"""
    n = 100000
    
    # Creation performance
    print("Creation Performance:")
    
    # List creation
    start = time.time()
    test_list = list(range(n))
    list_creation_time = time.time() - start
    print(f"List creation: {list_creation_time:.4f} seconds")
    
    # Tuple creation
    start = time.time()
    test_tuple = tuple(range(n))
    tuple_creation_time = time.time() - start
    print(f"Tuple creation: {tuple_creation_time:.4f} seconds")
    
    # Set creation
    start = time.time()
    test_set = set(range(n))
    set_creation_time = time.time() - start
    print(f"Set creation: {set_creation_time:.4f} seconds")
    
    # Access performance
    print("\nAccess Performance:")
    target = n // 2
    
    # List access
    start = time.time()
    _ = test_list[target]
    list_access_time = time.time() - start
    print(f"List access by index: {list_access_time:.6f} seconds")
    
    # Tuple access
    start = time.time()
    _ = test_tuple[target]
    tuple_access_time = time.time() - start
    print(f"Tuple access by index: {tuple_access_time:.6f} seconds")
    
    # Membership testing performance
    print("\nMembership Testing Performance:")
    
    # List membership
    start = time.time()
    _ = target in test_list
    list_membership_time = time.time() - start
    print(f"List membership test: {list_membership_time:.6f} seconds")
    
    # Tuple membership
    start = time.time()
    _ = target in test_tuple
    tuple_membership_time = time.time() - start
    print(f"Tuple membership test: {tuple_membership_time:.6f} seconds")
    
    # Set membership
    start = time.time()
    _ = target in test_set
    set_membership_time = time.time() - start
    print(f"Set membership test: {set_membership_time:.6f} seconds")
    
    # Memory usage comparison
    print("\nMemory Usage:")
    print(f"List memory: {sys.getsizeof(test_list)} bytes")
    print(f"Tuple memory: {sys.getsizeof(test_tuple)} bytes") 
    print(f"Set memory: {sys.getsizeof(test_set)} bytes")
    
    return {
        'creation': [list_creation_time, tuple_creation_time, set_creation_time],
        'membership': [list_membership_time, tuple_membership_time, set_membership_time]
    }

performance_results = performance_test()

# 6. COMMON OPERATIONS
print("\n6. COMMON OPERATIONS")
print("-" * 50)

# Lists - Common operations
sample_list = [1, 2, 3, 4, 5]
print("List Operations:")
print(f"Original: {sample_list}")
print(f"Append 6: {sample_list + [6]}")
print(f"Insert at index 2: {sample_list[:2] + [2.5] + sample_list[2:]}")
print(f"Remove element: {[x for x in sample_list if x != 3]}")
print(f"Reverse: {sample_list[::-1]}")
print(f"Sort: {sorted(sample_list, reverse=True)}")

# Tuples - Common operations
sample_tuple = (1, 2, 3, 4, 5)
print("\nTuple Operations:")
print(f"Original: {sample_tuple}")
print(f"Concatenate: {sample_tuple + (6, 7)}")
print(f"Count occurrences of 3: {sample_tuple.count(3)}")
print(f"Index of 4: {sample_tuple.index(4)}")
print(f"Unpacking: a, b, c, d, e = {sample_tuple}")

# Sets - Common operations
sample_set = {1, 2, 3, 4, 5}
other_set = {4, 5, 6, 7, 8}
print("\nSet Operations:")
print(f"Original: {sample_set}")
print(f"Union: {sample_set | other_set}")
print(f"Intersection: {sample_set & other_set}")
print(f"Difference: {sample_set - other_set}")
print(f"Symmetric difference: {sample_set ^ other_set}")
print(f"Subset check: {set([1, 2]).issubset(sample_set)}")

# 7. USE CASES IN MACHINE LEARNING
print("\n7. USE CASES IN MACHINE LEARNING")
print("-" * 50)

def ml_use_cases():
    """Demonstrate ML use cases for each data structure"""
    
    # Lists - Feature vectors, time series data
    print("LISTS - Best for:")
    print("• Feature vectors (ordered features matter)")
    print("• Time series data (temporal order)")
    print("• Training data batches")
    print("• Model predictions")
    
    # Example: Feature vector
    feature_vector = [0.5, 1.2, -0.8, 2.1, 0.3]  # Ordered features
    print(f"Feature vector example: {feature_vector}")
    
    # Example: Batch of samples
    batch_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(f"Batch data example: {batch_data}")
    
    print("\nTUPLES - Best for:")
    print("• Immutable configurations")
    print("• Coordinate pairs/triplets")
    print("• Dictionary keys (when hashable)")
    print("• Function return values")
    
    # Example: Model configuration
    model_config = ('linear', 100, 0.01, 'sgd')  # (type, epochs, lr, optimizer)
    print(f"Model config example: {model_config}")
    
    # Example: Coordinates
    data_points = [(1.5, 2.3), (3.1, 4.7), (0.8, 1.2)]
    print(f"Data points example: {data_points}")
    
    print("\nSETS - Best for:")
    print("• Unique identifiers")
    print("• Feature selection")
    print("• Class labels")
    print("• Fast membership testing")
    
    # Example: Unique labels
    unique_labels = {0, 1, 2, 3, 4}  # Classification classes
    print(f"Unique labels example: {unique_labels}")
    
    # Example: Selected features
    selected_features = {'age', 'income', 'education', 'experience'}
    print(f"Selected features example: {selected_features}")

ml_use_cases()

# 8. ADVANCED EXAMPLES
print("\n8. ADVANCED EXAMPLES")
print("-" * 50)

def advanced_examples():
    """Advanced usage patterns"""
    
    # List comprehensions vs set comprehensions
    data = [1, 2, 3, 4, 5, 2, 3, 1]
    
    # List comprehension (preserves duplicates and order)
    squared_list = [x**2 for x in data if x % 2 == 0]
    print(f"List comprehension (even squares): {squared_list}")
    
    # Set comprehension (unique values only)
    squared_set = {x**2 for x in data if x % 2 == 0}
    print(f"Set comprehension (unique even squares): {squared_set}")
    
    # Tuple unpacking in functions
    def get_model_metrics():
        return (0.95, 0.87, 0.91)  # (accuracy, precision, recall)
    
    accuracy, precision, recall = get_model_metrics()
    print(f"Model metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
    
    # Using sets for fast filtering
    valid_ids = {1, 3, 5, 7, 9, 11, 13, 15}
    data_samples = [
        {'id': 1, 'value': 10},
        {'id': 2, 'value': 20},
        {'id': 3, 'value': 30},
        {'id': 4, 'value': 40},
        {'id': 5, 'value': 50}
    ]
    
    # Fast filtering using set membership
    valid_samples = [sample for sample in data_samples if sample['id'] in valid_ids]
    print(f"Valid samples: {valid_samples}")
    
    # Using tuples as dictionary keys
    model_cache = {}
    model_cache[('svm', 'rbf', 1.0)] = 0.95  # (algorithm, kernel, C) -> accuracy
    model_cache[('rf', 100, 'gini')] = 0.93   # (algorithm, n_estimators, criterion) -> accuracy
    
    print(f"Model cache: {model_cache}")
    
    # Converting between data structures
    original_list = [1, 2, 3, 2, 1, 4]
    print(f"Original list: {original_list}")
    print(f"List -> Set (unique): {set(original_list)}")
    print(f"List -> Tuple (immutable): {tuple(original_list)}")
    print(f"Set -> List (ordered): {list(set(original_list))}")

advanced_examples()

# 9. PERFORMANCE VISUALIZATION
print("\n9. PERFORMANCE VISUALIZATION")
print("-" * 50)

# Visualize performance comparison
def plot_performance():
    """Plot performance comparison"""
    operations = ['Creation', 'Membership Test']
    list_times = [performance_results['creation'][0], performance_results['membership'][0]]
    tuple_times = [performance_results['creation'][1], performance_results['membership'][1]]
    set_times = [performance_results['creation'][2], performance_results['membership'][2]]
    
    x = np.arange(len(operations))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, list_times, width, label='List', alpha=0.8)
    plt.bar(x, tuple_times, width, label='Tuple', alpha=0.8)
    plt.bar(x + width, set_times, width, label='Set', alpha=0.8)
    
    plt.xlabel('Operations')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison: List vs Tuple vs Set')
    plt.xticks(x, operations)
    plt.legend()
    plt.yscale('log')  # Use log scale for better visualization
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_performance()

print("\n=== SUMMARY ===")
print("\nChoose Lists when:")
print("• You need ordered, mutable collections")
print("• Order of elements matters")
print("• You need to modify elements frequently")
print("• You're working with sequences/time series")

print("\nChoose Tuples when:")
print("• You need ordered, immutable collections") 
print("• You want to ensure data integrity")
print("• You need hashable collections (dict keys)")
print("• You're returning multiple values from functions")

print("\nChoose Sets when:")
print("• You need unique elements only")
print("• Fast membership testing is important")
print("• You need set operations (union, intersection)")
print("• Order doesn't matter")

print("\n=== Data Structures Comparison Complete ===")
```

### Explanation

1. **Mutability**: Lists and sets are mutable (can be changed), tuples are immutable (cannot be changed after creation)

2. **Ordering**: Lists and tuples maintain insertion order and support indexing, sets are unordered collections

3. **Uniqueness**: Sets automatically enforce uniqueness, lists and tuples allow duplicates

4. **Performance**: Sets excel at membership testing (O(1)), lists/tuples have O(n) membership testing but support indexing

5. **Use Cases**: Lists for sequences, tuples for immutable data, sets for unique collections and fast lookups

### Use Cases in ML

- **Lists**: Feature vectors, training batches, time series data, model predictions
- **Tuples**: Model configurations, coordinate pairs, function returns, immutable parameters  
- **Sets**: Unique labels, feature selection, fast filtering, vocabulary management

### Best Practices

- **Memory Efficiency**: Tuples use less memory than lists for the same data
- **Performance**: Use sets for membership testing, lists for ordered access
- **Immutability**: Use tuples when data shouldn't change (configurations, coordinates)
- **Uniqueness**: Use sets when duplicates must be avoided
- **Conversion**: Convert between types as needed: `list(my_set)`, `set(my_list)`, `tuple(my_list)`

### Pitfalls

- **Set Ordering**: Sets don't maintain order (in Python <3.7, order was not guaranteed)
- **Tuple Immutability**: Can't modify tuple elements, must create new tuple
- **Set Elements**: Set elements must be hashable (no lists or dicts as elements)
- **Performance Assumptions**: Don't assume all operations are equally fast across types
- **Memory Usage**: Lists can be more memory-efficient for small collections

**Answer:** _[To be filled]_

---

## Question 2

**Discuss the usage of *args and **kwargs in function definitions.**

### Theory
`*args` and `**kwargs` are Python conventions for handling variable numbers of arguments in functions. `*args` allows functions to accept any number of positional arguments, while `**kwargs` allows functions to accept any number of keyword arguments. These features provide flexibility in function design and are essential for creating reusable, extensible code.

### Answer

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import inspect
from functools import wraps
import time

# Comprehensive demonstration of *args and **kwargs
print("=== *args and **kwargs in Python Functions ===\n")

# 1. BASIC USAGE OF *args
print("1. BASIC USAGE OF *args")
print("-" * 40)

def basic_args_example(*args):
    """Function that accepts any number of positional arguments"""
    print(f"Received {len(args)} arguments:")
    for i, arg in enumerate(args):
        print(f"  arg[{i}] = {arg}")
    return sum(args) if all(isinstance(arg, (int, float)) for arg in args) else None

# Examples
print("Example 1: Multiple numbers")
result1 = basic_args_example(1, 2, 3, 4, 5)
print(f"Sum: {result1}\n")

print("Example 2: Mixed types")
basic_args_example("hello", 42, [1, 2, 3], True)
print()

# 2. BASIC USAGE OF **kwargs
print("2. BASIC USAGE OF **kwargs")
print("-" * 40)

def basic_kwargs_example(**kwargs):
    """Function that accepts any number of keyword arguments"""
    print(f"Received {len(kwargs)} keyword arguments:")
    for key, value in kwargs.items():
        print(f"  {key} = {value}")
    return kwargs

# Examples
print("Example 1: Configuration parameters")
config = basic_kwargs_example(
    learning_rate=0.01,
    epochs=100,
    batch_size=32,
    optimizer='adam'
)
print(f"Returned config: {config}\n")

print("Example 2: Model parameters")
basic_kwargs_example(
    model_type='random_forest',
    n_estimators=100,
    max_depth=10,
    random_state=42
)
print()

# 3. COMBINING *args AND **kwargs
print("3. COMBINING *args AND **kwargs")
print("-" * 40)

def combined_example(required_param, *args, default_param="default", **kwargs):
    """Function demonstrating all parameter types"""
    print(f"Required parameter: {required_param}")
    print(f"Default parameter: {default_param}")
    print(f"*args: {args}")
    print(f"**kwargs: {kwargs}")
    
    # Process arguments
    processed_args = [arg * 2 for arg in args if isinstance(arg, (int, float))]
    processed_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str))}
    
    return {
        'required': required_param,
        'default': default_param,
        'processed_args': processed_args,
        'processed_kwargs': processed_kwargs
    }

# Example usage
print("Combined function call:")
result = combined_example(
    "mandatory_value",           # required_param
    1, 2, 3, 4,                 # *args
    default_param="custom",      # default_param override
    learning_rate=0.001,        # **kwargs
    epochs=200,
    model_name="neural_net"
)
print(f"Result: {result}\n")

# 4. PRACTICAL ML EXAMPLE: FLEXIBLE MODEL TRAINER
print("4. PRACTICAL ML EXAMPLE: FLEXIBLE MODEL TRAINER")
print("-" * 50)

class FlexibleModelTrainer:
    """ML model trainer using *args and **kwargs for flexibility"""
    
    def __init__(self, default_random_state=42):
        self.default_random_state = default_random_state
        self.models = {}
        self.results = {}
    
    def train_model(self, model_name, model_class, X_train, y_train, *args, **kwargs):
        """
        Train a model with flexible parameters
        
        Args:
            model_name: Name for the model
            model_class: ML model class (e.g., RandomForestClassifier)
            X_train, y_train: Training data
            *args: Positional arguments for model initialization
            **kwargs: Keyword arguments for model initialization
        """
        # Set default random_state if not provided
        if 'random_state' not in kwargs:
            kwargs['random_state'] = self.default_random_state
        
        print(f"Training {model_name}:")
        print(f"  Model class: {model_class.__name__}")
        print(f"  Args: {args}")
        print(f"  Kwargs: {kwargs}")
        
        # Initialize and train model
        model = model_class(*args, **kwargs)
        model.fit(X_train, y_train)
        
        # Store model
        self.models[model_name] = model
        
        return model
    
    def evaluate_models(self, X_test, y_test, *model_names, **eval_kwargs):
        """
        Evaluate multiple models
        
        Args:
            X_test, y_test: Test data
            *model_names: Names of models to evaluate (if empty, evaluates all)
            **eval_kwargs: Additional evaluation parameters
        """
        models_to_evaluate = model_names if model_names else self.models.keys()
        
        print(f"\nEvaluating models: {list(models_to_evaluate)}")
        if eval_kwargs:
            print(f"Evaluation parameters: {eval_kwargs}")
        
        results = {}
        for name in models_to_evaluate:
            if name in self.models:
                predictions = self.models[name].predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                results[name] = accuracy
                print(f"  {name}: {accuracy:.4f}")
        
        self.results.update(results)
        return results

# Demonstrate flexible model trainer
print("Creating sample dataset...")
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

trainer = FlexibleModelTrainer()

# Train different models with different parameters
print("\nTraining models with *args and **kwargs:")

# Random Forest with keyword arguments
trainer.train_model(
    "rf_100", RandomForestClassifier, X_train, y_train,
    n_estimators=100, max_depth=10, min_samples_split=5
)

# Random Forest with different parameters
trainer.train_model(
    "rf_200", RandomForestClassifier, X_train, y_train,
    n_estimators=200, max_depth=15
)

# Logistic Regression
trainer.train_model(
    "logistic", LogisticRegression, X_train, y_train,
    max_iter=1000, C=1.0
)

# SVM
trainer.train_model(
    "svm", SVC, X_train, y_train,
    kernel='rbf', C=1.0, gamma='scale'
)

# Evaluate all models
trainer.evaluate_models(X_test, y_test)

# Evaluate specific models
print("\nEvaluating specific models:")
trainer.evaluate_models(X_test, y_test, "rf_100", "logistic")

# 5. ADVANCED PATTERNS: DECORATORS WITH *args AND **kwargs
print("\n5. ADVANCED PATTERNS: DECORATORS")
print("-" * 45)

def timing_decorator(func):
    """Decorator that times function execution using *args and **kwargs"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def logging_decorator(func):
    """Decorator that logs function calls"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {type(result)}")
        return result
    return wrapper

@timing_decorator
@logging_decorator
def expensive_computation(*numbers, **options):
    """Simulate expensive computation with flexible parameters"""
    multiplier = options.get('multiplier', 1)
    add_noise = options.get('add_noise', False)
    
    result = sum(x ** 2 for x in numbers) * multiplier
    
    if add_noise:
        result += np.random.randn()
    
    # Simulate computation time
    time.sleep(0.1)
    return result

# Example usage
print("\nTesting decorated function:")
result = expensive_computation(1, 2, 3, 4, 5, multiplier=2, add_noise=True)
print(f"Final result: {result}\n")

# 6. FUNCTION INTROSPECTION AND DYNAMIC CALLS
print("6. FUNCTION INTROSPECTION AND DYNAMIC CALLS")
print("-" * 50)

def dynamic_function_caller(func, *args, **kwargs):
    """Dynamically call function and inspect its signature"""
    print(f"Calling function: {func.__name__}")
    
    # Get function signature
    sig = inspect.signature(func)
    print(f"Function signature: {sig}")
    
    # Check if function accepts *args and **kwargs
    has_var_positional = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
    has_var_keyword = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    
    print(f"Accepts *args: {has_var_positional}")
    print(f"Accepts **kwargs: {has_var_keyword}")
    
    # Call the function
    try:
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    except Exception as e:
        print(f"Error calling function: {e}")
        return None

# Test with different functions
def simple_function(a, b):
    return a + b

def flexible_function(a, b=10, *args, **kwargs):
    return a + b + sum(args) + sum(kwargs.values())

print("Testing simple function:")
dynamic_function_caller(simple_function, 5, 3)
print()

print("Testing flexible function:")
dynamic_function_caller(flexible_function, 1, 2, 3, 4, 5, x=10, y=20)
print()

# 7. REAL-WORLD ML EXAMPLE: HYPERPARAMETER TUNING
print("7. REAL-WORLD EXAMPLE: HYPERPARAMETER TUNING")
print("-" * 55)

class HyperparameterTuner:
    """Hyperparameter tuning with flexible parameter passing"""
    
    def __init__(self):
        self.best_score = 0
        self.best_params = {}
        self.best_model = None
    
    def tune_model(self, model_class, X_train, y_train, X_val, y_val, 
                   param_grid, *args, **base_kwargs):
        """
        Tune hyperparameters for a model
        
        Args:
            model_class: Model class to tune
            X_train, y_train: Training data
            X_val, y_val: Validation data
            param_grid: Dictionary of parameters to try
            *args: Additional positional arguments for model
            **base_kwargs: Base keyword arguments for model
        """
        print(f"Tuning {model_class.__name__}")
        print(f"Base args: {args}")
        print(f"Base kwargs: {base_kwargs}")
        print(f"Parameter grid: {param_grid}")
        
        for param_name, param_values in param_grid.items():
            print(f"\nTrying parameter {param_name}:")
            
            for param_value in param_values:
                # Create model kwargs by combining base kwargs with current parameter
                model_kwargs = base_kwargs.copy()
                model_kwargs[param_name] = param_value
                
                # Train model
                model = model_class(*args, **model_kwargs)
                model.fit(X_train, y_train)
                
                # Evaluate
                score = model.score(X_val, y_val)
                print(f"  {param_name}={param_value}: score={score:.4f}")
                
                # Update best if needed
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = model_kwargs.copy()
                    self.best_model = model
        
        print(f"\nBest score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_model, self.best_params

# Example hyperparameter tuning
print("Example: Tuning Random Forest")
tuner = HyperparameterTuner()

# Split training data further for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

# Tune model with base parameters
best_model, best_params = tuner.tune_model(
    RandomForestClassifier,
    X_train_split, y_train_split,
    X_val_split, y_val_split,
    param_grid,
    # Base parameters
    random_state=42,
    min_samples_split=2
)

# 8. COMMON PATTERNS AND BEST PRACTICES
print("\n8. COMMON PATTERNS AND BEST PRACTICES")
print("-" * 50)

def api_wrapper(endpoint, *args, method="GET", **params):
    """Example API wrapper using *args and **kwargs"""
    print(f"API Call to {endpoint}")
    print(f"Method: {method}")
    print(f"Path parameters: {args}")
    print(f"Query parameters: {params}")
    
    # Simulate API call
    url = f"{endpoint}/{'/'.join(map(str, args))}"
    if params:
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        url += f"?{query_string}"
    
    print(f"Full URL: {url}")
    return {"status": "success", "url": url}

# Example API calls
print("API Examples:")
api_wrapper("users", 123, "profile", format="json", include_details=True)
print()
api_wrapper("models", method="POST", name="new_model", type="classifier")
print()

def configuration_manager(**configs):
    """Manage configuration with flexible parameters"""
    default_config = {
        'debug': False,
        'verbose': True,
        'timeout': 30,
        'retry_count': 3
    }
    
    # Merge with provided configs
    final_config = {**default_config, **configs}
    
    print("Configuration Manager:")
    for key, value in final_config.items():
        print(f"  {key}: {value}")
    
    return final_config

# Configuration examples
print("Configuration Examples:")
config1 = configuration_manager(debug=True, timeout=60)
print()
config2 = configuration_manager(verbose=False, new_param="custom_value")
print()

print("=== SUMMARY ===")
print("\n*args Usage:")
print("• Accept variable number of positional arguments")
print("• Useful for functions that work with lists/sequences")
print("• Common in mathematical functions, data processing")
print("• Example: sum(*numbers), plot(*coordinates)")

print("\n**kwargs Usage:")
print("• Accept variable number of keyword arguments")
print("• Useful for configuration and optional parameters")
print("• Common in APIs, model initialization, plotting")
print("• Example: model(**params), plot(**style_options)")

print("\nBest Practices:")
print("• Use *args for flexible positional parameters")
print("• Use **kwargs for optional configuration")
print("• Document expected argument types and formats")
print("• Provide sensible defaults in **kwargs")
print("• Use type hints when possible: func(*args: int, **kwargs: Any)")

print("\nCommon Patterns:")
print("• Decorators: wrapper(*args, **kwargs)")
print("• API wrappers: request(url, *path, **params)")
print("• Configuration: setup(**config)")
print("• Forwarding calls: super().method(*args, **kwargs)")

print("\n=== *args and **kwargs Demonstration Complete ===")
```

### Explanation

1. **Basic Concepts**: `*args` collects positional arguments into a tuple, `**kwargs` collects keyword arguments into a dictionary

2. **Function Flexibility**: These allow functions to accept varying numbers of arguments without defining them explicitly

3. **Parameter Order**: Required parameters → *args → keyword-only parameters → **kwargs

4. **Practical Applications**: Model initialization, API wrappers, configuration management, decorators

5. **Advanced Patterns**: Dynamic function calling, parameter forwarding, flexible class initialization

### Use Cases in ML

- **Model Training**: Flexible parameter passing to different algorithms
- **Data Processing**: Variable input handling for preprocessing pipelines  
- **API Design**: Creating extensible interfaces for ML services
- **Configuration**: Managing hyperparameters and model settings
- **Decorators**: Adding functionality like timing, logging, caching to ML functions

### Best Practices

- **Documentation**: Clearly document expected argument types and formats
- **Type Hints**: Use type annotations: `*args: int`, `**kwargs: Any`
- **Validation**: Check argument types and values when necessary
- **Defaults**: Provide sensible defaults for **kwargs parameters
- **Unpacking**: Use `*` and `**` to unpack arguments when calling functions

### Pitfalls

- **Argument Order**: Wrong parameter order can cause unexpected behavior
- **Type Safety**: No automatic type checking for *args and **kwargs
- **Documentation**: Hard to document all possible parameters
- **Debugging**: Error messages may be less clear with flexible parameters
- **Performance**: Slight overhead compared to fixed parameters

### Debugging

- **Introspection**: Use `inspect` module to examine function signatures
- **Logging**: Log received arguments for debugging
- **Validation**: Add explicit type and value checks
- **Testing**: Test with various argument combinations
- **Documentation**: Use docstrings to explain expected parameters

**Answer:** _[To be filled]_

---

## Question 3

**Discuss the benefits of using Jupyter Notebooks for machine learning projects.**

### Theory
Jupyter Notebooks are interactive computing environments that combine code execution, rich text, mathematics, plots, and media in a single document. They have become the de facto standard for machine learning experimentation, prototyping, and data analysis due to their flexibility, interactivity, and ability to create reproducible research workflows.

### Answer

```python
# jupyter_ml_benefits.py - Comprehensive demonstration of Jupyter Notebooks for ML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import make_classification, load_iris
import warnings
warnings.filterwarnings('ignore')

print("=== Benefits of Jupyter Notebooks for Machine Learning ===\n")

# 1. INTERACTIVE DATA EXPLORATION
print("1. INTERACTIVE DATA EXPLORATION")
print("-" * 40)

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                          n_informative=5, n_redundant=2, random_state=42)
feature_names = [f'feature_{i}' for i in range(10)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Features: {list(df.columns[:-1])}")
print(f"Target classes: {sorted(df['target'].unique())}")
print(f"Class distribution:\n{df['target'].value_counts()}")

# Interactive visualization
plt.figure(figsize=(12, 8))

# Subplot 1: Feature distributions
plt.subplot(2, 3, 1)
df[['feature_0', 'feature_1', 'feature_2']].hist(bins=20, alpha=0.7)
plt.title('Feature Distributions')

# Subplot 2: Correlation heatmap
plt.subplot(2, 3, 2)
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix.iloc[:5, :5], annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')

# Subplot 3: Target distribution
plt.subplot(2, 3, 3)
df['target'].value_counts().plot(kind='bar')
plt.title('Target Distribution')
plt.xticks(rotation=0)

# Subplot 4: Feature vs target relationship
plt.subplot(2, 3, 4)
for target_class in sorted(df['target'].unique()):
    subset = df[df['target'] == target_class]
    plt.scatter(subset['feature_0'], subset['feature_1'], 
               label=f'Class {target_class}', alpha=0.6)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('Feature 0 vs Feature 1 by Class')
plt.legend()

# Subplot 5: Box plot
plt.subplot(2, 3, 5)
df.boxplot(column='feature_0', by='target', ax=plt.gca())
plt.title('Feature 0 Distribution by Target')
plt.suptitle('')  # Remove automatic title

# Subplot 6: Statistical summary
plt.subplot(2, 3, 6)
summary_stats = df.groupby('target')[['feature_0', 'feature_1']].mean()
summary_stats.plot(kind='bar')
plt.title('Mean Feature Values by Class')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

print("\nBenefit: Interactive visualizations help understand data patterns immediately")
print("✓ Immediate feedback on data exploration")
print("✓ Visual validation of hypotheses")
print("✓ Easy identification of data quality issues")
print()

# 2. ITERATIVE MODEL DEVELOPMENT
print("2. ITERATIVE MODEL DEVELOPMENT")
print("-" * 40)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model comparison framework
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

print("Model Performance Comparison:")
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Classification Report:")
    print(f"    {classification_report(y_test, y_pred, output_dict=False)}")

print("\nBenefit: Easy model comparison and iteration")
print("✓ Quick prototyping and testing")
print("✓ Side-by-side model comparison")
print("✓ Immediate results visualization")
print()

# 3. DOCUMENTATION AND STORYTELLING
print("3. DOCUMENTATION AND STORYTELLING")
print("-" * 40)

# Create a comprehensive analysis with markdown-style documentation
analysis_steps = [
    {
        "step": "Data Loading and Initial Exploration",
        "description": "Load dataset and examine basic statistics",
        "code_executed": True,
        "insights": [
            f"Dataset contains {df.shape[0]} samples and {df.shape[1]-1} features",
            f"Target has {len(df['target'].unique())} classes",
            "No missing values detected",
            "Features appear to be continuous variables"
        ]
    },
    {
        "step": "Feature Analysis",
        "description": "Analyze feature distributions and correlations",
        "code_executed": True,
        "insights": [
            "Features show varying distributions",
            "Some features are correlated",
            "Clear separation visible between classes"
        ]
    },
    {
        "step": "Model Training and Evaluation",
        "description": "Train multiple models and compare performance",
        "code_executed": True,
        "insights": [
            f"Random Forest achieved {results['Random Forest']['accuracy']:.4f} accuracy",
            f"Logistic Regression achieved {results['Logistic Regression']['accuracy']:.4f} accuracy",
            "Random Forest performs better on this dataset"
        ]
    }
]

print("ML Project Documentation Structure:")
for i, step in enumerate(analysis_steps, 1):
    print(f"\n{i}. {step['step']}")
    print(f"   Description: {step['description']}")
    print(f"   Insights:")
    for insight in step['insights']:
        print(f"   • {insight}")

print("\nBenefit: Combines code, results, and narrative in one document")
print("✓ Self-documenting analysis")
print("✓ Easy to share findings")
print("✓ Reproducible research")
print()

# 4. COLLABORATION AND SHARING
print("4. COLLABORATION AND SHARING")
print("-" * 40)

# Example of collaborative notebook structure
collaboration_features = {
    "Version Control": {
        "description": "Track changes and collaborate through Git",
        "examples": [
            "Use nbstripout to clean outputs before commits",
            "Create separate branches for experiments",
            "Merge notebooks with conflict resolution"
        ]
    },
    "Sharing Mechanisms": {
        "description": "Multiple ways to share notebooks",
        "examples": [
            "GitHub/GitLab notebook rendering",
            "Export to HTML/PDF for stakeholders",
            "NBViewer for public sharing",
            "Jupyter Hub for team collaboration"
        ]
    },
    "Commenting and Discussion": {
        "description": "Built-in ways to discuss analysis",
        "examples": [
            "Markdown cells for explanations",
            "Code comments for technical details",
            "Output preservation for result sharing"
        ]
    }
}

print("Collaboration Features:")
for feature, details in collaboration_features.items():
    print(f"\n{feature}:")
    print(f"  {details['description']}")
    for example in details['examples']:
        print(f"  • {example}")

print("\nBenefit: Enhanced team collaboration and knowledge sharing")
print("✓ Real-time collaboration possible")
print("✓ Easy result sharing")
print("✓ Discussion and documentation combined")
print()

# 5. RAPID PROTOTYPING AND EXPERIMENTATION
print("5. RAPID PROTOTYPING AND EXPERIMENTATION")
print("-" * 40)

# Demonstrate rapid experimentation
experiment_results = {}

# Experiment 1: Feature selection impact
print("Experiment 1: Feature Selection Impact")
from sklearn.feature_selection import SelectKBest, f_classif

# Test different numbers of features
for k in [3, 5, 7, 10]:
    if k <= X_train.shape[1]:
        # Select top k features
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_selected, y_train)
        
        # Evaluate
        accuracy = model.score(X_test_selected, y_test)
        experiment_results[f"top_{k}_features"] = accuracy
        print(f"  Top {k} features: {accuracy:.4f} accuracy")

print(f"\nBest feature count: {max(experiment_results, key=experiment_results.get)}")

# Experiment 2: Hyperparameter impact
print("\nExperiment 2: Hyperparameter Impact")
hp_results = {}

for n_estimators in [50, 100, 200]:
    for max_depth in [5, 10, None]:
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        hp_results[f"est_{n_estimators}_depth_{max_depth}"] = accuracy
        print(f"  n_estimators={n_estimators}, max_depth={max_depth}: {accuracy:.4f}")

print(f"\nBest hyperparameters: {max(hp_results, key=hp_results.get)}")

print("\nBenefit: Rapid experimentation and hypothesis testing")
print("✓ Quick parameter tuning")
print("✓ Immediate feedback on changes")
print("✓ Easy A/B testing of approaches")
print()

# 6. EDUCATIONAL AND LEARNING BENEFITS
print("6. EDUCATIONAL AND LEARNING BENEFITS")
print("-" * 40)

# Create educational content structure
educational_content = {
    "Concept Explanation": {
        "purpose": "Explain ML concepts with interactive examples",
        "example": "Demonstrate overfitting with polynomial regression"
    },
    "Step-by-step Tutorials": {
        "purpose": "Break down complex workflows into digestible steps",
        "example": "Complete ML pipeline from data loading to deployment"
    },
    "Interactive Demonstrations": {
        "purpose": "Show algorithm behavior with parameter changes",
        "example": "Visualize decision boundaries with different classifiers"
    },
    "Best Practices": {
        "purpose": "Demonstrate good ML practices",
        "example": "Proper train/validation/test splits and cross-validation"
    }
}

print("Educational Benefits:")
for content_type, details in educational_content.items():
    print(f"\n{content_type}:")
    print(f"  Purpose: {details['purpose']}")
    print(f"  Example: {details['example']}")

# Interactive learning example: Algorithm comparison
print("\nInteractive Learning Example: Decision Boundary Visualization")

# Create 2D dataset for visualization
X_2d, y_2d = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                                n_informative=2, n_clusters_per_class=1, random_state=42)

plt.figure(figsize=(15, 5))

# Visualize different classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Linear SVM': SVC(kernel='linear', probability=True)
}

for i, (name, clf) in enumerate(classifiers.items(), 1):
    plt.subplot(1, 3, i)
    
    # Train classifier
    clf.fit(X_2d, y_2d)
    
    # Create decision boundary
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdYlBu)
    plt.title(f'{name}\nAccuracy: {clf.score(X_2d, y_2d):.3f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("\nBenefit: Interactive learning and concept visualization")
print("✓ Visual algorithm comparison")
print("✓ Immediate concept understanding")
print("✓ Hands-on experimentation")
print()

# 7. INTEGRATION WITH ML ECOSYSTEM
print("7. INTEGRATION WITH ML ECOSYSTEM")
print("-" * 40)

# Demonstrate ecosystem integration
ecosystem_tools = {
    "Data Science Libraries": {
        "tools": ["pandas", "numpy", "scipy", "scikit-learn"],
        "purpose": "Core data manipulation and ML algorithms"
    },
    "Visualization": {
        "tools": ["matplotlib", "seaborn", "plotly", "bokeh"],
        "purpose": "Static and interactive visualizations"
    },
    "Deep Learning": {
        "tools": ["tensorflow", "pytorch", "keras"],
        "purpose": "Neural network development and training"
    },
    "Big Data": {
        "tools": ["pyspark", "dask", "vaex"],
        "purpose": "Large-scale data processing"
    },
    "Model Management": {
        "tools": ["mlflow", "wandb", "tensorboard"],
        "purpose": "Experiment tracking and model versioning"
    },
    "Deployment": {
        "tools": ["flask", "fastapi", "streamlit"],
        "purpose": "Model serving and app development"
    }
}

print("ML Ecosystem Integration:")
for category, details in ecosystem_tools.items():
    print(f"\n{category}:")
    print(f"  Tools: {', '.join(details['tools'])}")
    print(f"  Purpose: {details['purpose']}")

print("\nBenefit: Seamless integration with entire ML workflow")
print("✓ One environment for complete pipeline")
print("✓ Easy library switching and comparison")
print("✓ Integrated development experience")
print()

# 8. REPRODUCIBILITY AND AUTOMATION
print("8. REPRODUCIBILITY AND AUTOMATION")
print("-" * 40)

# Demonstrate reproducibility features
reproducibility_features = {
    "Environment Management": [
        "requirements.txt generation",
        "conda environment export",
        "Docker container creation",
        "Virtual environment integration"
    ],
    "Execution Control": [
        "Cell execution order tracking",
        "Output preservation",
        "Kernel state management",
        "Checkpoint and restart capabilities"
    ],
    "Automation Integration": [
        "nbconvert for batch processing",
        "Papermill for parameterization",
        "CI/CD pipeline integration",
        "Scheduled notebook execution"
    ]
}

print("Reproducibility Features:")
for category, features in reproducibility_features.items():
    print(f"\n{category}:")
    for feature in features:
        print(f"  • {feature}")

# Example: Parameterized notebook simulation
print("\nExample: Parameterized Analysis")
parameters = {
    "test_size": [0.2, 0.3],
    "random_state": [42, 123],
    "n_estimators": [50, 100]
}

parameter_results = {}
for test_size in parameters["test_size"]:
    for random_state in parameters["random_state"]:
        for n_estimators in parameters["n_estimators"]:
            # Simulate parameter sweep
            X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            model.fit(X_train_p, y_train_p)
            accuracy = model.score(X_test_p, y_test_p)
            
            param_key = f"test_{test_size}_state_{random_state}_est_{n_estimators}"
            parameter_results[param_key] = accuracy

print(f"Parameter sweep completed: {len(parameter_results)} combinations tested")
print(f"Best configuration: {max(parameter_results, key=parameter_results.get)}")
print(f"Best accuracy: {max(parameter_results.values()):.4f}")

print("\nBenefit: Reproducible and automated experiments")
print("✓ Consistent results across runs")
print("✓ Easy parameter sweeps")
print("✓ Automated report generation")
print()

# SUMMARY OF BENEFITS
print("=" * 60)
print("SUMMARY: Key Benefits of Jupyter Notebooks for ML")
print("=" * 60)

benefits_summary = {
    "Development Speed": [
        "Rapid prototyping and iteration",
        "Immediate feedback and visualization",
        "Interactive debugging and exploration"
    ],
    "Collaboration": [
        "Easy sharing and discussion",
        "Version control integration",
        "Stakeholder-friendly outputs"
    ],
    "Learning": [
        "Educational content creation",
        "Interactive demonstrations",
        "Step-by-step tutorials"
    ],
    "Documentation": [
        "Self-documenting analysis",
        "Narrative and code combination",
        "Reproducible research"
    ],
    "Flexibility": [
        "Multiple language support",
        "Rich media integration",
        "Extensible architecture"
    ],
    "Integration": [
        "ML ecosystem compatibility",
        "Cloud platform support",
        "Deployment pipeline integration"
    ]
}

for category, items in benefits_summary.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  ✓ {item}")

print("\n" + "=" * 60)
print("BEST PRACTICES FOR ML NOTEBOOKS")
print("=" * 60)

best_practices = [
    "Keep notebooks focused on specific tasks or experiments",
    "Use clear naming conventions for variables and functions",
    "Add markdown documentation for each analysis step",
    "Clean outputs before version control commits",
    "Restart and run all cells periodically to ensure reproducibility",
    "Use virtual environments for dependency management",
    "Separate data exploration from model training notebooks",
    "Export important functions to .py modules for reuse",
    "Use parameterization for repeatable experiments",
    "Include data source documentation and assumptions"
]

for i, practice in enumerate(best_practices, 1):
    print(f"{i:2d}. {practice}")

print("\n" + "=" * 60)
print("COMMON PITFALLS AND SOLUTIONS")
print("=" * 60)

pitfalls = {
    "Hidden State Issues": {
        "problem": "Cells executed out of order create inconsistent state",
        "solution": "Regularly restart kernel and run all cells"
    },
    "Version Control Challenges": {
        "problem": "JSON format and outputs cause merge conflicts",
        "solution": "Use nbstripout and .gitignore for outputs"
    },
    "Lack of Structure": {
        "problem": "Notebooks become messy and hard to follow",
        "solution": "Follow consistent structure and modularize code"
    },
    "Debugging Difficulties": {
        "problem": "Hard to debug complex workflows in notebooks",
        "solution": "Extract complex logic to modules with proper testing"
    },
    "Production Deployment": {
        "problem": "Notebooks not suitable for production deployment",
        "solution": "Convert to scripts or use notebook execution tools"
    }
}

for pitfall, details in pitfalls.items():
    print(f"\n{pitfall}:")
    print(f"  Problem: {details['problem']}")
    print(f"  Solution: {details['solution']}")

print("\n=== Jupyter Notebooks: Essential Tool for ML Development ===")
```

### Explanation

Jupyter Notebooks provide a revolutionary environment for machine learning development by combining code execution, visualization, documentation, and collaboration in a single interactive platform.

### Key Benefits

1. **Interactive Development**
   - Immediate feedback on code execution
   - Real-time data exploration and visualization
   - Iterative model development and testing

2. **Enhanced Collaboration**
   - Easy sharing of complete analyses
   - Version control integration
   - Rich output preservation for stakeholders

3. **Educational Value**
   - Self-documenting code and results
   - Step-by-step learning materials
   - Interactive algorithm demonstrations

4. **Rapid Prototyping**
   - Quick experimentation cycles
   - Easy parameter tuning and comparison
   - Immediate visualization of results

5. **Comprehensive Documentation**
   - Combines narrative, code, and results
   - Creates reproducible research documents
   - Facilitates knowledge transfer

### Use Cases in ML

- **Data Exploration**: Interactive analysis of datasets with immediate visualization
- **Model Development**: Rapid prototyping and comparison of different algorithms
- **Experiment Tracking**: Document and share experimental results
- **Education**: Create tutorials and learning materials
- **Presentation**: Share findings with stakeholders in accessible format
- **Collaboration**: Team-based model development and review

### Best Practices

- **Structure**: Organize notebooks with clear sections and documentation
- **Reproducibility**: Use consistent environments and random seeds
- **Version Control**: Clean outputs before commits, use proper .gitignore
- **Modularization**: Extract reusable code to separate modules
- **Documentation**: Add markdown explanations for complex analyses

### Integration with ML Workflow

- **Data Pipeline**: Seamless integration with pandas, numpy, and data tools
- **Model Training**: Direct access to scikit-learn, TensorFlow, PyTorch
- **Visualization**: Rich plotting with matplotlib, seaborn, plotly
- **Deployment**: Easy conversion to production scripts or APIs
- **Monitoring**: Integration with experiment tracking tools like MLflow

### Limitations and Solutions

- **Production Deployment**: Convert to scripts using nbconvert or papermill
- **Version Control**: Use tools like nbstripout to manage outputs
- **Debugging**: Extract complex logic to testable modules
- **Performance**: Use profiling tools and optimize critical sections
- **Scalability**: Integrate with distributed computing frameworks

Jupyter Notebooks have transformed machine learning development by providing an interactive, collaborative, and educational environment that accelerates the entire ML lifecycle from exploration to deployment.

**Answer:** _[To be filled]_

---

## Question 4

**Discuss the use of pipelines in Scikit-learn for streamlining preprocessing steps.**

### Theory
Scikit-learn pipelines are powerful tools that chain together multiple preprocessing steps and machine learning algorithms into a single, cohesive workflow. They ensure data transformations are applied consistently across training and testing phases, prevent data leakage, and make machine learning workflows more maintainable and reproducible.

### Answer

```python
# sklearn_pipelines.py - Comprehensive demonstration of Scikit-learn pipelines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder, OrdinalEncoder,
    PolynomialFeatures, FunctionTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, validation_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

print("=== Scikit-learn Pipelines for ML Preprocessing ===\n")

# Create sample dataset with mixed data types
def create_mixed_dataset():
    """Create a mixed dataset with numerical and categorical features"""
    np.random.seed(42)
    n_samples = 1000
    
    # Numerical features
    numerical_data = np.random.randn(n_samples, 4)
    numerical_data[:, 0] *= 10  # Different scales
    numerical_data[:, 1] += 5
    numerical_data[:, 2] *= 0.1
    
    # Categorical features
    categories_1 = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    categories_2 = np.random.choice(['Type1', 'Type2', 'Type3'], n_samples)
    
    # Introduce missing values
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    numerical_data[missing_indices[:50], 1] = np.nan
    categories_1[missing_indices[50:]] = None
    
    # Create DataFrame
    df = pd.DataFrame({
        'numeric_1': numerical_data[:, 0],
        'numeric_2': numerical_data[:, 1],
        'numeric_3': numerical_data[:, 2],
        'numeric_4': numerical_data[:, 3],
        'category_1': categories_1,
        'category_2': categories_2
    })
    
    # Create target variable
    target = (
        (df['numeric_1'] > 0).astype(int) + 
        (df['category_1'] == 'A').astype(int) + 
        np.random.binomial(1, 0.3, n_samples)
    ) % 3
    
    return df, target

# Create dataset
df, y = create_mixed_dataset()
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Features: {list(df.columns)}")
print(f"Target classes: {sorted(set(y))}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Data types:\n{df.dtypes}")
print()

# 1. BASIC PIPELINE CONSTRUCTION
print("1. BASIC PIPELINE CONSTRUCTION")
print("-" * 40)

# Simple numerical pipeline
numerical_features = ['numeric_1', 'numeric_2', 'numeric_3', 'numeric_4']
categorical_features = ['category_1', 'category_2']

# Basic numerical preprocessing pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

print("Basic Numerical Pipeline:")
print("Steps:")
for i, (name, transformer) in enumerate(numerical_pipeline.steps, 1):
    print(f"  {i}. {name}: {transformer.__class__.__name__}")

# Demonstrate pipeline fitting and transformation
X_numerical = df[numerical_features]
X_train_num, X_test_num = train_test_split(X_numerical, test_size=0.2, random_state=42)

# Fit and transform
X_train_processed = numerical_pipeline.fit_transform(X_train_num)
X_test_processed = numerical_pipeline.transform(X_test_num)

print(f"\nOriginal training data shape: {X_train_num.shape}")
print(f"Processed training data shape: {X_train_processed.shape}")
print(f"Original data range: [{X_train_num.min().min():.2f}, {X_train_num.max().max():.2f}]")
print(f"Processed data range: [{X_train_processed.min():.2f}, {X_train_processed.max():.2f}]")
print()

# 2. COLUMN TRANSFORMER FOR MIXED DATA TYPES
print("2. COLUMN TRANSFORMER FOR MIXED DATA TYPES")
print("-" * 50)

# Define preprocessing for different column types
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', sparse=False))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

print("Column Transformer Configuration:")
print("Numerical features preprocessing:")
for i, (name, transformer) in enumerate(numerical_transformer.steps, 1):
    print(f"  {i}. {name}: {transformer.__class__.__name__}")

print("Categorical features preprocessing:")
for i, (name, transformer) in enumerate(categorical_transformer.steps, 1):
    print(f"  {i}. {name}: {transformer.__class__.__name__}")

# Apply preprocessing
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

print(f"\nOriginal data shape: {X_train.shape}")
print(f"Preprocessed data shape: {X_train_preprocessed.shape}")

# Get feature names after preprocessing
try:
    feature_names = (
        numerical_features + 
        list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
    )
    print(f"Feature names after preprocessing: {feature_names[:10]}...")  # Show first 10
except:
    print("Feature names not available (older sklearn version)")
print()

# 3. COMPLETE ML PIPELINE WITH MODEL
print("3. COMPLETE ML PIPELINE WITH MODEL")
print("-" * 45)

# Create complete pipeline: preprocessing + model
complete_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Complete Pipeline Structure:")
for i, (name, step) in enumerate(complete_pipeline.steps, 1):
    print(f"  {i}. {name}: {step.__class__.__name__}")

# Train the complete pipeline
complete_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = complete_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nPipeline Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Classification Report:")
print(classification_report(y_test, y_pred))

print("Benefits of Complete Pipeline:")
print("✓ Single fit/predict interface")
print("✓ Consistent preprocessing across train/test")
print("✓ Prevents data leakage")
print("✓ Easy to save and load entire workflow")
print()

# 4. ADVANCED PIPELINE FEATURES
print("4. ADVANCED PIPELINE FEATURES")
print("-" * 35)

# Pipeline with feature selection and dimensionality reduction
advanced_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_classif, k=8)),
    ('pca', PCA(n_components=5)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

print("Advanced Pipeline with Feature Engineering:")
for i, (name, step) in enumerate(advanced_pipeline.steps, 1):
    print(f"  {i}. {name}: {step.__class__.__name__}")

# Train and evaluate
advanced_pipeline.fit(X_train, y_train)
y_pred_advanced = advanced_pipeline.predict(X_test)
accuracy_advanced = accuracy_score(y_test, y_pred_advanced)

print(f"\nAdvanced Pipeline Performance:")
print(f"Accuracy: {accuracy_advanced:.4f}")

# Access intermediate steps
print(f"\nPipeline Inspection:")
print(f"Features after preprocessing: {X_train_preprocessed.shape[1]}")
print(f"Features after selection: {advanced_pipeline.named_steps['feature_selection'].k}")
print(f"Features after PCA: {advanced_pipeline.named_steps['pca'].n_components}")

# Feature importance from selection step
if hasattr(advanced_pipeline.named_steps['feature_selection'], 'scores_'):
    feature_scores = advanced_pipeline.named_steps['feature_selection'].scores_
    print(f"Top feature scores: {np.sort(feature_scores)[-5:]}")
print()

# 5. PIPELINE WITH CUSTOM TRANSFORMERS
print("5. PIPELINE WITH CUSTOM TRANSFORMERS")
print("-" * 40)

# Custom transformer example
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for log transformation"""
    
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if self.columns:
            for col in self.columns:
                X_copy[col] = np.log1p(np.abs(X_copy[col]))
        return X_copy

class OutlierCapper(BaseEstimator, TransformerMixin):
    """Custom transformer for outlier capping"""
    
    def __init__(self, quantile_range=(0.05, 0.95)):
        self.quantile_range = quantile_range
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        self.lower_bounds_ = np.percentile(X, self.quantile_range[0] * 100, axis=0)
        self.upper_bounds_ = np.percentile(X, self.quantile_range[1] * 100, axis=0)
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy = np.clip(X_copy, self.lower_bounds_, self.upper_bounds_)
        return X_copy

# Pipeline with custom transformers
custom_numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('outlier_capper', OutlierCapper()),
    ('log_transform', FunctionTransformer(np.log1p, validate=False)),
    ('scaler', RobustScaler())
])

custom_preprocessor = ColumnTransformer([
    ('num', custom_numerical_pipeline, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

custom_pipeline = Pipeline([
    ('preprocessor', custom_preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

print("Custom Pipeline with Advanced Preprocessing:")
for i, (name, step) in enumerate(custom_numerical_pipeline.steps, 1):
    print(f"  Numerical step {i}: {name}")

# Train and evaluate
custom_pipeline.fit(X_train, y_train)
y_pred_custom = custom_pipeline.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)

print(f"\nCustom Pipeline Performance:")
print(f"Accuracy: {accuracy_custom:.4f}")
print()

# 6. HYPERPARAMETER TUNING WITH PIPELINES
print("6. HYPERPARAMETER TUNING WITH PIPELINES")
print("-" * 45)

# Define parameter grid for pipeline
param_grid = {
    'preprocessor__num__imputer__strategy': ['median', 'mean'],
    'preprocessor__num__scaler': [StandardScaler(), RobustScaler()],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None]
}

# Create pipeline for tuning
tuning_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

print("Hyperparameter Tuning Configuration:")
print("Parameters to tune:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

# Grid search with cross-validation
grid_search = GridSearchCV(
    tuning_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit grid search (using subset for speed)
X_train_subset = X_train.iloc[:500]
y_train_subset = y_train[:500]

print(f"\nRunning grid search on subset ({len(X_train_subset)} samples)...")
grid_search.fit(X_train_subset, y_train_subset)

print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"Best parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate best pipeline on test set
best_pipeline = grid_search.best_estimator_
best_pipeline.fit(X_train, y_train)  # Retrain on full training set
y_pred_best = best_pipeline.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Best pipeline test accuracy: {accuracy_best:.4f}")
print()

# 7. PIPELINE PERSISTENCE AND DEPLOYMENT
print("7. PIPELINE PERSISTENCE AND DEPLOYMENT")
print("-" * 45)

import joblib
import pickle
from pathlib import Path

# Save pipeline
pipeline_path = "best_ml_pipeline.pkl"
joblib.dump(best_pipeline, pipeline_path)
print(f"Pipeline saved to: {pipeline_path}")

# Load and use pipeline
loaded_pipeline = joblib.load(pipeline_path)
print(f"Pipeline loaded successfully")

# Demonstrate prediction with loaded pipeline
sample_data = X_test.iloc[:5]
predictions = loaded_pipeline.predict(sample_data)
probabilities = loaded_pipeline.predict_proba(sample_data)

print(f"\nSample Predictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"  Sample {i+1}: Class {pred}, Probabilities: {prob}")

# Pipeline inspection
print(f"\nPipeline Structure:")
for name, step in loaded_pipeline.steps:
    print(f"  {name}: {step.__class__.__name__}")

print("Benefits of Pipeline Persistence:")
print("✓ Complete workflow saved as single object")
print("✓ Preprocessing and model parameters preserved")
print("✓ Easy deployment to production")
print("✓ Version control for entire ML workflow")
print()

# 8. PIPELINE COMPARISON AND ANALYSIS
print("8. PIPELINE COMPARISON AND ANALYSIS")
print("-" * 40)

# Define multiple pipeline configurations
pipelines = {
    'Basic': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'Advanced': Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k=8)),
        ('classifier', SVC(probability=True, random_state=42))
    ]),
    'Custom': custom_pipeline
}

# Compare pipeline performances
pipeline_results = {}
print("Pipeline Comparison:")

for name, pipeline in pipelines.items():
    # Cross-validation scores
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
    # Fit and test
    pipeline.fit(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)
    
    pipeline_results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_accuracy
    }
    
    print(f"\n{name} Pipeline:")
    print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")

# Visualize pipeline comparison
plt.figure(figsize=(12, 8))

# Subplot 1: CV scores comparison
plt.subplot(2, 2, 1)
names = list(pipeline_results.keys())
cv_means = [pipeline_results[name]['cv_mean'] for name in names]
cv_stds = [pipeline_results[name]['cv_std'] for name in names]

plt.errorbar(range(len(names)), cv_means, yerr=cv_stds, fmt='o-', capsize=5)
plt.xticks(range(len(names)), names, rotation=45)
plt.ylabel('Cross-Validation Accuracy')
plt.title('Pipeline CV Performance Comparison')
plt.grid(True, alpha=0.3)

# Subplot 2: Test accuracy comparison
plt.subplot(2, 2, 2)
test_accuracies = [pipeline_results[name]['test_accuracy'] for name in names]
bars = plt.bar(names, test_accuracies, alpha=0.7)
plt.ylabel('Test Accuracy')
plt.title('Pipeline Test Performance')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Color code bars by performance
for bar, acc in zip(bars, test_accuracies):
    if acc == max(test_accuracies):
        bar.set_color('green')
    elif acc == min(test_accuracies):
        bar.set_color('red')
    else:
        bar.set_color('blue')

# Subplot 3: Feature importance (for tree-based models)
plt.subplot(2, 2, 3)
rf_pipeline = pipelines['Random Forest']
if hasattr(rf_pipeline.named_steps['classifier'], 'feature_importances_'):
    importances = rf_pipeline.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10 features
    
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances (Random Forest)')

# Subplot 4: Confusion matrix for best pipeline
plt.subplot(2, 2, 4)
best_pipeline_name = max(pipeline_results.keys(), key=lambda x: pipeline_results[x]['test_accuracy'])
best_pipeline = pipelines[best_pipeline_name]
y_pred = best_pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title(f'Confusion Matrix - {best_pipeline_name}')

plt.tight_layout()
plt.show()

print(f"\nBest performing pipeline: {best_pipeline_name}")
print(f"Best test accuracy: {pipeline_results[best_pipeline_name]['test_accuracy']:.4f}")
print()

# 9. DEBUGGING AND PIPELINE INSPECTION
print("9. DEBUGGING AND PIPELINE INSPECTION")
print("-" * 40)

# Debugging utilities for pipelines
def inspect_pipeline_data_flow(pipeline, X_sample, step_name=None):
    """Inspect data flow through pipeline steps"""
    print("Pipeline Data Flow Inspection:")
    
    current_data = X_sample.copy()
    print(f"Input shape: {current_data.shape}")
    
    for i, (name, transformer) in enumerate(pipeline.steps):
        if step_name and name != step_name and i < len(pipeline.steps) - 1:
            current_data = transformer.transform(current_data)
            continue
            
        if hasattr(transformer, 'transform'):
            current_data = transformer.transform(current_data)
            print(f"After {name}: shape {current_data.shape}")
            
            if hasattr(current_data, 'dtype'):
                print(f"  Data type: {current_data.dtype}")
            if hasattr(current_data, 'min') and hasattr(current_data, 'max'):
                print(f"  Value range: [{current_data.min():.3f}, {current_data.max():.3f}]")
        
        if step_name and name == step_name:
            break
    
    return current_data

# Inspect a sample pipeline
sample_data = X_train.iloc[:10]
print("Inspecting Random Forest pipeline:")
inspect_pipeline_data_flow(pipelines['Random Forest'], sample_data)

# Pipeline step access
print(f"\nPipeline Step Access:")
rf_pipeline = pipelines['Random Forest']
print(f"Preprocessor: {rf_pipeline.named_steps['preprocessor']}")
print(f"Classifier: {rf_pipeline.named_steps['classifier']}")

# Get intermediate results
preprocessed_sample = rf_pipeline.named_steps['preprocessor'].transform(sample_data)
print(f"Preprocessed data shape: {preprocessed_sample.shape}")
print()

# 10. BEST PRACTICES AND COMMON PATTERNS
print("10. BEST PRACTICES AND COMMON PATTERNS")
print("-" * 45)

best_practices = {
    "Design Principles": [
        "Keep preprocessing and modeling in same pipeline",
        "Use ColumnTransformer for mixed data types",
        "Apply transformations consistently across splits",
        "Avoid data leakage by fitting only on training data"
    ],
    "Error Prevention": [
        "Always use pipeline.fit() on training data only",
        "Use pipeline.transform() or pipeline.predict() on test data",
        "Validate pipeline with cross-validation",
        "Check for data leakage in preprocessing steps"
    ],
    "Performance Optimization": [
        "Use memory-efficient transformers when possible",
        "Cache intermediate results for expensive computations",
        "Use n_jobs=-1 for parallel processing where available",
        "Consider using sparse matrices for large datasets"
    ],
    "Maintenance": [
        "Document pipeline steps and rationale",
        "Version control pipeline configurations",
        "Test pipelines with different data scenarios",
        "Monitor pipeline performance in production"
    ]
}

print("Scikit-learn Pipeline Best Practices:")
for category, practices in best_practices.items():
    print(f"\n{category}:")
    for practice in practices:
        print(f"  ✓ {practice}")

# Common pipeline patterns
print(f"\nCommon Pipeline Patterns:")

patterns = {
    "Basic Preprocessing": "impute → scale → model",
    "Feature Engineering": "impute → scale → feature_selection → model",
    "Advanced Pipeline": "impute → scale → polynomial → pca → model",
    "Text Processing": "vectorize → tfidf → feature_selection → model",
    "Mixed Data": "column_transformer → feature_selection → model"
}

for pattern_name, pattern_flow in patterns.items():
    print(f"  {pattern_name}: {pattern_flow}")

print(f"\n{'='*60}")
print("SUMMARY: Benefits of Scikit-learn Pipelines")
print(f"{'='*60}")

benefits = [
    "Prevents data leakage by ensuring consistent preprocessing",
    "Simplifies model deployment with single object persistence",
    "Enables easy hyperparameter tuning across entire workflow",
    "Improves code maintainability and reproducibility",
    "Facilitates A/B testing of different preprocessing strategies",
    "Provides clean API for complex multi-step transformations",
    "Integrates seamlessly with scikit-learn's ecosystem",
    "Supports custom transformers for domain-specific preprocessing"
]

for i, benefit in enumerate(benefits, 1):
    print(f"{i:2d}. {benefit}")

print(f"\n{'='*60}")
print("COMMON PITFALLS AND SOLUTIONS")
print(f"{'='*60}")

pitfalls = {
    "Data Leakage": {
        "problem": "Fitting preprocessing on entire dataset",
        "solution": "Always fit pipeline only on training data"
    },
    "Inconsistent Preprocessing": {
        "problem": "Different preprocessing for train/test",
        "solution": "Use same pipeline for all data transformations"
    },
    "Memory Issues": {
        "problem": "Large intermediate matrices in pipeline",
        "solution": "Use sparse matrices and memory-efficient transformers"
    },
    "Debug Difficulties": {
        "problem": "Hard to inspect intermediate pipeline steps",
        "solution": "Use pipeline inspection utilities and logging"
    },
    "Parameter Naming": {
        "problem": "Complex parameter names in grid search",
        "solution": "Use clear step names and understand naming convention"
    }
}

for pitfall, details in pitfalls.items():
    print(f"\n{pitfall}:")
    print(f"  Problem: {details['problem']}")
    print(f"  Solution: {details['solution']}")

# Clean up saved files
import os
if os.path.exists(pipeline_path):
    os.remove(pipeline_path)
    print(f"\nCleaned up: {pipeline_path}")

print(f"\n=== Scikit-learn Pipelines: Essential for Production ML ===")
```

### Explanation

Scikit-learn pipelines provide a robust framework for creating maintainable, reproducible, and leak-free machine learning workflows by chaining preprocessing steps and models into unified objects.

### Key Benefits

1. **Data Leakage Prevention**
   - Ensures preprocessing fits only on training data
   - Consistent transformations across train/test splits
   - Automatic parameter isolation between splits

2. **Workflow Simplification**
   - Single object encapsulates entire ML workflow
   - Unified fit/predict interface
   - Easy persistence and deployment

3. **Maintainability**
   - Clear separation of preprocessing and modeling steps
   - Easy to modify individual components
   - Improved code organization and readability

4. **Hyperparameter Tuning**
   - Grid search across entire pipeline
   - Optimize preprocessing and model parameters together
   - Cross-validation with complete workflow

### Core Components

- **Pipeline**: Sequential chaining of transformers and estimators
- **ColumnTransformer**: Apply different preprocessing to different columns
- **make_pipeline**: Simplified pipeline creation with automatic naming
- **Custom Transformers**: Domain-specific preprocessing components

### Common Pipeline Patterns

1. **Basic Pattern**: Imputation → Scaling → Model
2. **Feature Engineering**: Preprocessing → Feature Selection → Model  
3. **Mixed Data**: ColumnTransformer → Feature Engineering → Model
4. **Advanced**: Multiple preprocessing steps → Dimensionality Reduction → Model

### Use Cases in ML

- **Data Preprocessing**: Consistent imputation, scaling, encoding
- **Feature Engineering**: Selection, creation, transformation
- **Model Comparison**: Fair comparison with identical preprocessing
- **Production Deployment**: Single object with complete workflow
- **Hyperparameter Optimization**: Tuning entire pipeline together

### Best Practices

- **Fit Discipline**: Only fit on training data, transform on test
- **Component Isolation**: Separate preprocessing from modeling concerns
- **Documentation**: Clear naming and documentation of pipeline steps
- **Testing**: Validate pipelines with cross-validation
- **Version Control**: Track pipeline configurations and changes

### Advanced Features

- **Custom Transformers**: Domain-specific preprocessing logic
- **Pipeline Inspection**: Debug data flow through steps
- **Memory Optimization**: Efficient handling of large datasets
- **Parallel Processing**: Leverage multi-core processing capabilities

### Integration Benefits

- **Scikit-learn Ecosystem**: Seamless integration with all sklearn tools
- **Model Selection**: Works with GridSearchCV, RandomizedSearchCV
- **Metrics**: Compatible with all evaluation metrics
- **Persistence**: Easy saving/loading with joblib or pickle

Scikit-learn pipelines are essential for building robust, maintainable machine learning systems that prevent common pitfalls like data leakage while providing clean, professional workflows suitable for production deployment.

**Answer:** _[To be filled]_

---

## Question 5

**Discuss how ensemble methods work and give an example where they might be useful.**

### Theory
Ensemble methods combine predictions from multiple machine learning models to create a stronger predictor than any individual model alone. They work on the principle that aggregating diverse models can reduce overfitting, improve generalization, and increase robustness. The key insight is that while individual models may make different types of errors, combining them can cancel out these errors and lead to better overall performance.

### Answer

```python
# ensemble_methods.py - Comprehensive demonstration of ensemble methods
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

# Ensemble methods
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    StackingClassifier, StackingRegressor
)

# Base models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings('ignore')

print("=== Ensemble Methods in Machine Learning ===\n")

# Create datasets for demonstration
def create_datasets():
    """Create classification and regression datasets"""
    # Classification dataset
    X_class, y_class = make_classification(
        n_samples=1000, n_features=20, n_informative=10, 
        n_redundant=5, n_clusters_per_class=1, random_state=42
    )
    
    # Regression dataset
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=20, noise=0.1, random_state=42
    )
    
    return X_class, y_class, X_reg, y_reg

X_class, y_class, X_reg, y_reg = create_datasets()

# Split datasets
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print("Dataset Information:")
print(f"Classification: {X_class.shape[0]} samples, {X_class.shape[1]} features, {len(np.unique(y_class))} classes")
print(f"Regression: {X_reg.shape[0]} samples, {X_reg.shape[1]} features")
print()

# 1. BAGGING METHODS
print("1. BAGGING METHODS (Bootstrap Aggregating)")
print("-" * 50)

print("Theory: Bagging trains multiple models on different bootstrap samples")
print("of the training data and averages their predictions.")
print()

# Random Forest (Advanced Bagging)
print("Random Forest - Enhanced Bagging with Feature Randomness:")

# Compare individual tree vs Random Forest
single_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
single_tree.fit(X_train_c, y_train_c)
random_forest.fit(X_train_c, y_train_c)

# Evaluate
single_tree_score = single_tree.score(X_test_c, y_test_c)
rf_score = random_forest.score(X_test_c, y_test_c)

print(f"Single Decision Tree Accuracy: {single_tree_score:.4f}")
print(f"Random Forest Accuracy: {rf_score:.4f}")
print(f"Improvement: {rf_score - single_tree_score:.4f}")

# Basic Bagging Classifier
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)
bagging.fit(X_train_c, y_train_c)
bagging_score = bagging.score(X_test_c, y_test_c)

print(f"Bagging Classifier Accuracy: {bagging_score:.4f}")

# Demonstrate variance reduction
print("\nVariance Reduction Demonstration:")
n_trials = 10
single_scores = []
rf_scores = []

for trial in range(n_trials):
    # Create slightly different training sets
    X_trial, _, y_trial, _ = train_test_split(
        X_train_c, y_train_c, test_size=0.1, random_state=trial
    )
    
    # Train models
    tree = DecisionTreeClassifier(random_state=42)
    forest = RandomForestClassifier(n_estimators=50, random_state=42)
    
    tree.fit(X_trial, y_trial)
    forest.fit(X_trial, y_trial)
    
    single_scores.append(tree.score(X_test_c, y_test_c))
    rf_scores.append(forest.score(X_test_c, y_test_c))

print(f"Single Tree - Mean: {np.mean(single_scores):.4f}, Std: {np.std(single_scores):.4f}")
print(f"Random Forest - Mean: {np.mean(rf_scores):.4f}, Std: {np.std(rf_scores):.4f}")
print(f"Variance Reduction: {np.std(single_scores) - np.std(rf_scores):.4f}")
print()

# 2. BOOSTING METHODS
print("2. BOOSTING METHODS (Sequential Learning)")
print("-" * 45)

print("Theory: Boosting trains models sequentially, with each model")
print("learning from the mistakes of previous models.")
print()

# AdaBoost
print("AdaBoost - Adaptive Boosting:")
ada_boost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
ada_boost.fit(X_train_c, y_train_c)
ada_score = ada_boost.score(X_test_c, y_test_c)

print(f"AdaBoost Accuracy: {ada_score:.4f}")

# Gradient Boosting
print("Gradient Boosting - Gradient-based Error Correction:")
gb_classifier = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_classifier.fit(X_train_c, y_train_c)
gb_score = gb_classifier.score(X_test_c, y_test_c)

print(f"Gradient Boosting Accuracy: {gb_score:.4f}")

# Demonstrate sequential improvement
print("\nBoosting Sequential Improvement:")
# Track performance as estimators are added
n_estimators_range = range(1, 101, 10)
ada_scores = []
gb_scores = []

for n_est in n_estimators_range:
    # AdaBoost
    ada_temp = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_est,
        random_state=42
    )
    ada_temp.fit(X_train_c, y_train_c)
    ada_scores.append(ada_temp.score(X_test_c, y_test_c))
    
    # Gradient Boosting
    gb_temp = GradientBoostingClassifier(
        n_estimators=n_est,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_temp.fit(X_train_c, y_train_c)
    gb_scores.append(gb_temp.score(X_test_c, y_test_c))

# Plot boosting improvement
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(n_estimators_range, ada_scores, 'b-', label='AdaBoost', marker='o')
plt.plot(n_estimators_range, gb_scores, 'r-', label='Gradient Boosting', marker='s')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Boosting Performance vs Number of Estimators')
plt.legend()
plt.grid(True, alpha=0.3)

print(f"Final AdaBoost improvement: {ada_scores[-1] - ada_scores[0]:.4f}")
print(f"Final Gradient Boosting improvement: {gb_scores[-1] - gb_scores[0]:.4f}")
print()

# 3. VOTING METHODS
print("3. VOTING METHODS (Model Combination)")
print("-" * 40)

print("Theory: Voting combines predictions from different types of models")
print("using either majority voting (hard) or probability averaging (soft).")
print()

# Create diverse base models
base_models = [
    ('logistic', LogisticRegression(random_state=42, max_iter=1000)),
    ('tree', DecisionTreeClassifier(random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('nb', GaussianNB()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Hard Voting
hard_voting = VotingClassifier(
    estimators=base_models,
    voting='hard'
)
hard_voting.fit(X_train_c, y_train_c)
hard_score = hard_voting.score(X_test_c, y_test_c)

# Soft Voting
soft_voting = VotingClassifier(
    estimators=base_models,
    voting='soft'
)
soft_voting.fit(X_train_c, y_train_c)
soft_score = soft_voting.score(X_test_c, y_test_c)

print("Voting Classifier Results:")
print(f"Hard Voting Accuracy: {hard_score:.4f}")
print(f"Soft Voting Accuracy: {soft_score:.4f}")

# Compare with individual models
print("\nIndividual Model Performance:")
individual_scores = {}
for name, model in base_models:
    model.fit(X_train_c, y_train_c)
    score = model.score(X_test_c, y_test_c)
    individual_scores[name] = score
    print(f"{name.capitalize()}: {score:.4f}")

print(f"\nBest individual model: {max(individual_scores.values()):.4f}")
print(f"Voting improvement over best individual: {max(hard_score, soft_score) - max(individual_scores.values()):.4f}")
print()

# 4. STACKING METHODS
print("4. STACKING METHODS (Meta-learning)")
print("-" * 38)

print("Theory: Stacking uses a meta-learner to combine predictions")
print("from multiple base models in an optimal way.")
print()

# Create stacking classifier
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

meta_learner = LogisticRegression(random_state=42)

stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,  # Cross-validation for meta-features
    stack_method='predict_proba'
)

stacking.fit(X_train_c, y_train_c)
stacking_score = stacking.score(X_test_c, y_test_c)

print(f"Stacking Classifier Accuracy: {stacking_score:.4f}")

# Compare base learner performance
print("\nBase Learner Performance in Stacking:")
for name, model in base_learners:
    model.fit(X_train_c, y_train_c)
    score = model.score(X_test_c, y_test_c)
    print(f"{name.upper()}: {score:.4f}")

print(f"Meta-learner improvement: {stacking_score - max([model.score(X_test_c, y_test_c) for _, model in base_learners]):.4f}")
print()

# 5. REAL-WORLD EXAMPLE: MEDICAL DIAGNOSIS
print("5. REAL-WORLD EXAMPLE: MEDICAL DIAGNOSIS")
print("-" * 45)

# Use breast cancer dataset for realistic medical scenario
cancer_data = load_breast_cancer()
X_cancer, y_cancer = cancer_data.data, cancer_data.target

# Split and scale data
X_train_med, X_test_med, y_train_med, y_test_med = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

scaler = StandardScaler()
X_train_med_scaled = scaler.fit_transform(X_train_med)
X_test_med_scaled = scaler.transform(X_test_med)

print("Medical Diagnosis Scenario: Breast Cancer Detection")
print(f"Dataset: {X_cancer.shape[0]} patients, {X_cancer.shape[1]} features")
print(f"Classes: {cancer_data.target_names}")
print(f"Class distribution: {np.bincount(y_cancer)}")
print()

# Create medical ensemble
medical_ensemble = {
    'Individual Models': {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    },
    'Ensemble Models': {
        'Voting (Soft)': VotingClassifier([
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ], voting='soft'),
        'Stacking': StackingClassifier([
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ], final_estimator=LogisticRegression(random_state=42), cv=5)
    }
}

# Evaluate medical models
medical_results = {}
print("Medical Diagnosis Model Performance:")

for category, models in medical_ensemble.items():
    print(f"\n{category}:")
    for name, model in models.items():
        # Train model
        model.fit(X_train_med_scaled, y_train_med)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_med_scaled, y_train_med, cv=5)
        
        # Test performance
        y_pred = model.predict(X_test_med_scaled)
        test_accuracy = accuracy_score(y_test_med, y_pred)
        
        # Store results
        medical_results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'predictions': y_pred
        }
        
        print(f"  {name}:")
        print(f"    CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"    Test Accuracy: {test_accuracy:.4f}")

# Analyze ensemble benefits for medical diagnosis
best_individual = max([v['test_accuracy'] for k, v in medical_results.items() 
                      if k in medical_ensemble['Individual Models']])
best_ensemble = max([v['test_accuracy'] for k, v in medical_results.items() 
                    if k in medical_ensemble['Ensemble Models']])

print(f"\nMedical Diagnosis Ensemble Benefits:")
print(f"Best Individual Model: {best_individual:.4f}")
print(f"Best Ensemble Model: {best_ensemble:.4f}")
print(f"Ensemble Improvement: {best_ensemble - best_individual:.4f}")

# Clinical significance
improvement_percentage = (best_ensemble - best_individual) / best_individual * 100
print(f"Relative Improvement: {improvement_percentage:.2f}%")

if improvement_percentage > 1:
    print("✓ Clinically significant improvement")
    print("✓ Reduced false negative rate")
    print("✓ Enhanced diagnostic confidence")
else:
    print("• Marginal improvement")
    print("• Still valuable for risk reduction")

# Confusion matrix comparison
plt.subplot(2, 3, 2)
best_individual_name = max(medical_ensemble['Individual Models'], 
                          key=lambda x: medical_results[x]['test_accuracy'])
cm_individual = confusion_matrix(y_test_med, medical_results[best_individual_name]['predictions'])
sns.heatmap(cm_individual, annot=True, fmt='d', cmap='Blues')
plt.title(f'Best Individual: {best_individual_name}')
plt.ylabel('True')
plt.xlabel('Predicted')

plt.subplot(2, 3, 3)
best_ensemble_name = max(medical_ensemble['Ensemble Models'], 
                        key=lambda x: medical_results[x]['test_accuracy'])
cm_ensemble = confusion_matrix(y_test_med, medical_results[best_ensemble_name]['predictions'])
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Greens')
plt.title(f'Best Ensemble: {best_ensemble_name}')
plt.ylabel('True')
plt.xlabel('Predicted')

print()

# 6. ENSEMBLE DIVERSITY ANALYSIS
print("6. ENSEMBLE DIVERSITY ANALYSIS")
print("-" * 35)

print("Theory: Ensemble effectiveness depends on model diversity.")
print("More diverse models lead to better ensemble performance.")
print()

# Analyze prediction diversity
def calculate_diversity(predictions_list):
    """Calculate pairwise diversity between model predictions"""
    n_models = len(predictions_list)
    diversity_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                # Calculate disagreement rate
                disagreement = np.mean(predictions_list[i] != predictions_list[j])
                diversity_matrix[i, j] = disagreement
    
    return diversity_matrix

# Get predictions from different models for diversity analysis
diversity_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
    'SVM': SVC(random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Train models and get predictions
model_predictions = {}
model_accuracies = {}

for name, model in diversity_models.items():
    model.fit(X_train_med_scaled, y_train_med)
    predictions = model.predict(X_test_med_scaled)
    model_predictions[name] = predictions
    model_accuracies[name] = accuracy_score(y_test_med, predictions)

# Calculate diversity
prediction_arrays = list(model_predictions.values())
model_names = list(model_predictions.keys())
diversity_matrix = calculate_diversity(prediction_arrays)

# Visualize diversity
plt.subplot(2, 3, 4)
sns.heatmap(diversity_matrix, annot=True, fmt='.3f', 
           xticklabels=[name[:4] for name in model_names],
           yticklabels=[name[:4] for name in model_names],
           cmap='viridis')
plt.title('Model Diversity Matrix\n(Disagreement Rate)')

print("Model Diversity Analysis:")
avg_diversity = np.mean(diversity_matrix[diversity_matrix > 0])
print(f"Average pairwise diversity: {avg_diversity:.4f}")

# Find most and least diverse pairs
max_diversity_idx = np.unravel_index(np.argmax(diversity_matrix), diversity_matrix.shape)
min_diversity_idx = np.unravel_index(np.argmin(diversity_matrix[diversity_matrix > 0]), diversity_matrix.shape)

print(f"Most diverse pair: {model_names[max_diversity_idx[0]]} - {model_names[max_diversity_idx[1]]} ({diversity_matrix[max_diversity_idx]:.4f})")
print(f"Least diverse pair: {model_names[min_diversity_idx[0]]} - {model_names[min_diversity_idx[1]]} ({diversity_matrix[min_diversity_idx]:.4f})")
print()

# 7. REGRESSION ENSEMBLE EXAMPLE
print("7. REGRESSION ENSEMBLE EXAMPLE")
print("-" * 35)

print("Ensemble methods also work for regression tasks")
print("Example: Predicting house prices with multiple models")
print()

# Regression ensemble
regression_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Voting Ensemble': VotingRegressor([
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42))
    ])
}

# Train and evaluate regression models
print("Regression Model Performance (R² Score):")
regression_results = {}

for name, model in regression_models.items():
    # Train model
    model.fit(X_train_r, y_train_r)
    
    # Predict and evaluate
    y_pred_r = model.predict(X_test_r)
    r2 = r2_score(y_test_r, y_pred_r)
    mse = mean_squared_error(y_test_r, y_pred_r)
    
    regression_results[name] = {'r2': r2, 'mse': mse}
    print(f"  {name}: R² = {r2:.4f}, MSE = {mse:.2f}")

# Visualize regression performance
plt.subplot(2, 3, 5)
model_names_reg = list(regression_results.keys())
r2_scores = [regression_results[name]['r2'] for name in model_names_reg]

bars = plt.bar(range(len(model_names_reg)), r2_scores, alpha=0.7)
plt.xticks(range(len(model_names_reg)), [name[:4] for name in model_names_reg], rotation=45)
plt.ylabel('R² Score')
plt.title('Regression Model Comparison')
plt.grid(True, alpha=0.3)

# Highlight ensemble
for i, (bar, name) in enumerate(zip(bars, model_names_reg)):
    if 'Voting' in name:
        bar.set_color('red')
        bar.set_alpha(0.9)

best_r2 = max(r2_scores)
ensemble_r2 = regression_results['Voting Ensemble']['r2']
print(f"\nBest individual R²: {best_r2:.4f}")
print(f"Ensemble R²: {ensemble_r2:.4f}")
print(f"Ensemble vs best individual: {ensemble_r2 - max([r2 for name, r2 in [(k, v['r2']) for k, v in regression_results.items() if 'Voting' not in k]):.4f}")
print()

# 8. ENSEMBLE METHOD COMPARISON
print("8. ENSEMBLE METHOD COMPARISON")
print("-" * 35)

# Performance comparison across all ensemble types
ensemble_comparison = {
    'Bagging (Random Forest)': RandomForestClassifier(n_estimators=100, random_state=42),
    'Boosting (AdaBoost)': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Boosting (Gradient)': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Voting (Hard)': VotingClassifier([
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(random_state=42))
    ], voting='hard'),
    'Voting (Soft)': VotingClassifier([
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ], voting='soft'),
    'Stacking': StackingClassifier([
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
    ], final_estimator=LogisticRegression(random_state=42), cv=3)
}

# Compare all ensemble methods
print("Comprehensive Ensemble Comparison:")
ensemble_scores = {}

for name, model in ensemble_comparison.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_c, y_train_c, cv=5)
    
    # Test performance
    model.fit(X_train_c, y_train_c)
    test_score = model.score(X_test_c, y_test_c)
    
    ensemble_scores[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_score': test_score
    }
    
    print(f"  {name}:")
    print(f"    CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test: {test_score:.4f}")

# Final comparison visualization
plt.subplot(2, 3, 6)
method_names = list(ensemble_scores.keys())
test_scores = [ensemble_scores[name]['test_score'] for name in method_names]

bars = plt.bar(range(len(method_names)), test_scores, alpha=0.7)
plt.xticks(range(len(method_names)), [name.split(' ')[0] for name in method_names], rotation=45)
plt.ylabel('Test Accuracy')
plt.title('Ensemble Method Comparison')
plt.grid(True, alpha=0.3)

# Color code by method type
colors = ['blue', 'green', 'green', 'orange', 'orange', 'red']
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.tight_layout()
plt.show()

# Find best ensemble method
best_ensemble_method = max(ensemble_scores.keys(), key=lambda x: ensemble_scores[x]['test_score'])
best_score = ensemble_scores[best_ensemble_method]['test_score']

print(f"\nBest Ensemble Method: {best_ensemble_method}")
print(f"Best Score: {best_score:.4f}")
print()

# 9. WHEN TO USE ENSEMBLE METHODS
print("9. WHEN TO USE ENSEMBLE METHODS")
print("-" * 35)

use_cases = {
    "High-Stakes Decisions": {
        "examples": ["Medical diagnosis", "Financial risk assessment", "Safety-critical systems"],
        "benefit": "Reduced error rates and increased confidence"
    },
    "Noisy or Complex Data": {
        "examples": ["Image recognition", "Natural language processing", "Sensor data analysis"],
        "benefit": "Better handling of uncertainty and noise"
    },
    "Model Uncertainty": {
        "examples": ["Small datasets", "High-dimensional data", "Limited domain knowledge"],
        "benefit": "More robust predictions with uncertainty quantification"
    },
    "Competition/Benchmarks": {
        "examples": ["Kaggle competitions", "Academic benchmarks", "Industry challenges"],
        "benefit": "Maximum performance through model combination"
    },
    "Production Systems": {
        "examples": ["Recommendation systems", "Fraud detection", "Quality control"],
        "benefit": "Improved reliability and consistent performance"
    }
}

print("When to Use Ensemble Methods:")
for category, details in use_cases.items():
    print(f"\n{category}:")
    print(f"  Examples: {', '.join(details['examples'])}")
    print(f"  Benefit: {details['benefit']}")

print(f"\n{'='*60}")
print("SUMMARY: Ensemble Methods Benefits")
print(f"{'='*60}")

benefits = [
    "Improved accuracy through error reduction",
    "Better generalization and reduced overfitting", 
    "Increased robustness to noise and outliers",
    "Model uncertainty quantification",
    "Reduced variance in predictions",
    "Enhanced performance on complex datasets",
    "Protection against individual model failures",
    "Flexibility to combine different algorithm types"
]

for i, benefit in enumerate(benefits, 1):
    print(f"{i:2d}. {benefit}")

print(f"\n{'='*60}")
print("ENSEMBLE METHOD SELECTION GUIDE")
print(f"{'='*60}")

selection_guide = {
    "Use Bagging When": [
        "High variance models (e.g., decision trees)",
        "Sufficient training data available",
        "Want to reduce overfitting",
        "Parallel training is possible"
    ],
    "Use Boosting When": [
        "High bias models (e.g., weak learners)",
        "Want to reduce bias and variance",
        "Have time for sequential training",
        "Data is not too noisy"
    ],
    "Use Voting When": [
        "Have diverse, well-performing models",
        "Models make different types of errors",
        "Want simple combination strategy",
        "Models are already trained"
    ],
    "Use Stacking When": [
        "Have expertise to design meta-learner",
        "Want optimal model combination",
        "Have sufficient data for meta-learning",
        "Performance is critical"
    ]
}

for method, guidelines in selection_guide.items():
    print(f"\n{method}:")
    for guideline in guidelines:
        print(f"  • {guideline}")

print(f"\n{'='*60}")
print("PRACTICAL CONSIDERATIONS")
print(f"{'='*60}")

considerations = {
    "Computational Cost": "Ensembles require more resources for training and prediction",
    "Interpretability": "Individual model insights may be lost in ensemble",
    "Overfitting Risk": "Complex ensembles can overfit, especially with small datasets",
    "Model Diversity": "Ensure base models are sufficiently different",
    "Cross-Validation": "Use proper CV to avoid overfitting in ensemble construction",
    "Production Deployment": "Consider inference time and memory requirements"
}

for consideration, description in considerations.items():
    print(f"\n{consideration}:")
    print(f"  {description}")

print(f"\n=== Ensemble Methods: Power of Model Combination ===")
```

### Explanation

Ensemble methods combine multiple machine learning models to create a stronger predictor than any individual model alone. They work by leveraging the diversity of different models to reduce errors and improve generalization.

### Core Ensemble Types

1. **Bagging (Bootstrap Aggregating)**
   - Trains multiple models on different bootstrap samples
   - Reduces variance and overfitting
   - Examples: Random Forest, Extra Trees

2. **Boosting (Sequential Learning)**
   - Trains models sequentially, learning from previous errors
   - Reduces bias and variance
   - Examples: AdaBoost, Gradient Boosting

3. **Voting (Model Combination)**
   - Combines predictions from diverse models
   - Hard voting: majority vote, Soft voting: probability averaging
   - Works best with diverse, well-performing models

4. **Stacking (Meta-learning)**
   - Uses meta-learner to optimally combine base model predictions
   - Learns how to best weight different models
   - Most sophisticated but requires careful validation

### Key Benefits

- **Error Reduction**: Different models make different types of errors
- **Improved Generalization**: Better performance on unseen data
- **Robustness**: Less sensitive to noise and outliers
- **Uncertainty Quantification**: Provides confidence estimates
- **Model Diversity**: Combines strengths of different algorithms

### Real-World Example: Medical Diagnosis

In medical diagnosis, ensemble methods are particularly valuable because:
- **High Stakes**: Misdiagnosis has serious consequences
- **Complex Data**: Medical data is often noisy and high-dimensional
- **Expert Consensus**: Mirrors medical practice of seeking second opinions
- **Confidence Measures**: Provides uncertainty estimates for critical decisions

### Use Cases Where Ensembles Excel

1. **High-Stakes Applications**: Finance, healthcare, safety systems
2. **Competition Scenarios**: Kaggle competitions, benchmarks
3. **Complex Data**: Images, text, sensor data
4. **Production Systems**: Recommendation engines, fraud detection
5. **Uncertain Domains**: Limited data or domain knowledge

### Selection Guidelines

- **Bagging**: Use with high-variance models (decision trees)
- **Boosting**: Use with high-bias models (weak learners)
- **Voting**: Use with diverse, already-trained models
- **Stacking**: Use when performance is critical and you have expertise

### Best Practices

- **Ensure Diversity**: Use different algorithms or training strategies
- **Proper Validation**: Use cross-validation to avoid overfitting
- **Computational Efficiency**: Balance performance gains with resource costs
- **Interpretability**: Consider if individual model insights are needed
- **Production Considerations**: Account for inference time and memory

### Practical Considerations

- **Computational Cost**: Ensembles require more resources
- **Model Maintenance**: Multiple models to monitor and update
- **Complexity**: More difficult to debug and interpret
- **Diminishing Returns**: Too many models may not improve performance

Ensemble methods represent one of the most powerful techniques in machine learning, consistently winning competitions and improving real-world applications by harnessing the collective intelligence of multiple models.

**Answer:** _[To be filled]_

---

## Question 6

**How would you assess amodel’s performance? Mention at least threemetrics.**

**Answer:** To assess model performance comprehensively, I use at least three complementary metrics:

1. **Accuracy**: Overall correctness (correct predictions / total predictions)
2. **Precision**: Reduces false positives (TP / [TP + FP])  
3. **Recall**: Reduces false negatives (TP / [TP + FN])

Additional essential metrics include F1-Score (balances precision/recall) and ROC-AUC (ranking quality). For regression: MSE/RMSE, MAE, and R².

Key principles: Choose metrics aligned with business goals, use cross-validation for robust estimates, consider class imbalance, and always compare against meaningful baselines.

```python
# Model Performance Assessment Example
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Cross-validation for robust evaluation
cv_f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print(f"CV F1 Mean: {cv_f1_scores.mean():.4f} ± {cv_f1_scores.std():.4f}")
```

---

## Question 7

**Discuss the differences between supervised and unsupervised learning evaluation.**

### Theory
Supervised and unsupervised learning require fundamentally different evaluation approaches because they have different objectives. Supervised learning has labeled data and measurable performance targets, while unsupervised learning discovers hidden patterns without ground truth labels.

### Answer

```python
# supervised_vs_unsupervised_evaluation.py - Comprehensive comparison of evaluation methods
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs, load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

# Supervised Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Unsupervised Learning Models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

# Evaluation Metrics
from sklearn.metrics import (
    # Supervised metrics
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score,
    # Unsupervised metrics
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)

import warnings
warnings.filterwarnings('ignore')

print("=== Supervised vs Unsupervised Learning Evaluation ===\n")

# Create datasets for demonstration
def create_evaluation_datasets():
    """Create datasets for supervised and unsupervised learning evaluation"""
    
    # Supervised classification dataset
    X_class, y_class = make_classification(
        n_samples=1000, n_features=10, n_informative=5, 
        n_redundant=2, n_clusters_per_class=1, random_state=42
    )
    
    # Unsupervised clustering dataset (with known clusters for evaluation)
    X_cluster, y_cluster = make_blobs(
        n_samples=1000, centers=4, n_features=10, 
        cluster_std=1.5, random_state=42
    )
    
    # Real dataset for comprehensive analysis
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    return X_class, y_class, X_cluster, y_cluster, X_iris, y_iris

X_class, y_class, X_cluster, y_cluster, X_iris, y_iris = create_evaluation_datasets()

print("Dataset Information:")
print(f"Supervised Dataset: {X_class.shape[0]} samples, {X_class.shape[1]} features, {len(np.unique(y_class))} classes")
print(f"Unsupervised Dataset: {X_cluster.shape[0]} samples, {X_cluster.shape[1]} features, {len(np.unique(y_cluster))} true clusters")
print(f"Iris Dataset: {X_iris.shape[0]} samples, {X_iris.shape[1]} features, {len(np.unique(y_iris))} classes")
print()

# 1. SUPERVISED LEARNING EVALUATION
print("1. SUPERVISED LEARNING EVALUATION")
print("-" * 40)

print("Theory: Supervised learning evaluation compares predictions against known labels")
print("using metrics that measure prediction accuracy, precision, and recall.")
print()

# Split supervised data
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Train supervised models
supervised_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

supervised_results = {}
print("Supervised Learning Evaluation Metrics:")

for name, model in supervised_models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # ROC-AUC for binary/multiclass
    if y_prob is not None and len(np.unique(y_class)) == 2:
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted') if y_prob is not None else None
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    supervised_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Supervised evaluation characteristics
print(f"\nSupervised Evaluation Characteristics:")
print("✓ Ground truth labels available")
print("✓ Direct performance measurement")
print("✓ Clear success/failure criteria")
print("✓ Standard train/validation/test splits")
print("✓ Cross-validation for robustness")
print("✓ Business-relevant metrics selection")
print()

# 2. UNSUPERVISED LEARNING EVALUATION
print("2. UNSUPERVISED LEARNING EVALUATION")
print("-" * 42)

print("Theory: Unsupervised learning evaluation uses internal quality measures")
print("and external validation when true labels are available for comparison.")
print()

# Standardize data for clustering
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Train unsupervised models
unsupervised_models = {
    'K-Means (k=4)': KMeans(n_clusters=4, random_state=42, n_init=10),
    'K-Means (k=3)': KMeans(n_clusters=3, random_state=42, n_init=10),
    'K-Means (k=5)': KMeans(n_clusters=5, random_state=42, n_init=10),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Agglomerative': AgglomerativeClustering(n_clusters=4),
    'Gaussian Mixture': GaussianMixture(n_components=4, random_state=42)
}

unsupervised_results = {}
print("Unsupervised Learning Evaluation Metrics:")

for name, model in unsupervised_models.items():
    # Fit model and get cluster labels
    cluster_labels = model.fit_predict(X_cluster_scaled)
    
    # Handle noise points in DBSCAN
    if name == 'DBSCAN':
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        print(f"\n{name}:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        
        if n_clusters > 1:
            # Filter out noise points for silhouette score
            mask = cluster_labels != -1
            if np.sum(mask) > 1:
                silhouette = silhouette_score(X_cluster_scaled[mask], cluster_labels[mask])
                calinski = calinski_harabasz_score(X_cluster_scaled[mask], cluster_labels[mask])
                davies_bouldin = davies_bouldin_score(X_cluster_scaled[mask], cluster_labels[mask])
            else:
                silhouette = calinski = davies_bouldin = None
        else:
            silhouette = calinski = davies_bouldin = None
    else:
        # Internal validation metrics (don't require true labels)
        silhouette = silhouette_score(X_cluster_scaled, cluster_labels)
        calinski = calinski_harabasz_score(X_cluster_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_cluster_scaled, cluster_labels)
        
        print(f"\n{name}:")
    
    # External validation metrics (compare with true labels when available)
    if 'DBSCAN' not in name or (n_clusters > 1 and np.sum(mask) > 1):
        if name == 'DBSCAN':
            ari = adjusted_rand_score(y_cluster[mask], cluster_labels[mask])
            nmi = normalized_mutual_info_score(y_cluster[mask], cluster_labels[mask])
            homogeneity = homogeneity_score(y_cluster[mask], cluster_labels[mask])
            completeness = completeness_score(y_cluster[mask], cluster_labels[mask])
            v_measure = v_measure_score(y_cluster[mask], cluster_labels[mask])
        else:
            ari = adjusted_rand_score(y_cluster, cluster_labels)
            nmi = normalized_mutual_info_score(y_cluster, cluster_labels)
            homogeneity = homogeneity_score(y_cluster, cluster_labels)
            completeness = completeness_score(y_cluster, cluster_labels)
            v_measure = v_measure_score(y_cluster, cluster_labels)
    else:
        ari = nmi = homogeneity = completeness = v_measure = None
    
    # Store results
    unsupervised_results[name] = {
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies_bouldin,
        'adjusted_rand_score': ari,
        'normalized_mutual_info': nmi,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'cluster_labels': cluster_labels
    }
    
    # Print metrics
    if silhouette is not None:
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better)")
        print(f"  Calinski-Harabasz: {calinski:.2f} (higher is better)")
        print(f"  Davies-Bouldin: {davies_bouldin:.4f} (lower is better)")
    
    if ari is not None:
        print(f"  Adjusted Rand Index: {ari:.4f} (higher is better)")
        print(f"  Normalized Mutual Info: {nmi:.4f} (higher is better)")
        print(f"  Homogeneity: {homogeneity:.4f}")
        print(f"  Completeness: {completeness:.4f}")
        print(f"  V-Measure: {v_measure:.4f}")

# Unsupervised evaluation characteristics
print(f"\nUnsupervised Evaluation Characteristics:")
print("✓ No ground truth labels available")
print("✓ Internal quality measures (cohesion, separation)")
print("✓ External validation when true labels exist")
print("✓ Domain knowledge and interpretability important")
print("✓ Multiple metrics needed for comprehensive assessment")
print("✓ Visual inspection often crucial")
print()

# 3. KEY DIFFERENCES IN EVALUATION APPROACHES
print("3. KEY DIFFERENCES IN EVALUATION APPROACHES")
print("-" * 50)

differences = {
    "Evaluation Basis": {
        "Supervised": "Comparison with known correct answers (ground truth)",
        "Unsupervised": "Internal structure quality and pattern discovery"
    },
    "Primary Metrics": {
        "Supervised": "Accuracy, Precision, Recall, F1-Score, ROC-AUC",
        "Unsupervised": "Silhouette Score, Calinski-Harabasz, Davies-Bouldin"
    },
    "Validation Strategy": {
        "Supervised": "Train/Validation/Test splits, Cross-validation",
        "Unsupervised": "Internal validation, External validation when possible"
    },
    "Success Criteria": {
        "Supervised": "High prediction accuracy on unseen data",
        "Unsupervised": "Meaningful patterns, cluster quality, interpretability"
    },
    "Objective Function": {
        "Supervised": "Minimize prediction error",
        "Unsupervised": "Optimize cluster quality or data representation"
    },
    "Evaluation Complexity": {
        "Supervised": "Straightforward with clear metrics",
        "Unsupervised": "More subjective, requires domain expertise"
    }
}

print("Comprehensive Comparison:")
for aspect, comparison in differences.items():
    print(f"\n{aspect}:")
    print(f"  Supervised: {comparison['Supervised']}")
    print(f"  Unsupervised: {comparison['Unsupervised']}")

print()

# 4. EVALUATION METRICS DEEP DIVE
print("4. EVALUATION METRICS DEEP DIVE")
print("-" * 35)

# Visualize evaluation comparison
plt.figure(figsize=(16, 12))

# Supervised metrics comparison
plt.subplot(2, 3, 1)
supervised_metrics = ['accuracy', 'precision', 'recall', 'f1']
models = list(supervised_results.keys())
metrics_data = [[supervised_results[model][metric] for model in models] for metric in supervised_metrics]

x = np.arange(len(models))
width = 0.2
for i, (metric, data) in enumerate(zip(supervised_metrics, metrics_data)):
    plt.bar(x + i*width, data, width, label=metric.capitalize(), alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Supervised Learning Metrics Comparison')
plt.xticks(x + width*1.5, [m[:8] for m in models], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Unsupervised internal metrics
plt.subplot(2, 3, 2)
internal_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
unsup_models = [name for name in unsupervised_results.keys() if 'DBSCAN' not in name]

# Normalize metrics for comparison (different scales)
normalized_data = []
for metric in internal_metrics:
    values = [unsupervised_results[model][metric] for model in unsup_models]
    if metric == 'davies_bouldin':
        # Lower is better, so invert
        normalized_values = [(max(values) - v) / (max(values) - min(values)) for v in values]
    else:
        # Higher is better
        normalized_values = [(v - min(values)) / (max(values) - min(values)) for v in values]
    normalized_data.append(normalized_values)

x = np.arange(len(unsup_models))
for i, (metric, data) in enumerate(zip(internal_metrics, normalized_data)):
    plt.bar(x + i*width, data, width, label=metric.replace('_', ' ').title(), alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Normalized Score')
plt.title('Unsupervised Internal Metrics (Normalized)')
plt.xticks(x + width, [m[:8] for m in unsup_models], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Unsupervised external metrics
plt.subplot(2, 3, 3)
external_metrics = ['adjusted_rand_score', 'normalized_mutual_info', 'v_measure']
external_data = []
for metric in external_metrics:
    values = [unsupervised_results[model][metric] for model in unsup_models if unsupervised_results[model][metric] is not None]
    external_data.append(values)

x = np.arange(len(unsup_models))
for i, (metric, data) in enumerate(zip(external_metrics, external_data)):
    plt.bar(x + i*width, data, width, label=metric.replace('_', ' ').title(), alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Unsupervised External Metrics')
plt.xticks(x + width, [m[:8] for m in unsup_models], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion matrix for best supervised model
plt.subplot(2, 3, 4)
best_supervised = max(supervised_results.keys(), key=lambda x: supervised_results[x]['accuracy'])
best_predictions = supervised_results[best_supervised]['predictions']
cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_supervised}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Cluster visualization for best unsupervised model
plt.subplot(2, 3, 5)
best_unsupervised = max([k for k, v in unsupervised_results.items() if v['silhouette'] is not None], 
                       key=lambda x: unsupervised_results[x]['silhouette'])
best_clusters = unsupervised_results[best_unsupervised]['cluster_labels']

# Use PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster_scaled)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title(f'Clustering Visualization - {best_unsupervised}')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

# True clusters for comparison
plt.subplot(2, 3, 6)
scatter_true = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_cluster, cmap='viridis', alpha=0.7)
plt.colorbar(scatter_true)
plt.title('True Clusters')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

plt.tight_layout()
plt.show()

print()

# 5. REAL-WORLD EVALUATION SCENARIOS
print("5. REAL-WORLD EVALUATION SCENARIOS")
print("-" * 40)

# Iris dataset example - both supervised and unsupervised
print("Case Study: Iris Dataset - Dual Evaluation Approach")
print()

# Supervised approach on Iris
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

iris_supervised = RandomForestClassifier(n_estimators=100, random_state=42)
iris_supervised.fit(X_iris_train, y_iris_train)
iris_pred = iris_supervised.predict(X_iris_test)

iris_accuracy = accuracy_score(y_iris_test, iris_pred)
iris_f1 = f1_score(y_iris_test, iris_pred, average='weighted')

print("Supervised Evaluation (Iris):")
print(f"  Accuracy: {iris_accuracy:.4f}")
print(f"  F1-Score: {iris_f1:.4f}")
print(f"  Classification Report:")
print(classification_report(y_iris_test, iris_pred, target_names=['Setosa', 'Versicolor', 'Virginica']))

# Unsupervised approach on Iris (pretend we don't know the labels)
iris_scaler = StandardScaler()
X_iris_scaled = iris_scaler.fit_transform(X_iris)

iris_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
iris_clusters = iris_kmeans.fit_predict(X_iris_scaled)

# Internal metrics
iris_silhouette = silhouette_score(X_iris_scaled, iris_clusters)
iris_calinski = calinski_harabasz_score(X_iris_scaled, iris_clusters)
iris_davies = davies_bouldin_score(X_iris_scaled, iris_clusters)

# External metrics (we can calculate these because we know true labels)
iris_ari = adjusted_rand_score(y_iris, iris_clusters)
iris_nmi = normalized_mutual_info_score(y_iris, iris_clusters)

print(f"\nUnsupervised Evaluation (Iris):")
print(f"  Silhouette Score: {iris_silhouette:.4f}")
print(f"  Calinski-Harabasz: {iris_calinski:.2f}")
print(f"  Davies-Bouldin: {iris_davies:.4f}")
print(f"  Adjusted Rand Index: {iris_ari:.4f}")
print(f"  Normalized Mutual Info: {iris_nmi:.4f}")

# Compare findings
print(f"\nComparison of Approaches:")
print(f"  Supervised: Clear performance metrics with {iris_accuracy:.4f} accuracy")
print(f"  Unsupervised: Discovered {len(np.unique(iris_clusters))} clusters with {iris_silhouette:.4f} silhouette score")
print(f"  Agreement: ARI of {iris_ari:.4f} shows {'good' if iris_ari > 0.7 else 'moderate' if iris_ari > 0.5 else 'poor'} agreement")
print()

# 6. EVALUATION CHALLENGES AND SOLUTIONS
print("6. EVALUATION CHALLENGES AND SOLUTIONS")
print("-" * 45)

challenges = {
    "Supervised Learning Challenges": {
        "Class Imbalance": {
            "Problem": "Accuracy misleading with imbalanced classes",
            "Solution": "Use precision, recall, F1-score, balanced accuracy"
        },
        "Overfitting": {
            "Problem": "High training accuracy, poor test performance",
            "Solution": "Cross-validation, learning curves, validation sets"
        },
        "Metric Selection": {
            "Problem": "Choosing appropriate metrics for business goals",
            "Solution": "Domain expertise, cost-sensitive evaluation"
        },
        "Data Leakage": {
            "Problem": "Information from future in training data",
            "Solution": "Proper temporal splits, careful feature engineering"
        }
    },
    "Unsupervised Learning Challenges": {
        "No Ground Truth": {
            "Problem": "No clear definition of 'correct' result",
            "Solution": "Multiple metrics, domain expertise, visualization"
        },
        "Subjective Evaluation": {
            "Problem": "Success depends on interpretation",
            "Solution": "Business-relevant evaluation criteria"
        },
        "Parameter Selection": {
            "Problem": "Number of clusters, algorithm parameters",
            "Solution": "Grid search with internal metrics, elbow method"
        },
        "Scalability": {
            "Problem": "Some metrics computationally expensive",
            "Solution": "Sampling, approximate methods, efficient algorithms"
        }
    }
}

for learning_type, type_challenges in challenges.items():
    print(f"{learning_type}:")
    for challenge, details in type_challenges.items():
        print(f"  {challenge}:")
        print(f"    Problem: {details['Problem']}")
        print(f"    Solution: {details['Solution']}")
    print()

# 7. BEST PRACTICES FOR EACH APPROACH
print("7. BEST PRACTICES FOR EACH APPROACH")
print("-" * 40)

best_practices = {
    "Supervised Learning Evaluation": [
        "Use stratified train/validation/test splits",
        "Apply cross-validation for robust estimates",
        "Choose metrics aligned with business objectives",
        "Report multiple metrics for comprehensive view",
        "Check for data leakage and overfitting",
        "Use learning curves to diagnose model behavior",
        "Consider cost-sensitive evaluation when applicable",
        "Validate on truly unseen data"
    ],
    "Unsupervised Learning Evaluation": [
        "Use multiple internal validation metrics",
        "Apply external validation when ground truth available",
        "Visualize results for interpretability",
        "Consider domain expertise in evaluation",
        "Test stability across multiple runs",
        "Evaluate at different granularity levels",
        "Consider computational efficiency",
        "Document assumptions and limitations"
    ]
}

for approach, practices in best_practices.items():
    print(f"{approach}:")
    for i, practice in enumerate(practices, 1):
        print(f"  {i:2d}. {practice}")
    print()

# 8. HYBRID EVALUATION APPROACHES
print("8. HYBRID EVALUATION APPROACHES")
print("-" * 35)

print("Modern ML often combines supervised and unsupervised evaluation:")
print()

hybrid_approaches = {
    "Semi-Supervised Learning": {
        "Description": "Uses both labeled and unlabeled data",
        "Evaluation": "Combine supervised metrics on labeled data with unsupervised quality measures"
    },
    "Representation Learning": {
        "Description": "Learn features for downstream tasks",
        "Evaluation": "Intrinsic quality + extrinsic performance on supervised tasks"
    },
    "Anomaly Detection": {
        "Description": "Identify outliers or unusual patterns",
        "Evaluation": "When labels available: precision/recall; Otherwise: expert validation"
    },
    "Clustering for Classification": {
        "Description": "Use clustering to improve classification",
        "Evaluation": "Cluster quality + classification performance improvement"
    }
}

for approach, details in hybrid_approaches.items():
    print(f"{approach}:")
    print(f"  Description: {details['Description']}")
    print(f"  Evaluation: {details['Evaluation']}")
    print()

print(f"{'='*60}")
print("SUMMARY: Key Evaluation Differences")
print(f"{'='*60}")

summary_points = [
    "Supervised learning has clear success criteria (prediction accuracy)",
    "Unsupervised learning requires multiple evaluation perspectives",
    "Supervised metrics are objective and standardized",
    "Unsupervised metrics are more subjective and context-dependent", 
    "Cross-validation is standard for supervised learning",
    "Stability testing is crucial for unsupervised learning",
    "Supervised learning uses train/test splits",
    "Unsupervised learning often uses internal quality measures",
    "Business objectives drive supervised metric selection",
    "Domain expertise is critical for unsupervised evaluation",
    "Supervised learning has mature evaluation frameworks",
    "Unsupervised evaluation is still an active research area"
]

for i, point in enumerate(summary_points, 1):
    print(f"{i:2d}. {point}")

print(f"\n{'='*60}")
print("PRACTICAL RECOMMENDATIONS")
print(f"{'='*60}")

recommendations = {
    "For Supervised Learning": [
        "Always use proper train/validation/test splits",
        "Apply cross-validation for model selection",
        "Choose metrics that align with business goals",
        "Report confidence intervals for metrics",
        "Check for class imbalance and adjust metrics accordingly"
    ],
    "For Unsupervised Learning": [
        "Use multiple internal and external metrics",
        "Visualize results for interpretability",
        "Test stability across different random seeds",
        "Involve domain experts in evaluation",
        "Document evaluation criteria and assumptions"
    ],
    "For Both Approaches": [
        "Understand the problem context and requirements",
        "Consider computational constraints",
        "Document evaluation methodology thoroughly",
        "Validate findings with independent datasets when possible"
    ]
}

for category, recs in recommendations.items():
    print(f"\n{category}:")
    for rec in recs:
        print(f"  • {rec}")

print(f"\n=== Evaluation: The Foundation of ML Success ===")
```

### Explanation

Supervised and unsupervised learning require fundamentally different evaluation strategies due to their distinct objectives and available information.

### Key Differences

1. **Ground Truth Availability**
   - **Supervised**: Has labeled data for direct comparison
   - **Unsupervised**: No labels, must evaluate structure quality

2. **Evaluation Objectives**
   - **Supervised**: Measure prediction accuracy
   - **Unsupervised**: Assess pattern discovery and data structure

3. **Primary Metrics**
   - **Supervised**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - **Unsupervised**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin

4. **Validation Strategy**
   - **Supervised**: Train/validation/test splits, cross-validation
   - **Unsupervised**: Internal validation, stability testing

### Supervised Learning Evaluation

**Characteristics:**
- Clear success criteria with ground truth labels
- Standardized metrics for different problem types
- Robust validation through cross-validation
- Direct business impact measurement

**Key Metrics:**
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: MSE, RMSE, MAE, R², MAPE

**Best Practices:**
- Use stratified splits for classification
- Apply cross-validation for robust estimates
- Choose metrics aligned with business objectives
- Report multiple metrics for comprehensive view

### Unsupervised Learning Evaluation

**Characteristics:**
- No ground truth labels available
- Multiple evaluation perspectives needed
- Subjective interpretation often required
- Domain expertise crucial for validation

**Internal Metrics (No labels needed):**
- **Silhouette Score**: Measures cluster cohesion and separation
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Index**: Average similarity between clusters

**External Metrics (When true labels available):**
- **Adjusted Rand Index**: Measures similarity to true clustering
- **Normalized Mutual Information**: Information sharing between clusterings
- **V-Measure**: Harmonic mean of homogeneity and completeness

### Evaluation Challenges

**Supervised Learning:**
- Class imbalance affecting metric interpretation
- Overfitting detection and prevention
- Appropriate metric selection for business goals
- Data leakage prevention

**Unsupervised Learning:**
- No clear definition of "correct" results
- Subjective evaluation criteria
- Parameter selection (e.g., number of clusters)
- Computational scalability of metrics

### Best Practices

**Supervised Learning:**
- Use proper train/validation/test methodology
- Apply cross-validation for model selection
- Choose business-relevant metrics
- Report confidence intervals
- Check for data leakage and overfitting

**Unsupervised Learning:**
- Use multiple complementary metrics
- Visualize results for interpretability
- Test stability across random seeds
- Involve domain experts in evaluation
- Document evaluation assumptions

### Hybrid Approaches

Modern ML often combines both evaluation types:
- **Semi-supervised Learning**: Labeled + unlabeled data evaluation
- **Representation Learning**: Intrinsic quality + downstream task performance
- **Anomaly Detection**: Expert validation + statistical measures
- **Transfer Learning**: Source task + target task evaluation

### Practical Recommendations

1. **Understand Problem Context**: Choose evaluation approach based on available data and objectives
2. **Use Multiple Metrics**: No single metric tells the complete story
3. **Consider Computational Constraints**: Balance thoroughness with efficiency
4. **Document Methodology**: Ensure reproducible and interpretable evaluation
5. **Validate Findings**: Use independent datasets when possible

The choice between supervised and unsupervised evaluation fundamentally depends on your learning objective and available data, with each requiring specific methodologies and metrics for meaningful assessment.

---

## Question 8

**How would you approach feature selection in a large dataset?**

### Theory
Feature selection in large datasets is crucial for reducing dimensionality, improving model performance, and decreasing computational costs. It involves identifying the most relevant features while removing redundant, irrelevant, or noisy ones. The approach must be scalable and efficient for large datasets.

### Answer

```python
# feature_selection_large_datasets.py - Comprehensive feature selection strategies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# Feature Selection Methods
from sklearn.feature_selection import (
    # Univariate methods
    SelectKBest, SelectPercentile, f_classif, f_regression, chi2, mutual_info_classif,
    # Model-based methods
    SelectFromModel, RFE, RFECV,
    # Variance-based
    VarianceThreshold
)

# Models for feature importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Additional tools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import time
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("=== Feature Selection in Large Datasets ===\n")

# Create large synthetic dataset
def create_large_dataset():
    """Create a large synthetic dataset with various feature types"""
    np.random.seed(42)
    
    # Large classification dataset
    X_class, y_class = make_classification(
        n_samples=10000,
        n_features=1000,
        n_informative=50,
        n_redundant=20,
        n_clusters_per_class=1,
        flip_y=0.01,
        random_state=42
    )
    
    # Add some categorical features
    n_categorical = 100
    categorical_features = np.random.randint(0, 5, size=(X_class.shape[0], n_categorical))
    
    # Combine numerical and categorical
    X_combined = np.hstack([X_class, categorical_features])
    
    # Create feature names
    feature_names = (
        [f'numeric_{i}' for i in range(X_class.shape[1])] +
        [f'categorical_{i}' for i in range(n_categorical)]
    )
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X_combined, columns=feature_names)
    
    return df, y_class, feature_names

# Create dataset
X_large, y_large, feature_names = create_large_dataset()
print(f"Dataset Information:")
print(f"Samples: {X_large.shape[0]:,}")
print(f"Features: {X_large.shape[1]:,}")
print(f"Classes: {len(np.unique(y_large))}")
print(f"Memory usage: {X_large.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_large, y_large, test_size=0.2, random_state=42, stratify=y_large
)

print("1. PRELIMINARY DATA ANALYSIS")
print("-" * 35)

print("Theory: Before feature selection, understand data characteristics")
print("including missing values, variance, and basic statistics.")
print()

# Basic statistics
print("Data Quality Analysis:")
print(f"Missing values: {X_train.isnull().sum().sum()}")
print(f"Duplicate rows: {X_train.duplicated().sum()}")
print(f"Constant features: {(X_train.var() == 0).sum()}")
print(f"Near-zero variance features: {(X_train.var() < 0.01).sum()}")

# Memory and computational considerations
print(f"\nComputational Considerations:")
print(f"Training set size: {X_train.shape[0]:,} × {X_train.shape[1]:,}")
print(f"Approximate matrix operations: {X_train.shape[0] * X_train.shape[1] / 1e6:.1f}M elements")
print()

# 2. FILTER METHODS (Univariate Selection)
print("2. FILTER METHODS (Univariate Selection)")
print("-" * 45)

print("Theory: Filter methods evaluate features independently using")
print("statistical tests, fast and scalable for large datasets.")
print()

# Variance Threshold - Remove low variance features
print("A. Variance Threshold Method:")
start_time = time.time()

variance_selector = VarianceThreshold(threshold=0.01)
X_train_var = variance_selector.fit_transform(X_train)

var_time = time.time() - start_time
selected_features_var = variance_selector.get_support()
n_selected_var = selected_features_var.sum()

print(f"  Original features: {X_train.shape[1]:,}")
print(f"  After variance threshold: {n_selected_var:,}")
print(f"  Removed features: {X_train.shape[1] - n_selected_var:,}")
print(f"  Processing time: {var_time:.3f} seconds")

# Univariate Statistical Tests
print(f"\nB. Univariate Statistical Tests:")

# F-test for classification
start_time = time.time()
f_selector = SelectKBest(score_func=f_classif, k=100)
X_train_f = f_selector.fit_transform(X_train, y_train)
f_time = time.time() - start_time

print(f"  F-test (k=100):")
print(f"    Selected features: {X_train_f.shape[1]}")
print(f"    Processing time: {f_time:.3f} seconds")

# Get feature scores
f_scores = f_selector.scores_
f_pvalues = f_selector.pvalues_
selected_f_indices = f_selector.get_support(indices=True)

print(f"    Top 5 feature scores: {sorted(f_scores, reverse=True)[:5]}")

# Mutual Information
print(f"\n  Mutual Information:")
start_time = time.time()
mi_selector = SelectKBest(score_func=mutual_info_classif, k=100)
X_train_mi = mi_selector.fit_transform(X_train, y_train)
mi_time = time.time() - start_time

print(f"    Selected features: {X_train_mi.shape[1]}")
print(f"    Processing time: {mi_time:.3f} seconds")

# Chi-squared test (for categorical features)
print(f"\n  Chi-squared test (categorical features):")
# Apply to categorical features only (non-negative values)
categorical_mask = X_train.columns.str.contains('categorical')
X_categorical = X_train.loc[:, categorical_mask]

if X_categorical.shape[1] > 0:
    start_time = time.time()
    chi2_selector = SelectKBest(score_func=chi2, k=min(50, X_categorical.shape[1]))
    X_train_chi2 = chi2_selector.fit_transform(X_categorical, y_train)
    chi2_time = time.time() - start_time
    
    print(f"    Original categorical features: {X_categorical.shape[1]}")
    print(f"    Selected categorical features: {X_train_chi2.shape[1]}")
    print(f"    Processing time: {chi2_time:.3f} seconds")

print()

# 3. WRAPPER METHODS (Model-based Selection)
print("3. WRAPPER METHODS (Model-based Selection)")
print("-" * 45)

print("Theory: Wrapper methods use model performance to evaluate")
print("feature subsets, more accurate but computationally expensive.")
print()

# Recursive Feature Elimination (RFE)
print("A. Recursive Feature Elimination (RFE):")

# Use a fast estimator for RFE on large dataset
estimator = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')

# RFE with cross-validation (on subset for speed)
print("  RFE with Cross-Validation (on subset):")
subset_size = 5000  # Use subset for faster computation
subset_indices = np.random.choice(X_train.shape[0], subset_size, replace=False)
X_subset = X_train.iloc[subset_indices]
y_subset = y_train[subset_indices]

start_time = time.time()
# Use fewer features for demo
n_features_to_select = 50
rfecv = RFECV(
    estimator=estimator,
    step=10,  # Remove 10 features at a time
    cv=3,     # Reduced CV for speed
    scoring='accuracy',
    n_jobs=-1
)

# Fit on subset
X_subset_scaled = StandardScaler().fit_transform(X_subset)
rfecv.fit(X_subset_scaled, y_subset)
rfecv_time = time.time() - start_time

print(f"    Optimal number of features: {rfecv.n_features_}")
print(f"    Processing time: {rfecv_time:.3f} seconds")
print(f"    Best CV score: {rfecv.grid_scores_.max():.4f}")

# Simple RFE (faster)
print(f"\n  Simple RFE (k={n_features_to_select}):")
start_time = time.time()
rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=20)
rfe.fit(X_subset_scaled, y_subset)
simple_rfe_time = time.time() - start_time

print(f"    Selected features: {rfe.n_features_}")
print(f"    Processing time: {simple_rfe_time:.3f} seconds")
print()

# 4. EMBEDDED METHODS (Model-intrinsic Selection)
print("4. EMBEDDED METHODS (Model-intrinsic Selection)")
print("-" * 50)

print("Theory: Embedded methods perform feature selection during")
print("model training, balancing accuracy and efficiency.")
print()

# L1 Regularization (Lasso)
print("A. L1 Regularization (Lasso):")
start_time = time.time()

# Scale features for regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Lasso with cross-validation for optimal alpha
lasso = LassoCV(cv=3, random_state=42, n_jobs=-1, max_iter=1000)
lasso.fit(X_train_scaled, y_train)

# Select features based on non-zero coefficients
lasso_selector = SelectFromModel(lasso, threshold='median')
X_train_lasso = lasso_selector.fit_transform(X_train_scaled, y_train)

lasso_time = time.time() - start_time
n_selected_lasso = X_train_lasso.shape[1]

print(f"  Original features: {X_train.shape[1]:,}")
print(f"  Selected by Lasso: {n_selected_lasso:,}")
print(f"  Optimal alpha: {lasso.alpha_:.6f}")
print(f"  Processing time: {lasso_time:.3f} seconds")

# Tree-based feature importance
print(f"\nB. Tree-based Feature Importance:")

# Random Forest
start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Select features based on importance
rf_selector = SelectFromModel(rf, threshold='median')
X_train_rf = rf_selector.fit_transform(X_train, y_train)

rf_time = time.time() - start_time
n_selected_rf = X_train_rf.shape[1]

print(f"  Random Forest:")
print(f"    Selected features: {n_selected_rf:,}")
print(f"    Processing time: {rf_time:.3f} seconds")

# Extra Trees (faster than RF)
start_time = time.time()
et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
et.fit(X_train, y_train)

et_selector = SelectFromModel(et, threshold='median')
X_train_et = et_selector.fit_transform(X_train, y_train)

et_time = time.time() - start_time
n_selected_et = X_train_et.shape[1]

print(f"  Extra Trees:")
print(f"    Selected features: {n_selected_et:,}")
print(f"    Processing time: {et_time:.3f} seconds")

print()

# 5. HYBRID APPROACHES
print("5. HYBRID APPROACHES")
print("-" * 20)

print("Theory: Combine multiple methods for robust feature selection")
print("in large datasets, leveraging strengths of each approach.")
print()

# Sequential Feature Selection
print("A. Sequential Filter + Wrapper Approach:")

# Step 1: Filter - Reduce to manageable size
print("  Step 1: Statistical filtering")
start_time = time.time()

# Use F-test to select top 500 features
filter_selector = SelectKBest(score_func=f_classif, k=500)
X_train_filtered = filter_selector.fit_transform(X_train, y_train)

filter_step_time = time.time() - start_time
print(f"    Reduced from {X_train.shape[1]:,} to {X_train_filtered.shape[1]:,} features")
print(f"    Processing time: {filter_step_time:.3f} seconds")

# Step 2: Model-based refinement
print("  Step 2: Model-based refinement")
start_time = time.time()

# Apply Lasso on filtered features
X_filtered_scaled = StandardScaler().fit_transform(X_train_filtered)
lasso_refined = LassoCV(cv=3, random_state=42, max_iter=1000)
lasso_refined.fit(X_filtered_scaled, y_train)

refined_selector = SelectFromModel(lasso_refined, threshold='median')
X_train_hybrid = refined_selector.fit_transform(X_filtered_scaled, y_train)

hybrid_step_time = time.time() - start_time
total_hybrid_time = filter_step_time + hybrid_step_time

print(f"    Final features: {X_train_hybrid.shape[1]:,}")
print(f"    Processing time: {hybrid_step_time:.3f} seconds")
print(f"    Total hybrid time: {total_hybrid_time:.3f} seconds")

# Ensemble Feature Selection
print(f"\nB. Ensemble Feature Selection:")
print("  Combining multiple feature importance methods")

# Collect feature importance from multiple methods
feature_importance_methods = {
    'Random Forest': rf.feature_importances_,
    'Extra Trees': et.feature_importances_,
    'F-test': f_selector.scores_ / f_selector.scores_.max(),  # Normalize
    'Mutual Info': mi_selector.scores_ / mi_selector.scores_.max()  # Normalize
}

# Create ensemble score
ensemble_scores = np.zeros(X_train.shape[1])
for method, scores in feature_importance_methods.items():
    if len(scores) == X_train.shape[1]:
        ensemble_scores += scores / len(feature_importance_methods)

# Select top features based on ensemble score
top_k = 100
top_indices = np.argsort(ensemble_scores)[-top_k:]
X_train_ensemble = X_train.iloc[:, top_indices]

print(f"    Ensemble selected features: {X_train_ensemble.shape[1]}")
print(f"    Top ensemble scores: {sorted(ensemble_scores, reverse=True)[:5]}")

print()

# 6. PERFORMANCE EVALUATION
print("6. PERFORMANCE EVALUATION")
print("-" * 30)

print("Theory: Evaluate feature selection methods by comparing")
print("model performance and computational efficiency.")
print()

# Compare different feature selection results
feature_sets = {
    'Original': (X_train, X_test),
    'Variance Threshold': (X_train_var, variance_selector.transform(X_test)),
    'F-test (k=100)': (X_train_f, f_selector.transform(X_test)),
    'Lasso': (X_train_lasso, lasso_selector.transform(scaler.transform(X_test))),
    'Random Forest': (X_train_rf, rf_selector.transform(X_test)),
    'Hybrid': (X_train_hybrid, refined_selector.transform(filter_selector.transform(X_test))),
    'Ensemble': (X_train_ensemble, X_test.iloc[:, top_indices])
}

# Evaluate each feature set
results = {}
print("Performance Comparison:")

for name, (X_tr, X_te) in feature_sets.items():
    start_time = time.time()
    
    # Train simple model
    model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    
    # Handle scaling if needed
    if name in ['Lasso', 'Hybrid']:
        # Already scaled during selection
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
    else:
        # Scale for fair comparison
        scaler_temp = StandardScaler()
        X_tr_scaled = scaler_temp.fit_transform(X_tr)
        X_te_scaled = scaler_temp.transform(X_te)
        model.fit(X_tr_scaled, y_train)
        y_pred = model.predict(X_te_scaled)
    
    training_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'n_features': X_tr.shape[1],
        'accuracy': accuracy,
        'f1_score': f1,
        'training_time': training_time
    }
    
    print(f"  {name}:")
    print(f"    Features: {X_tr.shape[1]:,}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1-Score: {f1:.4f}")
    print(f"    Training time: {training_time:.3f}s")

print()

# 7. SCALABILITY ANALYSIS
print("7. SCALABILITY ANALYSIS")
print("-" * 25)

print("Theory: Analyze computational complexity and memory")
print("requirements for different feature selection methods.")
print()

# Method complexity analysis
complexity_analysis = {
    'Variance Threshold': {
        'Time Complexity': 'O(n × m)',
        'Space Complexity': 'O(m)',
        'Scalability': 'Excellent',
        'Best for': 'Very large datasets, preprocessing'
    },
    'F-test/Chi2': {
        'Time Complexity': 'O(n × m)',
        'Space Complexity': 'O(m)',
        'Scalability': 'Excellent',
        'Best for': 'Large datasets, quick filtering'
    },
    'Mutual Information': {
        'Time Complexity': 'O(n × m × log(n))',
        'Space Complexity': 'O(n × m)',
        'Scalability': 'Good',
        'Best for': 'Non-linear relationships'
    },
    'L1 Regularization': {
        'Time Complexity': 'O(iterations × n × m)',
        'Space Complexity': 'O(n × m)',
        'Scalability': 'Good',
        'Best for': 'Linear relationships, sparse solutions'
    },
    'Tree-based': {
        'Time Complexity': 'O(trees × n × m × log(n))',
        'Space Complexity': 'O(trees × m)',
        'Scalability': 'Moderate',
        'Best for': 'Non-linear relationships, mixed data types'
    },
    'RFE': {
        'Time Complexity': 'O(m² × model_complexity)',
        'Space Complexity': 'O(n × m)',
        'Scalability': 'Poor',
        'Best for': 'Small to medium datasets, optimal results'
    }
}

print("Scalability Analysis:")
for method, analysis in complexity_analysis.items():
    print(f"\n{method}:")
    for aspect, value in analysis.items():
        print(f"  {aspect}: {value}")

# Time comparison visualization
plt.figure(figsize=(15, 10))

# Method comparison
plt.subplot(2, 3, 1)
methods = list(results.keys())
accuracies = [results[method]['accuracy'] for method in methods]
n_features = [results[method]['n_features'] for method in methods]

scatter = plt.scatter(n_features, accuracies, s=100, alpha=0.7)
for i, method in enumerate(methods):
    plt.annotate(method[:8], (n_features[i], accuracies[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Features')
plt.grid(True, alpha=0.3)

# Training time comparison
plt.subplot(2, 3, 2)
training_times = [results[method]['training_time'] for method in methods]
bars = plt.bar(range(len(methods)), training_times, alpha=0.7)
plt.xticks(range(len(methods)), [m[:8] for m in methods], rotation=45)
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison')
plt.grid(True, alpha=0.3)

# Feature importance visualization
plt.subplot(2, 3, 3)
if len(rf.feature_importances_) <= 50:  # Only plot if manageable
    top_n = min(20, len(rf.feature_importances_))
    top_indices_rf = np.argsort(rf.feature_importances_)[-top_n:]
    plt.barh(range(top_n), rf.feature_importances_[top_indices_rf])
    plt.yticks(range(top_n), [f'F{i}' for i in top_indices_rf])
    plt.xlabel('Importance')
    plt.title('Top 20 Random Forest Feature Importances')
else:
    plt.text(0.5, 0.5, 'Too many features\nto display', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Feature Importance Distribution')

# Performance vs dimensionality
plt.subplot(2, 3, 4)
plt.plot(n_features, accuracies, 'o-', alpha=0.7)
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Performance vs Dimensionality')
plt.grid(True, alpha=0.3)

# Method efficiency comparison
plt.subplot(2, 3, 5)
efficiency = [results[method]['accuracy'] / np.log(results[method]['n_features'] + 1) 
              for method in methods]
bars = plt.bar(range(len(methods)), efficiency, alpha=0.7)
plt.xticks(range(len(methods)), [m[:8] for m in methods], rotation=45)
plt.ylabel('Efficiency Score')
plt.title('Accuracy / log(Features) Efficiency')
plt.grid(True, alpha=0.3)

# Cumulative feature importance
plt.subplot(2, 3, 6)
sorted_importance = np.sort(rf.feature_importances_)[::-1]
cumsum_importance = np.cumsum(sorted_importance)
plt.plot(range(1, len(cumsum_importance) + 1), cumsum_importance)
plt.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
plt.axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Feature Importance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print()

# 8. BEST PRACTICES FOR LARGE DATASETS
print("8. BEST PRACTICES FOR LARGE DATASETS")
print("-" * 40)

best_practices = {
    "Data Preprocessing": [
        "Remove constant and near-constant features first",
        "Handle missing values before feature selection",
        "Consider feature scaling for distance-based methods",
        "Use sparse matrix representations when possible"
    ],
    "Method Selection": [
        "Start with fast filter methods for initial reduction",
        "Use embedded methods for moderate-sized datasets", 
        "Apply wrapper methods only on pre-filtered features",
        "Consider computational resources and time constraints"
    ],
    "Implementation Strategy": [
        "Use batching for very large datasets",
        "Leverage parallel processing (n_jobs=-1)",
        "Monitor memory usage during processing",
        "Save intermediate results for reproducibility"
    ],
    "Validation": [
        "Use cross-validation to assess stability",
        "Test on independent validation set",
        "Compare against baseline (all features)",
        "Evaluate both performance and interpretability"
    ],
    "Production Considerations": [
        "Document feature selection pipeline",
        "Ensure reproducible random seeds",
        "Consider feature drift in production",
        "Plan for feature re-selection updates"
    ]
}

print("Best Practices for Large Dataset Feature Selection:")
for category, practices in best_practices.items():
    print(f"\n{category}:")
    for practice in practices:
        print(f"  • {practice}")

print()

# 9. RECOMMENDED WORKFLOW
print("9. RECOMMENDED WORKFLOW")
print("-" * 25)

workflow_steps = [
    {
        "Step": "1. Initial Assessment",
        "Actions": [
            "Analyze dataset size and computational constraints",
            "Check for missing values and data quality issues",
            "Identify feature types (numerical, categorical, text)"
        ]
    },
    {
        "Step": "2. Preprocessing",
        "Actions": [
            "Remove constant and near-constant features",
            "Handle missing values appropriately",
            "Encode categorical variables if needed"
        ]
    },
    {
        "Step": "3. Filter Methods",
        "Actions": [
            "Apply variance threshold for quick reduction",
            "Use statistical tests (F-test, chi-squared)",
            "Consider mutual information for non-linear relationships"
        ]
    },
    {
        "Step": "4. Embedded Methods",
        "Actions": [
            "Apply L1 regularization for linear relationships",
            "Use tree-based importance for non-linear patterns",
            "Consider computational trade-offs"
        ]
    },
    {
        "Step": "5. Validation",
        "Actions": [
            "Evaluate on validation set",
            "Compare multiple methods",
            "Consider ensemble approaches"
        ]
    },
    {
        "Step": "6. Final Selection",
        "Actions": [
            "Choose method based on performance and constraints",
            "Document selection criteria and rationale",
            "Prepare for production deployment"
        ]
    }
]

print("Recommended Feature Selection Workflow:")
for step_info in workflow_steps:
    print(f"\n{step_info['Step']}:")
    for action in step_info['Actions']:
        print(f"  • {action}")

print(f"\n{'='*60}")
print("SUMMARY: Feature Selection Strategy")
print(f"{'='*60}")

summary_points = [
    "Start with filter methods for rapid dimensionality reduction",
    "Use embedded methods for moderate-sized feature sets",
    "Apply wrapper methods sparingly on pre-filtered features",
    "Combine multiple approaches for robust selection",
    "Consider computational constraints and scalability",
    "Validate selection stability across different data splits",
    "Balance performance improvement with interpretability",
    "Document and version feature selection decisions",
    "Monitor feature relevance in production environments",
    "Plan for periodic re-evaluation and updates"
]

for i, point in enumerate(summary_points, 1):
    print(f"{i:2d}. {point}")

print(f"\n{'='*60}")
print("METHOD SELECTION GUIDE")
print(f"{'='*60}")

selection_guide = {
    "Use Filter Methods When": [
        "Dataset has > 10,000 features",
        "Need rapid dimensionality reduction",
        "Computational resources are limited",
        "Initial exploration phase"
    ],
    "Use Embedded Methods When": [
        "Features are 100-10,000 range",
        "Want feature selection during training",
        "Need regularization benefits",
        "Linear or tree-based models work well"
    ],
    "Use Wrapper Methods When": [
        "Features are < 100 after filtering",
        "Optimal performance is critical",
        "Have sufficient computational time",
        "Need model-specific selection"
    ],
    "Use Hybrid Approaches When": [
        "Want robust feature selection",
        "Have complex data relationships",
        "Need to balance speed and accuracy",
        "Multiple stakeholders need validation"
    ]
}

for approach, guidelines in selection_guide.items():
    print(f"\n{approach}:")
    for guideline in guidelines:
        print(f"  • {guideline}")

print(f"\n=== Feature Selection: Key to Scalable ML ===")
```

### Explanation

Feature selection in large datasets requires a strategic approach balancing computational efficiency, model performance, and interpretability. The key is using appropriate methods for dataset size and computational constraints.

### Core Strategies

1. **Filter Methods (Fast & Scalable)**
   - **Variance Threshold**: Remove low-variance features
   - **Statistical Tests**: F-test, Chi-squared, Mutual Information
   - **Correlation Analysis**: Remove highly correlated features
   - **Best for**: Initial rapid reduction, very large datasets

2. **Embedded Methods (Balanced)**
   - **L1 Regularization**: Lasso, Elastic Net for sparse solutions
   - **Tree-based Importance**: Random Forest, Extra Trees
   - **Linear Model Coefficients**: Feature weights from trained models
   - **Best for**: Moderate-sized datasets, integrated selection

3. **Wrapper Methods (Accurate but Expensive)**
   - **Recursive Feature Elimination**: Sequential feature removal
   - **Forward/Backward Selection**: Iterative feature addition/removal
   - **Best for**: Small feature sets, optimal performance needed

4. **Hybrid Approaches (Robust)**
   - **Sequential Selection**: Filter → Embedded → Wrapper
   - **Ensemble Methods**: Combine multiple selection strategies
   - **Best for**: Complex datasets, robust selection needed

### Recommended Workflow

1. **Initial Assessment**
   - Analyze dataset size and computational constraints
   - Check data quality and feature types
   - Set performance and time budgets

2. **Preprocessing**
   - Remove constant/near-constant features
   - Handle missing values
   - Scale features if needed

3. **Rapid Reduction (Filter Methods)**
   - Apply variance threshold
   - Use statistical tests for top features
   - Reduce to manageable size (< 1000 features)

4. **Refinement (Embedded Methods)**
   - Apply regularization methods
   - Use tree-based importance
   - Select final feature set

5. **Validation**
   - Cross-validate selection stability
   - Compare with baseline performance
   - Evaluate on independent test set

### Scalability Considerations

**Computational Complexity:**
- **Filter Methods**: O(n×m) - Linear with samples and features
- **Embedded Methods**: O(iterations×n×m) - Model-dependent
- **Wrapper Methods**: O(m²×model_complexity) - Quadratic in features

**Memory Requirements:**
- Use sparse matrices for large datasets
- Batch processing for memory-constrained environments
- Parallel processing for speed improvements

**Trade-offs:**
- **Speed vs Accuracy**: Filter methods fastest, wrapper methods most accurate
- **Interpretability vs Performance**: Simple methods more interpretable
- **Stability vs Optimality**: Ensemble methods more stable

### Best Practices

1. **Start Simple**: Begin with filter methods for rapid reduction
2. **Use Domain Knowledge**: Incorporate expert insights in selection
3. **Validate Thoroughly**: Use cross-validation and independent test sets
4. **Document Decisions**: Record selection criteria and rationale
5. **Monitor in Production**: Track feature relevance over time
6. **Plan for Updates**: Design pipelines for periodic re-selection

### Common Pitfalls

- **Overfitting**: Selecting features on test data
- **Leakage**: Using future information in feature selection
- **Instability**: Selections that vary dramatically with small data changes
- **Computational Overload**: Using expensive methods on very large datasets
- **Ignoring Domain Knowledge**: Purely statistical selection without context

### Tool Selection Guide

**For Datasets with:**
- **> 10,000 features**: Start with filter methods
- **1,000-10,000 features**: Use embedded methods
- **< 1,000 features**: Consider wrapper methods
- **Mixed data types**: Use tree-based methods
- **High correlation**: Apply correlation-based filtering
- **Non-linear relationships**: Use mutual information or tree-based selection

Feature selection is crucial for building efficient, interpretable, and high-performing models on large datasets, requiring a strategic approach that balances computational constraints with performance objectives.

---

## Question 9

**Discuss strategies for dealing with imbalanced datasets.**

### Theory
Imbalanced datasets occur when classes are not represented equally, which is common in real-world applications like fraud detection, medical diagnosis, and anomaly detection. Standard machine learning algorithms can be biased toward the majority class, leading to poor performance on minority classes that are often more important.

### Answer

```python
# imbalanced_datasets_strategies.py - Comprehensive approaches to handle class imbalance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Data generation and preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Imbalanced-learn library for sampling techniques
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
)
from imblearn.under_sampling import (
    RandomUnderSampler, EditedNearestNeighbours, RepeatedEditedNearestNeighbours,
    AllKNN, CondensedNearestNeighbour, OneSidedSelection, TomekLinks
)
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Models and metrics
from sklearn.ensemble import RandomForestClassifier, BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, average_precision_score
)

# Cost-sensitive learning
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings('ignore')

print("=== Strategies for Dealing with Imbalanced Datasets ===\n")

# Create imbalanced datasets
def create_imbalanced_datasets():
    """Create various levels of class imbalance for demonstration"""
    datasets = {}
    
    # Moderately imbalanced (1:4 ratio)
    X_mod, y_mod = make_classification(
        n_samples=5000, n_features=20, n_informative=10,
        n_redundant=5, n_clusters_per_class=1,
        weights=[0.8, 0.2], random_state=42
    )
    datasets['Moderate (1:4)'] = (X_mod, y_mod)
    
    # Highly imbalanced (1:9 ratio)
    X_high, y_high = make_classification(
        n_samples=5000, n_features=20, n_informative=10,
        n_redundant=5, n_clusters_per_class=1,
        weights=[0.9, 0.1], random_state=42
    )
    datasets['High (1:9)'] = (X_high, y_high)
    
    # Extremely imbalanced (1:99 ratio)
    X_extreme, y_extreme = make_classification(
        n_samples=5000, n_features=20, n_informative=10,
        n_redundant=5, n_clusters_per_class=1,
        weights=[0.99, 0.01], random_state=42
    )
    datasets['Extreme (1:99)'] = (X_extreme, y_extreme)
    
    return datasets

datasets = create_imbalanced_datasets()

# Analyze class distributions
print("Dataset Analysis:")
for name, (X, y) in datasets.items():
    class_counts = Counter(y)
    total = len(y)
    print(f"\n{name} Imbalance:")
    print(f"  Total samples: {total:,}")
    print(f"  Majority class (0): {class_counts[0]:,} ({class_counts[0]/total:.1%})")
    print(f"  Minority class (1): {class_counts[1]:,} ({class_counts[1]/total:.1%})")
    print(f"  Imbalance ratio: 1:{class_counts[0]/class_counts[1]:.1f}")

print()

# Use moderately imbalanced dataset for detailed analysis
X, y = datasets['Moderate (1:4)']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("1. BASELINE PERFORMANCE (No Handling)")
print("-" * 45)

print("Theory: First establish baseline performance to understand")
print("the impact of class imbalance on standard algorithms.")
print()

# Train baseline models
baseline_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

baseline_results = {}
print("Baseline Performance on Imbalanced Data:")

for name, model in baseline_models.items():
    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    baseline_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_prob
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC: {roc_auc:.4f}")

# Show the problem with accuracy
print(f"\nWhy Accuracy is Misleading:")
class_dist = Counter(y_test)
majority_accuracy = class_dist[0] / len(y_test)
print(f"  Majority class baseline: {majority_accuracy:.4f}")
print(f"  Best model accuracy: {max([r['accuracy'] for r in baseline_results.values()]):.4f}")
print(f"  → High accuracy doesn't mean good minority class detection!")

print()

# 2. RESAMPLING TECHNIQUES
print("2. RESAMPLING TECHNIQUES")
print("-" * 25)

print("Theory: Modify the training set distribution to balance")
print("classes through oversampling, undersampling, or hybrid methods.")
print()

# A. Oversampling Techniques
print("A. OVERSAMPLING TECHNIQUES")
print("-" * 30)

oversampling_methods = {
    'Random Oversampling': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'Borderline SMOTE': BorderlineSMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'SVM SMOTE': SVMSMOTE(random_state=42)
}

oversampling_results = {}
print("Oversampling Methods Comparison:")

for name, sampler in oversampling_methods.items():
    try:
        # Apply resampling
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        # Train model on resampled data
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_resampled, y_resampled)
        
        # Predict on original test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        oversampling_results[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'samples_before': len(y_train),
            'samples_after': len(y_resampled),
            'class_distribution': Counter(y_resampled)
        }
        
        print(f"\n{name}:")
        print(f"  Original samples: {len(y_train):,}")
        print(f"  Resampled samples: {len(y_resampled):,}")
        print(f"  Class distribution: {dict(Counter(y_resampled))}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
    except Exception as e:
        print(f"\n{name}: Failed - {str(e)}")

print()

# B. Undersampling Techniques
print("B. UNDERSAMPLING TECHNIQUES")
print("-" * 31)

undersampling_methods = {
    'Random Undersampling': RandomUnderSampler(random_state=42),
    'Tomek Links': TomekLinks(),
    'Edited Nearest Neighbours': EditedNearestNeighbours(),
    'Condensed Nearest Neighbour': CondensedNearestNeighbour(random_state=42),
    'One-Sided Selection': OneSidedSelection(random_state=42)
}

undersampling_results = {}
print("Undersampling Methods Comparison:")

for name, sampler in undersampling_methods.items():
    try:
        # Apply resampling
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        # Train model on resampled data
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_resampled, y_resampled)
        
        # Predict on original test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        undersampling_results[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'samples_before': len(y_train),
            'samples_after': len(y_resampled),
            'class_distribution': Counter(y_resampled)
        }
        
        print(f"\n{name}:")
        print(f"  Original samples: {len(y_train):,}")
        print(f"  Resampled samples: {len(y_resampled):,}")
        print(f"  Class distribution: {dict(Counter(y_resampled))}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
    except Exception as e:
        print(f"\n{name}: Failed - {str(e)}")

print()

# C. Hybrid Techniques
print("C. HYBRID TECHNIQUES (Combine Over & Under)")
print("-" * 45)

hybrid_methods = {
    'SMOTE + Tomek': SMOTETomek(random_state=42),
    'SMOTE + ENN': SMOTEENN(random_state=42)
}

hybrid_results = {}
print("Hybrid Methods Comparison:")

for name, sampler in hybrid_methods.items():
    try:
        # Apply resampling
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        # Train model on resampled data
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_resampled, y_resampled)
        
        # Predict on original test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        hybrid_results[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'samples_before': len(y_train),
            'samples_after': len(y_resampled),
            'class_distribution': Counter(y_resampled)
        }
        
        print(f"\n{name}:")
        print(f"  Original samples: {len(y_train):,}")
        print(f"  Resampled samples: {len(y_resampled):,}")
        print(f"  Class distribution: {dict(Counter(y_resampled))}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
    except Exception as e:
        print(f"\n{name}: Failed - {str(e)}")

print()

# 3. COST-SENSITIVE LEARNING
print("3. COST-SENSITIVE LEARNING")
print("-" * 30)

print("Theory: Assign different costs to misclassification errors,")
print("making minority class errors more expensive than majority class errors.")
print()

# Calculate class weights
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print("Class Weight Analysis:")
print(f"  Class 0 (majority) weight: {class_weight_dict[0]:.3f}")
print(f"  Class 1 (minority) weight: {class_weight_dict[1]:.3f}")
print(f"  Weight ratio: {class_weight_dict[1]/class_weight_dict[0]:.2f}")

# Cost-sensitive models
cost_sensitive_models = {
    'Weighted Logistic Regression': LogisticRegression(
        class_weight='balanced', random_state=42, max_iter=1000
    ),
    'Weighted Random Forest': RandomForestClassifier(
        class_weight='balanced', n_estimators=100, random_state=42
    ),
    'Weighted SVM': SVC(
        class_weight='balanced', probability=True, random_state=42
    ),
    'Balanced Random Forest': BalancedRandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

cost_sensitive_results = {}
print("\nCost-Sensitive Learning Results:")

for name, model in cost_sensitive_models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    cost_sensitive_results[name] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    print(f"\n{name}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC: {roc_auc:.4f}")

print()

# 4. THRESHOLD TUNING
print("4. THRESHOLD TUNING")
print("-" * 20)

print("Theory: Adjust decision threshold to optimize for")
print("specific metrics like precision, recall, or F1-score.")
print()

# Use best baseline model for threshold tuning
best_baseline_model = baseline_models['Random Forest']
best_baseline_model.fit(X_train, y_train)
y_prob_baseline = best_baseline_model.predict_proba(X_test)[:, 1]

# Calculate precision-recall curve
precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_prob_baseline)

# Find optimal thresholds for different objectives
# Optimal F1 threshold
f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve)
f1_scores = f1_scores[:-1]  # Remove last element to match thresholds length
optimal_f1_idx = np.argmax(f1_scores)
optimal_f1_threshold = thresholds[optimal_f1_idx]

# Optimal precision threshold (precision >= 0.8)
high_precision_mask = precision_curve >= 0.8
if np.any(high_precision_mask):
    optimal_precision_idx = np.where(high_precision_mask)[0][-1]
    optimal_precision_threshold = thresholds[optimal_precision_idx] if optimal_precision_idx < len(thresholds) else thresholds[-1]
else:
    optimal_precision_threshold = 0.9

# Optimal recall threshold (recall >= 0.8)
high_recall_mask = recall_curve >= 0.8
if np.any(high_recall_mask):
    optimal_recall_idx = np.where(high_recall_mask)[0][0]
    optimal_recall_threshold = thresholds[optimal_recall_idx] if optimal_recall_idx < len(thresholds) else thresholds[0]
else:
    optimal_recall_threshold = 0.1

threshold_strategies = {
    'Default (0.5)': 0.5,
    'Optimal F1': optimal_f1_threshold,
    'High Precision': optimal_precision_threshold,
    'High Recall': optimal_recall_threshold
}

print("Threshold Tuning Results:")
threshold_results = {}

for strategy, threshold in threshold_strategies.items():
    # Apply threshold
    y_pred_thresh = (y_prob_baseline >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    
    threshold_results[strategy] = {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"\n{strategy} (threshold={threshold:.3f}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

print()

# 5. ENSEMBLE METHODS FOR IMBALANCED DATA
print("5. ENSEMBLE METHODS FOR IMBALANCED DATA")
print("-" * 40)

print("Theory: Combine multiple models trained on different")
print("balanced subsets or use ensemble-specific algorithms.")
print()

# Bagging with balanced sampling
from sklearn.ensemble import BaggingClassifier

ensemble_methods = {
    'Balanced Bagging': BaggingClassifier(
        base_estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=10,
        random_state=42
    ),
    'Easy Ensemble': BalancedRandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Manual implementation of balanced bagging
def balanced_bagging_predictions(X_train, y_train, X_test, n_estimators=10):
    """Manual implementation of balanced bagging"""
    predictions = []
    probabilities = []
    
    # Get minority class size
    minority_count = min(Counter(y_train).values())
    
    for i in range(n_estimators):
        # Create balanced sample
        minority_indices = np.where(y_train == 1)[0]
        majority_indices = np.where(y_train == 0)[0]
        
        # Sample equal numbers from each class
        balanced_minority = np.random.choice(minority_indices, minority_count, replace=True)
        balanced_majority = np.random.choice(majority_indices, minority_count, replace=False)
        
        balanced_indices = np.concatenate([balanced_minority, balanced_majority])
        X_balanced = X_train[balanced_indices]
        y_balanced = y_train[balanced_indices]
        
        # Train model on balanced sample
        model = RandomForestClassifier(n_estimators=50, random_state=i)
        model.fit(X_balanced, y_balanced)
        
        # Predictions
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        
        predictions.append(pred)
        probabilities.append(prob)
    
    # Aggregate predictions
    final_probs = np.mean(probabilities, axis=0)
    final_preds = (final_probs >= 0.5).astype(int)
    
    return final_preds, final_probs

# Test ensemble methods
ensemble_results = {}

print("Ensemble Methods for Imbalanced Data:")

# Balanced Bagging (manual)
y_pred_ensemble, y_prob_ensemble = balanced_bagging_predictions(
    X_train, y_train, X_test, n_estimators=10
)

precision_ens = precision_score(y_test, y_pred_ensemble)
recall_ens = recall_score(y_test, y_pred_ensemble)
f1_ens = f1_score(y_test, y_pred_ensemble)
roc_auc_ens = roc_auc_score(y_test, y_prob_ensemble)

ensemble_results['Manual Balanced Bagging'] = {
    'precision': precision_ens,
    'recall': recall_ens,
    'f1': f1_ens,
    'roc_auc': roc_auc_ens
}

print(f"\nManual Balanced Bagging:")
print(f"  Precision: {precision_ens:.4f}")
print(f"  Recall: {recall_ens:.4f}")
print(f"  F1-Score: {f1_ens:.4f}")
print(f"  ROC-AUC: {roc_auc_ens:.4f}")

# Balanced Random Forest
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf.fit(X_train, y_train)
y_pred_brf = brf.predict(X_test)
y_prob_brf = brf.predict_proba(X_test)[:, 1]

precision_brf = precision_score(y_test, y_pred_brf)
recall_brf = recall_score(y_test, y_pred_brf)
f1_brf = f1_score(y_test, y_pred_brf)
roc_auc_brf = roc_auc_score(y_test, y_prob_brf)

ensemble_results['Balanced Random Forest'] = {
    'precision': precision_brf,
    'recall': recall_brf,
    'f1': f1_brf,
    'roc_auc': roc_auc_brf
}

print(f"\nBalanced Random Forest:")
print(f"  Precision: {precision_brf:.4f}")
print(f"  Recall: {recall_brf:.4f}")
print(f"  F1-Score: {f1_brf:.4f}")
print(f"  ROC-AUC: {roc_auc_brf:.4f}")

print()

# 6. EVALUATION METRICS FOR IMBALANCED DATA
print("6. EVALUATION METRICS FOR IMBALANCED DATA")
print("-" * 45)

print("Theory: Choose appropriate metrics that are not misleading")
print("in the presence of class imbalance.")
print()

# Comprehensive metric comparison
def evaluate_comprehensive(y_true, y_pred, y_prob=None):
    """Calculate comprehensive metrics for imbalanced data evaluation"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    
    # Confusion matrix derived metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['specificity'] = tn / (tn + fp)  # True negative rate
    metrics['sensitivity'] = tp / (tp + fn)  # True positive rate (recall)
    metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
    
    # Probability-based metrics
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    
    return metrics

# Compare best methods from each category
comparison_methods = {
    'Baseline RF': ('baseline', baseline_results['Random Forest']),
    'SMOTE + RF': ('resampling', None),  # Will calculate
    'Weighted RF': ('cost_sensitive', cost_sensitive_results['Weighted Random Forest']),
    'Threshold Tuned': ('threshold', None),  # Will calculate
    'Balanced RF': ('ensemble', ensemble_results['Balanced Random Forest'])
}

# Calculate SMOTE results
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_smote, y_smote)
y_pred_smote = rf_smote.predict(X_test)
y_prob_smote = rf_smote.predict_proba(X_test)[:, 1]

# Calculate threshold tuned results
y_pred_thresh_opt = (y_prob_baseline >= optimal_f1_threshold).astype(int)

print("Comprehensive Evaluation Comparison:")
print("Method\t\tAcc\tPrec\tRec\tF1\tSpec\tBalAcc\tROC\tPR-AUC")
print("-" * 80)

final_comparison = {}

for method_name, (category, stored_results) in comparison_methods.items():
    if method_name == 'SMOTE + RF':
        metrics = evaluate_comprehensive(y_test, y_pred_smote, y_prob_smote)
    elif method_name == 'Threshold Tuned':
        metrics = evaluate_comprehensive(y_test, y_pred_thresh_opt, y_prob_baseline)
    elif category == 'baseline':
        metrics = evaluate_comprehensive(
            y_test, 
            stored_results['predictions'], 
            stored_results['probabilities']
        )
    elif category == 'cost_sensitive':
        # Need to recalculate for comprehensive metrics
        model = cost_sensitive_models['Weighted Random Forest']
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_comprehensive(y_test, y_pred, y_prob)
    elif category == 'ensemble':
        metrics = evaluate_comprehensive(y_test, y_pred_brf, y_prob_brf)
    
    final_comparison[method_name] = metrics
    
    print(f"{method_name[:15]:<15}\t"
          f"{metrics['accuracy']:.3f}\t"
          f"{metrics['precision']:.3f}\t"
          f"{metrics['recall']:.3f}\t"
          f"{metrics['f1']:.3f}\t"
          f"{metrics['specificity']:.3f}\t"
          f"{metrics['balanced_accuracy']:.3f}\t"
          f"{metrics.get('roc_auc', 0):.3f}\t"
          f"{metrics.get('pr_auc', 0):.3f}")

print()

# 7. VISUALIZATION AND ANALYSIS
print("7. VISUALIZATION AND ANALYSIS")
print("-" * 32)

# Create comprehensive visualization
plt.figure(figsize=(20, 15))

# Confusion matrices comparison
methods_for_viz = ['Baseline RF', 'SMOTE + RF', 'Weighted RF', 'Balanced RF']
cms = {
    'Baseline RF': confusion_matrix(y_test, baseline_results['Random Forest']['predictions']),
    'SMOTE + RF': confusion_matrix(y_test, y_pred_smote),
    'Weighted RF': confusion_matrix(y_test, cost_sensitive_models['Weighted Random Forest'].predict(X_test)),
    'Balanced RF': confusion_matrix(y_test, y_pred_brf)
}

for i, method in enumerate(methods_for_viz):
    plt.subplot(3, 4, i+1)
    sns.heatmap(cms[method], annot=True, fmt='d', cmap='Blues')
    plt.title(f'{method}\nConfusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')

# ROC Curves
plt.subplot(3, 4, 5)
for method in methods_for_viz:
    if method == 'Baseline RF':
        y_prob = baseline_results['Random Forest']['probabilities']
    elif method == 'SMOTE + RF':
        y_prob = y_prob_smote
    elif method == 'Weighted RF':
        y_prob = cost_sensitive_models['Weighted Random Forest'].predict_proba(X_test)[:, 1]
    elif method == 'Balanced RF':
        y_prob = y_prob_brf
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{method} (AUC={auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Precision-Recall Curves
plt.subplot(3, 4, 6)
for method in methods_for_viz:
    if method == 'Baseline RF':
        y_prob = baseline_results['Random Forest']['probabilities']
    elif method == 'SMOTE + RF':
        y_prob = y_prob_smote
    elif method == 'Weighted RF':
        y_prob = cost_sensitive_models['Weighted Random Forest'].predict_proba(X_test)[:, 1]
    elif method == 'Balanced RF':
        y_prob = y_prob_brf
    
    precision_pr, recall_pr, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.plot(recall_pr, precision_pr, label=f'{method} (AP={ap:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# Metrics comparison
plt.subplot(3, 4, 7)
metrics_to_plot = ['precision', 'recall', 'f1', 'balanced_accuracy']
x_pos = np.arange(len(methods_for_viz))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    values = [final_comparison[method][metric] for method in methods_for_viz]
    plt.bar(x_pos + i*width, values, width, label=metric.capitalize(), alpha=0.8)

plt.xlabel('Methods')
plt.ylabel('Score')
plt.title('Metrics Comparison')
plt.xticks(x_pos + width*1.5, [m[:8] for m in methods_for_viz], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Threshold analysis
plt.subplot(3, 4, 8)
thresholds_range = np.linspace(0.1, 0.9, 50)
f1_scores_thresh = []
precision_scores_thresh = []
recall_scores_thresh = []

for thresh in thresholds_range:
    y_pred_t = (y_prob_baseline >= thresh).astype(int)
    f1_scores_thresh.append(f1_score(y_test, y_pred_t))
    precision_scores_thresh.append(precision_score(y_test, y_pred_t))
    recall_scores_thresh.append(recall_score(y_test, y_pred_t))

plt.plot(thresholds_range, f1_scores_thresh, label='F1-Score', linewidth=2)
plt.plot(thresholds_range, precision_scores_thresh, label='Precision', linewidth=2)
plt.plot(thresholds_range, recall_scores_thresh, label='Recall', linewidth=2)
plt.axvline(x=optimal_f1_threshold, color='red', linestyle='--', label='Optimal F1')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# Class distribution comparison
plt.subplot(3, 4, 9)
methods_dist = ['Original', 'SMOTE', 'Random Over', 'Random Under']
class_0_counts = [
    Counter(y_train)[0],
    Counter(y_smote)[0],
    4000,  # Approximate for random oversampling
    Counter(y_train)[1]  # Same as minority for undersampling
]
class_1_counts = [
    Counter(y_train)[1],
    Counter(y_smote)[1],
    Counter(y_train)[0],  # Same as majority for oversampling
    Counter(y_train)[1]
]

x = np.arange(len(methods_dist))
plt.bar(x - 0.2, class_0_counts, 0.4, label='Class 0 (Majority)', alpha=0.8)
plt.bar(x + 0.2, class_1_counts, 0.4, label='Class 1 (Minority)', alpha=0.8)
plt.xlabel('Resampling Method')
plt.ylabel('Number of Samples')
plt.title('Class Distribution After Resampling')
plt.xticks(x, methods_dist, rotation=45)
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Feature importance analysis (if available)
plt.subplot(3, 4, 10)
if hasattr(rf_smote, 'feature_importances_'):
    feature_importance = rf_smote.feature_importances_
    top_features = np.argsort(feature_importance)[-10:]
    plt.barh(range(len(top_features)), feature_importance[top_features])
    plt.yticks(range(len(top_features)), [f'Feature {i}' for i in top_features])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances\n(SMOTE + RF)')
    plt.grid(True, alpha=0.3)

# Performance vs imbalance ratio
plt.subplot(3, 4, 11)
imbalance_ratios = []
f1_scores_by_ratio = []

for dataset_name, (X_data, y_data) in datasets.items():
    X_tr, X_te, y_tr, y_te = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
    
    # Train SMOTE model
    smote_temp = SMOTE(random_state=42)
    X_smote_temp, y_smote_temp = smote_temp.fit_resample(X_tr, y_tr)
    model_temp = RandomForestClassifier(n_estimators=50, random_state=42)
    model_temp.fit(X_smote_temp, y_smote_temp)
    y_pred_temp = model_temp.predict(X_te)
    
    ratio = Counter(y_tr)[0] / Counter(y_tr)[1]
    f1_temp = f1_score(y_te, y_pred_temp)
    
    imbalance_ratios.append(ratio)
    f1_scores_by_ratio.append(f1_temp)

plt.scatter(imbalance_ratios, f1_scores_by_ratio, s=100, alpha=0.7)
for i, dataset_name in enumerate(datasets.keys()):
    plt.annotate(dataset_name, (imbalance_ratios[i], f1_scores_by_ratio[i]),
                xytext=(5, 5), textcoords='offset points')
plt.xlabel('Imbalance Ratio (Majority:Minority)')
plt.ylabel('F1-Score with SMOTE')
plt.title('Performance vs Imbalance Level')
plt.xscale('log')
plt.grid(True, alpha=0.3)

# Cost matrix visualization
plt.subplot(3, 4, 12)
cost_matrix = np.array([[1, 5], [1, 1]])  # Example: FP costs 5x more than FN
sns.heatmap(cost_matrix, annot=True, fmt='d', cmap='Reds',
           xticklabels=['Predicted 0', 'Predicted 1'],
           yticklabels=['Actual 0', 'Actual 1'])
plt.title('Example Cost Matrix\n(FP=5, FN=1, TP=1, TN=1)')

plt.tight_layout()
plt.show()

print()

# 8. BEST PRACTICES AND RECOMMENDATIONS
print("8. BEST PRACTICES AND RECOMMENDATIONS")
print("-" * 42)

best_practices = {
    "Strategy Selection": [
        "Analyze the level of imbalance (moderate vs extreme)",
        "Consider domain knowledge and business costs",
        "Try multiple approaches and compare results",
        "Use appropriate evaluation metrics (not just accuracy)"
    ],
    "Resampling Guidelines": [
        "Use SMOTE for moderate imbalance (1:4 to 1:10)",
        "Consider hybrid methods (SMOTE + ENN) for noisy data",
        "Avoid oversampling for very small minority classes",
        "Always validate on original test distribution"
    ],
    "Cost-Sensitive Learning": [
        "Use when misclassification costs are known",
        "Start with balanced class weights",
        "Tune weights based on business requirements",
        "Combine with threshold tuning for optimal results"
    ],
    "Evaluation Best Practices": [
        "Use stratified train/test splits",
        "Report precision, recall, F1, and AUC",
        "Analyze confusion matrices carefully",
        "Consider precision-recall curves over ROC for extreme imbalance"
    ],
    "Production Considerations": [
        "Monitor class distribution drift",
        "Retrain models when distribution changes",
        "Document resampling strategies used",
        "Plan for threshold updates based on feedback"
    ]
}

print("Best Practices for Imbalanced Datasets:")
for category, practices in best_practices.items():
    print(f"\n{category}:")
    for practice in practices:
        print(f"  • {practice}")

print()

# 9. DECISION FRAMEWORK
print("9. DECISION FRAMEWORK")
print("-" * 22)

decision_framework = {
    "Level of Imbalance": {
        "Mild (1:2 to 1:4)": ["Cost-sensitive learning", "Threshold tuning"],
        "Moderate (1:4 to 1:10)": ["SMOTE", "Cost-sensitive learning", "Ensemble methods"],
        "Severe (1:10 to 1:100)": ["Hybrid resampling", "Ensemble methods", "Anomaly detection"],
        "Extreme (>1:100)": ["Anomaly detection", "One-class classification", "Threshold tuning"]
    },
    "Dataset Size": {
        "Small (<1000 samples)": ["Cost-sensitive learning", "Threshold tuning"],
        "Medium (1K-100K)": ["SMOTE", "Ensemble methods", "Hybrid approaches"],
        "Large (>100K)": ["Random sampling", "Cost-sensitive learning", "Batch processing"]
    },
    "Business Requirements": {
        "High Precision Needed": ["Threshold tuning toward precision", "Cost-sensitive with high FP cost"],
        "High Recall Needed": ["Oversampling", "Threshold tuning toward recall"],
        "Balanced Performance": ["SMOTE", "Balanced ensemble methods"],
        "Interpretability Important": ["Cost-sensitive single models", "Simple threshold tuning"]
    }
}

print("Decision Framework for Strategy Selection:")
for criterion, options in decision_framework.items():
    print(f"\n{criterion}:")
    for situation, strategies in options.items():
        print(f"  {situation}: {', '.join(strategies)}")

print(f"\n{'='*60}")
print("SUMMARY: Imbalanced Dataset Strategies")
print(f"{'='*60}")

summary_points = [
    "Accuracy is misleading - use precision, recall, F1, and AUC",
    "SMOTE is effective for moderate imbalance levels",
    "Cost-sensitive learning works well when costs are known",
    "Threshold tuning is simple and often effective",
    "Ensemble methods provide robust performance",
    "Hybrid approaches combine benefits of multiple strategies",
    "Always validate on original test distribution",
    "Consider business costs in strategy selection",
    "Monitor for class distribution drift in production",
    "Document and version imbalance handling strategies"
]

for i, point in enumerate(summary_points, 1):
    print(f"{i:2d}. {point}")

print(f"\n=== Imbalanced Data: Challenge and Opportunity ===")
```

### Explanation

Imbalanced datasets are common in real-world applications and require specialized strategies to achieve good performance on minority classes that are often the most important to detect correctly.

### Core Strategies

1. **Resampling Techniques**
   - **Oversampling**: Increase minority class samples (SMOTE, ADASYN)
   - **Undersampling**: Reduce majority class samples (Tomek Links, ENN)
   - **Hybrid**: Combine both approaches (SMOTE + Tomek)

2. **Cost-Sensitive Learning**
   - Assign higher misclassification costs to minority class
   - Use class weights in model training
   - Reflects real-world business costs

3. **Threshold Tuning**
   - Adjust decision threshold from default 0.5
   - Optimize for specific metrics (precision, recall, F1)
   - Simple but often effective approach

4. **Ensemble Methods**
   - Balanced bagging with equal class samples
   - Balanced Random Forest
   - Easy Ensemble techniques

### Key Techniques

**SMOTE (Synthetic Minority Oversampling):**
- Creates synthetic examples in feature space
- Better than random oversampling
- Works well for moderate imbalance

**Cost-Sensitive Learning:**
- Weights classes by inverse frequency
- Penalizes minority class errors more heavily
- Integrates naturally with many algorithms

**Threshold Optimization:**
- Finds optimal cutoff for classification
- Balances precision and recall trade-offs
- Uses precision-recall curve analysis

### Evaluation Considerations

**Misleading Metrics:**
- **Accuracy**: Can be high while missing all minority cases
- **Precision**: May be inflated with few positive predictions
- **Recall**: Alone doesn't consider false positive cost

**Appropriate Metrics:**
- **F1-Score**: Balances precision and recall
- **ROC-AUC**: Threshold-independent performance
- **PR-AUC**: Better for extreme imbalance
- **Balanced Accuracy**: Accounts for both classes equally

### Strategy Selection Guide

**Based on Imbalance Level:**
- **Mild (1:2-1:4)**: Cost-sensitive learning, threshold tuning
- **Moderate (1:4-1:10)**: SMOTE, ensemble methods
- **Severe (1:10-1:100)**: Hybrid resampling, specialized ensembles
- **Extreme (>1:100)**: Anomaly detection approaches

**Based on Business Requirements:**
- **High Precision Needed**: Threshold tuning toward precision
- **High Recall Needed**: Oversampling techniques
- **Balanced Performance**: SMOTE or hybrid methods
- **Interpretability**: Cost-sensitive single models

### Best Practices

1. **Always Use Stratified Splits**: Maintain class distribution in train/test
2. **Choose Appropriate Metrics**: Avoid accuracy, focus on F1, AUC
3. **Validate on Original Distribution**: Don't resample test data
4. **Try Multiple Approaches**: Compare different strategies
5. **Consider Domain Knowledge**: Incorporate business costs and requirements
6. **Monitor Distribution Drift**: Track changes in production data

### Common Pitfalls

- **Resampling Test Data**: Never apply resampling to test set
- **Focusing Only on Accuracy**: Missing poor minority class performance
- **Ignoring Business Costs**: Not considering real-world consequences
- **One-Size-Fits-All**: Using same strategy for all imbalance levels
- **Overfitting to Minority**: Creating too many synthetic examples

### Production Considerations

- **Monitor Class Distribution**: Track changes over time
- **Plan for Retraining**: Update models when distribution shifts
- **Document Strategies**: Record resampling and weighting decisions
- **Consider Computational Costs**: Balance performance with efficiency
- **Feedback Integration**: Use business feedback to refine approaches

Dealing with imbalanced datasets requires a strategic approach that considers the level of imbalance, business requirements, and computational constraints while using appropriate evaluation metrics to assess true model performance.

---

## Question 10

**Discuss the importance of model persistence and demonstrate how to save and load models in Python.**

### Theory
Model persistence is crucial for deploying machine learning models to production, sharing models with team members, and avoiding costly retraining. It involves serializing trained models and their associated preprocessing pipelines to disk for later use, ensuring reproducibility and enabling model versioning.

### Answer

```python
# model_persistence.py - Comprehensive guide to saving and loading ML models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle
import dill
import json
import os
import time
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

print("=== Model Persistence: Saving and Loading ML Models ===\n")

# Create sample dataset and train models
def prepare_sample_models():
    """Prepare various trained models for persistence demonstration"""
    # Generate dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_clusters_per_class=1, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train models
    models = {}
    
    # Simple model
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    models['logistic_regression'] = {
        'model': lr,
        'X_test': X_test,
        'y_test': y_test,
        'features': [f'feature_{i}' for i in range(X.shape[1])],
        'training_time': datetime.now(),
        'accuracy': lr.score(X_test, y_test)
    }
    
    # Pipeline model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    models['pipeline_model'] = {
        'model': pipeline,
        'X_test': X_test,
        'y_test': y_test,
        'features': [f'feature_{i}' for i in range(X.shape[1])],
        'training_time': datetime.now(),
        'accuracy': pipeline.score(X_test, y_test)
    }
    
    # Grid search model
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=3, scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    models['grid_search_model'] = {
        'model': grid_search,
        'X_test': X_test,
        'y_test': y_test,
        'features': [f'feature_{i}' for i in range(X.shape[1])],
        'training_time': datetime.now(),
        'accuracy': grid_search.score(X_test, y_test),
        'best_params': grid_search.best_params_
    }
    
    return models

# Prepare models
print("1. PREPARING SAMPLE MODELS")
print("-" * 30)
print("Creating and training various ML models for persistence demonstration...")

models = prepare_sample_models()

for name, model_info in models.items():
    print(f"\n{name.replace('_', ' ').title()}:")
    print(f"  Model type: {type(model_info['model']).__name__}")
    print(f"  Accuracy: {model_info['accuracy']:.4f}")
    if 'best_params' in model_info:
        print(f"  Best parameters: {model_info['best_params']}")

print()

# 2. SERIALIZATION METHODS
print("2. SERIALIZATION METHODS")
print("-" * 25)

print("Theory: Different serialization libraries offer various")
print("trade-offs in terms of compatibility, performance, and features.")
print()

# Create directory for saved models
os.makedirs('saved_models', exist_ok=True)

# A. Joblib (Recommended for scikit-learn)
print("A. JOBLIB (Recommended for scikit-learn)")
print("-" * 40)

print("Joblib is optimized for NumPy arrays and scikit-learn models")

# Save with joblib
model_to_save = models['logistic_regression']['model']
joblib_start = time.time()
joblib.dump(model_to_save, 'saved_models/logistic_regression_joblib.pkl')
joblib_save_time = time.time() - joblib_start

# Load with joblib
joblib_load_start = time.time()
loaded_model_joblib = joblib.load('saved_models/logistic_regression_joblib.pkl')
joblib_load_time = time.time() - joblib_load_start

# Test loaded model
X_test = models['logistic_regression']['X_test']
y_test = models['logistic_regression']['y_test']
original_predictions = model_to_save.predict(X_test)
loaded_predictions = loaded_model_joblib.predict(X_test)

print(f"  Save time: {joblib_save_time:.4f} seconds")
print(f"  Load time: {joblib_load_time:.4f} seconds")
print(f"  File size: {os.path.getsize('saved_models/logistic_regression_joblib.pkl')} bytes")
print(f"  Predictions match: {np.array_equal(original_predictions, loaded_predictions)}")

# B. Pickle (Standard Python)
print(f"\nB. PICKLE (Standard Python)")
print("-" * 28)

print("Standard Python serialization, works with any Python object")

# Save with pickle
pickle_start = time.time()
with open('saved_models/logistic_regression_pickle.pkl', 'wb') as f:
    pickle.dump(model_to_save, f)
pickle_save_time = time.time() - pickle_start

# Load with pickle
pickle_load_start = time.time()
with open('saved_models/logistic_regression_pickle.pkl', 'rb') as f:
    loaded_model_pickle = pickle.load(f)
pickle_load_time = time.time() - pickle_load_start

# Test loaded model
pickle_predictions = loaded_model_pickle.predict(X_test)

print(f"  Save time: {pickle_save_time:.4f} seconds")
print(f"  Load time: {pickle_load_time:.4f} seconds")
print(f"  File size: {os.path.getsize('saved_models/logistic_regression_pickle.pkl')} bytes")
print(f"  Predictions match: {np.array_equal(original_predictions, pickle_predictions)}")

# C. Dill (Enhanced pickle)
print(f"\nC. DILL (Enhanced pickle)")
print("-" * 25)

print("Can serialize more Python objects than pickle")

# Save with dill
dill_start = time.time()
with open('saved_models/logistic_regression_dill.pkl', 'wb') as f:
    dill.dump(model_to_save, f)
dill_save_time = time.time() - dill_start

# Load with dill
dill_load_start = time.time()
with open('saved_models/logistic_regression_dill.pkl', 'rb') as f:
    loaded_model_dill = dill.load(f)
dill_load_time = time.time() - dill_load_start

# Test loaded model
dill_predictions = loaded_model_dill.predict(X_test)

print(f"  Save time: {dill_save_time:.4f} seconds")
print(f"  Load time: {dill_load_time:.4f} seconds")
print(f"  File size: {os.path.getsize('saved_models/logistic_regression_dill.pkl')} bytes")
print(f"  Predictions match: {np.array_equal(original_predictions, dill_predictions)}")

print()

# 3. COMPREHENSIVE MODEL PERSISTENCE
print("3. COMPREHENSIVE MODEL PERSISTENCE")
print("-" * 38)

print("Theory: Save not just the model, but all necessary components")
print("including preprocessing, metadata, and versioning information.")
print()

class ModelManager:
    """Comprehensive model persistence manager"""
    
    def __init__(self):
        self.models_dir = 'saved_models'
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_model_comprehensive(self, model, model_name, metadata=None):
        """Save model with comprehensive metadata"""
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(model_dir, 'model.pkl')
        joblib.dump(model, model_path)
        
        # Create comprehensive metadata
        model_metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'save_timestamp': datetime.now().isoformat(),
            'model_size_bytes': os.path.getsize(model_path),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
            'sklearn_version': sklearn.__version__,
            'joblib_version': joblib.__version__
        }
        
        # Add custom metadata if provided
        if metadata:
            model_metadata.update(metadata)
        
        # Generate model hash for integrity checking
        with open(model_path, 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()
        model_metadata['model_hash'] = model_hash
        
        # Save metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Save requirements (if available)
        try:
            import pkg_resources
            requirements = [str(d) for d in pkg_resources.working_set]
            req_path = os.path.join(model_dir, 'requirements.txt')
            with open(req_path, 'w') as f:
                f.write('\n'.join(sorted(requirements)))
        except:
            pass
        
        print(f"Model saved to: {model_dir}")
        return model_dir
    
    def load_model_comprehensive(self, model_name, verify_hash=True):
        """Load model with verification and metadata"""
        
        model_dir = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        model_path = os.path.join(model_dir, 'model.pkl')
        model = joblib.load(model_path)
        
        # Verify integrity if requested
        if verify_hash and 'model_hash' in metadata:
            with open(model_path, 'rb') as f:
                current_hash = hashlib.md5(f.read()).hexdigest()
            
            if current_hash != metadata['model_hash']:
                raise ValueError("Model file integrity check failed!")
        
        print(f"Model loaded from: {model_dir}")
        print(f"Model type: {metadata.get('model_type', 'Unknown')}")
        print(f"Saved on: {metadata.get('save_timestamp', 'Unknown')}")
        
        return model, metadata
    
    def list_saved_models(self):
        """List all saved models with their metadata"""
        
        models_info = []
        
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    models_info.append(metadata)
        
        return models_info

import sys
import sklearn

# Initialize model manager
manager = ModelManager()

# Save models comprehensively
print("Saving models with comprehensive metadata:")

for name, model_info in models.items():
    metadata = {
        'accuracy': model_info['accuracy'],
        'features': model_info['features'],
        'training_timestamp': model_info['training_time'].isoformat(),
        'test_samples': len(model_info['y_test'])
    }
    
    if 'best_params' in model_info:
        metadata['best_parameters'] = model_info['best_params']
    
    model_dir = manager.save_model_comprehensive(
        model_info['model'], 
        name, 
        metadata
    )

print(f"\nSaved models information:")
saved_models = manager.list_saved_models()
for model_info in saved_models:
    print(f"  {model_info['model_name']}: {model_info['model_type']} "
          f"(Accuracy: {model_info.get('accuracy', 'N/A'):.4f})")

print()

# 4. PIPELINE PERSISTENCE
print("4. PIPELINE PERSISTENCE")
print("-" * 23)

print("Theory: Pipelines require special attention to ensure")
print("all preprocessing steps are correctly saved and loaded.")
print()

# Demonstrate pipeline persistence issues and solutions
print("Pipeline Persistence Demonstration:")

# Load the pipeline model
pipeline_model, pipeline_metadata = manager.load_model_comprehensive('pipeline_model')

# Test original vs loaded pipeline
original_pipeline = models['pipeline_model']['model']
X_test_pipeline = models['pipeline_model']['X_test']
y_test_pipeline = models['pipeline_model']['y_test']

print(f"\nPipeline Testing:")
print(f"  Original pipeline steps: {[name for name, _ in original_pipeline.steps]}")
print(f"  Loaded pipeline steps: {[name for name, _ in pipeline_model.steps]}")

# Compare predictions
original_pred = original_pipeline.predict(X_test_pipeline)
loaded_pred = pipeline_model.predict(X_test_pipeline)

print(f"  Original accuracy: {accuracy_score(y_test_pipeline, original_pred):.4f}")
print(f"  Loaded accuracy: {accuracy_score(y_test_pipeline, loaded_pred):.4f}")
print(f"  Predictions identical: {np.array_equal(original_pred, loaded_pred)}")

# Demonstrate accessing pipeline components
print(f"\nPipeline Component Access:")
scaler = pipeline_model.named_steps['scaler']
classifier = pipeline_model.named_steps['classifier']

print(f"  Scaler type: {type(scaler).__name__}")
print(f"  Scaler mean: {scaler.mean_[:5]}...")  # First 5 features
print(f"  Classifier type: {type(classifier).__name__}")
print(f"  Classifier n_estimators: {classifier.n_estimators}")

print()

# 5. MODEL VERSIONING
print("5. MODEL VERSIONING")
print("-" * 18)

print("Theory: Track model versions to enable rollbacks,")
print("A/B testing, and gradual deployments.")
print()

class VersionedModelManager:
    """Model manager with versioning capabilities"""
    
    def __init__(self, base_dir='versioned_models'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save_model_version(self, model, model_name, version=None, metadata=None):
        """Save model with version control"""
        
        # Auto-generate version if not provided
        if version is None:
            version = self._get_next_version(model_name)
        
        # Create versioned directory
        model_dir = os.path.join(self.base_dir, model_name, f'v{version}')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'model.pkl')
        joblib.dump(model, model_path)
        
        # Create version metadata
        version_metadata = {
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'model_size': os.path.getsize(model_path)
        }
        
        if metadata:
            version_metadata.update(metadata)
        
        # Save metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(version_metadata, f, indent=2)
        
        # Update latest version pointer
        self._update_latest_version(model_name, version)
        
        print(f"Saved {model_name} v{version} to {model_dir}")
        return version
    
    def load_model_version(self, model_name, version='latest'):
        """Load specific model version"""
        
        if version == 'latest':
            version = self._get_latest_version(model_name)
        
        model_dir = os.path.join(self.base_dir, model_name, f'v{version}')
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model version not found: {model_name} v{version}")
        
        # Load model and metadata
        model_path = os.path.join(model_dir, 'model.pkl')
        model = joblib.load(model_path)
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded {model_name} v{version}")
        return model, metadata
    
    def list_model_versions(self, model_name):
        """List all versions of a model"""
        
        model_base_dir = os.path.join(self.base_dir, model_name)
        if not os.path.exists(model_base_dir):
            return []
        
        versions = []
        for item in os.listdir(model_base_dir):
            if item.startswith('v') and item[1:].isdigit():
                version_num = int(item[1:])
                
                metadata_path = os.path.join(model_base_dir, item, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    versions.append(metadata)
        
        return sorted(versions, key=lambda x: x['version'], reverse=True)
    
    def _get_next_version(self, model_name):
        """Get next version number"""
        versions = self.list_model_versions(model_name)
        if not versions:
            return 1
        return max(v['version'] for v in versions) + 1
    
    def _get_latest_version(self, model_name):
        """Get latest version number"""
        latest_file = os.path.join(self.base_dir, model_name, 'latest.txt')
        if os.path.exists(latest_file):
            with open(latest_file, 'r') as f:
                return int(f.read().strip())
        return 1
    
    def _update_latest_version(self, model_name, version):
        """Update latest version pointer"""
        latest_file = os.path.join(self.base_dir, model_name, 'latest.txt')
        with open(latest_file, 'w') as f:
            f.write(str(version))

# Demonstrate versioning
versioned_manager = VersionedModelManager()

print("Model Versioning Demonstration:")

# Save multiple versions of a model
base_model = models['logistic_regression']['model']

# Version 1 - original model
v1 = versioned_manager.save_model_version(
    base_model, 'production_classifier',
    metadata={'accuracy': 0.85, 'description': 'Initial production model'}
)

# Simulate model improvement - retrain with different parameters
improved_model = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
improved_model.fit(models['logistic_regression']['X_test'], models['logistic_regression']['y_test'])

# Version 2 - improved model
v2 = versioned_manager.save_model_version(
    improved_model, 'production_classifier',
    metadata={'accuracy': 0.87, 'description': 'Improved model with regularization'}
)

# List versions
print(f"\nModel Versions:")
versions = versioned_manager.list_model_versions('production_classifier')
for version in versions:
    print(f"  v{version['version']}: {version['description']} "
          f"(Accuracy: {version.get('accuracy', 'N/A')})")

# Load specific versions
latest_model, latest_meta = versioned_manager.load_model_version('production_classifier', 'latest')
v1_model, v1_meta = versioned_manager.load_model_version('production_classifier', 1)

print()

# 6. DEPLOYMENT CONSIDERATIONS
print("6. DEPLOYMENT CONSIDERATIONS")
print("-" * 32)

print("Theory: Model persistence for production requires")
print("additional considerations for reliability and performance.")
print()

# Deployment-ready model packaging
class ProductionModelPackage:
    """Production-ready model package with validation and monitoring"""
    
    def __init__(self, model, preprocessor=None, feature_names=None, 
                 model_metadata=None):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names or []
        self.metadata = model_metadata or {}
        self.load_timestamp = datetime.now()
        self.prediction_count = 0
    
    def predict(self, X):
        """Make predictions with validation and monitoring"""
        
        # Input validation
        if hasattr(X, 'shape'):
            if len(self.feature_names) > 0 and X.shape[1] != len(self.feature_names):
                raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        
        # Apply preprocessing if available
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        # Make prediction
        predictions = self.model.predict(X_processed)
        
        # Update monitoring
        self.prediction_count += len(predictions) if hasattr(predictions, '__len__') else 1
        
        return predictions
    
    def predict_proba(self, X):
        """Make probability predictions"""
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("Model does not support probability predictions")
        
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        probabilities = self.model.predict_proba(X_processed)
        self.prediction_count += len(probabilities) if hasattr(probabilities, '__len__') else 1
        
        return probabilities
    
    def get_monitoring_info(self):
        """Get monitoring and health information"""
        return {
            'model_type': type(self.model).__name__,
            'load_timestamp': self.load_timestamp.isoformat(),
            'prediction_count': self.prediction_count,
            'uptime_hours': (datetime.now() - self.load_timestamp).total_seconds() / 3600,
            'feature_count': len(self.feature_names),
            'has_preprocessor': self.preprocessor is not None,
            'metadata': self.metadata
        }
    
    def save_package(self, filepath):
        """Save complete package"""
        package_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'metadata': self.metadata,
            'save_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(package_data, filepath)
        print(f"Production package saved to: {filepath}")
    
    @classmethod
    def load_package(cls, filepath):
        """Load complete package"""
        package_data = joblib.load(filepath)
        
        package = cls(
            model=package_data['model'],
            preprocessor=package_data.get('preprocessor'),
            feature_names=package_data.get('feature_names', []),
            model_metadata=package_data.get('metadata', {})
        )
        
        print(f"Production package loaded from: {filepath}")
        return package

# Create production package
print("Production Model Package Demonstration:")

# Get pipeline model (has both preprocessor and classifier)
pipeline = models['pipeline_model']['model']
preprocessor = pipeline.named_steps['scaler']
classifier = pipeline.named_steps['classifier']

# Create production package
prod_package = ProductionModelPackage(
    model=classifier,
    preprocessor=preprocessor,
    feature_names=models['pipeline_model']['features'],
    model_metadata={
        'accuracy': models['pipeline_model']['accuracy'],
        'training_date': models['pipeline_model']['training_time'].isoformat(),
        'model_version': '1.0.0'
    }
)

# Save production package
prod_package.save_package('saved_models/production_package.pkl')

# Test the package
X_test_sample = models['pipeline_model']['X_test'][:5]
predictions = prod_package.predict(X_test_sample)
probabilities = prod_package.predict_proba(X_test_sample)

print(f"\nProduction Package Testing:")
print(f"  Sample predictions: {predictions}")
print(f"  Sample probabilities: {probabilities[0]}")

# Monitor package
monitoring_info = prod_package.get_monitoring_info()
print(f"\nMonitoring Information:")
for key, value in monitoring_info.items():
    if key != 'metadata':
        print(f"  {key}: {value}")

# Load package
loaded_package = ProductionModelPackage.load_package('saved_models/production_package.pkl')
loaded_predictions = loaded_package.predict(X_test_sample)

print(f"  Loaded predictions match: {np.array_equal(predictions, loaded_predictions)}")

print()

# 7. BEST PRACTICES AND VALIDATION
print("7. BEST PRACTICES AND VALIDATION")
print("-" * 35)

print("Theory: Implement validation and testing procedures")
print("to ensure model persistence works correctly.")
print()

def validate_model_persistence(original_model, X_test, y_test, save_path):
    """Comprehensive validation of model persistence"""
    
    validation_results = {
        'save_successful': False,
        'load_successful': False,
        'predictions_match': False,
        'performance_match': False,
        'file_size': None,
        'save_time': None,
        'load_time': None
    }
    
    try:
        # Test saving
        save_start = time.time()
        joblib.dump(original_model, save_path)
        validation_results['save_time'] = time.time() - save_start
        validation_results['save_successful'] = True
        validation_results['file_size'] = os.path.getsize(save_path)
        
        # Test loading
        load_start = time.time()
        loaded_model = joblib.load(save_path)
        validation_results['load_time'] = time.time() - load_start
        validation_results['load_successful'] = True
        
        # Test predictions
        original_pred = original_model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        validation_results['predictions_match'] = np.array_equal(original_pred, loaded_pred)
        
        # Test performance
        original_score = original_model.score(X_test, y_test)
        loaded_score = loaded_model.score(X_test, y_test)
        validation_results['performance_match'] = abs(original_score - loaded_score) < 1e-10
        
        validation_results['original_accuracy'] = original_score
        validation_results['loaded_accuracy'] = loaded_score
        
    except Exception as e:
        validation_results['error'] = str(e)
    
    return validation_results

# Validate all saved models
print("Model Persistence Validation:")
print("-" * 30)

validation_summary = {}

for model_name, model_info in models.items():
    save_path = f'saved_models/validation_{model_name}.pkl'
    
    validation_result = validate_model_persistence(
        model_info['model'],
        model_info['X_test'],
        model_info['y_test'],
        save_path
    )
    
    validation_summary[model_name] = validation_result
    
    print(f"\n{model_name.replace('_', ' ').title()}:")
    print(f"  Save successful: {validation_result['save_successful']}")
    print(f"  Load successful: {validation_result['load_successful']}")
    print(f"  Predictions match: {validation_result['predictions_match']}")
    print(f"  Performance match: {validation_result['performance_match']}")
    print(f"  File size: {validation_result['file_size']} bytes")
    print(f"  Save time: {validation_result['save_time']:.4f}s")
    print(f"  Load time: {validation_result['load_time']:.4f}s")
    
    if 'error' in validation_result:
        print(f"  Error: {validation_result['error']}")

print()

# 8. PERFORMANCE AND SIZE ANALYSIS
print("8. PERFORMANCE AND SIZE ANALYSIS")
print("-" * 35)

# Analyze file sizes and performance
plt.figure(figsize=(15, 10))

# Model file sizes
plt.subplot(2, 3, 1)
model_names = list(validation_summary.keys())
file_sizes = [validation_summary[name]['file_size'] for name in model_names]

bars = plt.bar(range(len(model_names)), file_sizes, alpha=0.7)
plt.xticks(range(len(model_names)), [name.replace('_', '\n') for name in model_names], rotation=45)
plt.ylabel('File Size (bytes)')
plt.title('Model File Sizes')
plt.grid(True, alpha=0.3)

# Save/Load times
plt.subplot(2, 3, 2)
save_times = [validation_summary[name]['save_time'] for name in model_names]
load_times = [validation_summary[name]['load_time'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, save_times, width, label='Save Time', alpha=0.8)
plt.bar(x + width/2, load_times, width, label='Load Time', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Time (seconds)')
plt.title('Save/Load Performance')
plt.xticks(x, [name.replace('_', '\n') for name in model_names], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Accuracy comparison
plt.subplot(2, 3, 3)
original_accs = [validation_summary[name]['original_accuracy'] for name in model_names]
loaded_accs = [validation_summary[name]['loaded_accuracy'] for name in model_names]

plt.scatter(original_accs, loaded_accs, s=100, alpha=0.7)
plt.plot([min(original_accs), max(original_accs)], [min(original_accs), max(original_accs)], 'r--', alpha=0.8)

for i, name in enumerate(model_names):
    plt.annotate(name.replace('_', ' '), (original_accs[i], loaded_accs[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('Original Model Accuracy')
plt.ylabel('Loaded Model Accuracy')
plt.title('Accuracy Preservation')
plt.grid(True, alpha=0.3)

# File size vs performance
plt.subplot(2, 3, 4)
plt.scatter(file_sizes, original_accs, s=100, alpha=0.7)

for i, name in enumerate(model_names):
    plt.annotate(name.replace('_', ' '), (file_sizes[i], original_accs[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('File Size (bytes)')
plt.ylabel('Model Accuracy')
plt.title('File Size vs Performance')
plt.grid(True, alpha=0.3)

# Serialization method comparison (from earlier)
plt.subplot(2, 3, 5)
methods = ['Joblib', 'Pickle', 'Dill']
save_times_methods = [joblib_save_time, pickle_save_time, dill_save_time]
load_times_methods = [joblib_load_time, pickle_load_time, dill_load_time]

x = np.arange(len(methods))
plt.bar(x - 0.2, save_times_methods, 0.4, label='Save', alpha=0.8)
plt.bar(x + 0.2, load_times_methods, 0.4, label='Load', alpha=0.8)

plt.xlabel('Serialization Method')
plt.ylabel('Time (seconds)')
plt.title('Serialization Method Performance')
plt.xticks(x, methods)
plt.legend()
plt.grid(True, alpha=0.3)

# Model complexity vs file size
plt.subplot(2, 3, 6)
complexities = []
for name in model_names:
    if 'pipeline' in name:
        complexities.append(3)  # Pipeline complexity
    elif 'grid_search' in name:
        complexities.append(4)  # Grid search complexity
    else:
        complexities.append(2)  # Simple model complexity

plt.scatter(complexities, file_sizes, s=100, alpha=0.7)

for i, name in enumerate(model_names):
    plt.annotate(name.replace('_', ' '), (complexities[i], file_sizes[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('Model Complexity (arbitrary scale)')
plt.ylabel('File Size (bytes)')
plt.title('Model Complexity vs File Size')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print()

# 9. PRODUCTION DEPLOYMENT CHECKLIST
print("9. PRODUCTION DEPLOYMENT CHECKLIST")
print("-" * 38)

deployment_checklist = {
    "Model Validation": [
        "✓ Verify predictions match between original and loaded models",
        "✓ Validate performance metrics are preserved",
        "✓ Test with various input data shapes and types",
        "✓ Check for any serialization warnings or errors"
    ],
    "Dependency Management": [
        "✓ Document exact library versions used",
        "✓ Test loading in clean environment",
        "✓ Verify compatibility across Python versions",
        "✓ Include requirements.txt with model artifacts"
    ],
    "File Management": [
        "✓ Use checksums for file integrity verification",
        "✓ Implement backup and recovery procedures",
        "✓ Set up file permission and access controls",
        "✓ Plan for model artifact storage and retrieval"
    ],
    "Monitoring Setup": [
        "✓ Track model loading times and failures",
        "✓ Monitor prediction latency and throughput",
        "✓ Log model version and deployment information",
        "✓ Set up alerts for model loading issues"
    ],
    "Version Control": [
        "✓ Implement model versioning strategy",
        "✓ Maintain rollback capabilities",
        "✓ Document model changes and improvements",
        "✓ Test deployment and rollback procedures"
    ]
}

print("Production Deployment Checklist:")
for category, items in deployment_checklist.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  {item}")

print()

# 10. SUMMARY AND RECOMMENDATIONS
print("10. SUMMARY AND RECOMMENDATIONS")
print("-" * 35)

print(f"{'='*60}")
print("MODEL PERSISTENCE SUMMARY")
print(f"{'='*60}")

summary_points = [
    "Use joblib for scikit-learn models (optimized for NumPy arrays)",
    "Save complete pipelines, not just final models",
    "Include comprehensive metadata with saved models",
    "Implement model versioning for production systems",
    "Validate model persistence with comprehensive testing",
    "Monitor file sizes and loading performance",
    "Plan for dependency management and compatibility",
    "Use checksums for file integrity verification",
    "Implement proper error handling and rollback procedures",
    "Document model artifacts and deployment procedures"
]

for i, point in enumerate(summary_points, 1):
    print(f"{i:2d}. {point}")

print(f"\n{'='*60}")
print("BEST PRACTICES BY USE CASE")
print(f"{'='*60}")

use_case_recommendations = {
    "Research and Experimentation": [
        "Use simple joblib.dump() for quick model saving",
        "Include basic metadata in filenames",
        "Focus on reproducibility with random seeds"
    ],
    "Team Collaboration": [
        "Use comprehensive metadata and documentation",
        "Include requirements.txt with model artifacts",
        "Implement consistent naming conventions"
    ],
    "Production Deployment": [
        "Use versioned model management systems",
        "Implement monitoring and health checks",
        "Plan for rollback and disaster recovery",
        "Use production-ready packaging classes"
    ],
    "Model Serving at Scale": [
        "Optimize model loading times",
        "Use model caching strategies",
        "Implement lazy loading for large models",
        "Consider model compression techniques"
    ]
}

for use_case, recommendations in use_case_recommendations.items():
    print(f"\n{use_case}:")
    for rec in recommendations:
        print(f"  • {rec}")

# Clean up created files
print(f"\nCleaning up demonstration files...")
import shutil

# Remove directories created during demonstration
directories_to_remove = ['saved_models', 'versioned_models']
for directory in directories_to_remove:
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"  Removed: {directory}")

print(f"\n=== Model Persistence: Foundation of ML Production ===")
```

### Explanation

Model persistence is essential for deploying machine learning models to production, enabling model sharing, and avoiding costly retraining. It involves saving trained models and their preprocessing pipelines to disk for later use.

### Key Components of Model Persistence

1. **Model Serialization**
   - **Joblib**: Optimized for NumPy arrays and scikit-learn models
   - **Pickle**: Standard Python serialization
   - **Dill**: Enhanced pickle with broader object support

2. **Complete Pipeline Persistence**
   - Save preprocessing steps alongside models
   - Maintain data transformation consistency
   - Ensure reproducible predictions

3. **Metadata Management**
   - Model version and training information
   - Feature names and data schema
   - Performance metrics and validation results

### Serialization Methods Comparison

**Joblib (Recommended for ML):**
- Optimized for NumPy arrays and scikit-learn
- Efficient compression for large arrays
- De facto standard for ML model persistence

**Pickle (Standard Python):**
- Built into Python standard library
- Works with any Python object
- Good for simple models and objects

**Dill (Enhanced Pickle):**
- Can serialize more complex objects
- Useful for custom classes and functions
- Larger file sizes but broader compatibility

### Production Considerations

1. **Model Packaging**
   - Include preprocessing, model, and metadata
   - Implement validation and monitoring
   - Create production-ready interfaces

2. **Version Management**
   - Track model versions and changes
   - Enable rollback capabilities
   - Support A/B testing and gradual deployments

3. **Validation and Testing**
   - Verify predictions match original model
   - Test loading in clean environments
   - Validate performance preservation

### Best Practices

**For Research:**
- Use simple joblib.dump() for quick saving
- Include basic metadata in filenames
- Focus on reproducibility

**For Production:**
- Implement comprehensive versioning
- Include monitoring and health checks
- Plan for rollback procedures
- Use production-ready packaging

**For Team Collaboration:**
- Document dependencies and requirements
- Use consistent naming conventions
- Include comprehensive metadata

### Common Pitfalls

- **Missing Dependencies**: Not documenting required libraries
- **Version Incompatibility**: Loading models with different library versions
- **Incomplete Pipelines**: Saving only models without preprocessing
- **No Validation**: Not testing loaded models against originals
- **Poor Organization**: Lack of versioning and metadata

### Deployment Checklist

1. **Model Validation**: Verify predictions and performance
2. **Dependency Management**: Document exact versions
3. **File Management**: Implement integrity checks
4. **Monitoring Setup**: Track loading and performance
5. **Version Control**: Enable rollback capabilities

### Advanced Features

- **Model Compression**: Reduce file sizes for faster loading
- **Lazy Loading**: Load model components on demand
- **Caching Strategies**: Keep frequently used models in memory
- **Health Monitoring**: Track model performance and degradation

Model persistence is the foundation of ML production systems, enabling reliable deployment, sharing, and maintenance of machine learning models while ensuring consistency and reproducibility across different environments.

---

## Question 11

**Discuss the impact of the GIL (Global Interpreter Lock) on Python concurrency in machine learning applications.**

### Theory
The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecodes simultaneously. While this simplifies memory management, it can limit performance in CPU-bound tasks but doesn't affect I/O-bound operations or C extensions that release the GIL.

### Answer

```python
# gil_impact_ml.py - Understanding GIL impact on ML applications
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import os
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import asyncio
import aiofiles
import warnings
warnings.filterwarnings('ignore')

print("=== GIL Impact on Python Concurrency in ML Applications ===\n")

# 1. UNDERSTANDING THE GIL
print("1. UNDERSTANDING THE GIL")
print("-" * 25)

print("Theory: The GIL ensures thread safety but limits true parallelism")
print("for CPU-bound tasks. However, it doesn't affect I/O operations or")
print("C extensions that properly release the GIL.")
print()

def demonstrate_gil_basics():
    """Demonstrate basic GIL behavior"""
    
    print("GIL Behavior Demonstration:")
    
    # CPU-bound task (affected by GIL)
    def cpu_bound_task(n):
        """Pure Python CPU-bound computation"""
        result = 0
        for i in range(n):
            result += i ** 2
        return result
    
    # I/O-bound task (not significantly affected by GIL)
    def io_bound_task(duration):
        """Simulated I/O operation"""
        time.sleep(duration)
        return f"I/O completed after {duration}s"
    
    # Test CPU-bound task
    print("\nA. CPU-bound Task Performance:")
    
    n = 1000000
    
    # Sequential execution
    start_time = time.time()
    results_sequential = [cpu_bound_task(n) for _ in range(4)]
    sequential_time = time.time() - start_time
    
    # Threading (limited by GIL)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results_threaded = list(executor.map(cpu_bound_task, [n] * 4))
    threaded_time = time.time() - start_time
    
    print(f"  Sequential time: {sequential_time:.3f} seconds")
    print(f"  Threaded time: {threaded_time:.3f} seconds")
    print(f"  Threading speedup: {sequential_time / threaded_time:.2f}x")
    print("  → Threading provides little benefit for CPU-bound tasks due to GIL")
    
    # Test I/O-bound task
    print(f"\nB. I/O-bound Task Performance:")
    
    duration = 0.1
    
    # Sequential execution
    start_time = time.time()
    results_sequential_io = [io_bound_task(duration) for _ in range(4)]
    sequential_time_io = time.time() - start_time
    
    # Threading (benefits from GIL release during I/O)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results_threaded_io = list(executor.map(io_bound_task, [duration] * 4))
    threaded_time_io = time.time() - start_time
    
    print(f"  Sequential time: {sequential_time_io:.3f} seconds")
    print(f"  Threaded time: {threaded_time_io:.3f} seconds")
    print(f"  Threading speedup: {sequential_time_io / threaded_time_io:.2f}x")
    print("  → Threading provides significant benefit for I/O-bound tasks")

demonstrate_gil_basics()
print()

# 2. GIL IMPACT ON ML WORKLOADS
print("2. GIL IMPACT ON ML WORKLOADS")
print("-" * 30)

print("Theory: Different ML operations are affected differently by the GIL.")
print("NumPy/scikit-learn operations often release GIL, while pure Python")
print("computations are constrained by it.")
print()

# Create sample datasets for testing
def create_ml_datasets():
    """Create datasets for ML GIL testing"""
    
    # Classification dataset
    X_class, y_class = make_classification(
        n_samples=10000, n_features=20, n_informative=10,
        n_redundant=5, n_clusters_per_class=1, random_state=42
    )
    
    # Regression dataset  
    X_reg, y_reg = make_regression(
        n_samples=10000, n_features=20, noise=0.1, random_state=42
    )
    
    return X_class, y_class, X_reg, y_reg

X_class, y_class, X_reg, y_reg = create_ml_datasets()
print(f"Dataset created: {X_class.shape[0]} samples, {X_class.shape[1]} features")

# A. Model Training Performance
print(f"\nA. MODEL TRAINING PERFORMANCE")
print("-" * 33)

def train_model_task(task_id, X, y, model_type='rf'):
    """Train a model - task for parallelization testing"""
    
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=task_id, n_jobs=1)
    elif model_type == 'lr':
        model = LogisticRegression(random_state=task_id, max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=50, random_state=task_id, n_jobs=1)
    
    start_time = time.time()
    model.fit(X, y)
    training_time = time.time() - start_time
    
    return {
        'task_id': task_id,
        'model_type': model_type,
        'training_time': training_time,
        'accuracy': model.score(X, y)
    }

# Test different parallelization approaches for model training
def test_model_training_parallelization():
    """Test model training with different parallelization approaches"""
    
    print("Model Training Parallelization Test:")
    
    n_models = 4
    models_to_test = ['rf', 'lr', 'rf', 'lr']
    
    # Sequential training
    print(f"  Training {n_models} models sequentially...")
    start_time = time.time()
    sequential_results = []
    for i, model_type in enumerate(models_to_test):
        result = train_model_task(i, X_class, y_class, model_type)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    # Threading (limited by GIL for pure Python, but NumPy operations may benefit)
    print(f"  Training {n_models} models with threading...")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        threaded_futures = []
        for i, model_type in enumerate(models_to_test):
            future = executor.submit(train_model_task, i, X_class, y_class, model_type)
            threaded_futures.append(future)
        threaded_results = [future.result() for future in threaded_futures]
    threaded_time = time.time() - start_time
    
    # Multiprocessing (bypasses GIL)
    print(f"  Training {n_models} models with multiprocessing...")
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        process_futures = []
        for i, model_type in enumerate(models_to_test):
            future = executor.submit(train_model_task, i, X_class, y_class, model_type)
            process_futures.append(future)
        process_results = [future.result() for future in process_futures]
    process_time = time.time() - start_time
    
    print(f"\n  Results:")
    print(f"    Sequential time: {sequential_time:.3f} seconds")
    print(f"    Threading time: {threaded_time:.3f} seconds")
    print(f"    Multiprocessing time: {process_time:.3f} seconds")
    print(f"    Threading speedup: {sequential_time / threaded_time:.2f}x")
    print(f"    Multiprocessing speedup: {sequential_time / process_time:.2f}x")
    
    return {
        'sequential_time': sequential_time,
        'threaded_time': threaded_time,
        'process_time': process_time
    }

training_results = test_model_training_parallelization()
print()

# B. Cross-validation Performance
print(f"B. CROSS-VALIDATION PERFORMANCE")
print("-" * 35)

def cv_task(task_id, X, y, cv_folds=5):
    """Perform cross-validation - task for parallelization testing"""
    
    model = RandomForestClassifier(n_estimators=50, random_state=task_id, n_jobs=1)
    
    start_time = time.time()
    cv_scores = cross_val_score(model, X, y, cv=cv_folds)
    cv_time = time.time() - start_time
    
    return {
        'task_id': task_id,
        'cv_time': cv_time,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

def test_cv_parallelization():
    """Test cross-validation with different parallelization approaches"""
    
    print("Cross-validation Parallelization Test:")
    
    n_cv_tasks = 4
    
    # Sequential CV
    print(f"  Running {n_cv_tasks} CV tasks sequentially...")
    start_time = time.time()
    sequential_cv = [cv_task(i, X_class, y_class) for i in range(n_cv_tasks)]
    sequential_cv_time = time.time() - start_time
    
    # Threading CV
    print(f"  Running {n_cv_tasks} CV tasks with threading...")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        threaded_cv = list(executor.map(lambda i: cv_task(i, X_class, y_class), range(n_cv_tasks)))
    threaded_cv_time = time.time() - start_time
    
    # Multiprocessing CV
    print(f"  Running {n_cv_tasks} CV tasks with multiprocessing...")
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        process_cv = list(executor.map(lambda i: cv_task(i, X_class, y_class), range(n_cv_tasks)))
    process_cv_time = time.time() - start_time
    
    print(f"\n  Results:")
    print(f"    Sequential time: {sequential_cv_time:.3f} seconds")
    print(f"    Threading time: {threaded_cv_time:.3f} seconds")
    print(f"    Multiprocessing time: {process_cv_time:.3f} seconds")
    print(f"    Threading speedup: {sequential_cv_time / threaded_cv_time:.2f}x")
    print(f"    Multiprocessing speedup: {sequential_cv_time / process_cv_time:.2f}x")
    
    return {
        'sequential_cv_time': sequential_cv_time,
        'threaded_cv_time': threaded_cv_time,
        'process_cv_time': process_cv_time
    }

cv_results = test_cv_parallelization()
print()

# 3. SCIKIT-LEARN AND GIL INTERACTION
print("3. SCIKIT-LEARN AND GIL INTERACTION")
print("-" * 36)

print("Theory: Scikit-learn algorithms often release the GIL during")
print("computationally intensive operations, allowing for better")
print("threading performance than pure Python code.")
print()

def test_sklearn_gil_behavior():
    """Test how scikit-learn algorithms interact with GIL"""
    
    print("Scikit-learn GIL Behavior Analysis:")
    
    # Test different algorithms with n_jobs parameter
    algorithms = {
        'Random Forest (n_jobs=1)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        'Random Forest (n_jobs=-1)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Test training times
    print(f"\n  Algorithm Training Performance:")
    algorithm_results = {}
    
    for name, algorithm in algorithms.items():
        start_time = time.time()
        algorithm.fit(X_class, y_class)
        training_time = time.time() - start_time
        accuracy = algorithm.score(X_class, y_class)
        
        algorithm_results[name] = {
            'training_time': training_time,
            'accuracy': accuracy
        }
        
        print(f"    {name}:")
        print(f"      Training time: {training_time:.3f} seconds")
        print(f"      Accuracy: {accuracy:.4f}")
    
    # Compare n_jobs effect
    rf_single = algorithm_results['Random Forest (n_jobs=1)']['training_time']
    rf_parallel = algorithm_results['Random Forest (n_jobs=-1)']['training_time']
    speedup = rf_single / rf_parallel
    
    print(f"\n  n_jobs Parameter Impact:")
    print(f"    Single thread (n_jobs=1): {rf_single:.3f} seconds")
    print(f"    Parallel (n_jobs=-1): {rf_parallel:.3f} seconds")
    print(f"    Speedup: {speedup:.2f}x")
    print(f"    → n_jobs bypasses GIL through internal parallelization")
    
    return algorithm_results

sklearn_results = test_sklearn_gil_behavior()
print()

# 4. DATA PROCESSING AND GIL
print("4. DATA PROCESSING AND GIL")
print("-" * 27)

print("Theory: Data preprocessing and feature engineering tasks")
print("may be affected differently by GIL depending on whether")
print("they use NumPy/Pandas (C extensions) or pure Python.")
print()

def test_data_processing_gil():
    """Test GIL impact on data processing operations"""
    
    print("Data Processing GIL Impact Analysis:")
    
    # Create larger dataset for processing
    large_data = pd.DataFrame(np.random.randn(50000, 10), 
                             columns=[f'feature_{i}' for i in range(10)])
    
    # Pure Python processing function
    def python_processing(data_chunk):
        """Pure Python data processing (affected by GIL)"""
        result = []
        for _, row in data_chunk.iterrows():
            # Pure Python computation
            processed_row = []
            for value in row:
                processed_value = value ** 2 + np.sin(value)
                processed_row.append(processed_value)
            result.append(processed_row)
        return np.array(result)
    
    # NumPy/Pandas processing function
    def numpy_processing(data_chunk):
        """NumPy-based processing (can release GIL)"""
        return data_chunk.values ** 2 + np.sin(data_chunk.values)
    
    # Split data into chunks
    chunk_size = len(large_data) // 4
    data_chunks = [large_data.iloc[i:i+chunk_size] for i in range(0, len(large_data), chunk_size)]
    
    # Test pure Python processing
    print(f"\n  Pure Python Processing:")
    
    # Sequential
    start_time = time.time()
    python_sequential = [python_processing(chunk) for chunk in data_chunks]
    python_seq_time = time.time() - start_time
    
    # Threading
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        python_threaded = list(executor.map(python_processing, data_chunks))
    python_thread_time = time.time() - start_time
    
    print(f"    Sequential: {python_seq_time:.3f} seconds")
    print(f"    Threading: {python_thread_time:.3f} seconds")
    print(f"    Threading speedup: {python_seq_time / python_thread_time:.2f}x")
    
    # Test NumPy processing
    print(f"\n  NumPy Processing:")
    
    # Sequential
    start_time = time.time()
    numpy_sequential = [numpy_processing(chunk) for chunk in data_chunks]
    numpy_seq_time = time.time() - start_time
    
    # Threading
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        numpy_threaded = list(executor.map(numpy_processing, data_chunks))
    numpy_thread_time = time.time() - start_time
    
    print(f"    Sequential: {numpy_seq_time:.3f} seconds")
    print(f"    Threading: {numpy_thread_time:.3f} seconds")
    print(f"    Threading speedup: {numpy_seq_time / numpy_thread_time:.2f}x")
    
    print(f"\n  Key Insights:")
    print(f"    → Pure Python: Limited threading benefit due to GIL")
    print(f"    → NumPy operations: Better threading performance")
    print(f"    → NumPy is {numpy_seq_time / python_seq_time:.1f}x faster overall")
    
    return {
        'python_seq_time': python_seq_time,
        'python_thread_time': python_thread_time,
        'numpy_seq_time': numpy_seq_time,
        'numpy_thread_time': numpy_thread_time
    }

processing_results = test_data_processing_gil()
print()

# 5. ASYNC PROGRAMMING IN ML
print("5. ASYNC PROGRAMMING IN ML")
print("-" * 26)

print("Theory: Async programming can be beneficial for ML workflows")
print("involving I/O operations like data loading, model serving,")
print("and API calls, even with the GIL.")
print()

async def async_model_prediction(model, X_batch, delay=0.01):
    """Async model prediction with simulated I/O delay"""
    
    # Simulate I/O operation (e.g., logging, database write)
    await asyncio.sleep(delay)
    
    # Make prediction
    predictions = model.predict(X_batch)
    
    # Simulate another I/O operation
    await asyncio.sleep(delay)
    
    return predictions

async def test_async_ml_serving():
    """Test async ML model serving"""
    
    print("Async ML Model Serving Test:")
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_class, y_class)
    
    # Create test batches
    X_test = np.random.randn(100, 20)
    batch_size = 25
    batches = [X_test[i:i+batch_size] for i in range(0, len(X_test), batch_size)]
    
    # Synchronous processing
    print(f"  Processing {len(batches)} batches synchronously...")
    start_time = time.time()
    sync_results = []
    for batch in batches:
        time.sleep(0.02)  # Simulate I/O delay
        result = model.predict(batch)
        time.sleep(0.02)  # Simulate I/O delay
        sync_results.append(result)
    sync_time = time.time() - start_time
    
    # Asynchronous processing
    print(f"  Processing {len(batches)} batches asynchronously...")
    start_time = time.time()
    async_tasks = [async_model_prediction(model, batch) for batch in batches]
    async_results = await asyncio.gather(*async_tasks)
    async_time = time.time() - start_time
    
    print(f"\n  Results:")
    print(f"    Synchronous time: {sync_time:.3f} seconds")
    print(f"    Asynchronous time: {async_time:.3f} seconds")
    print(f"    Async speedup: {sync_time / async_time:.2f}x")
    print(f"    → Async provides benefit when I/O operations are involved")
    
    return sync_time, async_time

# Run async test
async_sync_time, async_async_time = asyncio.run(test_async_ml_serving())
print()

# 6. WORKAROUNDS AND SOLUTIONS
print("6. WORKAROUNDS AND SOLUTIONS")
print("-" * 31)

print("Theory: Several strategies can mitigate GIL limitations")
print("in ML applications, each with specific use cases and trade-offs.")
print()

def demonstrate_gil_workarounds():
    """Demonstrate various GIL workaround strategies"""
    
    print("GIL Workaround Strategies:")
    
    # Strategy 1: Use libraries that release GIL
    print(f"\n  Strategy 1: Use GIL-releasing Libraries")
    print(f"  " + "-" * 40)
    
    # NumPy operations that release GIL
    large_array = np.random.randn(10000, 100)
    
    def numpy_computation(array_chunk):
        """NumPy computation that can release GIL"""
        return np.linalg.svd(array_chunk, full_matrices=False)
    
    # Split array for parallel processing
    chunks = np.array_split(large_array, 4)
    
    # Test NumPy threading performance
    start_time = time.time()
    sequential_numpy = [numpy_computation(chunk) for chunk in chunks]
    numpy_seq_time = time.time() - start_time
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        threaded_numpy = list(executor.map(numpy_computation, chunks))
    numpy_thread_time = time.time() - start_time
    
    print(f"    NumPy sequential: {numpy_seq_time:.3f} seconds")
    print(f"    NumPy threading: {numpy_thread_time:.3f} seconds")
    print(f"    Threading speedup: {numpy_seq_time / numpy_thread_time:.2f}x")
    
    # Strategy 2: Multiprocessing
    print(f"\n  Strategy 2: Multiprocessing")
    print(f"  " + "-" * 27)
    
    def cpu_intensive_ml_task(task_data):
        """CPU-intensive ML task for multiprocessing"""
        X, y, n_estimators = task_data
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=1)
        model.fit(X, y)
        return model.score(X, y)
    
    # Prepare tasks
    mp_tasks = [(X_class, y_class, 25) for _ in range(4)]
    
    # Sequential
    start_time = time.time()
    mp_sequential = [cpu_intensive_ml_task(task) for task in mp_tasks]
    mp_seq_time = time.time() - start_time
    
    # Multiprocessing
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        mp_parallel = list(executor.map(cpu_intensive_ml_task, mp_tasks))
    mp_parallel_time = time.time() - start_time
    
    print(f"    Sequential: {mp_seq_time:.3f} seconds")
    print(f"    Multiprocessing: {mp_parallel_time:.3f} seconds")
    print(f"    Multiprocessing speedup: {mp_seq_time / mp_parallel_time:.2f}x")
    
    # Strategy 3: Joblib parallel backends
    print(f"\n  Strategy 3: Joblib Parallel Backends")
    print(f"  " + "-" * 37)
    
    from joblib import Parallel, delayed
    
    def ml_task_joblib(task_id):
        """ML task for joblib parallelization"""
        model = RandomForestClassifier(n_estimators=25, random_state=task_id, n_jobs=1)
        model.fit(X_class, y_class)
        return model.score(X_class, y_class)
    
    # Threading backend
    start_time = time.time()
    joblib_threaded = Parallel(n_jobs=4, backend='threading')(
        delayed(ml_task_joblib)(i) for i in range(4)
    )
    joblib_thread_time = time.time() - start_time
    
    # Multiprocessing backend
    start_time = time.time()
    joblib_mp = Parallel(n_jobs=4, backend='multiprocessing')(
        delayed(ml_task_joblib)(i) for i in range(4)
    )
    joblib_mp_time = time.time() - start_time
    
    print(f"    Joblib threading: {joblib_thread_time:.3f} seconds")
    print(f"    Joblib multiprocessing: {joblib_mp_time:.3f} seconds")
    print(f"    → Joblib provides easy backend switching")
    
    return {
        'numpy_threading_speedup': numpy_seq_time / numpy_thread_time,
        'multiprocessing_speedup': mp_seq_time / mp_parallel_time,
        'joblib_thread_time': joblib_thread_time,
        'joblib_mp_time': joblib_mp_time
    }

workaround_results = demonstrate_gil_workarounds()
print()

# 7. PERFORMANCE ANALYSIS AND VISUALIZATION
print("7. PERFORMANCE ANALYSIS AND VISUALIZATION")
print("-" * 45)

# Collect all timing results for analysis
performance_data = {
    'Model Training': {
        'Sequential': training_results['sequential_time'],
        'Threading': training_results['threaded_time'],
        'Multiprocessing': training_results['process_time']
    },
    'Cross Validation': {
        'Sequential': cv_results['sequential_cv_time'],
        'Threading': cv_results['threaded_cv_time'],
        'Multiprocessing': cv_results['process_cv_time']
    },
    'Data Processing (Python)': {
        'Sequential': processing_results['python_seq_time'],
        'Threading': processing_results['python_thread_time'],
        'Multiprocessing': None  # Not tested
    },
    'Data Processing (NumPy)': {
        'Sequential': processing_results['numpy_seq_time'],
        'Threading': processing_results['numpy_thread_time'],
        'Multiprocessing': None  # Not tested
    },
    'Async Model Serving': {
        'Synchronous': async_sync_time,
        'Asynchronous': async_async_time,
        'Multiprocessing': None  # Not applicable
    }
}

# Create visualization
plt.figure(figsize=(16, 12))

# Speedup comparison
plt.subplot(2, 3, 1)
tasks = []
threading_speedups = []
mp_speedups = []

for task, times in performance_data.items():
    if times['Threading'] and times['Sequential']:
        tasks.append(task.replace(' ', '\n'))
        threading_speedups.append(times['Sequential'] / times['Threading'])
        
        if times['Multiprocessing']:
            mp_speedups.append(times['Sequential'] / times['Multiprocessing'])
        else:
            mp_speedups.append(0)

x = np.arange(len(tasks))
width = 0.35

bars1 = plt.bar(x - width/2, threading_speedups, width, label='Threading', alpha=0.8)
bars2 = plt.bar(x + width/2, mp_speedups, width, label='Multiprocessing', alpha=0.8)

plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
plt.xlabel('Task Type')
plt.ylabel('Speedup Factor')
plt.title('GIL Impact: Threading vs Multiprocessing Speedup')
plt.xticks(x, tasks, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

# Execution time comparison
plt.subplot(2, 3, 2)
sequential_times = [performance_data[task]['Sequential'] for task in ['Model Training', 'Cross Validation']]
threaded_times = [performance_data[task]['Threading'] for task in ['Model Training', 'Cross Validation']]
mp_times = [performance_data[task]['Multiprocessing'] for task in ['Model Training', 'Cross Validation']]

task_labels = ['Model\nTraining', 'Cross\nValidation']
x = np.arange(len(task_labels))

plt.bar(x - 0.25, sequential_times, 0.25, label='Sequential', alpha=0.8)
plt.bar(x, threaded_times, 0.25, label='Threading', alpha=0.8)
plt.bar(x + 0.25, mp_times, 0.25, label='Multiprocessing', alpha=0.8)

plt.xlabel('Task Type')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison')
plt.xticks(x, task_labels)
plt.legend()
plt.grid(True, alpha=0.3)

# GIL effect on different operations
plt.subplot(2, 3, 3)
operations = ['Pure Python\nProcessing', 'NumPy\nProcessing', 'ML Training\n(scikit-learn)']
gil_effects = [
    processing_results['python_seq_time'] / processing_results['python_thread_time'],
    processing_results['numpy_seq_time'] / processing_results['numpy_thread_time'],
    training_results['sequential_time'] / training_results['threaded_time']
]

colors = ['red' if effect < 1.5 else 'orange' if effect < 2.5 else 'green' for effect in gil_effects]
bars = plt.bar(operations, gil_effects, color=colors, alpha=0.7)

plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No benefit')
plt.axhline(y=2, color='blue', linestyle='--', alpha=0.5, label='Good parallel efficiency')
plt.ylabel('Threading Speedup Factor')
plt.title('GIL Effect on Different Operations')
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, effect in zip(bars, gil_effects):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{effect:.2f}x', ha='center', va='bottom')

# CPU core utilization simulation
plt.subplot(2, 3, 4)
cores = np.arange(1, 5)
ideal_speedup = cores
threading_speedup_sim = [1, 1.2, 1.3, 1.4]  # Limited by GIL
mp_speedup_sim = [1, 1.8, 2.7, 3.5]  # Near-linear scaling

plt.plot(cores, ideal_speedup, 'k--', label='Ideal Speedup', linewidth=2)
plt.plot(cores, threading_speedup_sim, 'r-o', label='Threading (GIL limited)', linewidth=2)
plt.plot(cores, mp_speedup_sim, 'g-s', label='Multiprocessing', linewidth=2)

plt.xlabel('Number of CPU Cores')
plt.ylabel('Speedup Factor')
plt.title('Scaling with CPU Cores')
plt.legend()
plt.grid(True, alpha=0.3)

# Memory vs speed trade-offs
plt.subplot(2, 3, 5)
approaches = ['Sequential', 'Threading', 'Multiprocessing', 'Async']
memory_usage = [1, 1.1, 3.5, 1.2]  # Relative memory usage
speed_improvement = [1, 1.3, 3.2, 2.1]  # Relative speed improvement

scatter = plt.scatter(memory_usage, speed_improvement, s=100, alpha=0.7)

for i, approach in enumerate(approaches):
    plt.annotate(approach, (memory_usage[i], speed_improvement[i]),
                xytext=(5, 5), textcoords='offset points')

plt.xlabel('Relative Memory Usage')
plt.ylabel('Relative Speed Improvement')
plt.title('Memory vs Speed Trade-offs')
plt.grid(True, alpha=0.3)

# Algorithm efficiency with parallel execution
plt.subplot(2, 3, 6)
sklearn_models = ['Random Forest\n(n_jobs=1)', 'Random Forest\n(n_jobs=-1)', 'Logistic\nRegression']
training_times = [
    sklearn_results['Random Forest (n_jobs=1)']['training_time'],
    sklearn_results['Random Forest (n_jobs=-1)']['training_time'],
    sklearn_results['Logistic Regression']['training_time']
]

colors = ['red', 'green', 'blue']
bars = plt.bar(sklearn_models, training_times, color=colors, alpha=0.7)

plt.ylabel('Training Time (seconds)')
plt.title('Scikit-learn Internal Parallelization')
plt.grid(True, alpha=0.3)

# Add time labels
for bar, time_val in zip(bars, training_times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{time_val:.3f}s', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print()

# 8. BEST PRACTICES AND RECOMMENDATIONS
print("8. BEST PRACTICES AND RECOMMENDATIONS")
print("-" * 42)

best_practices = {
    "Understanding GIL Impact": [
        "Profile your code to identify CPU vs I/O bound operations",
        "Measure actual performance gains, don't assume threading helps",
        "Use tools like cProfile and threading profilers",
        "Monitor CPU utilization to detect GIL bottlenecks"
    ],
    "Choosing Parallelization Strategy": [
        "Use threading for I/O-bound ML tasks (data loading, model serving)",
        "Use multiprocessing for CPU-bound tasks (training, inference)",
        "Leverage n_jobs parameter in scikit-learn algorithms",
        "Consider async programming for concurrent I/O operations"
    ],
    "Optimization Techniques": [
        "Use NumPy/Pandas operations that release the GIL",
        "Batch operations to reduce Python overhead",
        "Use compiled extensions (Cython, Numba) for critical paths",
        "Consider alternative Python implementations (PyPy, Jython)"
    ],
    "Production Considerations": [
        "Design stateless workers for easy multiprocessing",
        "Use message queues for distributed processing",
        "Monitor memory usage with multiprocessing",
        "Plan for graceful handling of worker failures"
    ],
    "Library-Specific Recommendations": [
        "Use joblib.Parallel for flexible backend selection",
        "Leverage Dask for larger-than-memory datasets",
        "Consider Ray for distributed ML workloads",
        "Use appropriate backend for different operations"
    ]
}

print("GIL-Aware ML Development Best Practices:")
for category, practices in best_practices.items():
    print(f"\n{category}:")
    for practice in practices:
        print(f"  • {practice}")

print()

# 9. DECISION FRAMEWORK
print("9. DECISION FRAMEWORK")
print("-" * 20)

decision_framework = {
    "Task Characteristics": {
        "CPU-bound, pure Python": ["Use multiprocessing", "Consider Cython/Numba"],
        "CPU-bound, NumPy/scikit-learn": ["Use n_jobs parameter", "Threading may help"],
        "I/O-bound operations": ["Use threading", "Consider async programming"],
        "Mixed workloads": ["Hybrid approach", "Profile each component"]
    },
    "Scale Requirements": {
        "Single machine, few cores": ["Threading for I/O", "n_jobs for ML algorithms"],
        "Single machine, many cores": ["Multiprocessing", "Process pools"],
        "Multiple machines": ["Distributed frameworks", "Message queues"],
        "Cloud/container environments": ["Horizontal scaling", "Serverless functions"]
    },
    "Performance Requirements": {
        "Low latency": ["Pre-trained models", "Async serving", "Caching"],
        "High throughput": ["Batch processing", "Multiprocessing", "Load balancing"],
        "Memory constrained": ["Threading over multiprocessing", "Streaming"],
        "CPU constrained": ["Multiprocessing", "Distributed computing"]
    }
}

print("GIL Impact Decision Framework:")
for criterion, scenarios in decision_framework.items():
    print(f"\n{criterion}:")
    for scenario, recommendations in scenarios.items():
        print(f"  {scenario}: {', '.join(recommendations)}")

print(f"\n{'='*60}")
print("SUMMARY: GIL Impact on ML Applications")
print(f"{'='*60}")

summary_points = [
    "GIL limits threading effectiveness for CPU-bound Python code",
    "NumPy/scikit-learn operations often release GIL, enabling better threading",
    "Multiprocessing bypasses GIL but has memory and startup overhead",
    "Use n_jobs parameter in scikit-learn for built-in parallelization",
    "Threading excels for I/O-bound operations (data loading, serving)",
    "Async programming helps with concurrent I/O without GIL issues",
    "Profile before optimizing - measure actual performance gains",
    "Consider memory usage trade-offs with multiprocessing",
    "Design for scalability with appropriate parallelization strategy",
    "Use appropriate tools (joblib, Dask, Ray) for different scales"
]

for i, point in enumerate(summary_points, 1):
    print(f"{i:2d}. {point}")

print(f"\n=== GIL: Understanding Limitations, Maximizing Performance ===")
```

### Explanation

The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecodes at once. While this ensures thread safety, it impacts concurrency in machine learning applications.

### Key GIL Impacts on ML

1. **CPU-bound Operations**
   - Pure Python computations are limited by GIL
   - Threading provides minimal benefit for CPU-intensive tasks
   - Multiprocessing bypasses GIL but has overhead

2. **I/O-bound Operations**
   - GIL is released during I/O operations
   - Threading provides significant benefits
   - Ideal for data loading, model serving, API calls

3. **NumPy/Scikit-learn Operations**
   - Many operations release GIL in C extensions
   - Threading can provide speedup for these operations
   - n_jobs parameter leverages internal parallelization

### GIL Interaction with ML Libraries

**Scikit-learn:**
- Algorithms with n_jobs parameter bypass GIL
- Random Forest, SVM, cross-validation support parallel execution
- Internal C implementations release GIL appropriately

**NumPy:**
- Mathematical operations release GIL
- Array operations can benefit from threading
- BLAS/LAPACK libraries handle parallelization

**Pandas:**
- Some operations release GIL
- I/O operations (reading files) benefit from threading
- Data processing may have mixed GIL behavior

### Strategies to Handle GIL

1. **Multiprocessing**
   - Bypass GIL completely with separate processes
   - Best for CPU-bound ML tasks
   - Higher memory usage and startup costs

2. **Threading**
   - Effective for I/O-bound operations
   - Good for model serving with I/O latency
   - Limited benefit for pure Python computation

3. **Async Programming**
   - Excellent for concurrent I/O operations
   - Model serving with database/API calls
   - Doesn't bypass GIL but handles I/O efficiently

4. **Library-specific Solutions**
   - Use n_jobs in scikit-learn algorithms
   - Leverage joblib for flexible parallelization
   - Consider Dask for larger-scale processing

### Performance Optimization Guidelines

**For Training:**
- Use n_jobs=-1 in scikit-learn algorithms
- Multiprocessing for multiple model training
- Consider distributed frameworks for large scale

**For Inference:**
- Threading for I/O-heavy serving applications
- Batch processing to amortize GIL overhead
- Async serving for high-concurrency applications

**For Data Processing:**
- Use NumPy/Pandas operations when possible
- Threading for file I/O and data loading
- Multiprocessing for CPU-intensive transformations

### Common Pitfalls

- **Assuming Threading Always Helps**: Profile first, measure actual gains
- **Ignoring Memory Overhead**: Multiprocessing can use significant memory
- **Over-parallelization**: Too many processes/threads can hurt performance
- **Not Using Library Features**: Missing n_jobs and built-in parallelization

### Best Practices

1. **Profile Before Optimizing**: Understand where time is spent
2. **Choose Right Tool**: Threading for I/O, multiprocessing for CPU
3. **Leverage Library Features**: Use n_jobs and parallel backends
4. **Design for Scale**: Consider distributed solutions for growth
5. **Monitor Resource Usage**: Track CPU, memory, and I/O utilization

### Production Considerations

- **Stateless Design**: Enable easy multiprocessing deployment
- **Error Handling**: Plan for worker process failures
- **Resource Monitoring**: Track memory usage with multiple processes
- **Load Balancing**: Distribute work effectively across resources

The GIL doesn't have to be a limitation if you understand its behavior and choose appropriate parallelization strategies for different types of ML workloads.

---

## Question 12

**Discuss the role of the collections module in managing data structures for machine learning.**

### Theory
Python's `collections` module provides specialized container datatypes beyond the basic built-in types (list, dict, tuple, set). These containers offer enhanced functionality and performance optimizations that are particularly valuable in machine learning workflows for data preprocessing, feature engineering, and model evaluation.

### Answer

```python
# collections_ml_applications.py - Comprehensive guide to collections module in ML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import (
    Counter, defaultdict, deque, namedtuple, OrderedDict, 
    ChainMap, UserDict, UserList, UserString
)
from sklearn.datasets import load_iris, make_classification, fetch_20newsgroups
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import memory_profiler
import sys
from typing import Dict, List, Tuple, Any
import json
import warnings
warnings.filterwarnings('ignore')

print("=== Python Collections Module in Machine Learning ===\n")

# Sample data creation
def create_ml_datasets():
    """Create various datasets for demonstrating collections usage"""
    
    # Classification dataset
    X_class, y_class = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_redundant=2, n_clusters_per_class=1, random_state=42
    )
    
    # Iris dataset
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    # Text data (subset for demo)
    try:
        newsgroups = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'sci.space'], 
                                      remove=('headers', 'footers', 'quotes'))
        text_data = newsgroups.data[:100]  # Small subset for demo
        text_labels = newsgroups.target[:100]
    except:
        # Fallback if fetch fails
        text_data = ["machine learning example", "data science project", "artificial intelligence"] * 30
        text_labels = [0, 1, 2] * 30
    
    return X_class, y_class, X_iris, y_iris, text_data, text_labels

X_class, y_class, X_iris, y_iris, text_data, text_labels = create_ml_datasets()

print("Dataset Information:")
print(f"Classification dataset: {X_class.shape[0]} samples, {X_class.shape[1]} features")
print(f"Iris dataset: {X_iris.shape[0]} samples, {X_iris.shape[1]} features")
print(f"Text dataset: {len(text_data)} documents")
print()

# 1. COUNTER - FREQUENCY ANALYSIS
print("1. COUNTER - FREQUENCY ANALYSIS")
print("-" * 35)

print("Theory: Counter is a dict subclass for counting hashable objects.")
print("Essential for exploratory data analysis and feature engineering.")
print()

# A. Label Distribution Analysis
print("A. Label Distribution Analysis:")

# Class distribution in classification dataset
label_counter = Counter(y_class)
print(f"Classification Dataset Label Distribution:")
for label, count in label_counter.most_common():
    percentage = (count / len(y_class)) * 100
    print(f"  Class {label}: {count:4d} samples ({percentage:5.1f}%)")

# Check for class imbalance
imbalance_ratio = max(label_counter.values()) / min(label_counter.values())
print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
if imbalance_ratio > 1.5:
    print("  ⚠️  Dataset shows class imbalance - consider resampling")
else:
    print("  ✓ Dataset is reasonably balanced")

# Iris dataset analysis
iris_counter = Counter(y_iris)
print(f"\nIris Dataset Label Distribution:")
iris_names = ['Setosa', 'Versicolor', 'Virginica']
for label, count in iris_counter.most_common():
    print(f"  {iris_names[label]}: {count} samples")

# B. Text Data Analysis with Counter
print(f"\nB. Text Data Analysis:")

# Word frequency analysis
all_words = []
for doc in text_data:
    # Simple tokenization
    words = doc.lower().replace('.', '').replace(',', '').split()
    all_words.extend(words)

word_counter = Counter(all_words)
print(f"Total unique words: {len(word_counter)}")
print(f"Most common words:")
for word, count in word_counter.most_common(10):
    print(f"  '{word}': {count}")

# Character frequency (useful for text preprocessing)
char_counter = Counter(''.join(text_data).lower())
print(f"\nCharacter frequency (top 10):")
for char, count in char_counter.most_common(10):
    if char.isalpha():
        print(f"  '{char}': {count}")

# C. Feature Value Distribution
print(f"\nC. Feature Value Distribution:")

# Analyze feature distributions in numerical data
feature_stats = {}
for i in range(X_class.shape[1]):
    feature_values = X_class[:, i]
    # Discretize for counting
    discretized = np.digitize(feature_values, bins=np.linspace(feature_values.min(), 
                                                             feature_values.max(), 6))
    feature_counter = Counter(discretized)
    feature_stats[f'Feature_{i}'] = feature_counter
    
    if i < 3:  # Show first 3 features
        print(f"  Feature {i} distribution (discretized):")
        for bin_idx, count in sorted(feature_counter.items()):
            print(f"    Bin {bin_idx}: {count} samples")

print()

# 2. DEFAULTDICT - AUTOMATIC INITIALIZATION
print("2. DEFAULTDICT - AUTOMATIC INITIALIZATION")
print("-" * 45)

print("Theory: defaultdict automatically creates missing keys with")
print("default values, simplifying data aggregation and grouping.")
print()

# A. Feature Grouping by Class
print("A. Feature Grouping by Class:")

# Group features by class using defaultdict
class_features = defaultdict(list)
for features, label in zip(X_iris, y_iris):
    class_features[label].extend(features)

print("Feature statistics by class:")
for class_label, feature_values in class_features.items():
    feature_array = np.array(feature_values).reshape(-1, 4)  # Iris has 4 features
    print(f"  Class {iris_names[class_label]}:")
    print(f"    Mean: {feature_array.mean(axis=0).round(2)}")
    print(f"    Std:  {feature_array.std(axis=0).round(2)}")

# B. Text Processing with defaultdict
print(f"\nB. Text Processing Applications:")

# Document term frequency by category
doc_terms = defaultdict(lambda: defaultdict(int))
for doc, label in zip(text_data, text_labels):
    words = doc.lower().split()
    for word in words:
        doc_terms[label][word] += 1

print("Top words by category:")
categories = ['Category_0', 'Category_1', 'Category_2']
for label in range(min(3, len(set(text_labels)))):
    if label in doc_terms:
        word_freq = doc_terms[label]
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  {categories[label]}: {dict(top_words)}")

# C. Model Performance Tracking
print(f"\nC. Model Performance Tracking:")

# Track model performance across different splits
performance_tracker = defaultdict(lambda: defaultdict(list))

# Simulate multiple experiments
for experiment in range(5):
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=experiment
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=experiment)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    performance_tracker['RandomForest']['train_scores'].append(train_score)
    performance_tracker['RandomForest']['test_scores'].append(test_score)

# Analyze results
rf_results = performance_tracker['RandomForest']
print("Model Performance Summary:")
print(f"  Train Score: {np.mean(rf_results['train_scores']):.4f} ± {np.std(rf_results['train_scores']):.4f}")
print(f"  Test Score:  {np.mean(rf_results['test_scores']):.4f} ± {np.std(rf_results['test_scores']):.4f}")

print()

# 3. DEQUE - EFFICIENT QUEUE OPERATIONS
print("3. DEQUE - EFFICIENT QUEUE OPERATIONS")
print("-" * 38)

print("Theory: deque provides O(1) append and pop operations from")
print("both ends, useful for sliding windows and batch processing.")
print()

# A. Sliding Window Feature Engineering
print("A. Sliding Window Feature Engineering:")

# Create time series data
time_series = np.random.randn(1000).cumsum()

def sliding_window_features(data, window_size=5):
    """Extract sliding window features using deque"""
    window = deque(maxlen=window_size)
    features = []
    
    for value in data:
        window.append(value)
        if len(window) == window_size:
            # Extract features from current window
            window_features = {
                'mean': np.mean(window),
                'std': np.std(window),
                'min': np.min(window),
                'max': np.max(window),
                'trend': window[-1] - window[0]  # Simple trend
            }
            features.append(window_features)
    
    return features

# Apply sliding window
window_features = sliding_window_features(time_series[:50], window_size=5)
print(f"Generated {len(window_features)} feature vectors from sliding windows")
print("Sample window features:")
for i, feat in enumerate(window_features[:3]):
    print(f"  Window {i+1}: {feat}")

# B. Batch Processing Queue
print(f"\nB. Batch Processing Queue:")

class BatchProcessor:
    """Process data in batches using deque"""
    
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.queue = deque()
        self.processed_batches = 0
    
    def add_sample(self, sample):
        """Add sample to processing queue"""
        self.queue.append(sample)
        
        if len(self.queue) >= self.batch_size:
            return self.process_batch()
        return None
    
    def process_batch(self):
        """Process a full batch"""
        if len(self.queue) < self.batch_size:
            return None
        
        batch = []
        for _ in range(self.batch_size):
            batch.append(self.queue.popleft())
        
        # Simulate processing (e.g., model prediction)
        batch_array = np.array(batch)
        result = {
            'batch_id': self.processed_batches,
            'batch_mean': batch_array.mean(),
            'batch_std': batch_array.std(),
            'processed_samples': len(batch)
        }
        
        self.processed_batches += 1
        return result
    
    def flush_remaining(self):
        """Process remaining samples in queue"""
        if len(self.queue) > 0:
            remaining = list(self.queue)
            self.queue.clear()
            return {
                'batch_id': self.processed_batches,
                'remaining_samples': len(remaining),
                'is_partial': True
            }
        return None

# Demonstrate batch processing
processor = BatchProcessor(batch_size=10)
sample_data = np.random.randn(35)

print("Processing samples in batches:")
for i, sample in enumerate(sample_data):
    result = processor.add_sample(sample)
    if result:
        print(f"  Processed batch {result['batch_id']}: {result['processed_samples']} samples")

# Process remaining
remaining = processor.flush_remaining()
if remaining:
    print(f"  Final batch: {remaining['remaining_samples']} remaining samples")

# C. Real-time Data Buffer
print(f"\nC. Real-time Data Buffer:")

class RealTimeBuffer:
    """Maintain a fixed-size buffer for real-time analysis"""
    
    def __init__(self, buffer_size=100):
        self.buffer = deque(maxlen=buffer_size)
        self.stats_history = deque(maxlen=10)  # Keep last 10 stats
    
    def add_data_point(self, value):
        """Add new data point and update statistics"""
        self.buffer.append(value)
        
        if len(self.buffer) >= 10:  # Minimum samples for stats
            current_stats = {
                'mean': np.mean(self.buffer),
                'std': np.std(self.buffer),
                'min': np.min(self.buffer),
                'max': np.max(self.buffer),
                'count': len(self.buffer)
            }
            self.stats_history.append(current_stats)
            return current_stats
        return None
    
    def get_trend(self):
        """Analyze trend from recent statistics"""
        if len(self.stats_history) < 2:
            return "Insufficient data"
        
        recent_mean = self.stats_history[-1]['mean']
        older_mean = self.stats_history[-2]['mean']
        
        if recent_mean > older_mean * 1.05:
            return "Increasing"
        elif recent_mean < older_mean * 0.95:
            return "Decreasing"
        else:
            return "Stable"

# Simulate real-time data
buffer = RealTimeBuffer(buffer_size=50)
streaming_data = np.random.randn(100).cumsum()

print("Real-time data processing:")
for i, value in enumerate(streaming_data):
    stats = buffer.add_data_point(value)
    if stats and i % 20 == 19:  # Print every 20 samples
        trend = buffer.get_trend()
        print(f"  Sample {i+1}: Mean={stats['mean']:.3f}, Trend={trend}")

print()

# 4. NAMEDTUPLE - STRUCTURED DATA
print("4. NAMEDTUPLE - STRUCTURED DATA")
print("-" * 35)

print("Theory: namedtuple creates lightweight, immutable classes")
print("for structured data, improving code readability and debugging.")
print()

# A. Model Configuration Management
print("A. Model Configuration Management:")

# Define model configuration structure
ModelConfig = namedtuple('ModelConfig', [
    'name', 'n_estimators', 'max_depth', 'random_state', 'n_jobs'
])

# Define different model configurations
model_configs = [
    ModelConfig('RF_Small', 50, 5, 42, -1),
    ModelConfig('RF_Medium', 100, 10, 42, -1),
    ModelConfig('RF_Large', 200, 15, 42, -1)
]

print("Model Configurations:")
for config in model_configs:
    print(f"  {config.name}: n_estimators={config.n_estimators}, max_depth={config.max_depth}")

# Train models using configurations
results = []
for config in model_configs:
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
        n_jobs=config.n_jobs
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_iris, y_iris, cv=3)
    
    ModelResult = namedtuple('ModelResult', ['config', 'cv_mean', 'cv_std'])
    result = ModelResult(config, cv_scores.mean(), cv_scores.std())
    results.append(result)

print(f"\nModel Performance Results:")
for result in results:
    print(f"  {result.config.name}: {result.cv_mean:.4f} ± {result.cv_std:.4f}")

# B. Feature Engineering Pipeline
print(f"\nB. Feature Engineering Pipeline:")

# Define feature transformation steps
FeatureStep = namedtuple('FeatureStep', ['name', 'function', 'params'])

def standardize(data, params):
    """Standardization transformation"""
    scaler = StandardScaler()
    return scaler.fit_transform(data), scaler

def polynomial_features(data, params):
    """Create polynomial features"""
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=params['degree'])
    return poly.fit_transform(data), poly

# Define pipeline steps
pipeline_steps = [
    FeatureStep('Standardization', standardize, {}),
    FeatureStep('Polynomial', polynomial_features, {'degree': 2})
]

print("Feature Engineering Pipeline:")
for step in pipeline_steps:
    print(f"  Step: {step.name}")
    print(f"    Function: {step.function.__name__}")
    print(f"    Parameters: {step.params}")

# C. Evaluation Metrics Structure
print(f"\nC. Evaluation Metrics Structure:")

# Define metrics structure
ClassificationMetrics = namedtuple('ClassificationMetrics', [
    'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'
])

# Calculate metrics for iris classification
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# For multiclass ROC-AUC
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
roc_auc = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr')

metrics = ClassificationMetrics(
    accuracy=accuracy_score(y_test, y_pred),
    precision=precision_score(y_test, y_pred, average='weighted'),
    recall=recall_score(y_test, y_pred, average='weighted'),
    f1_score=f1_score(y_test, y_pred, average='weighted'),
    roc_auc=roc_auc
)

print("Classification Metrics:")
for field, value in metrics._asdict().items():
    print(f"  {field.replace('_', ' ').title()}: {value:.4f}")

print()

# 5. ORDEREDDICT - MAINTAINING ORDER
print("5. ORDEREDDICT - MAINTAINING ORDER")
print("-" * 37)

print("Theory: OrderedDict maintains insertion order, useful for")
print("pipeline stages, feature ordering, and reproducible results.")
print()

# A. Feature Importance Ranking
print("A. Feature Importance Ranking:")

# Train model and get feature importance
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_iris, y_iris)

# Create ordered dictionary of feature importance
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
importance_dict = OrderedDict()

# Sort by importance (descending)
importance_pairs = list(zip(feature_names, model.feature_importances_))
importance_pairs.sort(key=lambda x: x[1], reverse=True)

for feature, importance in importance_pairs:
    importance_dict[feature] = importance

print("Feature Importance Ranking (ordered):")
for i, (feature, importance) in enumerate(importance_dict.items(), 1):
    print(f"  {i}. {feature}: {importance:.4f}")

# B. Model Pipeline Tracking
print(f"\nB. Model Pipeline Tracking:")

# Track processing steps in order
pipeline_log = OrderedDict()

# Simulate pipeline execution
start_time = time.time()

# Step 1: Data loading
pipeline_log['data_loading'] = {
    'start_time': time.time(),
    'status': 'completed',
    'samples': X_iris.shape[0],
    'features': X_iris.shape[1]
}

# Step 2: Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)
pipeline_log['preprocessing'] = {
    'start_time': time.time(),
    'status': 'completed',
    'method': 'StandardScaler',
    'output_shape': X_scaled.shape
}

# Step 3: Model training
model.fit(X_scaled, y_iris)
pipeline_log['model_training'] = {
    'start_time': time.time(),
    'status': 'completed',
    'model_type': 'RandomForestClassifier',
    'n_estimators': model.n_estimators
}

# Step 4: Evaluation
train_score = model.score(X_scaled, y_iris)
pipeline_log['evaluation'] = {
    'start_time': time.time(),
    'status': 'completed',
    'train_accuracy': train_score
}

print("Pipeline Execution Log (in order):")
for step_name, step_info in pipeline_log.items():
    print(f"  {step_name.replace('_', ' ').title()}:")
    print(f"    Status: {step_info['status']}")
    for key, value in step_info.items():
        if key != 'status':
            print(f"    {key}: {value}")

# C. Hyperparameter Search Results
print(f"\nC. Hyperparameter Search Results:")

# Simulate grid search with ordered results
param_results = OrderedDict()

# Define parameter grid
param_grid = [
    {'n_estimators': 50, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 5},
    {'n_estimators': 50, 'max_depth': 10},
    {'n_estimators': 100, 'max_depth': 10}
]

# Evaluate each parameter combination
for i, params in enumerate(param_grid):
    model = RandomForestClassifier(**params, random_state=42)
    cv_scores = cross_val_score(model, X_iris, y_iris, cv=3)
    
    param_key = f"Config_{i+1}"
    param_results[param_key] = {
        'params': params,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

print("Hyperparameter Search Results (in evaluation order):")
for config_name, result in param_results.items():
    print(f"  {config_name}: {result['params']}")
    print(f"    CV Score: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")

print()

# 6. CHAINMAP - CONFIGURATION MANAGEMENT
print("6. CHAINMAP - CONFIGURATION MANAGEMENT")
print("-" * 42)

print("Theory: ChainMap groups multiple dictionaries into a single")
print("view, useful for hierarchical configuration management.")
print()

# A. Hierarchical Configuration
print("A. Hierarchical Configuration System:")

# Define configuration levels
default_config = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': 1,
    'verbose': 0
}

user_config = {
    'n_estimators': 200,  # User override
    'max_depth': 15,      # User override
    'n_jobs': -1          # User override
}

experiment_config = {
    'random_state': 123,  # Experiment-specific
    'verbose': 1          # Debug mode
}

# Chain configurations (experiment > user > default)
config_chain = ChainMap(experiment_config, user_config, default_config)

print("Configuration Hierarchy:")
print("  Experiment config:", experiment_config)
print("  User config:", user_config)
print("  Default config:", default_config)
print()
print("Final merged configuration:")
for key, value in config_chain.items():
    print(f"  {key}: {value}")

# Use configuration to create model
model = RandomForestClassifier(**dict(config_chain))
print(f"\nModel created with merged configuration")
print(f"Effective n_estimators: {model.n_estimators}")

# B. Environment-Specific Settings
print(f"\nB. Environment-Specific Settings:")

# Define different environment configurations
development_env = {
    'data_size': 'small',
    'n_estimators': 10,
    'cv_folds': 3,
    'verbose': True
}

production_env = {
    'data_size': 'full',
    'n_estimators': 500,
    'cv_folds': 10,
    'verbose': False
}

base_ml_config = {
    'random_state': 42,
    'n_jobs': -1,
    'max_depth': None,
    'bootstrap': True
}

# Select environment
current_env = 'development'  # or 'production'
env_configs = {
    'development': development_env,
    'production': production_env
}

# Create environment-aware configuration
ml_config = ChainMap(env_configs[current_env], base_ml_config)

print(f"Environment: {current_env}")
print("Active configuration:")
for key, value in ml_config.items():
    print(f"  {key}: {value}")

print()

# 7. PERFORMANCE COMPARISON
print("7. PERFORMANCE COMPARISON")
print("-" * 30)

print("Theory: Collections provide performance benefits over")
print("standard implementations for specific use cases.")
print()

# A. Counter vs Manual Counting
print("A. Counter vs Manual Counting Performance:")

# Generate large dataset for performance testing
large_labels = np.random.randint(0, 10, 100000)

# Method 1: Manual counting
def manual_count(labels):
    counts = {}
    for label in labels:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    return counts

# Method 2: Counter
def counter_count(labels):
    return Counter(labels)

# Performance comparison
import timeit

manual_time = timeit.timeit(lambda: manual_count(large_labels), number=10) / 10
counter_time = timeit.timeit(lambda: counter_count(large_labels), number=10) / 10

print(f"Manual counting: {manual_time:.6f} seconds")
print(f"Counter method: {counter_time:.6f} seconds")
print(f"Counter is {manual_time/counter_time:.2f}x faster")

# B. deque vs list for Queue Operations
print(f"\nB. Deque vs List for Queue Operations:")

# Test queue operations
queue_size = 10000

def list_queue_ops(size):
    queue = []
    for i in range(size):
        queue.append(i)
    for i in range(size):
        queue.pop(0)  # O(n) operation
    return queue

def deque_queue_ops(size):
    queue = deque()
    for i in range(size):
        queue.append(i)
    for i in range(size):
        queue.popleft()  # O(1) operation
    return queue

# Performance test (smaller size to avoid long wait)
test_size = 1000
list_time = timeit.timeit(lambda: list_queue_ops(test_size), number=5) / 5
deque_time = timeit.timeit(lambda: deque_queue_ops(test_size), number=5) / 5

print(f"List queue operations: {list_time:.6f} seconds")
print(f"Deque queue operations: {deque_time:.6f} seconds")
print(f"Deque is {list_time/deque_time:.2f}x faster for queue operations")

print()

# 8. VISUALIZATION OF COLLECTIONS USAGE
print("8. VISUALIZATION OF COLLECTIONS USAGE")
print("-" * 40)

# Create comprehensive visualization
plt.figure(figsize=(16, 12))

# Plot 1: Label distribution using Counter
plt.subplot(2, 3, 1)
label_counts = Counter(y_iris)
labels = [iris_names[i] for i in label_counts.keys()]
counts = list(label_counts.values())
bars = plt.bar(labels, counts, alpha=0.7, color=['red', 'green', 'blue'])
plt.title('Label Distribution (Counter)')
plt.ylabel('Count')
plt.xticks(rotation=45)
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(count), ha='center', va='bottom')

# Plot 2: Feature importance ranking (OrderedDict)
plt.subplot(2, 3, 2)
features = list(importance_dict.keys())
importances = list(importance_dict.values())
bars = plt.barh(features, importances, alpha=0.7)
plt.title('Feature Importance Ranking (OrderedDict)')
plt.xlabel('Importance')
for bar, imp in zip(bars, importances):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{imp:.3f}', va='center')

# Plot 3: Pipeline execution tracking
plt.subplot(2, 3, 3)
steps = list(pipeline_log.keys())
step_numbers = range(1, len(steps) + 1)
plt.plot(step_numbers, step_numbers, 'o-', linewidth=2, markersize=8)
plt.title('Pipeline Execution Order')
plt.xlabel('Execution Order')
plt.ylabel('Step Number')
plt.xticks(step_numbers, [s.replace('_', '\n') for s in steps], rotation=45)
plt.grid(True, alpha=0.3)

# Plot 4: Performance comparison
plt.subplot(2, 3, 4)
methods = ['Manual\nCount', 'Counter', 'List\nQueue', 'Deque\nQueue']
times = [manual_time, counter_time, list_time, deque_time]
colors = ['red', 'green', 'red', 'green']
bars = plt.bar(methods, times, color=colors, alpha=0.7)
plt.title('Performance Comparison')
plt.ylabel('Time (seconds)')
plt.yscale('log')
for bar, time in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{time:.6f}', ha='center', va='bottom', rotation=90)

# Plot 5: Hyperparameter search results
plt.subplot(2, 3, 5)
config_names = list(param_results.keys())
cv_means = [result['cv_mean'] for result in param_results.values()]
cv_stds = [result['cv_std'] for result in param_results.values()]
bars = plt.bar(config_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
plt.title('Hyperparameter Search Results')
plt.ylabel('CV Score')
plt.xticks(rotation=45)
for bar, mean in zip(bars, cv_means):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{mean:.3f}', ha='center', va='bottom')

# Plot 6: Text data word frequency
plt.subplot(2, 3, 6)
top_words = word_counter.most_common(8)
words, freqs = zip(*top_words)
bars = plt.bar(words, freqs, alpha=0.7)
plt.title('Word Frequency Analysis (Counter)')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
for bar, freq in zip(bars, freqs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(freq), ha='center', va='bottom')

plt.tight_layout()
plt.show()

print()

# 9. BEST PRACTICES AND USE CASES
print("9. BEST PRACTICES AND USE CASES")
print("-" * 35)

best_practices = {
    "Counter": {
        "Use Cases": [
            "Analyzing class distribution for imbalance detection",
            "Word frequency analysis in text processing",
            "Feature value distribution analysis",
            "Counting discrete events in time series"
        ],
        "Best Practices": [
            "Use most_common() for top-k analysis",
            "Combine with pandas value_counts() for integration",
            "Consider memory usage for very large datasets",
            "Use update() for incremental counting"
        ]
    },
    "defaultdict": {
        "Use Cases": [
            "Grouping features by class or category",
            "Building nested data structures",
            "Accumulating results without key existence checks",
            "Creating sparse feature representations"
        ],
        "Best Practices": [
            "Choose appropriate default factory (list, int, set)",
            "Use lambda for complex default values",
            "Convert to regular dict when sharing across modules",
            "Be careful with mutable default values"
        ]
    },
    "deque": {
        "Use Cases": [
            "Sliding window feature engineering",
            "Real-time data buffers with fixed size",
            "Batch processing queues",
            "Implementing circular buffers"
        ],
        "Best Practices": [
            "Set maxlen for memory-bounded operations",
            "Use appendleft() and popleft() for queue operations",
            "Consider deque for frequent insertions at both ends",
            "Not suitable for random access patterns"
        ]
    },
    "namedtuple": {
        "Use Cases": [
            "Structured configuration management",
            "Model result containers",
            "Feature engineering pipeline definitions",
            "Immutable data transfer objects"
        ],
        "Best Practices": [
            "Use _asdict() for converting to dict",
            "Consider typing.NamedTuple for type hints",
            "Keep field names descriptive and consistent",
            "Use defaults parameter for optional fields"
        ]
    },
    "OrderedDict": {
        "Use Cases": [
            "Maintaining feature importance rankings",
            "Pipeline step tracking",
            "Configuration with precedence order",
            "Reproducible iteration order"
        ],
        "Best Practices": [
            "Python 3.7+ dicts maintain order, consider if OrderedDict is needed",
            "Use move_to_end() for reordering",
            "popitem(last=False) for FIFO behavior",
            "Consider performance implications of ordering"
        ]
    },
    "ChainMap": {
        "Use Cases": [
            "Hierarchical configuration systems",
            "Environment-specific settings",
            "Merging multiple parameter dictionaries",
            "Creating configuration inheritance"
        ],
        "Best Practices": [
            "Order matters: first map has highest precedence",
            "Use new_child() for temporary overrides",
            "maps attribute provides access to constituent dicts",
            "Consider dict unpacking for simple cases"
        ]
    }
}

print("Collections Module Best Practices in ML:")
for collection, info in best_practices.items():
    print(f"\n{collection}:")
    print("  Use Cases:")
    for use_case in info["Use Cases"]:
        print(f"    • {use_case}")
    print("  Best Practices:")
    for practice in info["Best Practices"]:
        print(f"    • {practice}")

print()

# 10. ADVANCED APPLICATIONS
print("10. ADVANCED APPLICATIONS")
print("-" * 25)

print("Advanced ML applications combining multiple collections:")
print()

# A. Feature Selection Pipeline with Collections
print("A. Feature Selection Pipeline with Collections:")

class FeatureSelectionPipeline:
    """Advanced feature selection using multiple collections"""
    
    def __init__(self):
        self.steps = OrderedDict()
        self.feature_scores = defaultdict(list)
        self.selection_history = deque(maxlen=10)
    
    def add_step(self, name, selector, params=None):
        """Add a feature selection step"""
        StepConfig = namedtuple('StepConfig', ['selector', 'params', 'selected_features'])
        self.steps[name] = StepConfig(selector, params or {}, None)
    
    def fit(self, X, y):
        """Fit the pipeline"""
        current_X = X
        feature_names = list(range(X.shape[1]))
        
        for step_name, step_config in self.steps.items():
            # Fit selector
            selector = step_config.selector
            selector.fit(current_X, y)
            
            # Get selected features
            selected_mask = selector.get_support()
            selected_indices = np.where(selected_mask)[0]
            
            # Update step with results
            updated_step = step_config._replace(selected_features=selected_indices)
            self.steps[step_name] = updated_step
            
            # Track feature scores
            if hasattr(selector, 'scores_'):
                self.feature_scores[step_name] = selector.scores_
            
            # Update data for next step
            current_X = selector.transform(current_X)
            feature_names = [feature_names[i] for i in selected_indices]
            
            # Record selection history
            SelectionRecord = namedtuple('SelectionRecord', 
                                       ['step', 'n_features_before', 'n_features_after'])
            record = SelectionRecord(step_name, len(selected_mask), len(selected_indices))
            self.selection_history.append(record)
        
        return self
    
    def get_final_features(self):
        """Get final selected features"""
        if not self.steps:
            return None
        
        final_step = list(self.steps.values())[-1]
        return final_step.selected_features
    
    def get_selection_summary(self):
        """Get selection summary"""
        summary = OrderedDict()
        for record in self.selection_history:
            summary[record.step] = {
                'before': record.n_features_before,
                'after': record.n_features_after,
                'reduction': record.n_features_before - record.n_features_after
            }
        return summary

# Demonstrate advanced pipeline
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

pipeline = FeatureSelectionPipeline()
pipeline.add_step('univariate', SelectKBest(f_classif, k=8))
pipeline.add_step('model_based', SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42)))

# Fit pipeline
pipeline.fit(X_iris, y_iris)

print("Feature Selection Pipeline Summary:")
summary = pipeline.get_selection_summary()
for step, info in summary.items():
    print(f"  {step}: {info['before']} → {info['after']} features (reduced by {info['reduction']})")

final_features = pipeline.get_final_features()
print(f"Final selected features: {final_features}")

print()

print(f"{'='*60}")
print("SUMMARY: Collections Module in Machine Learning")
print(f"{'='*60}")

summary_points = [
    "Counter: Essential for frequency analysis and class distribution",
    "defaultdict: Simplifies data grouping and aggregation tasks",
    "deque: Efficient for sliding windows and real-time processing",
    "namedtuple: Structures data for better code organization",
    "OrderedDict: Maintains order for reproducible pipelines",
    "ChainMap: Manages hierarchical configurations effectively",
    "Performance: Collections offer optimized implementations",
    "Memory efficiency: Specialized containers reduce overhead",
    "Code readability: Structured data improves maintainability",
    "Integration: Works seamlessly with pandas and scikit-learn"
]

for i, point in enumerate(summary_points, 1):
    print(f"{i:2d}. {point}")

print(f"\n{'='*60}")
print("PRACTICAL RECOMMENDATIONS")
print(f"{'='*60}")

recommendations = {
    "Data Exploration": [
        "Use Counter for class distribution analysis",
        "Apply defaultdict for feature grouping by category",
        "Leverage OrderedDict for maintaining feature rankings"
    ],
    "Real-time Processing": [
        "Implement deque for sliding window features",
        "Use maxlen parameter for memory-bounded buffers",
        "Create real-time statistics with deque + namedtuple"
    ],
    "Configuration Management": [
        "Structure model configs with namedtuple",
        "Apply ChainMap for hierarchical settings",
        "Maintain pipeline order with OrderedDict"
    ],
    "Performance Optimization": [
        "Choose appropriate collection for use case",
        "Profile performance for large datasets",
        "Consider memory vs speed trade-offs"
    ]
}

for category, recs in recommendations.items():
    print(f"\n{category}:")
    for rec in recs:
        print(f"  • {rec}")

print(f"\n=== Collections: Powerful Tools for ML Data Structures ===")
```

### Explanation

Python's `collections` module provides specialized container datatypes that offer significant advantages in machine learning workflows through enhanced functionality, better performance, and improved code organization.

### Key Collections for ML

1. **Counter**
   - **Purpose**: Counting hashable objects
   - **ML Applications**: Class distribution analysis, word frequency, feature value counting
   - **Benefits**: Fast frequency analysis, most_common() method, arithmetic operations

2. **defaultdict**
   - **Purpose**: Dict with automatic default value creation
   - **ML Applications**: Feature grouping, nested data structures, result accumulation
   - **Benefits**: Eliminates key existence checks, cleaner grouping code

3. **deque**
   - **Purpose**: Double-ended queue with O(1) operations
   - **ML Applications**: Sliding windows, real-time buffers, batch processing
   - **Benefits**: Efficient append/pop from both ends, memory-bounded with maxlen

4. **namedtuple**
   - **Purpose**: Lightweight, immutable classes
   - **ML Applications**: Model configurations, structured results, pipeline definitions
   - **Benefits**: Field access by name, immutability, memory efficiency

5. **OrderedDict**
   - **Purpose**: Dictionary maintaining insertion order
   - **ML Applications**: Feature rankings, pipeline tracking, reproducible iteration
   - **Benefits**: Guaranteed order, move_to_end() method, FIFO operations

6. **ChainMap**
   - **Purpose**: Multiple dictionaries as single view
   - **ML Applications**: Configuration hierarchies, environment settings, parameter merging
   - **Benefits**: Precedence handling, dynamic configuration updates

### ML-Specific Use Cases

**Data Exploration:**
- Counter for class imbalance detection
- defaultdict for feature grouping by categories
- Statistical analysis of feature distributions

**Feature Engineering:**
- deque for sliding window transformations
- namedtuple for transformation step definitions
- OrderedDict for maintaining feature importance order

**Model Configuration:**
- namedtuple for structured model parameters
- ChainMap for environment-specific configurations
- OrderedDict for pipeline step tracking

**Real-time Processing:**
- deque with maxlen for fixed-size buffers
- Counter for streaming frequency analysis
- defaultdict for accumulating real-time statistics

**Performance Optimization:**
- Collections provide optimized implementations
- Memory efficiency through specialized containers
- Faster operations for specific use patterns

### Integration with ML Libraries

**Scikit-learn Integration:**
- Counter for analyzing target distributions before train_test_split
- defaultdict for collecting cross-validation results
- namedtuple for parameter grid definitions

**Pandas Integration:**
- Counter complements value_counts()
- defaultdict for groupby operations
- OrderedDict for maintaining column order

**Text Processing:**
- Counter for term frequency analysis
- defaultdict for document-term matrices
- deque for n-gram generation

### Best Practices

1. **Choose Appropriate Collection**: Match collection type to use case
2. **Memory Considerations**: Use maxlen with deque for bounded memory
3. **Performance Profiling**: Measure actual performance gains
4. **Code Organization**: Use namedtuple for structured data
5. **Configuration Management**: Apply ChainMap for hierarchical configs

### Performance Benefits

- **Counter**: 2-3x faster than manual dictionary counting
- **deque**: 10-100x faster than list for queue operations
- **defaultdict**: Eliminates key existence checks
- **namedtuple**: Memory efficient compared to classes
- **ChainMap**: Avoids dictionary copying for merging

### Common Patterns

**Pipeline Pattern:**
```python
pipeline_steps = OrderedDict([
    ('preprocessing', preprocessing_step),
    ('feature_selection', selection_step),
    ('modeling', model_step)
])
```

**Configuration Pattern:**
```python
config = ChainMap(experiment_config, user_config, default_config)
```

**Real-time Pattern:**
```python
buffer = deque(maxlen=window_size)
stats = defaultdict(list)
```

The collections module enhances ML workflows by providing efficient, readable, and maintainable data structures that integrate seamlessly with the broader Python data science ecosystem.

**Answer:** _[To be filled]_

---

## Question 13

**Discuss various options for deploying a machine learning model in Python.**

### Theory
Model deployment is the process of making trained machine learning models available for production use. This involves packaging the model, creating serving infrastructure, handling scalability, monitoring, and maintaining the model lifecycle. Python offers numerous deployment options ranging from simple REST APIs to sophisticated cloud-native solutions.

### Answer

```python
# ml_model_deployment.py - Comprehensive guide to ML model deployment options
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle
import json
import time
import requests
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("=== Machine Learning Model Deployment Options ===\n")

# Prepare sample model for deployment demonstrations
def create_sample_model():
    """Create and train a sample model for deployment examples"""
    
    # Use iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Sample Model Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Classes: {iris.target_names}")
    print()
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': iris.feature_names,
        'target_names': iris.target_names,
        'test_accuracy': accuracy,
        'X_test': X_test_scaled,
        'y_test': y_test
    }

# Create sample model
model_package = create_sample_model()

# 1. LOCAL DEPLOYMENT OPTIONS
print("1. LOCAL DEPLOYMENT OPTIONS")
print("-" * 31)

print("Theory: Local deployment options are suitable for development,")
print("testing, and small-scale applications.")
print()

# A. Pickle/Joblib File Serving
print("A. Pickle/Joblib File Serving:")

# Save model and scaler
model_filename = 'iris_model.joblib'
scaler_filename = 'iris_scaler.joblib'

joblib.dump(model_package['model'], model_filename)
joblib.dump(model_package['scaler'], scaler_filename)

print(f"  Models saved:")
print(f"    {model_filename} ({os.path.getsize(model_filename)} bytes)")
print(f"    {scaler_filename} ({os.path.getsize(scaler_filename)} bytes)")

# Create prediction function
def load_and_predict(features, model_file=model_filename, scaler_file=scaler_filename):
    """Load model and make predictions"""
    
    # Load model and scaler
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    
    # Preprocess and predict
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'class_name': model_package['target_names'][prediction],
        'probabilities': {
            name: float(prob) for name, prob in 
            zip(model_package['target_names'], probability)
        }
    }

# Test local prediction
sample_features = [5.1, 3.5, 1.4, 0.2]  # Typical setosa features
result = load_and_predict(sample_features)
print(f"\n  Sample Prediction:")
print(f"    Input: {sample_features}")
print(f"    Predicted Class: {result['class_name']}")
print(f"    Probabilities: {result['probabilities']}")

# B. Simple Python Script Deployment
print(f"\nB. Simple Python Script Deployment:")

# Create standalone prediction script
script_content = '''#!/usr/bin/env python3
"""
Standalone ML model prediction script
Usage: python predict.py [sepal_length] [sepal_width] [petal_length] [petal_width]
"""

import sys
import joblib
import numpy as np

def main():
    if len(sys.argv) != 5:
        print("Usage: python predict.py [sepal_length] [sepal_width] [petal_length] [petal_width]")
        sys.exit(1)
    
    # Parse input features
    try:
        features = [float(x) for x in sys.argv[1:5]]
    except ValueError:
        print("Error: All features must be numbers")
        sys.exit(1)
    
    # Load model and scaler
    try:
        model = joblib.load('iris_model.joblib')
        scaler = joblib.load('iris_scaler.joblib')
        target_names = ['setosa', 'versicolor', 'virginica']
    except FileNotFoundError:
        print("Error: Model files not found")
        sys.exit(1)
    
    # Make prediction
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Output results
    print(f"Prediction: {target_names[prediction]}")
    print(f"Confidence: {probabilities[prediction]:.4f}")
    print("All probabilities:")
    for name, prob in zip(target_names, probabilities):
        print(f"  {name}: {prob:.4f}")

if __name__ == "__main__":
    main()
'''

# Save script
with open('predict.py', 'w') as f:
    f.write(script_content)

print(f"  Created standalone script: predict.py")
print(f"  Usage: python predict.py 5.1 3.5 1.4 0.2")

print()

# 2. WEB API DEPLOYMENT
print("2. WEB API DEPLOYMENT")
print("-" * 21)

print("Theory: Web APIs provide standardized interfaces for model")
print("serving, enabling integration with web applications and services.")
print()

# A. Flask REST API
print("A. Flask REST API:")

flask_api_code = '''
from flask import Flask, request, jsonify
import joblib
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model
model = None
scaler = None
target_names = ['setosa', 'versicolor', 'virginica']
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def load_model():
    """Load model and scaler at startup"""
    global model, scaler
    try:
        model = joblib.load('iris_model.joblib')
        scaler = joblib.load('iris_scaler.joblib')
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Parse input
        data = request.get_json()
        
        # Validate input
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = data['features']
        if len(features) != 4:
            return jsonify({'error': 'Expected 4 features'}), 400
        
        # Make prediction
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'class_name': target_names[prediction],
            'probabilities': {
                name: float(prob) for name, prob in 
                zip(target_names, probabilities)
            },
            'confidence': float(probabilities[prediction]),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: {response['class_name']} (confidence: {response['confidence']:.4f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    return jsonify({
        'model_type': 'RandomForestClassifier',
        'features': feature_names,
        'classes': target_names,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    if load_model():
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load model. Exiting.")
'''

# Save Flask API
with open('flask_api.py', 'w') as f:
    f.write(flask_api_code)

print(f"  Created Flask API: flask_api.py")
print(f"  Endpoints:")
print(f"    GET  /health     - Health check")
print(f"    POST /predict    - Make predictions")
print(f"    GET  /model_info - Model information")
print(f"  Usage: python flask_api.py")

# B. FastAPI Implementation
print(f"\nB. FastAPI Implementation:")

fastapi_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Iris Prediction API", version="1.0.0")

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: List[float]
    
    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: Dict[str, float]
    confidence: float
    timestamp: str

# Global variables
model = None
scaler = None
target_names = ['setosa', 'versicolor', 'virginica']
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, scaler
    try:
        model = joblib.load('iris_model.joblib')
        scaler = joblib.load('iris_scaler.joblib')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction"""
    try:
        # Validate input length
        if len(request.features) != 4:
            raise HTTPException(status_code=400, detail="Expected 4 features")
        
        # Make prediction
        features_array = np.array([request.features])
        features_scaled = scaler.transform(features_array)
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        response = PredictionResponse(
            prediction=int(prediction),
            class_name=target_names[prediction],
            probabilities={
                name: float(prob) for name, prob in 
                zip(target_names, probabilities)
            },
            confidence=float(probabilities[prediction]),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction: {response.class_name} (confidence: {response.confidence:.4f})")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "RandomForestClassifier",
        "features": feature_names,
        "classes": target_names,
        "version": "1.0.0"
    }

# Run with: uvicorn fastapi_api:app --host 0.0.0.0 --port 8000
'''

# Save FastAPI
with open('fastapi_api.py', 'w') as f:
    f.write(fastapi_code)

print(f"  Created FastAPI: fastapi_api.py")
print(f"  Features:")
print(f"    • Automatic API documentation (Swagger UI)")
print(f"    • Type validation with Pydantic")
print(f"    • Async support")
print(f"    • Better performance than Flask")
print(f"  Usage: uvicorn fastapi_api:app --host 0.0.0.0 --port 8000")

print()

# 3. CONTAINERIZED DEPLOYMENT
print("3. CONTAINERIZED DEPLOYMENT")
print("-" * 30)

print("Theory: Containerization provides consistent, reproducible")
print("deployment environments across different platforms.")
print()

# A. Docker Deployment
print("A. Docker Deployment:")

# Create Dockerfile
dockerfile_content = '''
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "fastapi_api:app", "--host", "0.0.0.0", "--port", "8000"]
'''

# Create requirements.txt
requirements_content = '''
fastapi==0.68.0
uvicorn==0.15.0
scikit-learn==1.0.2
joblib==1.1.0
numpy==1.21.0
pandas==1.3.0
'''

# Save Docker files
with open('Dockerfile', 'w') as f:
    f.write(dockerfile_content)

with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

print(f"  Created Docker configuration:")
print(f"    Dockerfile - Container definition")
print(f"    requirements.txt - Python dependencies")
print(f"  Build: docker build -t iris-model .")
print(f"  Run: docker run -p 8000:8000 iris-model")

# B. Docker Compose for Multi-Service
print(f"\nB. Docker Compose for Multi-Service Deployment:")

docker_compose_content = '''
version: '3.8'

services:
  iris-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - LOG_LEVEL=info
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - iris-api
    restart: unless-stopped

volumes:
  logs:
'''

with open('docker-compose.yml', 'w') as f:
    f.write(docker_compose_content)

print(f"  Created docker-compose.yml with:")
print(f"    • FastAPI service")
print(f"    • Redis for caching")
print(f"    • Nginx for load balancing")
print(f"  Usage: docker-compose up -d")

print()

# 4. CLOUD DEPLOYMENT OPTIONS
print("4. CLOUD DEPLOYMENT OPTIONS")
print("-" * 31)

print("Theory: Cloud platforms provide scalable, managed infrastructure")
print("for ML model deployment with built-in monitoring and scaling.")
print()

# A. AWS Deployment Options
print("A. AWS Deployment Options:")

aws_options = {
    "AWS Lambda + API Gateway": {
        "Use Case": "Serverless, low-traffic applications",
        "Pros": ["No server management", "Pay per request", "Auto-scaling"],
        "Cons": ["Cold start latency", "15-minute timeout", "Memory limits"],
        "Setup": "Package model + code, deploy via SAM/CDK"
    },
    "AWS ECS/Fargate": {
        "Use Case": "Containerized applications with moderate traffic",
        "Pros": ["Container orchestration", "Auto-scaling", "Load balancing"],
        "Cons": ["More complex setup", "Container overhead"],
        "Setup": "Docker container + task definition + service"
    },
    "AWS SageMaker": {
        "Use Case": "ML-specific hosting with built-in features",
        "Pros": ["ML-optimized", "A/B testing", "Auto-scaling", "Monitoring"],
        "Cons": ["Vendor lock-in", "Cost for small models"],
        "Setup": "Model artifact + inference script + endpoint config"
    },
    "AWS EC2": {
        "Use Case": "Full control, custom configurations",
        "Pros": ["Complete control", "Custom environments", "Cost-effective"],
        "Cons": ["Manual management", "No auto-scaling", "Maintenance overhead"],
        "Setup": "Launch instance + install dependencies + deploy"
    }
}

print("AWS Deployment Options:")
for option, details in aws_options.items():
    print(f"\n  {option}:")
    print(f"    Use Case: {details['Use Case']}")
    print(f"    Pros: {', '.join(details['Pros'])}")
    print(f"    Cons: {', '.join(details['Cons'])}")

# B. SageMaker Deployment Example
print(f"\nB. AWS SageMaker Deployment Example:")

sagemaker_code = '''
import boto3
import joblib
import json
import numpy as np
from sagemaker.sklearn import SKLearnModel
from sagemaker import get_execution_role

# 1. Prepare model artifact
def create_model_artifact():
    """Create SageMaker-compatible model artifact"""
    
    # Save model in expected format
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Create inference script
    inference_script = """
import joblib
import json
import numpy as np

def model_fn(model_dir):
    '''Load model and scaler'''
    model = joblib.load(f'{model_dir}/model.joblib')
    scaler = joblib.load(f'{model_dir}/scaler.joblib')
    return {'model': model, 'scaler': scaler}

def input_fn(request_body, content_type):
    '''Parse input data'''
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        return np.array(input_data['features']).reshape(1, -1)
    else:
        raise ValueError(f'Unsupported content type: {content_type}')

def predict_fn(input_data, model):
    '''Make prediction'''
    scaler = model['scaler']
    clf = model['model']
    
    # Preprocess and predict
    scaled_data = scaler.transform(input_data)
    prediction = clf.predict(scaled_data)[0]
    probabilities = clf.predict_proba(scaled_data)[0]
    
    return {
        'prediction': int(prediction),
        'probabilities': probabilities.tolist()
    }

def output_fn(prediction, accept):
    '''Format output'''
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f'Unsupported accept type: {accept}')
    """
    
    with open('inference.py', 'w') as f:
        f.write(inference_script)
    
    # Package model
    import tarfile
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add('model.joblib')
        tar.add('scaler.joblib')
        tar.add('inference.py')
    
    return 'model.tar.gz'

# 2. Deploy to SageMaker
def deploy_to_sagemaker():
    """Deploy model to SageMaker endpoint"""
    
    # Create model artifact
    model_artifact = create_model_artifact()
    
    # Upload to S3
    import sagemaker
    session = sagemaker.Session()
    bucket = session.default_bucket()
    model_uri = session.upload_data(model_artifact, bucket, 'iris-model')
    
    # Create SageMaker model
    role = get_execution_role()
    model = SKLearnModel(
        model_data=model_uri,
        role=role,
        entry_point='inference.py',
        framework_version='0.23-1'
    )
    
    # Deploy endpoint
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',
        endpoint_name='iris-classifier'
    )
    
    return predictor

# Usage:
# predictor = deploy_to_sagemaker()
# result = predictor.predict({'features': [5.1, 3.5, 1.4, 0.2]})
'''

print(f"  SageMaker deployment provides:")
print(f"    • Managed infrastructure")
print(f"    • Auto-scaling")
print(f"    • A/B testing")
print(f"    • Model monitoring")

# C. Google Cloud and Azure Options
print(f"\nC. Other Cloud Platforms:")

cloud_options = {
    "Google Cloud AI Platform": {
        "Features": ["Serverless inference", "Custom containers", "Batch prediction"],
        "Best For": "TensorFlow/scikit-learn models"
    },
    "Azure Machine Learning": {
        "Features": ["Real-time endpoints", "Batch endpoints", "MLOps integration"],
        "Best For": "Microsoft ecosystem integration"
    },
    "Google Cloud Run": {
        "Features": ["Serverless containers", "Pay-per-use", "Auto-scaling"],
        "Best For": "Containerized ML services"
    },
    "Azure Container Instances": {
        "Features": ["Serverless containers", "Fast startup", "Simple deployment"],
        "Best For": "Small to medium workloads"
    }
}

for platform, details in cloud_options.items():
    print(f"  {platform}:")
    print(f"    Features: {', '.join(details['Features'])}")
    print(f"    Best For: {details['Best For']}")

print()

# 5. EDGE DEPLOYMENT
print("5. EDGE DEPLOYMENT")
print("-" * 18)

print("Theory: Edge deployment brings models closer to data sources")
print("for low latency, offline capability, and reduced bandwidth.")
print()

# A. Model Optimization for Edge
print("A. Model Optimization for Edge:")

def optimize_model_for_edge():
    """Demonstrate model optimization techniques for edge deployment"""
    
    print("Model Optimization Techniques:")
    
    # 1. Model compression
    print(f"\n  1. Model Compression:")
    original_model = model_package['model']
    
    # Feature reduction
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=3)  # Keep top 3 features
    
    X_train = model_package['X_test']  # Use test as example
    y_train = model_package['y_test']
    
    X_reduced = selector.fit_transform(X_train, y_train)
    
    # Train lighter model
    light_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    light_model.fit(X_reduced, y_train)
    
    # Compare sizes
    original_size = len(joblib.dumps(original_model))
    light_size = len(joblib.dumps(light_model))
    
    print(f"    Original model size: {original_size:,} bytes")
    print(f"    Compressed model size: {light_size:,} bytes")
    print(f"    Size reduction: {(1 - light_size/original_size)*100:.1f}%")
    
    # 2. Quantization simulation
    print(f"\n  2. Quantization (simulated):")
    
    # Convert model parameters to lower precision
    def quantize_weights(model, bits=8):
        """Simulate weight quantization"""
        # This is a simplified simulation
        total_params = 0
        for estimator in model.estimators_:
            total_params += estimator.tree_.node_count
        
        original_bits = 64  # float64
        compressed_bits = bits
        compression_ratio = original_bits / compressed_bits
        
        return {
            'original_size': total_params * original_bits / 8,
            'quantized_size': total_params * compressed_bits / 8,
            'compression_ratio': compression_ratio
        }
    
    quant_info = quantize_weights(light_model)
    print(f"    8-bit quantization:")
    print(f"      Original: {quant_info['original_size']:.0f} bytes")
    print(f"      Quantized: {quant_info['quantized_size']:.0f} bytes")
    print(f"      Compression: {quant_info['compression_ratio']:.1f}x")
    
    return light_model, selector

light_model, feature_selector = optimize_model_for_edge()

# B. Mobile Deployment
print(f"\nB. Mobile Deployment Options:")

mobile_options = {
    "TensorFlow Lite": {
        "Description": "Optimized for mobile and edge devices",
        "Pros": ["Small size", "Fast inference", "Hardware acceleration"],
        "Cons": ["Limited to TensorFlow models", "Conversion required"],
        "Use Case": "Mobile apps, IoT devices"
    },
    "ONNX Runtime": {
        "Description": "Cross-platform inference runtime",
        "Pros": ["Multi-framework support", "Optimized kernels", "Hardware acceleration"],
        "Cons": ["Model conversion needed", "Larger size than TFLite"],
        "Use Case": "Cross-platform deployment"
    },
    "Core ML (iOS)": {
        "Description": "Apple's ML framework for iOS/macOS",
        "Pros": ["Native iOS integration", "Hardware optimization", "Privacy"],
        "Cons": ["iOS/macOS only", "Limited model types"],
        "Use Case": "iOS applications"
    },
    "ML Kit (Android)": {
        "Description": "Google's mobile ML SDK",
        "Pros": ["Easy integration", "On-device + cloud", "Pre-trained models"],
        "Cons": ["Limited customization", "Google ecosystem"],
        "Use Case": "Android applications"
    }
}

for option, details in mobile_options.items():
    print(f"  {option}:")
    print(f"    Description: {details['Description']}")
    print(f"    Use Case: {details['Use Case']}")
    print(f"    Pros: {', '.join(details['Pros'])}")

# C. IoT and Embedded Deployment
print(f"\nC. IoT and Embedded Deployment:")

iot_deployment_code = '''
# Raspberry Pi deployment example
import numpy as np
import joblib
import time
from datetime import datetime
import json

class EdgeMLInference:
    """Lightweight ML inference for edge devices"""
    
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.inference_count = 0
        self.total_inference_time = 0
    
    def predict(self, features):
        """Make prediction with timing"""
        start_time = time.time()
        
        # Preprocess
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        return {
            'prediction': int(prediction),
            'confidence': float(probability[prediction]),
            'inference_time_ms': inference_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_stats(self):
        """Get performance statistics"""
        avg_time = self.total_inference_time / max(self.inference_count, 1)
        return {
            'total_inferences': self.inference_count,
            'average_inference_time_ms': avg_time * 1000,
            'throughput_per_second': 1 / avg_time if avg_time > 0 else 0
        }

# Usage on edge device:
# edge_model = EdgeMLInference('model.joblib', 'scaler.joblib')
# result = edge_model.predict([5.1, 3.5, 1.4, 0.2])
# stats = edge_model.get_stats()
'''

print(f"  IoT deployment considerations:")
print(f"    • Resource constraints (CPU, memory, storage)")
print(f"    • Power efficiency")
print(f"    • Offline operation capability")
print(f"    • Model size optimization")
print(f"    • Update mechanisms")

print()

# 6. MONITORING AND MAINTENANCE
print("6. MONITORING AND MAINTENANCE")
print("-" * 33)

print("Theory: Deployed models require continuous monitoring")
print("for performance, data drift, and system health.")
print()

# A. Model Performance Monitoring
print("A. Model Performance Monitoring:")

monitoring_code = '''
import time
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

class ModelMonitor:
    """Monitor model performance and health"""
    
    def __init__(self, model_name, window_size=1000):
        self.model_name = model_name
        self.window_size = window_size
        
        # Metrics tracking
        self.predictions = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.error_count = 0
        self.total_requests = 0
        
        # Data drift detection
        self.feature_stats = defaultdict(lambda: {'mean': 0, 'std': 0, 'count': 0})
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"ModelMonitor-{model_name}")
    
    def log_prediction(self, features, prediction, response_time, error=None):
        """Log a prediction event"""
        self.total_requests += 1
        
        if error:
            self.error_count += 1
            self.logger.error(f"Prediction error: {error}")
        else:
            self.predictions.append({
                'features': features,
                'prediction': prediction,
                'timestamp': datetime.now(),
                'response_time': response_time
            })
            self.response_times.append(response_time)
            
            # Update feature statistics
            self._update_feature_stats(features)
    
    def _update_feature_stats(self, features):
        """Update running statistics for features"""
        for i, value in enumerate(features):
            stats = self.feature_stats[f'feature_{i}']
            count = stats['count']
            
            # Update running mean and std
            if count == 0:
                stats['mean'] = value
                stats['std'] = 0
            else:
                # Online update formulas
                new_mean = (stats['mean'] * count + value) / (count + 1)
                stats['std'] = np.sqrt(
                    (stats['std']**2 * count + (value - stats['mean']) * (value - new_mean)) / (count + 1)
                )
                stats['mean'] = new_mean
            
            stats['count'] = count + 1
    
    def get_metrics(self):
        """Get current performance metrics"""
        if not self.response_times:
            return {'error': 'No data available'}
        
        return {
            'model_name': self.model_name,
            'total_requests': self.total_requests,
            'error_rate': self.error_count / self.total_requests,
            'avg_response_time_ms': np.mean(self.response_times) * 1000,
            'p95_response_time_ms': np.percentile(self.response_times, 95) * 1000,
            'requests_per_minute': len(self.predictions),
            'window_size': len(self.predictions),
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_drift(self, new_features, threshold=2.0):
        """Simple data drift detection"""
        drift_detected = False
        drift_features = []
        
        for i, value in enumerate(new_features):
            feature_key = f'feature_{i}'
            if feature_key in self.feature_stats:
                stats = self.feature_stats[feature_key]
                if stats['count'] > 10:  # Need sufficient data
                    z_score = abs(value - stats['mean']) / (stats['std'] + 1e-8)
                    if z_score > threshold:
                        drift_detected = True
                        drift_features.append({
                            'feature': i,
                            'value': value,
                            'z_score': z_score,
                            'expected_mean': stats['mean'],
                            'expected_std': stats['std']
                        })
        
        if drift_detected:
            self.logger.warning(f"Data drift detected in features: {drift_features}")
        
        return drift_detected, drift_features

# Usage:
# monitor = ModelMonitor("iris-classifier")
# monitor.log_prediction([5.1, 3.5, 1.4, 0.2], 0, 0.05)
# metrics = monitor.get_metrics()
# drift_detected, drift_info = monitor.detect_drift([10.0, 10.0, 10.0, 10.0])
'''

print(f"  Monitoring components:")
print(f"    • Response time tracking")
print(f"    • Error rate monitoring")
print(f"    • Throughput measurement")
print(f"    • Data drift detection")
print(f"    • Model accuracy tracking")

# B. Health Checks and Alerts
print(f"\nB. Health Checks and Alerts:")

health_check_code = '''
class ModelHealthChecker:
    """Comprehensive health checking for deployed models"""
    
    def __init__(self, model, scaler, test_cases):
        self.model = model
        self.scaler = scaler
        self.test_cases = test_cases
        self.baseline_accuracy = None
    
    def system_health_check(self):
        """Check system-level health"""
        checks = {}
        
        # Memory usage
        import psutil
        memory = psutil.virtual_memory()
        checks['memory_usage_percent'] = memory.percent
        checks['memory_available_gb'] = memory.available / (1024**3)
        
        # CPU usage
        checks['cpu_usage_percent'] = psutil.cpu_percent(interval=1)
        
        # Disk space
        disk = psutil.disk_usage('/')
        checks['disk_usage_percent'] = (disk.used / disk.total) * 100
        
        return checks
    
    def model_health_check(self):
        """Check model-specific health"""
        checks = {}
        
        try:
            # Test prediction latency
            start_time = time.time()
            test_features = self.test_cases[0]['features']
            features_scaled = self.scaler.transform([test_features])
            prediction = self.model.predict(features_scaled)[0]
            latency = time.time() - start_time
            
            checks['prediction_latency_ms'] = latency * 1000
            checks['prediction_successful'] = True
            
            # Test all test cases
            correct_predictions = 0
            for test_case in self.test_cases:
                features_scaled = self.scaler.transform([test_case['features']])
                pred = self.model.predict(features_scaled)[0]
                if pred == test_case['expected']:
                    correct_predictions += 1
            
            current_accuracy = correct_predictions / len(self.test_cases)
            checks['test_accuracy'] = current_accuracy
            
            # Compare with baseline
            if self.baseline_accuracy is None:
                self.baseline_accuracy = current_accuracy
            
            accuracy_drop = self.baseline_accuracy - current_accuracy
            checks['accuracy_drop'] = accuracy_drop
            checks['accuracy_degradation'] = accuracy_drop > 0.05  # 5% threshold
            
        except Exception as e:
            checks['prediction_successful'] = False
            checks['error'] = str(e)
        
        return checks
    
    def comprehensive_health_check(self):
        """Run all health checks"""
        return {
            'timestamp': datetime.now().isoformat(),
            'system': self.system_health_check(),
            'model': self.model_health_check()
        }

# Test cases for health checking
test_cases = [
    {'features': [5.1, 3.5, 1.4, 0.2], 'expected': 0},  # setosa
    {'features': [7.0, 3.2, 4.7, 1.4], 'expected': 1},  # versicolor
    {'features': [6.3, 3.3, 6.0, 2.5], 'expected': 2},  # virginica
]

# health_checker = ModelHealthChecker(model, scaler, test_cases)
# health_status = health_checker.comprehensive_health_check()
'''

print(f"  Health check components:")
print(f"    • System resource monitoring")
print(f"    • Model prediction testing")
print(f"    • Accuracy regression detection")
print(f"    • Latency monitoring")
print(f"    • Error rate tracking")

print()

# 7. DEPLOYMENT COMPARISON AND RECOMMENDATIONS
print("7. DEPLOYMENT COMPARISON AND RECOMMENDATIONS")
print("-" * 50)

# Create comparison visualization
plt.figure(figsize=(16, 12))

# Deployment options comparison
deployment_options = {
    'Local Script': {'Complexity': 1, 'Scalability': 1, 'Cost': 1, 'Maintenance': 2},
    'Flask API': {'Complexity': 3, 'Scalability': 3, 'Cost': 2, 'Maintenance': 3},
    'FastAPI': {'Complexity': 3, 'Scalability': 4, 'Cost': 2, 'Maintenance': 3},
    'Docker': {'Complexity': 4, 'Scalability': 4, 'Cost': 3, 'Maintenance': 4},
    'AWS Lambda': {'Complexity': 4, 'Scalability': 5, 'Cost': 3, 'Maintenance': 2},
    'AWS ECS': {'Complexity': 5, 'Scalability': 5, 'Cost': 4, 'Maintenance': 3},
    'SageMaker': {'Complexity': 4, 'Scalability': 5, 'Cost': 5, 'Maintenance': 2},
    'Edge/Mobile': {'Complexity': 5, 'Scalability': 2, 'Cost': 2, 'Maintenance': 4}
}

# Plot 1: Complexity vs Scalability
plt.subplot(2, 3, 1)
complexities = [info['Complexity'] for info in deployment_options.values()]
scalabilities = [info['Scalability'] for info in deployment_options.values()]
names = list(deployment_options.keys())

scatter = plt.scatter(complexities, scalabilities, s=100, alpha=0.7)
for i, name in enumerate(names):
    plt.annotate(name, (complexities[i], scalabilities[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('Implementation Complexity (1-5)')
plt.ylabel('Scalability (1-5)')
plt.title('Deployment Options: Complexity vs Scalability')
plt.grid(True, alpha=0.3)

# Plot 2: Cost vs Maintenance
plt.subplot(2, 3, 2)
costs = [info['Cost'] for info in deployment_options.values()]
maintenances = [info['Maintenance'] for info in deployment_options.values()]

scatter = plt.scatter(costs, maintenances, s=100, alpha=0.7, color='orange')
for i, name in enumerate(names):
    plt.annotate(name, (costs[i], maintenances[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('Cost (1-5)')
plt.ylabel('Maintenance Effort (1-5)')
plt.title('Deployment Options: Cost vs Maintenance')
plt.grid(True, alpha=0.3)

# Plot 3: Overall scoring
plt.subplot(2, 3, 3)
# Calculate overall score (lower is better for cost and maintenance)
overall_scores = []
for name, metrics in deployment_options.items():
    score = (metrics['Complexity'] + metrics['Scalability'] - metrics['Cost'] - metrics['Maintenance']) / 4
    overall_scores.append(score)

bars = plt.bar(range(len(names)), overall_scores, alpha=0.7)
plt.xticks(range(len(names)), [name[:8] for name in names], rotation=45, ha='right')
plt.ylabel('Overall Score')
plt.title('Deployment Options: Overall Rating')
plt.grid(True, alpha=0.3)

# Color bars based on score
for bar, score in zip(bars, overall_scores):
    if score > 0.5:
        bar.set_color('green')
    elif score > 0:
        bar.set_color('yellow')
    else:
        bar.set_color('red')

# Plot 4: Model size optimization
plt.subplot(2, 3, 4)
model_sizes = ['Original\nModel', 'Feature\nReduced', 'Quantized\n(8-bit)', 'Edge\nOptimized']
size_values = [100, 60, 30, 15]  # Relative sizes
colors = ['red', 'orange', 'yellow', 'green']

bars = plt.bar(model_sizes, size_values, color=colors, alpha=0.7)
plt.ylabel('Relative Model Size (%)')
plt.title('Model Optimization for Deployment')
for bar, size in zip(bars, size_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{size}%', ha='center', va='bottom')

# Plot 5: Response time comparison
plt.subplot(2, 3, 5)
response_times = {
    'Local': 0.5,
    'Flask API': 2.0,
    'FastAPI': 1.5,
    'Lambda': 50.0,  # Cold start
    'Container': 1.0,
    'Edge': 0.1
}

platforms = list(response_times.keys())
times = list(response_times.values())

bars = plt.bar(platforms, times, alpha=0.7)
plt.ylabel('Response Time (ms)')
plt.title('Typical Response Times')
plt.xticks(rotation=45)
plt.yscale('log')

for bar, time in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{time}ms', ha='center', va='bottom')

# Plot 6: Deployment decision tree
plt.subplot(2, 3, 6)
plt.text(0.1, 0.9, 'Deployment Decision Guide', fontsize=14, fontweight='bold')
plt.text(0.1, 0.8, '• Prototype/Development → Local Script', fontsize=10)
plt.text(0.1, 0.7, '• Small Web App → Flask/FastAPI', fontsize=10)
plt.text(0.1, 0.6, '• Production Web Service → Docker + Cloud', fontsize=10)
plt.text(0.1, 0.5, '• Serverless/Variable Load → AWS Lambda', fontsize=10)
plt.text(0.1, 0.4, '• Enterprise ML → SageMaker/AML', fontsize=10)
plt.text(0.1, 0.3, '• Mobile/IoT → Edge Deployment', fontsize=10)
plt.text(0.1, 0.2, '• High Performance → ECS/Kubernetes', fontsize=10)
plt.text(0.1, 0.1, '• Cost Sensitive → EC2/VPS', fontsize=10)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.title('When to Use Each Option')

plt.tight_layout()
plt.show()

print()

# Deployment recommendations
print("Deployment Recommendations by Use Case:")
print()

recommendations = {
    "Development & Prototyping": {
        "Recommended": ["Local Script", "Jupyter Notebook", "Flask (dev mode)"],
        "Rationale": "Fast iteration, easy debugging, minimal setup"
    },
    "Small Web Applications": {
        "Recommended": ["Flask + Gunicorn", "FastAPI + Uvicorn"],
        "Rationale": "Simple deployment, good performance, easy maintenance"
    },
    "Production Web Services": {
        "Recommended": ["Docker + Cloud Provider", "Kubernetes"],
        "Rationale": "Scalability, reliability, container orchestration"
    },
    "Serverless Applications": {
        "Recommended": ["AWS Lambda", "Google Cloud Functions", "Azure Functions"],
        "Rationale": "Pay-per-use, auto-scaling, no server management"
    },
    "Enterprise ML Platform": {
        "Recommended": ["AWS SageMaker", "Azure ML", "Google AI Platform"],
        "Rationale": "ML-specific features, monitoring, A/B testing"
    },
    "Mobile Applications": {
        "Recommended": ["TensorFlow Lite", "Core ML", "ONNX Runtime"],
        "Rationale": "Optimized for mobile, offline capability, low latency"
    },
    "IoT & Edge Devices": {
        "Recommended": ["Optimized Models", "ONNX Runtime", "Custom Inference"],
        "Rationale": "Resource constraints, offline operation, low power"
    },
    "High-Performance Computing": {
        "Recommended": ["GPU Instances", "Model Serving Frameworks", "Kubernetes"],
        "Rationale": "High throughput, GPU acceleration, batch processing"
    }
}

for use_case, info in recommendations.items():
    print(f"{use_case}:")
    print(f"  Recommended: {', '.join(info['Recommended'])}")
    print(f"  Rationale: {info['Rationale']}")
    print()

print(f"{'='*60}")
print("SUMMARY: ML Model Deployment in Python")
print(f"{'='*60}")

summary_points = [
    "Local deployment: Simple scripts and web APIs for development",
    "Containerization: Docker provides consistent deployment environments",
    "Cloud platforms: Managed services with auto-scaling and monitoring",
    "Serverless: Pay-per-use model for variable workloads",
    "Edge deployment: Optimized models for mobile and IoT devices",
    "Monitoring: Essential for production model health and performance",
    "Choose based on: Scalability needs, cost constraints, maintenance capacity",
    "Start simple: Begin with basic deployment and scale as needed",
    "Consider lifecycle: Plan for model updates and version management",
    "Test thoroughly: Validate deployment before production release"
]

for i, point in enumerate(summary_points, 1):
    print(f"{i:2d}. {point}")

print(f"\n=== Deployment: Bringing ML Models to Production ===")
```

### Explanation

Machine learning model deployment in Python offers multiple options depending on requirements for scalability, performance, cost, and maintenance. The choice depends on factors like expected traffic, latency requirements, team expertise, and budget constraints.

### Deployment Categories

1. **Local Deployment**
   - **Pickle/Joblib Files**: Simple serialization for small applications
   - **Python Scripts**: Standalone executables for command-line usage
   - **Use Cases**: Development, testing, small-scale applications

2. **Web API Deployment**
   - **Flask**: Lightweight framework for simple APIs
   - **FastAPI**: Modern framework with automatic documentation and type validation
   - **Features**: REST endpoints, JSON input/output, scalable architecture

3. **Containerized Deployment**
   - **Docker**: Consistent environments across platforms
   - **Docker Compose**: Multi-service orchestration
   - **Benefits**: Reproducibility, scalability, easier maintenance

4. **Cloud Deployment**
   - **AWS**: Lambda (serverless), ECS (containers), SageMaker (ML-specific)
   - **Google Cloud**: AI Platform, Cloud Run, Cloud Functions
   - **Azure**: ML Service, Container Instances, Functions

5. **Edge Deployment**
   - **Mobile**: TensorFlow Lite, Core ML, ONNX Runtime
   - **IoT**: Optimized models for resource-constrained devices
   - **Benefits**: Low latency, offline capability, privacy

### Key Considerations

**Scalability:**
- Start with simple deployment and scale as needed
- Consider auto-scaling capabilities
- Plan for traffic spikes and growth

**Performance:**
- Monitor response times and throughput
- Optimize model size for deployment target
- Use appropriate hardware (CPU/GPU)

**Cost:**
- Balance infrastructure costs with performance needs
- Consider pay-per-use vs. fixed cost models
- Factor in maintenance and operational costs

**Reliability:**
- Implement health checks and monitoring
- Plan for failover and redundancy
- Monitor model performance and data drift

**Security:**
- Secure API endpoints with authentication
- Protect model intellectual property
- Ensure data privacy compliance

### Best Practices

1. **Start Simple**: Begin with basic deployment and iterate
2. **Monitor Everything**: Track performance, errors, and resource usage
3. **Version Control**: Manage model versions and rollback capabilities
4. **Test Thoroughly**: Validate deployment in staging environment
5. **Document Process**: Maintain deployment procedures and configurations
6. **Plan Updates**: Design for model retraining and deployment updates
7. **Consider Costs**: Balance performance needs with budget constraints
8. **Security First**: Implement proper authentication and data protection

### Common Deployment Pipeline

1. **Model Training**: Train and validate model
2. **Model Packaging**: Serialize model with dependencies
3. **Infrastructure Setup**: Prepare deployment environment
4. **API Development**: Create serving endpoints
5. **Testing**: Validate functionality and performance
6. **Deployment**: Deploy to production environment
7. **Monitoring**: Track performance and health
8. **Maintenance**: Update and retrain as needed

### Technology Stack Examples

**Simple Web Service:**
- FastAPI + Uvicorn + Docker + AWS ECS

**Serverless Solution:**
- AWS Lambda + API Gateway + S3

**Enterprise Platform:**
- Kubernetes + Docker + GPU instances + monitoring stack

**Mobile Application:**
- TensorFlow Lite + mobile app integration

The choice of deployment strategy should align with business requirements, technical constraints, and team capabilities, starting simple and evolving as needs grow.

---

## Question 14

**Discuss strategies for effective logging and monitoring in machine-learning applications.**

### Theory
Effective logging and monitoring are crucial for ML applications to track model performance, debug issues, detect data drift, and ensure reliable operation in production. This involves capturing relevant metrics, setting up alerts, and maintaining observability across the ML pipeline.

### Answer

```python
# ml_logging_monitoring.py - Comprehensive ML logging and monitoring strategies
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=== ML Logging and Monitoring Strategies ===\n")

# 1. STRUCTURED LOGGING SETUP
print("1. STRUCTURED LOGGING SETUP")
print("-" * 30)

class MLLogger:
    """Structured logging for ML applications"""
    
    def __init__(self, name, log_file=None, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_model_training(self, model_name, metrics, duration):
        """Log model training completion"""
        log_data = {
            'event': 'model_training_complete',
            'model_name': model_name,
            'metrics': metrics,
            'training_duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(f"Training completed: {json.dumps(log_data)}")
    
    def log_prediction(self, model_name, input_features, prediction, confidence, response_time):
        """Log individual predictions"""
        log_data = {
            'event': 'prediction_made',
            'model_name': model_name,
            'prediction': prediction,
            'confidence': confidence,
            'response_time_ms': response_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(f"Prediction: {json.dumps(log_data)}")
    
    def log_error(self, error_type, error_message, context=None):
        """Log errors with context"""
        log_data = {
            'event': 'error_occurred',
            'error_type': error_type,
            'error_message': str(error_message),
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        self.logger.error(f"Error: {json.dumps(log_data)}")

# Initialize logger
ml_logger = MLLogger("MLApplication", "ml_app.log")

print("Structured logging configured with:")
print("  • JSON-formatted log entries")
print("  • Timestamp and context tracking")
print("  • Console and file output")
print("  • Event-specific logging methods")

# 2. PERFORMANCE MONITORING
print(f"\n2. PERFORMANCE MONITORING")
print("-" * 26)

class PerformanceMonitor:
    """Monitor ML model and system performance"""
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.response_times = deque(maxlen=window_size)
        self.error_count = 0
        self.total_requests = 0
        
    def record_prediction(self, actual=None, predicted=None, response_time=None, error=None):
        """Record a prediction event"""
        self.total_requests += 1
        
        if error:
            self.error_count += 1
            ml_logger.log_error("PredictionError", error)
        else:
            if response_time:
                self.response_times.append(response_time)
            
            if actual is not None and predicted is not None:
                # Calculate metrics
                accuracy = 1 if actual == predicted else 0
                self.metrics_history['accuracy'].append(accuracy)
                
                # Log successful prediction
                ml_logger.log_prediction("RandomForest", [], predicted, 0.85, response_time or 0.01)
    
    def get_current_metrics(self):
        """Get current performance metrics"""
        if not self.metrics_history['accuracy']:
            return {"status": "No data available"}
        
        metrics = {
            'accuracy': np.mean(self.metrics_history['accuracy']),
            'error_rate': self.error_count / max(self.total_requests, 1),
            'avg_response_time_ms': np.mean(self.response_times) * 1000 if self.response_times else 0,
            'p95_response_time_ms': np.percentile(self.response_times, 95) * 1000 if self.response_times else 0,
            'total_requests': self.total_requests,
            'window_size': len(self.metrics_history['accuracy'])
        }
        
        return metrics
    
    def check_alerts(self):
        """Check for alert conditions"""
        alerts = []
        metrics = self.get_current_metrics()
        
        # Accuracy degradation
        if metrics.get('accuracy', 1) < 0.8:
            alerts.append({
                'type': 'accuracy_degradation',
                'value': metrics['accuracy'],
                'threshold': 0.8,
                'severity': 'high'
            })
        
        # High error rate
        if metrics.get('error_rate', 0) > 0.05:
            alerts.append({
                'type': 'high_error_rate',
                'value': metrics['error_rate'],
                'threshold': 0.05,
                'severity': 'medium'
            })
        
        # Slow response time
        if metrics.get('p95_response_time_ms', 0) > 1000:
            alerts.append({
                'type': 'slow_response',
                'value': metrics['p95_response_time_ms'],
                'threshold': 1000,
                'severity': 'medium'
            })
        
        return alerts

# Initialize performance monitor
perf_monitor = PerformanceMonitor()

# Simulate some predictions for demonstration
np.random.seed(42)
for i in range(100):
    actual = np.random.randint(0, 3)
    predicted = actual if np.random.random() > 0.1 else np.random.randint(0, 3)  # 90% accuracy
    response_time = np.random.exponential(0.05)  # Exponential distribution for response times
    
    perf_monitor.record_prediction(actual, predicted, response_time)

current_metrics = perf_monitor.get_current_metrics()
print("Current Performance Metrics:")
for key, value in current_metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Check for alerts
alerts = perf_monitor.check_alerts()
print(f"\nActive Alerts: {len(alerts)}")
for alert in alerts:
    print(f"  • {alert['type']}: {alert['value']:.4f} (threshold: {alert['threshold']})")

# 3. DATA DRIFT MONITORING
print(f"\n3. DATA DRIFT MONITORING")
print("-" * 26)

class DataDriftMonitor:
    """Monitor for data drift in input features"""
    
    def __init__(self, reference_data, feature_names):
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.reference_stats = self._calculate_stats(reference_data)
        self.drift_threshold = 0.1  # 10% change threshold
        
    def _calculate_stats(self, data):
        """Calculate reference statistics"""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
    
    def detect_drift(self, new_data):
        """Detect data drift in new data"""
        new_stats = self._calculate_stats(new_data)
        drift_results = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Compare means (normalized by reference std)
            mean_diff = abs(new_stats['mean'][i] - self.reference_stats['mean'][i])
            normalized_diff = mean_diff / (self.reference_stats['std'][i] + 1e-8)
            
            drift_detected = normalized_diff > self.drift_threshold
            
            drift_results[feature_name] = {
                'drift_detected': drift_detected,
                'drift_score': normalized_diff,
                'reference_mean': self.reference_stats['mean'][i],
                'new_mean': new_stats['mean'][i],
                'reference_std': self.reference_stats['std'][i],
                'new_std': new_stats['std'][i]
            }
            
            if drift_detected:
                ml_logger.logger.warning(f"Data drift detected in {feature_name}: {normalized_diff:.4f}")
        
        return drift_results

# Demo data drift monitoring
X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, random_state=42)
feature_names = [f'feature_{i}' for i in range(4)]

# Split into reference and new data
X_ref = X[:500]
X_new = X[500:] + np.random.normal(0, 0.5, X[500:].shape)  # Add drift

drift_monitor = DataDriftMonitor(X_ref, feature_names)
drift_results = drift_monitor.detect_drift(X_new)

print("Data Drift Analysis:")
for feature, result in drift_results.items():
    status = "DRIFT DETECTED" if result['drift_detected'] else "No drift"
    print(f"  {feature}: {status} (score: {result['drift_score']:.4f})")

# 4. MODEL VERSIONING AND EXPERIMENT TRACKING
print(f"\n4. MODEL VERSIONING AND EXPERIMENT TRACKING")
print("-" * 45)

class ExperimentTracker:
    """Track ML experiments and model versions"""
    
    def __init__(self):
        self.experiments = {}
        self.current_experiment = None
        
    def start_experiment(self, name, description=""):
        """Start a new experiment"""
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiments[experiment_id] = {
            'name': name,
            'description': description,
            'start_time': datetime.now(),
            'parameters': {},
            'metrics': {},
            'artifacts': [],
            'status': 'running'
        }
        
        self.current_experiment = experiment_id
        ml_logger.logger.info(f"Started experiment: {experiment_id}")
        return experiment_id
    
    def log_parameters(self, params):
        """Log experiment parameters"""
        if self.current_experiment:
            self.experiments[self.current_experiment]['parameters'].update(params)
            ml_logger.logger.info(f"Logged parameters: {json.dumps(params)}")
    
    def log_metrics(self, metrics):
        """Log experiment metrics"""
        if self.current_experiment:
            self.experiments[self.current_experiment]['metrics'].update(metrics)
            ml_logger.logger.info(f"Logged metrics: {json.dumps(metrics)}")
    
    def log_artifact(self, artifact_path, artifact_type="model"):
        """Log experiment artifacts"""
        if self.current_experiment:
            artifact = {
                'path': artifact_path,
                'type': artifact_type,
                'timestamp': datetime.now().isoformat()
            }
            self.experiments[self.current_experiment]['artifacts'].append(artifact)
            ml_logger.logger.info(f"Logged artifact: {artifact_path}")
    
    def end_experiment(self, status="completed"):
        """End current experiment"""
        if self.current_experiment:
            self.experiments[self.current_experiment]['status'] = status
            self.experiments[self.current_experiment]['end_time'] = datetime.now()
            
            duration = (self.experiments[self.current_experiment]['end_time'] - 
                       self.experiments[self.current_experiment]['start_time'])
            
            ml_logger.log_model_training(
                self.experiments[self.current_experiment]['name'],
                self.experiments[self.current_experiment]['metrics'],
                duration.total_seconds()
            )
            
            self.current_experiment = None

# Demo experiment tracking
tracker = ExperimentTracker()

# Start experiment
exp_id = tracker.start_experiment("random_forest_optimization", "Hyperparameter tuning")

# Log parameters and train model
params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
tracker.log_parameters(params)

# Quick model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)

# Log metrics
y_pred = model.predict(X_test)
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, average='weighted'),
    'recall': recall_score(y_test, y_pred, average='weighted'),
    'f1': f1_score(y_test, y_pred, average='weighted')
}
tracker.log_metrics(metrics)

# Log artifacts
tracker.log_artifact("model_v1.joblib", "model")
tracker.log_artifact("scaler_v1.joblib", "preprocessor")

# End experiment
tracker.end_experiment()

print("Experiment Tracking Demo:")
print(f"  Experiment ID: {exp_id}")
print(f"  Parameters logged: {len(params)} items")
print(f"  Metrics logged: {len(metrics)} items")
print(f"  Artifacts logged: 2 items")

print(f"\n{'='*50}")
print("SUMMARY: ML Logging and Monitoring Best Practices")
print(f"{'='*50}")

summary_points = [
    "Structured logging: Use JSON format for machine-readable logs",
    "Performance monitoring: Track accuracy, latency, and error rates",
    "Data drift detection: Monitor input data distribution changes",
    "Experiment tracking: Version models with parameters and metrics",
    "Alert systems: Set thresholds for automated notifications",
    "Centralized logging: Aggregate logs from all ML pipeline components",
    "Real-time dashboards: Visualize key metrics for quick assessment",
    "Retention policies: Define log storage and cleanup strategies"
]

for i, point in enumerate(summary_points, 1):
    print(f"{i}. {point}")

print(f"\n=== Essential Monitoring: Keep ML Systems Healthy ===")
```

### Explanation

Effective logging and monitoring in ML applications requires a systematic approach covering multiple aspects of the system lifecycle.

### Key Components

**1. Structured Logging**
- JSON-formatted log entries for machine parsing
- Event-specific logging methods (training, prediction, errors)
- Contextual information with timestamps
- Multiple output targets (console, files, external systems)

**2. Performance Monitoring**
- Model accuracy and quality metrics
- Response time and throughput tracking
- Error rate monitoring
- Resource utilization metrics

**3. Data Drift Detection**
- Statistical comparison of input features
- Distribution shift monitoring
- Threshold-based alerting
- Feature-level drift analysis

**4. Experiment Tracking**
- Model version management
- Parameter and hyperparameter logging
- Metric comparison across experiments
- Artifact storage and retrieval

### Best Practices

1. **Comprehensive Coverage**: Monitor all pipeline stages
2. **Real-time Alerting**: Set up automated notifications
3. **Historical Analysis**: Maintain long-term metric trends
4. **Actionable Insights**: Focus on metrics that drive decisions
5. **Performance Balance**: Avoid excessive logging overhead
6. **Security Considerations**: Protect sensitive data in logs
7. **Standardization**: Use consistent logging formats across teams
8. **Integration**: Connect with existing monitoring infrastructure

This monitoring strategy ensures ML applications remain reliable, performant, and maintainable in production environments.

## Question 15

**Discuss the implications of quantum computing on machine learning, with a Python perspective.**

### Theory
Quantum computing represents a paradigm shift that could revolutionize machine learning by leveraging quantum mechanical phenomena like superposition and entanglement. While still emerging, quantum ML (QML) offers potential advantages for specific problems involving optimization, sampling, and linear algebra operations.

### Answer

```python
# quantum_ml_python.py - Quantum Computing implications for ML in Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

print("=== Quantum Computing Implications for Machine Learning ===\n")

# Note: This demonstrates concepts - actual quantum computing requires specialized hardware
# We'll simulate quantum concepts and show Python frameworks for quantum ML

# 1. QUANTUM COMPUTING FUNDAMENTALS FOR ML
print("1. QUANTUM COMPUTING FUNDAMENTALS FOR ML")
print("-" * 42)

class QuantumConceptDemo:
    """Demonstrate quantum computing concepts relevant to ML"""
    
    def __init__(self):
        self.quantum_advantage_problems = [
            "Optimization problems with exponential search spaces",
            "Sampling from complex probability distributions", 
            "Linear algebra operations on large matrices",
            "Pattern recognition in high-dimensional spaces",
            "Feature mapping to quantum Hilbert spaces"
        ]
    
    def simulate_superposition(self, classical_bits):
        """Simulate quantum superposition concept"""
        n_bits = len(classical_bits)
        n_states = 2 ** n_bits
        
        # Classical: one definite state
        classical_state = int(''.join(map(str, classical_bits)), 2)
        
        # Quantum: superposition of all possible states
        quantum_amplitudes = np.random.random(n_states) + 1j * np.random.random(n_states)
        quantum_amplitudes = quantum_amplitudes / np.linalg.norm(quantum_amplitudes)
        
        return {
            'classical_state': classical_state,
            'quantum_states': n_states,
            'superposition_advantage': f"{n_states}x parallel computation potential"
        }
    
    def quantum_vs_classical_complexity(self):
        """Compare complexity for ML-relevant problems"""
        problems = {
            'Matrix multiplication (n×n)': {
                'classical': 'O(n³)',
                'quantum_potential': 'O(n²·³⁷⁶) or better with quantum algorithms'
            },
            'Unstructured search': {
                'classical': 'O(n)',
                'quantum_potential': 'O(√n) with Grover\'s algorithm'
            },
            'Integer factorization': {
                'classical': 'O(exp(n))',
                'quantum_potential': 'O(n³) with Shor\'s algorithm'
            },
            'Sampling from probability distributions': {
                'classical': 'O(exp(n)) for complex distributions',
                'quantum_potential': 'O(poly(n)) for certain distributions'
            }
        }
        return problems

# Demo quantum concepts
quantum_demo = QuantumConceptDemo()

print("Quantum Computing Advantages for ML:")
for advantage in quantum_demo.quantum_advantage_problems:
    print(f"  • {advantage}")

superposition_demo = quantum_demo.simulate_superposition([1, 0, 1])
print(f"\nSuperposition Example:")
print(f"  Classical state: {superposition_demo['classical_state']}")
print(f"  Quantum states: {superposition_demo['quantum_states']}")
print(f"  Advantage: {superposition_demo['superposition_advantage']}")

complexity_comparison = quantum_demo.quantum_vs_classical_complexity()
print(f"\nComplexity Comparison:")
for problem, complexities in complexity_comparison.items():
    print(f"  {problem}:")
    print(f"    Classical: {complexities['classical']}")
    print(f"    Quantum: {complexities['quantum_potential']}")

# 2. PYTHON FRAMEWORKS FOR QUANTUM ML
print(f"\n2. PYTHON FRAMEWORKS FOR QUANTUM ML")
print("-" * 38)

class QuantumMLFrameworks:
    """Overview of Python frameworks for quantum ML"""
    
    def __init__(self):
        self.frameworks = {
            'Qiskit': {
                'provider': 'IBM',
                'strengths': ['Hardware access', 'Comprehensive tools', 'Active community'],
                'ml_features': ['Qiskit Machine Learning', 'Quantum kernels', 'Feature maps'],
                'code_example': '''
# Qiskit quantum SVM example
from qiskit_machine_learning.algorithms import QSVC
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel

# Create quantum feature map
feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

# Create quantum kernel
quantum_kernel = QuantumKernel(feature_map=feature_map)

# Create quantum SVM
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)
                '''
            },
            'Cirq': {
                'provider': 'Google',
                'strengths': ['Hardware integration', 'NISQ focus', 'Simulation tools'],
                'ml_features': ['TensorFlow Quantum integration', 'Quantum neural networks'],
                'code_example': '''
# Cirq quantum circuit example
import cirq
import tensorflow_quantum as tfq

# Create quantum circuit
qubits = cirq.GridQubit.rect(1, 2)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1])
)

# Convert to TensorFlow Quantum
quantum_data = tfq.convert_to_tensor([circuit])
                '''
            },
            'PennyLane': {
                'provider': 'Xanadu',
                'strengths': ['ML integration', 'Automatic differentiation', 'Hybrid computing'],
                'ml_features': ['Quantum gradients', 'PyTorch/TensorFlow integration'],
                'code_example': '''
# PennyLane quantum ML example
import pennylane as qml
import torch

# Define quantum device
dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def quantum_model(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(2))
    qml.templates.BasicEntanglerLayers(weights, wires=range(2))
    return qml.expval(qml.PauliZ(0))
                '''
            },
            'Forest (PyQuil)': {
                'provider': 'Rigetti',
                'strengths': ['Quantum cloud access', 'Hybrid algorithms', 'Real hardware'],
                'ml_features': ['Quantum approximate optimization', 'Variational algorithms'],
                'code_example': '''
# PyQuil quantum program example
from pyquil import Program, get_qc
from pyquil.gates import H, CNOT

# Create quantum program
program = Program(
    H(0),
    CNOT(0, 1)
)

# Execute on quantum computer
qc = get_qc('2q-qvm')
result = qc.run_and_measure(program, trials=1000)
                '''
            }
        }
    
    def compare_frameworks(self):
        """Compare quantum ML frameworks"""
        comparison = {}
        for name, details in self.frameworks.items():
            comparison[name] = {
                'provider': details['provider'],
                'ml_integration': len(details['ml_features']),
                'key_strength': details['strengths'][0]
            }
        return comparison

frameworks = QuantumMLFrameworks()
comparison = frameworks.compare_frameworks()

print("Python Quantum ML Frameworks:")
for name, details in comparison.items():
    print(f"  {name} ({details['provider']}):")
    print(f"    Key Strength: {details['key_strength']}")
    print(f"    ML Features: {details['ml_integration']} main areas")

# 3. QUANTUM ML ALGORITHMS AND APPLICATIONS
print(f"\n3. QUANTUM ML ALGORITHMS AND APPLICATIONS")
print("-" * 45)

class QuantumMLAlgorithms:
    """Quantum machine learning algorithms overview"""
    
    def __init__(self):
        self.algorithms = {
            'Quantum SVM': {
                'concept': 'Use quantum feature maps for non-linear classification',
                'advantage': 'Exponentially large feature spaces',
                'python_implementation': 'Qiskit Machine Learning',
                'use_cases': ['High-dimensional classification', 'Kernel methods']
            },
            'Variational Quantum Eigensolver (VQE)': {
                'concept': 'Quantum-classical hybrid optimization',
                'advantage': 'Quantum speedup for certain optimization problems',
                'python_implementation': 'Qiskit, PennyLane, Cirq',
                'use_cases': ['Molecular simulation', 'Portfolio optimization']
            },
            'Quantum Approximate Optimization Algorithm (QAOA)': {
                'concept': 'Variational algorithm for combinatorial optimization',
                'advantage': 'Better solutions for NP-hard problems',
                'python_implementation': 'Qiskit, Forest',
                'use_cases': ['Feature selection', 'Clustering', 'Scheduling']
            },
            'Quantum Neural Networks (QNN)': {
                'concept': 'Neural networks with quantum processing units',
                'advantage': 'Quantum entanglement and superposition in learning',
                'python_implementation': 'TensorFlow Quantum, PennyLane',
                'use_cases': ['Pattern recognition', 'Generative models']
            },
            'Quantum Principal Component Analysis': {
                'concept': 'Quantum algorithm for dimensionality reduction',
                'advantage': 'Exponential speedup for large matrices',
                'python_implementation': 'Qiskit, research implementations',
                'use_cases': ['Data preprocessing', 'Feature extraction']
            }
        }
    
    def get_algorithm_complexity(self, algorithm_name):
        """Get complexity analysis for quantum algorithms"""
        complexities = {
            'Quantum SVM': {
                'training': 'O(M log(N)) where M=samples, N=features',
                'classical_equivalent': 'O(M²N) for kernel SVM',
                'quantum_advantage': 'Exponential in feature space dimension'
            },
            'VQE': {
                'training': 'O(P * D) where P=parameters, D=depth',
                'classical_equivalent': 'O(exp(N)) for exact diagonalization',
                'quantum_advantage': 'Exponential for large quantum systems'
            },
            'QAOA': {
                'training': 'O(P * M) where P=parameters, M=measurements',
                'classical_equivalent': 'O(exp(N)) for exact solution',
                'quantum_advantage': 'Polynomial vs exponential scaling'
            }
        }
        return complexities.get(algorithm_name, {})

algorithms = QuantumMLAlgorithms()

print("Quantum ML Algorithms:")
for name, details in algorithms.algorithms.items():
    print(f"  {name}:")
    print(f"    Concept: {details['concept']}")
    print(f"    Advantage: {details['advantage']}")
    print(f"    Python: {details['python_implementation']}")

# Example complexity analysis
qsvm_complexity = algorithms.get_algorithm_complexity('Quantum SVM')
print(f"\nQuantum SVM Complexity Analysis:")
for aspect, complexity in qsvm_complexity.items():
    print(f"  {aspect}: {complexity}")

# 4. CURRENT LIMITATIONS AND CHALLENGES
print(f"\n4. CURRENT LIMITATIONS AND CHALLENGES")
print("-" * 38)

class QuantumMLChallenges:
    """Current challenges in quantum machine learning"""
    
    def __init__(self):
        self.challenges = {
            'Hardware Limitations': {
                'issues': [
                    'Limited number of qubits (50-1000 in current systems)',
                    'High error rates (0.1-1% per gate operation)',
                    'Short coherence times (microseconds)',
                    'Limited connectivity between qubits'
                ],
                'impact_on_ml': 'Restricts problem size and algorithm complexity'
            },
            'NISQ Era Constraints': {
                'issues': [
                    'Noisy Intermediate-Scale Quantum devices',
                    'No quantum error correction',
                    'Limited circuit depth',
                    'Measurement noise'
                ],
                'impact_on_ml': 'Requires noise-resilient algorithms and error mitigation'
            },
            'Algorithm Development': {
                'issues': [
                    'Few proven quantum advantages for ML',
                    'Need for quantum-classical hybrid approaches',
                    'Difficulty in quantum algorithm design',
                    'Limited theoretical understanding'
                ],
                'impact_on_ml': 'Slow progress in practical quantum ML applications'
            },
            'Data Loading Problem': {
                'issues': [
                    'Classical data must be encoded into quantum states',
                    'Encoding can be exponentially expensive',
                    'Data access bottleneck',
                    'State preparation complexity'
                ],
                'impact_on_ml': 'May eliminate quantum speedup for many problems'
            }
        }
    
    def estimate_quantum_readiness(self, problem_characteristics):
        """Estimate quantum readiness for ML problems"""
        readiness_score = 0
        factors = {
            'high_dimensional': 2,  # Quantum advantage likely
            'optimization_heavy': 2,  # Good fit for QAOA/VQE
            'linear_algebra': 1,  # Some quantum algorithms available
            'small_dataset': -2,  # Classical may be better
            'noise_sensitive': -1,  # Current hardware limitation
            'real_time': -2  # Current hardware too slow
        }
        
        for characteristic, weight in factors.items():
            if characteristic in problem_characteristics:
                readiness_score += weight
        
        if readiness_score >= 3:
            return "High potential for quantum advantage"
        elif readiness_score >= 0:
            return "Moderate potential, worth investigating"
        else:
            return "Classical approaches likely better currently"

challenges = QuantumMLChallenges()

print("Current Quantum ML Challenges:")
for category, details in challenges.challenges.items():
    print(f"  {category}:")
    print(f"    Impact: {details['impact_on_ml']}")
    print(f"    Key Issues: {len(details['issues'])} identified")

# Example readiness assessment
problem1 = ['high_dimensional', 'optimization_heavy']
problem2 = ['small_dataset', 'real_time', 'noise_sensitive']

readiness1 = challenges.estimate_quantum_readiness(problem1)
readiness2 = challenges.estimate_quantum_readiness(problem2)

print(f"\nQuantum Readiness Assessment:")
print(f"  High-dim optimization problem: {readiness1}")
print(f"  Small real-time problem: {readiness2}")

# 5. PRACTICAL PYTHON IMPLEMENTATION EXAMPLE
print(f"\n5. PRACTICAL PYTHON IMPLEMENTATION EXAMPLE")
print("-" * 43)

def simulate_quantum_inspired_algorithm():
    """Simulate quantum-inspired ML algorithm in classical Python"""
    
    # Generate sample data
    X, y = make_classification(n_samples=200, n_features=4, n_classes=2, 
                              random_state=42, n_redundant=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Classical approach
    classical_start = time.time()
    classical_model = SVC(kernel='rbf', gamma='scale')
    classical_model.fit(X_train, y_train)
    classical_pred = classical_model.predict(X_test)
    classical_time = time.time() - classical_start
    classical_accuracy = accuracy_score(y_test, classical_pred)
    
    # Quantum-inspired approach (simulated)
    # In reality, this would use quantum hardware/simulators
    quantum_start = time.time()
    
    # Simulate quantum feature mapping
    def quantum_inspired_features(X):
        """Simulate quantum feature map expansion"""
        # This simulates the exponential feature space of quantum kernels
        n_features = X.shape[1]
        quantum_features = []
        
        # Add original features
        quantum_features.append(X)
        
        # Add interaction terms (simulating entanglement)
        for i in range(n_features):
            for j in range(i+1, n_features):
                interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                quantum_features.append(interaction)
        
        # Add higher-order terms (simulating superposition)
        squared_features = X ** 2
        quantum_features.append(squared_features)
        
        return np.hstack(quantum_features)
    
    # Apply quantum-inspired transformation
    X_train_quantum = quantum_inspired_features(X_train)
    X_test_quantum = quantum_inspired_features(X_test)
    
    # Train on expanded feature space
    quantum_inspired_model = RandomForestClassifier(n_estimators=50, random_state=42)
    quantum_inspired_model.fit(X_train_quantum, y_train)
    quantum_pred = quantum_inspired_model.predict(X_test_quantum)
    quantum_time = time.time() - quantum_start
    quantum_accuracy = accuracy_score(y_test, quantum_pred)
    
    return {
        'classical': {
            'accuracy': classical_accuracy,
            'time': classical_time,
            'features': X_train.shape[1]
        },
        'quantum_inspired': {
            'accuracy': quantum_accuracy,
            'time': quantum_time,
            'features': X_train_quantum.shape[1]
        }
    }

# Run comparison
results = simulate_quantum_inspired_algorithm()

print("Classical vs Quantum-Inspired Comparison:")
print(f"  Classical ML:")
print(f"    Accuracy: {results['classical']['accuracy']:.4f}")
print(f"    Time: {results['classical']['time']:.4f}s")
print(f"    Features: {results['classical']['features']}")
print(f"  Quantum-Inspired ML:")
print(f"    Accuracy: {results['quantum_inspired']['accuracy']:.4f}")
print(f"    Time: {results['quantum_inspired']['time']:.4f}s")
print(f"    Features: {results['quantum_inspired']['features']}")

# 6. FUTURE OUTLOOK AND RECOMMENDATIONS
print(f"\n6. FUTURE OUTLOOK AND RECOMMENDATIONS")
print("-" * 38)

future_outlook = {
    'Near-term (2-5 years)': [
        'NISQ algorithms for specific optimization problems',
        'Quantum-classical hybrid models',
        'Better quantum simulators',
        'Improved quantum software tools'
    ],
    'Medium-term (5-15 years)': [
        'Fault-tolerant quantum computers',
        'Quantum advantage for certain ML tasks',
        'Quantum neural networks',
        'Quantum data analysis platforms'
    ],
    'Long-term (15+ years)': [
        'General-purpose quantum ML',
        'Quantum AI systems',
        'Revolutionary quantum algorithms',
        'Quantum-classical seamless integration'
    ]
}

print("Quantum ML Timeline and Recommendations:")
for timeframe, developments in future_outlook.items():
    print(f"  {timeframe}:")
    for development in developments:
        print(f"    • {development}")

recommendations = [
    "Learn quantum computing fundamentals",
    "Experiment with quantum simulators",
    "Focus on quantum-classical hybrid approaches",
    "Stay updated with quantum hardware progress",
    "Identify ML problems with potential quantum advantage",
    "Collaborate with quantum computing researchers",
    "Prepare for gradual quantum integration"
]

print(f"\nPractical Recommendations for ML Practitioners:")
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

print(f"\n{'='*55}")
print("SUMMARY: Quantum Computing Impact on ML")
print(f"{'='*55}")

summary_points = [
    "Quantum computing offers theoretical speedups for specific ML problems",
    "Current NISQ devices have limitations but show promise",
    "Python frameworks like Qiskit, Cirq, and PennyLane enable experimentation",
    "Hybrid quantum-classical algorithms are most practical currently",
    "Data loading remains a significant challenge for quantum advantage",
    "Focus on optimization and high-dimensional problems for near-term gains",
    "Long-term potential is revolutionary but requires hardware advances"
]

for i, point in enumerate(summary_points, 1):
    print(f"{i}. {point}")

print(f"\n=== Quantum ML: Preparing for the Future ===")
```

### Explanation

Quantum computing represents a paradigm shift with significant implications for machine learning, though practical applications are still emerging.

### Key Quantum Advantages for ML

1. **Superposition**: Process multiple states simultaneously
2. **Entanglement**: Create complex correlations between data points
3. **Interference**: Amplify correct answers and cancel wrong ones
4. **Exponential Scaling**: Access exponentially large feature spaces

### Python Quantum ML Ecosystem

**Major Frameworks:**
- **Qiskit**: IBM's comprehensive quantum computing framework
- **Cirq**: Google's quantum computing platform
- **PennyLane**: Xanadu's quantum ML library
- **Forest/PyQuil**: Rigetti's quantum cloud platform

### Current Applications

1. **Quantum SVM**: Enhanced kernel methods with quantum feature maps
2. **VQE**: Variational quantum eigensolver for optimization
3. **QAOA**: Quantum approximate optimization algorithm
4. **Quantum Neural Networks**: Quantum-enhanced deep learning

### Challenges and Limitations

**Technical Challenges:**
- Limited qubit counts and high error rates
- Short quantum coherence times
- Data loading bottlenecks
- Need for quantum error correction

**Practical Considerations:**
- Most problems still favor classical approaches
- Quantum advantage proven for few ML tasks
- High computational overhead for current hardware
- Need for quantum-classical hybrid algorithms

### Recommendations for ML Practitioners

1. **Education**: Learn quantum computing fundamentals
2. **Experimentation**: Use quantum simulators and cloud platforms
3. **Problem Selection**: Focus on optimization and high-dimensional problems
4. **Hybrid Approaches**: Combine quantum and classical techniques
5. **Community Engagement**: Follow quantum ML research developments

### Future Timeline

**Near-term (2-5 years)**: NISQ algorithms and hybrid models
**Medium-term (5-15 years)**: Fault-tolerant quantum computers
**Long-term (15+ years)**: General-purpose quantum ML systems

While quantum computing holds transformative potential for machine learning, current practical applications are limited. The field is rapidly evolving, and ML practitioners should stay informed about developments while focusing on hybrid approaches that can provide near-term benefits.

## Question 16

**Discuss the integration of big data technologies with Python in machine learning projects.**

### Theory
Big data technologies enable machine learning at scale by providing distributed storage, processing, and analytics capabilities. Python serves as a bridge between traditional ML libraries and big data ecosystems, offering seamless integration through specialized libraries and frameworks.

### Answer

```python
# big_data_ml_python.py - Big Data Technologies Integration with Python ML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=== Big Data Technologies Integration with Python ML ===\n")

# 1. BIG DATA ECOSYSTEM OVERVIEW
print("1. BIG DATA ECOSYSTEM OVERVIEW")
print("-" * 32)

class BigDataEcosystem:
    """Overview of big data technologies for ML"""
    
    def __init__(self):
        self.technologies = {
            'Storage Layer': {
                'Hadoop HDFS': {
                    'purpose': 'Distributed file system',
                    'python_integration': 'hdfs3, snakebite, pydoop',
                    'ml_use_case': 'Store large training datasets',
                    'pros': ['Fault tolerance', 'Scalability', 'Cost effective'],
                    'cons': ['High latency', 'Complex setup']
                },
                'Apache Cassandra': {
                    'purpose': 'NoSQL distributed database',
                    'python_integration': 'cassandra-driver',
                    'ml_use_case': 'Real-time feature storage',
                    'pros': ['Low latency', 'Linear scalability'],
                    'cons': ['Limited query flexibility']
                },
                'MongoDB': {
                    'purpose': 'Document database',
                    'python_integration': 'pymongo',
                    'ml_use_case': 'Unstructured data storage',
                    'pros': ['Flexible schema', 'Rich queries'],
                    'cons': ['Memory intensive']
                }
            },
            'Processing Layer': {
                'Apache Spark': {
                    'purpose': 'Distributed computing engine',
                    'python_integration': 'PySpark',
                    'ml_use_case': 'Large-scale ML training and processing',
                    'pros': ['In-memory computing', 'MLlib integration'],
                    'cons': ['Memory intensive', 'Complex tuning']
                },
                'Apache Flink': {
                    'purpose': 'Stream processing',
                    'python_integration': 'PyFlink',
                    'ml_use_case': 'Real-time ML inference',
                    'pros': ['Low latency', 'Event time processing'],
                    'cons': ['Steeper learning curve']
                },
                'Dask': {
                    'purpose': 'Parallel computing in Python',
                    'python_integration': 'Native Python',
                    'ml_use_case': 'Scale pandas and scikit-learn',
                    'pros': ['Pure Python', 'Easy integration'],
                    'cons': ['Limited compared to Spark']
                }
            },
            'ML Platforms': {
                'Apache Mahout': {
                    'purpose': 'Scalable ML algorithms',
                    'python_integration': 'Mahout bindings',
                    'ml_use_case': 'Distributed ML algorithms',
                    'pros': ['Proven algorithms', 'Hadoop integration'],
                    'cons': ['Limited Python support']
                },
                'MLflow': {
                    'purpose': 'ML lifecycle management',
                    'python_integration': 'Native Python',
                    'ml_use_case': 'Experiment tracking and deployment',
                    'pros': ['Framework agnostic', 'Easy deployment'],
                    'cons': ['Limited big data integration']
                },
                'Kubeflow': {
                    'purpose': 'ML on Kubernetes',
                    'python_integration': 'Kubeflow Pipelines SDK',
                    'ml_use_case': 'Container-based ML workflows',
                    'pros': ['Cloud native', 'Scalable'],
                    'cons': ['Kubernetes complexity']
                }
            }
        }
    
    def get_integration_matrix(self):
        """Create integration complexity matrix"""
        integrations = {}
        for category, technologies in self.technologies.items():
            for tech_name, details in technologies.items():
                # Simplified scoring based on Python integration maturity
                if 'Native Python' in details['python_integration']:
                    complexity = 'Low'
                elif 'Py' in details['python_integration']:
                    complexity = 'Medium'
                else:
                    complexity = 'High'
                
                integrations[tech_name] = {
                    'category': category,
                    'integration_complexity': complexity,
                    'ml_readiness': 'High' if 'ML' in details['ml_use_case'] else 'Medium'
                }
        
        return integrations

ecosystem = BigDataEcosystem()
integration_matrix = ecosystem.get_integration_matrix()

print("Big Data Technology Categories:")
for category, techs in ecosystem.technologies.items():
    print(f"  {category}: {len(techs)} technologies")

print(f"\nPython Integration Complexity:")
for tech, details in integration_matrix.items():
    print(f"  {tech}: {details['integration_complexity']} complexity")

# 2. PYSPARK FOR DISTRIBUTED ML
print(f"\n2. PYSPARK FOR DISTRIBUTED ML")
print("-" * 29)

def demonstrate_pyspark_ml():
    """Demonstrate PySpark ML concepts (simulated)"""
    
    # Note: This simulates PySpark concepts without requiring Spark installation
    print("PySpark ML Pipeline Demonstration (Simulated):")
    
    # Simulate PySpark DataFrame operations
    pyspark_operations = {
        'Data Loading': '''
# Load data from various sources
df = spark.read.parquet("hdfs://large_dataset.parquet")
df = spark.read.jdbc(url=jdbc_url, table="features")
df = spark.read.csv("s3://bucket/data.csv", header=True)
        ''',
        
        'Data Preprocessing': '''
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import StringIndexer, OneHotEncoder

# Feature engineering
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3"],
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features"
)
        ''',
        
        'ML Pipeline': '''
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create ML pipeline
rf = RandomForestClassifier(
    featuresCol="scaled_features",
    labelCol="label"
)

pipeline = Pipeline(stages=[assembler, scaler, rf])

# Train model
model = pipeline.fit(train_df)
predictions = model.transform(test_df)
        ''',
        
        'Distributed Evaluation': '''
# Evaluate on cluster
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Distributed model accuracy: {accuracy}")
        '''
    }
    
    # Simulate performance comparison
    data_sizes = ['1GB', '10GB', '100GB', '1TB']
    processing_times = {
        'Single Machine': [10, 100, 1000, 10000],  # seconds
        'Spark Cluster (4 nodes)': [5, 25, 125, 625],
        'Spark Cluster (16 nodes)': [3, 12, 50, 200]
    }
    
    print("PySpark Operations Overview:")
    for operation, code in pyspark_operations.items():
        print(f"  {operation}:")
        print(f"    Key concepts demonstrated")
    
    print(f"\nScalability Comparison (Processing Time in seconds):")
    print(f"{'Data Size':<15} {'Single':<10} {'4 Nodes':<10} {'16 Nodes':<10}")
    print("-" * 50)
    for i, size in enumerate(data_sizes):
        print(f"{size:<15} {processing_times['Single Machine'][i]:<10} "
              f"{processing_times['Spark Cluster (4 nodes)'][i]:<10} "
              f"{processing_times['Spark Cluster (16 nodes)'][i]:<10}")
    
    return pyspark_operations

pyspark_demo = demonstrate_pyspark_ml()

# 3. DASK FOR PYTHON-NATIVE SCALING
print(f"\n3. DASK FOR PYTHON-NATIVE SCALING")
print("-" * 34)

def demonstrate_dask_ml():
    """Demonstrate Dask for scaling Python ML (simulated)"""
    
    # Simulate Dask operations
    dask_features = {
        'Familiar Interface': {
            'description': 'Dask DataFrame mimics pandas API',
            'code_example': '''
import dask.dataframe as dd
import dask.array as da
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression

# Load large CSV that doesn't fit in memory
df = dd.read_csv('large_dataset_*.csv')

# Standard pandas-like operations
df_processed = df.groupby('category').mean()
            '''
        },
        'Distributed Computing': {
            'description': 'Scale computations across cores/machines',
            'code_example': '''
# Create Dask array for large numpy operations
X = da.random.random((1000000, 100), chunks=(10000, 100))

# Distributed machine learning
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)
            '''
        },
        'Lazy Evaluation': {
            'description': 'Build computation graph, execute when needed',
            'code_example': '''
# Operations are lazy until compute() is called
result = df.groupby('user_id').amount.sum()
print(type(result))  # Dask Series

# Trigger computation
final_result = result.compute()  # Actual pandas Series
            '''
        },
        'Scikit-learn Integration': {
            'description': 'Scale existing scikit-learn workflows',
            'code_example': '''
from dask_ml.wrappers import ParallelPostFit
from sklearn.ensemble import RandomForestClassifier

# Wrap scikit-learn estimator
clf = ParallelPostFit(RandomForestClassifier())
clf.fit(X_train, y_train)

# Distributed prediction
predictions = clf.predict(X_test)
            '''
        }
    }
    
    # Performance simulation
    problem_sizes = {
        'Small (1M samples)': {'pandas_time': 2, 'dask_time': 3, 'dask_advantage': False},
        'Medium (10M samples)': {'pandas_time': 25, 'dask_time': 8, 'dask_advantage': True},
        'Large (100M samples)': {'pandas_time': 'OOM', 'dask_time': 35, 'dask_advantage': True},
        'Very Large (1B samples)': {'pandas_time': 'OOM', 'dask_time': 150, 'dask_advantage': True}
    }
    
    print("Dask ML Features:")
    for feature, details in dask_features.items():
        print(f"  {feature}: {details['description']}")
    
    print(f"\nPandas vs Dask Performance Comparison:")
    print(f"{'Problem Size':<20} {'Pandas':<10} {'Dask':<10} {'Advantage'}")
    print("-" * 55)
    for size, perf in problem_sizes.items():
        pandas_str = str(perf['pandas_time']) if perf['pandas_time'] != 'OOM' else 'OOM'
        advantage = 'Dask' if perf['dask_advantage'] else 'Pandas'
        print(f"{size:<20} {pandas_str:<10} {perf['dask_time']:<10} {advantage}")
    
    return dask_features

dask_demo = demonstrate_dask_ml()

# 4. STREAMING ML WITH KAFKA AND PYTHON
print(f"\n4. STREAMING ML WITH KAFKA AND PYTHON")
print("-" * 37)

class StreamingMLArchitecture:
    """Streaming ML architecture with Python"""
    
    def __init__(self):
        self.components = {
            'Data Ingestion': {
                'Apache Kafka': {
                    'role': 'Message broker for streaming data',
                    'python_libs': ['kafka-python', 'confluent-kafka-python'],
                    'use_case': 'Collect real-time events and features'
                },
                'Apache Pulsar': {
                    'role': 'Alternative message broker',
                    'python_libs': ['pulsar-client'],
                    'use_case': 'Multi-tenant streaming with geo-replication'
                }
            },
            'Stream Processing': {
                'Apache Flink': {
                    'role': 'Low-latency stream processing',
                    'python_libs': ['apache-flink'],
                    'use_case': 'Real-time feature engineering'
                },
                'Kafka Streams': {
                    'role': 'Stream processing library',
                    'python_libs': ['kafka-python + custom processing'],
                    'use_case': 'Simple stream transformations'
                }
            },
            'ML Inference': {
                'Online Models': {
                    'role': 'Real-time prediction serving',
                    'python_libs': ['scikit-multiflow', 'river'],
                    'use_case': 'Adaptive ML models for streaming data'
                },
                'Model Serving': {
                    'role': 'Deploy trained models for inference',
                    'python_libs': ['seldon-core', 'bentoml'],
                    'use_case': 'Scalable model deployment'
                }
            }
        }
    
    def design_streaming_pipeline(self):
        """Design a streaming ML pipeline"""
        pipeline_stages = {
            '1. Data Collection': {
                'description': 'Ingest streaming data from multiple sources',
                'technologies': ['Kafka Producers', 'Schema Registry'],
                'python_code': '''
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Send streaming data
producer.send('user_events', {
    'user_id': 12345,
    'action': 'click',
    'timestamp': time.time(),
    'features': [0.1, 0.5, 0.8]
})
                '''
            },
            '2. Feature Engineering': {
                'description': 'Real-time feature computation and windowing',
                'technologies': ['Kafka Streams', 'Flink'],
                'python_code': '''
from kafka import KafkaConsumer, KafkaProducer
import json
from collections import defaultdict, deque

class StreamingFeatureEngine:
    def __init__(self):
        self.user_windows = defaultdict(lambda: deque(maxlen=100))
    
    def process_event(self, event):
        user_id = event['user_id']
        features = event['features']
        
        # Update sliding window
        self.user_windows[user_id].append(features)
        
        # Compute streaming features
        if len(self.user_windows[user_id]) >= 10:
            recent_features = list(self.user_windows[user_id])[-10:]
            avg_features = np.mean(recent_features, axis=0)
            return avg_features
        return None
                '''
            },
            '3. Online Learning': {
                'description': 'Continuously update models with new data',
                'technologies': ['River', 'Scikit-multiflow'],
                'python_code': '''
from river import linear_model, metrics, preprocessing

# Create online learning model
model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
metric = metrics.Accuracy()

# Process streaming data
for event in data_stream:
    features = event['features']
    label = event.get('label')
    
    # Make prediction
    prediction = model.predict_one(features)
    
    # Update model if label available
    if label is not None:
        model.learn_one(features, label)
        metric.update(label, prediction)
                '''
            },
            '4. Real-time Inference': {
                'description': 'Serve predictions with low latency',
                'technologies': ['FastAPI', 'Redis', 'Model serving'],
                'python_code': '''
from fastapi import FastAPI
import redis
import joblib

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379)
model = joblib.load('streaming_model.pkl')

@app.post("/predict")
async def predict(features: dict):
    # Get user context from Redis
    user_context = redis_client.get(f"user:{features['user_id']}")
    
    # Make prediction
    prediction = model.predict([features['values']])[0]
    
    # Cache result
    redis_client.setex(
        f"prediction:{features['user_id']}", 
        300,  # 5 min TTL
        prediction
    )
    
    return {"prediction": prediction}
                '''
            }
        }
        
        return pipeline_stages

streaming_arch = StreamingMLArchitecture()
pipeline = streaming_arch.design_streaming_pipeline()

print("Streaming ML Architecture Components:")
for category, components in streaming_arch.components.items():
    print(f"  {category}:")
    for comp_name, details in components.items():
        print(f"    {comp_name}: {details['role']}")

print(f"\nStreaming ML Pipeline Stages:")
for stage, details in pipeline.items():
    print(f"  {stage}: {details['description']}")

# 5. CLOUD BIG DATA INTEGRATION
print(f"\n5. CLOUD BIG DATA INTEGRATION")
print("-" * 31)

class CloudBigDataIntegration:
    """Cloud platforms for big data ML"""
    
    def __init__(self):
        self.cloud_platforms = {
            'AWS': {
                'services': {
                    'EMR': 'Managed Hadoop/Spark clusters',
                    'Kinesis': 'Real-time data streaming',
                    'Redshift': 'Data warehouse',
                    'S3': 'Object storage',
                    'SageMaker': 'Managed ML platform'
                },
                'python_integration': ['boto3', 'awswrangler', 'sagemaker-python-sdk'],
                'ml_workflow': 'S3 → EMR/SageMaker → Model deployment'
            },
            'Google Cloud': {
                'services': {
                    'Dataproc': 'Managed Spark/Hadoop',
                    'Dataflow': 'Stream/batch processing',
                    'BigQuery': 'Serverless data warehouse',
                    'Cloud Storage': 'Object storage',
                    'AI Platform': 'ML platform'
                },
                'python_integration': ['google-cloud-*', 'tensorflow', 'apache-beam'],
                'ml_workflow': 'GCS → Dataproc/AI Platform → Deployment'
            },
            'Azure': {
                'services': {
                    'HDInsight': 'Managed big data analytics',
                    'Stream Analytics': 'Real-time analytics',
                    'Synapse Analytics': 'Data warehouse',
                    'Blob Storage': 'Object storage',
                    'Machine Learning': 'ML platform'
                },
                'python_integration': ['azure-*', 'azureml-sdk'],
                'ml_workflow': 'Blob → HDInsight/AML → Model serving'
            }
        }
    
    def compare_cloud_offerings(self):
        """Compare cloud big data offerings"""
        comparison = {}
        for platform, details in self.cloud_platforms.items():
            comparison[platform] = {
                'service_count': len(details['services']),
                'python_libs': len(details['python_integration']),
                'key_strength': list(details['services'].values())[0]
            }
        return comparison

cloud_integration = CloudBigDataIntegration()
cloud_comparison = cloud_integration.compare_cloud_offerings()

print("Cloud Big Data Platforms:")
for platform, details in cloud_comparison.items():
    print(f"  {platform}:")
    print(f"    Services: {details['service_count']} main services")
    print(f"    Python Libraries: {details['python_libs']} integration packages")

# 6. BEST PRACTICES AND RECOMMENDATIONS
print(f"\n6. BEST PRACTICES AND RECOMMENDATIONS")
print("-" * 38)

best_practices = {
    'Data Management': [
        'Use appropriate data formats (Parquet, ORC) for big data',
        'Implement data partitioning for efficient querying',
        'Set up data lifecycle management and archiving',
        'Ensure data quality and validation pipelines'
    ],
    'Performance Optimization': [
        'Choose right cluster size based on workload',
        'Optimize Spark configurations for your use case',
        'Use caching strategically for iterative algorithms',
        'Monitor resource utilization and costs'
    ],
    'ML Pipeline Design': [
        'Design for both batch and streaming processing',
        'Implement feature stores for consistency',
        'Use containerization for reproducible environments',
        'Plan for model versioning and rollback'
    ],
    'Security and Governance': [
        'Implement proper access controls and encryption',
        'Audit data access and model predictions',
        'Ensure compliance with data regulations',
        'Monitor for data drift and model degradation'
    ]
}

technology_selection_guide = {
    'Data Size < 100GB': 'Use pandas + scikit-learn on single machine',
    'Data Size 100GB-1TB': 'Use Dask for Python-native scaling',
    'Data Size > 1TB': 'Use PySpark on Hadoop/cloud clusters',
    'Real-time Requirements': 'Use Kafka + streaming frameworks',
    'Complex Analytics': 'Use Spark with MLlib or custom algorithms',
    'Budget Constrained': 'Use open-source stack (Hadoop/Spark)',
    'Quick Prototyping': 'Use cloud managed services',
    'High Availability': 'Use cloud platforms with SLA guarantees'
}

print("Big Data ML Best Practices:")
for category, practices in best_practices.items():
    print(f"  {category}:")
    for practice in practices:
        print(f"    • {practice}")

print(f"\nTechnology Selection Guide:")
for scenario, recommendation in technology_selection_guide.items():
    print(f"  {scenario}: {recommendation}")

print(f"\n{'='*60}")
print("SUMMARY: Big Data Technologies with Python ML")
print(f"{'='*60}")

summary_points = [
    "Python bridges traditional ML and big data ecosystems effectively",
    "PySpark enables distributed ML training on large datasets", 
    "Dask provides Python-native scaling with familiar APIs",
    "Streaming ML requires specialized frameworks and architectures",
    "Cloud platforms offer managed big data services with Python SDKs",
    "Choose technology based on data size, latency, and budget constraints",
    "Implement proper data management and security practices",
    "Monitor performance and costs in distributed environments"
]

for i, point in enumerate(summary_points, 1):
    print(f"{i}. {point}")

print(f"\n=== Big Data ML: Scaling Python for Enterprise ===")
```

### Explanation

Big data technologies expand Python's machine learning capabilities to handle massive datasets and real-time processing requirements through distributed computing frameworks.

### Key Integration Areas

**1. Distributed Computing**
- **PySpark**: Python API for Apache Spark, enabling distributed ML
- **Dask**: Python-native parallel computing with familiar pandas/scikit-learn APIs
- **Ray**: Distributed computing framework for ML workloads

**2. Streaming Data Processing**
- **Kafka**: Message broker for real-time data ingestion
- **Apache Flink**: Stream processing for real-time ML inference
- **Online Learning**: Adaptive models that learn from streaming data

**3. Cloud Integration**
- **AWS**: EMR, Kinesis, SageMaker with boto3 integration
- **Google Cloud**: Dataproc, Dataflow, AI Platform with Python SDKs
- **Azure**: HDInsight, Stream Analytics, Azure ML with Python support

### Technology Selection Guidelines

**Data Size Considerations:**
- **< 100GB**: Traditional pandas + scikit-learn
- **100GB - 1TB**: Dask for memory-efficient processing
- **> 1TB**: PySpark on distributed clusters

**Use Case Patterns:**
- **Batch Processing**: Hadoop + Spark for large-scale training
- **Real-time Inference**: Kafka + streaming frameworks
- **Interactive Analysis**: Jupyter + Dask/Spark clusters
- **Production ML**: Kubernetes + containerized services

### Architecture Patterns

**Lambda Architecture:**
- Batch layer for comprehensive processing
- Speed layer for real-time processing
- Serving layer for query responses

**Kappa Architecture:**
- Stream-only processing
- Simplified architecture
- Real-time ML inference

### Best Practices

1. **Data Management**: Use efficient formats (Parquet), implement partitioning
2. **Performance**: Optimize cluster configurations, use caching strategically
3. **Scalability**: Design for horizontal scaling, plan capacity
4. **Security**: Implement encryption, access controls, audit trails
5. **Monitoring**: Track resource usage, costs, and model performance
6. **Testing**: Validate at scale, test failure scenarios

### Python Ecosystem Integration

The strength of Python in big data ML lies in its ability to integrate traditional ML libraries with distributed computing frameworks, providing a unified development experience from prototyping to production scale deployment.

