# MATLAB Interview Questions - Scenario Based Questions

## Question 1

**Discuss MATLAB's support for different data types.**

### Answer

### Data Types Overview

| Type | Description | Example |
|------|-------------|---------|
| `double` | Default numeric (64-bit) | `x = 3.14` |
| `single` | 32-bit float | `x = single(3.14)` |
| `int8/16/32/64` | Signed integers | `x = int32(5)` |
| `uint8/16/32/64` | Unsigned integers | `x = uint8(255)` |
| `logical` | Boolean | `x = true` |
| `char` | Character array | `x = 'hello'` |
| `string` | String array | `x = "hello"` |
| `cell` | Cell array | `x = {1, 'a'}` |
| `struct` | Structure | `x.field = 1` |
| `table` | Tabular data | `x = table(...)` |

### MATLAB Code Example
```matlab
% Numeric types and memory
x_double = randn(1000);
x_single = single(randn(1000));
x_int32 = int32(randi(100, 1000));

whos x_double x_single x_int32

% Type conversion
a = 3.7;
b = int32(a);  % Truncates to 3
c = round(a);  % Rounds to 4

% String vs char
str = "Hello World";    % String
chr = 'Hello World';    % Char array

% String operations
words = split(str);
upper_str = upper(str);
contains_hello = contains(str, "Hello");

% Categorical for ML
categories = categorical({'low', 'medium', 'high', 'medium', 'low'});
summary(categories)

% Datetime
dt = datetime('now');
dates = datetime(2024, 1, 1:10);
duration = days(5) + hours(3);

% Choose appropriate type for memory efficiency
% Images: uint8 (0-255)
% Large matrices: single instead of double
% Categorical: categorical instead of strings
```

---

## Question 2

**Your MATLAB code is running slowly. How do you optimize it?**

### Answer

### Optimization Strategy

| Step | Action |
|------|--------|
| 1 | Profile to find bottleneck |
| 2 | Vectorize loops |
| 3 | Preallocate arrays |
| 4 | Use appropriate data types |
| 5 | Consider parallel computing |

### MATLAB Code Example
```matlab
%% Performance Optimization Example

n = 100000;

% SLOW: Growing array
tic
slow_result = [];
for i = 1:n
    slow_result = [slow_result; i^2];
end
slow_time = toc;

% BETTER: Preallocated loop
tic
better_result = zeros(n, 1);
for i = 1:n
    better_result(i) = i^2;
end
better_time = toc;

% BEST: Vectorized
tic
best_result = (1:n)'.^2;
best_time = toc;

fprintf('Growing array: %.4f s\n', slow_time);
fprintf('Preallocated: %.4f s\n', better_time);
fprintf('Vectorized: %.4f s\n', best_time);

% Profile code
profile on
my_slow_function();
profile viewer
profile off

% Use built-in functions (optimized C code)
A = randn(1000);

% Slow: manual sum
tic
s = 0;
for i = 1:numel(A)
    s = s + A(i);
end
manual_time = toc;

% Fast: built-in sum
tic
s = sum(A, 'all');
builtin_time = toc;

fprintf('Manual sum: %.6f s\n', manual_time);
fprintf('Built-in sum: %.6f s\n', builtin_time);

function my_slow_function()
    % Intentionally slow for profiling demo
    x = randn(1000);
    for i = 1:100
        y = x * x';
    end
end
```

---

## Question 3

**You need to process large datasets that don't fit in memory. What approaches can you use?**

### Answer

### Large Data Strategies

| Strategy | Use Case |
|----------|----------|
| `tall` arrays | Out-of-memory data |
| `datastore` | Iterate over files |
| Memory mapping | Direct file access |
| Chunked processing | Process in batches |

### MATLAB Code Example
```matlab
%% Large Data Processing

% Method 1: Datastore for large CSV files
ds = tabularTextDatastore('large_data/*.csv');
ds.SelectedVariableNames = {'Column1', 'Column2'};

% Process in chunks
while hasdata(ds)
    chunk = read(ds);
    % Process chunk
    chunk_mean = mean(chunk.Column1);
end

% Method 2: Tall arrays (lazy evaluation)
ds = tabularTextDatastore('large_data.csv');
tt = tall(ds);

% Operations are lazy - not executed yet
mean_val = mean(tt.Column1);
filtered = tt(tt.Column1 > 0, :);

% Execute all operations
results = gather(mean_val);

% Method 3: Memory-mapped files
% Create large binary file
data = randn(10000, 10000);
fid = fopen('large_data.bin', 'w');
fwrite(fid, data, 'double');
fclose(fid);

% Memory map for random access
m = memmapfile('large_data.bin', ...
    'Format', {'double', [10000 10000], 'data'});

% Access without loading entire file
partial_data = m.Data.data(1:100, 1:100);

% Method 4: Chunked processing
function process_large_file(filename, chunk_size)
    info = dir(filename);
    total_rows = info.bytes / 8;  % Assuming doubles
    
    fid = fopen(filename, 'r');
    result = 0;
    rows_processed = 0;
    
    while rows_processed < total_rows
        chunk = fread(fid, chunk_size, 'double');
        result = result + sum(chunk);
        rows_processed = rows_processed + length(chunk);
    end
    
    fclose(fid);
end
```

---

## Question 4

**How do you deploy a MATLAB machine learning model to production?**

### Answer

### Deployment Options

| Option | Use Case |
|--------|----------|
| MATLAB Compiler | Standalone application |
| MATLAB Production Server | Enterprise deployment |
| MATLAB Coder | C/C++ code generation |
| ONNX export | Cross-platform ML models |

### MATLAB Code Example
```matlab
%% Model Deployment

% Train model
load fisheriris
X = meas;
y = species;
mdl = fitcsvm(X, y);

% Method 1: Save and load model
save('svm_model.mat', 'mdl');

% Load in another script
loaded = load('svm_model.mat');
model = loaded.mdl;
y_pred = predict(model, X_new);

% Method 2: Export to PMML
% saveLearnerForCoder(mdl, 'svm_model');

% Method 3: Generate C code (MATLAB Coder)
% Create prediction function
function y = predict_wrapper(X, mdl)
    y = predict(mdl, X);
end

% Configure for code generation
% cfg = coder.config('lib');
% codegen predict_wrapper -args {X, mdl} -config cfg

% Method 4: Export deep learning model to ONNX
% For deep learning models
layers = [
    featureInputLayer(4)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];
options = trainingOptions('adam', 'MaxEpochs', 10);
net = trainNetwork(X, categorical(y), layers, options);

% Export to ONNX
exportONNXNetwork(net, 'model.onnx');

% Method 5: Create REST API (MATLAB Production Server)
% Package function
% Create deployable archive
% Deploy to MATLAB Production Server

% Method 6: Python integration
% Export model and call from Python
% Or use MATLAB Engine for Python
```

---

## Question 5

**How do you handle imbalanced datasets in MATLAB for classification?**

### Answer

### Handling Imbalanced Data

| Technique | Description |
|-----------|-------------|
| Oversampling | Duplicate minority class |
| Undersampling | Remove majority class samples |
| SMOTE | Synthetic samples |
| Class weights | Adjust loss function |

### MATLAB Code Example
```matlab
%% Handling Imbalanced Data

% Create imbalanced dataset
rng(42);
n_majority = 900;
n_minority = 100;

X_majority = randn(n_majority, 2);
X_minority = randn(n_minority, 2) + 3;
X = [X_majority; X_minority];
y = [zeros(n_majority, 1); ones(n_minority, 1)];

% Check class distribution
tabulate(y)

% Method 1: Class weights (Prior)
prior = [n_minority, n_majority] / (n_majority + n_minority);
mdl_weighted = fitcsvm(X, y, 'Prior', 'uniform');  % Equal weights

% Method 2: Cost-sensitive learning
cost = [0, 1; 9, 0];  % Higher cost for misclassifying minority
mdl_cost = fitcsvm(X, y, 'Cost', cost);

% Method 3: Oversampling (simple duplication)
oversample_ratio = round(n_majority / n_minority);
X_minority_oversampled = repmat(X_minority, oversample_ratio, 1);
y_minority_oversampled = ones(size(X_minority_oversampled, 1), 1);
X_balanced = [X_majority; X_minority_oversampled];
y_balanced = [zeros(n_majority, 1); y_minority_oversampled];

% Method 4: Undersampling
undersample_idx = randperm(n_majority, n_minority);
X_undersampled = [X_majority(undersample_idx, :); X_minority];
y_undersampled = [zeros(n_minority, 1); ones(n_minority, 1)];

% Method 5: SMOTE-like synthetic oversampling
function X_synthetic = simple_smote(X_minority, n_synthetic)
    n = size(X_minority, 1);
    X_synthetic = zeros(n_synthetic, size(X_minority, 2));
    
    for i = 1:n_synthetic
        % Pick random sample
        idx = randi(n);
        sample = X_minority(idx, :);
        
        % Find nearest neighbor
        dists = pdist2(sample, X_minority);
        [~, sorted_idx] = sort(dists);
        neighbor = X_minority(sorted_idx(2), :);  % Nearest (exclude self)
        
        % Interpolate
        alpha = rand();
        X_synthetic(i, :) = sample + alpha * (neighbor - sample);
    end
end

X_smote = simple_smote(X_minority, n_majority - n_minority);
X_balanced_smote = [X; X_smote];
y_balanced_smote = [y; ones(size(X_smote, 1), 1)];

% Evaluate with appropriate metrics
mdl = fitcsvm(X_balanced_smote, y_balanced_smote);
[y_pred, scores] = predict(mdl, X);

% Use AUC instead of accuracy
[~, ~, ~, auc] = perfcurve(y, scores(:,2), 1);
fprintf('AUC: %.4f\n', auc);

% Confusion matrix
figure;
confusionchart(y, y_pred);
title('Confusion Matrix');

% Precision, Recall, F1
tp = sum(y_pred == 1 & y == 1);
fp = sum(y_pred == 1 & y == 0);
fn = sum(y_pred == 0 & y == 1);

precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1 = 2 * precision * recall / (precision + recall);

fprintf('Precision: %.4f\n', precision);
fprintf('Recall: %.4f\n', recall);
fprintf('F1 Score: %.4f\n', f1);
```

