# MATLAB Interview Questions - Theory Questions

## Question 1

**What are the main features of MATLAB that make it suitable for machine learning?**

### Answer

**Definition**: MATLAB (Matrix Laboratory) is a high-level programming environment designed for numerical computing, visualization, and algorithm development.

### Key Features for ML

| Feature | Description |
|---------|-------------|
| **Matrix operations** | Native matrix/vector support |
| **Toolboxes** | Statistics, ML, Deep Learning toolboxes |
| **Visualization** | Built-in plotting functions |
| **Prototyping** | Rapid algorithm development |
| **Integration** | C/C++, Python, hardware integration |

### MATLAB Code Example
```matlab
% Load sample data
load fisheriris
X = meas;
y = species;

% Train classifier
mdl = fitcsvm(X, y);

% Cross-validation
cvmdl = crossval(mdl);
loss = kfoldLoss(cvmdl);
fprintf('CV Error: %.2f%%\n', loss*100);

% Visualize
gscatter(X(:,1), X(:,2), y);
xlabel('Sepal Length');
ylabel('Sepal Width');
title('Iris Dataset');
```

---

## Question 2

**Explain MATLAB's matrix operations and their importance.**

### Answer

### Core Matrix Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Addition | `A + B` | Element-wise addition |
| Multiplication | `A * B` | Matrix multiplication |
| Element-wise mult | `A .* B` | Element-wise multiplication |
| Transpose | `A'` | Matrix transpose |
| Inverse | `inv(A)` | Matrix inverse |
| Determinant | `det(A)` | Matrix determinant |

### MATLAB Code Example
```matlab
% Create matrices
A = [1 2; 3 4];
B = [5 6; 7 8];

% Matrix operations
C = A * B;           % Matrix multiplication
D = A .* B;          % Element-wise multiplication
E = A';              % Transpose
F = inv(A);          % Inverse

% Solve linear system Ax = b
b = [1; 2];
x = A \ b;           % More efficient than inv(A)*b

% Eigenvalues and eigenvectors
[V, D] = eig(A);

% SVD decomposition
[U, S, V] = svd(A);

fprintf('Matrix product:\n');
disp(C);
fprintf('Solution to Ax=b:\n');
disp(x);
```

### Interview Tip
Always use `\` (backslash) for solving linear systems instead of `inv()` - it's more numerically stable and efficient.

---

## Question 3

**What are MATLAB toolboxes and which ones are relevant for ML?**

### Answer

### ML-Related Toolboxes

| Toolbox | Purpose |
|---------|---------|
| **Statistics & ML** | Classical ML algorithms |
| **Deep Learning** | Neural networks, CNNs, RNNs |
| **Computer Vision** | Image processing, object detection |
| **Signal Processing** | Time series, filtering |
| **Optimization** | Convex optimization |
| **Parallel Computing** | GPU, distributed computing |

### MATLAB Code Example
```matlab
% Statistics and Machine Learning Toolbox
load carsmall
X = [Weight, Horsepower];
y = MPG;

% Remove NaN
idx = ~any(isnan([X y]), 2);
X = X(idx,:);
y = y(idx);

% Fit linear regression
mdl = fitlm(X, y);
disp(mdl);

% Random Forest
rf_mdl = TreeBagger(100, X, y, 'Method', 'regression');
y_pred = predict(rf_mdl, X);

% Deep Learning Toolbox
layers = [
    featureInputLayer(2)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

options = trainingOptions('adam', 'MaxEpochs', 100);
% net = trainNetwork(X, y, layers, options);
```

---

## Question 4

**Explain the difference between scripts and functions in MATLAB.**

### Answer

### Comparison

| Feature | Script | Function |
|---------|--------|----------|
| **Scope** | Workspace variables | Local variables |
| **Input/Output** | None | Arguments/returns |
| **Reusability** | Limited | High |
| **File** | `.m` file | `.m` file with function keyword |

### MATLAB Code Example
```matlab
% SCRIPT: my_script.m
% Operates on workspace variables
data = randn(100, 1);
mean_val = mean(data);
fprintf('Mean: %.2f\n', mean_val);

% FUNCTION: my_function.m
function [mean_val, std_val] = compute_stats(data)
    % Computes statistics for input data
    % Input: data - numeric array
    % Output: mean_val, std_val - statistics
    
    mean_val = mean(data);
    std_val = std(data);
end

% Anonymous function
square = @(x) x.^2;
result = square(5);  % Returns 25

% Function with validation
function y = safe_divide(a, b)
    arguments
        a double
        b double {mustBeNonzero}
    end
    y = a / b;
end
```

---

## Question 5

**What is vectorization in MATLAB and why is it important?**

### Answer

**Definition**: Vectorization is writing code that operates on entire arrays at once, avoiding explicit loops.

### Benefits

| Benefit | Description |
|---------|-------------|
| **Speed** | 10-100x faster than loops |
| **Readability** | Cleaner code |
| **Memory** | Better memory management |

### MATLAB Code Example
```matlab
n = 1000000;
x = randn(n, 1);

% SLOW: Loop-based
tic
y_loop = zeros(n, 1);
for i = 1:n
    y_loop(i) = x(i)^2 + 2*x(i) + 1;
end
loop_time = toc;

% FAST: Vectorized
tic
y_vec = x.^2 + 2*x + 1;
vec_time = toc;

fprintf('Loop time: %.4f s\n', loop_time);
fprintf('Vectorized time: %.4f s\n', vec_time);
fprintf('Speedup: %.1fx\n', loop_time/vec_time);

% More vectorization examples
A = randn(1000, 1000);
B = randn(1000, 1000);

% Matrix operations are inherently vectorized
C = A * B;

% Logical indexing (vectorized)
positive_vals = A(A > 0);

% Apply function to each element
result = arrayfun(@(x) x^2, A);  % Slower
result = A.^2;  % Faster, vectorized
```

---

## Question 6

**Explain memory management in MATLAB.**

### Answer

### Memory Concepts

| Concept | Description |
|---------|-------------|
| **Copy-on-write** | Data copied only when modified |
| **In-place operations** | Modify without copying |
| **Preallocation** | Allocate before loops |
| **Clear** | Release memory |

### MATLAB Code Example
```matlab
% BAD: Growing array in loop
tic
data_bad = [];
for i = 1:10000
    data_bad = [data_bad; i];  % Reallocates each iteration
end
bad_time = toc;

% GOOD: Preallocate
tic
data_good = zeros(10000, 1);
for i = 1:10000
    data_good(i) = i;
end
good_time = toc;

fprintf('Without preallocation: %.4f s\n', bad_time);
fprintf('With preallocation: %.4f s\n', good_time);

% Check memory usage
whos

% Clear variables
clear data_bad data_good

% Monitor memory
memory  % Windows only

% Use sparse matrices for large sparse data
I = speye(10000);  % Sparse identity matrix
whos I
```

---

## Question 7

**What are cell arrays and structures in MATLAB?**

### Answer

### Comparison

| Type | Use Case | Access |
|------|----------|--------|
| **Cell array** | Mixed data types | `{}` |
| **Structure** | Named fields | `.fieldname` |
| **Table** | Tabular data | `.VariableName` or `()` |

### MATLAB Code Example
```matlab
% Cell array - mixed types
cell_data = {1, 'text', [1 2 3], @sin};
disp(cell_data{1});      % Access element
disp(cell_data{2});      % 'text'

% Structure - named fields
person.name = 'Alice';
person.age = 30;
person.scores = [85, 90, 95];

fprintf('%s is %d years old\n', person.name, person.age);

% Structure array
people(1).name = 'Alice';
people(1).age = 30;
people(2).name = 'Bob';
people(2).age = 25;

% Access all names
names = {people.name};

% Table - best for ML data
T = table({'Alice'; 'Bob'}, [30; 25], [85; 90], ...
    'VariableNames', {'Name', 'Age', 'Score'});

disp(T);
disp(T.Age);  % Access column
disp(T(1,:)); % Access row
```

---

## Question 8

**Explain parallel computing in MATLAB.**

### Answer

### Parallel Options

| Feature | Use Case |
|---------|----------|
| **parfor** | Parallel for loops |
| **parfeval** | Async function execution |
| **GPU arrays** | GPU computing |
| **Distributed** | Cluster computing |

### MATLAB Code Example
```matlab
% Start parallel pool
pool = parpool('local', 4);

% Parallel for loop
n = 100;
results = zeros(n, 1);

parfor i = 1:n
    results(i) = expensive_computation(i);
end

% GPU computing
if gpuDeviceCount > 0
    A_gpu = gpuArray(randn(1000));
    B_gpu = gpuArray(randn(1000));
    C_gpu = A_gpu * B_gpu;  % Runs on GPU
    C = gather(C_gpu);  % Transfer back to CPU
end

% Async execution
f = parfeval(@expensive_computation, 1, 100);
% Do other work...
result = fetchOutputs(f);

% Delete pool when done
delete(pool);

function y = expensive_computation(x)
    pause(0.1);  % Simulate work
    y = x^2;
end
```

### Interview Tip
Not all loops can be parallelized with `parfor` - iterations must be independent.

