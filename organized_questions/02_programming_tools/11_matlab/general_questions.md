# MATLAB Interview Questions - General Questions

## Question 1

**How do you read and write data in MATLAB?**

### Answer

### File I/O Functions

| Function | File Type |
|----------|-----------|
| `readtable` | CSV, Excel, text |
| `load` | MAT files, text |
| `imread` | Images |
| `audioread` | Audio |
| `readmatrix` | Numeric data |

### MATLAB Code Example
```matlab
% CSV file
data = readtable('data.csv');
writetable(data, 'output.csv');

% Excel file
data = readtable('data.xlsx', 'Sheet', 'Sheet1');
writetable(data, 'output.xlsx');

% MAT file (MATLAB native)
save('data.mat', 'data');
loaded = load('data.mat');

% Text file
matrix = readmatrix('data.txt');
writematrix(matrix, 'output.txt');

% Image
img = imread('image.png');
imwrite(img, 'output.png');

% Custom text file
fid = fopen('custom.txt', 'w');
fprintf(fid, 'Value: %d\n', 42);
fclose(fid);

% Read line by line
fid = fopen('data.txt', 'r');
while ~feof(fid)
    line = fgetl(fid);
    disp(line);
end
fclose(fid);
```

---

## Question 2

**How do you handle missing data in MATLAB?**

### Answer

### Missing Data Functions

| Function | Purpose |
|----------|---------|
| `ismissing` | Detect missing values |
| `rmmissing` | Remove missing values |
| `fillmissing` | Fill missing values |
| `standardizeMissing` | Define missing indicators |

### MATLAB Code Example
```matlab
% Create data with missing values
data = [1 2 NaN 4 5 NaN 7];

% Detect missing
missing_idx = isnan(data);
fprintf('Missing count: %d\n', sum(missing_idx));

% Remove missing
clean_data = rmmissing(data);

% Fill missing values
filled_mean = fillmissing(data, 'constant', mean(data, 'omitnan'));
filled_interp = fillmissing(data, 'linear');
filled_prev = fillmissing(data, 'previous');

disp('Original:'); disp(data);
disp('Fill with mean:'); disp(filled_mean);
disp('Linear interpolation:'); disp(filled_interp);

% Table with missing values
T = table([1;2;NaN;4], {'A';'B';'';'D'}, ...
    'VariableNames', {'Num', 'Cat'});

% Define empty string as missing
T = standardizeMissing(T, '');

% Fill different columns differently
T.Num = fillmissing(T.Num, 'constant', 0);
T.Cat = fillmissing(T.Cat, 'constant', 'Unknown');
```

---

## Question 3

**How do you visualize data in MATLAB?**

### Answer

### Common Plot Functions

| Function | Use Case |
|----------|----------|
| `plot` | Line plots |
| `scatter` | Scatter plots |
| `histogram` | Distributions |
| `bar` | Bar charts |
| `heatmap` | Correlation matrices |
| `subplot` | Multiple plots |

### MATLAB Code Example
```matlab
% Generate data
x = linspace(0, 2*pi, 100);
y = sin(x);

% Line plot
figure;
plot(x, y, 'b-', 'LineWidth', 2);
xlabel('x');
ylabel('sin(x)');
title('Sine Wave');
grid on;

% Multiple plots
figure;
subplot(2, 2, 1);
plot(x, sin(x));
title('Sine');

subplot(2, 2, 2);
plot(x, cos(x));
title('Cosine');

subplot(2, 2, 3);
histogram(randn(1000, 1), 30);
title('Normal Distribution');

subplot(2, 2, 4);
scatter(randn(100, 1), randn(100, 1));
title('Scatter Plot');

% Heatmap for correlation
data = randn(100, 5);
corr_matrix = corrcoef(data);
figure;
heatmap(corr_matrix);
title('Correlation Heatmap');

% Save figure
saveas(gcf, 'figure.png');
```

---

## Question 4

**How do you debug MATLAB code?**

### Answer

### Debugging Tools

| Tool | Purpose |
|------|---------|
| `dbstop` | Set breakpoint |
| `dbcont` | Continue execution |
| `dbstep` | Step through code |
| `dbstack` | View call stack |
| `disp/fprintf` | Print values |

### MATLAB Code Example
```matlab
% Set breakpoint on error
dbstop if error

% Set breakpoint at line
dbstop in my_function at 10

% Conditional breakpoint
dbstop in my_function at 15 if x > 10

% Debug function
function result = buggy_function(data)
    % Add assertions for validation
    assert(~isempty(data), 'Data cannot be empty');
    assert(isnumeric(data), 'Data must be numeric');
    
    % Debug output
    fprintf('Input size: %s\n', mat2str(size(data)));
    
    % Try-catch for error handling
    try
        result = process_data(data);
    catch ME
        fprintf('Error: %s\n', ME.message);
        fprintf('Stack trace:\n');
        disp(ME.stack);
        result = [];
    end
end

% Use keyboard to pause execution
function result = interactive_debug(x)
    intermediate = x * 2;
    keyboard;  % Pauses here for inspection
    result = intermediate + 1;
end

% Clear breakpoints
dbclear all
```

---

## Question 5

**How do you optimize MATLAB code performance?**

### Answer

### Optimization Strategies

| Strategy | Impact |
|----------|--------|
| Vectorization | High |
| Preallocation | High |
| Built-in functions | Medium |
| Profiling | Diagnostic |
| Data types | Medium |

### MATLAB Code Example
```matlab
% Profile code
profile on
slow_function();
profile viewer

% Timing
tic
result = my_function(data);
elapsed = toc;
fprintf('Time: %.4f s\n', elapsed);

% Compare implementations
function benchmark()
    n = 1000000;
    x = randn(n, 1);
    
    % Method 1: Loop
    tic
    y1 = zeros(n, 1);
    for i = 1:n
        y1(i) = sqrt(abs(x(i)));
    end
    t1 = toc;
    
    % Method 2: Vectorized
    tic
    y2 = sqrt(abs(x));
    t2 = toc;
    
    % Method 3: arrayfun
    tic
    y3 = arrayfun(@(v) sqrt(abs(v)), x);
    t3 = toc;
    
    fprintf('Loop: %.4f s\n', t1);
    fprintf('Vectorized: %.4f s\n', t2);
    fprintf('arrayfun: %.4f s\n', t3);
end

% Use appropriate data types
x_single = single(randn(1000));  % Less memory than double
x_int = int32(1:1000);  % Integer type

% Sparse matrices for sparse data
A_sparse = sparse(eye(10000));  % Much less memory
```

