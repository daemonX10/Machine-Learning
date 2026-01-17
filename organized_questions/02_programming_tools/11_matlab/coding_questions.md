# MATLAB Interview Questions - Coding Questions

## Question 1

**Explain how to implement linear regression in MATLAB.**

### Answer

### MATLAB Code Example
```matlab
%% Linear Regression Implementation

% Generate sample data
n = 100;
X = randn(n, 2);  % 2 features
true_weights = [3; -2];
y = X * true_weights + 0.5 * randn(n, 1);  % Add noise

% Method 1: Using fitlm (Statistics Toolbox)
mdl = fitlm(X, y);
disp(mdl);
fprintf('R-squared: %.4f\n', mdl.Rsquared.Ordinary);

% Method 2: Normal equation (manual)
X_augmented = [ones(n, 1), X];  % Add bias term
weights_manual = (X_augmented' * X_augmented) \ (X_augmented' * y);
fprintf('Manual weights: %.4f, %.4f, %.4f\n', weights_manual);

% Method 3: Using regress
[b, bint, r, rint, stats] = regress(y, X_augmented);
fprintf('R-squared: %.4f\n', stats(1));

% Predictions
y_pred = predict(mdl, X);

% Visualization
figure;
subplot(1, 2, 1);
scatter(y, y_pred);
hold on;
plot([min(y), max(y)], [min(y), max(y)], 'r--');
xlabel('Actual');
ylabel('Predicted');
title('Actual vs Predicted');

subplot(1, 2, 2);
residuals = y - y_pred;
histogram(residuals, 20);
xlabel('Residual');
ylabel('Frequency');
title('Residual Distribution');
```

---

## Question 2

**Implement k-means clustering in MATLAB.**

### Answer

### MATLAB Code Example
```matlab
%% K-Means Clustering

% Generate sample data (3 clusters)
rng(42);
n_per_cluster = 100;
X1 = randn(n_per_cluster, 2) + [0, 0];
X2 = randn(n_per_cluster, 2) + [5, 5];
X3 = randn(n_per_cluster, 2) + [5, 0];
X = [X1; X2; X3];

% Method 1: Built-in kmeans
k = 3;
[idx, centroids] = kmeans(X, k, 'Replicates', 10);

% Method 2: Manual implementation
function [labels, centers] = my_kmeans(X, k, max_iter)
    [n, d] = size(X);
    
    % Random initialization
    rand_idx = randperm(n, k);
    centers = X(rand_idx, :);
    
    for iter = 1:max_iter
        % Assign points to nearest centroid
        distances = pdist2(X, centers);
        [~, labels] = min(distances, [], 2);
        
        % Update centroids
        new_centers = zeros(k, d);
        for j = 1:k
            new_centers(j, :) = mean(X(labels == j, :), 1);
        end
        
        % Check convergence
        if max(vecnorm(new_centers - centers, 2, 2)) < 1e-6
            break;
        end
        centers = new_centers;
    end
end

[my_idx, my_centers] = my_kmeans(X, 3, 100);

% Visualization
figure;
gscatter(X(:,1), X(:,2), idx);
hold on;
plot(centroids(:,1), centroids(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title('K-Means Clustering');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids');

% Elbow method for optimal k
inertia = zeros(10, 1);
for k = 1:10
    [~, ~, sumd] = kmeans(X, k, 'Replicates', 5);
    inertia(k) = sum(sumd);
end

figure;
plot(1:10, inertia, 'bo-');
xlabel('Number of Clusters');
ylabel('Inertia');
title('Elbow Method');
```

---

## Question 3

**Implement a neural network in MATLAB.**

### Answer

### MATLAB Code Example
```matlab
%% Neural Network for Classification

% Load data
load fisheriris
X = meas;
y = categorical(species);

% Split data
cv = cvpartition(y, 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

% Method 1: Using Deep Learning Toolbox
layers = [
    featureInputLayer(4, 'Name', 'input')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(32, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(3, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {X_test, y_test}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(X_train, y_train, layers, options);

% Evaluate
y_pred = classify(net, X_test);
accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix
figure;
confusionchart(y_test, y_pred);
title('Confusion Matrix');

% Method 2: Using patternnet (Neural Network Toolbox)
net2 = patternnet([20, 10]);
net2 = train(net2, X_train', dummyvar(y_train)');
```

---

## Question 4

**Implement PCA for dimensionality reduction.**

### Answer

### MATLAB Code Example
```matlab
%% Principal Component Analysis

% Load data
load fisheriris
X = meas;

% Standardize data
X_std = (X - mean(X)) ./ std(X);

% Method 1: Built-in pca
[coeff, score, latent, ~, explained] = pca(X_std);

fprintf('Variance explained:\n');
for i = 1:length(explained)
    fprintf('PC%d: %.2f%% (cumulative: %.2f%%)\n', ...
        i, explained(i), sum(explained(1:i)));
end

% Method 2: Manual implementation
function [V, D, scores] = my_pca(X)
    % Center data
    X_centered = X - mean(X);
    
    % Covariance matrix
    C = cov(X_centered);
    
    % Eigendecomposition
    [V, D] = eig(C);
    
    % Sort by eigenvalue (descending)
    [D, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    
    % Project data
    scores = X_centered * V;
end

[V_manual, D_manual, scores_manual] = my_pca(X_std);

% Visualization
figure;
subplot(1, 2, 1);
gscatter(score(:,1), score(:,2), species);
xlabel('PC1');
ylabel('PC2');
title('PCA: First 2 Components');

subplot(1, 2, 2);
bar(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Scree Plot');

% Biplot
figure;
biplot(coeff(:,1:2), 'Scores', score(:,1:2), 'VarLabels', ...
    {'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'});
title('PCA Biplot');
```

---

## Question 5

**Implement cross-validation for model evaluation.**

### Answer

### MATLAB Code Example
```matlab
%% Cross-Validation Implementation

% Load data
load fisheriris
X = meas;
y = categorical(species);

% Method 1: Built-in cross-validation
mdl = fitcsvm(X, y);
cv_mdl = crossval(mdl, 'KFold', 5);
cv_loss = kfoldLoss(cv_mdl);
fprintf('5-Fold CV Error: %.2f%%\n', cv_loss * 100);

% Method 2: Manual k-fold CV
function [mean_acc, std_acc] = manual_kfold_cv(X, y, k)
    n = size(X, 1);
    indices = crossvalind('Kfold', n, k);
    accuracies = zeros(k, 1);
    
    for i = 1:k
        % Split data
        test_idx = (indices == i);
        train_idx = ~test_idx;
        
        X_train = X(train_idx, :);
        y_train = y(train_idx);
        X_test = X(test_idx, :);
        y_test = y(test_idx);
        
        % Train and evaluate
        mdl = fitcsvm(X_train, y_train);
        y_pred = predict(mdl, X_test);
        accuracies(i) = sum(y_pred == y_test) / length(y_test);
    end
    
    mean_acc = mean(accuracies);
    std_acc = std(accuracies);
end

[mean_acc, std_acc] = manual_kfold_cv(X, y, 5);
fprintf('Manual 5-Fold CV: %.2f%% (+/- %.2f%%)\n', ...
    mean_acc * 100, std_acc * 100);

% Leave-one-out CV
cv_loo = crossval(mdl, 'Leaveout', 'on');
loo_loss = kfoldLoss(cv_loo);
fprintf('LOO CV Error: %.2f%%\n', loo_loss * 100);

% Stratified k-fold
cv_stratified = cvpartition(y, 'KFold', 5);
```

---

## Question 6

**Implement gradient descent optimization.**

### Answer

### MATLAB Code Example
```matlab
%% Gradient Descent Implementation

% Generate linear regression data
n = 100;
X = randn(n, 1);
y = 3 * X + 2 + 0.5 * randn(n, 1);

% Add bias term
X_aug = [ones(n, 1), X];

% Gradient descent for linear regression
function [weights, history] = gradient_descent(X, y, lr, epochs)
    [n, d] = size(X);
    weights = randn(d, 1);
    history = zeros(epochs, 1);
    
    for epoch = 1:epochs
        % Predictions
        y_pred = X * weights;
        
        % Compute loss (MSE)
        loss = mean((y - y_pred).^2);
        history(epoch) = loss;
        
        % Compute gradient
        gradient = (-2/n) * X' * (y - y_pred);
        
        % Update weights
        weights = weights - lr * gradient;
    end
end

% Train
learning_rate = 0.1;
epochs = 100;
[weights, loss_history] = gradient_descent(X_aug, y, learning_rate, epochs);

fprintf('Learned weights: bias=%.4f, slope=%.4f\n', weights(1), weights(2));

% Visualization
figure;
subplot(1, 2, 1);
scatter(X, y, 'b.');
hold on;
x_line = linspace(min(X), max(X), 100)';
y_line = [ones(100, 1), x_line] * weights;
plot(x_line, y_line, 'r-', 'LineWidth', 2);
xlabel('X');
ylabel('y');
title('Linear Regression Fit');

subplot(1, 2, 2);
plot(1:epochs, loss_history, 'b-');
xlabel('Epoch');
ylabel('MSE Loss');
title('Training Loss');
```

---

## Question 7

**Implement a decision tree classifier.**

### Answer

### MATLAB Code Example
```matlab
%% Decision Tree Classification

% Load data
load fisheriris
X = meas;
y = categorical(species);

% Split data
cv = cvpartition(y, 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

% Train decision tree
tree = fitctree(X_train, y_train);

% View tree
view(tree, 'Mode', 'graph');

% Predictions
y_pred = predict(tree, X_test);
accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Cross-validation
cv_tree = crossval(tree, 'KFold', 5);
cv_loss = kfoldLoss(cv_tree);
fprintf('CV Error: %.2f%%\n', cv_loss * 100);

% Hyperparameter tuning
min_leaf_sizes = [1, 5, 10, 20, 50];
cv_errors = zeros(length(min_leaf_sizes), 1);

for i = 1:length(min_leaf_sizes)
    tree_tuned = fitctree(X_train, y_train, ...
        'MinLeafSize', min_leaf_sizes(i));
    cv_tree_tuned = crossval(tree_tuned, 'KFold', 5);
    cv_errors(i) = kfoldLoss(cv_tree_tuned);
end

figure;
plot(min_leaf_sizes, cv_errors, 'bo-');
xlabel('Min Leaf Size');
ylabel('CV Error');
title('Hyperparameter Tuning');

% Feature importance
imp = predictorImportance(tree);
figure;
bar(imp);
xticklabels({'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'});
ylabel('Importance');
title('Feature Importance');
```

---

## Question 8

**Implement image classification with CNN.**

### Answer

### MATLAB Code Example
```matlab
%% CNN for Image Classification

% Load sample data
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', ...
    'nndemos', 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split data
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Define CNN architecture
layers = [
    imageInputLayer([28 28 1], 'Name', 'input')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.5, 'Name', 'dropout')
    
    fullyConnectedLayer(10, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train network
net = trainNetwork(imdsTrain, layers, options);

% Evaluate
y_pred = classify(net, imdsTest);
y_test = imdsTest.Labels;
accuracy = sum(y_pred == y_test) / numel(y_test);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix
figure;
confusionchart(y_test, y_pred);
title('Confusion Matrix');
```

