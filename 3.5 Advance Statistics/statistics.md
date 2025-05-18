## 1.1 Mathematical Foundations

### Linear Algebra Review

#### Eigenvalues and Eigenvectors in ML
- **Core Concepts**
  - Eigendecomposition and its significance in ML
  - Understanding eigenvalues as measures of variance along principal directions
  - Eigenvectors as directions of maximum variance
- **Applications**
  - PCA implementation from scratch using eigendecomposition
  - Understanding covariance matrices and their eigenstructure
  - Spectral methods in clustering (using eigenvalues for graph partitioning)
- **Advanced Topics**
  - Singular Value Decomposition (SVD) in detail
    - Full vs. reduced SVD
    - Relationship between SVD and PCA
  - Power iteration method for finding dominant eigenvalues
  - QR algorithm for computing all eigenvalues of a matrix
- **Implementation Focus**
  - Computing eigenvalues/vectors numerically using NumPy
  - Stability issues in eigendecomposition and their solutions
  - Implementing truncated SVD for dimensionality reduction

#### Matrix Decompositions
- **LU Decomposition**
  - Implementation and applications
  - Solving linear systems efficiently
- **QR Decomposition**
  - Orthogonalization process (Gram-Schmidt)
  - Applications in least squares problems
- **Cholesky Decomposition**
  - For positive definite matrices
  - Applications in sampling from multivariate Gaussians
- **Non-negative Matrix Factorization**
  - Algorithms for NMF (multiplicative update rules)
  - Applications in topic modeling and image decomposition
- **Tensor Decompositions**
  - CP decomposition
  - Tucker decomposition
  - Applications in higher-order data analysis

#### Advanced Transformations and Spaces
- **Vector Spaces and Subspaces**
  - Basis transformations and coordinate changes
  - Orthogonal complements and projections
- **Linear Mappings**
  - Kernel and image of linear transformations
  - Rank-nullity theorem and its implications
- **Inner Product Spaces**
  - Different inner products and their geometric interpretations
  - Gram matrices and kernel methods connection
- **Metrics and Norms**
  - L1, L2, and Lp norms and their properties
  - Frobenius norm for matrices
  - Nuclear norm and its relation to low-rank approximation
- **Matrix Calculus**
  - Derivatives with respect to vectors and matrices
  - Matrix differentiation rules
  - Applications in gradient-based optimization

### Calculus Extensions

#### Multivariable Calculus for Optimization
- **Gradients and Directional Derivatives**
  - Geometric interpretation of gradients
  - Computing and visualizing gradients
  - Steepest descent direction
- **Hessian Matrices**
  - Second-order derivatives and optimization
  - Eigenvalues of the Hessian and critical points
  - Newton's method using the Hessian
- **Taylor Series Expansions**
  - Multivariable Taylor series
  - Error analysis in approximations
  - Applications in numerical methods
- **Vector Calculus Identities**
  - Useful identities for ML derivations
  - Divergence, curl, and their interpretations
- **Line Integrals and Path Independence**
  - Relationship to conservative fields
  - Applications in physics-inspired ML methods

#### Lagrangian Multipliers and Constrained Optimization
- **Equality Constraints**
  - Lagrangian function formulation
  - First-order necessary conditions (KKT conditions)
  - Second-order sufficient conditions
- **Inequality Constraints**
  - Karush-Kuhn-Tucker (KKT) conditions in detail
  - Complementary slackness
  - Dual problem formulation
- **Applications in ML**
  - Support Vector Machines from optimization perspective
  - Constrained matrix factorization
  - Maximum entropy methods
- **Numerical Methods**
  - Penalty and barrier methods
  - Augmented Lagrangian methods
  - Sequential quadratic programming
- **Convex Optimization**
  - Convex sets and functions
  - Optimality conditions for convex problems
  - Strong and weak duality

#### Gradient Descent Variants and Convergence Properties
- **First-Order Methods**
  - Gradient descent with momentum
  - Nesterov accelerated gradient
  - Conjugate gradient method
- **Adaptive Learning Rate Methods**
  - AdaGrad, RMSprop, Adam in detail
  - Mathematical derivations and convergence guarantees
  - Hyperparameter sensitivity analysis
- **Convergence Analysis**
  - Convergence rates for different optimization landscapes
  - Effect of conditioning on convergence
  - Local vs. global convergence guarantees
- **Stochastic Methods**
  - Stochastic gradient descent analysis
  - Variance reduction techniques (SVRG, SAG, SAGA)
  - Mini-batch optimization strategies
- **Distributed Optimization**
  - Parallel and distributed gradient descent
  - Parameter server architecture
  - Federated optimization methods

### Probability Theory Mastery

#### Bayesian Statistics and Inference
- **Bayesian Framework**
  - Prior and posterior distributions
  - Conjugate priors and their applications
  - Hierarchical Bayesian models
- **Sampling Methods**
  - Rejection sampling
  - Importance sampling
  - Gibbs sampling implementation
  - Metropolis-Hastings algorithm
- **Variational Inference**
  - Evidence Lower Bound (ELBO)
  - Mean-field approximation
  - Stochastic variational inference
- **Bayesian Model Selection**
  - Bayes factors
  - Bayesian Information Criterion (BIC)
  - Minimum Description Length (MDL)
- **Applications**
  - Bayesian neural networks
  - Bayesian optimization for hyperparameter tuning
  - Bayesian nonparametrics (Dirichlet processes)

#### Advanced Sampling Methods
- **Markov Chain Monte Carlo (MCMC)**
  - Markov chain theory and stationary distributions
  - Mixing times and convergence diagnostics
  - Advanced MCMC algorithms:
    - Hamiltonian Monte Carlo
    - No-U-Turn Sampler (NUTS)
    - Slice sampling
- **Sequential Monte Carlo**
  - Particle filters
  - Sequential importance sampling with resampling
  - Applications in time series
- **Quasi-Monte Carlo Methods**
  - Low-discrepancy sequences
  - Error bounds and convergence rates
  - Applications in high-dimensional integration
- **Adaptive MCMC Methods**
  - Adaptive Metropolis algorithms
  - Adaptive proposal distributions
  - Reversible jump MCMC

#### Information Theory
- **Entropy Measures**
  - Shannon entropy
  - Differential entropy
  - Rényi entropy and generalizations
- **Divergence Measures**
  - Kullback-Leibler divergence properties
  - Jensen-Shannon divergence
  - f-divergences
  - Wasserstein distance
- **Mutual Information**
  - Properties and estimation techniques
  - Conditional mutual information
  - Total correlation (multivariate mutual information)
- **Maximum Entropy Principle**
  - MaxEnt distributions given constraints
  - Applications in natural language processing
  - Maximum entropy Markov models
- **Information Theory in ML**
  - Information bottleneck method
  - Minimum description length principle
  - Variational information maximization