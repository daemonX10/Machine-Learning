<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Eigenvalues and Eigenvectors in Machine Learning: Concepts, Applications, and Implementation

Eigenvalue decomposition stands as a foundational mathematical tool in machine learning, enabling powerful techniques for dimensionality reduction, feature extraction, and data transformation. This report explores the theoretical underpinnings, diverse applications, and implementation considerations of eigenvalues and eigenvectors in machine learning. Research indicates that eigendecomposition forms the backbone of essential algorithms including Principal Component Analysis (PCA), spectral clustering, and various matrix factorization techniques. Recent advances focus on addressing stability issues in eigendecomposition and developing innovative methods for handling large-scale matrices while improving computational efficiency. The eigenvalue decomposition of symmetric matrices continues to be central to numerous computer vision algorithms and dimensionality reduction approaches, while Singular Value Decomposition (SVD) offers a powerful generalization for non-square matrices with wide-ranging applications.

## Core Concepts

### Eigendecomposition and its Significance in ML

Eigendecomposition is a fundamental matrix factorization technique that decomposes a square matrix into a set of eigenvectors and eigenvalues. According to the research, eigendecomposition of symmetric matrices is at the heart of many computer vision algorithms and machine learning techniques[^5]. It is considered a foundational result in applied mathematics, motivated by shared structures found in inferential problems of recent interest, such as orthogonal tensor decompositions[^15].

The eigendecomposition of a square matrix A is represented as:
\$ A = PDP^{-1} \$

Where:

- P is a matrix whose columns are the eigenvectors of A
- D is a diagonal matrix containing the eigenvalues of A
- P^(-1) is the inverse of P

The significance of eigendecomposition in machine learning stems from its ability to reveal the underlying structure and properties of data. It enables dimensionality reduction while preserving the most important information, facilitates understanding of data variance and correlation, forms the basis for many unsupervised learning algorithms, and provides tools for solving optimization problems in learning algorithms[^2].

### Understanding Eigenvalues as Measures of Variance Along Principal Directions

Eigenvalues quantify the amount of variance captured by their corresponding eigenvectors. In the context of machine learning, particularly in techniques like PCA, eigenvalues represent how much information or variance is preserved along each principal component (which are eigenvectors of the covariance matrix)[^2].

The magnitude of an eigenvalue directly correlates with the importance of its associated eigenvector in representing the data. Larger eigenvalues indicate directions along which the data exhibits greater variation, thus containing more information. This concept is fundamental to dimensionality reduction techniques, where components with small eigenvalues can be discarded with minimal loss of information.

When working with covariance matrices, eigenvalues represent the variance of the data along the direction of the corresponding eigenvector. The sum of all eigenvalues equals the total variance in the dataset, making them valuable for determining how many components to retain in dimensionality reduction to preserve a certain percentage of the original data's variance.

### Eigenvectors as Directions of Maximum Variance

Eigenvectors represent the directions or axes in the feature space along which the data varies most significantly. In machine learning, eigenvectors often correspond to the principal components in PCA or the basis vectors in other spectral methods[^2].

An eigenvector v of matrix A satisfies the equation:
\$ Av = \lambda v \$

Where λ is the corresponding eigenvalue. This equation indicates that when matrix A operates on vector v, it only scales the vector by λ without changing its direction.

In the context of machine learning, eigenvectors have several key properties:

1. They are orthogonal (perpendicular) to each other when derived from symmetric matrices like covariance matrices
2. They form a new coordinate system that better represents the variation in the data
3. They can be ranked by their corresponding eigenvalues to identify the most important directions of variance
4. They provide a basis for data projection and transformation

Finding leading eigenvalues and eigenfunctions (eigenvectors in functional spaces) is a fundamental task in many machine learning and scientific computing problems[^13]. For high-dimensional eigenvalue problems, training neural networks to parameterize the eigenfunctions is considered a promising alternative to classical numerical linear algebra techniques.

## Applications

### PCA Implementation from Scratch Using Eigendecomposition

Principal Component Analysis (PCA) is one of the most direct applications of eigendecomposition in machine learning. It is a dimensionality reduction technique that uses the eigendecomposition of the covariance matrix to transform the data into a new coordinate system defined by the principal components.

The basic steps to implement PCA from scratch using eigendecomposition are:

1. **Data Preparation**: Center the data by subtracting the mean from each feature
2. **Compute Covariance Matrix**: Calculate the covariance matrix of the centered data
3. **Eigendecomposition**: Perform eigendecomposition of the covariance matrix to obtain eigenvalues and eigenvectors
4. **Sort Components**: Sort eigenvectors by their corresponding eigenvalues in descending order
5. **Project Data**: Project the original data onto the selected principal components

According to the research, PCA is one of the examples from machine learning that results in eigenvalue problems. The principal components are the eigenvectors of the covariance matrix of the data, and they represent directions of maximum variance in the data[^2].

Recent advances in PCA implementation include "In-Memory Principal Component Analysis by Analogue Closed-Loop Eigendecomposition," which introduces a novel closed-loop in-memory computing circuit to compute real eigenvalues and eigenvectors of a target matrix for PCA acceleration. This approach reportedly achieves comparable accuracy and throughput to commercial GPUs while securing $10^4$ times energy efficiency and $10^{2\div 4}$ times area efficiency improvements[^3].

### Understanding Covariance Matrices and Their Eigenstructure

Covariance matrices play a central role in machine learning algorithms, particularly those involving dimensionality reduction and multivariate analysis. The eigenstructure of a covariance matrix provides critical insights into the data's variance and correlation patterns.

A covariance matrix C for a dataset X is computed as:
\$ C = \frac{1}{n-1} (X - \bar{X})^T(X - \bar{X}) \$

Where n is the number of data points, X is the data matrix, and $\bar{X}$ is the mean of X.

Key properties of covariance matrices and their eigenstructure include:

1. **Symmetry**: Covariance matrices are symmetric, ensuring real eigenvalues and orthogonal eigenvectors
2. **Positive Semi-definiteness**: All eigenvalues are non-negative, reflecting that variance cannot be negative
3. **Eigenvectors as Principal Directions**: Eigenvectors of the covariance matrix point in directions of maximum variance
4. **Eigenvalues as Variance Measures**: Each eigenvalue quantifies the amount of variance along its corresponding eigenvector

Online eigendecomposition of a sample covariance matrix over a network has applications in decentralized Direction-of-Arrival (DoA) estimation and tracking applications[^17]. This demonstrates how the eigenstructure of covariance matrices can be utilized in practical applications beyond standard PCA.

For high-dimensional eigenvalue problems, training neural networks to parameterize the eigenfunctions is considered a promising alternative to classical numerical linear algebra techniques[^13], which could be applied to large covariance matrices as well.

### Spectral Methods in Clustering

Spectral clustering leverages eigenvalues and eigenvectors of matrices derived from the data to perform dimensionality reduction before clustering. It is particularly effective for identifying non-convex clusters where traditional methods like k-means might fail.

The key steps in spectral clustering include:

1. **Construct a Similarity Graph**: Create a graph where nodes represent data points and edges represent similarities
2. **Compute the Graph Laplacian**: Form the Laplacian matrix from the adjacency matrix and degree matrix
3. **Perform Eigendecomposition**: Compute the eigenvalues and eigenvectors of the Laplacian
4. **Use Eigenvectors for Clustering**: The eigenvectors corresponding to the smallest non-zero eigenvalues form a lower-dimensional representation on which traditional clustering is performed

Research on large graph clustering notes that "the convergence of iterative singular value decomposition approaches depends on the eigengaps of the spectrum of the given matrix, i.e., the difference between consecutive eigenvalues"[^6]. Recent work has introduced "a parallelizable approach to dilating the spectrum in order to accelerate SVD solvers and in turn, spectral clustering"[^6].

The online computation of the spectra of the graph Laplacian is important in Graph Fourier applications[^17], which relates closely to spectral clustering methods. This indicates the importance of efficient eigendecomposition techniques for graph-based clustering approaches.

## Advanced Topics

### Singular Value Decomposition (SVD) in Detail

Singular Value Decomposition (SVD) is a generalization of eigendecomposition that works for any rectangular matrix. It decomposes a matrix into three simpler matrices and is widely used in machine learning for dimensionality reduction, noise filtering, and data compression.

For a matrix A of size m×n, the SVD is represented as:
\$ A = U\Sigma V^T \$

Where:

- U is an m×m orthogonal matrix containing the left singular vectors
- Σ is an m×n diagonal matrix containing the singular values
- V^T is the transpose of an n×n orthogonal matrix V containing the right singular vectors


#### Full vs. Reduced SVD

**Full SVD** includes all singular values and corresponding singular vectors, while **Reduced SVD** (or Truncated SVD) includes only the k largest singular values and their corresponding vectors. The reduced form is often sufficient for approximating the original matrix while substantially reducing dimensionality.

Research on "Robust Differentiable SVD" addresses the derivatives of eigenvectors that "tend to be unstable" and proposes methods to approximate them[^5]. The instability "arises in the presence of eigenvalues that are close to each other," which "makes integrating eigendecomposition into deep networks difficult and often results in poor convergence, particularly when dealing with large matrices"[^5].

Recent advances include an optimization framework "based on the low-rank approximation characterization of a truncated singular value decomposition" along with "new techniques called nesting for learning the top-L singular [values/vectors]"[^13]. This suggests ongoing development in efficient computation of SVD, particularly for applications in neural networks.

#### Relationship between SVD and PCA

The connection between SVD and PCA lies in how they both capture the principal components of data. When PCA is performed by eigendecomposition of the covariance matrix, the eigenvectors are equivalent to the right singular vectors obtained from SVD of the centered data matrix, and the eigenvalues are related to the square of the singular values.

Specifically, if X is a centered data matrix with n samples and d features:

1. PCA computes eigenvectors of X^T X (which is proportional to the covariance matrix)
2. SVD decomposes X into U, Σ, and V^T
3. The right singular vectors in V are identical to the eigenvectors from PCA
4. The singular values in Σ are related to the eigenvalues λ by σ = √λ

This relationship makes SVD an alternative and often more numerically stable way to implement PCA, especially for high-dimensional data.

### Power Iteration Method for Finding Dominant Eigenvalues

The power iteration method is a simple iterative algorithm for finding the dominant eigenvalue (the eigenvalue with the largest absolute value) and its corresponding eigenvector of a matrix. It's particularly useful for large, sparse matrices where computing the full eigendecomposition would be computationally expensive.

The basic algorithm follows these steps:

1. Start with a random vector v₀
2. Repeatedly multiply by the matrix: vₖ₊₁ = A vₖ
3. Normalize the vector after each iteration: vₖ₊₁ = vₖ₊₁ / ||vₖ₊₁||
4. Continue until convergence

After sufficient iterations, vₖ converges to the eigenvector corresponding to the dominant eigenvalue, and the ratio ||A vₖ|| / ||vₖ|| converges to the dominant eigenvalue.

The convergence of iterative methods like power iteration depends on the eigengaps (differences between consecutive eigenvalues) of the matrix spectrum[^6]. Larger eigengaps lead to faster convergence, which is a key consideration when applying these methods to practical problems.

### QR Algorithm for Computing All Eigenvalues of a Matrix

The QR algorithm is a more sophisticated method for computing all eigenvalues and eigenvectors of a matrix. It's based on repeatedly performing QR decompositions and is one of the most widely used algorithms in numerical linear algebra for eigenvalue problems.

The basic QR algorithm proceeds as follows:

1. Start with the original matrix A₀ = A
2. For k = 0, 1, 2, ...:
a. Compute the QR decomposition of Aₖ: Aₖ = QₖRₖ
b. Form the next iterate: Aₖ₊₁ = RₖQₖ
3. Continue until Aₖ converges to an upper triangular matrix (or a block upper triangular matrix with small blocks on the diagonal)

The eigenvalues of A appear on the diagonal of the converged matrix. The QR algorithm is typically applied to a Hessenberg form of the matrix (which has zeros below the first subdiagonal) for efficiency.

Recent research proposes "a new method for computing the eigenvalue decomposition of a dense real normal matrix A through the decomposition of its skew-symmetric part"[^16]. This method "relies on algorithms that are known to be efficiently implemented, such as the bidiagonal singular value decomposition and the symmetric eigenvalue decomposition" and "in most cases, the method has the same operation count as the Hessenberg [method]"[^16], suggesting a potential alternative to the QR algorithm for specific types of matrices.

## Implementation Focus

### Computing Eigenvalues/Vectors Numerically Using NumPy

NumPy is a widely used Python library for scientific computing that provides efficient implementations of eigenvalue and eigenvector computations. When implementing eigendecomposition or SVD in practice, NumPy offers several key functions:

1. **np.linalg.eig()**: Computes the eigenvalues and eigenvectors of a general square matrix

```python
import numpy as np
eigenvalues, eigenvectors = np.linalg.eig(matrix)
```

2. **np.linalg.eigh()**: Computes the eigenvalues and eigenvectors of a Hermitian (symmetric) matrix, which is faster and more accurate than the general case

```python
eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
```

3. **np.linalg.svd()**: Computes the singular value decomposition of a matrix

```python
U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
```

4. **np.linalg.eigvals()**: Computes only the eigenvalues, which is more efficient if eigenvectors are not needed

```python
eigenvalues = np.linalg.eigvals(matrix)
```


These NumPy functions implement optimized versions of algorithms like the QR algorithm, making them efficient for most practical applications. For very large matrices or specialized needs, custom implementations or specialized libraries might be necessary.

Efficient implementations rely on algorithms that are known to be efficiently implemented, such as the bidiagonal singular value decomposition and the symmetric eigenvalue decomposition[^16]. These optimized implementations are crucial for applying eigendecomposition to large-scale machine learning problems.

### Stability Issues in Eigendecomposition and Their Solutions

Numerical stability is a critical concern when computing eigendecompositions in practice. Several stability issues can arise:

1. **Closely Spaced Eigenvalues**: When eigenvalues are close to each other, computing their corresponding eigenvectors can be numerically unstable
2. **Ill-Conditioned Matrices**: Matrices with high condition numbers can lead to significant errors in eigendecomposition
3. **Roundoff Errors**: Accumulation of floating-point errors during iterative algorithms can affect accuracy
4. **Zero or Near-Zero Eigenvalues**: Very small eigenvalues can cause instability in various applications

Research directly addresses these stability issues, noting that "the derivatives of the eigenvectors tend to be unstable" and this "instability arises in the presence of eigenvalues that are close to each other"[^5]. This "makes integrating eigendecomposition into deep networks difficult and often results in poor convergence, particularly when dealing with large matrices"[^5].

Solutions to these stability issues include:

1. **Robust Algorithms**: Using numerically stable algorithms like the QR algorithm with shifts
2. **Preconditioning**: Transforming the matrix to improve its condition number before eigendecomposition
3. **Regularization**: Adding small values to the diagonal to improve conditioning
4. **Double Precision**: Using higher precision arithmetic for critical calculations
5. **Alternative Formulations**: Using "eigendecomposition-free approach to training a deep network whose loss depends on the eigenvector"[^14] to avoid numerical instability introduced by differentiating eigendecomposition operations

Research notes that "performing eigendecomposition within a network requires the ability to differentiate this operation. While theoretically doable, this introduces numerical instability in the optimization process in practice"[^14]. To address this, researchers have introduced "eigendecomposition-free approaches" that avoid these stability issues.

### Implementing Truncated SVD for Dimensionality Reduction

Truncated SVD (also known as reduced SVD) is a powerful technique for dimensionality reduction that retains only the k largest singular values and their corresponding singular vectors. This approach is particularly useful for large datasets where full decomposition would be computationally expensive or unnecessary.

The basic implementation of truncated SVD for dimensionality reduction involves:

1. **Compute Partial SVD**: Calculate only the top k singular values and vectors

```python
import numpy as np
U, s, Vt = np.linalg.svd(X, full_matrices=False)
U_reduced = U[:, :k]
s_reduced = s[:k]
Vt_reduced = Vt[:k, :]
```

2. **Project Data**: Project the original data onto the reduced space

```python
X_reduced = X @ Vt_reduced.T
# Or equivalently:
X_reduced = U_reduced @ np.diag(s_reduced)
```

3. **Reconstruct Data (Optional)**: If needed, approximate the original data from the reduced representation

```python
X_reconstructed = X_reduced @ Vt_reduced
# Or equivalently:
X_reconstructed = U_reduced @ np.diag(s_reduced) @ Vt_reduced
```


For very large matrices, specialized algorithms like randomized SVD can provide further computational advantages.

Recent research mentions "an optimization framework based on the low-rank approximation characterization of a truncated singular value decomposition" along with "new techniques called nesting for learning the top-L singular [values/vectors]"[^13]. Additionally, approaches to "dilating the spectrum in order to accelerate SVD solvers"[^6] have been developed, which "significantly accelerates convergence" and can be "parallelized and stochastically approximated to scale with available [resources]"[^6].

## Conclusion

Eigenvalues and eigenvectors represent fundamental mathematical concepts with profound implications in machine learning. From the foundational technique of Principal Component Analysis to advanced spectral clustering methods, eigendecomposition provides powerful tools for understanding and transforming data in meaningful ways.

Current research highlights ongoing efforts to address key challenges in eigendecomposition, particularly regarding numerical stability and computational efficiency for large-scale applications. Alternative approaches like eigendecomposition-free methods and specialized algorithms for normal matrices demonstrate the continued evolution of these techniques to meet the demands of modern machine learning applications.

As machine learning models grow in complexity and scale, efficient and stable computation of eigenvalues and eigenvectors will remain crucial. Future directions may include further integration with deep learning frameworks, specialized hardware acceleration (as suggested by the in-memory computing approach[^3]), and continued algorithmic improvements for handling extremely large datasets.

The connection between classical linear algebra techniques and modern machine learning approaches underscores the enduring importance of eigendecomposition as a fundamental tool in the data scientist's toolkit. By understanding both the mathematical foundations and practical implementation details, practitioners can leverage these powerful concepts to develop more effective and efficient machine learning solutions.

<div style="text-align: center">⁂</div>

[^1]: https://www.semanticscholar.org/paper/1854f1fbf7120407e110bb284d6c182f5bbf62e6

[^2]: https://arxiv.org/pdf/1903.11240.pdf

[^3]: https://www.semanticscholar.org/paper/bc793ff0ebde1f97ad1e932a93454427d02a3bdb

[^4]: https://arxiv.org/abs/2310.16978

[^5]: https://arxiv.org/pdf/2104.03821.pdf

[^6]: https://arxiv.org/abs/2207.14589

[^7]: https://www.semanticscholar.org/paper/7159fea64be18e8b5b9ae66dd9aa61f005a25a99

[^8]: https://arxiv.org/abs/2204.06815

[^9]: https://www.semanticscholar.org/paper/aafb3d5a1f0db08e059ec214699df6a1df9f04f2

[^10]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7246183/

[^11]: https://www.semanticscholar.org/paper/79a5b88370268049820705c51761153fb24ddd1f

[^12]: https://www.semanticscholar.org/paper/18d1b08b1cca0f0d6f8121281d6d16f986ac3f61

[^13]: http://arxiv.org/pdf/2402.03655.pdf

[^14]: https://arxiv.org/pdf/2004.07931.pdf

[^15]: http://arxiv.org/pdf/1411.1420.pdf

[^16]: http://arxiv.org/pdf/2410.12421.pdf

[^17]: https://arxiv.org/pdf/2209.01257.pdf

[^18]: https://pubmed.ncbi.nlm.nih.gov/22903772/

[^19]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4327321/

[^20]: http://arxiv.org/pdf/1509.01893.pdf

[^21]: https://arxiv.org/html/2408.10099v1

[^22]: https://www.pythonkitchen.com/eigen-decomposition-in-machine-learning/

[^23]: https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix

[^24]: https://builtin.com/data-science/eigendecomposition

[^25]: https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/eigendecomposition.html

[^26]: https://blog.clairvoyantsoft.com/eigen-decomposition-and-pca-c50f4ca15501

[^27]: https://math.stackexchange.com/questions/2147211/why-are-the-eigenvalues-of-a-covariance-matrix-equal-to-the-variance-of-its-eige

[^28]: https://math.stackexchange.com/questions/23596/why-is-the-eigenvector-of-a-covariance-matrix-equal-to-a-principal-component

[^29]: https://stats.stackexchange.com/questions/346692/how-does-eigenvalues-measure-variance-along-the-principal-components-in-pca

[^30]: https://www.reddit.com/r/3Blue1Brown/comments/1bgsd4p/is_there_a_way_to_geometrically_understand_why/

[^31]: https://builtin.com/data-science/covariance-matrix

[^32]: https://www.semanticscholar.org/paper/3bad113f32a6cd7b2591912c510c8a1495873f6a

[^33]: https://www.semanticscholar.org/paper/cc78f685582b2c4b00f959d38e47472c0e8a3111

[^34]: http://arxiv.org/pdf/2208.11977.pdf

[^35]: https://arxiv.org/pdf/1911.08751.pdf

[^36]: https://www.datacamp.com/tutorial/eigendecomposition

[^37]: https://www.youtube.com/watch?v=oshZQtYAh84

[^38]: https://www.machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/

[^39]: https://stackoverflow.com/questions/11920264/principal-component-variance-given-by-eigenvalue-for-principal-eigenvector

[^40]: https://www.semanticscholar.org/paper/f0a949de0c9c2ab48bd3680db1f9aa6c123d4862

[^41]: https://www.semanticscholar.org/paper/3534f6d42d3e8709d960e6aad34729a20f5377fa

[^42]: https://math.stackexchange.com/questions/3211467/why-eigenvectors-with-the-highest-eigenvalues-maximize-the-variance-in-pca

[^43]: https://towardsdatascience.com/principal-component-analysis-part-1-the-different-formulations-6508f63a5553/

[^44]: https://codesignal.com/learn/courses/navigating-data-simplification-with-pca/lessons/mastering-pca-eigenvectors-eigenvalues-and-covariance-matrix-explained

[^45]: https://arxiv.org/abs/1809.01827

[^46]: https://arxiv.org/abs/2004.07931

[^47]: https://arxiv.org/abs/2102.04152v1

[^48]: https://math.stackexchange.com/questions/243533/how-to-intuitively-understand-eigenvalue-and-eigenvector

[^49]: https://www.youtube.com/watch?v=tL0wFZ9aJP8

[^50]: https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/

[^51]: https://www.linkedin.com/pulse/complete-guide-principal-component-analysis-pca-machine-tripathi

[^52]: https://en.wikipedia.org/wiki/Principal_component_analysis

[^53]: https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors

[^54]: https://www.semanticscholar.org/paper/cb913c0dfcdcff1f3e9e0ce2e0ea8824453d7d6a

[^55]: https://www.semanticscholar.org/paper/77da4337b7634ad04b6b2080ac57a67f23d6e38c

[^56]: https://www.youtube.com/watch?v=ihUr2LbdYlE

[^57]: https://www.reddit.com/r/AskStatistics/comments/5ekg7h/relationship_of_eigenvalues_to_variance_pcasvd/

[^58]: https://pubmed.ncbi.nlm.nih.gov/38480677/

[^59]: https://arxiv.org/abs/2401.01433

[^60]: http://arxiv.org/pdf/1207.1854.pdf

[^61]: https://bitmask93.github.io/ml-blog/Eigendecomposition-SVD-and-PCA/

[^62]: https://www.youtube.com/watch?v=KTKAp9Q3yWg

[^63]: https://online.stat.psu.edu/statprogram/reviews/matrix-algebra/eigendecomposition

[^64]: https://www.soest.hawaii.edu/martel/Courses/GG303/Lec.19.2019.pptx.pdf

[^65]: https://math.stackexchange.com/questions/3666489/why-are-the-eigen-vectors-of-the-shape-operator-the-principal-directions-of-curv

[^66]: https://arxiv.org/abs/2205.15215

[^67]: https://pubmed.ncbi.nlm.nih.gov/34279739/

[^68]: https://arxiv.org/pdf/1709.02373.pdf

[^69]: http://arxiv.org/pdf/1211.2671.pdf

[^70]: https://builtin.com/data-science/step-step-explanation-principal-component-analysis

[^71]: https://stats.oarc.ucla.edu/spss/seminars/efa-spss/

[^72]: https://www.youtube.com/watch?v=fKivxsVlycs

[^73]: https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html

[^74]: https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues

[^75]: https://www.semanticscholar.org/paper/2a1ac932100ff62f1d336e6143f3ca9e754a639e

[^76]: https://www.semanticscholar.org/paper/5168cec304d20170deb65a0fc987aac0d45ce0d5

[^77]: https://arxiv.org/abs/2405.12675

[^78]: https://www.semanticscholar.org/paper/ab9afc5d1bc53444acc2d04df7d9159c9620d263

[^79]: https://www.semanticscholar.org/paper/e04494f25a3bb75ded13bdc1214adcc29b7b4086

[^80]: https://www.semanticscholar.org/paper/8250dac886e40b40ac35be570f90d5eee818da22

[^81]: https://www.semanticscholar.org/paper/139eb72c3771b8e42e47f2f2919fc8cf7f88482b

[^82]: https://arxiv.org/abs/2308.13641

[^83]: http://arxiv.org/pdf/1905.04452.pdf

[^84]: https://arxiv.org/abs/2010.11625

[^85]: https://www.semanticscholar.org/paper/561238401974c41e54a549b046d6cab2b04e8526

[^86]: https://www.semanticscholar.org/paper/f91bf7b9f60c537b9ab234a2623e22b3032eac81

[^87]: https://arxiv.org/abs/1201.5338

[^88]: https://www.semanticscholar.org/paper/9c359a59059ae2c6db61242fc63abf69a91c22d9

[^89]: https://www.semanticscholar.org/paper/e6288838c821d316c766c5f9c104b43c4812affc

[^90]: https://www.semanticscholar.org/paper/926e5694a1c5448c88f9bc8cbd83a3c21ea147c5

[^91]: https://arxiv.org/abs/1902.10414

[^92]: https://www.semanticscholar.org/paper/b76ef2ca04f747002e764d1eb07d7ffcf78f28c2

[^93]: https://arxiv.org/abs/2411.02308

[^94]: https://arxiv.org/abs/2205.09191

[^95]: https://arxiv.org/abs/2402.12687

[^96]: https://arxiv.org/abs/2310.12487

[^97]: https://www.semanticscholar.org/paper/7f297ef569ae2f4243eb4984810a713ff4fb762d

[^98]: https://arxiv.org/pdf/2007.10205.pdf

[^99]: https://arxiv.org/abs/2311.06115

[^100]: https://www.semanticscholar.org/paper/97e58eb9718691752ccf62c08886bc910f50fa9c

[^101]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10376520/

[^102]: https://www.semanticscholar.org/paper/b6c0c67e0ec81cb16620e2cfbccc3d1d474fec1a

[^103]: https://www.semanticscholar.org/paper/a7b56ef0ddf47bd2ce4bc869490d6e49cbe16593

[^104]: https://www.semanticscholar.org/paper/8c7f114db19e385bb72009e496b8da92a4d05d4e

[^105]: https://www.semanticscholar.org/paper/d830eb6169b3c3d6b90d0b596d1fee6b826ab7e8

[^106]: https://www.semanticscholar.org/paper/5a45aed1abce7042ca73dc9f0bb9608432d84796

[^107]: https://www.semanticscholar.org/paper/6053c370221b6d07190de2b97fdcee6381e3ce69

[^108]: https://www.semanticscholar.org/paper/2ff62bbf75521cf37346b18a366138749c3f7cc5

[^109]: https://www.semanticscholar.org/paper/79b71a81ca03b8ec7e1cfcdb886431515a107c83

[^110]: https://www.semanticscholar.org/paper/bac39c5dc195dfdca7f59828fd2948c3dc226fe7

[^111]: https://arxiv.org/abs/2104.12356

[^112]: https://www.semanticscholar.org/paper/fbb438f5358a0e6f54cf1517643e3026c6a13987

[^113]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9843315/

[^114]: https://www.semanticscholar.org/paper/aebb1b87df732d60f7b9faeab7d29f16cb84339a

[^115]: http://arxiv.org/pdf/2205.00165.pdf

[^116]: https://arxiv.org/abs/2306.09917

[^117]: https://www.semanticscholar.org/paper/4438acc36941128ab8268e5c5a3c4daa693c9529

[^118]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10288619/

[^119]: https://arxiv.org/abs/2007.02759

[^120]: https://arxiv.org/abs/1703.09039

[^121]: https://www.semanticscholar.org/paper/9cab311dd67df14ac394f98cf09a94b99c7e4d04

[^122]: https://arxiv.org/abs/2209.00491

[^123]: https://www.semanticscholar.org/paper/91ba00ee7775aeef84f644f3b2dd1baabf2cf67b

[^124]: https://arxiv.org/pdf/1907.08560.pdf

[^125]: https://arxiv.org/abs/2108.04209v1

[^126]: https://arxiv.org/pdf/0906.2543.pdf

[^127]: https://www.semanticscholar.org/paper/64ca4b162284d244e9b4c108b7902c926e64ffe3

[^128]: https://www.semanticscholar.org/paper/e9d4816408a5db648127a4029a5174bf3316cb62

[^129]: https://www.semanticscholar.org/paper/596275c7e0b568cb58fead6b968941d0e7164ae2

[^130]: https://www.semanticscholar.org/paper/30e17fd69c484808a22ac62d03338ff305b37000

[^131]: https://www.semanticscholar.org/paper/aac72d594963dbb5569be842f4f615029603e5d6

[^132]: https://arxiv.org/abs/2311.13580

[^133]: https://arxiv.org/abs/2505.07030

[^134]: https://arxiv.org/abs/2210.06590

[^135]: https://arxiv.org/pdf/0901.3245.pdf

[^136]: https://arxiv.org/html/2309.13838v2

[^137]: https://arxiv.org/html/2311.13580

[^138]: https://arxiv.org/pdf/1606.00187.pdf

[^139]: https://arxiv.org/abs/2008.01496

[^140]: https://arxiv.org/pdf/2301.01543.pdf

[^141]: http://arxiv.org/pdf/1702.06488.pdf

[^142]: http://arxiv.org/pdf/1206.4560.pdf

