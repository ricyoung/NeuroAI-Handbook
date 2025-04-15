# Appendix A: Math & Python Mini-Refresher

## A.1 Mathematical Foundations

### Linear Algebra Essentials
- Vectors and matrices
- Matrix operations
- Eigenvalues and eigenvectors
- Matrix decompositions

### Calculus Fundamentals
- Derivatives and gradients
- Chain rule and backpropagation connection
- Optimization basics
- Integrals for probability

### Probability and Statistics
- Random variables and distributions
- Expectation and variance
- Bayes' theorem
- Statistical tests

## A.2 Python Fundamentals

### Core Python
- Data types and structures
- Control flow
- Functions and lambda expressions
- Classes and object-oriented programming

### Scientific Python Ecosystem
- NumPy for numerical computing
- Matplotlib for visualization
- Pandas for data manipulation
- SciPy for scientific computing

### Machine Learning Libraries
- TensorFlow/Keras basics
- PyTorch fundamentals
- Scikit-learn for classical ML
- Common APIs and patterns

## A.3 Neuroscience Data Processing

### Common Data Formats
- Spike trains and rasters
- Time series data
- Imaging data formats
- EEG/MEG/fMRI data structures

### Preprocessing Techniques
- Filtering methods
- Artifact removal
- Signal normalization
- Feature extraction

## A.4 Code Examples

```python
# NumPy example: Creating a correlation matrix
import numpy as np

# Generate random data (10 samples, 5 features)
data = np.random.randn(10, 5)

# Calculate correlation matrix
corr_matrix = np.corrcoef(data, rowvar=False)
print("Correlation matrix shape:", corr_matrix.shape)

# Matplotlib example: Plotting the correlation matrix
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```