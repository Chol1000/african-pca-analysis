# Principal Component Analysis (PCA) Implementation

A comprehensive implementation of Principal Component Analysis from scratch using NumPy, designed for dimensionality reduction on datasets with missing values and non-numeric columns.

## Overview

This project implements PCA to reduce dataset dimensionality while preserving maximum variance. The implementation handles real-world data challenges including missing values, non-numeric columns, and dynamic component selection based on explained variance.

## Features

- **From-scratch PCA implementation** using only NumPy
- **Automatic data preprocessing** for missing values and non-numeric columns
- **Dynamic component selection** based on variance threshold (95%)
- **Comprehensive visualizations** (original vs PCA space, scree plot)
- **Detailed analysis** with feature importance and mathematical insights

## Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd pca-implementation

# Install required packages
pip install numpy pandas matplotlib
```

## Usage

### Quick Start

```python
import numpy as np
import pandas as pd
from pca_implementation import PCAAnalyzer

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Initialize and run PCA
pca = PCAAnalyzer()
reduced_data = pca.fit_transform(df)

# Visualize results
pca.plot_results()
```

### Dataset Requirements

Your dataset must have:
- **Missing values** (NaN values)
- **At least 1 non-numeric column**
- **More than 10 columns** total
- **Africanized data** (African context preferred)

## Implementation Details

### Step 1: Data Standardization
```python
standardized_data = (data - mean) / std
```

### Step 2: Covariance Matrix
```python
cov_matrix = (1 / (n - 1)) * np.dot(standardized_data.T, standardized_data)
```

### Step 3: Eigendecomposition
```python
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
```

### Step 4: Component Selection
```python
# Sort by eigenvalues (descending)
sorted_indices = np.argsort(eigenvalues)[::-1]
# Select components for 95% variance
num_components = np.argmax(cumulative_variance >= 95.0) + 1
```

### Step 5: Data Projection
```python
reduced_data = np.dot(standardized_data, sorted_eigenvectors[:, :num_components])
```

## Results

- **Dimensionality reduction**: 65 → 3 features
- **Variance retained**: 96.46%
- **Compression ratio**: 21.7x
- **Data points**: 1,516 samples processed

## File Structure

```
├── pca_notebook.ipynb          # Main implementation notebook
├── README.md                   # This file
├── data/                       # Dataset directory
│   └── API_SSF_DS2_en_csv_v2_14635.csv
└── requirements.txt            # Dependencies
```

## Key Components

### Data Handling
- Identifies and handles missing values with mean imputation
- Processes non-numeric columns with encoding capabilities
- Validates data quality throughout preprocessing

### PCA Implementation
- Manual standardization without sklearn
- Eigendecomposition for principal components
- Dynamic variance-based component selection
- Comprehensive result validation

### Visualization
- Original feature space plot
- Principal component space with variance labels
- Scree plot for component selection
- Color-coded scatter plots for pattern recognition

## Mathematical Foundation

The implementation follows standard PCA mathematics:

1. **Standardization**: Z = (X - μ) / σ
2. **Covariance**: C = (1/(n-1)) × Z^T × Z
3. **Eigendecomposition**: C = VΛV^T
4. **Projection**: Y = Z × V_k

## Performance Metrics

- **Explained Variance**: PC1 (61.9%), PC2 (29.0%), PC3 (5.6%)
- **Cumulative Variance**: 96.46% with 3 components
- **Feature Reduction**: 95.4% dimensionality reduction
- **Processing Time**: Optimized for datasets up to 10K samples

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].

---

**Note**: This implementation is designed for educational purposes and demonstrates PCA fundamentals. For production use, consider scikit-learn's optimized implementations.
