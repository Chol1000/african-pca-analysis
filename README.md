# African PCA Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**Principal Component Analysis implementation from scratch for African socio-economic data**

*Dimensionality Reduction • Data Visualization • Statistical Analysis*

</div>

---

## Assignment Overview

This assignment implements **Principal Component Analysis (PCA)** from scratch using NumPy to analyze African socio-economic indicators. The implementation demonstrates dimensionality reduction techniques on real-world data with missing values and mixed data types.

## Assignment Requirements Met

### PCA Implementation
- **From-scratch implementation** using only NumPy (no sklearn)
- **Eigendecomposition** for principal component extraction
- **Variance-based component selection** (95% threshold)
- **Step-by-step mathematical calculations**

### Data Requirements
- **Missing values present** in dataset (67,609 handled via imputation)
- **Non-numeric columns** included (country names, indicator codes)
- **More than 10 columns** (87 features after preprocessing)
- **African context data** (Sub-Saharan Africa indicators)

### Analysis & Visualization
- **Scree plot** for component selection
- **Principal component visualization** 
- **Variance explanation** analysis
- **Dimensionality reduction demonstration**

---

## Dataset Information

**Source**: World Bank API - Sub-Saharan Africa Indicators  
**File**: `API_SSF_DS2_en_csv_v2_14635.csv`  
**Characteristics**:
- **Samples**: 1,516 data points
- **Original Features**: 70 columns, 87 after preprocessing
- **Missing Values**: 67,609 (handled via imputation)
- **Non-numeric Columns**: Country names, indicator codes
- **Time Period**: Multi-year African development data

---

## Project Structure

```
african-pca-analysis/
├── data/
│   └── API_SSF_DS2_en_csv_v2_14635.csv    # African socio-economic dataset
├── pca_notebook.ipynb                      # Main implementation notebook
├── requirements.txt                        # Python dependencies
└── README.md                              # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Chol1000/african-pca-analysis.git
cd african-pca-analysis

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook pca_notebook.ipynb
```

### Dependencies
```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## Implementation Details

### Step 1: Data Preprocessing
```python
# Handle missing values
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

# Encode non-numeric columns with one-hot encoding
for col in non_numeric_cols:
    if 2 <= unique_vals <= 20:
        encoded = pd.get_dummies(df_encoded[col], prefix=col, prefix_sep='_')
        df_encoded = pd.concat([df_encoded, encoded], axis=1)

# Standardize features (numpy only)
standardized_data = (data_array - mean) / std
```

### Step 2: Covariance Matrix Calculation
```python
# Manual covariance calculation
n = standardized_data.shape[0]
cov_matrix = (1 / (n - 1)) * np.dot(standardized_data.T, standardized_data)
```

### Step 3: Eigendecomposition
```python
# Extract eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalues (descending)
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
```

### Step 4: Component Selection
```python
# Calculate explained variance
sorted_eigenvalues_real = sorted_eigenvalues.real
total_variance = np.sum(sorted_eigenvalues_real)
explained_variance = (sorted_eigenvalues_real / total_variance) * 100
cumulative_variance = np.cumsum(explained_variance)

# Select components for 95% variance
variance_threshold = 95.0
num_components = np.argmax(cumulative_variance >= variance_threshold) + 1
```

### Step 5: Data Transformation
```python
# Project data onto principal components
reduced_data = np.dot(standardized_data, sorted_eigenvectors[:, :num_components].real)
```

---

## Results & Insights

### Dimensionality Reduction
- **Original Dimensions**: 87 features
- **Reduced Dimensions**: 12 components
- **Compression Ratio**: 7.25:1
- **Variance Retained**: 95.07%

### Principal Components Analysis
| Component | Variance Explained | Cumulative Variance |
|-----------|-------------------|-------------------|
| PC1       | 46.21%           | 46.21%           |
| PC2       | 21.62%           | 67.83%           |
| PC3       | 4.59%            | 72.42%           |
| PC4       | 4.24%            | 76.66%           |
| PC5       | 2.31%            | 78.97%           |

### Key Findings
- **PC1**: Captures overall economic development indicators (46.2% variance)
- **PC2**: Represents demographic and social factors (21.6% variance)
- **PC3-PC12**: Reflect infrastructure, governance, and sector-specific metrics
- **Data Efficiency**: 86.2% dimensionality reduction achieved

---

## Visualizations

### 1. Scree Plot
Shows eigenvalue distribution for component selection
```python
plt.plot(range(1, len(eigenvalues_sorted) + 1), eigenvalues_sorted, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot - Component Selection')
```

### 2. Before/After PCA Comparison
Displays original vs transformed feature space
```python
# Original feature space
plt.scatter(standardized_data[:, 0], standardized_data[:, 1], c='blue')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Feature Space (Before PCA)')

# PCA space
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='red')
plt.xlabel(f'PC1 ({explained_variance[0]:.1f}% Variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.1f}% Variance)')
plt.title('Principal Component Space (After PCA)')
```

### 3. Variance Explanation Chart
Shows cumulative variance by component
```python
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
plt.axhline(y=95, color='k', linestyle='--', label='95% Threshold')
```

---

## Mathematical Foundation

### PCA Algorithm Steps
1. **Standardization**: $Z = \frac{X - \mu}{\sigma}$
2. **Covariance Matrix**: $C = \frac{1}{n-1}Z^TZ$
3. **Eigendecomposition**: $C = V\Lambda V^T$
4. **Component Selection**: Choose $k$ components where $\sum_{i=1}^k \lambda_i \geq 0.95 \sum_{i=1}^p \lambda_i$
5. **Transformation**: $Y = ZV_k$

### Validation Metrics
- **Explained Variance Ratio**: $\frac{\lambda_i}{\sum_{j=1}^p \lambda_j}$
- **Cumulative Variance**: $\sum_{i=1}^k \frac{\lambda_i}{\sum_{j=1}^p \lambda_j}$
- **Reconstruction Error**: $||X - X_{reconstructed}||_F^2$

---

## Getting Started

### Running the Analysis
```python
# Open Jupyter Notebook
jupyter notebook pca_notebook.ipynb

# Follow the step-by-step implementation:
# 1. Data loading and exploration
# 2. Preprocessing (missing values, standardization)
# 3. PCA implementation from scratch
# 4. Component selection and analysis
# 5. Visualization and interpretation
```

### Key Code Sections
```python
# Load and preprocess African dataset
df = pd.read_csv('data/API_SSF_DS2_en_csv_v2_14635.csv', skiprows=4)

# Implement PCA from scratch
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Select components for 95% variance
num_components = np.argmax(cumulative_variance >= 95.0) + 1
```

---

## Assignment Deliverables

### Implementation Results
- **Dimensionality Reduction**: 87 → 12 features (86.2% reduction)
- **Variance Retained**: 95.07% with 12 components
- **Processing Efficiency**: Handles 1,516 samples effectively
- **Data Quality**: 100% missing value coverage (67,609 values imputed)

### Technical Achievements
- **Mathematical Rigor**: Complete eigendecomposition from scratch
- **Data Preprocessing**: Robust handling of mixed data types and missing values
- **Visualization Quality**: Clear before/after PCA comparison plots
- **Code Documentation**: Well-commented implementation steps

### Learning Outcomes
- Implemented PCA algorithm without external ML libraries
- Demonstrated understanding of eigenvalue decomposition
- Applied dimensionality reduction to real-world African data
- Created effective visualizations for statistical analysis

---

## Academic Information

**Course**: Data Science/Machine Learning  
**Assignment**: PCA Implementation from Scratch  
**Dataset**: African Socio-Economic Indicators (World Bank)  
**Implementation**: Pure NumPy (no sklearn)

---

<div align="center">

**Built for Educational Purposes**

*Demonstrating PCA fundamentals through hands-on implementation*

</div>

---

## Acknowledgments

- **World Bank**: For providing comprehensive African development datasets
- **NumPy Community**: For robust numerical computing foundations
- **Jupyter Project**: For interactive development environment
- **African Development Research**: For inspiring socio-economic analysis

---

<div align="center">

**Built with Python • Powered by Mathematics • Focused on Africa**

*Demonstrating dimensionality reduction techniques for real-world socio-economic analysis*

</div>
