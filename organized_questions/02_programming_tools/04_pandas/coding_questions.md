# ## Question 1

**Write a Pandas script to filter rows in a DataFrame based on a column's value being higher than a specified percentile.**

### Answer

#### Theory
Filtering DataFrame rows based on percentile values is a common data analysis task that allows you to identify outliers or focus on top/bottom performers. Pandas provides efficient methods to calculate percentiles using `quantile()` and filter data using boolean indexing.

#### Code Example
```python
import pandas as pd
import numpy as np

def filter_by_percentile(df, column_name, percentile_threshold=75):
    """
    Filter DataFrame rows where column values exceed specified percentile.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_name (str): Column to apply percentile filter
    percentile_threshold (float): Percentile threshold (0-100)
    
    Returns:
    pd.DataFrame: Filtered DataFrame
    """
    # Calculate percentile value
    percentile_value = df[column_name].quantile(percentile_threshold / 100)
    
    # Filter rows above percentile
    filtered_df = df[df[column_name] > percentile_value]
    
    return filtered_df, percentile_value

# Example usage
np.random.seed(42)
data = {
    'name': [f'Item_{i}' for i in range(1, 1001)],
    'value': np.random.normal(100, 20, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'score': np.random.uniform(0, 100, 1000)
}

df = pd.DataFrame(data)

# Filter by 90th percentile
high_performers, threshold = filter_by_percentile(df, 'value', 90)
print(f"90th percentile threshold: {threshold:.2f}")
print(f"Rows above 90th percentile: {len(high_performers)}")
print(high_performers.head())

# Advanced: Multiple column filtering
def filter_multiple_percentiles(df, column_percentile_dict):
    """
    Filter by multiple columns with different percentile thresholds.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_percentile_dict (dict): {column: percentile_threshold}
    
    Returns:
    pd.DataFrame: Filtered DataFrame
    """
    mask = pd.Series([True] * len(df))
    
    for column, percentile in column_percentile_dict.items():
        threshold = df[column].quantile(percentile / 100)
        mask &= (df[column] > threshold)
    
    return df[mask]

# Filter by multiple criteria
multi_filtered = filter_multiple_percentiles(df, {
    'value': 75,
    'score': 80
})
print(f"Multi-criteria filtered rows: {len(multi_filtered)}")
```

#### Explanation
1. **Percentile Calculation**: Use `quantile(percentile/100)` to convert percentile to decimal
2. **Boolean Indexing**: Create boolean mask with condition `df[column] > threshold`
3. **Flexible Function**: Parameterized function for reusability
4. **Multiple Columns**: Advanced function handles multiple percentile criteria simultaneously
5. **Performance**: Vectorized operations ensure efficient filtering

#### Use Cases
- **Sales Analysis**: Filter top-performing products/regions
- **Quality Control**: Identify items exceeding performance thresholds
- **Financial Analytics**: Focus on high-value transactions
- **Academic Research**: Study outliers in experimental data
- **Marketing Campaigns**: Target high-engagement customers

#### Best Practices
- **Validate Input**: Check column exists and contains numeric data
- **Handle Missing Values**: Use `dropna()` or `fillna()` before percentile calculation
- **Memory Efficiency**: Use `copy=False` for large DataFrames when possible
- **Documentation**: Clear parameter descriptions and return value explanations

#### Pitfalls
- **Empty Results**: Very high percentiles may return empty DataFrames
- **Data Types**: Ensure column contains numeric data for meaningful percentiles
- **Index Preservation**: Original index is maintained in filtered results
- **Memory Usage**: Large DataFrames may require chunking for memory management

#### Debugging
```python
# Debugging helpers
def debug_percentile_filter(df, column, percentile):
    threshold = df[column].quantile(percentile / 100)
    count_above = (df[column] > threshold).sum()
    
    print(f"Column: {column}")
    print(f"Percentile: {percentile}%")
    print(f"Threshold value: {threshold}")
    print(f"Rows above threshold: {count_above}")
    print(f"Percentage of data: {count_above/len(df)*100:.2f}%")
```

#### Optimization
```python
# Memory-efficient approach for large datasets
def filter_percentile_chunked(df, column_name, percentile_threshold, chunksize=10000):
    """Memory-efficient percentile filtering for large DataFrames."""
    # Calculate percentile on full dataset
    percentile_value = df[column_name].quantile(percentile_threshold / 100)
    
    # Process in chunks
    filtered_chunks = []
    for chunk in pd.read_csv('large_file.csv', chunksize=chunksize):
        filtered_chunk = chunk[chunk[column_name] > percentile_value]
        if not filtered_chunk.empty:
            filtered_chunks.append(filtered_chunk)
    
    return pd.concat(filtered_chunks, ignore_index=True)
```
---

## Question 2

**Code a function that concatenates two DataFrames and handles overlapping indices correctly.**

### Answer

#### Theory
Concatenating DataFrames with overlapping indices is a common challenge in data manipulation. Pandas provides several strategies to handle index conflicts, including resetting indices, using hierarchical indexing, or custom conflict resolution. Proper handling ensures data integrity and prevents unexpected behavior.

#### Code Example
```python
import pandas as pd
import numpy as np

def concatenate_with_overlapping_indices(df1, df2, index_strategy='reset', 
                                       conflict_resolution='raise', axis=0):
    """
    Concatenate DataFrames handling overlapping indices intelligently.
    
    Parameters:
    df1, df2 (pd.DataFrame): DataFrames to concatenate
    index_strategy (str): 'reset', 'hierarchical', 'preserve', 'suffix'
    conflict_resolution (str): 'raise', 'first', 'last', 'mean'
    axis (int): 0 for rows, 1 for columns
    
    Returns:
    pd.DataFrame: Concatenated DataFrame
    """
    
    # Check for overlapping indices
    overlapping_indices = df1.index.intersection(df2.index)
    
    if len(overlapping_indices) > 0 and conflict_resolution == 'raise':
        raise ValueError(f"Overlapping indices found: {overlapping_indices.tolist()}")
    
    if index_strategy == 'reset':
        # Reset indices before concatenation
        df1_copy = df1.reset_index(drop=True)
        df2_copy = df2.reset_index(drop=True)
        result = pd.concat([df1_copy, df2_copy], axis=axis, ignore_index=True)
        
    elif index_strategy == 'hierarchical':
        # Create hierarchical index
        result = pd.concat([df1, df2], axis=axis, keys=['df1', 'df2'])
        
    elif index_strategy == 'suffix':
        # Add suffixes to handle conflicts
        df1_copy = df1.copy()
        df2_copy = df2.copy()
        
        # Add suffix to overlapping indices in df2
        for idx in overlapping_indices:
            new_idx = f"{idx}_df2"
            df2_copy.rename(index={idx: new_idx}, inplace=True)
        
        result = pd.concat([df1_copy, df2_copy], axis=axis)
        
    elif index_strategy == 'preserve':
        # Handle overlapping data based on conflict_resolution
        if len(overlapping_indices) > 0:
            if conflict_resolution == 'first':
                # Keep first DataFrame's values for overlapping indices
                df2_filtered = df2.drop(overlapping_indices)
                result = pd.concat([df1, df2_filtered], axis=axis)
                
            elif conflict_resolution == 'last':
                # Keep second DataFrame's values for overlapping indices
                df1_filtered = df1.drop(overlapping_indices)
                result = pd.concat([df1_filtered, df2], axis=axis)
                
            elif conflict_resolution == 'mean':
                # Average overlapping values (numeric columns only)
                df1_non_overlap = df1.drop(overlapping_indices)
                df2_non_overlap = df2.drop(overlapping_indices)
                
                overlapping_data = []
                for idx in overlapping_indices:
                    row1 = df1.loc[idx]
                    row2 = df2.loc[idx]
                    
                    # Calculate mean for numeric columns
                    mean_row = {}
                    for col in df1.columns:
                        if pd.api.types.is_numeric_dtype(df1[col]):
                            mean_row[col] = (row1[col] + row2[col]) / 2
                        else:
                            mean_row[col] = row1[col]  # Keep first for non-numeric
                    
                    overlapping_data.append(mean_row)
                
                overlapping_df = pd.DataFrame(overlapping_data, index=overlapping_indices)
                result = pd.concat([df1_non_overlap, df2_non_overlap, overlapping_df], axis=axis)
        else:
            result = pd.concat([df1, df2], axis=axis)
    
    return result

# Example usage
np.random.seed(42)

# Create DataFrames with overlapping indices
df1 = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': ['a', 'b', 'c', 'd', 'e']
}, index=[0, 1, 2, 3, 4])

df2 = pd.DataFrame({
    'A': [6, 7, 8, 9],
    'B': [60, 70, 80, 90],
    'C': ['f', 'g', 'h', 'i']
}, index=[3, 4, 5, 6])  # Indices 3,4 overlap with df1

print("Original DataFrames:")
print("DF1:\n", df1)
print("\nDF2:\n", df2)

# Different concatenation strategies
print("\n=== Reset Index Strategy ===")
result1 = concatenate_with_overlapping_indices(df1, df2, 'reset')
print(result1)

print("\n=== Hierarchical Index Strategy ===")
result2 = concatenate_with_overlapping_indices(df1, df2, 'hierarchical')
print(result2)

print("\n=== Suffix Strategy ===")
result3 = concatenate_with_overlapping_indices(df1, df2, 'suffix')
print(result3)

print("\n=== Preserve with 'first' resolution ===")
result4 = concatenate_with_overlapping_indices(df1, df2, 'preserve', 'first')
print(result4)

print("\n=== Preserve with 'mean' resolution ===")
result5 = concatenate_with_overlapping_indices(df1, df2, 'preserve', 'mean')
print(result5)

# Advanced: Batch concatenation
def concatenate_multiple_dataframes(dataframes, **kwargs):
    """
    Concatenate multiple DataFrames with consistent overlap handling.
    
    Parameters:
    dataframes (list): List of DataFrames to concatenate
    **kwargs: Arguments passed to concatenate_with_overlapping_indices
    
    Returns:
    pd.DataFrame: Final concatenated DataFrame
    """
    if len(dataframes) < 2:
        return dataframes[0] if dataframes else pd.DataFrame()
    
    result = dataframes[0]
    for df in dataframes[1:]:
        result = concatenate_with_overlapping_indices(result, df, **kwargs)
    
    return result

# Example with multiple DataFrames
df3 = pd.DataFrame({
    'A': [10, 11, 12],
    'B': [100, 110, 120],
    'C': ['j', 'k', 'l']
}, index=[5, 6, 7])

all_dfs = [df1, df2, df3]
final_result = concatenate_multiple_dataframes(all_dfs, index_strategy='reset')
print("\n=== Multiple DataFrame Concatenation ===")
print(final_result)
```

#### Explanation
1. **Index Overlap Detection**: Use `df1.index.intersection(df2.index)` to identify conflicts
2. **Strategy Selection**: Different approaches based on use case requirements
3. **Conflict Resolution**: Handle overlapping data with various merge strategies
4. **Type Safety**: Preserve data types and handle non-numeric columns appropriately
5. **Extensibility**: Support for multiple DataFrames through batch processing

#### Use Cases
- **Time Series Merging**: Combining overlapping time periods
- **Data Pipeline Integration**: Merging outputs from different processing stages
- **Multi-source Data**: Consolidating data from various APIs or databases
- **Incremental Updates**: Adding new data while handling existing records
- **A/B Testing**: Combining experimental data with overlapping user IDs

#### Best Practices
- **Explicit Strategy**: Always specify how to handle overlaps rather than using defaults
- **Data Validation**: Check data consistency before concatenation
- **Index Naming**: Use meaningful index names for hierarchical strategies
- **Memory Management**: Consider memory usage for large DataFrames
- **Documentation**: Document chosen strategy for team understanding

#### Pitfalls
- **Silent Data Loss**: Overlapping indices may cause unexpected data loss
- **Type Mixing**: Concatenating incompatible data types
- **Index Alignment**: Misaligned indices can cause incorrect associations
- **Performance**: Large overlaps can significantly slow processing

#### Debugging
```python
def debug_concatenation(df1, df2):
    """Debug helper for concatenation issues."""
    print(f"DF1 shape: {df1.shape}")
    print(f"DF2 shape: {df2.shape}")
    print(f"DF1 index type: {type(df1.index)}")
    print(f"DF2 index type: {type(df2.index)}")
    
    overlaps = df1.index.intersection(df2.index)
    print(f"Overlapping indices: {len(overlaps)}")
    if len(overlaps) > 0:
        print(f"Overlap values: {overlaps.tolist()}")
    
    # Check column compatibility
    common_cols = set(df1.columns).intersection(set(df2.columns))
    print(f"Common columns: {common_cols}")
    
    # Check data types
    for col in common_cols:
        if df1[col].dtype != df2[col].dtype:
            print(f"Type mismatch in '{col}': {df1[col].dtype} vs {df2[col].dtype}")
```

#### Optimization
```python
# Memory-efficient concatenation for large DataFrames
def memory_efficient_concat(df1, df2, index_strategy='reset'):
    """
    Memory-efficient concatenation using categorical optimization.
    """
    # Convert string columns to categorical to save memory
    for df in [df1, df2]:
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
    
    # Use copy=False where possible
    if index_strategy == 'reset':
        return pd.concat([df1, df2], ignore_index=True, copy=False)
    else:
        return pd.concat([df1, df2], copy=False)
```

---

## Question 3

**Implement a data cleaning function that drops columns with more than 50% missing values and fills the remaining ones with column mean.**

### Answer

#### Theory
Data cleaning is a critical preprocessing step that involves handling missing values systematically. The strategy of dropping high-sparsity columns (>50% missing) removes low-information features, while mean imputation for remaining columns provides a simple yet effective approach for numeric data. This technique maintains data structure while improving model performance.

#### Code Example
```python
import pandas as pd
import numpy as np
from typing import Union, List, Dict

def clean_dataframe_advanced(df: pd.DataFrame, 
                           missing_threshold: float = 0.5,
                           imputation_strategy: str = 'mean',
                           numeric_only: bool = True,
                           exclude_columns: List[str] = None,
                           report: bool = True) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Advanced data cleaning function with comprehensive missing value handling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_threshold (float): Threshold for dropping columns (0.0-1.0)
    imputation_strategy (str): 'mean', 'median', 'mode', 'forward_fill', 'backward_fill'
    numeric_only (bool): Whether to apply imputation only to numeric columns
    exclude_columns (list): Columns to exclude from cleaning
    report (bool): Whether to generate cleaning report
    
    Returns:
    dict: Contains cleaned DataFrame and cleaning report
    """
    df_copy = df.copy()
    exclude_columns = exclude_columns or []
    
    # Initialize report
    cleaning_report = {
        'original_shape': df.shape,
        'columns_dropped': [],
        'columns_imputed': {},
        'missing_before': {},
        'missing_after': {}
    }
    
    # Record missing values before cleaning
    missing_before = df_copy.isnull().sum()
    missing_percentage = missing_before / len(df_copy)
    
    # Identify columns to drop
    columns_to_drop = []
    for col in df_copy.columns:
        if col not in exclude_columns:
            if missing_percentage[col] > missing_threshold:
                columns_to_drop.append(col)
                cleaning_report['columns_dropped'].append({
                    'column': col,
                    'missing_percentage': missing_percentage[col],
                    'missing_count': missing_before[col]
                })
    
    # Drop high-missing columns
    df_cleaned = df_copy.drop(columns=columns_to_drop)
    
    # Impute remaining missing values
    for col in df_cleaned.columns:
        if col not in exclude_columns and df_cleaned[col].isnull().any():
            
            if numeric_only and not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                continue
                
            original_missing = df_cleaned[col].isnull().sum()
            
            if imputation_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                fill_value = df_cleaned[col].mean()
                df_cleaned[col].fillna(fill_value, inplace=True)
                
            elif imputation_strategy == 'median' and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                fill_value = df_cleaned[col].median()
                df_cleaned[col].fillna(fill_value, inplace=True)
                
            elif imputation_strategy == 'mode':
                mode_value = df_cleaned[col].mode()
                if not mode_value.empty:
                    fill_value = mode_value.iloc[0]
                    df_cleaned[col].fillna(fill_value, inplace=True)
                    
            elif imputation_strategy == 'forward_fill':
                df_cleaned[col].fillna(method='ffill', inplace=True)
                fill_value = 'forward_fill'
                
            elif imputation_strategy == 'backward_fill':
                df_cleaned[col].fillna(method='bfill', inplace=True)
                fill_value = 'backward_fill'
            
            # Record imputation details
            if original_missing > 0:
                cleaning_report['columns_imputed'][col] = {
                    'strategy': imputation_strategy,
                    'fill_value': fill_value,
                    'missing_count': original_missing
                }
    
    # Final missing value check
    missing_after = df_cleaned.isnull().sum()
    cleaning_report['missing_before'] = missing_before.to_dict()
    cleaning_report['missing_after'] = missing_after.to_dict()
    cleaning_report['final_shape'] = df_cleaned.shape
    
    # Generate summary report
    if report:
        print_cleaning_report(cleaning_report)
    
    return {
        'cleaned_dataframe': df_cleaned,
        'report': cleaning_report
    }

def print_cleaning_report(report: Dict) -> None:
    """Print formatted cleaning report."""
    print("=" * 60)
    print("DATA CLEANING REPORT")
    print("=" * 60)
    print(f"Original shape: {report['original_shape']}")
    print(f"Final shape: {report['final_shape']}")
    print(f"Rows preserved: {report['final_shape'][0]}")
    print(f"Columns dropped: {len(report['columns_dropped'])}")
    print(f"Columns imputed: {len(report['columns_imputed'])}")
    
    if report['columns_dropped']:
        print("\nDropped Columns:")
        for col_info in report['columns_dropped']:
            print(f"  - {col_info['column']}: {col_info['missing_percentage']:.1%} missing")
    
    if report['columns_imputed']:
        print("\nImputed Columns:")
        for col, details in report['columns_imputed'].items():
            print(f"  - {col}: {details['missing_count']} values imputed with {details['strategy']}")

# Simple version of the function (as requested)
def clean_dataframe_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple data cleaning: drop columns >50% missing, fill remainder with mean.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_copy = df.copy()
    
    # Calculate missing percentage for each column
    missing_percentage = df_copy.isnull().sum() / len(df_copy)
    
    # Drop columns with more than 50% missing values
    columns_to_drop = missing_percentage[missing_percentage > 0.5].index
    df_cleaned = df_copy.drop(columns=columns_to_drop)
    
    # Fill remaining missing values with column mean (numeric columns only)
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_cleaned[col].isnull().any():
            mean_value = df_cleaned[col].mean()
            df_cleaned[col].fillna(mean_value, inplace=True)
    
    return df_cleaned

# Example usage and demonstration
np.random.seed(42)

# Create sample data with missing values
n_rows = 1000
data = {
    'id': range(1, n_rows + 1),
    'age': np.random.normal(35, 10, n_rows),
    'salary': np.random.normal(50000, 15000, n_rows),
    'score': np.random.uniform(0, 100, n_rows),
    'category': np.random.choice(['A', 'B', 'C', None], n_rows, p=[0.3, 0.3, 0.3, 0.1]),
    'high_missing_col': np.random.choice([1, 2, None], n_rows, p=[0.2, 0.2, 0.6]),  # 60% missing
    'some_missing_col': np.random.choice([10, 20, 30, None], n_rows, p=[0.25, 0.25, 0.25, 0.25])  # 25% missing
}

df = pd.DataFrame(data)

# Introduce additional missing values
missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[missing_indices, 'age'] = np.nan
df.loc[missing_indices[:50], 'salary'] = np.nan

print("Original DataFrame Info:")
print(f"Shape: {df.shape}")
print(f"Missing values per column:")
print(df.isnull().sum())
print(f"Missing percentage per column:")
print((df.isnull().sum() / len(df) * 100).round(2))

# Apply simple cleaning
cleaned_simple = clean_dataframe_simple(df)
print(f"\nSimple Cleaning Result:")
print(f"Shape: {cleaned_simple.shape}")
print(f"Remaining missing values: {cleaned_simple.isnull().sum().sum()}")

# Apply advanced cleaning
result = clean_dataframe_advanced(df, missing_threshold=0.5, imputation_strategy='mean')
cleaned_advanced = result['cleaned_dataframe']

# Validate cleaning results
print(f"\nValidation:")
print(f"Any missing values remaining: {cleaned_advanced.isnull().any().any()}")
print(f"Columns in original: {len(df.columns)}")
print(f"Columns after cleaning: {len(cleaned_advanced.columns)}")

# Performance comparison function
def compare_cleaning_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Compare different imputation strategies."""
    strategies = ['mean', 'median', 'mode']
    results = []
    
    for strategy in strategies:
        result = clean_dataframe_advanced(df, imputation_strategy=strategy, report=False)
        cleaned_df = result['cleaned_dataframe']
        
        results.append({
            'strategy': strategy,
            'final_shape': cleaned_df.shape,
            'columns_dropped': len(result['report']['columns_dropped']),
            'columns_imputed': len(result['report']['columns_imputed']),
            'missing_values': cleaned_df.isnull().sum().sum()
        })
    
    return pd.DataFrame(results)

comparison_df = compare_cleaning_strategies(df)
print(f"\nStrategy Comparison:")
print(comparison_df)
```

#### Explanation
1. **Missing Value Assessment**: Calculate percentage of missing values per column
2. **Column Filtering**: Drop columns exceeding 50% missing threshold
3. **Selective Imputation**: Apply mean imputation only to numeric columns
4. **Data Integrity**: Preserve original DataFrame with copy operation
5. **Comprehensive Reporting**: Track all cleaning operations for auditability

#### Use Cases
- **Machine Learning Preprocessing**: Prepare datasets for model training
- **Data Analysis**: Clean raw data before exploratory analysis
- **ETL Pipelines**: Standardize data cleaning across different sources
- **Research Data**: Handle survey data with incomplete responses
- **Business Intelligence**: Prepare operational data for reporting

#### Best Practices
- **Threshold Selection**: Choose missing value thresholds based on domain knowledge
- **Strategy Documentation**: Document imputation choices for reproducibility
- **Validation**: Always validate cleaning results before proceeding
- **Backup**: Keep original data intact with copy operations
- **Domain Consideration**: Consider business context when choosing imputation methods

#### Pitfalls
- **Information Loss**: High thresholds may remove important features
- **Mean Bias**: Mean imputation can reduce variance and introduce bias
- **Type Mixing**: Ensure imputation methods match data types
- **Temporal Dependencies**: Consider time-based patterns in missing data
- **Outlier Sensitivity**: Mean imputation is sensitive to outliers

#### Debugging
```python
def debug_missing_patterns(df: pd.DataFrame) -> None:
    """Analyze missing data patterns for debugging."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Missing value heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Missing Value Patterns')
    plt.tight_layout()
    plt.show()
    
    # Missing value statistics
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'Data_Type': df.dtypes
    })
    
    print("Missing Value Analysis:")
    print(missing_stats.sort_values('Missing_Percentage', ascending=False))

# Correlation analysis for imputation validation
def validate_imputation_impact(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    """Validate impact of imputation on data relationships."""
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        orig_corr = original_df[numeric_cols].corr()
        clean_corr = cleaned_df[numeric_cols].corr()
        
        corr_diff = abs(orig_corr - clean_corr).mean().mean()
        print(f"Average correlation change after imputation: {corr_diff:.4f}")
```

#### Optimization
```python
# Memory-efficient cleaning for large datasets
def clean_dataframe_memory_efficient(df: pd.DataFrame, chunksize: int = 10000) -> pd.DataFrame:
    """
    Memory-efficient cleaning for large DataFrames using chunking.
    """
    # First pass: identify columns to drop
    missing_percentage = df.isnull().sum() / len(df)
    columns_to_drop = missing_percentage[missing_percentage > 0.5].index
    
    # Calculate means for remaining numeric columns
    remaining_df = df.drop(columns=columns_to_drop)
    numeric_cols = remaining_df.select_dtypes(include=[np.number]).columns
    column_means = remaining_df[numeric_cols].mean()
    
    # Process in chunks
    cleaned_chunks = []
    for chunk in pd.read_csv('large_file.csv', chunksize=chunksize):
        # Drop high-missing columns
        chunk_cleaned = chunk.drop(columns=columns_to_drop)
        
        # Fill missing values with pre-calculated means
        for col in numeric_cols:
            if col in chunk_cleaned.columns:
                chunk_cleaned[col].fillna(column_means[col], inplace=True)
        
        cleaned_chunks.append(chunk_cleaned)
    
    return pd.concat(cleaned_chunks, ignore_index=True)
```

---

## Question 4

**Create a Pandas pipeline that ingests, processes, and summarizes time-series data from a CSV file.**

### Answer

#### Theory
Time-series data pipelines require careful handling of temporal indices, data validation, preprocessing, and aggregation. A well-designed pipeline ensures reproducible workflows with proper error handling, data quality checks, and flexible summarization capabilities. The pipeline pattern allows for modular processing steps that can be easily maintained and extended.

#### Code Example
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Union
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesPipeline:
    """
    Comprehensive time-series data processing pipeline.
    """
    
    def __init__(self, date_column: str = 'timestamp', 
                 freq: Optional[str] = None,
                 validation_rules: Optional[Dict] = None):
        """
        Initialize pipeline with configuration.
        
        Parameters:
        date_column (str): Name of the datetime column
        freq (str): Expected frequency ('D', 'H', 'M', etc.)
        validation_rules (dict): Data validation rules
        """
        self.date_column = date_column
        self.freq = freq
        self.validation_rules = validation_rules or {}
        self.processing_log = []
        
    def ingest_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Ingest time-series data from CSV with error handling.
        
        Parameters:
        file_path (str): Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
        Returns:
        pd.DataFrame: Raw ingested data
        """
        try:
            # Default CSV reading parameters for time-series
            default_params = {
                'parse_dates': [self.date_column] if self.date_column else None,
                'index_col': self.date_column if self.date_column else None,
                'date_parser': pd.to_datetime
            }
            default_params.update(kwargs)
            
            df = pd.read_csv(file_path, **default_params)
            
            self._log_step("Data Ingestion", f"Successfully loaded {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            self._log_step("Data Ingestion", f"Error: {str(e)}", is_error=True)
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate time-series data integrity.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        
        Returns:
        pd.DataFrame: Validated DataFrame
        """
        validation_results = {}
        
        # Check datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
                validation_results['datetime_conversion'] = 'Converted index to datetime'
            except:
                raise ValueError("Cannot convert index to datetime")
        
        # Check for duplicate timestamps
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            validation_results['duplicates'] = f"Found {duplicates} duplicate timestamps"
            df = df[~df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            validation_results['sorting'] = 'Sorted data by timestamp'
        
        # Check frequency consistency
        if self.freq:
            expected_periods = pd.date_range(
                start=df.index.min(), 
                end=df.index.max(), 
                freq=self.freq
            )
            missing_periods = len(expected_periods) - len(df)
            if missing_periods > 0:
                validation_results['missing_periods'] = f"{missing_periods} missing time periods"
        
        # Apply custom validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_result = rule_func(df)
                validation_results[rule_name] = rule_result
            except Exception as e:
                validation_results[rule_name] = f"Validation error: {str(e)}"
        
        self._log_step("Data Validation", validation_results)
        return df
    
    def process_data(self, df: pd.DataFrame, 
                    operations: List[Dict]) -> pd.DataFrame:
        """
        Apply processing operations to time-series data.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        operations (list): List of processing operations
        
        Returns:
        pd.DataFrame: Processed DataFrame
        """
        processed_df = df.copy()
        
        for operation in operations:
            op_type = operation.get('type')
            op_params = operation.get('params', {})
            
            if op_type == 'resample':
                processed_df = self._resample_data(processed_df, **op_params)
            elif op_type == 'fill_missing':
                processed_df = self._fill_missing_values(processed_df, **op_params)
            elif op_type == 'outlier_detection':
                processed_df = self._detect_outliers(processed_df, **op_params)
            elif op_type == 'feature_engineering':
                processed_df = self._engineer_features(processed_df, **op_params)
            elif op_type == 'smooth':
                processed_df = self._smooth_data(processed_df, **op_params)
            elif op_type == 'custom':
                func = op_params.get('function')
                if callable(func):
                    processed_df = func(processed_df, **op_params.get('kwargs', {}))
            
            self._log_step(f"Processing - {op_type}", f"Applied {op_type} operation")
        
        return processed_df
    
    def _resample_data(self, df: pd.DataFrame, freq: str, 
                      agg_func: Union[str, Dict] = 'mean') -> pd.DataFrame:
        """Resample time-series data to different frequency."""
        return df.resample(freq).agg(agg_func)
    
    def _fill_missing_values(self, df: pd.DataFrame, 
                           method: str = 'interpolate', **kwargs) -> pd.DataFrame:
        """Fill missing values in time-series data."""
        if method == 'interpolate':
            return df.interpolate(**kwargs)
        elif method == 'ffill':
            return df.fillna(method='ffill', **kwargs)
        elif method == 'bfill':
            return df.fillna(method='bfill', **kwargs)
        else:
            return df.fillna(**kwargs)
    
    def _detect_outliers(self, df: pd.DataFrame, columns: List[str], 
                        method: str = 'zscore', threshold: float = 3) -> pd.DataFrame:
        """Detect and flag outliers in time-series data."""
        df_copy = df.copy()
        
        for col in columns:
            if col in df_copy.columns:
                if method == 'zscore':
                    z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                    df_copy[f'{col}_outlier'] = z_scores > threshold
                elif method == 'iqr':
                    Q1 = df_copy[col].quantile(0.25)
                    Q3 = df_copy[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_copy[f'{col}_outlier'] = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
        
        return df_copy
    
    def _engineer_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Engineer time-based features."""
        df_copy = df.copy()
        
        for feature in features:
            if feature == 'hour':
                df_copy['hour'] = df_copy.index.hour
            elif feature == 'day_of_week':
                df_copy['day_of_week'] = df_copy.index.dayofweek
            elif feature == 'month':
                df_copy['month'] = df_copy.index.month
            elif feature == 'quarter':
                df_copy['quarter'] = df_copy.index.quarter
            elif feature == 'is_weekend':
                df_copy['is_weekend'] = df_copy.index.dayofweek.isin([5, 6])
            elif feature.startswith('rolling_'):
                window = int(feature.split('_')[1])
                metric = feature.split('_')[2]
                for col in df_copy.select_dtypes(include=[np.number]).columns:
                    if metric == 'mean':
                        df_copy[f'{col}_rolling_{window}'] = df_copy[col].rolling(window).mean()
                    elif metric == 'std':
                        df_copy[f'{col}_rolling_{window}'] = df_copy[col].rolling(window).std()
        
        return df_copy
    
    def _smooth_data(self, df: pd.DataFrame, columns: List[str], 
                    method: str = 'rolling', window: int = 3) -> pd.DataFrame:
        """Apply smoothing to time-series data."""
        df_copy = df.copy()
        
        for col in columns:
            if col in df_copy.columns:
                if method == 'rolling':
                    df_copy[f'{col}_smoothed'] = df_copy[col].rolling(window, center=True).mean()
                elif method == 'ewm':
                    df_copy[f'{col}_smoothed'] = df_copy[col].ewm(span=window).mean()
        
        return df_copy
    
    def summarize_data(self, df: pd.DataFrame, 
                      summary_config: Dict) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive summaries of time-series data.
        
        Parameters:
        df (pd.DataFrame): Processed DataFrame
        summary_config (dict): Summary configuration
        
        Returns:
        dict: Dictionary of summary DataFrames
        """
        summaries = {}
        
        # Basic statistics
        if summary_config.get('basic_stats', True):
            summaries['basic_stats'] = df.describe()
        
        # Temporal aggregations
        if 'aggregations' in summary_config:
            for agg_config in summary_config['aggregations']:
                freq = agg_config['frequency']
                metrics = agg_config.get('metrics', 'mean')
                name = agg_config.get('name', f'agg_{freq}')
                
                summaries[name] = df.resample(freq).agg(metrics)
        
        # Correlation analysis
        if summary_config.get('correlations', False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                summaries['correlations'] = df[numeric_cols].corr()
        
        # Missing value analysis
        if summary_config.get('missing_analysis', True):
            missing_data = pd.DataFrame({
                'Missing_Count': df.isnull().sum(),
                'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            summaries['missing_analysis'] = missing_data[missing_data['Missing_Count'] > 0]
        
        # Outlier summary
        outlier_cols = [col for col in df.columns if col.endswith('_outlier')]
        if outlier_cols and summary_config.get('outlier_summary', True):
            outlier_summary = df[outlier_cols].sum().to_frame('Outlier_Count')
            summaries['outlier_summary'] = outlier_summary
        
        self._log_step("Data Summarization", f"Generated {len(summaries)} summary reports")
        return summaries
    
    def run_pipeline(self, file_path: str, processing_operations: List[Dict],
                    summary_config: Dict, **read_kwargs) -> Dict:
        """
        Execute complete time-series pipeline.
        
        Parameters:
        file_path (str): Path to input CSV file
        processing_operations (list): Processing operations to apply
        summary_config (dict): Summary configuration
        **read_kwargs: Additional arguments for data ingestion
        
        Returns:
        dict: Complete pipeline results
        """
        try:
            # Execute pipeline steps
            raw_data = self.ingest_data(file_path, **read_kwargs)
            validated_data = self.validate_data(raw_data)
            processed_data = self.process_data(validated_data, processing_operations)
            summaries = self.summarize_data(processed_data, summary_config)
            
            results = {
                'raw_data': raw_data,
                'processed_data': processed_data,
                'summaries': summaries,
                'processing_log': self.processing_log.copy(),
                'pipeline_stats': {
                    'input_rows': len(raw_data),
                    'output_rows': len(processed_data),
                    'processing_steps': len(processing_operations),
                    'summary_reports': len(summaries)
                }
            }
            
            self._log_step("Pipeline Completion", "Pipeline executed successfully")
            return results
            
        except Exception as e:
            self._log_step("Pipeline Error", f"Pipeline failed: {str(e)}", is_error=True)
            raise
    
    def _log_step(self, step_name: str, message: Union[str, Dict], is_error: bool = False):
        """Log pipeline step execution."""
        log_entry = {
            'timestamp': datetime.now(),
            'step': step_name,
            'message': message,
            'status': 'ERROR' if is_error else 'SUCCESS'
        }
        self.processing_log.append(log_entry)

# Example usage and demonstration
def create_sample_timeseries_data(file_path: str = 'sample_timeseries.csv'):
    """Create sample time-series data for demonstration."""
    # Generate sample data
    date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    n_points = len(date_range)
    
    # Create realistic time-series patterns
    trend = np.linspace(0, 10, n_points)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))  # Weekly pattern
    noise = np.random.normal(0, 1, n_points)
    
    data = {
        'timestamp': date_range,
        'temperature': 20 + trend + seasonal + noise,
        'humidity': 50 + 10 * np.sin(2 * np.pi * np.arange(n_points) / 24) + 0.5 * noise,
        'pressure': 1013 + 5 * np.cos(2 * np.pi * np.arange(n_points) / (24 * 30)) + 0.2 * noise,
        'sales': np.maximum(0, 100 + trend * 10 + seasonal * 20 + 5 * noise)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values and outliers
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, 'temperature'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[outlier_indices, 'sales'] *= 3
    
    df.to_csv(file_path, index=False)
    return file_path

# Demonstration
if __name__ == "__main__":
    # Create sample data
    sample_file = create_sample_timeseries_data()
    
    # Define custom validation rule
    def check_temperature_range(df):
        temp_col = 'temperature'
        if temp_col in df.columns:
            invalid_temps = ((df[temp_col] < -50) | (df[temp_col] > 60)).sum()
            return f"Found {invalid_temps} temperatures outside valid range (-50°C to 60°C)"
        return "Temperature column not found"
    
    # Initialize pipeline
    pipeline = TimeSeriesPipeline(
        date_column='timestamp',
        freq='H',
        validation_rules={'temperature_range': check_temperature_range}
    )
    
    # Define processing operations
    processing_ops = [
        {
            'type': 'fill_missing',
            'params': {'method': 'interpolate', 'limit_direction': 'both'}
        },
        {
            'type': 'outlier_detection',
            'params': {'columns': ['sales', 'temperature'], 'method': 'zscore', 'threshold': 2.5}
        },
        {
            'type': 'feature_engineering',
            'params': {'features': ['hour', 'day_of_week', 'is_weekend', 'rolling_24_mean']}
        },
        {
            'type': 'resample',
            'params': {'freq': 'D', 'agg_func': {'temperature': 'mean', 'humidity': 'mean', 
                                               'pressure': 'mean', 'sales': 'sum'}}
        }
    ]
    
    # Define summary configuration
    summary_config = {
        'basic_stats': True,
        'correlations': True,
        'missing_analysis': True,
        'outlier_summary': True,
        'aggregations': [
            {'frequency': 'W', 'metrics': 'mean', 'name': 'weekly_averages'},
            {'frequency': 'M', 'metrics': {'sales': 'sum', 'temperature': 'mean'}, 'name': 'monthly_summary'}
        ]
    }
    
    # Run pipeline
    results = pipeline.run_pipeline(
        file_path=sample_file,
        processing_operations=processing_ops,
        summary_config=summary_config
    )
    
    # Display results
    print("Pipeline Results:")
    print(f"Input shape: {results['pipeline_stats']['input_rows']} rows")
    print(f"Output shape: {results['pipeline_stats']['output_rows']} rows")
    print(f"Processing steps: {results['pipeline_stats']['processing_steps']}")
    print(f"Summary reports: {results['pipeline_stats']['summary_reports']}")
    
    print("\nProcessed Data Sample:")
    print(results['processed_data'].head())
    
    print("\nBasic Statistics:")
    print(results['summaries']['basic_stats'])
    
    print("\nWeekly Averages (first 5 weeks):")
    print(results['summaries']['weekly_averages'].head())
```

#### Explanation
1. **Modular Design**: Pipeline class with separate methods for each processing stage
2. **Error Handling**: Comprehensive error handling with detailed logging
3. **Flexible Processing**: Configurable operations that can be chained together
4. **Data Validation**: Automatic validation of time-series data integrity
5. **Rich Summarization**: Multiple summary views for comprehensive analysis

#### Use Cases
- **IoT Data Processing**: Sensor data ingestion and analysis
- **Financial Data**: Stock price analysis and trading signal generation
- **Operations Analytics**: Manufacturing or service metrics monitoring
- **Weather Analysis**: Meteorological data processing and forecasting
- **Web Analytics**: User behavior and traffic pattern analysis

#### Best Practices
- **Configuration-Driven**: Use configuration files for pipeline parameters
- **Logging**: Comprehensive logging for debugging and audit trails
- **Modularity**: Separate concerns into distinct processing steps
- **Validation**: Always validate data quality at each step
- **Documentation**: Clear documentation of processing decisions

#### Pitfalls
- **Memory Usage**: Large time-series can consume significant memory
- **Time Zone Issues**: Inconsistent time zones can cause alignment problems
- **Missing Data**: Irregular missing data patterns can break assumptions
- **Frequency Changes**: Data with changing frequencies requires special handling
- **Outlier Impact**: Outliers can skew summary statistics significantly

#### Debugging
```python
def debug_pipeline_performance(pipeline_results: Dict) -> None:
    """Debug pipeline performance and data quality."""
    log = pipeline_results['processing_log']
    
    print("Pipeline Execution Log:")
    for entry in log:
        status_symbol = "❌" if entry['status'] == 'ERROR' else "✅"
        print(f"{status_symbol} {entry['step']}: {entry['message']}")
    
    # Data quality checks
    processed_data = pipeline_results['processed_data']
    print(f"\nData Quality Summary:")
    print(f"Missing values: {processed_data.isnull().sum().sum()}")
    print(f"Duplicate indices: {processed_data.index.duplicated().sum()}")
    print(f"Data types: {processed_data.dtypes.value_counts()}")
```

#### Optimization
```python
# Memory-efficient pipeline for large datasets
class MemoryEfficientTimeSeriesPipeline(TimeSeriesPipeline):
    """Memory-optimized version using chunked processing."""
    
    def process_large_file(self, file_path: str, chunksize: int = 10000, **kwargs):
        """Process large time-series files in chunks."""
        chunks = []
        
        for chunk in pd.read_csv(file_path, chunksize=chunksize, **kwargs):
            # Process each chunk
            processed_chunk = self.process_data(chunk, self.processing_operations)
            chunks.append(processed_chunk)
        
        # Combine processed chunks
        return pd.concat(chunks, ignore_index=True)
```

---

## Question 5

**Write a Python function that takes a DataFrame and computes the correlation matrix, then visualizes it using Seaborn’s heatmap.**

### Answer

#### Theory
Correlation analysis is fundamental in data science for understanding relationships between variables. The correlation matrix shows pairwise correlations between all numeric variables, with values ranging from -1 (perfect negative correlation) to +1 (perfect positive correlation). Visualizing correlations through heatmaps provides immediate insights into data patterns, multicollinearity issues, and feature relationships critical for machine learning and statistical analysis.

#### Code Example
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Union
import warnings
warnings.filterwarnings('ignore')

def analyze_and_visualize_correlations(df: pd.DataFrame,
                                     method: str = 'pearson',
                                     exclude_columns: Optional[List[str]] = None,
                                     include_only: Optional[List[str]] = None,
                                     threshold: Optional[float] = None,
                                     figsize: Tuple[int, int] = (12, 10),
                                     annot: bool = True,
                                     cmap: str = 'RdBu_r',
                                     center: float = 0,
                                     save_path: Optional[str] = None,
                                     return_high_correlations: bool = False) -> Dict[str, Union[pd.DataFrame, List]]:
    """
    Comprehensive correlation analysis and visualization function.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    method (str): Correlation method ('pearson', 'spearman', 'kendall')
    exclude_columns (list, optional): Columns to exclude from analysis
    include_only (list, optional): Only include specified columns
    threshold (float, optional): Highlight correlations above this threshold
    figsize (tuple): Figure size for the heatmap
    annot (bool): Whether to annotate heatmap cells with correlation values
    cmap (str): Colormap for the heatmap
    center (float): Value at which to center the colormap
    save_path (str, optional): Path to save the heatmap image
    return_high_correlations (bool): Whether to return high correlation pairs
    
    Returns:
    dict: Contains correlation matrix, plot object, and optional high correlations
    """
    
    # Data preprocessing
    df_analysis = df.copy()
    
    # Handle column selection
    if include_only:
        missing_cols = [col for col in include_only if col not in df_analysis.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        df_analysis = df_analysis[include_only]
    
    if exclude_columns:
        exclude_columns = [col for col in exclude_columns if col in df_analysis.columns]
        df_analysis = df_analysis.drop(columns=exclude_columns)
    
    # Select only numeric columns
    numeric_columns = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation analysis")
    
    df_numeric = df_analysis[numeric_columns]
    
    # Handle missing values
    if df_numeric.isnull().any().any():
        print(f"Warning: Found missing values. Using pairwise complete observations.")
        correlation_matrix = df_numeric.corr(method=method, min_periods=1)
    else:
        correlation_matrix = df_numeric.corr(method=method)
    
    # Create the visualization
    plt.figure(figsize=figsize)
    
    # Create mask for upper triangle (optional - removes redundancy)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate heatmap
    heatmap = sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=annot,
        cmap=cmap,
        center=center,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        fmt='.2f' if annot else None
    )
    
    plt.title(f'{method.capitalize()} Correlation Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    # Find high correlations if requested
    high_correlations = []
    if return_high_correlations and threshold:
        high_correlations = find_high_correlations(correlation_matrix, threshold)
    
    results = {
        'correlation_matrix': correlation_matrix,
        'heatmap_plot': heatmap,
        'numeric_columns_analyzed': numeric_columns,
        'missing_values_found': df_numeric.isnull().sum().sum()
    }
    
    if return_high_correlations:
        results['high_correlations'] = high_correlations
    
    return results

def find_high_correlations(corr_matrix: pd.DataFrame, threshold: float) -> List[Dict]:
    """
    Find correlation pairs above the specified threshold.
    
    Parameters:
    corr_matrix (pd.DataFrame): Correlation matrix
    threshold (float): Minimum correlation threshold
    
    Returns:
    list: List of high correlation pairs with details
    """
    high_corrs = []
    
    # Get upper triangle indices to avoid duplicates
    upper_triangle = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    
    for i, row in enumerate(corr_matrix.index):
        for j, col in enumerate(corr_matrix.columns):
            if upper_triangle[i, j]:
                correlation_value = corr_matrix.iloc[i, j]
                if abs(correlation_value) >= threshold:
                    high_corrs.append({
                        'feature_1': row,
                        'feature_2': col,
                        'correlation': correlation_value,
                        'abs_correlation': abs(correlation_value),
                        'relationship': 'positive' if correlation_value > 0 else 'negative'
                    })
    
    # Sort by absolute correlation value
    high_corrs.sort(key=lambda x: x['abs_correlation'], reverse=True)
    return high_corrs

# Simple function as requested
def compute_and_visualize_correlation(df: pd.DataFrame, 
                                    figsize: Tuple[int, int] = (10, 8),
                                    method: str = 'pearson') -> pd.DataFrame:
    """
    Simple function to compute correlation matrix and create heatmap visualization.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    figsize (tuple): Size of the figure
    method (str): Correlation method
    
    Returns:
    pd.DataFrame: Correlation matrix
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation analysis")
    
    # Compute correlation matrix
    correlation_matrix = numeric_df.corr(method=method)
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f')
    
    plt.title(f'{method.capitalize()} Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

# Example usage and demonstration
def create_sample_data_for_correlation():
    """Create sample data with known correlations for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create variables with different correlation patterns
    x1 = np.random.normal(0, 1, n_samples)
    x2 = 0.8 * x1 + np.random.normal(0, 0.6, n_samples)  # Strong positive correlation
    x3 = -0.6 * x1 + np.random.normal(0, 0.8, n_samples)  # Moderate negative correlation
    x4 = np.random.normal(0, 1, n_samples)  # Independent variable
    x5 = x1 ** 2 + np.random.normal(0, 0.1, n_samples)  # Non-linear relationship
    
    # Create categorical variable
    categories = np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.3, 0.3, 0.2, 0.2])
    
    # Create target variable
    target = 2 * x1 + 1.5 * x2 - x3 + 0.5 * x4 + np.random.normal(0, 0.5, n_samples)
    
    data = pd.DataFrame({
        'feature_1': x1,
        'feature_2': x2,
        'feature_3': x3,
        'feature_4': x4,
        'feature_5': x5,
        'category': categories,
        'target': target
    })
    
    return data

# Demonstration
if __name__ == "__main__":
    # Create sample data
    sample_data = create_sample_data_for_correlation()
    
    print("Sample Data Overview:")
    print(sample_data.head())
    print(f"\nData shape: {sample_data.shape}")
    print(f"Data types:\n{sample_data.dtypes}")
    
    # Simple correlation analysis
    print("\n" + "="*60)
    print("SIMPLE CORRELATION ANALYSIS")
    print("="*60)
    
    simple_corr = compute_and_visualize_correlation(sample_data)
    print("\nCorrelation Matrix:")
    print(simple_corr.round(3))
    
    # Advanced correlation analysis
    print("\n" + "="*60)
    print("COMPREHENSIVE CORRELATION ANALYSIS")
    print("="*60)
    
    results = analyze_and_visualize_correlations(
        sample_data,
        method='pearson',
        threshold=0.5,
        return_high_correlations=True,
        exclude_columns=['category']
    )
    
    print("High correlations found:")
    for corr in results['high_correlations']:
        print(f"  {corr['feature_1']} ↔ {corr['feature_2']}: {corr['correlation']:.3f}")
```

#### Explanation
1. **Data Preprocessing**: Automatically selects numeric columns and handles missing values appropriately
2. **Multiple Correlation Methods**: Supports Pearson, Spearman, and Kendall correlation coefficients
3. **Flexible Visualization**: Customizable heatmap with various styling options and triangle masking
4. **High Correlation Detection**: Identifies and reports correlations above specified thresholds
5. **Error Handling**: Validates input data and provides meaningful error messages
6. **Professional Styling**: Well-formatted heatmap with proper labels, colors, and annotations

#### Use Cases
- **Feature Selection**: Identify highly correlated features for dimensionality reduction
- **Multicollinearity Detection**: Detect redundant variables before model training
- **Exploratory Data Analysis**: Understand relationships between variables in datasets
- **Data Quality Assessment**: Identify unexpected correlations that might indicate data issues
- **Business Intelligence**: Analyze relationships between business metrics and KPIs

#### Best Practices
- **Method Selection**: Use Pearson for linear relationships, Spearman for monotonic relationships
- **Missing Value Handling**: Address missing values before correlation analysis
- **Threshold Setting**: Use domain knowledge to set meaningful correlation thresholds
- **Visual Clarity**: Use appropriate color schemes and annotations for clear interpretation
- **Statistical Significance**: Consider sample size and statistical significance of correlations

#### Pitfalls
- **Causation vs Correlation**: Correlation does not imply causation
- **Non-linear Relationships**: Pearson correlation may miss non-linear associations
- **Outlier Sensitivity**: Pearson correlation is sensitive to outliers
- **Sample Size**: Small samples can produce misleading correlation estimates
- **Multiple Testing**: High number of correlations increases false discovery risk

#### Debugging
```python
def debug_correlation_issues(df: pd.DataFrame) -> None:
    """Debug common correlation analysis problems."""
    print("Correlation Debugging Report:")
    print(f"DataFrame shape: {df.shape}")
    
    # Check data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Numeric columns: {len(numeric_cols)}")
    
    # Check for missing values
    missing_summary = df.isnull().sum()
    if missing_summary.sum() > 0:
        print("Missing values found:")
        print(missing_summary[missing_summary > 0])
    
    # Check for constant columns
    constant_cols = df.select_dtypes(include=[np.number]).columns[df.select_dtypes(include=[np.number]).std() == 0]
    if len(constant_cols) > 0:
        print(f"Constant columns (will cause correlation issues): {constant_cols.tolist()}")
```

#### Optimization
```python
# Memory-efficient correlation for large datasets
def correlation_for_large_datasets(df: pd.DataFrame, 
                                 chunksize: int = 10000,
                                 method: str = 'pearson') -> pd.DataFrame:
    """
    Compute correlations efficiently for large datasets using chunking.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df) <= chunksize:
        return numeric_df.corr(method=method)
    
    # For very large datasets, consider sampling or incremental computation
    sample_size = min(chunksize, len(numeric_df))
    sampled_df = numeric_df.sample(n=sample_size, random_state=42)
    
    return sampled_df.corr(method=method)
```

---

## Question 6

**If you have a DataFrame with multiple datetime columns, detail how you would create a new column combining them into the earliest datetime.**

### Answer

#### Theory
Combining multiple datetime columns to find the earliest datetime is a common task in data processing, especially when dealing with event data, timestamps from different sources, or multi-stage processes. This operation requires proper datetime handling, missing value consideration, and efficient vectorized operations to maintain performance on large datasets.

#### Code Example
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Union

def find_earliest_datetime(df: pd.DataFrame, 
                         datetime_columns: List[str],
                         new_column_name: str = 'earliest_datetime',
                         handle_missing: str = 'ignore',
                         return_source_column: bool = False) -> pd.DataFrame:
    """
    Create a new column with the earliest datetime from multiple datetime columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    datetime_columns (list): List of datetime column names
    new_column_name (str): Name for the new earliest datetime column
    handle_missing (str): How to handle missing values ('ignore', 'skip_row', 'fill_na')
    return_source_column (bool): Whether to include column indicating source of earliest datetime
    
    Returns:
    pd.DataFrame: DataFrame with new earliest datetime column
    """
    
    # Validate input
    df_copy = df.copy()
    missing_columns = [col for col in datetime_columns if col not in df_copy.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
    
    # Ensure all specified columns are datetime type
    for col in datetime_columns:
        if not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            try:
                df_copy[col] = pd.to_datetime(df_copy[col])
                print(f"Converted column '{col}' to datetime")
            except Exception as e:
                raise ValueError(f"Cannot convert column '{col}' to datetime: {str(e)}")
    
    # Handle different missing value strategies
    if handle_missing == 'ignore':
        # Use min() with skipna=True (default behavior)
        df_copy[new_column_name] = df_copy[datetime_columns].min(axis=1)
        
    elif handle_missing == 'skip_row':
        # Only process rows where all datetime columns have values
        mask = df_copy[datetime_columns].notna().all(axis=1)
        df_copy.loc[mask, new_column_name] = df_copy.loc[mask, datetime_columns].min(axis=1)
        df_copy.loc[~mask, new_column_name] = pd.NaT
        
    elif handle_missing == 'fill_na':
        # Fill missing values with a far future date before finding min
        far_future = pd.Timestamp('2099-12-31')
        temp_df = df_copy[datetime_columns].fillna(far_future)
        earliest = temp_df.min(axis=1)
        # Replace far future dates back to NaT if all were missing
        all_missing_mask = df_copy[datetime_columns].isna().all(axis=1)
        earliest.loc[all_missing_mask] = pd.NaT
        df_copy[new_column_name] = earliest
    
    # Optionally add source column indicating which column provided the earliest datetime
    if return_source_column:
        source_column_name = f'{new_column_name}_source'
        df_copy[source_column_name] = df_copy[datetime_columns].idxmin(axis=1)
    
    return df_copy

def find_earliest_datetime_advanced(df: pd.DataFrame,
                                  datetime_columns: List[str],
                                  new_column_name: str = 'earliest_datetime',
                                  weights: Optional[List[float]] = None,
                                  timezone_aware: bool = False,
                                  custom_logic: Optional[callable] = None) -> pd.DataFrame:
    """
    Advanced version with weights, timezone handling, and custom logic.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    datetime_columns (list): List of datetime column names
    new_column_name (str): Name for the new column
    weights (list, optional): Weights for tie-breaking when datetimes are equal
    timezone_aware (bool): Whether to handle timezone-aware datetimes
    custom_logic (callable, optional): Custom function for combining datetimes
    
    Returns:
    pd.DataFrame: DataFrame with new column
    """
    
    df_copy = df.copy()
    
    # Ensure datetime columns are properly formatted
    for col in datetime_columns:
        if not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = pd.to_datetime(df_copy[col])
    
    # Handle timezone awareness
    if timezone_aware:
        for col in datetime_columns:
            if df_copy[col].dt.tz is None:
                # Assume UTC if no timezone info
                df_copy[col] = df_copy[col].dt.tz_localize('UTC')
            else:
                # Convert all to UTC for comparison
                df_copy[col] = df_copy[col].dt.tz_convert('UTC')
    
    # Apply custom logic if provided
    if custom_logic:
        df_copy[new_column_name] = df_copy.apply(
            lambda row: custom_logic(row[datetime_columns]), axis=1
        )
    else:
        # Standard earliest datetime logic
        if weights:
            # Use weights for tie-breaking
            def weighted_earliest(row):
                valid_datetimes = row[datetime_columns].dropna()
                if valid_datetimes.empty:
                    return pd.NaT
                
                min_datetime = valid_datetimes.min()
                # Find all columns with the minimum datetime
                min_cols = valid_datetimes[valid_datetimes == min_datetime].index
                
                if len(min_cols) == 1:
                    return min_datetime
                else:
                    # Use weights to break ties
                    weighted_choice_idx = 0
                    max_weight = -1
                    for i, col in enumerate(min_cols):
                        col_idx = datetime_columns.index(col)
                        if col_idx < len(weights) and weights[col_idx] > max_weight:
                            max_weight = weights[col_idx]
                            weighted_choice_idx = i
                    return min_datetime
            
            df_copy[new_column_name] = df_copy.apply(weighted_earliest, axis=1)
        else:
            # Simple minimum
            df_copy[new_column_name] = df_copy[datetime_columns].min(axis=1)
    
    return df_copy

def combine_datetime_components(df: pd.DataFrame,
                              date_columns: List[str],
                              time_columns: List[str],
                              new_column_name: str = 'combined_datetime') -> pd.DataFrame:
    """
    Combine separate date and time columns into datetime and find earliest.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    date_columns (list): List of date column names
    time_columns (list): List of time column names
    new_column_name (str): Name for new datetime column
    
    Returns:
    pd.DataFrame: DataFrame with combined datetime column
    """
    
    df_copy = df.copy()
    datetime_columns = []
    
    # Combine date and time columns
    for i, (date_col, time_col) in enumerate(zip(date_columns, time_columns)):
        combined_col = f'datetime_combined_{i}'
        
        # Convert date and time to appropriate types
        df_copy[date_col] = pd.to_datetime(df_copy[date_col]).dt.date
        df_copy[time_col] = pd.to_datetime(df_copy[time_col], format='%H:%M:%S').dt.time
        
        # Combine date and time
        df_copy[combined_col] = pd.to_datetime(
            df_copy[date_col].astype(str) + ' ' + df_copy[time_col].astype(str)
        )
        datetime_columns.append(combined_col)
    
    # Find earliest combined datetime
    df_copy[new_column_name] = df_copy[datetime_columns].min(axis=1)
    
    # Clean up temporary columns
    df_copy = df_copy.drop(columns=datetime_columns)
    
    return df_copy

# Example usage and demonstration
def create_sample_datetime_data():
    """Create sample DataFrame with multiple datetime columns."""
    np.random.seed(42)
    n_rows = 1000
    
    # Base datetime
    base_date = datetime(2023, 1, 1)
    
    # Create multiple datetime columns with different patterns
    data = {
        'id': range(1, n_rows + 1),
        'created_at': [base_date + timedelta(days=x, hours=np.random.randint(0, 24)) 
                      for x in range(n_rows)],
        'updated_at': [base_date + timedelta(days=x, hours=np.random.randint(0, 24), minutes=30) 
                      for x in range(n_rows)],
        'published_at': [base_date + timedelta(days=x, hours=np.random.randint(0, 24), minutes=60) 
                        for x in range(n_rows)],
        'processed_at': [base_date + timedelta(days=x, hours=np.random.randint(0, 24), minutes=90) 
                        for x in range(n_rows)]
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    missing_indices = np.random.choice(df.index, size=100, replace=False)
    df.loc[missing_indices[:50], 'updated_at'] = pd.NaT
    df.loc[missing_indices[50:], 'published_at'] = pd.NaT
    
    return df

# Demonstration
if __name__ == "__main__":
    # Create sample data
    sample_df = create_sample_datetime_data()
    
    print("Sample DataFrame:")
    print(sample_df.head())
    print(f"\nDataFrame shape: {sample_df.shape}")
    print(f"Missing values per column:")
    print(sample_df.isnull().sum())
    
    # Find earliest datetime with different strategies
    datetime_cols = ['created_at', 'updated_at', 'published_at', 'processed_at']
    
    # Strategy 1: Ignore missing values
    result1 = find_earliest_datetime(
        sample_df, 
        datetime_cols, 
        'earliest_ignore_missing', 
        handle_missing='ignore',
        return_source_column=True
    )
    
    print("\n=== Strategy 1: Ignore Missing ===")
    print("First 5 rows with earliest datetime:")
    print(result1[['id'] + datetime_cols + ['earliest_ignore_missing', 'earliest_ignore_missing_source']].head())
    
    # Strategy 2: Skip rows with any missing values
    result2 = find_earliest_datetime(
        sample_df, 
        datetime_cols, 
        'earliest_skip_missing', 
        handle_missing='skip_row'
    )
    
    print(f"\n=== Strategy 2: Skip Missing ===")
    print(f"Rows with complete datetime data: {result2['earliest_skip_missing'].notna().sum()}")
    
    # Strategy 3: Advanced with weights
    weights = [1.0, 0.8, 0.6, 0.4]  # Prefer created_at in case of ties
    result3 = find_earliest_datetime_advanced(
        sample_df,
        datetime_cols,
        'earliest_weighted',
        weights=weights
    )
    
    print(f"\n=== Strategy 3: Weighted ===")
    print("First 5 rows with weighted earliest datetime:")
    print(result3[['id'] + datetime_cols + ['earliest_weighted']].head())
    
    # Custom logic example
    def custom_earliest_logic(datetime_series):
        """Custom logic: prefer created_at if within 1 hour of minimum."""
        valid_datetimes = datetime_series.dropna()
        if valid_datetimes.empty:
            return pd.NaT
        
        min_datetime = valid_datetimes.min()
        created_at = datetime_series.get('created_at')
        
        if pd.notna(created_at) and abs((created_at - min_datetime).total_seconds()) <= 3600:
            return created_at
        else:
            return min_datetime
    
    result4 = find_earliest_datetime_advanced(
        sample_df,
        datetime_cols,
        'earliest_custom',
        custom_logic=custom_earliest_logic
    )
    
    print(f"\n=== Strategy 4: Custom Logic ===")
    print("First 5 rows with custom logic:")
    print(result4[['id'] + datetime_cols + ['earliest_custom']].head())
    
    # Performance comparison
    print(f"\n=== Performance Summary ===")
    print(f"Original missing values: {sample_df[datetime_cols].isnull().sum().sum()}")
    print(f"Ignore missing result NaT count: {result1['earliest_ignore_missing'].isna().sum()}")
    print(f"Skip missing result NaT count: {result2['earliest_skip_missing'].isna().sum()}")
    print(f"Custom logic result NaT count: {result4['earliest_custom'].isna().sum()}")
```

#### Explanation
1. **Multiple Strategies**: Different approaches for handling missing values in datetime columns
2. **Type Validation**: Automatic conversion and validation of datetime columns
3. **Source Tracking**: Optional column to track which original column provided the earliest datetime
4. **Advanced Features**: Support for weights, timezone awareness, and custom logic
5. **Performance Optimization**: Vectorized operations for efficient processing

#### Use Cases
- **Event Timeline Analysis**: Finding first occurrence across multiple event types
- **Data Pipeline Timestamps**: Tracking earliest processing time across stages
- **User Activity Tracking**: Identifying first interaction across multiple channels
- **System Monitoring**: Finding earliest alert or error timestamp
- **Content Management**: Tracking earliest publication or modification time

#### Best Practices
- **Data Validation**: Always validate datetime column types before processing
- **Missing Value Strategy**: Choose appropriate strategy based on business logic
- **Timezone Handling**: Be explicit about timezone assumptions and conversions
- **Performance**: Use vectorized operations for large datasets
- **Documentation**: Clearly document the logic for combining datetimes

#### Pitfalls
- **Mixed Timezones**: Different timezone data can cause incorrect comparisons
- **Data Type Issues**: Non-datetime columns may cause unexpected results
- **Missing Value Logic**: Inconsistent handling can lead to biased results
- **Performance**: Row-wise operations can be slow on large datasets
- **Edge Cases**: All-missing rows need special consideration

#### Debugging
```python
def debug_datetime_combination(df: pd.DataFrame, datetime_columns: List[str]) -> None:
    """Debug datetime combination issues."""
    print("Datetime Column Analysis:")
    
    for col in datetime_columns:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Data type: {df[col].dtype}")
            print(f"  Missing values: {df[col].isnull().sum()}")
            print(f"  Unique values: {df[col].nunique()}")
            
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                print(f"  Date range: {df[col].min()} to {df[col].max()}")
                print(f"  Timezone info: {df[col].dt.tz}")
            else:
                print(f"  Sample values: {df[col].dropna().head(3).tolist()}")
        else:
            print(f"\n{col}: Column not found!")
    
    # Check for timezone inconsistencies
    tz_info = {}
    for col in datetime_columns:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            tz_info[col] = df[col].dt.tz
    
    unique_timezones = set(tz_info.values())
    if len(unique_timezones) > 1:
        print(f"\nWARNING: Mixed timezones detected: {tz_info}")
```

#### Optimization
```python
# Optimized version for large datasets
def find_earliest_datetime_optimized(df: pd.DataFrame, 
                                   datetime_columns: List[str],
                                   new_column_name: str = 'earliest_datetime') -> pd.DataFrame:
    """
    Memory and performance optimized version for large datasets.
    """
    # Use numpy operations for better performance
    datetime_arrays = []
    
    for col in datetime_columns:
        if col in df.columns:
            datetime_arrays.append(df[col].values)
    
    # Stack arrays and find minimum along axis
    stacked_arrays = np.column_stack(datetime_arrays)
    earliest_indices = np.nanargmin(stacked_arrays, axis=1)
    
    # Create result using advanced indexing
    result = df.copy()
    earliest_values = np.array([stacked_arrays[i, idx] if not np.isnan(stacked_arrays[i, idx]) else pd.NaT 
                               for i, idx in enumerate(earliest_indices)])
    
    result[new_column_name] = pd.to_datetime(earliest_values)
    
    return result
```

---

## Question 7

**Develop a routine in Pandas to detect and flag rows that deviate by more than three standard deviations from the mean of specific columns.**

### Answer

#### Theory
Statistical outlier detection using standard deviations is based on the assumption that data follows a normal distribution. In a normal distribution, approximately 99.7% of values lie within three standard deviations of the mean. Values beyond this threshold are considered statistical outliers and may indicate data quality issues, measurement errors, or genuinely unusual observations that require special attention.

#### Code Example
```python
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class StandardDeviationOutlierDetector:
    """
    Comprehensive toolkit for detecting rows with values that deviate 
    by more than specified standard deviations from the mean.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.outlier_results = {}
        self.column_stats = {}
    
    def calculate_column_statistics(self, columns: List[str]) -> Dict:
        """
        Calculate mean and standard deviation for specified columns.
        
        Parameters:
        columns (list): List of column names
        
        Returns:
        dict: Statistics for each column
        """
        stats_dict = {}
        
        for column in columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                raise ValueError(f"Column '{column}' must be numeric")
            
            col_data = self.df[column].dropna()
            
            stats_dict[column] = {
                'mean': col_data.mean(),
                'std': col_data.std(),
                'count': len(col_data),
                'missing_count': self.df[column].isnull().sum(),
                'min': col_data.min(),
                'max': col_data.max(),
                'median': col_data.median(),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis()
            }
        
        self.column_stats.update(stats_dict)
        return stats_dict
    
    def detect_outlier_rows(self, columns: List[str], 
                           threshold: float = 3.0,
                           method: str = 'any',
                           return_details: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Detect rows with values deviating by more than threshold standard deviations.
        
        Parameters:
        columns (list): List of numeric columns to check
        threshold (float): Number of standard deviations as threshold
        method (str): 'any' (outlier in any column) or 'all' (outlier in all columns)
        return_details (bool): Whether to return detailed outlier information
        
        Returns:
        pd.Series or tuple: Boolean series indicating outlier rows, optionally with details
        """
        # Calculate statistics if not already done
        if not any(col in self.column_stats for col in columns):
            self.calculate_column_statistics(columns)
        
        # Calculate z-scores for each column
        z_scores_df = pd.DataFrame(index=self.df.index)
        outlier_flags_df = pd.DataFrame(index=self.df.index)
        
        for column in columns:
            if column not in self.column_stats:
                self.calculate_column_statistics([column])
            
            stats = self.column_stats[column]
            
            # Calculate z-scores
            z_scores = (self.df[column] - stats['mean']) / stats['std']
            z_scores_df[f'{column}_zscore'] = z_scores
            
            # Flag outliers
            outlier_flags_df[column] = np.abs(z_scores) > threshold
        
        # Combine outlier flags based on method
        if method == 'any':
            # Row is outlier if ANY column has outlier
            outlier_rows = outlier_flags_df[columns].any(axis=1)
        elif method == 'all':
            # Row is outlier if ALL columns have outliers
            outlier_rows = outlier_flags_df[columns].all(axis=1)
        else:
            raise ValueError("Method must be 'any' or 'all'")
        
        # Store results
        result_key = f"{'_'.join(columns)}_{method}_{threshold}"
        self.outlier_results[result_key] = {
            'columns': columns,
            'threshold': threshold,
            'method': method,
            'outlier_count': outlier_rows.sum(),
            'outlier_percentage': (outlier_rows.sum() / len(outlier_rows)) * 100,
            'z_scores': z_scores_df,
            'outlier_flags': outlier_flags_df
        }
        
        if return_details:
            # Create detailed information about outliers
            details_df = self.df[outlier_rows].copy()
            
            # Add z-scores and deviation information
            for column in columns:
                details_df[f'{column}_zscore'] = z_scores_df.loc[outlier_rows, f'{column}_zscore']
                details_df[f'{column}_deviation'] = np.abs(z_scores_df.loc[outlier_rows, f'{column}_zscore'])
            
            return outlier_rows, details_df
        
        return outlier_rows
    
    def detect_outlier_rows_robust(self, columns: List[str],
                                  threshold: float = 3.0,
                                  use_median: bool = True,
                                  method: str = 'any') -> pd.Series:
        """
        Robust outlier detection using median and MAD (Median Absolute Deviation).
        
        Parameters:
        columns (list): List of numeric columns
        threshold (float): Threshold for modified z-score
        use_median (bool): Use median and MAD instead of mean and std
        method (str): 'any' or 'all'
        
        Returns:
        pd.Series: Boolean series indicating outlier rows
        """
        outlier_flags_df = pd.DataFrame(index=self.df.index)
        
        for column in columns:
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                raise ValueError(f"Column '{column}' must be numeric")
            
            col_data = self.df[column]
            
            if use_median:
                # Use median and MAD for robust detection
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                
                # Avoid division by zero
                if mad == 0:
                    mad = 1.4826 * np.median(np.abs(col_data - median))
                
                # Calculate modified z-scores
                modified_z_scores = 0.6745 * (col_data - median) / mad
                outlier_flags_df[column] = np.abs(modified_z_scores) > threshold
            else:
                # Use mean and std (same as regular method)
                mean = col_data.mean()
                std = col_data.std()
                z_scores = (col_data - mean) / std
                outlier_flags_df[column] = np.abs(z_scores) > threshold
        
        # Combine flags
        if method == 'any':
            outlier_rows = outlier_flags_df[columns].any(axis=1)
        else:
            outlier_rows = outlier_flags_df[columns].all(axis=1)
        
        return outlier_rows
    
    def flag_outlier_rows_with_scores(self, columns: List[str],
                                     threshold: float = 3.0,
                                     flag_column: str = 'outlier_flag') -> pd.DataFrame:
        """
        Add outlier flag and scores directly to the DataFrame.
        
        Parameters:
        columns (list): List of columns to check
        threshold (float): Standard deviation threshold
        flag_column (str): Name for the outlier flag column
        
        Returns:
        pd.DataFrame: DataFrame with outlier flags and scores added
        """
        result_df = self.df.copy()
        
        # Calculate statistics
        self.calculate_column_statistics(columns)
        
        # Calculate z-scores and flags for each column
        for column in columns:
            stats = self.column_stats[column]
            z_scores = (self.df[column] - stats['mean']) / stats['std']
            
            # Add z-score column
            result_df[f'{column}_zscore'] = z_scores
            
            # Add individual column outlier flag
            result_df[f'{column}_outlier'] = np.abs(z_scores) > threshold
        
        # Add overall outlier flag (outlier in any column)
        outlier_cols = [f'{col}_outlier' for col in columns]
        result_df[flag_column] = result_df[outlier_cols].any(axis=1)
        
        # Add maximum absolute z-score
        zscore_cols = [f'{col}_zscore' for col in columns]
        result_df['max_abs_zscore'] = result_df[zscore_cols].abs().max(axis=1)
        
        return result_df
    
    def analyze_outlier_patterns(self, columns: List[str], threshold: float = 3.0) -> Dict:
        """
        Comprehensive analysis of outlier patterns across columns.
        
        Parameters:
        columns (list): List of columns to analyze
        threshold (float): Standard deviation threshold
        
        Returns:
        dict: Detailed analysis results
        """
        analysis = {}
        
        # Get outlier information
        outlier_rows, details_df = self.detect_outlier_rows(
            columns, threshold=threshold, return_details=True
        )
        
        analysis['summary'] = {
            'total_rows': len(self.df),
            'outlier_rows': outlier_rows.sum(),
            'outlier_percentage': (outlier_rows.sum() / len(self.df)) * 100,
            'columns_analyzed': columns,
            'threshold': threshold
        }
        
        # Per-column analysis
        analysis['per_column'] = {}
        for column in columns:
            col_outliers = self.outlier_results[f"{'_'.join(columns)}_any_{threshold}"]['outlier_flags'][column]
            
            analysis['per_column'][column] = {
                'outlier_count': col_outliers.sum(),
                'outlier_percentage': (col_outliers.sum() / len(col_outliers)) * 100,
                'mean': self.column_stats[column]['mean'],
                'std': self.column_stats[column]['std'],
                'outlier_values': self.df.loc[col_outliers, column].tolist()
            }
        
        # Pattern analysis
        if len(columns) > 1:
            outlier_flags_df = self.outlier_results[f"{'_'.join(columns)}_any_{threshold}"]['outlier_flags']
            
            # Count outliers per row
            outliers_per_row = outlier_flags_df[columns].sum(axis=1)
            
            analysis['patterns'] = {
                'single_column_outliers': (outliers_per_row == 1).sum(),
                'multiple_column_outliers': (outliers_per_row > 1).sum(),
                'max_outliers_per_row': outliers_per_row.max(),
                'outliers_per_row_distribution': outliers_per_row.value_counts().to_dict()
            }
            
            # Correlation between outlier occurrences
            outlier_correlation = outlier_flags_df[columns].corr()
            analysis['outlier_correlation'] = outlier_correlation.to_dict()
        
        return analysis
    
    def visualize_outliers(self, columns: List[str], threshold: float = 3.0,
                          figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Visualize outlier detection results.
        
        Parameters:
        columns (list): List of columns to visualize
        threshold (float): Standard deviation threshold
        figsize (tuple): Figure size
        """
        outlier_rows = self.detect_outlier_rows(columns, threshold=threshold)
        
        n_cols = len(columns)
        n_rows = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Outlier Detection (±{threshold} Standard Deviations)', fontsize=16)
        
        for i, column in enumerate(columns):
            # Calculate z-scores
            stats = self.column_stats[column]
            z_scores = (self.df[column] - stats['mean']) / stats['std']
            
            # Top row: Histogram with outliers highlighted
            ax1 = axes[0, i]
            ax1.hist(self.df[column].dropna(), bins=50, alpha=0.7, label='All data')
            
            # Highlight outliers
            outlier_data = self.df.loc[outlier_rows, column].dropna()
            if len(outlier_data) > 0:
                ax1.hist(outlier_data, bins=20, alpha=0.8, color='red', label='Outliers')
            
            ax1.axvline(stats['mean'], color='green', linestyle='--', label='Mean')
            ax1.axvline(stats['mean'] + threshold * stats['std'], color='orange', 
                       linestyle='--', label=f'±{threshold}σ')
            ax1.axvline(stats['mean'] - threshold * stats['std'], color='orange', linestyle='--')
            
            ax1.set_title(f'{column} - Distribution')
            ax1.set_xlabel(column)
            ax1.set_ylabel('Frequency')
            ax1.legend()
            
            # Bottom row: Z-scores
            ax2 = axes[1, i]
            ax2.scatter(self.df.index, z_scores, alpha=0.6, s=20)
            ax2.axhline(threshold, color='red', linestyle='--', label=f'±{threshold}σ threshold')
            ax2.axhline(-threshold, color='red', linestyle='--')
            ax2.axhline(0, color='green', linestyle='-', alpha=0.5, label='Mean')
            
            # Highlight outlier points
            outlier_indices = self.df.index[outlier_rows]
            outlier_zscores = z_scores[outlier_rows]
            ax2.scatter(outlier_indices, outlier_zscores, color='red', s=40, alpha=0.8)
            
            ax2.set_title(f'{column} - Z-scores')
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Z-score')
            ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary of all outlier detection results.
        
        Returns:
        pd.DataFrame: Summary DataFrame
        """
        if not self.outlier_results:
            return pd.DataFrame()
        
        summary_data = []
        for key, info in self.outlier_results.items():
            summary_data.append({
                'Analysis': key,
                'Columns': ', '.join(info['columns']),
                'Threshold': info['threshold'],
                'Method': info['method'],
                'Outlier_Count': info['outlier_count'],
                'Outlier_Percentage': round(info['outlier_percentage'], 2)
            })
        
        return pd.DataFrame(summary_data)

def detect_multivariate_outliers(df: pd.DataFrame, columns: List[str],
                                threshold: float = 3.0) -> pd.Series:
    """
    Detect multivariate outliers using Mahalanobis distance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of numeric columns
    threshold (float): Threshold for Mahalanobis distance
    
    Returns:
    pd.Series: Boolean series indicating outlier rows
    """
    # Extract numeric data
    data = df[columns].dropna()
    
    if len(data) == 0:
        return pd.Series(False, index=df.index)
    
    # Calculate mean and covariance matrix
    mean = data.mean()
    cov_matrix = data.cov()
    
    # Calculate Mahalanobis distance
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        
        mahal_distances = []
        for _, row in data.iterrows():
            diff = row - mean
            mahal_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
            mahal_distances.append(mahal_dist)
        
        mahal_distances = np.array(mahal_distances)
        
        # Detect outliers
        outliers = pd.Series(False, index=df.index)
        outliers.loc[data.index] = mahal_distances > threshold
        
    except np.linalg.LinAlgError:
        # Fallback to univariate detection if covariance matrix is singular
        print("Warning: Singular covariance matrix, falling back to univariate detection")
        detector = StandardDeviationOutlierDetector(df)
        outliers = detector.detect_outlier_rows(columns, threshold=threshold)
    
    return outliers

# Example usage and demonstration
def create_sample_data_with_outliers():
    """Create sample DataFrame with outliers for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate correlated variables
    mean = [50, 100, 200]
    cov = [[100, 30, 50], [30, 150, 60], [50, 60, 200]]
    data = np.random.multivariate_normal(mean, cov, n_samples)
    
    df = pd.DataFrame(data, columns=['variable1', 'variable2', 'variable3'])
    
    # Add some extreme outliers
    outlier_indices = np.random.choice(n_samples, 50, replace=False)
    
    # Make some extreme outliers
    df.loc[outlier_indices[:20], 'variable1'] += 200  # Far from mean
    df.loc[outlier_indices[20:35], 'variable2'] -= 300
    df.loc[outlier_indices[35:], 'variable3'] += 400
    
    # Add some moderate outliers
    moderate_indices = np.random.choice(n_samples, 30, replace=False)
    df.loc[moderate_indices, 'variable1'] += np.random.normal(0, 100, 30)
    
    # Add categorical column for context
    df['category'] = np.random.choice(['A', 'B', 'C'], n_samples)
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, 20, replace=False)
    df.loc[missing_indices[:10], 'variable2'] = np.nan
    df.loc[missing_indices[10:], 'variable3'] = np.nan
    
    return df

# Demonstration
if __name__ == "__main__":
    # Create sample data
    sample_df = create_sample_data_with_outliers()
    
    print("Sample DataFrame with Outliers:")
    print(sample_df.describe())
    
    # Initialize detector
    detector = StandardDeviationOutlierDetector(sample_df)
    
    # Analyze specific columns
    columns_to_analyze = ['variable1', 'variable2', 'variable3']
    
    print(f"\n=== Column Statistics ===")
    stats = detector.calculate_column_statistics(columns_to_analyze)
    for col, stat in stats.items():
        print(f"{col}: mean={stat['mean']:.2f}, std={stat['std']:.2f}, skew={stat['skewness']:.2f}")
    
    # Detect outliers with different methods
    print(f"\n=== Outlier Detection Results ===")
    
    # Method 1: Any column outlier (3 standard deviations)
    outliers_any = detector.detect_outlier_rows(columns_to_analyze, threshold=3.0, method='any')
    print(f"Rows with outliers (any column, 3σ): {outliers_any.sum()} ({outliers_any.sum()/len(outliers_any)*100:.1f}%)")
    
    # Method 2: All columns outliers
    outliers_all = detector.detect_outlier_rows(columns_to_analyze, threshold=3.0, method='all')
    print(f"Rows with outliers (all columns, 3σ): {outliers_all.sum()} ({outliers_all.sum()/len(outliers_all)*100:.1f}%)")
    
    # Method 3: More conservative threshold
    outliers_conservative = detector.detect_outlier_rows(columns_to_analyze, threshold=2.5, method='any')
    print(f"Rows with outliers (any column, 2.5σ): {outliers_conservative.sum()} ({outliers_conservative.sum()/len(outliers_conservative)*100:.1f}%)")
    
    # Method 4: Robust detection
    outliers_robust = detector.detect_outlier_rows_robust(columns_to_analyze, threshold=3.0, use_median=True)
    print(f"Rows with outliers (robust method): {outliers_robust.sum()} ({outliers_robust.sum()/len(outliers_robust)*100:.1f}%)")
    
    # Detailed analysis
    print(f"\n=== Detailed Analysis ===")
    outliers_with_details, details_df = detector.detect_outlier_rows(
        columns_to_analyze, threshold=3.0, return_details=True
    )
    
    print("Sample outlier rows with z-scores:")
    print(details_df[['variable1', 'variable2', 'variable3', 
                     'variable1_zscore', 'variable2_zscore', 'variable3_zscore']].head())
    
    # Pattern analysis
    patterns = detector.analyze_outlier_patterns(columns_to_analyze, threshold=3.0)
    print(f"\n=== Pattern Analysis ===")
    print(f"Total outlier rows: {patterns['summary']['outlier_rows']}")
    print(f"Single-column outliers: {patterns['patterns']['single_column_outliers']}")
    print(f"Multi-column outliers: {patterns['patterns']['multiple_column_outliers']}")
    
    # Flag outliers in DataFrame
    flagged_df = detector.flag_outlier_rows_with_scores(columns_to_analyze, threshold=3.0)
    print(f"\nDataFrame with outlier flags shape: {flagged_df.shape}")
    print("Columns added:", [col for col in flagged_df.columns if col not in sample_df.columns])
    
    # Multivariate outlier detection
    multivariate_outliers = detect_multivariate_outliers(sample_df, columns_to_analyze, threshold=3.0)
    print(f"Multivariate outliers: {multivariate_outliers.sum()} ({multivariate_outliers.sum()/len(multivariate_outliers)*100:.1f}%)")
    
    # Summary of all results
    summary = detector.get_outlier_summary()
    print(f"\n=== Summary of All Analyses ===")
    print(summary)
    
    # Visualize results (uncomment to show plots)
    # detector.visualize_outliers(columns_to_analyze, threshold=3.0)
```

#### Explanation
1. **Statistical Foundation**: Uses z-score calculation to identify values beyond specified standard deviations
2. **Multiple Detection Methods**: Supports 'any' or 'all' column criteria for row-level outlier flagging
3. **Robust Detection**: Includes median-based methods for non-normal distributions
4. **Comprehensive Analysis**: Provides detailed statistics and pattern analysis
5. **Multivariate Support**: Includes Mahalanobis distance for multivariate outlier detection

#### Use Cases
- **Data Quality Control**: Identifying rows with suspicious values across multiple measurements
- **Sensor Data Validation**: Flagging sensor readings that deviate significantly from normal ranges
- **Financial Analysis**: Detecting unusual patterns in financial metrics or ratios
- **Scientific Research**: Identifying experimental results that may need investigation
- **Customer Analytics**: Finding customers with unusual behavior patterns

#### Best Practices
- **Distribution Check**: Verify data follows normal distribution before using z-score method
- **Threshold Selection**: Choose appropriate threshold based on domain knowledge and data characteristics
- **Multiple Methods**: Use both univariate and multivariate approaches for comprehensive detection
- **Documentation**: Record outlier detection criteria and decisions for reproducibility
- **Domain Context**: Consider business or scientific context when interpreting outliers

#### Pitfalls
- **Normal Distribution Assumption**: Z-score method assumes normal distribution
- **Threshold Sensitivity**: Different thresholds can dramatically change results
- **Missing Data**: Missing values need careful handling in calculations
- **Correlated Variables**: High correlation between variables can affect multivariate detection
- **Small Sample Size**: Statistical measures become unreliable with small datasets

#### Debugging
```python
def debug_outlier_detection_routine(df: pd.DataFrame, columns: List[str]) -> None:
    """Debug outlier detection issues."""
    print("Debugging Standard Deviation Outlier Detection")
    
    for column in columns:
        if column not in df.columns:
            print(f"ERROR: Column '{column}' not found!")
            continue
        
        col_data = df[column]
        print(f"\n{column}:")
        print(f"  Data type: {col_data.dtype}")
        print(f"  Shape: {col_data.shape}")
        print(f"  Missing values: {col_data.isnull().sum()}")
        
        if pd.api.types.is_numeric_dtype(col_data):
            valid_data = col_data.dropna()
            print(f"  Valid values: {len(valid_data)}")
            print(f"  Mean: {valid_data.mean():.2f}")
            print(f"  Std: {valid_data.std():.2f}")
            print(f"  Range: {valid_data.min():.2f} to {valid_data.max():.2f}")
            
            # Check for zero standard deviation
            if valid_data.std() == 0:
                print(f"  WARNING: Zero standard deviation - all values are identical!")
            
            # Check for infinite values
            inf_count = np.isinf(valid_data).sum()
            if inf_count > 0:
                print(f"  WARNING: {inf_count} infinite values detected!")
            
            # Normality test
            if len(valid_data) > 5:
                statistic, p_value = stats.shapiro(valid_data.sample(min(5000, len(valid_data))))
                print(f"  Normality test p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print(f"  WARNING: Data may not be normally distributed!")
        else:
            print(f"  ERROR: Column is not numeric!")
```

#### Optimization
```python
# Optimized version for large datasets
def detect_outliers_vectorized(df: pd.DataFrame, columns: List[str],
                             threshold: float = 3.0, chunk_size: int = 10000) -> pd.Series:
    """
    Vectorized outlier detection for large datasets.
    """
    # Pre-calculate statistics
    means = df[columns].mean()
    stds = df[columns].std()
    
    outlier_mask = pd.Series(False, index=df.index)
    
    # Process in chunks for memory efficiency
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        
        # Vectorized z-score calculation
        z_scores = (chunk[columns] - means) / stds
        
        # Check if any column exceeds threshold
        chunk_outliers = (np.abs(z_scores) > threshold).any(axis=1)
        outlier_mask.iloc[i:i+chunk_size] = chunk_outliers
    
    return outlier_mask
```

---

## Question 8

**Outline how to merge multiple time series datasets effectively in Pandas, ensuring correct alignment and handling missing values.**

### Answer

#### Theory
Merging time series datasets requires careful consideration of temporal alignment, different sampling frequencies, missing data patterns, and time zone handling. Effective time series merging ensures that temporal relationships are preserved while handling data quality issues like gaps, overlaps, and misaligned timestamps. The process involves temporal indexing, resampling, interpolation strategies, and robust handling of missing values.

#### Code Example
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Union, Optional, Tuple
import warnings
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns

class TimeSeriesMerger:
    """
    Comprehensive toolkit for merging multiple time series datasets with 
    proper alignment and missing value handling.
    """
    
    def __init__(self, timezone: str = 'UTC'):
        """
        Initialize the time series merger.
        
        Parameters:
        timezone (str): Default timezone for time series operations
        """
        self.timezone = timezone
        self.merged_data = None
        self.merge_info = {}
        self.original_datasets = {}
    
    def prepare_time_series(self, df: pd.DataFrame, 
                           datetime_column: str,
                           value_columns: List[str] = None,
                           freq: str = None,
                           timezone: str = None) -> pd.DataFrame:
        """
        Prepare a DataFrame for time series merging.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        datetime_column (str): Name of datetime column
        value_columns (list): List of value columns to keep
        freq (str): Target frequency for resampling
        timezone (str): Timezone to convert to
        
        Returns:
        pd.DataFrame: Prepared time series DataFrame
        """
        # Create a copy to avoid modifying original
        ts_df = df.copy()
        
        # Ensure datetime column is properly formatted
        if not pd.api.types.is_datetime64_any_dtype(ts_df[datetime_column]):
            ts_df[datetime_column] = pd.to_datetime(ts_df[datetime_column])
        
        # Handle timezone
        tz = timezone or self.timezone
        if ts_df[datetime_column].dt.tz is None:
            ts_df[datetime_column] = ts_df[datetime_column].dt.tz_localize(tz)
        else:
            ts_df[datetime_column] = ts_df[datetime_column].dt.tz_convert(tz)
        
        # Set datetime as index
        ts_df = ts_df.set_index(datetime_column)
        
        # Select value columns if specified
        if value_columns:
            available_columns = [col for col in value_columns if col in ts_df.columns]
            if len(available_columns) != len(value_columns):
                missing = set(value_columns) - set(available_columns)
                warnings.warn(f"Columns not found: {missing}")
            ts_df = ts_df[available_columns]
        
        # Remove duplicated timestamps (keep last)
        if ts_df.index.duplicated().any():
            warnings.warn("Duplicate timestamps found, keeping last occurrence")
            ts_df = ts_df[~ts_df.index.duplicated(keep='last')]
        
        # Sort by index
        ts_df = ts_df.sort_index()
        
        # Resample if frequency is specified
        if freq:
            ts_df = self._resample_series(ts_df, freq)
        
        return ts_df
    
    def _resample_series(self, df: pd.DataFrame, freq: str, 
                        agg_method: str = 'mean') -> pd.DataFrame:
        """
        Resample time series to specified frequency.
        
        Parameters:
        df (pd.DataFrame): Time series DataFrame
        freq (str): Target frequency
        agg_method (str): Aggregation method for downsampling
        
        Returns:
        pd.DataFrame: Resampled DataFrame
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        
        resampled_parts = []
        
        # Handle numeric columns
        if len(numeric_columns) > 0:
            if agg_method == 'mean':
                numeric_resampled = df[numeric_columns].resample(freq).mean()
            elif agg_method == 'sum':
                numeric_resampled = df[numeric_columns].resample(freq).sum()
            elif agg_method == 'last':
                numeric_resampled = df[numeric_columns].resample(freq).last()
            elif agg_method == 'first':
                numeric_resampled = df[numeric_columns].resample(freq).first()
            else:
                numeric_resampled = df[numeric_columns].resample(freq).mean()
            
            resampled_parts.append(numeric_resampled)
        
        # Handle non-numeric columns
        if len(non_numeric_columns) > 0:
            non_numeric_resampled = df[non_numeric_columns].resample(freq).last()
            resampled_parts.append(non_numeric_resampled)
        
        if resampled_parts:
            return pd.concat(resampled_parts, axis=1)
        else:
            return df.resample(freq).mean()  # Fallback
    
    def merge_time_series_datasets(self, datasets: List[pd.DataFrame],
                                  dataset_names: List[str] = None,
                                  merge_method: str = 'outer',
                                  column_suffixes: List[str] = None,
                                  common_frequency: str = None,
                                  fill_method: str = 'none') -> pd.DataFrame:
        """
        Merge multiple time series datasets.
        
        Parameters:
        datasets (list): List of prepared time series DataFrames
        dataset_names (list): Names for each dataset
        merge_method (str): Merge method ('outer', 'inner', 'left', 'right')
        column_suffixes (list): Suffixes for columns from each dataset
        common_frequency (str): Common frequency to resample all datasets
        fill_method (str): Method to handle missing values after merge
        
        Returns:
        pd.DataFrame: Merged time series DataFrame
        """
        if not datasets:
            raise ValueError("No datasets provided")
        
        # Generate default names and suffixes if not provided
        n_datasets = len(datasets)
        if dataset_names is None:
            dataset_names = [f'dataset_{i+1}' for i in range(n_datasets)]
        
        if column_suffixes is None:
            column_suffixes = [f'_{name}' for name in dataset_names]
        
        # Store original datasets for reference
        for i, (df, name) in enumerate(zip(datasets, dataset_names)):
            self.original_datasets[name] = df.copy()
        
        # Resample all datasets to common frequency if specified
        if common_frequency:
            datasets = [self._resample_series(df, common_frequency) for df in datasets]
        
        # Add suffixes to column names to avoid conflicts
        renamed_datasets = []
        for i, (df, suffix) in enumerate(zip(datasets, column_suffixes)):
            df_renamed = df.copy()
            df_renamed.columns = [f'{col}{suffix}' for col in df.columns]
            renamed_datasets.append(df_renamed)
        
        # Perform the merge
        if merge_method == 'outer':
            # Outer join - include all timestamps
            merged = pd.concat(renamed_datasets, axis=1, join='outer', sort=True)
        elif merge_method == 'inner':
            # Inner join - only common timestamps
            merged = renamed_datasets[0]
            for df in renamed_datasets[1:]:
                merged = merged.join(df, how='inner')
        elif merge_method == 'left':
            # Left join - use first dataset's timestamps
            merged = renamed_datasets[0]
            for df in renamed_datasets[1:]:
                merged = merged.join(df, how='left')
        elif merge_method == 'right':
            # Right join - use last dataset's timestamps
            merged = renamed_datasets[-1]
            for df in reversed(renamed_datasets[:-1]):
                merged = df.join(merged, how='right')
        else:
            raise ValueError(f"Unknown merge method: {merge_method}")
        
        # Handle missing values according to specified method
        if fill_method != 'none':
            merged = self._handle_missing_values(merged, fill_method)
        
        # Store merge information
        self.merge_info = {
            'dataset_names': dataset_names,
            'merge_method': merge_method,
            'common_frequency': common_frequency,
            'fill_method': fill_method,
            'original_shapes': {name: df.shape for name, df in zip(dataset_names, datasets)},
            'merged_shape': merged.shape,
            'date_range': (merged.index.min(), merged.index.max()),
            'missing_percentage': (merged.isnull().sum() / len(merged) * 100).to_dict()
        }
        
        self.merged_data = merged
        return merged
    
    def _handle_missing_values(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Handle missing values in merged time series.
        
        Parameters:
        df (pd.DataFrame): DataFrame with missing values
        method (str): Method to handle missing values
        
        Returns:
        pd.DataFrame: DataFrame with missing values handled
        """
        if method == 'forward_fill':
            return df.fillna(method='ffill')
        elif method == 'backward_fill':
            return df.fillna(method='bfill')
        elif method == 'interpolate_linear':
            return df.interpolate(method='linear')
        elif method == 'interpolate_time':
            return df.interpolate(method='time')
        elif method == 'interpolate_spline':
            return df.interpolate(method='spline', order=2)
        elif method == 'drop_missing':
            return df.dropna()
        elif method == 'fill_zero':
            return df.fillna(0)
        elif method == 'fill_mean':
            return df.fillna(df.mean())
        elif method == 'combined':
            # Combined approach: forward fill, then interpolate, then backward fill
            return df.fillna(method='ffill').interpolate().fillna(method='bfill')
        else:
            return df
    
    def merge_with_alignment(self, datasets: Dict[str, pd.DataFrame],
                           target_frequency: str = '1min',
                           alignment_method: str = 'nearest',
                           tolerance: str = '30s') -> pd.DataFrame:
        """
        Advanced merge with precise temporal alignment.
        
        Parameters:
        datasets (dict): Dictionary of dataset_name -> DataFrame
        target_frequency (str): Target frequency for alignment
        alignment_method (str): Method for timestamp alignment
        tolerance (str): Maximum tolerance for alignment
        
        Returns:
        pd.DataFrame: Precisely aligned merged DataFrame
        """
        # Create target time index
        all_start_times = [df.index.min() for df in datasets.values()]
        all_end_times = [df.index.max() for df in datasets.values()]
        
        start_time = min(all_start_times)
        end_time = max(all_end_times)
        
        target_index = pd.date_range(start=start_time, end=end_time, freq=target_frequency)
        
        aligned_datasets = {}
        
        for name, df in datasets.items():
            if alignment_method == 'nearest':
                # Use nearest neighbor alignment
                aligned = df.reindex(target_index, method='nearest', tolerance=pd.Timedelta(tolerance))
            elif alignment_method == 'linear':
                # Use linear interpolation
                aligned = df.reindex(target_index).interpolate(method='linear')
            elif alignment_method == 'forward':
                # Forward fill
                aligned = df.reindex(target_index, method='ffill')
            elif alignment_method == 'backward':
                # Backward fill
                aligned = df.reindex(target_index, method='bfill')
            else:
                aligned = df.reindex(target_index)
            
            # Add suffix to column names
            aligned.columns = [f'{col}_{name}' for col in aligned.columns]
            aligned_datasets[name] = aligned
        
        # Combine all aligned datasets
        merged = pd.concat(aligned_datasets.values(), axis=1)
        
        return merged
    
    def merge_asof(self, left_df: pd.DataFrame, right_df: pd.DataFrame,
                  tolerance: str = '1min', direction: str = 'nearest') -> pd.DataFrame:
        """
        Perform as-of merge for time series with different timestamps.
        
        Parameters:
        left_df (pd.DataFrame): Left DataFrame (must have DatetimeIndex)
        right_df (pd.DataFrame): Right DataFrame (must have DatetimeIndex)
        tolerance (str): Maximum time difference for matching
        direction (str): Direction for matching ('backward', 'forward', 'nearest')
        
        Returns:
        pd.DataFrame: Merged DataFrame using as-of join
        """
        # Reset index to use merge_asof
        left_reset = left_df.reset_index()
        right_reset = right_df.reset_index()
        
        # Add suffixes to avoid column conflicts
        right_reset.columns = [f'{col}_right' if col != right_reset.columns[0] else col 
                              for col in right_reset.columns]
        
        # Perform as-of merge
        merged = pd.merge_asof(
            left_reset, 
            right_reset,
            left_index=True, 
            right_index=True,
            tolerance=pd.Timedelta(tolerance),
            direction=direction
        )
        
        # Set index back to datetime
        merged = merged.set_index(left_reset.columns[0])
        
        return merged
    
    def analyze_merge_quality(self) -> Dict:
        """
        Analyze the quality of the merged time series.
        
        Returns:
        dict: Quality analysis results
        """
        if self.merged_data is None:
            raise ValueError("No merged data available. Perform merge first.")
        
        analysis = {}
        df = self.merged_data
        
        # Basic statistics
        analysis['basic_stats'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_range': (df.index.min(), df.index.max()),
            'frequency_analysis': self._analyze_frequency(df.index)
        }
        
        # Missing value analysis
        missing_info = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_info[col] = {
                'missing_count': missing_count,
                'missing_percentage': (missing_count / len(df)) * 100,
                'consecutive_missing': self._find_consecutive_missing(df[col])
            }
        
        analysis['missing_values'] = missing_info
        
        # Gap analysis
        analysis['gaps'] = self._analyze_gaps(df.index)
        
        # Overlap analysis for original datasets
        if self.original_datasets:
            analysis['overlap_analysis'] = self._analyze_overlaps()
        
        return analysis
    
    def _analyze_frequency(self, index: pd.DatetimeIndex) -> Dict:
        """Analyze the frequency pattern of the datetime index."""
        if len(index) < 2:
            return {'frequency': 'insufficient_data'}
        
        # Calculate time differences
        diffs = index.to_series().diff().dropna()
        
        # Find most common frequency
        mode_freq = diffs.mode()
        
        return {
            'inferred_frequency': pd.infer_freq(index),
            'most_common_interval': mode_freq.iloc[0] if len(mode_freq) > 0 else None,
            'min_interval': diffs.min(),
            'max_interval': diffs.max(),
            'frequency_std': diffs.std()
        }
    
    def _find_consecutive_missing(self, series: pd.Series) -> Dict:
        """Find consecutive missing value patterns."""
        is_missing = series.isnull()
        
        # Find consecutive groups
        groups = (is_missing != is_missing.shift()).cumsum()
        missing_groups = is_missing.groupby(groups).apply(lambda x: x.all() and len(x))
        consecutive_missing = missing_groups[missing_groups > 0]
        
        if len(consecutive_missing) == 0:
            return {'max_consecutive': 0, 'count_gaps': 0}
        
        return {
            'max_consecutive': consecutive_missing.max(),
            'count_gaps': len(consecutive_missing),
            'gap_lengths': consecutive_missing.tolist()
        }
    
    def _analyze_gaps(self, index: pd.DatetimeIndex) -> Dict:
        """Analyze gaps in the time series index."""
        if len(index) < 2:
            return {'gaps': []}
        
        # Expected frequency based on most common interval
        diffs = index.to_series().diff().dropna()
        expected_freq = diffs.mode().iloc[0] if len(diffs.mode()) > 0 else diffs.median()
        
        # Find gaps larger than expected
        large_gaps = diffs[diffs > expected_freq * 1.5]
        
        gaps = []
        for timestamp, gap_size in large_gaps.items():
            gaps.append({
                'start_time': timestamp - gap_size,
                'end_time': timestamp,
                'gap_duration': gap_size,
                'expected_points': int(gap_size / expected_freq) - 1
            })
        
        return {
            'gap_count': len(gaps),
            'total_gap_duration': sum(gap['gap_duration'] for gap in gaps),
            'gaps': gaps
        }
    
    def _analyze_overlaps(self) -> Dict:
        """Analyze temporal overlaps between original datasets."""
        overlaps = {}
        dataset_names = list(self.original_datasets.keys())
        
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                df1 = self.original_datasets[name1]
                df2 = self.original_datasets[name2]
                
                start1, end1 = df1.index.min(), df1.index.max()
                start2, end2 = df2.index.min(), df2.index.max()
                
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                
                if overlap_start <= overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    overlaps[f'{name1}_vs_{name2}'] = {
                        'overlap_duration': overlap_duration,
                        'overlap_start': overlap_start,
                        'overlap_end': overlap_end,
                        'overlap_percentage_1': (overlap_duration / (end1 - start1)) * 100,
                        'overlap_percentage_2': (overlap_duration / (end2 - start2)) * 100
                    }
                else:
                    overlaps[f'{name1}_vs_{name2}'] = {'overlap_duration': pd.Timedelta(0)}
        
        return overlaps
    
    def visualize_merge_results(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Visualize the merge results and data quality.
        
        Parameters:
        figsize (tuple): Figure size for plots
        """
        if self.merged_data is None:
            raise ValueError("No merged data available. Perform merge first.")
        
        df = self.merged_data
        
        # Create subplot layout
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('Time Series Merge Analysis', fontsize=16)
        
        # Plot 1: Time series overview
        ax1 = axes[0, 0]
        for col in df.select_dtypes(include=[np.number]).columns[:5]:  # Limit to 5 series
            ax1.plot(df.index, df[col], label=col, alpha=0.7)
        ax1.set_title('Merged Time Series Overview')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Values')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Missing values heatmap
        ax2 = axes[0, 1]
        missing_data = df.isnull()
        if missing_data.any().any():
            sns.heatmap(missing_data.T, cbar=True, ax=ax2, cmap='viridis')
            ax2.set_title('Missing Values Pattern')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Missing Values Pattern')
        
        # Plot 3: Data availability timeline
        ax3 = axes[1, 0]
        availability = (~df.isnull()).astype(int)
        for i, col in enumerate(availability.columns[:10]):  # Limit to 10 columns
            ax3.fill_between(df.index, i, i + availability[col], alpha=0.7, label=col)
        ax3.set_title('Data Availability Timeline')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Dataset')
        
        # Plot 4: Missing value statistics
        ax4 = axes[1, 1]
        missing_percentages = (df.isnull().sum() / len(df) * 100)
        if len(missing_percentages) > 0:
            missing_percentages.plot(kind='bar', ax=ax4)
            ax4.set_title('Missing Value Percentage by Column')
            ax4.set_xlabel('Columns')
            ax4.set_ylabel('Missing %')
            ax4.tick_params(axis='x', rotation=45)
        
        # Plot 5: Frequency analysis
        ax5 = axes[2, 0]
        if len(df.index) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            time_diffs_seconds = time_diffs.dt.total_seconds()
            ax5.hist(time_diffs_seconds, bins=50, alpha=0.7)
            ax5.set_title('Time Interval Distribution')
            ax5.set_xlabel('Seconds between observations')
            ax5.set_ylabel('Frequency')
        
        # Plot 6: Dataset overlap visualization
        ax6 = axes[2, 1]
        if self.original_datasets:
            y_pos = 0
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.original_datasets)))
            
            for (name, df_orig), color in zip(self.original_datasets.items(), colors):
                start_time = df_orig.index.min()
                end_time = df_orig.index.max()
                ax6.barh(y_pos, (end_time - start_time).total_seconds() / 3600, 
                        left=(start_time - df.index.min()).total_seconds() / 3600,
                        height=0.5, color=color, alpha=0.7, label=name)
                y_pos += 1
            
            ax6.set_title('Dataset Temporal Coverage')
            ax6.set_xlabel('Hours from start')
            ax6.set_ylabel('Datasets')
            ax6.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage and utility functions
def create_sample_time_series_datasets():
    """Create multiple sample time series datasets for demonstration."""
    # Dataset 1: High frequency sensor data
    start_time = pd.Timestamp('2023-01-01 00:00:00', tz='UTC')
    dates1 = pd.date_range(start=start_time, periods=1000, freq='1min')
    
    np.random.seed(42)
    data1 = pd.DataFrame({
        'temperature': np.random.normal(20, 5, 1000) + 10 * np.sin(np.arange(1000) * 2 * np.pi / 144),
        'humidity': np.random.normal(60, 10, 1000),
        'sensor_id': 'TEMP01'
    }, index=dates1)
    
    # Introduce some missing values
    data1.loc[data1.index[100:120], 'temperature'] = np.nan
    data1.loc[data1.index[500:510], 'humidity'] = np.nan
    
    # Dataset 2: Lower frequency weather data
    dates2 = pd.date_range(start=start_time, periods=200, freq='5min')
    data2 = pd.DataFrame({
        'pressure': np.random.normal(1013, 20, 200),
        'wind_speed': np.random.exponential(5, 200),
        'station_id': 'WS01'
    }, index=dates2)
    
    # Dataset 3: Event-based data (irregular timestamps)
    event_times = pd.to_datetime(['2023-01-01 00:15:00', '2023-01-01 00:45:00', 
                                 '2023-01-01 01:20:00', '2023-01-01 02:30:00',
                                 '2023-01-01 03:45:00', '2023-01-01 05:10:00'], utc=True)
    data3 = pd.DataFrame({
        'rainfall': [0.5, 1.2, 0.8, 2.1, 0.3, 1.8],
        'event_type': ['light', 'moderate', 'light', 'heavy', 'light', 'moderate']
    }, index=event_times)
    
    return {'sensors': data1, 'weather': data2, 'events': data3}

# Demonstration
if __name__ == "__main__":
    # Create sample datasets
    datasets = create_sample_time_series_datasets()
    
    print("Sample Time Series Datasets:")
    for name, df in datasets.items():
        print(f"\n{name.upper()}:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Missing values: {df.isnull().sum().sum()}")
    
    # Initialize merger
    merger = TimeSeriesMerger(timezone='UTC')
    
    # Prepare datasets
    prepared_datasets = []
    for name, df in datasets.items():
        # Different preparation for different dataset types
        if name == 'sensors':
            prepared = merger.prepare_time_series(df, df.index.name or 'timestamp', 
                                                ['temperature', 'humidity'], freq='1min')
        elif name == 'weather':
            prepared = merger.prepare_time_series(df, df.index.name or 'timestamp',
                                                ['pressure', 'wind_speed'], freq='5min')
        else:  # events
            prepared = merger.prepare_time_series(df, df.index.name or 'timestamp',
                                                ['rainfall'])
        
        prepared_datasets.append(prepared)
    
    # Method 1: Simple outer merge
    print(f"\n=== Method 1: Simple Outer Merge ===")
    merged_outer = merger.merge_time_series_datasets(
        prepared_datasets,
        dataset_names=['sensors', 'weather', 'events'],
        merge_method='outer',
        fill_method='combined'
    )
    
    print(f"Merged shape: {merged_outer.shape}")
    print(f"Missing values after merge: {merged_outer.isnull().sum().sum()}")
    print(f"Date range: {merged_outer.index.min()} to {merged_outer.index.max()}")
    
    # Method 2: Aligned merge with common frequency
    print(f"\n=== Method 2: Aligned Merge (5min frequency) ===")
    merged_aligned = merger.merge_with_alignment(
        {name: df for name, df in zip(['sensors', 'weather', 'events'], prepared_datasets)},
        target_frequency='5min',
        alignment_method='linear'
    )
    
    print(f"Aligned shape: {merged_aligned.shape}")
    print(f"Missing values after alignment: {merged_aligned.isnull().sum().sum()}")
    
    # Method 3: As-of merge for event-based joining
    print(f"\n=== Method 3: As-of Merge ===")
    asof_merged = merger.merge_asof(
        prepared_datasets[0],  # sensors (left)
        prepared_datasets[2],  # events (right)
        tolerance='30min',
        direction='backward'
    )
    
    print(f"As-of merged shape: {asof_merged.shape}")
    print("Sample of as-of merged data:")
    print(asof_merged.head())
    
    # Quality analysis
    print(f"\n=== Merge Quality Analysis ===")
    quality_analysis = merger.analyze_merge_quality()
    
    print("Basic Statistics:")
    for key, value in quality_analysis['basic_stats'].items():
        print(f"  {key}: {value}")
    
    print("\nMissing Value Summary:")
    for col, info in quality_analysis['missing_values'].items():
        print(f"  {col}: {info['missing_percentage']:.1f}% missing, "
              f"max consecutive: {info['consecutive_missing']['max_consecutive']}")
    
    print(f"\nGap Analysis:")
    print(f"  Number of gaps: {quality_analysis['gaps']['gap_count']}")
    
    # Visualize results (uncomment to show plots)
    # merger.visualize_merge_results()
    
    print(f"\n=== Performance Comparison ===")
    print("Merge Methods Summary:")
    print(f"  Outer merge: {merged_outer.shape[0]} rows, {merged_outer.isnull().sum().sum()} missing")
    print(f"  Aligned merge: {merged_aligned.shape[0]} rows, {merged_aligned.isnull().sum().sum()} missing")
    print(f"  As-of merge: {asof_merged.shape[0]} rows, {asof_merged.isnull().sum().sum()} missing")
```

#### Explanation
1. **Temporal Alignment**: Ensures proper timestamp matching across datasets with different frequencies
2. **Missing Value Handling**: Multiple strategies for dealing with gaps and missing data points
3. **Flexible Merging**: Supports various merge methods including outer, inner, and as-of joins
4. **Quality Analysis**: Comprehensive analysis of merge results and data quality metrics
5. **Resampling Support**: Automatic resampling to common frequencies for better alignment

#### Use Cases
- **IoT Sensor Integration**: Combining data from multiple sensors with different sampling rates
- **Financial Data Analysis**: Merging stock prices, economic indicators, and news sentiment
- **Environmental Monitoring**: Combining weather, pollution, and traffic data
- **System Monitoring**: Merging logs, metrics, and events from different system components
- **Research Data**: Combining experimental measurements from different instruments

#### Best Practices
- **Timezone Consistency**: Ensure all datasets use consistent timezone handling
- **Frequency Planning**: Choose appropriate target frequency based on analysis requirements
- **Missing Value Strategy**: Select filling methods based on data characteristics and domain knowledge
- **Memory Management**: Use chunking for large time series datasets
- **Quality Validation**: Always analyze merge quality and data coverage

#### Pitfalls
- **Clock Drift**: Different data sources may have slight time synchronization issues
- **Frequency Mismatch**: Naive merging without resampling can create sparse datasets
- **Missing Data Bias**: Poor missing value handling can introduce analysis bias
- **Memory Issues**: Large time series merges can consume significant memory
- **Timezone Confusion**: Mixed timezones can cause alignment errors

#### Debugging
```python
def debug_time_series_merge(datasets: List[pd.DataFrame], names: List[str]) -> None:
    """Debug common time series merge issues."""
    print("Time Series Merge Debugging")
    
    for i, (df, name) in enumerate(zip(datasets, names)):
        print(f"\n=== Dataset {i+1}: {name} ===")
        
        # Check index type
        print(f"Index type: {type(df.index)}")
        print(f"Is DatetimeIndex: {isinstance(df.index, pd.DatetimeIndex)}")
        
        if isinstance(df.index, pd.DatetimeIndex):
            print(f"Timezone: {df.index.tz}")
            print(f"Frequency: {df.index.freq}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            # Check for duplicates
            duplicates = df.index.duplicated().sum()
            print(f"Duplicate timestamps: {duplicates}")
            
            # Check for gaps
            if len(df.index) > 1:
                time_diffs = df.index.to_series().diff().dropna()
                print(f"Time intervals - min: {time_diffs.min()}, max: {time_diffs.max()}")
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types: {df.dtypes.to_dict()}")
        print(f"Missing values: {df.isnull().sum().to_dict()}")
```

#### Optimization
```python
# Memory-efficient merge for large datasets
def merge_large_time_series(datasets: List[pd.DataFrame], 
                           chunk_size: str = '1D') -> pd.DataFrame:
    """
    Memory-efficient merge for large time series datasets.
    """
    # Find common date range
    min_date = min(df.index.min() for df in datasets)
    max_date = max(df.index.max() for df in datasets)
    
    # Process in chunks
    merged_chunks = []
    current_date = min_date
    
    while current_date < max_date:
        chunk_end = current_date + pd.Timedelta(chunk_size)
        
        # Extract chunks from each dataset
        chunk_datasets = []
        for df in datasets:
            chunk = df[(df.index >= current_date) & (df.index < chunk_end)]
            if not chunk.empty:
                chunk_datasets.append(chunk)
        
        # Merge chunk
        if chunk_datasets:
            chunk_merged = pd.concat(chunk_datasets, axis=1, join='outer')
            merged_chunks.append(chunk_merged)
        
        current_date = chunk_end
    
    # Combine all chunks
    return pd.concat(merged_chunks, axis=0).sort_index()
```

---



