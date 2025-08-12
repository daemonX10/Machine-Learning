# Pandas Interview Questions - Theory Questions

## Question 1

**What is Pandas in Python and why is it used for data analysis?**

### Answer

#### Theory
Pandas (Python Data Analysis Library) is a powerful, fast, and flexible open-source data analysis and manipulation library built on top of NumPy. It provides high-performance, easy-to-use data structures and data analysis tools designed to make working with structured and time series data both fast and intuitive. Pandas is essentially the foundational pillar of the Python data science ecosystem.

#### Core Purpose and Features

**Primary Purpose:**
- **Data Manipulation**: Provides tools for reading, writing, filtering, grouping, and transforming data
- **Data Analysis**: Offers statistical operations, aggregations, and analytical functions
- **Data Cleaning**: Handles missing data, duplicates, and data type conversions
- **Data Integration**: Merges, joins, and combines data from multiple sources

**Key Features:**
1. **Powerful Data Structures**: Series (1D) and DataFrame (2D) that handle both homogeneous and heterogeneous data
2. **Flexible Data I/O**: Read and write data from/to multiple formats (CSV, Excel, JSON, SQL, HDF5, etc.)
3. **Data Alignment**: Automatic and explicit data alignment based on labels
4. **Missing Data Handling**: Robust handling of missing data with various strategies
5. **Time Series Functionality**: Comprehensive date/time handling and time series analysis
6. **Performance**: Built on NumPy and Cython for high performance operations

#### Why Pandas is Essential for Data Analysis

**1. Intuitive Data Structures:**
```python
import pandas as pd
import numpy as np

# Series - 1D labeled array
series = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print("Series:\n", series)

# DataFrame - 2D labeled data structure
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})
print("\nDataFrame:\n", df)
```

**2. Easy Data Import/Export:**
```python
# Reading from various sources
df_csv = pd.read_csv('data.csv')
df_excel = pd.read_excel('data.xlsx')
df_json = pd.read_json('data.json')

# Writing to various formats
df.to_csv('output.csv')
df.to_excel('output.xlsx')
df.to_json('output.json')
```

**3. Powerful Data Manipulation:**
```python
# Filtering
young_employees = df[df['age'] < 30]

# Grouping
age_groups = df.groupby('department')['salary'].mean()

# Merging
merged_df = pd.merge(df1, df2, on='common_column')
```

#### Advantages Over Base Python

**1. Performance:**
- Vectorized operations (much faster than Python loops)
- Optimized memory usage
- Built on NumPy's efficient array operations

**2. Functionality:**
- Rich set of data manipulation functions
- Built-in statistical operations
- Automatic data alignment
- Comprehensive missing data handling

**3. Ease of Use:**
- Intuitive syntax similar to SQL and Excel
- Extensive documentation and community support
- Integration with visualization libraries

#### Real-World Applications

**1. Business Analytics:**
- Sales analysis and reporting
- Customer segmentation
- Financial modeling
- Market research

**2. Scientific Research:**
- Experimental data analysis
- Statistical modeling
- Data preprocessing for machine learning
- Time series analysis

**3. Data Engineering:**
- ETL (Extract, Transform, Load) operations
- Data cleaning and validation
- Data pipeline development
- Database interactions

#### Integration with Data Science Ecosystem

**Core Integrations:**
- **NumPy**: Foundation for numerical operations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning preprocessing
- **Jupyter**: Interactive data analysis
- **SQLAlchemy**: Database connectivity

#### Best Practices
- Use vectorized operations instead of loops
- Leverage built-in functions for common operations
- Handle missing data appropriately
- Use appropriate data types for memory efficiency
- Take advantage of indexing for performance

#### Common Use Cases
- Loading and cleaning messy datasets
- Performing exploratory data analysis (EDA)
- Aggregating and summarizing data
- Preparing data for machine learning models
- Creating reports and dashboards

---

## Question 2

**Explain the difference between a Series and a DataFrame in Pandas.**

### Answer

#### Theory
Series and DataFrame are the two primary data structures in Pandas, designed to handle different types of data organization. Understanding their differences is fundamental to effective Pandas usage, as they serve different purposes in data analysis workflows and have distinct characteristics in terms of dimensionality, indexing, and functionality.

#### Series - 1D Labeled Array

**Definition:**
A Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.). It's essentially a column of data with an associated index.

**Key Characteristics:**
- **One-dimensional**: Contains data in a single column
- **Homogeneous**: All elements should be of the same data type (though mixed types are possible)
- **Labeled**: Each element has an associated index label
- **Size immutable**: Length cannot be changed after creation (but values can be modified)

**Structure:**
```python
import pandas as pd
import numpy as np

# Creating a Series
series = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print("Series structure:")
print(series)
print(f"Data type: {series.dtype}")
print(f"Index: {series.index}")
print(f"Values: {series.values}")
```

#### DataFrame - 2D Labeled Data Structure

**Definition:**
A DataFrame is a two-dimensional labeled data structure with columns of potentially different data types. It's like a spreadsheet or SQL table, or a dictionary of Series objects.

**Key Characteristics:**
- **Two-dimensional**: Contains data in rows and columns
- **Heterogeneous**: Different columns can have different data types
- **Labeled**: Both rows and columns have labels (index and column names)
- **Size mutable**: Can add/remove rows and columns
- **Primary data structure**: Most commonly used structure in Pandas

**Structure:**
```python
# Creating a DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000.0, 60000.0, 70000.0],
    'active': [True, False, True]
})
print("DataFrame structure:")
print(df)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Index: {list(df.index)}")
print(f"Data types:\n{df.dtypes}")
```

#### Detailed Comparison

**1. Dimensionality:**
```python
# Series - 1D
series = pd.Series([1, 2, 3, 4])
print(f"Series dimensions: {series.ndim}")  # Output: 1
print(f"Series shape: {series.shape}")      # Output: (4,)

# DataFrame - 2D
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
print(f"DataFrame dimensions: {df.ndim}")   # Output: 2
print(f"DataFrame shape: {df.shape}")       # Output: (2, 2)
```

**2. Data Type Consistency:**
```python
# Series - typically homogeneous
homogeneous_series = pd.Series([1, 2, 3, 4])  # All integers
print(f"Series dtype: {homogeneous_series.dtype}")

# Mixed types in Series (becomes object dtype)
mixed_series = pd.Series([1, 'hello', 3.14, True])
print(f"Mixed Series dtype: {mixed_series.dtype}")

# DataFrame - heterogeneous columns
df = pd.DataFrame({
    'integers': [1, 2, 3],
    'floats': [1.1, 2.2, 3.3],
    'strings': ['a', 'b', 'c'],
    'booleans': [True, False, True]
})
print("DataFrame dtypes:")
print(df.dtypes)
```

**3. Indexing and Selection:**
```python
# Series indexing
series = pd.Series([10, 20, 30], index=['x', 'y', 'z'])
print(f"Single value: {series['x']}")           # 10
print(f"Multiple values: {series[['x', 'z']]}")  # Series with x and z

# DataFrame indexing
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})
print("Single column (returns Series):")
print(df['A'])
print("\nMultiple columns (returns DataFrame):")
print(df[['A', 'C']])
print("\nRow selection:")
print(df.iloc[0])  # First row as Series
```

**4. Operations and Methods:**
```python
# Series operations
series = pd.Series([1, 2, 3, 4, 5])
print(f"Series sum: {series.sum()}")
print(f"Series mean: {series.mean()}")
print(f"Series max: {series.max()}")

# DataFrame operations
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
print("DataFrame operations:")
print(f"Column-wise sum:\n{df.sum()}")      # Returns Series
print(f"Row-wise sum:\n{df.sum(axis=1)}")   # Returns Series
print(f"Overall sum: {df.sum().sum()}")     # Single value
```

#### Relationship Between Series and DataFrame

**1. DataFrame as Collection of Series:**
```python
# DataFrame can be viewed as a dictionary of Series
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Each column is a Series
column_a = df['A']
print(f"Column A type: {type(column_a)}")  # pandas.core.series.Series
print(f"Column A:\n{column_a}")

# Row selection returns a Series
row_0 = df.iloc[0]
print(f"Row 0 type: {type(row_0)}")        # pandas.core.series.Series
print(f"Row 0:\n{row_0}")
```

**2. Converting Between Series and DataFrame:**
```python
# Series to DataFrame
series = pd.Series([1, 2, 3], name='values')
df_from_series = series.to_frame()
print("Series to DataFrame:")
print(df_from_series)

# Multiple Series to DataFrame
series1 = pd.Series([1, 2, 3], name='A')
series2 = pd.Series([4, 5, 6], name='B')
df_from_multiple = pd.concat([series1, series2], axis=1)
print("Multiple Series to DataFrame:")
print(df_from_multiple)

# DataFrame column to Series
df = pd.DataFrame({'X': [1, 2, 3], 'Y': [4, 5, 6]})
series_from_df = df['X']
print("DataFrame column to Series:")
print(series_from_df)
```

#### Memory and Performance Considerations

**1. Memory Usage:**
```python
# Series memory usage
series = pd.Series(range(1000000))
print(f"Series memory usage: {series.memory_usage(deep=True)} bytes")

# DataFrame memory usage
df = pd.DataFrame({
    'A': range(1000000),
    'B': range(1000000)
})
print(f"DataFrame memory usage:")
print(df.memory_usage(deep=True))
```

**2. Performance Characteristics:**
```python
import time

# Series operations are generally faster for single-column operations
series = pd.Series(range(1000000))
start = time.time()
result_series = series * 2
end = time.time()
print(f"Series operation time: {end - start:.4f} seconds")

# DataFrame operations on single column
df = pd.DataFrame({'A': range(1000000)})
start = time.time()
result_df = df['A'] * 2
end = time.time()
print(f"DataFrame single column operation time: {end - start:.4f} seconds")
```

#### When to Use Series vs DataFrame

**Use Series when:**
- Working with a single column of data
- Performing mathematical operations on one-dimensional data
- Creating indices for DataFrames
- Working with time series data (single metric)
- Need simpler, more direct operations

**Use DataFrame when:**
- Working with tabular data (multiple columns)
- Need to maintain relationships between different variables
- Performing complex data analysis with multiple dimensions
- Working with heterogeneous data types
- Need advanced data manipulation features

#### Advanced Concepts

**1. Index Alignment:**
```python
# Series index alignment
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['b', 'c', 'd'])
result = s1 + s2  # Aligns on index automatically
print("Series alignment result:")
print(result)

# DataFrame index alignment
df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
df2 = pd.DataFrame({'A': [3, 4]}, index=[1, 2])
result = df1 + df2
print("DataFrame alignment result:")
print(result)
```

**2. Broadcasting:**
```python
# Series broadcasting
series = pd.Series([1, 2, 3])
result = series + 10  # Broadcasts scalar to all elements
print("Series broadcasting:")
print(result)

# DataFrame broadcasting
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = df + 10  # Broadcasts to all elements
print("DataFrame broadcasting:")
print(result)

# DataFrame with Series broadcasting
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
series = pd.Series([10, 20])
result = df + series  # Broadcasts along columns
print("DataFrame + Series broadcasting:")
print(result)
```

#### Best Practices

**1. Choosing the Right Structure:**
- Use Series for single-variable analysis
- Use DataFrame for multi-variable analysis
- Consider memory usage for large datasets
- Use appropriate indexing strategies

**2. Performance Optimization:**
- Prefer vectorized operations over loops
- Use appropriate data types
- Consider using categorical data for repeated strings
- Leverage built-in methods for common operations

**3. Code Readability:**
- Use descriptive column names and indices
- Document complex operations
- Use consistent naming conventions
- Consider chaining operations for readability

---

## Question 3

**What are Pandas indexes, and how are they used?**

### Answer

#### Theory
Pandas indexes are fundamental data structures that provide labels for rows and columns in Series and DataFrames. They serve as immutable arrays that hold axis labels and other metadata, enabling powerful data alignment, selection, and manipulation operations. Indexes are crucial for Pandas' automatic data alignment capabilities and provide the foundation for many advanced operations.

#### Understanding Pandas Indexes

**Definition and Purpose:**
An index in Pandas is an immutable array-like structure that labels the rows (and columns) of a Series or DataFrame. It provides:
- **Data Alignment**: Automatic alignment of data based on labels
- **Data Selection**: Label-based and position-based selection
- **Performance**: Optimized operations through indexing
- **Data Integrity**: Ensures data relationships are maintained

**Types of Indexes:**
```python
import pandas as pd
import numpy as np
from datetime import datetime, date

# 1. RangeIndex (default integer index)
df_range = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print("RangeIndex:")
print(f"Index type: {type(df_range.index)}")
print(f"Index: {df_range.index}")

# 2. Index (custom labels)
df_custom = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
print("\nCustom Index:")
print(f"Index: {df_custom.index}")

# 3. DatetimeIndex (for time series)
dates = pd.date_range('2023-01-01', periods=3, freq='D')
df_datetime = pd.DataFrame({'values': [1, 2, 3]}, index=dates)
print("\nDatetimeIndex:")
print(f"Index type: {type(df_datetime.index)}")
print(f"Index: {df_datetime.index}")

# 4. MultiIndex (hierarchical index)
multi_idx = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)])
df_multi = pd.DataFrame({'values': [10, 20, 30]}, index=multi_idx)
print("\nMultiIndex:")
print(f"Index: {df_multi.index}")
```

#### Index Properties and Characteristics

**1. Immutability:**
```python
# Indexes are immutable
df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
original_index = df.index

# This creates a new DataFrame with new index
df_new = df.set_index(pd.Index(['x', 'y', 'z']))
print(f"Original index unchanged: {original_index}")
print(f"New DataFrame index: {df_new.index}")

# Attempting to modify index directly raises error
try:
    df.index[0] = 'new_value'  # This will raise TypeError
except TypeError as e:
    print(f"Index modification error: {e}")
```

**2. Index Attributes:**
```python
df = pd.DataFrame({'A': [1, 2, 3, 2, 1]}, index=['a', 'b', 'c', 'd', 'e'])

print("Index attributes:")
print(f"Name: {df.index.name}")
print(f"Size: {df.index.size}")
print(f"Shape: {df.index.shape}")
print(f"Data type: {df.index.dtype}")
print(f"Is unique: {df.index.is_unique}")
print(f"Is monotonic: {df.index.is_monotonic}")
print(f"Has duplicates: {df.index.has_duplicates}")
```

#### Data Alignment with Indexes

**1. Automatic Alignment:**
```python
# Series alignment based on index
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6, 7], index=['b', 'c', 'd', 'e'])

# Addition aligns on index automatically
result = s1 + s2
print("Automatic alignment:")
print(result)  # Only 'b' and 'c' will have values, others NaN

# DataFrame alignment
df1 = pd.DataFrame({'X': [1, 2]}, index=['a', 'b'])
df2 = pd.DataFrame({'X': [3, 4]}, index=['b', 'c'])
result_df = df1 + df2
print("\nDataFrame alignment:")
print(result_df)
```

**2. Reindexing:**
```python
df = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])

# Reindex to new set of labels
new_index = ['x', 'y', 'z', 'w']
df_reindexed = df.reindex(new_index)
print("Reindexed DataFrame:")
print(df_reindexed)

# Reindex with fill value
df_filled = df.reindex(new_index, fill_value=0)
print("\nReindexed with fill value:")
print(df_filled)

# Reindex with method
df_method = df.reindex(new_index, method='ffill')
print("\nReindexed with forward fill:")
print(df_method)
```

#### Index-Based Selection

**1. Label-Based Selection (.loc):**
```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
}, index=['w', 'x', 'y', 'z'])

# Single row selection
row = df.loc['x']
print("Single row selection:")
print(row)

# Multiple rows
rows = df.loc[['w', 'y']]
print("\nMultiple rows:")
print(rows)

# Row and column selection
subset = df.loc['x':'y', 'A']
print("\nRow and column selection:")
print(subset)

# Boolean indexing with labels
boolean_mask = df['A'] > 2
filtered = df.loc[boolean_mask]
print("\nBoolean indexing:")
print(filtered)
```

**2. Position-Based Selection (.iloc):**
```python
# Position-based selection
first_row = df.iloc[0]
print("First row (position-based):")
print(first_row)

# Multiple positions
rows_pos = df.iloc[[0, 2]]
print("\nMultiple rows by position:")
print(rows_pos)

# Slicing by position
slice_pos = df.iloc[1:3, 0:1]
print("\nSlicing by position:")
print(slice_pos)
```

#### Special Index Types

**1. DatetimeIndex:**
```python
# Creating DatetimeIndex
dates = pd.date_range('2023-01-01', periods=5, freq='D')
ts = pd.Series(range(5), index=dates)

print("Time series with DatetimeIndex:")
print(ts)

# Date-based selection
print("\nDate selection:")
print(ts['2023-01-02'])
print(ts['2023-01-02':'2023-01-04'])

# Partial string indexing
monthly_data = pd.Series(
    range(30),
    index=pd.date_range('2023-01-01', periods=30, freq='D')
)
print("\nPartial string indexing:")
print(monthly_data['2023-01'])  # All January data
```

**2. MultiIndex (Hierarchical Index):**
```python
# Creating MultiIndex
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
multi_index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df_multi = pd.DataFrame(np.random.randn(4, 2), index=multi_index, columns=['X', 'Y'])

print("MultiIndex DataFrame:")
print(df_multi)

# Accessing with MultiIndex
print("\nLevel 'A' data:")
print(df_multi.loc['A'])

# Cross-section
print("\nCross-section (second level = 1):")
print(df_multi.xs(1, level='second'))

# Swapping levels
print("\nSwapped levels:")
df_swapped = df_multi.swaplevel('first', 'second')
print(df_swapped)
```

#### Index Operations and Methods

**1. Index Manipulation:**
```python
df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])

# Setting new index
df_new_index = df.set_index(pd.Index(['x', 'y', 'z'], name='new_index'))
print("New index set:")
print(df_new_index)

# Resetting index (moving index to column)
df_reset = df_new_index.reset_index()
print("\nIndex reset:")
print(df_reset)

# Setting column as index
df_with_col = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})
df_name_index = df_with_col.set_index('name')
print("\nColumn as index:")
print(df_name_index)
```

**2. Index Set Operations:**
```python
idx1 = pd.Index(['a', 'b', 'c', 'd'])
idx2 = pd.Index(['c', 'd', 'e', 'f'])

print("Index set operations:")
print(f"Union: {idx1.union(idx2)}")
print(f"Intersection: {idx1.intersection(idx2)}")
print(f"Difference: {idx1.difference(idx2)}")
print(f"Symmetric difference: {idx1.symmetric_difference(idx2)}")
```

#### Performance Benefits of Indexing

**1. Hash-based lookups:**
```python
import time

# Large DataFrame
large_df = pd.DataFrame({
    'value': np.random.randn(1000000),
    'key': [f'key_{i}' for i in range(1000000)]
})

# Without index (sequential search)
start_time = time.time()
result1 = large_df[large_df['key'] == 'key_500000']
time_without_index = time.time() - start_time

# With index (hash lookup)
indexed_df = large_df.set_index('key')
start_time = time.time()
result2 = indexed_df.loc['key_500000']
time_with_index = time.time() - start_time

print(f"Time without index: {time_without_index:.4f} seconds")
print(f"Time with index: {time_with_index:.4f} seconds")
print(f"Speedup: {time_without_index / time_with_index:.1f}x")
```

#### Best Practices for Index Usage

**1. Index Design:**
```python
# Good: Meaningful index names
df = pd.DataFrame({
    'sales': [100, 200, 150],
    'profit': [20, 40, 30]
}, index=pd.Index(['Q1', 'Q2', 'Q3'], name='quarter'))

# Good: Appropriate index type for data
time_series = pd.DataFrame({
    'temperature': [20, 22, 25]
}, index=pd.date_range('2023-01-01', periods=3, freq='D', name='date'))

# Good: MultiIndex for hierarchical data
hierarchical = pd.DataFrame({
    'value': [10, 20, 30, 40]
}, index=pd.MultiIndex.from_tuples([
    ('North', 'A'), ('North', 'B'), ('South', 'A'), ('South', 'B')
], names=['region', 'category']))
```

**2. Index Operations:**
```python
# Efficient: Use .loc for label-based selection
efficient_selection = df.loc['Q2']

# Efficient: Use .iloc for position-based selection
efficient_position = df.iloc[1]

# Avoid: Using loops for index operations
# Instead of:
# for idx in df.index:
#     process(df.loc[idx])

# Use vectorized operations:
processed = df.apply(lambda x: x * 2)
```

#### Common Pitfalls and Solutions

**1. Index Alignment Issues:**
```python
# Problem: Misaligned indexes
s1 = pd.Series([1, 2, 3], index=[0, 1, 2])
s2 = pd.Series([4, 5, 6], index=[1, 2, 3])
result = s1 + s2  # Results in NaN for non-matching indexes

# Solution: Explicit alignment
aligned_result = s1.add(s2, fill_value=0)
print("Aligned addition with fill_value:")
print(aligned_result)
```

**2. Index Duplication:**
```python
# Problem: Duplicate index values
df_dup = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'a', 'b'])
print("DataFrame with duplicate index:")
print(df_dup)

# Accessing duplicate index returns multiple rows
print("\nAccessing duplicate index:")
print(df_dup.loc['a'])

# Solution: Make index unique
df_unique = df_dup.reset_index(drop=True)
print("\nUnique index:")
print(df_unique)
```

#### Advanced Index Concepts

**1. Index Frequency (for DatetimeIndex):**
```python
# DatetimeIndex with frequency
freq_index = pd.date_range('2023-01-01', periods=10, freq='B')  # Business days
ts_freq = pd.Series(range(10), index=freq_index)
print("Business day frequency:")
print(ts_freq)
print(f"Frequency: {ts_freq.index.freq}")
```

**2. Index Memory Usage:**
```python
# Compare memory usage of different index types
range_idx = pd.RangeIndex(1000000)
int_idx = pd.Index(range(1000000))
str_idx = pd.Index([f'item_{i}' for i in range(100000)])

print("Index memory usage:")
print(f"RangeIndex: {range_idx.memory_usage(deep=True)} bytes")
print(f"Integer Index: {int_idx.memory_usage(deep=True)} bytes") 
print(f"String Index: {str_idx.memory_usage(deep=True)} bytes")
```

---

## Question 4

**Explain the concept of data alignment and broadcasting in Pandas.**

### Answer

#### Theory
Data alignment and broadcasting are fundamental concepts in Pandas that enable automatic and intelligent handling of operations between data structures with different shapes, sizes, or indexes. These mechanisms ensure that operations are performed correctly even when data structures don't perfectly match, making Pandas both powerful and intuitive for data manipulation.

#### Data Alignment

**Definition:**
Data alignment is Pandas' automatic process of matching data based on index labels before performing operations. When operations are performed between Series or DataFrames, Pandas automatically aligns the data using the index labels, ensuring that operations are performed on corresponding elements.

**Key Principles:**
- Operations align on index labels, not positions
- Missing labels result in NaN values
- Both row and column alignment occur in DataFrames
- Alignment happens automatically before arithmetic operations

**1. Series Alignment:**
```python
import pandas as pd
import numpy as np

# Create Series with different indexes
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6, 7], index=['b', 'c', 'd', 'e'])

print("Series 1:")
print(s1)
print("\nSeries 2:")
print(s2)

# Addition with automatic alignment
result = s1 + s2
print("\nAligned addition (s1 + s2):")
print(result)
# Only 'b' and 'c' have values from both series
# 'a' gets NaN (only in s1), 'd' and 'e' get NaN (only in s2)
```

**2. DataFrame Alignment:**
```python
# DataFrames with different indexes and columns
df1 = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}, index=['x', 'y', 'z'])

df2 = pd.DataFrame({
    'A': [7, 8],
    'C': [9, 10]
}, index=['y', 'z'])

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Addition with alignment on both rows and columns
result = df1 + df2
print("\nAligned addition (df1 + df2):")
print(result)
# Column 'A' aligns where indexes match
# Column 'B' and 'C' become NaN where no alignment exists
```

**3. Controlling Alignment with Fill Values:**
```python
# Using fill_value to handle missing alignments
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5], index=['b', 'c'])

# Default alignment (produces NaN)
default_result = s1 + s2
print("Default alignment:")
print(default_result)

# Using add method with fill_value
filled_result = s1.add(s2, fill_value=0)
print("\nAlignment with fill_value=0:")
print(filled_result)

# DataFrame alignment with fill_value
df_result = df1.add(df2, fill_value=0)
print("\nDataFrame alignment with fill_value:")
print(df_result)
```

#### Broadcasting

**Definition:**
Broadcasting in Pandas extends NumPy's broadcasting rules to work with labeled data. It allows operations between data structures of different shapes by automatically expanding the smaller structure to match the larger one, while respecting index alignment.

**Broadcasting Rules:**
1. Scalar values broadcast to all elements
2. Series broadcasts along matching axis in DataFrames
3. Lower-dimensional structures broadcast to higher dimensions
4. Index labels must be compatible for broadcasting

**1. Scalar Broadcasting:**
```python
# Scalar with Series
series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
scalar_result = series + 10
print("Scalar broadcasting with Series:")
print(scalar_result)

# Scalar with DataFrame
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df_scalar_result = df * 2
print("\nScalar broadcasting with DataFrame:")
print(df_scalar_result)
```

**2. Series-DataFrame Broadcasting:**
```python
# DataFrame and Series broadcasting
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}, index=['x', 'y', 'z'])

# Series with matching column names (broadcasts along rows)
col_series = pd.Series([10, 20, 30], index=['A', 'B', 'C'])
print("DataFrame:")
print(df)
print("\nColumn Series:")
print(col_series)
print("\nBroadcasting along columns:")
print(df + col_series)

# Series with matching index (broadcasts along columns)
row_series = pd.Series([100, 200, 300], index=['x', 'y', 'z'])
print("\nRow Series:")
print(row_series)
print("\nBroadcasting along rows:")
print(df.add(row_series, axis=0))  # Explicit axis specification
```

**3. Controlling Broadcasting Direction:**
```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}, index=['x', 'y', 'z'])

series = pd.Series([10, 20], index=['A', 'B'])

# Broadcasting along columns (default for Series)
result_cols = df + series
print("Broadcasting along columns:")
print(result_cols)

# Broadcasting along rows (explicit axis=0)
row_series = pd.Series([100, 200, 300], index=['x', 'y', 'z'])
result_rows = df.add(row_series, axis=0)
print("\nBroadcasting along rows:")
print(result_rows)
```

#### Advanced Alignment and Broadcasting

**1. Multi-Level Index Alignment:**
```python
# MultiIndex alignment
multi_idx1 = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)])
multi_idx2 = pd.MultiIndex.from_tuples([('A', 1), ('B', 1), ('B', 2)])

s_multi1 = pd.Series([10, 20, 30], index=multi_idx1)
s_multi2 = pd.Series([5, 15, 25], index=multi_idx2)

print("MultiIndex Series 1:")
print(s_multi1)
print("\nMultiIndex Series 2:")
print(s_multi2)
print("\nMultiIndex alignment:")
print(s_multi1 + s_multi2)
```

**2. DateTime Index Alignment:**
```python
# Time series alignment
dates1 = pd.date_range('2023-01-01', periods=3, freq='D')
dates2 = pd.date_range('2023-01-02', periods=3, freq='D')

ts1 = pd.Series([1, 2, 3], index=dates1)
ts2 = pd.Series([10, 20, 30], index=dates2)

print("Time Series 1:")
print(ts1)
print("\nTime Series 2:")
print(ts2)
print("\nTime series alignment:")
print(ts1 + ts2)
```

#### Practical Applications

**1. Data Cleaning and Normalization:**
```python
# Normalizing data using broadcasting
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50],
    'feature3': [100, 200, 300, 400, 500]
})

# Calculate mean for each column
means = data.mean()
print("Original data:")
print(data)
print("\nColumn means:")
print(means)

# Center data by subtracting mean (broadcasting)
centered_data = data - means
print("\nCentered data:")
print(centered_data)

# Standardize by dividing by standard deviation
stds = data.std()
standardized_data = (data - means) / stds
print("\nStandardized data:")
print(standardized_data)
```

**2. Financial Data Analysis:**
```python
# Stock returns calculation with alignment
prices = pd.DataFrame({
    'AAPL': [150, 155, 152, 158],
    'GOOGL': [2800, 2850, 2820, 2900],
    'MSFT': [300, 305, 298, 310]
}, index=pd.date_range('2023-01-01', periods=4, freq='D'))

# Calculate returns using alignment
returns = (prices / prices.shift(1) - 1) * 100
print("Stock prices:")
print(prices)
print("\nDaily returns (%):")
print(returns)

# Portfolio weights
weights = pd.Series([0.4, 0.3, 0.3], index=['AAPL', 'GOOGL', 'MSFT'])
weighted_returns = returns * weights  # Broadcasting
print("\nWeighted returns:")
print(weighted_returns)
```

#### Performance Considerations

**1. Alignment Overhead:**
```python
import time

# Large aligned operation
large_s1 = pd.Series(np.random.randn(1000000), 
                     index=range(1000000))
large_s2 = pd.Series(np.random.randn(1000000), 
                     index=range(500000, 1500000))  # Partial overlap

# Measure alignment time
start_time = time.time()
result = large_s1 + large_s2
alignment_time = time.time() - start_time
print(f"Alignment operation time: {alignment_time:.4f} seconds")

# Compare with NumPy array operation (no alignment)
arr1 = np.random.randn(1000000)
arr2 = np.random.randn(1000000)

start_time = time.time()
arr_result = arr1 + arr2
numpy_time = time.time() - start_time
print(f"NumPy operation time: {numpy_time:.4f} seconds")
print(f"Alignment overhead: {alignment_time / numpy_time:.1f}x")
```

**2. Memory Efficiency in Broadcasting:**
```python
# Memory-efficient broadcasting
df = pd.DataFrame(np.random.randn(1000, 10))
series = pd.Series(np.random.randn(10))

# Broadcasting doesn't create intermediate copies
print(f"DataFrame memory usage: {df.memory_usage(deep=True).sum()} bytes")
print(f"Series memory usage: {series.memory_usage(deep=True)} bytes")

# The operation is memory efficient
result = df + series
print(f"Result memory usage: {result.memory_usage(deep=True).sum()} bytes")
```

#### Best Practices

**1. Index Design for Alignment:**
```python
# Good: Meaningful, consistent indexes
good_index = pd.Index(['Q1_2023', 'Q2_2023', 'Q3_2023'], name='quarter')
df_good = pd.DataFrame({'revenue': [100, 120, 110]}, index=good_index)

# Good: Using DatetimeIndex for time series
time_index = pd.date_range('2023-01-01', periods=3, freq='M', name='month')
ts_good = pd.Series([1.5, 1.7, 1.6], index=time_index)

# Operations align correctly
aligned_result = df_good['revenue'] * ts_good
print("Well-aligned time series operation:")
print(aligned_result)
```

**2. Explicit Control When Needed:**
```python
# When alignment isn't desired, use values
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['x', 'y', 'z'])

# This creates all NaN due to no index overlap
bad_result = s1 + s2
print("Unintended alignment result:")
print(bad_result)

# Use .values to bypass alignment
good_result = pd.Series(s1.values + s2.values, 
                       index=s1.index)
print("\nBypassing alignment with .values:")
print(good_result)
```

#### Common Pitfalls and Solutions

**1. Unexpected NaN Results:**
```python
# Problem: Misaligned indexes creating NaN
df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
df2 = pd.DataFrame({'A': [3, 4]}, index=[1, 2])
problematic = df1 + df2
print("Problematic alignment:")
print(problematic)

# Solution: Explicit reindexing or fill_value
solution1 = df1.add(df2, fill_value=0)
print("\nSolution with fill_value:")
print(solution1)

# Solution: Reindex to common index
common_index = df1.index.union(df2.index)
solution2 = df1.reindex(common_index, fill_value=0) + df2.reindex(common_index, fill_value=0)
print("\nSolution with reindexing:")
print(solution2)
```

**2. Broadcasting Dimension Mismatch:**
```python
# Problem: Incompatible dimensions
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
wrong_series = pd.Series([10, 20, 30])  # Wrong length

try:
    wrong_result = df + wrong_series
except ValueError as e:
    print(f"Broadcasting error: {e}")

# Solution: Ensure compatible dimensions
correct_series = pd.Series([10, 20], index=['A', 'B'])
correct_result = df + correct_series
print("\nCorrect broadcasting:")
print(correct_result)
```

---

## Question 5

**What is data slicing in Pandas, and how does it differ from filtering?**

### Answer

#### Theory
Data slicing and filtering are two fundamental data access patterns in Pandas that serve different purposes. **Slicing** refers to selecting contiguous subsets of data based on position or label ranges, while **filtering** involves selecting data based on conditional criteria. Understanding the distinction is crucial for efficient data manipulation and query optimization.

#### Data Slicing

**Definition:** Slicing extracts contiguous portions of data using position-based or label-based ranges.

**Key Characteristics:**
- Returns contiguous subsets
- Uses range syntax (start:stop:step)
- Can be position-based (.iloc) or label-based (.loc)
- Preserves original data structure

```python
import pandas as pd
import numpy as np

# Create sample DataFrame
df = pd.DataFrame({
    'A': range(10),
    'B': range(10, 20),
    'C': range(20, 30),
    'D': list('abcdefghij')
}, index=list('pqrstuvwxy'))

print("Original DataFrame:")
print(df)

# Position-based slicing with .iloc
print("\n=== Position-based Slicing (.iloc) ===")
print("First 5 rows:")
print(df.iloc[:5])

print("\nRows 2-6:")
print(df.iloc[2:7])

print("\nEvery second row:")
print(df.iloc[::2])

print("\nColumns 1-3:")
print(df.iloc[:, 1:4])

print("\nRows 3-7, Columns 0-2:")
print(df.iloc[3:8, 0:3])

# Label-based slicing with .loc
print("\n=== Label-based Slicing (.loc) ===")
print("Index 'p' to 't':")
print(df.loc['p':'t'])

print("\nColumns 'A' to 'C':")
print(df.loc[:, 'A':'C'])

print("\nRows 'r' to 'v', Columns 'B' to 'D':")
print(df.loc['r':'v', 'B':'D'])

# Series slicing
series = pd.Series(range(20), index=range(100, 120))
print("\n=== Series Slicing ===")
print("Original series (first 5):")
print(series.head())

print("\nSlice indices 102-107:")
print(series[102:108])

print("\nSlice positions 5-10:")
print(series.iloc[5:11])
```

#### Data Filtering

**Definition:** Filtering selects data based on boolean conditions or criteria, potentially returning non-contiguous subsets.

**Key Characteristics:**
- Returns data meeting specific conditions
- Uses boolean indexing
- Can return non-contiguous results
- Flexible conditional logic

```python
# Boolean filtering examples
print("\n=== Boolean Filtering ===")

# Single condition
print("Rows where column A > 5:")
filtered1 = df[df['A'] > 5]
print(filtered1)

# Multiple conditions
print("\nRows where A > 3 AND B < 18:")
filtered2 = df[(df['A'] > 3) & (df['B'] < 18)]
print(filtered2)

print("\nRows where A < 3 OR A > 8:")
filtered3 = df[(df['A'] < 3) | (df['A'] > 8)]
print(filtered3)

# String filtering
print("\nRows where D contains vowels:")
vowel_filter = df['D'].str.contains('[aeiou]', regex=True)
filtered4 = df[vowel_filter]
print(filtered4)

# Using isin() for filtering
print("\nRows where D is in ['a', 'c', 'e']:")
filtered5 = df[df['D'].isin(['a', 'c', 'e'])]
print(filtered5)

# Complex filtering with query()
print("\nUsing query() method:")
filtered6 = df.query('A > 4 and B < 17')
print(filtered6)
```

#### Advanced Slicing and Filtering

```python
# DateTime slicing
dates = pd.date_range('2023-01-01', periods=10, freq='D')
ts = pd.DataFrame({
    'value': np.random.randn(10),
    'category': np.random.choice(['X', 'Y'], 10)
}, index=dates)

print("\n=== DateTime Slicing ===")
print("Time series data:")
print(ts)

print("\nSlice January 3-7:")
print(ts.loc['2023-01-03':'2023-01-07'])

print("\nPartial string slicing (all January):")
ts_month = pd.DataFrame({
    'value': np.random.randn(31)
}, index=pd.date_range('2023-01-01', periods=31, freq='D'))
print(ts_month.loc['2023-01'].head())

# Multi-level indexing
multi_index = pd.MultiIndex.from_tuples([
    ('A', 1), ('A', 2), ('A', 3),
    ('B', 1), ('B', 2), ('B', 3),
    ('C', 1), ('C', 2), ('C', 3)
], names=['letter', 'number'])

multi_df = pd.DataFrame({
    'value': range(9),
    'score': range(10, 19)
}, index=multi_index)

print("\n=== Multi-level Index Slicing ===")
print("Multi-index DataFrame:")
print(multi_df)

print("\nSlice level 'A' to 'B':")
print(multi_df.loc['A':'B'])

print("\nCross-section slice:")
print(multi_df.loc[(slice('A', 'B'), slice(1, 2)), :])
```

#### Key Differences

```python
# Demonstrating key differences
print("\n=== Key Differences Demonstration ===")

# 1. Contiguity
print("1. CONTIGUITY:")
print("Slicing (contiguous):")
slice_result = df.iloc[2:6]
print(f"Slice indices: {slice_result.index.tolist()}")

print("Filtering (potentially non-contiguous):")
filter_result = df[df['A'] % 2 == 0]
print(f"Filter indices: {filter_result.index.tolist()}")

# 2. Predictability
print("\n2. PREDICTABILITY:")
print("Slicing always returns same number of rows for same range")
print(f"df.iloc[1:4] always returns {len(df.iloc[1:4])} rows")

print("Filtering returns variable number based on data")
condition_results = []
for threshold in [2, 5, 8]:
    result = df[df['A'] > threshold]
    condition_results.append(len(result))
print(f"df[df['A'] > threshold] returns: {condition_results} rows for thresholds [2,5,8]")

# 3. Performance characteristics
import time

large_df = pd.DataFrame({
    'A': np.random.randint(0, 100, 100000),
    'B': np.random.randn(100000)
})

# Timing slicing
start = time.time()
slice_result = large_df.iloc[10000:20000]
slice_time = time.time() - start

# Timing filtering
start = time.time()
filter_result = large_df[large_df['A'] > 50]
filter_time = time.time() - start

print(f"\n3. PERFORMANCE:")
print(f"Slicing 10K rows: {slice_time:.6f} seconds")
print(f"Filtering ~50K rows: {filter_time:.6f} seconds")
```

#### When to Use Each Approach

```python
# Use case demonstrations
print("\n=== USE CASE GUIDELINES ===")

# Slicing use cases
print("SLICING - Use when:")
print("1. Getting first/last N records")
top_5 = df.head()  # Equivalent to df.iloc[:5]
print("Top 5 records:")
print(top_5)

print("\n2. Sampling regular intervals")
every_third = df.iloc[::3]
print("Every 3rd row:")
print(every_third)

print("\n3. Working with sorted data ranges")
sorted_df = df.sort_values('A')
middle_range = sorted_df.iloc[3:7]
print("Middle range of sorted data:")
print(middle_range)

# Filtering use cases
print("\nFILTERING - Use when:")
print("1. Finding data meeting criteria")
high_values = df[df['A'] > df['A'].mean()]
print("Above-average values:")
print(high_values)

print("\n2. Data quality checks")
valid_data = df[df['A'].notna() & (df['A'] >= 0)]
print("Valid data (non-null, non-negative):")
print(valid_data)

print("\n3. Business rule application")
business_filter = df[df['D'].str.contains('[aeiou]') & (df['A'] % 2 == 0)]
print("Business rule: vowels AND even numbers:")
print(business_filter)
```

#### Advanced Techniques

```python
# Combining slicing and filtering
print("\n=== COMBINING TECHNIQUES ===")

# Filter then slice
filtered_then_sliced = df[df['A'] > 5].iloc[:3]
print("Filter A > 5, then take first 3:")
print(filtered_then_sliced)

# Slice then filter
sliced_then_filtered = df.iloc[2:8][df.iloc[2:8]['B'] < 17]
print("Slice rows 2-7, then filter B < 17:")
print(sliced_then_filtered)

# Conditional slicing
def conditional_slice(df, condition_col, threshold, slice_size):
    """Get first N rows that meet condition."""
    filtered = df[df[condition_col] > threshold]
    return filtered.iloc[:slice_size]

cond_slice = conditional_slice(df, 'A', 4, 3)
print("First 3 rows where A > 4:")
print(cond_slice)

# Dynamic slicing based on data
def dynamic_slice(df, fraction=0.5):
    """Slice middle fraction of data."""
    n = len(df)
    start = int(n * (1 - fraction) / 2)
    end = int(n * (1 + fraction) / 2)
    return df.iloc[start:end]

dynamic_result = dynamic_slice(df, 0.6)
print(f"Middle 60% of data (rows {len(df) - len(dynamic_result)})")
print(dynamic_result)
```

#### Performance Optimization

```python
# Optimization techniques
print("\n=== OPTIMIZATION TECHNIQUES ===")

# 1. Index-based operations are faster
df_indexed = df.set_index('A')
print("1. Index-based filtering:")

start = time.time()
indexed_filter = df_indexed.loc[df_indexed.index > 5]
indexed_time = time.time() - start

start = time.time()
column_filter = df[df['A'] > 5]
column_time = time.time() - start

print(f"Index-based: {indexed_time:.6f}s")
print(f"Column-based: {column_time:.6f}s")

# 2. Use query() for complex conditions
complex_condition = "A > 3 and B < 18 and D in ['c', 'd', 'e']"

start = time.time()
query_result = df.query(complex_condition)
query_time = time.time() - start

start = time.time()
manual_result = df[(df['A'] > 3) & (df['B'] < 18) & df['D'].isin(['c', 'd', 'e'])]
manual_time = time.time() - start

print(f"\n2. Complex filtering:")
print(f"query(): {query_time:.6f}s")
print(f"manual: {manual_time:.6f}s")

# 3. Vectorized operations
print("\n3. Vectorized vs iterative:")

def iterative_filter(df):
    result = []
    for idx, row in df.iterrows():
        if row['A'] > 5:
            result.append(row)
    return pd.DataFrame(result)

start = time.time()
vectorized = df[df['A'] > 5]
vec_time = time.time() - start

start = time.time()
iterative = iterative_filter(df)
iter_time = time.time() - start

print(f"Vectorized: {vec_time:.6f}s")
print(f"Iterative: {iter_time:.6f}s")
print(f"Speedup: {iter_time/vec_time:.1f}x")
```

#### Best Practices

**For Slicing:**
1. Use `.iloc` for position-based operations
2. Use `.loc` for label-based operations
3. Prefer slicing for sorted data ranges
4. Use slicing for pagination and sampling

**For Filtering:**
1. Use vectorized boolean operations
2. Combine conditions with `&`, `|`, `~`
3. Use `.query()` for complex readable conditions
4. Consider indexing for repeated filter operations

#### Common Pitfalls

```python
# Common mistakes and solutions
print("\n=== COMMON PITFALLS ===")

# 1. Chained assignment warning
print("1. Chained assignment (WARNING):")
try:
    # This can cause SettingWithCopyWarning
    df[df['A'] > 5]['B'] = 999  # DON'T DO THIS
except Exception as e:
    print(f"Warning prevented: {e}")

# Correct approach
df_copy = df.copy()
df_copy.loc[df_copy['A'] > 5, 'B'] = 999
print("Correct: Use .loc for conditional assignment")

# 2. Index alignment in slicing
print("\n2. Index alignment issues:")
df_shuffled = df.sample(frac=1)  # Shuffle rows
print("Original order slice:")
print(df.iloc[2:5]['A'].values)
print("Shuffled order slice:")
print(df_shuffled.iloc[2:5]['A'].values)
print("Use .sort_index() if order matters")

# 3. Boolean indexing with NaN
df_with_nan = df.copy()
df_with_nan.loc[3:5, 'A'] = np.nan
print("\n3. NaN in boolean conditions:")
print("Data with NaN:")
print(df_with_nan[['A', 'B']])
print("Filter A > 5 (NaN excluded automatically):")
print(df_with_nan[df_with_nan['A'] > 5][['A', 'B']])
```

#### Summary

| Aspect | Slicing | Filtering |
|--------|---------|-----------|
| **Purpose** | Extract contiguous ranges | Select by conditions |
| **Result** | Always contiguous | Potentially scattered |
| **Syntax** | `[start:stop:step]` | `[boolean_condition]` |
| **Performance** | Fast, predictable | Variable, depends on selectivity |
| **Use Cases** | Pagination, sampling, ranges | Data quality, business rules |
| **Predictability** | Deterministic size | Data-dependent size |

---

## Question 6

**Describe how joining and merging data works in Pandas.**

### Answer

#### Theory
Joining and merging are fundamental operations for combining data from multiple DataFrames in Pandas. These operations enable the integration of related datasets based on common keys or indices, similar to SQL joins. Understanding the different types of joins and their appropriate use cases is essential for effective data analysis and preparation workflows.

#### Core Concepts

**Merging vs Joining:**
- **Merge**: General-purpose operation using `pd.merge()` or `DataFrame.merge()`
- **Join**: Index-based operation using `DataFrame.join()` 
- **Concatenate**: Stacking operations using `pd.concat()`

```python
import pandas as pd
import numpy as np

# Create sample datasets for demonstration
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105, 106],
    'customer_id': [1, 2, 2, 3, 6, 1],  # Note: customer_id 6 doesn't exist in customers
    'product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Mouse', 'Keyboard'],
    'amount': [1200, 800, 400, 300, 25, 150]
})

products = pd.DataFrame({
    'product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Headphones'],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Audio'],
    'price': [1200, 800, 400, 300, 100]
})

print("Sample Datasets:")
print("Customers:")
print(customers)
print("\nOrders:")
print(orders)
print("\nProducts:")
print(products)
```

#### Types of Merges

```python
print("\n=== MERGE TYPES ===")

# 1. Inner Join (default)
print("1. INNER JOIN - Only matching records:")
inner_merge = pd.merge(customers, orders, on='customer_id', how='inner')
print(inner_merge)

# 2. Left Join
print("\n2. LEFT JOIN - All customers, matching orders:")
left_merge = pd.merge(customers, orders, on='customer_id', how='left')
print(left_merge)

# 3. Right Join
print("\n3. RIGHT JOIN - All orders, matching customers:")
right_merge = pd.merge(customers, orders, on='customer_id', how='right')
print(right_merge)

# 4. Outer Join
print("\n4. OUTER JOIN - All records from both datasets:")
outer_merge = pd.merge(customers, orders, on='customer_id', how='outer')
print(outer_merge)

# Summary of merge results
merge_types = ['inner', 'left', 'right', 'outer']
for merge_type in merge_types:
    result = pd.merge(customers, orders, on='customer_id', how=merge_type)
    print(f"{merge_type.upper()} join: {len(result)} rows")
```

#### Advanced Merging Techniques

```python
print("\n=== ADVANCED MERGING ===")

# 1. Multiple column joins
sales_data = pd.DataFrame({
    'product': ['Laptop', 'Phone', 'Tablet'],
    'region': ['North', 'South', 'East'],
    'sales': [100, 150, 80]
})

inventory_data = pd.DataFrame({
    'product': ['Laptop', 'Phone', 'Tablet', 'Monitor'],
    'region': ['North', 'South', 'East', 'West'],
    'stock': [50, 75, 30, 20]
})

print("Multiple column merge:")
multi_merge = pd.merge(sales_data, inventory_data, on=['product', 'region'], how='outer')
print(multi_merge)

# 2. Different column names
customers_alt = customers.rename(columns={'customer_id': 'cust_id'})
print("\n2. Different column names:")
diff_names_merge = pd.merge(customers_alt, orders, 
                           left_on='cust_id', right_on='customer_id', how='inner')
print(diff_names_merge)

# 3. Index-based merging
customers_indexed = customers.set_index('customer_id')
orders_indexed = orders.set_index('customer_id')

print("\n3. Index-based merge:")
index_merge = pd.merge(customers_indexed, orders_indexed, 
                      left_index=True, right_index=True, how='inner')
print(index_merge)

# 4. Suffix handling for duplicate columns
customers_with_amount = customers.copy()
customers_with_amount['amount'] = [5000, 3000, 7000, 4000, 6000]  # Customer credit limit

print("\n4. Suffix handling:")
suffix_merge = pd.merge(customers_with_amount, orders, on='customer_id', 
                       suffixes=('_customer', '_order'))
print(suffix_merge[['name', 'amount_customer', 'product', 'amount_order']])
```

#### Join Operations

```python
print("\n=== JOIN OPERATIONS ===")

# DataFrame.join() - primarily for index-based joins
customers_idx = customers.set_index('customer_id')
orders_grouped = orders.groupby('customer_id').agg({
    'order_id': 'count',
    'amount': 'sum'
}).rename(columns={'order_id': 'total_orders', 'amount': 'total_spent'})

print("1. Basic join (left join by default):")
basic_join = customers_idx.join(orders_grouped)
print(basic_join)

print("\n2. Inner join:")
inner_join = customers_idx.join(orders_grouped, how='inner')
print(inner_join)

# Multiple DataFrame joins
payment_methods = pd.DataFrame({
    'customer_id': [1, 2, 3, 4],
    'preferred_payment': ['Credit', 'Debit', 'Credit', 'PayPal']
}).set_index('customer_id')

print("\n3. Multiple DataFrame join:")
multi_join = customers_idx.join([orders_grouped, payment_methods])
print(multi_join)
```

#### Concatenation

```python
print("\n=== CONCATENATION ===")

# Vertical concatenation (stacking rows)
customers_2023 = pd.DataFrame({
    'customer_id': [6, 7, 8],
    'name': ['Frank', 'Grace', 'Henry'],
    'city': ['Seattle', 'Portland', 'Denver']
})

print("1. Vertical concatenation:")
vertical_concat = pd.concat([customers, customers_2023], ignore_index=True)
print(vertical_concat)

# Horizontal concatenation (adding columns)
customer_details = pd.DataFrame({
    'age': [25, 30, 35, 28, 32],
    'income': [50000, 75000, 60000, 55000, 80000]
})

print("\n2. Horizontal concatenation:")
horizontal_concat = pd.concat([customers, customer_details], axis=1)
print(horizontal_concat)

# Concatenation with keys
print("\n3. Concatenation with hierarchical index:")
keyed_concat = pd.concat([customers, customers_2023], keys=['2022', '2023'])
print(keyed_concat)

# Handling mismatched columns
customers_extended = customers.copy()
customers_extended['email'] = ['alice@email.com', 'bob@email.com', 'charlie@email.com', 
                              'diana@email.com', 'eve@email.com']

print("\n4. Mismatched columns (outer join behavior):")
mismatched_concat = pd.concat([customers, customers_extended], ignore_index=True)
print(mismatched_concat)
```

#### Complex Merge Scenarios

```python
print("\n=== COMPLEX SCENARIOS ===")

# 1. Many-to-many relationships
products_suppliers = pd.DataFrame({
    'product': ['Laptop', 'Laptop', 'Phone', 'Phone', 'Tablet'],
    'supplier': ['SupplierA', 'SupplierB', 'SupplierA', 'SupplierC', 'SupplierB'],
    'cost': [1000, 950, 600, 650, 350]
})

product_orders = pd.DataFrame({
    'product': ['Laptop', 'Phone', 'Phone', 'Tablet'],
    'quantity': [10, 5, 8, 12]
})

print("1. Many-to-many merge:")
many_to_many = pd.merge(products_suppliers, product_orders, on='product')
print(many_to_many)

# 2. Merge with validation
print("\n2. Merge validation:")
try:
    # This will raise an error if not one-to-one
    validated_merge = pd.merge(customers, orders, on='customer_id', validate='one_to_one')
except Exception as e:
    print(f"Validation error: {e}")

# Correct validation for one-to-many
validated_merge = pd.merge(customers, orders, on='customer_id', validate='one_to_many')
print("Validated one-to-many merge successful")

# 3. Merge indicator
print("\n3. Merge with indicator:")
indicator_merge = pd.merge(customers, orders, on='customer_id', how='outer', indicator=True)
print(indicator_merge[['name', 'product', '_merge']])

merge_summary = indicator_merge['_merge'].value_counts()
print(f"\nMerge summary:\n{merge_summary}")

# 4. Fuzzy matching (approximate joins)
print("\n4. Approximate string matching:")
customers_typos = pd.DataFrame({
    'name': ['Alice', 'Bobby', 'Charle', 'Diana'],  # Slight variations
    'phone': ['123-456-7890', '234-567-8901', '345-678-9012', '456-789-0123']
})

# Using merge_asof for sorted data
time_series1 = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=5, freq='H'),
    'sensor1': [20, 22, 21, 23, 19]
})

time_series2 = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01 00:30:00', periods=4, freq='H'),
    'sensor2': [15, 17, 16, 18]
})

print("\n5. merge_asof for time-based joins:")
asof_merge = pd.merge_asof(time_series1, time_series2, on='timestamp', direction='nearest')
print(asof_merge)
```

#### Performance Optimization

```python
print("\n=== PERFORMANCE OPTIMIZATION ===")

# Create larger datasets for performance testing
import time

large_df1 = pd.DataFrame({
    'key': np.random.randint(1, 10000, 50000),
    'value1': np.random.randn(50000)
})

large_df2 = pd.DataFrame({
    'key': np.random.randint(1, 10000, 30000),
    'value2': np.random.randn(30000)
})

# 1. Index-based joins are faster
print("1. Performance comparison:")

# Merge on column (slower)
start_time = time.time()
column_merge = pd.merge(large_df1, large_df2, on='key')
column_time = time.time() - start_time

# Merge on index (faster)
large_df1_indexed = large_df1.set_index('key')
large_df2_indexed = large_df2.set_index('key')

start_time = time.time()
index_merge = large_df1_indexed.join(large_df2_indexed)
index_time = time.time() - start_time

print(f"Column-based merge: {column_time:.4f} seconds")
print(f"Index-based join: {index_time:.4f} seconds")
print(f"Speedup: {column_time/index_time:.1f}x")

# 2. Sort optimization
print("\n2. Sort optimization:")
unsorted_df1 = large_df1.sample(frac=1)  # Shuffle
unsorted_df2 = large_df2.sample(frac=1)  # Shuffle

start_time = time.time()
unsorted_merge = pd.merge(unsorted_df1, unsorted_df2, on='key')
unsorted_time = time.time() - start_time

sorted_df1 = large_df1.sort_values('key')
sorted_df2 = large_df2.sort_values('key')

start_time = time.time()
sorted_merge = pd.merge(sorted_df1, sorted_df2, on='key')
sorted_time = time.time() - start_time

print(f"Unsorted merge: {unsorted_time:.4f} seconds")
print(f"Sorted merge: {sorted_time:.4f} seconds")

# 3. Memory-efficient merging
def memory_efficient_merge(df1, df2, on_col, chunk_size=10000):
    """Merge large DataFrames in chunks to save memory."""
    result_chunks = []
    
    for i in range(0, len(df1), chunk_size):
        chunk = df1.iloc[i:i+chunk_size]
        merged_chunk = pd.merge(chunk, df2, on=on_col)
        result_chunks.append(merged_chunk)
    
    return pd.concat(result_chunks, ignore_index=True)

print("\n3. Memory-efficient merging implemented")
```

#### Best Practices and Guidelines

```python
print("\n=== BEST PRACTICES ===")

# 1. Choose appropriate merge type
def recommend_join_type(left_df, right_df, join_key):
    """Recommend join type based on data characteristics."""
    left_unique = left_df[join_key].nunique()
    right_unique = right_df[join_key].nunique()
    left_total = len(left_df)
    right_total = len(right_df)
    
    recommendations = []
    
    if left_total == left_unique and right_total == right_unique:
        recommendations.append("Consider inner join (1:1 relationship)")
    elif left_total == left_unique:
        recommendations.append("Consider left join (1:many relationship)")
    elif right_total == right_unique:
        recommendations.append("Consider right join (many:1 relationship)")
    else:
        recommendations.append("Many:many relationship - use outer join with caution")
    
    return recommendations

recommendations = recommend_join_type(customers, orders, 'customer_id')
print("Join recommendations:")
for rec in recommendations:
    print(f"- {rec}")

# 2. Data validation before merging
def validate_merge_keys(df1, df2, key_col):
    """Validate merge keys before performing merge."""
    validation_report = {}
    
    # Check for missing values
    validation_report['df1_missing'] = df1[key_col].isnull().sum()
    validation_report['df2_missing'] = df2[key_col].isnull().sum()
    
    # Check data types
    validation_report['df1_dtype'] = df1[key_col].dtype
    validation_report['df2_dtype'] = df2[key_col].dtype
    
    # Check for duplicates
    validation_report['df1_duplicates'] = df1[key_col].duplicated().sum()
    validation_report['df2_duplicates'] = df2[key_col].duplicated().sum()
    
    # Check overlap
    overlap = set(df1[key_col]) & set(df2[key_col])
    validation_report['overlap_count'] = len(overlap)
    validation_report['overlap_percentage'] = len(overlap) / max(df1[key_col].nunique(), df2[key_col].nunique()) * 100
    
    return validation_report

print("\n2. Merge validation:")
validation = validate_merge_keys(customers, orders, 'customer_id')
for key, value in validation.items():
    print(f"{key}: {value}")

# 3. Error handling and debugging
def safe_merge(df1, df2, **merge_kwargs):
    """Safely merge DataFrames with error handling."""
    try:
        # Validate inputs
        if df1.empty or df2.empty:
            print("Warning: One or both DataFrames are empty")
            return pd.DataFrame()
        
        # Perform merge
        result = pd.merge(df1, df2, **merge_kwargs)
        
        # Log results
        print(f"Merge successful: {len(result)} rows in result")
        print(f"Left: {len(df1)} rows, Right: {len(df2)} rows")
        
        return result
        
    except Exception as e:
        print(f"Merge failed: {str(e)}")
        return None

print("\n3. Safe merge example:")
safe_result = safe_merge(customers, orders, on='customer_id', how='left')
```

#### Common Pitfalls and Solutions

```python
print("\n=== COMMON PITFALLS ===")

# 1. Cartesian product explosion
print("1. Cartesian Product Warning:")
duplicate_customers = pd.DataFrame({
    'customer_id': [1, 1, 2, 2],  # Duplicates
    'name': ['Alice', 'Alice', 'Bob', 'Bob'],
    'segment': ['Premium', 'Regular', 'Premium', 'Regular']
})

cartesian_result = pd.merge(duplicate_customers, orders, on='customer_id')
print(f"Cartesian explosion: {len(cartesian_result)} rows from {len(duplicate_customers)} × {len(orders)}")

# Solution: Remove duplicates or use appropriate aggregation
deduplicated = duplicate_customers.drop_duplicates(subset=['customer_id'])
safe_result = pd.merge(deduplicated, orders, on='customer_id')
print(f"After deduplication: {len(safe_result)} rows")

# 2. Data type mismatches
print("\n2. Data Type Mismatch:")
orders_str_id = orders.copy()
orders_str_id['customer_id'] = orders_str_id['customer_id'].astype(str)

try:
    type_mismatch_merge = pd.merge(customers, orders_str_id, on='customer_id')
    print("Merge succeeded despite type difference")
except Exception as e:
    print(f"Type mismatch error: {e}")

# 3. Memory issues with large merges
print("\n3. Memory Management:")
def estimate_merge_memory(df1, df2):
    """Estimate memory requirements for merge."""
    df1_memory = df1.memory_usage(deep=True).sum()
    df2_memory = df2.memory_usage(deep=True).sum()
    
    # Rough estimate: worst case is cartesian product
    worst_case_rows = len(df1) * len(df2)
    avg_row_size = (df1_memory + df2_memory) / (len(df1) + len(df2))
    estimated_memory = worst_case_rows * avg_row_size
    
    return estimated_memory / (1024**2)  # Convert to MB

estimated_mb = estimate_merge_memory(customers, orders)
print(f"Estimated worst-case memory usage: {estimated_mb:.2f} MB")
```

#### Summary Guide

| Operation | Use Case | Syntax | Key Points |
|-----------|----------|--------|------------|
| **merge()** | General purpose joining | `pd.merge(df1, df2, on='key')` | Most flexible, supports all join types |
| **join()** | Index-based joining | `df1.join(df2)` | Fast for index operations, left join default |
| **concat()** | Stacking/combining | `pd.concat([df1, df2])` | For concatenation, not key-based joins |
| **merge_asof()** | Time-series joins | `pd.merge_asof(df1, df2, on='time')` | For sorted time-based data |

**Key Decision Points:**
- Use **merge()** for most scenarios requiring key-based joins
- Use **join()** when working with indexed data for better performance
- Use **concat()** for simple stacking without key matching
- Consider **merge_asof()** for time-series data with tolerance requirements

---

## Question 7

**How do you convert categorical data to numerical data in Pandas?**

### Answer

#### Theory
Converting categorical data to numerical format is a fundamental preprocessing step in data analysis and machine learning. Categorical data can be ordinal (with inherent order) or nominal (without order), and different encoding techniques are appropriate for each type. Pandas provides several built-in methods to handle these conversions efficiently.

#### Core Techniques Overview

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Create comprehensive sample dataset
np.random.seed(42)
sample_data = pd.DataFrame({
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'color': np.random.choice(['Red', 'Blue', 'Green', 'Yellow'], 1000),
    'size': np.random.choice(['Small', 'Medium', 'Large'], 1000),
    'grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 1000),
    'satisfaction': np.random.choice(['Very Low', 'Low', 'Medium', 'High', 'Very High'], 1000),
    'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 1000),
    'income': np.random.normal(50000, 15000, 1000)
})

print("Original Categorical Data:")
print(sample_data.head(10))
print(f"\nDataset shape: {sample_data.shape}")
print(f"\nData types:\n{sample_data.dtypes}")
```

#### 1. Label Encoding (Ordinal Encoding)

```python
print("\n=== LABEL ENCODING ===")

# Method 1: Using pd.Categorical with ordered categories
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
sample_data['education_label'] = pd.Categorical(
    sample_data['education'], 
    categories=education_order, 
    ordered=True
).codes

size_order = ['Small', 'Medium', 'Large']
sample_data['size_label'] = pd.Categorical(
    sample_data['size'], 
    categories=size_order, 
    ordered=True
).codes

grade_order = ['F', 'D', 'C', 'B', 'A']
sample_data['grade_label'] = pd.Categorical(
    sample_data['grade'], 
    categories=grade_order, 
    ordered=True
).codes

satisfaction_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
sample_data['satisfaction_label'] = pd.Categorical(
    sample_data['satisfaction'], 
    categories=satisfaction_order, 
    ordered=True
).codes

print("Label Encoding Results:")
label_comparison = sample_data[['education', 'education_label', 'size', 'size_label']].head()
print(label_comparison)

# Method 2: Using map() for custom encoding
education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
sample_data['education_mapped'] = sample_data['education'].map(education_mapping)

print("\nCustom mapping example:")
print(sample_data[['education', 'education_mapped']].value_counts().head())

# Method 3: Using sklearn LabelEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
sample_data['color_label'] = le.fit_transform(sample_data['color'])

# Store the mapping for inverse transformation
color_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"\nColor mapping: {color_mapping}")

# Inverse transformation
sample_data['color_decoded'] = le.inverse_transform(sample_data['color_label'])
print(f"Inverse transformation successful: {(sample_data['color'] == sample_data['color_decoded']).all()}")
```

#### 2. One-Hot Encoding (Dummy Variables)

```python
print("\n=== ONE-HOT ENCODING ===")

# Method 1: Using pd.get_dummies()
print("1. Using pd.get_dummies():")
color_dummies = pd.get_dummies(sample_data['color'], prefix='color')
region_dummies = pd.get_dummies(sample_data['region'], prefix='region')

print("Color dummies shape:", color_dummies.shape)
print(color_dummies.head())

# Combining with original data
sample_with_dummies = pd.concat([sample_data, color_dummies, region_dummies], axis=1)
print(f"\nExpanded dataset shape: {sample_with_dummies.shape}")

# Method 2: Drop first category to avoid multicollinearity
color_dummies_drop_first = pd.get_dummies(sample_data['color'], prefix='color', drop_first=True)
print(f"\n2. Drop first category (shape: {color_dummies_drop_first.shape}):")
print(color_dummies_drop_first.head())

# Method 3: Using sklearn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, drop='first')
region_encoded = ohe.fit_transform(sample_data[['region']])
region_columns = [f'region_{cat}' for cat in ohe.categories_[0][1:]]  # Skip first due to drop='first'

region_ohe_df = pd.DataFrame(region_encoded, columns=region_columns, index=sample_data.index)
print(f"\n3. Sklearn OneHotEncoder shape: {region_ohe_df.shape}")
print(region_ohe_df.head())

# Method 4: Handling multiple categorical columns simultaneously
categorical_columns = ['color', 'region']
all_dummies = pd.get_dummies(sample_data[categorical_columns], drop_first=True)
print(f"\n4. Multiple columns at once shape: {all_dummies.shape}")
print(all_dummies.columns.tolist())
```

#### 3. Binary Encoding

```python
print("\n=== BINARY ENCODING ===")

# Implementing binary encoding manually
def binary_encode(series, n_bits=None):
    """Convert categorical series to binary encoding."""
    # Create label encoding first
    le = LabelEncoder()
    label_encoded = le.fit_transform(series)
    
    # Determine number of bits needed
    max_val = max(label_encoded)
    if n_bits is None:
        n_bits = len(bin(max_val)[2:])  # Remove '0b' prefix
    
    # Convert to binary
    binary_df = pd.DataFrame(index=series.index)
    for i in range(n_bits):
        binary_df[f'{series.name}_bin_{i}'] = (label_encoded >> i) & 1
    
    return binary_df, le

# Apply binary encoding
color_binary, color_le = binary_encode(sample_data['color'])
print("Binary encoding for color:")
print(color_binary.head())

# Comparison of memory usage
print(f"\nMemory usage comparison:")
print(f"Original color column: {sample_data['color'].memory_usage(deep=True)} bytes")
print(f"One-hot encoded: {pd.get_dummies(sample_data['color']).memory_usage(deep=True).sum()} bytes")
print(f"Binary encoded: {color_binary.memory_usage(deep=True).sum()} bytes")
print(f"Label encoded: {sample_data['color_label'].memory_usage(deep=True)} bytes")
```

#### 4. Target Encoding (Mean Encoding)

```python
print("\n=== TARGET ENCODING ===")

# Create a target variable for demonstration
np.random.seed(42)
sample_data['target'] = (
    sample_data['education_label'] * 0.3 + 
    sample_data['satisfaction_label'] * 0.4 + 
    np.random.normal(0, 1, len(sample_data))
)

def target_encode(df, categorical_col, target_col, smoothing=1.0):
    """
    Perform target encoding with smoothing to prevent overfitting.
    
    smoothing: Higher values increase regularization
    """
    # Calculate global mean
    global_mean = df[target_col].mean()
    
    # Calculate category means and counts
    category_stats = df.groupby(categorical_col)[target_col].agg(['mean', 'count'])
    
    # Apply smoothing formula
    smoothed_means = (
        category_stats['count'] * category_stats['mean'] + smoothing * global_mean
    ) / (category_stats['count'] + smoothing)
    
    # Map back to original data
    encoded_values = df[categorical_col].map(smoothed_means)
    
    return encoded_values, smoothed_means

# Apply target encoding
region_target_encoded, region_mapping = target_encode(sample_data, 'region', 'target', smoothing=10)
sample_data['region_target_encoded'] = region_target_encoded

print("Target encoding mapping:")
print(region_mapping.sort_values(ascending=False))

# Cross-validation target encoding to prevent overfitting
def cv_target_encode(df, categorical_col, target_col, cv_folds=5, smoothing=1.0):
    """Target encoding with cross-validation to prevent data leakage."""
    from sklearn.model_selection import KFold
    
    encoded_values = np.zeros(len(df))
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Calculate encoding on training fold
        global_mean = train_df[target_col].mean()
        category_stats = train_df.groupby(categorical_col)[target_col].agg(['mean', 'count'])
        
        smoothed_means = (
            category_stats['count'] * category_stats['mean'] + smoothing * global_mean
        ) / (category_stats['count'] + smoothing)
        
        # Apply to validation fold
        encoded_values[val_idx] = val_df[categorical_col].map(smoothed_means).fillna(global_mean)
    
    return encoded_values

sample_data['region_cv_target_encoded'] = cv_target_encode(sample_data, 'region', 'target')

print(f"\nTarget encoding comparison:")
print(f"Standard target encoding mean: {sample_data['region_target_encoded'].mean():.4f}")
print(f"CV target encoding mean: {sample_data['region_cv_target_encoded'].mean():.4f}")
print(f"Target variable mean: {sample_data['target'].mean():.4f}")
```

#### 5. Frequency/Count Encoding

```python
print("\n=== FREQUENCY ENCODING ===")

# Method 1: Simple frequency encoding
color_counts = sample_data['color'].value_counts()
sample_data['color_frequency'] = sample_data['color'].map(color_counts)

region_counts = sample_data['region'].value_counts()
sample_data['region_frequency'] = sample_data['region'].map(region_counts)

print("Frequency encoding results:")
freq_comparison = sample_data[['color', 'color_frequency', 'region', 'region_frequency']].head()
print(freq_comparison)

# Method 2: Normalized frequency encoding
total_samples = len(sample_data)
sample_data['color_freq_normalized'] = sample_data['color_frequency'] / total_samples
sample_data['region_freq_normalized'] = sample_data['region_frequency'] / total_samples

print(f"\nNormalized frequencies (sum to 1.0):")
print(f"Color frequencies sum: {sample_data['color_freq_normalized'].sum():.4f}")
print(f"Region frequencies sum: {sample_data['region_freq_normalized'].sum():.4f}")

# Method 3: Rank-based frequency encoding
color_ranks = sample_data['color'].value_counts().rank(method='dense', ascending=False)
sample_data['color_rank'] = sample_data['color'].map(color_ranks)

print(f"\nRank-based encoding:")
rank_comparison = sample_data[['color', 'color_frequency', 'color_rank']].drop_duplicates().sort_values('color_frequency', ascending=False)
print(rank_comparison)
```

#### 6. Advanced Encoding Techniques

```python
print("\n=== ADVANCED TECHNIQUES ===")

# 1. Hashing Encoding (useful for high cardinality)
def hash_encode(series, n_features=8):
    """Simple hash encoding implementation."""
    hash_df = pd.DataFrame(index=series.index)
    
    for i in range(n_features):
        hash_df[f'{series.name}_hash_{i}'] = series.apply(
            lambda x: hash(str(x) + str(i)) % 2
        )
    
    return hash_df

# Apply hash encoding
color_hashed = hash_encode(sample_data['color'], n_features=6)
print("1. Hash encoding (first 5 rows):")
print(color_hashed.head())

# 2. Embedding-like encoding using random projections
def random_projection_encode(series, n_dimensions=5, random_state=42):
    """Create random projection encoding for categorical variables."""
    np.random.seed(random_state)
    unique_values = series.unique()
    
    # Create random embeddings for each unique value
    embeddings = {}
    for value in unique_values:
        embeddings[value] = np.random.normal(0, 1, n_dimensions)
    
    # Map to DataFrame
    embedding_df = pd.DataFrame(index=series.index)
    for i in range(n_dimensions):
        embedding_df[f'{series.name}_embed_{i}'] = series.map({v: emb[i] for v, emb in embeddings.items()})
    
    return embedding_df, embeddings

region_embeddings, region_embedding_map = random_projection_encode(sample_data['region'], n_dimensions=4)
print(f"\n2. Random projection encoding shape: {region_embeddings.shape}")
print(region_embeddings.head())

# 3. Polynomial features for categorical interactions
def categorical_polynomial_features(df, cat_cols, degree=2):
    """Create polynomial features from categorical variables."""
    # First convert to dummy variables
    dummies_list = []
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        dummies_list.append(dummies)
    
    # Combine all dummies
    all_dummies = pd.concat(dummies_list, axis=1)
    
    # Create interaction terms
    if degree >= 2:
        interactions = pd.DataFrame(index=df.index)
        cols = all_dummies.columns
        
        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                interaction_name = f"{col1}_x_{col2}"
                interactions[interaction_name] = all_dummies[col1] * all_dummies[col2]
        
        all_dummies = pd.concat([all_dummies, interactions], axis=1)
    
    return all_dummies

poly_features = categorical_polynomial_features(sample_data, ['color', 'size'], degree=2)
print(f"\n3. Polynomial features shape: {poly_features.shape}")
print(f"Sample interaction columns: {poly_features.columns[-5:].tolist()}")
```

#### 7. Choosing the Right Encoding Method

```python
print("\n=== ENCODING SELECTION GUIDE ===")

def encoding_selector(series, target=None, cardinality_threshold=10):
    """
    Recommend encoding method based on data characteristics.
    """
    cardinality = series.nunique()
    has_order = False  # This would need domain knowledge in practice
    has_target = target is not None
    
    recommendations = []
    
    # Check for natural ordering (simplified heuristic)
    unique_vals = series.unique()
    ordinal_patterns = ['small', 'medium', 'large', 'low', 'high', 'a', 'b', 'c', 'd', 'f']
    if any(any(pattern in str(val).lower() for pattern in ordinal_patterns) for val in unique_vals):
        has_order = True
    
    if cardinality == 2:
        recommendations.append("Label Encoding (binary)")
    elif cardinality <= cardinality_threshold and not has_order:
        recommendations.append("One-Hot Encoding")
    elif cardinality <= cardinality_threshold and has_order:
        recommendations.append("Ordinal/Label Encoding")
    elif cardinality > cardinality_threshold:
        recommendations.append("Target Encoding or Frequency Encoding")
        if cardinality > 50:
            recommendations.append("Hash Encoding or Embeddings")
    
    if has_target:
        recommendations.append("Consider Target Encoding")
    
    return {
        'cardinality': cardinality,
        'has_natural_order': has_order,
        'recommendations': recommendations
    }

# Analyze each categorical column
categorical_columns = ['education', 'color', 'size', 'grade', 'satisfaction', 'region']

print("Encoding recommendations:")
for col in categorical_columns:
    analysis = encoding_selector(sample_data[col], target=sample_data['target'])
    print(f"\n{col}:")
    print(f"  Cardinality: {analysis['cardinality']}")
    print(f"  Has order: {analysis['has_natural_order']}")
    print(f"  Recommendations: {', '.join(analysis['recommendations'])}")

# Performance comparison
print(f"\n=== PERFORMANCE COMPARISON ===")

# Create a larger dataset for timing
large_categorical = pd.Series(np.random.choice(['A', 'B', 'C', 'D', 'E'], 100000))

import time

# Label encoding timing
start_time = time.time()
label_result = large_categorical.astype('category').cat.codes
label_time = time.time() - start_time

# One-hot encoding timing
start_time = time.time()
onehot_result = pd.get_dummies(large_categorical)
onehot_time = time.time() - start_time

# Frequency encoding timing
start_time = time.time()
freq_counts = large_categorical.value_counts()
freq_result = large_categorical.map(freq_counts)
freq_time = time.time() - start_time

print(f"Performance on 100K samples:")
print(f"Label encoding: {label_time:.4f} seconds")
print(f"One-hot encoding: {onehot_time:.4f} seconds")
print(f"Frequency encoding: {freq_time:.4f} seconds")

# Memory usage comparison
print(f"\nMemory usage comparison:")
print(f"Original: {large_categorical.memory_usage(deep=True):,} bytes")
print(f"Label encoded: {label_result.memory_usage(deep=True):,} bytes")
print(f"One-hot encoded: {onehot_result.memory_usage(deep=True).sum():,} bytes")
print(f"Frequency encoded: {freq_result.memory_usage(deep=True):,} bytes")
```

#### 8. Complete Preprocessing Pipeline

```python
print("\n=== COMPLETE PREPROCESSING PIPELINE ===")

class CategoricalEncoder:
    """
    Comprehensive categorical encoding pipeline.
    """
    
    def __init__(self):
        self.encoding_maps = {}
        self.encoders = {}
        self.target_encodings = {}
        
    def fit_transform(self, df, encoding_config, target_col=None):
        """
        Apply multiple encoding strategies based on configuration.
        
        encoding_config: dict with column names as keys and encoding types as values
        """
        df_encoded = df.copy()
        
        for col, encoding_type in encoding_config.items():
            if col not in df.columns:
                continue
                
            if encoding_type == 'label':
                df_encoded[f'{col}_label'] = pd.Categorical(df[col]).codes
                self.encoding_maps[col] = dict(enumerate(pd.Categorical(df[col]).categories))
                
            elif encoding_type == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                self.encoding_maps[col] = dummies.columns.tolist()
                
            elif encoding_type == 'frequency':
                freq_map = df[col].value_counts()
                df_encoded[f'{col}_freq'] = df[col].map(freq_map)
                self.encoding_maps[col] = freq_map.to_dict()
                
            elif encoding_type == 'target' and target_col:
                encoded_vals, mapping = target_encode(df, col, target_col)
                df_encoded[f'{col}_target'] = encoded_vals
                self.target_encodings[col] = mapping
                
            elif encoding_type == 'ordinal':
                # Requires predefined order - simplified example
                if col == 'education':
                    order = ['High School', 'Bachelor', 'Master', 'PhD']
                elif col == 'size':
                    order = ['Small', 'Medium', 'Large']
                elif col == 'grade':
                    order = ['F', 'D', 'C', 'B', 'A']
                else:
                    order = sorted(df[col].unique())
                
                df_encoded[f'{col}_ordinal'] = pd.Categorical(
                    df[col], categories=order, ordered=True
                ).codes
                self.encoding_maps[col] = dict(enumerate(order))
        
        return df_encoded
    
    def transform(self, df):
        """Apply learned encodings to new data."""
        df_encoded = df.copy()
        
        for col, mapping in self.encoding_maps.items():
            if col in df.columns:
                if isinstance(mapping, dict) and all(isinstance(k, int) for k in mapping.keys()):
                    # Label encoding
                    reverse_map = {v: k for k, v in mapping.items()}
                    df_encoded[f'{col}_label'] = df[col].map(reverse_map)
                elif isinstance(mapping, list):
                    # One-hot encoding
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    for dummy_col in mapping:
                        if dummy_col not in dummies.columns:
                            dummies[dummy_col] = 0
                    df_encoded = pd.concat([df_encoded, dummies[mapping]], axis=1)
        
        return df_encoded

# Example usage
encoder = CategoricalEncoder()

encoding_config = {
    'education': 'ordinal',
    'color': 'onehot',
    'size': 'ordinal',
    'grade': 'ordinal',
    'region': 'target',
    'satisfaction': 'frequency'
}

# Apply encoding
sample_encoded = encoder.fit_transform(sample_data, encoding_config, target_col='target')

print("Pipeline encoding results:")
print(f"Original shape: {sample_data.shape}")
print(f"Encoded shape: {sample_encoded.shape}")
print(f"New columns: {[col for col in sample_encoded.columns if col not in sample_data.columns]}")

# Validation
print(f"\nEncoding validation:")
print(f"No missing values in encoded data: {sample_encoded.isnull().sum().sum() == 0}")
print(f"All categorical columns processed: {all(col in encoding_config for col in categorical_columns)}")
```

#### Best Practices and Common Pitfalls

```python
print("\n=== BEST PRACTICES ===")

# 1. Handle missing values before encoding
def handle_missing_before_encoding(df, strategy='mode'):
    """Handle missing values in categorical data."""
    df_clean = df.copy()
    
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().any():
            if strategy == 'mode':
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            elif strategy == 'unknown':
                fill_value = 'Unknown'
            else:
                fill_value = strategy
            
            df_clean[col] = df[col].fillna(fill_value)
            print(f"Filled {col} missing values with: {fill_value}")
    
    return df_clean

# 2. Prevent data leakage in target encoding
def safe_target_encoding(X_train, X_test, y_train, categorical_cols, smoothing=1.0):
    """Perform target encoding without data leakage."""
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    global_mean = y_train.mean()
    encodings = {}
    
    for col in categorical_cols:
        # Calculate encoding on training data only
        category_stats = X_train.groupby(col)[y_train.name].agg(['mean', 'count'])
        
        smoothed_means = (
            category_stats['count'] * category_stats['mean'] + smoothing * global_mean
        ) / (category_stats['count'] + smoothing)
        
        encodings[col] = smoothed_means
        
        # Apply to both train and test
        X_train_encoded[f'{col}_target'] = X_train[col].map(smoothed_means).fillna(global_mean)
        X_test_encoded[f'{col}_target'] = X_test[col].map(smoothed_means).fillna(global_mean)
    
    return X_train_encoded, X_test_encoded, encodings

print("Best practices implemented for safe encoding")

# 3. Common pitfalls and solutions
print("\nCommon Pitfalls and Solutions:")
pitfalls = [
    "1. Data leakage in target encoding → Use cross-validation or separate train/test",
    "2. Curse of dimensionality with one-hot → Use target/frequency encoding for high cardinality",
    "3. Information loss with label encoding → Preserve ordinal relationships where they exist",
    "4. Overfitting with target encoding → Apply smoothing and regularization",
    "5. Memory issues with sparse data → Use sparse matrices for one-hot encoding",
    "6. Inconsistent encodings → Always fit on training data and transform test data",
    "7. Missing category handling → Plan for unseen categories in production"
]

for pitfall in pitfalls:
    print(pitfall)
```

#### Summary Decision Matrix

| Encoding Method | Best For | Pros | Cons | Use When |
|----------------|----------|------|------|----------|
| **Label Encoding** | Ordinal data, binary categories | Memory efficient, preserves order | Assumes ordinality | Natural ordering exists |
| **One-Hot Encoding** | Nominal data, low cardinality | No false ordinality, interpretable | High dimensionality, sparse | < 10 categories, nominal data |
| **Target Encoding** | High cardinality, supervised learning | Captures target relationship | Overfitting risk, requires target | High cardinality + target available |
| **Frequency Encoding** | Any categorical data | Simple, no overfitting | May not capture relationships | Quick baseline, unsupervised |
| **Binary Encoding** | Medium-high cardinality | Reduces dimensionality vs one-hot | Less interpretable | 10-50 categories |
| **Hash Encoding** | Very high cardinality | Handles any cardinality | Hash collisions possible | > 100 categories |

**Key Decision Rules:**
- **Cardinality < 5**: One-hot encoding
- **Cardinality 5-15**: One-hot or target encoding
- **Cardinality > 15**: Target, frequency, or hash encoding
- **Ordinal data**: Always use ordinal/label encoding
- **With target variable**: Consider target encoding
- **Memory constraints**: Use label or frequency encoding

---

## Question 8

**What is the purpose of the apply() function in Pandas?**

### Answer

#### Theory
The `apply()` function is one of the most powerful and versatile methods in Pandas for data transformation. It allows you to apply custom functions to DataFrame columns, rows, or entire DataFrames, providing flexibility for complex data manipulations that can't be achieved with built-in vectorized operations. Understanding `apply()` is crucial for advanced data processing workflows.

#### Core Concepts and Basic Usage

```python
import pandas as pd
import numpy as np
import time
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

# Create comprehensive sample data
np.random.seed(42)
sample_data = pd.DataFrame({
    'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Wilson', 'Eve Davis'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 75000, 60000, 55000, 80000],
    'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Engineering'],
    'start_date': pd.to_datetime(['2020-01-15', '2019-03-22', '2021-06-10', '2020-09-05', '2018-11-30']),
    'performance_score': [8.5, 7.2, 9.1, 6.8, 8.9],
    'bonus_eligible': [True, False, True, True, False]
})

# Add more complex data
sample_data['full_address'] = [
    '123 Main St, New York, NY 10001',
    '456 Oak Ave, Los Angeles, CA 90210', 
    '789 Pine Rd, Chicago, IL 60601',
    '321 Elm St, Houston, TX 77001',
    '654 Maple Dr, Phoenix, AZ 85001'
]

sample_data['skills'] = [
    ['Python', 'SQL', 'Machine Learning'],
    ['Marketing', 'Analytics', 'Communication'],
    ['Python', 'Java', 'DevOps'],
    ['Sales', 'CRM', 'Negotiation'],
    ['Python', 'Data Science', 'Statistics']
]

print("Sample Dataset:")
print(sample_data)
print(f"\nDataFrame shape: {sample_data.shape}")
print(f"Data types:\n{sample_data.dtypes}")
```

#### 1. Column-wise Apply (axis=0)

```python
print("\n=== COLUMN-WISE APPLY (axis=0) ===")

# Basic column operations
print("1. Basic statistical operations:")
numeric_stats = sample_data[['age', 'salary', 'performance_score']].apply(np.mean)
print("Column means:")
print(numeric_stats)

# Custom function on columns
def coefficient_of_variation(series):
    """Calculate coefficient of variation (CV)."""
    return series.std() / series.mean() if series.mean() != 0 else 0

cv_results = sample_data[['age', 'salary', 'performance_score']].apply(coefficient_of_variation)
print(f"\nCoefficient of Variation:")
print(cv_results)

# Multiple statistics at once
def comprehensive_stats(series):
    """Return comprehensive statistics for a numeric series."""
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series({
            'count': len(series),
            'unique': series.nunique(),
            'mode': series.mode().iloc[0] if not series.mode().empty else None
        })
    
    return pd.Series({
        'count': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'range': series.max() - series.min(),
        'cv': series.std() / series.mean() if series.mean() != 0 else 0,
        'skewness': series.skew(),
        'outliers': len(series[np.abs(series - series.mean()) > 2 * series.std()])
    })

print("\n2. Comprehensive statistics:")
all_stats = sample_data[['age', 'salary', 'performance_score']].apply(comprehensive_stats)
print(all_stats)

# String operations on columns
print("\n3. String operations:")
string_stats = sample_data[['name', 'department']].apply(lambda col: {
    'avg_length': col.str.len().mean(),
    'max_length': col.str.len().max(),
    'contains_uppercase': col.str.contains(r'[A-Z]').sum(),
    'unique_words': len(set(' '.join(col).split()))
})

for col, stats in string_stats.items():
    print(f"{col}: {stats}")
```

#### 2. Row-wise Apply (axis=1)

```python
print("\n=== ROW-WISE APPLY (axis=1) ===")

# Basic row operations
print("1. Creating derived columns:")

# Calculate experience years
def calculate_experience(row):
    """Calculate years of experience from start date."""
    today = pd.Timestamp.now()
    return (today - row['start_date']).days / 365.25

sample_data['experience_years'] = sample_data.apply(calculate_experience, axis=1)

# Salary grade based on multiple factors
def determine_salary_grade(row):
    """Determine salary grade based on multiple factors."""
    base_score = 0
    
    # Age factor
    if row['age'] < 30:
        base_score += 1
    elif row['age'] < 35:
        base_score += 2
    else:
        base_score += 3
    
    # Performance factor
    if row['performance_score'] >= 9:
        base_score += 3
    elif row['performance_score'] >= 8:
        base_score += 2
    elif row['performance_score'] >= 7:
        base_score += 1
    
    # Experience factor
    if row['experience_years'] >= 3:
        base_score += 2
    elif row['experience_years'] >= 1:
        base_score += 1
    
    # Determine grade
    if base_score >= 7:
        return 'Senior'
    elif base_score >= 5:
        return 'Mid-Level'
    else:
        return 'Junior'

sample_data['salary_grade'] = sample_data.apply(determine_salary_grade, axis=1)

print("Derived columns:")
print(sample_data[['name', 'age', 'performance_score', 'experience_years', 'salary_grade']])

# Complex string processing
print("\n2. Complex string processing:")

def extract_address_components(row):
    """Extract city and state from full address."""
    address_parts = row['full_address'].split(', ')
    if len(address_parts) >= 3:
        city = address_parts[1]
        state_zip = address_parts[2].split(' ')
        state = state_zip[0] if state_zip else 'Unknown'
        return pd.Series({'city': city, 'state': state})
    return pd.Series({'city': 'Unknown', 'state': 'Unknown'})

address_components = sample_data.apply(extract_address_components, axis=1)
sample_data = pd.concat([sample_data, address_components], axis=1)

print("Address extraction:")
print(sample_data[['name', 'full_address', 'city', 'state']])

# Working with lists in rows
print("\n3. Processing list data:")

def analyze_skills(row):
    """Analyze employee skills."""
    skills = row['skills']
    return pd.Series({
        'skill_count': len(skills),
        'has_python': 'Python' in skills,
        'technical_skills': sum(1 for skill in skills if skill.lower() in ['python', 'sql', 'java', 'machine learning', 'data science', 'statistics', 'devops']),
        'primary_skill': skills[0] if skills else None
    })

skill_analysis = sample_data.apply(analyze_skills, axis=1)
sample_data = pd.concat([sample_data, skill_analysis], axis=1)

print("Skill analysis:")
print(sample_data[['name', 'skills', 'skill_count', 'has_python', 'technical_skills']])
```

#### 3. Apply vs Other Methods Comparison

```python
print("\n=== APPLY vs OTHER METHODS ===")

# Create larger dataset for performance comparison
large_data = pd.DataFrame({
    'values': np.random.randn(100000),
    'categories': np.random.choice(['A', 'B', 'C'], 100000),
    'multiplier': np.random.randint(1, 10, 100000)
})

print("Performance comparison on 100K rows:")

# Method 1: Using apply
start_time = time.time()
result_apply = large_data['values'].apply(lambda x: x**2 if x > 0 else 0)
apply_time = time.time() - start_time

# Method 2: Using vectorized operations
start_time = time.time()
result_vectorized = np.where(large_data['values'] > 0, large_data['values']**2, 0)
vectorized_time = time.time() - start_time

# Method 3: Using numpy directly
start_time = time.time()
mask = large_data['values'].values > 0
result_numpy = np.where(mask, large_data['values'].values**2, 0)
numpy_time = time.time() - start_time

print(f"Apply method: {apply_time:.4f} seconds")
print(f"Vectorized method: {vectorized_time:.4f} seconds")
print(f"NumPy method: {numpy_time:.4f} seconds")
print(f"Vectorized speedup: {apply_time/vectorized_time:.1f}x")
print(f"NumPy speedup: {apply_time/numpy_time:.1f}x")

# Verify results are identical
print(f"Results identical: {np.allclose(result_apply, result_vectorized)}")

# When apply is necessary
print("\n=== WHEN APPLY IS NECESSARY ===")

# Complex conditional logic that can't be vectorized easily
def complex_business_logic(row):
    """Complex business logic that requires apply."""
    base_value = row['values']
    category = row['categories']
    multiplier = row['multiplier']
    
    # Multi-step conditional logic
    if category == 'A':
        if base_value > 0:
            result = base_value * multiplier * 1.5
        else:
            result = abs(base_value) * multiplier * 0.8
    elif category == 'B':
        if base_value > 1:
            result = np.log(base_value) * multiplier
        elif base_value > 0:
            result = base_value**2 * multiplier
        else:
            result = 0
    else:  # Category C
        result = base_value * multiplier + np.sin(base_value)
    
    return min(max(result, -100), 100)  # Clamp between -100 and 100

# This type of logic is difficult to vectorize
start_time = time.time()
complex_result = large_data.apply(complex_business_logic, axis=1)
complex_time = time.time() - start_time

print(f"Complex business logic with apply: {complex_time:.4f} seconds")
print(f"Sample results: {complex_result.head()}")
```

#### 4. Advanced Apply Patterns

```python
print("\n=== ADVANCED APPLY PATTERNS ===")

# 1. Apply with additional arguments
def calculate_adjusted_salary(row, inflation_rate=0.03, bonus_multiplier=1.2):
    """Calculate adjusted salary with external parameters."""
    base_salary = row['salary']
    experience = row['experience_years']
    performance = row['performance_score']
    
    # Adjust for inflation
    adjusted_salary = base_salary * (1 + inflation_rate * experience)
    
    # Performance bonus
    if performance >= 8.5:
        adjusted_salary *= bonus_multiplier
    elif performance >= 7.5:
        adjusted_salary *= (bonus_multiplier - 0.1)
    
    return adjusted_salary

sample_data['adjusted_salary'] = sample_data.apply(
    calculate_adjusted_salary, 
    axis=1, 
    inflation_rate=0.035,
    bonus_multiplier=1.25
)

print("1. Apply with additional arguments:")
print(sample_data[['name', 'salary', 'adjusted_salary']].head())

# 2. Apply returning multiple values
def comprehensive_employee_analysis(row):
    """Return multiple calculated values."""
    return pd.Series({
        'salary_per_experience': row['salary'] / max(row['experience_years'], 0.5),
        'performance_rank': 'High' if row['performance_score'] >= 8.5 else 'Medium' if row['performance_score'] >= 7.0 else 'Low',
        'retention_risk': 'High' if row['salary'] < 60000 and row['performance_score'] > 8.0 else 'Low',
        'promotion_candidate': row['performance_score'] > 8.0 and row['experience_years'] > 2
    })

employee_analysis = sample_data.apply(comprehensive_employee_analysis, axis=1)
sample_data = pd.concat([sample_data, employee_analysis], axis=1)

print("\n2. Multiple return values:")
print(sample_data[['name', 'salary_per_experience', 'performance_rank', 'retention_risk', 'promotion_candidate']].head())

# 3. Conditional apply with error handling
def safe_division_with_logging(row):
    """Safely perform division with error handling."""
    try:
        result = row['salary'] / row['age']
        if np.isinf(result) or np.isnan(result):
            return 0
        return result
    except ZeroDivisionError:
        print(f"Division by zero for row: {row.name}")
        return 0
    except Exception as e:
        print(f"Unexpected error for row {row.name}: {e}")
        return None

sample_data['salary_age_ratio'] = sample_data.apply(safe_division_with_logging, axis=1)

print("\n3. Safe operations:")
print(sample_data[['name', 'salary', 'age', 'salary_age_ratio']].head())

# 4. Apply with groupby
print("\n4. Apply with GroupBy:")

def department_statistics(group):
    """Calculate department-level statistics."""
    return pd.Series({
        'avg_salary': group['salary'].mean(),
        'salary_std': group['salary'].std(),
        'avg_performance': group['performance_score'].mean(),
        'headcount': len(group),
        'total_experience': group['experience_years'].sum(),
        'python_developers': group['has_python'].sum()
    })

dept_stats = sample_data.groupby('department').apply(department_statistics)
print("Department statistics:")
print(dept_stats)

# 5. Rolling apply for time series
print("\n5. Rolling apply:")

# Create time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'sales': np.random.normal(1000, 200, 100) + np.sin(np.arange(100) * 2 * np.pi / 30) * 100
})

def custom_volatility(window):
    """Calculate custom volatility measure."""
    if len(window) < 2:
        return np.nan
    return np.std(np.diff(window)) / np.mean(window)

ts_data['volatility'] = ts_data['sales'].rolling(window=7).apply(custom_volatility)

print("Time series with custom rolling function:")
print(ts_data.head(10))
```

#### 5. Lambda Functions with Apply

```python
print("\n=== LAMBDA FUNCTIONS WITH APPLY ===")

# Simple lambda examples
print("1. Simple lambda transformations:")

# String operations
sample_data['name_length'] = sample_data['name'].apply(lambda x: len(x))
sample_data['first_name'] = sample_data['name'].apply(lambda x: x.split()[0])
sample_data['initials'] = sample_data['name'].apply(lambda x: ''.join([word[0] for word in x.split()]))

print("String transformations:")
print(sample_data[['name', 'name_length', 'first_name', 'initials']].head())

# Numeric operations
sample_data['salary_category'] = sample_data['salary'].apply(
    lambda x: 'High' if x >= 70000 else 'Medium' if x >= 55000 else 'Low'
)

sample_data['performance_letter'] = sample_data['performance_score'].apply(
    lambda x: 'A' if x >= 9 else 'B' if x >= 8 else 'C' if x >= 7 else 'D'
)

print("\n2. Categorical transformations:")
print(sample_data[['name', 'salary', 'salary_category', 'performance_score', 'performance_letter']].head())

# Complex lambda with conditional logic
sample_data['bonus_amount'] = sample_data.apply(
    lambda row: row['salary'] * 0.15 if row['performance_score'] >= 9 
                else row['salary'] * 0.10 if row['performance_score'] >= 8 
                else row['salary'] * 0.05 if row['performance_score'] >= 7 
                else 0,
    axis=1
)

print("\n3. Complex conditional lambda:")
print(sample_data[['name', 'salary', 'performance_score', 'bonus_amount']].head())

# Lambda with external data
department_multipliers = {'Engineering': 1.2, 'Marketing': 1.1, 'Sales': 1.15}

sample_data['adjusted_bonus'] = sample_data.apply(
    lambda row: row['bonus_amount'] * department_multipliers.get(row['department'], 1.0),
    axis=1
)

print("\n4. Lambda with external dictionary:")
print(sample_data[['name', 'department', 'bonus_amount', 'adjusted_bonus']].head())
```

#### 6. Performance Optimization Techniques

```python
print("\n=== PERFORMANCE OPTIMIZATION ===")

# 1. Avoid apply when vectorized operations are possible
print("1. Vectorization vs Apply:")

test_data = pd.DataFrame({
    'A': np.random.randn(50000),
    'B': np.random.randn(50000)
})

# Slow: Using apply
start_time = time.time()
result_slow = test_data.apply(lambda row: row['A'] + row['B'], axis=1)
slow_time = time.time() - start_time

# Fast: Vectorized operation
start_time = time.time()
result_fast = test_data['A'] + test_data['B']
fast_time = time.time() - start_time

print(f"Apply method: {slow_time:.4f} seconds")
print(f"Vectorized method: {fast_time:.4f} seconds")
print(f"Speedup: {slow_time/fast_time:.1f}x")

# 2. Use raw=True for numpy operations
print("\n2. raw=True optimization:")

start_time = time.time()
result_pandas = test_data['A'].apply(np.sqrt)
pandas_time = time.time() - start_time

start_time = time.time()
result_raw = test_data['A'].apply(np.sqrt, raw=True)
raw_time = time.time() - start_time

print(f"Without raw=True: {pandas_time:.4f} seconds")
print(f"With raw=True: {raw_time:.4f} seconds")
print(f"Speedup: {pandas_time/raw_time:.1f}x")

# 3. Consider map() for element-wise transformations
mapping_dict = {i: i**2 for i in range(-1000, 1001)}
test_series = pd.Series(np.random.randint(-1000, 1000, 10000))

start_time = time.time()
result_apply = test_series.apply(lambda x: x**2)
apply_time = time.time() - start_time

start_time = time.time()
result_map = test_series.map(mapping_dict)
map_time = time.time() - start_time

print(f"\n3. Map vs Apply for dictionary lookup:")
print(f"Apply method: {apply_time:.4f} seconds")
print(f"Map method: {map_time:.4f} seconds")
print(f"Map speedup: {apply_time/map_time:.1f}x")

# 4. Memory usage considerations
print(f"\n4. Memory usage:")
print(f"Original DataFrame memory: {sample_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
print(f"Large test DataFrame memory: {test_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
```

#### 7. Error Handling and Debugging

```python
print("\n=== ERROR HANDLING AND DEBUGGING ===")

# Create data with potential issues
problematic_data = pd.DataFrame({
    'numbers': [1, 2, 0, -1, np.nan, 'invalid', 5],
    'strings': ['hello', '', None, 'world', 'test', 123, 'end'],
    'mixed': [1, 'two', 3.0, None, [1, 2], {'key': 'value'}, 'seven']
})

print("Problematic data:")
print(problematic_data)

# Robust apply function with error handling
def robust_string_processing(value):
    """Process string values with comprehensive error handling."""
    try:
        # Convert to string first
        str_value = str(value)
        
        # Handle None/nan
        if str_value.lower() in ['none', 'nan']:
            return 'MISSING'
        
        # Handle empty strings
        if len(str_value.strip()) == 0:
            return 'EMPTY'
        
        # Process valid strings
        return str_value.upper()
        
    except Exception as e:
        print(f"Error processing value {value}: {e}")
        return 'ERROR'

problematic_data['processed_strings'] = problematic_data['strings'].apply(robust_string_processing)

print("\nRobust string processing:")
print(problematic_data[['strings', 'processed_strings']])

# Debugging apply functions
def debug_numeric_operation(value):
    """Numeric operation with debugging."""
    print(f"Processing value: {value} (type: {type(value)})")
    
    try:
        # Try to convert to numeric
        numeric_value = pd.to_numeric(value, errors='coerce')
        
        if pd.isna(numeric_value):
            print(f"  → Could not convert to numeric, returning 0")
            return 0
        
        # Perform operation
        result = numeric_value ** 2
        print(f"  → Squared result: {result}")
        return result
        
    except Exception as e:
        print(f"  → Error: {e}")
        return -1

print("\n7. Debugging apply (showing first 3 rows):")
debug_results = problematic_data['numbers'].head(3).apply(debug_numeric_operation)

# Using progress bars for long operations
try:
    from tqdm import tqdm
    tqdm.pandas()
    
    def slow_operation(x):
        """Simulate slow operation."""
        time.sleep(0.001)  # Simulate processing time
        return x ** 2
    
    print("\n8. Progress bar example (if tqdm is available):")
    large_series = pd.Series(range(100))
    # result_with_progress = large_series.progress_apply(slow_operation)
    print("Progress bar would show here with tqdm.pandas()")
    
except ImportError:
    print("\n8. Install tqdm for progress bars: pip install tqdm")
```

#### 8. Best Practices and Common Pitfalls

```python
print("\n=== BEST PRACTICES ===")

best_practices = [
    "1. Use vectorized operations when possible (e.g., +, -, *, /, ==, &, |)",
    "2. Reserve apply() for complex logic that can't be vectorized",
    "3. Use raw=True for numpy functions to improve performance",
    "4. Consider map() for dictionary lookups or simple transformations",
    "5. Use lambda for simple one-line functions, named functions for complex logic",
    "6. Always handle errors in apply functions, especially with real-world data",
    "7. Profile your code - apply can be a performance bottleneck",
    "8. Use result_type parameter when returning different data types",
    "9. Consider using numba.jit for computational intensive functions",
    "10. Test apply functions on small datasets before scaling up"
]

print("Best Practices:")
for practice in best_practices:
    print(practice)

print("\nCommon Pitfalls:")
pitfalls = [
    "1. Using apply when vectorized operations would work → Performance loss",
    "2. Not handling missing/invalid data → Runtime errors",
    "3. Using complex lambda functions → Reduced readability",
    "4. Forgetting axis parameter → Operating on wrong dimension", 
    "5. Not considering memory usage → Out of memory errors",
    "6. Ignoring return types → Unexpected data structures",
    "7. Not testing edge cases → Production failures",
    "8. Using apply in loops → Exponential performance degradation"
]

for pitfall in pitfalls:
    print(pitfall)

# Performance guidelines
print(f"\n=== PERFORMANCE GUIDELINES ===")

performance_guide = pd.DataFrame({
    'Operation': ['Element-wise arithmetic', 'Dictionary lookup', 'Complex conditions', 'String operations', 'Mathematical functions'],
    'Vectorized': ['df["A"] + df["B"]', 'df["col"].map(dict)', 'np.where()', 'df["col"].str.method()', 'np.sqrt(df["col"])'],
    'Apply': ['df.apply(lambda x: x["A"] + x["B"])', 'df["col"].apply(lambda x: dict[x])', 'df.apply(complex_logic)', 'df["col"].apply(str_func)', 'df["col"].apply(np.sqrt)'],
    'Recommendation': ['Always vectorize', 'Use map() instead', 'Use apply when necessary', 'Use str accessor', 'Vectorize or use raw=True']
})

print("Performance comparison guide:")
print(performance_guide.to_string(index=False))
```

#### Summary and Decision Matrix

| Scenario | Method | Syntax | Performance | Use Case |
|----------|--------|--------|-------------|----------|
| **Element-wise arithmetic** | Vectorized | `df['A'] + df['B']` | ⭐⭐⭐⭐⭐ | Simple math operations |
| **Dictionary mapping** | map() | `series.map(dict)` | ⭐⭐⭐⭐ | Value transformations |
| **String operations** | str accessor | `series.str.method()` | ⭐⭐⭐⭐ | String manipulations |
| **Simple conditions** | np.where | `np.where(condition, if_true, if_false)` | ⭐⭐⭐⭐⭐ | Binary choices |
| **Complex row logic** | apply(axis=1) | `df.apply(func, axis=1)` | ⭐⭐⭐ | Multi-column calculations |
| **Column statistics** | apply(axis=0) | `df.apply(func, axis=0)` | ⭐⭐⭐⭐ | Column-wise aggregations |
| **Mathematical functions** | Vectorized/raw | `np.func(series)` or `series.apply(func, raw=True)` | ⭐⭐⭐⭐⭐ | Math transformations |

**Key Decision Rules:**
- **Can it be vectorized?** → Always choose vectorized operations
- **Simple mapping?** → Use `map()` instead of `apply()`
- **String operations?** → Use `.str` accessor
- **Complex multi-column logic?** → Use `apply(axis=1)`
- **Column aggregations?** → Use `apply(axis=0)` or built-in methods
- **Performance critical?** → Profile and optimize, consider numba/cython

---

## Question 9

**Explain the usage and differences between astype, to_numeric, and pd.to_datetime.**

### Answer

#### Theory
Data type conversion is fundamental in data preprocessing and analysis. Pandas provides several methods for converting data types, each optimized for specific scenarios. Understanding the differences between `astype()`, `to_numeric()`, and `pd.to_datetime()` is crucial for robust data processing workflows that handle real-world data inconsistencies and type mismatches.

#### Core Concepts and Method Overview

```python
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

# Create sample data with mixed types and potential conversion issues
sample_data = pd.DataFrame({
    'integers_as_strings': ['1', '2', '3', '4', '5'],
    'floats_as_strings': ['1.5', '2.7', '3.14', '4.0', '5.99'],
    'mixed_numeric': ['10', '20.5', 'invalid', '30', '40.0'],
    'dates_as_strings': ['2023-01-01', '2023-02-15', '2023-03-20', '2023-04-10', '2023-05-30'],
    'messy_dates': ['Jan 1, 2023', '15/02/2023', '2023-03-20', 'invalid_date', '30-May-2023'],
    'boolean_strings': ['True', 'False', 'true', 'FALSE', '1'],
    'categorical_numbers': ['1', '2', '1', '3', '2'],
    'currency_strings': ['$100', '$250.50', '$1,000', '$75.25', '$500'],
    'percentages': ['10%', '25.5%', '100%', '0.5%', '75%']
})

print("Original Data with Mixed Types:")
print(sample_data)
print(f"\nData types:\n{sample_data.dtypes}")
print(f"\nSample values from each column:")
for col in sample_data.columns:
    print(f"{col}: {sample_data[col].iloc[0]} (type: {type(sample_data[col].iloc[0])})")
```

#### 1. astype() Method - Direct Type Conversion

```python
print("\n=== ASTYPE() METHOD ===")

# Basic astype conversions
print("1. Basic astype conversions:")

# Simple numeric conversions
df_astype = sample_data.copy()

# Convert clean string numbers to integers
df_astype['integers_converted'] = df_astype['integers_as_strings'].astype(int)
print(f"String to int: {df_astype['integers_converted'].dtype}")

# Convert clean string floats to floats
df_astype['floats_converted'] = df_astype['floats_as_strings'].astype(float)
print(f"String to float: {df_astype['floats_converted'].dtype}")

# Convert to categorical
df_astype['categorical_converted'] = df_astype['categorical_numbers'].astype('category')
print(f"String to category: {df_astype['categorical_converted'].dtype}")

print("Conversion results:")
print(df_astype[['integers_as_strings', 'integers_converted', 'floats_as_strings', 'floats_converted']].head())

# Advanced astype usage
print("\n2. Advanced astype patterns:")

# Convert with nullable integer types
df_astype['nullable_int'] = df_astype['integers_as_strings'].astype('Int64')  # Nullable integer
print(f"Nullable integer type: {df_astype['nullable_int'].dtype}")

# Convert to specific numpy dtypes
df_astype['int32_column'] = df_astype['integers_as_strings'].astype(np.int32)
df_astype['float32_column'] = df_astype['floats_as_strings'].astype(np.float32)

print("Specific dtypes:")
print(f"int32: {df_astype['int32_column'].dtype}")
print(f"float32: {df_astype['float32_column'].dtype}")

# Batch conversion with astype
type_conversions = {
    'integers_as_strings': 'int64',
    'floats_as_strings': 'float64',
    'categorical_numbers': 'category'
}

df_batch_converted = sample_data.astype(type_conversions)
print(f"\n3. Batch conversions:")
print(df_batch_converted.dtypes[list(type_conversions.keys())])

# Error handling with astype
print("\n4. Error handling:")
try:
    # This will fail due to 'invalid' value
    df_astype['mixed_numeric'].astype(float)
except ValueError as e:
    print(f"astype error: {e}")

# Copy vs inplace conversion
print("\n5. Copy vs inplace:")
original_memory = df_astype.memory_usage(deep=True).sum()

# Copy (default)
df_copy = df_astype.astype({'integers_as_strings': 'int64'})
copy_memory = df_copy.memory_usage(deep=True).sum()

print(f"Original memory: {original_memory} bytes")
print(f"After copy conversion: {copy_memory} bytes")
print(f"Memory change: {copy_memory - original_memory} bytes")
```

#### 2. to_numeric() Method - Robust Numeric Conversion

```python
print("\n=== TO_NUMERIC() METHOD ===")

# Basic to_numeric usage
print("1. Basic to_numeric conversions:")

# Clean conversion
clean_numeric = pd.to_numeric(sample_data['floats_as_strings'])
print(f"Clean conversion result: {clean_numeric.dtype}")
print(clean_numeric.head())

# Handling errors with different strategies
print("\n2. Error handling strategies:")

# Strategy 1: errors='raise' (default)
try:
    pd.to_numeric(sample_data['mixed_numeric'], errors='raise')
except ValueError as e:
    print(f"errors='raise': {e}")

# Strategy 2: errors='coerce'
coerced_numeric = pd.to_numeric(sample_data['mixed_numeric'], errors='coerce')
print(f"\nerrors='coerce' result:")
print(coerced_numeric)
print(f"NaN count: {coerced_numeric.isna().sum()}")

# Strategy 3: errors='ignore'
ignored_result = pd.to_numeric(sample_data['mixed_numeric'], errors='ignore')
print(f"\nerrors='ignore' result:")
print(ignored_result)
print(f"Result type: {type(ignored_result.iloc[0])}")

# Downcast option for memory efficiency
print("\n3. Downcasting for memory efficiency:")

large_integers = pd.Series(['100', '200', '300', '400', '500'])
regular_conversion = pd.to_numeric(large_integers)
downcast_conversion = pd.to_numeric(large_integers, downcast='integer')

print(f"Regular conversion: {regular_conversion.dtype}")
print(f"Downcast conversion: {downcast_conversion.dtype}")
print(f"Memory saved: {regular_conversion.memory_usage() - downcast_conversion.memory_usage()} bytes")

# Float downcasting
float_series = pd.Series(['1.1', '2.2', '3.3', '4.4', '5.5'])
float_regular = pd.to_numeric(float_series)
float_downcast = pd.to_numeric(float_series, downcast='float')

print(f"\nFloat regular: {float_regular.dtype}")
print(f"Float downcast: {float_downcast.dtype}")

# Complex data cleaning with to_numeric
print("\n4. Complex data cleaning:")

def clean_currency_column(series):
    """Clean currency strings and convert to numeric."""
    # Remove currency symbols and commas
    cleaned = series.str.replace('$', '', regex=False)
    cleaned = cleaned.str.replace(',', '', regex=False)
    return pd.to_numeric(cleaned, errors='coerce')

def clean_percentage_column(series):
    """Clean percentage strings and convert to numeric."""
    # Remove % symbol and convert to decimal
    cleaned = series.str.replace('%', '', regex=False)
    numeric_values = pd.to_numeric(cleaned, errors='coerce')
    return numeric_values / 100  # Convert to decimal

sample_data['currency_cleaned'] = clean_currency_column(sample_data['currency_strings'])
sample_data['percentage_cleaned'] = clean_percentage_column(sample_data['percentages'])

print("Currency and percentage cleaning:")
comparison = sample_data[['currency_strings', 'currency_cleaned', 'percentages', 'percentage_cleaned']]
print(comparison)

# Performance comparison: to_numeric vs astype
print("\n5. Performance comparison:")
import time

large_string_numbers = pd.Series([str(i) for i in range(100000)])

# Using to_numeric
start_time = time.time()
result_to_numeric = pd.to_numeric(large_string_numbers)
to_numeric_time = time.time() - start_time

# Using astype (when data is clean)
start_time = time.time()
result_astype = large_string_numbers.astype(int)
astype_time = time.time() - start_time

print(f"to_numeric time: {to_numeric_time:.4f} seconds")
print(f"astype time: {astype_time:.4f} seconds")
print(f"astype speedup: {to_numeric_time/astype_time:.1f}x")
```

#### 3. pd.to_datetime() Method - Date/Time Conversion

```python
print("\n=== PD.TO_DATETIME() METHOD ===")

# Basic datetime conversion
print("1. Basic datetime conversions:")

# Clean date conversion
clean_dates = pd.to_datetime(sample_data['dates_as_strings'])
print(f"Clean dates conversion:")
print(clean_dates)
print(f"Result dtype: {clean_dates.dtype}")

# Handling messy dates
print("\n2. Handling messy date formats:")

# Default behavior with mixed formats
try:
    messy_default = pd.to_datetime(sample_data['messy_dates'])
    print("Default conversion succeeded:")
    print(messy_default)
except Exception as e:
    print(f"Default conversion failed: {e}")

# Using errors='coerce' for messy dates
messy_coerced = pd.to_datetime(sample_data['messy_dates'], errors='coerce')
print(f"\nCoerced conversion:")
print(messy_coerced)
print(f"NaT count: {messy_coerced.isna().sum()}")

# Infer format automatically
messy_infer = pd.to_datetime(sample_data['messy_dates'], errors='coerce', infer_datetime_format=True)
print(f"\nWith format inference:")
print(messy_infer)

# Specific format specification
print("\n3. Format specification:")

# Create data with specific format
specific_format_dates = pd.Series(['20230101', '20230215', '20230320', '20230410', '20230530'])
formatted_dates = pd.to_datetime(specific_format_dates, format='%Y%m%d')
print(f"Specific format conversion:")
print(formatted_dates)

# Multiple format handling
mixed_formats = pd.Series(['2023-01-01', '01/15/2023', '2023.03.20', '2023-04-10', '30-May-2023'])

def flexible_date_parsing(date_series):
    """Parse dates with multiple possible formats."""
    formats = ['%Y-%m-%d', '%m/%d/%Y', '%Y.%m.%d', '%d-%b-%Y']
    
    result = pd.Series(index=date_series.index, dtype='datetime64[ns]')
    
    for fmt in formats:
        mask = result.isna()
        if mask.any():
            try:
                result[mask] = pd.to_datetime(date_series[mask], format=fmt, errors='coerce')
            except:
                continue
    
    return result

flexible_parsed = flexible_date_parsing(mixed_formats)
print(f"\nFlexible parsing result:")
print(flexible_parsed)

# Advanced datetime features
print("\n4. Advanced datetime features:")

# Unix timestamp conversion
unix_timestamps = pd.Series([1640995200, 1672531200, 1704067200, 1735689600])  # Unix timestamps
unix_converted = pd.to_datetime(unix_timestamps, unit='s')
print(f"Unix timestamp conversion:")
print(unix_converted)

# Origin parameter
excel_dates = pd.Series([44197, 44228, 44256])  # Excel serial dates
excel_converted = pd.to_datetime(excel_dates, unit='D', origin='1899-12-30')
print(f"\nExcel date conversion:")
print(excel_converted)

# UTC and timezone handling
utc_dates = pd.to_datetime(sample_data['dates_as_strings'], utc=True)
print(f"\nUTC conversion:")
print(utc_dates)

# Timezone conversion
local_dates = utc_dates.dt.tz_convert('US/Eastern')
print(f"\nTimezone converted:")
print(local_dates)

# Date component extraction
print("\n5. Date component extraction:")
sample_datetime = pd.to_datetime(sample_data['dates_as_strings'])

date_components = pd.DataFrame({
    'original': sample_datetime,
    'year': sample_datetime.dt.year,
    'month': sample_datetime.dt.month,
    'day': sample_datetime.dt.day,
    'dayofweek': sample_datetime.dt.dayofweek,
    'dayname': sample_datetime.dt.day_name(),
    'quarter': sample_datetime.dt.quarter,
    'is_weekend': sample_datetime.dt.dayofweek >= 5
})

print("Date component extraction:")
print(date_components)

# Performance optimization
print("\n6. Performance optimization:")

# Large dataset for timing
large_date_strings = pd.Series(['2023-01-01'] * 100000)

# Default conversion
start_time = time.time()
result_default = pd.to_datetime(large_date_strings)
default_time = time.time() - start_time

# With format specification
start_time = time.time()
result_format = pd.to_datetime(large_date_strings, format='%Y-%m-%d')
format_time = time.time() - start_time

print(f"Default parsing: {default_time:.4f} seconds")
print(f"Format specified: {format_time:.4f} seconds")
print(f"Format speedup: {default_time/format_time:.1f}x")
```

#### 4. Comparison and Best Practices

```python
print("\n=== COMPARISON AND BEST PRACTICES ===")

# Create test data for comparison
test_data = pd.DataFrame({
    'clean_integers': ['1', '2', '3', '4', '5'],
    'messy_numbers': ['1', '2.5', 'invalid', '4', '5.0'],
    'clean_dates': ['2023-01-01', '2023-02-01', '2023-03-01'],
    'messy_dates': ['2023-01-01', 'invalid', '2023-03-01'],
    'mixed_types': ['1', '2.5', 'text', '2023-01-01', 'True']
})

print("Method comparison matrix:")

comparison_results = {}

# astype() results
comparison_results['astype'] = {}
try:
    comparison_results['astype']['clean_integers'] = test_data['clean_integers'].astype(int).dtype
except:
    comparison_results['astype']['clean_integers'] = 'ERROR'

try:
    comparison_results['astype']['messy_numbers'] = test_data['messy_numbers'].astype(float).dtype
except:
    comparison_results['astype']['messy_numbers'] = 'ERROR'

# to_numeric() results
comparison_results['to_numeric'] = {}
comparison_results['to_numeric']['clean_integers'] = pd.to_numeric(test_data['clean_integers']).dtype
comparison_results['to_numeric']['messy_numbers'] = pd.to_numeric(test_data['messy_numbers'], errors='coerce').dtype

# to_datetime() results
comparison_results['to_datetime'] = {}
comparison_results['to_datetime']['clean_dates'] = pd.to_datetime(test_data['clean_dates']).dtype
comparison_results['to_datetime']['messy_dates'] = pd.to_datetime(test_data['messy_dates'], errors='coerce').dtype

print("\nComparison Results:")
for method, results in comparison_results.items():
    print(f"\n{method}:")
    for data_type, result in results.items():
        print(f"  {data_type}: {result}")

# Best practices guide
print("\n=== BEST PRACTICES GUIDE ===")

best_practices = {
    'astype()': [
        "✓ Use for clean data with known, consistent types",
        "✓ Fastest conversion method when data is clean",
        "✓ Good for batch conversions with dictionary mapping",
        "✓ Best for converting between compatible numeric types",
        "✗ No built-in error handling for invalid data",
        "✗ Fails completely if any value can't be converted"
    ],
    'to_numeric()': [
        "✓ Best for converting strings to numbers with error handling",
        "✓ Excellent for messy real-world numeric data",
        "✓ Provides downcast options for memory efficiency",
        "✓ Flexible error handling (raise, coerce, ignore)",
        "✓ Can handle currency, percentages with preprocessing",
        "✗ Slower than astype() for clean data",
        "✗ Only works for numeric conversions"
    ],
    'pd.to_datetime()': [
        "✓ Specialized for date/time conversions",
        "✓ Handles multiple date formats automatically",
        "✓ Supports timezone operations",
        "✓ Can parse Unix timestamps and Excel dates",
        "✓ Provides extensive error handling options",
        "✗ Can be slow without format specification",
        "✗ Only for datetime conversions"
    ]
}

for method, practices in best_practices.items():
    print(f"\n{method}:")
    for practice in practices:
        print(f"  {practice}")

# Decision flowchart
print("\n=== DECISION FLOWCHART ===")

def recommend_conversion_method(data_type, data_quality, specific_requirements=None):
    """Recommend the best conversion method based on context."""
    
    recommendations = []
    
    if data_type == 'numeric':
        if data_quality == 'clean':
            recommendations.append("astype() - Fastest for clean numeric data")
        else:
            recommendations.append("to_numeric() - Best error handling for messy data")
    
    elif data_type == 'datetime':
        recommendations.append("pd.to_datetime() - Specialized datetime conversion")
        if data_quality == 'messy':
            recommendations.append("  → Use errors='coerce' parameter")
        if specific_requirements == 'performance':
            recommendations.append("  → Specify format parameter for speed")
    
    elif data_type == 'categorical':
        recommendations.append("astype('category') - For categorical data")
    
    elif data_type == 'boolean':
        if data_quality == 'clean':
            recommendations.append("astype(bool) - For clean boolean data")
        else:
            recommendations.append("Custom function + to_numeric() for messy boolean data")
    
    return recommendations

# Example recommendations
scenarios = [
    ('numeric', 'clean', None),
    ('numeric', 'messy', None),
    ('datetime', 'clean', None),
    ('datetime', 'messy', 'performance'),
    ('categorical', 'clean', None),
    ('boolean', 'messy', None)
]

print("Conversion recommendations:")
for data_type, quality, requirements in scenarios:
    recs = recommend_conversion_method(data_type, quality, requirements)
    print(f"\n{data_type.title()} data ({quality}):")
    for rec in recs:
        print(f"  {rec}")

# Performance comparison summary
print(f"\n=== PERFORMANCE SUMMARY ===")

performance_data = pd.DataFrame({
    'Method': ['astype()', 'to_numeric()', 'pd.to_datetime()'],
    'Speed': ['Fastest', 'Medium', 'Slow-Medium'],
    'Error Handling': ['None', 'Excellent', 'Good'],
    'Flexibility': ['Low', 'High', 'High'],
    'Memory Efficiency': ['Good', 'Very Good (downcast)', 'Good'],
    'Use Case': ['Clean, known types', 'Messy numeric data', 'Date/time data']
})

print(performance_data.to_string(index=False))
```

#### 5. Advanced Usage Patterns

```python
print("\n=== ADVANCED USAGE PATTERNS ===")

# 1. Chained conversions
print("1. Chained conversions:")

messy_financial_data = pd.DataFrame({
    'revenue': ['$1,000.50', '$2,500.75', 'invalid', '$3,200.00'],
    'date': ['2023-01-01', '2023-02-01', '2023-03-01', 'invalid'],
    'percentage': ['10.5%', '15.2%', '20.0%', 'invalid']
})

def clean_financial_data(df):
    """Clean financial data with robust error handling."""
    df_clean = df.copy()
    
    # Clean revenue
    revenue_clean = df_clean['revenue'].str.replace('[$,]', '', regex=True)
    df_clean['revenue_numeric'] = pd.to_numeric(revenue_clean, errors='coerce')
    
    # Clean dates
    df_clean['date_parsed'] = pd.to_datetime(df_clean['date'], errors='coerce')
    
    # Clean percentages
    percentage_clean = df_clean['percentage'].str.replace('%', '', regex=False)
    df_clean['percentage_numeric'] = pd.to_numeric(percentage_clean, errors='coerce') / 100
    
    return df_clean

cleaned_financial = clean_financial_data(messy_financial_data)
print("Financial data cleaning:")
print(cleaned_financial)

# 2. Conditional conversions
print("\n2. Conditional conversions:")

mixed_data = pd.DataFrame({
    'values': ['1', '2.5', '2023-01-01', 'True', 'invalid']
})

def smart_type_detection(series):
    """Attempt to detect and convert appropriate types."""
    results = pd.DataFrame(index=series.index)
    results['original'] = series
    
    # Try numeric conversion
    numeric_result = pd.to_numeric(series, errors='coerce')
    results['as_numeric'] = numeric_result
    results['is_numeric'] = ~numeric_result.isna()
    
    # Try datetime conversion
    datetime_result = pd.to_datetime(series, errors='coerce')
    results['as_datetime'] = datetime_result
    results['is_datetime'] = ~datetime_result.isna()
    
    # Try boolean conversion
    boolean_mask = series.str.lower().isin(['true', 'false', '1', '0'])
    results['is_boolean'] = boolean_mask
    
    return results

smart_results = smart_type_detection(mixed_data['values'])
print("Smart type detection:")
print(smart_results)

# 3. Batch processing with error tracking
print("\n3. Batch processing with error tracking:")

def batch_convert_with_tracking(df, conversion_config):
    """
    Batch convert columns with detailed error tracking.
    
    conversion_config: dict with column names and conversion specifications
    """
    results = {}
    df_converted = df.copy()
    
    for col, config in conversion_config.items():
        if col not in df.columns:
            results[col] = {'status': 'column_not_found', 'errors': 0}
            continue
        
        try:
            method = config['method']
            params = config.get('params', {})
            
            if method == 'astype':
                converted = df[col].astype(config['target_type'])
                error_count = 0
                
            elif method == 'to_numeric':
                converted = pd.to_numeric(df[col], **params)
                error_count = converted.isna().sum() - df[col].isna().sum()
                
            elif method == 'to_datetime':
                converted = pd.to_datetime(df[col], **params)
                error_count = converted.isna().sum() - df[col].isna().sum()
                
            df_converted[col] = converted
            results[col] = {
                'status': 'success',
                'errors': error_count,
                'new_type': str(converted.dtype)
            }
            
        except Exception as e:
            results[col] = {
                'status': 'failed',
                'error_message': str(e),
                'errors': len(df)
            }
    
    return df_converted, results

# Example batch conversion
batch_config = {
    'integers_as_strings': {
        'method': 'astype',
        'target_type': 'int64'
    },
    'mixed_numeric': {
        'method': 'to_numeric',
        'params': {'errors': 'coerce'}
    },
    'messy_dates': {
        'method': 'to_datetime',
        'params': {'errors': 'coerce'}
    }
}

batch_result, batch_tracking = batch_convert_with_tracking(sample_data, batch_config)

print("Batch conversion tracking:")
for col, result in batch_tracking.items():
    print(f"{col}: {result}")
```

#### Summary Decision Matrix

| Scenario | Method | Parameters | Use Case |
|----------|--------|------------|----------|
| **Clean string numbers** | `astype()` | `astype(int)` or `astype(float)` | Fastest conversion, guaranteed clean data |
| **Messy numeric data** | `to_numeric()` | `errors='coerce'` | Real-world data with invalid values |
| **Memory optimization** | `to_numeric()` | `downcast='integer'/'float'` | Large datasets requiring memory efficiency |
| **Clean date strings** | `pd.to_datetime()` | `format='%Y-%m-%d'` | Known date format for speed |
| **Mixed date formats** | `pd.to_datetime()` | `errors='coerce', infer_datetime_format=True` | Flexible date parsing |
| **Unix timestamps** | `pd.to_datetime()` | `unit='s'` | Converting timestamp data |
| **Categorical data** | `astype()` | `astype('category')` | Creating categorical variables |
| **Batch conversions** | `astype()` | Dictionary mapping | Multiple columns, clean data |

**Key Decision Points:**
- **Data Quality**: Clean data → `astype()`, Messy data → `to_numeric()`/`pd.to_datetime()`
- **Error Handling**: Need robust handling → `to_numeric()`/`pd.to_datetime()` with `errors='coerce'`
- **Performance**: Speed critical + clean data → `astype()`
- **Memory**: Large datasets → `to_numeric()` with `downcast` parameter
- **Datetime**: Any date/time conversion → `pd.to_datetime()` exclusively

---

## Question 10

**Explain the different types of data ranking available in Pandas.**

### Answer

#### Theory
Data ranking is a fundamental operation in data analysis that assigns ordinal positions to values based on their magnitude or specified criteria. Pandas provides flexible ranking capabilities through the `rank()` method, supporting different ranking strategies to handle ties, missing values, and sorting preferences. Understanding ranking is essential for percentile calculations, competitive analysis, performance evaluation, and creating ordered categorical variables.

#### Core Ranking Concepts

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create comprehensive sample data for ranking demonstrations
np.random.seed(42)
sample_data = pd.DataFrame({
    'student_id': range(1, 16),
    'math_score': [95, 87, 87, 92, 78, 95, 83, 90, 75, 88, 91, 85, 79, 94, 86],
    'science_score': [88, 92, 85, 89, 95, 87, 91, 84, 90, 86, 93, 88, 82, 91, 87],
    'english_score': [92, 85, 88, 87, 82, 90, 86, 93, 79, 91, 89, 84, 87, 88, 85],
    'attendance': [0.95, 0.88, 0.92, 0.97, 0.85, 0.98, 0.90, 0.93, 0.87, 0.91, 0.96, 0.89, 0.84, 0.94, 0.86]
})

# Add some missing values for demonstration
sample_data.loc[2, 'science_score'] = np.nan
sample_data.loc[7, 'english_score'] = np.nan
sample_data.loc[12, 'math_score'] = np.nan

print("Sample Student Data with Rankings:")
print(sample_data.head(10))
print(f"\nData shape: {sample_data.shape}")
print(f"Missing values:\n{sample_data.isnull().sum()}")
```

#### 1. Basic Ranking Methods

```python
print("\n=== BASIC RANKING METHODS ===")

# Default ranking (average method for ties)
print("1. Default ranking (average method):")
sample_data['math_rank_default'] = sample_data['math_score'].rank()
sample_data['science_rank_default'] = sample_data['science_score'].rank()

print("Math scores with default ranking:")
math_ranking = sample_data[['student_id', 'math_score', 'math_rank_default']].sort_values('math_score', ascending=False)
print(math_ranking)

# Identify ties in the data
math_scores = sample_data['math_score'].dropna()
ties = math_scores[math_scores.duplicated(keep=False)].sort_values()
print(f"\nTies in math scores: {ties.tolist()}")

# Different ranking methods for handling ties
print("\n2. Different methods for handling ties:")

ranking_methods = ['average', 'min', 'max', 'first', 'dense']
tie_comparison = pd.DataFrame()
tie_comparison['score'] = sample_data['math_score']

for method in ranking_methods:
    tie_comparison[f'rank_{method}'] = sample_data['math_score'].rank(method=method)

# Show only rows with ties for clarity
tied_scores = sample_data['math_score'].dropna()
tie_values = tied_scores[tied_scores.duplicated(keep=False)].unique()

print("Comparison of ranking methods for tied values:")
for tie_val in tie_values:
    print(f"\nScore {tie_val}:")
    tied_rows = tie_comparison[tie_comparison['score'] == tie_val]
    print(tied_rows[['score'] + [f'rank_{method}' for method in ranking_methods]])

# Detailed explanation of each method
print("\n3. Method explanations with examples:")
explanations = {
    'average': "Average of ranks that would be assigned to tied values",
    'min': "Minimum rank among tied values (all get the lowest rank)",
    'max': "Maximum rank among tied values (all get the highest rank)", 
    'first': "Ranks assigned in order of appearance in the data",
    'dense': "Like 'min' but ranks always increase by 1 between groups"
}

for method, explanation in explanations.items():
    print(f"{method}: {explanation}")
```

#### 2. Ranking Direction and Sorting

```python
print("\n=== RANKING DIRECTION ===")

# Ascending vs Descending rankings
print("1. Ascending vs Descending rankings:")

sample_data['math_rank_asc'] = sample_data['math_score'].rank(ascending=True)
sample_data['math_rank_desc'] = sample_data['math_score'].rank(ascending=False)

ranking_direction = sample_data[['student_id', 'math_score', 'math_rank_asc', 'math_rank_desc']].sort_values('math_score', ascending=False)
print("Math score rankings (ascending vs descending):")
print(ranking_direction.head(10))

print(f"\nAscending: Higher scores get higher ranks (rank {sample_data['math_rank_asc'].max()} is best)")
print(f"Descending: Higher scores get lower ranks (rank 1 is best)")

# Percentage rankings (percentile ranks)
print("\n2. Percentage rankings:")

sample_data['math_percentile'] = sample_data['math_score'].rank(pct=True)
sample_data['science_percentile'] = sample_data['science_score'].rank(pct=True)

percentile_view = sample_data[['student_id', 'math_score', 'math_percentile', 'science_score', 'science_percentile']].sort_values('math_score', ascending=False)
print("Percentile rankings (0-1 scale):")
print(percentile_view.head(10))

# Convert to traditional percentiles (0-100)
sample_data['math_percentile_100'] = sample_data['math_percentile'] * 100
print(f"\nTop student math percentile: {sample_data['math_percentile_100'].max():.1f}th percentile")
print(f"Bottom student math percentile: {sample_data['math_percentile_100'].min():.1f}th percentile")
```

#### 3. Advanced Ranking Techniques

```python
print("\n=== ADVANCED RANKING TECHNIQUES ===")

# 1. Multi-column ranking
print("1. Multi-column composite ranking:")

# Create weighted composite score
weights = {'math_score': 0.4, 'science_score': 0.35, 'english_score': 0.25}

def calculate_weighted_score(row):
    """Calculate weighted composite score handling missing values."""
    total_weight = 0
    weighted_sum = 0
    
    for subject, weight in weights.items():
        if not pd.isna(row[subject]):
            weighted_sum += row[subject] * weight
            total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else np.nan

sample_data['composite_score'] = sample_data.apply(calculate_weighted_score, axis=1)
sample_data['composite_rank'] = sample_data['composite_score'].rank(ascending=False, method='dense')

print("Composite score ranking:")
composite_ranking = sample_data[['student_id', 'math_score', 'science_score', 'english_score', 'composite_score', 'composite_rank']].sort_values('composite_rank')
print(composite_ranking.head(10))

# 2. Group-wise ranking
print("\n2. Group-wise ranking:")

# Create performance categories for grouping
sample_data['performance_category'] = pd.cut(
    sample_data['composite_score'], 
    bins=3, 
    labels=['Below Average', 'Average', 'Above Average']
)

# Rank within each performance category
sample_data['rank_within_category'] = sample_data.groupby('performance_category')['math_score'].rank(ascending=False)

group_ranking = sample_data[['student_id', 'performance_category', 'math_score', 'rank_within_category']].sort_values(['performance_category', 'rank_within_category'])
print("Ranking within performance categories:")
print(group_ranking)

# 3. Conditional ranking
print("\n3. Conditional ranking with attendance threshold:")

# Rank only students with good attendance (>= 90%)
def conditional_rank(df, score_col, condition_col, threshold):
    """Rank only students meeting attendance criteria."""
    eligible_mask = df[condition_col] >= threshold
    ranking = pd.Series(index=df.index, dtype=float)
    
    if eligible_mask.any():
        eligible_scores = df.loc[eligible_mask, score_col]
        eligible_ranks = eligible_scores.rank(ascending=False, method='dense')
        ranking.loc[eligible_mask] = eligible_ranks
    
    return ranking

sample_data['math_rank_eligible'] = conditional_rank(
    sample_data, 'math_score', 'attendance', 0.90
)

conditional_ranking = sample_data[['student_id', 'math_score', 'attendance', 'math_rank_eligible']].sort_values('math_rank_eligible')
print("Rankings for students with >= 90% attendance:")
print(conditional_ranking.dropna(subset=['math_rank_eligible']))

# 4. Rolling/windowed rankings
print("\n4. Time-based rolling rankings:")

# Create time series data for rolling rankings
dates = pd.date_range('2023-01-01', periods=20, freq='D')
time_series_data = pd.DataFrame({
    'date': dates,
    'daily_score': np.random.normal(85, 10, 20)
})

# 7-day rolling rank
time_series_data['rolling_rank'] = time_series_data['daily_score'].rolling(window=7).rank(ascending=False)

print("Rolling 7-day rankings:")
print(time_series_data.tail(10))
```

#### 4. Handling Missing Values in Rankings

```python
print("\n=== HANDLING MISSING VALUES ===")

# Different strategies for missing values
print("1. Missing value strategies:")

missing_strategies = ['keep', 'top', 'bottom']

for strategy in missing_strategies:
    col_name = f'rank_na_{strategy}'
    sample_data[col_name] = sample_data['science_score'].rank(na_option=strategy, ascending=False)

missing_comparison = sample_data[['student_id', 'science_score'] + [f'rank_na_{strategy}' for strategy in missing_strategies]]
print("Missing value handling comparison:")
print(missing_comparison[missing_comparison['science_score'].isna() | missing_comparison.index.isin([0, 1, 2, 3, 4])])

print("\nMissing value strategies explained:")
print("keep: Missing values remain NaN in rankings")
print("top: Missing values get rank 1 (best rank)")
print("bottom: Missing values get worst rank")

# Custom missing value handling
print("\n2. Custom missing value imputation before ranking:")

def smart_impute_for_ranking(df, score_col, reference_cols):
    """Impute missing values based on correlation with other subjects."""
    df_imputed = df.copy()
    
    # Calculate correlation matrix
    corr_matrix = df[reference_cols + [score_col]].corr()
    
    # For each missing value, impute based on highest correlated subject
    missing_mask = df[score_col].isna()
    
    for idx in df[missing_mask].index:
        # Find the most correlated non-missing subject
        available_scores = {}
        for ref_col in reference_cols:
            if not pd.isna(df.loc[idx, ref_col]):
                available_scores[ref_col] = df.loc[idx, ref_col]
        
        if available_scores:
            # Use the score from the most correlated subject
            best_corr = -1
            best_ref = None
            for ref_col in available_scores:
                if abs(corr_matrix.loc[score_col, ref_col]) > best_corr:
                    best_corr = abs(corr_matrix.loc[score_col, ref_col])
                    best_ref = ref_col
            
            # Simple linear imputation based on correlation
            if best_ref:
                correlation = corr_matrix.loc[score_col, best_ref]
                mean_ratio = df[score_col].mean() / df[best_ref].mean()
                imputed_value = available_scores[best_ref] * mean_ratio
                df_imputed.loc[idx, score_col] = imputed_value
    
    return df_imputed

# Apply smart imputation
imputed_data = smart_impute_for_ranking(sample_data, 'science_score', ['math_score', 'english_score'])
sample_data['science_imputed'] = imputed_data['science_score']
sample_data['science_rank_imputed'] = sample_data['science_imputed'].rank(ascending=False)

print("Imputation-based ranking:")
imputation_comparison = sample_data[['student_id', 'science_score', 'science_imputed', 'science_rank_imputed']]
print(imputation_comparison[imputation_comparison['science_score'].isna() | imputation_comparison.index.isin([0, 1, 2, 3, 4])])
```

#### 5. Ranking Applications and Use Cases

```python
print("\n=== RANKING APPLICATIONS ===")

# 1. Quartile and decile rankings
print("1. Quartile and decile rankings:")

sample_data['math_quartile'] = pd.qcut(
    sample_data['math_score'].rank(method='first'), 
    q=4, 
    labels=['Q1 (Bottom 25%)', 'Q2', 'Q3', 'Q4 (Top 25%)']
)

sample_data['math_decile'] = pd.qcut(
    sample_data['math_score'].rank(method='first'), 
    q=10, 
    labels=[f'D{i}' for i in range(1, 11)]
)

quartile_summary = sample_data.groupby('math_quartile')['math_score'].agg(['count', 'mean', 'min', 'max'])
print("Math score quartile analysis:")
print(quartile_summary)

# 2. Performance tier assignment
print("\n2. Performance tier assignment:")

def assign_performance_tier(rank, total_count):
    """Assign performance tier based on rank percentile."""
    percentile = (total_count - rank + 1) / total_count
    
    if percentile >= 0.9:
        return 'Excellent (Top 10%)'
    elif percentile >= 0.75:
        return 'Good (Top 25%)'
    elif percentile >= 0.5:
        return 'Average (Top 50%)'
    elif percentile >= 0.25:
        return 'Below Average (Bottom 50%)'
    else:
        return 'Needs Improvement (Bottom 25%)'

total_students = sample_data['math_score'].count()
sample_data['performance_tier'] = sample_data['math_rank_desc'].apply(
    lambda x: assign_performance_tier(x, total_students) if not pd.isna(x) else 'Incomplete'
)

tier_distribution = sample_data['performance_tier'].value_counts()
print("Performance tier distribution:")
print(tier_distribution)

# 3. Competitive ranking with ties
print("\n3. Competitive ranking simulation:")

def competitive_ranking(scores, tie_method='min'):
    """
    Simulate competitive ranking like Olympic standings.
    """
    ranks = scores.rank(ascending=False, method=tie_method)
    
    # Create ranking summary
    ranking_summary = pd.DataFrame({
        'participant': scores.index,
        'score': scores.values,
        'rank': ranks.values
    }).sort_values('rank')
    
    # Add medals for top 3
    ranking_summary['medal'] = ranking_summary['rank'].apply(
        lambda x: 'Gold 🥇' if x == 1 else 'Silver 🥈' if x == 2 else 'Bronze 🥉' if x == 3 else ''
    )
    
    return ranking_summary

math_competition = competitive_ranking(sample_data.set_index('student_id')['math_score'].dropna())
print("Math competition results:")
print(math_competition.head(10))

# 4. Relative ranking and z-scores
print("\n4. Relative performance analysis:")

# Calculate z-scores for relative performance
sample_data['math_zscore'] = (sample_data['math_score'] - sample_data['math_score'].mean()) / sample_data['math_score'].std()
sample_data['science_zscore'] = (sample_data['science_score'] - sample_data['science_score'].mean()) / sample_data['science_score'].std()

# Rank based on z-scores for fair comparison across subjects
sample_data['math_relative_rank'] = sample_data['math_zscore'].rank(ascending=False)
sample_data['science_relative_rank'] = sample_data['science_zscore'].rank(ascending=False)

relative_performance = sample_data[['student_id', 'math_zscore', 'math_relative_rank', 'science_zscore', 'science_relative_rank']].sort_values('math_relative_rank')
print("Relative performance ranking (z-score based):")
print(relative_performance.head(8))

# 5. Dynamic ranking with updates
print("\n5. Dynamic ranking system:")

class DynamicRanking:
    """A class to maintain dynamic rankings as new data arrives."""
    
    def __init__(self, initial_data=None):
        self.data = initial_data.copy() if initial_data is not None else pd.DataFrame()
        self.rankings = pd.Series(dtype=float)
        
    def add_score(self, participant_id, score):
        """Add a new score and update rankings."""
        self.data.loc[participant_id] = score
        self._update_rankings()
        
    def _update_rankings(self):
        """Recalculate all rankings."""
        self.rankings = self.data.rank(ascending=False, method='dense')
        
    def get_ranking(self, participant_id=None):
        """Get ranking for specific participant or all."""
        if participant_id:
            return self.rankings.get(participant_id, None)
        return self.rankings.sort_values()
    
    def get_top_n(self, n=5):
        """Get top N participants."""
        top_ranks = self.rankings.sort_values().head(n)
        return pd.DataFrame({
            'participant': top_ranks.index,
            'score': [self.data.loc[p] for p in top_ranks.index],
            'rank': top_ranks.values
        })

# Demonstrate dynamic ranking
initial_scores = sample_data.set_index('student_id')['math_score'].dropna().head(8)
dynamic_ranker = DynamicRanking(initial_scores)

print("Initial leaderboard:")
print(dynamic_ranker.get_top_n(5))

# Add new participant
dynamic_ranker.add_score('new_student_1', 98)
print("\nAfter adding high-performing student:")
print(dynamic_ranker.get_top_n(5))

dynamic_ranker.add_score('new_student_2', 70)
print("\nAfter adding lower-performing student:")
print(dynamic_ranker.get_top_n(5))
```

#### 6. Performance Optimization and Best Practices

```python
print("\n=== PERFORMANCE OPTIMIZATION ===")

# Performance comparison of different ranking methods
import time

# Create large dataset for performance testing
large_data = pd.Series(np.random.normal(50, 15, 100000))

print("Performance comparison on 100K values:")

ranking_methods = ['average', 'min', 'max', 'first', 'dense']
performance_results = {}

for method in ranking_methods:
    start_time = time.time()
    result = large_data.rank(method=method)
    end_time = time.time()
    performance_results[method] = end_time - start_time

print("Ranking method performance:")
for method, time_taken in performance_results.items():
    print(f"{method:>8}: {time_taken:.4f} seconds")

fastest_method = min(performance_results, key=performance_results.get)
print(f"\nFastest method: {fastest_method}")

# Memory usage comparison
print(f"\n Memory usage:")
print(f"Original data: {large_data.memory_usage(deep=True) / 1024:.2f} KB")
print(f"Ranking result: {result.memory_usage(deep=True) / 1024:.2f} KB")

# Best practices guide
print("\n=== BEST PRACTICES ===")

best_practices = [
    "1. Choose appropriate ranking method for ties based on business logic",
    "2. Use 'dense' ranking for leaderboards to avoid gaps",
    "3. Use 'min' ranking for competitive scenarios (traditional)",
    "4. Consider 'first' ranking when order of occurrence matters",
    "5. Use percentage rankings for percentile analysis",
    "6. Handle missing values explicitly based on domain knowledge",
    "7. Use ascending=False for 'higher is better' scenarios",
    "8. Consider memory usage for large datasets",
    "9. Validate ranking results with domain experts",
    "10. Document ranking methodology for reproducibility"
]

for practice in best_practices:
    print(practice)

# Common pitfalls
print("\nCommon Pitfalls:")
pitfalls = [
    "1. Inconsistent handling of ties across analyses",
    "2. Not considering missing values impact on rankings", 
    "3. Using wrong ascending parameter direction",
    "4. Ignoring the effect of outliers on rankings",
    "5. Not validating ranking results make business sense",
    "6. Mixing ranking methods within the same analysis",
    "7. Not documenting ranking methodology",
    "8. Forgetting to handle new data in dynamic rankings"
]

for pitfall in pitfalls:
    print(pitfall)
```

#### Summary Decision Matrix

| Ranking Scenario | Method | Parameters | Use Case |
|------------------|--------|------------|----------|
| **Traditional competition** | `rank()` | `method='min', ascending=False` | Sports, academic competitions |
| **Dense leaderboards** | `rank()` | `method='dense', ascending=False` | Gaming, performance tracking |
| **Percentile analysis** | `rank()` | `pct=True` | Statistical analysis, benchmarking |
| **Order-sensitive** | `rank()` | `method='first'` | First-come-first-served scenarios |
| **Fair tie handling** | `rank()` | `method='average'` | Academic grading, performance reviews |
| **Missing data inclusive** | `rank()` | `na_option='keep'` | Incomplete datasets |
| **Group comparisons** | `groupby().rank()` | Various | Within-category rankings |
| **Time series** | `rolling().rank()` | `window=n` | Moving rankings, trends |

**Key Decision Points:**
- **Ties matter**: Use `min` for traditional, `dense` for no gaps, `average` for fairness
- **Missing values**: Choose `keep`, `top`, or `bottom` based on business logic  
- **Direction**: `ascending=False` when higher values should get lower ranks (rank 1 = best)
- **Percentiles**: Use `pct=True` for standardized 0-1 scale comparisons
- **Performance**: `first` method is typically fastest, `average` requires more computation

---

## Question 11

**What is a crosstab in Pandas, and when would you use it?**

### Answer

#### Theory
A crosstab (cross-tabulation) is a powerful statistical tool for examining relationships between categorical variables by creating a contingency table that shows the frequency distribution of variables in a matrix format. Pandas' `pd.crosstab()` function provides an efficient way to create these tables, enabling analysis of associations, dependencies, and patterns between categorical variables. It's essential for exploratory data analysis, hypothesis testing, and understanding data relationships.

#### Basic Crosstab Usage

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample dataset for crosstab demonstrations
np.random.seed(42)
sample_data = pd.DataFrame({
    'gender': np.random.choice(['Male', 'Female'], 1000),
    'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], 1000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], 1000),
    'satisfaction': np.random.choice(['Low', 'Medium', 'High'], 1000),
    'salary_range': np.random.choice(['<50K', '50-75K', '75-100K', '>100K'], 1000),
    'performance': np.random.choice(['Below', 'Meets', 'Exceeds'], 1000, p=[0.2, 0.6, 0.2])
})

print("Sample Data Overview:")
print(sample_data.head())
print(f"\nDataset shape: {sample_data.shape}")

# Basic crosstab example
print("\n=== BASIC CROSSTAB ===")
basic_crosstab = pd.crosstab(sample_data['gender'], sample_data['department'])
print("Gender vs Department Crosstab:")
print(basic_crosstab)

# Add margins (totals)
crosstab_with_margins = pd.crosstab(
    sample_data['gender'], 
    sample_data['department'], 
    margins=True
)
print("\nWith margins (totals):")
print(crosstab_with_margins)
```

#### Advanced Crosstab Features

```python
print("\n=== ADVANCED CROSSTAB FEATURES ===")

# 1. Percentage calculations
print("1. Percentage calculations:")

# Column percentages
col_percentages = pd.crosstab(
    sample_data['gender'], 
    sample_data['department'], 
    normalize='columns'  # Normalize by columns
) * 100

print("Column percentages (% within each department):")
print(col_percentages.round(1))

# Row percentages  
row_percentages = pd.crosstab(
    sample_data['gender'], 
    sample_data['department'], 
    normalize='index'  # Normalize by rows
) * 100

print("\nRow percentages (% within each gender):")
print(row_percentages.round(1))

# Overall percentages
all_percentages = pd.crosstab(
    sample_data['gender'], 
    sample_data['department'], 
    normalize='all'  # Normalize by total
) * 100

print("\nOverall percentages (% of total):")
print(all_percentages.round(1))

# 2. Three-way crosstabs
print("\n2. Three-way crosstabs:")

three_way = pd.crosstab(
    [sample_data['gender'], sample_data['age_group']], 
    sample_data['department'],
    margins=True
)

print("Gender + Age Group vs Department:")
print(three_way.head(10))

# 3. Crosstab with aggregation values
print("\n3. Crosstab with aggregated values:")

# Add numeric values for aggregation
sample_data['salary'] = np.random.normal(75000, 20000, 1000)
sample_data['years_experience'] = np.random.randint(1, 20, 1000)

# Mean salary by gender and department
salary_crosstab = pd.crosstab(
    sample_data['gender'], 
    sample_data['department'],
    values=sample_data['salary'],
    aggfunc='mean'
)

print("Average salary by gender and department:")
print(salary_crosstab.round(0))

# Multiple aggregation functions
multi_agg = pd.crosstab(
    sample_data['gender'], 
    sample_data['department'],
    values=sample_data['salary'],
    aggfunc=['mean', 'std', 'count']
)

print("\nMultiple aggregations:")
print(multi_agg.round(0))
```

#### Use Cases and Applications

```python
print("\n=== USE CASES AND APPLICATIONS ===")

# 1. Market Research Analysis
print("1. Market Research Analysis:")

# Customer segmentation analysis
customer_analysis = pd.crosstab(
    sample_data['age_group'], 
    sample_data['satisfaction'],
    normalize='index',
    margins=True
) * 100

print("Customer satisfaction by age group (%):")
print(customer_analysis.round(1))

# 2. A/B Testing Analysis
print("\n2. A/B Testing Simulation:")

# Simulate A/B test data
ab_test_data = pd.DataFrame({
    'variant': np.random.choice(['A', 'B'], 1000),
    'conversion': np.random.choice(['Yes', 'No'], 1000, p=[0.15, 0.85]),
    'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], 1000),
    'source': np.random.choice(['Organic', 'Paid', 'Social'], 1000)
})

# Conversion rates by variant
conversion_analysis = pd.crosstab(
    ab_test_data['variant'], 
    ab_test_data['conversion'],
    normalize='index'
) * 100

print("Conversion rates by variant:")
print(conversion_analysis.round(2))

# Conversion by variant and device
device_conversion = pd.crosstab(
    [ab_test_data['variant'], ab_test_data['device']], 
    ab_test_data['conversion'],
    normalize='index'
) * 100

print("\nConversion rates by variant and device:")
print(device_conversion.round(2))

# 3. Quality Control Analysis
print("\n3. Quality Control Analysis:")

# Simulate manufacturing data
quality_data = pd.DataFrame({
    'shift': np.random.choice(['Morning', 'Afternoon', 'Night'], 500),
    'machine': np.random.choice(['Machine_A', 'Machine_B', 'Machine_C'], 500),
    'quality': np.random.choice(['Pass', 'Fail'], 500, p=[0.85, 0.15]),
    'operator': np.random.choice(['Op1', 'Op2', 'Op3', 'Op4'], 500)
})

# Quality rates by shift and machine
quality_analysis = pd.crosstab(
    quality_data['shift'], 
    quality_data['machine'],
    values=quality_data['quality'] == 'Pass',
    aggfunc='mean'
) * 100

print("Pass rates by shift and machine (%):")
print(quality_analysis.round(1))

# 4. Survey Data Analysis
print("\n4. Survey Response Analysis:")

# Demographic response patterns
demo_response = pd.crosstab(
    sample_data['education'], 
    sample_data['satisfaction'],
    normalize='index'
) * 100

print("Satisfaction levels by education (%):")
print(demo_response.round(1))
```

#### Statistical Testing with Crosstabs

```python
print("\n=== STATISTICAL TESTING ===")

# Chi-square test of independence
from scipy.stats import chi2_contingency

print("1. Chi-square test of independence:")

# Test independence between gender and department
contingency_table = pd.crosstab(sample_data['gender'], sample_data['department'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

print("\nExpected frequencies:")
expected_df = pd.DataFrame(
    expected, 
    index=contingency_table.index, 
    columns=contingency_table.columns
)
print(expected_df.round(1))

# 2. Effect size calculation (Cramér's V)
print("\n2. Effect size (Cramér's V):")

def cramers_v(confusion_matrix):
    """Calculate Cramér's V effect size."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

cramers_v_value = cramers_v(contingency_table)
print(f"Cramér's V: {cramers_v_value:.4f}")

effect_size_interpretation = {
    0.1: "Small effect",
    0.3: "Medium effect", 
    0.5: "Large effect"
}

interpretation = "Small effect"
for threshold, desc in effect_size_interpretation.items():
    if cramers_v_value >= threshold:
        interpretation = desc

print(f"Effect size interpretation: {interpretation}")

# 3. Residuals analysis
print("\n3. Standardized residuals:")

residuals = (contingency_table - expected_df) / np.sqrt(expected_df)
print("Standardized residuals (>±2 indicates significant association):")
print(residuals.round(2))
```

#### Advanced Crosstab Techniques

```python
print("\n=== ADVANCED TECHNIQUES ===")

# 1. Custom aggregation functions
print("1. Custom aggregation functions:")

def coefficient_of_variation(x):
    """Calculate coefficient of variation."""
    return (x.std() / x.mean()) * 100 if x.mean() != 0 else 0

cv_crosstab = pd.crosstab(
    sample_data['department'], 
    sample_data['education'],
    values=sample_data['salary'],
    aggfunc=coefficient_of_variation
)

print("Salary coefficient of variation by department and education:")
print(cv_crosstab.round(1))

# 2. Time series crosstabs
print("\n2. Time series crosstabs:")

# Generate time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
time_data = pd.DataFrame({
    'date': np.random.choice(dates, 1000),
    'product': np.random.choice(['Product_A', 'Product_B', 'Product_C'], 1000),
    'sales_channel': np.random.choice(['Online', 'Store', 'Phone'], 1000),
    'units_sold': np.random.poisson(10, 1000)
})

time_data['month'] = time_data['date'].dt.month_name()
time_data['quarter'] = 'Q' + time_data['date'].dt.quarter.astype(str)

monthly_sales = pd.crosstab(
    time_data['month'], 
    time_data['product'],
    values=time_data['units_sold'],
    aggfunc='sum'
)

print("Monthly sales by product:")
print(monthly_sales.head())

# 3. Conditional crosstabs
print("\n3. Conditional crosstabs:")

# High performers only
high_performers = sample_data[sample_data['performance'] == 'Exceeds']
high_perf_crosstab = pd.crosstab(
    high_performers['department'], 
    high_performers['education'],
    normalize='index'
) * 100

print("Education distribution among high performers by department (%):")
print(high_perf_crosstab.round(1))

# 4. Weighted crosstabs
print("\n4. Weighted crosstabs:")

# Add weights (e.g., survey weights)
sample_data['weight'] = np.random.uniform(0.5, 2.0, len(sample_data))

def weighted_crosstab(df, row_var, col_var, weight_var):
    """Create weighted crosstab."""
    # Create combinations
    combinations = df.groupby([row_var, col_var])[weight_var].sum().reset_index()
    
    # Pivot to crosstab format
    weighted_ct = combinations.pivot(index=row_var, columns=col_var, values=weight_var)
    return weighted_ct.fillna(0)

weighted_ct = weighted_crosstab(sample_data, 'gender', 'department', 'weight')
print("Weighted crosstab (gender vs department):")
print(weighted_ct.round(2))
```

#### Visualization and Reporting

```python
print("\n=== VISUALIZATION EXAMPLES ===")

# 1. Heatmap visualization
def create_crosstab_heatmap(crosstab_data, title="Crosstab Heatmap"):
    """Create heatmap visualization of crosstab."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(crosstab_data, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title(title)
    plt.tight_layout()
    return plt

# Example: Department vs Gender heatmap
dept_gender_ct = pd.crosstab(sample_data['department'], sample_data['gender'])
print("Creating heatmap visualization...")
print("Department vs Gender counts:")
print(dept_gender_ct)

# 2. Percentage heatmap
perc_ct = pd.crosstab(sample_data['department'], sample_data['satisfaction'], normalize='index') * 100
print("\nSatisfaction percentages by department:")
print(perc_ct.round(1))

# 3. Comprehensive reporting function
def crosstab_report(df, row_var, col_var, title="Crosstab Analysis"):
    """Generate comprehensive crosstab report."""
    
    report = f"\n{'='*50}\n{title}\n{'='*50}\n"
    
    # Basic crosstab
    ct = pd.crosstab(df[row_var], df[col_var], margins=True)
    report += f"\n1. Frequency Table:\n{ct}\n"
    
    # Row percentages
    row_pct = pd.crosstab(df[row_var], df[col_var], normalize='index') * 100
    report += f"\n2. Row Percentages:\n{row_pct.round(1)}\n"
    
    # Column percentages
    col_pct = pd.crosstab(df[row_var], df[col_var], normalize='columns') * 100
    report += f"\n3. Column Percentages:\n{col_pct.round(1)}\n"
    
    # Chi-square test
    chi2, p_val, dof, expected = chi2_contingency(ct.iloc[:-1, :-1])  # Exclude margins
    report += f"\n4. Statistical Test:\n"
    report += f"   Chi-square: {chi2:.4f}\n"
    report += f"   p-value: {p_val:.4f}\n"
    report += f"   Significant: {'Yes' if p_val < 0.05 else 'No'}\n"
    
    return report

# Generate report
analysis_report = crosstab_report(sample_data, 'education', 'performance', 
                                 "Education vs Performance Analysis")
print(analysis_report)
```

#### Best Practices and Common Use Cases

```python
print("\n=== BEST PRACTICES ===")

best_practices = [
    "1. Use margins=True to include row/column totals for context",
    "2. Choose appropriate normalization: 'index' for row %, 'columns' for column %",
    "3. Apply statistical tests (chi-square) to assess significance",
    "4. Use visualizations (heatmaps) for better interpretation",
    "5. Consider sample sizes - small cells (<5) may affect statistical validity",
    "6. Document assumptions and methodology for reproducibility",
    "7. Use aggfunc parameter for numerical variable analysis",
    "8. Handle missing values explicitly before creating crosstabs",
    "9. Validate results make business/domain sense",
    "10. Consider effect sizes alongside statistical significance"
]

print("Best Practices:")
for practice in best_practices:
    print(practice)

print("\nCommon Use Cases:")
use_cases = [
    "• Market research: Customer demographics vs preferences",
    "• A/B testing: Treatment groups vs outcomes", 
    "• Quality control: Process factors vs defect rates",
    "• Survey analysis: Respondent characteristics vs responses",
    "• Medical research: Risk factors vs health outcomes",
    "• Education: Student characteristics vs performance",
    "• HR analytics: Employee attributes vs satisfaction/performance",
    "• Sales analysis: Customer segments vs purchase behavior",
    "• Social science: Demographic factors vs attitudes/behaviors",
    "• Risk analysis: Risk factors vs incident occurrence"
]

for use_case in use_cases:
    print(use_case)

print("\nCommon Pitfalls:")
pitfalls = [
    "• Interpreting correlation as causation",
    "• Ignoring small cell counts in statistical tests",
    "• Not considering confounding variables",
    "• Using inappropriate normalization for the question",
    "• Overlooking missing data patterns",
    "• Not validating statistical assumptions",
    "• Misinterpreting percentages vs. raw counts",
    "• Ignoring the effect size when significant results exist"
]

for pitfall in pitfalls:
    print(pitfall)
```

#### Summary Decision Guide

| Analysis Goal | Crosstab Type | Parameters | Statistical Test |
|---------------|---------------|------------|------------------|
| **Frequency analysis** | Basic counts | `margins=True` | Chi-square test |
| **Proportion comparison** | Row/column % | `normalize='index'/'columns'` | Chi-square test |
| **Market segmentation** | Percentage | `normalize='index'` | Chi-square + Cramér's V |
| **A/B testing** | Conversion rates | `normalize='index'` | Chi-square test |
| **Survey analysis** | Response patterns | `normalize='index'` | Chi-square test |
| **Quality control** | Defect rates | `aggfunc='mean'` | Control charts |
| **Customer analysis** | Behavior patterns | `normalize='columns'` | Association rules |
| **Risk assessment** | Incident rates | `aggfunc='mean'` | Risk ratios |

**Key Decision Points:**
- **Categorical vs Numerical**: Use basic crosstab for categorical, aggfunc for numerical values
- **Independence testing**: Always include chi-square test for association analysis
- **Percentage interpretation**: Row % for "given X, what's the distribution of Y?"
- **Sample size**: Ensure adequate cell counts (>5) for valid statistical tests
- **Effect size**: Calculate Cramér's V for practical significance beyond statistical significance

---

## Question 12

**Describe how to perform a multi-index query on a DataFrame.**

### Answer

#### Theory
Multi-index DataFrames (hierarchical indexes) allow for sophisticated data organization and querying capabilities in Pandas. They enable representation of higher-dimensional data in a 2D structure and provide powerful methods for data selection, filtering, and aggregation across multiple levels. Understanding multi-index querying is essential for working with complex datasets like time series, panel data, and grouped statistics.

#### Creating Multi-Index DataFrames

```python
import pandas as pd
import numpy as np

# Create sample multi-index data
np.random.seed(42)

# Method 1: Create from tuples
index_tuples = [
    ('Company_A', 'Q1', '2023'), ('Company_A', 'Q2', '2023'), ('Company_A', 'Q3', '2023'), ('Company_A', 'Q4', '2023'),
    ('Company_A', 'Q1', '2024'), ('Company_A', 'Q2', '2024'),
    ('Company_B', 'Q1', '2023'), ('Company_B', 'Q2', '2023'), ('Company_B', 'Q3', '2023'), ('Company_B', 'Q4', '2023'),
    ('Company_B', 'Q1', '2024'), ('Company_B', 'Q2', '2024'),
    ('Company_C', 'Q1', '2023'), ('Company_C', 'Q2', '2023'), ('Company_C', 'Q3', '2023'), ('Company_C', 'Q4', '2023'),
    ('Company_C', 'Q1', '2024'), ('Company_C', 'Q2', '2024')
]

multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['Company', 'Quarter', 'Year'])

financial_data = pd.DataFrame({
    'Revenue': np.random.normal(1000000, 200000, len(index_tuples)),
    'Profit': np.random.normal(150000, 50000, len(index_tuples)),
    'Employees': np.random.randint(100, 1000, len(index_tuples)),
    'Growth_Rate': np.random.normal(0.05, 0.15, len(index_tuples))
}, index=multi_index)

print("Multi-Index DataFrame:")
print(financial_data)
print(f"\nIndex levels: {financial_data.index.names}")
print(f"Index shape: {financial_data.index.shape}")

# Method 2: Create using set_index
flat_data = pd.DataFrame({
    'Company': ['A', 'A', 'B', 'B', 'C', 'C'] * 3,
    'Region': ['North', 'South'] * 9,
    'Product': ['Product1', 'Product1', 'Product2', 'Product2', 'Product3', 'Product3'] * 3,
    'Sales': np.random.normal(50000, 10000, 18),
    'Units': np.random.randint(100, 1000, 18)
})

multi_df = flat_data.set_index(['Company', 'Region', 'Product'])
print(f"\nAlternative Multi-Index DataFrame:")
print(multi_df.head(10))
```

#### Basic Multi-Index Querying

```python
print("\n=== BASIC MULTI-INDEX QUERYING ===")

# 1. Single level selection
print("1. Single level selection:")
company_a_data = financial_data.loc['Company_A']
print("All Company_A data:")
print(company_a_data)

# 2. Multiple level selection with tuples
print("\n2. Tuple-based selection:")
specific_quarter = financial_data.loc[('Company_A', 'Q1')]
print("Company_A Q1 data:")
print(specific_quarter)

# Full specification
exact_record = financial_data.loc[('Company_A', 'Q1', '2023')]
print(f"\nExact record (Company_A, Q1, 2023):")
print(exact_record)

# 3. Slice notation
print("\n3. Slice notation:")
slice_data = financial_data.loc['Company_A':'Company_B']
print("Companies A through B:")
print(slice_data.head())

# 4. Cross-section (xs) method
print("\n4. Cross-section method:")
q1_all_companies = financial_data.xs('Q1', level='Quarter')
print("Q1 data for all companies:")
print(q1_all_companies)

# Cross-section with multiple levels
year_2023_data = financial_data.xs('2023', level='Year')
print(f"\n2023 data for all companies:")
print(year_2023_data.head())
```

#### Advanced Querying Techniques

```python
print("\n=== ADVANCED QUERYING TECHNIQUES ===")

# 1. IndexSlice for complex selections
print("1. Using IndexSlice:")
idx = pd.IndexSlice

# Select specific combinations
complex_selection = financial_data.loc[idx[['Company_A', 'Company_C'], 'Q1', :]]
print("Company A & C, Q1, all years:")
print(complex_selection)

# Range selections
range_selection = financial_data.loc[idx[:, 'Q1':'Q3', '2023']]
print(f"\nAll companies, Q1-Q3, 2023:")
print(range_selection)

# 2. Boolean indexing with multi-index
print("\n2. Boolean indexing:")

# Filter by values
high_revenue = financial_data[financial_data['Revenue'] > 1200000]
print("High revenue records:")
print(high_revenue)

# Combine index selection with boolean filtering
profitable_company_a = financial_data.loc['Company_A'][financial_data.loc['Company_A']['Profit'] > 150000]
print(f"\nProfitable Company_A quarters:")
print(profitable_company_a)

# 3. Query method with multi-index
print("\n3. Query method:")

# Reset index temporarily for query
temp_df = financial_data.reset_index()
query_result = temp_df.query("Company == 'Company_A' and Revenue > 1000000 and Year == '2023'")
print("Query result:")
print(query_result[['Company', 'Quarter', 'Year', 'Revenue']])

# 4. isin() method for multiple values
print("\n4. isin() method:")

# Select multiple specific combinations
companies_of_interest = ['Company_A', 'Company_C']
quarters_of_interest = ['Q1', 'Q4']

mask = (financial_data.index.get_level_values('Company').isin(companies_of_interest) & 
        financial_data.index.get_level_values('Quarter').isin(quarters_of_interest))

filtered_data = financial_data[mask]
print("Companies A & C, Q1 & Q4:")
print(filtered_data)
```

#### Aggregation and Grouping with Multi-Index

```python
print("\n=== AGGREGATION WITH MULTI-INDEX ===")

# 1. Level-based aggregation
print("1. Level-based aggregation:")

# Aggregate by company (level 0)
company_totals = financial_data.groupby(level='Company').agg({
    'Revenue': ['sum', 'mean'],
    'Profit': ['sum', 'mean'],
    'Employees': 'mean',
    'Growth_Rate': 'mean'
})

print("Company-level aggregations:")
print(company_totals)

# Aggregate by year (level 2)
yearly_totals = financial_data.groupby(level='Year').agg({
    'Revenue': 'sum',
    'Profit': 'sum',
    'Employees': 'sum'
})

print(f"\nYearly totals:")
print(yearly_totals)

# 2. Multiple level aggregation
print("\n2. Multiple level aggregation:")

company_year_agg = financial_data.groupby(level=['Company', 'Year']).agg({
    'Revenue': 'sum',
    'Profit': 'mean',
    'Growth_Rate': 'mean'
})

print("Company-Year aggregations:")
print(company_year_agg)

# 3. Custom aggregation functions
print("\n3. Custom aggregation:")

def growth_consistency(series):
    """Calculate coefficient of variation as consistency measure."""
    return series.std() / series.mean() if series.mean() != 0 else np.nan

consistency_analysis = financial_data.groupby(level='Company').agg({
    'Growth_Rate': [growth_consistency, 'std', 'mean']
})

print("Growth consistency by company:")
print(consistency_analysis.round(4))

# 4. Pivot-like operations
print("\n4. Pivot-like operations:")

# Unstack to create pivot-like structure
unstacked = financial_data['Revenue'].unstack(level='Quarter')
print("Revenue by company and quarter:")
print(unstacked)

# Multiple level unstacking
double_unstacked = financial_data['Revenue'].unstack(level=['Quarter', 'Year'])
print(f"\nRevenue unstacked by quarter and year:")
print(double_unstacked.head())
```

#### Complex Query Patterns

```python
print("\n=== COMPLEX QUERY PATTERNS ===")

# 1. Conditional multi-level querying
print("1. Conditional multi-level queries:")

def get_top_performers(df, metric='Revenue', top_n=2):
    """Get top N performers for each company."""
    return df.groupby(level='Company').apply(
        lambda x: x.nlargest(top_n, metric)
    )

top_revenue_quarters = get_top_performers(financial_data, 'Revenue', 2)
print("Top 2 revenue quarters per company:")
print(top_revenue_quarters[['Revenue', 'Profit']])

# 2. Time-based queries
print("\n2. Time-based analysis:")

# Calculate quarter-over-quarter growth
def calculate_qoq_growth(group):
    """Calculate quarter-over-quarter growth."""
    return group.pct_change()

qoq_growth = financial_data.groupby(level='Company')['Revenue'].apply(calculate_qoq_growth)
financial_data['QoQ_Growth'] = qoq_growth

print("Quarter-over-quarter growth:")
print(financial_data[['Revenue', 'QoQ_Growth']].head(8))

# 3. Cross-level comparisons
print("\n3. Cross-level comparisons:")

# Compare Q4 performance across years
q4_comparison = financial_data.xs('Q4', level='Quarter')[['Revenue', 'Profit']]
print("Q4 performance comparison:")
print(q4_comparison)

# Year-over-year comparison for same quarters
def yoy_comparison(df, quarter):
    """Compare year-over-year for specific quarter."""
    quarter_data = df.xs(quarter, level='Quarter')
    
    # If we have multiple years, calculate YoY growth
    if len(quarter_data.index.get_level_values('Year').unique()) > 1:
        yoy_growth = quarter_data.groupby(level='Company')['Revenue'].pct_change()
        return yoy_growth
    else:
        return None

q1_yoy = yoy_comparison(financial_data, 'Q1')
if q1_yoy is not None:
    print(f"\nQ1 Year-over-Year growth:")
    print(q1_yoy.dropna())

# 4. Multi-condition filtering
print("\n4. Multi-condition filtering:")

# Complex conditions across levels
def multi_condition_filter(df):
    """Apply multiple conditions across different index levels."""
    conditions = (
        (df.index.get_level_values('Company').isin(['Company_A', 'Company_B'])) &
        (df.index.get_level_values('Year') == '2023') &
        (df['Revenue'] > df['Revenue'].median()) &
        (df['Profit'] > 0)
    )
    return df[conditions]

filtered_results = multi_condition_filter(financial_data)
print("Multi-condition filtered results:")
print(filtered_results[['Revenue', 'Profit']])
```

#### Performance Optimization

```python
print("\n=== PERFORMANCE OPTIMIZATION ===")

# 1. Index sorting for better performance
print("1. Index sorting importance:")
import time

# Create unsorted multi-index
unsorted_data = financial_data.sample(frac=1)  # Shuffle rows
print("Unsorted index:")
print(f"Is sorted: {unsorted_data.index.is_monotonic_increasing}")

# Timing unsorted access
start_time = time.time()
result_unsorted = unsorted_data.loc[('Company_A', 'Q1')]
unsorted_time = time.time() - start_time

# Sort and time again
sorted_data = unsorted_data.sort_index()
start_time = time.time()
result_sorted = sorted_data.loc[('Company_A', 'Q1')]
sorted_time = time.time() - start_time

print(f"Unsorted access time: {unsorted_time:.6f} seconds")
print(f"Sorted access time: {sorted_time:.6f} seconds")
print(f"Speedup: {unsorted_time/sorted_time:.1f}x")

# 2. Level-specific operations
print("\n2. Efficient level operations:")

# Using get_level_values for filtering
companies = financial_data.index.get_level_values('Company')
target_companies = companies.isin(['Company_A', 'Company_B'])

efficient_filter = financial_data[target_companies]
print("Efficient company filtering:")
print(f"Filtered data shape: {efficient_filter.shape}")

# 3. Memory usage considerations
print("\n3. Memory usage:")
print(f"Original DataFrame memory: {financial_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
print(f"Index memory usage: {financial_data.index.memory_usage(deep=True) / 1024:.2f} KB")

# Optimizing with categorical data
financial_data_cat = financial_data.copy()
for level in ['Company', 'Quarter', 'Year']:
    level_values = financial_data_cat.index.get_level_values(level).astype('category')
    
print("Memory optimization with categorical indexes implemented")
```

#### Best Practices and Common Patterns

```python
print("\n=== BEST PRACTICES ===")

# 1. Safe multi-index operations
def safe_multiindex_query(df, query_dict):
    """Safely query multi-index DataFrame with error handling."""
    try:
        # Build selection tuple
        selection = []
        for level in df.index.names:
            if level in query_dict:
                selection.append(query_dict[level])
            else:
                selection.append(slice(None))  # Select all for this level
        
        result = df.loc[tuple(selection)]
        return result
    
    except KeyError as e:
        print(f"Key error in query: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error in multi-index query: {e}")
        return pd.DataFrame()

# Example usage
safe_query = {'Company': 'Company_A', 'Quarter': 'Q1'}
safe_result = safe_multiindex_query(financial_data, safe_query)
print("Safe query result:")
print(safe_result.head())

# 2. Validation functions
def validate_multiindex_structure(df, expected_levels):
    """Validate multi-index structure."""
    validation_results = {
        'has_multiindex': isinstance(df.index, pd.MultiIndex),
        'level_count': len(df.index.names) if isinstance(df.index, pd.MultiIndex) else 1,
        'expected_levels': len(expected_levels),
        'level_names_match': df.index.names == expected_levels if isinstance(df.index, pd.MultiIndex) else False,
        'is_sorted': df.index.is_monotonic_increasing if isinstance(df.index, pd.MultiIndex) else True
    }
    
    return validation_results

validation = validate_multiindex_structure(financial_data, ['Company', 'Quarter', 'Year'])
print(f"\n2. Multi-index validation:")
for key, value in validation.items():
    print(f"   {key}: {value}")

# 3. Common utility functions
def flatten_multiindex_columns(df):
    """Flatten multi-level column names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

def reset_specific_levels(df, levels_to_reset):
    """Reset only specific index levels."""
    remaining_levels = [level for level in df.index.names if level not in levels_to_reset]
    return df.reset_index(level=levels_to_reset).set_index(remaining_levels, append=True)

print("\n3. Utility functions implemented")

# Best practices summary
best_practices = [
    "1. Always sort multi-index for optimal performance (sort_index())",
    "2. Use meaningful names for index levels (names parameter)",
    "3. Prefer xs() for cross-sections over complex loc selections",
    "4. Use IndexSlice (pd.IndexSlice) for complex selections",
    "5. Consider memory usage - use categorical for repeated string values",
    "6. Validate index structure before complex operations",
    "7. Use get_level_values() for efficient level-based filtering",
    "8. Document multi-index structure and query patterns",
    "9. Use swaplevel() and reorder_levels() for index manipulation",
    "10. Consider unstacking for pivot-like analysis"
]

print("\nBest Practices:")
for practice in best_practices:
    print(practice)

print("\nCommon Pitfalls:")
pitfalls = [
    "• Not sorting multi-index leading to performance issues",
    "• Confusing tuple syntax in .loc selections",
    "• Forgetting to handle missing levels in queries", 
    "• Memory inefficiency with string-based indexes",
    "• Not validating index structure consistency",
    "• Overcomplicating queries when simpler methods exist",
    "• Ignoring level names in aggregations",
    "• Not considering alternative data structures when appropriate"
]

for pitfall in pitfalls:
    print(pitfall)
```

#### Summary Query Reference

| Operation | Method | Syntax | Use Case |
|-----------|--------|--------|----------|
| **Single level** | `.loc[]` | `df.loc['level1_value']` | Select all data for one level |
| **Multiple levels** | `.loc[]` | `df.loc[('val1', 'val2')]` | Exact multi-level selection |
| **Cross-section** | `.xs()` | `df.xs('value', level='name')` | Slice across specific level |
| **Range selection** | `IndexSlice` | `df.loc[idx['val1':'val2', :]]` | Range across levels |
| **Boolean filtering** | Standard | `df[condition]` | Value-based filtering |
| **Level aggregation** | `.groupby()` | `df.groupby(level='name')` | Aggregate by index level |
| **Multiple conditions** | Combined | `df[mask1 & mask2]` | Complex filtering |
| **Pivot-like** | `.unstack()` | `df.unstack(level='name')` | Reshape multi-index data |

**Key Decision Points:**
- **Performance**: Always sort_index() for repeated queries
- **Complexity**: Use xs() for simple cross-sections, IndexSlice for complex selections  
- **Memory**: Consider categorical indexes for repeated string values
- **Readability**: Prefer named levels over numeric level references
- **Flexibility**: Design index structure to match common query patterns

---

## Question 13

**Explain how you would export a DataFrame to different file formats for reporting purposes.**

### Answer

#### Theory
Exporting DataFrames to various file formats is crucial for data sharing, reporting, and integration with different systems. Pandas provides comprehensive export capabilities for multiple formats including CSV, Excel, JSON, Parquet, HTML, and database formats. Each format has specific advantages, use cases, and configuration options that optimize data preservation, file size, and compatibility with target systems.

#### Core Export Methods Overview

```python
import pandas as pd
import numpy as np
from datetime import datetime, date
import os

# Create comprehensive sample data for export demonstrations
np.random.seed(42)
sample_data = pd.DataFrame({
    'employee_id': range(1, 101),
    'name': [f'Employee_{i}' for i in range(1, 101)],
    'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], 100),
    'salary': np.random.normal(75000, 20000, 100).round(2),
    'hire_date': pd.date_range('2020-01-01', periods=100, freq='W'),
    'performance_score': np.random.uniform(1, 10, 100).round(2),
    'is_remote': np.random.choice([True, False], 100),
    'bonus_percentage': np.random.uniform(0, 0.2, 100).round(4),
    'projects_completed': np.random.randint(0, 50, 100),
    'notes': [f'Performance notes for employee {i}' if i % 3 == 0 else '' for i in range(100)]
})

# Add some missing values
sample_data.loc[5:7, 'performance_score'] = np.nan
sample_data.loc[10:12, 'bonus_percentage'] = np.nan

print("Sample Employee Data for Export:")
print(sample_data.head(10))
print(f"\nData shape: {sample_data.shape}")
print(f"Data types:\n{sample_data.dtypes}")
print(f"Missing values:\n{sample_data.isnull().sum()}")

# Create output directory for exports
output_dir = "data_exports"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
```

#### 1. CSV Export - Universal Compatibility

```python
print("\n=== CSV EXPORT ===")

# Basic CSV export
print("1. Basic CSV export:")
csv_file = f"{output_dir}/employee_data_basic.csv"
sample_data.to_csv(csv_file)
print(f"Basic CSV exported to: {csv_file}")

# Advanced CSV export with options
print("\n2. Advanced CSV export:")
advanced_csv = f"{output_dir}/employee_data_advanced.csv"
sample_data.to_csv(
    advanced_csv,
    index=False,              # Don't include row numbers
    sep=';',                  # Use semicolon separator
    encoding='utf-8',         # Specify encoding
    na_rep='N/A',            # Replace NaN with 'N/A'
    date_format='%Y-%m-%d',  # Format dates
    float_format='%.2f',     # Format floats to 2 decimal places
    columns=['employee_id', 'name', 'department', 'salary', 'hire_date']  # Select specific columns
)
print(f"Advanced CSV exported to: {advanced_csv}")

# CSV export with custom formatting
print("\n3. CSV with preprocessing:")
formatted_data = sample_data.copy()
formatted_data['salary_formatted'] = formatted_data['salary'].apply(lambda x: f"${x:,.2f}")
formatted_data['hire_date_formatted'] = formatted_data['hire_date'].dt.strftime('%B %d, %Y')

formatted_csv = f"{output_dir}/employee_data_formatted.csv"
formatted_data[['employee_id', 'name', 'department', 'salary_formatted', 'hire_date_formatted']].to_csv(
    formatted_csv, 
    index=False
)
print(f"Formatted CSV exported to: {formatted_csv}")

# Handle large files with chunking
print("\n4. Large file export with chunking:")
def export_large_csv_chunks(df, filename, chunk_size=1000):
    """Export large DataFrame in chunks to manage memory."""
    total_rows = len(df)
    chunks_written = 0
    
    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        mode = 'w' if i == 0 else 'a'
        header = i == 0
        
        chunk.to_csv(filename, mode=mode, header=header, index=False)
        chunks_written += 1
    
    return chunks_written

chunks = export_large_csv_chunks(sample_data, f"{output_dir}/chunked_export.csv", chunk_size=25)
print(f"Exported in {chunks} chunks")
```

#### 2. Excel Export - Rich Formatting Options

```python
print("\n=== EXCEL EXPORT ===")

# Basic Excel export
print("1. Basic Excel export:")
excel_file = f"{output_dir}/employee_data.xlsx"
sample_data.to_excel(excel_file, index=False, sheet_name='Employee_Data')
print(f"Basic Excel exported to: {excel_file}")

# Multiple sheets export
print("\n2. Multiple sheets export:")
multi_sheet_file = f"{output_dir}/employee_analysis.xlsx"

# Create different views of the data
summary_by_dept = sample_data.groupby('department').agg({
    'salary': ['mean', 'median', 'std'],
    'performance_score': 'mean',
    'projects_completed': 'sum',
    'employee_id': 'count'
}).round(2)

summary_by_dept.columns = ['_'.join(col).strip() for col in summary_by_dept.columns]

high_performers = sample_data[sample_data['performance_score'] > 8.0].copy()

# Export to multiple sheets
with pd.ExcelWriter(multi_sheet_file, engine='openpyxl') as writer:
    sample_data.to_excel(writer, sheet_name='Raw_Data', index=False)
    summary_by_dept.to_excel(writer, sheet_name='Department_Summary')
    high_performers.to_excel(writer, sheet_name='High_Performers', index=False)

print(f"Multi-sheet Excel exported to: {multi_sheet_file}")

# Advanced Excel formatting
print("\n3. Advanced Excel formatting:")
try:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows
    
    formatted_excel = f"{output_dir}/formatted_employee_data.xlsx"
    
    # Export basic data first
    sample_data.to_excel(formatted_excel, index=False, sheet_name='Employees')
    
    # Load and format
    wb = load_workbook(formatted_excel)
    ws = wb['Employees']
    
    # Header formatting
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    
    for col in range(1, len(sample_data.columns) + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    
    # Format salary column
    salary_col = sample_data.columns.get_loc('salary') + 1
    for row in range(2, len(sample_data) + 2):
        ws.cell(row=row, column=salary_col).number_format = '"$"#,##0.00'
    
    # Format date column
    date_col = sample_data.columns.get_loc('hire_date') + 1
    for row in range(2, len(sample_data) + 2):
        ws.cell(row=row, column=date_col).number_format = 'MM/DD/YYYY'
    
    wb.save(formatted_excel)
    print(f"Formatted Excel exported to: {formatted_excel}")
    
except ImportError:
    print("openpyxl not available - basic Excel export only")

# Excel with charts and pivot tables
print("\n4. Excel with analysis:")
analysis_file = f"{output_dir}/employee_analysis_charts.xlsx"

# Create pivot table data
pivot_data = sample_data.pivot_table(
    values=['salary', 'performance_score'],
    index='department',
    aggfunc={'salary': 'mean', 'performance_score': 'mean'}
).round(2)

with pd.ExcelWriter(analysis_file, engine='openpyxl') as writer:
    sample_data.to_excel(writer, sheet_name='Data', index=False)
    pivot_data.to_excel(writer, sheet_name='Pivot_Analysis')

print(f"Analysis Excel exported to: {analysis_file}")
```

#### 3. JSON Export - API and Web Integration

```python
print("\n=== JSON EXPORT ===")

# Basic JSON export
print("1. Basic JSON export:")
json_file = f"{output_dir}/employee_data.json"
sample_data.to_json(json_file, orient='records', date_format='iso', indent=2)
print(f"JSON exported to: {json_file}")

# Different JSON orientations
print("\n2. Different JSON orientations:")
orientations = ['records', 'index', 'values', 'columns']

for orient in orientations:
    orient_file = f"{output_dir}/employee_data_{orient}.json"
    sample_data.head(5).to_json(orient_file, orient=orient, date_format='iso', indent=2)
    print(f"JSON ({orient} orientation) exported to: {orient_file}")

# Nested JSON structure
print("\n3. Nested JSON structure:")
nested_data = sample_data.groupby('department').apply(
    lambda x: x[['employee_id', 'name', 'salary', 'performance_score']].to_dict('records')
).to_dict()

nested_json_file = f"{output_dir}/employee_data_nested.json"
import json
with open(nested_json_file, 'w') as f:
    json.dump(nested_data, f, indent=2, default=str)

print(f"Nested JSON exported to: {nested_json_file}")

# JSON Lines format (for streaming)
print("\n4. JSON Lines format:")
jsonl_file = f"{output_dir}/employee_data.jsonl"
sample_data.to_json(jsonl_file, orient='records', lines=True, date_format='iso')
print(f"JSON Lines exported to: {jsonl_file}")
```

#### 4. Parquet Export - Big Data Optimized

```python
print("\n=== PARQUET EXPORT ===")

try:
    # Basic Parquet export
    print("1. Basic Parquet export:")
    parquet_file = f"{output_dir}/employee_data.parquet"
    sample_data.to_parquet(parquet_file)
    print(f"Parquet exported to: {parquet_file}")
    
    # Parquet with compression
    print("\n2. Compressed Parquet:")
    compressed_parquet = f"{output_dir}/employee_data_compressed.parquet"
    sample_data.to_parquet(compressed_parquet, compression='gzip')
    
    # Compare file sizes
    basic_size = os.path.getsize(parquet_file)
    compressed_size = os.path.getsize(compressed_parquet)
    
    print(f"Basic Parquet size: {basic_size:,} bytes")
    print(f"Compressed Parquet size: {compressed_size:,} bytes")
    print(f"Compression ratio: {basic_size/compressed_size:.2f}x")
    
    # Partitioned Parquet export
    print("\n3. Partitioned Parquet:")
    partitioned_dir = f"{output_dir}/partitioned_data"
    
    # Export partitioned by department
    for dept in sample_data['department'].unique():
        dept_data = sample_data[sample_data['department'] == dept]
        dept_file = f"{partitioned_dir}/department={dept}/data.parquet"
        os.makedirs(os.path.dirname(dept_file), exist_ok=True)
        dept_data.to_parquet(dept_file)
    
    print(f"Partitioned Parquet exported to: {partitioned_dir}")
    
except ImportError:
    print("PyArrow not available - Parquet export requires PyArrow")
```

#### 5. HTML Export - Web Reports

```python
print("\n=== HTML EXPORT ===")

# Basic HTML table
print("1. Basic HTML table:")
html_file = f"{output_dir}/employee_report.html"
sample_data.head(20).to_html(html_file, index=False)
print(f"HTML table exported to: {html_file}")

# Styled HTML report
print("\n2. Styled HTML report:")
styled_html = f"{output_dir}/styled_employee_report.html"

# Create styled version
styled_df = sample_data.head(20).style.format({
    'salary': '${:,.2f}',
    'performance_score': '{:.2f}',
    'bonus_percentage': '{:.2%}',
    'hire_date': lambda x: x.strftime('%Y-%m-%d')
}).background_gradient(subset=['salary', 'performance_score'], cmap='RdYlGn')

styled_df.to_html(styled_html)
print(f"Styled HTML exported to: {styled_html}")

# Complete HTML report with summary
print("\n3. Complete HTML report:")
complete_html = f"{output_dir}/complete_employee_report.html"

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Employee Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #f9f9f9; padding: 20px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Employee Data Report</h1>
    <div class="summary">
        <h2>Summary Statistics</h2>
        <p>Total Employees: {len(sample_data)}</p>
        <p>Average Salary: ${sample_data['salary'].mean():.2f}</p>
        <p>Average Performance Score: {sample_data['performance_score'].mean():.2f}</p>
        <p>Departments: {', '.join(sample_data['department'].unique())}</p>
    </div>
    
    <h2>Department Summary</h2>
    {summary_by_dept.to_html()}
    
    <h2>Top 20 Employees</h2>
    {sample_data.nlargest(20, 'performance_score')[['name', 'department', 'salary', 'performance_score']].to_html(index=False)}
</body>
</html>
"""

with open(complete_html, 'w') as f:
    f.write(html_content)

print(f"Complete HTML report exported to: {complete_html}")
```

#### 6. Database Export

```python
print("\n=== DATABASE EXPORT ===")

try:
    import sqlite3
    
    # SQLite database export
    print("1. SQLite database export:")
    db_file = f"{output_dir}/employee_database.db"
    
    # Create connection
    conn = sqlite3.connect(db_file)
    
    # Export to database
    sample_data.to_sql('employees', conn, if_exists='replace', index=False)
    
    # Create additional tables
    summary_by_dept.to_sql('department_summary', conn, if_exists='replace')
    
    # Verify export
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM employees")
    row_count = cursor.fetchone()[0]
    
    print(f"SQLite database exported to: {db_file}")
    print(f"Records in database: {row_count}")
    
    conn.close()
    
    # PostgreSQL/MySQL export example (connection string format)
    print("\n2. SQL database export (connection string example):")
    connection_examples = {
        'postgresql': 'postgresql://username:password@localhost:5432/database',
        'mysql': 'mysql+pymysql://username:password@localhost:3306/database',
        'mssql': 'mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server'
    }
    
    print("Connection string examples:")
    for db_type, conn_str in connection_examples.items():
        print(f"  {db_type}: {conn_str}")
    
    # Example code (commented out - requires actual database)
    print("\nExample database export code:")
    print("""
    from sqlalchemy import create_engine
    
    # Create engine
    engine = create_engine(connection_string)
    
    # Export DataFrame
    sample_data.to_sql(
        'employees', 
        engine, 
        if_exists='replace',
        index=False,
        method='multi',  # For better performance
        chunksize=1000   # Process in chunks
    )
    """)
    
except ImportError:
    print("SQLite not available in this environment")
```

#### 7. Specialized Formats

```python
print("\n=== SPECIALIZED FORMATS ===")

# Feather format (fast binary format)
print("1. Feather format:")
try:
    feather_file = f"{output_dir}/employee_data.feather"
    sample_data.to_feather(feather_file)
    print(f"Feather exported to: {feather_file}")
except ImportError:
    print("PyArrow required for Feather format")

# Pickle format (Python objects)
print("\n2. Pickle format:")
pickle_file = f"{output_dir}/employee_data.pkl"
sample_data.to_pickle(pickle_file)
print(f"Pickle exported to: {pickle_file}")

# XML export
print("\n3. XML format:")
xml_file = f"{output_dir}/employee_data.xml"
sample_data.head(10).to_xml(xml_file, index=False)
print(f"XML exported to: {xml_file}")

# HDF5 format (hierarchical data)
print("\n4. HDF5 format:")
try:
    hdf_file = f"{output_dir}/employee_data.h5"
    
    # Export multiple datasets
    sample_data.to_hdf(hdf_file, key='employees', mode='w')
    summary_by_dept.to_hdf(hdf_file, key='department_summary', mode='a')
    
    print(f"HDF5 exported to: {hdf_file}")
    
    # List contents
    with pd.HDFStore(hdf_file, 'r') as store:
        print(f"HDF5 keys: {list(store.keys())}")
        
except ImportError:
    print("PyTables required for HDF5 format")

# Stata format
print("\n5. Stata format:")
try:
    stata_file = f"{output_dir}/employee_data.dta"
    
    # Stata has restrictions on variable names and types
    stata_data = sample_data.copy()
    stata_data.columns = [col.replace(' ', '_').lower() for col in stata_data.columns]
    
    # Convert object columns to string
    for col in stata_data.select_dtypes(include=['object']).columns:
        stata_data[col] = stata_data[col].astype(str)
    
    stata_data.to_stata(stata_file, write_index=False)
    print(f"Stata exported to: {stata_file}")
    
except ImportError:
    print("Stata export not available")
```

#### 8. Performance Optimization and Best Practices

```python
print("\n=== PERFORMANCE OPTIMIZATION ===")

# File size comparison
print("1. File size comparison:")
formats_to_test = ['csv', 'json', 'parquet', 'pickle']
file_sizes = {}

test_data = sample_data.head(50)  # Use subset for demonstration

for fmt in formats_to_test:
    try:
        test_file = f"{output_dir}/size_test.{fmt}"
        
        if fmt == 'csv':
            test_data.to_csv(test_file, index=False)
        elif fmt == 'json':
            test_data.to_json(test_file, orient='records')
        elif fmt == 'parquet':
            test_data.to_parquet(test_file)
        elif fmt == 'pickle':
            test_data.to_pickle(test_file)
        
        file_sizes[fmt] = os.path.getsize(test_file)
        
    except Exception as e:
        print(f"Error with {fmt}: {e}")

print("File size comparison (50 records):")
for fmt, size in sorted(file_sizes.items(), key=lambda x: x[1]):
    print(f"  {fmt}: {size:,} bytes")

# Export performance timing
print("\n2. Export performance timing:")
import time

large_data = pd.concat([sample_data] * 10, ignore_index=True)  # 1000 rows

performance_results = {}

formats_to_time = ['csv', 'json', 'parquet', 'pickle']

for fmt in formats_to_time:
    try:
        start_time = time.time()
        
        if fmt == 'csv':
            large_data.to_csv(f"{output_dir}/perf_test.csv", index=False)
        elif fmt == 'json':
            large_data.to_json(f"{output_dir}/perf_test.json", orient='records')
        elif fmt == 'parquet':
            large_data.to_parquet(f"{output_dir}/perf_test.parquet")
        elif fmt == 'pickle':
            large_data.to_pickle(f"{output_dir}/perf_test.pkl")
        
        performance_results[fmt] = time.time() - start_time
        
    except Exception as e:
        print(f"Error timing {fmt}: {e}")

print("Export performance (1000 records):")
for fmt, time_taken in sorted(performance_results.items(), key=lambda x: x[1]):
    print(f"  {fmt}: {time_taken:.4f} seconds")

# Memory-efficient export strategies
print("\n3. Memory-efficient export:")

def chunk_export_csv(df, filename, chunk_size=1000):
    """Export large DataFrame in chunks."""
    total_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        mode = 'w' if i == 0 else 'a'
        header = i == 0
        chunk.to_csv(filename, mode=mode, header=header, index=False)
    
    return total_chunks

# Best practices summary
print("\n=== BEST PRACTICES SUMMARY ===")

best_practices = {
    'CSV': [
        "• Use for maximum compatibility",
        "• Specify encoding (UTF-8) for international characters",
        "• Use appropriate separators for locale",
        "• Handle missing values explicitly",
        "• Consider compression for large files"
    ],
    'Excel': [
        "• Use for business users and formatted reports", 
        "• Leverage multiple sheets for organization",
        "• Apply formatting for readability",
        "• Be aware of row limits (1M rows in modern Excel)",
        "• Use xlsxwriter for advanced formatting"
    ],
    'JSON': [
        "• Use for web APIs and JavaScript integration",
        "• Choose appropriate orientation for use case",
        "• Use date_format='iso' for standard dates",
        "• Consider JSON Lines for streaming",
        "• Validate JSON structure before export"
    ],
    'Parquet': [
        "• Use for big data and analytics workflows",
        "• Leverage compression for storage efficiency", 
        "• Use partitioning for large datasets",
        "• Preserves data types and schema",
        "• Excellent for columnar analysis"
    ],
    'Database': [
        "• Use for persistent storage and queries",
        "• Consider indexing strategy",
        "• Use chunking for large datasets",
        "• Handle data type conversions",
        "• Implement proper error handling"
    ]
}

for format_type, practices in best_practices.items():
    print(f"\n{format_type}:")
    for practice in practices:
        print(f"  {practice}")

print(f"\n=== FORMAT SELECTION GUIDE ===")

selection_guide = pd.DataFrame({
    'Format': ['CSV', 'Excel', 'JSON', 'Parquet', 'Database', 'HTML'],
    'Use Case': [
        'Data exchange, simple reports',
        'Business reports, formatted output',
        'Web APIs, JavaScript apps',
        'Big data, analytics pipelines',
        'Persistent storage, queries',
        'Web reports, documentation'
    ],
    'Pros': [
        'Universal, simple, human-readable',
        'Rich formatting, multiple sheets',
        'Web-native, structured',
        'Efficient, preserves types',
        'Queryable, persistent',
        'Web-ready, styled'
    ],
    'Cons': [
        'No data types, large size',
        'Proprietary, size limits',
        'Can be large, limited types',
        'Requires libraries',
        'Setup complexity',
        'Static, no data types'
    ]
})

print(selection_guide.to_string(index=False))
```

#### Summary Decision Matrix

| Export Goal | Primary Format | Secondary Format | Key Considerations |
|-------------|---------------|------------------|-------------------|
| **Data Exchange** | CSV | JSON | Compatibility, encoding |
| **Business Reports** | Excel | HTML | Formatting, multiple sheets |
| **Web Integration** | JSON | CSV | API compatibility, size |
| **Big Data Analytics** | Parquet | CSV | Compression, performance |
| **Archival Storage** | Parquet | Database | Long-term preservation |
| **Dashboard/BI** | Database | Parquet | Query performance |
| **Documentation** | HTML | Excel | Presentation, styling |
| **Data Science** | Pickle | Parquet | Type preservation |

**Key Decision Factors:**
- **Audience**: Technical (Parquet/JSON) vs Business (Excel/CSV)
- **Volume**: Small (Excel/CSV) vs Large (Parquet/Database)
- **Usage**: Analysis (Parquet) vs Exchange (CSV) vs Presentation (Excel/HTML)
- **Integration**: APIs (JSON) vs BI tools (Database/Parquet)
- **Longevity**: Standards-based (CSV/JSON) vs Optimized (Parquet)

---

## Question 14

**How does one use Dask or Modin to handle larger-than-memory data in Pandas?**

### Answer

#### Theory
When datasets exceed available RAM, traditional Pandas operations fail with memory errors. Dask and Modin are two primary solutions for scaling Pandas beyond memory limitations. **Dask** provides lazy evaluation and parallel computing with familiar Pandas-like API, while **Modin** aims for drop-in Pandas replacement with automatic parallelization. Both enable processing datasets that are 10x-100x larger than available memory through chunking, lazy evaluation, and distributed computing.

#### Understanding the Challenge

```python
import pandas as pd
import numpy as np
import psutil
import os
from typing import Optional, List, Dict, Any

# Memory analysis utilities
def get_memory_info():
    """Get current memory usage information."""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent_used': memory.percent
    }

def estimate_dataframe_memory(rows: int, cols: int, avg_string_length: int = 20) -> float:
    """Estimate DataFrame memory usage in GB."""
    # Rough estimation: numeric=8 bytes, string=avg_length bytes
    numeric_cols = cols * 0.7  # Assume 70% numeric
    string_cols = cols * 0.3   # Assume 30% string
    
    memory_bytes = rows * (numeric_cols * 8 + string_cols * avg_string_length)
    return memory_bytes / (1024**3)

print("=== MEMORY CONSTRAINTS ANALYSIS ===")
memory_info = get_memory_info()
print(f"System Memory: {memory_info['total_gb']:.1f} GB total, {memory_info['available_gb']:.1f} GB available")

# Example memory requirements
sample_datasets = [
    {"name": "Small Dataset", "rows": 100_000, "cols": 50},
    {"name": "Medium Dataset", "rows": 1_000_000, "cols": 100},
    {"name": "Large Dataset", "rows": 10_000_000, "cols": 200},
    {"name": "Very Large Dataset", "rows": 100_000_000, "cols": 500},
]

print("\nEstimated Memory Requirements:")
for dataset in sample_datasets:
    memory_needed = estimate_dataframe_memory(dataset["rows"], dataset["cols"])
    print(f"  {dataset['name']}: {memory_needed:.2f} GB")
    if memory_needed > memory_info['available_gb']:
        print(f"    ⚠️  Exceeds available memory - requires Dask/Modin")
    else:
        print(f"    ✅ Fits in memory - regular Pandas OK")

# Demonstrate memory limitations
print("\n=== PANDAS MEMORY LIMITATIONS ===")

def create_memory_heavy_dataframe(rows: int = 1_000_000):
    """Create a memory-intensive DataFrame for demonstration."""
    np.random.seed(42)
    
    print(f"Creating DataFrame with {rows:,} rows...")
    initial_memory = psutil.Process().memory_info().rss / (1024**2)
    
    try:
        df = pd.DataFrame({
            'id': range(rows),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], rows),
            'value1': np.random.randn(rows),
            'value2': np.random.randn(rows),
            'value3': np.random.randn(rows),
            'timestamp': pd.date_range('2020-01-01', periods=rows, freq='1min'),
            'description': [f'Item_{i}_{"_" * 20}' for i in range(rows)]  # Memory intensive strings
        })
        
        final_memory = psutil.Process().memory_info().rss / (1024**2)
        memory_used = final_memory - initial_memory
        
        print(f"DataFrame created successfully!")
        print(f"Memory used: {memory_used:.1f} MB")
        print(f"DataFrame info:")
        print(f"  Shape: {df.shape}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
        
        return df
        
    except MemoryError:
        print("❌ MemoryError: DataFrame too large for available memory")
        return None

# Create manageable test DataFrame
test_df = create_memory_heavy_dataframe(500_000)  # Start with manageable size
```

#### 1. Dask - Distributed Computing Solution

```python
print("\n=== DASK IMPLEMENTATION ===")

try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster
    
    print("1. Setting up Dask cluster:")
    
    # Create local cluster for demonstration
    def setup_dask_cluster(n_workers: int = 2, memory_limit: str = '2GB'):
        """Set up local Dask cluster."""
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=2,
            memory_limit=memory_limit,
            dashboard_address=':8787'
        )
        client = Client(cluster)
        
        print(f"Dask cluster created:")
        print(f"  Dashboard: {client.dashboard_link}")
        print(f"  Workers: {n_workers}")
        print(f"  Memory per worker: {memory_limit}")
        
        return client, cluster
    
    # Setup cluster
    client, cluster = setup_dask_cluster()
    
    print("\n2. Creating large Dask DataFrame:")
    
    # Create Dask DataFrame from partitions
    def create_large_dask_dataframe(total_rows: int = 5_000_000, chunk_size: int = 100_000):
        """Create large Dask DataFrame using chunking."""
        
        print(f"Creating Dask DataFrame with {total_rows:,} rows in chunks of {chunk_size:,}")
        
        # Function to create a single partition
        def make_partition(partition_id, chunk_size):
            np.random.seed(partition_id)  # Reproducible partitions
            rows = min(chunk_size, total_rows - partition_id * chunk_size)
            
            return pd.DataFrame({
                'id': range(partition_id * chunk_size, partition_id * chunk_size + rows),
                'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], rows),
                'value1': np.random.randn(rows),
                'value2': np.random.randn(rows),
                'value3': np.random.randn(rows),
                'score': np.random.uniform(0, 100, rows),
                'timestamp': pd.date_range('2020-01-01', periods=rows, freq='1min')
            })
        
        # Create delayed objects for each partition
        partitions = []
        n_partitions = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)
        
        for i in range(n_partitions):
            partition = dask.delayed(make_partition)(i, chunk_size)
            partitions.append(partition)
        
        # Convert to Dask DataFrame
        ddf = dd.from_delayed(partitions)
        
        print(f"Dask DataFrame created:")
        print(f"  Partitions: {ddf.npartitions}")
        print(f"  Estimated size: {estimate_dataframe_memory(total_rows, 7):.2f} GB")
        
        return ddf
    
    # Create large Dask DataFrame
    large_ddf = create_large_dask_dataframe()
    
    print("\n3. Dask operations (lazy evaluation):")
    
    # Basic operations
    print("Basic information (computed lazily):")
    print(f"  Columns: {list(large_ddf.columns)}")
    print(f"  Partitions: {large_ddf.npartitions}")
    
    # Lazy operations
    print("\nLazy operations:")
    mean_score = large_ddf['score'].mean()
    max_value = large_ddf['value1'].max()
    category_counts = large_ddf['category'].value_counts()
    
    print(f"  Mean score (lazy): {type(mean_score)}")
    print(f"  Max value (lazy): {type(max_value)}")
    
    # Compute results
    print("\nComputing results:")
    import time
    start_time = time.time()
    
    mean_result = mean_score.compute()
    max_result = max_value.compute()
    counts_result = category_counts.compute()
    
    compute_time = time.time() - start_time
    
    print(f"  Mean score: {mean_result:.2f}")
    print(f"  Max value: {max_result:.2f}")
    print(f"  Category counts: {dict(counts_result)}")
    print(f"  Computation time: {compute_time:.2f} seconds")
    
    print("\n4. Advanced Dask operations:")
    
    # Groupby operations
    print("Groupby operations:")
    groupby_result = large_ddf.groupby('category').agg({
        'score': ['mean', 'std', 'count'],
        'value1': 'sum'
    }).compute()
    
    print("Category statistics:")
    print(groupby_result)
    
    # Filtering and transformation
    print("\nFiltering and transformation:")
    high_scores = large_ddf[large_ddf['score'] > 90]
    transformed = large_ddf.assign(
        score_category=large_ddf['score'].apply(
            lambda x: 'High' if x > 80 else 'Medium' if x > 50 else 'Low',
            meta=('score_category', 'object')
        )
    )
    
    high_score_count = len(high_scores)
    score_category_counts = transformed['score_category'].value_counts().compute()
    
    print(f"High scores (>90): {high_score_count:,}")
    print(f"Score categories: {dict(score_category_counts)}")
    
    print("\n5. Data persistence and export:")
    
    # Save to multiple formats
    print("Saving Dask DataFrame:")
    
    # Parquet (recommended for Dask)
    output_dir = "dask_output"
    os.makedirs(output_dir, exist_ok=True)
    
    parquet_path = f"{output_dir}/large_dataset.parquet"
    large_ddf.to_parquet(parquet_path, engine='pyarrow')
    print(f"  Saved to Parquet: {parquet_path}")
    
    # CSV (partitioned)
    csv_path = f"{output_dir}/large_dataset_csv"
    large_ddf.to_csv(f"{csv_path}/*.csv", index=False)
    print(f"  Saved to CSV: {csv_path}")
    
    # Reading back
    print("Reading back from storage:")
    loaded_ddf = dd.read_parquet(parquet_path)
    print(f"  Loaded DataFrame shape: {loaded_ddf.compute().shape}")
    
    # Clean up
    client.close()
    cluster.close()
    
except ImportError:
    print("Dask not installed. Install with: pip install dask[complete]")
except Exception as e:
    print(f"Dask setup error: {e}")
```

#### 2. Modin - Drop-in Pandas Replacement

```python
print("\n=== MODIN IMPLEMENTATION ===")

try:
    # Modin with Ray backend
    import modin.pandas as mpd
    import ray
    
    print("1. Setting up Modin with Ray:")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=4, object_store_memory=1000000000)  # 1GB object store
    
    print("Ray cluster initialized")
    print(f"  Available CPUs: {ray.cluster_resources().get('CPU', 0)}")
    
    print("\n2. Creating large Modin DataFrame:")
    
    def create_large_modin_dataframe(rows: int = 2_000_000):
        """Create large Modin DataFrame."""
        print(f"Creating Modin DataFrame with {rows:,} rows...")
        
        # Modin handles memory management automatically
        start_time = time.time()
        
        # Create DataFrame (similar to Pandas syntax)
        mdf = mpd.DataFrame({
            'id': range(rows),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], rows),
            'value1': np.random.randn(rows),
            'value2': np.random.randn(rows),
            'score': np.random.uniform(0, 100, rows),
            'timestamp': mpd.date_range('2020-01-01', periods=rows, freq='1min')
        })
        
        creation_time = time.time() - start_time
        print(f"Modin DataFrame created in {creation_time:.2f} seconds")
        print(f"Shape: {mdf.shape}")
        
        return mdf
    
    large_mdf = create_large_modin_dataframe()
    
    print("\n3. Modin operations (automatic parallelization):")
    
    # Basic operations (identical to Pandas)
    print("Basic operations:")
    start_time = time.time()
    
    mean_score = large_mdf['score'].mean()
    std_score = large_mdf['score'].std()
    category_counts = large_mdf['category'].value_counts()
    
    operation_time = time.time() - start_time
    
    print(f"  Mean score: {mean_score:.2f}")
    print(f"  Std score: {std_score:.2f}")
    print(f"  Category counts: {dict(category_counts)}")
    print(f"  Operation time: {operation_time:.2f} seconds")
    
    # Advanced operations
    print("\nAdvanced operations:")
    
    # Groupby (automatically parallelized)
    groupby_stats = large_mdf.groupby('category').agg({
        'score': ['mean', 'std', 'count'],
        'value1': ['min', 'max']
    })
    
    print("Groupby statistics:")
    print(groupby_stats)
    
    # Filtering and transformation
    high_performers = large_mdf[large_mdf['score'] > 95]
    print(f"High performers (score > 95): {len(high_performers):,}")
    
    # Apply operations
    large_mdf['score_rank'] = large_mdf['score'].rank(method='dense')
    large_mdf['normalized_value'] = (large_mdf['value1'] - large_mdf['value1'].mean()) / large_mdf['value1'].std()
    
    print("Added derived columns: score_rank, normalized_value")
    
    print("\n4. Converting between Modin and Pandas:")
    
    # Modin to Pandas (for small results)
    sample_pandas = large_mdf.head(1000)._to_pandas()
    print(f"Converted sample to Pandas: {type(sample_pandas)}")
    
    # Pandas to Modin
    pandas_df = pd.DataFrame({'test': range(1000)})
    modin_from_pandas = mpd.DataFrame(pandas_df)
    print(f"Converted Pandas to Modin: {type(modin_from_pandas)}")
    
    print("\n5. Modin I/O operations:")
    
    # Reading/writing files (parallelized)
    output_file = "modin_output/large_dataset.csv"
    os.makedirs("modin_output", exist_ok=True)
    
    # Write CSV
    start_time = time.time()
    large_mdf.to_csv(output_file, index=False)
    write_time = time.time() - start_time
    print(f"CSV write time: {write_time:.2f} seconds")
    
    # Read CSV
    start_time = time.time()
    loaded_mdf = mpd.read_csv(output_file)
    read_time = time.time() - start_time
    print(f"CSV read time: {read_time:.2f} seconds")
    print(f"Loaded shape: {loaded_mdf.shape}")
    
    # Shutdown Ray
    ray.shutdown()
    
except ImportError:
    print("Modin not installed. Install with: pip install modin[ray] or modin[dask]")
except Exception as e:
    print(f"Modin setup error: {e}")
```

#### 3. Performance Comparison

```python
print("\n=== PERFORMANCE COMPARISON ===")

def benchmark_operations(size: int = 1_000_000):
    """Benchmark Pandas vs Dask vs Modin operations."""
    
    print(f"Benchmarking with {size:,} rows...")
    
    # Create test data
    np.random.seed(42)
    data = {
        'id': range(size),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size),
        'value1': np.random.randn(size),
        'value2': np.random.randn(size),
        'score': np.random.uniform(0, 100, size)
    }
    
    results = {}
    
    # Pandas benchmark
    try:
        print("\nPandas operations:")
        start_time = time.time()
        
        pandas_df = pd.DataFrame(data)
        pandas_mean = pandas_df['score'].mean()
        pandas_groupby = pandas_df.groupby('category')['score'].mean()
        pandas_filter = pandas_df[pandas_df['score'] > 80]
        
        pandas_time = time.time() - start_time
        results['Pandas'] = pandas_time
        
        print(f"  Time: {pandas_time:.3f} seconds")
        print(f"  Mean score: {pandas_mean:.2f}")
        print(f"  Filtered rows: {len(pandas_filter):,}")
        
    except MemoryError:
        print("  ❌ Pandas: Memory error")
        results['Pandas'] = None
    
    # Dask benchmark
    try:
        import dask.dataframe as dd
        
        print("\nDask operations:")
        start_time = time.time()
        
        # Create Dask DataFrame with smaller partitions
        dask_df = dd.from_pandas(pd.DataFrame(data), npartitions=4)
        dask_mean = dask_df['score'].mean().compute()
        dask_groupby = dask_df.groupby('category')['score'].mean().compute()
        dask_filter = dask_df[dask_df['score'] > 80].compute()
        
        dask_time = time.time() - start_time
        results['Dask'] = dask_time
        
        print(f"  Time: {dask_time:.3f} seconds")
        print(f"  Mean score: {dask_mean:.2f}")
        print(f"  Filtered rows: {len(dask_filter):,}")
        
    except ImportError:
        print("  ⚠️  Dask not available")
        results['Dask'] = None
    
    # Modin benchmark
    try:
        import modin.pandas as mpd
        
        print("\nModin operations:")
        start_time = time.time()
        
        modin_df = mpd.DataFrame(data)
        modin_mean = modin_df['score'].mean()
        modin_groupby = modin_df.groupby('category')['score'].mean()
        modin_filter = modin_df[modin_df['score'] > 80]
        
        modin_time = time.time() - start_time
        results['Modin'] = modin_time
        
        print(f"  Time: {modin_time:.3f} seconds")
        print(f"  Mean score: {modin_mean:.2f}")
        print(f"  Filtered rows: {len(modin_filter):,}")
        
    except ImportError:
        print("  ⚠️  Modin not available")
        results['Modin'] = None
    
    # Performance summary
    print("\nPerformance Summary:")
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        fastest = min(valid_results, key=valid_results.get)
        for tool, time_taken in sorted(valid_results.items(), key=lambda x: x[1]):
            speedup = valid_results[fastest] / time_taken if time_taken > 0 else 1
            print(f"  {tool}: {time_taken:.3f}s ({speedup:.2f}x vs fastest)")
    
    return results

# Run benchmark with manageable size
benchmark_results = benchmark_operations(500_000)
```

#### 4. Best Practices and Implementation Strategies

```python
print("\n=== BEST PRACTICES ===")

class LargeDataHandler:
    """Comprehensive handler for large datasets using Dask/Modin."""
    
    def __init__(self, backend: str = 'dask', memory_limit: str = '2GB'):
        self.backend = backend
        self.memory_limit = memory_limit
        self.client = None
        
        if backend == 'dask':
            self._setup_dask()
        elif backend == 'modin':
            self._setup_modin()
    
    def _setup_dask(self):
        """Setup Dask cluster."""
        try:
            import dask.dataframe as dd
            from dask.distributed import Client, LocalCluster
            
            cluster = LocalCluster(
                memory_limit=self.memory_limit,
                n_workers=2,
                threads_per_worker=2
            )
            self.client = Client(cluster)
            self.dd = dd
            print(f"Dask cluster ready: {self.client.dashboard_link}")
            
        except ImportError:
            raise ImportError("Dask not available")
    
    def _setup_modin(self):
        """Setup Modin with Ray."""
        try:
            import modin.pandas as mpd
            import ray
            
            if not ray.is_initialized():
                ray.init(num_cpus=4)
            
            self.mpd = mpd
            print("Modin with Ray ready")
            
        except ImportError:
            raise ImportError("Modin not available")
    
    def read_large_file(self, filepath: str, **kwargs) -> Any:
        """Read large file using appropriate backend."""
        
        print(f"Reading large file: {filepath}")
        
        if self.backend == 'dask':
            if filepath.endswith('.csv'):
                return self.dd.read_csv(filepath, **kwargs)
            elif filepath.endswith('.parquet'):
                return self.dd.read_parquet(filepath, **kwargs)
            else:
                raise ValueError("Unsupported file format for Dask")
        
        elif self.backend == 'modin':
            if filepath.endswith('.csv'):
                return self.mpd.read_csv(filepath, **kwargs)
            else:
                # Modin has limited file format support
                return self.mpd.read_csv(filepath, **kwargs)
    
    def process_in_chunks(self, data, operation, chunk_size: int = 100_000):
        """Process data in chunks to manage memory."""
        
        if self.backend == 'dask':
            # Dask handles chunking automatically
            return operation(data)
        
        elif self.backend == 'modin':
            # Modin handles parallelization automatically
            return operation(data)
        
        else:  # Fallback to manual chunking
            results = []
            total_rows = len(data)
            
            for i in range(0, total_rows, chunk_size):
                chunk = data.iloc[i:i+chunk_size]
                result = operation(chunk)
                results.append(result)
            
            return pd.concat(results, ignore_index=True)
    
    def optimize_dtypes(self, df):
        """Optimize DataFrame dtypes to reduce memory usage."""
        
        print("Optimizing data types...")
        
        optimization_map = {}
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                # Try to convert to category if beneficial
                unique_ratio = len(df[col].unique()) / len(df[col])
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimization_map[col] = 'category'
            
            elif col_type in ['int64', 'int32']:
                # Downcast integers
                col_min, col_max = df[col].min(), df[col].max()
                if col_min >= 0:
                    if col_max < 255:
                        optimization_map[col] = 'uint8'
                    elif col_max < 65535:
                        optimization_map[col] = 'uint16'
                    elif col_max < 4294967295:
                        optimization_map[col] = 'uint32'
                else:
                    if col_min > -128 and col_max < 127:
                        optimization_map[col] = 'int8'
                    elif col_min > -32768 and col_max < 32767:
                        optimization_map[col] = 'int16'
            
            elif col_type in ['float64']:
                # Downcast floats
                optimization_map[col] = 'float32'
        
        print(f"Optimization suggestions: {optimization_map}")
        return optimization_map
    
    def memory_efficient_groupby(self, df, groupby_cols: List[str], agg_dict: Dict):
        """Memory-efficient groupby operation."""
        
        if self.backend == 'dask':
            return df.groupby(groupby_cols).agg(agg_dict)
        
        elif self.backend == 'modin':
            return df.groupby(groupby_cols).agg(agg_dict)
        
        else:
            # Manual chunked groupby for regular Pandas
            chunk_size = 100_000
            results = []
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                chunk_result = chunk.groupby(groupby_cols).agg(agg_dict)
                results.append(chunk_result)
            
            # Combine results
            combined = pd.concat(results).groupby(groupby_cols).sum()
            return combined
    
    def export_efficiently(self, df, output_path: str, format: str = 'parquet'):
        """Export large DataFrame efficiently."""
        
        print(f"Exporting to {format}: {output_path}")
        
        if format == 'parquet':
            if self.backend == 'dask':
                df.to_parquet(output_path, engine='pyarrow')
            elif self.backend == 'modin':
                df.to_parquet(output_path)
            else:
                df.to_parquet(output_path, engine='pyarrow')
        
        elif format == 'csv':
            if self.backend == 'dask':
                df.to_csv(f"{output_path}/*.csv", index=False)
            else:
                df.to_csv(output_path, index=False)
        
        print(f"Export completed to: {output_path}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.client:
            self.client.close()
        
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except ImportError:
            pass

# Demonstrate usage
print("\n=== LARGE DATA HANDLER DEMO ===")

try:
    # Choose backend based on availability
    backend = 'dask'  # Change to 'modin' to test Modin
    
    handler = LargeDataHandler(backend=backend)
    
    print(f"Using {backend} backend")
    
    # Example workflow
    print("Large data processing workflow:")
    print("1. ✅ Setup cluster/backend")
    print("2. 📂 Read large files with chunking")
    print("3. 🔧 Optimize data types")
    print("4. 📊 Perform memory-efficient operations")
    print("5. 💾 Export results efficiently")
    print("6. 🧹 Cleanup resources")
    
    handler.cleanup()
    
except Exception as e:
    print(f"Demo error: {e}")

# Decision matrix
print("\n=== DASK VS MODIN DECISION MATRIX ===")

comparison_df = pd.DataFrame({
    'Aspect': [
        'API Compatibility',
        'Learning Curve',
        'Performance',
        'Memory Efficiency',
        'Distributed Computing',
        'File Format Support',
        'Ecosystem Integration',
        'Debugging/Monitoring',
        'Production Readiness',
        'Community Support'
    ],
    'Dask': [
        'Similar to Pandas',
        'Moderate (lazy evaluation)',
        'Excellent for large data',
        'Very good (chunking)',
        'Excellent (built-in)',
        'Excellent (many formats)',
        'Very good (ML, viz)',
        'Excellent (dashboard)',
        'Very good',
        'Large, active'
    ],
    'Modin': [
        'Near-identical to Pandas',
        'Low (drop-in replacement)',
        'Good for medium data',
        'Good (automatic)',
        'Good (Ray/Dask backend)',
        'Limited (mainly CSV)',
        'Growing',
        'Limited',
        'Moderate',
        'Growing'
    ],
    'Use Case': [
        'Both suitable',
        'Modin for easy migration',
        'Dask for very large data',
        'Both good',
        'Dask for complex workflows',
        'Dask for variety',
        'Dask more mature',
        'Dask better tools',
        'Dask more proven',
        'Dask more established'
    ]
})

print(comparison_df.to_string(index=False))

print("\n=== IMPLEMENTATION RECOMMENDATIONS ===")

recommendations = {
    'Data Size': {
        '1-10 GB': 'Try Modin first (easier migration)',
        '10-100 GB': 'Use Dask (better memory management)',
        '100+ GB': 'Use Dask with distributed cluster',
        'Streaming': 'Use Dask (better streaming support)'
    },
    'Use Case': {
        'Drop-in replacement': 'Modin (minimal code changes)',
        'Complex workflows': 'Dask (better control)',
        'Machine learning': 'Dask (better ML ecosystem)',
        'ETL pipelines': 'Dask (more robust)',
        'Ad-hoc analysis': 'Either (Modin easier)'
    },
    'Infrastructure': {
        'Single machine': 'Either (both work well)',
        'Multiple machines': 'Dask (distributed by design)',
        'Cloud environments': 'Dask (better cloud integration)',
        'Limited resources': 'Dask (better resource management)'
    }
}

for category, items in recommendations.items():
    print(f"\n{category}:")
    for situation, recommendation in items.items():
        print(f"  {situation}: {recommendation}")
```

#### Key Takeaways

**When to Use Each Solution:**

1. **Choose Dask when:**
   - Working with 10+ GB datasets
   - Need distributed computing
   - Require complex workflows
   - Want extensive monitoring/debugging
   - Working with various file formats

2. **Choose Modin when:**
   - Want minimal code changes
   - Working with 1-10 GB datasets  
   - Need quick performance boost
   - Team is new to parallel computing
   - Mainly using CSV files

3. **Implementation Strategy:**
   - Start with memory profiling
   - Test with subset of data first
   - Monitor cluster resources
   - Implement proper error handling
   - Use appropriate file formats (Parquet recommended)

---

## Question 15

**Describe how you could use Pandas to preprocess data for a machine learning model.**

### Answer

#### Theory
Data preprocessing is a critical phase in machine learning pipelines where raw data is transformed into a format suitable for model training. Pandas provides comprehensive tools for data cleaning, feature engineering, encoding, scaling, and splitting that are essential for successful ML workflows. Proper preprocessing can significantly impact model performance, interpretability, and generalization capabilities.

#### Code Example

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class MLDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for machine learning.
    """
    
    def __init__(self, target_column: str, 
                 test_size: float = 0.2,
                 random_state: int = 42,
                 validation_split: bool = True):
        """
        Initialize preprocessor with configuration.
        
        Parameters:
        target_column (str): Name of target variable column
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        validation_split (bool): Whether to create validation set
        """
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.validation_split = validation_split
        
        # Store preprocessing components
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        
        # Track preprocessing steps
        self.preprocessing_log = []
        self.feature_names = []
        
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality analysis.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        
        Returns:
        dict: Data quality report
        """
        
        quality_report = {
            'shape': df.shape,
            'missing_values': {},
            'data_types': df.dtypes.to_dict(),
            'duplicates': df.duplicated().sum(),
            'target_distribution': {},
            'feature_statistics': {},
            'correlation_insights': {},
            'outlier_detection': {}
        }
        
        # Missing value analysis
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df) * 100).round(2)
        
        quality_report['missing_values'] = {
            'counts': missing_counts.to_dict(),
            'percentages': missing_percentages.to_dict(),
            'total_missing': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist()
        }
        
        # Target variable analysis
        if self.target_column in df.columns:
            target_data = df[self.target_column]
            
            if target_data.dtype in ['object', 'category']:
                # Classification target
                target_dist = target_data.value_counts()
                quality_report['target_distribution'] = {
                    'type': 'categorical',
                    'classes': target_dist.index.tolist(),
                    'counts': target_dist.values.tolist(),
                    'class_balance': (target_dist / len(df)).round(3).to_dict()
                }
            else:
                # Regression target
                quality_report['target_distribution'] = {
                    'type': 'continuous',
                    'mean': float(target_data.mean()),
                    'std': float(target_data.std()),
                    'min': float(target_data.min()),
                    'max': float(target_data.max()),
                    'skewness': float(target_data.skew())
                }
        
        # Feature statistics
        numeric_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=['object', 'category']).columns
        
        quality_report['feature_statistics'] = {
            'numeric_features': len(numeric_features),
            'categorical_features': len(categorical_features),
            'datetime_features': len(df.select_dtypes(include=['datetime64']).columns),
            'total_features': len(df.columns) - 1  # Exclude target
        }
        
        # Correlation analysis for numeric features
        if len(numeric_features) > 1:
            corr_matrix = df[numeric_features].corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:  # High correlation threshold
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': round(corr_val, 3)
                        })
            
            quality_report['correlation_insights'] = {
                'high_correlation_pairs': high_corr_pairs,
                'max_correlation': float(corr_matrix.abs().unstack().sort_values(ascending=False).iloc[1])
            }
        
        # Basic outlier detection using IQR
        outlier_counts = {}
        for col in numeric_features:
            if col != self.target_column:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_counts[col] = outliers
        
        quality_report['outlier_detection'] = outlier_counts
        
        self._log_step("Data Quality Analysis", f"Analyzed {df.shape[0]} rows, {df.shape[1]} columns")
        return quality_report
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values with different strategies per column type.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        strategy (dict): Strategy mapping for different columns
        
        Returns:
        pd.DataFrame: DataFrame with handled missing values
        """
        
        df_processed = df.copy()
        
        # Default strategies
        default_strategy = {
            'numeric': 'median',
            'categorical': 'mode',
            'datetime': 'forward_fill'
        }
        
        strategy = strategy or {}
        
        # Handle numeric columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                col_strategy = strategy.get(col, default_strategy['numeric'])
                
                if col_strategy == 'mean':
                    fill_value = df_processed[col].mean()
                elif col_strategy == 'median':
                    fill_value = df_processed[col].median()
                elif col_strategy == 'mode':
                    fill_value = df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 0
                elif col_strategy == 'knn':
                    # Use KNN imputer for more sophisticated imputation
                    if col not in self.imputers:
                        self.imputers[col] = KNNImputer(n_neighbors=5)
                        df_processed[col] = self.imputers[col].fit_transform(df_processed[[col]]).flatten()
                    else:
                        df_processed[col] = self.imputers[col].transform(df_processed[[col]]).flatten()
                    continue
                elif col_strategy == 'drop':
                    df_processed = df_processed.dropna(subset=[col])
                    continue
                else:
                    fill_value = col_strategy  # Custom value
                
                df_processed[col].fillna(fill_value, inplace=True)
        
        # Handle categorical columns
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != self.target_column]
        
        for col in categorical_cols:
            if df_processed[col].isnull().any():
                col_strategy = strategy.get(col, default_strategy['categorical'])
                
                if col_strategy == 'mode':
                    fill_value = df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 'Unknown'
                elif col_strategy == 'unknown':
                    fill_value = 'Unknown'
                elif col_strategy == 'drop':
                    df_processed = df_processed.dropna(subset=[col])
                    continue
                else:
                    fill_value = col_strategy  # Custom value
                
                df_processed[col].fillna(fill_value, inplace=True)
        
        # Handle datetime columns
        datetime_cols = df_processed.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            if df_processed[col].isnull().any():
                col_strategy = strategy.get(col, default_strategy['datetime'])
                
                if col_strategy == 'forward_fill':
                    df_processed[col].fillna(method='ffill', inplace=True)
                elif col_strategy == 'backward_fill':
                    df_processed[col].fillna(method='bfill', inplace=True)
                elif col_strategy == 'drop':
                    df_processed = df_processed.dropna(subset=[col])
        
        missing_before = df.isnull().sum().sum()
        missing_after = df_processed.isnull().sum().sum()
        self._log_step("Missing Value Handling", 
                      f"Reduced missing values from {missing_before} to {missing_after}")
        
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame,
                                  encoding_strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Encode categorical features using various strategies.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        encoding_strategy (dict): Encoding strategy per column
        
        Returns:
        pd.DataFrame: DataFrame with encoded categorical features
        """
        
        df_processed = df.copy()
        
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != self.target_column]
        
        encoding_strategy = encoding_strategy or {}
        
        for col in categorical_cols:
            strategy = encoding_strategy.get(col, 'auto')
            
            # Auto-detect best encoding strategy
            if strategy == 'auto':
                unique_count = df_processed[col].nunique()
                if unique_count == 2:
                    strategy = 'label'
                elif unique_count <= 10:
                    strategy = 'onehot'
                else:
                    strategy = 'target' if self.target_column in df.columns else 'label'
            
            if strategy == 'label':
                # Label encoding for ordinal or binary categories
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_processed[col] = self.encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    df_processed[col] = self.encoders[col].transform(df_processed[col].astype(str))
            
            elif strategy == 'onehot':
                # One-hot encoding for nominal categories
                if col not in self.encoders:
                    self.encoders[col] = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded_features = self.encoders[col].fit_transform(df_processed[[col]])
                    feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                else:
                    encoded_features = self.encoders[col].transform(df_processed[[col]])
                    feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df_processed.index)
                
                # Replace original column with encoded features
                df_processed = df_processed.drop(columns=[col])
                df_processed = pd.concat([df_processed, encoded_df], axis=1)
            
            elif strategy == 'target':
                # Target encoding (mean encoding)
                if self.target_column in df.columns:
                    target_means = df_processed.groupby(col)[self.target_column].mean()
                    df_processed[f"{col}_target_encoded"] = df_processed[col].map(target_means)
                    df_processed = df_processed.drop(columns=[col])
            
            elif strategy == 'frequency':
                # Frequency encoding
                freq_map = df_processed[col].value_counts().to_dict()
                df_processed[f"{col}_frequency"] = df_processed[col].map(freq_map)
                df_processed = df_processed.drop(columns=[col])
        
        encoded_features = len(df_processed.columns) - len(df.columns)
        self._log_step("Categorical Encoding", f"Added {encoded_features} encoded features")
        
        return df_processed
    
    def engineer_features(self, df: pd.DataFrame,
                         feature_engineering_config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Engineer new features from existing ones.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        feature_engineering_config (dict): Configuration for feature engineering
        
        Returns:
        pd.DataFrame: DataFrame with engineered features
        """
        
        df_processed = df.copy()
        config = feature_engineering_config or {}
        
        # Polynomial features
        if config.get('polynomial_features', False):
            from sklearn.preprocessing import PolynomialFeatures
            
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != self.target_column]
            
            if len(numeric_cols) >= 2:
                poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
                poly_features = poly.fit_transform(df_processed[numeric_cols])
                poly_feature_names = poly.get_feature_names_out(numeric_cols)
                
                # Add only interaction terms (not squared terms)
                interaction_indices = [i for i, name in enumerate(poly_feature_names) 
                                     if ' ' in name and '^2' not in name]
                
                for idx in interaction_indices:
                    feature_name = poly_feature_names[idx].replace(' ', '_')
                    df_processed[f"interaction_{feature_name}"] = poly_features[:, idx]
        
        # Binning/Discretization
        if config.get('binning', {}):
            for col, bins in config['binning'].items():
                if col in df_processed.columns:
                    df_processed[f"{col}_binned"] = pd.cut(df_processed[col], bins=bins, labels=False)
        
        # Log transformations
        if config.get('log_transform', []):
            for col in config['log_transform']:
                if col in df_processed.columns and (df_processed[col] > 0).all():
                    df_processed[f"{col}_log"] = np.log1p(df_processed[col])
        
        # Square root transformations
        if config.get('sqrt_transform', []):
            for col in config['sqrt_transform']:
                if col in df_processed.columns and (df_processed[col] >= 0).all():
                    df_processed[f"{col}_sqrt"] = np.sqrt(df_processed[col])
        
        # Date/time feature extraction
        datetime_cols = df_processed.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            if config.get('datetime_features', True):
                df_processed[f"{col}_year"] = df_processed[col].dt.year
                df_processed[f"{col}_month"] = df_processed[col].dt.month
                df_processed[f"{col}_day"] = df_processed[col].dt.day
                df_processed[f"{col}_dayofweek"] = df_processed[col].dt.dayofweek
                df_processed[f"{col}_hour"] = df_processed[col].dt.hour
                df_processed[f"{col}_is_weekend"] = df_processed[col].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Aggregation features
        if config.get('aggregation_features', {}):
            for group_col, agg_configs in config['aggregation_features'].items():
                if group_col in df_processed.columns:
                    for agg_col, agg_funcs in agg_configs.items():
                        if agg_col in df_processed.columns:
                            for func in agg_funcs:
                                grouped = df_processed.groupby(group_col)[agg_col].transform(func)
                                df_processed[f"{group_col}_{agg_col}_{func}"] = grouped
        
        new_features = len(df_processed.columns) - len(df.columns)
        self._log_step("Feature Engineering", f"Created {new_features} new features")
        
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, 
                      scaling_method: str = 'standard',
                      exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        scaling_method (str): 'standard', 'minmax', 'robust', or 'none'
        exclude_columns (list): Columns to exclude from scaling
        
        Returns:
        pd.DataFrame: DataFrame with scaled features
        """
        
        if scaling_method == 'none':
            return df
        
        df_processed = df.copy()
        exclude_columns = exclude_columns or []
        exclude_columns.append(self.target_column)
        
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col not in exclude_columns]
        
        if not cols_to_scale:
            return df_processed
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        # Fit and transform
        if 'scaler' not in self.scalers:
            self.scalers['scaler'] = scaler
            df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
        else:
            df_processed[cols_to_scale] = self.scalers['scaler'].transform(df_processed[cols_to_scale])
        
        self._log_step("Feature Scaling", f"Applied {scaling_method} scaling to {len(cols_to_scale)} features")
        
        return df_processed
    
    def handle_outliers(self, df: pd.DataFrame,
                       outlier_method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in numerical features.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        outlier_method (str): 'iqr', 'zscore', 'isolation_forest', or 'none'
        threshold (float): Threshold for outlier detection
        
        Returns:
        pd.DataFrame: DataFrame with handled outliers
        """
        
        if outlier_method == 'none':
            return df
        
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        outlier_count = 0
        
        for col in numeric_cols:
            if outlier_method == 'iqr':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                
            elif outlier_method == 'zscore':
                z_scores = np.abs(stats.zscore(df_processed[col]))
                outlier_mask = z_scores > threshold
            
            # Cap outliers instead of removing them
            outlier_indices = df_processed[outlier_mask].index
            outlier_count += len(outlier_indices)
            
            if outlier_method == 'iqr':
                df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
            elif outlier_method == 'zscore':
                # Cap at mean ± threshold * std
                mean_val = df_processed[col].mean()
                std_val = df_processed[col].std()
                df_processed.loc[df_processed[col] > mean_val + threshold * std_val, col] = mean_val + threshold * std_val
                df_processed.loc[df_processed[col] < mean_val - threshold * std_val, col] = mean_val - threshold * std_val
        
        self._log_step("Outlier Handling", f"Capped {outlier_count} outliers using {outlier_method} method")
        
        return df_processed
    
    def select_features(self, df: pd.DataFrame,
                       selection_method: str = 'correlation',
                       k_features: int = None) -> pd.DataFrame:
        """
        Select most relevant features for the model.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        selection_method (str): 'correlation', 'mutual_info', 'f_test', or 'none'
        k_features (int): Number of features to select
        
        Returns:
        pd.DataFrame: DataFrame with selected features
        """
        
        if selection_method == 'none':
            return df
        
        df_processed = df.copy()
        
        if self.target_column not in df_processed.columns:
            self._log_step("Feature Selection", "Target column not found, skipping feature selection")
            return df_processed
        
        feature_cols = [col for col in df_processed.columns if col != self.target_column]
        
        if not feature_cols:
            return df_processed
        
        X = df_processed[feature_cols]
        y = df_processed[self.target_column]
        
        # Handle non-numeric features for feature selection
        X_numeric = X.select_dtypes(include=[np.number])
        
        if len(X_numeric.columns) == 0:
            self._log_step("Feature Selection", "No numeric features found for selection")
            return df_processed
        
        k_features = k_features or min(10, len(X_numeric.columns))
        
        if selection_method == 'correlation':
            # Select features based on correlation with target
            correlations = X_numeric.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(k_features).index.tolist()
            
        elif selection_method == 'mutual_info':
            # Mutual information feature selection
            if 'mutual_info_selector' not in self.feature_selectors:
                self.feature_selectors['mutual_info_selector'] = SelectKBest(
                    score_func=mutual_info_classif if y.dtype == 'object' else mutual_info_classif,
                    k=k_features
                )
                self.feature_selectors['mutual_info_selector'].fit(X_numeric, y)
            
            selected_mask = self.feature_selectors['mutual_info_selector'].get_support()
            selected_features = X_numeric.columns[selected_mask].tolist()
            
        elif selection_method == 'f_test':
            # F-test feature selection
            if 'f_test_selector' not in self.feature_selectors:
                self.feature_selectors['f_test_selector'] = SelectKBest(
                    score_func=f_classif,
                    k=k_features
                )
                self.feature_selectors['f_test_selector'].fit(X_numeric, y)
            
            selected_mask = self.feature_selectors['f_test_selector'].get_support()
            selected_features = X_numeric.columns[selected_mask].tolist()
        
        # Keep selected features plus target and non-numeric features
        non_numeric_features = [col for col in feature_cols if col not in X_numeric.columns]
        final_features = selected_features + non_numeric_features + [self.target_column]
        
        df_processed = df_processed[final_features]
        
        self._log_step("Feature Selection", 
                      f"Selected {len(selected_features)} features using {selection_method}")
        
        return df_processed
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                  pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Parameters:
        df (pd.DataFrame): Preprocessed DataFrame
        
        Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y if y.dtype == 'object' else None
        )
        
        if self.validation_split:
            # Second split: train vs val
            val_size = self.test_size / (1 - self.test_size)  # Adjust validation size
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=self.random_state,
                stratify=y_temp if y_temp.dtype == 'object' else None
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, pd.DataFrame(), y_temp, pd.Series()
        
        self._log_step("Data Splitting", 
                      f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of all preprocessing steps performed.
        
        Returns:
        dict: Preprocessing summary
        """
        
        return {
            'steps_performed': [step['step'] for step in self.preprocessing_log],
            'total_steps': len(self.preprocessing_log),
            'preprocessing_log': self.preprocessing_log,
            'components_fitted': {
                'scalers': list(self.scalers.keys()),
                'encoders': list(self.encoders.keys()),
                'imputers': list(self.imputers.keys()),
                'feature_selectors': list(self.feature_selectors.keys())
            }
        }
    
    def preprocess_pipeline(self, df: pd.DataFrame,
                          missing_strategy: Dict[str, str] = None,
                          encoding_strategy: Dict[str, str] = None,
                          feature_engineering_config: Dict[str, Any] = None,
                          scaling_method: str = 'standard',
                          outlier_method: str = 'iqr',
                          selection_method: str = 'correlation',
                          k_features: int = None) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Parameters:
        df (pd.DataFrame): Raw input DataFrame
        missing_strategy (dict): Missing value handling strategy
        encoding_strategy (dict): Categorical encoding strategy
        feature_engineering_config (dict): Feature engineering configuration
        scaling_method (str): Feature scaling method
        outlier_method (str): Outlier handling method
        selection_method (str): Feature selection method
        k_features (int): Number of features to select
        
        Returns:
        dict: Complete preprocessing results
        """
        
        # Step 1: Data quality analysis
        quality_report = self.analyze_data_quality(df)
        
        # Step 2: Handle missing values
        df_processed = self.handle_missing_values(df, missing_strategy)
        
        # Step 3: Encode categorical features
        df_processed = self.encode_categorical_features(df_processed, encoding_strategy)
        
        # Step 4: Engineer features
        df_processed = self.engineer_features(df_processed, feature_engineering_config)
        
        # Step 5: Handle outliers
        df_processed = self.handle_outliers(df_processed, outlier_method)
        
        # Step 6: Scale features
        df_processed = self.scale_features(df_processed, scaling_method)
        
        # Step 7: Select features
        df_processed = self.select_features(df_processed, selection_method, k_features)
        
        # Step 8: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df_processed)
        
        # Compile results
        results = {
            'quality_report': quality_report,
            'preprocessed_data': df_processed,
            'train_data': (X_train, y_train),
            'validation_data': (X_val, y_val) if not X_val.empty else None,
            'test_data': (X_test, y_test),
            'preprocessing_summary': self.get_preprocessing_summary(),
            'feature_names': df_processed.drop(columns=[self.target_column]).columns.tolist()
        }
        
        return results
    
    def _log_step(self, step_name: str, message: str):
        """Log preprocessing step."""
        self.preprocessing_log.append({
            'step': step_name,
            'message': message,
            'timestamp': pd.Timestamp.now()
        })

# Example usage and demonstration
def create_sample_ml_dataset():
    """Create a realistic sample dataset for ML preprocessing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = {
        'numeric_feature_1': np.random.normal(50, 15, n_samples),
        'numeric_feature_2': np.random.exponential(2, n_samples),
        'numeric_feature_3': np.random.uniform(0, 100, n_samples),
        'categorical_feature_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'categorical_feature_2': np.random.choice(['High', 'Medium', 'Low'], n_samples, p=[0.2, 0.5, 0.3]),
        'binary_feature': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'date_feature': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'high_cardinality_cat': [f'Category_{i}' for i in np.random.randint(0, 50, n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on features
    target_score = (
        0.3 * df['numeric_feature_1'] +
        0.2 * df['numeric_feature_2'] +
        0.1 * df['numeric_feature_3'] +
        0.2 * df['binary_feature'] * 50 +
        np.random.normal(0, 10, n_samples)
    )
    
    # Create binary classification target
    df['target'] = (target_score > target_score.median()).astype(int)
    
    # Introduce missing values
    missing_indices = np.random.choice(df.index, size=int(0.1 * n_samples), replace=False)
    df.loc[missing_indices[:50], 'numeric_feature_1'] = np.nan
    df.loc[missing_indices[50:100], 'categorical_feature_1'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[outlier_indices, 'numeric_feature_2'] *= 10
    
    return df

# Demonstration
if __name__ == "__main__":
    # Create sample dataset
    sample_data = create_sample_ml_dataset()
    
    print("Sample ML Dataset:")
    print(f"Shape: {sample_data.shape}")
    print(sample_data.head())
    print(f"\nData types:\n{sample_data.dtypes}")
    print(f"\nTarget distribution:\n{sample_data['target'].value_counts()}")
    
    # Initialize preprocessor
    preprocessor = MLDataPreprocessor(
        target_column='target',
        test_size=0.2,
        validation_split=True
    )
    
    # Configure preprocessing
    missing_strategy = {
        'numeric_feature_1': 'median',
        'categorical_feature_1': 'mode'
    }
    
    encoding_strategy = {
        'categorical_feature_1': 'onehot',
        'categorical_feature_2': 'label',
        'high_cardinality_cat': 'target'
    }
    
    feature_engineering_config = {
        'polynomial_features': True,
        'log_transform': ['numeric_feature_2'],
        'datetime_features': True,
        'binning': {'numeric_feature_3': 5}
    }
    
    # Run preprocessing pipeline
    results = preprocessor.preprocess_pipeline(
        df=sample_data,
        missing_strategy=missing_strategy,
        encoding_strategy=encoding_strategy,
        feature_engineering_config=feature_engineering_config,
        scaling_method='standard',
        outlier_method='iqr',
        selection_method='correlation',
        k_features=15
    )
    
    # Display results
    print("\n" + "="*60)
    print("PREPROCESSING RESULTS")
    print("="*60)
    
    print(f"\nOriginal shape: {sample_data.shape}")
    print(f"Processed shape: {results['preprocessed_data'].shape}")
    print(f"Selected features: {len(results['feature_names'])}")
    
    print("\nData splits:")
    X_train, y_train = results['train_data']
    X_test, y_test = results['test_data']
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    if results['validation_data']:
        X_val, y_val = results['validation_data']
        print(f"  Validation: {len(X_val)} samples")
    
    print("\nQuality Report Summary:")
    quality = results['quality_report']
    print(f"  Missing values handled: {quality['missing_values']['total_missing']}")
    print(f"  Duplicates: {quality['duplicates']}")
    print(f"  Numeric features: {quality['feature_statistics']['numeric_features']}")
    print(f"  Categorical features: {quality['feature_statistics']['categorical_features']}")
    
    print("\nPreprocessing Steps:")
    for step in results['preprocessing_summary']['steps_performed']:
        print(f"  ✅ {step}")
    
    print("\nFinal Feature Names:")
    for i, feature in enumerate(results['feature_names'][:10]):  # Show first 10
        print(f"  {i+1}. {feature}")
    if len(results['feature_names']) > 10:
        print(f"  ... and {len(results['feature_names']) - 10} more features")
    
    print("\nPreprocessed Data Sample:")
    print(results['preprocessed_data'].head())
```

#### Explanation
1. **Comprehensive Pipeline**: End-to-end preprocessing from raw data to ML-ready format
2. **Modular Design**: Each preprocessing step is isolated and configurable
3. **Quality Analysis**: Detailed data quality assessment before preprocessing
4. **Flexible Strategies**: Multiple approaches for each preprocessing task
5. **Component Persistence**: Fitted transformers saved for consistent test data processing
6. **Automated Decisions**: Intelligent defaults with manual override options

#### Use Cases
- **Supervised Learning**: Classification and regression model preparation
- **Feature Engineering**: Creating informative features from raw data
- **Data Quality**: Cleaning and validating datasets
- **Production Pipelines**: Consistent preprocessing across environments
- **Exploratory Analysis**: Understanding data characteristics and relationships

#### Best Practices
- **Data Leakage Prevention**: Fit transformers only on training data
- **Reproducibility**: Use random seeds and version control for preprocessing code
- **Validation**: Always validate preprocessing steps with domain experts
- **Documentation**: Maintain clear documentation of preprocessing decisions
- **Testing**: Test preprocessing pipeline with edge cases and validation data

#### Pitfalls
- **Data Leakage**: Using information from test set during preprocessing
- **Over-preprocessing**: Removing too much information through aggressive cleaning
- **Feature Selection Bias**: Selecting features that don't generalize
- **Scaling Issues**: Incorrect scaling can hurt distance-based algorithms
- **Target Encoding**: Can cause overfitting if not properly validated

#### Debugging
```python
def debug_preprocessing_pipeline(preprocessor, df):
    """Debug preprocessing pipeline issues."""
    
    print("Debugging Preprocessing Pipeline")
    print("="*40)
    
    # Check data types
    print("Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # Check missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\nMissing Values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"Infinite values in {col}: {inf_count}")
    
    # Check preprocessing log
    print(f"\nPreprocessing Log:")
    for step in preprocessor.preprocessing_log:
        print(f"  {step['step']}: {step['message']}")

def validate_preprocessing_quality(original_df, processed_df, target_column):
    """Validate preprocessing quality."""
    
    print("Preprocessing Quality Validation")
    print("="*35)
    
    # Check data preservation
    original_samples = len(original_df)
    processed_samples = len(processed_df)
    print(f"Sample preservation: {processed_samples}/{original_samples} ({processed_samples/original_samples*100:.1f}%)")
    
    # Check feature count
    original_features = len([col for col in original_df.columns if col != target_column])
    processed_features = len([col for col in processed_df.columns if col != target_column])
    print(f"Feature count: {original_features} → {processed_features}")
    
    # Check missing values
    original_missing = original_df.isnull().sum().sum()
    processed_missing = processed_df.isnull().sum().sum()
    print(f"Missing values: {original_missing} → {processed_missing}")
    
    # Check target distribution preservation
    if target_column in both DataFrames:
        original_dist = original_df[target_column].value_counts(normalize=True)
        processed_dist = processed_df[target_column].value_counts(normalize=True)
        print(f"Target distribution preserved: {np.allclose(original_dist.values, processed_dist.values, rtol=0.1)}")
```

#### Optimization
```python
# Memory-efficient preprocessing for large datasets
class MemoryEfficientPreprocessor(MLDataPreprocessor):
    """Memory-optimized version for large datasets."""
    
    def preprocess_large_dataset(self, file_path: str, chunksize: int = 10000):
        """Process large datasets in chunks."""
        
        # First pass: analyze data and fit transformers
        sample_chunk = pd.read_csv(file_path, nrows=1000)
        self.analyze_data_quality(sample_chunk)
        
        # Process in chunks
        processed_chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            processed_chunk = self.handle_missing_values(chunk)
            processed_chunk = self.encode_categorical_features(processed_chunk)
            processed_chunks.append(processed_chunk)
        
        return pd.concat(processed_chunks, ignore_index=True)

# Automated preprocessing with AutoML principles
def auto_preprocess(df: pd.DataFrame, target_column: str):
    """Automated preprocessing with intelligent defaults."""
    
    preprocessor = MLDataPreprocessor(target_column)
    
    # Analyze data to determine best strategies
    quality_report = preprocessor.analyze_data_quality(df)
    
    # Auto-determine strategies based on data characteristics
    missing_strategy = {}
    encoding_strategy = {}
    
    for col in df.columns:
        if col != target_column:
            missing_pct = quality_report['missing_values']['percentages'][col]
            
            if missing_pct > 50:
                missing_strategy[col] = 'drop'
            elif df[col].dtype in ['object', 'category']:
                missing_strategy[col] = 'mode'
                unique_ratio = df[col].nunique() / len(df)
                encoding_strategy[col] = 'onehot' if unique_ratio < 0.1 else 'label'
            else:
                missing_strategy[col] = 'median'
    
    return preprocessor.preprocess_pipeline(
        df=df,
        missing_strategy=missing_strategy,
        encoding_strategy=encoding_strategy
    )
```

**Key Preprocessing Strategies:**

| Task | Strategy Options | Best For |
|------|-----------------|----------|
| **Missing Values** | Mean/Median/Mode, KNN, Drop | Depends on % missing |
| **Categorical Encoding** | Label, One-Hot, Target, Frequency | Cardinality & type |
| **Feature Scaling** | Standard, MinMax, Robust | Algorithm requirements |
| **Outlier Handling** | IQR, Z-score, Isolation Forest | Data distribution |
| **Feature Selection** | Correlation, Mutual Info, F-test | Dataset size & type |

This comprehensive preprocessing framework ensures robust, reproducible ML pipelines with proper data handling and feature engineering.

---

## Question 16

**What are some strategies for optimizing Pandas code performance?**

### Answer

#### Theory
Pandas performance optimization involves understanding the underlying data structures, memory management, and computational patterns. Effective optimization combines algorithmic improvements, memory efficiency techniques, vectorization strategies, and appropriate data types. Performance gains can range from 2x to 100x depending on the optimization techniques applied and the nature of the operations.

#### Code Example

```python
import pandas as pd
import numpy as np
import time
import psutil
import gc
from typing import List, Dict, Any, Callable
import warnings
warnings.filterwarnings('ignore')

class PandasPerformanceOptimizer:
    """
    Comprehensive Pandas performance optimization toolkit.
    """
    
    def __init__(self):
        self.performance_metrics = []
        
    def benchmark_operation(self, operation: Callable, data: Any, 
                          operation_name: str, iterations: int = 1) -> Dict[str, float]:
        """
        Benchmark a Pandas operation and collect performance metrics.
        
        Parameters:
        operation (callable): Function to benchmark
        data: Input data for the operation
        operation_name (str): Name for the operation
        iterations (int): Number of iterations to run
        
        Returns:
        dict: Performance metrics
        """
        
        # Measure initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Warm up
        _ = operation(data)
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(iterations):
            result = operation(data)
        
        end_time = time.time()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        metrics = {
            'operation': operation_name,
            'execution_time': (end_time - start_time) / iterations,
            'memory_used': final_memory - initial_memory,
            'iterations': iterations
        }
        
        self.performance_metrics.append(metrics)
        return metrics

# 1. Data Type Optimization
def optimize_data_types_comprehensive():
    """Demonstrate comprehensive data type optimization techniques."""
    
    print("=== DATA TYPE OPTIMIZATION ===")
    
    # Create sample data with suboptimal types
    np.random.seed(42)
    n_rows = 1_000_000
    
    # Inefficient data types
    data_inefficient = pd.DataFrame({
        'small_int': np.random.randint(0, 100, n_rows),  # Could be uint8
        'medium_int': np.random.randint(0, 10000, n_rows),  # Could be uint16
        'float_data': np.random.randn(n_rows),  # float64 by default
        'category_data': np.random.choice(['A', 'B', 'C', 'D'], n_rows),  # object
        'boolean_like': np.random.choice([0, 1], n_rows),  # int64
        'date_string': pd.date_range('2020-01-01', periods=n_rows, freq='H').astype(str)  # object
    })
    
    print(f"Original memory usage: {data_inefficient.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("Original data types:")
    print(data_inefficient.dtypes)
    
    # Optimized data types
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            if col_type == 'object':
                # Check if it's actually a date
                try:
                    df_optimized[col] = pd.to_datetime(df_optimized[col])
                    continue
                except:
                    pass
                
                # Check if it should be categorical
                unique_ratio = len(df_optimized[col].unique()) / len(df_optimized[col])
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df_optimized[col] = df_optimized[col].astype('category')
            
            elif col_type in ['int64', 'int32']:
                # Downcast integers
                c_min, c_max = df_optimized[col].min(), df_optimized[col].max()
                
                if c_min >= 0:  # Unsigned integers
                    if c_max < 255:
                        df_optimized[col] = df_optimized[col].astype('uint8')
                    elif c_max < 65535:
                        df_optimized[col] = df_optimized[col].astype('uint16')
                    elif c_max < 4294967295:
                        df_optimized[col] = df_optimized[col].astype('uint32')
                else:  # Signed integers
                    if c_min > -128 and c_max < 127:
                        df_optimized[col] = df_optimized[col].astype('int8')
                    elif c_min > -32768 and c_max < 32767:
                        df_optimized[col] = df_optimized[col].astype('int16')
                    elif c_min > -2147483648 and c_max < 2147483647:
                        df_optimized[col] = df_optimized[col].astype('int32')
            
            elif col_type == 'float64':
                # Try to downcast to float32
                if df_optimized[col].between(-3.4e38, 3.4e38).all():
                    df_optimized[col] = df_optimized[col].astype('float32')
        
        return df_optimized
    
    # Apply optimization
    data_optimized = optimize_dtypes(data_inefficient)
    
    print(f"\nOptimized memory usage: {data_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("Optimized data types:")
    print(data_optimized.dtypes)
    
    memory_reduction = (1 - data_optimized.memory_usage(deep=True).sum() / 
                       data_inefficient.memory_usage(deep=True).sum()) * 100
    print(f"\nMemory reduction: {memory_reduction:.1f}%")
    
    return data_inefficient, data_optimized

# 2. Vectorization vs. Iteration
def demonstrate_vectorization():
    """Compare vectorized operations vs. loops and apply functions."""
    
    print("\n=== VECTORIZATION OPTIMIZATION ===")
    
    optimizer = PandasPerformanceOptimizer()
    
    # Create test data
    df = pd.DataFrame({
        'a': np.random.randn(100_000),
        'b': np.random.randn(100_000),
        'c': np.random.choice(['X', 'Y', 'Z'], 100_000)
    })
    
    # Method 1: Inefficient - iterrows
    def method_iterrows(df):
        result = []
        for idx, row in df.iterrows():
            if row['c'] == 'X':
                result.append(row['a'] + row['b'])
            else:
                result.append(row['a'] * row['b'])
        return pd.Series(result)
    
    # Method 2: Apply function
    def method_apply(df):
        return df.apply(lambda row: row['a'] + row['b'] if row['c'] == 'X' else row['a'] * row['b'], axis=1)
    
    # Method 3: Vectorized with numpy.where
    def method_vectorized_where(df):
        return np.where(df['c'] == 'X', df['a'] + df['b'], df['a'] * df['b'])
    
    # Method 4: Vectorized with boolean indexing
    def method_vectorized_boolean(df):
        result = pd.Series(index=df.index, dtype=float)
        mask_x = df['c'] == 'X'
        result[mask_x] = df.loc[mask_x, 'a'] + df.loc[mask_x, 'b']
        result[~mask_x] = df.loc[~mask_x, 'a'] * df.loc[~mask_x, 'b']
        return result
    
    # Method 5: Using pandas.where
    def method_pandas_where(df):
        return df['a'].where(df['c'] == 'X', df['a'] * df['b']) + df['b'].where(df['c'] == 'X', 0)
    
    # Benchmark methods (skip iterrows for large datasets)
    methods = [
        (method_apply, "Apply Function"),
        (method_vectorized_where, "NumPy Where"),
        (method_vectorized_boolean, "Boolean Indexing"),
        (method_pandas_where, "Pandas Where")
    ]
    
    print("Performance comparison (100k rows):")
    results = {}
    
    for method, name in methods:
        metrics = optimizer.benchmark_operation(method, df, name)
        results[name] = metrics['execution_time']
        print(f"{name}: {metrics['execution_time']:.4f} seconds")
    
    # Show speedup
    baseline = max(results.values())
    print(f"\nSpeedup factors (vs slowest):")
    for name, time_taken in sorted(results.items(), key=lambda x: x[1]):
        speedup = baseline / time_taken
        print(f"{name}: {speedup:.1f}x faster")

# 3. Memory Optimization Techniques
def demonstrate_memory_optimization():
    """Demonstrate various memory optimization techniques."""
    
    print("\n=== MEMORY OPTIMIZATION ===")
    
    # Chunked processing
    def process_large_file_chunked(file_path: str, chunk_size: int = 10000):
        """Process large files in chunks to manage memory."""
        
        results = []
        total_rows = 0
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Process chunk
            processed_chunk = chunk.groupby('category').agg({'value': 'mean'})
            results.append(processed_chunk)
            total_rows += len(chunk)
            
            # Clear memory
            del chunk
            gc.collect()
        
        # Combine results
        final_result = pd.concat(results).groupby(level=0).mean()
        return final_result, total_rows
    
    # Memory-efficient aggregations
    def memory_efficient_groupby():
        """Demonstrate memory-efficient groupby operations."""
        
        # Create large dataset
        n_rows = 1_000_000
        df = pd.DataFrame({
            'group': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
            'value1': np.random.randn(n_rows),
            'value2': np.random.randn(n_rows),
            'value3': np.random.randn(n_rows)
        })
        
        print(f"Dataset size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Inefficient: Multiple separate groupby operations
        def inefficient_aggregation(df):
            result1 = df.groupby('group')['value1'].mean()
            result2 = df.groupby('group')['value2'].sum()
            result3 = df.groupby('group')['value3'].std()
            return pd.concat([result1, result2, result3], axis=1)
        
        # Efficient: Single groupby with multiple aggregations
        def efficient_aggregation(df):
            return df.groupby('group').agg({
                'value1': 'mean',
                'value2': 'sum',
                'value3': 'std'
            })
        
        optimizer = PandasPerformanceOptimizer()
        
        # Benchmark both approaches
        inefficient_time = optimizer.benchmark_operation(
            inefficient_aggregation, df, "Inefficient Aggregation"
        )['execution_time']
        
        efficient_time = optimizer.benchmark_operation(
            efficient_aggregation, df, "Efficient Aggregation"
        )['execution_time']
        
        print(f"Inefficient aggregation: {inefficient_time:.4f} seconds")
        print(f"Efficient aggregation: {efficient_time:.4f} seconds")
        print(f"Speedup: {inefficient_time/efficient_time:.1f}x")
        
        return df
    
    # Index optimization
    def demonstrate_index_optimization():
        """Show the importance of proper indexing."""
        
        # Create dataset
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100_000, freq='H'),
            'category': np.random.choice(['A', 'B', 'C'], 100_000),
            'value': np.random.randn(100_000)
        })
        
        # Query without index
        def query_without_index(df):
            return df[df['category'] == 'A']['value'].mean()
        
        # Query with index
        df_indexed = df.set_index('category')
        def query_with_index(df_indexed):
            return df_indexed.loc['A', 'value'].mean()
        
        optimizer = PandasPerformanceOptimizer()
        
        no_index_time = optimizer.benchmark_operation(
            query_without_index, df, "Query Without Index", iterations=100
        )['execution_time']
        
        with_index_time = optimizer.benchmark_operation(
            query_with_index, df_indexed, "Query With Index", iterations=100
        )['execution_time']
        
        print(f"\nIndex optimization results:")
        print(f"Without index: {no_index_time:.6f} seconds")
        print(f"With index: {with_index_time:.6f} seconds")
        print(f"Speedup: {no_index_time/with_index_time:.1f}x")
    
    # Run demonstrations
    memory_efficient_groupby()
    demonstrate_index_optimization()

# 4. String Operations Optimization
def optimize_string_operations():
    """Demonstrate string operation optimizations."""
    
    print("\n=== STRING OPERATIONS OPTIMIZATION ===")
    
    # Create string data
    n_rows = 100_000
    string_data = pd.Series([f"text_data_{i}_example" for i in range(n_rows)])
    
    optimizer = PandasPerformanceOptimizer()
    
    # Method 1: Python string methods
    def python_string_method(series):
        return series.apply(lambda x: x.upper().replace('_', '-'))
    
    # Method 2: Vectorized string methods
    def vectorized_string_method(series):
        return series.str.upper().str.replace('_', '-', regex=False)
    
    # Method 3: Optimized with categorical
    string_categorical = string_data.astype('category')
    def categorical_string_method(series):
        return series.str.upper().str.replace('_', '-', regex=False)
    
    # Benchmark methods
    python_time = optimizer.benchmark_operation(
        python_string_method, string_data, "Python String Methods"
    )['execution_time']
    
    vectorized_time = optimizer.benchmark_operation(
        vectorized_string_method, string_data, "Vectorized String Methods"
    )['execution_time']
    
    categorical_time = optimizer.benchmark_operation(
        categorical_string_method, string_categorical, "Categorical String Methods"
    )['execution_time']
    
    print(f"Python string methods: {python_time:.4f} seconds")
    print(f"Vectorized string methods: {vectorized_time:.4f} seconds")
    print(f"Categorical string methods: {categorical_time:.4f} seconds")
    
    print(f"\nString optimization speedups:")
    print(f"Vectorized vs Python: {python_time/vectorized_time:.1f}x")
    print(f"Categorical vs Python: {python_time/categorical_time:.1f}x")

# 5. Join and Merge Optimization
def optimize_joins_and_merges():
    """Demonstrate join and merge optimization techniques."""
    
    print("\n=== JOIN AND MERGE OPTIMIZATION ===")
    
    # Create test datasets
    n_left = 50_000
    n_right = 30_000
    
    df_left = pd.DataFrame({
        'key': np.random.randint(0, 10_000, n_left),
        'value_left': np.random.randn(n_left)
    })
    
    df_right = pd.DataFrame({
        'key': np.random.randint(0, 10_000, n_right),
        'value_right': np.random.randn(n_right)
    })
    
    optimizer = PandasPerformanceOptimizer()
    
    # Method 1: Basic merge
    def basic_merge(left, right):
        return pd.merge(left, right, on='key', how='inner')
    
    # Method 2: Merge with sorted data
    df_left_sorted = df_left.sort_values('key')
    df_right_sorted = df_right.sort_values('key')
    
    def sorted_merge(left, right):
        return pd.merge(left, right, on='key', how='inner')
    
    # Method 3: Merge with indexed data
    df_left_indexed = df_left.set_index('key')
    df_right_indexed = df_right.set_index('key')
    
    def indexed_join(left, right):
        return left.join(right, how='inner')
    
    # Benchmark approaches
    basic_time = optimizer.benchmark_operation(
        basic_merge, (df_left, df_right), "Basic Merge"
    )['execution_time']
    
    sorted_time = optimizer.benchmark_operation(
        sorted_merge, (df_left_sorted, df_right_sorted), "Sorted Merge"
    )['execution_time']
    
    indexed_time = optimizer.benchmark_operation(
        indexed_join, (df_left_indexed, df_right_indexed), "Indexed Join"
    )['execution_time']
    
    print(f"Basic merge: {basic_time:.4f} seconds")
    print(f"Sorted merge: {sorted_time:.4f} seconds")
    print(f"Indexed join: {indexed_time:.4f} seconds")
    
    print(f"\nJoin optimization speedups:")
    print(f"Sorted vs Basic: {basic_time/sorted_time:.1f}x")
    print(f"Indexed vs Basic: {basic_time/indexed_time:.1f}x")

# 6. Comprehensive Performance Testing
def comprehensive_performance_analysis():
    """Run comprehensive performance analysis."""
    
    print("\n=== COMPREHENSIVE PERFORMANCE ANALYSIS ===")
    
    # Performance testing framework
    class PerformanceTester:
        def __init__(self):
            self.results = []
        
        def test_operation(self, operation_func, data, name, iterations=3):
            """Test an operation multiple times and record statistics."""
            
            times = []
            for _ in range(iterations):
                start_time = time.time()
                result = operation_func(data)
                end_time = time.time()
                times.append(end_time - start_time)
            
            self.results.append({
                'operation': name,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            })
        
        def get_summary(self):
            """Get performance summary."""
            return pd.DataFrame(self.results).sort_values('mean_time')
    
    # Create test data
    df = pd.DataFrame({
        'group': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100_000),
        'value1': np.random.randn(100_000),
        'value2': np.random.randn(100_000),
        'category': np.random.choice(['X', 'Y', 'Z'], 100_000)
    })
    
    tester = PerformanceTester()
    
    # Test various operations
    operations = [
        (lambda df: df.groupby('group')['value1'].sum(), "GroupBy Sum"),
        (lambda df: df.sort_values('value1'), "Sort Values"),
        (lambda df: df.drop_duplicates(), "Drop Duplicates"),
        (lambda df: df.fillna(0), "Fill NA"),
        (lambda df: df.query('value1 > 0'), "Query Filter"),
        (lambda df: df[df['value1'] > 0], "Boolean Filter"),
        (lambda df: df.pivot_table(index='group', columns='category', values='value1'), "Pivot Table"),
        (lambda df: pd.get_dummies(df['category']), "Get Dummies")
    ]
    
    for operation, name in operations:
        tester.test_operation(operation, df, name)
    
    summary = tester.get_summary()
    print("Performance Summary (100k rows):")
    print(summary.to_string(index=False, float_format='%.4f'))

# 7. Advanced Optimization Techniques
def advanced_optimization_techniques():
    """Demonstrate advanced optimization techniques."""
    
    print("\n=== ADVANCED OPTIMIZATION TECHNIQUES ===")
    
    # Technique 1: Use eval() for complex expressions
    df = pd.DataFrame({
        'a': np.random.randn(100_000),
        'b': np.random.randn(100_000),
        'c': np.random.randn(100_000),
        'd': np.random.randn(100_000)
    })
    
    optimizer = PandasPerformanceOptimizer()
    
    # Standard calculation
    def standard_calculation(df):
        return df['a'] + df['b'] * df['c'] - df['d'] / 2
    
    # Using eval()
    def eval_calculation(df):
        return df.eval('a + b * c - d / 2')
    
    standard_time = optimizer.benchmark_operation(
        standard_calculation, df, "Standard Calculation"
    )['execution_time']
    
    eval_time = optimizer.benchmark_operation(
        eval_calculation, df, "Eval Calculation"
    )['execution_time']
    
    print(f"Standard calculation: {standard_time:.4f} seconds")
    print(f"Eval calculation: {eval_time:.4f} seconds")
    print(f"Eval speedup: {standard_time/eval_time:.1f}x")
    
    # Technique 2: Numba acceleration
    try:
        from numba import jit
        
        @jit(nopython=True)
        def numba_calculation(a, b, c, d):
            result = np.empty_like(a)
            for i in range(len(a)):
                result[i] = a[i] + b[i] * c[i] - d[i] / 2
            return result
        
        def numba_pandas_calculation(df):
            return numba_calculation(df['a'].values, df['b'].values, 
                                   df['c'].values, df['d'].values)
        
        # Warm up numba
        _ = numba_pandas_calculation(df.head(100))
        
        numba_time = optimizer.benchmark_operation(
            numba_pandas_calculation, df, "Numba Calculation"
        )['execution_time']
        
        print(f"Numba calculation: {numba_time:.4f} seconds")
        print(f"Numba speedup: {standard_time/numba_time:.1f}x")
        
    except ImportError:
        print("Numba not available - install with 'pip install numba'")

# Performance optimization best practices summary
def print_optimization_best_practices():
    """Print comprehensive optimization best practices."""
    
    print("\n" + "="*60)
    print("PANDAS PERFORMANCE OPTIMIZATION BEST PRACTICES")
    print("="*60)
    
    practices = {
        "Data Types": [
            "• Use appropriate numeric types (int8, int16, int32 vs int64)",
            "• Convert string columns to categorical when possible",
            "• Use datetime types for date/time data",
            "• Consider nullable integer types (Int64) for missing data",
            "• Downcast float64 to float32 when precision allows"
        ],
        "Vectorization": [
            "• Avoid iterrows() and apply() when vectorized alternatives exist",
            "• Use numpy.where() for conditional operations",
            "• Leverage broadcasting for element-wise operations",
            "• Use built-in pandas methods over custom functions",
            "• Prefer pandas.eval() for complex expressions"
        ],
        "Memory Management": [
            "• Process large files in chunks",
            "• Use del and gc.collect() to free memory",
            "• Avoid creating unnecessary copies",
            "• Use memory_usage(deep=True) to monitor usage",
            "• Consider using sparse data structures for sparse data"
        ],
        "Indexing & Querying": [
            "• Set appropriate indexes for frequent lookups",
            "• Use .loc[] and .iloc[] instead of chained indexing",
            "• Sort data before range queries",
            "• Use query() method for complex filtering",
            "• Leverage MultiIndex for hierarchical data"
        ],
        "Aggregations": [
            "• Combine multiple aggregations in single groupby",
            "• Use transform() instead of apply() when possible",
            "• Pre-sort data for groupby operations",
            "• Use agg() with dictionary for multiple functions",
            "• Consider using pivot_table() for cross-tabulations"
        ],
        "String Operations": [
            "• Use vectorized string methods (.str accessor)",
            "• Convert to categorical for repeated string operations",
            "• Avoid regex when simple string operations suffice",
            "• Use string methods that return boolean for filtering",
            "• Consider using cat() for string concatenation"
        ],
        "I/O Operations": [
            "• Use efficient file formats (Parquet, HDF5, Feather)",
            "• Specify data types when reading CSV files",
            "• Use compression for storage (gzip, snappy)",
            "• Read only required columns",
            "• Use chunking for large files"
        ],
        "Advanced Techniques": [
            "• Use Numba for numerical computations",
            "• Leverage Dask for out-of-core processing",
            "• Consider CuDF for GPU acceleration",
            "• Use sparse arrays for sparse data",
            "• Profile code with cProfile or line_profiler"
        ]
    }
    
    for category, tips in practices.items():
        print(f"\n{category}:")
        for tip in tips:
            print(f"  {tip}")

# Main demonstration function
def run_performance_optimization_demo():
    """Run complete performance optimization demonstration."""
    
    print("PANDAS PERFORMANCE OPTIMIZATION COMPREHENSIVE DEMO")
    print("="*55)
    
    # Run all demonstrations
    optimize_data_types_comprehensive()
    demonstrate_vectorization()
    demonstrate_memory_optimization()
    optimize_string_operations()
    optimize_joins_and_merges()
    comprehensive_performance_analysis()
    advanced_optimization_techniques()
    print_optimization_best_practices()

# Execute demonstration
if __name__ == "__main__":
    run_performance_optimization_demo()
```

#### Explanation
1. **Systematic Benchmarking**: Framework for measuring and comparing performance improvements
2. **Memory Optimization**: Comprehensive data type optimization and memory management
3. **Vectorization Strategies**: Multiple approaches for eliminating loops and improving computation speed
4. **Index Optimization**: Proper indexing strategies for fast data access
5. **Advanced Techniques**: Eval(), Numba integration, and specialized optimization methods

#### Use Cases
- **Large Dataset Processing**: Handling datasets that challenge memory and computation limits
- **Real-time Analytics**: Optimizing code for low-latency data processing
- **ETL Pipelines**: Improving throughput in data transformation workflows
- **Financial Analysis**: High-frequency data processing with performance requirements
- **Scientific Computing**: Large-scale numerical computations with pandas

#### Best Practices
- **Profile First**: Always measure before optimizing to identify bottlenecks
- **Incremental Optimization**: Apply optimizations systematically and measure impact
- **Memory Monitoring**: Track memory usage throughout the optimization process
- **Vectorization Priority**: Focus on eliminating loops and apply functions first
- **Data Type Optimization**: Often provides the biggest memory and speed improvements

#### Pitfalls
- **Premature Optimization**: Optimizing code before identifying actual bottlenecks
- **Over-optimization**: Making code complex for marginal performance gains
- **Memory vs Speed Trade-offs**: Some optimizations increase memory usage for speed
- **Platform Dependencies**: Some optimizations work differently across platforms
- **Maintenance Overhead**: Highly optimized code can be harder to maintain

#### Debugging
```python
def debug_performance_issues(df: pd.DataFrame):
    """Debug common performance issues in pandas operations."""
    
    print("Performance Debugging Checklist:")
    print("="*35)
    
    # Check data types
    memory_usage = df.memory_usage(deep=True)
    print(f"Total memory usage: {memory_usage.sum() / 1024**2:.2f} MB")
    
    # Identify large columns
    large_columns = memory_usage[memory_usage > memory_usage.mean() * 2]
    if not large_columns.empty:
        print("Large memory columns:")
        for col, usage in large_columns.items():
            print(f"  {col}: {usage / 1024**2:.2f} MB")
    
    # Check for object columns that could be categorical
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.5:
            print(f"Consider converting '{col}' to categorical (unique ratio: {unique_ratio:.3f})")
    
    # Check index efficiency
    if not df.index.is_monotonic_increasing:
        print("Consider sorting index for better query performance")
    
    # Memory optimization suggestions
    print(f"\nOptimization suggestions:")
    print(f"• Current shape: {df.shape}")
    print(f"• Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"• Object columns: {len(object_columns)}")
    
def profile_pandas_operation(operation_func, data, operation_name):
    """Profile a pandas operation for performance analysis."""
    import cProfile
    import pstats
    from io import StringIO
    
    pr = cProfile.Profile()
    pr.enable()
    
    result = operation_func(data)
    
    pr.disable()
    
    # Get profiling results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    print(f"Profiling results for {operation_name}:")
    print(s.getvalue())
    
    return result
```

#### Optimization

**Performance Optimization Decision Matrix:**

| Optimization Type | Memory Impact | Speed Improvement | Implementation Complexity | Best For |
|------------------|---------------|-------------------|---------------------------|----------|
| **Data Type Optimization** | -50 to -80% | +10 to +30% | Low | All datasets |
| **Vectorization** | Neutral | +2x to +100x | Medium | Loop-heavy operations |
| **Indexing** | +5 to +20% | +10x to +1000x | Low | Query-heavy workloads |
| **Chunking** | -80 to -95% | Variable | Medium | Large datasets |
| **Categorical Types** | -50 to -90% | +20 to +50% | Low | String/object columns |
| **eval() Expressions** | Neutral | +20 to +50% | Low | Complex calculations |
| **Numba Integration** | Neutral | +10x to +100x | High | Numerical computations |

**Quick Performance Checklist:**
1. ✅ **Data Types**: Optimize numeric types, use categorical for strings
2. ✅ **Vectorization**: Replace loops with vectorized operations  
3. ✅ **Indexing**: Set appropriate indexes for queries
4. ✅ **Memory**: Monitor usage and use chunking for large data
5. ✅ **I/O**: Use efficient formats (Parquet) and specify dtypes
6. ✅ **Profiling**: Measure performance before and after optimization

---

## Question 17

**Explain the importance of using categorical data types, especially when working with a large number of unique values.**

### Answer

#### Theory
Categorical data types in Pandas are essential for memory efficiency and performance optimization, particularly when dealing with string data or limited discrete values. Unlike object dtype which stores each string value separately, categorical dtype stores unique values once and uses integer codes for references. This approach can reduce memory usage by 50-90% and significantly improve performance for operations like groupby, sorting, and filtering.

#### Code Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import time
import psutil
import gc
import warnings
warnings.filterwarnings('ignore')

class CategoricalDataAnalyzer:
    """
    Comprehensive analysis and optimization toolkit for categorical data.
    """
    
    def __init__(self):
        self.memory_comparisons = []
        self.performance_metrics = []
        
    def analyze_categorical_potential(self, df: pd.DataFrame, 
                                   cardinality_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze DataFrame for categorical conversion potential.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        cardinality_threshold (float): Threshold for unique value ratio
        
        Returns:
        dict: Analysis results with recommendations
        """
        
        analysis = {
            'candidates': {},
            'current_memory': df.memory_usage(deep=True).sum(),
            'potential_savings': 0,
            'recommendations': []
        }
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'string':
                # Calculate cardinality metrics
                total_values = len(df[col])
                unique_values = df[col].nunique()
                null_values = df[col].isnull().sum()
                cardinality_ratio = unique_values / (total_values - null_values) if total_values > null_values else 1
                
                # Memory usage analysis
                current_memory = df[col].memory_usage(deep=True)
                
                # Estimate categorical memory usage
                if unique_values > 0:
                    # Memory for categories + memory for codes
                    categories_memory = df[col].dropna().drop_duplicates().memory_usage(deep=True)
                    codes_memory = total_values * 1 if unique_values <= 256 else (
                        total_values * 2 if unique_values <= 65536 else total_values * 4
                    )
                    estimated_categorical_memory = categories_memory + codes_memory
                else:
                    estimated_categorical_memory = current_memory
                
                potential_savings = current_memory - estimated_categorical_memory
                savings_percentage = (potential_savings / current_memory) * 100 if current_memory > 0 else 0
                
                analysis['candidates'][col] = {
                    'current_dtype': str(df[col].dtype),
                    'unique_values': unique_values,
                    'total_values': total_values,
                    'cardinality_ratio': cardinality_ratio,
                    'current_memory_mb': current_memory / 1024**2,
                    'estimated_categorical_memory_mb': estimated_categorical_memory / 1024**2,
                    'potential_savings_mb': potential_savings / 1024**2,
                    'savings_percentage': savings_percentage,
                    'recommendation': 'Convert to categorical' if cardinality_ratio < cardinality_threshold and savings_percentage > 10 else 'Keep as object'
                }
                
                if cardinality_ratio < cardinality_threshold and savings_percentage > 10:
                    analysis['potential_savings'] += potential_savings
                    analysis['recommendations'].append(col)
        
        analysis['total_potential_savings_mb'] = analysis['potential_savings'] / 1024**2
        analysis['total_savings_percentage'] = (analysis['potential_savings'] / analysis['current_memory']) * 100
        
        return analysis
    
    def convert_to_categorical_optimized(self, df: pd.DataFrame, 
                                       columns: List[str] = None,
                                       auto_detect: bool = True) -> pd.DataFrame:
        """
        Convert specified columns to categorical with optimization.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        columns (list): Specific columns to convert
        auto_detect (bool): Automatically detect good candidates
        
        Returns:
        pd.DataFrame: DataFrame with optimized categorical columns
        """
        
        df_optimized = df.copy()
        
        if auto_detect:
            analysis = self.analyze_categorical_potential(df_optimized)
            auto_columns = analysis['recommendations']
            columns = list(set((columns or []) + auto_columns))
        
        if not columns:
            return df_optimized
        
        conversion_log = []
        
        for col in columns:
            if col not in df_optimized.columns:
                continue
                
            try:
                # Store original info
                original_memory = df_optimized[col].memory_usage(deep=True)
                original_dtype = df_optimized[col].dtype
                
                # Convert to categorical
                if df_optimized[col].dtype == 'object':
                    # Handle missing values appropriately
                    df_optimized[col] = df_optimized[col].astype('category')
                
                # Record conversion info
                new_memory = df_optimized[col].memory_usage(deep=True)
                memory_saved = original_memory - new_memory
                savings_percentage = (memory_saved / original_memory) * 100 if original_memory > 0 else 0
                
                conversion_log.append({
                    'column': col,
                    'original_dtype': original_dtype,
                    'new_dtype': df_optimized[col].dtype,
                    'memory_saved_mb': memory_saved / 1024**2,
                    'savings_percentage': savings_percentage,
                    'unique_values': df_optimized[col].nunique()
                })
                
            except Exception as e:
                print(f"Failed to convert {col} to categorical: {e}")
        
        # Summary
        total_memory_saved = sum(log['memory_saved_mb'] for log in conversion_log)
        print(f"Categorical conversion summary:")
        print(f"  Columns converted: {len(conversion_log)}")
        print(f"  Total memory saved: {total_memory_saved:.2f} MB")
        
        for log in conversion_log:
            print(f"  {log['column']}: {log['savings_percentage']:.1f}% reduction")
        
        return df_optimized

def demonstrate_memory_efficiency():
    """Demonstrate memory efficiency gains from categorical data."""
    
    print("=== CATEGORICAL DATA MEMORY EFFICIENCY ===")
    
    # Create test dataset with various cardinality levels
    np.random.seed(42)
    n_rows = 500_000
    
    # Different cardinality scenarios
    data = {
        'low_cardinality': np.random.choice(['A', 'B', 'C', 'D'], n_rows),  # 4 unique
        'medium_cardinality': np.random.choice([f'Cat_{i}' for i in range(100)], n_rows),  # 100 unique
        'high_cardinality': np.random.choice([f'Item_{i}' for i in range(10000)], n_rows),  # 10k unique
        'very_high_cardinality': [f'ID_{i}_{j}' for i, j in enumerate(np.random.randint(0, 100, n_rows))],  # ~500k unique
        'numeric_data': np.random.randn(n_rows),
        'boolean_data': np.random.choice([True, False], n_rows)
    }
    
    df_original = pd.DataFrame(data)
    
    print(f"Original Dataset:")
    print(f"  Shape: {df_original.shape}")
    print(f"  Memory usage: {df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Analyze each column
    print(f"\nDetailed analysis by column:")
    for col in df_original.columns:
        if df_original[col].dtype == 'object':
            memory_mb = df_original[col].memory_usage(deep=True) / 1024**2
            unique_count = df_original[col].nunique()
            cardinality_ratio = unique_count / len(df_original[col])
            
            print(f"  {col}:")
            print(f"    Memory: {memory_mb:.2f} MB")
            print(f"    Unique values: {unique_count:,}")
            print(f"    Cardinality ratio: {cardinality_ratio:.4f}")
    
    # Convert appropriate columns to categorical
    categorical_candidates = ['low_cardinality', 'medium_cardinality', 'high_cardinality']
    
    df_categorical = df_original.copy()
    memory_savings = {}
    
    for col in categorical_candidates:
        original_memory = df_categorical[col].memory_usage(deep=True)
        df_categorical[col] = df_categorical[col].astype('category')
        new_memory = df_categorical[col].memory_usage(deep=True)
        
        memory_savings[col] = {
            'original_mb': original_memory / 1024**2,
            'categorical_mb': new_memory / 1024**2,
            'savings_mb': (original_memory - new_memory) / 1024**2,
            'savings_percentage': ((original_memory - new_memory) / original_memory) * 100
        }
    
    print(f"\nMemory savings from categorical conversion:")
    total_original = sum(s['original_mb'] for s in memory_savings.values())
    total_categorical = sum(s['categorical_mb'] for s in memory_savings.values())
    total_savings = total_original - total_categorical
    
    for col, savings in memory_savings.items():
        print(f"  {col}:")
        print(f"    Original: {savings['original_mb']:.2f} MB")
        print(f"    Categorical: {savings['categorical_mb']:.2f} MB")
        print(f"    Savings: {savings['savings_mb']:.2f} MB ({savings['savings_percentage']:.1f}%)")
    
    print(f"\nTotal savings: {total_savings:.2f} MB ({(total_savings/total_original)*100:.1f}%)")
    print(f"Overall memory reduction: {df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB → {df_categorical.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df_original, df_categorical

def demonstrate_performance_improvements():
    """Demonstrate performance improvements with categorical data."""
    
    print("\n=== CATEGORICAL DATA PERFORMANCE IMPROVEMENTS ===")
    
    # Create performance test dataset
    np.random.seed(42)
    n_rows = 200_000
    
    categories = [f'Category_{i}' for i in range(20)]
    
    df_object = pd.DataFrame({
        'category_col': np.random.choice(categories, n_rows),
        'value_col': np.random.randn(n_rows),
        'id_col': range(n_rows)
    })
    
    df_categorical = df_object.copy()
    df_categorical['category_col'] = df_categorical['category_col'].astype('category')
    
    print(f"Performance test dataset: {n_rows:,} rows, {len(categories)} unique categories")
    
    # Performance testing function
    def time_operation(operation_func, data, operation_name, iterations=3):
        """Time an operation multiple times and return average."""
        times = []
        for _ in range(iterations):
            start_time = time.time()
            result = operation_func(data)
            end_time = time.time()
            times.append(end_time - start_time)
        return np.mean(times)
    
    # Test various operations
    operations = [
        ('GroupBy Mean', lambda df: df.groupby('category_col')['value_col'].mean()),
        ('Value Counts', lambda df: df['category_col'].value_counts()),
        ('Sort by Category', lambda df: df.sort_values('category_col')),
        ('Filter Category', lambda df: df[df['category_col'].isin(categories[:5])]),
        ('Unique Values', lambda df: df['category_col'].unique()),
        ('Category Frequency', lambda df: df['category_col'].value_counts().to_dict())
    ]
    
    print(f"\nPerformance comparison (object vs categorical):")
    
    performance_results = []
    
    for op_name, operation in operations:
        # Time object dtype operation
        object_time = time_operation(operation, df_object, f"{op_name} (object)")
        
        # Time categorical dtype operation
        categorical_time = time_operation(operation, df_categorical, f"{op_name} (categorical)")
        
        # Calculate speedup
        speedup = object_time / categorical_time if categorical_time > 0 else 1
        
        performance_results.append({
            'Operation': op_name,
            'Object Time (s)': object_time,
            'Categorical Time (s)': categorical_time,
            'Speedup': speedup
        })
        
        print(f"  {op_name}:")
        print(f"    Object: {object_time:.4f}s")
        print(f"    Categorical: {categorical_time:.4f}s")
        print(f"    Speedup: {speedup:.2f}x")
    
    # Summary
    avg_speedup = np.mean([r['Speedup'] for r in performance_results])
    print(f"\nAverage speedup with categorical: {avg_speedup:.2f}x")
    
    return performance_results

def demonstrate_categorical_operations():
    """Demonstrate specialized categorical operations."""
    
    print("\n=== CATEGORICAL DATA OPERATIONS ===")
    
    # Create sample categorical data
    categories = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
    data = pd.DataFrame({
        'customer_tier': pd.Categorical(np.random.choice(categories, 1000), 
                                      categories=categories, ordered=True),
        'region': pd.Categorical(np.random.choice(['North', 'South', 'East', 'West'], 1000)),
        'satisfaction': np.random.uniform(1, 10, 1000)
    })
    
    print("Sample categorical data:")
    print(data.head())
    print(f"\nData types:")
    print(data.dtypes)
    
    # 1. Ordered categorical operations
    print(f"\n1. Ordered categorical operations:")
    print(f"   Customer tier levels: {data['customer_tier'].cat.categories}")
    print(f"   Is ordered: {data['customer_tier'].cat.ordered}")
    
    # Filter by category comparison
    premium_customers = data[data['customer_tier'] >= 'Gold']
    print(f"   Premium customers (Gold+): {len(premium_customers)}")
    
    # 2. Category management
    print(f"\n2. Category management:")
    print(f"   Original categories: {list(data['region'].cat.categories)}")
    
    # Add new category
    data['region'] = data['region'].cat.add_categories(['Central'])
    print(f"   After adding 'Central': {list(data['region'].cat.categories)}")
    
    # Remove unused categories
    data['region'] = data['region'].cat.remove_unused_categories()
    print(f"   After removing unused: {list(data['region'].cat.categories)}")
    
    # 3. Category renaming and reordering
    print(f"\n3. Category manipulation:")
    
    # Rename categories
    data['region'] = data['region'].cat.rename_categories({
        'North': 'Northern', 'South': 'Southern', 
        'East': 'Eastern', 'West': 'Western'
    })
    print(f"   Renamed categories: {list(data['region'].cat.categories)}")
    
    # Reorder categories
    data['region'] = data['region'].cat.reorder_categories(['Northern', 'Eastern', 'Southern', 'Western'])
    print(f"   Reordered categories: {list(data['region'].cat.categories)}")
    
    # 4. Categorical aggregations
    print(f"\n4. Categorical aggregations:")
    
    # GroupBy with categorical preserves categories
    region_stats = data.groupby('region', observed=False)['satisfaction'].agg(['mean', 'count'])
    print("   Region statistics (preserves all categories):")
    print(region_stats)
    
    # Customer tier analysis
    tier_analysis = data.groupby('customer_tier', observed=False).agg({
        'satisfaction': ['mean', 'std', 'count']
    }).round(2)
    print("\n   Customer tier analysis:")
    print(tier_analysis)
    
    return data

def demonstrate_advanced_categorical_techniques():
    """Demonstrate advanced categorical data techniques."""
    
    print("\n=== ADVANCED CATEGORICAL TECHNIQUES ===")
    
    # 1. Handling large cardinality with frequency-based categoricals
    print("1. Frequency-based categorical optimization:")
    
    # Create dataset with high cardinality
    np.random.seed(42)
    n_rows = 100_000
    
    # Simulate realistic high-cardinality data (e.g., product IDs)
    high_card_data = pd.DataFrame({
        'product_id': [f'PROD_{i:06d}' for i in np.random.randint(0, 50000, n_rows)],
        'sales': np.random.exponential(100, n_rows)
    })
    
    print(f"   Original unique products: {high_card_data['product_id'].nunique():,}")
    print(f"   Memory usage: {high_card_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Strategy: Keep only top N categories, group rest as "Other"
    def optimize_high_cardinality_categorical(series: pd.Series, top_n: int = 1000) -> pd.Series:
        """Optimize high cardinality categorical by keeping only top N categories."""
        
        value_counts = series.value_counts()
        top_categories = value_counts.head(top_n).index
        
        # Replace less frequent categories with "Other"
        optimized_series = series.where(series.isin(top_categories), 'Other')
        
        # Convert to categorical
        return optimized_series.astype('category')
    
    # Apply optimization
    high_card_data['product_id_optimized'] = optimize_high_cardinality_categorical(
        high_card_data['product_id'], top_n=1000
    )
    
    print(f"   Optimized unique products: {high_card_data['product_id_optimized'].nunique():,}")
    
    # Memory comparison
    original_memory = high_card_data['product_id'].memory_usage(deep=True) / 1024**2
    optimized_memory = high_card_data['product_id_optimized'].memory_usage(deep=True) / 1024**2
    
    print(f"   Memory savings: {original_memory:.2f} MB → {optimized_memory:.2f} MB ({((original_memory-optimized_memory)/original_memory)*100:.1f}% reduction)")
    
    # 2. Temporal categorical data
    print(f"\n2. Temporal categorical optimization:")
    
    # Create time-based categorical data
    date_range = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    temporal_data = pd.DataFrame({
        'date': np.random.choice(date_range, 10000),
        'event_type': np.random.choice(['login', 'purchase', 'logout', 'view'], 10000),
        'value': np.random.randn(10000)
    })
    
    # Extract temporal features as categorical
    temporal_data['year'] = temporal_data['date'].dt.year.astype('category')
    temporal_data['month'] = temporal_data['date'].dt.month.astype('category')
    temporal_data['day_of_week'] = temporal_data['date'].dt.day_name().astype('category')
    temporal_data['quarter'] = temporal_data['date'].dt.quarter.astype('category')
    
    # Order day of week properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    temporal_data['day_of_week'] = temporal_data['day_of_week'].cat.reorder_categories(day_order)
    
    print("   Temporal categorical features created:")
    print(f"   Years: {sorted(temporal_data['year'].cat.categories)}")
    print(f"   Day of week: {list(temporal_data['day_of_week'].cat.categories)}")
    
    # Temporal analysis
    temporal_analysis = temporal_data.groupby(['day_of_week', 'event_type'], observed=False)['value'].mean().unstack(fill_value=0)
    print("\n   Day of week vs event type analysis:")
    print(temporal_analysis.round(3))
    
    # 3. Categorical encoding for machine learning
    print(f"\n3. Categorical encoding for ML:")
    
    # Create sample dataset
    ml_data = pd.DataFrame({
        'category_low_card': pd.Categorical(np.random.choice(['A', 'B', 'C'], 1000)),
        'category_medium_card': pd.Categorical(np.random.choice([f'Cat_{i}' for i in range(10)], 1000)),
        'category_high_card': pd.Categorical(np.random.choice([f'Item_{i}' for i in range(100)], 1000)),
        'target': np.random.randn(1000)
    })
    
    # Different encoding strategies
    encoding_strategies = {}
    
    # One-hot encoding for low cardinality
    encoding_strategies['onehot_low'] = pd.get_dummies(ml_data['category_low_card'], prefix='low')
    
    # Label encoding for medium cardinality
    encoding_strategies['label_medium'] = ml_data['category_medium_card'].cat.codes
    
    # Target encoding for high cardinality
    target_means = ml_data.groupby('category_high_card')['target'].mean()
    encoding_strategies['target_high'] = ml_data['category_high_card'].map(target_means)
    
    print("   Encoding strategies applied:")
    print(f"   One-hot (low cardinality): {encoding_strategies['onehot_low'].shape[1]} features")
    print(f"   Label encoding (medium): 1 feature")
    print(f"   Target encoding (high): 1 feature")
    
    return high_card_data, temporal_data, ml_data

def categorical_best_practices_guide():
    """Comprehensive guide to categorical data best practices."""
    
    print("\n" + "="*60)
    print("CATEGORICAL DATA BEST PRACTICES GUIDE")
    print("="*60)
    
    best_practices = {
        "When to Use Categorical": [
            "• String data with limited unique values (< 50% cardinality ratio)",
            "• Repeated string values in large datasets",
            "• Ordinal data with natural ordering",
            "• Data that will be used for grouping operations",
            "• Memory-constrained environments"
        ],
        "Memory Optimization": [
            "• Convert object columns with < 50% unique values",
            "• Use ordered categoricals for natural ordering",
            "• Consider frequency-based grouping for high cardinality",
            "• Remove unused categories after filtering",
            "• Monitor memory usage before/after conversion"
        ],
        "Performance Optimization": [
            "• Use observed=False in groupby to preserve all categories",
            "• Sort categorical data for better performance",
            "• Use .cat accessor for category-specific operations",
            "• Leverage vectorized categorical operations",
            "• Pre-define categories when possible"
        ],
        "Data Quality": [
            "• Handle missing values before converting to categorical",
            "• Validate category ordering for ordinal data",
            "• Use consistent category names across datasets",
            "• Document category meanings and orderings",
            "• Test for category stability across time"
        ],
        "Machine Learning": [
            "• Choose appropriate encoding strategy by cardinality",
            "• One-hot encode low cardinality (< 10 categories)",
            "• Label encode ordinal or medium cardinality (10-50)",
            "• Target encode high cardinality (> 50 categories)",
            "• Preserve categorical information in pipelines"
        ],
        "Common Pitfalls": [
            "• Don't convert high cardinality data without strategy",
            "• Avoid categorical for truly unique identifiers",
            "• Don't ignore missing values during conversion",
            "• Don't assume all string data should be categorical",
            "• Don't forget to handle new categories in production"
        ]
    }
    
    for category, practices in best_practices.items():
        print(f"\n{category}:")
        for practice in practices:
            print(f"  {practice}")
    
    # Decision matrix
    print(f"\n" + "="*60)
    print("CATEGORICAL CONVERSION DECISION MATRIX")
    print("="*60)
    
    decision_matrix = pd.DataFrame({
        'Cardinality': ['Very Low (2-5)', 'Low (5-20)', 'Medium (20-100)', 'High (100-1000)', 'Very High (1000+)'],
        'Recommendation': ['Always Convert', 'Convert', 'Convert with Analysis', 'Frequency Grouping', 'Avoid or Group'],
        'Memory Savings': ['60-90%', '40-80%', '20-60%', '10-40%', '0-20%'],
        'Performance Gain': ['High', 'High', 'Medium', 'Low', 'None/Negative'],
        'Best Encoding for ML': ['One-Hot', 'One-Hot/Label', 'Label/Target', 'Target/Hash', 'Hash/Embeddings']
    })
    
    print(decision_matrix.to_string(index=False))

def run_comprehensive_categorical_demo():
    """Run comprehensive categorical data demonstration."""
    
    print("CATEGORICAL DATA COMPREHENSIVE ANALYSIS")
    print("="*45)
    
    # Initialize analyzer
    analyzer = CategoricalDataAnalyzer()
    
    # Run demonstrations
    df_original, df_categorical = demonstrate_memory_efficiency()
    performance_results = demonstrate_performance_improvements()
    categorical_ops_data = demonstrate_categorical_operations()
    advanced_results = demonstrate_advanced_categorical_techniques()
    categorical_best_practices_guide()
    
    # Final summary
    print(f"\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*60)
    
    print("Key Benefits of Categorical Data Types:")
    print("• Memory reduction: 50-90% for appropriate data")
    print("• Performance improvement: 2-10x for common operations")
    print("• Better data integrity with defined categories")
    print("• Optimized storage and computation")
    print("• Enhanced groupby and aggregation performance")
    
    print("\nOptimal Use Cases:")
    print("• String columns with repetitive values")
    print("• Ordinal data with natural ordering")
    print("• Data used frequently in groupby operations")
    print("• Memory-constrained environments")
    print("• Large datasets with categorical features")

# Execute demonstration
if __name__ == "__main__":
    run_comprehensive_categorical_demo()
```

#### Explanation
1. **Memory Efficiency Analysis**: Comprehensive analysis of memory usage patterns and optimization potential
2. **Performance Benchmarking**: Systematic comparison of operations on object vs categorical data
3. **Advanced Techniques**: Strategies for high-cardinality data and specialized categorical operations
4. **ML Integration**: Categorical encoding strategies for different machine learning scenarios
5. **Best Practices Framework**: Decision matrix and guidelines for optimal categorical usage

#### Use Cases
- **Large Dataset Processing**: Reducing memory footprint in big data scenarios
- **Repeated String Analysis**: Optimizing performance for groupby and aggregation operations
- **Machine Learning Pipelines**: Efficient categorical encoding for model training
- **Data Warehousing**: Storage optimization for dimensional data
- **Time Series Analysis**: Efficient handling of temporal categorical features

#### Best Practices
- **Cardinality Analysis**: Evaluate unique value ratio before conversion
- **Memory Monitoring**: Track memory usage before and after conversion
- **Performance Testing**: Benchmark operations on categorical vs object types
- **Ordered Categories**: Use ordered categoricals for natural orderings
- **Category Management**: Maintain category consistency across datasets

#### Pitfalls
- **High Cardinality Conversion**: Converting data with too many unique values can hurt performance
- **Missing Category Handling**: New categories in production data can cause errors
- **Memory Overhead**: Very high cardinality can increase memory usage
- **Type Conversion Issues**: Inappropriate conversions can break downstream operations
- **Category Ordering**: Incorrect ordering in ordinal data affects analysis

#### Debugging
```python
def debug_categorical_issues(df: pd.DataFrame):
    """Debug common categorical data issues."""
    
    print("Categorical Data Debugging:")
    print("="*30)
    
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            cat_col = df[col]
            
            print(f"\nColumn: {col}")
            print(f"  Categories: {len(cat_col.cat.categories)}")
            print(f"  Ordered: {cat_col.cat.ordered}")
            print(f"  Memory usage: {cat_col.memory_usage(deep=True) / 1024**2:.2f} MB")
            
            # Check for unused categories
            used_categories = set(cat_col.dropna().unique())
            all_categories = set(cat_col.cat.categories)
            unused = all_categories - used_categories
            
            if unused:
                print(f"  Unused categories: {len(unused)} ({list(unused)[:5]}...)")
            
            # Check for missing values
            missing_count = cat_col.isnull().sum()
            if missing_count > 0:
                print(f"  Missing values: {missing_count} ({missing_count/len(cat_col)*100:.1f}%)")

def optimize_categorical_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize categorical columns for memory efficiency."""
    
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        if pd.api.types.is_categorical_dtype(df_optimized[col]):
            # Remove unused categories
            df_optimized[col] = df_optimized[col].cat.remove_unused_categories()
            
            # Optimize category dtype if possible
            n_categories = len(df_optimized[col].cat.categories)
            if n_categories <= 256:
                # Can use uint8 for codes
                df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized
```

#### Optimization

**Categorical Optimization Decision Tree:**

```
Is the column string/object type?
├── No → Keep as is
└── Yes → Calculate cardinality ratio
    ├── > 50% unique → Consider other strategies
    │   ├── High frequency patterns → Group by frequency
    │   ├── Hierarchical structure → Create hierarchy
    │   └── Truly unique → Keep as object or use hash encoding
    └── ≤ 50% unique → Convert to categorical
        ├── Natural ordering → Use ordered categorical
        ├── ML feature → Choose encoding strategy by cardinality
        └── General use → Use unordered categorical
```

**Memory Impact by Cardinality:**

| Unique Values | Memory Reduction | Performance Gain | Recommendation |
|---------------|------------------|------------------|----------------|
| **2-10** | 70-90% | Very High | Always convert |
| **10-50** | 50-80% | High | Convert |
| **50-200** | 30-60% | Medium | Analyze first |
| **200-1000** | 10-40% | Low | Use frequency grouping |
| **1000+** | 0-20% | None/Negative | Avoid or use advanced techniques |

This comprehensive approach to categorical data ensures optimal memory usage and performance while maintaining data integrity and analysis capabilities.

---

