# Pandas Interview Questions - Theory Questions

## Question 1

**What is Pandas in Python and why is it used for data analysis?**

### Definition
Pandas is an open-source Python library for data manipulation and analysis. Built on NumPy, it provides high-performance, easy-to-use data structures (DataFrame and Series) for working with structured data.

### Why It's Essential for Data Analysis

| Feature | Description |
|---------|-------------|
| **DataFrame** | 2D table structure like Excel/SQL |
| **Data Alignment** | Automatic alignment based on labels |
| **Missing Data** | Built-in handling of NaN values |
| **Reshaping** | Pivot, melt, stack operations |
| **Grouping** | SQL-like GROUP BY operations |
| **Time Series** | First-class datetime support |
| **Integration** | Works with NumPy, Matplotlib, Scikit-learn |

### Code Example
```python
import pandas as pd

# Read data
df = pd.read_csv('sales.csv')

# Quick analysis
print(df.head())           # First 5 rows
print(df.describe())       # Statistics
print(df.info())           # Data types and memory

# Grouping and aggregation
sales_by_region = df.groupby('region')['amount'].sum()
```

---

## Question 2

**Explain the difference between a Series and a DataFrame.**

### Comparison

| Feature | Series | DataFrame |
|---------|--------|-----------|
| Dimensions | 1D (column) | 2D (table) |
| Analogy | One column | Entire spreadsheet |
| Data Types | Homogeneous | Heterogeneous (per column) |
| Components | Values + Index | Values + Index + Columns |

### Code Example
```python
import pandas as pd

# Series: 1D labeled array
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='values')
print(s)
# a    10
# b    20
# c    30

# DataFrame: 2D table
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30],
    'salary': [50000, 60000]
})

# Accessing a column returns a Series
print(type(df['name']))  # <class 'pandas.core.series.Series'>
```

---

## Question 3

**What are Pandas indexes, and how are they used?**

### Definition
An index is an immutable array of labels that identifies rows (and columns). It enables fast lookups, data alignment, and relational operations.

### Key Uses

| Use | Description |
|-----|-------------|
| Label-based Selection | `df.loc['row_label']` |
| Data Alignment | Auto-aligns during operations |
| Fast Lookups | O(1) for hash-based indexes |
| Joins/Merges | Acts as keys for combining data |
| MultiIndex | Hierarchical indexing for complex data |

### Code Example
```python
import pandas as pd

df = pd.DataFrame({
    'product_id': ['P101', 'P102', 'P103'],
    'price': [50, 75, 120]
})

# Set meaningful index
df.set_index('product_id', inplace=True)

# Fast label-based lookup
print(df.loc['P102', 'price'])  # 75

# Data alignment works on index
prices_new = pd.Series([60, 80], index=['P101', 'P102'])
df['new_price'] = prices_new  # Auto-aligns by index
```

---

## Question 4

**Explain data alignment and broadcasting in Pandas.**

### Data Alignment
Pandas automatically aligns data based on index labels before operations. Mismatched labels result in NaN.

```python
import pandas as pd

s1 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
s2 = pd.Series([100, 200, 300], index=['c', 'b', 'd'])

result = s1 + s2
# a      NaN   (only in s1)
# b    220.0   (20 + 200)
# c    130.0   (30 + 100)
# d      NaN   (only in s2)
```

### Broadcasting
Operations between DataFrame and Series broadcast across rows/columns.

```python
df = pd.DataFrame([[1, 2, 3],
                   [4, 5, 6]], columns=['A', 'B', 'C'])

row = df.iloc[0]  # Series: A=1, B=2, C=3

# Subtract first row from all rows
result = df - row
#    A  B  C
# 0  0  0  0
# 1  3  3  3
```

---

## Question 5

**What is data slicing vs filtering in Pandas?**

### Comparison

| Aspect | Slicing | Filtering |
|--------|---------|-----------|
| Purpose | Select by position/range | Select by condition |
| Syntax | `df.iloc[1:4]`, `df.loc['a':'c']` | `df[df['col'] > 5]` |
| Mechanism | Index/position based | Boolean mask based |

### Code Example
```python
import pandas as pd

df = pd.DataFrame({'A': [10, 20, 30, 40, 50],
                   'B': ['x', 'y', 'x', 'y', 'z']})

# --- Slicing ---
# Positional: rows 1-3
print(df.iloc[1:4])

# Label-based (if string index)
df.index = list('abcde')
print(df.loc['b':'d'])

# --- Filtering ---
# Rows where A > 25
print(df[df['A'] > 25])

# Multiple conditions
print(df[(df['A'] > 20) & (df['B'] == 'y')])
```

---

## Question 6

**Describe joining and merging in Pandas.**

### Join Types

| Type | Returns |
|------|---------|
| `inner` | Only matching keys |
| `left` | All from left + matched from right |
| `right` | All from right + matched from left |
| `outer` | All from both |

### Code Example
```python
import pandas as pd

customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103],
    'customer_id': [1, 2, 4],
    'amount': [50, 75, 120]
})

# Left join: keep all customers
result = pd.merge(customers, orders, on='customer_id', how='left')
#    customer_id     name  order_id  amount
# 0            1    Alice     101.0    50.0
# 1            2      Bob     102.0    75.0
# 2            3  Charlie       NaN     NaN
```

### Methods
- `pd.merge()`: Database-style joins on columns
- `df.join()`: Join on index
- `pd.concat()`: Stack DataFrames (not a true join)

---

## Question 7

**How do you convert categorical data to numerical?**

### Three Methods

| Method | Use Case | Result |
|--------|----------|--------|
| One-Hot Encoding | Nominal (no order) | Binary columns |
| Label Encoding | Ordinal (has order) | Integer codes |
| Manual Mapping | Custom order | Defined integers |

### Code Examples
```python
import pandas as pd

df = pd.DataFrame({'color': ['Red', 'Green', 'Blue', 'Green'],
                   'size': ['Small', 'Large', 'Medium', 'Small']})

# 1. One-Hot Encoding (nominal)
one_hot = pd.get_dummies(df['color'], prefix='color')
#    color_Blue  color_Green  color_Red
# 0           0            0          1
# 1           0            1          0

# 2. Label Encoding (ordinal)
size_order = pd.Categorical(df['size'], 
                            categories=['Small', 'Medium', 'Large'],
                            ordered=True)
df['size_encoded'] = size_order.codes  # 0, 2, 1, 0

# 3. Manual Mapping
risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['risk_code'] = df['risk'].map(risk_map)
```

---

## Question 8

**What is the purpose of the apply() function?**

### Definition
`apply()` applies a function along an axis of a DataFrame (row-wise or column-wise). It's flexible but slower than vectorized operations.

### Axis Parameter
- `axis=0`: Apply to each column
- `axis=1`: Apply to each row

### Code Example
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]})

# Row-wise custom function
def custom_calc(row):
    if row['A'] > 1:
        return row['A'] * row['B']
    return row['B']

df['C'] = df.apply(custom_calc, axis=1)
#    A   B    C
# 0  1  10   10
# 1  2  20   40
# 2  3  30   90
```

### Performance Note
Prefer vectorized operations when possible:
```python
# Slow (apply)
df['result'] = df.apply(lambda row: row['A'] + row['B'], axis=1)

# Fast (vectorized)
df['result'] = df['A'] + df['B']
```

---

## Question 9

**Explain astype, to_numeric, and pd.to_datetime.**

### Comparison

| Function | Purpose | Error Handling |
|----------|---------|----------------|
| `astype()` | General type conversion | Raises error on failure |
| `pd.to_numeric()` | Convert to number | `errors='coerce'` → NaN |
| `pd.to_datetime()` | Convert to datetime | `errors='coerce'` → NaT |

### Code Examples
```python
import pandas as pd

# astype: simple conversion
df['age'] = df['age'].astype('int32')
df['category'] = df['category'].astype('category')

# to_numeric: handles errors gracefully
s = pd.Series(['1', '2', 'bad', '4'])
pd.to_numeric(s, errors='coerce')  # [1, 2, NaN, 4]

# to_datetime: parse dates
s = pd.Series(['2023-01-01', '2023-02-15', 'invalid'])
pd.to_datetime(s, errors='coerce')  # [2023-01-01, 2023-02-15, NaT]
```

---

## Question 10

**Explain data ranking in Pandas.**

### Ranking Methods

| Method | Description |
|--------|-------------|
| `average` | Mean of tied ranks (default) |
| `min` | Lowest rank for ties |
| `max` | Highest rank for ties |
| `first` | Ranks in order of appearance |
| `dense` | Like min, but no gaps |

### Code Example
```python
import pandas as pd

s = pd.Series([3, 1, 4, 1, 5])

print(s.rank(method='average'))  # [3.0, 1.5, 4.0, 1.5, 5.0]
print(s.rank(method='min'))      # [3.0, 1.0, 4.0, 1.0, 5.0]
print(s.rank(method='dense'))    # [2.0, 1.0, 3.0, 1.0, 4.0]
```

---

## Question 11

**What is a crosstab in Pandas?**

### Definition
`pd.crosstab()` computes a cross-tabulation (frequency table) of two or more categorical variables. It's useful for exploring relationships between categories.

### Code Example
```python
import pandas as pd

df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    'product': ['A', 'A', 'B', 'A', 'B', 'B'],
    'purchased': [1, 1, 0, 1, 1, 0]
})

# Simple frequency table
print(pd.crosstab(df['gender'], df['product']))
# product  A  B
# gender       
# F        2  1
# M        1  2

# With aggregation
print(pd.crosstab(df['gender'], df['product'], 
                  values=df['purchased'], aggfunc='sum'))
```

---

## Question 12

**How do you perform a MultiIndex query?**

### Definition
MultiIndex (hierarchical index) allows multiple levels of indexing. Query using `.loc` with tuples or `xs()` for cross-section selection.

### Code Example
```python
import pandas as pd

# Create MultiIndex DataFrame
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
index = pd.MultiIndex.from_arrays(arrays, names=['letter', 'number'])
df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=index)

#                 value
# letter number       
# A      1           10
#        2           20
# B      1           30
#        2           40

# Query specific level
print(df.loc['A'])           # All rows where letter='A'
print(df.loc[('A', 1)])      # Specific combination
print(df.xs(1, level='number'))  # All rows where number=1
```

---

## Question 13

**How do you export a DataFrame to different file formats?**

### Export Methods

| Format | Method | Common Parameters |
|--------|--------|-------------------|
| CSV | `df.to_csv()` | `index=False`, `encoding='utf-8'` |
| Excel | `df.to_excel()` | `sheet_name`, `index=False` |
| JSON | `df.to_json()` | `orient='records'` |
| Parquet | `df.to_parquet()` | `compression='snappy'` |
| SQL | `df.to_sql()` | `if_exists='replace'` |

### Code Example
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

# CSV
df.to_csv('output.csv', index=False)

# Excel
df.to_excel('output.xlsx', sheet_name='Data', index=False)

# JSON
df.to_json('output.json', orient='records')

# Parquet (efficient for large data)
df.to_parquet('output.parquet')
```

---

## Question 14

**How to handle larger-than-memory data with Dask or Modin?**

### Dask
Dask provides parallel computing and out-of-core processing. It uses lazy evaluation.

```python
import dask.dataframe as dd

# Read large CSV in chunks
df = dd.read_csv('large_file.csv')

# Operations are lazy (not executed yet)
result = df.groupby('category').sum()

# Execute and get pandas DataFrame
result_pd = result.compute()
```

### Modin
Modin is a drop-in replacement that parallelizes Pandas operations.

```python
import modin.pandas as pd  # Just change the import!

df = pd.read_csv('large_file.csv')  # Automatically parallelized
df.groupby('category').mean()  # Uses all CPU cores
```

---

## Question 15

**How to use Pandas to preprocess data for ML?**

### Common Preprocessing Steps

```python
import pandas as pd
import numpy as np

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df.dropna(subset=['target'], inplace=True)

# 3. Encode categorical variables
df = pd.get_dummies(df, columns=['category'])

# 4. Feature scaling
df['age_scaled'] = (df['age'] - df['age'].mean()) / df['age'].std()

# 5. Create features
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100],
                         labels=['child', 'young', 'middle', 'senior'])

# 6. Split features and target
X = df.drop('target', axis=1)
y = df['target']
```

---

## Question 16

**What are strategies for optimizing Pandas performance?**

### Optimization Strategies

| Strategy | Impact |
|----------|--------|
| Use `category` dtype | 90%+ memory reduction for low cardinality |
| Downcast numeric types | int64→int32, float64→float32 |
| Use vectorized operations | 10-100x faster than apply() |
| Use `query()` for filtering | More readable, sometimes faster |
| Avoid chained indexing | Use `.loc[]` instead |

### Code Example
```python
import pandas as pd

# 1. Use category dtype
df['country'] = df['country'].astype('category')

# 2. Downcast numerics
df['id'] = pd.to_numeric(df['id'], downcast='integer')

# 3. Vectorized vs apply
# Slow
df['result'] = df.apply(lambda x: x['a'] + x['b'], axis=1)
# Fast
df['result'] = df['a'] + df['b']

# 4. Use query() for readability
df.query('age > 30 and salary > 50000')
```

---

## Question 17

**Explain the importance of categorical data types.**

### Benefits

| Benefit | Description |
|---------|-------------|
| Memory Efficiency | Stores as integers, not strings |
| Performance | Faster groupby, sorting |
| Ordinality | Can define order for ordinal data |
| Type Safety | Prevents invalid categories |

### Code Example
```python
import pandas as pd

# Before: object dtype (inefficient)
df = pd.DataFrame({'country': ['USA', 'UK', 'USA', 'UK'] * 100000})
print(f"Object memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# After: category dtype (efficient)
df['country'] = df['country'].astype('category')
print(f"Category memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
# Typically 90%+ reduction!

# Ordinal categories
size_type = pd.CategoricalDtype(categories=['S', 'M', 'L', 'XL'], ordered=True)
df['size'] = df['size'].astype(size_type)

# Now comparisons work
df[df['size'] > 'M']  # Returns L and XL
```

