# Pandas Interview Questions - General Questions

## Question 1

**How can you read and write data from/to a CSV file in Pandas?**

### Reading CSV
```python
import pandas as pd

# Basic read
df = pd.read_csv('data.csv')

# With options
df = pd.read_csv('data.csv',
                 sep=',',              # Delimiter
                 header=0,             # Row number for header
                 index_col='id',       # Column to use as index
                 usecols=['A', 'B'],   # Only read specific columns
                 dtype={'A': 'int32'}, # Specify dtypes
                 na_values=['NA', ''], # Treat as NaN
                 parse_dates=['date'], # Parse date columns
                 nrows=1000)           # Read only first 1000 rows
```

### Writing CSV
```python
df.to_csv('output.csv',
          index=False,           # Don't write index
          columns=['A', 'B'],    # Only write specific columns
          encoding='utf-8')
```

---

## Question 2

**How do you handle missing data in a DataFrame?**

### Detection
```python
import pandas as pd

# Check for missing values
df.isnull()              # Boolean mask
df.isnull().sum()        # Count per column
df.isnull().sum().sum()  # Total count
```

### Handling Methods
```python
# 1. Drop rows with NaN
df.dropna()                    # Drop any row with NaN
df.dropna(subset=['col1'])     # Only check specific column
df.dropna(thresh=3)            # Keep rows with at least 3 non-NaN

# 2. Fill with value
df.fillna(0)                   # Fill all with 0
df['col'].fillna(df['col'].mean())  # Fill with mean
df.fillna(method='ffill')      # Forward fill
df.fillna(method='bfill')      # Backward fill

# 3. Interpolate
df.interpolate()               # Linear interpolation
```

---

## Question 3

**How do you apply a function to all elements in a DataFrame column?**

### Methods
```python
import pandas as pd

df = pd.DataFrame({'text': ['hello', 'world', 'pandas']})

# 1. map() - for element-wise operations on Series
df['upper'] = df['text'].map(str.upper)
df['upper'] = df['text'].map(lambda x: x.upper())

# 2. apply() - more flexible
df['length'] = df['text'].apply(len)

# 3. Vectorized string methods (preferred for strings)
df['upper'] = df['text'].str.upper()
df['first_char'] = df['text'].str[0]
```

### Performance Tip
```python
# Slow (apply)
df['result'] = df['num'].apply(lambda x: x * 2)

# Fast (vectorized)
df['result'] = df['num'] * 2
```

---

## Question 4

**How do you handle duplicate rows in a DataFrame?**

### Detection and Removal
```python
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 2, 3],
    'B': ['x', 'y', 'y', 'z']
})

# Detect duplicates
df.duplicated()                    # Boolean mask
df.duplicated(subset=['A'])        # Check only column A
df.duplicated(keep='last')         # Mark first occurrence as duplicate

# Remove duplicates
df.drop_duplicates()               # Keep first by default
df.drop_duplicates(keep='last')    # Keep last
df.drop_duplicates(subset=['A'])   # Based on column A only

# Count duplicates
df.duplicated().sum()
```

---

## Question 5

**How can you pivot data in a DataFrame?**

### pivot_table()
```python
import pandas as pd

df = pd.DataFrame({
    'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 150, 120, 180]
})

# Create pivot table
pivot = pd.pivot_table(df,
                       values='sales',      # Values to aggregate
                       index='date',        # Rows
                       columns='product',   # Columns
                       aggfunc='sum')       # Aggregation function

#          product    A    B
# date                      
# 2023-01          100  150
# 2023-02          120  180
```

### melt() (unpivot)
```python
# Wide to long format
df_long = pd.melt(pivot.reset_index(),
                  id_vars=['date'],
                  var_name='product',
                  value_name='sales')
```

---

## Question 6

**How do you apply conditional logic using where()?**

### where() Method
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

# Keep values where condition is True, else replace
df['A'].where(df['A'] > 2, other=0)  # [0, 0, 3, 4, 5]

# np.where for if-else logic
df['category'] = np.where(df['A'] > 2, 'high', 'low')
# ['low', 'low', 'high', 'high', 'high']

# Multiple conditions with np.select
conditions = [
    df['A'] <= 2,
    df['A'] <= 4,
    df['A'] > 4
]
choices = ['low', 'medium', 'high']
df['category'] = np.select(conditions, choices)
```

---

## Question 7

**How do you reshape a DataFrame using stack and unstack?**

### stack() and unstack()
```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
}, index=['row1', 'row2'])

# stack(): columns → rows (wide to long)
stacked = df.stack()
# row1  A    1
#       B    3
# row2  A    2
#       B    4

# unstack(): rows → columns (long to wide)
unstacked = stacked.unstack()
# Back to original shape
```

### Use Case
```python
# MultiIndex DataFrame
df = pd.DataFrame({
    ('Sales', 'Q1'): [100, 200],
    ('Sales', 'Q2'): [150, 250]
}, index=['Product A', 'Product B'])

df.stack(level=1)  # Move quarter level to rows
```

---

## Question 8

**How do you perform statistical aggregation on DataFrame groups?**

### groupby() Aggregation
```python
import pandas as pd

df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'value': [10, 20, 30, 40],
    'count': [1, 2, 3, 4]
})

# Single aggregation
df.groupby('category')['value'].sum()

# Multiple aggregations
df.groupby('category').agg({
    'value': ['sum', 'mean', 'max'],
    'count': 'sum'
})

# Named aggregations (cleaner)
df.groupby('category').agg(
    total_value=('value', 'sum'),
    avg_value=('value', 'mean'),
    total_count=('count', 'sum')
)
```

---

## Question 9

**How do you use window functions for running calculations?**

### Rolling Window
```python
import pandas as pd

df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

# Rolling average (window of 3)
df['rolling_avg'] = df['value'].rolling(window=3).mean()

# Expanding window (cumulative)
df['cumulative_avg'] = df['value'].expanding().mean()

# Exponential weighted moving average
df['ewm'] = df['value'].ewm(span=3).mean()

# Rolling with min_periods
df['rolling_avg'] = df['value'].rolling(window=3, min_periods=1).mean()
```

---

## Question 10

**How do you normalize data within a DataFrame column?**

### Normalization Methods
```python
import pandas as pd

df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})

# 1. Z-score normalization (standardization)
df['z_score'] = (df['value'] - df['value'].mean()) / df['value'].std()

# 2. Min-Max normalization (scale to 0-1)
df['min_max'] = (df['value'] - df['value'].min()) / (df['value'].max() - df['value'].min())

# 3. Max normalization (scale to 0-max)
df['max_norm'] = df['value'] / df['value'].max()

# 4. Group-wise normalization
df['group_norm'] = df.groupby('category')['value'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

---

## Question 11

**How do you create simple plots from a DataFrame?**

### Pandas Visualization
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr'],
    'sales': [100, 150, 120, 180],
    'profit': [20, 35, 25, 45]
})

# Line plot
df.plot(x='month', y='sales', kind='line')

# Bar plot
df.plot(x='month', y='sales', kind='bar')

# Multiple columns
df.plot(x='month', y=['sales', 'profit'], kind='bar')

# Histogram
df['sales'].hist(bins=10)

# Box plot
df[['sales', 'profit']].boxplot()

# Scatter plot
df.plot.scatter(x='sales', y='profit')

plt.show()
```

### Quick Statistics Visualization
```python
# Distribution of all numeric columns
df.hist(figsize=(10, 6))

# Correlation heatmap
import seaborn as sns
sns.heatmap(df.corr(), annot=True)
```

