# Pandas Interview Questions - Scenario-Based Questions

## Scenario 1

**Discuss the use of groupby in Pandas and provide an example.**

### Concept
`groupby()` splits data into groups based on criteria, applies a function to each group, and combines results. It follows the "split-apply-combine" pattern.

### Code Example
```python
import pandas as pd

# Sample sales data
df = pd.DataFrame({
    'region': ['East', 'East', 'West', 'West', 'North', 'North'],
    'product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'sales': [100, 150, 200, 250, 120, 180],
    'quantity': [10, 15, 20, 25, 12, 18]
})

# Basic groupby
print("--- Sales by Region ---")
print(df.groupby('region')['sales'].sum())

# Multiple aggregations
print("\n--- Multiple Stats by Region ---")
print(df.groupby('region').agg({
    'sales': ['sum', 'mean', 'max'],
    'quantity': 'sum'
}))

# Groupby multiple columns
print("\n--- Sales by Region and Product ---")
print(df.groupby(['region', 'product'])['sales'].sum().unstack())

# Transform: apply function but keep original shape
df['sales_pct_of_region'] = df.groupby('region')['sales'].transform(
    lambda x: x / x.sum() * 100
)

# Filter groups
high_sales_regions = df.groupby('region').filter(lambda x: x['sales'].sum() > 300)
```

### Common Use Cases
| Method | Purpose |
|--------|---------|
| `.sum()`, `.mean()` | Aggregate values |
| `.agg()` | Multiple aggregations |
| `.transform()` | Keep original shape |
| `.filter()` | Filter groups by condition |
| `.apply()` | Custom group operations |

---

## Scenario 2

**Discuss how to deal with time series data in Pandas.**

### Key Operations

```python
import pandas as pd
import numpy as np

# Create time series
dates = pd.date_range('2023-01-01', periods=365, freq='D')
df = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(365).cumsum() + 100
})
df.set_index('date', inplace=True)

# 1. Resampling (change frequency)
monthly = df.resample('M').agg({
    'value': ['first', 'last', 'mean', 'std']
})
weekly = df.resample('W').mean()

# 2. Rolling calculations
df['rolling_7d'] = df['value'].rolling(window=7).mean()
df['rolling_30d'] = df['value'].rolling(window=30).mean()

# 3. Shift (lag/lead values)
df['prev_day'] = df['value'].shift(1)   # Yesterday's value
df['daily_change'] = df['value'] - df['value'].shift(1)
df['pct_change'] = df['value'].pct_change()

# 4. Date component extraction
df['year'] = df.index.year
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = df.index.dayofweek >= 5

# 5. Filtering by date
jan_data = df['2023-01']  # All January data
q1_data = df['2023-01':'2023-03']  # Q1 data

# 6. Business day operations
business_days = pd.bdate_range('2023-01-01', '2023-12-31')
```

### Time Series Best Practices
- Always convert to datetime: `pd.to_datetime()`
- Set datetime as index for efficient slicing
- Use `resample()` instead of `groupby()` for time-based grouping
- Handle timezone if applicable: `df.index = df.index.tz_localize('UTC')`

---

## Scenario 3

**Discuss how Pandas integrates with Matplotlib and Seaborn for visualization.**

### Pandas + Matplotlib
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'sales': [100, 150, 120, 180, 200],
    'profit': [20, 30, 24, 36, 40]
})

# Pandas built-in plotting (uses Matplotlib backend)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Line plot
df.plot(x='month', y=['sales', 'profit'], ax=axes[0, 0], title='Line Plot')

# Bar plot
df.plot(x='month', y='sales', kind='bar', ax=axes[0, 1], title='Bar Plot')

# Area plot
df.plot.area(x='month', y=['sales', 'profit'], ax=axes[1, 0], title='Area Plot')

# Pie chart
df.set_index('month')['sales'].plot.pie(ax=axes[1, 1], title='Pie Chart')

plt.tight_layout()
plt.show()
```

### Pandas + Seaborn
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'C', 'C'] * 50,
    'value': np.random.randn(300),
    'size': np.random.randint(10, 100, 300)
})

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Box plot
sns.boxplot(data=df, x='category', y='value', ax=axes[0, 0])

# Violin plot
sns.violinplot(data=df, x='category', y='value', ax=axes[0, 1])

# Histogram with KDE
sns.histplot(data=df, x='value', hue='category', kde=True, ax=axes[1, 0])

# Correlation heatmap
numeric_df = df.pivot_table(index='category', values=['value', 'size'], aggfunc='mean')
sns.heatmap(numeric_df.T.corr(), annot=True, ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

---

## Scenario 4

**How would you use Pandas to prepare and clean e-commerce sales data for insights into customer purchasing patterns?**

### Complete Solution
```python
import pandas as pd
import numpy as np

# Sample e-commerce data
np.random.seed(42)
n = 10000

df = pd.DataFrame({
    'order_id': range(1, n+1),
    'customer_id': np.random.randint(1, 1001, n),
    'order_date': pd.date_range('2023-01-01', periods=n, freq='H'),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n),
    'amount': np.random.exponential(50, n),
    'quantity': np.random.randint(1, 5, n)
})

# Add some realistic issues
df.loc[df.sample(frac=0.05).index, 'amount'] = np.nan  # 5% missing
df.loc[0:10, 'amount'] = -50  # Invalid values

# === DATA CLEANING ===

# 1. Handle missing values
df['amount'].fillna(df['amount'].median(), inplace=True)

# 2. Remove invalid values
df = df[df['amount'] > 0]

# 3. Remove duplicates
df.drop_duplicates(subset=['order_id'], inplace=True)

# 4. Parse dates
df['order_date'] = pd.to_datetime(df['order_date'])
df['order_month'] = df['order_date'].dt.to_period('M')
df['day_of_week'] = df['order_date'].dt.day_name()
df['hour'] = df['order_date'].dt.hour

# === CUSTOMER ANALYSIS ===

# RFM Analysis (Recency, Frequency, Monetary)
rfm = df.groupby('customer_id').agg({
    'order_date': lambda x: (df['order_date'].max() - x.max()).days,  # Recency
    'order_id': 'count',  # Frequency
    'amount': 'sum'       # Monetary
}).rename(columns={
    'order_date': 'recency',
    'order_id': 'frequency',
    'amount': 'monetary'
})

# Customer segments based on percentiles
rfm['r_score'] = pd.qcut(rfm['recency'], 4, labels=[4, 3, 2, 1])
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['m_score'] = pd.qcut(rfm['monetary'], 4, labels=[1, 2, 3, 4])

# === SALES INSIGHTS ===

# Monthly sales trend
monthly_sales = df.groupby('order_month').agg({
    'amount': 'sum',
    'order_id': 'count',
    'customer_id': 'nunique'
}).rename(columns={
    'amount': 'total_sales',
    'order_id': 'total_orders',
    'customer_id': 'unique_customers'
})

# Sales by category
category_analysis = df.groupby('product_category').agg({
    'amount': ['sum', 'mean'],
    'order_id': 'count'
})

# Peak shopping hours
hourly_pattern = df.groupby('hour')['amount'].agg(['sum', 'count'])

print("=== RFM Summary ===")
print(rfm.describe())

print("\n=== Monthly Sales ===")
print(monthly_sales.head())

print("\n=== Category Performance ===")
print(category_analysis)
```

---

## Scenario 5

**Discuss the advantages of vectorized operations in Pandas over iteration.**

### Comparison

| Aspect | Iteration (loops) | Vectorized |
|--------|------------------|------------|
| **Speed** | Slow (Python overhead) | Fast (C implementation) |
| **Code** | Verbose | Concise |
| **Memory** | May create copies | Optimized |
| **Readability** | Clear logic | Requires knowledge |

### Performance Demonstration
```python
import pandas as pd
import numpy as np
import time

# Create large DataFrame
n = 1_000_000
df = pd.DataFrame({
    'A': np.random.randn(n),
    'B': np.random.randn(n)
})

# === METHOD 1: For Loop (SLOW) ===
start = time.time()
results_loop = []
for idx, row in df.iterrows():
    results_loop.append(row['A'] + row['B'])
df['C_loop'] = results_loop
print(f"For loop: {time.time() - start:.2f}s")

# === METHOD 2: Apply (MEDIUM) ===
start = time.time()
df['C_apply'] = df.apply(lambda row: row['A'] + row['B'], axis=1)
print(f"Apply: {time.time() - start:.2f}s")

# === METHOD 3: Vectorized (FAST) ===
start = time.time()
df['C_vectorized'] = df['A'] + df['B']
print(f"Vectorized: {time.time() - start:.4f}s")

# Typical results:
# For loop: ~60s
# Apply: ~15s
# Vectorized: ~0.01s (1000x faster!)
```

### Best Practices
```python
# AVOID: Iterating
for i in range(len(df)):
    df.loc[i, 'new'] = df.loc[i, 'A'] * 2

# PREFER: Vectorized
df['new'] = df['A'] * 2

# AVOID: Apply for simple operations
df['new'] = df['A'].apply(lambda x: x * 2)

# PREFER: Built-in operations
df['new'] = df['A'] * 2

# When you NEED apply (complex row logic):
def complex_logic(row):
    if row['A'] > 0 and row['B'] < 0:
        return row['A'] * row['B']
    elif row['A'] < 0:
        return row['A'] + row['B']
    else:
        return 0

# Use np.where or np.select when possible
df['new'] = np.where(df['A'] > 0, df['A'] * 2, df['A'])

# For multiple conditions:
conditions = [df['A'] > 0, df['A'] < 0]
choices = [df['A'] * 2, df['A'] / 2]
df['new'] = np.select(conditions, choices, default=0)
```

### Key Takeaways
1. **Avoid `iterrows()`** - Use only for debugging
2. **Avoid `apply()`** for simple operations - Use vectorized alternatives
3. **Use `np.where()`** for conditional operations
4. **Use `np.select()`** for multiple conditions
5. **Use built-in Pandas/NumPy functions** whenever possible
