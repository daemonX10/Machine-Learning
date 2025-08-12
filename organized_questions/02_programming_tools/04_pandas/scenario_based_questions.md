# Pandas Interview Questions - Scenario_Based Questions

## Question 1

**Discuss the use of groupby in Pandas and provide an example.**

### Answer

#### Theory
The `groupby` operation in Pandas is one of the most powerful data manipulation tools, implementing the "split-apply-combine" paradigm. It allows you to split data into groups based on criteria, apply functions to each group independently, and combine the results. This operation is essential for data aggregation, statistical analysis, and exploratory data analysis, enabling sophisticated data transformations and insights across different categories or segments of your dataset.

#### Code Example

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Callable
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ======================== BASIC GROUPBY OPERATIONS ========================

def demonstrate_basic_groupby():
    """Demonstrate fundamental groupby operations."""
    
    print("=== BASIC GROUPBY OPERATIONS ===")
    
    # Create comprehensive sales dataset
    df = create_comprehensive_sales_dataset()
    print("Sales Dataset:")
    print(df.head(10))
    print(f"Dataset shape: {df.shape}")
    
    # 1. Simple groupby with single column
    print(f"\n1. SINGLE COLUMN GROUPBY:")
    
    # Group by region and calculate basic statistics
    region_stats = df.groupby('region').agg({
        'sales': ['count', 'sum', 'mean', 'std'],
        'profit': ['sum', 'mean'],
        'quantity': 'sum'
    })
    
    print("Sales statistics by region:")
    print(region_stats.round(2))
    
    # 2. Multiple column groupby
    print(f"\n2. MULTIPLE COLUMN GROUPBY:")
    
    # Group by region and product category
    multi_group = df.groupby(['region', 'product_category']).agg({
        'sales': 'sum',
        'profit': 'mean',
        'quantity': 'sum'
    }).round(2)
    
    print("Sales by region and product category:")
    print(multi_group.head(10))
    
    # 3. Groupby with custom aggregation functions
    print(f"\n3. CUSTOM AGGREGATION FUNCTIONS:")
    
    def coefficient_of_variation(series):
        """Calculate coefficient of variation."""
        return series.std() / series.mean() if series.mean() != 0 else 0
    
    def profit_margin(group):
        """Calculate profit margin."""
        return (group['profit'].sum() / group['sales'].sum() * 100) if group['sales'].sum() != 0 else 0
    
    custom_stats = df.groupby('region').agg({
        'sales': ['mean', coefficient_of_variation],
        'profit': 'sum'
    })
    
    # Add profit margin calculation
    profit_margins = df.groupby('region').apply(profit_margin)
    
    print("Custom statistics by region:")
    print(custom_stats.round(3))
    print(f"\nProfit margins by region:")
    print(profit_margins.round(2))
    
    return df

def demonstrate_advanced_groupby():
    """Demonstrate advanced groupby techniques."""
    
    print(f"\n=== ADVANCED GROUPBY TECHNIQUES ===")
    
    df = create_comprehensive_sales_dataset()
    
    # 1. Transform operations
    print(f"\n1. TRANSFORM OPERATIONS:")
    
    # Add group-based features
    df['region_avg_sales'] = df.groupby('region')['sales'].transform('mean')
    df['region_total_sales'] = df.groupby('region')['sales'].transform('sum')
    df['sales_vs_region_avg'] = df['sales'] - df['region_avg_sales']
    df['pct_of_region_sales'] = (df['sales'] / df['region_total_sales'] * 100).round(2)
    
    print("Data with group-based features:")
    print(df[['region', 'sales', 'region_avg_sales', 'sales_vs_region_avg', 'pct_of_region_sales']].head())
    
    # 2. Ranking within groups
    print(f"\n2. RANKING WITHIN GROUPS:")
    
    df['sales_rank'] = df.groupby('region')['sales'].rank(method='dense', ascending=False)
    df['profit_rank'] = df.groupby('region')['profit'].rank(method='dense', ascending=False)
    
    # Find top performers in each region
    top_performers = df[df['sales_rank'] <= 3].sort_values(['region', 'sales_rank'])
    print("Top 3 sales performers by region:")
    print(top_performers[['region', 'salesperson', 'sales', 'sales_rank']].head(15))
    
    # 3. Rolling operations within groups
    print(f"\n3. ROLLING OPERATIONS WITHIN GROUPS:")
    
    # Sort by date within each region for time-based analysis
    df_sorted = df.sort_values(['region', 'date'])
    
    # Calculate rolling averages within each region
    df_sorted['rolling_7d_sales'] = df_sorted.groupby('region')['sales'].rolling(
        window=7, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    df_sorted['rolling_7d_profit'] = df_sorted.groupby('region')['profit'].rolling(
        window=7, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    print("Rolling averages within regions:")
    print(df_sorted[['region', 'date', 'sales', 'rolling_7d_sales', 'profit', 'rolling_7d_profit']].head())
    
    # 4. Filter operations
    print(f"\n4. FILTER OPERATIONS:")
    
    # Filter groups by group characteristics
    high_volume_regions = df.groupby('region').filter(lambda x: x['sales'].sum() > 50000)
    print(f"High-volume regions (sales > 50k):")
    print(f"Regions: {high_volume_regions['region'].unique()}")
    print(f"Records: {len(high_volume_regions)}")
    
    # Filter by group size
    large_teams = df.groupby('region').filter(lambda x: x['salesperson'].nunique() >= 5)
    print(f"\nRegions with large sales teams (>= 5 people):")
    print(f"Regions: {large_teams['region'].unique()}")
    
    return df_sorted

def demonstrate_business_scenarios():
    """Demonstrate real-world business scenarios using groupby."""
    
    print(f"\n=== BUSINESS SCENARIO APPLICATIONS ===")
    
    df = create_comprehensive_sales_dataset()
    
    # 1. Sales Performance Analysis
    print(f"\n1. SALES PERFORMANCE ANALYSIS:")
    
    # Regional performance comparison
    regional_performance = df.groupby('region').agg({
        'sales': ['sum', 'mean', 'count'],
        'profit': 'sum',
        'quantity': 'sum',
        'salesperson': 'nunique'
    })
    
    # Flatten column names
    regional_performance.columns = ['_'.join(col).strip() for col in regional_performance.columns]
    
    # Calculate derived metrics
    regional_performance['avg_deal_size'] = (
        regional_performance['sales_sum'] / regional_performance['sales_count']
    ).round(2)
    
    regional_performance['profit_margin'] = (
        regional_performance['profit_sum'] / regional_performance['sales_sum'] * 100
    ).round(2)
    
    regional_performance['sales_per_person'] = (
        regional_performance['sales_sum'] / regional_performance['salesperson_nunique']
    ).round(2)
    
    print("Regional Performance Metrics:")
    print(regional_performance)
    
    # 2. Product Category Analysis
    print(f"\n2. PRODUCT CATEGORY ANALYSIS:")
    
    category_analysis = df.groupby('product_category').agg({
        'sales': ['sum', 'count'],
        'profit': 'sum',
        'quantity': 'sum'
    })
    
    category_analysis.columns = ['_'.join(col).strip() for col in category_analysis.columns]
    
    # Calculate market share
    total_sales = df['sales'].sum()
    category_analysis['market_share'] = (
        category_analysis['sales_sum'] / total_sales * 100
    ).round(2)
    
    # Sort by market share
    category_analysis = category_analysis.sort_values('market_share', ascending=False)
    
    print("Product Category Analysis:")
    print(category_analysis)
    
    # 3. Time-based Analysis
    print(f"\n3. TIME-BASED ANALYSIS:")
    
    # Add time-based grouping columns
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Monthly trends
    monthly_trends = df.groupby('month').agg({
        'sales': 'sum',
        'profit': 'sum',
        'quantity': 'sum'
    }).round(2)
    
    print("Monthly Sales Trends:")
    print(monthly_trends)
    
    # Day of week patterns
    dow_patterns = df.groupby('day_of_week').agg({
        'sales': 'mean',
        'quantity': 'mean'
    }).round(2)
    
    dow_patterns.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    print("\nDay of Week Patterns (Average):")
    print(dow_patterns)
    
    # 4. Customer Segmentation
    print(f"\n4. CUSTOMER SEGMENTATION:")
    
    # Group by customer and calculate customer metrics
    customer_metrics = df.groupby('customer_id').agg({
        'sales': ['sum', 'count', 'mean'],
        'profit': 'sum',
        'date': ['min', 'max']
    })
    
    customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns]
    
    # Calculate customer lifetime and frequency
    customer_metrics['days_active'] = (
        customer_metrics['date_max'] - customer_metrics['date_min']
    ).dt.days + 1
    
    customer_metrics['purchase_frequency'] = (
        customer_metrics['sales_count'] / customer_metrics['days_active'] * 30
    ).round(2)  # Purchases per month
    
    # Customer segmentation based on total value
    def categorize_customer(total_sales):
        if total_sales >= 5000:
            return 'High Value'
        elif total_sales >= 2000:
            return 'Medium Value'
        else:
            return 'Low Value'
    
    customer_metrics['segment'] = customer_metrics['sales_sum'].apply(categorize_customer)
    
    # Segment analysis
    segment_analysis = customer_metrics.groupby('segment').agg({
        'sales_sum': ['count', 'mean', 'sum'],
        'purchase_frequency': 'mean'
    }).round(2)
    
    print("Customer Segment Analysis:")
    print(segment_analysis)
    
    return df, customer_metrics

# ======================== GROUPBY UTILITY CLASS ========================

class GroupByAnalyzer:
    """Advanced groupby analysis utility for business intelligence."""
    
    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with DataFrame."""
        self.df = dataframe.copy()
        self.analysis_results = {}
    
    def perform_comprehensive_analysis(self, group_columns: Union[str, List[str]], 
                                     value_columns: Optional[List[str]] = None) -> Dict:
        """Perform comprehensive groupby analysis."""
        
        if isinstance(group_columns, str):
            group_columns = [group_columns]
        
        if value_columns is None:
            value_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove group columns from value columns
        value_columns = [col for col in value_columns if col not in group_columns]
        
        results = {}
        
        # 1. Basic statistics
        basic_stats = self.df.groupby(group_columns)[value_columns].agg([
            'count', 'sum', 'mean', 'median', 'std', 'min', 'max'
        ])
        results['basic_statistics'] = basic_stats
        
        # 2. Group sizes
        group_sizes = self.df.groupby(group_columns).size()
        results['group_sizes'] = group_sizes
        
        # 3. Group composition
        for col in value_columns:
            if col in self.df.columns:
                # Calculate percentiles
                percentiles = self.df.groupby(group_columns)[col].quantile([0.25, 0.5, 0.75]).unstack()
                results[f'{col}_percentiles'] = percentiles
        
        # 4. Relative metrics
        if len(value_columns) > 0:
            totals = self.df[value_columns].sum()
            group_totals = self.df.groupby(group_columns)[value_columns].sum()
            
            relative_shares = {}
            for col in value_columns:
                if totals[col] != 0:
                    relative_shares[col] = (group_totals[col] / totals[col] * 100).round(2)
            
            results['relative_shares'] = pd.DataFrame(relative_shares)
        
        self.analysis_results[str(group_columns)] = results
        return results
    
    def identify_outliers_by_group(self, group_columns: Union[str, List[str]], 
                                  value_column: str, method: str = 'iqr') -> pd.DataFrame:
        """Identify outliers within each group."""
        
        def detect_outliers_iqr(group):
            Q1 = group[value_column].quantile(0.25)
            Q3 = group[value_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            group['is_outlier'] = (group[value_column] < lower_bound) | (group[value_column] > upper_bound)
            group['outlier_type'] = 'normal'
            group.loc[group[value_column] < lower_bound, 'outlier_type'] = 'low_outlier'
            group.loc[group[value_column] > upper_bound, 'outlier_type'] = 'high_outlier'
            
            return group
        
        def detect_outliers_zscore(group):
            mean_val = group[value_column].mean()
            std_val = group[value_column].std()
            
            if std_val == 0:
                group['is_outlier'] = False
                group['outlier_type'] = 'normal'
            else:
                z_scores = np.abs((group[value_column] - mean_val) / std_val)
                group['is_outlier'] = z_scores > 3
                group['outlier_type'] = 'normal'
                group.loc[group['is_outlier'], 'outlier_type'] = 'z_outlier'
            
            return group
        
        if method == 'iqr':
            result = self.df.groupby(group_columns).apply(detect_outliers_iqr).reset_index(drop=True)
        elif method == 'zscore':
            result = self.df.groupby(group_columns).apply(detect_outliers_zscore).reset_index(drop=True)
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        return result
    
    def trend_analysis_by_group(self, group_columns: Union[str, List[str]], 
                               time_column: str, value_column: str) -> pd.DataFrame:
        """Perform trend analysis within each group."""
        
        def calculate_trend(group):
            # Sort by time
            group_sorted = group.sort_values(time_column)
            
            # Calculate trend metrics
            x = np.arange(len(group_sorted))
            y = group_sorted[value_column].values
            
            if len(y) > 1:
                # Linear regression
                slope, intercept = np.polyfit(x, y, 1)
                
                # R-squared
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                return pd.Series({
                    'trend_slope': slope,
                    'trend_intercept': intercept,
                    'r_squared': r_squared,
                    'start_value': y[0],
                    'end_value': y[-1],
                    'total_change': y[-1] - y[0],
                    'percent_change': ((y[-1] - y[0]) / y[0] * 100) if y[0] != 0 else 0
                })
            else:
                return pd.Series({
                    'trend_slope': 0,
                    'trend_intercept': 0,
                    'r_squared': 0,
                    'start_value': y[0] if len(y) > 0 else 0,
                    'end_value': y[0] if len(y) > 0 else 0,
                    'total_change': 0,
                    'percent_change': 0
                })
        
        trends = self.df.groupby(group_columns).apply(calculate_trend)
        return trends
    
    def generate_insights_report(self, group_columns: Union[str, List[str]]) -> Dict:
        """Generate automated insights from groupby analysis."""
        
        if str(group_columns) not in self.analysis_results:
            self.perform_comprehensive_analysis(group_columns)
        
        results = self.analysis_results[str(group_columns)]
        insights = {}
        
        # Group size insights
        group_sizes = results['group_sizes']
        insights['group_size_insights'] = {
            'largest_group': group_sizes.idxmax(),
            'smallest_group': group_sizes.idxmin(),
            'avg_group_size': group_sizes.mean(),
            'group_size_variation': group_sizes.std() / group_sizes.mean() if group_sizes.mean() != 0 else 0
        }
        
        # Performance insights
        if 'basic_statistics' in results:
            basic_stats = results['basic_statistics']
            
            # Find top and bottom performers for each metric
            for col in basic_stats.columns.levels[0]:  # First level of MultiIndex
                if 'sum' in basic_stats[col].columns:
                    top_performer = basic_stats[col]['sum'].idxmax()
                    bottom_performer = basic_stats[col]['sum'].idxmin()
                    
                    insights[f'{col}_performance'] = {
                        'top_performer': top_performer,
                        'bottom_performer': bottom_performer,
                        'performance_gap': basic_stats[col]['sum'].max() - basic_stats[col]['sum'].min()
                    }
        
        return insights

# ======================== HELPER FUNCTIONS ========================

def create_comprehensive_sales_dataset():
    """Create comprehensive sales dataset for groupby demonstrations."""
    
    np.random.seed(42)
    
    # Define data dimensions
    regions = ['North', 'South', 'East', 'West']
    product_categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']
    salespeople = [f'Sales_{i}' for i in range(1, 21)]
    
    # Generate date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    # Generate data
    n_records = 2000
    
    data = {
        'date': np.random.choice(date_range, n_records),
        'region': np.random.choice(regions, n_records),
        'product_category': np.random.choice(product_categories, n_records),
        'salesperson': np.random.choice(salespeople, n_records),
        'customer_id': np.random.randint(1, 501, n_records),  # 500 unique customers
        'sales': np.random.exponential(1000, n_records).round(2),
        'quantity': np.random.poisson(5, n_records),
        'unit_price': np.random.normal(50, 15, n_records).round(2)
    }
    
    # Ensure positive prices
    data['unit_price'] = np.abs(data['unit_price'])
    
    # Calculate derived fields
    df = pd.DataFrame(data)
    df['total_revenue'] = df['quantity'] * df['unit_price']
    df['profit'] = df['sales'] * np.random.uniform(0.1, 0.3, n_records)  # 10-30% profit margin
    
    return df

# ======================== MAIN DEMONSTRATION ========================

def run_comprehensive_groupby_demo():
    """Run comprehensive groupby demonstration."""
    
    print("PANDAS GROUPBY COMPREHENSIVE GUIDE")
    print("="*35)
    
    # Basic groupby operations
    df_basic = demonstrate_basic_groupby()
    
    # Advanced groupby techniques
    df_advanced = demonstrate_advanced_groupby()
    
    # Business scenarios
    df_business, customer_metrics = demonstrate_business_scenarios()
    
    # Demonstrate GroupByAnalyzer class
    print(f"\n=== GROUPBY ANALYZER CLASS ===")
    
    analyzer = GroupByAnalyzer(df_business)
    
    # Comprehensive analysis
    results = analyzer.perform_comprehensive_analysis(['region'], ['sales', 'profit'])
    print("Group sizes by region:")
    print(results['group_sizes'])
    
    # Outlier detection
    outliers = analyzer.identify_outliers_by_group(['region'], 'sales', method='iqr')
    outlier_count = outliers['is_outlier'].sum()
    print(f"\nOutliers detected: {outlier_count}")
    
    # Generate insights
    insights = analyzer.generate_insights_report(['region'])
    print(f"\nKey Insights:")
    print(f"Largest region: {insights['group_size_insights']['largest_group']}")
    print(f"Top sales performer: {insights.get('sales_performance', {}).get('top_performer', 'N/A')}")

# Execute demonstration
if __name__ == "__main__":
    run_comprehensive_groupby_demo()
```

#### Explanation
1. **Split-Apply-Combine**: Groupby splits data into groups, applies functions, and combines results
2. **Aggregation Functions**: Built-in and custom functions for statistical analysis
3. **Transform Operations**: Add group-level statistics back to original DataFrame
4. **Filter Operations**: Select groups based on group-level conditions
5. **Business Applications**: Real-world scenarios for sales analysis and customer segmentation

#### Use Cases
- **Sales Analysis**: Performance by region, product, or salesperson
- **Customer Segmentation**: Grouping customers by behavior or value
- **Financial Reporting**: Aggregating financial metrics by departments or time periods
- **A/B Testing**: Comparing treatment groups in experiments
- **Time Series Analysis**: Analyzing trends within different categories

#### Best Practices
- **Choose Appropriate Aggregation**: Select functions that match your analytical goals
- **Handle Missing Data**: Decide how to treat NaN values in groups
- **Use Transform Wisely**: Transform operations preserve DataFrame structure
- **Optimize Performance**: Sort data and use categorical types for better performance
- **Meaningful Grouping**: Ensure grouping variables make business sense

#### Pitfalls
- **Memory Usage**: Large groupby operations can consume significant memory
- **Lost Index Information**: Some operations may lose important index data
- **Hierarchical Columns**: Multi-level column names can be confusing
- **Empty Groups**: Some combinations may result in empty groups
- **Performance Issues**: Complex custom functions can be slow

#### Debugging
```python
def debug_groupby_operation(df: pd.DataFrame, group_cols: List[str]):
    """Debug groupby operations."""
    
    print("GroupBy Debug Information:")
    print(f"DataFrame shape: {df.shape}")
    print(f"Group columns: {group_cols}")
    
    # Check group formation
    groups = df.groupby(group_cols)
    print(f"Number of groups: {groups.ngroups}")
    
    # Group sizes
    group_sizes = groups.size()
    print(f"Group sizes - Min: {group_sizes.min()}, Max: {group_sizes.max()}, Mean: {group_sizes.mean():.1f}")
    
    # Check for empty groups
    empty_groups = group_sizes[group_sizes == 0]
    if len(empty_groups) > 0:
        print(f"Warning: {len(empty_groups)} empty groups found")
    
    # Memory usage estimate
    print(f"Estimated memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

#### Optimization

**GroupBy Performance Tips:**

| Scenario | Optimization Strategy |
|----------|----------------------|
| **Large DataFrames** | Use categorical data types for group columns |
| **Many Groups** | Consider chunking or sampling for exploration |
| **Complex Aggregations** | Use built-in functions over custom functions when possible |
| **Memory Constraints** | Process groups sequentially or use Dask for out-of-core |
| **Repeated Operations** | Cache groupby objects for reuse |

**Memory and Performance:**
- Convert string columns to categorical before grouping
- Sort data by group columns for better performance
- Use appropriate data types (int32 vs int64, float32 vs float64)
- Consider using `.agg()` with dictionaries for efficiency

---

## Question 2

**Discuss how to deal withtime series datainPandas.**

### Answer

#### Theory
Time series data analysis in Pandas involves handling data points indexed by time. Pandas provides comprehensive tools for working with temporal data including parsing dates, resampling, time zone handling, rolling windows, and frequency analysis. The key concepts include DatetimeIndex, period operations, time-based slicing, and temporal aggregations. Effective time series analysis requires understanding how to clean, transform, and analyze temporal patterns in data for forecasting, trend analysis, and seasonal decomposition.

#### Code Example

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ======================== TIME SERIES DATA CREATION ========================

def create_time_series_dataset():
    """Create comprehensive time series dataset for analysis."""
    
    # Generate date range
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)
    n_days = len(date_range)
    
    # Create base trend
    trend = np.linspace(100, 200, n_days)
    
    # Add seasonality (yearly and weekly)
    yearly_seasonality = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Add noise
    noise = np.random.normal(0, 15, n_days)
    
    # Combine components
    values = trend + yearly_seasonality + weekly_seasonality + noise
    
    # Add some missing values randomly
    missing_indices = np.random.choice(n_days, size=int(n_days * 0.02), replace=False)
    values[missing_indices] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'value': values,
        'category': np.random.choice(['A', 'B', 'C'], n_days),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_days)
    })
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    # Add additional time series with different patterns
    df['volume'] = np.random.poisson(50, n_days)
    df['price'] = 10 + 5 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + np.random.normal(0, 2, n_days)
    df['revenue'] = df['value'] * df['price'] * (1 + np.random.normal(0, 0.1, n_days))
    
    return df

def demonstrate_datetime_operations():
    """Demonstrate basic datetime operations."""
    
    print("=== DATETIME OPERATIONS ===")
    
    df = create_time_series_dataset()
    print(f"Time series dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Index type: {type(df.index)}")
    
    # 1. Date component extraction
    print(f"\n1. DATE COMPONENT EXTRACTION:")
    
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['quarter'] = df.index.quarter
    df['is_weekend'] = df.index.dayofweek >= 5
    
    print("Date components sample:")
    print(df[['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend']].head())
    
    # 2. Time-based indexing and slicing
    print(f"\n2. TIME-BASED INDEXING:")
    
    # Slice by year
    data_2022 = df['2022']
    print(f"2022 data shape: {data_2022.shape}")
    
    # Slice by month
    jan_2022 = df['2022-01']
    print(f"January 2022 shape: {jan_2022.shape}")
    
    # Slice by date range
    q1_2022 = df['2022-01-01':'2022-03-31']
    print(f"Q1 2022 shape: {q1_2022.shape}")
    
    # Boolean indexing with dates
    weekends = df[df['is_weekend']]
    print(f"Weekend data points: {len(weekends)}")
    
    # 3. Date filtering and selection
    print(f"\n3. DATE FILTERING:")
    
    # Filter by specific conditions
    recent_data = df[df.index >= '2023-01-01']
    print(f"Recent data (2023+) shape: {recent_data.shape}")
    
    # Filter by day of week
    mondays = df[df.index.dayofweek == 0]
    print(f"Monday data points: {len(mondays)}")
    
    # Filter by month
    summer_months = df[df.index.month.isin([6, 7, 8])]
    print(f"Summer months data: {len(summer_months)}")
    
    return df

def demonstrate_resampling_operations():
    """Demonstrate resampling and frequency conversion."""
    
    print(f"\n=== RESAMPLING OPERATIONS ===")
    
    df = create_time_series_dataset()
    
    # 1. Basic resampling
    print(f"\n1. BASIC RESAMPLING:")
    
    # Monthly aggregation
    monthly_agg = df.resample('M').agg({
        'value': ['mean', 'sum', 'std'],
        'volume': 'sum',
        'price': 'mean',
        'revenue': 'sum'
    })
    
    print("Monthly aggregation sample:")
    print(monthly_agg.head())
    
    # Weekly aggregation
    weekly_agg = df.resample('W').agg({
        'value': 'mean',
        'volume': 'sum',
        'revenue': 'sum'
    })
    
    print(f"\nWeekly aggregation shape: {weekly_agg.shape}")
    
    # 2. Different resampling frequencies
    print(f"\n2. DIFFERENT FREQUENCIES:")
    
    # Quarterly
    quarterly = df.resample('Q')['value'].agg(['mean', 'min', 'max'])
    print("Quarterly statistics:")
    print(quarterly.head())
    
    # Business days only
    business_daily = df.resample('B')['value'].mean()
    print(f"Business days data points: {len(business_daily)}")
    
    # 3. Forward fill and backward fill
    print(f"\n3. FILLING STRATEGIES:")
    
    # Handle missing values during resampling
    daily_filled = df.resample('D').agg({
        'value': 'mean',
        'volume': 'sum'
    })
    
    # Forward fill
    daily_ffill = daily_filled.fillna(method='ffill')
    
    # Backward fill
    daily_bfill = daily_filled.fillna(method='bfill')
    
    # Interpolation
    daily_interpolated = daily_filled.interpolate(method='linear')
    
    print("Filling strategies comparison:")
    missing_dates = daily_filled[daily_filled.isnull().any(axis=1)]
    if not missing_dates.empty:
        print(f"Missing data points: {len(missing_dates)}")
    
    # 4. Custom resampling functions
    print(f"\n4. CUSTOM RESAMPLING:")
    
    def weighted_average(series):
        """Calculate weighted average."""
        weights = np.arange(1, len(series) + 1)
        return np.average(series.dropna(), weights=weights[:len(series.dropna())])
    
    def volatility(series):
        """Calculate volatility (standard deviation)."""
        return series.std()
    
    custom_resample = df.resample('M').agg({
        'value': [weighted_average, volatility],
        'price': ['mean', volatility]
    })
    
    print("Custom resampling functions:")
    print(custom_resample.head())
    
    return monthly_agg, weekly_agg

def demonstrate_rolling_operations():
    """Demonstrate rolling window operations."""
    
    print(f"\n=== ROLLING WINDOW OPERATIONS ===")
    
    df = create_time_series_dataset()
    
    # 1. Basic rolling windows
    print(f"\n1. BASIC ROLLING WINDOWS:")
    
    # Simple moving averages
    df['sma_7'] = df['value'].rolling(window=7).mean()
    df['sma_30'] = df['value'].rolling(window=30).mean()
    df['sma_90'] = df['value'].rolling(window=90).mean()
    
    # Rolling standard deviation
    df['volatility_30'] = df['value'].rolling(window=30).std()
    
    print("Rolling statistics sample:")
    print(df[['value', 'sma_7', 'sma_30', 'volatility_30']].head(10))
    
    # 2. Different rolling functions
    print(f"\n2. ROLLING FUNCTIONS:")
    
    # Rolling min and max
    df['rolling_min_30'] = df['value'].rolling(window=30).min()
    df['rolling_max_30'] = df['value'].rolling(window=30).max()
    
    # Rolling quantiles
    df['rolling_median'] = df['value'].rolling(window=30).median()
    df['rolling_q25'] = df['value'].rolling(window=30).quantile(0.25)
    df['rolling_q75'] = df['value'].rolling(window=30).quantile(0.75)
    
    # Rolling correlation
    if 'price' in df.columns:
        df['rolling_corr'] = df['value'].rolling(window=30).corr(df['price'])
    
    print("Rolling functions sample:")
    print(df[['value', 'rolling_min_30', 'rolling_max_30', 'rolling_median']].tail())
    
    # 3. Expanding windows
    print(f"\n3. EXPANDING WINDOWS:")
    
    # Expanding mean (cumulative average)
    df['expanding_mean'] = df['value'].expanding().mean()
    df['expanding_std'] = df['value'].expanding().std()
    df['expanding_min'] = df['value'].expanding().min()
    df['expanding_max'] = df['value'].expanding().max()
    
    print("Expanding statistics sample:")
    print(df[['value', 'expanding_mean', 'expanding_std']].tail())
    
    # 4. Exponentially weighted functions
    print(f"\n4. EXPONENTIALLY WEIGHTED:")
    
    # EWM with different span parameters
    df['ewm_10'] = df['value'].ewm(span=10).mean()
    df['ewm_30'] = df['value'].ewm(span=30).mean()
    
    # EWM standard deviation
    df['ewm_std'] = df['value'].ewm(span=30).std()
    
    print("EWM statistics sample:")
    print(df[['value', 'ewm_10', 'ewm_30', 'ewm_std']].tail())
    
    return df

def demonstrate_time_zone_operations():
    """Demonstrate time zone operations."""
    
    print(f"\n=== TIME ZONE OPERATIONS ===")
    
    # Create dataset with timezone-naive dates
    df = create_time_series_dataset()
    
    # 1. Timezone localization
    print(f"\n1. TIMEZONE LOCALIZATION:")
    
    # Localize to UTC
    df_utc = df.copy()
    df_utc.index = df_utc.index.tz_localize('UTC')
    print(f"UTC timezone: {df_utc.index.tz}")
    
    # Localize to specific timezone
    df_ny = df.copy()
    df_ny.index = df_ny.index.tz_localize('America/New_York')
    print(f"New York timezone: {df_ny.index.tz}")
    
    # 2. Timezone conversion
    print(f"\n2. TIMEZONE CONVERSION:")
    
    # Convert UTC to different timezones
    df_london = df_utc.tz_convert('Europe/London')
    df_tokyo = df_utc.tz_convert('Asia/Tokyo')
    df_sydney = df_utc.tz_convert('Australia/Sydney')
    
    print("Timezone conversion sample:")
    sample_date = df_utc.index[100]
    print(f"UTC:     {sample_date}")
    print(f"London:  {df_london.index[100]}")
    print(f"Tokyo:   {df_tokyo.index[100]}")
    print(f"Sydney:  {df_sydney.index[100]}")
    
    # 3. Business day operations
    print(f"\n3. BUSINESS DAY OPERATIONS:")
    
    # Filter business days
    business_days = df[df.index.dayofweek < 5]  # Monday=0, Friday=4
    print(f"Business days count: {len(business_days)}")
    
    # Create business day range
    bdate_range = pd.bdate_range(start='2023-01-01', end='2023-12-31')
    print(f"Business days in 2023: {len(bdate_range)}")
    
    return df_utc, df_london

# ======================== TIME SERIES ANALYSIS CLASS ========================

class TimeSeriesAnalyzer:
    """Comprehensive time series analysis utility."""
    
    def __init__(self, data: pd.DataFrame, date_column: Optional[str] = None):
        """Initialize with time series data."""
        self.df = data.copy()
        
        if date_column:
            self.df[date_column] = pd.to_datetime(self.df[date_column])
            self.df.set_index(date_column, inplace=True)
        
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
        
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
    
    def detect_seasonality(self, column: str, periods: List[int] = [7, 30, 365]) -> Dict:
        """Detect seasonal patterns in the data."""
        
        results = {}
        
        for period in periods:
            if len(self.df) >= period * 2:  # Need at least 2 cycles
                # Group by period
                period_groups = self.df.groupby(self.df.index.dayofyear % period)[column]
                
                # Calculate statistics
                period_means = period_groups.mean()
                period_stds = period_groups.std()
                
                # Calculate coefficient of variation
                cv = period_stds.std() / period_means.mean() if period_means.mean() != 0 else 0
                
                # Seasonality strength (higher values indicate stronger seasonality)
                seasonality_strength = 1 - np.var(period_means) / np.var(self.df[column].dropna())
                
                results[f'period_{period}'] = {
                    'coefficient_of_variation': cv,
                    'seasonality_strength': seasonality_strength,
                    'period_means': period_means,
                    'period_stds': period_stds
                }
        
        return results
    
    def detect_trends(self, column: str, method: str = 'linear') -> Dict:
        """Detect trends in time series data."""
        
        # Remove NaN values
        clean_data = self.df[column].dropna()
        
        if len(clean_data) < 3:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Convert dates to numeric for regression
        x = np.arange(len(clean_data))
        y = clean_data.values
        
        if method == 'linear':
            # Linear trend
            slope, intercept = np.polyfit(x, y, 1)
            trend_line = slope * x + intercept
            
            # R-squared
            ss_res = np.sum((y - trend_line) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'method': 'linear',
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat',
                'trend_strength': abs(slope)
            }
        
        elif method == 'polynomial':
            # Polynomial trend (degree 2)
            coeffs = np.polyfit(x, y, 2)
            trend_line = np.polyval(coeffs, x)
            
            # R-squared
            ss_res = np.sum((y - trend_line) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'method': 'polynomial',
                'coefficients': coeffs,
                'r_squared': r_squared,
                'curvature': coeffs[0]  # Second derivative
            }
    
    def calculate_technical_indicators(self, column: str) -> pd.DataFrame:
        """Calculate common technical indicators."""
        
        result = pd.DataFrame(index=self.df.index)
        series = self.df[column]
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            result[f'SMA_{window}'] = series.rolling(window=window).mean()
        
        # Exponential Moving Averages
        for span in [12, 26]:
            result[f'EMA_{span}'] = series.ewm(span=span).mean()
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = series.ewm(span=12).mean()
        ema_26 = series.ewm(span=26).mean()
        result['MACD'] = ema_12 - ema_26
        result['MACD_signal'] = result['MACD'].ewm(span=9).mean()
        result['MACD_histogram'] = result['MACD'] - result['MACD_signal']
        
        # RSI (Relative Strength Index)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_20 = series.rolling(window=20).mean()
        std_20 = series.rolling(window=20).std()
        result['BB_upper'] = sma_20 + (std_20 * 2)
        result['BB_lower'] = sma_20 - (std_20 * 2)
        result['BB_width'] = result['BB_upper'] - result['BB_lower']
        
        # Volatility
        result['volatility_30'] = series.rolling(window=30).std()
        
        return result
    
    def decompose_time_series(self, column: str, model: str = 'additive', period: int = 365) -> Dict:
        """Decompose time series into trend, seasonal, and residual components."""
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Clean data
            clean_series = self.df[column].dropna()
            
            if len(clean_series) < period * 2:
                return {'error': f'Insufficient data for decomposition. Need at least {period * 2} points.'}
            
            # Perform decomposition
            decomposition = seasonal_decompose(clean_series, model=model, period=period)
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed,
                'model': model,
                'period': period
            }
            
        except ImportError:
            return {'error': 'statsmodels not available for decomposition'}
    
    def detect_anomalies(self, column: str, method: str = 'iqr', window: int = 30) -> pd.DataFrame:
        """Detect anomalies in time series data."""
        
        result = self.df[[column]].copy()
        
        if method == 'iqr':
            # IQR-based anomaly detection
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            result['is_anomaly'] = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
            result['anomaly_type'] = 'normal'
            result.loc[self.df[column] < lower_bound, 'anomaly_type'] = 'low_outlier'
            result.loc[self.df[column] > upper_bound, 'anomaly_type'] = 'high_outlier'
        
        elif method == 'rolling_zscore':
            # Rolling Z-score based detection
            rolling_mean = self.df[column].rolling(window=window).mean()
            rolling_std = self.df[column].rolling(window=window).std()
            z_scores = np.abs((self.df[column] - rolling_mean) / rolling_std)
            
            result['z_score'] = z_scores
            result['is_anomaly'] = z_scores > 3
            result['anomaly_type'] = 'normal'
            result.loc[result['is_anomaly'], 'anomaly_type'] = 'z_outlier'
        
        return result

# ======================== MAIN DEMONSTRATION ========================

def run_comprehensive_time_series_demo():
    """Run comprehensive time series demonstration."""
    
    print("PANDAS TIME SERIES COMPREHENSIVE GUIDE")
    print("="*35)
    
    # Basic datetime operations
    df_datetime = demonstrate_datetime_operations()
    
    # Resampling operations
    monthly_agg, weekly_agg = demonstrate_resampling_operations()
    
    # Rolling operations
    df_rolling = demonstrate_rolling_operations()
    
    # Timezone operations
    df_utc, df_london = demonstrate_time_zone_operations()
    
    # Demonstrate TimeSeriesAnalyzer class
    print(f"\n=== TIME SERIES ANALYZER CLASS ===")
    
    analyzer = TimeSeriesAnalyzer(df_datetime)
    
    # Seasonality detection
    seasonality = analyzer.detect_seasonality('value')
    print(f"Seasonality analysis completed for {len(seasonality)} periods")
    
    # Trend detection
    trend_analysis = analyzer.detect_trends('value')
    print(f"Trend direction: {trend_analysis.get('trend_direction', 'unknown')}")
    print(f"R-squared: {trend_analysis.get('r_squared', 0):.3f}")
    
    # Technical indicators
    indicators = analyzer.calculate_technical_indicators('value')
    print(f"Technical indicators calculated: {len(indicators.columns)} indicators")
    
    # Anomaly detection
    anomalies = analyzer.detect_anomalies('value')
    anomaly_count = anomalies['is_anomaly'].sum()
    print(f"Anomalies detected: {anomaly_count}")

# Execute demonstration
if __name__ == "__main__":
    run_comprehensive_time_series_demo()
```

#### Explanation
1. **DatetimeIndex**: Foundation for time series analysis with built-in temporal operations
2. **Resampling**: Frequency conversion and temporal aggregation for different time periods
3. **Rolling Operations**: Moving windows for trend analysis and smoothing
4. **Time Zones**: Handling global data with proper timezone conversions
5. **Technical Analysis**: Financial and statistical indicators for time series

#### Use Cases
- **Financial Analysis**: Stock price analysis, portfolio tracking, risk assessment
- **IoT Data Processing**: Sensor data aggregation, anomaly detection
- **Business Analytics**: Sales forecasting, seasonal analysis, performance metrics
- **Scientific Research**: Experimental data analysis, climate studies
- **Web Analytics**: User behavior tracking, traffic pattern analysis

#### Best Practices
- **Consistent Frequency**: Ensure regular time intervals for analysis
- **Handle Missing Data**: Use appropriate filling strategies (forward fill, interpolation)
- **Timezone Awareness**: Always consider timezone implications for global data
- **Memory Optimization**: Use appropriate data types and chunking for large datasets
- **Validation**: Check for data quality issues like duplicates and gaps

#### Pitfalls
- **Timezone Confusion**: Mixing timezone-aware and naive datetimes
- **Irregular Frequencies**: Gaps in time series can affect analysis
- **Memory Issues**: Large time series datasets can consume significant memory
- **Seasonality Assumptions**: Not all patterns are truly seasonal
- **Overfitting**: Complex models may not generalize well

#### Debugging
```python
def debug_time_series_data(df: pd.DataFrame):
    """Debug time series data quality."""
    
    print("Time Series Debug Information:")
    print(f"Index type: {type(df.index)}")
    print(f"Is monotonic: {df.index.is_monotonic}")
    print(f"Timezone: {df.index.tz}")
    
    # Check for gaps
    freq = pd.infer_freq(df.index)
    print(f"Inferred frequency: {freq}")
    
    # Missing data
    missing_count = df.isnull().sum().sum()
    print(f"Missing values: {missing_count}")
    
    # Duplicated dates
    duplicate_dates = df.index.duplicated().sum()
    print(f"Duplicate dates: {duplicate_dates}")
```

#### Optimization

**Time Series Performance Tips:**

| Scenario | Optimization Strategy |
|----------|----------------------|
| **Large Datasets** | Use chunking and efficient data types |
| **Memory Constraints** | Process data in smaller time windows |
| **Complex Calculations** | Vectorize operations and use built-in functions |
| **Frequent Resampling** | Cache intermediate results |
| **Multi-series Analysis** | Use groupby operations efficiently |

**Memory and Performance:**
- Use categorical data types for repeated string values
- Leverage NumPy operations where possible
- Consider using Dask for out-of-core processing
- Optimize rolling window operations with appropriate parameters

---

## Question 3

**Discuss howPandasintegrates withMatplotlibandSeabornfordata visualization.**

### Answer

#### Theory
Pandas seamlessly integrates with Matplotlib and Seaborn to provide powerful data visualization capabilities. Pandas DataFrames and Series have built-in plotting methods that use Matplotlib as the backend, making it easy to create quick visualizations. Seaborn builds on Matplotlib and provides high-level statistical visualization functions that work excellently with Pandas data structures. This integration allows for rapid exploratory data analysis, statistical visualization, and publication-ready plots directly from Pandas objects.

#### Code Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ======================== DATA PREPARATION ========================

def create_visualization_dataset():
    """Create comprehensive dataset for visualization demonstrations."""
    
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    
    # Create base dataset
    data = {
        'date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'sales': np.random.exponential(1000, n_samples),
        'profit': np.random.normal(500, 200, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], n_samples),
        'customer_satisfaction': np.random.normal(4.2, 0.8, n_samples),
        'marketing_spend': np.random.exponential(200, n_samples),
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 5, n_samples),
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlations
    df['sales'] = df['sales'] + 0.3 * df['marketing_spend'] + np.random.normal(0, 100, n_samples)
    df['profit'] = 0.2 * df['sales'] + np.random.normal(0, 50, n_samples)
    
    # Ensure positive values where needed
    df['sales'] = np.abs(df['sales'])
    df['marketing_spend'] = np.abs(df['marketing_spend'])
    df['customer_satisfaction'] = np.clip(df['customer_satisfaction'], 1, 5)
    
    return df

def demonstrate_pandas_plotting():
    """Demonstrate built-in Pandas plotting capabilities."""
    
    print("=== PANDAS BUILT-IN PLOTTING ===")
    
    df = create_visualization_dataset()
    
    # 1. Basic line plots
    print(f"\n1. LINE PLOTS:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series line plot
    daily_sales = df.groupby('date')['sales'].sum()
    daily_sales.plot(ax=axes[0,0], title='Daily Sales Over Time', color='blue')
    axes[0,0].set_ylabel('Sales ($)')
    
    # Multiple series line plot
    metrics = df.groupby('date')[['sales', 'profit', 'marketing_spend']].mean()
    metrics.plot(ax=axes[0,1], title='Key Metrics Over Time')
    axes[0,1].set_ylabel('Value')
    
    # Subplots for different metrics
    df.groupby('date')['sales'].sum().plot(ax=axes[1,0], title='Sales', color='green')
    df.groupby('date')['profit'].sum().plot(ax=axes[1,1], title='Profit', color='orange')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Bar plots
    print(f"\n2. BAR PLOTS:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sales by region
    region_sales = df.groupby('region')['sales'].sum()
    region_sales.plot(kind='bar', ax=axes[0,0], title='Sales by Region', color='skyblue')
    axes[0,0].set_ylabel('Total Sales ($)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Horizontal bar plot
    product_sales = df.groupby('product_category')['sales'].mean()
    product_sales.plot(kind='barh', ax=axes[0,1], title='Average Sales by Product', color='lightcoral')
    axes[0,1].set_xlabel('Average Sales ($)')
    
    # Stacked bar plot
    sales_by_region_product = df.groupby(['region', 'product_category'])['sales'].sum().unstack()
    sales_by_region_product.plot(kind='bar', stacked=True, ax=axes[1,0], title='Sales by Region and Product')
    axes[1,0].set_ylabel('Total Sales ($)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Grouped bar plot
    sales_by_region_product.plot(kind='bar', ax=axes[1,1], title='Sales by Region and Product (Grouped)')
    axes[1,1].set_ylabel('Total Sales ($)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Distribution plots
    print(f"\n3. DISTRIBUTION PLOTS:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram
    df['sales'].plot(kind='hist', bins=50, ax=axes[0,0], title='Sales Distribution', alpha=0.7)
    axes[0,0].set_xlabel('Sales ($)')
    axes[0,0].set_ylabel('Frequency')
    
    # Box plot
    df.boxplot(column='sales', by='region', ax=axes[0,1])
    axes[0,1].set_title('Sales Distribution by Region')
    axes[0,1].set_ylabel('Sales ($)')
    
    # Density plot
    df['profit'].plot(kind='density', ax=axes[1,0], title='Profit Density')
    axes[1,0].set_xlabel('Profit ($)')
    axes[1,0].set_ylabel('Density')
    
    # Multiple distributions
    for region in df['region'].unique():
        region_data = df[df['region'] == region]['sales']
        region_data.plot(kind='density', ax=axes[1,1], alpha=0.7, label=region)
    axes[1,1].set_title('Sales Density by Region')
    axes[1,1].set_xlabel('Sales ($)')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 4. Scatter plots
    print(f"\n4. SCATTER PLOTS:")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Basic scatter plot
    df.plot(kind='scatter', x='marketing_spend', y='sales', ax=axes[0], 
            title='Sales vs Marketing Spend', alpha=0.6)
    axes[0].set_xlabel('Marketing Spend ($)')
    axes[0].set_ylabel('Sales ($)')
    
    # Scatter plot with color mapping
    df.plot(kind='scatter', x='marketing_spend', y='sales', c='profit', 
            colormap='viridis', ax=axes[1], title='Sales vs Marketing (colored by Profit)')
    axes[1].set_xlabel('Marketing Spend ($)')
    axes[1].set_ylabel('Sales ($)')
    
    plt.tight_layout()
    plt.show()
    
    return df

def demonstrate_matplotlib_integration():
    """Demonstrate advanced Matplotlib integration with Pandas."""
    
    print(f"\n=== MATPLOTLIB INTEGRATION ===")
    
    df = create_visualization_dataset()
    
    # 1. Custom styling with Matplotlib
    print(f"\n1. CUSTOM STYLING:")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Custom line plot with annotations
    monthly_sales = df.groupby(df['date'].dt.to_period('M'))['sales'].sum()
    monthly_sales.plot(ax=axes[0,0], linewidth=2, marker='o', markersize=4)
    axes[0,0].set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Sales ($)', fontsize=12)
    axes[0,0].grid(True, alpha=0.3)
    
    # Add trend line
    x_vals = np.arange(len(monthly_sales))
    z = np.polyfit(x_vals, monthly_sales.values, 1)
    p = np.poly1d(z)
    axes[0,0].plot(monthly_sales.index, p(x_vals), "--", alpha=0.8, color='red', label='Trend')
    axes[0,0].legend()
    
    # Custom bar plot with error bars
    region_stats = df.groupby('region')['sales'].agg(['mean', 'std'])
    region_stats['mean'].plot(kind='bar', ax=axes[0,1], yerr=region_stats['std'], 
                             capsize=5, color='lightblue', edgecolor='navy')
    axes[0,1].set_title('Average Sales by Region (with error bars)', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Average Sales ($)', fontsize=12)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Subplot within subplot
    axes[1,0].remove()
    gs = fig.add_gridspec(2, 2)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_hist = ax_main.inset_axes([0.65, 0.65, 0.3, 0.3])
    
    # Main scatter plot
    df.plot(kind='scatter', x='marketing_spend', y='sales', ax=ax_main, alpha=0.6)
    ax_main.set_title('Sales vs Marketing Spend with Distribution', fontsize=14, fontweight='bold')
    
    # Inset histogram
    df['marketing_spend'].plot(kind='hist', ax=ax_hist, bins=20, alpha=0.7, color='orange')
    ax_hist.set_title('Marketing Spend Dist.', fontsize=8)
    ax_hist.tick_params(axis='both', labelsize=6)
    
    # Correlation heatmap using Matplotlib
    correlation_matrix = df[['sales', 'profit', 'marketing_spend', 'customer_satisfaction']].corr()
    im = axes[1,1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1,1].set_xticks(range(len(correlation_matrix.columns)))
    axes[1,1].set_yticks(range(len(correlation_matrix.columns)))
    axes[1,1].set_xticklabels(correlation_matrix.columns, rotation=45)
    axes[1,1].set_yticklabels(correlation_matrix.columns)
    axes[1,1].set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Add correlation values as text
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text = axes[1,1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Multiple y-axes
    print(f"\n2. MULTIPLE Y-AXES:")
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Primary y-axis
    monthly_data = df.groupby(df['date'].dt.to_period('M')).agg({
        'sales': 'sum',
        'marketing_spend': 'sum',
        'customer_satisfaction': 'mean'
    })
    
    color = 'tab:blue'
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Sales ($)', color=color)
    monthly_data['sales'].plot(ax=ax1, color=color, linewidth=2, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Secondary y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Marketing Spend ($)', color=color)
    monthly_data['marketing_spend'].plot(ax=ax2, color=color, linewidth=2, marker='s')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Sales and Marketing Spend Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return df

def demonstrate_seaborn_integration():
    """Demonstrate Seaborn integration with Pandas."""
    
    print(f"\n=== SEABORN INTEGRATION ===")
    
    df = create_visualization_dataset()
    
    # 1. Statistical plots
    print(f"\n1. STATISTICAL PLOTS:")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Regression plot
    sns.regplot(data=df, x='marketing_spend', y='sales', ax=axes[0,0])
    axes[0,0].set_title('Sales vs Marketing Spend (with regression line)')
    
    # Residual plot
    sns.residplot(data=df, x='marketing_spend', y='sales', ax=axes[0,1])
    axes[0,1].set_title('Residual Plot')
    
    # Joint plot equivalent using subplots
    sns.scatterplot(data=df, x='marketing_spend', y='sales', hue='region', ax=axes[1,0])
    axes[1,0].set_title('Sales vs Marketing by Region')
    
    # Correlation heatmap
    numeric_cols = ['sales', 'profit', 'marketing_spend', 'customer_satisfaction', 'temperature']
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Categorical plots
    print(f"\n2. CATEGORICAL PLOTS:")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Box plot
    sns.boxplot(data=df, x='region', y='sales', ax=axes[0,0])
    axes[0,0].set_title('Sales Distribution by Region')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Violin plot
    sns.violinplot(data=df, x='region', y='sales', ax=axes[0,1])
    axes[0,1].set_title('Sales Distribution by Region (Violin)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Strip plot
    sns.stripplot(data=df, x='region', y='sales', ax=axes[0,2], alpha=0.6)
    axes[0,2].set_title('Sales by Region (Strip Plot)')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # Bar plot with error bars
    sns.barplot(data=df, x='region', y='sales', estimator=np.mean, ci=95, ax=axes[1,0])
    axes[1,0].set_title('Average Sales by Region (95% CI)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Point plot
    sns.pointplot(data=df, x='region', y='sales', estimator=np.mean, ci=95, ax=axes[1,1])
    axes[1,1].set_title('Sales Trends by Region')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Count plot
    sns.countplot(data=df, x='region', ax=axes[1,2])
    axes[1,2].set_title('Data Points by Region')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Multi-dimensional plots
    print(f"\n3. MULTI-DIMENSIONAL PLOTS:")
    
    # Pair plot for numeric variables
    numeric_subset = df[['sales', 'profit', 'marketing_spend', 'customer_satisfaction']].sample(200)
    g = sns.pairplot(numeric_subset, diag_kind='kde')
    g.fig.suptitle('Pair Plot of Numeric Variables', y=1.02)
    plt.show()
    
    # Facet grid
    g = sns.FacetGrid(df, col='region', row='season', margin_titles=True, height=4)
    g.map(sns.scatterplot, 'marketing_spend', 'sales', alpha=0.6)
    g.add_legend()
    plt.show()
    
    # 4. Time series with Seaborn
    print(f"\n4. TIME SERIES WITH SEABORN:")
    
    # Prepare time series data
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    monthly_avg = df.groupby(['year', 'month', 'region'])['sales'].mean().reset_index()
    monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))
    
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=monthly_avg, x='date', y='sales', hue='region', marker='o')
    plt.title('Monthly Sales Trends by Region')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return df

# ======================== VISUALIZATION UTILITY CLASS ========================

class PandasVisualizationHelper:
    """Advanced visualization helper for Pandas data."""
    
    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with DataFrame."""
        self.df = dataframe.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def quick_overview(self, figsize: tuple = (16, 12)):
        """Create a comprehensive overview of the dataset."""
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Data shape and missing values
        ax1 = fig.add_subplot(gs[0, 0])
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            missing_data[missing_data > 0].plot(kind='bar', ax=ax1, color='orange')
            ax1.set_title('Missing Values by Column')
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Missing Values Status')
        
        # 2. Data types distribution
        ax2 = fig.add_subplot(gs[0, 1])
        dtype_counts = self.df.dtypes.value_counts()
        dtype_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('Data Types Distribution')
        
        # 3. Numeric columns correlation
        if len(self.numeric_columns) > 1:
            ax3 = fig.add_subplot(gs[0, 2])
            corr_matrix = self.df[self.numeric_columns].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_title('Correlation Matrix')
        
        # 4. Distribution of first numeric column
        if len(self.numeric_columns) > 0:
            ax4 = fig.add_subplot(gs[1, 0])
            self.df[self.numeric_columns[0]].plot(kind='hist', bins=30, ax=ax4, alpha=0.7)
            ax4.set_title(f'Distribution of {self.numeric_columns[0]}')
        
        # 5. Box plot of numeric columns
        if len(self.numeric_columns) > 1:
            ax5 = fig.add_subplot(gs[1, 1])
            self.df[self.numeric_columns].boxplot(ax=ax5)
            ax5.set_title('Box Plot of Numeric Columns')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. Categorical column counts
        if len(self.categorical_columns) > 0:
            ax6 = fig.add_subplot(gs[1, 2])
            first_cat = self.categorical_columns[0]
            self.df[first_cat].value_counts().plot(kind='bar', ax=ax6)
            ax6.set_title(f'Count of {first_cat}')
            ax6.tick_params(axis='x', rotation=45)
        
        # 7. Time series if date column exists
        date_columns = self.df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0 and len(self.numeric_columns) > 0:
            ax7 = fig.add_subplot(gs[2, :])
            if self.df.index.name in date_columns or isinstance(self.df.index, pd.DatetimeIndex):
                self.df[self.numeric_columns[0]].plot(ax=ax7)
                ax7.set_title(f'Time Series: {self.numeric_columns[0]}')
            else:
                date_col = date_columns[0]
                temp_df = self.df.set_index(date_col)
                temp_df[self.numeric_columns[0]].plot(ax=ax7)
                ax7.set_title(f'Time Series: {self.numeric_columns[0]}')
        
        plt.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        plt.show()
    
    def correlation_analysis(self, method: str = 'pearson', figsize: tuple = (12, 10)):
        """Detailed correlation analysis."""
        
        if len(self.numeric_columns) < 2:
            print("Need at least 2 numeric columns for correlation analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Correlation heatmap
        corr_matrix = self.df[self.numeric_columns].corr(method=method)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,0])
        axes[0,0].set_title(f'{method.capitalize()} Correlation Matrix')
        
        # 2. Correlation with target (assume first numeric column is target)
        target_col = self.numeric_columns[0]
        correlations = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
        correlations.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title(f'Correlations with {target_col}')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Scatter plot of highest correlation
        if len(correlations) > 0:
            highest_corr_col = correlations.index[0]
            self.df.plot(kind='scatter', x=highest_corr_col, y=target_col, ax=axes[1,0], alpha=0.6)
            axes[1,0].set_title(f'{target_col} vs {highest_corr_col}')
        
        # 4. Distribution comparison
        if len(self.numeric_columns) >= 2:
            axes[1,1].hist([self.df[self.numeric_columns[0]].dropna(), 
                           self.df[self.numeric_columns[1]].dropna()], 
                          bins=30, alpha=0.7, label=self.numeric_columns[:2])
            axes[1,1].set_title('Distribution Comparison')
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return corr_matrix
    
    def categorical_analysis(self, figsize: tuple = (15, 10)):
        """Analyze categorical variables."""
        
        if len(self.categorical_columns) == 0:
            print("No categorical columns found")
            return
        
        n_cats = len(self.categorical_columns)
        n_nums = len(self.numeric_columns)
        
        if n_cats == 1 and n_nums >= 1:
            # Single categorical vs numeric
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            cat_col = self.categorical_columns[0]
            num_col = self.numeric_columns[0]
            
            # Count plot
            sns.countplot(data=self.df, x=cat_col, ax=axes[0,0])
            axes[0,0].set_title(f'Count of {cat_col}')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Box plot
            sns.boxplot(data=self.df, x=cat_col, y=num_col, ax=axes[0,1])
            axes[0,1].set_title(f'{num_col} by {cat_col}')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Bar plot with means
            sns.barplot(data=self.df, x=cat_col, y=num_col, estimator=np.mean, ci=95, ax=axes[1,0])
            axes[1,0].set_title(f'Average {num_col} by {cat_col}')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Violin plot
            sns.violinplot(data=self.df, x=cat_col, y=num_col, ax=axes[1,1])
            axes[1,1].set_title(f'{num_col} Distribution by {cat_col}')
            axes[1,1].tick_params(axis='x', rotation=45)
            
        elif n_cats >= 2:
            # Multiple categorical analysis
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            cat1, cat2 = self.categorical_columns[0], self.categorical_columns[1]
            
            # Cross-tabulation
            crosstab = pd.crosstab(self.df[cat1], self.df[cat2])
            sns.heatmap(crosstab, annot=True, fmt='d', ax=axes[0,0])
            axes[0,0].set_title(f'{cat1} vs {cat2} Cross-tabulation')
            
            # Stacked bar plot
            crosstab_pct = pd.crosstab(self.df[cat1], self.df[cat2], normalize='index') * 100
            crosstab_pct.plot(kind='bar', stacked=True, ax=axes[0,1])
            axes[0,1].set_title(f'{cat1} vs {cat2} (Percentage)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Individual count plots
            sns.countplot(data=self.df, x=cat1, ax=axes[1,0])
            axes[1,0].set_title(f'Count of {cat1}')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            sns.countplot(data=self.df, x=cat2, ax=axes[1,1])
            axes[1,1].set_title(f'Count of {cat2}')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def time_series_analysis(self, date_column: str = None, value_columns: list = None, 
                           figsize: tuple = (15, 10)):
        """Analyze time series data."""
        
        # Identify date column
        if date_column is None:
            date_columns = self.df.select_dtypes(include=['datetime64']).columns
            if len(date_columns) > 0:
                date_column = date_columns[0]
            elif isinstance(self.df.index, pd.DatetimeIndex):
                # Use index as date column
                temp_df = self.df.copy()
            else:
                print("No datetime column found")
                return
        
        if date_column and date_column in self.df.columns:
            temp_df = self.df.set_index(date_column)
        else:
            temp_df = self.df.copy()
        
        if value_columns is None:
            value_columns = self.numeric_columns[:3]  # Take first 3 numeric columns
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Time series plot
        for col in value_columns:
            if col in temp_df.columns:
                temp_df[col].plot(ax=axes[0,0], label=col, alpha=0.8)
        axes[0,0].set_title('Time Series Plot')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Monthly aggregation
        if len(value_columns) > 0:
            monthly_data = temp_df.resample('M')[value_columns[0]].sum()
            monthly_data.plot(ax=axes[0,1], marker='o')
            axes[0,1].set_title(f'Monthly {value_columns[0]}')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Seasonal decomposition (simplified)
        if len(value_columns) > 0:
            # Rolling statistics
            rolling_mean = temp_df[value_columns[0]].rolling(window=30).mean()
            rolling_std = temp_df[value_columns[0]].rolling(window=30).std()
            
            axes[1,0].plot(temp_df.index, temp_df[value_columns[0]], alpha=0.6, label='Original')
            axes[1,0].plot(temp_df.index, rolling_mean, label='30-day Moving Average')
            axes[1,0].fill_between(temp_df.index, 
                                  rolling_mean - rolling_std, 
                                  rolling_mean + rolling_std, 
                                  alpha=0.3, label='±1 Std Dev')
            axes[1,0].set_title('Rolling Statistics')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Distribution over time
        if len(value_columns) > 0:
            temp_df['month'] = temp_df.index.month
            monthly_dist = temp_df.groupby('month')[value_columns[0]].mean()
            monthly_dist.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title(f'Average {value_columns[0]} by Month')
            axes[1,1].set_xlabel('Month')
        
        plt.tight_layout()
        plt.show()

# ======================== MAIN DEMONSTRATION ========================

def run_comprehensive_visualization_demo():
    """Run comprehensive visualization demonstration."""
    
    print("PANDAS VISUALIZATION COMPREHENSIVE GUIDE")
    print("="*40)
    
    # Pandas built-in plotting
    df_pandas = demonstrate_pandas_plotting()
    
    # Matplotlib integration
    df_matplotlib = demonstrate_matplotlib_integration()
    
    # Seaborn integration
    df_seaborn = demonstrate_seaborn_integration()
    
    # Demonstrate PandasVisualizationHelper class
    print(f"\n=== VISUALIZATION HELPER CLASS ===")
    
    viz_helper = PandasVisualizationHelper(df_pandas)
    
    # Quick overview
    print("Creating dataset overview...")
    viz_helper.quick_overview()
    
    # Correlation analysis
    print("Performing correlation analysis...")
    correlations = viz_helper.correlation_analysis()
    
    # Categorical analysis
    print("Analyzing categorical variables...")
    viz_helper.categorical_analysis()
    
    # Time series analysis
    print("Analyzing time series patterns...")
    viz_helper.time_series_analysis(date_column='date', value_columns=['sales', 'profit'])

# Execute demonstration
if __name__ == "__main__":
    run_comprehensive_visualization_demo()
```

#### Explanation
1. **Built-in Plotting**: Pandas provides `.plot()` method for quick visualizations using Matplotlib backend
2. **Matplotlib Integration**: Direct access to Matplotlib for custom styling and complex layouts
3. **Seaborn Integration**: High-level statistical plots with excellent Pandas DataFrame support
4. **Customization**: Combining all three libraries for publication-ready visualizations
5. **Interactive Features**: Enhanced plotting capabilities for exploratory data analysis

#### Use Cases
- **Exploratory Data Analysis**: Quick visualizations for data understanding
- **Statistical Analysis**: Correlation plots, distribution analysis, hypothesis testing
- **Business Reporting**: Dashboard creation, KPI visualization, trend analysis
- **Scientific Research**: Publication-ready plots, experimental data visualization
- **Time Series Analysis**: Trend visualization, seasonality detection, forecasting plots

#### Best Practices
- **Choose Right Plot Type**: Match visualization to data type and analysis goal
- **Consistent Styling**: Use consistent colors, fonts, and layouts across plots
- **Clear Labels**: Always include titles, axis labels, and legends
- **Appropriate Scale**: Use log scales, normalization when necessary
- **Color Accessibility**: Consider colorblind-friendly palettes

#### Pitfalls
- **Overplotting**: Too many data points can obscure patterns
- **Poor Color Choices**: Misleading or inaccessible color schemes
- **Missing Context**: Plots without proper titles or explanations
- **Scale Issues**: Inappropriate axis scaling can mislead interpretation
- **Performance Issues**: Large datasets can slow down interactive plots

#### Debugging
```python
def debug_plot_issues(df: pd.DataFrame, plot_type: str):
    """Debug common plotting issues."""
    
    print(f"Debugging {plot_type} plot:")
    print(f"DataFrame shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    
    # Check for missing values
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print(f"Missing values detected:\n{missing_data[missing_data > 0]}")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"Infinite values in {col}: {inf_count}")
    
    # Memory usage
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

#### Optimization

**Visualization Performance Tips:**

| Scenario | Optimization Strategy |
|----------|----------------------|
| **Large Datasets** | Sample data or use aggregation before plotting |
| **Multiple Plots** | Use subplots efficiently, close figures to free memory |
| **Interactive Plots** | Consider using Plotly or Bokeh for better performance |
| **High-Resolution** | Adjust DPI settings and use vector formats |
| **Real-time Updates** | Use blitting for animation, update data efficiently |

**Memory and Performance:**
- Use appropriate figure sizes and DPI settings
- Close matplotlib figures after saving to prevent memory leaks
- Sample large datasets for exploration before creating final plots
- Use efficient data types and avoid unnecessary data copying

---

## Question 4

**How would you usePandasto prepare and cleanecommerce sales datafor better insight into customer purchasing patterns?**

### Answer

#### Theory
E-commerce sales data preparation and cleaning using Pandas involves multiple stages: data ingestion, quality assessment, handling missing values, data type conversions, outlier detection, feature engineering, and data transformation. The goal is to create a clean, consistent dataset that enables accurate analysis of customer purchasing patterns, seasonal trends, product performance, and business metrics. This process typically involves dealing with transactional data, customer information, product catalogs, and temporal patterns.

#### Code Example

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ======================== E-COMMERCE DATA SIMULATION ========================

def create_ecommerce_dataset():
    """Create realistic e-commerce sales dataset with quality issues."""
    
    np.random.seed(42)
    
    # Generate base data
    n_orders = 10000
    n_customers = 2000
    n_products = 500
    
    # Date range (2 years of data)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Product categories and brands
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty', 'Toys']
    brands = ['Brand_A', 'Brand_B', 'Brand_C', 'Brand_D', 'Brand_E', 'Other']
    
    # Generate orders
    orders_data = []
    for i in range(n_orders):
        order_id = f"ORD_{i+1:06d}"
        customer_id = f"CUST_{np.random.randint(1, n_customers+1):05d}"
        
        # Random date with some seasonal patterns
        base_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))
        # Add seasonal boost for holidays
        if base_date.month in [11, 12]:  # Holiday season
            base_date = base_date if np.random.random() < 0.3 else base_date
        
        # Order details
        num_items = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
        
        for j in range(num_items):
            product_id = f"PROD_{np.random.randint(1, n_products+1):04d}"
            category = np.random.choice(categories)
            brand = np.random.choice(brands)
            
            # Price with some category-based patterns
            base_price = {
                'Electronics': 200, 'Clothing': 50, 'Home & Garden': 80,
                'Books': 15, 'Sports': 60, 'Beauty': 30, 'Toys': 25
            }
            
            price = np.random.exponential(base_price[category]) + 5
            quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            
            # Introduce some data quality issues
            order_date = base_date
            
            # Occasionally corrupt data
            if np.random.random() < 0.02:  # 2% bad data
                if np.random.random() < 0.5:
                    price = -price  # Negative price
                else:
                    quantity = 0    # Zero quantity
            
            # Sometimes missing data
            brand_value = brand if np.random.random() > 0.05 else np.nan
            category_value = category if np.random.random() > 0.02 else np.nan
            
            orders_data.append({
                'order_id': order_id,
                'customer_id': customer_id,
                'order_date': order_date,
                'product_id': product_id,
                'category': category_value,
                'brand': brand_value,
                'quantity': quantity,
                'unit_price': price,
                'total_amount': price * quantity
            })
    
    df_orders = pd.DataFrame(orders_data)
    
    # Generate customer data
    customers_data = []
    for i in range(n_customers):
        customer_id = f"CUST_{i+1:05d}"
        
        # Customer demographics
        age = np.random.randint(18, 80)
        gender = np.random.choice(['M', 'F', 'Other'])
        city = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                                'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Other'])
        
        # Registration date
        reg_date = start_date + timedelta(days=np.random.randint(-365, 365))
        
        # Introduce some quality issues
        age_value = age if np.random.random() > 0.03 else np.nan
        city_value = city if np.random.random() > 0.05 else np.nan
        
        customers_data.append({
            'customer_id': customer_id,
            'age': age_value,
            'gender': gender,
            'city': city_value,
            'registration_date': reg_date
        })
    
    df_customers = pd.DataFrame(customers_data)
    
    # Generate product data
    products_data = []
    for i in range(n_products):
        product_id = f"PROD_{i+1:04d}"
        category = np.random.choice(categories)
        brand = np.random.choice(brands)
        
        # Product details
        weight = np.random.exponential(0.5) + 0.1  # kg
        dimensions = f"{np.random.randint(5, 50)}x{np.random.randint(5, 50)}x{np.random.randint(2, 20)}"
        
        # Introduce quality issues
        weight_value = weight if np.random.random() > 0.04 else np.nan
        brand_value = brand if np.random.random() > 0.03 else np.nan
        
        products_data.append({
            'product_id': product_id,
            'category': category,
            'brand': brand_value,
            'weight_kg': weight_value,
            'dimensions': dimensions
        })
    
    df_products = pd.DataFrame(products_data)
    
    return df_orders, df_customers, df_products

def assess_data_quality(df_orders, df_customers, df_products):
    """Assess data quality issues in e-commerce datasets."""
    
    print("=== DATA QUALITY ASSESSMENT ===")
    
    datasets = {
        'Orders': df_orders,
        'Customers': df_customers,
        'Products': df_products
    }
    
    quality_report = {}
    
    for name, df in datasets.items():
        print(f"\n{name} Dataset:")
        print(f"Shape: {df.shape}")
        
        # Missing values
        missing_values = df.isnull().sum()
        missing_pct = (missing_values / len(df) * 100).round(2)
        
        print(f"Missing values:")
        for col in missing_values.index:
            if missing_values[col] > 0:
                print(f"  {col}: {missing_values[col]} ({missing_pct[col]}%)")
        
        # Duplicates
        duplicates = df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        # Data types
        print(f"Data types:")
        for dtype in df.dtypes.unique():
            cols = df.select_dtypes(include=[dtype]).columns.tolist()
            print(f"  {dtype}: {len(cols)} columns")
        
        quality_report[name] = {
            'shape': df.shape,
            'missing_values': missing_values.to_dict(),
            'duplicates': duplicates,
            'data_types': df.dtypes.to_dict()
        }
    
    # Business logic violations
    print(f"\n=== BUSINESS LOGIC VIOLATIONS ===")
    
    # Orders dataset violations
    negative_prices = (df_orders['unit_price'] < 0).sum()
    zero_quantities = (df_orders['quantity'] <= 0).sum()
    negative_amounts = (df_orders['total_amount'] < 0).sum()
    
    print(f"Orders violations:")
    print(f"  Negative prices: {negative_prices}")
    print(f"  Zero/negative quantities: {zero_quantities}")
    print(f"  Negative total amounts: {negative_amounts}")
    
    # Consistency checks
    amount_mismatch = (abs(df_orders['total_amount'] - 
                          df_orders['unit_price'] * df_orders['quantity']) > 0.01).sum()
    print(f"  Amount calculation mismatches: {amount_mismatch}")
    
    return quality_report

# ======================== DATA CLEANING AND PREPARATION ========================

class EcommerceDataCleaner:
    """Comprehensive e-commerce data cleaning and preparation utility."""
    
    def __init__(self, df_orders: pd.DataFrame, df_customers: pd.DataFrame, 
                 df_products: pd.DataFrame):
        """Initialize with e-commerce datasets."""
        self.df_orders = df_orders.copy()
        self.df_customers = df_customers.copy()
        self.df_products = df_products.copy()
        
        self.cleaning_log = []
        self.cleaned_data = {}
    
    def clean_orders_data(self) -> pd.DataFrame:
        """Clean and prepare orders dataset."""
        
        print("=== CLEANING ORDERS DATA ===")
        df = self.df_orders.copy()
        initial_rows = len(df)
        
        # 1. Fix data types
        print(f"1. Fixing data types...")
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
        df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
        
        # 2. Remove invalid records
        print(f"2. Removing invalid records...")
        
        # Remove negative prices
        negative_price_mask = df['unit_price'] < 0
        negative_price_count = negative_price_mask.sum()
        df = df[~negative_price_mask]
        self.cleaning_log.append(f"Removed {negative_price_count} records with negative prices")
        
        # Remove zero or negative quantities
        invalid_qty_mask = df['quantity'] <= 0
        invalid_qty_count = invalid_qty_mask.sum()
        df = df[~invalid_qty_mask]
        self.cleaning_log.append(f"Removed {invalid_qty_count} records with invalid quantities")
        
        # 3. Fix calculation errors
        print(f"3. Fixing calculation errors...")
        df['total_amount'] = df['unit_price'] * df['quantity']
        
        # 4. Handle missing values
        print(f"4. Handling missing values...")
        
        # Missing categories - use mode imputation by brand
        if df['category'].isnull().any():
            for brand in df['brand'].unique():
                if pd.notna(brand):
                    brand_mode_category = df[df['brand'] == brand]['category'].mode()
                    if not brand_mode_category.empty:
                        mask = (df['brand'] == brand) & df['category'].isnull()
                        df.loc[mask, 'category'] = brand_mode_category.iloc[0]
            
            # Fill remaining missing categories with overall mode
            overall_mode = df['category'].mode().iloc[0] if not df['category'].mode().empty else 'Unknown'
            df['category'].fillna(overall_mode, inplace=True)
        
        # Missing brands
        df['brand'].fillna('Unknown', inplace=True)
        
        # 5. Detect and handle outliers
        print(f"5. Detecting outliers...")
        
        # Price outliers using IQR method
        Q1_price = df['unit_price'].quantile(0.25)
        Q3_price = df['unit_price'].quantile(0.75)
        IQR_price = Q3_price - Q1_price
        price_outlier_mask = (df['unit_price'] > (Q3_price + 3 * IQR_price))
        
        # Cap extreme outliers
        outlier_count = price_outlier_mask.sum()
        df.loc[price_outlier_mask, 'unit_price'] = Q3_price + 3 * IQR_price
        self.cleaning_log.append(f"Capped {outlier_count} extreme price outliers")
        
        # 6. Add derived features
        print(f"6. Adding derived features...")
        
        df['order_year'] = df['order_date'].dt.year
        df['order_month'] = df['order_date'].dt.month
        df['order_quarter'] = df['order_date'].dt.quarter
        df['order_day_of_week'] = df['order_date'].dt.dayofweek
        df['order_day_name'] = df['order_date'].dt.day_name()
        df['is_weekend'] = df['order_day_of_week'].isin([5, 6])
        
        # Holiday flags
        df['is_holiday_season'] = df['order_month'].isin([11, 12])
        df['is_summer'] = df['order_month'].isin([6, 7, 8])
        
        final_rows = len(df)
        print(f"Orders cleaning completed: {initial_rows} -> {final_rows} rows ({final_rows/initial_rows:.2%} retained)")
        
        self.cleaned_data['orders'] = df
        return df
    
    def clean_customers_data(self) -> pd.DataFrame:
        """Clean and prepare customers dataset."""
        
        print(f"\n=== CLEANING CUSTOMERS DATA ===")
        df = self.df_customers.copy()
        initial_rows = len(df)
        
        # 1. Fix data types
        print(f"1. Fixing data types...")
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        
        # 2. Handle missing values
        print(f"2. Handling missing values...")
        
        # Age imputation based on gender
        for gender in df['gender'].unique():
            if pd.notna(gender):
                gender_median_age = df[df['gender'] == gender]['age'].median()
                mask = (df['gender'] == gender) & df['age'].isnull()
                if pd.notna(gender_median_age):
                    df.loc[mask, 'age'] = gender_median_age
        
        # Fill remaining missing ages with overall median
        overall_median_age = df['age'].median()
        df['age'].fillna(overall_median_age, inplace=True)
        
        # City imputation
        df['city'].fillna('Unknown', inplace=True)
        
        # 3. Validate age ranges
        print(f"3. Validating age ranges...")
        invalid_age_mask = (df['age'] < 13) | (df['age'] > 100)
        invalid_age_count = invalid_age_mask.sum()
        
        # Replace invalid ages with median
        df.loc[invalid_age_mask, 'age'] = df['age'].median()
        self.cleaning_log.append(f"Fixed {invalid_age_count} invalid age values")
        
        # 4. Standardize categorical values
        print(f"4. Standardizing categorical values...")
        
        # Standardize gender values
        gender_mapping = {
            'M': 'Male', 'F': 'Female', 'Other': 'Other',
            'm': 'Male', 'f': 'Female', 'male': 'Male', 'female': 'Female'
        }
        df['gender'] = df['gender'].map(gender_mapping).fillna(df['gender'])
        
        # 5. Add derived features
        print(f"5. Adding derived features...")
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 25, 35, 50, 65, 100], 
                                labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        # Registration recency
        current_date = df['registration_date'].max()
        df['days_since_registration'] = (current_date - df['registration_date']).dt.days
        df['registration_recency'] = pd.cut(df['days_since_registration'],
                                           bins=[0, 90, 365, 730, np.inf],
                                           labels=['New', 'Recent', 'Established', 'Long-term'])
        
        final_rows = len(df)
        print(f"Customers cleaning completed: {initial_rows} -> {final_rows} rows")
        
        self.cleaned_data['customers'] = df
        return df
    
    def clean_products_data(self) -> pd.DataFrame:
        """Clean and prepare products dataset."""
        
        print(f"\n=== CLEANING PRODUCTS DATA ===")
        df = self.df_products.copy()
        initial_rows = len(df)
        
        # 1. Fix data types
        print(f"1. Fixing data types...")
        df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
        
        # 2. Handle missing values
        print(f"2. Handling missing values...")
        
        # Weight imputation by category
        for category in df['category'].unique():
            if pd.notna(category):
                category_median_weight = df[df['category'] == category]['weight_kg'].median()
                mask = (df['category'] == category) & df['weight_kg'].isnull()
                if pd.notna(category_median_weight):
                    df.loc[mask, 'weight_kg'] = category_median_weight
        
        # Brand imputation
        df['brand'].fillna('Unknown', inplace=True)
        
        # 3. Add derived features
        print(f"3. Adding derived features...")
        
        # Weight categories
        df['weight_category'] = pd.cut(df['weight_kg'],
                                      bins=[0, 0.5, 2, 10, np.inf],
                                      labels=['Light', 'Medium', 'Heavy', 'Very Heavy'])
        
        final_rows = len(df)
        print(f"Products cleaning completed: {initial_rows} -> {final_rows} rows")
        
        self.cleaned_data['products'] = df
        return df
    
    def merge_datasets(self) -> pd.DataFrame:
        """Merge cleaned datasets for analysis."""
        
        print(f"\n=== MERGING DATASETS ===")
        
        if 'orders' not in self.cleaned_data:
            self.clean_orders_data()
        if 'customers' not in self.cleaned_data:
            self.clean_customers_data()
        if 'products' not in self.cleaned_data:
            self.clean_products_data()
        
        # Start with orders as base
        merged_df = self.cleaned_data['orders'].copy()
        
        # Merge with customers
        merged_df = merged_df.merge(
            self.cleaned_data['customers'],
            on='customer_id',
            how='left',
            suffixes=('', '_customer')
        )
        
        # Merge with products
        merged_df = merged_df.merge(
            self.cleaned_data['products'][['product_id', 'brand', 'weight_kg', 'weight_category']],
            on='product_id',
            how='left',
            suffixes=('', '_product')
        )
        
        # Handle conflicts (use product table data over orders table)
        merged_df['brand'] = merged_df['brand_product'].fillna(merged_df['brand'])
        merged_df.drop(['brand_product'], axis=1, inplace=True)
        
        print(f"Merged dataset shape: {merged_df.shape}")
        
        self.cleaned_data['merged'] = merged_df
        return merged_df
    
    def generate_cleaning_report(self) -> Dict:
        """Generate comprehensive cleaning report."""
        
        report = {
            'cleaning_log': self.cleaning_log,
            'datasets_processed': list(self.cleaned_data.keys()),
            'data_quality_metrics': {}
        }
        
        for name, df in self.cleaned_data.items():
            if df is not None:
                report['data_quality_metrics'][name] = {
                    'shape': df.shape,
                    'missing_values': df.isnull().sum().to_dict(),
                    'data_types': df.dtypes.to_dict(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
                }
        
        return report

# ======================== CUSTOMER INSIGHTS ANALYSIS ========================

class CustomerInsightsAnalyzer:
    """Analyze customer purchasing patterns from cleaned e-commerce data."""
    
    def __init__(self, merged_df: pd.DataFrame):
        """Initialize with merged e-commerce dataset."""
        self.df = merged_df.copy()
    
    def analyze_customer_behavior(self) -> Dict:
        """Analyze customer purchasing behavior patterns."""
        
        print("=== CUSTOMER BEHAVIOR ANALYSIS ===")
        
        # 1. Customer-level aggregations
        customer_metrics = self.df.groupby('customer_id').agg({
            'order_id': 'nunique',  # Number of orders
            'product_id': 'nunique',  # Product variety
            'total_amount': ['sum', 'mean', 'count'],
            'quantity': 'sum',
            'order_date': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns]
        
        # Calculate derived metrics
        customer_metrics['customer_lifetime_days'] = (
            customer_metrics['order_date_max'] - customer_metrics['order_date_min']
        ).dt.days + 1
        
        customer_metrics['avg_order_value'] = (
            customer_metrics['total_amount_sum'] / customer_metrics['order_id_nunique']
        ).round(2)
        
        customer_metrics['purchase_frequency'] = (
            customer_metrics['order_id_nunique'] / customer_metrics['customer_lifetime_days'] * 30
        ).round(3)  # Orders per month
        
        # 2. Customer segmentation
        print(f"\n1. CUSTOMER SEGMENTATION:")
        
        # RFM Analysis (Recency, Frequency, Monetary)
        current_date = self.df['order_date'].max()
        
        rfm = self.df.groupby('customer_id').agg({
            'order_date': lambda x: (current_date - x.max()).days,  # Recency
            'order_id': 'nunique',  # Frequency
            'total_amount': 'sum'  # Monetary
        }).round(2)
        
        rfm.columns = ['recency', 'frequency', 'monetary']
        
        # Create RFM scores (1-5 scale)
        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1])  # Lower recency = higher score
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5])
        
        # Combine scores
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
        
        # Customer segments based on RFM
        def segment_customers(row):
            if row['rfm_score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['rfm_score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['rfm_score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif row['rfm_score'] in ['533', '532', '531', '523', '522', '521', '515', '514']:
                return 'New Customers'
            elif row['rfm_score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'At Risk'
            elif row['rfm_score'] in ['155', '154', '145', '143', '142', '135', '134']:
                return "Can't Lose Them"
            else:
                return 'Others'
        
        rfm['segment'] = rfm.apply(segment_customers, axis=1)
        
        segment_summary = rfm['segment'].value_counts()
        print("Customer segments:")
        for segment, count in segment_summary.items():
            print(f"  {segment}: {count} customers ({count/len(rfm)*100:.1f}%)")
        
        # 3. Product preferences by customer segments
        print(f"\n2. PRODUCT PREFERENCES BY SEGMENT:")
        
        df_with_segments = self.df.merge(rfm[['segment']], left_on='customer_id', right_index=True)
        
        segment_preferences = df_with_segments.groupby(['segment', 'category'])['total_amount'].sum().unstack()
        segment_preferences_pct = segment_preferences.div(segment_preferences.sum(axis=1), axis=0) * 100
        
        print("Category preferences by segment (% of total spending):")
        print(segment_preferences_pct.round(1))
        
        # 4. Seasonal patterns
        print(f"\n3. SEASONAL PATTERNS:")
        
        seasonal_analysis = self.df.groupby(['order_month', 'segment'])['total_amount'].sum().unstack()
        seasonal_analysis_pct = seasonal_analysis.div(seasonal_analysis.sum(axis=0), axis=1) * 100
        
        print("Monthly spending distribution by segment:")
        print(seasonal_analysis_pct.round(1).head())
        
        return {
            'customer_metrics': customer_metrics,
            'rfm_analysis': rfm,
            'segment_summary': segment_summary,
            'product_preferences': segment_preferences_pct,
            'seasonal_patterns': seasonal_analysis_pct
        }
    
    def analyze_product_performance(self) -> Dict:
        """Analyze product and category performance."""
        
        print(f"\n=== PRODUCT PERFORMANCE ANALYSIS ===")
        
        # 1. Category analysis
        category_performance = self.df.groupby('category').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'order_id': 'nunique',
            'customer_id': 'nunique'
        }).round(2)
        
        category_performance['avg_order_value'] = (
            category_performance['total_amount'] / category_performance['order_id']
        ).round(2)
        
        category_performance['market_share'] = (
            category_performance['total_amount'] / category_performance['total_amount'].sum() * 100
        ).round(2)
        
        print("Category performance:")
        print(category_performance.sort_values('total_amount', ascending=False))
        
        # 2. Brand analysis
        brand_performance = self.df.groupby('brand').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'customer_id': 'nunique'
        }).round(2).sort_values('total_amount', ascending=False).head(10)
        
        print(f"\nTop 10 brands by revenue:")
        print(brand_performance)
        
        # 3. Product lifecycle analysis
        product_lifecycle = self.df.groupby('product_id').agg({
            'order_date': ['min', 'max', 'count'],
            'total_amount': 'sum',
            'customer_id': 'nunique'
        })
        
        product_lifecycle.columns = ['_'.join(col).strip() for col in product_lifecycle.columns]
        product_lifecycle['product_lifespan_days'] = (
            product_lifecycle['order_date_max'] - product_lifecycle['order_date_min']
        ).dt.days + 1
        
        print(f"\nProduct lifecycle summary:")
        print(f"Average product lifespan: {product_lifecycle['product_lifespan_days'].mean():.0f} days")
        print(f"Products sold to single customer: {(product_lifecycle['customer_id_nunique'] == 1).sum()}")
        
        return {
            'category_performance': category_performance,
            'brand_performance': brand_performance,
            'product_lifecycle': product_lifecycle
        }
    
    def generate_business_insights(self) -> List[str]:
        """Generate actionable business insights."""
        
        insights = []
        
        # Revenue insights
        total_revenue = self.df['total_amount'].sum()
        avg_order_value = self.df.groupby('order_id')['total_amount'].sum().mean()
        insights.append(f"Total revenue: ${total_revenue:,.2f} with average order value: ${avg_order_value:.2f}")
        
        # Customer insights
        total_customers = self.df['customer_id'].nunique()
        avg_customer_value = total_revenue / total_customers
        insights.append(f"Total customers: {total_customers:,} with average customer value: ${avg_customer_value:.2f}")
        
        # Seasonal insights
        holiday_revenue = self.df[self.df['is_holiday_season']]['total_amount'].sum()
        holiday_pct = holiday_revenue / total_revenue * 100
        insights.append(f"Holiday season accounts for {holiday_pct:.1f}% of total revenue")
        
        # Category insights
        top_category = self.df.groupby('category')['total_amount'].sum().idxmax()
        top_category_pct = self.df.groupby('category')['total_amount'].sum().max() / total_revenue * 100
        insights.append(f"Top category '{top_category}' accounts for {top_category_pct:.1f}% of revenue")
        
        # Customer behavior insights
        repeat_customers = self.df.groupby('customer_id')['order_id'].nunique()
        repeat_customer_pct = (repeat_customers > 1).sum() / len(repeat_customers) * 100
        insights.append(f"{repeat_customer_pct:.1f}% of customers have made repeat purchases")
        
        return insights

# ======================== MAIN DEMONSTRATION ========================

def run_ecommerce_analysis_demo():
    """Run comprehensive e-commerce data analysis demonstration."""
    
    print("E-COMMERCE DATA PREPARATION & ANALYSIS")
    print("="*35)
    
    # 1. Create sample data
    print("1. Creating sample e-commerce dataset...")
    df_orders, df_customers, df_products = create_ecommerce_dataset()
    
    # 2. Assess data quality
    print("\n2. Assessing data quality...")
    quality_report = assess_data_quality(df_orders, df_customers, df_products)
    
    # 3. Clean data
    print("\n3. Cleaning and preparing data...")
    cleaner = EcommerceDataCleaner(df_orders, df_customers, df_products)
    
    # Clean individual datasets
    cleaned_orders = cleaner.clean_orders_data()
    cleaned_customers = cleaner.clean_customers_data()
    cleaned_products = cleaner.clean_products_data()
    
    # Merge datasets
    merged_df = cleaner.merge_datasets()
    
    # Generate cleaning report
    cleaning_report = cleaner.generate_cleaning_report()
    print(f"\nCleaning completed. Log entries: {len(cleaning_report['cleaning_log'])}")
    
    # 4. Analyze customer insights
    print("\n4. Analyzing customer purchasing patterns...")
    analyzer = CustomerInsightsAnalyzer(merged_df)
    
    customer_behavior = analyzer.analyze_customer_behavior()
    product_performance = analyzer.analyze_product_performance()
    business_insights = analyzer.generate_business_insights()
    
    print(f"\n=== BUSINESS INSIGHTS ===")
    for i, insight in enumerate(business_insights, 1):
        print(f"{i}. {insight}")
    
    return {
        'raw_data': (df_orders, df_customers, df_products),
        'cleaned_data': merged_df,
        'analysis_results': {
            'customer_behavior': customer_behavior,
            'product_performance': product_performance,
            'business_insights': business_insights
        }
    }

# Execute demonstration
if __name__ == "__main__":
    results = run_ecommerce_analysis_demo()
```

#### Explanation
1. **Data Quality Assessment**: Systematic evaluation of missing values, duplicates, and business logic violations
2. **Data Cleaning Pipeline**: Structured approach to fixing data types, handling outliers, and imputing missing values
3. **Feature Engineering**: Creating derived features for temporal analysis and customer segmentation
4. **Customer Segmentation**: RFM analysis to identify high-value customer segments
5. **Business Intelligence**: Generating actionable insights from cleaned and prepared data

#### Use Cases
- **Customer Analytics**: Understanding purchasing patterns, customer lifetime value, churn prediction
- **Product Analytics**: Performance tracking, inventory optimization, recommendation systems
- **Marketing Analytics**: Campaign effectiveness, customer segmentation, personalization
- **Business Intelligence**: Revenue analysis, trend identification, strategic planning
- **Operational Analytics**: Supply chain optimization, demand forecasting, fraud detection

#### Best Practices
- **Systematic Approach**: Follow structured data cleaning pipeline with documentation
- **Domain Knowledge**: Apply business rules and constraints during cleaning
- **Data Validation**: Implement checks for data consistency and business logic
- **Feature Engineering**: Create meaningful derived features for analysis
- **Reproducibility**: Document all cleaning steps and transformations

#### Pitfalls
- **Over-cleaning**: Removing too much data can bias analysis results
- **Assumption Errors**: Incorrect imputation strategies can introduce bias
- **Lost Context**: Important business context can be lost during cleaning
- **Performance Issues**: Large datasets require efficient processing strategies
- **Data Leakage**: Using future information in historical analysis

#### Debugging
```python
def debug_ecommerce_data(df: pd.DataFrame):
    """Debug e-commerce data quality issues."""
    
    print("E-commerce Data Debug Report:")
    print(f"Dataset shape: {df.shape}")
    
    # Check key business metrics
    if 'total_amount' in df.columns:
        print(f"Revenue range: ${df['total_amount'].min():.2f} - ${df['total_amount'].max():.2f}")
        negative_revenue = (df['total_amount'] < 0).sum()
        if negative_revenue > 0:
            print(f"WARNING: {negative_revenue} negative revenue records")
    
    # Check temporal consistency
    if 'order_date' in df.columns:
        date_range = df['order_date'].max() - df['order_date'].min()
        print(f"Date range: {date_range.days} days")
        
        future_dates = (df['order_date'] > pd.Timestamp.now()).sum()
        if future_dates > 0:
            print(f"WARNING: {future_dates} future dates found")
    
    # Check customer data consistency
    if 'customer_id' in df.columns:
        customer_count = df['customer_id'].nunique()
        order_count = df.shape[0]
        print(f"Customer-to-order ratio: {order_count/customer_count:.2f} orders per customer")
```

#### Optimization

**E-commerce Data Processing Tips:**

| Scenario | Optimization Strategy |
|----------|----------------------|
| **Large Transaction Data** | Use chunking and incremental processing |
| **Real-time Analytics** | Implement streaming data pipelines |
| **Customer Segmentation** | Cache segmentation results and update periodically |
| **Product Recommendations** | Use sparse matrices and efficient similarity calculations |
| **Time Series Forecasting** | Optimize feature engineering and model training |

**Memory and Performance:**
- Use appropriate data types (category for repeated strings)
- Implement data partitioning by date or customer segments
- Use vectorized operations for feature engineering
- Consider using Dask for out-of-core processing of large datasets

---

## Question 5

**Discuss the advantages ofvectorized operationsinPandasover iteration.**

### Answer

#### Theory
Vectorized operations in Pandas refer to operations that are applied to entire arrays or Series at once, rather than iterating through individual elements. These operations leverage optimized C and Cython implementations under the hood, along with NumPy's efficient array operations. Vectorization provides significant performance improvements, cleaner code, and better memory usage compared to explicit Python loops. This approach aligns with the "array programming" paradigm and is fundamental to efficient data analysis in Pandas.

#### Code Example

```python
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Callable
import warnings
warnings.filterwarnings('ignore')

# ======================== PERFORMANCE COMPARISON UTILITIES ========================

def create_performance_dataset(size: int = 100000):
    """Create dataset for performance testing."""
    
    np.random.seed(42)
    
    data = {
        'value1': np.random.randn(size),
        'value2': np.random.randn(size),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size),
        'price': np.random.uniform(10, 1000, size),
        'quantity': np.random.randint(1, 100, size),
        'date': pd.date_range('2020-01-01', periods=size, freq='H'),
        'is_active': np.random.choice([True, False], size),
        'score': np.random.uniform(0, 100, size)
    }
    
    return pd.DataFrame(data)

def time_function(func: Callable, *args, **kwargs) -> float:
    """Time function execution."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result

def demonstrate_basic_vectorization():
    """Demonstrate basic vectorized operations vs loops."""
    
    print("=== BASIC VECTORIZATION COMPARISON ===")
    
    df = create_performance_dataset(50000)
    
    # 1. Simple arithmetic operations
    print(f"\n1. ARITHMETIC OPERATIONS:")
    
    # Iterative approach
    def add_iterative(series1, series2):
        result = []
        for i in range(len(series1)):
            result.append(series1.iloc[i] + series2.iloc[i])
        return pd.Series(result)
    
    # Vectorized approach
    def add_vectorized(series1, series2):
        return series1 + series2
    
    # Time comparison
    time_iter, result_iter = time_function(add_iterative, df['value1'], df['value2'])
    time_vect, result_vect = time_function(add_vectorized, df['value1'], df['value2'])
    
    print(f"Iterative addition: {time_iter:.4f} seconds")
    print(f"Vectorized addition: {time_vect:.4f} seconds")
    print(f"Speedup: {time_iter/time_vect:.1f}x faster")
    print(f"Results match: {np.allclose(result_iter, result_vect)}")
    
    # 2. Conditional operations
    print(f"\n2. CONDITIONAL OPERATIONS:")
    
    # Iterative approach
    def conditional_iterative(series, threshold):
        result = []
        for i in range(len(series)):
            if series.iloc[i] > threshold:
                result.append('High')
            else:
                result.append('Low')
        return pd.Series(result)
    
    # Vectorized approach
    def conditional_vectorized(series, threshold):
        return np.where(series > threshold, 'High', 'Low')
    
    threshold = df['score'].mean()
    time_iter, result_iter = time_function(conditional_iterative, df['score'], threshold)
    time_vect, result_vect = time_function(conditional_vectorized, df['score'], threshold)
    
    print(f"Iterative conditional: {time_iter:.4f} seconds")
    print(f"Vectorized conditional: {time_vect:.4f} seconds")
    print(f"Speedup: {time_iter/time_vect:.1f}x faster")
    print(f"Results match: {(result_iter == result_vect).all()}")
    
    # 3. String operations
    print(f"\n3. STRING OPERATIONS:")
    
    # Create string data
    string_data = pd.Series([f"Item_{i}_Category_{np.random.choice(['A', 'B', 'C'])}" 
                           for i in range(10000)])
    
    # Iterative approach
    def extract_iterative(series):
        result = []
        for item in series:
            parts = item.split('_')
            result.append(parts[1] if len(parts) > 1 else '')
        return pd.Series(result)
    
    # Vectorized approach
    def extract_vectorized(series):
        return series.str.split('_').str[1]
    
    time_iter, result_iter = time_function(extract_iterative, string_data)
    time_vect, result_vect = time_function(extract_vectorized, string_data)
    
    print(f"Iterative string extraction: {time_iter:.4f} seconds")
    print(f"Vectorized string extraction: {time_vect:.4f} seconds")
    print(f"Speedup: {time_iter/time_vect:.1f}x faster")
    print(f"Results match: {result_iter.equals(result_vect)}")
    
    return df

def demonstrate_advanced_vectorization():
    """Demonstrate advanced vectorized operations."""
    
    print(f"\n=== ADVANCED VECTORIZATION TECHNIQUES ===")
    
    df = create_performance_dataset(30000)
    
    # 1. Complex mathematical operations
    print(f"\n1. COMPLEX MATHEMATICAL OPERATIONS:")
    
    # Iterative approach
    def complex_math_iterative(price, quantity, score):
        result = []
        for i in range(len(price)):
            value = (price.iloc[i] * quantity.iloc[i]) * np.sqrt(score.iloc[i] / 100)
            result.append(np.log1p(value) if value > 0 else 0)
        return pd.Series(result)
    
    # Vectorized approach
    def complex_math_vectorized(price, quantity, score):
        value = (price * quantity) * np.sqrt(score / 100)
        return np.where(value > 0, np.log1p(value), 0)
    
    time_iter, result_iter = time_function(complex_math_iterative, df['price'], df['quantity'], df['score'])
    time_vect, result_vect = time_function(complex_math_vectorized, df['price'], df['quantity'], df['score'])
    
    print(f"Iterative complex math: {time_iter:.4f} seconds")
    print(f"Vectorized complex math: {time_vect:.4f} seconds")
    print(f"Speedup: {time_iter/time_vect:.1f}x faster")
    print(f"Results close: {np.allclose(result_iter, result_vect, rtol=1e-10)}")
    
    # 2. Date operations
    print(f"\n2. DATE OPERATIONS:")
    
    # Iterative approach
    def date_operations_iterative(dates):
        result = []
        for date in dates:
            is_weekend = date.weekday() >= 5
            is_month_end = (date.replace(day=1) + pd.DateOffset(months=1) - pd.DateOffset(days=1)).day == date.day
            result.append('Weekend_MonthEnd' if is_weekend and is_month_end 
                         else 'Weekend' if is_weekend 
                         else 'MonthEnd' if is_month_end 
                         else 'Regular')
        return pd.Series(result)
    
    # Vectorized approach
    def date_operations_vectorized(dates):
        is_weekend = dates.dt.weekday >= 5
        is_month_end = dates.dt.is_month_end
        
        return np.select(
            [is_weekend & is_month_end, is_weekend, is_month_end],
            ['Weekend_MonthEnd', 'Weekend', 'MonthEnd'],
            default='Regular'
        )
    
    sample_dates = df['date'].head(5000)  # Smaller sample for date operations
    time_iter, result_iter = time_function(date_operations_iterative, sample_dates)
    time_vect, result_vect = time_function(date_operations_vectorized, sample_dates)
    
    print(f"Iterative date operations: {time_iter:.4f} seconds")
    print(f"Vectorized date operations: {time_vect:.4f} seconds")
    print(f"Speedup: {time_iter/time_vect:.1f}x faster")
    print(f"Results match: {(result_iter == result_vect).all()}")
    
    # 3. Groupby operations
    print(f"\n3. GROUPBY OPERATIONS:")
    
    # Iterative approach (simulated)
    def groupby_iterative(df, group_col, value_col):
        result = {}
        for category in df[group_col].unique():
            mask = df[group_col] == category
            subset = df[mask][value_col]
            result[category] = {
                'mean': subset.mean(),
                'std': subset.std(),
                'count': len(subset)
            }
        return result
    
    # Vectorized approach
    def groupby_vectorized(df, group_col, value_col):
        return df.groupby(group_col)[value_col].agg(['mean', 'std', 'count'])
    
    time_iter, result_iter = time_function(groupby_iterative, df, 'category', 'price')
    time_vect, result_vect = time_function(groupby_vectorized, df, 'category', 'price')
    
    print(f"Iterative groupby: {time_iter:.4f} seconds")
    print(f"Vectorized groupby: {time_vect:.4f} seconds")
    print(f"Speedup: {time_iter/time_vect:.1f}x faster")
    
    return df

def demonstrate_memory_efficiency():
    """Demonstrate memory efficiency of vectorized operations."""
    
    print(f"\n=== MEMORY EFFICIENCY COMPARISON ===")
    
    import psutil
    import os
    
    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    # Create larger dataset
    df = create_performance_dataset(100000)
    
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    # 1. Memory usage during iterative operations
    print(f"\n1. ITERATIVE MEMORY USAGE:")
    
    initial_memory = get_memory_usage()
    
    # Iterative approach - creates temporary lists
    def memory_intensive_iterative(series):
        temp_results = []
        intermediate_results = []
        
        for i, value in enumerate(series):
            temp_calc = value ** 2 + np.log1p(abs(value))
            temp_results.append(temp_calc)
            
            if i % 1000 == 0:
                intermediate_results.append(sum(temp_results))
                temp_results = []  # Clear temp list
        
        return pd.Series(intermediate_results)
    
    result_iter = memory_intensive_iterative(df['value1'])
    iterative_memory = get_memory_usage()
    
    print(f"Memory after iterative operation: {iterative_memory:.1f} MB")
    print(f"Memory increase: {iterative_memory - initial_memory:.1f} MB")
    
    # Reset memory baseline
    del result_iter
    import gc
    gc.collect()
    
    # 2. Memory usage during vectorized operations
    print(f"\n2. VECTORIZED MEMORY USAGE:")
    
    baseline_memory = get_memory_usage()
    
    # Vectorized approach - operates on arrays directly
    def memory_efficient_vectorized(series):
        # Chunked processing for large operations
        chunk_size = 10000
        results = []
        
        for i in range(0, len(series), chunk_size):
            chunk = series.iloc[i:i+chunk_size]
            chunk_result = (chunk ** 2 + np.log1p(np.abs(chunk))).sum()
            results.append(chunk_result)
        
        return pd.Series(results)
    
    result_vect = memory_efficient_vectorized(df['value1'])
    vectorized_memory = get_memory_usage()
    
    print(f"Memory after vectorized operation: {vectorized_memory:.1f} MB")
    print(f"Memory increase: {vectorized_memory - baseline_memory:.1f} MB")
    
    # 3. Demonstrate in-place operations
    print(f"\n3. IN-PLACE OPERATIONS:")
    
    df_copy = df.copy()
    initial_copy_memory = get_memory_usage()
    
    # Not in-place (creates new Series)
    df_copy['new_column'] = df_copy['price'] * df_copy['quantity']
    not_inplace_memory = get_memory_usage()
    
    # Reset
    df_copy = df.copy()
    gc.collect()
    
    # In-place operations where possible
    df_copy['price'] *= df_copy['quantity']  # Modify existing column
    inplace_memory = get_memory_usage()
    
    print(f"Not in-place memory increase: {not_inplace_memory - initial_copy_memory:.1f} MB")
    print(f"In-place memory increase: {inplace_memory - initial_copy_memory:.1f} MB")
    
    return df

# ======================== VECTORIZATION PATTERNS CLASS ========================

class VectorizationOptimizer:
    """Utility class for optimizing Pandas operations with vectorization."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with DataFrame."""
        self.df = df.copy()
        self.optimization_log = []
    
    def optimize_conditions(self, conditions: List[Dict]) -> pd.Series:
        """Optimize multiple conditional operations."""
        
        # Example conditions format:
        # [{'column': 'price', 'op': '>', 'value': 100, 'result': 'High'},
        #  {'column': 'score', 'op': '<', 'value': 50, 'result': 'Low'}]
        
        if not conditions:
            return pd.Series(index=self.df.index, dtype='object')
        
        # Build conditions and choices for np.select
        condition_arrays = []
        choices = []
        
        for condition in conditions:
            col = condition['column']
            op = condition['op']
            value = condition['value']
            result = condition['result']
            
            if op == '>':
                cond_array = self.df[col] > value
            elif op == '<':
                cond_array = self.df[col] < value
            elif op == '>=':
                cond_array = self.df[col] >= value
            elif op == '<=':
                cond_array = self.df[col] <= value
            elif op == '==':
                cond_array = self.df[col] == value
            elif op == '!=':
                cond_array = self.df[col] != value
            else:
                continue
            
            condition_arrays.append(cond_array)
            choices.append(result)
        
        # Use np.select for efficient conditional logic
        result = np.select(condition_arrays, choices, default='Other')
        
        self.optimization_log.append(f"Optimized {len(conditions)} conditions using np.select")
        return pd.Series(result, index=self.df.index)
    
    def optimize_string_operations(self, column: str, operations: List[str]) -> pd.DataFrame:
        """Optimize multiple string operations on a column."""
        
        result_df = pd.DataFrame(index=self.df.index)
        series = self.df[column].astype(str)
        
        # Batch string operations
        for operation in operations:
            if operation == 'upper':
                result_df[f'{column}_upper'] = series.str.upper()
            elif operation == 'lower':
                result_df[f'{column}_lower'] = series.str.lower()
            elif operation == 'length':
                result_df[f'{column}_length'] = series.str.len()
            elif operation.startswith('contains_'):
                pattern = operation.split('_', 1)[1]
                result_df[f'{column}_contains_{pattern}'] = series.str.contains(pattern, na=False)
            elif operation.startswith('extract_'):
                pattern = operation.split('_', 1)[1]
                result_df[f'{column}_extract_{pattern}'] = series.str.extract(f'({pattern})')
        
        self.optimization_log.append(f"Optimized {len(operations)} string operations on {column}")
        return result_df
    
    def optimize_date_operations(self, date_column: str) -> pd.DataFrame:
        """Optimize date-related feature extraction."""
        
        result_df = pd.DataFrame(index=self.df.index)
        date_series = pd.to_datetime(self.df[date_column])
        
        # Vectorized date operations
        result_df[f'{date_column}_year'] = date_series.dt.year
        result_df[f'{date_column}_month'] = date_series.dt.month
        result_df[f'{date_column}_day'] = date_series.dt.day
        result_df[f'{date_column}_dayofweek'] = date_series.dt.dayofweek
        result_df[f'{date_column}_quarter'] = date_series.dt.quarter
        result_df[f'{date_column}_is_weekend'] = date_series.dt.dayofweek >= 5
        result_df[f'{date_column}_is_month_end'] = date_series.dt.is_month_end
        result_df[f'{date_column}_is_month_start'] = date_series.dt.is_month_start
        
        # Business day calculations
        result_df[f'{date_column}_is_business_day'] = date_series.dt.dayofweek < 5
        
        self.optimization_log.append(f"Optimized date operations on {date_column}")
        return result_df
    
    def optimize_numerical_transformations(self, columns: List[str], 
                                         transformations: List[str]) -> pd.DataFrame:
        """Optimize numerical transformations using vectorization."""
        
        result_df = pd.DataFrame(index=self.df.index)
        
        for column in columns:
            if column not in self.df.columns:
                continue
                
            series = self.df[column]
            
            for transformation in transformations:
                if transformation == 'log':
                    result_df[f'{column}_log'] = np.log1p(np.maximum(series, 0))
                elif transformation == 'sqrt':
                    result_df[f'{column}_sqrt'] = np.sqrt(np.maximum(series, 0))
                elif transformation == 'square':
                    result_df[f'{column}_square'] = series ** 2
                elif transformation == 'normalize':
                    result_df[f'{column}_norm'] = (series - series.mean()) / series.std()
                elif transformation == 'minmax':
                    result_df[f'{column}_minmax'] = (series - series.min()) / (series.max() - series.min())
                elif transformation == 'rank':
                    result_df[f'{column}_rank'] = series.rank()
                elif transformation == 'percentile':
                    result_df[f'{column}_percentile'] = series.rank(pct=True)
        
        self.optimization_log.append(f"Optimized {len(transformations)} transformations on {len(columns)} columns")
        return result_df
    
    def benchmark_operation(self, operation_func: Callable, *args, iterations: int = 3) -> Dict:
        """Benchmark an operation multiple times."""
        
        times = []
        for _ in range(iterations):
            start_time = time.time()
            result = operation_func(*args)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'result_shape': getattr(result, 'shape', None)
        }

# ======================== BEST PRACTICES DEMONSTRATION ========================

def demonstrate_best_practices():
    """Demonstrate best practices for vectorization."""
    
    print(f"\n=== VECTORIZATION BEST PRACTICES ===")
    
    df = create_performance_dataset(20000)
    
    # 1. Use built-in methods when available
    print(f"\n1. USE BUILT-IN METHODS:")
    
    # Custom implementation vs built-in
    def custom_cumsum(series):
        result = []
        total = 0
        for value in series:
            total += value
            result.append(total)
        return pd.Series(result)
    
    # Built-in method
    def builtin_cumsum(series):
        return series.cumsum()
    
    sample_series = df['value1'].head(5000)
    time_custom, result_custom = time_function(custom_cumsum, sample_series)
    time_builtin, result_builtin = time_function(builtin_cumsum, sample_series)
    
    print(f"Custom cumsum: {time_custom:.4f} seconds")
    print(f"Built-in cumsum: {time_builtin:.4f} seconds")
    print(f"Speedup: {time_custom/time_builtin:.1f}x faster")
    
    # 2. Avoid apply when vectorization is possible
    print(f"\n2. AVOID APPLY WHEN POSSIBLE:")
    
    # Using apply
    def using_apply(df):
        return df.apply(lambda x: x['price'] * x['quantity'] + x['score'], axis=1)
    
    # Pure vectorization
    def using_vectorization(df):
        return df['price'] * df['quantity'] + df['score']
    
    sample_df = df.head(10000)
    time_apply, result_apply = time_function(using_apply, sample_df)
    time_vect, result_vect = time_function(using_vectorization, sample_df)
    
    print(f"Using apply: {time_apply:.4f} seconds")
    print(f"Pure vectorization: {time_vect:.4f} seconds")
    print(f"Speedup: {time_apply/time_vect:.1f}x faster")
    print(f"Results match: {np.allclose(result_apply, result_vect)}")
    
    # 3. Use appropriate data types
    print(f"\n3. OPTIMIZE DATA TYPES:")
    
    # Create dataset with suboptimal types
    df_inefficient = df.copy()
    df_inefficient['category'] = df_inefficient['category'].astype(str)
    df_inefficient['is_active'] = df_inefficient['is_active'].astype(object)
    
    # Optimize data types
    df_efficient = df.copy()
    df_efficient['category'] = df_efficient['category'].astype('category')
    df_efficient['is_active'] = df_efficient['is_active'].astype(bool)
    
    # Compare memory usage
    inefficient_memory = df_inefficient.memory_usage(deep=True).sum() / 1024**2
    efficient_memory = df_efficient.memory_usage(deep=True).sum() / 1024**2
    
    print(f"Inefficient types memory: {inefficient_memory:.1f} MB")
    print(f"Efficient types memory: {efficient_memory:.1f} MB")
    print(f"Memory reduction: {(1 - efficient_memory/inefficient_memory)*100:.1f}%")
    
    # Test operation speed
    def operation_test(df):
        return df.groupby('category')['price'].mean()
    
    time_inefficient, _ = time_function(operation_test, df_inefficient)
    time_efficient, _ = time_function(operation_test, df_efficient)
    
    print(f"Inefficient types operation: {time_inefficient:.4f} seconds")
    print(f"Efficient types operation: {time_efficient:.4f} seconds")
    print(f"Speedup: {time_inefficient/time_efficient:.1f}x faster")

# ======================== MAIN DEMONSTRATION ========================

def run_comprehensive_vectorization_demo():
    """Run comprehensive vectorization demonstration."""
    
    print("PANDAS VECTORIZATION COMPREHENSIVE GUIDE")
    print("="*40)
    
    # Basic vectorization
    df_basic = demonstrate_basic_vectorization()
    
    # Advanced vectorization
    df_advanced = demonstrate_advanced_vectorization()
    
    # Memory efficiency
    df_memory = demonstrate_memory_efficiency()
    
    # Best practices
    demonstrate_best_practices()
    
    # Demonstrate VectorizationOptimizer class
    print(f"\n=== VECTORIZATION OPTIMIZER CLASS ===")
    
    optimizer = VectorizationOptimizer(df_basic)
    
    # Optimize conditions
    conditions = [
        {'column': 'price', 'op': '>', 'value': 500, 'result': 'High_Price'},
        {'column': 'score', 'op': '<', 'value': 30, 'result': 'Low_Score'}
    ]
    
    optimized_conditions = optimizer.optimize_conditions(conditions)
    print(f"Condition optimization completed: {len(optimized_conditions)} results")
    
    # Optimize date operations
    date_features = optimizer.optimize_date_operations('date')
    print(f"Date optimization completed: {date_features.shape[1]} features created")
    
    # Optimization log
    print(f"\nOptimization log:")
    for log_entry in optimizer.optimization_log:
        print(f"  - {log_entry}")

# Execute demonstration
if __name__ == "__main__":
    run_comprehensive_vectorization_demo()
```

#### Explanation
1. **Performance Benefits**: Vectorized operations are significantly faster than loops due to optimized C implementations
2. **Memory Efficiency**: Vectorization reduces memory overhead by avoiding temporary Python objects
3. **Code Simplicity**: Vectorized code is more concise and readable than explicit loops
4. **NumPy Integration**: Leverages NumPy's efficient array operations and broadcasting
5. **Built-in Optimizations**: Pandas methods are specifically optimized for vectorized operations

#### Use Cases
- **Data Transformation**: Applying mathematical operations to entire columns
- **Feature Engineering**: Creating derived features efficiently
- **Data Cleaning**: Applying transformations and validations across datasets
- **Statistical Analysis**: Computing aggregations and statistical measures
- **Conditional Logic**: Implementing business rules across large datasets

#### Best Practices
- **Use Built-in Methods**: Prefer Pandas/NumPy built-in functions over custom implementations
- **Avoid Apply When Possible**: Use pure vectorization instead of apply() for simple operations
- **Optimize Data Types**: Use appropriate dtypes (category, bool) for better performance
- **Leverage Broadcasting**: Take advantage of NumPy's broadcasting for array operations
- **Batch Operations**: Group related operations to minimize overhead

#### Pitfalls
- **Memory Constraints**: Very large vectorized operations can consume significant memory
- **Over-vectorization**: Some complex logic may be clearer with explicit loops
- **Type Coercion**: Unexpected type conversions in vectorized operations
- **NaN Propagation**: Missing values can propagate through vectorized calculations
- **Index Alignment**: Misaligned indices can cause unexpected results

#### Debugging
```python
def debug_vectorization_performance(func_vectorized, func_iterative, data, iterations=3):
    """Debug and compare vectorization performance."""
    
    print("Vectorization Performance Debug:")
    
    # Test correctness first
    result_vect = func_vectorized(data)
    result_iter = func_iterative(data)
    
    if hasattr(result_vect, '__len__') and hasattr(result_iter, '__len__'):
        if len(result_vect) == len(result_iter):
            if isinstance(result_vect, (pd.Series, np.ndarray)):
                match = np.allclose(result_vect, result_iter, equal_nan=True)
            else:
                match = (result_vect == result_iter).all()
            print(f"Results match: {match}")
        else:
            print(f"Length mismatch: {len(result_vect)} vs {len(result_iter)}")
    
    # Performance comparison
    times_vect = []
    times_iter = []
    
    for _ in range(iterations):
        start = time.time()
        func_vectorized(data)
        times_vect.append(time.time() - start)
        
        start = time.time()
        func_iterative(data)
        times_iter.append(time.time() - start)
    
    avg_vect = np.mean(times_vect)
    avg_iter = np.mean(times_iter)
    
    print(f"Vectorized avg time: {avg_vect:.4f}s (±{np.std(times_vect):.4f})")
    print(f"Iterative avg time: {avg_iter:.4f}s (±{np.std(times_iter):.4f})")
    print(f"Speedup: {avg_iter/avg_vect:.1f}x")
```

#### Optimization

**Vectorization Performance Tips:**

| Scenario | Optimization Strategy |
|----------|----------------------|
| **Large DataFrames** | Use chunking for memory-intensive operations |
| **Complex Conditions** | Use np.select() for multiple conditional logic |
| **String Operations** | Batch string operations using .str accessor |
| **Date Operations** | Extract multiple date features in single pass |
| **Mathematical Operations** | Use NumPy functions directly when possible |

**Memory and Performance:**
- Use appropriate data types (category for strings, bool for boolean)
- Avoid chained operations that create intermediate results
- Consider using numba for custom vectorized functions
- Use eval() for complex arithmetic expressions on large datasets

**When to Avoid Vectorization:**
- Complex conditional logic that's clearer with explicit loops
- Operations requiring external API calls or file I/O
- Very small datasets where overhead exceeds benefits
- Operations that require stateful processing across rows

---

