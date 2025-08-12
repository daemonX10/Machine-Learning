# Sql Ml Interview Questions - Coding Questions

## Question 1

**Write a SQL query that joins two tables and retrieves only the rows with matching keys.**

### Answer:

#### Theory
SQL JOINs are fundamental operations that combine rows from two or more tables based on related columns. An INNER JOIN specifically returns only the rows where there's a match in both tables, effectively filtering out non-matching records.

#### Code Example
```sql
-- Basic INNER JOIN example
-- Tables: customers and orders
SELECT 
    c.customer_id,
    c.customer_name,
    c.email,
    o.order_id,
    o.order_date,
    o.total_amount
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
ORDER BY o.order_date DESC;

-- Multiple table JOIN with filtering
SELECT 
    c.customer_name,
    p.product_name,
    oi.quantity,
    oi.unit_price,
    (oi.quantity * oi.unit_price) AS line_total
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
INNER JOIN products p ON oi.product_id = p.product_id
WHERE o.order_date >= '2024-01-01'
    AND c.customer_status = 'ACTIVE';

-- JOIN with aggregation for ML features
SELECT 
    c.customer_id,
    c.customer_segment,
    COUNT(o.order_id) AS total_orders,
    AVG(o.total_amount) AS avg_order_value,
    MAX(o.order_date) AS last_order_date,
    SUM(o.total_amount) AS lifetime_value
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_segment
HAVING COUNT(o.order_id) > 1;
```

#### Explanation
1. **INNER JOIN Syntax**: `table1 INNER JOIN table2 ON table1.column = table2.column`
2. **ON Clause**: Specifies the join condition using matching keys
3. **Table Aliases**: Use aliases (c, o) for cleaner, more readable code
4. **Multiple JOINs**: Chain multiple INNER JOINs for complex relationships
5. **Filtering**: WHERE clause applies after the JOIN operation
6. **Aggregation**: GROUP BY enables feature engineering for ML models

#### Use Cases
- **Customer Analytics**: Join customer data with transaction history
- **Feature Engineering**: Create ML features by combining dimensional and fact tables
- **Data Validation**: Ensure referential integrity by finding only matching records
- **Reporting**: Generate comprehensive reports from normalized databases

#### Best Practices
- Always use table aliases for readability
- Index the columns used in JOIN conditions
- Place most selective filters in WHERE clause
- Consider JOIN order for performance optimization
- Use explicit INNER JOIN syntax for clarity

#### Pitfalls
- **Performance Issues**: JOINs on non-indexed columns can be slow
- **Cartesian Products**: Missing or incorrect ON conditions create massive result sets
- **NULL Handling**: INNER JOINs exclude rows with NULL in join columns
- **Data Loss**: INNER JOINs may eliminate important records if no match exists

#### Debugging
```sql
-- Check for missing matches
SELECT COUNT(*) FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.customer_id IS NULL;

-- Verify JOIN cardinality
SELECT 
    'customers' AS table_name, COUNT(*) AS row_count
FROM customers
UNION ALL
SELECT 
    'joined_result', COUNT(*)
FROM customers c INNER JOIN orders o ON c.customer_id = o.customer_id;
```

#### Optimization
- **Index Strategy**: Create composite indexes on JOIN columns
- **Query Hints**: Use query hints for complex multi-table JOINs
- **Partitioning**: Partition large tables to improve JOIN performance
- **Statistics**: Keep table statistics updated for optimal execution plans

---

## Question 2

**Create a SQL query to pivot a table transforming rows into columns.**

### Answer:

#### Theory
PIVOT operations transform unique values from rows into columns, creating a cross-tabular view of data. This is essential for feature engineering in ML, creating dummy variables, and reshaping data for analysis. Modern SQL provides both PIVOT syntax and manual aggregation methods.

#### Code Example
```sql
-- Method 1: Using PIVOT syntax (SQL Server, Oracle)
SELECT *
FROM (
    SELECT 
        customer_id,
        product_category,
        sales_amount
    FROM sales_data
) source_table
PIVOT (
    SUM(sales_amount)
    FOR product_category IN ([Electronics], [Clothing], [Books], [Home])
) AS pivot_table;

-- Method 2: Manual PIVOT using CASE statements (Universal SQL)
SELECT 
    customer_id,
    SUM(CASE WHEN product_category = 'Electronics' THEN sales_amount ELSE 0 END) AS Electronics,
    SUM(CASE WHEN product_category = 'Clothing' THEN sales_amount ELSE 0 END) AS Clothing,
    SUM(CASE WHEN product_category = 'Books' THEN sales_amount ELSE 0 END) AS Books,
    SUM(CASE WHEN product_category = 'Home' THEN sales_amount ELSE 0 END) AS Home,
    COUNT(DISTINCT product_category) AS category_diversity,
    SUM(sales_amount) AS total_sales
FROM sales_data
GROUP BY customer_id;

-- Dynamic PIVOT for ML feature matrix
WITH category_stats AS (
    SELECT DISTINCT product_category 
    FROM sales_data 
    WHERE sales_date >= '2024-01-01'
),
pivoted_features AS (
    SELECT 
        s.customer_id,
        s.sales_date,
        -- Aggregated features per category
        SUM(CASE WHEN s.product_category = 'Electronics' THEN s.sales_amount ELSE 0 END) AS electronics_sales,
        SUM(CASE WHEN s.product_category = 'Clothing' THEN s.sales_amount ELSE 0 END) AS clothing_sales,
        SUM(CASE WHEN s.product_category = 'Books' THEN s.sales_amount ELSE 0 END) AS books_sales,
        SUM(CASE WHEN s.product_category = 'Home' THEN s.sales_amount ELSE 0 END) AS home_sales,
        -- Behavioral features
        COUNT(DISTINCT s.product_category) AS categories_purchased,
        COUNT(*) AS transaction_count,
        AVG(s.sales_amount) AS avg_transaction_value,
        MAX(s.sales_amount) AS max_transaction_value
    FROM sales_data s
    WHERE s.sales_date >= '2024-01-01'
    GROUP BY s.customer_id, s.sales_date
)
SELECT 
    customer_id,
    electronics_sales,
    clothing_sales,
    books_sales,
    home_sales,
    categories_purchased,
    transaction_count,
    avg_transaction_value,
    max_transaction_value,
    -- Derived ML features
    CASE 
        WHEN categories_purchased >= 3 THEN 'High_Diversity'
        WHEN categories_purchased = 2 THEN 'Medium_Diversity'
        ELSE 'Low_Diversity'
    END AS customer_diversity_segment
FROM pivoted_features;

-- Advanced: Pivoting time series for ML features
SELECT 
    product_id,
    -- Sales by month (last 12 months)
    SUM(CASE WHEN EXTRACT(MONTH FROM sales_date) = 1 THEN sales_amount ELSE 0 END) AS jan_sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM sales_date) = 2 THEN sales_amount ELSE 0 END) AS feb_sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM sales_date) = 3 THEN sales_amount ELSE 0 END) AS mar_sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM sales_date) = 4 THEN sales_amount ELSE 0 END) AS apr_sales,
    -- Statistical features
    AVG(sales_amount) AS avg_monthly_sales,
    STDDEV(sales_amount) AS sales_volatility,
    MAX(sales_amount) - MIN(sales_amount) AS sales_range
FROM sales_data
WHERE sales_date >= DATE_SUB(CURRENT_DATE, INTERVAL 12 MONTH)
GROUP BY product_id;
```

#### Explanation
1. **PIVOT Syntax**: Uses FOR clause to specify pivot column and IN clause for values
2. **Aggregation**: PIVOT requires an aggregate function (SUM, COUNT, AVG, etc.)
3. **CASE Statements**: Manual method works across all SQL dialects
4. **GROUP BY**: Essential for aggregating data at desired granularity
5. **NULL Handling**: Use COALESCE or ISNULL for missing category combinations

#### Use Cases
- **Feature Engineering**: Create dummy variables for categorical data
- **Customer Segmentation**: Pivot purchase behavior across product categories
- **Time Series**: Transform temporal data into feature vectors
- **Cross-tabulation**: Business intelligence and reporting
- **ML Data Preparation**: Convert long format to wide format for algorithms

#### Best Practices
- Always specify explicit column names for predictable output
- Use meaningful aliases for pivoted columns
- Handle NULL values appropriately for ML models
- Consider data cardinality - too many pivot columns can cause performance issues
- Validate that pivot categories are complete and expected

#### Pitfalls
- **Memory Usage**: High cardinality pivot columns can consume excessive memory
- **Dynamic Columns**: Hard to predict column names with unknown categories
- **NULL vs Zero**: PIVOT may return NULL where manual CASE returns 0
- **Performance**: Large pivots can be resource-intensive
- **Maintenance**: Adding new categories requires query modification

#### Debugging
```sql
-- Check category distribution before pivoting
SELECT 
    product_category,
    COUNT(*) AS row_count,
    COUNT(DISTINCT customer_id) AS unique_customers
FROM sales_data
GROUP BY product_category
ORDER BY row_count DESC;

-- Verify pivot results
SELECT 
    COUNT(*) AS total_customers,
    AVG(Electronics + Clothing + Books + Home) AS avg_total_sales,
    COUNT(CASE WHEN Electronics > 0 THEN 1 END) AS electronics_customers
FROM pivoted_sales_data;
```

#### Optimization
- **Indexing**: Index source columns used in PIVOT operations
- **Pre-filtering**: Filter data before pivoting to reduce processing
- **Materialized Views**: Store pivot results for frequently accessed data
- **Partitioning**: Partition source tables by pivot dimensions

---

## Question 3

**Write a SQL query to calculate moving averages.**

### Answer:

#### Theory
Moving averages are essential time series features that smooth out short-term fluctuations and reveal underlying trends. They're crucial for ML models dealing with temporal data, providing momentum indicators, trend signals, and reducing noise in sequential data.

#### Code Example
```sql
-- Simple Moving Average using Window Functions
SELECT 
    product_id,
    sales_date,
    daily_sales,
    -- 7-day moving average
    AVG(daily_sales) OVER (
        PARTITION BY product_id 
        ORDER BY sales_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS ma_7_day,
    -- 30-day moving average
    AVG(daily_sales) OVER (
        PARTITION BY product_id 
        ORDER BY sales_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS ma_30_day,
    -- Exponentially Weighted Moving Average (approximation)
    AVG(daily_sales) OVER (
        PARTITION BY product_id 
        ORDER BY sales_date 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS ema_10_day
FROM daily_sales
ORDER BY product_id, sales_date;

-- Advanced Moving Averages with Multiple Time Windows
WITH enhanced_moving_averages AS (
    SELECT 
        customer_id,
        transaction_date,
        purchase_amount,
        -- Multiple moving averages for ML features
        AVG(purchase_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS ma_7_day,
        AVG(purchase_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS ma_14_day,
        AVG(purchase_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS ma_30_day,
        -- Moving standard deviation
        STDDEV(purchase_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS ma_30_day_stddev,
        -- Moving count for frequency analysis
        COUNT(*) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS transaction_frequency_30_day
    FROM customer_transactions
)
SELECT 
    customer_id,
    transaction_date,
    purchase_amount,
    ma_7_day,
    ma_14_day,
    ma_30_day,
    ma_30_day_stddev,
    transaction_frequency_30_day,
    -- Derived ML features
    CASE 
        WHEN purchase_amount > ma_7_day * 1.5 THEN 'Above_Trend'
        WHEN purchase_amount < ma_7_day * 0.5 THEN 'Below_Trend'
        ELSE 'Normal'
    END AS spending_trend_signal,
    -- Volatility indicator
    CASE 
        WHEN ma_30_day_stddev > ma_30_day * 0.3 THEN 'High_Volatility'
        ELSE 'Low_Volatility'
    END AS spending_volatility,
    -- Momentum features
    (ma_7_day - ma_30_day) / ma_30_day AS short_vs_long_momentum
FROM enhanced_moving_averages
WHERE transaction_date >= '2024-01-01';

-- Weighted Moving Average for Recent Data Emphasis
SELECT 
    stock_symbol,
    trade_date,
    closing_price,
    -- Linear weighted moving average (recent data has more weight)
    (
        closing_price * 5 +
        LAG(closing_price, 1) OVER (PARTITION BY stock_symbol ORDER BY trade_date) * 4 +
        LAG(closing_price, 2) OVER (PARTITION BY stock_symbol ORDER BY trade_date) * 3 +
        LAG(closing_price, 3) OVER (PARTITION BY stock_symbol ORDER BY trade_date) * 2 +
        LAG(closing_price, 4) OVER (PARTITION BY stock_symbol ORDER BY trade_date) * 1
    ) / 15 AS weighted_ma_5_day,
    -- Exponential Moving Average approximation
    AVG(closing_price) OVER (
        PARTITION BY stock_symbol 
        ORDER BY trade_date 
        ROWS BETWEEN 8 PRECEDING AND CURRENT ROW
    ) AS ema_9_day,
    -- Bollinger Bands using moving average
    AVG(closing_price) OVER (
        PARTITION BY stock_symbol 
        ORDER BY trade_date 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) + 2 * STDDEV(closing_price) OVER (
        PARTITION BY stock_symbol 
        ORDER BY trade_date 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) AS bollinger_upper,
    AVG(closing_price) OVER (
        PARTITION BY stock_symbol 
        ORDER BY trade_date 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) - 2 * STDDEV(closing_price) OVER (
        PARTITION BY stock_symbol 
        ORDER BY trade_date 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) AS bollinger_lower
FROM stock_prices
ORDER BY stock_symbol, trade_date;

-- Moving Average for Seasonal Data
SELECT 
    region,
    sales_month,
    monthly_revenue,
    -- Year-over-year moving average
    AVG(monthly_revenue) OVER (
        PARTITION BY region, EXTRACT(MONTH FROM sales_month)
        ORDER BY sales_month
        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    ) AS seasonal_ma_12_month,
    -- Quarter-over-quarter moving average
    AVG(monthly_revenue) OVER (
        PARTITION BY region
        ORDER BY sales_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS quarterly_ma_3_month,
    -- Trend vs seasonal component
    monthly_revenue - AVG(monthly_revenue) OVER (
        PARTITION BY region, EXTRACT(MONTH FROM sales_month)
        ORDER BY sales_month
        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    ) AS seasonal_deviation
FROM monthly_sales
ORDER BY region, sales_month;
```

#### Explanation
1. **Window Functions**: Use OVER clause with PARTITION BY and ORDER BY
2. **Frame Specification**: ROWS BETWEEN defines the moving window size
3. **PRECEDING**: Includes previous N rows in calculation
4. **Multiple Averages**: Different window sizes capture various trend patterns
5. **LAG Function**: Access previous row values for weighted calculations

#### Use Cases
- **Trend Analysis**: Identify long-term patterns in time series data
- **Signal Processing**: Smooth noisy data for ML model inputs
- **Anomaly Detection**: Compare current values against historical averages
- **Financial Analysis**: Technical indicators for trading algorithms
- **Demand Forecasting**: Create features for predictive models

#### Best Practices
- Choose window sizes based on data frequency and business cycles
- Use multiple moving averages to capture different trend horizons
- Handle edge cases where insufficient historical data exists
- Consider seasonal patterns when selecting window sizes
- Combine with other statistical measures (standard deviation, percentiles)

#### Pitfalls
- **Lag Effect**: Moving averages inherently lag behind actual trends
- **Insufficient Data**: Early records may have incomplete windows
- **Boundary Effects**: Edge cases need special handling
- **Performance**: Large windows on big datasets can be resource-intensive
- **Overfitting**: Too many MA features can lead to model complexity

#### Debugging
```sql
-- Check data completeness for moving averages
SELECT 
    product_id,
    COUNT(*) AS total_days,
    MIN(sales_date) AS first_date,
    MAX(sales_date) AS last_date,
    COUNT(CASE WHEN ma_30_day IS NOT NULL THEN 1 END) AS valid_ma_30_records
FROM daily_sales_with_ma
GROUP BY product_id
HAVING COUNT(*) != COUNT(CASE WHEN ma_30_day IS NOT NULL THEN 1 END);

-- Validate moving average calculations
SELECT 
    sales_date,
    daily_sales,
    ma_7_day,
    -- Manual verification for recent 7 days
    (SELECT AVG(daily_sales) 
     FROM daily_sales d2 
     WHERE d2.product_id = d1.product_id 
       AND d2.sales_date BETWEEN d1.sales_date - INTERVAL 6 DAY AND d1.sales_date
    ) AS manual_ma_7_day
FROM daily_sales d1
WHERE product_id = 'PROD001'
ORDER BY sales_date DESC
LIMIT 10;
```

#### Optimization
- **Indexing**: Create indexes on partition and order columns
- **Incremental Calculation**: Update only new data points instead of recalculating all
- **Materialized Views**: Store moving averages for frequently accessed timeframes
- **Partitioning**: Partition tables by date ranges for faster window operations

---

## Question 4

**How can you create lagged features in SQL?**

### Answer:

#### Theory
Lagged features are time-shifted versions of existing variables that capture temporal dependencies and enable models to learn from historical patterns. They're fundamental for time series analysis, sequence prediction, and creating autoregressive features in ML models.

#### Code Example
```sql
-- Basic Lag Features using LAG() Window Function
SELECT 
    customer_id,
    purchase_date,
    purchase_amount,
    -- Simple lag features
    LAG(purchase_amount, 1) OVER (
        PARTITION BY customer_id 
        ORDER BY purchase_date
    ) AS prev_purchase_amount,
    LAG(purchase_amount, 7) OVER (
        PARTITION BY customer_id 
        ORDER BY purchase_date
    ) AS purchase_amount_7_days_ago,
    LAG(purchase_amount, 30) OVER (
        PARTITION BY customer_id 
        ORDER BY purchase_date
    ) AS purchase_amount_30_days_ago,
    -- Lead features (future values for validation)
    LEAD(purchase_amount, 1) OVER (
        PARTITION BY customer_id 
        ORDER BY purchase_date
    ) AS next_purchase_amount
FROM customer_purchases
ORDER BY customer_id, purchase_date;

-- Multiple Lag Features for ML Feature Engineering
WITH lagged_features AS (
    SELECT 
        product_id,
        sales_date,
        daily_sales,
        -- Historical sales lags
        LAG(daily_sales, 1) OVER (PARTITION BY product_id ORDER BY sales_date) AS sales_lag_1,
        LAG(daily_sales, 2) OVER (PARTITION BY product_id ORDER BY sales_date) AS sales_lag_2,
        LAG(daily_sales, 7) OVER (PARTITION BY product_id ORDER BY sales_date) AS sales_lag_7,
        LAG(daily_sales, 14) OVER (PARTITION BY product_id ORDER BY sales_date) AS sales_lag_14,
        LAG(daily_sales, 30) OVER (PARTITION BY product_id ORDER BY sales_date) AS sales_lag_30,
        -- Seasonal lags (weekly, monthly, yearly)
        LAG(daily_sales, 7) OVER (PARTITION BY product_id ORDER BY sales_date) AS same_day_last_week,
        LAG(daily_sales, 365) OVER (PARTITION BY product_id ORDER BY sales_date) AS same_day_last_year,
        -- Price lags
        LAG(unit_price, 1) OVER (PARTITION BY product_id ORDER BY sales_date) AS price_lag_1,
        LAG(unit_price, 7) OVER (PARTITION BY product_id ORDER BY sales_date) AS price_lag_7
    FROM daily_product_sales
),
derived_features AS (
    SELECT 
        product_id,
        sales_date,
        daily_sales,
        sales_lag_1,
        sales_lag_7,
        sales_lag_30,
        same_day_last_week,
        same_day_last_year,
        price_lag_1,
        price_lag_7,
        -- Derived lag-based features
        daily_sales - sales_lag_1 AS day_over_day_change,
        daily_sales - sales_lag_7 AS week_over_week_change,
        daily_sales - same_day_last_week AS weekly_seasonal_change,
        daily_sales - same_day_last_year AS yearly_seasonal_change,
        -- Percentage changes
        CASE 
            WHEN sales_lag_1 > 0 THEN (daily_sales - sales_lag_1) / sales_lag_1 * 100
            ELSE NULL 
        END AS pct_change_1_day,
        CASE 
            WHEN sales_lag_7 > 0 THEN (daily_sales - sales_lag_7) / sales_lag_7 * 100
            ELSE NULL 
        END AS pct_change_7_day,
        -- Moving trend indicators
        CASE 
            WHEN sales_lag_1 < sales_lag_2 AND daily_sales > sales_lag_1 THEN 'Upward_Trend'
            WHEN sales_lag_1 > sales_lag_2 AND daily_sales < sales_lag_1 THEN 'Downward_Trend'
            ELSE 'Stable'
        END AS trend_direction
    FROM lagged_features
)
SELECT 
    product_id,
    sales_date,
    daily_sales,
    sales_lag_1,
    sales_lag_7,
    sales_lag_30,
    day_over_day_change,
    week_over_week_change,
    weekly_seasonal_change,
    yearly_seasonal_change,
    pct_change_1_day,
    pct_change_7_day,
    trend_direction,
    -- Additional ML features
    CASE 
        WHEN ABS(pct_change_1_day) > 20 THEN 'High_Volatility'
        WHEN ABS(pct_change_1_day) > 10 THEN 'Medium_Volatility'
        ELSE 'Low_Volatility'
    END AS volatility_category
FROM derived_features
WHERE sales_date >= '2024-01-01';

-- Advanced: Variable-Length Lag Features
SELECT 
    user_id,
    session_date,
    page_views,
    session_duration,
    -- Dynamic lag based on business logic
    LAG(page_views, 
        CASE 
            WHEN EXTRACT(DOW FROM session_date) = 1 THEN 3  -- Monday looks back to Friday
            ELSE 1  -- Other days look back 1 day
        END
    ) OVER (PARTITION BY user_id ORDER BY session_date) AS previous_session_views,
    
    -- Multiple lag horizons for ensemble features
    LAG(session_duration, 1) OVER (PARTITION BY user_id ORDER BY session_date) AS duration_lag_1,
    LAG(session_duration, 3) OVER (PARTITION BY user_id ORDER BY session_date) AS duration_lag_3,
    LAG(session_duration, 7) OVER (PARTITION BY user_id ORDER BY session_date) AS duration_lag_7,
    
    -- Lag with aggregation
    AVG(LAG(page_views, 1) OVER (PARTITION BY user_id ORDER BY session_date)) OVER (
        PARTITION BY user_id 
        ORDER BY session_date 
        ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
    ) AS avg_previous_7_sessions
FROM user_sessions
ORDER BY user_id, session_date;

-- Time-based Lag Features with Date Logic
WITH time_based_lags AS (
    SELECT 
        stock_symbol,
        trade_date,
        closing_price,
        volume,
        -- Exact time-based lags
        LAG(closing_price, 1) OVER (
            PARTITION BY stock_symbol 
            ORDER BY trade_date
        ) AS price_1_day_ago,
        
        -- Self-join approach for specific date lags
        prev_week.closing_price AS price_1_week_ago,
        prev_month.closing_price AS price_1_month_ago,
        prev_year.closing_price AS price_1_year_ago
    FROM stock_prices sp
    LEFT JOIN stock_prices prev_week ON 
        sp.stock_symbol = prev_week.stock_symbol 
        AND prev_week.trade_date = sp.trade_date - INTERVAL 7 DAY
    LEFT JOIN stock_prices prev_month ON 
        sp.stock_symbol = prev_month.stock_symbol 
        AND prev_month.trade_date = sp.trade_date - INTERVAL 1 MONTH
    LEFT JOIN stock_prices prev_year ON 
        sp.stock_symbol = prev_year.stock_symbol 
        AND prev_year.trade_date = sp.trade_date - INTERVAL 1 YEAR
)
SELECT 
    stock_symbol,
    trade_date,
    closing_price,
    price_1_day_ago,
    price_1_week_ago,
    price_1_month_ago,
    price_1_year_ago,
    -- Technical indicators from lag features
    closing_price - price_1_day_ago AS daily_change,
    (closing_price - price_1_week_ago) / price_1_week_ago * 100 AS weekly_return,
    (closing_price - price_1_month_ago) / price_1_month_ago * 100 AS monthly_return,
    (closing_price - price_1_year_ago) / price_1_year_ago * 100 AS annual_return
FROM time_based_lags
WHERE trade_date >= '2024-01-01';
```

#### Explanation
1. **LAG Function**: `LAG(column, offset)` retrieves value from previous row
2. **PARTITION BY**: Ensures lags are calculated within logical groups
3. **ORDER BY**: Defines the sequence for lag calculations
4. **Offset Parameter**: Specifies how many rows back to look
5. **NULL Handling**: Early rows may have NULL lag values

#### Use Cases
- **Time Series Forecasting**: Create autoregressive features for prediction models
- **Customer Behavior**: Track changes in purchasing patterns over time
- **Anomaly Detection**: Compare current values with historical baselines
- **Market Analysis**: Technical indicators using price and volume lags
- **Cohort Analysis**: Compare user behavior across different time periods

#### Best Practices
- Choose lag periods based on business cycles and data frequency
- Create multiple lag horizons (1-day, 1-week, 1-month, 1-year)
- Handle NULL values appropriately for ML models
- Consider seasonal lags (same day last week/month/year)
- Combine with moving averages for robust feature sets

#### Pitfalls
- **Data Gaps**: Missing dates can create incorrect lag relationships
- **Boundary Effects**: Early observations will have NULL lag values
- **Computation Cost**: Multiple lag features can be computationally expensive
- **Overfitting**: Too many lag features may lead to model overfitting
- **Leakage**: Ensure no future information leaks into lag features

#### Debugging
```sql
-- Validate lag feature calculations
SELECT 
    product_id,
    sales_date,
    daily_sales,
    sales_lag_1,
    -- Verify lag alignment
    LAG(sales_date, 1) OVER (PARTITION BY product_id ORDER BY sales_date) AS prev_date,
    sales_date - LAG(sales_date, 1) OVER (PARTITION BY product_id ORDER BY sales_date) AS date_gap
FROM product_sales_with_lags
WHERE product_id = 'PROD001'
ORDER BY sales_date DESC
LIMIT 10;

-- Check for data completeness
SELECT 
    product_id,
    COUNT(*) AS total_records,
    COUNT(sales_lag_1) AS lag_1_records,
    COUNT(sales_lag_7) AS lag_7_records,
    COUNT(sales_lag_30) AS lag_30_records
FROM product_sales_with_lags
GROUP BY product_id;
```

#### Optimization
- **Indexing**: Create indexes on partition and order columns
- **Window Size**: Limit window calculations to necessary ranges
- **Materialized Views**: Pre-calculate lag features for large datasets
- **Incremental Processing**: Update only new lag values instead of recalculating all

---

## Question 5

**Describe how to compute a ratio feature within groups using SQL.**

### Answer:

#### Theory
Ratio features normalize values relative to group totals, enabling meaningful comparisons across different scales and contexts. They're essential for creating features like market share, percentage contribution, relative performance metrics, and normalized behavioral indicators in ML models.

#### Code Example
```sql
-- Basic Ratio Features using Window Functions
SELECT 
    customer_id,
    product_category,
    purchase_amount,
    purchase_date,
    -- Ratio of individual purchase to customer's total spending
    purchase_amount / SUM(purchase_amount) OVER (
        PARTITION BY customer_id
    ) AS share_of_customer_total,
    
    -- Ratio within product category
    purchase_amount / SUM(purchase_amount) OVER (
        PARTITION BY product_category
    ) AS share_of_category_total,
    
    -- Ratio within time period
    purchase_amount / SUM(purchase_amount) OVER (
        PARTITION BY DATE_TRUNC('month', purchase_date)
    ) AS share_of_monthly_total,
    
    -- Multiple group ratios
    purchase_amount / SUM(purchase_amount) OVER (
        PARTITION BY customer_id, product_category
    ) AS share_of_customer_category_total
FROM customer_purchases
ORDER BY customer_id, purchase_date;

-- Advanced Ratio Features for ML
WITH customer_metrics AS (
    SELECT 
        customer_id,
        product_category,
        region,
        purchase_date,
        purchase_amount,
        quantity_purchased,
        -- Customer-level ratios
        purchase_amount / SUM(purchase_amount) OVER (
            PARTITION BY customer_id
        ) AS customer_spend_ratio,
        
        -- Category-level ratios
        purchase_amount / AVG(purchase_amount) OVER (
            PARTITION BY product_category
        ) AS vs_category_average_ratio,
        
        -- Regional ratios
        purchase_amount / SUM(purchase_amount) OVER (
            PARTITION BY region, product_category
        ) AS regional_category_share,
        
        -- Time-based ratios
        purchase_amount / SUM(purchase_amount) OVER (
            PARTITION BY customer_id, DATE_TRUNC('quarter', purchase_date)
        ) AS quarterly_spend_ratio,
        
        -- Quantity vs amount ratios
        purchase_amount / NULLIF(quantity_purchased, 0) AS avg_unit_price,
        quantity_purchased / SUM(quantity_purchased) OVER (
            PARTITION BY customer_id, product_category
        ) AS quantity_share_in_category
    FROM sales_transactions
),
enriched_ratios AS (
    SELECT 
        customer_id,
        product_category,
        region,
        purchase_date,
        purchase_amount,
        customer_spend_ratio,
        vs_category_average_ratio,
        regional_category_share,
        quarterly_spend_ratio,
        avg_unit_price,
        quantity_share_in_category,
        
        -- Derived ratio features
        CASE 
            WHEN customer_spend_ratio > 0.3 THEN 'High_Category_Concentration'
            WHEN customer_spend_ratio > 0.1 THEN 'Medium_Category_Concentration'
            ELSE 'Low_Category_Concentration'
        END AS category_concentration_level,
        
        CASE 
            WHEN vs_category_average_ratio > 2.0 THEN 'High_Spender'
            WHEN vs_category_average_ratio > 1.5 THEN 'Above_Average_Spender'
            WHEN vs_category_average_ratio > 0.5 THEN 'Average_Spender'
            ELSE 'Below_Average_Spender'
        END AS spending_tier,
        
        -- Relative performance indicators
        vs_category_average_ratio * regional_category_share AS composite_performance_score
    FROM customer_metrics
)
SELECT 
    customer_id,
    product_category,
    purchase_amount,
    customer_spend_ratio,
    vs_category_average_ratio,
    regional_category_share,
    category_concentration_level,
    spending_tier,
    composite_performance_score
FROM enriched_ratios
WHERE purchase_date >= '2024-01-01';

-- Time-series Ratio Features
SELECT 
    product_id,
    sales_month,
    monthly_sales,
    monthly_units_sold,
    
    -- Year-over-year ratios
    monthly_sales / LAG(monthly_sales, 12) OVER (
        PARTITION BY product_id 
        ORDER BY sales_month
    ) AS yoy_sales_ratio,
    
    -- Month-over-month ratios
    monthly_sales / LAG(monthly_sales, 1) OVER (
        PARTITION BY product_id 
        ORDER BY sales_month
    ) AS mom_sales_ratio,
    
    -- Market share ratios
    monthly_sales / SUM(monthly_sales) OVER (
        PARTITION BY sales_month
    ) AS market_share,
    
    -- Category performance ratios
    monthly_sales / SUM(monthly_sales) OVER (
        PARTITION BY product_category, sales_month
    ) AS category_share,
    
    -- Rolling average ratios
    monthly_sales / AVG(monthly_sales) OVER (
        PARTITION BY product_id 
        ORDER BY sales_month 
        ROWS BETWEEN 11 PRECEDING AND 1 PRECEDING
    ) AS vs_trailing_12_avg_ratio,
    
    -- Efficiency ratios
    monthly_sales / NULLIF(monthly_units_sold, 0) AS revenue_per_unit,
    monthly_units_sold / SUM(monthly_units_sold) OVER (
        PARTITION BY product_category, sales_month
    ) AS unit_market_share
FROM monthly_product_sales
ORDER BY product_id, sales_month;

-- Multi-dimensional Ratio Analysis
WITH hierarchical_ratios AS (
    SELECT 
        employee_id,
        department,
        job_level,
        region,
        performance_score,
        salary,
        
        -- Department ratios
        performance_score / AVG(performance_score) OVER (
            PARTITION BY department
        ) AS vs_dept_avg_performance,
        
        salary / AVG(salary) OVER (
            PARTITION BY department
        ) AS vs_dept_avg_salary,
        
        -- Job level ratios
        performance_score / AVG(performance_score) OVER (
            PARTITION BY job_level
        ) AS vs_level_avg_performance,
        
        salary / AVG(salary) OVER (
            PARTITION BY job_level
        ) AS vs_level_avg_salary,
        
        -- Regional ratios
        salary / AVG(salary) OVER (
            PARTITION BY region
        ) AS vs_regional_avg_salary,
        
        -- Combined context ratios
        salary / AVG(salary) OVER (
            PARTITION BY department, job_level
        ) AS vs_dept_level_avg_salary,
        
        -- Percentile-based ratios
        PERCENT_RANK() OVER (
            PARTITION BY department 
            ORDER BY performance_score
        ) AS dept_performance_percentile,
        
        PERCENT_RANK() OVER (
            PARTITION BY job_level 
            ORDER BY salary
        ) AS level_salary_percentile
    FROM employee_data
)
SELECT 
    employee_id,
    department,
    job_level,
    performance_score,
    salary,
    vs_dept_avg_performance,
    vs_dept_avg_salary,
    vs_level_avg_performance,
    vs_level_avg_salary,
    vs_regional_avg_salary,
    vs_dept_level_avg_salary,
    dept_performance_percentile,
    level_salary_percentile,
    
    -- Complex ratio features for ML
    (vs_dept_avg_performance + vs_level_avg_performance) / 2 AS avg_relative_performance,
    vs_dept_avg_salary / vs_level_avg_salary AS dept_vs_level_salary_ratio,
    
    -- Outlier detection using ratios
    CASE 
        WHEN vs_dept_avg_performance > 1.5 AND vs_dept_avg_salary < 0.8 THEN 'Undervalued_High_Performer'
        WHEN vs_dept_avg_performance < 0.7 AND vs_dept_avg_salary > 1.2 THEN 'Overpaid_Low_Performer'
        ELSE 'Balanced'
    END AS performance_salary_alignment
FROM hierarchical_ratios;
```

#### Explanation
1. **Window Functions**: Use SUM(), AVG(), COUNT() with OVER clause for group calculations
2. **PARTITION BY**: Defines the groups for ratio calculations
3. **Multiple Partitions**: Create ratios across different dimensional contexts
4. **NULLIF**: Prevents division by zero errors
5. **Percentile Functions**: PERCENT_RANK() for relative positioning

#### Use Cases
- **Market Share Analysis**: Calculate product/company share within markets
- **Customer Segmentation**: Identify high-value customers relative to peers
- **Performance Metrics**: Normalize metrics across different business units
- **Feature Engineering**: Create relative features for ML models
- **Anomaly Detection**: Identify outliers based on relative performance

#### Best Practices
- Always handle division by zero with NULLIF or CASE statements
- Choose appropriate group dimensions based on business logic
- Consider multiple ratio perspectives (absolute vs relative)
- Use meaningful variable names for complex ratio calculations
- Validate ratio ranges to ensure they make business sense

#### Pitfalls
- **Division by Zero**: Groups with zero totals cause calculation errors
- **Small Denominators**: Very small group totals can create misleading ratios
- **Temporal Misalignment**: Ensure time periods are correctly aligned
- **Sparse Data**: Groups with few observations may have unreliable ratios
- **Scale Sensitivity**: Very large or small ratios may need transformation

#### Debugging
```sql
-- Validate ratio calculations
SELECT 
    customer_id,
    SUM(purchase_amount) AS total_customer_spend,
    SUM(share_of_customer_total) AS sum_of_ratios,  -- Should equal 1.0
    COUNT(*) AS transaction_count
FROM customer_purchase_ratios
GROUP BY customer_id
HAVING ABS(SUM(share_of_customer_total) - 1.0) > 0.01;

-- Check for division by zero cases
SELECT 
    product_category,
    COUNT(*) AS record_count,
    SUM(purchase_amount) AS category_total,
    AVG(vs_category_average_ratio) AS avg_ratio
FROM category_ratios
GROUP BY product_category
ORDER BY category_total;
```

#### Optimization
- **Indexing**: Index partition columns for efficient window operations
- **Pre-aggregation**: Calculate group totals in advance for complex ratios
- **Materialized Views**: Store frequently used ratio calculations
- **Query Rewriting**: Use CTEs to avoid recalculating the same window functions

---

## Question 6

**Write a SQL query that identifies and removes duplicate records from a dataset.**

### Answer:

#### Theory
Duplicate removal is crucial for data quality in ML pipelines. Duplicates can skew model training, create data leakage, and produce biased results. SQL provides multiple approaches: window functions with ROW_NUMBER(), DISTINCT operations, and self-joins for complex deduplication logic.

#### Code Example
```sql
-- Method 1: Using ROW_NUMBER() for Complete Deduplication
WITH deduplicated_data AS (
    SELECT 
        customer_id,
        email,
        first_name,
        last_name,
        registration_date,
        ROW_NUMBER() OVER (
            PARTITION BY email 
            ORDER BY registration_date DESC, customer_id DESC
        ) AS row_num
    FROM customers
)
SELECT 
    customer_id,
    email,
    first_name,
    last_name,
    registration_date
FROM deduplicated_data
WHERE row_num = 1;

-- Method 2: Identifying Duplicates with Window Functions
WITH duplicate_analysis AS (
    SELECT 
        transaction_id,
        customer_id,
        product_id,
        transaction_date,
        amount,
        -- Count duplicates based on business logic
        COUNT(*) OVER (
            PARTITION BY customer_id, product_id, transaction_date, amount
        ) AS duplicate_count,
        -- Rank duplicates by most recent or reliable source
        ROW_NUMBER() OVER (
            PARTITION BY customer_id, product_id, transaction_date, amount
            ORDER BY transaction_id DESC  -- Keep the latest transaction_id
        ) AS dedup_rank,
        -- Flag exact duplicates
        CASE 
            WHEN COUNT(*) OVER (
                PARTITION BY customer_id, product_id, transaction_date, amount
            ) > 1 THEN 'Duplicate'
            ELSE 'Unique'
        END AS duplicate_flag
    FROM transactions
)
SELECT 
    transaction_id,
    customer_id,
    product_id,
    transaction_date,
    amount,
    duplicate_count,
    duplicate_flag
FROM duplicate_analysis
WHERE dedup_rank = 1  -- Keep only the first occurrence
ORDER BY customer_id, transaction_date;

-- Method 3: Fuzzy Deduplication for Similar Records
WITH fuzzy_duplicates AS (
    SELECT 
        c1.customer_id AS id1,
        c2.customer_id AS id2,
        c1.first_name,
        c1.last_name,
        c1.email AS email1,
        c2.email AS email2,
        -- Calculate similarity scores
        CASE 
            WHEN LOWER(c1.email) = LOWER(c2.email) THEN 100
            WHEN LEVENSHTEIN(LOWER(c1.email), LOWER(c2.email)) <= 2 THEN 80
            ELSE 0
        END AS email_similarity,
        CASE 
            WHEN LOWER(CONCAT(c1.first_name, c1.last_name)) = 
                 LOWER(CONCAT(c2.first_name, c2.last_name)) THEN 100
            WHEN LEVENSHTEIN(
                LOWER(CONCAT(c1.first_name, c1.last_name)), 
                LOWER(CONCAT(c2.first_name, c2.last_name))
            ) <= 3 THEN 70
            ELSE 0
        END AS name_similarity
    FROM customers c1
    JOIN customers c2 ON c1.customer_id < c2.customer_id  -- Avoid self-joins and duplicates
    WHERE 
        -- Quick filters to reduce comparison space
        (LOWER(c1.email) = LOWER(c2.email) OR
         LOWER(c1.first_name) = LOWER(c2.first_name) OR
         LOWER(c1.last_name) = LOWER(c2.last_name))
),
potential_duplicates AS (
    SELECT 
        id1,
        id2,
        email1,
        email2,
        email_similarity,
        name_similarity,
        (email_similarity + name_similarity) / 2 AS composite_similarity
    FROM fuzzy_duplicates
    WHERE (email_similarity + name_similarity) / 2 >= 70  -- Threshold for likely duplicates
),
deduplication_decisions AS (
    SELECT 
        id1,
        id2,
        composite_similarity,
        -- Decision logic: keep the record with more complete data
        CASE 
            WHEN c1.email IS NOT NULL AND c2.email IS NULL THEN id1
            WHEN c1.email IS NULL AND c2.email IS NOT NULL THEN id2
            WHEN c1.registration_date < c2.registration_date THEN id1  -- Keep older record
            ELSE id2
        END AS keep_id,
        CASE 
            WHEN c1.email IS NOT NULL AND c2.email IS NULL THEN id2
            WHEN c1.email IS NULL AND c2.email IS NOT NULL THEN id1
            WHEN c1.registration_date < c2.registration_date THEN id2
            ELSE id1
        END AS remove_id
    FROM potential_duplicates pd
    JOIN customers c1 ON pd.id1 = c1.customer_id
    JOIN customers c2 ON pd.id2 = c2.customer_id
)
-- Final clean dataset
SELECT c.*
FROM customers c
WHERE c.customer_id NOT IN (SELECT remove_id FROM deduplication_decisions);

-- Method 4: Advanced Deduplication with Data Quality Scoring
WITH data_quality_scores AS (
    SELECT 
        customer_id,
        email,
        first_name,
        last_name,
        phone,
        address,
        registration_date,
        last_activity_date,
        -- Calculate data completeness score
        (
            CASE WHEN email IS NOT NULL AND email != '' THEN 25 ELSE 0 END +
            CASE WHEN phone IS NOT NULL AND phone != '' THEN 20 ELSE 0 END +
            CASE WHEN address IS NOT NULL AND address != '' THEN 15 ELSE 0 END +
            CASE WHEN first_name IS NOT NULL AND first_name != '' THEN 20 ELSE 0 END +
            CASE WHEN last_name IS NOT NULL AND last_name != '' THEN 20 ELSE 0 END
        ) AS completeness_score,
        -- Calculate data freshness score
        CASE 
            WHEN last_activity_date >= CURRENT_DATE - INTERVAL 30 DAY THEN 30
            WHEN last_activity_date >= CURRENT_DATE - INTERVAL 90 DAY THEN 20
            WHEN last_activity_date >= CURRENT_DATE - INTERVAL 365 DAY THEN 10
            ELSE 0
        END AS freshness_score
    FROM customers
),
ranked_duplicates AS (
    SELECT 
        customer_id,
        email,
        first_name,
        last_name,
        phone,
        address,
        registration_date,
        completeness_score,
        freshness_score,
        (completeness_score + freshness_score) AS total_quality_score,
        -- Rank within duplicate groups by quality
        ROW_NUMBER() OVER (
            PARTITION BY LOWER(email)
            ORDER BY (completeness_score + freshness_score) DESC, 
                     registration_date ASC
        ) AS quality_rank
    FROM data_quality_scores
    WHERE email IS NOT NULL AND email != ''
)
SELECT 
    customer_id,
    email,
    first_name,
    last_name,
    phone,
    address,
    registration_date,
    total_quality_score
FROM ranked_duplicates
WHERE quality_rank = 1  -- Keep highest quality record per email
ORDER BY total_quality_score DESC;

-- Method 5: Deduplication with Audit Trail
CREATE TABLE duplicate_removal_log (
    removal_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    removed_id INT,
    kept_id INT,
    removal_reason VARCHAR(200),
    similarity_score DECIMAL(5,2)
);

-- Remove duplicates and log the actions
WITH dedup_with_logging AS (
    SELECT 
        customer_id,
        email,
        ROW_NUMBER() OVER (
            PARTITION BY LOWER(email) 
            ORDER BY last_activity_date DESC, customer_id ASC
        ) AS keep_rank,
        FIRST_VALUE(customer_id) OVER (
            PARTITION BY LOWER(email) 
            ORDER BY last_activity_date DESC, customer_id ASC
        ) AS primary_id
    FROM customers
    WHERE email IS NOT NULL
)
-- Insert removal log entries
INSERT INTO duplicate_removal_log (removed_id, kept_id, removal_reason)
SELECT 
    customer_id AS removed_id,
    primary_id AS kept_id,
    'Email duplicate - kept most recent activity' AS removal_reason
FROM dedup_with_logging
WHERE keep_rank > 1;

-- Keep only the deduplicated records
DELETE FROM customers 
WHERE customer_id IN (
    SELECT removed_id FROM duplicate_removal_log
);
```

#### Explanation
1. **ROW_NUMBER()**: Assigns sequential numbers within partitions for ranking duplicates
2. **PARTITION BY**: Defines duplicate groups based on business logic
3. **ORDER BY**: Determines which record to keep (newest, most complete, etc.)
4. **Self-joins**: Compare records within the same table for similarity
5. **Quality Scoring**: Prioritize records based on data completeness and freshness

#### Use Cases
- **Customer Data**: Remove duplicate customer registrations
- **Transaction Cleaning**: Eliminate duplicate payment records
- **Product Catalogs**: Consolidate similar product entries
- **ML Data Preparation**: Ensure training data quality
- **Data Integration**: Merge data from multiple sources

#### Best Practices
- Define clear business rules for what constitutes a duplicate
- Consider partial matches and fuzzy deduplication for real-world data
- Preserve audit trails of deduplication decisions
- Test deduplication logic on sample data before production runs
- Monitor duplicate rates over time to identify data quality issues

#### Pitfalls
- **Over-deduplication**: Removing legitimate similar records
- **Under-deduplication**: Missing subtle duplicates due to data variations
- **Performance Issues**: Large cartesian products in similarity comparisons
- **Data Loss**: Accidentally removing important information
- **Incomplete Logic**: Not accounting for all duplicate scenarios

#### Debugging
```sql
-- Analyze duplicate patterns
SELECT 
    'By Email' AS duplicate_type,
    COUNT(*) AS total_records,
    COUNT(DISTINCT email) AS unique_emails,
    COUNT(*) - COUNT(DISTINCT email) AS duplicates
FROM customers
WHERE email IS NOT NULL
UNION ALL
SELECT 
    'By Phone' AS duplicate_type,
    COUNT(*) AS total_records,
    COUNT(DISTINCT phone) AS unique_phones,
    COUNT(*) - COUNT(DISTINCT phone) AS duplicates
FROM customers
WHERE phone IS NOT NULL;

-- Validate deduplication results
SELECT 
    email,
    COUNT(*) AS remaining_count
FROM customers_deduplicated
GROUP BY email
HAVING COUNT(*) > 1;
```

#### Optimization
- **Indexing**: Create indexes on columns used for duplicate detection
- **Batch Processing**: Process large datasets in chunks
- **Parallel Processing**: Use parallel execution for independent duplicate groups
- **Approximate Matching**: Use efficient similarity algorithms for fuzzy matching

---

## Question 7

**In SQL, how would you format strings or concatenate columns for text-based Machine Learning features?**

### Answer:

#### Theory
String manipulation and concatenation are fundamental for creating text features in ML models. This includes cleaning text data, normalizing formats, creating composite features, and preparing text for NLP algorithms like TF-IDF, word embeddings, or sentiment analysis.

#### Code Example
```sql
-- Basic String Concatenation and Formatting
SELECT 
    customer_id,
    first_name,
    last_name,
    email,
    -- Standard concatenation
    CONCAT(first_name, ' ', last_name) AS full_name,
    CONCAT(first_name, '.', last_name, '@company.com') AS normalized_email,
    
    -- Concatenation with null handling
    COALESCE(CONCAT(first_name, ' ', last_name), first_name, last_name, 'Unknown') AS safe_full_name,
    
    -- SQL Server style concatenation
    first_name + ' ' + ISNULL(middle_name + ' ', '') + last_name AS full_name_with_middle,
    
    -- Formatting for ML features
    UPPER(CONCAT(SUBSTRING(first_name, 1, 1), SUBSTRING(last_name, 1, 1))) AS initials,
    LOWER(REPLACE(REPLACE(email, '.', ''), '@', '_at_')) AS email_normalized
FROM customers;

-- Advanced Text Feature Engineering
WITH text_features AS (
    SELECT 
        review_id,
        product_id,
        customer_id,
        review_title,
        review_text,
        rating,
        
        -- Text cleaning and normalization
        LOWER(TRIM(review_text)) AS cleaned_text,
        REGEXP_REPLACE(
            LOWER(TRIM(review_text)), 
            '[^a-z0-9\s]', 
            ' '
        ) AS alphanumeric_only,
        
        -- Length-based features
        LENGTH(review_text) AS text_length,
        LENGTH(review_text) - LENGTH(REPLACE(review_text, ' ', '')) + 1 AS word_count,
        ARRAY_LENGTH(SPLIT(review_text, ' '), 1) AS word_count_alt,
        
        -- Sentiment and keyword features
        CASE 
            WHEN LOWER(review_text) LIKE '%excellent%' OR LOWER(review_text) LIKE '%amazing%' 
                 OR LOWER(review_text) LIKE '%perfect%' THEN 'positive_keywords'
            WHEN LOWER(review_text) LIKE '%terrible%' OR LOWER(review_text) LIKE '%awful%' 
                 OR LOWER(review_text) LIKE '%worst%' THEN 'negative_keywords'
            ELSE 'neutral_keywords'
        END AS keyword_sentiment,
        
        -- Text pattern features
        LENGTH(review_text) - LENGTH(REPLACE(UPPER(review_text), 'A', '')) AS letter_a_count,
        (LENGTH(review_text) - LENGTH(REPLACE(review_text, '!', ''))) AS exclamation_count,
        (LENGTH(review_text) - LENGTH(REPLACE(review_text, '?', ''))) AS question_count,
        
        -- Composite text features
        CONCAT(
            CAST(rating AS STRING), '_',
            CASE 
                WHEN LENGTH(review_text) > 500 THEN 'long'
                WHEN LENGTH(review_text) > 100 THEN 'medium'
                ELSE 'short'
            END, '_',
            CASE 
                WHEN LOWER(review_text) LIKE '%recommend%' THEN 'recommends'
                ELSE 'no_recommendation'
            END
        ) AS composite_review_feature
    FROM product_reviews
),
enhanced_features AS (
    SELECT 
        review_id,
        product_id,
        customer_id,
        cleaned_text,
        text_length,
        word_count,
        keyword_sentiment,
        exclamation_count,
        question_count,
        composite_review_feature,
        
        -- N-gram features (bigrams from title and text)
        CONCAT(
            SPLIT(LOWER(review_title), ' ')[SAFE_OFFSET(0)], '_',
            SPLIT(LOWER(review_title), ' ')[SAFE_OFFSET(1)]
        ) AS title_bigram_1,
        
        -- Text statistical features
        CAST(text_length AS FLOAT64) / NULLIF(word_count, 0) AS avg_word_length,
        CAST(exclamation_count AS FLOAT64) / NULLIF(text_length, 0) * 1000 AS exclamation_density,
        
        -- Regular expression features
        REGEXP_CONTAINS(LOWER(cleaned_text), r'\b(buy|purchase|order)\b') AS contains_purchase_intent,
        REGEXP_CONTAINS(LOWER(cleaned_text), r'\b(return|refund|exchange)\b') AS contains_return_intent,
        
        -- Text preprocessing for ML
        REGEXP_REPLACE(
            REGEXP_REPLACE(cleaned_text, r'\s+', ' '),  -- Multiple spaces to single
            r'[^\w\s]', ''  -- Remove punctuation
        ) AS ml_ready_text
    FROM text_features
)
SELECT 
    review_id,
    product_id,
    -- Final feature set for ML model
    text_length,
    word_count,
    avg_word_length,
    exclamation_density,
    keyword_sentiment,
    contains_purchase_intent,
    contains_return_intent,
    composite_review_feature,
    title_bigram_1,
    ml_ready_text
FROM enhanced_features;

-- Text Aggregation for Customer Profiles
SELECT 
    customer_id,
    -- Concatenate all reviews for customer profiling
    STRING_AGG(DISTINCT product_category, ', ' ORDER BY product_category) AS categories_reviewed,
    STRING_AGG(review_text, ' | ' ORDER BY review_date DESC LIMIT 5) AS recent_reviews,
    
    -- Text-based behavioral features
    AVG(LENGTH(review_text)) AS avg_review_length,
    COUNT(CASE WHEN LOWER(review_text) LIKE '%recommend%' THEN 1 END) AS recommendation_count,
    COUNT(CASE WHEN rating >= 4 THEN 1 END) AS positive_review_count,
    
    -- Create customer text signature
    CONCAT(
        'customer_',
        COUNT(*), '_reviews_',
        ROUND(AVG(rating), 1), '_avg_rating_',
        CASE 
            WHEN AVG(LENGTH(review_text)) > 200 THEN 'verbose'
            WHEN AVG(LENGTH(review_text)) > 50 THEN 'moderate'
            ELSE 'brief'
        END, '_reviewer'
    ) AS customer_profile_signature,
    
    -- Most common words in reviews (simplified)
    SPLIT(STRING_AGG(LOWER(review_text), ' '), ' ') AS all_words
FROM product_reviews
GROUP BY customer_id;

-- Advanced Text Processing for NLP Features
WITH tokenized_reviews AS (
    SELECT 
        review_id,
        customer_id,
        product_id,
        review_text,
        -- Tokenization
        SPLIT(
            REGEXP_REPLACE(
                LOWER(TRIM(review_text)), 
                r'[^\w\s]', 
                ' '
            ), 
            ' '
        ) AS tokens,
        -- Remove stop words (simplified list)
        ARRAY(
            SELECT token 
            FROM UNNEST(SPLIT(
                REGEXP_REPLACE(LOWER(TRIM(review_text)), r'[^\w\s]', ' '), 
                ' '
            )) AS token
            WHERE token NOT IN ('the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should')
            AND LENGTH(token) > 2
        ) AS filtered_tokens
    FROM product_reviews
),
term_frequency AS (
    SELECT 
        review_id,
        token,
        COUNT(*) AS term_freq
    FROM tokenized_reviews, UNNEST(filtered_tokens) AS token
    GROUP BY review_id, token
),
document_frequency AS (
    SELECT 
        token,
        COUNT(DISTINCT review_id) AS doc_freq,
        COUNT(*) AS total_freq
    FROM term_frequency
    GROUP BY token
),
tfidf_features AS (
    SELECT 
        tf.review_id,
        tf.token,
        tf.term_freq,
        df.doc_freq,
        -- Simple TF-IDF calculation
        tf.term_freq * LOG((SELECT COUNT(DISTINCT review_id) FROM tokenized_reviews) / df.doc_freq) AS tfidf_score
    FROM term_frequency tf
    JOIN document_frequency df ON tf.token = df.token
    WHERE df.doc_freq >= 5  -- Filter rare terms
)
SELECT 
    review_id,
    -- Top TF-IDF terms per review
    STRING_AGG(
        CONCAT(token, ':', ROUND(tfidf_score, 3)), 
        ', ' 
        ORDER BY tfidf_score DESC 
        LIMIT 10
    ) AS top_tfidf_terms
FROM tfidf_features
GROUP BY review_id;
```

#### Explanation
1. **CONCAT Function**: Combines multiple strings with optional separators
2. **String Functions**: UPPER, LOWER, TRIM, SUBSTRING for text manipulation
3. **Regular Expressions**: REGEXP_REPLACE, REGEXP_CONTAINS for pattern matching
4. **Null Handling**: COALESCE, ISNULL for safe string operations
5. **Array Functions**: SPLIT, STRING_AGG for tokenization and aggregation

#### Use Cases
- **Customer Profiling**: Create unified customer identifiers and signatures
- **NLP Feature Engineering**: Prepare text data for machine learning models
- **Data Standardization**: Normalize text formats across different sources
- **Search Optimization**: Create searchable text indices and keywords
- **Content Analysis**: Extract meaningful patterns from unstructured text

#### Best Practices
- Always handle NULL values in string concatenation
- Use consistent text normalization (case, punctuation, whitespace)
- Consider character encoding issues in multilingual text
- Validate string length limits for target systems
- Use appropriate collation for text comparisons

#### Pitfalls
- **Performance Issues**: String operations can be CPU-intensive on large datasets
- **Memory Usage**: Concatenating large text fields consumes significant memory
- **Character Limits**: Some databases have string length limitations
- **Encoding Problems**: Mixed character encodings can cause corruption
- **Null Propagation**: NULL values in concatenation can result in NULL output

#### Debugging
```sql
-- Check for text data quality issues
SELECT 
    'Null Values' AS issue_type,
    COUNT(CASE WHEN review_text IS NULL THEN 1 END) AS count
FROM product_reviews
UNION ALL
SELECT 
    'Empty Strings',
    COUNT(CASE WHEN TRIM(review_text) = '' THEN 1 END)
FROM product_reviews
UNION ALL
SELECT 
    'Very Long Texts',
    COUNT(CASE WHEN LENGTH(review_text) > 10000 THEN 1 END)
FROM product_reviews;

-- Validate string concatenation results
SELECT 
    customer_id,
    first_name,
    last_name,
    CONCAT(first_name, ' ', last_name) AS concatenated,
    LENGTH(CONCAT(first_name, ' ', last_name)) AS concat_length
FROM customers
WHERE first_name IS NOT NULL AND last_name IS NOT NULL
LIMIT 10;
```

#### Optimization
- **Indexing**: Create functional indexes on commonly used string expressions
- **Materialized Columns**: Store computed string values for frequent access
- **Text Search**: Use full-text search capabilities for efficient text queries
- **Chunking**: Process large text datasets in manageable batches

---

## Question 8

**Write a SQL stored procedure that calls a Machine Learning scoring function.**

### Answer:

#### Theory
SQL stored procedures can integrate ML models directly into database operations, enabling real-time scoring, batch predictions, and automated model deployment. This approach reduces data movement, improves latency, and creates seamless ML-enabled applications.

#### Code Example
```sql
-- SQL Server ML Services Example
CREATE PROCEDURE sp_predict_customer_churn
    @customer_id INT = NULL,
    @batch_mode BIT = 0,
    @model_version VARCHAR(20) = 'latest'
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @sql NVARCHAR(MAX);
    DECLARE @model_data VARBINARY(MAX);
    DECLARE @input_data NVARCHAR(MAX);
    
    -- Retrieve the trained model from model store
    SELECT @model_data = model_object 
    FROM ml_models 
    WHERE model_name = 'customer_churn_rf' 
      AND model_version = @model_version
      AND is_active = 1;
    
    IF @model_data IS NULL
    BEGIN
        RAISERROR('Model not found or inactive', 16, 1);
        RETURN;
    END
    
    -- Prepare feature data
    IF @batch_mode = 1
    BEGIN
        -- Batch prediction for all active customers
        WITH customer_features AS (
            SELECT 
                c.customer_id,
                c.account_age_days,
                c.total_purchases,
                c.avg_order_value,
                c.days_since_last_purchase,
                c.support_tickets_count,
                c.payment_failures_count,
                CAST(c.is_premium AS INT) AS is_premium
            FROM customer_features_view c
            WHERE c.is_active = 1
        )
        SELECT @input_data = (
            SELECT customer_id, account_age_days, total_purchases, avg_order_value,
                   days_since_last_purchase, support_tickets_count, payment_failures_count, is_premium
            FROM customer_features
            FOR JSON AUTO
        );
    END
    ELSE
    BEGIN
        -- Single customer prediction
        SELECT @input_data = (
            SELECT customer_id, account_age_days, total_purchases, avg_order_value,
                   days_since_last_purchase, support_tickets_count, payment_failures_count, is_premium
            FROM customer_features_view
            WHERE customer_id = @customer_id
            FOR JSON AUTO
        );
    END
    
    -- Execute R/Python script for prediction
    SET @sql = N'
    import pandas as pd
    import pickle
    import json
    from sklearn.ensemble import RandomForestClassifier
    
    # Load model
    model = pickle.loads(model_data)
    
    # Parse input data
    input_df = pd.read_json(input_data)
    
    # Feature engineering
    feature_columns = ["account_age_days", "total_purchases", "avg_order_value", 
                      "days_since_last_purchase", "support_tickets_count", 
                      "payment_failures_count", "is_premium"]
    
    X = input_df[feature_columns]
    
    # Make predictions
    churn_probability = model.predict_proba(X)[:, 1]
    churn_prediction = model.predict(X)
    
    # Prepare results
    results = input_df.copy()
    results["churn_probability"] = churn_probability
    results["churn_prediction"] = churn_prediction
    results["risk_score"] = churn_probability * 100
    results["risk_category"] = pd.cut(churn_probability, 
                                    bins=[0, 0.3, 0.7, 1.0], 
                                    labels=["Low", "Medium", "High"])
    
    # Return results
    OutputDataSet = results[["customer_id", "churn_probability", "churn_prediction", 
                           "risk_score", "risk_category"]]
    ';
    
    -- Execute Python script
    EXEC sp_execute_external_script
        @language = N'Python',
        @script = @sql,
        @params = N'@model_data VARBINARY(MAX), @input_data NVARCHAR(MAX)',
        @model_data = @model_data,
        @input_data = @input_data
    WITH RESULT SETS ((
        customer_id INT,
        churn_probability FLOAT,
        churn_prediction INT,
        risk_score FLOAT,
        risk_category VARCHAR(10)
    ));
    
    -- Log prediction activity
    INSERT INTO ml_prediction_log (
        model_name, model_version, prediction_date, 
        customer_count, batch_mode
    )
    VALUES (
        'customer_churn_rf', @model_version, GETDATE(),
        CASE WHEN @batch_mode = 1 THEN (SELECT COUNT(*) FROM customer_features_view WHERE is_active = 1)
             ELSE 1 END,
        @batch_mode
    );
END;

-- PostgreSQL with PL/Python Example
CREATE OR REPLACE FUNCTION predict_product_demand(
    product_id INTEGER DEFAULT NULL,
    forecast_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    product_id INTEGER,
    forecast_date DATE,
    predicted_demand NUMERIC,
    confidence_interval_lower NUMERIC,
    confidence_interval_upper NUMERIC
)
LANGUAGE plpythonu
AS $$
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    
    # Get historical sales data
    if product_id:
        query = f"""
        SELECT product_id, sales_date, daily_sales, 
               LAG(daily_sales, 1) OVER (ORDER BY sales_date) as sales_lag_1,
               LAG(daily_sales, 7) OVER (ORDER BY sales_date) as sales_lag_7,
               LAG(daily_sales, 30) OVER (ORDER BY sales_date) as sales_lag_30,
               EXTRACT(DOW FROM sales_date) as day_of_week,
               EXTRACT(MONTH FROM sales_date) as month
        FROM daily_sales 
        WHERE product_id = {product_id}
        ORDER BY sales_date
        """
    else:
        return "Product ID required"
    
    # Execute query and get data
    rv = plpy.execute(query)
    df = pd.DataFrame(rv)
    
    # Load pre-trained model
    model_query = """
    SELECT model_object FROM ml_models 
    WHERE model_name = 'demand_forecast' AND is_active = true
    """
    model_result = plpy.execute(model_query)
    
    if not model_result:
        return "Model not found"
    
    # Feature engineering
    df['sales_ma_7'] = df['daily_sales'].rolling(7).mean()
    df['sales_ma_30'] = df['daily_sales'].rolling(30).mean()
    df = df.dropna()
    
    # Prepare features
    feature_cols = ['sales_lag_1', 'sales_lag_7', 'sales_lag_30', 
                   'sales_ma_7', 'sales_ma_30', 'day_of_week', 'month']
    
    if len(df) < 30:
        return "Insufficient historical data"
    
    # Use last known values for forecasting
    last_date = pd.to_datetime(df['sales_date'].iloc[-1])
    last_features = df[feature_cols].iloc[-1:].values
    
    # Generate forecasts
    results = []
    for i in range(forecast_days):
        forecast_date = last_date + timedelta(days=i+1)
        
        # Simple prediction logic (replace with actual model)
        base_prediction = np.mean(df['daily_sales'].tail(30))
        seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * forecast_date.dayofyear / 365)
        trend_factor = 1.0 + 0.001 * i  # Small upward trend
        
        predicted_demand = base_prediction * seasonal_factor * trend_factor
        
        # Add some uncertainty
        confidence_lower = predicted_demand * 0.85
        confidence_upper = predicted_demand * 1.15
        
        results.append({
            'product_id': product_id,
            'forecast_date': forecast_date.date(),
            'predicted_demand': round(predicted_demand, 2),
            'confidence_interval_lower': round(confidence_lower, 2),
            'confidence_interval_upper': round(confidence_upper, 2)
        })
    
    return results
$$;

-- Oracle APEX with ML Models
CREATE OR REPLACE PROCEDURE score_fraud_transaction(
    p_transaction_id IN NUMBER,
    p_fraud_score OUT NUMBER,
    p_fraud_risk OUT VARCHAR2
)
IS
    v_amount NUMBER;
    v_merchant_risk NUMBER;
    v_customer_velocity NUMBER;
    v_time_since_last NUMBER;
    v_location_risk NUMBER;
    v_sql CLOB;
    v_result NUMBER;
BEGIN
    -- Extract transaction features
    SELECT 
        t.amount,
        m.risk_score,
        c.transaction_velocity_24h,
        EXTRACT(HOUR FROM (SYSDATE - t.transaction_time)) * 60 + 
        EXTRACT(MINUTE FROM (SYSDATE - t.transaction_time)) AS minutes_since_last,
        l.risk_score
    INTO v_amount, v_merchant_risk, v_customer_velocity, v_time_since_last, v_location_risk
    FROM transactions t
    JOIN merchants m ON t.merchant_id = m.merchant_id
    JOIN customers c ON t.customer_id = c.customer_id
    JOIN locations l ON t.location_id = l.location_id
    WHERE t.transaction_id = p_transaction_id;
    
    -- Simple rule-based ML scoring (replace with actual ML model)
    v_result := 
        CASE 
            WHEN v_amount > 5000 THEN 0.3
            WHEN v_amount > 1000 THEN 0.1
            ELSE 0.0
        END +
        CASE 
            WHEN v_merchant_risk > 0.7 THEN 0.4
            WHEN v_merchant_risk > 0.3 THEN 0.2
            ELSE 0.0
        END +
        CASE 
            WHEN v_customer_velocity > 10 THEN 0.3
            WHEN v_customer_velocity > 5 THEN 0.1
            ELSE 0.0
        END +
        CASE 
            WHEN v_time_since_last < 5 THEN 0.2
            WHEN v_time_since_last < 60 THEN 0.1
            ELSE 0.0
        END +
        CASE 
            WHEN v_location_risk > 0.8 THEN 0.3
            WHEN v_location_risk > 0.5 THEN 0.1
            ELSE 0.0
        END;
    
    p_fraud_score := LEAST(v_result, 1.0);
    
    p_fraud_risk := 
        CASE 
            WHEN p_fraud_score >= 0.8 THEN 'HIGH'
            WHEN p_fraud_score >= 0.5 THEN 'MEDIUM'
            WHEN p_fraud_score >= 0.2 THEN 'LOW'
            ELSE 'MINIMAL'
        END;
    
    -- Log the scoring
    INSERT INTO fraud_scoring_log (
        transaction_id, fraud_score, fraud_risk, scoring_date
    ) VALUES (
        p_transaction_id, p_fraud_score, p_fraud_risk, SYSDATE
    );
    
    COMMIT;
    
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        p_fraud_score := 0.5;  -- Default medium risk
        p_fraud_risk := 'MEDIUM';
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
```

#### Explanation
1. **External Scripts**: Use sp_execute_external_script for Python/R integration
2. **Model Storage**: Store serialized models in database tables
3. **Feature Preparation**: Extract and transform features within the procedure
4. **Batch vs Single**: Support both individual and batch predictions
5. **Error Handling**: Implement proper exception handling and logging

#### Use Cases
- **Real-time Scoring**: Score transactions, customers, or products in real-time
- **Automated Decisions**: Integrate ML predictions into business workflows
- **A/B Testing**: Deploy multiple model versions for comparison
- **Monitoring**: Track model performance and prediction distributions
- **ETL Integration**: Embed ML scoring in data pipeline processes

#### Best Practices
- Version control your models and track performance metrics
- Implement model fallback strategies for system resilience
- Use appropriate security measures for model access
- Monitor prediction latency and throughput
- Validate input data quality before scoring

#### Pitfalls
- **Performance Impact**: ML scoring can slow down database operations
- **Model Drift**: Models may become stale without regular updates
- **Memory Usage**: Large models consume significant database memory
- **Dependency Management**: External libraries may have compatibility issues
- **Security Risks**: Stored procedures with ML code require careful access control

#### Debugging
```sql
-- Test the stored procedure
EXEC sp_predict_customer_churn @customer_id = 12345, @batch_mode = 0;

-- Check model performance logs
SELECT 
    prediction_date,
    model_version,
    customer_count,
    AVG(churn_probability) as avg_churn_prob
FROM ml_prediction_log 
WHERE model_name = 'customer_churn_rf'
GROUP BY prediction_date, model_version, customer_count
ORDER BY prediction_date DESC;
```

#### Optimization
- **Model Caching**: Cache frequently used models in memory
- **Parallel Processing**: Use parallel execution for batch predictions
- **Result Caching**: Cache predictions for frequently requested entities
- **Resource Management**: Monitor and limit resource usage of ML operations

---

## Question 9

**How would you construct a complex SQL query to extract time series features for a Machine Learning model?**

### Answer:

#### Theory
Time series feature extraction transforms temporal data into meaningful predictors for ML models. This involves creating lagged variables, rolling statistics, seasonal decomposition, trend analysis, and cyclical patterns that capture the underlying temporal dynamics essential for forecasting and classification tasks.

#### Code Example
```sql
-- Comprehensive Time Series Feature Engineering
WITH base_time_series AS (
    SELECT 
        product_id,
        sales_date,
        daily_sales,
        daily_units,
        avg_price,
        -- Date components
        EXTRACT(YEAR FROM sales_date) AS year,
        EXTRACT(MONTH FROM sales_date) AS month,
        EXTRACT(DAY FROM sales_date) AS day,
        EXTRACT(DOW FROM sales_date) AS day_of_week,
        EXTRACT(DOY FROM sales_date) AS day_of_year,
        EXTRACT(WEEK FROM sales_date) AS week_of_year,
        -- Business calendar features
        CASE 
            WHEN EXTRACT(DOW FROM sales_date) IN (6, 0) THEN 1 
            ELSE 0 
        END AS is_weekend,
        CASE 
            WHEN EXTRACT(MONTH FROM sales_date) IN (12, 1, 2) THEN 1 
            ELSE 0 
        END AS is_winter,
        CASE 
            WHEN EXTRACT(MONTH FROM sales_date) IN (11, 12) THEN 1 
            ELSE 0 
        END AS is_holiday_season
    FROM daily_product_sales
    WHERE sales_date >= '2020-01-01'
),
lagged_features AS (
    SELECT 
        product_id,
        sales_date,
        daily_sales,
        year, month, day_of_week, day_of_year,
        is_weekend, is_winter, is_holiday_season,
        
        -- Short-term lags (1-7 days)
        LAG(daily_sales, 1) OVER (PARTITION BY product_id ORDER BY sales_date) AS lag_1d,
        LAG(daily_sales, 2) OVER (PARTITION BY product_id ORDER BY sales_date) AS lag_2d,
        LAG(daily_sales, 3) OVER (PARTITION BY product_id ORDER BY sales_date) AS lag_3d,
        LAG(daily_sales, 7) OVER (PARTITION BY product_id ORDER BY sales_date) AS lag_7d,
        
        -- Medium-term lags (2-4 weeks)
        LAG(daily_sales, 14) OVER (PARTITION BY product_id ORDER BY sales_date) AS lag_14d,
        LAG(daily_sales, 21) OVER (PARTITION BY product_id ORDER BY sales_date) AS lag_21d,
        LAG(daily_sales, 28) OVER (PARTITION BY product_id ORDER BY sales_date) AS lag_28d,
        
        -- Long-term lags (seasonal)
        LAG(daily_sales, 90) OVER (PARTITION BY product_id ORDER BY sales_date) AS lag_90d,
        LAG(daily_sales, 180) OVER (PARTITION BY product_id ORDER BY sales_date) AS lag_180d,
        LAG(daily_sales, 365) OVER (PARTITION BY product_id ORDER BY sales_date) AS lag_365d,
        
        -- Same day previous periods
        LAG(daily_sales, 7) OVER (PARTITION BY product_id, day_of_week ORDER BY sales_date) AS same_dow_last_week,
        LAG(daily_sales, 28) OVER (PARTITION BY product_id, day_of_week ORDER BY sales_date) AS same_dow_4weeks_ago,
        LAG(daily_sales, 365) OVER (PARTITION BY product_id, EXTRACT(DOY FROM sales_date) ORDER BY sales_date) AS same_doy_last_year
    FROM base_time_series
),
rolling_features AS (
    SELECT 
        product_id,
        sales_date,
        daily_sales,
        lag_1d, lag_7d, lag_14d, lag_28d, lag_365d,
        same_dow_last_week, same_doy_last_year,
        year, month, day_of_week, is_weekend, is_holiday_season,
        
        -- Rolling averages (multiple windows)
        AVG(daily_sales) OVER (
            PARTITION BY product_id 
            ORDER BY sales_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS ma_7d,
        AVG(daily_sales) OVER (
            PARTITION BY product_id 
            ORDER BY sales_date 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS ma_14d,
        AVG(daily_sales) OVER (
            PARTITION BY product_id 
            ORDER BY sales_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS ma_30d,
        AVG(daily_sales) OVER (
            PARTITION BY product_id 
            ORDER BY sales_date 
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        ) AS ma_90d,
        
        -- Rolling standard deviations
        STDDEV(daily_sales) OVER (
            PARTITION BY product_id 
            ORDER BY sales_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS std_30d,
        STDDEV(daily_sales) OVER (
            PARTITION BY product_id 
            ORDER BY sales_date 
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        ) AS std_90d,
        
        -- Rolling min/max
        MIN(daily_sales) OVER (
            PARTITION BY product_id 
            ORDER BY sales_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS min_30d,
        MAX(daily_sales) OVER (
            PARTITION BY product_id 
            ORDER BY sales_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS max_30d,
        
        -- Rolling sums for aggregated features
        SUM(daily_sales) OVER (
            PARTITION BY product_id 
            ORDER BY sales_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS sum_30d,
        
        -- Count of non-zero sales days
        COUNT(CASE WHEN daily_sales > 0 THEN 1 END) OVER (
            PARTITION BY product_id 
            ORDER BY sales_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS active_days_30d
    FROM lagged_features
),
derived_features AS (
    SELECT 
        product_id,
        sales_date,
        daily_sales,
        lag_1d, lag_7d, lag_28d, lag_365d,
        ma_7d, ma_14d, ma_30d, ma_90d,
        std_30d, std_90d,
        min_30d, max_30d, sum_30d,
        same_dow_last_week, same_doy_last_year,
        year, month, day_of_week, is_weekend, is_holiday_season,
        active_days_30d,
        
        -- Trend features
        daily_sales - lag_1d AS daily_change,
        daily_sales - lag_7d AS weekly_change,
        daily_sales - lag_28d AS monthly_change,
        daily_sales - same_doy_last_year AS yearly_change,
        
        -- Momentum features
        CASE WHEN lag_7d > 0 THEN (daily_sales - lag_7d) / lag_7d ELSE 0 END AS weekly_growth_rate,
        CASE WHEN lag_28d > 0 THEN (daily_sales - lag_28d) / lag_28d ELSE 0 END AS monthly_growth_rate,
        CASE WHEN same_doy_last_year > 0 THEN (daily_sales - same_doy_last_year) / same_doy_last_year ELSE 0 END AS yoy_growth_rate,
        
        -- Moving average ratios
        CASE WHEN ma_7d > 0 THEN daily_sales / ma_7d ELSE 0 END AS vs_ma7_ratio,
        CASE WHEN ma_30d > 0 THEN daily_sales / ma_30d ELSE 0 END AS vs_ma30_ratio,
        CASE WHEN ma_90d > 0 THEN ma_7d / ma_90d ELSE 0 END AS ma7_vs_ma90_ratio,
        
        -- Volatility features
        CASE WHEN std_30d > 0 THEN ABS(daily_sales - ma_30d) / std_30d ELSE 0 END AS zscore_30d,
        CASE WHEN ma_30d > 0 THEN std_30d / ma_30d ELSE 0 END AS coeff_variation_30d,
        
        -- Range features
        CASE WHEN (max_30d - min_30d) > 0 THEN (daily_sales - min_30d) / (max_30d - min_30d) ELSE 0 END AS normalized_position,
        max_30d - min_30d AS range_30d,
        
        -- Seasonal features
        SIN(2 * PI() * EXTRACT(DOY FROM sales_date) / 365.25) AS seasonal_sin,
        COS(2 * PI() * EXTRACT(DOY FROM sales_date) / 365.25) AS seasonal_cos,
        SIN(2 * PI() * EXTRACT(DOW FROM sales_date) / 7) AS weekly_sin,
        COS(2 * PI() * EXTRACT(DOW FROM sales_date) / 7) AS weekly_cos,
        
        -- Business cycle features
        CASE 
            WHEN EXTRACT(DAY FROM sales_date) <= 7 THEN 'Week1'
            WHEN EXTRACT(DAY FROM sales_date) <= 14 THEN 'Week2'
            WHEN EXTRACT(DAY FROM sales_date) <= 21 THEN 'Week3'
            ELSE 'Week4+'
        END AS week_of_month,
        
        -- Interaction features
        day_of_week * is_holiday_season AS dow_holiday_interaction,
        month * is_weekend AS month_weekend_interaction
    FROM rolling_features
),
advanced_features AS (
    SELECT 
        product_id,
        sales_date,
        daily_sales,
        -- Core temporal features
        lag_1d, lag_7d, lag_28d, ma_7d, ma_30d, std_30d,
        daily_change, weekly_change, monthly_change,
        weekly_growth_rate, monthly_growth_rate, yoy_growth_rate,
        vs_ma7_ratio, vs_ma30_ratio, zscore_30d, coeff_variation_30d,
        seasonal_sin, seasonal_cos, weekly_sin, weekly_cos,
        
        -- Calendar features
        year, month, day_of_week, is_weekend, is_holiday_season,
        week_of_month, dow_holiday_interaction,
        
        -- Advanced pattern features
        CASE 
            WHEN weekly_change > 0 AND monthly_change > 0 THEN 'Accelerating_Up'
            WHEN weekly_change > 0 AND monthly_change <= 0 THEN 'Recent_Recovery'
            WHEN weekly_change <= 0 AND monthly_change > 0 THEN 'Recent_Decline'
            ELSE 'Declining'
        END AS trend_pattern,
        
        CASE 
            WHEN zscore_30d > 2 THEN 'Unusually_High'
            WHEN zscore_30d < -2 THEN 'Unusually_Low'
            ELSE 'Normal'
        END AS anomaly_flag,
        
        CASE 
            WHEN coeff_variation_30d > 0.5 THEN 'High_Volatility'
            WHEN coeff_variation_30d > 0.2 THEN 'Medium_Volatility'
            ELSE 'Low_Volatility'
        END AS volatility_regime,
        
        -- Time since events
        ROW_NUMBER() OVER (
            PARTITION BY product_id, 
            CASE WHEN daily_sales = 0 THEN 1 ELSE 0 END 
            ORDER BY sales_date DESC
        ) AS days_since_zero_sales,
        
        -- Cumulative features
        SUM(daily_sales) OVER (
            PARTITION BY product_id, year 
            ORDER BY sales_date
        ) AS cumulative_sales_ytd,
        
        ROW_NUMBER() OVER (
            PARTITION BY product_id 
            ORDER BY sales_date
        ) AS time_index
    FROM derived_features
)
SELECT 
    product_id,
    sales_date,
    daily_sales,
    -- Essential features for ML model
    lag_1d, lag_7d, lag_28d,
    ma_7d, ma_30d, std_30d,
    daily_change, weekly_change, monthly_change,
    weekly_growth_rate, monthly_growth_rate,
    vs_ma7_ratio, vs_ma30_ratio,
    zscore_30d, coeff_variation_30d,
    seasonal_sin, seasonal_cos,
    is_weekend, is_holiday_season,
    trend_pattern, anomaly_flag, volatility_regime,
    time_index
FROM advanced_features
WHERE sales_date >= '2021-01-01'  -- Ensure sufficient lag data
ORDER BY product_id, sales_date;
```

#### Explanation
1. **Temporal Decomposition**: Extract date components, seasonal patterns, and cyclical features
2. **Lag Features**: Multiple time horizons capture short, medium, and long-term dependencies
3. **Rolling Statistics**: Moving averages, standard deviations, and aggregations smooth noise
4. **Derived Features**: Growth rates, ratios, and normalized metrics reveal patterns
5. **Interaction Terms**: Combine temporal and categorical features for richer representations

#### Use Cases
- **Demand Forecasting**: Predict future sales based on historical patterns
- **Anomaly Detection**: Identify unusual patterns in time series data
- **Seasonality Analysis**: Understand cyclical business patterns
- **Trend Analysis**: Capture long-term growth or decline signals
- **Financial Modeling**: Create features for stock price or economic predictions

#### Best Practices
- Include multiple time horizons to capture different patterns
- Handle missing dates and irregular time series appropriately
- Create both absolute and relative (ratio) features
- Consider business calendar effects (holidays, seasons, events)
- Validate feature importance and remove redundant variables

#### Pitfalls
- **Look-ahead Bias**: Ensure features don't include future information
- **Overfitting**: Too many time-based features can lead to overfitting
- **Data Leakage**: Careful with aggregations that include target values
- **Computational Cost**: Complex window functions can be resource-intensive
- **Sparse Data**: Handle periods with missing or zero values appropriately

#### Debugging
```sql
-- Validate feature completeness
SELECT 
    product_id,
    COUNT(*) AS total_rows,
    COUNT(lag_7d) AS lag_7d_count,
    COUNT(ma_30d) AS ma_30d_count,
    MIN(sales_date) AS first_date,
    MAX(sales_date) AS last_date
FROM time_series_features
GROUP BY product_id
ORDER BY total_rows DESC;

-- Check for data quality issues
SELECT 
    COUNT(CASE WHEN daily_sales < 0 THEN 1 END) AS negative_sales,
    COUNT(CASE WHEN daily_sales IS NULL THEN 1 END) AS null_sales,
    COUNT(CASE WHEN ABS(zscore_30d) > 5 THEN 1 END) AS extreme_outliers
FROM time_series_features;
```

#### Optimization
- **Indexing**: Create indexes on (product_id, sales_date) for efficient window operations
- **Partitioning**: Partition tables by date ranges for better performance
- **Incremental Processing**: Update only new data points instead of recalculating all features
- **Materialized Views**: Store computed time series features for repeated access

---

## Question 10

**Discuss ways to implement regular expressions in SQL for natural language processing tasks.**

### Answer:

#### Theory
Regular expressions (regex) in SQL enable powerful pattern matching and text extraction for NLP preprocessing. They're essential for cleaning text data, extracting entities, validating formats, and preparing unstructured text for machine learning models.

#### Code Example
```sql
-- Basic Regular Expression Patterns for NLP
SELECT 
    text_id,
    original_text,
    -- Email extraction
    REGEXP_EXTRACT(original_text, r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}') AS extracted_email,
    
    -- Phone number extraction (US format)
    REGEXP_EXTRACT(original_text, r'(\+1-?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}') AS phone_number,
    
    -- URL extraction
    REGEXP_EXTRACT(original_text, r'https?://[^\s]+') AS url,
    
    -- Extract hashtags
    REGEXP_EXTRACT_ALL(original_text, r'#\w+') AS hashtags,
    
    -- Extract mentions
    REGEXP_EXTRACT_ALL(original_text, r'@\w+') AS mentions,
    
    -- Clean text - remove special characters
    REGEXP_REPLACE(original_text, r'[^\w\s]', ' ') AS cleaned_text,
    
    -- Remove extra whitespace
    TRIM(REGEXP_REPLACE(original_text, r'\s+', ' ')) AS normalized_whitespace,
    
    -- Extract numeric values
    REGEXP_EXTRACT_ALL(original_text, r'\$?\d+\.?\d*') AS numeric_values
FROM social_media_posts;

-- Advanced Text Preprocessing with Regex
WITH text_preprocessing AS (
    SELECT 
        review_id,
        product_id,
        review_text,
        
        -- Remove HTML tags
        REGEXP_REPLACE(review_text, r'<[^>]+>', '') AS html_removed,
        
        -- Extract sentences (simplified)
        REGEXP_EXTRACT_ALL(review_text, r'[^.!?]+[.!?]') AS sentences,
        
        -- Word extraction (alphanumeric words only)
        REGEXP_EXTRACT_ALL(
            LOWER(review_text), 
            r'\b[a-z]{3,}\b'
        ) AS words,
        
        -- Extract capitalized words (potential proper nouns)
        REGEXP_EXTRACT_ALL(review_text, r'\b[A-Z][a-z]+\b') AS proper_nouns,
        
        -- Count exclamation marks
        ARRAY_LENGTH(REGEXP_EXTRACT_ALL(review_text, r'!')) AS exclamation_count,
        
        -- Extract years (4-digit numbers 1900-2099)
        REGEXP_EXTRACT_ALL(review_text, r'\b(19|20)\d{2}\b') AS years,
        
        -- Sentiment indicators
        CASE 
            WHEN REGEXP_CONTAINS(LOWER(review_text), r'\b(excellent|amazing|fantastic|perfect|love)\b') THEN 'positive'
            WHEN REGEXP_CONTAINS(LOWER(review_text), r'\b(terrible|awful|hate|worst|horrible)\b') THEN 'negative'
            ELSE 'neutral'
        END AS sentiment_pattern
    FROM product_reviews
),
enhanced_features AS (
    SELECT 
        review_id,
        product_id,
        review_text,
        html_removed,
        ARRAY_LENGTH(sentences) AS sentence_count,
        ARRAY_LENGTH(words) AS word_count,
        ARRAY_LENGTH(proper_nouns) AS proper_noun_count,
        exclamation_count,
        ARRAY_LENGTH(years) AS year_mention_count,
        sentiment_pattern,
        
        -- Text quality indicators
        CASE 
            WHEN REGEXP_CONTAINS(html_removed, r'^[A-Z][a-z\s]+[.!?]$') THEN 'well_formatted'
            WHEN REGEXP_CONTAINS(html_removed, r'^[a-z\s]*$') THEN 'no_capitalization'
            WHEN REGEXP_CONTAINS(html_removed, r'^[A-Z\s]*$') THEN 'all_caps'
            ELSE 'mixed_formatting'
        END AS text_format_quality,
        
        -- Language detection (simplified)
        CASE 
            WHEN REGEXP_CONTAINS(html_removed, r'\b(the|and|or|but|for|to|of|in|on|at|by|with)\b') THEN 'likely_english'
            ELSE 'other_language'
        END AS language_indicator,
        
        -- Content type classification
        CASE 
            WHEN REGEXP_CONTAINS(LOWER(html_removed), r'\b(buy|purchase|order|price|cost|dollar|money)\b') THEN 'commercial'
            WHEN REGEXP_CONTAINS(LOWER(html_removed), r'\b(problem|issue|error|bug|fail|broken)\b') THEN 'issue_report'
            WHEN REGEXP_CONTAINS(LOWER(html_removed), r'\b(recommend|suggest|advice|tip|guide)\b') THEN 'recommendation'
            ELSE 'general'
        END AS content_type
    FROM text_preprocessing
)
SELECT 
    review_id,
    word_count,
    sentence_count,
    proper_noun_count,
    exclamation_count,
    sentiment_pattern,
    text_format_quality,
    language_indicator,
    content_type
FROM enhanced_features;

-- Entity Extraction with Named Entity Recognition Patterns
SELECT 
    document_id,
    document_text,
    
    -- Extract person names (simplified pattern)
    REGEXP_EXTRACT_ALL(
        document_text, 
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
    ) AS potential_names,
    
    -- Extract dates (various formats)
    REGEXP_EXTRACT_ALL(
        document_text, 
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+ \d{1,2}, \d{4})\b'
    ) AS dates,
    
    -- Extract monetary amounts
    REGEXP_EXTRACT_ALL(
        document_text, 
        r'\$[0-9,]+\.?\d{0,2}'
    ) AS monetary_amounts,
    
    -- Extract company names (basic pattern)
    REGEXP_EXTRACT_ALL(
        document_text, 
        r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company|Co)\b'
    ) AS company_names,
    
    -- Extract product codes or IDs
    REGEXP_EXTRACT_ALL(
        document_text, 
        r'\b[A-Z]{2,4}-?\d{3,6}\b'
    ) AS product_codes,
    
    -- Extract technical terms (CamelCase or specific patterns)
    REGEXP_EXTRACT_ALL(
        document_text, 
        r'\b[A-Z][a-z]+[A-Z][a-z]+\w*\b'
    ) AS technical_terms
FROM legal_documents;

-- Text Validation and Quality Assessment
WITH text_validation AS (
    SELECT 
        comment_id,
        comment_text,
        
        -- Check for spam patterns
        CASE 
            WHEN REGEXP_CONTAINS(LOWER(comment_text), r'\b(viagra|casino|lottery|free money|click here)\b') THEN 1
            ELSE 0
        END AS spam_indicator,
        
        -- Check for all caps (shouting)
        CASE 
            WHEN REGEXP_CONTAINS(comment_text, r'^[A-Z\s!?.,]{10,}$') THEN 1
            ELSE 0
        END AS all_caps_flag,
        
        -- Check for excessive repetition
        CASE 
            WHEN REGEXP_CONTAINS(comment_text, r'(.)\1{4,}') THEN 1  -- 5+ repeated characters
            ELSE 0
        END AS excessive_repetition,
        
        -- Check for valid email format
        CASE 
            WHEN REGEXP_CONTAINS(comment_text, r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$') THEN 1
            ELSE 0
        END AS contains_email,
        
        -- Profanity detection (simplified)
        CASE 
            WHEN REGEXP_CONTAINS(LOWER(comment_text), r'\b(badword1|badword2|inappropriate)\b') THEN 1
            ELSE 0
        END AS profanity_flag,
        
        -- Text complexity assessment
        ARRAY_LENGTH(REGEXP_EXTRACT_ALL(comment_text, r'\b\w{8,}\b')) AS complex_word_count,
        ARRAY_LENGTH(REGEXP_EXTRACT_ALL(comment_text, r'[.!?]')) AS sentence_end_count
    FROM user_comments
)
SELECT 
    comment_id,
    comment_text,
    spam_indicator,
    all_caps_flag,
    excessive_repetition,
    profanity_flag,
    complex_word_count,
    sentence_end_count,
    -- Overall quality score
    CASE 
        WHEN spam_indicator + all_caps_flag + excessive_repetition + profanity_flag = 0 
             AND complex_word_count > 0 THEN 'high_quality'
        WHEN spam_indicator + all_caps_flag + excessive_repetition + profanity_flag <= 1 THEN 'medium_quality'
        ELSE 'low_quality'
    END AS text_quality_score
FROM text_validation;
```

#### Explanation
1. **Pattern Matching**: REGEXP_CONTAINS checks if text matches a pattern
2. **Extraction**: REGEXP_EXTRACT and REGEXP_EXTRACT_ALL pull specific patterns
3. **Replacement**: REGEXP_REPLACE substitutes matched patterns with new text
4. **Quantifiers**: Use +, *, ?, {n,m} to specify repetition patterns
5. **Character Classes**: [a-zA-Z], \w, \d, \s for different character types

#### Use Cases
- **Data Cleaning**: Remove unwanted characters and normalize text format
- **Entity Extraction**: Pull names, dates, phone numbers, emails from text
- **Content Classification**: Categorize text based on linguistic patterns
- **Spam Detection**: Identify suspicious or inappropriate content
- **Feature Engineering**: Create text-based features for ML models

#### Best Practices
- Test regex patterns thoroughly with sample data
- Use raw strings (r'pattern') to avoid escape character issues
- Consider performance implications of complex patterns on large datasets
- Validate extracted entities for accuracy
- Document regex patterns with comments for maintainability

#### Pitfalls
- **Performance Issues**: Complex regex can be slow on large text columns
- **False Positives**: Overly broad patterns may match unintended text
- **Missing Matches**: Overly specific patterns may miss valid cases
- **Encoding Issues**: Regex behavior may vary with different character encodings
- **Language Limitations**: Patterns designed for English may not work for other languages

#### Debugging
```sql
-- Test regex patterns
SELECT 
    'test@example.com' AS test_text,
    REGEXP_CONTAINS('test@example.com', r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}') AS email_match,
    REGEXP_EXTRACT('Call me at 555-123-4567', r'\d{3}-\d{3}-\d{4}') AS phone_extract;

-- Validate extraction results
SELECT 
    original_text,
    extracted_emails,
    ARRAY_LENGTH(extracted_emails) AS email_count
FROM text_extractions
WHERE ARRAY_LENGTH(extracted_emails) > 3  -- Check for over-extraction
LIMIT 10;
```

#### Optimization
- **Indexing**: Create functional indexes on commonly searched text patterns
- **Pre-filtering**: Use simple string functions before applying complex regex
- **Compiled Patterns**: Some databases allow pre-compiling regex for better performance
- **Chunking**: Process large text datasets in smaller batches to manage memory

---

## Question 11

**Write a SQL script to identify and replace missing values with the column mean.**

### Answer:

#### Theory
Missing value imputation is crucial for ML model performance. Mean imputation is a simple strategy that replaces NULL values with the column average, maintaining data completeness while preserving the overall distribution characteristics for numerical features.

#### Code Example
```sql
-- Method 1: Simple Mean Imputation
WITH column_means AS (
    SELECT 
        AVG(age) AS mean_age,
        AVG(income) AS mean_income,
        AVG(credit_score) AS mean_credit_score,
        AVG(account_balance) AS mean_account_balance
    FROM customers
),
imputed_data AS (
    SELECT 
        customer_id,
        first_name,
        last_name,
        -- Replace NULLs with column means
        COALESCE(age, (SELECT mean_age FROM column_means)) AS age_imputed,
        COALESCE(income, (SELECT mean_income FROM column_means)) AS income_imputed,
        COALESCE(credit_score, (SELECT mean_credit_score FROM column_means)) AS credit_score_imputed,
        COALESCE(account_balance, (SELECT mean_account_balance FROM column_means)) AS account_balance_imputed,
        -- Track which values were imputed
        CASE WHEN age IS NULL THEN 1 ELSE 0 END AS age_was_imputed,
        CASE WHEN income IS NULL THEN 1 ELSE 0 END AS income_was_imputed,
        CASE WHEN credit_score IS NULL THEN 1 ELSE 0 END AS credit_score_was_imputed,
        CASE WHEN account_balance IS NULL THEN 1 ELSE 0 END AS account_balance_was_imputed
    FROM customers
)
SELECT * FROM imputed_data;

-- Method 2: Group-wise Mean Imputation
WITH group_means AS (
    SELECT 
        customer_segment,
        region,
        AVG(age) AS segment_mean_age,
        AVG(income) AS segment_mean_income,
        AVG(credit_score) AS segment_mean_credit_score
    FROM customers
    WHERE age IS NOT NULL OR income IS NOT NULL OR credit_score IS NOT NULL
    GROUP BY customer_segment, region
),
global_means AS (
    SELECT 
        AVG(age) AS global_mean_age,
        AVG(income) AS global_mean_income,
        AVG(credit_score) AS global_mean_credit_score
    FROM customers
),
advanced_imputation AS (
    SELECT 
        c.customer_id,
        c.customer_segment,
        c.region,
        c.age,
        c.income,
        c.credit_score,
        -- Group-wise imputation with global fallback
        COALESCE(
            c.age,
            gm.segment_mean_age,
            (SELECT global_mean_age FROM global_means)
        ) AS age_imputed,
        COALESCE(
            c.income,
            gm.segment_mean_income,
            (SELECT global_mean_income FROM global_means)
        ) AS income_imputed,
        COALESCE(
            c.credit_score,
            gm.segment_mean_credit_score,
            (SELECT global_mean_credit_score FROM global_means)
        ) AS credit_score_imputed,
        -- Imputation method tracking
        CASE 
            WHEN c.age IS NOT NULL THEN 'original'
            WHEN gm.segment_mean_age IS NOT NULL THEN 'group_mean'
            ELSE 'global_mean'
        END AS age_imputation_method
    FROM customers c
    LEFT JOIN group_means gm ON c.customer_segment = gm.customer_segment 
                              AND c.region = gm.region
)
SELECT * FROM advanced_imputation;

-- Method 3: Dynamic Imputation with Window Functions
SELECT 
    transaction_id,
    customer_id,
    merchant_id,
    transaction_date,
    amount,
    
    -- Impute using customer-specific mean
    COALESCE(
        amount,
        AVG(amount) OVER (PARTITION BY customer_id)
    ) AS amount_customer_mean,
    
    -- Impute using merchant-specific mean
    COALESCE(
        amount,
        AVG(amount) OVER (PARTITION BY merchant_id)
    ) AS amount_merchant_mean,
    
    -- Impute using time-window mean (last 30 days)
    COALESCE(
        amount,
        AVG(amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND 1 PRECEDING
        )
    ) AS amount_time_window_mean,
    
    -- Hybrid imputation strategy
    COALESCE(
        amount,
        -- Try customer mean first
        AVG(amount) OVER (PARTITION BY customer_id),
        -- Fallback to merchant mean
        AVG(amount) OVER (PARTITION BY merchant_id),
        -- Final fallback to global mean
        AVG(amount) OVER ()
    ) AS amount_hybrid_imputed
FROM transactions
ORDER BY customer_id, transaction_date;

-- Method 4: Comprehensive Missing Value Treatment
CREATE TABLE imputation_metadata (
    table_name VARCHAR(100),
    column_name VARCHAR(100),
    imputation_method VARCHAR(50),
    imputation_value DECIMAL(15,4),
    missing_count INT,
    total_count INT,
    missing_percentage DECIMAL(5,2),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analysis and imputation procedure
WITH missing_analysis AS (
    SELECT 
        'customers' AS table_name,
        'age' AS column_name,
        COUNT(*) AS total_count,
        COUNT(age) AS non_null_count,
        COUNT(*) - COUNT(age) AS missing_count,
        ROUND((COUNT(*) - COUNT(age)) * 100.0 / COUNT(*), 2) AS missing_percentage,
        AVG(age) AS mean_value,
        STDDEV(age) AS std_value,
        MIN(age) AS min_value,
        MAX(age) AS max_value
    FROM customers
    
    UNION ALL
    
    SELECT 
        'customers' AS table_name,
        'income' AS column_name,
        COUNT(*) AS total_count,
        COUNT(income) AS non_null_count,
        COUNT(*) - COUNT(income) AS missing_count,
        ROUND((COUNT(*) - COUNT(income)) * 100.0 / COUNT(*), 2) AS missing_percentage,
        AVG(income) AS mean_value,
        STDDEV(income) AS std_value,
        MIN(income) AS min_value,
        MAX(income) AS max_value
    FROM customers
    
    UNION ALL
    
    SELECT 
        'customers' AS table_name,
        'credit_score' AS column_name,
        COUNT(*) AS total_count,
        COUNT(credit_score) AS non_null_count,
        COUNT(*) - COUNT(credit_score) AS missing_count,
        ROUND((COUNT(*) - COUNT(credit_score)) * 100.0 / COUNT(*), 2) AS missing_percentage,
        AVG(credit_score) AS mean_value,
        STDDEV(credit_score) AS std_value,
        MIN(credit_score) AS min_value,
        MAX(credit_score) AS max_value
    FROM customers
),
imputation_strategy AS (
    SELECT 
        table_name,
        column_name,
        missing_count,
        total_count,
        missing_percentage,
        mean_value,
        CASE 
            WHEN missing_percentage < 5 THEN 'mean_imputation'
            WHEN missing_percentage < 15 THEN 'group_mean_imputation'
            WHEN missing_percentage < 30 THEN 'predictive_imputation'
            ELSE 'flag_as_missing'
        END AS recommended_strategy
    FROM missing_analysis
)
SELECT 
    table_name,
    column_name,
    missing_count,
    total_count,
    missing_percentage,
    ROUND(mean_value, 2) AS imputation_value,
    recommended_strategy
FROM imputation_strategy;

-- Create imputed dataset with audit trail
CREATE TABLE customers_imputed AS
WITH imputation_values AS (
    SELECT 
        AVG(age) AS mean_age,
        AVG(income) AS mean_income,
        AVG(credit_score) AS mean_credit_score
    FROM customers
    WHERE age IS NOT NULL AND income IS NOT NULL AND credit_score IS NOT NULL
)
SELECT 
    customer_id,
    first_name,
    last_name,
    customer_segment,
    region,
    -- Original values
    age AS original_age,
    income AS original_income,
    credit_score AS original_credit_score,
    -- Imputed values
    COALESCE(age, (SELECT mean_age FROM imputation_values)) AS age,
    COALESCE(income, (SELECT mean_income FROM imputation_values)) AS income,
    COALESCE(credit_score, (SELECT mean_credit_score FROM imputation_values)) AS credit_score,
    -- Imputation flags
    CASE WHEN age IS NULL THEN 1 ELSE 0 END AS age_imputed,
    CASE WHEN income IS NULL THEN 1 ELSE 0 END AS income_imputed,
    CASE WHEN credit_score IS NULL THEN 1 ELSE 0 END AS credit_score_imputed,
    CURRENT_TIMESTAMP AS imputation_timestamp
FROM customers;

-- Insert imputation metadata
INSERT INTO imputation_metadata (table_name, column_name, imputation_method, imputation_value, missing_count, total_count, missing_percentage)
SELECT 
    'customers' AS table_name,
    'age' AS column_name,
    'mean_imputation' AS imputation_method,
    AVG(age) AS imputation_value,
    COUNT(*) - COUNT(age) AS missing_count,
    COUNT(*) AS total_count,
    ROUND((COUNT(*) - COUNT(age)) * 100.0 / COUNT(*), 2) AS missing_percentage
FROM customers;
```

#### Explanation
1. **COALESCE Function**: Returns first non-NULL value from a list
2. **Window Functions**: Calculate means within specific partitions
3. **Conditional Logic**: Choose imputation strategy based on data characteristics
4. **Audit Trail**: Track which values were imputed and methods used
5. **Fallback Strategy**: Multiple levels of imputation (group → global)

#### Use Cases
- **ML Preprocessing**: Ensure complete datasets for model training
- **Data Quality**: Improve dataset completeness for analysis
- **Real-time Systems**: Handle missing values in streaming data
- **A/B Testing**: Compare different imputation strategies
- **Regulatory Compliance**: Maintain data completeness requirements

#### Best Practices
- Analyze missing data patterns before choosing imputation method
- Document imputation decisions and maintain audit trails
- Consider domain knowledge when selecting imputation strategies
- Test multiple imputation methods and compare model performance
- Create indicator variables to flag imputed values

#### Pitfalls
- **Bias Introduction**: Mean imputation can reduce variance and alter distributions
- **Correlation Distortion**: May weaken relationships between variables
- **Outlier Sensitivity**: Means can be skewed by extreme values
- **Temporal Patterns**: Simple means ignore time-based trends
- **Group Differences**: Global means may not represent subgroup characteristics

#### Debugging
```sql
-- Validate imputation results
SELECT 
    'Original' AS dataset,
    COUNT(*) AS total_rows,
    COUNT(age) AS age_complete,
    COUNT(income) AS income_complete,
    AVG(age) AS avg_age
FROM customers
UNION ALL
SELECT 
    'Imputed' AS dataset,
    COUNT(*) AS total_rows,
    COUNT(age) AS age_complete,
    COUNT(income) AS income_complete,
    AVG(age) AS avg_age
FROM customers_imputed;

-- Check imputation distribution
SELECT 
    age_imputed,
    COUNT(*) AS count,
    AVG(age) AS avg_age,
    STDDEV(age) AS std_age
FROM customers_imputed
GROUP BY age_imputed;
```

#### Optimization
- **Batch Processing**: Process large datasets in chunks to manage memory
- **Indexing**: Create indexes on grouping columns for faster aggregations
- **Materialized Views**: Store computed means for repeated imputation tasks
- **Parallel Processing**: Use parallel execution for independent imputation operations

---

## Question 12

**Create a SQL query that normalizes a column (scales between 0 and 1).**

### Answer:

#### Theory
Normalization (Min-Max scaling) transforms features to a common scale between 0 and 1, which is essential for ML algorithms sensitive to feature magnitude. This prevents features with larger scales from dominating the model and ensures equal contribution from all variables.

#### Code Example
```sql
-- Basic Min-Max Normalization
WITH normalization_params AS (
    SELECT 
        MIN(salary) AS min_salary,
        MAX(salary) AS max_salary,
        MIN(age) AS min_age,
        MAX(age) AS max_age,
        MIN(experience_years) AS min_experience,
        MAX(experience_years) AS max_experience
    FROM employees
),
normalized_data AS (
    SELECT 
        employee_id,
        first_name,
        last_name,
        department,
        salary,
        age,
        experience_years,
        -- Min-Max normalization formula: (x - min) / (max - min)
        CASE 
            WHEN (SELECT max_salary - min_salary FROM normalization_params) > 0 
            THEN (salary - (SELECT min_salary FROM normalization_params)) / 
                 (SELECT max_salary - min_salary FROM normalization_params)
            ELSE 0.5  -- If all values are the same
        END AS salary_normalized,
        
        CASE 
            WHEN (SELECT max_age - min_age FROM normalization_params) > 0 
            THEN (age - (SELECT min_age FROM normalization_params)) / 
                 (SELECT max_age - min_age FROM normalization_params)
            ELSE 0.5
        END AS age_normalized,
        
        CASE 
            WHEN (SELECT max_experience - min_experience FROM normalization_params) > 0 
            THEN (experience_years - (SELECT min_experience FROM normalization_params)) / 
                 (SELECT max_experience - min_experience FROM normalization_params)
            ELSE 0.5
        END AS experience_normalized
    FROM employees
)
SELECT 
    employee_id,
    first_name,
    last_name,
    department,
    -- Original values
    salary,
    age,
    experience_years,
    -- Normalized values (0-1 scale)
    ROUND(salary_normalized, 4) AS salary_norm,
    ROUND(age_normalized, 4) AS age_norm,
    ROUND(experience_normalized, 4) AS experience_norm
FROM normalized_data
ORDER BY employee_id;

-- Group-wise Normalization
WITH department_stats AS (
    SELECT 
        department,
        MIN(salary) AS dept_min_salary,
        MAX(salary) AS dept_max_salary,
        MIN(performance_score) AS dept_min_performance,
        MAX(performance_score) AS dept_max_performance
    FROM employees
    GROUP BY department
),
department_normalized AS (
    SELECT 
        e.employee_id,
        e.department,
        e.salary,
        e.performance_score,
        -- Normalize within department groups
        CASE 
            WHEN (ds.dept_max_salary - ds.dept_min_salary) > 0 
            THEN (e.salary - ds.dept_min_salary) / (ds.dept_max_salary - ds.dept_min_salary)
            ELSE 0.5
        END AS salary_dept_normalized,
        
        CASE 
            WHEN (ds.dept_max_performance - ds.dept_min_performance) > 0 
            THEN (e.performance_score - ds.dept_min_performance) / (ds.dept_max_performance - ds.dept_min_performance)
            ELSE 0.5
        END AS performance_dept_normalized
    FROM employees e
    JOIN department_stats ds ON e.department = ds.department
)
SELECT 
    employee_id,
    department,
    salary,
    performance_score,
    ROUND(salary_dept_normalized, 4) AS salary_dept_norm,
    ROUND(performance_dept_normalized, 4) AS performance_dept_norm
FROM department_normalized;

-- Robust Normalization using Percentiles
WITH percentile_stats AS (
    SELECT 
        PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY revenue) AS revenue_p5,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY revenue) AS revenue_p95,
        PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY customer_count) AS customers_p5,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY customer_count) AS customers_p95
    FROM company_metrics
),
robust_normalized AS (
    SELECT 
        company_id,
        company_name,
        revenue,
        customer_count,
        -- Robust normalization using 5th and 95th percentiles
        CASE 
            WHEN (SELECT revenue_p95 - revenue_p5 FROM percentile_stats) > 0 
            THEN GREATEST(0, LEAST(1, 
                (revenue - (SELECT revenue_p5 FROM percentile_stats)) / 
                (SELECT revenue_p95 - revenue_p5 FROM percentile_stats)
            ))
            ELSE 0.5
        END AS revenue_robust_norm,
        
        CASE 
            WHEN (SELECT customers_p95 - customers_p5 FROM percentile_stats) > 0 
            THEN GREATEST(0, LEAST(1,
                (customer_count - (SELECT customers_p5 FROM percentile_stats)) / 
                (SELECT customers_p95 - customers_p5 FROM percentile_stats)
            ))
            ELSE 0.5
        END AS customers_robust_norm
    FROM company_metrics
)
SELECT 
    company_id,
    company_name,
    revenue,
    customer_count,
    ROUND(revenue_robust_norm, 4) AS revenue_norm,
    ROUND(customers_robust_norm, 4) AS customers_norm
FROM robust_normalized;

-- Time-series Normalization with Rolling Windows
SELECT 
    product_id,
    sales_date,
    daily_sales,
    
    -- Rolling min-max normalization (30-day window)
    (daily_sales - MIN(daily_sales) OVER (
        PARTITION BY product_id 
        ORDER BY sales_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    )) / NULLIF(
        MAX(daily_sales) OVER (
            PARTITION BY product_id 
            ORDER BY sales_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) - MIN(daily_sales) OVER (
            PARTITION BY product_id 
            ORDER BY sales_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ), 0
    ) AS sales_rolling_normalized,
    
    -- Global normalization for comparison
    (daily_sales - MIN(daily_sales) OVER (PARTITION BY product_id)) / 
    NULLIF(
        MAX(daily_sales) OVER (PARTITION BY product_id) - 
        MIN(daily_sales) OVER (PARTITION BY product_id), 0
    ) AS sales_global_normalized
FROM daily_product_sales
ORDER BY product_id, sales_date;

-- Comprehensive Normalization Function
CREATE OR REPLACE FUNCTION normalize_column(
    table_name TEXT,
    column_name TEXT,
    method TEXT DEFAULT 'minmax'
)
RETURNS TABLE (
    original_value NUMERIC,
    normalized_value NUMERIC
) AS $$
DECLARE
    min_val NUMERIC;
    max_val NUMERIC;
    mean_val NUMERIC;
    std_val NUMERIC;
    sql_query TEXT;
BEGIN
    -- Get statistics
    sql_query := FORMAT('SELECT MIN(%s), MAX(%s), AVG(%s), STDDEV(%s) FROM %s',
                       column_name, column_name, column_name, column_name, table_name);
    EXECUTE sql_query INTO min_val, max_val, mean_val, std_val;
    
    -- Apply normalization based on method
    IF method = 'minmax' THEN
        sql_query := FORMAT(
            'SELECT %s, CASE WHEN %s > 0 THEN (%s - %s) / %s ELSE 0.5 END FROM %s',
            column_name, 
            max_val - min_val,
            column_name, min_val, max_val - min_val,
            table_name
        );
    ELSIF method = 'zscore' THEN
        sql_query := FORMAT(
            'SELECT %s, CASE WHEN %s > 0 THEN (%s - %s) / %s ELSE 0 END FROM %s',
            column_name,
            std_val,
            column_name, mean_val, std_val,
            table_name
        );
    END IF;
    
    RETURN QUERY EXECUTE sql_query;
END;
$$ LANGUAGE plpgsql;

-- Advanced Normalization with Multiple Methods
WITH advanced_normalization AS (
    SELECT 
        customer_id,
        purchase_amount,
        transaction_count,
        days_since_last_purchase,
        
        -- Method 1: Standard Min-Max (0-1)
        (purchase_amount - MIN(purchase_amount) OVER ()) / 
        NULLIF(MAX(purchase_amount) OVER () - MIN(purchase_amount) OVER (), 0) AS amount_minmax,
        
        -- Method 2: Z-score normalization
        (purchase_amount - AVG(purchase_amount) OVER ()) / 
        NULLIF(STDDEV(purchase_amount) OVER (), 0) AS amount_zscore,
        
        -- Method 3: Decimal scaling
        purchase_amount / POWER(10, CEIL(LOG10(MAX(ABS(purchase_amount)) OVER ()))) AS amount_decimal_scaled,
        
        -- Method 4: Unit vector scaling
        purchase_amount / SQRT(SUM(POWER(purchase_amount, 2)) OVER ()) AS amount_unit_vector,
        
        -- Method 5: Quantile-based normalization
        PERCENT_RANK() OVER (ORDER BY purchase_amount) AS amount_quantile_norm
    FROM customer_transactions
)
SELECT 
    customer_id,
    purchase_amount,
    ROUND(amount_minmax, 4) AS minmax_norm,
    ROUND(amount_zscore, 4) AS zscore_norm,
    ROUND(amount_decimal_scaled, 4) AS decimal_norm,
    ROUND(amount_unit_vector, 6) AS unit_vector_norm,
    ROUND(amount_quantile_norm, 4) AS quantile_norm
FROM advanced_normalization
ORDER BY customer_id;
```

#### Explanation
1. **Min-Max Formula**: (x - min) / (max - min) scales values to [0,1] range
2. **Window Functions**: Calculate statistics across entire dataset or groups
3. **NULLIF**: Prevents division by zero when max equals min
4. **GREATEST/LEAST**: Clamp values to ensure they stay within [0,1] bounds
5. **Group Normalization**: Normalize within specific categories or time windows

#### Use Cases
- **Machine Learning**: Prepare features for algorithms sensitive to scale
- **Data Visualization**: Create comparable scales for multi-variable charts
- **Performance Metrics**: Normalize KPIs across different business units
- **Anomaly Detection**: Standardize features for outlier detection
- **Recommendation Systems**: Normalize user ratings and preferences

#### Best Practices
- Handle edge cases where all values are identical (zero variance)
- Consider robust normalization using percentiles for outlier resistance
- Document normalization parameters for inverse transformation
- Apply same normalization to training and test datasets
- Monitor for data drift that may affect normalization parameters

#### Pitfalls
- **Information Loss**: Min-max normalization is sensitive to outliers
- **Zero Variance**: Constant columns cannot be normalized meaningfully
- **Data Leakage**: Using future data to calculate normalization parameters
- **Scale Drift**: Distribution changes over time may require re-normalization
- **Domain Boundaries**: Some domains may require different scaling ranges

#### Debugging
```sql
-- Verify normalization results
SELECT 
    MIN(salary_norm) AS min_normalized,
    MAX(salary_norm) AS max_normalized,
    AVG(salary_norm) AS avg_normalized,
    STDDEV(salary_norm) AS std_normalized
FROM normalized_employees;

-- Check for edge cases
SELECT 
    COUNT(CASE WHEN salary_norm < 0 THEN 1 END) AS below_zero,
    COUNT(CASE WHEN salary_norm > 1 THEN 1 END) AS above_one,
    COUNT(CASE WHEN salary_norm IS NULL THEN 1 END) AS null_values
FROM normalized_employees;
```

#### Optimization
- **Materialized Views**: Store normalization parameters for reuse
- **Indexing**: Index original columns for faster min/max calculations
- **Batch Processing**: Normalize large datasets in manageable chunks
- **Caching**: Cache computed statistics for repeated normalization operations

---

## Question 13

**Generate a feature that is a count over a rolling time window using SQL.**

### Answer:

#### Theory
Rolling window counts create temporal features that capture activity patterns and trends over time. These features are essential for time series analysis, user behavior modeling, and detecting changes in activity levels for ML applications.

#### Code Example
```sql
-- Basic Rolling Count Features
SELECT 
    customer_id,
    transaction_date,
    transaction_amount,
    
    -- Rolling count of transactions (7-day window)
    COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS transactions_7d,
    
    -- Rolling count of transactions (30-day window)
    COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS transactions_30d,
    
    -- Rolling count of high-value transactions (>$1000)
    COUNT(CASE WHEN transaction_amount > 1000 THEN 1 END) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS high_value_transactions_30d,
    
    -- Rolling distinct merchant count
    COUNT(DISTINCT merchant_id) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS unique_merchants_30d
FROM customer_transactions
ORDER BY customer_id, transaction_date;

-- Advanced Rolling Features with Multiple Windows
WITH rolling_features AS (
    SELECT 
        user_id,
        login_date,
        session_duration,
        pages_viewed,
        
        -- Multiple rolling window counts
        COUNT(*) OVER (
            PARTITION BY user_id 
            ORDER BY login_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS logins_7d,
        
        COUNT(*) OVER (
            PARTITION BY user_id 
            ORDER BY login_date 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS logins_14d,
        
        COUNT(*) OVER (
            PARTITION BY user_id 
            ORDER BY login_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS logins_30d,
        
        -- Rolling count of long sessions (>30 minutes)
        COUNT(CASE WHEN session_duration > 30 THEN 1 END) OVER (
            PARTITION BY user_id 
            ORDER BY login_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS long_sessions_30d,
        
        -- Rolling count of highly engaged sessions (>10 pages)
        COUNT(CASE WHEN pages_viewed > 10 THEN 1 END) OVER (
            PARTITION BY user_id 
            ORDER BY login_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS engaged_sessions_30d,
        
        -- Rolling sum of pages viewed
        SUM(pages_viewed) OVER (
            PARTITION BY user_id 
            ORDER BY login_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS total_pages_7d
    FROM user_sessions
),
derived_features AS (
    SELECT 
        user_id,
        login_date,
        logins_7d,
        logins_14d,
        logins_30d,
        long_sessions_30d,
        engaged_sessions_30d,
        total_pages_7d,
        
        -- Derived ratio features
        CASE 
            WHEN logins_30d > 0 THEN CAST(long_sessions_30d AS FLOAT) / logins_30d 
            ELSE 0 
        END AS long_session_rate_30d,
        
        CASE 
            WHEN logins_30d > 0 THEN CAST(engaged_sessions_30d AS FLOAT) / logins_30d 
            ELSE 0 
        END AS engagement_rate_30d,
        
        -- Activity trend indicators
        CASE 
            WHEN logins_14d > 0 THEN CAST(logins_7d AS FLOAT) / (logins_14d - logins_7d + 1)
            ELSE 0
        END AS recent_activity_ratio,
        
        -- User engagement categories
        CASE 
            WHEN logins_30d >= 20 THEN 'high_frequency'
            WHEN logins_30d >= 10 THEN 'medium_frequency'
            WHEN logins_30d >= 5 THEN 'low_frequency'
            ELSE 'rare_user'
        END AS user_activity_category
    FROM rolling_features
)
SELECT * FROM derived_features
ORDER BY user_id, login_date;

-- Time-based Rolling Counts (Date Range Windows)
SELECT 
    product_id,
    order_date,
    quantity_sold,
    
    -- Count orders in last 7 days (date-based window)
    (SELECT COUNT(*) 
     FROM product_orders po2 
     WHERE po2.product_id = po1.product_id 
       AND po2.order_date BETWEEN po1.order_date - INTERVAL 7 DAY AND po1.order_date
    ) AS orders_last_7_days,
    
    -- Count orders in last 30 days
    (SELECT COUNT(*) 
     FROM product_orders po2 
     WHERE po2.product_id = po1.product_id 
       AND po2.order_date BETWEEN po1.order_date - INTERVAL 30 DAY AND po1.order_date
    ) AS orders_last_30_days,
    
    -- Count unique customers in last 30 days
    (SELECT COUNT(DISTINCT customer_id) 
     FROM product_orders po2 
     WHERE po2.product_id = po1.product_id 
       AND po2.order_date BETWEEN po1.order_date - INTERVAL 30 DAY AND po1.order_date
    ) AS unique_customers_30d,
    
    -- Count weekend orders in last 30 days
    (SELECT COUNT(*) 
     FROM product_orders po2 
     WHERE po2.product_id = po1.product_id 
       AND po2.order_date BETWEEN po1.order_date - INTERVAL 30 DAY AND po1.order_date
       AND EXTRACT(DOW FROM po2.order_date) IN (0, 6)
    ) AS weekend_orders_30d
FROM product_orders po1
ORDER BY product_id, order_date;

-- Multi-dimensional Rolling Counts
WITH multi_dimensional_counts AS (
    SELECT 
        event_date,
        user_id,
        event_type,
        channel,
        device_type,
        
        -- Count by event type in rolling window
        COUNT(CASE WHEN event_type = 'page_view' THEN 1 END) OVER (
            PARTITION BY user_id 
            ORDER BY event_date 
            RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW
        ) AS page_views_7d,
        
        COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) OVER (
            PARTITION BY user_id 
            ORDER BY event_date 
            RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW
        ) AS purchases_7d,
        
        COUNT(CASE WHEN event_type = 'search' THEN 1 END) OVER (
            PARTITION BY user_id 
            ORDER BY event_date 
            RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW
        ) AS searches_7d,
        
        -- Count by channel in rolling window
        COUNT(CASE WHEN channel = 'mobile' THEN 1 END) OVER (
            PARTITION BY user_id 
            ORDER BY event_date 
            RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW
        ) AS mobile_events_7d,
        
        COUNT(CASE WHEN channel = 'web' THEN 1 END) OVER (
            PARTITION BY user_id 
            ORDER BY event_date 
            RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW
        ) AS web_events_7d,
        
        -- Count distinct dimensions
        COUNT(DISTINCT channel) OVER (
            PARTITION BY user_id 
            ORDER BY event_date 
            RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW
        ) AS channels_used_30d,
        
        COUNT(DISTINCT device_type) OVER (
            PARTITION BY user_id 
            ORDER BY event_date 
            RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW
        ) AS devices_used_30d
    FROM user_events
),
behavioral_features AS (
    SELECT 
        user_id,
        event_date,
        page_views_7d,
        purchases_7d,
        searches_7d,
        mobile_events_7d,
        web_events_7d,
        channels_used_30d,
        devices_used_30d,
        
        -- Behavioral ratios
        CASE 
            WHEN page_views_7d > 0 THEN CAST(purchases_7d AS FLOAT) / page_views_7d 
            ELSE 0 
        END AS conversion_rate_7d,
        
        CASE 
            WHEN page_views_7d > 0 THEN CAST(searches_7d AS FLOAT) / page_views_7d 
            ELSE 0 
        END AS search_rate_7d,
        
        CASE 
            WHEN mobile_events_7d + web_events_7d > 0 
            THEN CAST(mobile_events_7d AS FLOAT) / (mobile_events_7d + web_events_7d)
            ELSE 0 
        END AS mobile_preference_7d,
        
        -- User behavior categories
        CASE 
            WHEN purchases_7d >= 3 THEN 'frequent_buyer'
            WHEN purchases_7d >= 1 THEN 'occasional_buyer'
            WHEN page_views_7d >= 10 THEN 'browser'
            ELSE 'inactive'
        END AS user_behavior_segment
    FROM multi_dimensional_counts
)
SELECT * FROM behavioral_features
ORDER BY user_id, event_date;

-- Complex Rolling Count with Conditions and Aggregations
SELECT 
    customer_id,
    transaction_date,
    amount,
    merchant_category,
    
    -- Rolling count with amount thresholds
    COUNT(CASE WHEN amount >= 100 THEN 1 END) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS large_transactions_30d,
    
    -- Rolling count by merchant category
    COUNT(CASE WHEN merchant_category = 'grocery' THEN 1 END) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS grocery_transactions_30d,
    
    COUNT(CASE WHEN merchant_category = 'gas' THEN 1 END) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS gas_transactions_30d,
    
    -- Rolling count of weekend transactions
    COUNT(CASE WHEN EXTRACT(DOW FROM transaction_date) IN (0, 6) THEN 1 END) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS weekend_transactions_30d,
    
    -- Rolling velocity features
    COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS velocity_7d,
    
    COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 13 PRECEDING AND 7 PRECEDING
    ) AS velocity_prev_7d,
    
    -- Acceleration (change in velocity)
    COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) - COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 13 PRECEDING AND 7 PRECEDING
    ) AS velocity_acceleration
FROM financial_transactions
ORDER BY customer_id, transaction_date;
```

#### Explanation
1. **Window Functions**: Use COUNT() with OVER clause for rolling counts
2. **ROWS BETWEEN**: Defines the rolling window size (n PRECEDING)
3. **RANGE BETWEEN**: Uses actual date/time intervals for windows
4. **Conditional Counts**: COUNT(CASE WHEN condition THEN 1 END) for filtered counts
5. **DISTINCT Counts**: COUNT(DISTINCT column) for unique value counts

#### Use Cases
- **User Activity Tracking**: Monitor engagement patterns over time
- **Fraud Detection**: Identify unusual transaction patterns
- **Customer Segmentation**: Classify users based on activity levels
- **Demand Forecasting**: Analyze purchase frequency trends
- **Marketing Attribution**: Track campaign response patterns

#### Best Practices
- Choose appropriate window sizes based on business cycles
- Consider multiple time horizons for comprehensive feature engineering
- Use conditional counts to capture specific event types
- Combine with ratio features for normalized comparisons
- Handle edge cases where windows contain insufficient data

#### Pitfalls
- **Performance Issues**: Large windows can be computationally expensive
- **Boundary Effects**: Early records may have incomplete windows
- **Data Sparsity**: Windows may contain zero counts for inactive periods
- **Memory Usage**: Complex window functions can consume significant resources
- **Temporal Alignment**: Ensure proper ordering for accurate rolling calculations

#### Debugging
```sql
-- Validate rolling count calculations
SELECT 
    customer_id,
    transaction_date,
    transactions_7d,
    -- Manual count for verification
    (SELECT COUNT(*) 
     FROM customer_transactions ct2 
     WHERE ct2.customer_id = ct1.customer_id 
       AND ct2.transaction_date BETWEEN ct1.transaction_date - INTERVAL 6 DAY 
                                    AND ct1.transaction_date
    ) AS manual_count_7d
FROM customer_transaction_features ct1
WHERE customer_id = 12345
ORDER BY transaction_date DESC
LIMIT 10;
```

#### Optimization
- **Indexing**: Create indexes on partition and order columns
- **Window Size**: Limit window sizes to necessary ranges
- **Incremental Processing**: Update only new rolling counts instead of recalculating all
- **Partitioning**: Partition tables by date for better performance

---

## Question 14

**Code an SQL function that categorizes continuous variables into bins.**

### Answer:

#### Theory
Binning (discretization) transforms continuous variables into categorical intervals, reducing noise, handling outliers, and creating interpretable features for ML models. This technique is essential for decision trees, association rules, and when dealing with non-linear relationships.

#### Code Example
```sql
-- Method 1: Equal-Width Binning
SELECT 
    customer_id,
    age,
    income,
    credit_score,
    
    -- Equal-width age bins
    CASE 
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 45 THEN '35-44'
        WHEN age < 55 THEN '45-54'
        WHEN age < 65 THEN '55-64'
        ELSE '65+'
    END AS age_bin,
    
    -- Equal-width income bins (10 bins)
    CASE 
        WHEN income < 20000 THEN 'Income_01_0-20k'
        WHEN income < 40000 THEN 'Income_02_20-40k'
        WHEN income < 60000 THEN 'Income_03_40-60k'
        WHEN income < 80000 THEN 'Income_04_60-80k'
        WHEN income < 100000 THEN 'Income_05_80-100k'
        WHEN income < 120000 THEN 'Income_06_100-120k'
        WHEN income < 140000 THEN 'Income_07_120-140k'
        WHEN income < 160000 THEN 'Income_08_140-160k'
        WHEN income < 180000 THEN 'Income_09_160-180k'
        ELSE 'Income_10_180k+'
    END AS income_bin,
    
    -- Credit score bins (standard ranges)
    CASE 
        WHEN credit_score < 580 THEN 'Poor'
        WHEN credit_score < 670 THEN 'Fair'
        WHEN credit_score < 740 THEN 'Good'
        WHEN credit_score < 800 THEN 'Very_Good'
        ELSE 'Excellent'
    END AS credit_bin
FROM customers;

-- Method 2: Equal-Frequency (Quantile) Binning
WITH quantile_bins AS (
    SELECT 
        customer_id,
        income,
        account_balance,
        -- Calculate quantiles for equal-frequency binning
        NTILE(5) OVER (ORDER BY income) AS income_quintile,
        NTILE(10) OVER (ORDER BY account_balance) AS balance_decile,
        NTILE(4) OVER (ORDER BY income) AS income_quartile,
        
        -- Get actual quantile values
        PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY income) OVER () AS income_p20,
        PERCENTILE_CONT(0.4) WITHIN GROUP (ORDER BY income) OVER () AS income_p40,
        PERCENTILE_CONT(0.6) WITHIN GROUP (ORDER BY income) OVER () AS income_p60,
        PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY income) OVER () AS income_p80
    FROM customers
),
labeled_quantiles AS (
    SELECT 
        customer_id,
        income,
        account_balance,
        income_quintile,
        balance_decile,
        income_quartile,
        
        -- Create descriptive labels for quantile bins
        CASE income_quintile
            WHEN 1 THEN 'Bottom_20pct'
            WHEN 2 THEN 'Lower_Middle_20pct'
            WHEN 3 THEN 'Middle_20pct'
            WHEN 4 THEN 'Upper_Middle_20pct'
            WHEN 5 THEN 'Top_20pct'
        END AS income_quintile_label,
        
        CASE income_quartile
            WHEN 1 THEN 'Q1_Bottom_25pct'
            WHEN 2 THEN 'Q2_Lower_Middle_25pct'
            WHEN 3 THEN 'Q3_Upper_Middle_25pct'
            WHEN 4 THEN 'Q4_Top_25pct'
        END AS income_quartile_label,
        
        'Decile_' || LPAD(balance_decile::TEXT, 2, '0') AS balance_decile_label
    FROM quantile_bins
)
SELECT * FROM labeled_quantiles;

-- Method 3: Custom Business Logic Binning
WITH business_logic_bins AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount,
        transaction_frequency_30d,
        days_since_last_transaction,
        
        -- Transaction amount bins (business-specific thresholds)
        CASE 
            WHEN transaction_amount <= 10 THEN 'Micro'
            WHEN transaction_amount <= 50 THEN 'Small'
            WHEN transaction_amount <= 200 THEN 'Medium'
            WHEN transaction_amount <= 1000 THEN 'Large'
            WHEN transaction_amount <= 5000 THEN 'Very_Large'
            ELSE 'Exceptional'
        END AS amount_category,
        
        -- Frequency bins (activity level)
        CASE 
            WHEN transaction_frequency_30d = 0 THEN 'Inactive'
            WHEN transaction_frequency_30d <= 2 THEN 'Low_Activity'
            WHEN transaction_frequency_30d <= 10 THEN 'Medium_Activity'
            WHEN transaction_frequency_30d <= 25 THEN 'High_Activity'
            ELSE 'Very_High_Activity'
        END AS frequency_category,
        
        -- Recency bins
        CASE 
            WHEN days_since_last_transaction <= 1 THEN 'Very_Recent'
            WHEN days_since_last_transaction <= 7 THEN 'Recent'
            WHEN days_since_last_transaction <= 30 THEN 'Moderate'
            WHEN days_since_last_transaction <= 90 THEN 'Old'
            ELSE 'Very_Old'
        END AS recency_category,
        
        -- Combined RFM-style binning
        CONCAT(
            CASE 
                WHEN transaction_frequency_30d <= 2 THEN 'L'
                WHEN transaction_frequency_30d <= 10 THEN 'M'
                ELSE 'H'
            END,
            CASE 
                WHEN days_since_last_transaction <= 7 THEN 'R'
                WHEN days_since_last_transaction <= 30 THEN 'M'
                ELSE 'O'
            END,
            CASE 
                WHEN transaction_amount <= 50 THEN 'S'
                WHEN transaction_amount <= 200 THEN 'M'
                ELSE 'L'
            END
        ) AS rfm_segment
    FROM customer_transaction_features
)
SELECT * FROM business_logic_bins;

-- Method 4: Adaptive Binning Function
CREATE OR REPLACE FUNCTION adaptive_binning(
    input_value NUMERIC,
    min_value NUMERIC,
    max_value NUMERIC,
    num_bins INTEGER DEFAULT 5,
    method VARCHAR(20) DEFAULT 'equal_width'
)
RETURNS VARCHAR(50) AS $$
DECLARE
    bin_width NUMERIC;
    bin_number INTEGER;
    result VARCHAR(50);
BEGIN
    -- Handle NULL values
    IF input_value IS NULL THEN
        RETURN 'Unknown';
    END IF;
    
    -- Handle edge cases
    IF min_value = max_value THEN
        RETURN 'Single_Value';
    END IF;
    
    IF method = 'equal_width' THEN
        bin_width := (max_value - min_value) / num_bins;
        bin_number := LEAST(
            num_bins, 
            GREATEST(1, CEIL((input_value - min_value) / bin_width))
        );
        
        result := 'Bin_' || LPAD(bin_number::TEXT, 2, '0') || '_' ||
                 ROUND(min_value + (bin_number - 1) * bin_width, 0)::TEXT || '_to_' ||
                 ROUND(min_value + bin_number * bin_width, 0)::TEXT;
                 
    ELSIF method = 'log_scale' THEN
        -- Logarithmic binning for skewed distributions
        IF input_value <= 0 THEN
            result := 'Bin_00_Zero_or_Negative';
        ELSE
            bin_number := LEAST(
                num_bins,
                GREATEST(1, CEIL(LOG(input_value / min_value) / LOG(max_value / min_value) * num_bins))
            );
            result := 'LogBin_' || LPAD(bin_number::TEXT, 2, '0');
        END IF;
    END IF;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Using the adaptive binning function
WITH binning_stats AS (
    SELECT 
        MIN(salary) AS min_salary,
        MAX(salary) AS max_salary,
        MIN(experience_years) AS min_experience,
        MAX(experience_years) AS max_experience
    FROM employees
)
SELECT 
    employee_id,
    salary,
    experience_years,
    adaptive_binning(salary, 
                    (SELECT min_salary FROM binning_stats), 
                    (SELECT max_salary FROM binning_stats), 
                    8, 'equal_width') AS salary_bin,
    adaptive_binning(experience_years, 
                    (SELECT min_experience FROM binning_stats), 
                    (SELECT max_experience FROM binning_stats), 
                    5, 'equal_width') AS experience_bin
FROM employees;

-- Method 5: Statistical Binning with Outlier Handling
WITH outlier_robust_bins AS (
    SELECT 
        product_id,
        price,
        sales_volume,
        -- Calculate percentiles for robust binning
        PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY price) OVER () AS price_p5,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) OVER () AS price_q1,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) OVER () AS price_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) OVER () AS price_q3,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY price) OVER () AS price_p95,
        
        -- Z-score for outlier detection
        (price - AVG(price) OVER ()) / STDDEV(price) OVER () AS price_zscore
    FROM products
),
robust_binned AS (
    SELECT 
        product_id,
        price,
        sales_volume,
        price_zscore,
        
        -- Outlier-aware binning
        CASE 
            WHEN ABS(price_zscore) > 3 THEN 'Outlier'
            WHEN price <= price_q1 THEN 'Low'
            WHEN price <= price_median THEN 'Medium_Low'
            WHEN price <= price_q3 THEN 'Medium_High'
            ELSE 'High'
        END AS price_category,
        
        -- Percentile-based binning (excluding outliers)
        CASE 
            WHEN price < price_p5 THEN 'Bottom_5pct'
            WHEN price <= price_q1 THEN 'Low_Quartile'
            WHEN price <= price_median THEN 'Second_Quartile'
            WHEN price <= price_q3 THEN 'Third_Quartile'
            WHEN price <= price_p95 THEN 'Top_Quartile'
            ELSE 'Top_5pct'
        END AS price_percentile_bin
    FROM outlier_robust_bins
)
SELECT * FROM robust_binned;

-- Method 6: Dynamic Multi-Column Binning
CREATE TABLE binning_rules (
    rule_id SERIAL PRIMARY KEY,
    column_name VARCHAR(50),
    bin_type VARCHAR(20),
    bin_boundaries JSONB,
    bin_labels JSONB,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert binning rules
INSERT INTO binning_rules (column_name, bin_type, bin_boundaries, bin_labels)
VALUES 
    ('age', 'custom', '[18, 25, 35, 45, 55, 65, 100]', '["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]'),
    ('income', 'quantile', '[0.2, 0.4, 0.6, 0.8, 1.0]', '["Low", "Medium_Low", "Medium", "Medium_High", "High"]');

-- Apply dynamic binning
WITH dynamic_binning AS (
    SELECT 
        c.customer_id,
        c.age,
        c.income,
        br_age.bin_boundaries AS age_boundaries,
        br_age.bin_labels AS age_labels,
        br_income.bin_boundaries AS income_boundaries,
        br_income.bin_labels AS income_labels
    FROM customers c
    CROSS JOIN binning_rules br_age WHERE br_age.column_name = 'age'
    CROSS JOIN binning_rules br_income WHERE br_income.column_name = 'income'
)
SELECT 
    customer_id,
    age,
    income,
    -- Apply age binning
    CASE 
        WHEN age < (age_boundaries->0)::NUMERIC THEN 'Under_18'
        WHEN age < (age_boundaries->1)::NUMERIC THEN (age_labels->0)::TEXT
        WHEN age < (age_boundaries->2)::NUMERIC THEN (age_labels->1)::TEXT
        WHEN age < (age_boundaries->3)::NUMERIC THEN (age_labels->2)::TEXT
        WHEN age < (age_boundaries->4)::NUMERIC THEN (age_labels->3)::TEXT
        WHEN age < (age_boundaries->5)::NUMERIC THEN (age_labels->4)::TEXT
        ELSE (age_labels->5)::TEXT
    END AS age_bin_dynamic
FROM dynamic_binning;
```

#### Explanation
1. **CASE Statements**: Create custom bin boundaries and labels
2. **NTILE Function**: Divides data into equal-frequency bins
3. **PERCENTILE_CONT**: Calculates quantile boundaries for robust binning
4. **Statistical Methods**: Use mean, standard deviation for outlier-aware binning
5. **Dynamic Rules**: Store binning logic in tables for flexibility

#### Use Cases
- **Credit Scoring**: Categorize risk levels and credit worthiness
- **Customer Segmentation**: Group customers by behavior and demographics
- **A/B Testing**: Create balanced test groups based on user characteristics
- **Recommendation Systems**: Bin user preferences and item attributes
- **Fraud Detection**: Categorize transaction patterns and risk levels

#### Best Practices
- Choose bin boundaries based on domain knowledge and data distribution
- Ensure sufficient observations in each bin for statistical validity
- Handle edge cases and outliers appropriately
- Use descriptive labels that are interpretable by business users
- Document binning logic and validate bin distributions

#### Pitfalls
- **Information Loss**: Binning reduces precision and may lose important patterns
- **Arbitrary Boundaries**: Poorly chosen bins can introduce bias
- **Empty Bins**: Some bins may have very few or zero observations
- **Temporal Stability**: Bin boundaries may need adjustment over time
- **Overfitting**: Too many bins can lead to overfitting in ML models

#### Debugging
```sql
-- Validate bin distributions
SELECT 
    age_bin,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
    MIN(age) AS min_age,
    MAX(age) AS max_age,
    AVG(age) AS avg_age
FROM customer_bins
GROUP BY age_bin
ORDER BY MIN(age);
```

#### Optimization
- **Indexed Lookups**: Create indexes on frequently binned columns
- **Pre-computed Bins**: Store bin assignments to avoid repeated calculations
- **Batch Processing**: Apply binning to large datasets in manageable chunks
- **Memory Management**: Consider memory usage for complex binning operations

---

## Question 15

**Implement a SQL solution to compute the TF-IDF score for text data.**

### Answer:

#### Theory
TF-IDF (Term Frequency-Inverse Document Frequency) measures the importance of words in documents relative to a collection of documents. It combines how frequently a term appears in a document (TF) with how rare the term is across all documents (IDF), making it fundamental for text mining, search engines, and NLP feature engineering.

#### Code Example
```sql
-- Step 1: Text Preprocessing and Tokenization
WITH document_tokens AS (
    SELECT 
        document_id,
        document_text,
        -- Extract words (simplified tokenization)
        REGEXP_SPLIT_TO_TABLE(
            REGEXP_REPLACE(
                LOWER(TRIM(document_text)), 
                '[^a-z0-9\s]', ' ', 'g'
            ), 
            '\s+'
        ) AS token
    FROM documents
    WHERE document_text IS NOT NULL 
      AND LENGTH(TRIM(document_text)) > 0
),
-- Remove stop words and filter tokens
filtered_tokens AS (
    SELECT 
        document_id,
        token
    FROM document_tokens
    WHERE token NOT IN (
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
        'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 
        'might', 'must', 'can', 'shall', 'this', 'that', 'these', 'those', 'a', 
        'an', 'as', 'if', 'it', 'its', 'me', 'my', 'we', 'us', 'you', 'your', 
        'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their'
    )
    AND LENGTH(token) >= 3  -- Filter short tokens
    AND LENGTH(token) <= 20 -- Filter very long tokens
    AND token ~ '^[a-z]+$'  -- Only alphabetic tokens
),
-- Step 2: Calculate Term Frequency (TF)
term_frequency AS (
    SELECT 
        document_id,
        token,
        COUNT(*) AS term_count,
        -- Calculate total words per document
        SUM(COUNT(*)) OVER (PARTITION BY document_id) AS total_words_in_doc,
        -- TF = (term_count) / (total_words_in_document)
        CAST(COUNT(*) AS FLOAT) / SUM(COUNT(*)) OVER (PARTITION BY document_id) AS tf
    FROM filtered_tokens
    GROUP BY document_id, token
),
-- Step 3: Calculate Document Frequency and Inverse Document Frequency
document_frequency AS (
    SELECT 
        token,
        COUNT(DISTINCT document_id) AS document_frequency,
        -- Total number of documents
        (SELECT COUNT(DISTINCT document_id) FROM documents) AS total_documents,
        -- IDF = log(total_documents / document_frequency)
        LOG(
            CAST((SELECT COUNT(DISTINCT document_id) FROM documents) AS FLOAT) / 
            COUNT(DISTINCT document_id)
        ) AS idf
    FROM term_frequency
    GROUP BY token
),
-- Step 4: Calculate TF-IDF
tfidf_scores AS (
    SELECT 
        tf.document_id,
        tf.token,
        tf.term_count,
        tf.total_words_in_doc,
        ROUND(tf.tf, 6) AS tf,
        df.document_frequency,
        df.total_documents,
        ROUND(df.idf, 6) AS idf,
        -- TF-IDF = TF * IDF
        ROUND(tf.tf * df.idf, 6) AS tfidf_score
    FROM term_frequency tf
    JOIN document_frequency df ON tf.token = df.token
),
-- Step 5: Rank terms by TF-IDF within each document
ranked_terms AS (
    SELECT 
        document_id,
        token,
        term_count,
        tf,
        idf,
        tfidf_score,
        -- Rank terms within each document
        ROW_NUMBER() OVER (
            PARTITION BY document_id 
            ORDER BY tfidf_score DESC, term_count DESC
        ) AS term_rank,
        -- Calculate percentile rank
        PERCENT_RANK() OVER (
            PARTITION BY document_id 
            ORDER BY tfidf_score
        ) AS tfidf_percentile
    FROM tfidf_scores
)
SELECT 
    document_id,
    token,
    term_count,
    tf,
    idf,
    tfidf_score,
    term_rank,
    ROUND(tfidf_percentile, 4) AS tfidf_percentile
FROM ranked_terms
WHERE term_rank <= 20  -- Top 20 terms per document
ORDER BY document_id, term_rank;

-- Advanced TF-IDF with Multiple Variants
WITH advanced_tfidf AS (
    SELECT 
        tf.document_id,
        tf.token,
        tf.term_count,
        tf.tf,
        df.idf,
        
        -- Standard TF-IDF
        tf.tf * df.idf AS tfidf_standard,
        
        -- Log-normalized TF-IDF
        (1 + LOG(tf.term_count)) * df.idf AS tfidf_log_normalized,
        
        -- Sublinear TF scaling
        (0.5 + 0.5 * tf.tf / MAX(tf.tf) OVER (PARTITION BY tf.document_id)) * df.idf AS tfidf_sublinear,
        
        -- Smooth IDF
        LOG(1 + CAST(df.total_documents AS FLOAT) / df.document_frequency) AS smooth_idf,
        
        -- Smooth TF-IDF
        tf.tf * LOG(1 + CAST(df.total_documents AS FLOAT) / df.document_frequency) AS tfidf_smooth
    FROM term_frequency tf
    JOIN document_frequency df ON tf.token = df.token
)
SELECT 
    document_id,
    token,
    term_count,
    ROUND(tfidf_standard, 6) AS tfidf_standard,
    ROUND(tfidf_log_normalized, 6) AS tfidf_log_norm,
    ROUND(tfidf_sublinear, 6) AS tfidf_sublinear,
    ROUND(tfidf_smooth, 6) AS tfidf_smooth
FROM advanced_tfidf
ORDER BY document_id, tfidf_standard DESC;

-- TF-IDF for Document Similarity and Clustering
WITH document_vectors AS (
    -- Create TF-IDF vectors for each document
    SELECT 
        document_id,
        -- Create a vector of top terms and their TF-IDF scores
        STRING_AGG(
            token || ':' || ROUND(tfidf_score, 4), 
            '|' 
            ORDER BY tfidf_score DESC 
            LIMIT 50
        ) AS tfidf_vector,
        -- Create dense vector for top terms
        ARRAY_AGG(
            tfidf_score 
            ORDER BY tfidf_score DESC 
            LIMIT 50
        ) AS tfidf_array
    FROM (
        SELECT 
            document_id,
            token,
            tf * idf AS tfidf_score
        FROM term_frequency tf
        JOIN document_frequency df ON tf.token = df.token
    ) tfidf_data
    GROUP BY document_id
),
-- Calculate document similarity using cosine similarity approximation
document_similarity AS (
    SELECT 
        d1.document_id AS doc1_id,
        d2.document_id AS doc2_id,
        -- Simple vector similarity (using common terms)
        (
            SELECT COUNT(DISTINCT t1.token)
            FROM (
                SELECT document_id, token, tfidf_score
                FROM tfidf_scores 
                WHERE document_id = d1.document_id 
                  AND term_rank <= 20
            ) t1
            JOIN (
                SELECT document_id, token, tfidf_score
                FROM tfidf_scores 
                WHERE document_id = d2.document_id 
                  AND term_rank <= 20
            ) t2 ON t1.token = t2.token
        ) AS common_important_terms,
        -- Jaccard similarity of top terms
        (
            SELECT 
                CAST(COUNT(DISTINCT t1.token) AS FLOAT) / 
                (COUNT(DISTINCT t1.token) + COUNT(DISTINCT t2.token) - COUNT(DISTINCT t1.token))
            FROM (
                SELECT token FROM tfidf_scores 
                WHERE document_id = d1.document_id AND term_rank <= 20
            ) t1
            FULL OUTER JOIN (
                SELECT token FROM tfidf_scores 
                WHERE document_id = d2.document_id AND term_rank <= 20
            ) t2 ON t1.token = t2.token
        ) AS jaccard_similarity
    FROM document_vectors d1
    CROSS JOIN document_vectors d2
    WHERE d1.document_id < d2.document_id  -- Avoid duplicates
)
SELECT 
    doc1_id,
    doc2_id,
    common_important_terms,
    ROUND(jaccard_similarity, 4) AS jaccard_similarity
FROM document_similarity
WHERE jaccard_similarity > 0.1  -- Filter for similar documents
ORDER BY jaccard_similarity DESC;

-- TF-IDF-based Feature Engineering for ML
WITH ml_features AS (
    SELECT 
        document_id,
        -- Aggregate TF-IDF statistics per document
        COUNT(DISTINCT token) AS unique_terms,
        AVG(tfidf_score) AS avg_tfidf,
        MAX(tfidf_score) AS max_tfidf,
        STDDEV(tfidf_score) AS tfidf_std,
        SUM(tfidf_score) AS total_tfidf_score,
        
        -- Top term features
        MAX(CASE WHEN term_rank = 1 THEN tfidf_score END) AS top_term_tfidf,
        MAX(CASE WHEN term_rank = 1 THEN token END) AS top_term,
        
        -- Category-specific terms (example: sentiment)
        SUM(CASE WHEN token IN ('excellent', 'amazing', 'great', 'perfect', 'love') 
                THEN tfidf_score ELSE 0 END) AS positive_sentiment_tfidf,
        SUM(CASE WHEN token IN ('terrible', 'awful', 'hate', 'worst', 'horrible') 
                THEN tfidf_score ELSE 0 END) AS negative_sentiment_tfidf,
        
        -- Technical terms indicator
        SUM(CASE WHEN LENGTH(token) > 8 THEN tfidf_score ELSE 0 END) AS complex_terms_tfidf,
        
        -- Create feature vector string for top 10 terms
        STRING_AGG(
            token || '=' || ROUND(tfidf_score, 4),
            ','
            ORDER BY tfidf_score DESC
            LIMIT 10
        ) AS top_10_features
    FROM ranked_terms
    WHERE term_rank <= 100  -- Consider top 100 terms per document
    GROUP BY document_id
)
SELECT 
    document_id,
    unique_terms,
    ROUND(avg_tfidf, 4) AS avg_tfidf,
    ROUND(max_tfidf, 4) AS max_tfidf,
    ROUND(tfidf_std, 4) AS tfidf_std,
    ROUND(total_tfidf_score, 4) AS total_tfidf,
    ROUND(top_term_tfidf, 4) AS top_term_tfidf,
    top_term,
    ROUND(positive_sentiment_tfidf, 4) AS positive_sentiment,
    ROUND(negative_sentiment_tfidf, 4) AS negative_sentiment,
    ROUND(complex_terms_tfidf, 4) AS complexity_score,
    top_10_features
FROM ml_features
ORDER BY document_id;

-- Real-time TF-IDF Computation for New Documents
CREATE OR REPLACE FUNCTION compute_document_tfidf(
    new_document_text TEXT,
    reference_corpus_name VARCHAR(100) DEFAULT 'main_corpus'
)
RETURNS TABLE (
    token VARCHAR(100),
    tf NUMERIC,
    idf NUMERIC,
    tfidf_score NUMERIC
) AS $$
DECLARE
    total_docs INTEGER;
BEGIN
    -- Get total document count from reference corpus
    SELECT COUNT(DISTINCT document_id) INTO total_docs
    FROM documents 
    WHERE corpus_name = reference_corpus_name;
    
    -- Tokenize and compute TF-IDF for new document
    RETURN QUERY
    WITH new_doc_tokens AS (
        SELECT 
            REGEXP_SPLIT_TO_TABLE(
                REGEXP_REPLACE(LOWER(TRIM(new_document_text)), '[^a-z0-9\s]', ' ', 'g'), 
                '\s+'
            ) AS token
    ),
    new_doc_tf AS (
        SELECT 
            token,
            COUNT(*) AS term_count,
            CAST(COUNT(*) AS FLOAT) / SUM(COUNT(*)) OVER () AS tf
        FROM new_doc_tokens
        WHERE LENGTH(token) >= 3
        GROUP BY token
    ),
    corpus_df AS (
        SELECT 
            token,
            COUNT(DISTINCT document_id) AS doc_freq
        FROM (
            SELECT 
                document_id,     z 
                REGEXP_SPLIT_TO_TABLE(
                    REGEXP_REPLACE(LOWER(TRIM(document_text)), '[^a-z0-9\s]', ' ', 'g'), 
                    '\s+'
                ) AS token
            FROM documents 
            WHERE corpus_name = reference_corpus_name
        ) corpus_tokens
        WHERE LENGTH(token) >= 3
        GROUP BY token
    )
    SELECT 
        ndt.token::VARCHAR(100),
        ROUND(ndt.tf, 6) AS tf,
        ROUND(LOG(total_docs::FLOAT / COALESCE(cdf.doc_freq, 1)), 6) AS idf,
        ROUND(ndt.tf * LOG(total_docs::FLOAT / COALESCE(cdf.doc_freq, 1)), 6) AS tfidf_score
    FROM new_doc_tf ndt
    LEFT JOIN corpus_df cdf ON ndt.token = cdf.token
    ORDER BY tfidf_score DESC;
END;
$$ LANGUAGE plpgsql;

-- Usage example for real-time TF-IDF
SELECT * FROM compute_document_tfidf(
    'This is a sample document about machine learning and data science algorithms.'
);
```

#### Explanation
1. **Term Frequency (TF)**: Frequency of term in document divided by total terms in document
2. **Document Frequency (DF)**: Number of documents containing the term
3. **Inverse Document Frequency (IDF)**: log(total_documents / document_frequency)
4. **TF-IDF Score**: TF × IDF combines term importance within document and across corpus
5. **Normalization**: Various TF-IDF variants handle different text characteristics

#### Use Cases
- **Search Engines**: Rank document relevance for search queries
- **Document Classification**: Create feature vectors for ML text classification
- **Content Recommendation**: Find similar documents based on TF-IDF vectors
- **Keyword Extraction**: Identify important terms in documents
- **Information Retrieval**: Build text search and discovery systems

#### Best Practices
- Remove stop words and apply proper text preprocessing
- Handle rare terms that may have inflated IDF scores
- Consider document length normalization for fair comparison
- Use appropriate TF-IDF variants based on your use case
- Validate results with domain experts and sample documents

#### Pitfalls
- **Sparse Vectors**: TF-IDF creates high-dimensional sparse feature spaces
- **Vocabulary Growth**: New terms require recomputing IDF values
- **Poor for Short Texts**: TF-IDF works better with longer documents
- **No Semantic Understanding**: Doesn't capture word relationships or context
- **Computational Complexity**: Large corpora can be memory and computation intensive

#### Debugging
```sql
-- Validate TF-IDF calculations
SELECT 
    token,
    COUNT(DISTINCT document_id) AS appears_in_docs,
    AVG(tfidf_score) AS avg_tfidf,
    MAX(tfidf_score) AS max_tfidf,
    MIN(tfidf_score) AS min_tfidf
FROM tfidf_scores
GROUP BY token
ORDER BY max_tfidf DESC
LIMIT 20;

-- Check for anomalies
SELECT document_id, COUNT(*) as term_count
FROM tfidf_scores
GROUP BY document_id
HAVING COUNT(*) < 5;  -- Documents with very few terms
```

#### Optimization
- **Indexing**: Create indexes on document_id and token columns
- **Materialized Views**: Store computed TF-IDF scores for large corpora
- **Incremental Updates**: Update TF-IDF scores efficiently when adding new documents
- **Parallel Processing**: Compute TF-IDF for different document subsets in parallel

---

