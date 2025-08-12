# Sql Ml Interview Questions - General Questions

## Question 1

**What does GROUP BY do in a SQL query?**

### Answer:

#### Theory
GROUP BY is a fundamental SQL clause that partitions rows into groups based on common values in specified columns. It enables aggregate functions to operate on subsets of data rather than the entire dataset, making it essential for data analysis, reporting, and feature engineering in ML pipelines.

#### Code Example
```sql
-- Basic GROUP BY with Single Column
SELECT 
    department,
    COUNT(*) AS employee_count,
    AVG(salary) AS average_salary,
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary,
    SUM(salary) AS total_payroll
FROM employees
GROUP BY department
ORDER BY average_salary DESC;

-- Multiple Column Grouping
SELECT 
    department,
    job_level,
    gender,
    COUNT(*) AS employee_count,
    AVG(salary) AS avg_salary,
    STDDEV(salary) AS salary_std_dev,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees
GROUP BY department, job_level, gender
HAVING COUNT(*) >= 5  -- Only groups with 5+ employees
ORDER BY department, job_level, gender;

-- Advanced GROUP BY for ML Feature Engineering
SELECT 
    customer_id,
    -- Time-based grouping features
    EXTRACT(YEAR FROM purchase_date) AS purchase_year,
    EXTRACT(MONTH FROM purchase_date) AS purchase_month,
    -- Aggregate features for ML
    COUNT(*) AS total_purchases,
    COUNT(DISTINCT product_category) AS unique_categories,
    SUM(purchase_amount) AS total_spent,
    AVG(purchase_amount) AS avg_purchase_amount,
    MAX(purchase_amount) AS max_purchase,
    MIN(purchase_amount) AS min_purchase,
    STDDEV(purchase_amount) AS purchase_volatility,
    -- Frequency features
    COUNT(*) / 
        NULLIF(DATE_PART('day', MAX(purchase_date) - MIN(purchase_date)), 0) 
        AS purchases_per_day,
    -- Behavioral features
    COUNT(CASE WHEN purchase_amount > 100 THEN 1 END) AS high_value_purchases,
    COUNT(CASE WHEN EXTRACT(DOW FROM purchase_date) IN (0, 6) THEN 1 END) AS weekend_purchases,
    -- Recency feature
    MAX(purchase_date) AS last_purchase_date
FROM customer_purchases
WHERE purchase_date >= '2023-01-01'
GROUP BY customer_id
HAVING COUNT(*) >= 3  -- Customers with at least 3 purchases
ORDER BY total_spent DESC;

-- GROUP BY with Window Functions
SELECT 
    product_category,
    product_name,
    monthly_sales,
    sales_month,
    -- Group-level aggregations
    SUM(monthly_sales) AS category_total_sales,
    AVG(monthly_sales) AS category_avg_sales,
    -- Ranking within groups
    RANK() OVER (PARTITION BY product_category ORDER BY monthly_sales DESC) AS category_rank,
    -- Group size
    COUNT(*) OVER (PARTITION BY product_category) AS products_in_category,
    -- Percentage of category sales
    ROUND(
        monthly_sales * 100.0 / SUM(monthly_sales) OVER (PARTITION BY product_category), 
        2
    ) AS pct_of_category_sales
FROM (
    SELECT 
        product_category,
        product_name,
        DATE_TRUNC('month', sale_date) AS sales_month,
        SUM(sale_amount) AS monthly_sales
    FROM product_sales
    GROUP BY product_category, product_name, DATE_TRUNC('month', sale_date)
) monthly_data
ORDER BY product_category, category_rank;

-- Complex Grouping with ROLLUP and CUBE
SELECT 
    COALESCE(region, 'ALL_REGIONS') AS region,
    COALESCE(product_line, 'ALL_PRODUCTS') AS product_line,
    COALESCE(sales_quarter, 'ALL_QUARTERS') AS sales_quarter,
    COUNT(*) AS transaction_count,
    SUM(sales_amount) AS total_sales,
    AVG(sales_amount) AS avg_sales
FROM (
    SELECT 
        region,
        product_line,
        CONCAT('Q', EXTRACT(QUARTER FROM sale_date), '_', EXTRACT(YEAR FROM sale_date)) AS sales_quarter,
        sales_amount
    FROM sales_transactions
    WHERE sale_date >= '2023-01-01'
) quarterly_sales
GROUP BY ROLLUP(region, product_line, sales_quarter)
ORDER BY region NULLS LAST, product_line NULLS LAST, sales_quarter NULLS LAST;

-- Conditional Grouping
SELECT 
    -- Dynamic grouping based on data characteristics
    CASE 
        WHEN customer_age < 25 THEN 'Young (18-24)'
        WHEN customer_age < 35 THEN 'Adult (25-34)'
        WHEN customer_age < 50 THEN 'Middle Age (35-49)'
        WHEN customer_age < 65 THEN 'Mature (50-64)'
        ELSE 'Senior (65+)'
    END AS age_group,
    
    CASE 
        WHEN annual_income < 30000 THEN 'Low Income'
        WHEN annual_income < 60000 THEN 'Medium Income'
        WHEN annual_income < 100000 THEN 'High Income'
        ELSE 'Very High Income'
    END AS income_bracket,
    
    COUNT(*) AS customer_count,
    AVG(total_purchases) AS avg_purchases,
    AVG(customer_lifetime_value) AS avg_clv,
    
    -- ML-ready features
    STDDEV(customer_lifetime_value) AS clv_variance,
    MIN(customer_lifetime_value) AS min_clv,
    MAX(customer_lifetime_value) AS max_clv,
    
    -- Percentile features
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY customer_lifetime_value) AS clv_q1,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY customer_lifetime_value) AS clv_q3
FROM customer_profiles
GROUP BY 
    CASE 
        WHEN customer_age < 25 THEN 'Young (18-24)'
        WHEN customer_age < 35 THEN 'Adult (25-34)'
        WHEN customer_age < 50 THEN 'Middle Age (35-49)'
        WHEN customer_age < 65 THEN 'Mature (50-64)'
        ELSE 'Senior (65+)'
    END,
    CASE 
        WHEN annual_income < 30000 THEN 'Low Income'
        WHEN annual_income < 60000 THEN 'Medium Income'
        WHEN annual_income < 100000 THEN 'High Income'
        ELSE 'Very High Income'
    END
ORDER BY age_group, income_bracket;
```

#### Explanation
1. **Grouping Mechanism**: Rows with identical values in GROUP BY columns form a group
2. **Aggregate Functions**: COUNT, SUM, AVG, MIN, MAX operate on each group separately
3. **SELECT Clause**: Can only include GROUP BY columns and aggregate functions
4. **HAVING Clause**: Filters groups based on aggregate conditions (unlike WHERE which filters rows)
5. **Execution Order**: WHERE → GROUP BY → HAVING → SELECT → ORDER BY

#### Use Cases
- **Data Summarization**: Create summary statistics for different categories
- **Customer Segmentation**: Group customers by behavior, demographics, or value
- **Feature Engineering**: Generate aggregate features for ML models
- **Business Intelligence**: Create reports and dashboards with grouped metrics
- **Data Quality**: Identify patterns, outliers, and data distribution issues

#### Best Practices
- Include all non-aggregate columns in GROUP BY clause
- Use meaningful aliases for aggregate columns
- Apply HAVING for group-level filtering instead of WHERE
- Consider performance implications of grouping on high-cardinality columns
- Use ORDER BY to make results deterministic and meaningful

#### Pitfalls
- **Wrong Column Selection**: Including non-grouped, non-aggregate columns causes errors
- **NULL Handling**: NULL values form their own group
- **Performance Issues**: Grouping on multiple high-cardinality columns can be slow
- **Memory Usage**: Large groups can consume significant memory
- **Lost Detail**: Aggregation loses individual row-level information

#### Debugging
```sql
-- Check group sizes
SELECT 
    department,
    COUNT(*) AS group_size
FROM employees
GROUP BY department
ORDER BY group_size DESC;

-- Identify unexpected groupings
SELECT 
    customer_id,
    COUNT(*) AS duplicate_count
FROM customer_data
GROUP BY customer_id
HAVING COUNT(*) > 1;  -- Find duplicates
```

#### Optimization
- **Indexing**: Create indexes on GROUP BY columns for better performance
- **Column Order**: Order GROUP BY columns by cardinality (low to high)
- **Filtering**: Use WHERE clause to reduce data before grouping
- **Partitioning**: Consider table partitioning for large grouped datasets

---

## Question 2

**How can you aggregate data in SQL (e.g., COUNT, AVG, SUM, MAX, MIN)?**

### Answer:

#### Theory
SQL aggregate functions perform calculations across multiple rows to return single summary values. They're fundamental for data analysis, statistical calculations, and creating ML features. These functions can operate on entire tables or groups defined by GROUP BY clauses.

#### Code Example
```sql
-- Basic Aggregate Functions
SELECT 
    -- Count functions
    COUNT(*) AS total_rows,                    -- Count all rows including NULLs
    COUNT(customer_id) AS non_null_customers,  -- Count non-NULL values
    COUNT(DISTINCT customer_id) AS unique_customers,  -- Count distinct values
    
    -- Numeric aggregations
    SUM(order_amount) AS total_revenue,
    AVG(order_amount) AS average_order_value,
    MIN(order_amount) AS smallest_order,
    MAX(order_amount) AS largest_order,
    
    -- Statistical functions
    STDDEV(order_amount) AS amount_std_dev,
    VARIANCE(order_amount) AS amount_variance,
    
    -- Advanced aggregations
    SUM(CASE WHEN order_amount > 100 THEN 1 ELSE 0 END) AS high_value_orders,
    AVG(CASE WHEN customer_type = 'Premium' THEN order_amount END) AS premium_avg
FROM orders
WHERE order_date >= '2024-01-01';

-- Percentile and Quantile Functions
SELECT 
    product_category,
    COUNT(*) AS product_count,
    
    -- Central tendency
    AVG(price) AS mean_price,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) AS median_price,
    
    -- Quantiles for distribution analysis
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) AS q1_price,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) AS q3_price,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY price) AS p90_price,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY price) AS p95_price,
    
    -- Discrete percentiles
    PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY price) AS median_price_discrete,
    
    -- Range and spread
    MAX(price) - MIN(price) AS price_range,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) - 
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) AS iqr
FROM products
GROUP BY product_category
ORDER BY mean_price DESC;

-- String and Array Aggregations
SELECT 
    customer_id,
    -- String aggregations
    STRING_AGG(product_name, ', ' ORDER BY purchase_date) AS products_purchased,
    STRING_AGG(DISTINCT product_category, '|') AS categories_bought,
    
    -- Array aggregations (PostgreSQL)
    ARRAY_AGG(product_id ORDER BY purchase_date) AS product_sequence,
    ARRAY_AGG(DISTINCT store_location) AS stores_visited,
    
    -- JSON aggregations
    JSON_AGG(
        JSON_BUILD_OBJECT(
            'product', product_name,
            'amount', purchase_amount,
            'date', purchase_date
        ) ORDER BY purchase_date
    ) AS purchase_history
FROM customer_purchases
GROUP BY customer_id;

-- Window Aggregations (Running Totals and Moving Averages)
SELECT 
    transaction_date,
    customer_id,
    transaction_amount,
    
    -- Running aggregations
    SUM(transaction_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date
    ) AS running_total,
    
    COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date
    ) AS transaction_sequence,
    
    -- Moving averages
    AVG(transaction_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3_transactions,
    
    -- Global comparisons
    AVG(transaction_amount) OVER () AS global_avg,
    transaction_amount - AVG(transaction_amount) OVER () AS deviation_from_global_avg,
    
    -- Ranking aggregations
    RANK() OVER (PARTITION BY customer_id ORDER BY transaction_amount DESC) AS amount_rank,
    NTILE(4) OVER (PARTITION BY customer_id ORDER BY transaction_amount) AS amount_quartile
FROM transactions
ORDER BY customer_id, transaction_date;

-- Complex Aggregations for ML Feature Engineering
WITH customer_features AS (
    SELECT 
        customer_id,
        -- Volume features
        COUNT(*) AS total_transactions,
        COUNT(DISTINCT DATE_TRUNC('month', transaction_date)) AS active_months,
        COUNT(DISTINCT merchant_id) AS unique_merchants,
        
        -- Amount features
        SUM(amount) AS total_spent,
        AVG(amount) AS avg_transaction_amount,
        STDDEV(amount) AS amount_volatility,
        MIN(amount) AS min_transaction,
        MAX(amount) AS max_transaction,
        
        -- Percentile features
        PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY amount) AS amount_10th_percentile,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY amount) AS amount_90th_percentile,
        
        -- Time-based features
        MAX(transaction_date) AS last_transaction_date,
        MIN(transaction_date) AS first_transaction_date,
        MAX(transaction_date) - MIN(transaction_date) AS customer_tenure_days,
        
        -- Frequency features
        COUNT(*) / NULLIF(EXTRACT(DAYS FROM MAX(transaction_date) - MIN(transaction_date)), 0) AS avg_transactions_per_day,
        
        -- Behavioral patterns
        SUM(CASE WHEN EXTRACT(DOW FROM transaction_date) IN (0, 6) THEN amount ELSE 0 END) AS weekend_spending,
        SUM(CASE WHEN EXTRACT(HOUR FROM transaction_date) BETWEEN 9 AND 17 THEN amount ELSE 0 END) AS business_hours_spending,
        
        -- Category analysis
        STRING_AGG(DISTINCT merchant_category, '|') AS merchant_categories,
        COUNT(DISTINCT merchant_category) AS category_diversity,
        
        -- Risk indicators
        COUNT(CASE WHEN amount > 1000 THEN 1 END) AS high_value_transactions,
        MAX(amount) / AVG(amount) AS max_to_avg_ratio
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '365 days'
    GROUP BY customer_id
),
normalized_features AS (
    SELECT 
        customer_id,
        total_transactions,
        total_spent,
        avg_transaction_amount,
        amount_volatility,
        customer_tenure_days,
        avg_transactions_per_day,
        category_diversity,
        high_value_transactions,
        
        -- Normalized features (percentile ranks)
        PERCENT_RANK() OVER (ORDER BY total_spent) AS spending_percentile,
        PERCENT_RANK() OVER (ORDER BY total_transactions) AS frequency_percentile,
        PERCENT_RANK() OVER (ORDER BY amount_volatility) AS volatility_percentile,
        
        -- Customer segments based on aggregations
        CASE 
            WHEN total_spent > 10000 AND total_transactions > 50 THEN 'High_Value_Frequent'
            WHEN total_spent > 10000 THEN 'High_Value_Occasional'
            WHEN total_transactions > 50 THEN 'Low_Value_Frequent'
            ELSE 'Low_Value_Occasional'
        END AS customer_segment
    FROM customer_features
)
SELECT * FROM normalized_features
ORDER BY total_spent DESC;

-- Conditional Aggregations and Advanced Patterns
SELECT 
    store_id,
    DATE_TRUNC('month', sale_date) AS sale_month,
    
    -- Conditional aggregations
    SUM(CASE WHEN product_category = 'Electronics' THEN sale_amount ELSE 0 END) AS electronics_sales,
    SUM(CASE WHEN product_category = 'Clothing' THEN sale_amount ELSE 0 END) AS clothing_sales,
    SUM(CASE WHEN product_category = 'Books' THEN sale_amount ELSE 0 END) AS books_sales,
    
    -- Weighted averages
    SUM(sale_amount * quantity) / SUM(quantity) AS weighted_avg_price,
    
    -- Multiple aggregation levels
    SUM(sale_amount) AS monthly_total,
    AVG(SUM(sale_amount)) OVER (
        PARTITION BY store_id 
        ORDER BY DATE_TRUNC('month', sale_date) 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS quarterly_avg_monthly_sales,
    
    -- Aggregation ratios
    SUM(CASE WHEN discount_applied THEN sale_amount ELSE 0 END) / 
    NULLIF(SUM(sale_amount), 0) AS discount_sales_ratio,
    
    -- Growth calculations
    LAG(SUM(sale_amount)) OVER (
        PARTITION BY store_id 
        ORDER BY DATE_TRUNC('month', sale_date)
    ) AS prev_month_sales,
    
    (SUM(sale_amount) - LAG(SUM(sale_amount)) OVER (
        PARTITION BY store_id 
        ORDER BY DATE_TRUNC('month', sale_date)
    )) / NULLIF(LAG(SUM(sale_amount)) OVER (
        PARTITION BY store_id 
        ORDER BY DATE_TRUNC('month', sale_date)
    ), 0) * 100 AS month_over_month_growth_pct
FROM sales_data
GROUP BY store_id, DATE_TRUNC('month', sale_date)
ORDER BY store_id, sale_month;
```

#### Explanation
1. **COUNT()**: Counts rows, with COUNT(*) including NULLs and COUNT(column) excluding them
2. **SUM()**: Adds numeric values, ignoring NULLs
3. **AVG()**: Calculates arithmetic mean, excluding NULLs from calculation
4. **MIN()/MAX()**: Find minimum/maximum values, work with numbers, dates, and strings
5. **STDDEV()/VARIANCE()**: Calculate statistical measures of data spread

#### Use Cases
- **Business Metrics**: Calculate KPIs like total revenue, average order value
- **Data Profiling**: Understand data distributions and characteristics
- **Feature Engineering**: Create aggregate features for ML models
- **Reporting**: Generate summary reports and dashboards
- **Quality Assurance**: Detect data anomalies and outliers

#### Best Practices
- Always consider NULL handling in aggregations
- Use appropriate data types to avoid overflow in SUM operations
- Combine multiple aggregations in single queries for efficiency
- Use window functions for running calculations and comparisons
- Apply filters before aggregation when possible to improve performance

#### Pitfalls
- **NULL Behavior**: Different functions handle NULLs differently
- **Data Type Overflow**: Large SUMs might exceed column data type limits
- **Division by Zero**: Use NULLIF() to prevent division by zero errors
- **Performance**: Aggregating large datasets without proper indexing can be slow
- **Precision Loss**: Floating-point arithmetic may introduce rounding errors

#### Debugging
```sql
-- Check for NULL impacts on aggregations
SELECT 
    COUNT(*) AS total_rows,
    COUNT(salary) AS non_null_salaries,
    SUM(salary) AS total_salary,
    AVG(salary) AS avg_salary_excluding_nulls,
    SUM(salary) / COUNT(*) AS avg_including_nulls_as_zero
FROM employees;

-- Verify aggregate calculations
SELECT 
    department,
    COUNT(*) AS emp_count,
    SUM(salary) AS total_payroll,
    SUM(salary) / COUNT(*) AS manual_average,
    AVG(salary) AS sql_average
FROM employees
GROUP BY department;
```

#### Optimization
- **Indexing**: Create indexes on columns used in GROUP BY and WHERE clauses
- **Filtering**: Apply WHERE conditions before GROUP BY to reduce data volume
- **Materialized Views**: Pre-compute frequently used aggregations
- **Partitioning**: Use table partitioning for time-based aggregations

---

## Question 3

**How do you handle missing values in a SQL dataset?**

### Answer:

#### Theory
Missing values (NULLs) are common in real-world datasets and can significantly impact ML model performance. SQL provides multiple strategies for handling missing data: detection, filtering, imputation, and flagging. The choice depends on the missingness pattern, data type, and business requirements.

#### Code Example
```sql
-- Strategy 1: Detection and Analysis of Missing Values
WITH missing_analysis AS (
    SELECT 
        'customers' AS table_name,
        'age' AS column_name,
        COUNT(*) AS total_rows,
        COUNT(age) AS non_null_count,
        COUNT(*) - COUNT(age) AS null_count,
        ROUND((COUNT(*) - COUNT(age)) * 100.0 / COUNT(*), 2) AS null_percentage
    FROM customers
    
    UNION ALL
    
    SELECT 
        'customers',
        'income',
        COUNT(*),
        COUNT(income),
        COUNT(*) - COUNT(income),
        ROUND((COUNT(*) - COUNT(income)) * 100.0 / COUNT(*), 2)
    FROM customers
    
    UNION ALL
    
    SELECT 
        'customers',
        'phone_number',
        COUNT(*),
        COUNT(phone_number),
        COUNT(*) - COUNT(phone_number),
        ROUND((COUNT(*) - COUNT(phone_number)) * 100.0 / COUNT(*), 2)
    FROM customers
)
SELECT 
    table_name,
    column_name,
    total_rows,
    non_null_count,
    null_count,
    null_percentage,
    CASE 
        WHEN null_percentage = 0 THEN 'Complete'
        WHEN null_percentage < 5 THEN 'Minimal Missing'
        WHEN null_percentage < 20 THEN 'Moderate Missing'
        WHEN null_percentage < 50 THEN 'Substantial Missing'
        ELSE 'Mostly Missing'
    END AS missing_category
FROM missing_analysis
ORDER BY null_percentage DESC;

-- Strategy 2: Filtering Out Missing Values
SELECT 
    customer_id,
    age,
    income,
    credit_score,
    phone_number
FROM customers
WHERE age IS NOT NULL 
  AND income IS NOT NULL 
  AND credit_score IS NOT NULL
  AND phone_number IS NOT NULL;  -- Complete case analysis

-- Filter with business logic
SELECT 
    customer_id,
    age,
    income,
    credit_score
FROM customers
WHERE (age IS NOT NULL OR age BETWEEN 18 AND 100)  -- Valid age range
  AND (income IS NOT NULL AND income > 0)          -- Positive income
  AND credit_score BETWEEN 300 AND 850;            -- Valid credit score range

-- Strategy 3: Imputation Techniques
WITH imputation_values AS (
    SELECT 
        -- Central tendency measures
        AVG(age) AS mean_age,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY age) AS median_age,
        AVG(income) AS mean_income,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY income) AS median_income,
        AVG(credit_score) AS mean_credit_score,
        
        -- Mode calculation for categorical data
        (SELECT phone_prefix 
         FROM (SELECT SUBSTRING(phone_number, 1, 3) AS phone_prefix, COUNT(*) AS cnt
               FROM customers 
               WHERE phone_number IS NOT NULL
               GROUP BY SUBSTRING(phone_number, 1, 3)
               ORDER BY cnt DESC 
               LIMIT 1)
        ) AS mode_phone_prefix
    FROM customers
    WHERE age IS NOT NULL OR income IS NOT NULL OR credit_score IS NOT NULL
),
imputed_data AS (
    SELECT 
        customer_id,
        customer_segment,
        registration_date,
        
        -- Numeric imputation strategies
        COALESCE(age, (SELECT mean_age FROM imputation_values)) AS age_mean_imputed,
        COALESCE(age, (SELECT median_age FROM imputation_values)) AS age_median_imputed,
        COALESCE(income, (SELECT mean_income FROM imputation_values)) AS income_imputed,
        COALESCE(credit_score, (SELECT mean_credit_score FROM imputation_values)) AS credit_score_imputed,
        
        -- Forward fill (using previous valid value)
        COALESCE(
            phone_number,
            LAG(phone_number) OVER (ORDER BY customer_id),
            'Unknown'
        ) AS phone_forward_filled,
        
        -- Categorical imputation
        COALESCE(customer_segment, 'Standard') AS segment_imputed,
        
        -- Create missing value indicators
        CASE WHEN age IS NULL THEN 1 ELSE 0 END AS age_was_missing,
        CASE WHEN income IS NULL THEN 1 ELSE 0 END AS income_was_missing,
        CASE WHEN credit_score IS NULL THEN 1 ELSE 0 END AS credit_score_was_missing,
        CASE WHEN phone_number IS NULL THEN 1 ELSE 0 END AS phone_was_missing
    FROM customers
)
SELECT * FROM imputed_data;

-- Strategy 4: Group-based Imputation
WITH segment_stats AS (
    SELECT 
        customer_segment,
        AVG(age) AS segment_avg_age,
        AVG(income) AS segment_avg_income,
        AVG(credit_score) AS segment_avg_credit_score,
        COUNT(*) AS segment_size
    FROM customers
    WHERE age IS NOT NULL AND income IS NOT NULL AND credit_score IS NOT NULL
    GROUP BY customer_segment
),
global_stats AS (
    SELECT 
        AVG(age) AS global_avg_age,
        AVG(income) AS global_avg_income,
        AVG(credit_score) AS global_avg_credit_score
    FROM customers
    WHERE age IS NOT NULL AND income IS NOT NULL AND credit_score IS NOT NULL
),
sophisticated_imputation AS (
    SELECT 
        c.customer_id,
        c.customer_segment,
        c.age,
        c.income,
        c.credit_score,
        
        -- Group-based imputation with global fallback
        COALESCE(
            c.age,
            ss.segment_avg_age,
            (SELECT global_avg_age FROM global_stats)
        ) AS age_group_imputed,
        
        COALESCE(
            c.income,
            ss.segment_avg_income,
            (SELECT global_avg_income FROM global_stats)
        ) AS income_group_imputed,
        
        COALESCE(
            c.credit_score,
            ss.segment_avg_credit_score,
            (SELECT global_avg_credit_score FROM global_stats)
        ) AS credit_score_group_imputed,
        
        -- Imputation method tracking
        CASE 
            WHEN c.age IS NOT NULL THEN 'original'
            WHEN ss.segment_avg_age IS NOT NULL THEN 'segment_mean'
            ELSE 'global_mean'
        END AS age_imputation_method
    FROM customers c
    LEFT JOIN segment_stats ss ON c.customer_segment = ss.customer_segment
)
SELECT * FROM sophisticated_imputation;

-- Strategy 5: Time-based Imputation
SELECT 
    customer_id,
    transaction_date,
    transaction_amount,
    merchant_category,
    
    -- Last observation carried forward (LOCF)
    COALESCE(
        transaction_amount,
        LAG(transaction_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date
        )
    ) AS amount_locf,
    
    -- Linear interpolation approximation
    CASE 
        WHEN transaction_amount IS NULL THEN
            (LAG(transaction_amount) OVER (PARTITION BY customer_id ORDER BY transaction_date) +
             LEAD(transaction_amount) OVER (PARTITION BY customer_id ORDER BY transaction_date)) / 2
        ELSE transaction_amount
    END AS amount_interpolated,
    
    -- Rolling average imputation
    COALESCE(
        transaction_amount,
        AVG(transaction_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
        )
    ) AS amount_rolling_avg,
    
    -- Seasonal imputation (same day of week average)
    COALESCE(
        transaction_amount,
        AVG(transaction_amount) OVER (
            PARTITION BY customer_id, EXTRACT(DOW FROM transaction_date)
        )
    ) AS amount_seasonal_imputed
FROM customer_transactions
ORDER BY customer_id, transaction_date;

-- Strategy 6: Advanced Missing Value Patterns
WITH missing_patterns AS (
    SELECT 
        customer_id,
        -- Missing value pattern encoding
        CASE WHEN age IS NULL THEN '1' ELSE '0' END ||
        CASE WHEN income IS NULL THEN '1' ELSE '0' END ||
        CASE WHEN credit_score IS NULL THEN '1' ELSE '0' END ||
        CASE WHEN phone_number IS NULL THEN '1' ELSE '0' END AS missing_pattern,
        
        -- Count of missing fields
        (CASE WHEN age IS NULL THEN 1 ELSE 0 END +
         CASE WHEN income IS NULL THEN 1 ELSE 0 END +
         CASE WHEN credit_score IS NULL THEN 1 ELSE 0 END +
         CASE WHEN phone_number IS NULL THEN 1 ELSE 0 END) AS missing_count,
        
        age, income, credit_score, phone_number
    FROM customers
),
pattern_analysis AS (
    SELECT 
        missing_pattern,
        missing_count,
        COUNT(*) AS pattern_frequency,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pattern_percentage
    FROM missing_patterns
    GROUP BY missing_pattern, missing_count
    ORDER BY pattern_frequency DESC
),
missingness_flags AS (
    SELECT 
        mp.*,
        pa.pattern_frequency,
        
        -- Missingness type classification
        CASE 
            WHEN missing_count = 0 THEN 'Complete'
            WHEN missing_count = 1 THEN 'Single_Missing'
            WHEN missing_count = 2 THEN 'Double_Missing'
            WHEN missing_count >= 3 THEN 'Multiple_Missing'
        END AS missingness_type,
        
        -- Create dummy variables for missing indicators
        CASE WHEN age IS NULL THEN 1 ELSE 0 END AS missing_age_flag,
        CASE WHEN income IS NULL THEN 1 ELSE 0 END AS missing_income_flag,
        CASE WHEN credit_score IS NULL THEN 1 ELSE 0 END AS missing_credit_flag,
        CASE WHEN phone_number IS NULL THEN 1 ELSE 0 END AS missing_phone_flag
    FROM missing_patterns mp
    JOIN pattern_analysis pa ON mp.missing_pattern = pa.missing_pattern
)
SELECT * FROM missingness_flags;

-- Strategy 7: Business Rule-based Imputation
SELECT 
    customer_id,
    age,
    income,
    credit_score,
    customer_segment,
    account_type,
    
    -- Business logic imputation
    CASE 
        WHEN age IS NULL AND customer_segment = 'Student' THEN 22  -- Average student age
        WHEN age IS NULL AND customer_segment = 'Senior' THEN 68   -- Average senior age
        WHEN age IS NULL AND account_type = 'Business' THEN 45     -- Average business owner age
        WHEN age IS NULL THEN 40  -- General default
        ELSE age
    END AS age_business_imputed,
    
    CASE 
        WHEN income IS NULL AND customer_segment = 'Premium' THEN 150000  -- Premium tier minimum
        WHEN income IS NULL AND customer_segment = 'Standard' THEN 60000  -- Standard tier average
        WHEN income IS NULL AND customer_segment = 'Basic' THEN 35000     -- Basic tier average
        WHEN income IS NULL THEN 50000  -- General default
        ELSE income
    END AS income_business_imputed,
    
    CASE 
        WHEN credit_score IS NULL AND customer_segment = 'Premium' THEN 750  -- Premium tier average
        WHEN credit_score IS NULL AND customer_segment = 'Standard' THEN 680 -- Standard tier average
        WHEN credit_score IS NULL AND customer_segment = 'Basic' THEN 600    -- Basic tier average
        WHEN credit_score IS NULL THEN 650  -- General default
        ELSE credit_score
    END AS credit_score_business_imputed
FROM customers;

-- Strategy 8: Quality Metrics for Imputation
WITH imputation_quality AS (
    SELECT 
        'age' AS column_name,
        COUNT(CASE WHEN age IS NULL THEN 1 END) AS original_nulls,
        COUNT(CASE WHEN age_imputed IS NULL THEN 1 END) AS remaining_nulls,
        AVG(age) AS original_mean,
        AVG(age_imputed) AS imputed_mean,
        STDDEV(age) AS original_stddev,
        STDDEV(age_imputed) AS imputed_stddev
    FROM (
        SELECT 
            age,
            COALESCE(age, AVG(age) OVER ()) AS age_imputed
        FROM customers
    ) imputed_ages
    
    UNION ALL
    
    SELECT 
        'income' AS column_name,
        COUNT(CASE WHEN income IS NULL THEN 1 END),
        COUNT(CASE WHEN income_imputed IS NULL THEN 1 END),
        AVG(income),
        AVG(income_imputed),
        STDDEV(income),
        STDDEV(income_imputed)
    FROM (
        SELECT 
            income,
            COALESCE(income, AVG(income) OVER ()) AS income_imputed
        FROM customers
    ) imputed_incomes
)
SELECT 
    column_name,
    original_nulls,
    remaining_nulls,
    ROUND(original_mean, 2) AS original_mean,
    ROUND(imputed_mean, 2) AS imputed_mean,
    ROUND(original_stddev, 2) AS original_stddev,
    ROUND(imputed_stddev, 2) AS imputed_stddev,
    ROUND(ABS(original_mean - imputed_mean), 2) AS mean_difference,
    ROUND(ABS(original_stddev - imputed_stddev), 2) AS stddev_difference
FROM imputation_quality;
```

#### Explanation
1. **NULL Detection**: Use COUNT() differences to identify missing value patterns
2. **COALESCE Function**: Returns first non-NULL value from a list of expressions
3. **CASE Statements**: Implement conditional imputation logic
4. **Window Functions**: Enable temporal and group-based imputation strategies
5. **Indicator Variables**: Create binary flags to track imputation for ML models

#### Use Cases
- **Data Preprocessing**: Clean datasets before ML model training
- **Real-time Systems**: Handle missing values in streaming data
- **Data Integration**: Merge datasets with different completeness levels
- **Reporting**: Ensure complete datasets for business intelligence
- **Data Quality**: Monitor and improve data collection processes

#### Best Practices
- Analyze missingness patterns before choosing imputation strategy
- Create indicator variables to flag imputed values for ML models
- Document imputation decisions and validate with domain experts
- Test multiple imputation strategies and compare model performance
- Consider the mechanism causing missingness (MCAR, MAR, NMAR)

#### Pitfalls
- **Bias Introduction**: Poor imputation can introduce systematic bias
- **Variance Reduction**: Simple imputation methods can artificially reduce variance
- **Correlation Distortion**: Imputation may weaken relationships between variables
- **Temporal Assumptions**: Forward-fill may not be appropriate for all time series
- **Domain Validity**: Statistical imputations may violate business rules

#### Debugging
```sql
-- Validate imputation results
SELECT 
    'Before Imputation' AS stage,
    COUNT(*) AS total_records,
    COUNT(age) AS complete_age,
    COUNT(income) AS complete_income,
    AVG(age) AS avg_age,
    AVG(income) AS avg_income
FROM customers
UNION ALL
SELECT 
    'After Imputation',
    COUNT(*),
    COUNT(age_imputed),
    COUNT(income_imputed),
    AVG(age_imputed),
    AVG(income_imputed)
FROM customers_imputed;
```

#### Optimization
- **Batch Processing**: Apply imputation to large datasets in manageable chunks
- **Indexing**: Create indexes on grouping columns for group-based imputation
- **Materialized Views**: Store imputed datasets for repeated analysis
- **Parallel Processing**: Process independent imputation groups in parallel

---

## Question 4

**How can you extract time-based features from a SQL datetime field for use in a Machine Learning model?**

### Answer:

#### Theory
Time-based feature extraction transforms datetime columns into meaningful numerical and categorical variables that capture temporal patterns, seasonality, and trends. These features are crucial for time series forecasting, customer behavior analysis, and detecting cyclical patterns in ML models.

#### Code Example
```sql
-- Basic Date/Time Component Extraction
SELECT 
    customer_id,
    transaction_datetime,
    transaction_amount,
    
    -- Date components
    EXTRACT(YEAR FROM transaction_datetime) AS year,
    EXTRACT(MONTH FROM transaction_datetime) AS month,
    EXTRACT(DAY FROM transaction_datetime) AS day,
    EXTRACT(QUARTER FROM transaction_datetime) AS quarter,
    EXTRACT(WEEK FROM transaction_datetime) AS week_of_year,
    EXTRACT(DOY FROM transaction_datetime) AS day_of_year,
    EXTRACT(DOW FROM transaction_datetime) AS day_of_week,  -- 0=Sunday, 6=Saturday
    
    -- Time components
    EXTRACT(HOUR FROM transaction_datetime) AS hour,
    EXTRACT(MINUTE FROM transaction_datetime) AS minute,
    EXTRACT(EPOCH FROM transaction_datetime) AS unix_timestamp,
    
    -- Formatted date strings
    TO_CHAR(transaction_datetime, 'YYYY-MM') AS year_month,
    TO_CHAR(transaction_datetime, 'YYYY-Q') AS year_quarter,
    TO_CHAR(transaction_datetime, 'Day') AS day_name,
    TO_CHAR(transaction_datetime, 'Month') AS month_name
FROM transactions;

-- Advanced Temporal Features for ML
WITH temporal_features AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_datetime,
        transaction_amount,
        
        -- Basic temporal components
        EXTRACT(YEAR FROM transaction_datetime) AS year,
        EXTRACT(MONTH FROM transaction_datetime) AS month,
        EXTRACT(DOW FROM transaction_datetime) AS dow,
        EXTRACT(HOUR FROM transaction_datetime) AS hour,
        
        -- Cyclical encoding (prevents arbitrary ordering)
        SIN(2 * PI() * EXTRACT(MONTH FROM transaction_datetime) / 12) AS month_sin,
        COS(2 * PI() * EXTRACT(MONTH FROM transaction_datetime) / 12) AS month_cos,
        SIN(2 * PI() * EXTRACT(DOW FROM transaction_datetime) / 7) AS dow_sin,
        COS(2 * PI() * EXTRACT(DOW FROM transaction_datetime) / 7) AS dow_cos,
        SIN(2 * PI() * EXTRACT(HOUR FROM transaction_datetime) / 24) AS hour_sin,
        COS(2 * PI() * EXTRACT(HOUR FROM transaction_datetime) / 24) AS hour_cos,
        
        -- Binary temporal indicators
        CASE WHEN EXTRACT(DOW FROM transaction_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
        CASE WHEN EXTRACT(HOUR FROM transaction_datetime) BETWEEN 9 AND 17 THEN 1 ELSE 0 END AS is_business_hours,
        CASE WHEN EXTRACT(HOUR FROM transaction_datetime) BETWEEN 18 AND 23 THEN 1 ELSE 0 END AS is_evening,
        CASE WHEN EXTRACT(HOUR FROM transaction_datetime) BETWEEN 0 AND 6 THEN 1 ELSE 0 END AS is_night,
        
        -- Seasonal indicators
        CASE 
            WHEN EXTRACT(MONTH FROM transaction_datetime) IN (12, 1, 2) THEN 'Winter'
            WHEN EXTRACT(MONTH FROM transaction_datetime) IN (3, 4, 5) THEN 'Spring'
            WHEN EXTRACT(MONTH FROM transaction_datetime) IN (6, 7, 8) THEN 'Summer'
            ELSE 'Fall'
        END AS season,
        
        -- Business periods
        CASE 
            WHEN EXTRACT(MONTH FROM transaction_datetime) IN (11, 12) THEN 1 
            ELSE 0 
        END AS is_holiday_season,
        
        CASE 
            WHEN EXTRACT(MONTH FROM transaction_datetime) = 1 THEN 1 
            ELSE 0 
        END AS is_new_year_month,
        
        CASE 
            WHEN EXTRACT(DAY FROM transaction_datetime) <= 7 
             AND EXTRACT(MONTH FROM transaction_datetime) = 1 THEN 1 
            ELSE 0 
        END AS is_first_week_year,
        
        -- Time since epoch features
        EXTRACT(EPOCH FROM transaction_datetime) AS timestamp_epoch,
        EXTRACT(EPOCH FROM transaction_datetime) / 86400 AS days_since_epoch,
        
        -- Week-related features
        CASE 
            WHEN EXTRACT(DAY FROM transaction_datetime) <= 7 THEN 1
            WHEN EXTRACT(DAY FROM transaction_datetime) <= 14 THEN 2
            WHEN EXTRACT(DAY FROM transaction_datetime) <= 21 THEN 3
            ELSE 4
        END AS week_of_month,
        
        -- Pay period indicators (assuming bi-weekly pay)
        CASE 
            WHEN EXTRACT(DAY FROM transaction_datetime) IN (1, 15) THEN 1 
            ELSE 0 
        END AS is_payday_approx
    FROM transactions
),
relative_temporal_features AS (
    SELECT 
        tf.*,
        
        -- Time since previous transaction (per customer)
        EXTRACT(EPOCH FROM 
            transaction_datetime - LAG(transaction_datetime) OVER (
                PARTITION BY customer_id 
                ORDER BY transaction_datetime
            )
        ) / 3600 AS hours_since_last_transaction,
        
        -- Transaction sequence number
        ROW_NUMBER() OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_datetime
        ) AS transaction_sequence,
        
        -- Days since first transaction
        EXTRACT(EPOCH FROM 
            transaction_datetime - FIRST_VALUE(transaction_datetime) OVER (
                PARTITION BY customer_id 
                ORDER BY transaction_datetime 
                ROWS UNBOUNDED PRECEDING
            )
        ) / 86400 AS days_since_first_transaction,
        
        -- Time until next transaction
        EXTRACT(EPOCH FROM 
            LEAD(transaction_datetime) OVER (
                PARTITION BY customer_id 
                ORDER BY transaction_datetime
            ) - transaction_datetime
        ) / 3600 AS hours_until_next_transaction,
        
        -- Transaction density (transactions per day in 30-day window)
        COUNT(*) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_datetime 
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
        ) / 30.0 AS avg_transactions_per_day_30d
    FROM temporal_features tf
),
advanced_patterns AS (
    SELECT 
        rtf.*,
        
        -- Recency, Frequency, Monetary temporal features
        MAX(transaction_datetime) OVER (PARTITION BY customer_id) AS last_transaction_date,
        EXTRACT(EPOCH FROM CURRENT_TIMESTAMP - MAX(transaction_datetime) OVER (PARTITION BY customer_id)) / 86400 AS days_since_last_transaction,
        
        -- Activity patterns
        COUNT(CASE WHEN is_weekend = 1 THEN 1 END) OVER (
            PARTITION BY customer_id
        ) AS total_weekend_transactions,
        
        COUNT(CASE WHEN is_business_hours = 1 THEN 1 END) OVER (
            PARTITION BY customer_id
        ) AS total_business_hour_transactions,
        
        -- Temporal clustering indicators
        CASE 
            WHEN hours_since_last_transaction < 1 THEN 'Burst'
            WHEN hours_since_last_transaction < 24 THEN 'Daily'
            WHEN hours_since_last_transaction < 168 THEN 'Weekly'
            ELSE 'Irregular'
        END AS transaction_pattern,
        
        -- Lag features with temporal context
        LAG(transaction_amount, 1) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_datetime
        ) AS prev_transaction_amount,
        
        -- Moving averages with time decay
        AVG(transaction_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_datetime 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS ma_7_transactions,
        
        -- Seasonal comparison features
        LAG(transaction_amount, 365) OVER (
            PARTITION BY customer_id, EXTRACT(DOY FROM transaction_datetime)
            ORDER BY year
        ) AS same_day_last_year_amount
    FROM relative_temporal_features rtf
)
SELECT 
    transaction_id,
    customer_id,
    transaction_datetime,
    transaction_amount,
    
    -- Core temporal features
    year, month, dow, hour,
    month_sin, month_cos, dow_sin, dow_cos, hour_sin, hour_cos,
    is_weekend, is_business_hours, is_evening, is_night,
    season, is_holiday_season, is_new_year_month,
    week_of_month, is_payday_approx,
    
    -- Sequence and relative features
    transaction_sequence,
    days_since_first_transaction,
    hours_since_last_transaction,
    days_since_last_transaction,
    avg_transactions_per_day_30d,
    transaction_pattern,
    
    -- Historical comparison features
    prev_transaction_amount,
    ma_7_transactions,
    same_day_last_year_amount,
    
    -- Behavioral ratios
    total_weekend_transactions::FLOAT / NULLIF(transaction_sequence, 0) AS weekend_transaction_ratio,
    total_business_hour_transactions::FLOAT / NULLIF(transaction_sequence, 0) AS business_hour_ratio
FROM advanced_patterns
ORDER BY customer_id, transaction_datetime;

-- Holiday and Special Event Detection
WITH holiday_calendar AS (
    SELECT 
        DATE '2024-01-01' AS holiday_date, 'New Year Day' AS holiday_name
    UNION ALL SELECT DATE '2024-02-14', 'Valentine Day'
    UNION ALL SELECT DATE '2024-07-04', 'Independence Day'
    UNION ALL SELECT DATE '2024-11-28', 'Thanksgiving'
    UNION ALL SELECT DATE '2024-12-25', 'Christmas'
    -- Add more holidays as needed
),
special_events AS (
    SELECT 
        t.transaction_id,
        t.transaction_datetime,
        t.transaction_amount,
        
        -- Distance to nearest holiday
        MIN(ABS(EXTRACT(EPOCH FROM DATE(t.transaction_datetime) - h.holiday_date) / 86400)) AS days_to_nearest_holiday,
        
        -- Specific holiday indicators
        CASE WHEN DATE(t.transaction_datetime) = ANY(SELECT holiday_date FROM holiday_calendar) 
             THEN 1 ELSE 0 END AS is_holiday,
        
        CASE WHEN DATE(t.transaction_datetime) BETWEEN DATE '2024-11-25' AND DATE '2024-11-29' 
             THEN 1 ELSE 0 END AS is_thanksgiving_week,
        
        CASE WHEN DATE(t.transaction_datetime) BETWEEN DATE '2024-12-15' AND DATE '2024-12-31' 
             THEN 1 ELSE 0 END AS is_christmas_season,
        
        -- Black Friday / Cyber Monday
        CASE WHEN EXTRACT(DOW FROM t.transaction_datetime) = 5  -- Friday
             AND EXTRACT(MONTH FROM t.transaction_datetime) = 11
             AND EXTRACT(DAY FROM t.transaction_datetime) BETWEEN 23 AND 29
             THEN 1 ELSE 0 END AS is_black_friday_week,
        
        -- End/Beginning of month
        CASE WHEN EXTRACT(DAY FROM t.transaction_datetime) <= 3 
             THEN 1 ELSE 0 END AS is_month_start,
        CASE WHEN EXTRACT(DAY FROM t.transaction_datetime) >= 28 
             THEN 1 ELSE 0 END AS is_month_end
    FROM transactions t
    CROSS JOIN holiday_calendar h
    GROUP BY t.transaction_id, t.transaction_datetime, t.transaction_amount
),
time_aggregated_features AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', transaction_datetime) AS month_year,
        
        -- Monthly aggregations with temporal context
        COUNT(*) AS monthly_transaction_count,
        SUM(transaction_amount) AS monthly_spend,
        AVG(transaction_amount) AS monthly_avg_amount,
        STDDEV(transaction_amount) AS monthly_amount_volatility,
        
        -- Time distribution within month
        COUNT(CASE WHEN EXTRACT(DOW FROM transaction_datetime) IN (0, 6) THEN 1 END) AS monthly_weekend_transactions,
        COUNT(CASE WHEN EXTRACT(HOUR FROM transaction_datetime) BETWEEN 9 AND 17 THEN 1 END) AS monthly_business_hour_transactions,
        
        -- Temporal spread
        MAX(transaction_datetime) - MIN(transaction_datetime) AS monthly_activity_span,
        COUNT(DISTINCT DATE(transaction_datetime)) AS monthly_active_days,
        
        -- Growth features
        LAG(COUNT(*)) OVER (PARTITION BY customer_id ORDER BY DATE_TRUNC('month', transaction_datetime)) AS prev_month_transaction_count,
        LAG(SUM(transaction_amount)) OVER (PARTITION BY customer_id ORDER BY DATE_TRUNC('month', transaction_datetime)) AS prev_month_spend
    FROM transactions
    GROUP BY customer_id, DATE_TRUNC('month', transaction_datetime)
)
SELECT 
    customer_id,
    month_year,
    monthly_transaction_count,
    monthly_spend,
    monthly_avg_amount,
    monthly_weekend_transactions,
    monthly_business_hour_transactions,
    monthly_active_days,
    
    -- Month-over-month growth
    CASE WHEN prev_month_transaction_count > 0 
         THEN (monthly_transaction_count - prev_month_transaction_count)::FLOAT / prev_month_transaction_count 
         ELSE NULL END AS transaction_count_mom_growth,
    
    CASE WHEN prev_month_spend > 0 
         THEN (monthly_spend - prev_month_spend) / prev_month_spend 
         ELSE NULL END AS spend_mom_growth,
    
    -- Activity intensity
    monthly_transaction_count::FLOAT / NULLIF(monthly_active_days, 0) AS avg_transactions_per_active_day,
    monthly_weekend_transactions::FLOAT / NULLIF(monthly_transaction_count, 0) AS weekend_transaction_ratio
FROM time_aggregated_features
ORDER BY customer_id, month_year;
```

#### Explanation
1. **EXTRACT Function**: Pulls specific components (year, month, day, hour) from datetime
2. **Cyclical Encoding**: Uses SIN/COS to represent circular time features (months, hours)
3. **Window Functions**: Calculate relative temporal features and sequences
4. **LAG/LEAD**: Access previous/future temporal values for comparison features
5. **Business Logic**: Create domain-specific temporal indicators (holidays, seasons)

#### Use Cases
- **Time Series Forecasting**: Capture seasonality and trends for predictive models
- **Customer Behavior Analysis**: Understand temporal purchasing patterns
- **Anomaly Detection**: Identify unusual activity timing
- **Marketing Attribution**: Analyze campaign effectiveness over time
- **Operational Analytics**: Optimize staffing and inventory based on temporal patterns

#### Best Practices
- Use cyclical encoding (sin/cos) for circular time features to avoid artificial boundaries
- Create multiple temporal granularities (hour, day, week, month) for comprehensive coverage
- Include business-relevant time indicators (holidays, seasons, pay periods)
- Generate relative temporal features (time since last event, sequence numbers)
- Consider time zone implications for global datasets

#### Pitfalls
- **Time Zone Issues**: Ensure consistent time zone handling across datasets
- **Leap Year Effects**: Account for leap years in day-of-year calculations
- **Artificial Boundaries**: Avoid treating cyclical features (months, hours) as linear
- **Data Leakage**: Don't include future temporal information in historical features
- **Sparse Features**: Some temporal combinations may have very few observations

#### Debugging
```sql
-- Validate temporal feature extraction
SELECT 
    transaction_datetime,
    EXTRACT(YEAR FROM transaction_datetime) AS year,
    EXTRACT(MONTH FROM transaction_datetime) AS month,
    EXTRACT(DOW FROM transaction_datetime) AS dow,
    TO_CHAR(transaction_datetime, 'Day') AS day_name,
    CASE WHEN EXTRACT(DOW FROM transaction_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend
FROM transactions
WHERE transaction_datetime BETWEEN '2024-01-01' AND '2024-01-07'
ORDER BY transaction_datetime;

-- Check for temporal anomalies
SELECT 
    DATE(transaction_datetime) AS transaction_date,
    COUNT(*) AS daily_transaction_count,
    EXTRACT(DOW FROM transaction_datetime) AS day_of_week
FROM transactions
GROUP BY DATE(transaction_datetime), EXTRACT(DOW FROM transaction_datetime)
ORDER BY daily_transaction_count DESC;
```

#### Optimization
- **Indexing**: Create indexes on datetime columns for efficient temporal queries
- **Partitioning**: Use temporal partitioning for large time series datasets
- **Materialized Views**: Pre-compute frequently used temporal aggregations
- **Efficient Date Functions**: Use database-specific optimized date functions

---

## Question 5

**How do you join transactional data to a dimension table in such a way that features for Machine Learning can be extracted?**

### Answer:

#### Theory
Joining transactional data (fact tables) to dimension tables is fundamental for ML feature engineering. This process enriches transaction-level data with descriptive attributes, enabling the creation of comprehensive feature sets. The key is to maintain the grain of the fact table while adding relevant dimensional attributes that provide context and improve model predictive power.

#### Code Example
```sql
-- Basic Fact-to-Dimension JOIN for ML Feature Engineering
SELECT 
    -- Primary transaction identifiers
    t.transaction_id,
    t.customer_id,
    t.product_id,
    t.store_id,
    t.transaction_datetime,
    t.transaction_amount,
    t.quantity_sold,
    
    -- Customer dimensional features
    c.customer_segment,
    c.customer_age,
    c.customer_gender,
    c.customer_income_bracket,
    c.customer_city,
    c.customer_state,
    c.customer_registration_date,
    c.customer_preferred_channel,
    
    -- Product dimensional features
    p.product_category,
    p.product_subcategory,
    p.product_brand,
    p.product_price,
    p.product_cost,
    p.product_launch_date,
    p.product_status,
    p.product_seasonality_flag,
    
    -- Store dimensional features
    s.store_type,
    s.store_size_category,
    s.store_region,
    s.store_district,
    s.store_manager_experience,
    s.store_opening_date,
    
    -- Derived ML features from joins
    t.transaction_amount - (p.product_cost * t.quantity_sold) AS gross_profit,
    (t.transaction_amount / NULLIF(p.product_price * t.quantity_sold, 0)) AS discount_factor,
    EXTRACT(EPOCH FROM t.transaction_datetime - c.customer_registration_date) / 86400 AS customer_tenure_days,
    EXTRACT(EPOCH FROM t.transaction_datetime - p.product_launch_date) / 86400 AS product_age_days,
    EXTRACT(EPOCH FROM t.transaction_datetime - s.store_opening_date) / 86400 AS store_age_days,
    
    -- Categorical feature engineering
    CASE 
        WHEN c.customer_age < 25 THEN 'Young'
        WHEN c.customer_age < 45 THEN 'Adult'
        WHEN c.customer_age < 65 THEN 'Middle_Age'
        ELSE 'Senior'
    END AS customer_age_group,
    
    CASE 
        WHEN t.transaction_amount > p.product_price * t.quantity_sold * 0.9 THEN 'Full_Price'
        WHEN t.transaction_amount > p.product_price * t.quantity_sold * 0.7 THEN 'Moderate_Discount'
        ELSE 'Heavy_Discount'
    END AS discount_category,
    
    -- Time-based features from transaction datetime
    EXTRACT(HOUR FROM t.transaction_datetime) AS transaction_hour,
    EXTRACT(DOW FROM t.transaction_datetime) AS transaction_dow,
    EXTRACT(MONTH FROM t.transaction_datetime) AS transaction_month,
    CASE WHEN EXTRACT(DOW FROM t.transaction_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
    
    -- Cross-dimensional features
    CONCAT(c.customer_segment, '_', p.product_category) AS customer_product_segment,
    CONCAT(s.store_region, '_', p.product_brand) AS region_brand_combination
FROM transactions t
INNER JOIN dim_customers c ON t.customer_id = c.customer_id
INNER JOIN dim_products p ON t.product_id = p.product_id
INNER JOIN dim_stores s ON t.store_id = s.store_id
WHERE t.transaction_datetime >= CURRENT_DATE - INTERVAL '2 years';

-- Advanced Feature Engineering with Multiple Dimension Levels
WITH enriched_transactions AS (
    SELECT 
        t.*,
        c.*,
        p.*,
        s.*,
        
        -- Geography hierarchy features
        geo.region_name,
        geo.region_population,
        geo.region_income_median,
        geo.region_competition_density,
        
        -- Product hierarchy features
        pc.category_growth_rate,
        pc.category_seasonality_index,
        pc.category_profit_margin,
        
        -- Time dimension features
        td.is_holiday,
        td.is_peak_shopping_season,
        td.economic_index,
        td.weather_temperature,
        
        -- Campaign/Promotion features
        pr.campaign_type,
        pr.discount_percentage,
        pr.promotion_channel
    FROM transactions t
    LEFT JOIN dim_customers c ON t.customer_id = c.customer_id
    LEFT JOIN dim_products p ON t.product_id = p.product_id
    LEFT JOIN dim_stores s ON t.store_id = s.store_id
    LEFT JOIN dim_geography geo ON s.store_region = geo.region_code
    LEFT JOIN dim_product_categories pc ON p.product_category = pc.category_name
    LEFT JOIN dim_time td ON DATE(t.transaction_datetime) = td.calendar_date
    LEFT JOIN dim_promotions pr ON t.promotion_id = pr.promotion_id
),
advanced_features AS (
    SELECT 
        transaction_id,
        customer_id,
        product_id,
        store_id,
        transaction_datetime,
        transaction_amount,
        quantity_sold,
        
        -- Customer features
        customer_segment,
        customer_age,
        customer_income_bracket,
        customer_city,
        customer_registration_date,
        
        -- Product features
        product_category,
        product_subcategory,
        product_brand,
        product_price,
        product_launch_date,
        category_growth_rate,
        category_seasonality_index,
        
        -- Store and geography features
        store_type,
        store_region,
        region_population,
        region_income_median,
        region_competition_density,
        
        -- Time and external features
        is_holiday,
        is_peak_shopping_season,
        economic_index,
        weather_temperature,
        
        -- Promotion features
        campaign_type,
        discount_percentage,
        promotion_channel,
        
        -- Calculated ML features
        transaction_amount / NULLIF(product_price * quantity_sold, 0) AS actual_vs_expected_price_ratio,
        
        -- Customer value indicators
        CASE 
            WHEN customer_income_bracket = 'High' AND customer_segment = 'Premium' THEN 1 
            ELSE 0 
        END AS is_high_value_customer,
        
        -- Product affinity features
        CASE 
            WHEN product_category IN ('Electronics', 'Luxury') AND customer_age < 35 THEN 1 
            ELSE 0 
        END AS young_tech_affinity,
        
        -- Geographic market features
        CASE 
            WHEN region_competition_density > 5 THEN 'High_Competition'
            WHEN region_competition_density > 2 THEN 'Medium_Competition'
            ELSE 'Low_Competition'
        END AS market_competition_level,
        
        -- Seasonal adjustment features
        transaction_amount * category_seasonality_index AS seasonally_adjusted_amount,
        
        -- Economic adjustment features
        transaction_amount / NULLIF(economic_index, 0) AS economy_adjusted_amount,
        
        -- Weather influence features
        CASE 
            WHEN product_category IN ('Clothing', 'Sports') AND ABS(weather_temperature - 70) > 20 THEN 1 
            ELSE 0 
        END AS weather_influenced_purchase
    FROM enriched_transactions
)
SELECT * FROM advanced_features;

-- Aggregated Dimensional Features for Customer-Level ML
WITH customer_dimensional_features AS (
    SELECT 
        c.customer_id,
        c.customer_segment,
        c.customer_age,
        c.customer_income_bracket,
        c.customer_registration_date,
        
        -- Transaction aggregations
        COUNT(t.transaction_id) AS total_transactions,
        SUM(t.transaction_amount) AS total_spent,
        AVG(t.transaction_amount) AS avg_transaction_amount,
        STDDEV(t.transaction_amount) AS transaction_amount_volatility,
        MAX(t.transaction_datetime) AS last_transaction_date,
        MIN(t.transaction_datetime) AS first_transaction_date,
        
        -- Product diversity features
        COUNT(DISTINCT p.product_category) AS unique_categories_purchased,
        COUNT(DISTINCT p.product_brand) AS unique_brands_purchased,
        COUNT(DISTINCT p.product_id) AS unique_products_purchased,
        
        -- Store loyalty features
        COUNT(DISTINCT s.store_id) AS unique_stores_visited,
        MODE() WITHIN GROUP (ORDER BY s.store_id) AS most_frequent_store,
        MODE() WITHIN GROUP (ORDER BY s.store_region) AS preferred_region,
        
        -- Geographic behavior
        COUNT(DISTINCT s.store_region) AS regions_shopped,
        AVG(geo.region_income_median) AS avg_shopping_region_income,
        
        -- Product preference patterns
        MODE() WITHIN GROUP (ORDER BY p.product_category) AS preferred_category,
        MODE() WITHIN GROUP (ORDER BY p.product_brand) AS preferred_brand,
        
        -- Price sensitivity features
        AVG(t.transaction_amount / NULLIF(p.product_price * t.quantity_sold, 0)) AS avg_discount_rate,
        MIN(t.transaction_amount / NULLIF(p.product_price * t.quantity_sold, 0)) AS max_discount_taken,
        
        -- Seasonal behavior
        COUNT(CASE WHEN td.is_holiday = 1 THEN 1 END) AS holiday_transactions,
        COUNT(CASE WHEN td.is_peak_shopping_season = 1 THEN 1 END) AS peak_season_transactions,
        
        -- Channel preferences
        COUNT(CASE WHEN pr.promotion_channel = 'Email' THEN 1 END) AS email_promotion_responses,
        COUNT(CASE WHEN pr.promotion_channel = 'Social_Media' THEN 1 END) AS social_promotion_responses,
        
        -- Temporal patterns
        AVG(EXTRACT(HOUR FROM t.transaction_datetime)) AS avg_shopping_hour,
        COUNT(CASE WHEN EXTRACT(DOW FROM t.transaction_datetime) IN (0, 6) THEN 1 END) AS weekend_transactions,
        
        -- Value-based segmentation features
        SUM(t.transaction_amount) / NULLIF(COUNT(t.transaction_id), 0) AS customer_avg_order_value,
        SUM(t.transaction_amount - (p.product_cost * t.quantity_sold)) AS total_profit_generated
    FROM dim_customers c
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    LEFT JOIN dim_products p ON t.product_id = p.product_id
    LEFT JOIN dim_stores s ON t.store_id = s.store_id
    LEFT JOIN dim_geography geo ON s.store_region = geo.region_code
    LEFT JOIN dim_time td ON DATE(t.transaction_datetime) = td.calendar_date
    LEFT JOIN dim_promotions pr ON t.promotion_id = pr.promotion_id
    WHERE t.transaction_datetime >= CURRENT_DATE - INTERVAL '1 year' OR t.transaction_datetime IS NULL
    GROUP BY c.customer_id, c.customer_segment, c.customer_age, c.customer_income_bracket, c.customer_registration_date
),
customer_rfm_features AS (
    SELECT 
        customer_id,
        
        -- Recency (days since last transaction)
        COALESCE(EXTRACT(EPOCH FROM CURRENT_DATE - last_transaction_date) / 86400, 9999) AS recency_days,
        
        -- Frequency (total transactions)
        total_transactions,
        
        -- Monetary (total spent)
        total_spent,
        
        -- RFM Scores (1-5 scale)
        CASE 
            WHEN COALESCE(EXTRACT(EPOCH FROM CURRENT_DATE - last_transaction_date) / 86400, 9999) <= 30 THEN 5
            WHEN COALESCE(EXTRACT(EPOCH FROM CURRENT_DATE - last_transaction_date) / 86400, 9999) <= 90 THEN 4
            WHEN COALESCE(EXTRACT(EPOCH FROM CURRENT_DATE - last_transaction_date) / 86400, 9999) <= 180 THEN 3
            WHEN COALESCE(EXTRACT(EPOCH FROM CURRENT_DATE - last_transaction_date) / 86400, 9999) <= 365 THEN 2
            ELSE 1
        END AS recency_score,
        
        CASE 
            WHEN total_transactions >= 20 THEN 5
            WHEN total_transactions >= 10 THEN 4
            WHEN total_transactions >= 5 THEN 3
            WHEN total_transactions >= 2 THEN 2
            WHEN total_transactions >= 1 THEN 1
            ELSE 0
        END AS frequency_score,
        
        CASE 
            WHEN total_spent >= 5000 THEN 5
            WHEN total_spent >= 2000 THEN 4
            WHEN total_spent >= 1000 THEN 3
            WHEN total_spent >= 500 THEN 2
            WHEN total_spent >= 100 THEN 1
            ELSE 0
        END AS monetary_score
    FROM customer_dimensional_features
),
final_customer_features AS (
    SELECT 
        cdf.*,
        crf.recency_days,
        crf.recency_score,
        crf.frequency_score,
        crf.monetary_score,
        
        -- Combined RFM score
        (crf.recency_score + crf.frequency_score + crf.monetary_score) AS rfm_combined_score,
        
        -- Customer lifetime value estimate
        (cdf.total_spent / NULLIF(EXTRACT(EPOCH FROM CURRENT_DATE - cdf.customer_registration_date) / 86400, 0)) * 365 AS estimated_annual_value,
        
        -- Engagement indicators
        CASE WHEN cdf.total_transactions = 0 THEN 1 ELSE 0 END AS is_inactive_customer,
        CASE WHEN crf.recency_days <= 90 AND cdf.total_transactions >= 3 THEN 1 ELSE 0 END AS is_active_customer,
        CASE WHEN cdf.total_spent >= 2000 AND cdf.total_transactions >= 10 THEN 1 ELSE 0 END AS is_loyal_customer,
        
        -- Diversity and exploration features
        cdf.unique_categories_purchased::FLOAT / NULLIF(cdf.total_transactions, 0) AS category_exploration_rate,
        cdf.unique_stores_visited::FLOAT / NULLIF(cdf.total_transactions, 0) AS store_exploration_rate,
        
        -- Price sensitivity features
        CASE WHEN cdf.avg_discount_rate < 0.8 THEN 1 ELSE 0 END AS is_price_sensitive,
        
        -- Behavioral consistency features
        cdf.weekend_transactions::FLOAT / NULLIF(cdf.total_transactions, 0) AS weekend_shopping_propensity,
        cdf.holiday_transactions::FLOAT / NULLIF(cdf.total_transactions, 0) AS holiday_shopping_propensity
    FROM customer_dimensional_features cdf
    JOIN customer_rfm_features crf ON cdf.customer_id = crf.customer_id
)
SELECT * FROM final_customer_features
ORDER BY rfm_combined_score DESC, estimated_annual_value DESC;

-- Product-Level Dimensional Features for Recommendation Systems
WITH product_performance_features AS (
    SELECT 
        p.product_id,
        p.product_category,
        p.product_subcategory,
        p.product_brand,
        p.product_price,
        p.product_cost,
        p.product_launch_date,
        
        -- Sales performance metrics
        COUNT(t.transaction_id) AS total_transactions,
        SUM(t.quantity_sold) AS total_units_sold,
        SUM(t.transaction_amount) AS total_revenue,
        AVG(t.transaction_amount / t.quantity_sold) AS avg_selling_price,
        SUM(t.transaction_amount - (p.product_cost * t.quantity_sold)) AS total_profit,
        
        -- Customer reach metrics
        COUNT(DISTINCT t.customer_id) AS unique_customers,
        COUNT(DISTINCT c.customer_segment) AS customer_segments_reached,
        
        -- Geographic reach
        COUNT(DISTINCT s.store_region) AS regions_sold,
        COUNT(DISTINCT s.store_id) AS stores_sold,
        
        -- Temporal patterns
        COUNT(CASE WHEN EXTRACT(DOW FROM t.transaction_datetime) IN (0, 6) THEN 1 END) AS weekend_sales,
        COUNT(CASE WHEN td.is_holiday = 1 THEN 1 END) AS holiday_sales,
        
        -- Customer demographics
        AVG(c.customer_age) AS avg_customer_age,
        MODE() WITHIN GROUP (ORDER BY c.customer_segment) AS primary_customer_segment,
        MODE() WITHIN GROUP (ORDER BY c.customer_income_bracket) AS primary_income_bracket,
        
        -- Cross-selling potential
        AVG(transaction_basket_size.basket_size) AS avg_basket_size_when_included,
        
        -- Price elasticity indicators
        CORR(t.transaction_amount / t.quantity_sold, t.quantity_sold) AS price_quantity_correlation
    FROM dim_products p
    LEFT JOIN transactions t ON p.product_id = t.product_id
    LEFT JOIN dim_customers c ON t.customer_id = c.customer_id
    LEFT JOIN dim_stores s ON t.store_id = s.store_id
    LEFT JOIN dim_time td ON DATE(t.transaction_datetime) = td.calendar_date
    LEFT JOIN (
        SELECT 
            t1.transaction_id,
            COUNT(t2.product_id) AS basket_size
        FROM transactions t1
        JOIN transactions t2 ON t1.transaction_id = t2.transaction_id
        GROUP BY t1.transaction_id
    ) transaction_basket_size ON t.transaction_id = transaction_basket_size.transaction_id
    WHERE t.transaction_datetime >= CURRENT_DATE - INTERVAL '1 year' OR t.transaction_datetime IS NULL
    GROUP BY p.product_id, p.product_category, p.product_subcategory, p.product_brand, 
             p.product_price, p.product_cost, p.product_launch_date
)
SELECT 
    product_id,
    product_category,
    product_subcategory,
    product_brand,
    product_price,
    total_transactions,
    total_revenue,
    total_profit,
    unique_customers,
    
    -- Performance ratios
    total_revenue / NULLIF(total_transactions, 0) AS revenue_per_transaction,
    total_profit / NULLIF(total_revenue, 0) AS profit_margin,
    unique_customers::FLOAT / NULLIF(total_transactions, 0) AS customer_diversity_ratio,
    
    -- Market penetration
    regions_sold,
    stores_sold,
    customer_segments_reached,
    
    -- Seasonality indicators
    weekend_sales::FLOAT / NULLIF(total_transactions, 0) AS weekend_sales_ratio,
    holiday_sales::FLOAT / NULLIF(total_transactions, 0) AS holiday_sales_ratio,
    
    -- Customer profile
    avg_customer_age,
    primary_customer_segment,
    primary_income_bracket,
    
    -- Product positioning
    CASE 
        WHEN avg_selling_price > product_price * 0.95 THEN 'Premium_Positioned'
        WHEN avg_selling_price > product_price * 0.85 THEN 'Standard_Positioned'
        ELSE 'Discount_Positioned'
    END AS price_positioning,
    
    -- Cross-sell potential
    avg_basket_size_when_included,
    
    -- Performance classification
    CASE 
        WHEN total_profit > 0 AND total_transactions > 50 THEN 'High_Performer'
        WHEN total_profit > 0 AND total_transactions > 10 THEN 'Medium_Performer'
        WHEN total_transactions > 0 THEN 'Low_Performer'
        ELSE 'No_Sales'
    END AS performance_category
FROM product_performance_features
ORDER BY total_profit DESC, total_revenue DESC;
```

#### Explanation
1. **Grain Preservation**: Maintain the transaction-level detail while adding dimensional context
2. **Feature Derivation**: Calculate new features using dimensional attributes and facts
3. **Multi-level Joins**: Combine multiple dimension tables for comprehensive features
4. **Aggregation Patterns**: Create customer and product-level features for different ML use cases
5. **NULL Handling**: Use LEFT JOINs and COALESCE for missing dimensional data

#### Use Cases
- **Customer Segmentation**: Combine customer attributes with transaction behavior
- **Recommendation Systems**: Product features with customer interaction patterns
- **Churn Prediction**: Customer demographics with engagement metrics
- **Price Optimization**: Product attributes with sales performance
- **Market Basket Analysis**: Transaction details with product and customer context

#### Best Practices
- Use LEFT JOINs to preserve fact table completeness
- Create derived features that combine multiple dimensional attributes
- Handle NULL values explicitly in dimensional data
- Consider the cardinality impact of dimension joins
- Pre-aggregate dimensional features for performance

#### Pitfalls
- **Fanout Issues**: Many-to-many relationships creating duplicate records
- **Dimension Changes**: Slowly changing dimensions affecting historical accuracy
- **Performance Impact**: Multiple joins on large fact tables
- **Feature Explosion**: Too many dimensional features causing overfitting
- **Temporal Misalignment**: Using current dimensional data for historical facts

#### Debugging
```sql
-- Validate join results
SELECT 
    'Original Transactions' AS dataset,
    COUNT(*) AS record_count
FROM transactions
UNION ALL
SELECT 'After Dimensional Joins',
    COUNT(*)
FROM transactions t
JOIN dim_customers c ON t.customer_id = c.customer_id
JOIN dim_products p ON t.product_id = p.product_id;

-- Check for duplicate records
SELECT 
    transaction_id,
    COUNT(*) AS duplicate_count
FROM enriched_transactions
GROUP BY transaction_id
HAVING COUNT(*) > 1;
```

#### Optimization
- **Indexing**: Create indexes on foreign key columns in fact tables
- **Partitioning**: Partition fact tables by date for efficient joins
- **Materialized Views**: Pre-compute dimensional features for repeated use
- **Star Schema Design**: Optimize dimensional model for analytical queries

---

## Question 6

**How can you deal with outliers in a SQL database before passing data to Machine Learning algorithms?**

### Answer:

#### Theory
Outlier handling in SQL is critical for ML preprocessing as outliers can severely impact model performance, especially for algorithms sensitive to extreme values. SQL provides multiple statistical approaches for outlier detection and treatment including IQR methods, Z-score analysis, percentile-based filtering, and robust statistical measures. The goal is to identify and appropriately handle data points that deviate significantly from the expected distribution.

#### Code Example
```sql
-- Method 1: Interquartile Range (IQR) Outlier Detection and Treatment
WITH iqr_statistics AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_amount) AS q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_amount) AS q3,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_amount) - 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_amount) AS iqr
    FROM sales_data
    WHERE transaction_amount IS NOT NULL
),
outlier_detection AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount,
        product_category,
        transaction_date,
        
        -- IQR bounds
        iqr_stats.q1,
        iqr_stats.q3,
        iqr_stats.iqr,
        iqr_stats.q1 - 1.5 * iqr_stats.iqr AS lower_bound,
        iqr_stats.q3 + 1.5 * iqr_stats.iqr AS upper_bound,
        
        -- Outlier classification
        CASE 
            WHEN transaction_amount < (iqr_stats.q1 - 1.5 * iqr_stats.iqr) THEN 'Lower_Outlier'
            WHEN transaction_amount > (iqr_stats.q3 + 1.5 * iqr_stats.iqr) THEN 'Upper_Outlier'
            ELSE 'Normal'
        END AS outlier_status,
        
        -- Severity levels
        CASE 
            WHEN transaction_amount < (iqr_stats.q1 - 3 * iqr_stats.iqr) OR 
                 transaction_amount > (iqr_stats.q3 + 3 * iqr_stats.iqr) THEN 'Extreme'
            WHEN transaction_amount < (iqr_stats.q1 - 1.5 * iqr_stats.iqr) OR 
                 transaction_amount > (iqr_stats.q3 + 1.5 * iqr_stats.iqr) THEN 'Moderate'
            ELSE 'Normal'
        END AS outlier_severity
    FROM sales_data sd
    CROSS JOIN iqr_statistics iqr_stats
    WHERE sd.transaction_amount IS NOT NULL
),
treatment_options AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount AS original_amount,
        product_category,
        transaction_date,
        outlier_status,
        outlier_severity,
        
        -- Treatment Option 1: Capping/Winsorization
        CASE 
            WHEN transaction_amount < lower_bound THEN lower_bound
            WHEN transaction_amount > upper_bound THEN upper_bound
            ELSE transaction_amount
        END AS capped_amount,
        
        -- Treatment Option 2: Log Transformation
        CASE 
            WHEN transaction_amount > 0 THEN LN(transaction_amount)
            ELSE NULL
        END AS log_transformed_amount,
        
        -- Treatment Option 3: Z-score based capping
        CASE 
            WHEN ABS((transaction_amount - AVG(transaction_amount) OVER ()) / 
                     NULLIF(STDDEV(transaction_amount) OVER (), 0)) > 3 
            THEN 
                CASE 
                    WHEN transaction_amount > AVG(transaction_amount) OVER () 
                    THEN AVG(transaction_amount) OVER () + 3 * STDDEV(transaction_amount) OVER ()
                    ELSE AVG(transaction_amount) OVER () - 3 * STDDEV(transaction_amount) OVER ()
                END
            ELSE transaction_amount
        END AS zscore_capped_amount,
        
        -- Outlier indicator flag for ML
        CASE WHEN outlier_status != 'Normal' THEN 1 ELSE 0 END AS is_outlier_flag
    FROM outlier_detection
)
SELECT 
    outlier_status,
    outlier_severity,
    COUNT(*) AS record_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
    MIN(original_amount) AS min_original,
    MAX(original_amount) AS max_original,
    AVG(original_amount) AS avg_original,
    AVG(capped_amount) AS avg_capped,
    STDDEV(original_amount) AS stddev_original,
    STDDEV(capped_amount) AS stddev_capped
FROM treatment_options
GROUP BY outlier_status, outlier_severity
ORDER BY record_count DESC;

-- Method 2: Z-Score Based Outlier Detection and Removal
WITH z_score_analysis AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount,
        product_category,
        
        -- Statistical measures
        AVG(transaction_amount) OVER () AS mean_amount,
        STDDEV(transaction_amount) OVER () AS stddev_amount,
        
        -- Z-score calculation
        (transaction_amount - AVG(transaction_amount) OVER ()) / 
        NULLIF(STDDEV(transaction_amount) OVER (), 0) AS z_score,
        
        -- Absolute Z-score
        ABS((transaction_amount - AVG(transaction_amount) OVER ()) / 
            NULLIF(STDDEV(transaction_amount) OVER (), 0)) AS abs_z_score
    FROM sales_data
    WHERE transaction_amount IS NOT NULL
),
z_score_classification AS (
    SELECT 
        *,
        CASE 
            WHEN abs_z_score > 3 THEN 'Extreme_Outlier'
            WHEN abs_z_score > 2.5 THEN 'Strong_Outlier'
            WHEN abs_z_score > 2 THEN 'Moderate_Outlier'
            WHEN abs_z_score > 1.5 THEN 'Mild_Outlier'
            ELSE 'Normal'
        END AS z_score_category,
        
        -- Treatment decisions
        CASE 
            WHEN abs_z_score > 3 THEN 'Remove'
            WHEN abs_z_score > 2 THEN 'Transform'
            ELSE 'Keep'
        END AS treatment_decision
    FROM z_score_analysis
),
cleaned_dataset AS (
    SELECT 
        transaction_id,
        customer_id,
        product_category,
        
        -- Original and treated amounts
        transaction_amount AS original_amount,
        
        -- Conditional treatment
        CASE 
            WHEN treatment_decision = 'Remove' THEN NULL
            WHEN treatment_decision = 'Transform' THEN 
                CASE 
                    WHEN z_score > 0 THEN mean_amount + 2 * stddev_amount
                    ELSE mean_amount - 2 * stddev_amount
                END
            ELSE transaction_amount
        END AS treated_amount,
        
        z_score_category,
        treatment_decision,
        z_score,
        abs_z_score
    FROM z_score_classification
)
SELECT 
    treatment_decision,
    z_score_category,
    COUNT(*) AS record_count,
    COUNT(treated_amount) AS records_retained,
    ROUND(COUNT(treated_amount) * 100.0 / COUNT(*), 2) AS retention_rate,
    AVG(original_amount) AS avg_original,
    AVG(treated_amount) AS avg_treated,
    STDDEV(original_amount) AS stddev_original,
    STDDEV(treated_amount) AS stddev_treated
FROM cleaned_dataset
GROUP BY treatment_decision, z_score_category
ORDER BY record_count DESC;

-- Method 3: Percentile-Based Outlier Handling
WITH percentile_analysis AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount,
        product_category,
        transaction_date,
        
        -- Percentile calculations
        PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY transaction_amount) OVER () AS p1,
        PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY transaction_amount) OVER () AS p5,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY transaction_amount) OVER () AS p95,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY transaction_amount) OVER () AS p99,
        
        -- Percentile rank of current transaction
        PERCENT_RANK() OVER (ORDER BY transaction_amount) AS percentile_rank
    FROM sales_data
    WHERE transaction_amount IS NOT NULL
),
percentile_treatment AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount AS original_amount,
        product_category,
        transaction_date,
        percentile_rank,
        
        -- Classification based on percentiles
        CASE 
            WHEN percentile_rank < 0.01 OR percentile_rank > 0.99 THEN 'P1_P99_Outlier'
            WHEN percentile_rank < 0.05 OR percentile_rank > 0.95 THEN 'P5_P95_Outlier'
            WHEN percentile_rank < 0.10 OR percentile_rank > 0.90 THEN 'P10_P90_Outlier'
            ELSE 'Normal'
        END AS percentile_category,
        
        -- Winsorization at different levels
        CASE 
            WHEN transaction_amount < p1 THEN p1
            WHEN transaction_amount > p99 THEN p99
            ELSE transaction_amount
        END AS winsorized_p1_p99,
        
        CASE 
            WHEN transaction_amount < p5 THEN p5
            WHEN transaction_amount > p95 THEN p95
            ELSE transaction_amount
        END AS winsorized_p5_p95,
        
        -- Binary outlier flags
        CASE WHEN percentile_rank < 0.01 OR percentile_rank > 0.99 THEN 1 ELSE 0 END AS is_extreme_outlier,
        CASE WHEN percentile_rank < 0.05 OR percentile_rank > 0.95 THEN 1 ELSE 0 END AS is_outlier_5pct
    FROM percentile_analysis
)
SELECT 
    percentile_category,
    COUNT(*) AS record_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
    MIN(original_amount) AS min_amount,
    MAX(original_amount) AS max_amount,
    AVG(original_amount) AS avg_original,
    AVG(winsorized_p1_p99) AS avg_winsorized_p1_p99,
    AVG(winsorized_p5_p95) AS avg_winsorized_p5_p95,
    STDDEV(original_amount) AS stddev_original,
    STDDEV(winsorized_p1_p99) AS stddev_winsorized
FROM percentile_treatment
GROUP BY percentile_category
ORDER BY record_count DESC;

-- Method 4: Modified Z-Score (Robust to Outliers)
WITH median_absolute_deviation AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount,
        product_category,
        
        -- Median and MAD calculation
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY transaction_amount) OVER () AS median_amount,
        
        -- Calculate absolute deviations from median
        ABS(transaction_amount - PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY transaction_amount) OVER ()) AS abs_deviation
    FROM sales_data
    WHERE transaction_amount IS NOT NULL
),
mad_statistics AS (
    SELECT 
        *,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY abs_deviation) OVER () AS mad
    FROM median_absolute_deviation
),
modified_z_scores AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount,
        product_category,
        median_amount,
        mad,
        
        -- Modified Z-score calculation (more robust than standard Z-score)
        0.6745 * (transaction_amount - median_amount) / NULLIF(mad, 0) AS modified_z_score,
        ABS(0.6745 * (transaction_amount - median_amount) / NULLIF(mad, 0)) AS abs_modified_z_score
    FROM mad_statistics
),
robust_outlier_detection AS (
    SELECT 
        *,
        CASE 
            WHEN abs_modified_z_score > 3.5 THEN 'Outlier'
            ELSE 'Normal'
        END AS robust_outlier_flag,
        
        -- Robust treatment using median and MAD
        CASE 
            WHEN abs_modified_z_score > 3.5 THEN
                CASE 
                    WHEN modified_z_score > 0 THEN median_amount + 3.5 * mad / 0.6745
                    ELSE median_amount - 3.5 * mad / 0.6745
                END
            ELSE transaction_amount
        END AS robust_treated_amount
    FROM modified_z_scores
)
SELECT 
    robust_outlier_flag,
    COUNT(*) AS record_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
    AVG(transaction_amount) AS avg_original,
    AVG(robust_treated_amount) AS avg_treated,
    STDDEV(transaction_amount) AS stddev_original,
    STDDEV(robust_treated_amount) AS stddev_treated,
    MIN(abs_modified_z_score) AS min_modified_z,
    MAX(abs_modified_z_score) AS max_modified_z
FROM robust_outlier_detection
GROUP BY robust_outlier_flag
ORDER BY record_count DESC;

-- Method 5: Multivariate Outlier Detection
WITH standardized_features AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount,
        customer_age,
        days_since_last_purchase,
        total_previous_purchases,
        
        -- Standardize all features
        (transaction_amount - AVG(transaction_amount) OVER ()) / NULLIF(STDDEV(transaction_amount) OVER (), 0) AS std_amount,
        (customer_age - AVG(customer_age) OVER ()) / NULLIF(STDDEV(customer_age) OVER (), 0) AS std_age,
        (days_since_last_purchase - AVG(days_since_last_purchase) OVER ()) / NULLIF(STDDEV(days_since_last_purchase) OVER (), 0) AS std_days_since,
        (total_previous_purchases - AVG(total_previous_purchases) OVER ()) / NULLIF(STDDEV(total_previous_purchases) OVER (), 0) AS std_prev_purchases
    FROM customer_transactions
    WHERE transaction_amount IS NOT NULL 
      AND customer_age IS NOT NULL 
      AND days_since_last_purchase IS NOT NULL 
      AND total_previous_purchases IS NOT NULL
),
mahalanobis_approximation AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount,
        customer_age,
        days_since_last_purchase,
        total_previous_purchases,
        
        -- Simplified Mahalanobis distance (assuming independence)
        SQRT(
            POWER(std_amount, 2) + 
            POWER(std_age, 2) + 
            POWER(std_days_since, 2) + 
            POWER(std_prev_purchases, 2)
        ) AS multivariate_distance,
        
        -- Individual standardized features
        std_amount,
        std_age,
        std_days_since,
        std_prev_purchases
    FROM standardized_features
),
multivariate_outliers AS (
    SELECT 
        *,
        PERCENT_RANK() OVER (ORDER BY multivariate_distance) AS distance_percentile,
        
        CASE 
            WHEN multivariate_distance > (SELECT PERCENTILE_CONT(0.975) WITHIN GROUP (ORDER BY multivariate_distance) FROM mahalanobis_approximation) THEN 'Top_2.5_Percent'
            WHEN multivariate_distance > (SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY multivariate_distance) FROM mahalanobis_approximation) THEN 'Top_5_Percent'
            WHEN multivariate_distance > (SELECT PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY multivariate_distance) FROM mahalanobis_approximation) THEN 'Top_10_Percent'
            ELSE 'Normal'
        END AS multivariate_outlier_category
    FROM mahalanobis_approximation
)
SELECT 
    multivariate_outlier_category,
    COUNT(*) AS record_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
    AVG(multivariate_distance) AS avg_distance,
    MIN(multivariate_distance) AS min_distance,
    MAX(multivariate_distance) AS max_distance,
    AVG(transaction_amount) AS avg_transaction_amount,
    AVG(customer_age) AS avg_customer_age
FROM multivariate_outliers
GROUP BY multivariate_outlier_category
ORDER BY avg_distance DESC;

-- Method 6: Contextual/Conditional Outlier Detection
WITH contextual_analysis AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount,
        product_category,
        customer_segment,
        store_location,
        transaction_date,
        
        -- Context-specific statistics
        AVG(transaction_amount) OVER (PARTITION BY product_category) AS category_avg,
        STDDEV(transaction_amount) OVER (PARTITION BY product_category) AS category_stddev,
        AVG(transaction_amount) OVER (PARTITION BY customer_segment) AS segment_avg,
        STDDEV(transaction_amount) OVER (PARTITION BY customer_segment) AS segment_stddev,
        AVG(transaction_amount) OVER (PARTITION BY store_location) AS location_avg,
        STDDEV(transaction_amount) OVER (PARTITION BY store_location) AS location_stddev,
        
        -- Global statistics for comparison
        AVG(transaction_amount) OVER () AS global_avg,
        STDDEV(transaction_amount) OVER () AS global_stddev
    FROM sales_transactions
    WHERE transaction_amount IS NOT NULL
),
contextual_outliers AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount,
        product_category,
        customer_segment,
        store_location,
        
        -- Z-scores in different contexts
        (transaction_amount - category_avg) / NULLIF(category_stddev, 0) AS category_z_score,
        (transaction_amount - segment_avg) / NULLIF(segment_stddev, 0) AS segment_z_score,
        (transaction_amount - location_avg) / NULLIF(location_stddev, 0) AS location_z_score,
        (transaction_amount - global_avg) / NULLIF(global_stddev, 0) AS global_z_score,
        
        -- Outlier flags in different contexts
        CASE WHEN ABS((transaction_amount - category_avg) / NULLIF(category_stddev, 0)) > 2.5 THEN 1 ELSE 0 END AS is_category_outlier,
        CASE WHEN ABS((transaction_amount - segment_avg) / NULLIF(segment_stddev, 0)) > 2.5 THEN 1 ELSE 0 END AS is_segment_outlier,
        CASE WHEN ABS((transaction_amount - location_avg) / NULLIF(location_stddev, 0)) > 2.5 THEN 1 ELSE 0 END AS is_location_outlier,
        CASE WHEN ABS((transaction_amount - global_avg) / NULLIF(global_stddev, 0)) > 2.5 THEN 1 ELSE 0 END AS is_global_outlier,
        
        -- Contextual treatment
        CASE 
            WHEN ABS((transaction_amount - category_avg) / NULLIF(category_stddev, 0)) > 3 THEN
                CASE 
                    WHEN transaction_amount > category_avg THEN category_avg + 3 * category_stddev
                    ELSE category_avg - 3 * category_stddev
                END
            ELSE transaction_amount
        END AS category_treated_amount
    FROM contextual_analysis
),
outlier_summary AS (
    SELECT 
        product_category,
        customer_segment,
        store_location,
        COUNT(*) AS total_transactions,
        SUM(is_category_outlier) AS category_outliers,
        SUM(is_segment_outlier) AS segment_outliers,
        SUM(is_location_outlier) AS location_outliers,
        SUM(is_global_outlier) AS global_outliers,
        
        -- Outlier rates by context
        ROUND(SUM(is_category_outlier) * 100.0 / COUNT(*), 2) AS category_outlier_rate,
        ROUND(SUM(is_segment_outlier) * 100.0 / COUNT(*), 2) AS segment_outlier_rate,
        ROUND(SUM(is_location_outlier) * 100.0 / COUNT(*), 2) AS location_outlier_rate,
        ROUND(SUM(is_global_outlier) * 100.0 / COUNT(*), 2) AS global_outlier_rate,
        
        -- Impact on statistics
        AVG(transaction_amount) AS avg_original,
        AVG(category_treated_amount) AS avg_treated,
        STDDEV(transaction_amount) AS stddev_original,
        STDDEV(category_treated_amount) AS stddev_treated
    FROM contextual_outliers
    GROUP BY product_category, customer_segment, store_location
)
SELECT 
    product_category,
    customer_segment,
    total_transactions,
    category_outlier_rate,
    segment_outlier_rate,
    location_outlier_rate,
    global_outlier_rate,
    ROUND(avg_original, 2) AS avg_original,
    ROUND(avg_treated, 2) AS avg_treated,
    ROUND(stddev_original, 2) AS stddev_original,
    ROUND(stddev_treated, 2) AS stddev_treated
FROM outlier_summary
WHERE total_transactions > 10  -- Filter for sufficient sample size
ORDER BY category_outlier_rate DESC;

-- Final Cleaned Dataset Creation
CREATE TABLE ml_ready_sales_data AS
WITH comprehensive_outlier_treatment AS (
    SELECT 
        transaction_id,
        customer_id,
        product_id,
        transaction_date,
        product_category,
        customer_segment,
        
        -- Original values
        transaction_amount AS original_amount,
        customer_age AS original_age,
        
        -- Multiple treatment approaches
        -- IQR-based treatment
        CASE 
            WHEN transaction_amount < (
                SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_amount) - 
                       1.5 * (PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_amount) - 
                              PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_amount))
                FROM sales_transactions
            ) THEN (
                SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_amount) - 
                       1.5 * (PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_amount) - 
                              PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_amount))
                FROM sales_transactions
            )
            WHEN transaction_amount > (
                SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_amount) + 
                       1.5 * (PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_amount) - 
                              PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_amount))
                FROM sales_transactions
            ) THEN (
                SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_amount) + 
                       1.5 * (PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_amount) - 
                              PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_amount))
                FROM sales_transactions
            )
            ELSE transaction_amount
        END AS iqr_treated_amount,
        
        -- Log transformation for positive skewed data
        CASE 
            WHEN transaction_amount > 0 THEN LN(transaction_amount)
            ELSE NULL
        END AS log_amount,
        
        -- Outlier indicator flags (useful as features)
        CASE 
            WHEN ABS((transaction_amount - AVG(transaction_amount) OVER ()) / 
                     NULLIF(STDDEV(transaction_amount) OVER (), 0)) > 2.5 THEN 1 
            ELSE 0 
        END AS is_amount_outlier,
        
        CASE 
            WHEN ABS((customer_age - AVG(customer_age) OVER ()) / 
                     NULLIF(STDDEV(customer_age) OVER (), 0)) > 2.5 THEN 1 
            ELSE 0 
        END AS is_age_outlier,
        
        -- Treatment method used
        CASE 
            WHEN transaction_amount != CASE 
                WHEN transaction_amount < (
                    SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_amount) - 
                           1.5 * (PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_amount) - 
                                  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_amount))
                    FROM sales_transactions
                ) OR transaction_amount > (
                    SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_amount) + 
                           1.5 * (PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_amount) - 
                                  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_amount))
                    FROM sales_transactions
                ) THEN 'IQR_Capped'
                ELSE 'Original'
            END
            THEN 'IQR_Capped'
            ELSE 'Original'
        END AS treatment_applied
    FROM sales_transactions st
    WHERE transaction_amount IS NOT NULL AND customer_age IS NOT NULL
)
SELECT * FROM comprehensive_outlier_treatment;

-- Quality Assessment of Outlier Treatment
WITH treatment_impact AS (
    SELECT 
        'Original Data' AS dataset_stage,
        COUNT(*) AS record_count,
        AVG(original_amount) AS mean_amount,
        STDDEV(original_amount) AS stddev_amount,
        MIN(original_amount) AS min_amount,
        MAX(original_amount) AS max_amount,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY original_amount) AS q1_amount,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY original_amount) AS q3_amount
    FROM ml_ready_sales_data
    
    UNION ALL
    
    SELECT 
        'IQR Treated Data',
        COUNT(*),
        AVG(iqr_treated_amount),
        STDDEV(iqr_treated_amount),
        MIN(iqr_treated_amount),
        MAX(iqr_treated_amount),
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY iqr_treated_amount),
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY iqr_treated_amount)
    FROM ml_ready_sales_data
    
    UNION ALL
    
    SELECT 
        'Log Transformed Data',
        COUNT(log_amount),
        AVG(log_amount),
        STDDEV(log_amount),
        MIN(log_amount),
        MAX(log_amount),
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY log_amount),
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY log_amount)
    FROM ml_ready_sales_data
    WHERE log_amount IS NOT NULL
)
SELECT 
    dataset_stage,
    record_count,
    ROUND(mean_amount, 2) AS mean_amount,
    ROUND(stddev_amount, 2) AS stddev_amount,
    ROUND(min_amount, 2) AS min_amount,
    ROUND(max_amount, 2) AS max_amount,
    ROUND(q1_amount, 2) AS q1_amount,
    ROUND(q3_amount, 2) AS q3_amount,
    ROUND(stddev_amount / NULLIF(mean_amount, 0), 3) AS coefficient_of_variation
FROM treatment_impact
ORDER BY 
    CASE dataset_stage
        WHEN 'Original Data' THEN 1
        WHEN 'IQR Treated Data' THEN 2
        WHEN 'Log Transformed Data' THEN 3
    END;
```

#### Explanation
1. **IQR Method**: Uses quartiles to identify outliers beyond 1.5×IQR from Q1/Q3
2. **Z-Score Analysis**: Identifies outliers based on standard deviations from mean
3. **Modified Z-Score**: Uses median and MAD for robust outlier detection
4. **Percentile-Based**: Uses percentile thresholds for outlier identification
5. **Multivariate Detection**: Considers multiple variables simultaneously
6. **Contextual Analysis**: Identifies outliers within specific business contexts

#### Use Cases
- **Data Preprocessing**: Clean datasets before ML model training
- **Fraud Detection**: Identify unusual transaction patterns
- **Quality Control**: Remove measurement errors and data entry mistakes
- **Customer Segmentation**: Handle extreme customer behavior patterns
- **Financial Analysis**: Detect anomalous trading or spending patterns

#### Best Practices
- Compare multiple outlier detection methods before choosing approach
- Consider business domain knowledge when defining outliers
- Create outlier indicator variables as additional ML features
- Document outlier treatment decisions for model interpretability
- Validate treatment impact on model performance

#### Pitfalls
- **Over-aggressive Removal**: Losing legitimate but extreme observations
- **Method Selection**: Using parametric methods on non-normal data
- **Context Ignorance**: Not considering business-specific outlier definitions
- **Treatment Bias**: Introducing systematic bias through inappropriate treatment
- **Multivariate Neglect**: Only considering univariate outliers

#### Debugging
```sql
-- Outlier treatment validation
SELECT 
    treatment_applied,
    COUNT(*) AS record_count,
    AVG(original_amount) AS avg_original,
    AVG(iqr_treated_amount) AS avg_treated,
    SUM(is_amount_outlier) AS outlier_count
FROM ml_ready_sales_data
GROUP BY treatment_applied;

-- Compare distributions before and after treatment
SELECT 
    'Before Treatment' AS stage,
    MIN(original_amount) AS min_val,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY original_amount) AS q1,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY original_amount) AS median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY original_amount) AS q3,
    MAX(original_amount) AS max_val
FROM ml_ready_sales_data
UNION ALL
SELECT 
    'After Treatment',
    MIN(iqr_treated_amount),
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY iqr_treated_amount),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY iqr_treated_amount),
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY iqr_treated_amount),
    MAX(iqr_treated_amount)
FROM ml_ready_sales_data;
```

#### Optimization
- **Indexing**: Create indexes on numerical columns for efficient statistical calculations
- **Sampling**: Use statistical sampling for outlier detection on very large datasets
- **Parallel Processing**: Compute outlier statistics using window functions for better performance
- **Materialized Views**: Pre-compute outlier boundaries for real-time applications

---

## Question 7

**How can you execute a Machine Learning model stored in a database (such as a SQL Server with R or Python integration)?**

### Answer:

#### Theory
Modern database systems like SQL Server, PostgreSQL, and Oracle provide integrated ML capabilities allowing you to execute machine learning models directly within the database environment. This approach reduces data movement, provides better security, and enables real-time scoring. Popular integrations include SQL Server ML Services (R/Python), PostgreSQL with PL/Python, and Oracle Advanced Analytics.

#### Code Example
```sql
-- SQL Server ML Services with Python Integration
-- Enable external scripts execution
EXEC sp_configure 'external scripts enabled', 1;
RECONFIGURE WITH OVERRIDE;

-- Create a stored procedure that executes a Python ML model
CREATE PROCEDURE PredictCustomerChurn
    @InputData NVARCHAR(MAX)
AS
BEGIN
    DECLARE @Script NVARCHAR(MAX);
    SET @Script = N'
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
with open("C:/Models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler
with open("C:/Models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Process input data
df = pd.DataFrame(InputDataSet)

# Feature engineering
df["avg_monthly_spend"] = df["total_spend"] / df["months_active"]
df["transactions_per_month"] = df["total_transactions"] / df["months_active"]
df["days_since_last_transaction"] = (pd.Timestamp.now() - pd.to_datetime(df["last_transaction_date"])).dt.days

# Select features for prediction
feature_columns = [
    "customer_age", "total_spend", "total_transactions", 
    "avg_monthly_spend", "transactions_per_month", 
    "days_since_last_transaction", "customer_segment_encoded"
]

X = df[feature_columns]

# Scale features
X_scaled = scaler.transform(X)

# Make predictions
predictions = model.predict_proba(X_scaled)[:, 1]  # Probability of churn
prediction_labels = model.predict(X_scaled)

# Prepare output
df["churn_probability"] = predictions
df["churn_prediction"] = prediction_labels
df["risk_category"] = pd.cut(predictions, 
                           bins=[0, 0.3, 0.7, 1.0], 
                           labels=["Low", "Medium", "High"])

OutputDataSet = df[["customer_id", "churn_probability", "churn_prediction", "risk_category"]]
    ';

    EXEC sp_execute_external_script
        @language = N'Python',
        @script = @Script,
        @input_data_1 = @InputData,
        @input_data_1_name = N'InputDataSet';
END;

-- Execute the model on customer data
DECLARE @CustomerData NVARCHAR(MAX) = N'
SELECT 
    customer_id,
    customer_age,
    total_spend,
    total_transactions,
    months_active,
    last_transaction_date,
    CASE customer_segment 
        WHEN ''Premium'' THEN 3
        WHEN ''Standard'' THEN 2
        WHEN ''Basic'' THEN 1
        ELSE 0
    END AS customer_segment_encoded
FROM customer_metrics
WHERE last_transaction_date >= DATEADD(month, -12, GETDATE())';

EXEC PredictCustomerChurn @CustomerData;

-- SQL Server with R Integration
CREATE PROCEDURE CalculateCustomerLifetimeValue
    @CustomerId INT
AS
BEGIN
    DECLARE @RScript NVARCHAR(MAX) = N'
library(survival)
library(dplyr)

# Create customer data frame
customer_data <- InputDataSet

# Calculate CLV using survival analysis
# Fit survival model for customer lifetime
surv_data <- customer_data %>%
    mutate(
        time_to_churn = ifelse(is_churned == 1, days_active, days_active),
        event = is_churned
    )

surv_fit <- survfit(Surv(time_to_churn, event) ~ customer_segment, data = surv_data)

# Predict remaining lifetime
predicted_lifetime <- predict(surv_fit, newdata = customer_data)

# Calculate CLV components
customer_data$predicted_lifetime_days <- predicted_lifetime
customer_data$monthly_value <- customer_data$avg_monthly_spend
customer_data$predicted_lifetime_months <- customer_data$predicted_lifetime_days / 30
customer_data$gross_clv <- customer_data$monthly_value * customer_data$predicted_lifetime_months
customer_data$net_clv <- customer_data$gross_clv * 0.1  # 10% profit margin

# Apply discount rate (assuming 10% annual discount rate)
discount_rate <- 0.10 / 12  # Monthly rate
customer_data$discounted_clv <- customer_data$net_clv / 
                               (1 + discount_rate)^customer_data$predicted_lifetime_months

OutputDataSet <- customer_data[, c("customer_id", "gross_clv", "net_clv", 
                                  "discounted_clv", "predicted_lifetime_months")]
    ';

    EXEC sp_execute_external_script
        @language = N'R',
        @script = @RScript,
        @input_data_1 = N'
            SELECT 
                customer_id,
                customer_segment,
                avg_monthly_spend,
                days_active,
                is_churned
            FROM customer_analytics 
            WHERE customer_id = @CustomerId',
        @params = N'@CustomerId INT',
        @CustomerId = @CustomerId;
END;

-- PostgreSQL with PL/Python Integration
-- Create PL/Python function for price optimization
CREATE OR REPLACE FUNCTION optimize_product_price(
    product_id_input INTEGER,
    current_price DECIMAL,
    sales_data JSON
) 
RETURNS TABLE(
    optimal_price DECIMAL,
    predicted_demand INTEGER,
    predicted_revenue DECIMAL,
    price_elasticity DECIMAL
) AS $$
import json
import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd

# Parse sales data
sales_df = pd.DataFrame(json.loads(sales_data))

# Calculate price elasticity
prices = sales_df['price'].values
quantities = sales_df['quantity_sold'].values

# Log-log regression for elasticity (simplified)
log_prices = np.log(prices)
log_quantities = np.log(quantities)

# Calculate elasticity coefficient
price_elasticity = np.polyfit(log_prices, log_quantities, 1)[0]

# Define demand function
def demand_function(price):
    base_demand = np.mean(quantities)
    return base_demand * (price / current_price) ** price_elasticity

# Define revenue function
def revenue_function(price):
    return price * demand_function(price)

# Find optimal price
result = minimize_scalar(lambda p: -revenue_function(p), 
                        bounds=(current_price * 0.5, current_price * 1.5),
                        method='bounded')

optimal_price = result.x
predicted_demand = int(demand_function(optimal_price))
predicted_revenue = revenue_function(optimal_price)

return [(optimal_price, predicted_demand, predicted_revenue, price_elasticity)]
$$ LANGUAGE plpython3u;

-- Execute price optimization
SELECT * FROM optimize_product_price(
    123,
    29.99,
    '[{"price": 25.99, "quantity_sold": 150}, 
      {"price": 29.99, "quantity_sold": 120}, 
      {"price": 34.99, "quantity_sold": 85}]'::JSON
);

-- Oracle Advanced Analytics with R
-- Create R script for customer segmentation
CREATE OR REPLACE FUNCTION customer_segmentation(
    cursor_input SYS_REFCURSOR
) 
RETURN SYS_REFCURSOR 
PIPELINED 
USING ODMRutil;

-- R implementation in Oracle
BEGIN
  sys.rqScriptCreate('CustomerSegmentation', '
library(cluster)
library(ORE)

# Get data from cursor
customer_data <- ore.pull(cursor_input)

# Prepare features for clustering
features <- customer_data[, c("total_spend", "frequency", "recency", 
                             "avg_order_value", "customer_tenure")]

# Standardize features
features_scaled <- scale(features)

# Perform K-means clustering
set.seed(123)
kmeans_result <- kmeans(features_scaled, centers = 4, nstart = 25)

# Add cluster assignments
customer_data$cluster_id <- kmeans_result$cluster

# Calculate cluster statistics
cluster_stats <- aggregate(. ~ cluster_id, data = customer_data, mean)

# Assign cluster names based on characteristics
cluster_names <- c("High_Value_Loyal", "Medium_Value_Regular", 
                  "Low_Value_Occasional", "At_Risk_Customers")
customer_data$cluster_name <- cluster_names[customer_data$cluster_id]

ore.create(customer_data, "CUSTOMER_SEGMENTS_OUTPUT")
  ');
END;

-- Execute Oracle R script
SELECT * FROM TABLE(customer_segmentation(
    CURSOR(SELECT customer_id, total_spend, frequency, recency, 
                  avg_order_value, customer_tenure 
           FROM customer_metrics)
));

-- Real-time Model Scoring with Streaming Data
CREATE PROCEDURE StreamingFraudDetection
AS
BEGIN
    DECLARE @PythonScript NVARCHAR(MAX) = N'
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load pre-trained fraud detection model
fraud_model = joblib.load("C:/Models/fraud_detection_model.pkl")
feature_scaler = joblib.load("C:/Models/fraud_scaler.pkl")

# Process streaming transaction data
df = pd.DataFrame(InputDataSet)

# Real-time feature engineering
df["hour"] = pd.to_datetime(df["transaction_datetime"]).dt.hour
df["day_of_week"] = pd.to_datetime(df["transaction_datetime"]).dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)

# Amount-based features
df["amount_log"] = np.log1p(df["transaction_amount"])
df["amount_zscore"] = (df["transaction_amount"] - df["transaction_amount"].mean()) / df["transaction_amount"].std()

# Velocity features (requires historical data lookup)
# This would typically involve caching recent transactions
df["transactions_last_hour"] = 1  # Simplified - would query recent history
df["amount_last_hour"] = df["transaction_amount"]  # Simplified

# Geographic features
df["is_international"] = (df["merchant_country"] != df["customer_country"]).astype(int)

# Select features for model
model_features = [
    "transaction_amount", "amount_log", "amount_zscore",
    "hour", "day_of_week", "is_weekend", "is_night",
    "transactions_last_hour", "amount_last_hour", "is_international"
]

X = df[model_features].fillna(0)
X_scaled = feature_scaler.transform(X)

# Generate fraud predictions
fraud_probabilities = fraud_model.predict_proba(X_scaled)[:, 1]
fraud_predictions = fraud_model.predict(X_scaled)

# Create risk categories
df["fraud_probability"] = fraud_probabilities
df["fraud_prediction"] = fraud_predictions
df["risk_level"] = pd.cut(fraud_probabilities, 
                         bins=[0, 0.1, 0.3, 0.7, 1.0],
                         labels=["Very_Low", "Low", "Medium", "High"])

# Flag high-risk transactions for immediate review
df["requires_review"] = (fraud_probabilities > 0.7).astype(int)
df["auto_decline"] = (fraud_probabilities > 0.9).astype(int)

OutputDataSet = df[["transaction_id", "fraud_probability", "fraud_prediction", 
                   "risk_level", "requires_review", "auto_decline"]]
    ';

    -- Process new transactions in real-time
    EXEC sp_execute_external_script
        @language = N'Python',
        @script = @PythonScript,
        @input_data_1 = N'
            SELECT 
                transaction_id,
                customer_id,
                merchant_id,
                transaction_amount,
                transaction_datetime,
                merchant_country,
                customer_country
            FROM transactions_stream 
            WHERE processed = 0
            ORDER BY transaction_datetime DESC';

    -- Update processed flag
    UPDATE transactions_stream 
    SET processed = 1 
    WHERE processed = 0;
END;

-- Automated Model Execution with Scheduling
-- Create job to run model predictions daily
USE msdb;
GO

EXEC dbo.sp_add_job
    @job_name = N'Daily Customer Churn Prediction';

EXEC sp_add_jobstep
    @job_name = N'Daily Customer Churn Prediction',
    @step_name = N'Execute Churn Model',
    @command = N'
        INSERT INTO customer_churn_predictions (customer_id, prediction_date, churn_probability, risk_category)
        SELECT 
            customer_id,
            GETDATE() as prediction_date,
            churn_probability,
            risk_category
        FROM (
            EXEC PredictCustomerChurn @InputData = N''
                SELECT customer_id, customer_age, total_spend, total_transactions, 
                       months_active, last_transaction_date, customer_segment_encoded
                FROM customer_metrics 
                WHERE last_prediction_date < DATEADD(day, -1, GETDATE())
                   OR last_prediction_date IS NULL''
        ) AS predictions;
        
        UPDATE customer_metrics 
        SET last_prediction_date = GETDATE()
        WHERE last_prediction_date < DATEADD(day, -1, GETDATE())
           OR last_prediction_date IS NULL;
    ',
    @database_name = N'CustomerAnalytics';

EXEC dbo.sp_add_schedule
    @schedule_name = N'Daily at 2 AM',
    @freq_type = 4,
    @freq_interval = 1,
    @active_start_time = 020000;

EXEC sp_attach_schedule
    @job_name = N'Daily Customer Churn Prediction',
    @schedule_name = N'Daily at 2 AM';

EXEC dbo.sp_add_jobserver
    @job_name = N'Daily Customer Churn Prediction';

-- Model Performance Monitoring
CREATE VIEW model_performance_metrics AS
WITH prediction_accuracy AS (
    SELECT 
        model_name,
        prediction_date,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN predicted_class = actual_class THEN 1 ELSE 0 END) as correct_predictions,
        AVG(ABS(predicted_probability - actual_outcome)) as mean_absolute_error,
        COUNT(CASE WHEN predicted_class = 1 AND actual_class = 1 THEN 1 END) as true_positives,
        COUNT(CASE WHEN predicted_class = 1 AND actual_class = 0 THEN 1 END) as false_positives,
        COUNT(CASE WHEN predicted_class = 0 AND actual_class = 1 THEN 1 END) as false_negatives,
        COUNT(CASE WHEN predicted_class = 0 AND actual_class = 0 THEN 1 END) as true_negatives
    FROM model_predictions mp
    JOIN actual_outcomes ao ON mp.record_id = ao.record_id 
                           AND mp.prediction_date <= ao.outcome_date
    WHERE ao.outcome_date BETWEEN mp.prediction_date AND DATEADD(day, 30, mp.prediction_date)
    GROUP BY model_name, prediction_date
)
SELECT 
    model_name,
    prediction_date,
    total_predictions,
    ROUND(correct_predictions * 100.0 / total_predictions, 2) as accuracy_percentage,
    ROUND(mean_absolute_error, 4) as mae,
    ROUND(true_positives * 1.0 / NULLIF(true_positives + false_positives, 0), 4) as precision,
    ROUND(true_positives * 1.0 / NULLIF(true_positives + false_negatives, 0), 4) as recall,
    ROUND(2.0 * (true_positives * 1.0 / NULLIF(true_positives + false_positives, 0)) * 
              (true_positives * 1.0 / NULLIF(true_positives + false_negatives, 0)) /
          NULLIF((true_positives * 1.0 / NULLIF(true_positives + false_positives, 0)) + 
                 (true_positives * 1.0 / NULLIF(true_positives + false_negatives, 0)), 0), 4) as f1_score
FROM prediction_accuracy;

-- Query model performance trends
SELECT 
    model_name,
    prediction_date,
    accuracy_percentage,
    precision,
    recall,
    f1_score,
    LAG(accuracy_percentage) OVER (PARTITION BY model_name ORDER BY prediction_date) as prev_accuracy,
    accuracy_percentage - LAG(accuracy_percentage) OVER (PARTITION BY model_name ORDER BY prediction_date) as accuracy_change
FROM model_performance_metrics
WHERE prediction_date >= DATEADD(month, -3, GETDATE())
ORDER BY model_name, prediction_date DESC;
```

#### Explanation
1. **sp_execute_external_script**: SQL Server's interface for executing Python/R code
2. **PL/Python**: PostgreSQL's procedural language for Python integration
3. **Oracle R Enterprise**: Oracle's R integration for advanced analytics
4. **Model Serialization**: Loading pre-trained models (pickle, joblib) within database
5. **Real-time Scoring**: Processing streaming data with immediate model predictions

#### Use Cases
- **Real-time Fraud Detection**: Score transactions as they occur
- **Customer Churn Prevention**: Daily scoring for proactive retention
- **Price Optimization**: Dynamic pricing based on demand models
- **Recommendation Systems**: Generate personalized recommendations in-database
- **Risk Assessment**: Credit scoring and loan approval automation

#### Best Practices
- Store serialized models in accessible file systems or database BLOBs
- Implement error handling and logging within stored procedures
- Monitor model performance and accuracy over time
- Use connection pooling for high-throughput scoring scenarios
- Implement model versioning for A/B testing and rollbacks

#### Pitfalls
- **Performance Impact**: Complex models can slow database operations
- **Resource Contention**: ML processing competing with transactional workloads
- **Model Drift**: Models becoming stale without regular retraining
- **Security Concerns**: External script execution requires careful permission management
- **Dependency Management**: Python/R package dependencies need maintenance

#### Debugging
```sql
-- Check external script configuration
SELECT name, value FROM sys.configurations WHERE name = 'external scripts enabled';

-- Monitor model execution performance
SELECT 
    session_id,
    start_time,
    end_time,
    DATEDIFF(second, start_time, end_time) as duration_seconds,
    external_script_name
FROM sys.dm_external_script_requests
WHERE start_time >= DATEADD(hour, -24, GETDATE())
ORDER BY start_time DESC;

-- Validate model predictions
SELECT 
    model_name,
    AVG(prediction_confidence) as avg_confidence,
    COUNT(CASE WHEN prediction_confidence < 0.6 THEN 1 END) as low_confidence_predictions
FROM model_predictions
WHERE prediction_date = CAST(GETDATE() AS DATE)
GROUP BY model_name;
```

#### Optimization
- **Resource Pools**: Configure resource pools for ML workloads
- **Parallel Execution**: Process batches in parallel for large datasets
- **Model Caching**: Cache frequently used models in memory
- **Asynchronous Processing**: Use message queues for non-real-time scoring

---

## Question 8

**Can you update a Machine Learning model directly from SQL? If so, how might you do it?**

### Answer:

#### Theory
Yes, you can update machine learning models directly from SQL using several approaches: incremental learning algorithms, model retraining with new data, online learning techniques, and hybrid approaches. Modern databases with ML integration support model updating through stored procedures, external scripts, and specialized ML functions. The choice of approach depends on the algorithm type, data volume, and real-time requirements.

#### Code Example
```sql
-- Method 1: Incremental Model Updates with SQL Server ML Services
CREATE PROCEDURE UpdateChurnModelIncremental
    @NewDataStartDate DATE,
    @ModelVersion VARCHAR(50) = NULL
AS
BEGIN
    DECLARE @PythonScript NVARCHAR(MAX) = N'
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime

# Load existing model and scaler
try:
    with open("C:/Models/churn_model_sgd.pkl", "rb") as f:
        model = pickle.load(f)
    with open("C:/Models/churn_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model_exists = True
    print("Loaded existing model")
except FileNotFoundError:
    model = SGDClassifier(random_state=42, learning_rate="constant", eta0=0.01)
    scaler = StandardScaler()
    model_exists = False
    print("Creating new model")

# Get new training data
df = pd.DataFrame(InputDataSet)

# Feature engineering
df["avg_monthly_spend"] = df["total_spend"] / np.maximum(df["months_active"], 1)
df["transactions_per_month"] = df["total_transactions"] / np.maximum(df["months_active"], 1)
df["recency_score"] = np.where(df["days_since_last_transaction"] <= 30, 4,
                      np.where(df["days_since_last_transaction"] <= 90, 3,
                      np.where(df["days_since_last_transaction"] <= 180, 2, 1)))

# Customer segment encoding
segment_mapping = {"Premium": 3, "Standard": 2, "Basic": 1}
df["customer_segment_encoded"] = df["customer_segment"].map(segment_mapping).fillna(0)

# Prepare features
feature_columns = [
    "customer_age", "total_spend", "total_transactions", "months_active",
    "avg_monthly_spend", "transactions_per_month", "recency_score",
    "customer_segment_encoded", "days_since_last_transaction"
]

X = df[feature_columns].fillna(0)
y = df["is_churned"]

# Update scaler incrementally or fit on new data
if model_exists:
    # For incremental learning, we can use partial_fit if available
    X_scaled = scaler.transform(X)
    
    # Use partial_fit for incremental learning
    model.partial_fit(X_scaled, y)
    
    print(f"Model updated with {len(X)} new samples")
else:
    # Initial training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit scaler on training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train initial model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Initial model accuracy: {accuracy:.3f}")

# Save updated model and scaler
with open("C:/Models/churn_model_sgd.pkl", "wb") as f:
    pickle.dump(model, f)
with open("C:/Models/churn_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Generate model metadata
model_metadata = {
    "model_version": model_version if model_version else datetime.now().strftime("%Y%m%d_%H%M%S"),
    "update_date": datetime.now().isoformat(),
    "training_samples": len(X),
    "feature_columns": feature_columns,
    "model_type": "SGDClassifier",
    "model_exists_before_update": model_exists
}

# Output results
OutputDataSet = pd.DataFrame([model_metadata])
    ';

    EXEC sp_execute_external_script
        @language = N'Python',
        @script = @PythonScript,
        @input_data_1 = N'
            SELECT 
                customer_id,
                customer_age,
                customer_segment,
                total_spend,
                total_transactions,
                months_active,
                days_since_last_transaction,
                is_churned
            FROM customer_training_data 
            WHERE created_date >= @NewDataStartDate
              AND is_churned IS NOT NULL',
        @params = N'@NewDataStartDate DATE, @ModelVersion VARCHAR(50)',
        @NewDataStartDate = @NewDataStartDate,
        @ModelVersion = @ModelVersion;

    -- Log the model update
    INSERT INTO model_update_log (model_name, update_date, data_start_date, update_type)
    VALUES ('churn_model_sgd', GETDATE(), @NewDataStartDate, 'incremental');
END;

-- Execute incremental update
EXEC UpdateChurnModelIncremental 
    @NewDataStartDate = '2024-01-01',
    @ModelVersion = 'v2.1_incremental';

-- Method 2: Online Learning with Streaming Updates
CREATE PROCEDURE OnlineLearningUpdate
AS
BEGIN
    DECLARE @PythonScript NVARCHAR(MAX) = N'
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# Load existing online learning model
try:
    with open("C:/Models/price_optimization_online.pkl", "rb") as f:
        model = pickle.load(f)
    with open("C:/Models/price_scaler_online.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Loaded existing online model")
except FileNotFoundError:
    model = PassiveAggressiveRegressor(random_state=42)
    scaler = MinMaxScaler()
    print("Creating new online model")

# Process streaming data batch
df = pd.DataFrame(InputDataSet)

if len(df) > 0:
    # Feature engineering
    df["price_per_unit"] = df["total_price"] / np.maximum(df["quantity"], 1)
    df["discount_rate"] = 1 - (df["sale_price"] / df["list_price"])
    df["hour_of_day"] = pd.to_datetime(df["sale_datetime"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["sale_datetime"]).dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Prepare features and target
    feature_columns = ["list_price", "discount_rate", "hour_of_day", "day_of_week", "is_weekend"]
    X = df[feature_columns].fillna(0)
    y = df["quantity"]  # Target: quantity sold
    
    # Scale features
    X_scaled = scaler.fit_transform(X) if not hasattr(model, "coef_") else scaler.transform(X)
    
    # Perform online learning (sample by sample)
    for i in range(len(X_scaled)):
        model.partial_fit([X_scaled[i]], [y.iloc[i]])
    
    # Save updated model
    with open("C:/Models/price_optimization_online.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("C:/Models/price_scaler_online.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"Model updated with {len(X)} new samples online")
    
    # Generate predictions for validation
    predictions = model.predict(X_scaled)
    df["predicted_quantity"] = predictions
    df["prediction_error"] = np.abs(df["quantity"] - predictions)
    
    OutputDataSet = df[["product_id", "sale_datetime", "quantity", "predicted_quantity", "prediction_error"]]
else:
    OutputDataSet = pd.DataFrame()
    ';

    EXEC sp_execute_external_script
        @language = N'Python',
        @script = @PythonScript,
        @input_data_1 = N'
            SELECT 
                product_id,
                sale_datetime,
                list_price,
                sale_price,
                total_price,
                quantity
            FROM sales_stream 
            WHERE processed_for_learning = 0
            ORDER BY sale_datetime';

    -- Mark data as processed
    UPDATE sales_stream 
    SET processed_for_learning = 1 
    WHERE processed_for_learning = 0;
END;

-- Method 3: Model Retraining with Validation
CREATE PROCEDURE RetrainModelWithValidation
    @ModelName VARCHAR(100),
    @RetrainingThreshold DECIMAL(5,4) = 0.05  -- Retrain if accuracy drops by 5%
AS
BEGIN
    DECLARE @CurrentAccuracy DECIMAL(5,4);
    DECLARE @BaselineAccuracy DECIMAL(5,4);
    DECLARE @ShouldRetrain BIT = 0;

    -- Check current model performance
    SELECT @CurrentAccuracy = accuracy_score
    FROM model_performance_log
    WHERE model_name = @ModelName
      AND evaluation_date = (SELECT MAX(evaluation_date) 
                             FROM model_performance_log 
                             WHERE model_name = @ModelName);

    -- Get baseline accuracy
    SELECT @BaselineAccuracy = accuracy_score
    FROM model_baseline_metrics
    WHERE model_name = @ModelName;

    -- Determine if retraining is needed
    IF @CurrentAccuracy < (@BaselineAccuracy - @RetrainingThreshold)
        SET @ShouldRetrain = 1;

    IF @ShouldRetrain = 1
    BEGIN
        DECLARE @PythonScript NVARCHAR(MAX) = N'
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from datetime import datetime

# Load current model for comparison
try:
    with open("C:/Models/" + model_name + ".pkl", "rb") as f:
        old_model = pickle.load(f)
    with open("C:/Models/" + model_name + "_scaler.pkl", "rb") as f:
        old_scaler = pickle.load(f)
    print("Loaded existing model for comparison")
except:
    old_model = None
    old_scaler = None

# Prepare training data
df = pd.DataFrame(InputDataSet)

# Feature engineering
df["feature_1_log"] = np.log1p(df["feature_1"])
df["feature_2_squared"] = df["feature_2"] ** 2
df["feature_interaction"] = df["feature_1"] * df["feature_2"]

# Prepare features and target
feature_columns = ["feature_1", "feature_2", "feature_3", "feature_1_log", 
                  "feature_2_squared", "feature_interaction"]
X = df[feature_columns].fillna(0)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Train new model
new_scaler = StandardScaler()
X_train_scaled = new_scaler.fit_transform(X_train)
X_test_scaled = new_scaler.transform(X_test)

new_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
new_model.fit(X_train_scaled, y_train)

# Evaluate new model
y_pred_new = new_model.predict(X_test_scaled)
new_accuracy = accuracy_score(y_test, y_pred_new)

# Compare with old model if available
old_accuracy = 0
if old_model is not None and old_scaler is not None:
    try:
        X_test_old_scaled = old_scaler.transform(X_test)
        y_pred_old = old_model.predict(X_test_old_scaled)
        old_accuracy = accuracy_score(y_test, y_pred_old)
    except:
        old_accuracy = 0

# Cross-validation score
cv_scores = cross_val_score(new_model, X_train_scaled, y_train, cv=5)
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

# Decide whether to deploy new model
deploy_new_model = new_accuracy > old_accuracy

if deploy_new_model:
    # Save new model
    with open("C:/Models/" + model_name + ".pkl", "wb") as f:
        pickle.dump(new_model, f)
    with open("C:/Models/" + model_name + "_scaler.pkl", "wb") as f:
        pickle.dump(new_scaler, f)
    
    # Backup old model
    if old_model is not None:
        with open("C:/Models/" + model_name + "_backup.pkl", "wb") as f:
            pickle.dump(old_model, f)

# Generate detailed metrics
feature_importance = new_model.feature_importances_
feature_importance_dict = dict(zip(feature_columns, feature_importance))

results = {
    "model_name": model_name,
    "retrain_date": datetime.now().isoformat(),
    "old_accuracy": float(old_accuracy),
    "new_accuracy": float(new_accuracy),
    "cv_mean_accuracy": float(cv_mean),
    "cv_std_accuracy": float(cv_std),
    "improvement": float(new_accuracy - old_accuracy),
    "deployed": deploy_new_model,
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "feature_importance": feature_importance_dict
}

OutputDataSet = pd.DataFrame([results])
        ';

        EXEC sp_execute_external_script
            @language = N'Python',
            @script = @PythonScript,
            @input_data_1 = N'
                SELECT *
                FROM model_training_data 
                WHERE created_date >= DATEADD(month, -6, GETDATE())',
            @params = N'@ModelName VARCHAR(100)',
            @ModelName = @ModelName;

        -- Log retraining activity
        INSERT INTO model_retrain_log (model_name, retrain_date, reason, triggered_by)
        VALUES (@ModelName, GETDATE(), 'Performance degradation', 'automated_threshold');

        PRINT 'Model retrained due to performance degradation';
    END
    ELSE
    BEGIN
        PRINT 'Model performance within acceptable range. No retraining needed.';
    END
END;

-- Method 4: A/B Testing Framework for Model Updates
CREATE PROCEDURE DeployModelABTest
    @ModelNameA VARCHAR(100),
    @ModelNameB VARCHAR(100),
    @TrafficSplit DECIMAL(3,2) = 0.5  -- 50% split by default
AS
BEGIN
    -- Create A/B test configuration
    INSERT INTO model_ab_tests (
        test_name, model_a, model_b, traffic_split, start_date, status
    )
    VALUES (
        CONCAT(@ModelNameA, '_vs_', @ModelNameB, '_', FORMAT(GETDATE(), 'yyyyMMdd')),
        @ModelNameA, @ModelNameB, @TrafficSplit, GETDATE(), 'active'
    );

    DECLARE @TestId INT = SCOPE_IDENTITY();

    -- Create routing logic for A/B testing
    DECLARE @PythonScript NVARCHAR(MAX) = N'
import pandas as pd
import pickle
import numpy as np
import hashlib

# Load both models
with open("C:/Models/" + model_a + ".pkl", "rb") as f:
    model_a_obj = pickle.load(f)
with open("C:/Models/" + model_a + "_scaler.pkl", "rb") as f:
    scaler_a = pickle.load(f)

with open("C:/Models/" + model_b + ".pkl", "rb") as f:
    model_b_obj = pickle.load(f)
with open("C:/Models/" + model_b + "_scaler.pkl", "rb") as f:
    scaler_b = pickle.load(f)

# Process input data
df = pd.DataFrame(InputDataSet)

# Determine model assignment based on customer_id hash
def assign_model(customer_id, split_ratio):
    hash_val = int(hashlib.md5(str(customer_id).encode()).hexdigest(), 16)
    return "A" if (hash_val % 100) / 100 < split_ratio else "B"

df["model_assignment"] = df["customer_id"].apply(
    lambda x: assign_model(x, traffic_split)
)

# Prepare features
feature_columns = ["feature_1", "feature_2", "feature_3"]
X = df[feature_columns].fillna(0)

# Make predictions with assigned models
predictions_a = model_a_obj.predict_proba(scaler_a.transform(X))[:, 1]
predictions_b = model_b_obj.predict_proba(scaler_b.transform(X))[:, 1]

# Assign final predictions based on model assignment
df["prediction"] = np.where(
    df["model_assignment"] == "A", 
    predictions_a, 
    predictions_b
)

df["model_used"] = np.where(
    df["model_assignment"] == "A", 
    model_a, 
    model_b
)

df["test_id"] = test_id

OutputDataSet = df[["customer_id", "prediction", "model_used", "model_assignment", "test_id"]]
    ';

    EXEC sp_execute_external_script
        @language = N'Python',
        @script = @PythonScript,
        @input_data_1 = N'
            SELECT customer_id, feature_1, feature_2, feature_3
            FROM prediction_requests 
            WHERE processed = 0',
        @params = N'@ModelNameA VARCHAR(100), @ModelNameB VARCHAR(100), @TrafficSplit DECIMAL(3,2), @TestId INT',
        @ModelNameA = @ModelNameA,
        @ModelNameB = @ModelNameB,
        @TrafficSplit = @TrafficSplit,
        @TestId = @TestId;

    PRINT CONCAT('A/B test deployed for models: ', @ModelNameA, ' vs ', @ModelNameB);
END;

-- Method 5: Automated Model Update Pipeline
CREATE PROCEDURE AutomatedModelUpdatePipeline
AS
BEGIN
    DECLARE @ModelList TABLE (model_name VARCHAR(100), last_update DATE, update_frequency_days INT);
    
    INSERT INTO @ModelList
    SELECT model_name, last_update, update_frequency_days
    FROM model_configuration
    WHERE is_active = 1;

    DECLARE @ModelName VARCHAR(100);
    DECLARE @LastUpdate DATE;
    DECLARE @UpdateFrequency INT;
    DECLARE @DaysSinceUpdate INT;

    DECLARE model_cursor CURSOR FOR
    SELECT model_name, last_update, update_frequency_days FROM @ModelList;

    OPEN model_cursor;
    FETCH NEXT FROM model_cursor INTO @ModelName, @LastUpdate, @UpdateFrequency;

    WHILE @@FETCH_STATUS = 0
    BEGIN
        SET @DaysSinceUpdate = DATEDIFF(day, @LastUpdate, GETDATE());

        IF @DaysSinceUpdate >= @UpdateFrequency
        BEGIN
            PRINT CONCAT('Updating model: ', @ModelName);

            -- Check data availability
            DECLARE @NewDataCount INT;
            SELECT @NewDataCount = COUNT(*)
            FROM model_training_data
            WHERE model_name = @ModelName
              AND created_date > @LastUpdate;

            IF @NewDataCount >= 100  -- Minimum threshold for update
            BEGIN
                -- Execute incremental update
                EXEC UpdateChurnModelIncremental 
                    @NewDataStartDate = @LastUpdate,
                    @ModelVersion = CONCAT(@ModelName, '_auto_', FORMAT(GETDATE(), 'yyyyMMdd'));

                -- Update configuration
                UPDATE model_configuration
                SET last_update = GETDATE()
                WHERE model_name = @ModelName;

                PRINT CONCAT('Successfully updated: ', @ModelName);
            END
            ELSE
            BEGIN
                PRINT CONCAT('Insufficient new data for: ', @ModelName, ' (', @NewDataCount, ' samples)');
            END
        END

        FETCH NEXT FROM model_cursor INTO @ModelName, @LastUpdate, @UpdateFrequency;
    END

    CLOSE model_cursor;
    DEALLOCATE model_cursor;
END;

-- Schedule automated updates
USE msdb;
EXEC dbo.sp_add_job
    @job_name = N'Automated Model Updates';

EXEC sp_add_jobstep
    @job_name = N'Automated Model Updates',
    @step_name = N'Run Update Pipeline',
    @command = N'EXEC AutomatedModelUpdatePipeline',
    @database_name = N'MLDatabase';

EXEC dbo.sp_add_schedule
    @schedule_name = N'Weekly Model Updates',
    @freq_type = 4,
    @freq_interval = 7;

EXEC sp_attach_schedule
    @job_name = N'Automated Model Updates',
    @schedule_name = N'Weekly Model Updates';

-- Model Rollback Capability
CREATE PROCEDURE RollbackModel
    @ModelName VARCHAR(100),
    @RollbackReason VARCHAR(500)
AS
BEGIN
    DECLARE @BackupExists BIT = 0;

    -- Check if backup exists
    DECLARE @PythonScript NVARCHAR(MAX) = N'
import os
import shutil
from datetime import datetime

model_path = "C:/Models/" + model_name + ".pkl"
backup_path = "C:/Models/" + model_name + "_backup.pkl"
scaler_path = "C:/Models/" + model_name + "_scaler.pkl"
backup_scaler_path = "C:/Models/" + model_name + "_scaler_backup.pkl"

backup_exists = os.path.exists(backup_path)

if backup_exists:
    # Create timestamp backup of current model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_backup_path = f"C:/Models/{model_name}_rollback_{timestamp}.pkl"
    
    if os.path.exists(model_path):
        shutil.copy2(model_path, current_backup_path)
    
    # Restore from backup
    shutil.copy2(backup_path, model_path)
    if os.path.exists(backup_scaler_path):
        shutil.copy2(backup_scaler_path, scaler_path)
    
    print(f"Model {model_name} rolled back successfully")
    status = "success"
else:
    print(f"No backup found for model {model_name}")
    status = "failed"

import pandas as pd
OutputDataSet = pd.DataFrame([{"model_name": model_name, "rollback_status": status, "backup_exists": backup_exists}])
    ';

    EXEC sp_execute_external_script
        @language = N'Python',
        @script = @PythonScript,
        @params = N'@ModelName VARCHAR(100)',
        @ModelName = @ModelName;

    -- Log rollback
    INSERT INTO model_rollback_log (model_name, rollback_date, reason, performed_by)
    VALUES (@ModelName, GETDATE(), @RollbackReason, SYSTEM_USER);
END;

-- Monitor Model Update Success
CREATE VIEW model_update_summary AS
SELECT 
    model_name,
    COUNT(*) as total_updates,
    MAX(update_date) as last_update,
    AVG(CASE WHEN update_status = 'success' THEN 1.0 ELSE 0.0 END) as success_rate,
    SUM(CASE WHEN update_date >= DATEADD(day, -30, GETDATE()) THEN 1 ELSE 0 END) as updates_last_30_days
FROM model_update_log
GROUP BY model_name;

SELECT * FROM model_update_summary ORDER BY last_update DESC;
```

#### Explanation
1. **Incremental Learning**: Using algorithms like SGDClassifier that support partial_fit()
2. **Online Learning**: Real-time model updates with streaming data
3. **Model Retraining**: Complete retraining with performance validation
4. **A/B Testing**: Gradual deployment of updated models
5. **Automated Pipelines**: Scheduled and trigger-based model updates

#### Use Cases
- **Fraud Detection**: Continuous learning from new fraud patterns
- **Recommendation Systems**: Adapting to changing user preferences
- **Price Optimization**: Updating demand models with market changes
- **Customer Behavior**: Learning from evolving customer patterns
- **Risk Assessment**: Incorporating new risk factors and market conditions

#### Best Practices
- Implement model versioning and backup strategies
- Monitor model performance before and after updates
- Use incremental learning for large datasets
- Validate updates before deployment
- Maintain audit trails of all model changes

#### Pitfalls
- **Model Drift**: Gradual performance degradation without detection
- **Catastrophic Forgetting**: Losing previously learned patterns
- **Data Quality**: Poor quality new data degrading model performance
- **Overfitting**: Models becoming too specific to recent data
- **Resource Constraints**: Updates consuming excessive computational resources

#### Debugging
```sql
-- Check model update history
SELECT 
    model_name,
    update_date,
    update_type,
    data_start_date,
    DATEDIFF(day, data_start_date, update_date) as data_window_days
FROM model_update_log
WHERE update_date >= DATEADD(month, -3, GETDATE())
ORDER BY update_date DESC;

-- Monitor update performance
SELECT 
    model_name,
    AVG(CASE WHEN accuracy_after > accuracy_before THEN 1.0 ELSE 0.0 END) as improvement_rate,
    AVG(accuracy_after - accuracy_before) as avg_accuracy_change
FROM model_update_performance
GROUP BY model_name;
```

#### Optimization
- **Batch Updates**: Process multiple updates together for efficiency
- **Smart Scheduling**: Update models during low-traffic periods
- **Resource Management**: Allocate dedicated resources for model updates
- **Parallel Processing**: Update multiple models simultaneously when possible

---

## Question 9

**What strategies can be used to efficiently update a large SQL-based Machine Learning model?**

**Answer:**

### Theory

Efficiently updating large SQL-based ML models requires balancing computational resources, data consistency, and model performance. The key challenges include handling large datasets, minimizing downtime, ensuring data integrity, and optimizing computational resources.

### Implementation Strategies

#### 1. Incremental Learning Approach

```sql
-- Create incremental update framework
CREATE TABLE model_versions (
    version_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_data_hash VARCHAR(64),
    performance_metrics JSONB,
    is_active BOOLEAN DEFAULT FALSE
);

-- Incremental data processing
CREATE OR REPLACE FUNCTION incremental_model_update(
    p_model_name VARCHAR,
    p_batch_size INT DEFAULT 10000
)
RETURNS VOID AS $$
DECLARE
    last_processed_id INT;
    batch_count INT := 0;
BEGIN
    -- Get last processed record
    SELECT COALESCE(MAX(last_processed_id), 0) 
    INTO last_processed_id
    FROM model_training_log 
    WHERE model_name = p_model_name;
    
    -- Process data in batches
    WHILE EXISTS (
        SELECT 1 FROM training_data 
        WHERE id > last_processed_id 
        LIMIT 1
    ) LOOP
        -- Update model with batch
        WITH batch_data AS (
            SELECT * FROM training_data 
            WHERE id > last_processed_id 
            ORDER BY id 
            LIMIT p_batch_size
        )
        INSERT INTO model_coefficients (model_name, feature_name, coefficient, version)
        SELECT 
            p_model_name,
            feature_name,
            -- Apply incremental learning algorithm
            CASE 
                WHEN existing_coef.coefficient IS NULL THEN new_coefficient
                ELSE existing_coef.coefficient * 0.9 + new_coefficient * 0.1
            END,
            (SELECT MAX(version_id) FROM model_versions WHERE model_name = p_model_name)
        FROM (
            SELECT 
                unnest(ARRAY['feature1', 'feature2', 'feature3']) AS feature_name,
                unnest(calculate_coefficients(batch_data.*)) AS new_coefficient
            FROM batch_data
        ) new_coefs
        LEFT JOIN model_coefficients existing_coef 
            ON existing_coef.feature_name = new_coefs.feature_name
            AND existing_coef.model_name = p_model_name;
        
        -- Update last processed ID
        UPDATE model_training_log 
        SET last_processed_id = (
            SELECT MAX(id) FROM training_data 
            WHERE id <= last_processed_id + p_batch_size
        )
        WHERE model_name = p_model_name;
        
        batch_count := batch_count + 1;
        
        -- Commit every 10 batches
        IF batch_count % 10 = 0 THEN
            COMMIT;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

#### 2. Parallel Processing Strategy

```sql
-- Partition-based parallel updates
CREATE OR REPLACE FUNCTION parallel_model_update(
    p_model_name VARCHAR,
    p_num_partitions INT DEFAULT 4
)
RETURNS VOID AS $$
DECLARE
    partition_size INT;
    i INT;
BEGIN
    -- Calculate partition size
    SELECT CEIL(COUNT(*)::FLOAT / p_num_partitions) 
    INTO partition_size
    FROM training_data;
    
    -- Create parallel update jobs
    FOR i IN 1..p_num_partitions LOOP
        -- Use background workers or stored procedures
        PERFORM pg_background_launch(
            format('
                SELECT update_model_partition(''%s'', %s, %s)',
                p_model_name,
                (i-1) * partition_size,
                i * partition_size
            )
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Partition update function
CREATE OR REPLACE FUNCTION update_model_partition(
    p_model_name VARCHAR,
    p_start_row INT,
    p_end_row INT
)
RETURNS VOID AS $$
BEGIN
    WITH partition_data AS (
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (ORDER BY id) as rn
            FROM training_data
        ) t
        WHERE rn BETWEEN p_start_row AND p_end_row
    )
    -- Update model coefficients for this partition
    INSERT INTO temp_model_updates (model_name, feature_name, partial_coefficient)
    SELECT 
        p_model_name,
        feature_name,
        SUM(feature_value * target_value) / COUNT(*)
    FROM partition_data
    GROUP BY feature_name;
END;
$$ LANGUAGE plpgsql;
```

#### 3. Shadow Model Strategy

```sql
-- Create shadow model for safe updates
CREATE OR REPLACE FUNCTION create_shadow_model(
    p_original_model VARCHAR,
    p_shadow_version VARCHAR
)
RETURNS VOID AS $$
BEGIN
    -- Copy existing model structure
    INSERT INTO model_coefficients (model_name, feature_name, coefficient, version)
    SELECT 
        p_shadow_version,
        feature_name,
        coefficient,
        p_shadow_version
    FROM model_coefficients 
    WHERE model_name = p_original_model 
    AND is_active = TRUE;
    
    -- Update shadow model with new data
    WITH new_training_data AS (
        SELECT * FROM training_data 
        WHERE created_at > (
            SELECT last_training_date 
            FROM model_metadata 
            WHERE model_name = p_original_model
        )
    )
    UPDATE model_coefficients 
    SET coefficient = (
        SELECT calculate_updated_coefficient(
            coefficient, 
            new_data.feature_values
        )
        FROM new_training_data new_data
        WHERE new_data.feature_name = model_coefficients.feature_name
    )
    WHERE model_name = p_shadow_version;
END;
$$ LANGUAGE plpgsql;

-- Atomic model swap
CREATE OR REPLACE FUNCTION swap_model_versions(
    p_model_name VARCHAR,
    p_new_version VARCHAR
)
RETURNS VOID AS $$
BEGIN
    -- Validate new model performance
    IF (SELECT validate_model_performance(p_new_version)) THEN
        BEGIN
            -- Atomic swap
            UPDATE model_coefficients 
            SET is_active = FALSE 
            WHERE model_name = p_model_name AND is_active = TRUE;
            
            UPDATE model_coefficients 
            SET is_active = TRUE, model_name = p_model_name
            WHERE model_name = p_new_version;
            
            -- Log the swap
            INSERT INTO model_deployment_log (
                model_name, old_version, new_version, swapped_at
            ) VALUES (
                p_model_name, 
                (SELECT version FROM model_coefficients WHERE model_name = p_model_name AND is_active = FALSE LIMIT 1),
                p_new_version,
                CURRENT_TIMESTAMP
            );
        EXCEPTION WHEN OTHERS THEN
            RAISE EXCEPTION 'Model swap failed: %', SQLERRM;
        END;
    ELSE
        RAISE EXCEPTION 'New model failed validation';
    END IF;
END;
$$ LANGUAGE plpgsql;
```

#### 4. Memory-Optimized Updates

```sql
-- Memory-efficient batch processing
CREATE OR REPLACE FUNCTION memory_efficient_update(
    p_model_name VARCHAR,
    p_memory_limit_mb INT DEFAULT 500
)
RETURNS VOID AS $$
DECLARE
    batch_size INT;
    current_batch INT := 0;
    total_rows INT;
BEGIN
    -- Calculate optimal batch size based on memory limit
    SELECT COUNT(*) INTO total_rows FROM training_data;
    batch_size := LEAST(
        p_memory_limit_mb * 1000, -- Approximate rows per MB
        total_rows / 10 -- At least 10 batches
    );
    
    -- Create temporary aggregation table
    CREATE TEMP TABLE IF NOT EXISTS temp_feature_aggregates (
        feature_name VARCHAR(100),
        sum_product NUMERIC,
        sum_squares NUMERIC,
        count_values INT,
        PRIMARY KEY (feature_name)
    );
    
    -- Process in memory-efficient batches
    FOR current_batch IN 0..(total_rows / batch_size) LOOP
        -- Aggregate features in batch
        INSERT INTO temp_feature_aggregates (
            feature_name, sum_product, sum_squares, count_values
        )
        SELECT 
            feature_name,
            SUM(feature_value * target_value),
            SUM(feature_value * feature_value),
            COUNT(*)
        FROM (
            SELECT * FROM training_data 
            OFFSET current_batch * batch_size 
            LIMIT batch_size
        ) batch_data
        CROSS JOIN LATERAL unnest_features(batch_data.*) AS t(feature_name, feature_value, target_value)
        GROUP BY feature_name
        ON CONFLICT (feature_name) DO UPDATE SET
            sum_product = temp_feature_aggregates.sum_product + EXCLUDED.sum_product,
            sum_squares = temp_feature_aggregates.sum_squares + EXCLUDED.sum_squares,
            count_values = temp_feature_aggregates.count_values + EXCLUDED.count_values;
        
        -- Clear memory periodically
        IF current_batch % 5 = 0 THEN
            VACUUM temp_feature_aggregates;
        END IF;
    END LOOP;
    
    -- Update model coefficients from aggregates
    UPDATE model_coefficients mc
    SET coefficient = (
        SELECT 
            CASE 
                WHEN tfa.sum_squares > 0 
                THEN tfa.sum_product / tfa.sum_squares
                ELSE mc.coefficient
            END
        FROM temp_feature_aggregates tfa
        WHERE tfa.feature_name = mc.feature_name
    )
    WHERE mc.model_name = p_model_name;
    
    DROP TABLE temp_feature_aggregates;
END;
$$ LANGUAGE plpgsql;
```

### Real-World Applications

1. **E-commerce Recommendation Systems**: Update product recommendation models with new user interactions
2. **Financial Risk Models**: Incorporate new transaction data for fraud detection
3. **Healthcare Predictive Models**: Update patient outcome predictions with new clinical data
4. **Marketing Attribution Models**: Refresh attribution coefficients with new campaign data

### Best Practices

1. **Data Validation**: Always validate new data before model updates
2. **Backup Strategies**: Maintain model version history for rollback capability
3. **Performance Monitoring**: Track update performance and optimize batch sizes
4. **Testing Pipelines**: Implement A/B testing for model updates
5. **Resource Management**: Monitor CPU, memory, and I/O during updates

### Common Pitfalls

1. **Lock Contention**: Long-running updates can block read operations
2. **Memory Overflow**: Processing too much data at once can cause out-of-memory errors
3. **Data Drift**: Not accounting for concept drift in incremental updates
4. **Inconsistent States**: Partial updates due to failures can corrupt models
5. **Performance Degradation**: Poorly optimized updates can impact system performance

### Debugging and Optimization

```sql
-- Monitor update performance
CREATE VIEW model_update_performance AS
SELECT 
    model_name,
    update_method,
    avg_duration,
    success_rate,
    avg_memory_usage,
    last_update
FROM (
    SELECT 
        model_name,
        update_method,
        AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_duration,
        COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END)::FLOAT / COUNT(*) as success_rate,
        AVG(peak_memory_mb) as avg_memory_usage,
        MAX(start_time) as last_update
    FROM model_update_log
    GROUP BY model_name, update_method
) stats;

-- Optimize batch sizes based on performance
CREATE OR REPLACE FUNCTION optimize_batch_size(
    p_model_name VARCHAR
)
RETURNS INT AS $$
DECLARE
    optimal_size INT;
BEGIN
    SELECT 
        CASE 
            WHEN avg_duration < 30 THEN batch_size * 2  -- Too fast, increase
            WHEN avg_duration > 300 THEN batch_size / 2  -- Too slow, decrease
            ELSE batch_size  -- Just right
        END
    INTO optimal_size
    FROM model_update_performance 
    WHERE model_name = p_model_name;
    
    RETURN COALESCE(optimal_size, 10000);  -- Default fallback
END;
$$ LANGUAGE plpgsql;
```

### Performance Considerations

- **Incremental Learning**: 60-80% faster than full retraining
- **Parallel Processing**: Can achieve 3-4x speedup with proper partitioning
- **Memory Optimization**: Reduces memory usage by 70-90%
- **Shadow Updates**: Eliminates downtime but doubles storage requirements

---

## Question 10

**How do you ensure the consistency and reliability of SQL data used for Machine Learning?**

**Answer:**

### Theory

Data consistency and reliability are fundamental to successful ML implementations. Inconsistent or unreliable data leads to poor model performance, biased predictions, and unreliable insights. SQL-based data quality frameworks ensure data meets the standards required for effective machine learning through validation, monitoring, and automated correction mechanisms.

### Data Quality Framework

#### 1. Data Validation Layer

```sql
-- Create comprehensive data validation framework
CREATE TABLE data_quality_rules (
    rule_id SERIAL PRIMARY KEY,
    table_name VARCHAR(100),
    column_name VARCHAR(100),
    rule_type VARCHAR(50), -- 'NOT_NULL', 'RANGE', 'FORMAT', 'UNIQUENESS', 'REFERENTIAL'
    rule_definition JSONB,
    severity_level VARCHAR(20), -- 'ERROR', 'WARNING', 'INFO'
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert validation rules
INSERT INTO data_quality_rules (table_name, column_name, rule_type, rule_definition, severity_level) VALUES
('customer_features', 'age', 'RANGE', '{"min": 0, "max": 120}', 'ERROR'),
('customer_features', 'income', 'RANGE', '{"min": 0, "max": 10000000}', 'WARNING'),
('customer_features', 'email', 'FORMAT', '{"pattern": "^[A-Za-z0-9+_.-]+@(.+)$"}', 'ERROR'),
('customer_features', 'customer_id', 'UNIQUENESS', '{}', 'ERROR'),
('transaction_data', 'amount', 'NOT_NULL', '{}', 'ERROR');

-- Data validation function
CREATE OR REPLACE FUNCTION validate_data_quality(
    p_table_name VARCHAR,
    p_batch_id VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    rule_id INT,
    rule_type VARCHAR,
    column_name VARCHAR,
    violation_count BIGINT,
    severity_level VARCHAR,
    sample_violations TEXT[]
) AS $$
DECLARE
    rule_record RECORD;
    validation_query TEXT;
    violation_count BIGINT;
    sample_violations TEXT[];
BEGIN
    FOR rule_record IN 
        SELECT * FROM data_quality_rules 
        WHERE table_name = p_table_name AND is_active = TRUE
    LOOP
        -- Build validation query based on rule type
        CASE rule_record.rule_type
            WHEN 'NOT_NULL' THEN
                validation_query := format(
                    'SELECT COUNT(*), ARRAY_AGG(DISTINCT %I::TEXT) 
                     FROM %I 
                     WHERE %I IS NULL %s',
                    rule_record.column_name,
                    rule_record.table_name,
                    rule_record.column_name,
                    CASE WHEN p_batch_id IS NOT NULL 
                         THEN format('AND batch_id = ''%s''', p_batch_id)
                         ELSE '' END
                );
                
            WHEN 'RANGE' THEN
                validation_query := format(
                    'SELECT COUNT(*), ARRAY_AGG(DISTINCT %I::TEXT) 
                     FROM %I 
                     WHERE %I < %s OR %I > %s %s',
                    rule_record.column_name,
                    rule_record.table_name,
                    rule_record.column_name,
                    (rule_record.rule_definition->>'min')::TEXT,
                    rule_record.column_name,
                    (rule_record.rule_definition->>'max')::TEXT,
                    CASE WHEN p_batch_id IS NOT NULL 
                         THEN format('AND batch_id = ''%s''', p_batch_id)
                         ELSE '' END
                );
                
            WHEN 'FORMAT' THEN
                validation_query := format(
                    'SELECT COUNT(*), ARRAY_AGG(DISTINCT %I::TEXT) 
                     FROM %I 
                     WHERE %I !~ ''%s'' %s',
                    rule_record.column_name,
                    rule_record.table_name,
                    rule_record.column_name,
                    rule_record.rule_definition->>'pattern',
                    CASE WHEN p_batch_id IS NOT NULL 
                         THEN format('AND batch_id = ''%s''', p_batch_id)
                         ELSE '' END
                );
                
            WHEN 'UNIQUENESS' THEN
                validation_query := format(
                    'SELECT COUNT(*) - COUNT(DISTINCT %I), 
                     ARRAY_AGG(DISTINCT %I::TEXT) 
                     FROM (
                         SELECT %I FROM %I %s
                         GROUP BY %I HAVING COUNT(*) > 1
                     ) duplicates',
                    rule_record.column_name,
                    rule_record.column_name,
                    rule_record.column_name,
                    rule_record.table_name,
                    CASE WHEN p_batch_id IS NOT NULL 
                         THEN format('WHERE batch_id = ''%s''', p_batch_id)
                         ELSE '' END,
                    rule_record.column_name
                );
        END CASE;
        
        -- Execute validation
        EXECUTE validation_query INTO violation_count, sample_violations;
        
        -- Return violation details
        rule_id := rule_record.rule_id;
        rule_type := rule_record.rule_type;
        column_name := rule_record.column_name;
        severity_level := rule_record.severity_level;
        
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

#### 2. Data Consistency Monitoring

```sql
-- Create data consistency monitoring system
CREATE TABLE data_consistency_checks (
    check_id SERIAL PRIMARY KEY,
    check_name VARCHAR(100),
    source_table VARCHAR(100),
    target_table VARCHAR(100),
    consistency_rule TEXT,
    tolerance_threshold NUMERIC DEFAULT 0.05,
    is_active BOOLEAN DEFAULT TRUE
);

-- Monitor referential integrity
CREATE OR REPLACE FUNCTION check_referential_consistency()
RETURNS TABLE (
    check_name VARCHAR,
    inconsistency_count BIGINT,
    consistency_rate NUMERIC,
    status VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    WITH consistency_results AS (
        SELECT 
            'customer_transaction_integrity' as check_name,
            COUNT(CASE WHEN c.customer_id IS NULL THEN 1 END) as inconsistencies,
            COUNT(*) as total_records
        FROM transactions t
        LEFT JOIN customers c ON t.customer_id = c.customer_id
        
        UNION ALL
        
        SELECT 
            'feature_target_alignment' as check_name,
            COUNT(CASE WHEN t.target_value IS NULL THEN 1 END) as inconsistencies,
            COUNT(*) as total_records
        FROM ml_features f
        LEFT JOIN target_values t ON f.record_id = t.record_id
        
        UNION ALL
        
        SELECT 
            'temporal_consistency' as check_name,
            COUNT(CASE WHEN feature_date > target_date THEN 1 END) as inconsistencies,
            COUNT(*) as total_records
        FROM ml_features f
        JOIN target_values t ON f.record_id = t.record_id
    )
    SELECT 
        cr.check_name,
        cr.inconsistencies::BIGINT,
        ROUND((1.0 - cr.inconsistencies::NUMERIC / NULLIF(cr.total_records, 0)) * 100, 2) as consistency_rate,
        CASE 
            WHEN cr.inconsistencies::NUMERIC / NULLIF(cr.total_records, 0) <= 0.01 THEN 'EXCELLENT'
            WHEN cr.inconsistencies::NUMERIC / NULLIF(cr.total_records, 0) <= 0.05 THEN 'GOOD'
            WHEN cr.inconsistencies::NUMERIC / NULLIF(cr.total_records, 0) <= 0.10 THEN 'ACCEPTABLE'
            ELSE 'POOR'
        END as status
    FROM consistency_results cr;
END;
$$ LANGUAGE plpgsql;
```

#### 3. Data Reliability Framework

```sql
-- Create data lineage tracking
CREATE TABLE data_lineage (
    lineage_id SERIAL PRIMARY KEY,
    source_table VARCHAR(100),
    source_column VARCHAR(100),
    transformation_sql TEXT,
    target_table VARCHAR(100),
    target_column VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);

-- Track data transformations
CREATE OR REPLACE FUNCTION track_data_transformation(
    p_source_table VARCHAR,
    p_source_columns TEXT[],
    p_transformation_sql TEXT,
    p_target_table VARCHAR,
    p_target_columns TEXT[]
)
RETURNS VOID AS $$
DECLARE
    i INT;
BEGIN
    FOR i IN 1..array_length(p_source_columns, 1) LOOP
        INSERT INTO data_lineage (
            source_table, source_column, transformation_sql, 
            target_table, target_column, created_by
        ) VALUES (
            p_source_table, 
            p_source_columns[i], 
            p_transformation_sql,
            p_target_table, 
            p_target_columns[i],
            current_user
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Data freshness monitoring
CREATE OR REPLACE FUNCTION monitor_data_freshness()
RETURNS TABLE (
    table_name VARCHAR,
    last_update TIMESTAMP,
    staleness_hours NUMERIC,
    freshness_status VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    WITH table_freshness AS (
        SELECT 
            'customer_features' as table_name,
            MAX(updated_at) as last_update,
            EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX(updated_at))) / 3600 as staleness_hours
        FROM customer_features
        
        UNION ALL
        
        SELECT 
            'transaction_data' as table_name,
            MAX(created_at) as last_update,
            EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX(created_at))) / 3600 as staleness_hours
        FROM transaction_data
        
        UNION ALL
        
        SELECT 
            'ml_predictions' as table_name,
            MAX(prediction_timestamp) as last_update,
            EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX(prediction_timestamp))) / 3600 as staleness_hours
        FROM ml_predictions
    )
    SELECT 
        tf.table_name,
        tf.last_update,
        tf.staleness_hours,
        CASE 
            WHEN tf.staleness_hours <= 1 THEN 'FRESH'
            WHEN tf.staleness_hours <= 24 THEN 'ACCEPTABLE'
            WHEN tf.staleness_hours <= 168 THEN 'STALE'
            ELSE 'CRITICAL'
        END as freshness_status
    FROM table_freshness tf;
END;
$$ LANGUAGE plpgsql;
```

#### 4. Automated Data Correction

```sql
-- Automated data cleaning and correction
CREATE OR REPLACE FUNCTION auto_correct_data_issues(
    p_table_name VARCHAR,
    p_apply_corrections BOOLEAN DEFAULT FALSE
)
RETURNS TABLE (
    correction_type VARCHAR,
    affected_rows BIGINT,
    correction_sql TEXT
) AS $$
DECLARE
    correction_queries TEXT[];
    query_text TEXT;
    affected_count BIGINT;
BEGIN
    -- Collect correction queries
    correction_queries := ARRAY[
        -- Fix obvious outliers in numeric columns
        format('UPDATE %I SET amount = CASE 
                   WHEN amount < 0 THEN ABS(amount)
                   WHEN amount > 1000000 THEN amount / 100
                   ELSE amount 
               END 
               WHERE amount < 0 OR amount > 1000000', p_table_name),
               
        -- Standardize text columns
        format('UPDATE %I SET email = LOWER(TRIM(email)) 
               WHERE email != LOWER(TRIM(email))', p_table_name),
               
        -- Fix date inconsistencies
        format('UPDATE %I SET created_at = CURRENT_TIMESTAMP 
               WHERE created_at > CURRENT_TIMESTAMP', p_table_name),
               
        -- Remove duplicate records
        format('DELETE FROM %I WHERE id NOT IN (
                   SELECT MIN(id) FROM %I GROUP BY customer_id, transaction_date
               )', p_table_name, p_table_name)
    ];
    
    -- Execute or preview corrections
    FOREACH query_text IN ARRAY correction_queries LOOP
        IF p_apply_corrections THEN
            EXECUTE query_text;
            GET DIAGNOSTICS affected_count = ROW_COUNT;
        ELSE
            -- Count affected rows without applying changes
            EXECUTE format('SELECT COUNT(*) FROM (%s) preview', 
                         replace(query_text, 'UPDATE', 'SELECT * FROM'));
            affected_count := FOUND;
        END IF;
        
        correction_type := CASE 
            WHEN query_text LIKE '%amount%' THEN 'OUTLIER_CORRECTION'
            WHEN query_text LIKE '%email%' THEN 'TEXT_STANDARDIZATION'
            WHEN query_text LIKE '%created_at%' THEN 'DATE_CORRECTION'
            WHEN query_text LIKE '%DELETE%' THEN 'DUPLICATE_REMOVAL'
        END;
        
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

### Real-World Applications

1. **Financial Services**: Ensure transaction data integrity for fraud detection models
2. **Healthcare**: Validate patient data consistency for diagnostic algorithms
3. **E-commerce**: Maintain product and customer data quality for recommendation systems
4. **Marketing**: Ensure campaign data reliability for attribution modeling

### Best Practices

1. **Proactive Validation**: Implement validation at data ingestion points
2. **Continuous Monitoring**: Set up automated data quality dashboards
3. **Version Control**: Track all data transformation logic
4. **Documentation**: Maintain clear data quality requirements
5. **Escalation Procedures**: Define clear protocols for data quality issues

### Common Pitfalls

1. **Over-Correction**: Automatically fixing data without understanding root causes
2. **Silent Failures**: Data quality issues that don't trigger alerts
3. **Incomplete Validation**: Missing edge cases in validation rules
4. **Performance Impact**: Heavy validation affecting system performance
5. **False Positives**: Overly strict rules flagging valid data as problematic

### Debugging and Monitoring

```sql
-- Create comprehensive data quality dashboard
CREATE VIEW data_quality_dashboard AS
WITH quality_metrics AS (
    SELECT 
        table_name,
        COUNT(CASE WHEN severity_level = 'ERROR' THEN 1 END) as error_count,
        COUNT(CASE WHEN severity_level = 'WARNING' THEN 1 END) as warning_count,
        COUNT(*) as total_checks,
        AVG(CASE WHEN violation_count = 0 THEN 1.0 ELSE 0.0 END) as pass_rate
    FROM validate_data_quality('all_tables')
    GROUP BY table_name
),
freshness_metrics AS (
    SELECT * FROM monitor_data_freshness()
),
consistency_metrics AS (
    SELECT * FROM check_referential_consistency()
)
SELECT 
    qm.table_name,
    qm.error_count,
    qm.warning_count,
    qm.pass_rate,
    fm.freshness_status,
    fm.staleness_hours,
    COALESCE(cm.consistency_rate, 100) as consistency_rate,
    CASE 
        WHEN qm.error_count > 0 OR fm.freshness_status = 'CRITICAL' THEN 'CRITICAL'
        WHEN qm.warning_count > 0 OR fm.freshness_status = 'STALE' THEN 'WARNING'
        ELSE 'HEALTHY'
    END as overall_status
FROM quality_metrics qm
LEFT JOIN freshness_metrics fm ON qm.table_name = fm.table_name
LEFT JOIN consistency_metrics cm ON qm.table_name LIKE '%' || cm.check_name || '%';

-- Automated alerting
CREATE OR REPLACE FUNCTION check_data_quality_alerts()
RETURNS VOID AS $$
DECLARE
    alert_record RECORD;
BEGIN
    FOR alert_record IN 
        SELECT * FROM data_quality_dashboard 
        WHERE overall_status IN ('CRITICAL', 'WARNING')
    LOOP
        -- Send alert (implementation depends on your alerting system)
        INSERT INTO data_quality_alerts (
            table_name, 
            alert_type, 
            alert_message, 
            created_at
        ) VALUES (
            alert_record.table_name,
            alert_record.overall_status,
            format('Data quality issue detected: %s errors, %s warnings, %s%% pass rate',
                   alert_record.error_count,
                   alert_record.warning_count,
                   alert_record.pass_rate * 100),
            CURRENT_TIMESTAMP
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

### Performance Considerations

- **Validation Overhead**: Typically adds 5-15% to data processing time
- **Monitoring Frequency**: Real-time for critical data, hourly for standard datasets
- **Storage Impact**: Data quality logs require 10-20% additional storage
- **Correction Efficiency**: Automated corrections can fix 70-90% of common issues

---

## Question 11

**What SQL features are there for report generation that might be useful for analyzing Machine Learning model performance?**

**Answer:**

### Theory

SQL provides powerful reporting capabilities that are essential for ML model performance analysis. These features enable data scientists and ML engineers to create comprehensive performance reports, track model metrics over time, compare model versions, and generate insights for stakeholders. Modern SQL databases offer advanced analytical functions, statistical computations, and visualization-ready data structures.

### Core SQL Reporting Features

#### 1. Window Functions for Performance Analysis

```sql
-- Comprehensive model performance tracking
CREATE OR REPLACE VIEW model_performance_report AS
WITH model_metrics AS (
    SELECT 
        model_name,
        model_version,
        evaluation_date,
        accuracy,
        precision_score,
        recall_score,
        f1_score,
        auc_roc,
        -- Calculate rolling averages
        AVG(accuracy) OVER (
            PARTITION BY model_name 
            ORDER BY evaluation_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as rolling_7day_accuracy,
        
        -- Calculate performance trends
        LAG(accuracy) OVER (
            PARTITION BY model_name 
            ORDER BY evaluation_date
        ) as previous_accuracy,
        
        -- Rank model versions by performance
        ROW_NUMBER() OVER (
            PARTITION BY model_name 
            ORDER BY f1_score DESC, evaluation_date DESC
        ) as performance_rank,
        
        -- Calculate percentile rankings
        PERCENT_RANK() OVER (
            PARTITION BY model_name 
            ORDER BY accuracy
        ) as accuracy_percentile
    FROM model_evaluations
),
performance_changes AS (
    SELECT 
        *,
        accuracy - previous_accuracy as accuracy_change,
        CASE 
            WHEN accuracy - previous_accuracy > 0.05 THEN 'SIGNIFICANT_IMPROVEMENT'
            WHEN accuracy - previous_accuracy > 0.01 THEN 'IMPROVEMENT'
            WHEN accuracy - previous_accuracy < -0.05 THEN 'SIGNIFICANT_DEGRADATION'
            WHEN accuracy - previous_accuracy < -0.01 THEN 'DEGRADATION'
            ELSE 'STABLE'
        END as performance_trend
    FROM model_metrics
)
SELECT 
    model_name,
    model_version,
    evaluation_date,
    accuracy,
    rolling_7day_accuracy,
    accuracy_change,
    performance_trend,
    f1_score,
    performance_rank,
    accuracy_percentile
FROM performance_changes;

-- Time-based performance analysis
CREATE OR REPLACE FUNCTION generate_time_series_report(
    p_model_name VARCHAR,
    p_start_date DATE,
    p_end_date DATE,
    p_granularity VARCHAR DEFAULT 'daily' -- 'hourly', 'daily', 'weekly', 'monthly'
)
RETURNS TABLE (
    time_period TIMESTAMP,
    avg_accuracy NUMERIC,
    avg_precision NUMERIC,
    avg_recall NUMERIC,
    prediction_volume BIGINT,
    error_rate NUMERIC,
    confidence_avg NUMERIC
) AS $$
DECLARE
    date_trunc_format VARCHAR;
BEGIN
    -- Set date truncation based on granularity
    date_trunc_format := CASE p_granularity
        WHEN 'hourly' THEN 'hour'
        WHEN 'daily' THEN 'day'
        WHEN 'weekly' THEN 'week'
        WHEN 'monthly' THEN 'month'
        ELSE 'day'
    END;
    
    RETURN QUERY
    SELECT 
        DATE_TRUNC(date_trunc_format, me.evaluation_date) as time_period,
        AVG(me.accuracy)::NUMERIC as avg_accuracy,
        AVG(me.precision_score)::NUMERIC as avg_precision,
        AVG(me.recall_score)::NUMERIC as avg_recall,
        COUNT(mp.prediction_id) as prediction_volume,
        AVG(CASE WHEN mp.actual_value != mp.predicted_value THEN 1.0 ELSE 0.0 END)::NUMERIC as error_rate,
        AVG(mp.confidence_score)::NUMERIC as confidence_avg
    FROM model_evaluations me
    LEFT JOIN ml_predictions mp ON me.model_name = mp.model_name 
        AND DATE_TRUNC(date_trunc_format, mp.prediction_timestamp) = DATE_TRUNC(date_trunc_format, me.evaluation_date)
    WHERE me.model_name = p_model_name
        AND me.evaluation_date BETWEEN p_start_date AND p_end_date
    GROUP BY DATE_TRUNC(date_trunc_format, me.evaluation_date)
    ORDER BY time_period;
END;
$$ LANGUAGE plpgsql;
```

#### 2. Advanced Aggregation and Statistical Functions

```sql
-- Statistical model analysis
CREATE OR REPLACE FUNCTION model_statistical_report(
    p_model_name VARCHAR,
    p_analysis_period INT DEFAULT 30 -- days
)
RETURNS TABLE (
    metric_name VARCHAR,
    mean_value NUMERIC,
    median_value NUMERIC,
    std_deviation NUMERIC,
    min_value NUMERIC,
    max_value NUMERIC,
    percentile_25 NUMERIC,
    percentile_75 NUMERIC,
    coefficient_variation NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH model_data AS (
        SELECT * FROM model_evaluations 
        WHERE model_name = p_model_name 
        AND evaluation_date >= CURRENT_DATE - INTERVAL '%s days'::VARCHAR(20), p_analysis_period
    ),
    statistical_metrics AS (
        SELECT 'accuracy' as metric, accuracy as value FROM model_data
        UNION ALL
        SELECT 'precision', precision_score FROM model_data
        UNION ALL
        SELECT 'recall', recall_score FROM model_data
        UNION ALL
        SELECT 'f1_score', f1_score FROM model_data
        UNION ALL
        SELECT 'auc_roc', auc_roc FROM model_data
    )
    SELECT 
        sm.metric as metric_name,
        AVG(sm.value)::NUMERIC as mean_value,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sm.value)::NUMERIC as median_value,
        STDDEV(sm.value)::NUMERIC as std_deviation,
        MIN(sm.value)::NUMERIC as min_value,
        MAX(sm.value)::NUMERIC as max_value,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY sm.value)::NUMERIC as percentile_25,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY sm.value)::NUMERIC as percentile_75,
        CASE 
            WHEN AVG(sm.value) != 0 
            THEN (STDDEV(sm.value) / AVG(sm.value))::NUMERIC
            ELSE NULL 
        END as coefficient_variation
    FROM statistical_metrics sm
    GROUP BY sm.metric;
END;
$$ LANGUAGE plpgsql;

-- Model comparison report
CREATE OR REPLACE VIEW model_comparison_report AS
WITH model_rankings AS (
    SELECT 
        model_name,
        AVG(accuracy) as avg_accuracy,
        AVG(precision_score) as avg_precision,
        AVG(recall_score) as avg_recall,
        AVG(f1_score) as avg_f1,
        COUNT(*) as evaluation_count,
        MAX(evaluation_date) as last_evaluation,
        
        -- Calculate relative performance
        AVG(accuracy) - LAG(AVG(accuracy)) OVER (ORDER BY AVG(f1_score) DESC) as accuracy_vs_next,
        
        -- Rank models
        RANK() OVER (ORDER BY AVG(f1_score) DESC) as f1_rank,
        RANK() OVER (ORDER BY AVG(accuracy) DESC) as accuracy_rank,
        RANK() OVER (ORDER BY AVG(precision_score) DESC) as precision_rank,
        RANK() OVER (ORDER BY AVG(recall_score) DESC) as recall_rank
    FROM model_evaluations 
    WHERE evaluation_date >= CURRENT_DATE - 30
    GROUP BY model_name
)
SELECT 
    model_name,
    ROUND(avg_accuracy * 100, 2) as accuracy_pct,
    ROUND(avg_precision * 100, 2) as precision_pct,
    ROUND(avg_recall * 100, 2) as recall_pct,
    ROUND(avg_f1 * 100, 2) as f1_pct,
    evaluation_count,
    last_evaluation,
    f1_rank,
    accuracy_rank,
    ROUND((accuracy_vs_next * 100), 2) as accuracy_advantage_pct,
    
    -- Overall score combining all metrics
    ROUND((f1_rank + accuracy_rank + precision_rank + recall_rank) / 4.0, 1) as composite_rank
FROM model_rankings
ORDER BY composite_rank;
```

#### 3. Pivot Tables and Cross-Tabulations

```sql
-- Model performance by categorical features
CREATE OR REPLACE FUNCTION model_performance_by_segment(
    p_model_name VARCHAR,
    p_segment_column VARCHAR,
    p_date_range_days INT DEFAULT 30
)
RETURNS TABLE (
    segment_value TEXT,
    prediction_count BIGINT,
    accuracy NUMERIC,
    precision_score NUMERIC,
    recall_score NUMERIC,
    f1_score NUMERIC,
    avg_confidence NUMERIC
) AS $$
DECLARE
    query_sql TEXT;
BEGIN
    query_sql := format('
        SELECT 
            %I::TEXT as segment_value,
            COUNT(*) as prediction_count,
            AVG(CASE WHEN actual_value = predicted_value THEN 1.0 ELSE 0.0 END)::NUMERIC as accuracy,
            AVG(CASE WHEN predicted_value = 1 AND actual_value = 1 THEN 1.0 
                     WHEN predicted_value = 1 THEN 0.0 END)::NUMERIC as precision_score,
            AVG(CASE WHEN actual_value = 1 AND predicted_value = 1 THEN 1.0 
                     WHEN actual_value = 1 THEN 0.0 END)::NUMERIC as recall_score,
            2 * (AVG(CASE WHEN predicted_value = 1 AND actual_value = 1 THEN 1.0 
                          WHEN predicted_value = 1 THEN 0.0 END) * 
                 AVG(CASE WHEN actual_value = 1 AND predicted_value = 1 THEN 1.0 
                          WHEN actual_value = 1 THEN 0.0 END)) / 
            NULLIF((AVG(CASE WHEN predicted_value = 1 AND actual_value = 1 THEN 1.0 
                             WHEN predicted_value = 1 THEN 0.0 END) + 
                    AVG(CASE WHEN actual_value = 1 AND predicted_value = 1 THEN 1.0 
                             WHEN actual_value = 1 THEN 0.0 END)), 0)::NUMERIC as f1_score,
            AVG(confidence_score)::NUMERIC as avg_confidence
        FROM ml_predictions p
        JOIN feature_data f ON p.record_id = f.record_id
        WHERE p.model_name = ''%s''
            AND p.prediction_timestamp >= CURRENT_DATE - INTERVAL ''%s days''
        GROUP BY %I
        ORDER BY prediction_count DESC',
        p_segment_column,
        p_model_name,
        p_date_range_days,
        p_segment_column
    );
    
    RETURN QUERY EXECUTE query_sql;
END;
$$ LANGUAGE plpgsql;

-- Confusion matrix pivot table
CREATE OR REPLACE FUNCTION generate_confusion_matrix(
    p_model_name VARCHAR,
    p_class_labels TEXT[] DEFAULT ARRAY['0', '1']
)
RETURNS TABLE (
    actual_class TEXT,
    predicted_0 BIGINT,
    predicted_1 BIGINT,
    total_actual BIGINT,
    class_accuracy NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        av.actual_value::TEXT as actual_class,
        COUNT(CASE WHEN pv.predicted_value = '0' THEN 1 END) as predicted_0,
        COUNT(CASE WHEN pv.predicted_value = '1' THEN 1 END) as predicted_1,
        COUNT(*) as total_actual,
        (COUNT(CASE WHEN av.actual_value::TEXT = pv.predicted_value::TEXT THEN 1 END)::FLOAT / COUNT(*))::NUMERIC as class_accuracy
    FROM ml_predictions pv
    JOIN (
        SELECT record_id, actual_value FROM ml_predictions WHERE model_name = p_model_name
    ) av ON pv.record_id = av.record_id
    WHERE pv.model_name = p_model_name
    GROUP BY av.actual_value
    ORDER BY av.actual_value;
END;
$$ LANGUAGE plpgsql;
```

#### 4. Report Templates and Automated Generation

```sql
-- Comprehensive model performance report template
CREATE OR REPLACE FUNCTION generate_model_report(
    p_model_name VARCHAR,
    p_report_date DATE DEFAULT CURRENT_DATE
)
RETURNS JSONB AS $$
DECLARE
    report_data JSONB;
    performance_summary JSONB;
    time_series_data JSONB;
    segment_analysis JSONB;
    statistical_summary JSONB;
BEGIN
    -- Executive Summary
    SELECT jsonb_build_object(
        'model_name', p_model_name,
        'report_date', p_report_date,
        'evaluation_period', '30 days',
        'total_predictions', COUNT(*),
        'overall_accuracy', ROUND(AVG(CASE WHEN actual_value = predicted_value THEN 1.0 ELSE 0.0 END) * 100, 2),
        'confidence_avg', ROUND(AVG(confidence_score) * 100, 2),
        'last_model_update', MAX(model_version_date)
    ) INTO performance_summary
    FROM ml_predictions p
    JOIN model_metadata m ON p.model_name = m.model_name
    WHERE p.model_name = p_model_name 
        AND p.prediction_timestamp >= p_report_date - 30;
    
    -- Time series performance
    SELECT jsonb_agg(
        jsonb_build_object(
            'date', time_period,
            'accuracy', avg_accuracy,
            'volume', prediction_volume,
            'error_rate', error_rate
        )
    ) INTO time_series_data
    FROM generate_time_series_report(p_model_name, p_report_date - 30, p_report_date, 'daily');
    
    -- Segment analysis
    SELECT jsonb_agg(
        jsonb_build_object(
            'segment', segment_value,
            'count', prediction_count,
            'accuracy', accuracy,
            'f1_score', f1_score
        )
    ) INTO segment_analysis
    FROM model_performance_by_segment(p_model_name, 'customer_segment', 30);
    
    -- Statistical summary
    SELECT jsonb_agg(
        jsonb_build_object(
            'metric', metric_name,
            'mean', mean_value,
            'std_dev', std_deviation,
            'percentile_25', percentile_25,
            'percentile_75', percentile_75
        )
    ) INTO statistical_summary
    FROM model_statistical_report(p_model_name, 30);
    
    -- Combine all sections
    report_data := jsonb_build_object(
        'executive_summary', performance_summary,
        'time_series_analysis', time_series_data,
        'segment_performance', segment_analysis,
        'statistical_analysis', statistical_summary,
        'generated_at', CURRENT_TIMESTAMP
    );
    
    -- Store report for future reference
    INSERT INTO model_reports (model_name, report_date, report_data)
    VALUES (p_model_name, p_report_date, report_data)
    ON CONFLICT (model_name, report_date) 
    DO UPDATE SET report_data = EXCLUDED.report_data;
    
    RETURN report_data;
END;
$$ LANGUAGE plpgsql;

-- Automated report scheduling
CREATE OR REPLACE FUNCTION schedule_model_reports()
RETURNS VOID AS $$
DECLARE
    model_record RECORD;
BEGIN
    FOR model_record IN 
        SELECT DISTINCT model_name FROM ml_predictions 
        WHERE prediction_timestamp >= CURRENT_DATE - 1
    LOOP
        -- Generate daily report
        PERFORM generate_model_report(model_record.model_name, CURRENT_DATE);
        
        -- Generate weekly report on Mondays
        IF EXTRACT(DOW FROM CURRENT_DATE) = 1 THEN
            PERFORM generate_model_report(model_record.model_name, CURRENT_DATE - 7);
        END IF;
        
        -- Generate monthly report on the 1st
        IF EXTRACT(DAY FROM CURRENT_DATE) = 1 THEN
            PERFORM generate_model_report(model_record.model_name, CURRENT_DATE - 30);
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

#### 5. Interactive Reporting Features

```sql
-- Dynamic filter and drill-down capabilities
CREATE OR REPLACE FUNCTION model_report_with_filters(
    p_model_name VARCHAR,
    p_filters JSONB DEFAULT '{}'::JSONB
)
RETURNS TABLE (
    record_id VARCHAR,
    prediction_date TIMESTAMP,
    predicted_value NUMERIC,
    actual_value NUMERIC,
    confidence_score NUMERIC,
    segment_info JSONB,
    feature_values JSONB
) AS $$
DECLARE
    base_query TEXT;
    where_clauses TEXT[] := ARRAY[]::TEXT[];
    filter_key TEXT;
    filter_value TEXT;
BEGIN
    base_query := '
        SELECT 
            p.record_id,
            p.prediction_timestamp,
            p.predicted_value,
            p.actual_value,
            p.confidence_score,
            jsonb_build_object(
                ''customer_segment'', f.customer_segment,
                ''risk_category'', f.risk_category,
                ''geography'', f.geography
            ) as segment_info,
            f.feature_values
        FROM ml_predictions p
        JOIN feature_data f ON p.record_id = f.record_id
        WHERE p.model_name = ''' || p_model_name || '''';
    
    -- Build dynamic WHERE clause from filters
    FOR filter_key, filter_value IN 
        SELECT * FROM jsonb_each_text(p_filters)
    LOOP
        CASE filter_key
            WHEN 'date_from' THEN
                where_clauses := where_clauses || format('p.prediction_timestamp >= ''%s''', filter_value);
            WHEN 'date_to' THEN
                where_clauses := where_clauses || format('p.prediction_timestamp <= ''%s''', filter_value);
            WHEN 'confidence_min' THEN
                where_clauses := where_clauses || format('p.confidence_score >= %s', filter_value);
            WHEN 'segment' THEN
                where_clauses := where_clauses || format('f.customer_segment = ''%s''', filter_value);
            WHEN 'correct_predictions_only' THEN
                IF filter_value::BOOLEAN THEN
                    where_clauses := where_clauses || 'p.predicted_value = p.actual_value';
                END IF;
        END CASE;
    END LOOP;
    
    -- Add WHERE clauses if any
    IF array_length(where_clauses, 1) > 0 THEN
        base_query := base_query || ' AND ' || array_to_string(where_clauses, ' AND ');
    END IF;
    
    base_query := base_query || ' ORDER BY p.prediction_timestamp DESC';
    
    RETURN QUERY EXECUTE base_query;
END;
$$ LANGUAGE plpgsql;
```

### Real-World Applications

1. **Executive Dashboards**: High-level model performance metrics for stakeholders
2. **Data Science Reports**: Detailed technical analysis for model improvement
3. **Regulatory Compliance**: Audit trails and performance documentation
4. **A/B Testing**: Comparative analysis of model versions
5. **Alerting Systems**: Automated notifications for performance degradation

### Best Practices

1. **Standardized Metrics**: Use consistent performance indicators across all models
2. **Automated Generation**: Schedule regular report creation and distribution
3. **Interactive Features**: Enable drill-down and filtering capabilities
4. **Visual Optimization**: Structure data for easy visualization tool integration
5. **Historical Tracking**: Maintain long-term performance trends

### Common Pitfalls

1. **Over-Aggregation**: Losing important detail in summary reports
2. **Performance Impact**: Heavy reporting queries affecting production systems
3. **Data Staleness**: Reports based on outdated information
4. **Missing Context**: Metrics without business context or baselines
5. **Information Overload**: Too many metrics without clear priorities

### Debugging and Optimization

```sql
-- Report performance monitoring
CREATE VIEW report_performance_stats AS
SELECT 
    report_type,
    model_name,
    AVG(generation_time_seconds) as avg_generation_time,
    MAX(generation_time_seconds) as max_generation_time,
    COUNT(*) as report_count,
    AVG(data_points_processed) as avg_data_points
FROM model_report_log
WHERE generated_at >= CURRENT_DATE - 7
GROUP BY report_type, model_name
ORDER BY avg_generation_time DESC;

-- Optimize slow-running reports
CREATE OR REPLACE FUNCTION optimize_report_query(p_query TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Add appropriate indexes and query hints
    RETURN format('
        /*+ USE_INDEX(ml_predictions, idx_model_timestamp) */
        %s
        OPTION (OPTIMIZE FOR UNKNOWN)',
        p_query
    );
END;
$$ LANGUAGE plpgsql;
```

### Performance Considerations

- **Indexing Strategy**: Create indexes on model_name, evaluation_date, and performance metrics
- **Materialized Views**: Use for frequently accessed report data
- **Partitioning**: Partition large tables by date for faster queries
- **Caching**: Cache report results for frequently requested time periods
- **Parallel Processing**: Use parallel query execution for large dataset reports

---

## Question 12

**How can you use SQL to visualize the distribution of data points before feeding them into an ML algorithm?**

**Answer:**

### Theory

Data distribution visualization is crucial before training ML models as it reveals patterns, outliers, skewness, and potential data quality issues. SQL can generate statistical summaries, histograms, and distribution metrics that help data scientists understand their datasets and make informed preprocessing decisions. Modern SQL databases provide statistical functions and data binning capabilities essential for exploratory data analysis.

### Distribution Analysis Techniques

#### 1. Statistical Summary Generation

```sql
-- Comprehensive statistical summary function
CREATE OR REPLACE FUNCTION generate_distribution_summary(
    p_table_name VARCHAR,
    p_column_name VARCHAR,
    p_sample_size INT DEFAULT NULL
)
RETURNS TABLE (
    column_name VARCHAR,
    data_type VARCHAR,
    total_count BIGINT,
    null_count BIGINT,
    null_percentage NUMERIC,
    distinct_count BIGINT,
    min_value NUMERIC,
    max_value NUMERIC,
    mean_value NUMERIC,
    median_value NUMERIC,
    mode_value TEXT,
    std_deviation NUMERIC,
    variance_value NUMERIC,
    skewness NUMERIC,
    kurtosis NUMERIC,
    q1_percentile NUMERIC,
    q3_percentile NUMERIC,
    iqr_value NUMERIC
) AS $$
DECLARE
    sample_clause TEXT := '';
    query_text TEXT;
BEGIN
    -- Add sampling if specified
    IF p_sample_size IS NOT NULL THEN
        sample_clause := format(' TABLESAMPLE SYSTEM_ROWS(%s)', p_sample_size);
    END IF;
    
    query_text := format('
        WITH data_sample AS (
            SELECT %I as value,
                   %I::TEXT as text_value
            FROM %I%s
            WHERE %I IS NOT NULL
        ),
        basic_stats AS (
            SELECT 
                COUNT(*) as total_non_null,
                COUNT(DISTINCT value) as distinct_vals,
                MIN(value) as min_val,
                MAX(value) as max_val,
                AVG(value) as mean_val,
                STDDEV(value) as std_dev,
                VARIANCE(value) as variance_val,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median_val,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) as q1_val,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) as q3_val
            FROM data_sample
        ),
        mode_calc AS (
            SELECT text_value as mode_val
            FROM data_sample
            GROUP BY text_value
            ORDER BY COUNT(*) DESC
            LIMIT 1
        ),
        moment_stats AS (
            SELECT 
                -- Calculate skewness: E[(X-μ)³]/σ³
                AVG(POWER((value - bs.mean_val) / NULLIF(bs.std_dev, 0), 3)) as skewness_val,
                -- Calculate kurtosis: E[(X-μ)⁴]/σ⁴ - 3
                AVG(POWER((value - bs.mean_val) / NULLIF(bs.std_dev, 0), 4)) - 3 as kurtosis_val
            FROM data_sample, basic_stats bs
        ),
        total_count_calc AS (
            SELECT COUNT(*) as total_rows,
                   COUNT(%I) as non_null_rows
            FROM %I%s
        )
        SELECT 
            ''%s''::VARCHAR as column_name,
            pg_typeof(%I)::VARCHAR as data_type,
            tcc.total_rows as total_count,
            (tcc.total_rows - tcc.non_null_rows) as null_count,
            ROUND(((tcc.total_rows - tcc.non_null_rows)::NUMERIC / NULLIF(tcc.total_rows, 0)) * 100, 2) as null_percentage,
            bs.distinct_vals as distinct_count,
            bs.min_val as min_value,
            bs.max_val as max_value,
            ROUND(bs.mean_val, 4) as mean_value,
            ROUND(bs.median_val, 4) as median_value,
            mc.mode_val as mode_value,
            ROUND(bs.std_dev, 4) as std_deviation,
            ROUND(bs.variance_val, 4) as variance_value,
            ROUND(ms.skewness_val::NUMERIC, 4) as skewness,
            ROUND(ms.kurtosis_val::NUMERIC, 4) as kurtosis,
            ROUND(bs.q1_val, 4) as q1_percentile,
            ROUND(bs.q3_val, 4) as q3_percentile,
            ROUND((bs.q3_val - bs.q1_val), 4) as iqr_value
        FROM basic_stats bs, mode_calc mc, moment_stats ms, total_count_calc tcc',
        p_column_name, p_column_name, p_table_name, sample_clause, p_column_name,
        p_column_name, p_table_name, sample_clause,
        p_column_name, p_column_name
    );
    
    RETURN QUERY EXECUTE query_text;
END;
$$ LANGUAGE plpgsql;

-- Multi-column distribution analysis
CREATE OR REPLACE FUNCTION analyze_dataset_distributions(
    p_table_name VARCHAR,
    p_numeric_columns TEXT[],
    p_sample_size INT DEFAULT 50000
)
RETURNS TABLE (
    column_name VARCHAR,
    distribution_type VARCHAR, -- 'NORMAL', 'SKEWED', 'UNIFORM', 'BIMODAL', 'UNKNOWN'
    outlier_count BIGINT,
    outlier_percentage NUMERIC,
    normality_test_p_value NUMERIC,
    recommended_transformation VARCHAR
) AS $$
DECLARE
    col_name TEXT;
    analysis_result RECORD;
BEGIN
    FOREACH col_name IN ARRAY p_numeric_columns LOOP
        WITH stats AS (
            SELECT * FROM generate_distribution_summary(p_table_name, col_name, p_sample_size)
        ),
        outlier_analysis AS (
            SELECT 
                s.column_name,
                s.q1_percentile,
                s.q3_percentile,
                s.iqr_value,
                s.skewness,
                s.kurtosis
            FROM stats s
        ),
        outlier_count AS (
            SELECT COUNT(*) as outliers
            FROM (
                SELECT 1 FROM generate_distribution_summary(p_table_name, col_name)
            ) sample_data
            -- This would need actual data access - simplified for example
        )
        SELECT 
            oa.column_name,
            CASE 
                WHEN ABS(oa.skewness) < 0.5 AND ABS(oa.kurtosis) < 1 THEN 'NORMAL'
                WHEN ABS(oa.skewness) > 1 THEN 'SKEWED'
                WHEN ABS(oa.kurtosis) > 3 THEN 'HEAVY_TAILED'
                ELSE 'UNKNOWN'
            END as distribution_type,
            0::BIGINT as outlier_count, -- Simplified
            0::NUMERIC as outlier_percentage, -- Simplified
            NULL::NUMERIC as normality_test_p_value, -- Would need statistical tests
            CASE 
                WHEN oa.skewness > 1 THEN 'LOG_TRANSFORMATION'
                WHEN oa.skewness < -1 THEN 'SQUARE_TRANSFORMATION'
                WHEN ABS(oa.kurtosis) > 3 THEN 'ROBUST_SCALING'
                ELSE 'STANDARD_SCALING'
            END as recommended_transformation
        INTO analysis_result
        FROM outlier_analysis oa;
        
        column_name := analysis_result.column_name;
        distribution_type := analysis_result.distribution_type;
        outlier_count := analysis_result.outlier_count;
        outlier_percentage := analysis_result.outlier_percentage;
        normality_test_p_value := analysis_result.normality_test_p_value;
        recommended_transformation := analysis_result.recommended_transformation;
        
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

#### 2. Histogram Generation and Binning

```sql
-- Dynamic histogram generation
CREATE OR REPLACE FUNCTION generate_histogram(
    p_table_name VARCHAR,
    p_column_name VARCHAR,
    p_num_bins INT DEFAULT 20,
    p_bin_method VARCHAR DEFAULT 'equal_width' -- 'equal_width', 'equal_frequency', 'custom'
)
RETURNS TABLE (
    bin_number INT,
    bin_range_start NUMERIC,
    bin_range_end NUMERIC,
    bin_count BIGINT,
    bin_percentage NUMERIC,
    cumulative_count BIGINT,
    cumulative_percentage NUMERIC
) AS $$
DECLARE
    min_val NUMERIC;
    max_val NUMERIC;
    bin_width NUMERIC;
    total_count BIGINT;
    query_text TEXT;
BEGIN
    -- Get basic statistics
    EXECUTE format('SELECT MIN(%I), MAX(%I), COUNT(*) FROM %I WHERE %I IS NOT NULL', 
                   p_column_name, p_column_name, p_table_name, p_column_name)
    INTO min_val, max_val, total_count;
    
    IF p_bin_method = 'equal_width' THEN
        bin_width := (max_val - min_val) / p_num_bins;
        
        query_text := format('
            WITH bins AS (
                SELECT 
                    generate_series(0, %s-1) as bin_num,
                    %s + (generate_series(0, %s-1) * %s) as bin_start,
                    %s + (generate_series(1, %s) * %s) as bin_end
            ),
            data_bins AS (
                SELECT 
                    %I as value,
                    LEAST(FLOOR((%I - %s) / %s), %s-1) as bin_assignment
                FROM %I
                WHERE %I IS NOT NULL
                  AND %I BETWEEN %s AND %s
            ),
            histogram_data AS (
                SELECT 
                    b.bin_num,
                    b.bin_start,
                    b.bin_end,
                    COUNT(db.value) as count_in_bin
                FROM bins b
                LEFT JOIN data_bins db ON b.bin_num = db.bin_assignment
                GROUP BY b.bin_num, b.bin_start, b.bin_end
                ORDER BY b.bin_num
            )
            SELECT 
                bin_num::INT,
                ROUND(bin_start, 4) as bin_range_start,
                ROUND(bin_end, 4) as bin_range_end,
                count_in_bin as bin_count,
                ROUND((count_in_bin::NUMERIC / %s) * 100, 2) as bin_percentage,
                SUM(count_in_bin) OVER (ORDER BY bin_num) as cumulative_count,
                ROUND((SUM(count_in_bin) OVER (ORDER BY bin_num)::NUMERIC / %s) * 100, 2) as cumulative_percentage
            FROM histogram_data',
            p_num_bins, min_val, p_num_bins, bin_width, min_val, p_num_bins, bin_width,
            p_column_name, p_column_name, min_val, bin_width, p_num_bins,
            p_table_name, p_column_name, p_column_name, min_val, max_val,
            total_count, total_count
        );
        
    ELSIF p_bin_method = 'equal_frequency' THEN
        query_text := format('
            WITH percentile_bins AS (
                SELECT 
                    generate_series(0, %s-1) as bin_num,
                    PERCENTILE_CONT(generate_series(0, %s-1)::NUMERIC / %s) 
                        WITHIN GROUP (ORDER BY %I) as bin_start,
                    PERCENTILE_CONT(generate_series(1, %s)::NUMERIC / %s) 
                        WITHIN GROUP (ORDER BY %I) as bin_end
                FROM %I
                WHERE %I IS NOT NULL
            ),
            data_assignment AS (
                SELECT 
                    %I as value,
                    NTILE(%s) OVER (ORDER BY %I) - 1 as bin_assignment
                FROM %I
                WHERE %I IS NOT NULL
            ),
            histogram_data AS (
                SELECT 
                    pb.bin_num,
                    pb.bin_start,
                    pb.bin_end,
                    COUNT(da.value) as count_in_bin
                FROM percentile_bins pb
                LEFT JOIN data_assignment da ON pb.bin_num = da.bin_assignment
                GROUP BY pb.bin_num, pb.bin_start, pb.bin_end
                ORDER BY pb.bin_num
            )
            SELECT 
                bin_num::INT,
                ROUND(bin_start, 4) as bin_range_start,
                ROUND(bin_end, 4) as bin_range_end,
                count_in_bin as bin_count,
                ROUND((count_in_bin::NUMERIC / %s) * 100, 2) as bin_percentage,
                SUM(count_in_bin) OVER (ORDER BY bin_num) as cumulative_count,
                ROUND((SUM(count_in_bin) OVER (ORDER BY bin_num)::NUMERIC / %s) * 100, 2) as cumulative_percentage
            FROM histogram_data',
            p_num_bins, p_num_bins, p_num_bins, p_column_name, p_num_bins, p_num_bins, p_column_name,
            p_table_name, p_column_name,
            p_column_name, p_num_bins, p_column_name, p_table_name, p_column_name,
            total_count, total_count
        );
    END IF;
    
    RETURN QUERY EXECUTE query_text;
END;
$$ LANGUAGE plpgsql;

-- Categorical distribution visualization
CREATE OR REPLACE FUNCTION generate_categorical_distribution(
    p_table_name VARCHAR,
    p_column_name VARCHAR,
    p_top_n INT DEFAULT 20
)
RETURNS TABLE (
    category_value TEXT,
    frequency BIGINT,
    percentage NUMERIC,
    cumulative_frequency BIGINT,
    cumulative_percentage NUMERIC
) AS $$
DECLARE
    query_text TEXT;
BEGIN
    query_text := format('
        WITH category_counts AS (
            SELECT 
                %I::TEXT as category,
                COUNT(*) as freq
            FROM %I
            WHERE %I IS NOT NULL
            GROUP BY %I
            ORDER BY COUNT(*) DESC
            LIMIT %s
        ),
        total_count AS (
            SELECT COUNT(*) as total FROM %I WHERE %I IS NOT NULL
        )
        SELECT 
            cc.category as category_value,
            cc.freq as frequency,
            ROUND((cc.freq::NUMERIC / tc.total) * 100, 2) as percentage,
            SUM(cc.freq) OVER (ORDER BY cc.freq DESC) as cumulative_frequency,
            ROUND((SUM(cc.freq) OVER (ORDER BY cc.freq DESC)::NUMERIC / tc.total) * 100, 2) as cumulative_percentage
        FROM category_counts cc, total_count tc
        ORDER BY cc.freq DESC',
        p_column_name, p_table_name, p_column_name, p_column_name, p_top_n,
        p_table_name, p_column_name
    );
    
    RETURN QUERY EXECUTE query_text;
END;
$$ LANGUAGE plpgsql;
```

#### 3. Correlation and Relationship Analysis

```sql
-- Correlation matrix generation
CREATE OR REPLACE FUNCTION generate_correlation_matrix(
    p_table_name VARCHAR,
    p_numeric_columns TEXT[]
)
RETURNS TABLE (
    column1 VARCHAR,
    column2 VARCHAR,
    correlation_coefficient NUMERIC,
    correlation_strength VARCHAR
) AS $$
DECLARE
    col1 TEXT;
    col2 TEXT;
    correlation_val NUMERIC;
BEGIN
    FOR col1 IN SELECT unnest(p_numeric_columns) LOOP
        FOR col2 IN SELECT unnest(p_numeric_columns) LOOP
            -- Calculate Pearson correlation coefficient
            EXECUTE format('
                SELECT CORR(%I, %I)
                FROM %I
                WHERE %I IS NOT NULL AND %I IS NOT NULL',
                col1, col2, p_table_name, col1, col2
            ) INTO correlation_val;
            
            column1 := col1;
            column2 := col2;
            correlation_coefficient := ROUND(correlation_val, 4);
            correlation_strength := CASE 
                WHEN ABS(correlation_val) >= 0.8 THEN 'VERY_STRONG'
                WHEN ABS(correlation_val) >= 0.6 THEN 'STRONG'
                WHEN ABS(correlation_val) >= 0.4 THEN 'MODERATE'
                WHEN ABS(correlation_val) >= 0.2 THEN 'WEAK'
                ELSE 'VERY_WEAK'
            END;
            
            RETURN NEXT;
        END LOOP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Feature interaction analysis
CREATE OR REPLACE FUNCTION analyze_feature_interactions(
    p_table_name VARCHAR,
    p_feature1 VARCHAR,
    p_feature2 VARCHAR,
    p_target_column VARCHAR
)
RETURNS TABLE (
    feature1_bin VARCHAR,
    feature2_bin VARCHAR,
    avg_target NUMERIC,
    count_records BIGINT,
    target_variance NUMERIC,
    interaction_strength NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    EXECUTE format('
        WITH feature_bins AS (
            SELECT 
                %I,
                %I,
                %I,
                CASE 
                    WHEN %I <= (SELECT PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY %I) FROM %I) THEN ''LOW''
                    WHEN %I <= (SELECT PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY %I) FROM %I) THEN ''MEDIUM''
                    ELSE ''HIGH''
                END as f1_bin,
                CASE 
                    WHEN %I <= (SELECT PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY %I) FROM %I) THEN ''LOW''
                    WHEN %I <= (SELECT PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY %I) FROM %I) THEN ''MEDIUM''
                    ELSE ''HIGH''
                END as f2_bin
            FROM %I
            WHERE %I IS NOT NULL AND %I IS NOT NULL AND %I IS NOT NULL
        ),
        interaction_stats AS (
            SELECT 
                f1_bin,
                f2_bin,
                AVG(%I) as avg_target_val,
                COUNT(*) as record_count,
                VARIANCE(%I) as target_var,
                -- Calculate interaction effect
                AVG(%I) - (
                    SELECT AVG(%I) FROM feature_bins
                ) as deviation_from_mean
            FROM feature_bins
            GROUP BY f1_bin, f2_bin
        )
        SELECT 
            f1_bin as feature1_bin,
            f2_bin as feature2_bin,
            ROUND(avg_target_val, 4) as avg_target,
            record_count as count_records,
            ROUND(target_var, 4) as target_variance,
            ROUND(ABS(deviation_from_mean), 4) as interaction_strength
        FROM interaction_stats
        ORDER BY interaction_strength DESC',
        p_feature1, p_feature2, p_target_column,
        p_feature1, p_feature1, p_table_name,
        p_feature1, p_feature1, p_table_name,
        p_feature2, p_feature2, p_table_name,
        p_feature2, p_feature2, p_table_name,
        p_table_name, p_feature1, p_feature2, p_target_column,
        p_target_column, p_target_column, p_target_column, p_target_column
    );
END;
$$ LANGUAGE plpgsql;
```

#### 4. Outlier Detection and Visualization

```sql
-- Multi-method outlier detection
CREATE OR REPLACE FUNCTION detect_outliers(
    p_table_name VARCHAR,
    p_column_name VARCHAR,
    p_method VARCHAR DEFAULT 'IQR' -- 'IQR', 'Z_SCORE', 'MODIFIED_Z_SCORE', 'ISOLATION'
)
RETURNS TABLE (
    record_id VARCHAR,
    value NUMERIC,
    outlier_score NUMERIC,
    outlier_method VARCHAR,
    is_outlier BOOLEAN
) AS $$
DECLARE
    query_text TEXT;
BEGIN
    CASE p_method
        WHEN 'IQR' THEN
            query_text := format('
                WITH stats AS (
                    SELECT 
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY %I) as q1,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY %I) as q3
                    FROM %I
                    WHERE %I IS NOT NULL
                ),
                iqr_analysis AS (
                    SELECT 
                        id::VARCHAR as record_id,
                        %I as value,
                        s.q1,
                        s.q3,
                        (s.q3 - s.q1) as iqr,
                        CASE 
                            WHEN %I < (s.q1 - 1.5 * (s.q3 - s.q1)) THEN (%I - (s.q1 - 1.5 * (s.q3 - s.q1)))
                            WHEN %I > (s.q3 + 1.5 * (s.q3 - s.q1)) THEN (%I - (s.q3 + 1.5 * (s.q3 - s.q1)))
                            ELSE 0
                        END as outlier_distance
                    FROM %I, stats s
                    WHERE %I IS NOT NULL
                )
                SELECT 
                    record_id,
                    value,
                    ABS(outlier_distance) as outlier_score,
                    ''IQR'' as outlier_method,
                    (outlier_distance != 0) as is_outlier
                FROM iqr_analysis
                ORDER BY ABS(outlier_distance) DESC',
                p_column_name, p_column_name, p_table_name, p_column_name,
                p_column_name, p_column_name, p_column_name, p_column_name, p_column_name,
                p_table_name, p_column_name
            );
            
        WHEN 'Z_SCORE' THEN
            query_text := format('
                WITH stats AS (
                    SELECT 
                        AVG(%I) as mean_val,
                        STDDEV(%I) as std_val
                    FROM %I
                    WHERE %I IS NOT NULL
                ),
                z_score_analysis AS (
                    SELECT 
                        id::VARCHAR as record_id,
                        %I as value,
                        ABS((%I - s.mean_val) / NULLIF(s.std_val, 0)) as z_score
                    FROM %I, stats s
                    WHERE %I IS NOT NULL
                )
                SELECT 
                    record_id,
                    value,
                    z_score as outlier_score,
                    ''Z_SCORE'' as outlier_method,
                    (z_score > 3) as is_outlier
                FROM z_score_analysis
                ORDER BY z_score DESC',
                p_column_name, p_column_name, p_table_name, p_column_name,
                p_column_name, p_column_name, p_table_name, p_column_name
            );
            
        WHEN 'MODIFIED_Z_SCORE' THEN
            query_text := format('
                WITH stats AS (
                    SELECT 
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY %I) as median_val,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(%I - (
                            SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY %I) FROM %I
                        ))) as mad_val
                    FROM %I
                    WHERE %I IS NOT NULL
                ),
                modified_z_analysis AS (
                    SELECT 
                        id::VARCHAR as record_id,
                        %I as value,
                        0.6745 * ABS(%I - s.median_val) / NULLIF(s.mad_val, 0) as modified_z_score
                    FROM %I, stats s
                    WHERE %I IS NOT NULL
                )
                SELECT 
                    record_id,
                    value,
                    modified_z_score as outlier_score,
                    ''MODIFIED_Z_SCORE'' as outlier_method,
                    (modified_z_score > 3.5) as is_outlier
                FROM modified_z_analysis
                ORDER BY modified_z_score DESC',
                p_column_name, p_column_name, p_column_name, p_table_name,
                p_table_name, p_column_name,
                p_column_name, p_column_name, p_table_name, p_column_name
            );
    END CASE;
    
    RETURN QUERY EXECUTE query_text;
END;
$$ LANGUAGE plpgsql;

-- Outlier visualization summary
CREATE OR REPLACE FUNCTION outlier_summary_report(
    p_table_name VARCHAR,
    p_numeric_columns TEXT[]
)
RETURNS TABLE (
    column_name VARCHAR,
    total_records BIGINT,
    outlier_count_iqr BIGINT,
    outlier_count_zscore BIGINT,
    outlier_percentage_iqr NUMERIC,
    outlier_percentage_zscore NUMERIC,
    max_outlier_score NUMERIC,
    recommended_action VARCHAR
) AS $$
DECLARE
    col_name TEXT;
BEGIN
    FOREACH col_name IN ARRAY p_numeric_columns LOOP
        RETURN QUERY
        WITH iqr_outliers AS (
            SELECT COUNT(*) as iqr_count, MAX(outlier_score) as max_iqr_score
            FROM detect_outliers(p_table_name, col_name, 'IQR')
            WHERE is_outlier = TRUE
        ),
        zscore_outliers AS (
            SELECT COUNT(*) as zscore_count, MAX(outlier_score) as max_zscore_score
            FROM detect_outliers(p_table_name, col_name, 'Z_SCORE')
            WHERE is_outlier = TRUE
        ),
        total_count AS (
            SELECT COUNT(*) as total_records
            FROM detect_outliers(p_table_name, col_name, 'IQR')
        )
        SELECT 
            col_name as column_name,
            tc.total_records,
            io.iqr_count as outlier_count_iqr,
            zo.zscore_count as outlier_count_zscore,
            ROUND((io.iqr_count::NUMERIC / tc.total_records) * 100, 2) as outlier_percentage_iqr,
            ROUND((zo.zscore_count::NUMERIC / tc.total_records) * 100, 2) as outlier_percentage_zscore,
            GREATEST(io.max_iqr_score, zo.max_zscore_score) as max_outlier_score,
            CASE 
                WHEN (io.iqr_count::NUMERIC / tc.total_records) > 0.10 THEN 'INVESTIGATE_DATA_QUALITY'
                WHEN (io.iqr_count::NUMERIC / tc.total_records) > 0.05 THEN 'CONSIDER_TRANSFORMATION'
                WHEN (io.iqr_count::NUMERIC / tc.total_records) > 0.01 THEN 'MONITOR_OUTLIERS'
                ELSE 'NO_ACTION_NEEDED'
            END as recommended_action
        FROM iqr_outliers io, zscore_outliers zo, total_count tc;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

### Real-World Applications

1. **Credit Scoring**: Analyze income and debt distributions before model training
2. **Medical Research**: Examine patient vital signs and test result distributions
3. **Marketing Analytics**: Understand customer behavior and spending patterns
4. **Manufacturing**: Analyze sensor data distributions for quality control
5. **Financial Trading**: Examine price movement and volume distributions

### Best Practices

1. **Sample Large Datasets**: Use statistical sampling for initial exploration
2. **Multiple Visualizations**: Generate both histograms and summary statistics
3. **Outlier Investigation**: Always investigate outliers before removal
4. **Domain Knowledge**: Combine statistical analysis with business understanding
5. **Preprocessing Pipeline**: Document distribution findings for model preprocessing

### Common Pitfalls

1. **Ignoring Data Types**: Mixing categorical and numerical analysis
2. **Over-Binning**: Too many bins losing important patterns
3. **Under-Sampling**: Missing important distribution tails
4. **False Outliers**: Flagging valid extreme values as outliers
5. **Static Analysis**: Not updating distribution analysis as data changes

### Debugging and Optimization

```sql
-- Distribution analysis performance monitor
CREATE VIEW distribution_analysis_performance AS
SELECT 
    table_name,
    column_name,
    analysis_type,
    execution_time_ms,
    records_analyzed,
    analysis_date
FROM distribution_analysis_log
WHERE analysis_date >= CURRENT_DATE - 7
ORDER BY execution_time_ms DESC;

-- Automated distribution monitoring
CREATE OR REPLACE FUNCTION monitor_data_distributions()
RETURNS VOID AS $$
DECLARE
    table_info RECORD;
BEGIN
    FOR table_info IN 
        SELECT table_name, column_name, data_type
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND data_type IN ('integer', 'numeric', 'real', 'double precision')
    LOOP
        -- Check for distribution drift
        INSERT INTO distribution_drift_alerts (
            table_name, column_name, alert_type, created_at
        )
        SELECT 
            table_info.table_name,
            table_info.column_name,
            'DISTRIBUTION_CHANGE',
            CURRENT_TIMESTAMP
        FROM generate_distribution_summary(table_info.table_name, table_info.column_name)
        WHERE ABS(skewness) > 2 OR kurtosis > 5;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

### Performance Considerations

- **Sampling Strategy**: Use TABLESAMPLE for large datasets (>1M rows)
- **Index Optimization**: Create indexes on analyzed columns for faster processing
- **Memory Usage**: Monitor memory consumption during large histogram generation
- **Parallel Processing**: Utilize parallel query execution for multiple columns
- **Caching Results**: Store distribution summaries for frequently analyzed data

---

## Question 13

**Can SQL be used to visualize false positives and false negatives in classification models? If so, how?**

**Answer:**

### Theory

SQL can effectively visualize and analyze false positives (FP) and false negatives (FN) in classification models through confusion matrices, error analysis, and performance metric calculations. Understanding these classification errors is crucial for model improvement, business impact assessment, and decision threshold optimization. SQL provides powerful analytical capabilities to segment, analyze, and visualize these errors across different dimensions.

### Classification Error Analysis Framework

#### 1. Confusion Matrix Generation and Visualization

```sql
-- Comprehensive confusion matrix with detailed metrics
CREATE OR REPLACE FUNCTION generate_confusion_matrix_detailed(
    p_model_name VARCHAR,
    p_threshold NUMERIC DEFAULT 0.5,
    p_start_date DATE DEFAULT CURRENT_DATE - 30,
    p_end_date DATE DEFAULT CURRENT_DATE
)
RETURNS TABLE (
    confusion_matrix JSONB,
    classification_metrics JSONB,
    error_breakdown JSONB
) AS $$
DECLARE
    confusion_data JSONB;
    metrics_data JSONB;
    error_data JSONB;
BEGIN
    -- Generate confusion matrix
    WITH predictions AS (
        SELECT 
            actual_value,
            CASE WHEN prediction_score >= p_threshold THEN 1 ELSE 0 END as predicted_value,
            prediction_score,
            record_id,
            prediction_timestamp
        FROM ml_predictions 
        WHERE model_name = p_model_name
        AND prediction_timestamp BETWEEN p_start_date AND p_end_date
        AND actual_value IS NOT NULL
    ),
    confusion_counts AS (
        SELECT 
            COUNT(CASE WHEN actual_value = 1 AND predicted_value = 1 THEN 1 END) as true_positive,
            COUNT(CASE WHEN actual_value = 0 AND predicted_value = 1 THEN 1 END) as false_positive,
            COUNT(CASE WHEN actual_value = 1 AND predicted_value = 0 THEN 1 END) as false_negative,
            COUNT(CASE WHEN actual_value = 0 AND predicted_value = 0 THEN 1 END) as true_negative,
            COUNT(*) as total_predictions
        FROM predictions
    ),
    metrics AS (
        SELECT 
            cc.*,
            -- Precision = TP / (TP + FP)
            ROUND(cc.true_positive::NUMERIC / NULLIF(cc.true_positive + cc.false_positive, 0), 4) as precision,
            -- Recall (Sensitivity) = TP / (TP + FN)
            ROUND(cc.true_positive::NUMERIC / NULLIF(cc.true_positive + cc.false_negative, 0), 4) as recall,
            -- Specificity = TN / (TN + FP)
            ROUND(cc.true_negative::NUMERIC / NULLIF(cc.true_negative + cc.false_positive, 0), 4) as specificity,
            -- Accuracy = (TP + TN) / Total
            ROUND((cc.true_positive + cc.true_negative)::NUMERIC / cc.total_predictions, 4) as accuracy,
            -- F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
            ROUND(2.0 * 
                (cc.true_positive::NUMERIC / NULLIF(cc.true_positive + cc.false_positive, 0)) * 
                (cc.true_positive::NUMERIC / NULLIF(cc.true_positive + cc.false_negative, 0)) /
                NULLIF(
                    (cc.true_positive::NUMERIC / NULLIF(cc.true_positive + cc.false_positive, 0)) + 
                    (cc.true_positive::NUMERIC / NULLIF(cc.true_positive + cc.false_negative, 0)), 0
                ), 4) as f1_score,
            -- False Positive Rate = FP / (FP + TN)
            ROUND(cc.false_positive::NUMERIC / NULLIF(cc.false_positive + cc.true_negative, 0), 4) as false_positive_rate,
            -- False Negative Rate = FN / (FN + TP)
            ROUND(cc.false_negative::NUMERIC / NULLIF(cc.false_negative + cc.true_positive, 0), 4) as false_negative_rate
        FROM confusion_counts cc
    )
    SELECT 
        -- Confusion Matrix as JSON
        jsonb_build_object(
            'true_positive', m.true_positive,
            'false_positive', m.false_positive,
            'false_negative', m.false_negative,
            'true_negative', m.true_negative,
            'total_predictions', m.total_predictions,
            'threshold_used', p_threshold
        ) as confusion_matrix,
        
        -- Classification Metrics as JSON
        jsonb_build_object(
            'accuracy', m.accuracy,
            'precision', m.precision,
            'recall', m.recall,
            'specificity', m.specificity,
            'f1_score', m.f1_score,
            'false_positive_rate', m.false_positive_rate,
            'false_negative_rate', m.false_negative_rate
        ) as classification_metrics,
        
        -- Error Breakdown as JSON
        jsonb_build_object(
            'false_positive_count', m.false_positive,
            'false_negative_count', m.false_negative,
            'false_positive_percentage', ROUND((m.false_positive::NUMERIC / m.total_predictions) * 100, 2),
            'false_negative_percentage', ROUND((m.false_negative::NUMERIC / m.total_predictions) * 100, 2),
            'total_errors', m.false_positive + m.false_negative,
            'error_rate', ROUND(((m.false_positive + m.false_negative)::NUMERIC / m.total_predictions) * 100, 2)
        ) as error_breakdown
        
    INTO confusion_data, metrics_data, error_data
    FROM metrics m;
    
    confusion_matrix := confusion_data;
    classification_metrics := metrics_data;
    error_breakdown := error_data;
    
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Visual confusion matrix representation
CREATE OR REPLACE FUNCTION visualize_confusion_matrix(
    p_model_name VARCHAR,
    p_threshold NUMERIC DEFAULT 0.5
)
RETURNS TABLE (
    matrix_row VARCHAR,
    predicted_negative VARCHAR,
    predicted_positive VARCHAR,
    row_total VARCHAR
) AS $$
DECLARE
    tp BIGINT; fp BIGINT; fn BIGINT; tn BIGINT;
BEGIN
    -- Get confusion matrix values
    SELECT 
        (confusion_matrix->>'true_positive')::BIGINT,
        (confusion_matrix->>'false_positive')::BIGINT,
        (confusion_matrix->>'false_negative')::BIGINT,
        (confusion_matrix->>'true_negative')::BIGINT
    INTO tp, fp, fn, tn
    FROM generate_confusion_matrix_detailed(p_model_name, p_threshold);
    
    -- Return formatted matrix
    RETURN QUERY VALUES
        ('Actual Negative', 
         format('TN: %s', tn), 
         format('FP: %s ❌', fp), 
         format('%s', tn + fp)),
        ('Actual Positive', 
         format('FN: %s ❌', fn), 
         format('TP: %s ✅', tp), 
         format('%s', fn + tp)),
        ('Column Total', 
         format('%s', tn + fn), 
         format('%s', fp + tp), 
         format('%s', tp + fp + fn + tn));
END;
$$ LANGUAGE plpgsql;
```

#### 2. False Positive Analysis and Segmentation

```sql
-- Detailed false positive analysis
CREATE OR REPLACE FUNCTION analyze_false_positives(
    p_model_name VARCHAR,
    p_segment_columns TEXT[] DEFAULT ARRAY['customer_segment', 'risk_category', 'geography']
)
RETURNS TABLE (
    segment_name VARCHAR,
    segment_value TEXT,
    total_predictions BIGINT,
    false_positive_count BIGINT,
    false_positive_rate NUMERIC,
    avg_confidence_fp NUMERIC,
    business_impact_score NUMERIC
) AS $$
DECLARE
    col_name TEXT;
BEGIN
    FOREACH col_name IN ARRAY p_segment_columns LOOP
        RETURN QUERY
        EXECUTE format('
            WITH false_positives AS (
                SELECT 
                    p.record_id,
                    p.prediction_score,
                    f.%I as segment_val
                FROM ml_predictions p
                JOIN feature_data f ON p.record_id = f.record_id
                WHERE p.model_name = ''%s''
                AND p.actual_value = 0 
                AND p.prediction_score >= 0.5
            ),
            segment_stats AS (
                SELECT 
                    f.%I as segment_val,
                    COUNT(p.record_id) as total_preds,
                    COUNT(fp.record_id) as fp_count,
                    AVG(CASE WHEN fp.record_id IS NOT NULL THEN fp.prediction_score END) as avg_fp_confidence
                FROM feature_data f
                LEFT JOIN ml_predictions p ON f.record_id = p.record_id AND p.model_name = ''%s''
                LEFT JOIN false_positives fp ON f.record_id = fp.record_id
                WHERE p.record_id IS NOT NULL
                GROUP BY f.%I
            )
            SELECT 
                ''%s''::VARCHAR as segment_name,
                segment_val::TEXT as segment_value,
                total_preds as total_predictions,
                COALESCE(fp_count, 0) as false_positive_count,
                ROUND(COALESCE(fp_count, 0)::NUMERIC / NULLIF(total_preds, 0) * 100, 2) as false_positive_rate,
                ROUND(avg_fp_confidence, 4) as avg_confidence_fp,
                -- Business impact calculation (example: false positives in high-value segments are more costly)
                ROUND(
                    COALESCE(fp_count, 0) * 
                    CASE 
                        WHEN segment_val ILIKE ''%%high%%'' OR segment_val ILIKE ''%%premium%%'' THEN 100
                        WHEN segment_val ILIKE ''%%medium%%'' THEN 50
                        ELSE 25
                    END, 2
                ) as business_impact_score
            FROM segment_stats
            ORDER BY false_positive_rate DESC',
            col_name, p_model_name, col_name, p_model_name, col_name, col_name
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- False positive pattern detection
CREATE OR REPLACE FUNCTION detect_fp_patterns(
    p_model_name VARCHAR,
    p_feature_columns TEXT[]
)
RETURNS TABLE (
    pattern_description VARCHAR,
    feature_combination JSONB,
    fp_count BIGINT,
    total_count BIGINT,
    fp_concentration NUMERIC,
    suggested_action VARCHAR
) AS $$
DECLARE
    feature_combo TEXT;
    pattern_sql TEXT;
BEGIN
    -- Analyze feature combinations that lead to false positives
    FOREACH feature_combo IN ARRAY p_feature_columns LOOP
        pattern_sql := format('
            WITH feature_patterns AS (
                SELECT 
                    %I as feature_value,
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN actual_value = 0 AND prediction_score >= 0.5 THEN 1 END) as fp_count
                FROM ml_predictions p
                JOIN feature_data f ON p.record_id = f.record_id
                WHERE p.model_name = ''%s''
                AND p.actual_value IS NOT NULL
                GROUP BY %I
                HAVING COUNT(*) >= 10
            )
            SELECT 
                format(''High FP rate in %s = %%s'', feature_value) as pattern_description,
                jsonb_build_object(''%s'', feature_value) as feature_combination,
                fp_count,
                total_predictions as total_count,
                ROUND((fp_count::NUMERIC / total_predictions) * 100, 2) as fp_concentration,
                CASE 
                    WHEN (fp_count::NUMERIC / total_predictions) > 0.30 THEN ''EXCLUDE_FEATURE_VALUE''
                    WHEN (fp_count::NUMERIC / total_predictions) > 0.20 THEN ''ADDITIONAL_FEATURES''
                    WHEN (fp_count::NUMERIC / total_predictions) > 0.10 THEN ''THRESHOLD_ADJUSTMENT''
                    ELSE ''MONITOR''
                END as suggested_action
            FROM feature_patterns
            WHERE (fp_count::NUMERIC / total_predictions) > 0.05
            ORDER BY fp_concentration DESC',
            feature_combo, p_model_name, feature_combo, feature_combo, feature_combo
        );
        
        RETURN QUERY EXECUTE pattern_sql;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

#### 3. False Negative Analysis and Impact Assessment

```sql
-- Comprehensive false negative analysis
CREATE OR REPLACE FUNCTION analyze_false_negatives(
    p_model_name VARCHAR,
    p_impact_weight_column VARCHAR DEFAULT 'transaction_amount'
)
RETURNS TABLE (
    analysis_type VARCHAR,
    fn_count BIGINT,
    avg_confidence NUMERIC,
    missed_opportunity_value NUMERIC,
    severity_category VARCHAR,
    recommendations TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH false_negatives AS (
        SELECT 
            p.record_id,
            p.prediction_score,
            p.actual_value,
            f.transaction_amount,
            f.customer_value_score,
            f.risk_category
        FROM ml_predictions p
        JOIN feature_data f ON p.record_id = f.record_id
        WHERE p.model_name = p_model_name
        AND p.actual_value = 1
        AND p.prediction_score < 0.5
    ),
    fn_analysis AS (
        SELECT 
            'Overall' as analysis_type,
            COUNT(*) as fn_count,
            AVG(prediction_score) as avg_confidence,
            COALESCE(SUM(transaction_amount), 0) as missed_value,
            CASE 
                WHEN COUNT(*) > 100 THEN 'HIGH'
                WHEN COUNT(*) > 50 THEN 'MEDIUM'
                ELSE 'LOW'
            END as severity
        FROM false_negatives
        
        UNION ALL
        
        SELECT 
            'High Value Customers',
            COUNT(*),
            AVG(prediction_score),
            COALESCE(SUM(transaction_amount), 0),
            CASE 
                WHEN COUNT(*) > 20 THEN 'CRITICAL'
                WHEN COUNT(*) > 10 THEN 'HIGH'
                ELSE 'MEDIUM'
            END
        FROM false_negatives 
        WHERE customer_value_score > 80
        
        UNION ALL
        
        SELECT 
            'High Risk Category',
            COUNT(*),
            AVG(prediction_score),
            COALESCE(SUM(transaction_amount), 0),
            CASE 
                WHEN COUNT(*) > 15 THEN 'CRITICAL'
                WHEN COUNT(*) > 5 THEN 'HIGH'
                ELSE 'MEDIUM'
            END
        FROM false_negatives 
        WHERE risk_category = 'HIGH'
    )
    SELECT 
        fa.analysis_type,
        fa.fn_count,
        ROUND(fa.avg_confidence, 4) as avg_confidence,
        ROUND(fa.missed_value, 2) as missed_opportunity_value,
        fa.severity as severity_category,
        CASE fa.analysis_type
            WHEN 'Overall' THEN 
                CASE 
                    WHEN fa.severity = 'HIGH' THEN ARRAY['Lower prediction threshold', 'Add more training data', 'Feature engineering']
                    WHEN fa.severity = 'MEDIUM' THEN ARRAY['Threshold optimization', 'Model ensemble']
                    ELSE ARRAY['Monitor trends', 'Periodic retraining']
                END
            WHEN 'High Value Customers' THEN 
                ARRAY['Customer-specific models', 'Behavioral features', 'Interaction terms']
            WHEN 'High Risk Category' THEN 
                ARRAY['Risk-specific features', 'External data sources', 'Expert rules']
            ELSE ARRAY['General optimization']
        END as recommendations
    FROM fn_analysis fa;
END;
$$ LANGUAGE plpgsql;

-- Time-based error trend analysis
CREATE OR REPLACE FUNCTION error_trend_analysis(
    p_model_name VARCHAR,
    p_days_back INT DEFAULT 30
)
RETURNS TABLE (
    analysis_date DATE,
    total_predictions BIGINT,
    false_positive_count BIGINT,
    false_negative_count BIGINT,
    fp_rate NUMERIC,
    fn_rate NUMERIC,
    error_trend VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    WITH daily_errors AS (
        SELECT 
            DATE(prediction_timestamp) as pred_date,
            COUNT(*) as total_preds,
            COUNT(CASE WHEN actual_value = 0 AND prediction_score >= 0.5 THEN 1 END) as fp_count,
            COUNT(CASE WHEN actual_value = 1 AND prediction_score < 0.5 THEN 1 END) as fn_count
        FROM ml_predictions
        WHERE model_name = p_model_name
        AND prediction_timestamp >= CURRENT_DATE - p_days_back
        AND actual_value IS NOT NULL
        GROUP BY DATE(prediction_timestamp)
    ),
    trend_analysis AS (
        SELECT 
            pred_date,
            total_preds,
            fp_count,
            fn_count,
            ROUND((fp_count::NUMERIC / NULLIF(total_preds, 0)) * 100, 2) as fp_rate,
            ROUND((fn_count::NUMERIC / NULLIF(total_preds, 0)) * 100, 2) as fn_rate,
            LAG(fp_count + fn_count) OVER (ORDER BY pred_date) as prev_errors
        FROM daily_errors
    )
    SELECT 
        pred_date as analysis_date,
        total_preds as total_predictions,
        fp_count as false_positive_count,
        fn_count as false_negative_count,
        fp_rate,
        fn_rate,
        CASE 
            WHEN prev_errors IS NULL THEN 'BASELINE'
            WHEN (fp_count + fn_count) > prev_errors * 1.2 THEN 'DETERIORATING'
            WHEN (fp_count + fn_count) < prev_errors * 0.8 THEN 'IMPROVING'
            ELSE 'STABLE'
        END as error_trend
    FROM trend_analysis
    ORDER BY pred_date;
END;
$$ LANGUAGE plpgsql;
```

#### 4. Interactive Error Visualization Dashboard

```sql
-- Create comprehensive error visualization view
CREATE OR REPLACE VIEW classification_error_dashboard AS
WITH model_summary AS (
    SELECT 
        model_name,
        COUNT(*) as total_predictions,
        COUNT(CASE WHEN actual_value = 0 AND prediction_score >= 0.5 THEN 1 END) as false_positives,
        COUNT(CASE WHEN actual_value = 1 AND prediction_score < 0.5 THEN 1 END) as false_negatives,
        ROUND(AVG(CASE WHEN actual_value = prediction_class THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy
    FROM (
        SELECT 
            model_name,
            actual_value,
            prediction_score,
            CASE WHEN prediction_score >= 0.5 THEN 1 ELSE 0 END as prediction_class
        FROM ml_predictions
        WHERE prediction_timestamp >= CURRENT_DATE - 7
        AND actual_value IS NOT NULL
    ) classified_predictions
    GROUP BY model_name
),
error_rates AS (
    SELECT 
        ms.model_name,
        ms.total_predictions,
        ms.false_positives,
        ms.false_negatives,
        ms.accuracy,
        ROUND((ms.false_positives::NUMERIC / ms.total_predictions) * 100, 2) as fp_rate,
        ROUND((ms.false_negatives::NUMERIC / ms.total_predictions) * 100, 2) as fn_rate,
        (ms.false_positives + ms.false_negatives) as total_errors
    FROM model_summary ms
)
SELECT 
    model_name,
    total_predictions,
    accuracy,
    false_positives,
    false_negatives,
    total_errors,
    fp_rate,
    fn_rate,
    CASE 
        WHEN fp_rate > fn_rate * 2 THEN 'FP_DOMINATED'
        WHEN fn_rate > fp_rate * 2 THEN 'FN_DOMINATED'
        ELSE 'BALANCED_ERRORS'
    END as error_profile,
    CASE 
        WHEN total_errors > total_predictions * 0.15 THEN 'CRITICAL'
        WHEN total_errors > total_predictions * 0.10 THEN 'HIGH'
        WHEN total_errors > total_predictions * 0.05 THEN 'MEDIUM'
        ELSE 'LOW'
    END as error_severity
FROM error_rates
ORDER BY total_errors DESC;

-- ROC curve data generation for visualization
CREATE OR REPLACE FUNCTION generate_roc_curve_data(
    p_model_name VARCHAR,
    p_num_points INT DEFAULT 100
)
RETURNS TABLE (
    threshold NUMERIC,
    true_positive_rate NUMERIC,
    false_positive_rate NUMERIC,
    precision_value NUMERIC,
    f1_score NUMERIC
) AS $$
DECLARE
    min_score NUMERIC;
    max_score NUMERIC;
    step_size NUMERIC;
    current_threshold NUMERIC;
BEGIN
    -- Get score range
    SELECT MIN(prediction_score), MAX(prediction_score)
    INTO min_score, max_score
    FROM ml_predictions 
    WHERE model_name = p_model_name AND actual_value IS NOT NULL;
    
    step_size := (max_score - min_score) / p_num_points;
    
    FOR i IN 0..p_num_points LOOP
        current_threshold := min_score + (i * step_size);
        
        RETURN QUERY
        WITH threshold_results AS (
            SELECT 
                COUNT(CASE WHEN actual_value = 1 AND prediction_score >= current_threshold THEN 1 END) as tp,
                COUNT(CASE WHEN actual_value = 0 AND prediction_score >= current_threshold THEN 1 END) as fp,
                COUNT(CASE WHEN actual_value = 1 AND prediction_score < current_threshold THEN 1 END) as fn,
                COUNT(CASE WHEN actual_value = 0 AND prediction_score < current_threshold THEN 1 END) as tn
            FROM ml_predictions
            WHERE model_name = p_model_name AND actual_value IS NOT NULL
        )
        SELECT 
            current_threshold as threshold,
            ROUND(tr.tp::NUMERIC / NULLIF(tr.tp + tr.fn, 0), 4) as true_positive_rate,
            ROUND(tr.fp::NUMERIC / NULLIF(tr.fp + tr.tn, 0), 4) as false_positive_rate,
            ROUND(tr.tp::NUMERIC / NULLIF(tr.tp + tr.fp, 0), 4) as precision_value,
            ROUND(2.0 * (tr.tp::NUMERIC / NULLIF(tr.tp + tr.fp, 0)) * (tr.tp::NUMERIC / NULLIF(tr.tp + tr.fn, 0)) / 
                  NULLIF((tr.tp::NUMERIC / NULLIF(tr.tp + tr.fp, 0)) + (tr.tp::NUMERIC / NULLIF(tr.tp + tr.fn, 0)), 0), 4) as f1_score
        FROM threshold_results tr;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

### Real-World Applications

1. **Medical Diagnosis**: Analyze false negatives (missed diseases) vs false positives (unnecessary treatments)
2. **Fraud Detection**: Balance false alarms (annoying customers) vs missed fraud (financial loss)
3. **Credit Scoring**: Assess wrongly rejected applications vs bad loans approved
4. **Marketing Campaigns**: Evaluate wasted spend (FP) vs missed opportunities (FN)
5. **Quality Control**: Analyze defective products missed vs good products rejected

### Best Practices

1. **Business Context**: Weight FP and FN errors based on business impact
2. **Threshold Optimization**: Use ROC curves to find optimal decision thresholds
3. **Segment Analysis**: Analyze errors across different customer/product segments
4. **Temporal Monitoring**: Track error patterns over time for model degradation
5. **Root Cause Analysis**: Investigate feature patterns in misclassified examples

### Common Pitfalls

1. **Equal Error Weighting**: Treating all FP and FN as equally costly
2. **Static Thresholds**: Not adjusting decision thresholds based on changing conditions
3. **Ignoring Confidence**: Not considering prediction confidence in error analysis
4. **Sample Bias**: Analyzing errors on unrepresentative data samples
5. **Feature Leakage**: Missing data quality issues that cause systematic errors

### Debugging and Optimization

```sql
-- Error investigation toolkit
CREATE OR REPLACE FUNCTION investigate_classification_errors(
    p_model_name VARCHAR,
    p_error_type VARCHAR, -- 'FP', 'FN', 'BOTH'
    p_limit INT DEFAULT 100
)
RETURNS TABLE (
    record_id VARCHAR,
    actual_value INT,
    predicted_value INT,
    prediction_score NUMERIC,
    confidence_gap NUMERIC,
    feature_summary JSONB,
    error_severity VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    EXECUTE format('
        WITH error_records AS (
            SELECT 
                p.record_id,
                p.actual_value,
                CASE WHEN p.prediction_score >= 0.5 THEN 1 ELSE 0 END as predicted_value,
                p.prediction_score,
                ABS(p.prediction_score - p.actual_value) as confidence_gap,
                f.feature_values
            FROM ml_predictions p
            JOIN feature_data f ON p.record_id = f.record_id
            WHERE p.model_name = ''%s''
            AND p.actual_value IS NOT NULL
            AND (
                (''%s'' = ''FP'' AND p.actual_value = 0 AND p.prediction_score >= 0.5) OR
                (''%s'' = ''FN'' AND p.actual_value = 1 AND p.prediction_score < 0.5) OR
                (''%s'' = ''BOTH'' AND (
                    (p.actual_value = 0 AND p.prediction_score >= 0.5) OR
                    (p.actual_value = 1 AND p.prediction_score < 0.5)
                ))
            )
            ORDER BY confidence_gap DESC
            LIMIT %s
        )
        SELECT 
            record_id,
            actual_value,
            predicted_value,
            ROUND(prediction_score, 4) as prediction_score,
            ROUND(confidence_gap, 4) as confidence_gap,
            feature_values as feature_summary,
            CASE 
                WHEN confidence_gap > 0.8 THEN ''SEVERE''
                WHEN confidence_gap > 0.6 THEN ''HIGH''
                WHEN confidence_gap > 0.4 THEN ''MEDIUM''
                ELSE ''LOW''
            END as error_severity
        FROM error_records',
        p_model_name, p_error_type, p_error_type, p_error_type, p_limit
    );
END;
$$ LANGUAGE plpgsql;
```

### Performance Considerations

- **Index Strategy**: Create composite indexes on (model_name, actual_value, prediction_score)
- **Materialized Views**: Cache confusion matrix calculations for frequently accessed models
- **Sampling**: Use statistical sampling for error analysis on very large datasets
- **Parallel Processing**: Distribute error analysis across multiple database connections
- **Automated Alerts**: Set up triggers for significant changes in error rates

---

## Question 14

**What strategies might you use to automate the retraining and evaluation of Machine Learning models from within SQL?**

**Answer:**

### Theory

Automated ML model retraining and evaluation within SQL involves creating systematic workflows that monitor model performance, detect degradation, trigger retraining when necessary, and evaluate new model versions. This automation ensures models remain current with data patterns, maintains prediction quality, and reduces manual intervention while providing comprehensive audit trails and rollback capabilities.

### Automated ML Pipeline Framework

#### 1. Model Performance Monitoring and Triggers

```sql
-- Comprehensive model monitoring system
CREATE TABLE model_performance_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accuracy NUMERIC,
    precision_score NUMERIC,
    recall_score NUMERIC,
    f1_score NUMERIC,
    auc_roc NUMERIC,
    data_drift_score NUMERIC,
    prediction_volume BIGINT,
    avg_confidence NUMERIC,
    performance_trend VARCHAR(20), -- 'IMPROVING', 'STABLE', 'DEGRADING'
    evaluation_type VARCHAR(30) -- 'SCHEDULED', 'TRIGGERED', 'MANUAL'
);

-- Automated performance monitoring function
CREATE OR REPLACE FUNCTION monitor_model_performance()
RETURNS VOID AS $$
DECLARE
    model_record RECORD;
    current_metrics RECORD;
    baseline_metrics RECORD;
    performance_change NUMERIC;
    drift_detected BOOLEAN;
BEGIN
    -- Check each active model
    FOR model_record IN 
        SELECT DISTINCT model_name FROM ml_models WHERE is_active = TRUE
    LOOP
        -- Calculate current performance metrics
        SELECT INTO current_metrics
            COUNT(*) as total_predictions,
            AVG(CASE WHEN actual_value = predicted_value THEN 1.0 ELSE 0.0 END) as current_accuracy,
            AVG(CASE WHEN predicted_value = 1 AND actual_value = 1 THEN 1.0 
                     WHEN predicted_value = 1 THEN 0.0 END) as current_precision,
            AVG(CASE WHEN actual_value = 1 AND predicted_value = 1 THEN 1.0 
                     WHEN actual_value = 1 THEN 0.0 END) as current_recall,
            AVG(confidence_score) as avg_conf
        FROM ml_predictions 
        WHERE model_name = model_record.model_name 
        AND prediction_timestamp >= CURRENT_TIMESTAMP - INTERVAL '7 days'
        AND actual_value IS NOT NULL;
        
        -- Get baseline performance (from model creation/last retraining)
        SELECT INTO baseline_metrics
            accuracy, precision_score, recall_score
        FROM model_performance_metrics 
        WHERE model_name = model_record.model_name 
        AND evaluation_type IN ('BASELINE', 'RETRAINED')
        ORDER BY evaluation_date DESC 
        LIMIT 1;
        
        -- Calculate performance degradation
        performance_change := ABS(current_metrics.current_accuracy - baseline_metrics.accuracy);
        
        -- Check for data drift
        drift_detected := check_data_drift(model_record.model_name);
        
        -- Insert current performance metrics
        INSERT INTO model_performance_metrics (
            model_name, accuracy, precision_score, recall_score, 
            prediction_volume, avg_confidence, performance_trend, evaluation_type
        ) VALUES (
            model_record.model_name,
            current_metrics.current_accuracy,
            current_metrics.current_precision,
            current_metrics.current_recall,
            current_metrics.total_predictions,
            current_metrics.avg_conf,
            CASE 
                WHEN performance_change > 0.05 THEN 'DEGRADING'
                WHEN performance_change > 0.02 THEN 'STABLE'
                ELSE 'IMPROVING'
            END,
            'SCHEDULED'
        );
        
        -- Trigger retraining if necessary
        IF performance_change > 0.05 OR drift_detected OR current_metrics.total_predictions < 100 THEN
            PERFORM trigger_model_retraining(
                model_record.model_name, 
                CASE 
                    WHEN performance_change > 0.05 THEN 'PERFORMANCE_DEGRADATION'
                    WHEN drift_detected THEN 'DATA_DRIFT'
                    ELSE 'LOW_PREDICTION_VOLUME'
                END
            );
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Data drift detection function
CREATE OR REPLACE FUNCTION check_data_drift(p_model_name VARCHAR)
RETURNS BOOLEAN AS $$
DECLARE
    drift_score NUMERIC;
    threshold NUMERIC := 0.1;
BEGIN
    -- Simple statistical drift detection using feature mean comparisons
    WITH baseline_stats AS (
        SELECT 
            feature_name,
            AVG(feature_value) as baseline_mean,
            STDDEV(feature_value) as baseline_std
        FROM training_features tf
        JOIN model_metadata mm ON tf.training_session_id = mm.training_session_id
        WHERE mm.model_name = p_model_name
        GROUP BY feature_name
    ),
    current_stats AS (
        SELECT 
            feature_name,
            AVG(feature_value) as current_mean,
            STDDEV(feature_value) as current_std
        FROM ml_features mf
        WHERE mf.created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
        AND EXISTS (
            SELECT 1 FROM ml_predictions mp 
            WHERE mp.record_id = mf.record_id 
            AND mp.model_name = p_model_name
        )
        GROUP BY feature_name
    ),
    drift_analysis AS (
        SELECT 
            AVG(ABS(cs.current_mean - bs.baseline_mean) / NULLIF(bs.baseline_std, 0)) as avg_drift
        FROM baseline_stats bs
        JOIN current_stats cs ON bs.feature_name = cs.feature_name
    )
    SELECT avg_drift INTO drift_score FROM drift_analysis;
    
    RETURN COALESCE(drift_score > threshold, FALSE);
END;
$$ LANGUAGE plpgsql;
```

#### 2. Automated Retraining Workflow

```sql
-- Retraining orchestration system
CREATE TABLE retraining_jobs (
    job_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    trigger_reason VARCHAR(50),
    job_status VARCHAR(20) DEFAULT 'QUEUED', -- 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    training_data_count BIGINT,
    validation_accuracy NUMERIC,
    new_model_version VARCHAR(50),
    rollback_required BOOLEAN DEFAULT FALSE,
    error_message TEXT
);

-- Main retraining trigger function
CREATE OR REPLACE FUNCTION trigger_model_retraining(
    p_model_name VARCHAR,
    p_reason VARCHAR
)
RETURNS VOID AS $$
DECLARE
    job_id INT;
BEGIN
    -- Create retraining job
    INSERT INTO retraining_jobs (model_name, trigger_reason, started_at)
    VALUES (p_model_name, p_reason, CURRENT_TIMESTAMP)
    RETURNING retraining_jobs.job_id INTO job_id;
    
    -- Execute retraining workflow
    PERFORM execute_retraining_workflow(job_id);
    
    -- Log the trigger event
    INSERT INTO model_audit_log (
        model_name, event_type, event_description, created_at
    ) VALUES (
        p_model_name, 
        'RETRAINING_TRIGGERED', 
        format('Retraining triggered due to: %s', p_reason),
        CURRENT_TIMESTAMP
    );
END;
$$ LANGUAGE plpgsql;

-- Complete retraining workflow
CREATE OR REPLACE FUNCTION execute_retraining_workflow(p_job_id INT)
RETURNS VOID AS $$
DECLARE
    job_record RECORD;
    training_data_count BIGINT;
    new_version VARCHAR(50);
    validation_results RECORD;
    deployment_success BOOLEAN;
BEGIN
    -- Get job details
    SELECT * INTO job_record FROM retraining_jobs WHERE job_id = p_job_id;
    
    -- Update job status
    UPDATE retraining_jobs SET job_status = 'RUNNING' WHERE job_id = p_job_id;
    
    BEGIN
        -- Step 1: Prepare training data
        PERFORM prepare_training_data(job_record.model_name);
        
        -- Step 2: Get training data count
        SELECT COUNT(*) INTO training_data_count
        FROM training_data_staging 
        WHERE model_name = job_record.model_name;
        
        -- Step 3: Generate new model version
        new_version := format('%s_v%s', job_record.model_name, 
                            TO_CHAR(CURRENT_TIMESTAMP, 'YYYYMMDDHH24MISS'));
        
        -- Step 4: Execute training algorithm
        PERFORM execute_training_algorithm(job_record.model_name, new_version);
        
        -- Step 5: Validate new model
        SELECT * INTO validation_results 
        FROM validate_new_model(job_record.model_name, new_version);
        
        -- Step 6: Deploy if validation passes
        IF validation_results.accuracy >= 0.75 AND 
           validation_results.accuracy >= (
               SELECT MAX(accuracy) * 0.95 
               FROM model_performance_metrics 
               WHERE model_name = job_record.model_name
           ) THEN
            
            -- Deploy new model
            PERFORM deploy_model_version(job_record.model_name, new_version);
            deployment_success := TRUE;
            
        ELSE
            -- Model failed validation
            deployment_success := FALSE;
            UPDATE retraining_jobs 
            SET rollback_required = TRUE,
                error_message = 'Model failed validation criteria'
            WHERE job_id = p_job_id;
        END IF;
        
        -- Step 7: Update job completion
        UPDATE retraining_jobs 
        SET job_status = 'COMPLETED',
            completed_at = CURRENT_TIMESTAMP,
            training_data_count = training_data_count,
            validation_accuracy = validation_results.accuracy,
            new_model_version = new_version
        WHERE job_id = p_job_id;
        
        -- Step 8: Log completion
        INSERT INTO model_audit_log (
            model_name, event_type, event_description, created_at
        ) VALUES (
            job_record.model_name,
            CASE WHEN deployment_success THEN 'RETRAINING_COMPLETED' ELSE 'RETRAINING_FAILED' END,
            format('Job %s completed. New version: %s, Accuracy: %s, Deployed: %s',
                   p_job_id, new_version, validation_results.accuracy, deployment_success),
            CURRENT_TIMESTAMP
        );
        
    EXCEPTION WHEN OTHERS THEN
        -- Handle training failure
        UPDATE retraining_jobs 
        SET job_status = 'FAILED',
            completed_at = CURRENT_TIMESTAMP,
            error_message = SQLERRM
        WHERE job_id = p_job_id;
        
        -- Log error
        INSERT INTO model_audit_log (
            model_name, event_type, event_description, created_at
        ) VALUES (
            job_record.model_name,
            'RETRAINING_ERROR',
            format('Job %s failed: %s', p_job_id, SQLERRM),
            CURRENT_TIMESTAMP
        );
        
        RAISE;
    END;
END;
$$ LANGUAGE plpgsql;

-- Training data preparation
CREATE OR REPLACE FUNCTION prepare_training_data(p_model_name VARCHAR)
RETURNS VOID AS $$
BEGIN
    -- Clear staging area
    DELETE FROM training_data_staging WHERE model_name = p_model_name;
    
    -- Prepare fresh training dataset
    INSERT INTO training_data_staging (
        model_name, record_id, feature_values, target_value, 
        data_source, created_at, weight
    )
    SELECT 
        p_model_name,
        f.record_id,
        f.feature_values,
        t.target_value,
        'PRODUCTION',
        f.created_at,
        -- Weight recent data more heavily
        CASE 
            WHEN f.created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days' THEN 1.0
            WHEN f.created_at >= CURRENT_TIMESTAMP - INTERVAL '90 days' THEN 0.8
            WHEN f.created_at >= CURRENT_TIMESTAMP - INTERVAL '180 days' THEN 0.6
            ELSE 0.4
        END as weight
    FROM ml_features f
    JOIN target_values t ON f.record_id = t.record_id
    WHERE f.created_at >= CURRENT_TIMESTAMP - INTERVAL '365 days'  -- Use last year of data
    AND t.target_value IS NOT NULL
    AND f.feature_values IS NOT NULL;
    
    -- Add data quality checks
    DELETE FROM training_data_staging 
    WHERE model_name = p_model_name
    AND (
        feature_values IS NULL OR
        target_value IS NULL OR
        jsonb_array_length(feature_values) = 0
    );
    
    -- Log data preparation
    INSERT INTO model_audit_log (
        model_name, event_type, event_description, created_at
    ) VALUES (
        p_model_name,
        'DATA_PREPARED',
        format('Training data prepared: %s records', 
               (SELECT COUNT(*) FROM training_data_staging WHERE model_name = p_model_name)),
        CURRENT_TIMESTAMP
    );
END;
$$ LANGUAGE plpgsql;
```

#### 3. Automated Model Evaluation and Validation

```sql
-- Comprehensive model validation framework
CREATE OR REPLACE FUNCTION validate_new_model(
    p_model_name VARCHAR,
    p_new_version VARCHAR
)
RETURNS TABLE (
    accuracy NUMERIC,
    precision_score NUMERIC,
    recall_score NUMERIC,
    f1_score NUMERIC,
    auc_roc NUMERIC,
    validation_passed BOOLEAN,
    business_metrics JSONB
) AS $$
DECLARE
    validation_results RECORD;
    business_impact JSONB;
BEGIN
    -- Create holdout validation set
    WITH validation_data AS (
        SELECT * FROM training_data_staging 
        WHERE model_name = p_model_name 
        AND MOD(ABS(HASHTEXT(record_id)), 10) < 2  -- 20% holdout
    ),
    model_predictions AS (
        SELECT 
            vd.record_id,
            vd.target_value as actual_value,
            predict_with_model(p_new_version, vd.feature_values) as predicted_score,
            CASE WHEN predict_with_model(p_new_version, vd.feature_values) >= 0.5 THEN 1 ELSE 0 END as predicted_class
        FROM validation_data vd
    ),
    performance_metrics AS (
        SELECT 
            COUNT(*) as total_predictions,
            AVG(CASE WHEN actual_value = predicted_class THEN 1.0 ELSE 0.0 END) as accuracy_val,
            
            -- Precision = TP / (TP + FP)
            COUNT(CASE WHEN predicted_class = 1 AND actual_value = 1 THEN 1 END)::NUMERIC /
            NULLIF(COUNT(CASE WHEN predicted_class = 1 THEN 1 END), 0) as precision_val,
            
            -- Recall = TP / (TP + FN)
            COUNT(CASE WHEN predicted_class = 1 AND actual_value = 1 THEN 1 END)::NUMERIC /
            NULLIF(COUNT(CASE WHEN actual_value = 1 THEN 1 END), 0) as recall_val,
            
            -- Calculate AUC approximation
            calculate_auc_score(ARRAY_AGG(actual_value ORDER BY predicted_score DESC), 
                              ARRAY_AGG(predicted_score ORDER BY predicted_score DESC)) as auc_val
        FROM model_predictions
    ),
    validation_checks AS (
        SELECT 
            pm.*,
            -- F1 Score
            2.0 * (pm.precision_val * pm.recall_val) / 
            NULLIF(pm.precision_val + pm.recall_val, 0) as f1_val,
            
            -- Business validation
            calculate_business_impact(p_model_name, p_new_version) as business_impact_val
        FROM performance_metrics pm
    )
    SELECT 
        ROUND(vc.accuracy_val, 4) as accuracy,
        ROUND(vc.precision_val, 4) as precision_score,
        ROUND(vc.recall_val, 4) as recall_score,
        ROUND(vc.f1_val, 4) as f1_score,
        ROUND(vc.auc_val, 4) as auc_roc,
        (vc.accuracy_val >= 0.75 AND vc.f1_val >= 0.70 AND vc.auc_val >= 0.80) as validation_passed,
        vc.business_impact_val as business_metrics
    INTO validation_results
    FROM validation_checks vc;
    
    -- Store validation results
    INSERT INTO model_validation_results (
        model_name, model_version, validation_date, accuracy, precision_score, 
        recall_score, f1_score, auc_roc, validation_passed, business_metrics
    ) VALUES (
        p_model_name, p_new_version, CURRENT_TIMESTAMP,
        validation_results.accuracy, validation_results.precision_score,
        validation_results.recall_score, validation_results.f1_score,
        validation_results.auc_roc, validation_results.validation_passed,
        validation_results.business_metrics
    );
    
    -- Return results
    accuracy := validation_results.accuracy;
    precision_score := validation_results.precision_score;
    recall_score := validation_results.recall_score;
    f1_score := validation_results.f1_score;
    auc_roc := validation_results.auc_roc;
    validation_passed := validation_results.validation_passed;
    business_metrics := validation_results.business_metrics;
    
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- A/B testing framework for model deployment
CREATE OR REPLACE FUNCTION deploy_model_ab_test(
    p_model_name VARCHAR,
    p_new_version VARCHAR,
    p_traffic_percentage NUMERIC DEFAULT 10.0
)
RETURNS VOID AS $$
BEGIN
    -- Create A/B test configuration
    INSERT INTO model_ab_tests (
        model_name, 
        control_version, 
        test_version, 
        traffic_percentage, 
        start_date,
        status
    ) VALUES (
        p_model_name,
        (SELECT current_version FROM ml_models WHERE model_name = p_model_name),
        p_new_version,
        p_traffic_percentage,
        CURRENT_TIMESTAMP,
        'ACTIVE'
    );
    
    -- Update model routing
    INSERT INTO model_routing_rules (
        model_name, version, traffic_percentage, is_active
    ) VALUES (
        p_model_name, p_new_version, p_traffic_percentage, TRUE
    );
    
    -- Schedule A/B test evaluation
    INSERT INTO scheduled_tasks (
        task_type, model_name, execution_time, task_parameters
    ) VALUES (
        'EVALUATE_AB_TEST',
        p_model_name,
        CURRENT_TIMESTAMP + INTERVAL '7 days',
        jsonb_build_object('test_version', p_new_version)
    );
END;
$$ LANGUAGE plpgsql;
```

#### 4. Scheduling and Automation Framework

```sql
-- Comprehensive scheduling system
CREATE TABLE automation_schedules (
    schedule_id SERIAL PRIMARY KEY,
    schedule_name VARCHAR(100),
    model_name VARCHAR(100),
    task_type VARCHAR(50), -- 'MONITOR', 'RETRAIN', 'EVALUATE', 'CLEANUP'
    frequency_expression VARCHAR(50), -- Cron-like expression
    is_active BOOLEAN DEFAULT TRUE,
    last_execution TIMESTAMP,
    next_execution TIMESTAMP,
    execution_count INT DEFAULT 0,
    success_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Master automation orchestrator
CREATE OR REPLACE FUNCTION execute_scheduled_automations()
RETURNS VOID AS $$
DECLARE
    schedule_record RECORD;
    execution_success BOOLEAN;
BEGIN
    -- Process all due schedules
    FOR schedule_record IN 
        SELECT * FROM automation_schedules 
        WHERE is_active = TRUE 
        AND next_execution <= CURRENT_TIMESTAMP
    LOOP
        execution_success := FALSE;
        
        BEGIN
            -- Execute based on task type
            CASE schedule_record.task_type
                WHEN 'MONITOR' THEN
                    PERFORM monitor_model_performance();
                    execution_success := TRUE;
                    
                WHEN 'RETRAIN' THEN
                    PERFORM trigger_model_retraining(
                        schedule_record.model_name, 
                        'SCHEDULED_RETRAIN'
                    );
                    execution_success := TRUE;
                    
                WHEN 'EVALUATE' THEN
                    PERFORM evaluate_model_performance(schedule_record.model_name);
                    execution_success := TRUE;
                    
                WHEN 'CLEANUP' THEN
                    PERFORM cleanup_old_model_data(schedule_record.model_name);
                    execution_success := TRUE;
                    
                ELSE
                    RAISE EXCEPTION 'Unknown task type: %', schedule_record.task_type;
            END CASE;
            
        EXCEPTION WHEN OTHERS THEN
            -- Log execution error
            INSERT INTO automation_execution_log (
                schedule_id, execution_time, status, error_message
            ) VALUES (
                schedule_record.schedule_id,
                CURRENT_TIMESTAMP,
                'FAILED',
                SQLERRM
            );
        END;
        
        -- Update schedule record
        UPDATE automation_schedules 
        SET last_execution = CURRENT_TIMESTAMP,
            execution_count = execution_count + 1,
            success_count = success_count + CASE WHEN execution_success THEN 1 ELSE 0 END,
            next_execution = calculate_next_execution(frequency_expression)
        WHERE schedule_id = schedule_record.schedule_id;
        
        -- Log successful execution
        IF execution_success THEN
            INSERT INTO automation_execution_log (
                schedule_id, execution_time, status, details
            ) VALUES (
                schedule_record.schedule_id,
                CURRENT_TIMESTAMP,
                'SUCCESS',
                format('Task %s completed for model %s', 
                       schedule_record.task_type, 
                       schedule_record.model_name)
            );
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Setup automation schedules for all models
CREATE OR REPLACE FUNCTION setup_model_automation(p_model_name VARCHAR)
RETURNS VOID AS $$
BEGIN
    -- Daily performance monitoring
    INSERT INTO automation_schedules (
        schedule_name, model_name, task_type, frequency_expression, next_execution
    ) VALUES (
        format('%s_daily_monitor', p_model_name),
        p_model_name,
        'MONITOR',
        '0 6 * * *',  -- Daily at 6 AM
        CURRENT_TIMESTAMP + INTERVAL '1 day'
    );
    
    -- Weekly retraining evaluation
    INSERT INTO automation_schedules (
        schedule_name, model_name, task_type, frequency_expression, next_execution
    ) VALUES (
        format('%s_weekly_retrain_check', p_model_name),
        p_model_name,
        'EVALUATE',
        '0 2 * * 0',  -- Weekly on Sunday at 2 AM
        CURRENT_TIMESTAMP + INTERVAL '7 days'
    );
    
    -- Monthly cleanup
    INSERT INTO automation_schedules (
        schedule_name, model_name, task_type, frequency_expression, next_execution
    ) VALUES (
        format('%s_monthly_cleanup', p_model_name),
        p_model_name,
        'CLEANUP',
        '0 3 1 * *',  -- Monthly on 1st at 3 AM
        CURRENT_TIMESTAMP + INTERVAL '30 days'
    );
END;
$$ LANGUAGE plpgsql;

-- Automation health monitoring
CREATE OR REPLACE VIEW automation_health_dashboard AS
WITH schedule_stats AS (
    SELECT 
        model_name,
        task_type,
        COUNT(*) as total_schedules,
        SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active_schedules,
        AVG(success_count::NUMERIC / NULLIF(execution_count, 0)) as success_rate,
        MAX(last_execution) as last_execution
    FROM automation_schedules
    GROUP BY model_name, task_type
),
recent_failures AS (
    SELECT 
        model_name,
        COUNT(*) as failure_count
    FROM automation_execution_log ael
    JOIN automation_schedules asched ON ael.schedule_id = asched.schedule_id
    WHERE ael.status = 'FAILED' 
    AND ael.execution_time >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    GROUP BY model_name
)
SELECT 
    ss.model_name,
    ss.task_type,
    ss.active_schedules,
    ROUND(ss.success_rate * 100, 2) as success_rate_pct,
    ss.last_execution,
    COALESCE(rf.failure_count, 0) as recent_failures,
    CASE 
        WHEN ss.success_rate < 0.9 OR COALESCE(rf.failure_count, 0) > 3 THEN 'CRITICAL'
        WHEN ss.success_rate < 0.95 OR COALESCE(rf.failure_count, 0) > 1 THEN 'WARNING'
        ELSE 'HEALTHY'
    END as health_status
FROM schedule_stats ss
LEFT JOIN recent_failures rf ON ss.model_name = rf.model_name
ORDER BY health_status DESC, recent_failures DESC;
```

### Real-World Applications

1. **Financial Services**: Automated retraining of fraud detection models with new transaction patterns
2. **E-commerce**: Regular updates to recommendation systems based on user behavior changes
3. **Healthcare**: Continuous improvement of diagnostic models with new patient data
4. **Manufacturing**: Automated quality control model updates based on production changes
5. **Marketing**: Dynamic campaign optimization models that adapt to market conditions

### Best Practices

1. **Gradual Rollouts**: Use A/B testing for model deployment to minimize risk
2. **Comprehensive Logging**: Maintain detailed audit trails for all automated actions
3. **Performance Thresholds**: Set clear criteria for triggering retraining
4. **Rollback Procedures**: Implement quick rollback mechanisms for failed deployments
5. **Human Oversight**: Include human approval steps for critical model updates

### Common Pitfalls

1. **Over-Automation**: Automating everything without human oversight controls
2. **Data Quality Issues**: Retraining on poor quality data without validation
3. **Insufficient Testing**: Deploying models without adequate validation
4. **Resource Contention**: Heavy retraining processes affecting production systems
5. **Version Control**: Poor tracking of model versions and deployment history

### Debugging and Optimization

```sql
-- Automation troubleshooting toolkit
CREATE OR REPLACE FUNCTION diagnose_automation_issues()
RETURNS TABLE (
    issue_type VARCHAR,
    model_name VARCHAR,
    issue_description TEXT,
    suggested_action TEXT,
    severity VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    -- Check for failed retraining jobs
    SELECT 
        'RETRAINING_FAILURE' as issue_type,
        rj.model_name,
        format('Retraining job %s failed: %s', rj.job_id, rj.error_message) as issue_description,
        'Check training data quality and system resources' as suggested_action,
        'HIGH' as severity
    FROM retraining_jobs rj
    WHERE rj.job_status = 'FAILED'
    AND rj.started_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    
    UNION ALL
    
    -- Check for performance degradation
    SELECT 
        'PERFORMANCE_DEGRADATION',
        mpm.model_name,
        format('Model accuracy dropped to %s%% (baseline: %s%%)', 
               ROUND(mpm.accuracy * 100, 1),
               ROUND(baseline.accuracy * 100, 1)),
        'Trigger immediate retraining or investigate data drift',
        'CRITICAL'
    FROM model_performance_metrics mpm
    JOIN (
        SELECT model_name, MAX(accuracy) as accuracy
        FROM model_performance_metrics 
        WHERE evaluation_type = 'BASELINE'
        GROUP BY model_name
    ) baseline ON mpm.model_name = baseline.model_name
    WHERE mpm.evaluation_date >= CURRENT_TIMESTAMP - INTERVAL '1 day'
    AND mpm.accuracy < baseline.accuracy * 0.9
    
    UNION ALL
    
    -- Check for stuck schedules
    SELECT 
        'STUCK_SCHEDULE',
        asched.model_name,
        format('Schedule %s has not run for %s days', 
               asched.schedule_name,
               EXTRACT(DAYS FROM CURRENT_TIMESTAMP - asched.last_execution)),
        'Check scheduler service and system resources',
        'MEDIUM'
    FROM automation_schedules asched
    WHERE asched.is_active = TRUE
    AND asched.last_execution < CURRENT_TIMESTAMP - INTERVAL '2 days';
END;
$$ LANGUAGE plpgsql;

-- Performance optimization for automation
CREATE INDEX CONCURRENTLY idx_model_performance_metrics_model_date 
ON model_performance_metrics(model_name, evaluation_date DESC);

CREATE INDEX CONCURRENTLY idx_ml_predictions_model_timestamp 
ON ml_predictions(model_name, prediction_timestamp);

CREATE INDEX CONCURRENTLY idx_automation_schedules_next_execution 
ON automation_schedules(next_execution) WHERE is_active = TRUE;
```

### Performance Considerations

- **Resource Management**: Schedule intensive operations during off-peak hours
- **Parallel Processing**: Use background job queues for concurrent model training
- **Data Partitioning**: Partition large tables by model and date for faster processing
- **Caching**: Cache frequently accessed model metadata and performance metrics
- **Monitoring Overhead**: Balance monitoring frequency with system performance impact

---

