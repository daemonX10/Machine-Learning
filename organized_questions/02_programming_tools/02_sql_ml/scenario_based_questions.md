# Sql Ml Interview Questions - Scenario_Based Questions

## Question 1

**How would you write a SQL query to selectdistinct valuesfrom a column?**

**Answer:**

### Basic DISTINCT Syntax

The `DISTINCT` keyword is used to return only unique values from a column, eliminating duplicates.

#### 1. Simple DISTINCT Query
```sql
-- Basic distinct values from a single column
SELECT DISTINCT column_name
FROM table_name;

-- Example: Get unique customer cities
SELECT DISTINCT city
FROM customers;
```

#### 2. DISTINCT with Multiple Columns
```sql
-- Distinct combinations of multiple columns
SELECT DISTINCT column1, column2, column3
FROM table_name;

-- Example: Unique combinations of product category and supplier
SELECT DISTINCT category, supplier_id
FROM products;
```

#### 3. DISTINCT with ORDER BY
```sql
-- Ordered distinct values
SELECT DISTINCT city
FROM customers
ORDER BY city ASC;
```

### Advanced DISTINCT Scenarios

#### 4. DISTINCT with WHERE Clause
```sql
-- Distinct values with filtering
SELECT DISTINCT category
FROM products
WHERE price > 50.00
ORDER BY category;
```

#### 5. COUNT DISTINCT
```sql
-- Count unique values
SELECT COUNT(DISTINCT customer_id) AS unique_customers
FROM orders;

-- Count distinct with grouping
SELECT 
    category,
    COUNT(DISTINCT supplier_id) AS unique_suppliers
FROM products
GROUP BY category;
```

#### 6. DISTINCT with CASE Statements
```sql
-- Distinct categorical transformations
SELECT DISTINCT 
    CASE 
        WHEN price < 20 THEN 'Budget'
        WHEN price BETWEEN 20 AND 100 THEN 'Mid-range'
        ELSE 'Premium'
    END AS price_category
FROM products;
```

### Machine Learning Context Applications

#### 7. Feature Value Analysis
```sql
-- Analyze distinct values for categorical features
SELECT 
    'gender' AS feature_name,
    COUNT(DISTINCT gender) AS unique_values,
    STRING_AGG(DISTINCT gender, ', ') AS value_list
FROM customer_features
UNION ALL
SELECT 
    'age_group' AS feature_name,
    COUNT(DISTINCT age_group) AS unique_values,
    STRING_AGG(DISTINCT age_group, ', ') AS value_list
FROM customer_features;
```

#### 8. Data Quality Checks
```sql
-- Check for unexpected values in categorical columns
SELECT DISTINCT 
    status,
    COUNT(*) as frequency
FROM model_predictions
WHERE status NOT IN ('success', 'failed', 'pending')
GROUP BY status;
```

#### 9. Feature Engineering with DISTINCT
```sql
-- Create lookup tables for categorical encoding
CREATE VIEW category_mapping AS
SELECT DISTINCT 
    category,
    ROW_NUMBER() OVER (ORDER BY category) AS category_id
FROM products;

-- Use for one-hot encoding preparation
SELECT DISTINCT 
    customer_id,
    category,
    1 AS has_category
FROM purchase_history
PIVOT (
    MAX(has_category) 
    FOR category IN ('Electronics', 'Clothing', 'Books')
) AS pivot_table;
```

### Performance Optimization

#### 10. DISTINCT with Indexes
```sql
-- Create index for better DISTINCT performance
CREATE INDEX idx_product_category ON products(category);

-- Efficient distinct query with index
SELECT DISTINCT category
FROM products;
```

#### 11. DISTINCT vs GROUP BY Performance
```sql
-- Using GROUP BY instead of DISTINCT (sometimes faster)
SELECT customer_city
FROM customers
GROUP BY customer_city;

-- Equivalent to:
SELECT DISTINCT customer_city
FROM customers;
```

#### 12. Window Functions Alternative
```sql
-- Using ROW_NUMBER() to get distinct records with additional info
SELECT 
    customer_id,
    first_purchase_date,
    city
FROM (
    SELECT 
        customer_id,
        first_purchase_date,
        city,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY first_purchase_date) as rn
    FROM customer_orders
) ranked
WHERE rn = 1;
```

### Common Pitfalls and Best Practices

#### Best Practices:
1. **Use indexes** on columns with DISTINCT operations
2. **Consider GROUP BY** for better performance in some cases
3. **Limit result sets** when possible with WHERE clauses
4. **Use DISTINCT COUNT** instead of COUNT(DISTINCT) when appropriate

#### Common Pitfalls:
1. **Performance impact** on large datasets without proper indexing
2. **Memory usage** with DISTINCT on multiple columns
3. **NULL handling** - DISTINCT treats all NULLs as the same value
4. **Unexpected results** when using DISTINCT with calculated columns

#### 13. Handling NULLs with DISTINCT
```sql
-- DISTINCT includes NULL as a unique value
SELECT DISTINCT customer_rating
FROM feedback; -- Will include NULL if present

-- Exclude NULLs explicitly
SELECT DISTINCT customer_rating
FROM feedback
WHERE customer_rating IS NOT NULL;
```

This comprehensive approach to using DISTINCT provides both basic functionality and advanced applications relevant to data analysis and machine learning workflows.

---

## Question 2

**How would youoptimizea SQL query that seems to be running slowly?**

**Answer:**

### Query Optimization Strategy Framework

SQL query optimization follows a systematic approach involving analysis, identification of bottlenecks, and implementing targeted solutions.

#### 1. Query Analysis and Profiling

```sql
-- Enable query execution plan analysis
SET STATISTICS IO ON;
SET STATISTICS TIME ON;

-- Analyze query execution plan
EXPLAIN (ANALYZE, BUFFERS) 
SELECT customer_id, SUM(order_amount)
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE order_date >= '2023-01-01'
GROUP BY customer_id;
```

#### 2. Index Optimization

```sql
-- Identify missing indexes
SELECT 
    DB_NAME() AS database_name,
    OBJECT_NAME(s.object_id) AS table_name,
    i.name AS index_name,
    s.user_seeks,
    s.user_scans,
    s.user_lookups
FROM sys.dm_db_index_usage_stats s
JOIN sys.indexes i ON s.object_id = i.object_id AND s.index_id = i.index_id
WHERE s.database_id = DB_ID()
ORDER BY s.user_seeks + s.user_scans + s.user_lookups DESC;

-- Create optimal indexes
CREATE INDEX idx_orders_date_customer 
ON orders(order_date, customer_id) 
INCLUDE (order_amount);

-- Composite index for multi-column searches
CREATE INDEX idx_customer_features_ml 
ON customer_features(age_group, income_bracket, region)
INCLUDE (customer_lifetime_value);
```

#### 3. Query Rewriting Techniques

```sql
-- BEFORE: Inefficient subquery
SELECT * FROM customers c
WHERE customer_id IN (
    SELECT customer_id FROM orders 
    WHERE order_date >= '2023-01-01'
);

-- AFTER: Optimized with EXISTS
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.customer_id 
    AND o.order_date >= '2023-01-01'
);

-- BEFORE: Inefficient OR conditions
SELECT * FROM products
WHERE category = 'Electronics' OR category = 'Computers';

-- AFTER: Optimized with IN
SELECT * FROM products
WHERE category IN ('Electronics', 'Computers');
```

#### 4. JOIN Optimization

```sql
-- Optimize JOIN order and conditions
-- BEFORE: Inefficient JOIN
SELECT c.customer_name, o.order_total, p.product_name
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE c.registration_date >= '2023-01-01';

-- AFTER: Optimized with filtering first
SELECT c.customer_name, o.order_total, p.product_name
FROM (
    SELECT customer_id, customer_name
    FROM customers
    WHERE registration_date >= '2023-01-01'
) c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id;
```

### Machine Learning Specific Optimizations

#### 5. Large Dataset Handling for ML

```sql
-- Partitioned table for time-series ML data
CREATE TABLE ml_training_data (
    record_id BIGINT,
    feature_vector NVARCHAR(MAX),
    target_value FLOAT,
    created_date DATE
) 
PARTITION BY RANGE (created_date);

-- Optimized sampling for ML training
WITH stratified_sample AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY target_class 
               ORDER BY NEWID()
           ) as rn
    FROM training_dataset
)
SELECT * FROM stratified_sample
WHERE rn <= 1000; -- Sample 1000 per class
```

#### 6. Feature Computation Optimization

```sql
-- BEFORE: Slow aggregation
SELECT 
    customer_id,
    AVG(purchase_amount) as avg_purchase,
    COUNT(*) as purchase_count,
    MAX(purchase_date) as last_purchase
FROM purchases
GROUP BY customer_id;

-- AFTER: Optimized with materialized view
CREATE MATERIALIZED VIEW customer_features_agg AS
SELECT 
    customer_id,
    AVG(purchase_amount) as avg_purchase,
    COUNT(*) as purchase_count,
    MAX(purchase_date) as last_purchase
FROM purchases
GROUP BY customer_id;

-- Use incremental refresh for real-time updates
REFRESH MATERIALIZED VIEW customer_features_agg;
```

#### 7. Window Function Optimization

```sql
-- Optimize ranking and moving averages for ML features
SELECT 
    customer_id,
    purchase_amount,
    purchase_date,
    -- Optimized moving average
    AVG(purchase_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY purchase_date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as moving_avg_3,
    -- Optimized ranking
    DENSE_RANK() OVER (
        PARTITION BY customer_id 
        ORDER BY purchase_amount DESC
    ) as purchase_rank
FROM purchases
WHERE purchase_date >= DATEADD(month, -12, GETDATE());
```

### Advanced Optimization Techniques

#### 8. Query Plan Caching

```sql
-- Use parameterized queries for plan reuse
DECLARE @start_date DATE = '2023-01-01';
DECLARE @end_date DATE = '2023-12-31';

SELECT customer_id, SUM(order_amount)
FROM orders
WHERE order_date BETWEEN @start_date AND @end_date
GROUP BY customer_id;

-- Create query plan guides for consistent performance
CREATE PLAN GUIDE guide_customer_analysis
FOR 
    STATEMENT N'SELECT customer_id, SUM(order_amount)
                FROM orders
                WHERE order_date BETWEEN @start_date AND @end_date
                GROUP BY customer_id'
    OPTION (FORCE ORDER, LOOP JOIN);
```

#### 9. Parallel Query Processing

```sql
-- Enable parallel processing for large ML datasets
SELECT /*+ PARALLEL(4) */ 
    feature_category,
    COUNT(*) as record_count,
    AVG(target_value) as avg_target
FROM ml_training_data
WHERE created_date >= '2023-01-01'
GROUP BY feature_category;

-- Optimize parallel execution
SET max_parallel_workers_per_gather = 4;
SET parallel_tuple_cost = 0.1;
```

#### 10. Memory and Resource Optimization

```sql
-- Configure memory settings for large queries
SET work_mem = '256MB';
SET effective_cache_size = '4GB';

-- Use CTEs to break complex queries
WITH feature_base AS (
    SELECT 
        customer_id,
        purchase_date,
        purchase_amount,
        product_category
    FROM purchases
    WHERE purchase_date >= DATEADD(month, -12, GETDATE())
),
customer_metrics AS (
    SELECT 
        customer_id,
        COUNT(*) as purchase_frequency,
        AVG(purchase_amount) as avg_amount,
        COUNT(DISTINCT product_category) as category_diversity
    FROM feature_base
    GROUP BY customer_id
)
SELECT * FROM customer_metrics;
```

### Performance Monitoring and Maintenance

#### 11. Continuous Monitoring

```sql
-- Monitor query performance over time
CREATE TABLE query_performance_log (
    query_id VARCHAR(100),
    execution_time_ms INT,
    cpu_time_ms INT,
    logical_reads BIGINT,
    execution_date DATETIME,
    query_plan_hash VARBINARY(8)
);

-- Log query metrics
INSERT INTO query_performance_log
SELECT 
    'customer_feature_extraction' as query_id,
    total_elapsed_time/1000 as execution_time_ms,
    total_worker_time/1000 as cpu_time_ms,
    total_logical_reads,
    GETDATE(),
    query_plan_hash
FROM sys.dm_exec_query_stats qs
CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) st
WHERE st.text LIKE '%customer_feature_extraction%';
```

#### 12. Automated Optimization

```sql
-- Auto-update statistics for ML tables
CREATE PROCEDURE sp_update_ml_statistics
AS
BEGIN
    -- Update statistics on key ML tables
    UPDATE STATISTICS ml_training_data WITH FULLSCAN;
    UPDATE STATISTICS customer_features WITH FULLSCAN;
    UPDATE STATISTICS model_predictions WITH FULLSCAN;
    
    -- Rebuild fragmented indexes
    DECLARE @sql NVARCHAR(MAX);
    SELECT @sql = STRING_AGG(
        'ALTER INDEX ' + i.name + ' ON ' + t.name + ' REBUILD;', 
        CHAR(13)
    )
    FROM sys.indexes i
    JOIN sys.tables t ON i.object_id = t.object_id
    JOIN sys.dm_db_index_physical_stats(DB_ID(), NULL, NULL, NULL, 'LIMITED') ps
        ON i.object_id = ps.object_id AND i.index_id = ps.index_id
    WHERE ps.avg_fragmentation_in_percent > 30
        AND t.name IN ('ml_training_data', 'customer_features');
    
    EXEC sp_executesql @sql;
END;

-- Schedule regular maintenance
-- EXEC msdb.dbo.sp_add_job @job_name = 'ML_Statistics_Update'
```

### Best Practices for ML Query Optimization

1. **Profile Before Optimizing**: Always analyze execution plans first
2. **Index Strategy**: Create composite indexes for multi-column ML feature queries
3. **Partition Large Tables**: Use date-based partitioning for time-series ML data
4. **Materialized Views**: Pre-compute complex feature aggregations
5. **Parallel Processing**: Enable parallel execution for large dataset operations
6. **Memory Management**: Configure appropriate memory settings for ML workloads
7. **Regular Maintenance**: Update statistics and rebuild indexes regularly
8. **Query Caching**: Use parameterized queries for repeated ML operations

### Common Performance Pitfalls

1. **Over-indexing**: Too many indexes can slow DML operations
2. **Inefficient JOINs**: Wrong JOIN order or missing WHERE clauses
3. **Function Calls in WHERE**: Avoid functions on filtered columns
4. **SELECT ***: Always specify required columns
5. **Improper Data Types**: Use appropriate data types for ML features
6. **Lack of Statistics**: Outdated statistics lead to poor execution plans

This comprehensive optimization approach ensures optimal performance for both standard SQL operations and machine learning specific workloads.

---

## Question 3

**How would youmergemultiple result sets in SQL without duplicates?**

**Answer:**

### Set Operations for Merging Without Duplicates

SQL provides several set operations to combine result sets while handling duplicates appropriately.

#### 1. UNION - Basic Merge Without Duplicates

```sql
-- Basic UNION automatically removes duplicates
SELECT customer_id, customer_name, email
FROM customers_2022
UNION
SELECT customer_id, customer_name, email
FROM customers_2023;

-- Multi-table UNION
SELECT product_id, product_name, 'Electronics' as source
FROM electronics_products
UNION
SELECT product_id, product_name, 'Clothing' as source
FROM clothing_products
UNION
SELECT product_id, product_name, 'Books' as source
FROM book_products;
```

#### 2. UNION ALL with Manual Deduplication

```sql
-- When you need control over deduplication logic
WITH combined_data AS (
    SELECT customer_id, customer_name, email, registration_date, 'source1' as source
    FROM customers_table1
    UNION ALL
    SELECT customer_id, customer_name, email, registration_date, 'source2' as source
    FROM customers_table2
    UNION ALL
    SELECT customer_id, customer_name, email, registration_date, 'source3' as source
    FROM customers_table3
),
deduplicated AS (
    SELECT 
        customer_id,
        customer_name,
        email,
        registration_date,
        source,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id 
            ORDER BY registration_date DESC, source
        ) as rn
    FROM combined_data
)
SELECT customer_id, customer_name, email, registration_date, source
FROM deduplicated
WHERE rn = 1;
```

#### 3. Complex Merging with Priority Rules

```sql
-- Merge with business logic for handling conflicts
WITH data_sources AS (
    SELECT 
        customer_id,
        customer_name,
        email,
        phone,
        last_updated,
        'CRM' as source,
        1 as priority
    FROM crm_customers
    UNION ALL
    SELECT 
        customer_id,
        customer_name,
        email,
        phone,
        last_updated,
        'Marketing' as source,
        2 as priority
    FROM marketing_customers
    UNION ALL
    SELECT 
        customer_id,
        customer_name,
        email,
        phone,
        last_updated,
        'Sales' as source,
        3 as priority
    FROM sales_customers
),
ranked_data AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY customer_id 
               ORDER BY priority ASC, last_updated DESC
           ) as rn
    FROM data_sources
)
SELECT 
    customer_id,
    customer_name,
    email,
    phone,
    last_updated,
    source as data_source
FROM ranked_data
WHERE rn = 1;
```

### Machine Learning Data Merging Scenarios

#### 4. Feature Set Merging for ML

```sql
-- Merge features from different sources for ML model
WITH demographic_features AS (
    SELECT 
        customer_id,
        age,
        gender,
        income_bracket,
        education_level
    FROM customer_demographics
),
behavioral_features AS (
    SELECT 
        customer_id,
        avg_purchase_amount,
        purchase_frequency,
        preferred_category,
        loyalty_score
    FROM customer_behavior
),
transaction_features AS (
    SELECT 
        customer_id,
        total_transactions,
        avg_transaction_value,
        last_transaction_date,
        payment_method_preference
    FROM customer_transactions
)
-- Full outer join to combine all features
SELECT 
    COALESCE(d.customer_id, b.customer_id, t.customer_id) as customer_id,
    d.age,
    d.gender,
    d.income_bracket,
    d.education_level,
    b.avg_purchase_amount,
    b.purchase_frequency,
    b.preferred_category,
    b.loyalty_score,
    t.total_transactions,
    t.avg_transaction_value,
    t.last_transaction_date,
    t.payment_method_preference
FROM demographic_features d
FULL OUTER JOIN behavioral_features b ON d.customer_id = b.customer_id
FULL OUTER JOIN transaction_features t ON COALESCE(d.customer_id, b.customer_id) = t.customer_id;
```

#### 5. Training Data Merging from Multiple Experiments

```sql
-- Combine training data from different ML experiments
WITH experiment_data AS (
    SELECT 
        'experiment_1' as experiment_id,
        customer_id,
        feature_vector,
        target_value,
        model_version,
        created_date
    FROM ml_experiment_1_data
    WHERE data_quality_score > 0.8
    
    UNION
    
    SELECT 
        'experiment_2' as experiment_id,
        customer_id,
        feature_vector,
        target_value,
        model_version,
        created_date
    FROM ml_experiment_2_data
    WHERE data_quality_score > 0.8
    
    UNION
    
    SELECT 
        'experiment_3' as experiment_id,
        customer_id,
        feature_vector,
        target_value,
        model_version,
        created_date
    FROM ml_experiment_3_data
    WHERE data_quality_score > 0.8
),
-- Remove duplicates based on customer and feature similarity
deduplicated_training AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY customer_id, 
                          LEFT(feature_vector, 100) -- Consider feature similarity
               ORDER BY created_date DESC
           ) as rn
    FROM experiment_data
)
SELECT 
    experiment_id,
    customer_id,
    feature_vector,
    target_value,
    model_version,
    created_date
FROM deduplicated_training
WHERE rn = 1;
```

#### 6. Cross-Database Merging

```sql
-- Merge data from different databases/schemas
SELECT customer_id, order_amount, order_date, 'production' as environment
FROM production_db.orders.customer_orders
WHERE order_date >= '2023-01-01'

UNION

SELECT customer_id, order_amount, order_date, 'staging' as environment
FROM staging_db.orders.customer_orders
WHERE order_date >= '2023-01-01'
  AND customer_id NOT IN (
      SELECT customer_id 
      FROM production_db.orders.customer_orders
      WHERE order_date >= '2023-01-01'
  );
```

### Advanced Merging Techniques

#### 7. Fuzzy Matching Merge

```sql
-- Merge records with fuzzy matching for duplicate detection
WITH potential_matches AS (
    SELECT 
        a.customer_id as id1,
        b.customer_id as id2,
        a.customer_name as name1,
        b.customer_name as name2,
        a.email as email1,
        b.email as email2,
        -- Calculate similarity scores
        CASE 
            WHEN UPPER(a.email) = UPPER(b.email) THEN 100
            WHEN SOUNDEX(a.customer_name) = SOUNDEX(b.customer_name) THEN 80
            WHEN DIFFERENCE(a.customer_name, b.customer_name) >= 3 THEN 60
            ELSE 0
        END as similarity_score
    FROM customers_source1 a
    CROSS JOIN customers_source2 b
    WHERE a.customer_id <> b.customer_id
),
high_similarity_pairs AS (
    SELECT id1, id2, similarity_score
    FROM potential_matches
    WHERE similarity_score >= 80
),
merged_customers AS (
    SELECT 
        customer_id,
        customer_name,
        email,
        phone,
        'source1' as source
    FROM customers_source1
    WHERE customer_id NOT IN (SELECT id1 FROM high_similarity_pairs)
    
    UNION
    
    SELECT 
        customer_id,
        customer_name,
        email,
        phone,
        'source2' as source
    FROM customers_source2
    WHERE customer_id NOT IN (SELECT id2 FROM high_similarity_pairs)
    
    UNION
    
    -- Keep only one record from each similar pair (prefer source1)
    SELECT 
        c1.customer_id,
        c1.customer_name,
        COALESCE(c1.email, c2.email) as email,
        COALESCE(c1.phone, c2.phone) as phone,
        'merged' as source
    FROM high_similarity_pairs hsp
    JOIN customers_source1 c1 ON hsp.id1 = c1.customer_id
    JOIN customers_source2 c2 ON hsp.id2 = c2.customer_id
)
SELECT * FROM merged_customers;
```

#### 8. Incremental Merging for Large Datasets

```sql
-- Efficient incremental merging for streaming data
CREATE PROCEDURE sp_incremental_merge_customers
    @last_sync_date DATETIME
AS
BEGIN
    -- Create temporary table for new data
    CREATE TABLE #temp_merged_customers (
        customer_id INT,
        customer_name VARCHAR(100),
        email VARCHAR(100),
        last_updated DATETIME,
        source_system VARCHAR(50)
    );
    
    -- Insert new/updated records from each source
    INSERT INTO #temp_merged_customers
    SELECT customer_id, customer_name, email, last_updated, 'CRM'
    FROM crm_customers
    WHERE last_updated > @last_sync_date
    
    UNION
    
    SELECT customer_id, customer_name, email, last_updated, 'ERP'
    FROM erp_customers
    WHERE last_updated > @last_sync_date;
    
    -- Merge with existing data using MERGE statement
    MERGE target_customers AS target
    USING (
        SELECT 
            customer_id,
            customer_name,
            email,
            last_updated,
            source_system,
            ROW_NUMBER() OVER (
                PARTITION BY customer_id 
                ORDER BY last_updated DESC
            ) as rn
        FROM #temp_merged_customers
    ) AS source ON target.customer_id = source.customer_id
    WHEN MATCHED AND source.rn = 1 AND source.last_updated > target.last_updated THEN
        UPDATE SET 
            customer_name = source.customer_name,
            email = source.email,
            last_updated = source.last_updated,
            source_system = source.source_system
    WHEN NOT MATCHED AND source.rn = 1 THEN
        INSERT (customer_id, customer_name, email, last_updated, source_system)
        VALUES (source.customer_id, source.customer_name, source.email, 
                source.last_updated, source.source_system);
    
    DROP TABLE #temp_merged_customers;
END;
```

#### 9. Statistical Merging for ML Features

```sql
-- Merge statistical aggregations from different time periods
WITH monthly_stats AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', transaction_date) as month,
        COUNT(*) as transaction_count,
        AVG(amount) as avg_amount,
        STDDEV(amount) as amount_stddev
    FROM transactions
    WHERE transaction_date >= DATE_SUB(CURRENT_DATE, INTERVAL 12 MONTH)
    GROUP BY customer_id, DATE_TRUNC('month', transaction_date)
),
quarterly_stats AS (
    SELECT 
        customer_id,
        DATE_TRUNC('quarter', transaction_date) as quarter,
        COUNT(*) as transaction_count,
        AVG(amount) as avg_amount,
        STDDEV(amount) as amount_stddev
    FROM transactions
    WHERE transaction_date >= DATE_SUB(CURRENT_DATE, INTERVAL 12 MONTH)
    GROUP BY customer_id, DATE_TRUNC('quarter', transaction_date)
),
yearly_stats AS (
    SELECT 
        customer_id,
        DATE_TRUNC('year', transaction_date) as year,
        COUNT(*) as transaction_count,
        AVG(amount) as avg_amount,
        STDDEV(amount) as amount_stddev
    FROM transactions
    WHERE transaction_date >= DATE_SUB(CURRENT_DATE, INTERVAL 2 YEAR)
    GROUP BY customer_id, DATE_TRUNC('year', transaction_date)
)
-- Combine all time-based features
SELECT DISTINCT
    COALESCE(m.customer_id, q.customer_id, y.customer_id) as customer_id,
    AVG(m.avg_amount) as monthly_avg_amount,
    AVG(q.avg_amount) as quarterly_avg_amount,
    AVG(y.avg_amount) as yearly_avg_amount,
    STDDEV(m.amount_stddev) as monthly_volatility,
    STDDEV(q.amount_stddev) as quarterly_volatility,
    STDDEV(y.amount_stddev) as yearly_volatility
FROM monthly_stats m
FULL OUTER JOIN quarterly_stats q ON m.customer_id = q.customer_id
FULL OUTER JOIN yearly_stats y ON COALESCE(m.customer_id, q.customer_id) = y.customer_id
GROUP BY COALESCE(m.customer_id, q.customer_id, y.customer_id);
```

### Best Practices for Merging Without Duplicates

#### 10. Performance Optimization

```sql
-- Use appropriate indexes for merge operations
CREATE INDEX idx_customers_merge 
ON customers (customer_id, last_updated, email);

-- Partition large tables for efficient merging
CREATE TABLE customer_transactions_partitioned (
    customer_id INT,
    transaction_date DATE,
    amount DECIMAL(10,2)
) PARTITION BY RANGE (transaction_date);
```

### Common Pitfalls and Solutions

1. **Performance Issues**: Use indexes on join/merge columns
2. **Data Type Mismatches**: Ensure consistent data types across sources
3. **NULL Handling**: Use COALESCE for proper NULL management
4. **Memory Limitations**: Use incremental processing for large datasets
5. **Business Logic**: Define clear rules for conflict resolution
6. **Data Quality**: Validate data before merging operations

### Key Considerations for ML Applications

1. **Feature Consistency**: Ensure feature definitions are consistent across sources
2. **Temporal Alignment**: Handle time-based features appropriately
3. **Missing Values**: Plan for missing value imputation strategies
4. **Data Lineage**: Track source systems for merged features
5. **Validation**: Implement data quality checks post-merge
6. **Scalability**: Design for growing datasets and new data sources

This comprehensive approach ensures reliable merging of multiple result sets while maintaining data quality and performance for both standard analytics and machine learning applications.

---

## Question 4

**How would you handle very large datasets in SQL for Machine Learning purposes?**

**Answer:**

### Comprehensive Strategy for Large-Scale ML Data Handling

Handling very large datasets in SQL for machine learning requires a multi-layered approach involving data architecture, query optimization, and specialized techniques.

#### 1. Data Partitioning Strategies

```sql
-- Horizontal partitioning by date for time-series ML data
CREATE TABLE ml_training_data (
    record_id BIGINT,
    customer_id INT,
    feature_vector TEXT,
    target_value FLOAT,
    created_date DATE,
    model_version VARCHAR(20)
) 
PARTITION BY RANGE (created_date) (
    PARTITION p2023_q1 VALUES LESS THAN ('2023-04-01'),
    PARTITION p2023_q2 VALUES LESS THAN ('2023-07-01'),
    PARTITION p2023_q3 VALUES LESS THAN ('2023-10-01'),
    PARTITION p2023_q4 VALUES LESS THAN ('2024-01-01'),
    PARTITION p2024_q1 VALUES LESS THAN ('2024-04-01')
);

-- Hash partitioning for customer-based ML features
CREATE TABLE customer_features_large (
    customer_id BIGINT,
    demographic_features JSON,
    behavioral_features JSON,
    transaction_features JSON,
    last_updated TIMESTAMP
)
PARTITION BY HASH (customer_id) PARTITIONS 16;

-- Vertical partitioning for different feature types
CREATE TABLE customer_demographic_features (
    customer_id BIGINT PRIMARY KEY,
    age INT,
    gender VARCHAR(10),
    income_bracket VARCHAR(20),
    education_level VARCHAR(30)
);

CREATE TABLE customer_behavioral_features (
    customer_id BIGINT PRIMARY KEY,
    avg_session_duration FLOAT,
    page_views_per_session FLOAT,
    bounce_rate FLOAT,
    conversion_rate FLOAT
);
```

#### 2. Incremental Data Processing

```sql
-- Incremental feature engineering for large datasets
CREATE PROCEDURE sp_incremental_feature_processing
    @batch_size INT = 100000,
    @start_date DATE = NULL
AS
BEGIN
    DECLARE @last_processed_date DATE;
    DECLARE @current_batch_start BIGINT = 0;
    DECLARE @current_batch_end BIGINT;
    
    -- Get last processed date
    SELECT @last_processed_date = COALESCE(@start_date, MAX(processed_date))
    FROM feature_processing_log;
    
    -- Process in batches
    WHILE EXISTS (
        SELECT 1 FROM raw_customer_data 
        WHERE created_date > @last_processed_date 
        AND record_id > @current_batch_start
    )
    BEGIN
        SET @current_batch_end = @current_batch_start + @batch_size;
        
        -- Process current batch
        INSERT INTO customer_features_processed (
            customer_id,
            recency_score,
            frequency_score,
            monetary_score,
            engagement_score,
            processed_date
        )
        SELECT 
            customer_id,
            DATEDIFF(day, MAX(last_purchase_date), GETDATE()) as recency_score,
            COUNT(DISTINCT order_id) as frequency_score,
            SUM(order_value) as monetary_score,
            AVG(engagement_rating) as engagement_score,
            GETDATE() as processed_date
        FROM raw_customer_data
        WHERE created_date > @last_processed_date
          AND record_id BETWEEN @current_batch_start AND @current_batch_end
        GROUP BY customer_id;
        
        -- Log progress
        INSERT INTO feature_processing_log (batch_start, batch_end, processed_date, record_count)
        SELECT @current_batch_start, @current_batch_end, GETDATE(), @@ROWCOUNT;
        
        SET @current_batch_start = @current_batch_end;
        
        -- Prevent infinite loop
        IF @current_batch_start > (SELECT MAX(record_id) FROM raw_customer_data)
            BREAK;
    END;
END;
```

#### 3. Sampling Strategies for ML

```sql
-- Stratified sampling for balanced ML datasets
WITH target_distribution AS (
    SELECT 
        target_class,
        COUNT(*) as class_count,
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as class_proportion
    FROM large_ml_dataset
    GROUP BY target_class
),
sample_sizes AS (
    SELECT 
        target_class,
        LEAST(class_count, 50000) as max_sample_size, -- Cap at 50k per class
        class_proportion
    FROM target_distribution
),
stratified_sample AS (
    SELECT 
        d.*,
        ROW_NUMBER() OVER (
            PARTITION BY d.target_class 
            ORDER BY NEWID()
        ) as rn
    FROM large_ml_dataset d
    JOIN sample_sizes s ON d.target_class = s.target_class
)
SELECT 
    customer_id,
    feature_vector,
    target_class,
    target_value
FROM stratified_sample s1
JOIN sample_sizes s2 ON s1.target_class = s2.target_class
WHERE s1.rn <= s2.max_sample_size;

-- Time-based sampling for temporal ML models
WITH monthly_samples AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY DATE_TRUNC('month', created_date)
               ORDER BY RANDOM()
           ) as monthly_rn
    FROM time_series_ml_data
    WHERE created_date >= DATE_SUB(CURRENT_DATE, INTERVAL 24 MONTH)
)
SELECT * FROM monthly_samples
WHERE monthly_rn <= 10000; -- 10k samples per month
```

#### 4. Optimized Aggregation Techniques

```sql
-- Pre-computed aggregations using materialized views
CREATE MATERIALIZED VIEW customer_monthly_aggregates AS
SELECT 
    customer_id,
    DATE_TRUNC('month', transaction_date) as month,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    STDDEV(amount) as amount_stddev,
    COUNT(DISTINCT product_category) as category_diversity,
    MAX(transaction_date) as last_transaction_date
FROM customer_transactions
WHERE transaction_date >= DATE_SUB(CURRENT_DATE, INTERVAL 36 MONTH)
GROUP BY customer_id, DATE_TRUNC('month', transaction_date);

-- Refresh strategy for real-time updates
CREATE PROCEDURE sp_refresh_ml_aggregates
AS
BEGIN
    -- Incremental refresh for recent data
    DELETE FROM customer_monthly_aggregates
    WHERE month >= DATE_SUB(CURRENT_DATE, INTERVAL 2 MONTH);
    
    INSERT INTO customer_monthly_aggregates
    SELECT 
        customer_id,
        DATE_TRUNC('month', transaction_date) as month,
        COUNT(*) as transaction_count,
        SUM(amount) as total_amount,
        AVG(amount) as avg_amount,
        STDDEV(amount) as amount_stddev,
        COUNT(DISTINCT product_category) as category_diversity,
        MAX(transaction_date) as last_transaction_date
    FROM customer_transactions
    WHERE transaction_date >= DATE_SUB(CURRENT_DATE, INTERVAL 2 MONTH)
    GROUP BY customer_id, DATE_TRUNC('month', transaction_date);
END;
```

#### 5. Feature Store Implementation

```sql
-- Create feature store tables for large-scale ML
CREATE TABLE feature_store_metadata (
    feature_group_id VARCHAR(100) PRIMARY KEY,
    feature_group_name VARCHAR(200),
    description TEXT,
    owner VARCHAR(100),
    creation_date TIMESTAMP,
    last_updated TIMESTAMP,
    feature_count INT,
    record_count BIGINT,
    storage_format VARCHAR(50)
);

CREATE TABLE feature_store_features (
    feature_id VARCHAR(100) PRIMARY KEY,
    feature_group_id VARCHAR(100),
    feature_name VARCHAR(200),
    feature_type VARCHAR(50),
    description TEXT,
    computation_logic TEXT,
    dependencies JSON,
    FOREIGN KEY (feature_group_id) REFERENCES feature_store_metadata(feature_group_id)
);

-- Partitioned feature values table
CREATE TABLE feature_store_values (
    entity_id BIGINT,
    feature_group_id VARCHAR(100),
    feature_values JSON,
    event_timestamp TIMESTAMP,
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITION BY RANGE (event_timestamp);

-- Feature retrieval with caching
CREATE PROCEDURE sp_get_features_for_inference
    @entity_ids VARCHAR(MAX), -- Comma-separated list
    @feature_groups VARCHAR(MAX), -- Comma-separated list
    @point_in_time TIMESTAMP = NULL
AS
BEGIN
    DECLARE @sql NVARCHAR(MAX);
    DECLARE @pit TIMESTAMP = COALESCE(@point_in_time, CURRENT_TIMESTAMP);
    
    -- Build dynamic query for multiple feature groups
    WITH entity_list AS (
        SELECT CAST(value AS BIGINT) as entity_id
        FROM STRING_SPLIT(@entity_ids, ',')
    ),
    feature_group_list AS (
        SELECT TRIM(value) as feature_group_id
        FROM STRING_SPLIT(@feature_groups, ',')
    ),
    latest_features AS (
        SELECT 
            fsv.entity_id,
            fsv.feature_group_id,
            fsv.feature_values,
            ROW_NUMBER() OVER (
                PARTITION BY fsv.entity_id, fsv.feature_group_id
                ORDER BY fsv.event_timestamp DESC
            ) as rn
        FROM feature_store_values fsv
        JOIN entity_list el ON fsv.entity_id = el.entity_id
        JOIN feature_group_list fgl ON fsv.feature_group_id = fgl.feature_group_id
        WHERE fsv.event_timestamp <= @pit
    )
    SELECT 
        entity_id,
        feature_group_id,
        feature_values
    FROM latest_features
    WHERE rn = 1;
END;
```

#### 6. Parallel Processing for Large ML Operations

```sql
-- Parallel feature computation using table-valued functions
CREATE FUNCTION fn_compute_customer_features_parallel
(
    @customer_batch TABLE (customer_id BIGINT)
)
RETURNS TABLE
AS
RETURN
(
    WITH customer_transactions AS (
        SELECT 
            cb.customer_id,
            ct.transaction_date,
            ct.amount,
            ct.product_category
        FROM @customer_batch cb
        JOIN customer_transactions ct ON cb.customer_id = ct.customer_id
        WHERE ct.transaction_date >= DATEADD(year, -2, GETDATE())
    ),
    customer_features AS (
        SELECT 
            customer_id,
            COUNT(*) as transaction_count,
            AVG(amount) as avg_transaction_amount,
            STDDEV(amount) as transaction_amount_stddev,
            COUNT(DISTINCT product_category) as category_diversity,
            DATEDIFF(day, MAX(transaction_date), GETDATE()) as days_since_last_transaction,
            COUNT(DISTINCT DATE_TRUNC('month', transaction_date)) as active_months
        FROM customer_transactions
        GROUP BY customer_id
    )
    SELECT * FROM customer_features
);

-- Parallel execution wrapper
CREATE PROCEDURE sp_process_features_parallel
    @thread_count INT = 8
AS
BEGIN
    DECLARE @batch_size INT = 100000;
    DECLARE @customer_count BIGINT;
    DECLARE @batches_per_thread INT;
    
    SELECT @customer_count = COUNT(*) FROM customers;
    SET @batches_per_thread = CEILING(@customer_count / (@thread_count * @batch_size));
    
    -- Create temp table for batch assignments
    CREATE TABLE #customer_batches (
        thread_id INT,
        batch_id INT,
        customer_id BIGINT
    );
    
    -- Assign customers to threads and batches
    INSERT INTO #customer_batches
    SELECT 
        (ROW_NUMBER() OVER (ORDER BY customer_id) - 1) / @batch_size % @thread_count + 1 as thread_id,
        (ROW_NUMBER() OVER (ORDER BY customer_id) - 1) / @batch_size as batch_id,
        customer_id
    FROM customers;
    
    -- Process each thread's batches (this would be executed in parallel)
    DECLARE @thread_id INT = 1;
    WHILE @thread_id <= @thread_count
    BEGIN
        INSERT INTO customer_features_computed
        SELECT cf.*
        FROM fn_compute_customer_features_parallel(
            SELECT customer_id FROM #customer_batches WHERE thread_id = @thread_id
        ) cf;
        
        SET @thread_id = @thread_id + 1;
    END;
    
    DROP TABLE #customer_batches;
END;
```

#### 7. Memory-Efficient Query Patterns

```sql
-- Streaming aggregation for large datasets
WITH RECURSIVE customer_feature_stream AS (
    -- Base case: first batch
    SELECT 
        customer_id,
        SUM(amount) as total_amount,
        COUNT(*) as transaction_count,
        1 as batch_number
    FROM customer_transactions
    WHERE transaction_id BETWEEN 1 AND 100000
    GROUP BY customer_id
    
    UNION ALL
    
    -- Recursive case: subsequent batches
    SELECT 
        COALESCE(cfs.customer_id, new_batch.customer_id) as customer_id,
        COALESCE(cfs.total_amount, 0) + COALESCE(new_batch.total_amount, 0) as total_amount,
        COALESCE(cfs.transaction_count, 0) + COALESCE(new_batch.transaction_count, 0) as transaction_count,
        cfs.batch_number + 1
    FROM customer_feature_stream cfs
    FULL OUTER JOIN (
        SELECT 
            customer_id,
            SUM(amount) as total_amount,
            COUNT(*) as transaction_count
        FROM customer_transactions
        WHERE transaction_id BETWEEN (cfs.batch_number * 100000 + 1) AND ((cfs.batch_number + 1) * 100000)
        GROUP BY customer_id
    ) new_batch ON cfs.customer_id = new_batch.customer_id
    WHERE cfs.batch_number < (SELECT MAX(transaction_id) FROM customer_transactions) / 100000
)
SELECT 
    customer_id,
    total_amount,
    transaction_count,
    total_amount / transaction_count as avg_transaction_amount
FROM customer_feature_stream
WHERE batch_number = (SELECT MAX(batch_number) FROM customer_feature_stream);
```

#### 8. Data Pipeline Orchestration

```sql
-- Create pipeline status tracking
CREATE TABLE ml_pipeline_status (
    pipeline_id VARCHAR(100),
    stage_name VARCHAR(100),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20), -- 'running', 'completed', 'failed'
    record_count BIGINT,
    error_message TEXT,
    PRIMARY KEY (pipeline_id, stage_name)
);

-- Pipeline orchestration procedure
CREATE PROCEDURE sp_execute_ml_pipeline
    @pipeline_id VARCHAR(100)
AS
BEGIN
    DECLARE @stage_name VARCHAR(100);
    DECLARE @start_time TIMESTAMP;
    DECLARE @error_message TEXT;
    
    BEGIN TRY
        -- Stage 1: Data Validation
        SET @stage_name = 'data_validation';
        SET @start_time = CURRENT_TIMESTAMP;
        
        INSERT INTO ml_pipeline_status VALUES (@pipeline_id, @stage_name, @start_time, NULL, 'running', 0, NULL);
        
        -- Validate data quality
        IF EXISTS (
            SELECT 1 FROM raw_customer_data 
            WHERE created_date >= DATEADD(day, -1, GETDATE())
            AND (customer_id IS NULL OR amount < 0)
        )
        BEGIN
            THROW 50001, 'Data quality validation failed', 1;
        END;
        
        UPDATE ml_pipeline_status 
        SET end_time = CURRENT_TIMESTAMP, status = 'completed'
        WHERE pipeline_id = @pipeline_id AND stage_name = @stage_name;
        
        -- Stage 2: Feature Engineering
        SET @stage_name = 'feature_engineering';
        SET @start_time = CURRENT_TIMESTAMP;
        
        INSERT INTO ml_pipeline_status VALUES (@pipeline_id, @stage_name, @start_time, NULL, 'running', 0, NULL);
        
        EXEC sp_incremental_feature_processing;
        
        UPDATE ml_pipeline_status 
        SET end_time = CURRENT_TIMESTAMP, status = 'completed', record_count = @@ROWCOUNT
        WHERE pipeline_id = @pipeline_id AND stage_name = @stage_name;
        
        -- Stage 3: Model Training Data Preparation
        SET @stage_name = 'training_data_prep';
        SET @start_time = CURRENT_TIMESTAMP;
        
        INSERT INTO ml_pipeline_status VALUES (@pipeline_id, @stage_name, @start_time, NULL, 'running', 0, NULL);
        
        -- Create training dataset
        INSERT INTO ml_training_dataset_current
        SELECT 
            customer_id,
            feature_vector,
            target_value
        FROM customer_features_processed cfp
        JOIN customer_targets ct ON cfp.customer_id = ct.customer_id
        WHERE cfp.processed_date >= DATEADD(day, -7, GETDATE());
        
        UPDATE ml_pipeline_status 
        SET end_time = CURRENT_TIMESTAMP, status = 'completed', record_count = @@ROWCOUNT
        WHERE pipeline_id = @pipeline_id AND stage_name = @stage_name;
        
    END TRY
    BEGIN CATCH
        SET @error_message = ERROR_MESSAGE();
        
        UPDATE ml_pipeline_status 
        SET end_time = CURRENT_TIMESTAMP, status = 'failed', error_message = @error_message
        WHERE pipeline_id = @pipeline_id AND stage_name = @stage_name;
        
        THROW;
    END CATCH;
END;
```

### Best Practices for Large-Scale ML Data Handling

#### 9. Monitoring and Alerting

```sql
-- Performance monitoring for large dataset operations
CREATE TABLE query_performance_metrics (
    query_id VARCHAR(100),
    execution_date TIMESTAMP,
    duration_seconds INT,
    rows_processed BIGINT,
    memory_used_mb INT,
    cpu_time_seconds INT,
    io_operations BIGINT
);

-- Alert system for performance degradation
CREATE PROCEDURE sp_check_ml_performance_alerts
AS
BEGIN
    DECLARE @alert_threshold_seconds INT = 3600; -- 1 hour
    DECLARE @memory_threshold_mb INT = 8192; -- 8GB
    
    -- Check for long-running queries
    INSERT INTO performance_alerts (alert_type, message, severity, created_date)
    SELECT 
        'long_running_query',
        'ML query ' + query_id + ' exceeded ' + CAST(@alert_threshold_seconds AS VARCHAR) + ' seconds',
        'high',
        CURRENT_TIMESTAMP
    FROM query_performance_metrics
    WHERE execution_date >= DATEADD(hour, -1, CURRENT_TIMESTAMP)
      AND duration_seconds > @alert_threshold_seconds;
    
    -- Check for high memory usage
    INSERT INTO performance_alerts (alert_type, message, severity, created_date)
    SELECT 
        'high_memory_usage',
        'ML query ' + query_id + ' used ' + CAST(memory_used_mb AS VARCHAR) + ' MB memory',
        'medium',
        CURRENT_TIMESTAMP
    FROM query_performance_metrics
    WHERE execution_date >= DATEADD(hour, -1, CURRENT_TIMESTAMP)
      AND memory_used_mb > @memory_threshold_mb;
END;
```

### Key Strategies Summary

1. **Partitioning**: Horizontal, vertical, and hash partitioning for scalability
2. **Incremental Processing**: Batch-based processing to handle memory constraints
3. **Sampling**: Stratified and temporal sampling for manageable dataset sizes
4. **Materialized Views**: Pre-computed aggregations for faster feature access
5. **Feature Store**: Centralized feature management and versioning
6. **Parallel Processing**: Multi-threaded computation for large operations
7. **Memory Management**: Streaming and chunked processing patterns
8. **Pipeline Orchestration**: Automated, monitored data processing workflows
9. **Performance Monitoring**: Continuous monitoring and alerting systems

This comprehensive approach enables efficient handling of very large datasets while maintaining performance and reliability for machine learning applications.

---

## Question 5

**Discuss how you would design a system to regularly feed a Machine Learning model with SQL data.**

**Answer:**

### Comprehensive ML Data Pipeline Architecture

Designing a robust system to regularly feed ML models with SQL data requires a well-architected pipeline that ensures data quality, reliability, and scalability.

#### 1. Pipeline Architecture Overview

```sql
-- Create pipeline configuration table
CREATE TABLE ml_pipeline_config (
    pipeline_id VARCHAR(100) PRIMARY KEY,
    model_name VARCHAR(100),
    source_tables JSON, -- List of source tables
    feature_config JSON, -- Feature engineering configuration
    schedule_cron VARCHAR(50), -- Cron expression for scheduling
    batch_size INT DEFAULT 10000,
    max_parallel_jobs INT DEFAULT 4,
    data_retention_days INT DEFAULT 90,
    is_active BOOLEAN DEFAULT TRUE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert example pipeline configuration
INSERT INTO ml_pipeline_config (
    pipeline_id,
    model_name,
    source_tables,
    feature_config,
    schedule_cron,
    batch_size
) VALUES (
    'customer_churn_pipeline',
    'churn_prediction_model_v2',
    '["customers", "transactions", "customer_support_tickets"]',
    '{
        "demographic_features": ["age", "gender", "location", "income_bracket"],
        "behavioral_features": ["avg_monthly_spend", "transaction_frequency", "support_tickets_count"],
        "derived_features": ["customer_lifetime_value", "engagement_score", "churn_risk_score"]
    }',
    '0 2 * * *', -- Run daily at 2 AM
    50000
);
```

#### 2. Data Ingestion and Validation Layer

```sql
-- Create data quality validation framework
CREATE TABLE data_quality_rules (
    rule_id VARCHAR(100) PRIMARY KEY,
    table_name VARCHAR(100),
    column_name VARCHAR(100),
    rule_type VARCHAR(50), -- 'not_null', 'range', 'format', 'uniqueness'
    rule_parameters JSON,
    severity VARCHAR(20), -- 'error', 'warning'
    is_active BOOLEAN DEFAULT TRUE
);

-- Example data quality rules
INSERT INTO data_quality_rules VALUES
('customer_id_not_null', 'customers', 'customer_id', 'not_null', '{}', 'error', TRUE),
('transaction_amount_range', 'transactions', 'amount', 'range', '{"min": 0, "max": 100000}', 'error', TRUE),
('email_format', 'customers', 'email', 'format', '{"pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"}', 'warning', TRUE);

-- Data validation procedure
CREATE PROCEDURE sp_validate_data_quality
    @pipeline_id VARCHAR(100),
    @validation_date DATE = NULL
AS
BEGIN
    DECLARE @validation_date_param DATE = COALESCE(@validation_date, CAST(GETDATE() AS DATE));
    DECLARE @source_tables JSON;
    
    SELECT @source_tables = source_tables 
    FROM ml_pipeline_config 
    WHERE pipeline_id = @pipeline_id;
    
    -- Create validation results table
    CREATE TABLE #validation_results (
        rule_id VARCHAR(100),
        table_name VARCHAR(100),
        column_name VARCHAR(100),
        failed_records INT,
        total_records INT,
        failure_rate DECIMAL(5,4),
        severity VARCHAR(20)
    );
    
    -- Execute validation rules
    DECLARE @rule_id VARCHAR(100), @table_name VARCHAR(100), @column_name VARCHAR(100);
    DECLARE @rule_type VARCHAR(50), @rule_parameters JSON, @severity VARCHAR(20);
    
    DECLARE validation_cursor CURSOR FOR
    SELECT rule_id, table_name, column_name, rule_type, rule_parameters, severity
    FROM data_quality_rules
    WHERE is_active = TRUE;
    
    OPEN validation_cursor;
    FETCH NEXT FROM validation_cursor INTO @rule_id, @table_name, @column_name, @rule_type, @rule_parameters, @severity;
    
    WHILE @@FETCH_STATUS = 0
    BEGIN
        DECLARE @sql NVARCHAR(MAX);
        DECLARE @failed_count INT, @total_count INT;
        
        -- Build validation query based on rule type
        IF @rule_type = 'not_null'
        BEGIN
            SET @sql = N'
                SELECT 
                    @failed_count = COUNT(*),
                    @total_count = (SELECT COUNT(*) FROM ' + QUOTENAME(@table_name) + ' WHERE created_date >= @validation_date)
                FROM ' + QUOTENAME(@table_name) + '
                WHERE ' + QUOTENAME(@column_name) + ' IS NULL 
                AND created_date >= @validation_date';
        END
        ELSE IF @rule_type = 'range'
        BEGIN
            DECLARE @min_value DECIMAL(18,2) = JSON_VALUE(@rule_parameters, '$.min');
            DECLARE @max_value DECIMAL(18,2) = JSON_VALUE(@rule_parameters, '$.max');
            
            SET @sql = N'
                SELECT 
                    @failed_count = COUNT(*),
                    @total_count = (SELECT COUNT(*) FROM ' + QUOTENAME(@table_name) + ' WHERE created_date >= @validation_date)
                FROM ' + QUOTENAME(@table_name) + '
                WHERE (' + QUOTENAME(@column_name) + ' < ' + CAST(@min_value AS VARCHAR) + ' 
                   OR ' + QUOTENAME(@column_name) + ' > ' + CAST(@max_value AS VARCHAR) + ')
                AND created_date >= @validation_date';
        END;
        
        -- Execute validation
        EXEC sp_executesql @sql, 
             N'@failed_count INT OUTPUT, @total_count INT OUTPUT, @validation_date DATE',
             @failed_count OUTPUT, @total_count OUTPUT, @validation_date_param;
        
        -- Store results
        INSERT INTO #validation_results 
        VALUES (@rule_id, @table_name, @column_name, @failed_count, @total_count, 
                CASE WHEN @total_count > 0 THEN CAST(@failed_count AS DECIMAL(5,4)) / @total_count ELSE 0 END, 
                @severity);
        
        FETCH NEXT FROM validation_cursor INTO @rule_id, @table_name, @column_name, @rule_type, @rule_parameters, @severity;
    END;
    
    CLOSE validation_cursor;
    DEALLOCATE validation_cursor;
    
    -- Log validation results
    INSERT INTO data_quality_log (pipeline_id, validation_date, rule_id, failed_records, total_records, failure_rate, severity)
    SELECT @pipeline_id, @validation_date_param, rule_id, failed_records, total_records, failure_rate, severity
    FROM #validation_results;
    
    -- Check for critical failures
    IF EXISTS (SELECT 1 FROM #validation_results WHERE severity = 'error' AND failure_rate > 0.05)
    BEGIN
        THROW 50002, 'Data quality validation failed with critical errors', 1;
    END;
    
    DROP TABLE #validation_results;
END;
```

#### 3. Feature Engineering Pipeline

```sql
-- Create feature engineering configuration
CREATE TABLE feature_engineering_steps (
    step_id VARCHAR(100) PRIMARY KEY,
    pipeline_id VARCHAR(100),
    step_order INT,
    step_type VARCHAR(50), -- 'aggregation', 'transformation', 'derived', 'encoding'
    source_tables JSON,
    target_table VARCHAR(100),
    transformation_logic TEXT,
    dependencies JSON, -- List of previous step_ids
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (pipeline_id) REFERENCES ml_pipeline_config(pipeline_id)
);

-- Feature engineering execution procedure
CREATE PROCEDURE sp_execute_feature_engineering
    @pipeline_id VARCHAR(100),
    @processing_date DATE = NULL
AS
BEGIN
    DECLARE @processing_date_param DATE = COALESCE(@processing_date, CAST(GETDATE() AS DATE));
    DECLARE @step_id VARCHAR(100), @step_order INT, @step_type VARCHAR(50);
    DECLARE @source_tables JSON, @target_table VARCHAR(100), @transformation_logic TEXT;
    
    -- Execute steps in order
    DECLARE feature_cursor CURSOR FOR
    SELECT step_id, step_order, step_type, source_tables, target_table, transformation_logic
    FROM feature_engineering_steps
    WHERE pipeline_id = @pipeline_id AND is_active = TRUE
    ORDER BY step_order;
    
    OPEN feature_cursor;
    FETCH NEXT FROM feature_cursor INTO @step_id, @step_order, @step_type, @source_tables, @target_table, @transformation_logic;
    
    WHILE @@FETCH_STATUS = 0
    BEGIN
        BEGIN TRY
            -- Log step start
            INSERT INTO pipeline_execution_log (pipeline_id, step_id, start_time, status)
            VALUES (@pipeline_id, @step_id, GETDATE(), 'running');
            
            -- Execute transformation based on step type
            IF @step_type = 'aggregation'
            BEGIN
                -- Example: Customer transaction aggregations
                EXEC sp_execute_sql @transformation_logic;
            END
            ELSE IF @step_type = 'derived'
            BEGIN
                -- Example: Calculate derived features like RFM scores
                DECLARE @sql NVARCHAR(MAX) = REPLACE(@transformation_logic, '@processing_date', '''' + CAST(@processing_date_param AS VARCHAR) + '''');
                EXEC sp_executesql @sql;
            END;
            
            -- Log step completion
            UPDATE pipeline_execution_log 
            SET end_time = GETDATE(), status = 'completed', record_count = @@ROWCOUNT
            WHERE pipeline_id = @pipeline_id AND step_id = @step_id AND start_time = (
                SELECT MAX(start_time) FROM pipeline_execution_log 
                WHERE pipeline_id = @pipeline_id AND step_id = @step_id
            );
            
        END TRY
        BEGIN CATCH
            -- Log step failure
            UPDATE pipeline_execution_log 
            SET end_time = GETDATE(), status = 'failed', error_message = ERROR_MESSAGE()
            WHERE pipeline_id = @pipeline_id AND step_id = @step_id AND start_time = (
                SELECT MAX(start_time) FROM pipeline_execution_log 
                WHERE pipeline_id = @pipeline_id AND step_id = @step_id
            );
            
            THROW;
        END CATCH;
        
        FETCH NEXT FROM feature_cursor INTO @step_id, @step_order, @step_type, @source_tables, @target_table, @transformation_logic;
    END;
    
    CLOSE feature_cursor;
    DEALLOCATE feature_cursor;
END;
```

#### 4. Incremental Processing and Change Detection

```sql
-- Create change detection framework
CREATE TABLE table_change_tracking (
    table_name VARCHAR(100),
    last_processed_timestamp TIMESTAMP,
    last_processed_id BIGINT,
    change_detection_method VARCHAR(50), -- 'timestamp', 'sequence', 'checksum'
    PRIMARY KEY (table_name)
);

-- Incremental processing procedure
CREATE PROCEDURE sp_process_incremental_changes
    @pipeline_id VARCHAR(100),
    @table_name VARCHAR(100)
AS
BEGIN
    DECLARE @last_processed_timestamp TIMESTAMP;
    DECLARE @last_processed_id BIGINT;
    DECLARE @current_max_timestamp TIMESTAMP;
    DECLARE @current_max_id BIGINT;
    
    -- Get last processed markers
    SELECT 
        @last_processed_timestamp = last_processed_timestamp,
        @last_processed_id = last_processed_id
    FROM table_change_tracking
    WHERE table_name = @table_name;
    
    -- Handle first run
    IF @last_processed_timestamp IS NULL
    BEGIN
        SET @last_processed_timestamp = DATEADD(day, -30, GETDATE()); -- Start with 30 days ago
        SET @last_processed_id = 0;
    END;
    
    -- Process changes based on table
    IF @table_name = 'customers'
    BEGIN
        -- Process new/updated customers
        INSERT INTO customer_features_staging (
            customer_id,
            demographic_score,
            behavioral_score,
            engagement_score,
            processing_date
        )
        SELECT 
            c.customer_id,
            -- Demographic scoring
            CASE 
                WHEN c.age BETWEEN 25 AND 45 AND c.income_bracket IN ('high', 'medium') THEN 0.8
                WHEN c.age BETWEEN 18 AND 65 THEN 0.6
                ELSE 0.3
            END as demographic_score,
            -- Behavioral scoring (requires join with transactions)
            COALESCE(b.avg_monthly_spend / 1000.0, 0) as behavioral_score,
            -- Engagement scoring
            CASE 
                WHEN c.last_login_date >= DATEADD(day, -7, GETDATE()) THEN 1.0
                WHEN c.last_login_date >= DATEADD(day, -30, GETDATE()) THEN 0.6
                ELSE 0.2
            END as engagement_score,
            GETDATE() as processing_date
        FROM customers c
        LEFT JOIN (
            SELECT 
                customer_id,
                AVG(amount) as avg_monthly_spend
            FROM transactions
            WHERE transaction_date >= DATEADD(month, -3, GETDATE())
            GROUP BY customer_id
        ) b ON c.customer_id = b.customer_id
        WHERE c.updated_timestamp > @last_processed_timestamp
           OR c.customer_id > @last_processed_id;
    END;
    
    -- Update tracking information
    SELECT 
        @current_max_timestamp = MAX(updated_timestamp),
        @current_max_id = MAX(customer_id)
    FROM customers
    WHERE updated_timestamp > @last_processed_timestamp;
    
    UPDATE table_change_tracking
    SET 
        last_processed_timestamp = @current_max_timestamp,
        last_processed_id = @current_max_id
    WHERE table_name = @table_name;
    
    -- If no existing record, insert one
    IF @@ROWCOUNT = 0
    BEGIN
        INSERT INTO table_change_tracking (table_name, last_processed_timestamp, last_processed_id, change_detection_method)
        VALUES (@table_name, @current_max_timestamp, @current_max_id, 'timestamp');
    END;
END;
```

#### 5. Model Serving Pipeline

```sql
-- Create model metadata table
CREATE TABLE ml_model_metadata (
    model_id VARCHAR(100) PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    model_type VARCHAR(50), -- 'classification', 'regression', 'clustering'
    feature_schema JSON, -- Expected feature schema
    model_artifact_path VARCHAR(500),
    deployment_status VARCHAR(50), -- 'active', 'staged', 'retired'
    performance_metrics JSON,
    created_date TIMESTAMP,
    deployed_date TIMESTAMP,
    retired_date TIMESTAMP
);

-- Create feature serving table
CREATE TABLE feature_serving_cache (
    entity_id BIGINT,
    entity_type VARCHAR(50), -- 'customer', 'product', etc.
    feature_vector JSON,
    feature_timestamp TIMESTAMP,
    cache_expiry TIMESTAMP,
    PRIMARY KEY (entity_id, entity_type)
);

-- Real-time feature serving procedure
CREATE PROCEDURE sp_get_features_for_prediction
    @entity_ids VARCHAR(MAX), -- Comma-separated entity IDs
    @entity_type VARCHAR(50),
    @model_id VARCHAR(100)
AS
BEGIN
    DECLARE @feature_schema JSON;
    DECLARE @cache_duration_minutes INT = 60; -- Cache features for 1 hour
    
    -- Get required feature schema for the model
    SELECT @feature_schema = feature_schema
    FROM ml_model_metadata
    WHERE model_id = @model_id AND deployment_status = 'active';
    
    IF @feature_schema IS NULL
    BEGIN
        THROW 50003, 'Model not found or not active', 1;
    END;
    
    -- Check cache first
    WITH entity_list AS (
        SELECT CAST(value AS BIGINT) as entity_id
        FROM STRING_SPLIT(@entity_ids, ',')
    ),
    cached_features AS (
        SELECT 
            el.entity_id,
            fsc.feature_vector,
            fsc.feature_timestamp,
            CASE WHEN fsc.cache_expiry > GETDATE() THEN 1 ELSE 0 END as is_valid
        FROM entity_list el
        LEFT JOIN feature_serving_cache fsc ON el.entity_id = fsc.entity_id AND fsc.entity_type = @entity_type
    ),
    missing_entities AS (
        SELECT entity_id 
        FROM cached_features 
        WHERE feature_vector IS NULL OR is_valid = 0
    )
    -- Compute features for missing entities
    INSERT INTO feature_serving_cache (entity_id, entity_type, feature_vector, feature_timestamp, cache_expiry)
    SELECT 
        me.entity_id,
        @entity_type,
        (
            SELECT 
                demographic_score,
                behavioral_score,
                engagement_score,
                recency_score,
                frequency_score,
                monetary_score
            FROM customer_features_current cfc
            WHERE cfc.customer_id = me.entity_id
            FOR JSON PATH, WITHOUT_ARRAY_WRAPPER
        ) as feature_vector,
        GETDATE() as feature_timestamp,
        DATEADD(minute, @cache_duration_minutes, GETDATE()) as cache_expiry
    FROM missing_entities me
    ON CONFLICT (entity_id, entity_type) 
    DO UPDATE SET 
        feature_vector = EXCLUDED.feature_vector,
        feature_timestamp = EXCLUDED.feature_timestamp,
        cache_expiry = EXCLUDED.cache_expiry;
    
    -- Return all requested features
    SELECT 
        el.entity_id,
        fsc.feature_vector,
        fsc.feature_timestamp
    FROM entity_list el
    JOIN feature_serving_cache fsc ON el.entity_id = fsc.entity_id AND fsc.entity_type = @entity_type
    WHERE fsc.cache_expiry > GETDATE();
END;
```

#### 6. Monitoring and Alerting System

```sql
-- Create monitoring tables
CREATE TABLE pipeline_health_metrics (
    metric_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    pipeline_id VARCHAR(100),
    metric_name VARCHAR(100),
    metric_value DECIMAL(18,4),
    metric_timestamp TIMESTAMP DEFAULT GETDATE(),
    threshold_min DECIMAL(18,4),
    threshold_max DECIMAL(18,4),
    alert_severity VARCHAR(20) -- 'info', 'warning', 'error', 'critical'
);

CREATE TABLE pipeline_alerts (
    alert_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    pipeline_id VARCHAR(100),
    alert_type VARCHAR(100),
    alert_message TEXT,
    severity VARCHAR(20),
    created_date TIMESTAMP DEFAULT GETDATE(),
    resolved_date TIMESTAMP,
    is_acknowledged BOOLEAN DEFAULT FALSE
);

-- Monitoring procedure
CREATE PROCEDURE sp_monitor_pipeline_health
    @pipeline_id VARCHAR(100)
AS
BEGIN
    -- Check data freshness
    INSERT INTO pipeline_health_metrics (pipeline_id, metric_name, metric_value, threshold_max, alert_severity)
    SELECT 
        @pipeline_id,
        'data_freshness_hours',
        DATEDIFF(hour, MAX(feature_timestamp), GETDATE()),
        24, -- Alert if data is more than 24 hours old
        'warning'
    FROM feature_serving_cache;
    
    -- Check pipeline execution frequency
    INSERT INTO pipeline_health_metrics (pipeline_id, metric_name, metric_value, threshold_max, alert_severity)
    SELECT 
        @pipeline_id,
        'last_execution_hours_ago',
        DATEDIFF(hour, MAX(start_time), GETDATE()),
        26, -- Alert if pipeline hasn't run in 26 hours
        'error'
    FROM pipeline_execution_log
    WHERE pipeline_id = @pipeline_id AND status = 'completed';
    
    -- Check error rate
    INSERT INTO pipeline_health_metrics (pipeline_id, metric_name, metric_value, threshold_max, alert_severity)
    SELECT 
        @pipeline_id,
        'error_rate_24h',
        CAST(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS DECIMAL) / COUNT(*),
        0.1, -- Alert if error rate > 10%
        'warning'
    FROM pipeline_execution_log
    WHERE pipeline_id = @pipeline_id 
      AND start_time >= DATEADD(hour, -24, GETDATE());
    
    -- Generate alerts for threshold violations
    INSERT INTO pipeline_alerts (pipeline_id, alert_type, alert_message, severity)
    SELECT 
        pipeline_id,
        'threshold_violation',
        'Metric ' + metric_name + ' value ' + CAST(metric_value AS VARCHAR) + ' exceeds threshold ' + CAST(threshold_max AS VARCHAR),
        alert_severity
    FROM pipeline_health_metrics
    WHERE metric_timestamp >= DATEADD(minute, -5, GETDATE())
      AND metric_value > threshold_max
      AND pipeline_id = @pipeline_id;
END;
```

#### 7. Orchestration and Scheduling

```sql
-- Create job scheduling table
CREATE TABLE pipeline_schedule (
    schedule_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    pipeline_id VARCHAR(100),
    schedule_type VARCHAR(50), -- 'cron', 'interval', 'event_driven'
    schedule_expression VARCHAR(100),
    next_execution_time TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    max_retries INT DEFAULT 3,
    retry_delay_minutes INT DEFAULT 15
);

-- Main orchestration procedure
CREATE PROCEDURE sp_orchestrate_ml_pipeline
    @pipeline_id VARCHAR(100),
    @execution_mode VARCHAR(50) = 'scheduled' -- 'scheduled', 'manual', 'retry'
AS
BEGIN
    DECLARE @start_time TIMESTAMP = GETDATE();
    DECLARE @execution_id BIGINT;
    
    BEGIN TRY
        -- Create execution record
        INSERT INTO pipeline_execution_master (pipeline_id, execution_mode, start_time, status)
        VALUES (@pipeline_id, @execution_mode, @start_time, 'running');
        SET @execution_id = SCOPE_IDENTITY();
        
        -- Step 1: Data Validation
        EXEC sp_validate_data_quality @pipeline_id;
        
        -- Step 2: Incremental Processing
        DECLARE @table_name VARCHAR(100);
        DECLARE @source_tables JSON;
        
        SELECT @source_tables = source_tables 
        FROM ml_pipeline_config 
        WHERE pipeline_id = @pipeline_id;
        
        DECLARE table_cursor CURSOR FOR
        SELECT value as table_name
        FROM OPENJSON(@source_tables);
        
        OPEN table_cursor;
        FETCH NEXT FROM table_cursor INTO @table_name;
        
        WHILE @@FETCH_STATUS = 0
        BEGIN
            EXEC sp_process_incremental_changes @pipeline_id, @table_name;
            FETCH NEXT FROM table_cursor INTO @table_name;
        END;
        
        CLOSE table_cursor;
        DEALLOCATE table_cursor;
        
        -- Step 3: Feature Engineering
        EXEC sp_execute_feature_engineering @pipeline_id;
        
        -- Step 4: Update Model Serving Cache
        EXEC sp_refresh_feature_serving_cache @pipeline_id;
        
        -- Step 5: Monitor Health
        EXEC sp_monitor_pipeline_health @pipeline_id;
        
        -- Mark execution as completed
        UPDATE pipeline_execution_master
        SET end_time = GETDATE(), status = 'completed'
        WHERE execution_id = @execution_id;
        
        -- Schedule next execution
        UPDATE pipeline_schedule
        SET next_execution_time = CASE
            WHEN schedule_type = 'cron' THEN dbo.fn_calculate_next_cron_time(schedule_expression, GETDATE())
            WHEN schedule_type = 'interval' THEN DATEADD(minute, CAST(schedule_expression AS INT), GETDATE())
        END
        WHERE pipeline_id = @pipeline_id;
        
    END TRY
    BEGIN CATCH
        -- Mark execution as failed
        UPDATE pipeline_execution_master
        SET end_time = GETDATE(), status = 'failed', error_message = ERROR_MESSAGE()
        WHERE execution_id = @execution_id;
        
        -- Create critical alert
        INSERT INTO pipeline_alerts (pipeline_id, alert_type, alert_message, severity)
        VALUES (@pipeline_id, 'execution_failure', 'Pipeline execution failed: ' + ERROR_MESSAGE(), 'critical');
        
        THROW;
    END CATCH;
END;
```

### Best Practices and Key Considerations

#### 8. Data Lineage and Versioning

```sql
-- Track data lineage
CREATE TABLE data_lineage (
    lineage_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    source_table VARCHAR(100),
    target_table VARCHAR(100),
    transformation_type VARCHAR(50),
    pipeline_id VARCHAR(100),
    execution_timestamp TIMESTAMP,
    record_count BIGINT
);

-- Version control for features
CREATE TABLE feature_versions (
    version_id VARCHAR(100) PRIMARY KEY,
    feature_group VARCHAR(100),
    feature_schema JSON,
    computation_logic TEXT,
    created_date TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

This comprehensive system design ensures:

1. **Reliability**: Robust error handling and retry mechanisms
2. **Scalability**: Incremental processing and parallel execution
3. **Observability**: Comprehensive monitoring and alerting
4. **Data Quality**: Built-in validation and quality checks
5. **Flexibility**: Configurable pipelines and feature engineering
6. **Performance**: Caching and optimized data access patterns
7. **Governance**: Data lineage tracking and version control

The system can handle both batch and real-time ML model feeding requirements while maintaining high availability and data quality standards.

---

## Question 6

**How would you extract and prepare aconfusion matrixfor a classification problem using SQL?**

**Answer:**

### Comprehensive Confusion Matrix Implementation in SQL

Building confusion matrices in SQL requires systematic approaches to handle predictions, actuals, and various classification scenarios including binary, multi-class, and multi-label problems.

#### 1. Basic Binary Classification Confusion Matrix

```sql
-- Create sample prediction results table
CREATE TABLE model_predictions (
    prediction_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    customer_id BIGINT,
    actual_label VARCHAR(20),
    predicted_label VARCHAR(20),
    prediction_probability DECIMAL(5,4),
    prediction_date TIMESTAMP DEFAULT GETDATE(),
    model_version VARCHAR(50)
);

-- Insert sample data
INSERT INTO model_predictions (customer_id, actual_label, predicted_label, prediction_probability, model_version)
VALUES 
(1001, 'Churn', 'Churn', 0.8500, 'churn_model_v1.2'),
(1002, 'Stay', 'Stay', 0.9200, 'churn_model_v1.2'),
(1003, 'Churn', 'Stay', 0.3500, 'churn_model_v1.2'),
(1004, 'Stay', 'Churn', 0.6500, 'churn_model_v1.2'),
(1005, 'Stay', 'Stay', 0.7800, 'churn_model_v1.2');

-- Basic 2x2 Confusion Matrix
WITH confusion_matrix_data AS (
    SELECT 
        actual_label,
        predicted_label,
        COUNT(*) as count
    FROM model_predictions
    WHERE model_version = 'churn_model_v1.2'
    GROUP BY actual_label, predicted_label
),
confusion_matrix_pivot AS (
    SELECT 
        actual_label,
        SUM(CASE WHEN predicted_label = 'Churn' THEN count ELSE 0 END) as Predicted_Churn,
        SUM(CASE WHEN predicted_label = 'Stay' THEN count ELSE 0 END) as Predicted_Stay
    FROM confusion_matrix_data
    GROUP BY actual_label
)
SELECT 
    'Confusion Matrix' as Matrix_Type,
    actual_label as Actual_Label,
    Predicted_Churn,
    Predicted_Stay,
    (Predicted_Churn + Predicted_Stay) as Total
FROM confusion_matrix_pivot
UNION ALL
SELECT 
    'Total' as Matrix_Type,
    'All' as Actual_Label,
    SUM(Predicted_Churn) as Predicted_Churn,
    SUM(Predicted_Stay) as Predicted_Stay,
    SUM(Predicted_Churn + Predicted_Stay) as Total
FROM confusion_matrix_pivot;
```

#### 2. Multi-Class Confusion Matrix

```sql
-- Multi-class classification example (Product Category Prediction)
CREATE TABLE product_classification_results (
    prediction_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    product_id BIGINT,
    actual_category VARCHAR(50),
    predicted_category VARCHAR(50),
    confidence_score DECIMAL(5,4),
    prediction_timestamp TIMESTAMP DEFAULT GETDATE(),
    model_id VARCHAR(100)
);

-- Generate multi-class confusion matrix
CREATE PROCEDURE sp_generate_multiclass_confusion_matrix
    @model_id VARCHAR(100),
    @start_date DATE = NULL,
    @end_date DATE = NULL
AS
BEGIN
    DECLARE @start_date_param DATE = COALESCE(@start_date, DATEADD(day, -30, GETDATE()));
    DECLARE @end_date_param DATE = COALESCE(@end_date, GETDATE());
    
    -- Get all unique classes
    WITH unique_classes AS (
        SELECT DISTINCT actual_category as class_name
        FROM product_classification_results
        WHERE model_id = @model_id
          AND prediction_timestamp BETWEEN @start_date_param AND @end_date_param
        UNION
        SELECT DISTINCT predicted_category as class_name
        FROM product_classification_results
        WHERE model_id = @model_id
          AND prediction_timestamp BETWEEN @start_date_param AND @end_date_param
    ),
    confusion_data AS (
        SELECT 
            actual_category,
            predicted_category,
            COUNT(*) as prediction_count
        FROM product_classification_results
        WHERE model_id = @model_id
          AND prediction_timestamp BETWEEN @start_date_param AND @end_date_param
        GROUP BY actual_category, predicted_category
    ),
    -- Create full matrix with zeros for missing combinations
    full_matrix AS (
        SELECT 
            uc1.class_name as actual_class,
            uc2.class_name as predicted_class,
            COALESCE(cd.prediction_count, 0) as count
        FROM unique_classes uc1
        CROSS JOIN unique_classes uc2
        LEFT JOIN confusion_data cd ON uc1.class_name = cd.actual_category 
                                   AND uc2.class_name = cd.predicted_category
    )
    -- Pivot the confusion matrix
    SELECT 
        actual_class,
        SUM(CASE WHEN predicted_class = 'Electronics' THEN count ELSE 0 END) as Pred_Electronics,
        SUM(CASE WHEN predicted_class = 'Clothing' THEN count ELSE 0 END) as Pred_Clothing,
        SUM(CASE WHEN predicted_class = 'Books' THEN count ELSE 0 END) as Pred_Books,
        SUM(CASE WHEN predicted_class = 'Home_Garden' THEN count ELSE 0 END) as Pred_Home_Garden,
        SUM(CASE WHEN predicted_class = 'Sports' THEN count ELSE 0 END) as Pred_Sports,
        SUM(count) as Total_Actual
    FROM full_matrix
    GROUP BY actual_class
    
    UNION ALL
    
    -- Add totals row
    SELECT 
        'Total_Predicted' as actual_class,
        SUM(CASE WHEN predicted_class = 'Electronics' THEN count ELSE 0 END) as Pred_Electronics,
        SUM(CASE WHEN predicted_class = 'Clothing' THEN count ELSE 0 END) as Pred_Clothing,
        SUM(CASE WHEN predicted_class = 'Books' THEN count ELSE 0 END) as Pred_Books,
        SUM(CASE WHEN predicted_class = 'Home_Garden' THEN count ELSE 0 END) as Pred_Home_Garden,
        SUM(CASE WHEN predicted_class = 'Sports' THEN count ELSE 0 END) as Pred_Sports,
        SUM(count) as Total_Actual
    FROM full_matrix;
END;
```

#### 3. Dynamic Confusion Matrix for Any Number of Classes

```sql
-- Dynamic confusion matrix generator
CREATE PROCEDURE sp_generate_dynamic_confusion_matrix
    @table_name VARCHAR(100),
    @actual_column VARCHAR(100),
    @predicted_column VARCHAR(100),
    @filter_conditions VARCHAR(500) = ''
AS
BEGIN
    DECLARE @sql NVARCHAR(MAX);
    DECLARE @pivot_columns NVARCHAR(MAX);
    DECLARE @case_statements NVARCHAR(MAX);
    
    -- Get unique class names
    SET @sql = N'
    WITH unique_classes AS (
        SELECT DISTINCT ' + QUOTENAME(@actual_column) + ' as class_name
        FROM ' + QUOTENAME(@table_name) + 
        CASE WHEN @filter_conditions <> '' THEN ' WHERE ' + @filter_conditions ELSE '' END + '
        UNION
        SELECT DISTINCT ' + QUOTENAME(@predicted_column) + ' as class_name
        FROM ' + QUOTENAME(@table_name) + 
        CASE WHEN @filter_conditions <> '' THEN ' WHERE ' + @filter_conditions ELSE '' END + '
    )
    SELECT @pivot_columns = STRING_AGG(''Pred_'' + REPLACE(class_name, '' '', ''_''), '',''),
           @case_statements = STRING_AGG(
               ''SUM(CASE WHEN '' + QUOTENAME(@predicted_column) + '' = '''''' + class_name + '''''' THEN count ELSE 0 END) as Pred_'' + REPLACE(class_name, '' '', ''_''),
               '','' + CHAR(13)
           )
    FROM unique_classes';
    
    EXEC sp_executesql @sql, 
         N'@pivot_columns NVARCHAR(MAX) OUTPUT, @case_statements NVARCHAR(MAX) OUTPUT', 
         @pivot_columns OUTPUT, @case_statements OUTPUT;
    
    -- Build final confusion matrix query
    SET @sql = N'
    WITH confusion_data AS (
        SELECT 
            ' + QUOTENAME(@actual_column) + ' as actual_class,
            ' + QUOTENAME(@predicted_column) + ' as predicted_class,
            COUNT(*) as count
        FROM ' + QUOTENAME(@table_name) + 
        CASE WHEN @filter_conditions <> '' THEN ' WHERE ' + @filter_conditions ELSE '' END + '
        GROUP BY ' + QUOTENAME(@actual_column) + ', ' + QUOTENAME(@predicted_column) + '
    )
    SELECT 
        actual_class,
        ' + @case_statements + ',
        SUM(count) as Total_Actual
    FROM confusion_data
    GROUP BY actual_class
    ORDER BY actual_class';
    
    EXEC sp_executesql @sql;
END;
```

#### 4. Confusion Matrix with Performance Metrics

```sql
-- Calculate comprehensive metrics from confusion matrix
CREATE FUNCTION fn_calculate_classification_metrics
(
    @true_positives INT,
    @false_positives INT,
    @true_negatives INT,
    @false_negatives INT
)
RETURNS TABLE
AS
RETURN
(
    SELECT 
        @true_positives as True_Positives,
        @false_positives as False_Positives,
        @true_negatives as True_Negatives,
        @false_negatives as False_Negatives,
        
        -- Accuracy
        CAST((@true_positives + @true_negatives) AS FLOAT) / 
        NULLIF((@true_positives + @false_positives + @true_negatives + @false_negatives), 0) as Accuracy,
        
        -- Precision
        CAST(@true_positives AS FLOAT) / NULLIF((@true_positives + @false_positives), 0) as Precision,
        
        -- Recall (Sensitivity)
        CAST(@true_positives AS FLOAT) / NULLIF((@true_positives + @false_negatives), 0) as Recall,
        
        -- Specificity
        CAST(@true_negatives AS FLOAT) / NULLIF((@true_negatives + @false_positives), 0) as Specificity,
        
        -- F1 Score
        2.0 * 
        (CAST(@true_positives AS FLOAT) / NULLIF((@true_positives + @false_positives), 0)) *
        (CAST(@true_positives AS FLOAT) / NULLIF((@true_positives + @false_negatives), 0)) /
        NULLIF((
            (CAST(@true_positives AS FLOAT) / NULLIF((@true_positives + @false_positives), 0)) +
            (CAST(@true_positives AS FLOAT) / NULLIF((@true_positives + @false_negatives), 0))
        ), 0) as F1_Score
);

-- Comprehensive confusion matrix analysis
CREATE PROCEDURE sp_analyze_binary_classification
    @model_id VARCHAR(100),
    @positive_class VARCHAR(50),
    @evaluation_date DATE = NULL
AS
BEGIN
    DECLARE @evaluation_date_param DATE = COALESCE(@evaluation_date, GETDATE());
    
    WITH confusion_matrix AS (
        SELECT 
            SUM(CASE WHEN actual_label = @positive_class AND predicted_label = @positive_class THEN 1 ELSE 0 END) as TP,
            SUM(CASE WHEN actual_label != @positive_class AND predicted_label = @positive_class THEN 1 ELSE 0 END) as FP,
            SUM(CASE WHEN actual_label != @positive_class AND predicted_label != @positive_class THEN 1 ELSE 0 END) as TN,
            SUM(CASE WHEN actual_label = @positive_class AND predicted_label != @positive_class THEN 1 ELSE 0 END) as FN
        FROM model_predictions
        WHERE model_version = @model_id
          AND CAST(prediction_date AS DATE) = @evaluation_date_param
    )
    SELECT 
        'Confusion Matrix Analysis' as Analysis_Type,
        cm.TP,
        cm.FP,
        cm.TN,
        cm.FN,
        metrics.*
    FROM confusion_matrix cm
    CROSS APPLY fn_calculate_classification_metrics(cm.TP, cm.FP, cm.TN, cm.FN) metrics;
END;
```

#### 5. Multi-Label Classification Confusion Matrix

```sql
-- Multi-label classification (e.g., product can belong to multiple categories)
CREATE TABLE multilabel_predictions (
    prediction_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    product_id BIGINT,
    actual_labels VARCHAR(500), -- Comma-separated labels
    predicted_labels VARCHAR(500), -- Comma-separated labels
    prediction_scores JSON, -- {"electronics": 0.8, "gadgets": 0.6}
    model_version VARCHAR(50),
    prediction_date TIMESTAMP DEFAULT GETDATE()
);

-- Multi-label confusion matrix
CREATE PROCEDURE sp_multilabel_confusion_matrix
    @model_version VARCHAR(50),
    @threshold DECIMAL(3,2) = 0.5
AS
BEGIN
    -- Normalize multi-label data into individual label rows
    WITH normalized_actuals AS (
        SELECT 
            prediction_id,
            product_id,
            TRIM(value) as actual_label,
            1 as actual_value
        FROM multilabel_predictions
        CROSS APPLY STRING_SPLIT(actual_labels, ',')
        WHERE model_version = @model_version
    ),
    normalized_predictions AS (
        SELECT 
            p.prediction_id,
            p.product_id,
            labels.label_name as predicted_label,
            CASE WHEN CAST(labels.label_score AS DECIMAL(3,2)) >= @threshold THEN 1 ELSE 0 END as predicted_value
        FROM multilabel_predictions p
        CROSS APPLY (
            SELECT 
                JSON_VALUE(prediction_scores, '$.' + key) as label_score,
                key as label_name
            FROM OPENJSON(prediction_scores)
        ) labels
        WHERE model_version = @model_version
    ),
    all_labels AS (
        SELECT DISTINCT actual_label as label_name FROM normalized_actuals
        UNION
        SELECT DISTINCT predicted_label as label_name FROM normalized_predictions
    ),
    label_combinations AS (
        SELECT 
            al.label_name,
            p.product_id,
            COALESCE(na.actual_value, 0) as actual_value,
            COALESCE(np.predicted_value, 0) as predicted_value
        FROM all_labels al
        CROSS JOIN (SELECT DISTINCT product_id FROM multilabel_predictions WHERE model_version = @model_version) p
        LEFT JOIN normalized_actuals na ON al.label_name = na.actual_label AND p.product_id = na.product_id
        LEFT JOIN normalized_predictions np ON al.label_name = np.predicted_label AND p.product_id = np.product_id
    ),
    per_label_metrics AS (
        SELECT 
            label_name,
            SUM(CASE WHEN actual_value = 1 AND predicted_value = 1 THEN 1 ELSE 0 END) as TP,
            SUM(CASE WHEN actual_value = 0 AND predicted_value = 1 THEN 1 ELSE 0 END) as FP,
            SUM(CASE WHEN actual_value = 0 AND predicted_value = 0 THEN 1 ELSE 0 END) as TN,
            SUM(CASE WHEN actual_value = 1 AND predicted_value = 0 THEN 1 ELSE 0 END) as FN
        FROM label_combinations
        GROUP BY label_name
    )
    SELECT 
        label_name,
        TP as True_Positives,
        FP as False_Positives,
        TN as True_Negatives,
        FN as False_Negatives,
        CAST((TP + TN) AS FLOAT) / NULLIF((TP + FP + TN + FN), 0) as Accuracy,
        CAST(TP AS FLOAT) / NULLIF((TP + FP), 0) as Precision,
        CAST(TP AS FLOAT) / NULLIF((TP + FN), 0) as Recall,
        2.0 * (CAST(TP AS FLOAT) / NULLIF((TP + FP), 0)) * (CAST(TP AS FLOAT) / NULLIF((TP + FN), 0)) /
        NULLIF((CAST(TP AS FLOAT) / NULLIF((TP + FP), 0)) + (CAST(TP AS FLOAT) / NULLIF((TP + FN), 0)), 0) as F1_Score
    FROM per_label_metrics
    ORDER BY label_name;
END;
```

#### 6. Time-Series Confusion Matrix Analysis

```sql
-- Temporal analysis of model performance
CREATE PROCEDURE sp_temporal_confusion_matrix
    @model_id VARCHAR(100),
    @time_granularity VARCHAR(20) = 'daily', -- 'hourly', 'daily', 'weekly', 'monthly'
    @start_date DATE,
    @end_date DATE
AS
BEGIN
    DECLARE @date_format VARCHAR(50);
    
    SET @date_format = CASE @time_granularity
        WHEN 'hourly' THEN 'yyyy-MM-dd HH'
        WHEN 'daily' THEN 'yyyy-MM-dd'
        WHEN 'weekly' THEN 'yyyy-ww'
        WHEN 'monthly' THEN 'yyyy-MM'
        ELSE 'yyyy-MM-dd'
    END;
    
    WITH temporal_confusion AS (
        SELECT 
            FORMAT(prediction_date, @date_format) as time_period,
            actual_label,
            predicted_label,
            COUNT(*) as prediction_count
        FROM model_predictions
        WHERE model_version = @model_id
          AND CAST(prediction_date AS DATE) BETWEEN @start_date AND @end_date
        GROUP BY FORMAT(prediction_date, @date_format), actual_label, predicted_label
    ),
    temporal_metrics AS (
        SELECT 
            time_period,
            SUM(CASE WHEN actual_label = 'Positive' AND predicted_label = 'Positive' THEN prediction_count ELSE 0 END) as TP,
            SUM(CASE WHEN actual_label = 'Negative' AND predicted_label = 'Positive' THEN prediction_count ELSE 0 END) as FP,
            SUM(CASE WHEN actual_label = 'Negative' AND predicted_label = 'Negative' THEN prediction_count ELSE 0 END) as TN,
            SUM(CASE WHEN actual_label = 'Positive' AND predicted_label = 'Negative' THEN prediction_count ELSE 0 END) as FN
        FROM temporal_confusion
        GROUP BY time_period
    )
    SELECT 
        time_period,
        TP, FP, TN, FN,
        CAST((TP + TN) AS FLOAT) / NULLIF((TP + FP + TN + FN), 0) as Accuracy,
        CAST(TP AS FLOAT) / NULLIF((TP + FP), 0) as Precision,
        CAST(TP AS FLOAT) / NULLIF((TP + FN), 0) as Recall,
        2.0 * (CAST(TP AS FLOAT) / NULLIF((TP + FP), 0)) * (CAST(TP AS FLOAT) / NULLIF((TP + FN), 0)) /
        NULLIF((CAST(TP AS FLOAT) / NULLIF((TP + FP), 0)) + (CAST(TP AS FLOAT) / NULLIF((TP + FN), 0)), 0) as F1_Score
    FROM temporal_metrics
    ORDER BY time_period;
END;
```

#### 7. Confusion Matrix Visualization Data

```sql
-- Generate data for confusion matrix heatmap visualization
CREATE PROCEDURE sp_confusion_matrix_heatmap_data
    @model_id VARCHAR(100),
    @normalize_type VARCHAR(20) = 'none' -- 'none', 'true', 'pred', 'all'
AS
BEGIN
    WITH confusion_raw AS (
        SELECT 
            actual_label,
            predicted_label,
            COUNT(*) as count
        FROM model_predictions
        WHERE model_version = @model_id
        GROUP BY actual_label, predicted_label
    ),
    confusion_totals AS (
        SELECT 
            cr.*,
            SUM(count) OVER (PARTITION BY actual_label) as actual_total,
            SUM(count) OVER (PARTITION BY predicted_label) as predicted_total,
            SUM(count) OVER () as grand_total
        FROM confusion_raw cr
    )
    SELECT 
        actual_label,
        predicted_label,
        count as raw_count,
        CASE @normalize_type
            WHEN 'true' THEN CAST(count AS FLOAT) / NULLIF(actual_total, 0)
            WHEN 'pred' THEN CAST(count AS FLOAT) / NULLIF(predicted_total, 0)
            WHEN 'all' THEN CAST(count AS FLOAT) / NULLIF(grand_total, 0)
            ELSE CAST(count AS FLOAT)
        END as normalized_value,
        @normalize_type as normalization_type
    FROM confusion_totals
    ORDER BY actual_label, predicted_label;
END;
```

#### 8. Model Comparison via Confusion Matrices

```sql
-- Compare multiple models using confusion matrix metrics
CREATE PROCEDURE sp_compare_model_performance
    @model_list VARCHAR(500), -- Comma-separated model IDs
    @evaluation_date DATE = NULL
AS
BEGIN
    DECLARE @evaluation_date_param DATE = COALESCE(@evaluation_date, GETDATE());
    
    WITH model_list_cte AS (
        SELECT TRIM(value) as model_id
        FROM STRING_SPLIT(@model_list, ',')
    ),
    model_confusion_metrics AS (
        SELECT 
            mp.model_version as model_id,
            SUM(CASE WHEN actual_label = 'Positive' AND predicted_label = 'Positive' THEN 1 ELSE 0 END) as TP,
            SUM(CASE WHEN actual_label = 'Negative' AND predicted_label = 'Positive' THEN 1 ELSE 0 END) as FP,
            SUM(CASE WHEN actual_label = 'Negative' AND predicted_label = 'Negative' THEN 1 ELSE 0 END) as TN,
            SUM(CASE WHEN actual_label = 'Positive' AND predicted_label = 'Negative' THEN 1 ELSE 0 END) as FN,
            COUNT(*) as total_predictions
        FROM model_predictions mp
        JOIN model_list_cte ml ON mp.model_version = ml.model_id
        WHERE CAST(mp.prediction_date AS DATE) = @evaluation_date_param
        GROUP BY mp.model_version
    )
    SELECT 
        model_id,
        TP, FP, TN, FN,
        total_predictions,
        CAST((TP + TN) AS FLOAT) / total_predictions as Accuracy,
        CAST(TP AS FLOAT) / NULLIF((TP + FP), 0) as Precision,
        CAST(TP AS FLOAT) / NULLIF((TP + FN), 0) as Recall,
        CAST(TN AS FLOAT) / NULLIF((TN + FP), 0) as Specificity,
        2.0 * (CAST(TP AS FLOAT) / NULLIF((TP + FP), 0)) * (CAST(TP AS FLOAT) / NULLIF((TP + FN), 0)) /
        NULLIF((CAST(TP AS FLOAT) / NULLIF((TP + FP), 0)) + (CAST(TP AS FLOAT) / NULLIF((TP + FN), 0)), 0) as F1_Score,
        -- Matthew's Correlation Coefficient
        (CAST(TP AS FLOAT) * TN - CAST(FP AS FLOAT) * FN) / 
        NULLIF(SQRT(CAST((TP + FP) AS FLOAT) * (TP + FN) * (TN + FP) * (TN + FN)), 0) as MCC
    FROM model_confusion_metrics
    ORDER BY F1_Score DESC;
END;
```

### Best Practices for Confusion Matrix Implementation

#### Key Considerations:
1. **Data Quality**: Ensure predictions and actuals are properly aligned
2. **Class Imbalance**: Consider stratified sampling for imbalanced datasets
3. **Temporal Validation**: Track performance over time to detect model drift
4. **Multi-class Handling**: Use appropriate aggregation for overall metrics
5. **Threshold Optimization**: For probabilistic models, optimize decision thresholds
6. **Missing Values**: Handle cases where predictions or actuals are missing
7. **Performance**: Use appropriate indexing for large prediction datasets

#### Common Pitfalls:
1. **Label Mismatch**: Inconsistent label formats between actual and predicted
2. **Data Leakage**: Using future information in historical evaluations
3. **Sample Bias**: Non-representative evaluation samples
4. **Metric Misinterpretation**: Understanding context-appropriate metrics

This comprehensive SQL-based approach provides robust confusion matrix analysis capabilities for various classification scenarios in machine learning applications.

---

## Question 7

**How would youlog and track predictionsmade by a Machine Learning model within a SQL environment?**

**Answer:**

### Comprehensive ML Prediction Logging and Tracking System

Building a robust prediction logging and tracking system in SQL requires comprehensive infrastructure for capturing, storing, analyzing, and monitoring machine learning model predictions in production environments.

#### 1. Core Prediction Logging Schema

```sql
-- Core prediction logging infrastructure
CREATE SCHEMA ml_prediction_logging;

-- Model registry for tracking deployed models
CREATE TABLE ml_prediction_logging.model_registry (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'classification', 'regression', 'clustering'
    algorithm_name VARCHAR(100),
    
    -- Model metadata
    feature_schema JSONB NOT NULL, -- Expected input features
    output_schema JSONB NOT NULL, -- Expected output format
    model_parameters JSONB, -- Hyperparameters and configuration
    training_dataset_id UUID,
    
    -- Deployment information
    deployment_environment VARCHAR(50), -- 'development', 'staging', 'production'
    deployment_date TIMESTAMP,
    deployment_status VARCHAR(20) DEFAULT 'active', -- 'active', 'inactive', 'retired'
    
    -- Performance expectations
    expected_accuracy DECIMAL(5,4),
    expected_latency_ms INTEGER,
    expected_throughput_rps INTEGER,
    
    -- Audit trail
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, model_version),
    INDEX idx_model_registry_status (deployment_status, deployment_environment),
    INDEX idx_model_registry_created (created_at DESC)
);

-- Comprehensive prediction logging table
CREATE TABLE ml_prediction_logging.prediction_logs (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES ml_prediction_logging.model_registry(model_id),
    
    -- Request context
    request_id UUID, -- For grouping related predictions
    session_id VARCHAR(255), -- User session tracking
    user_id UUID, -- User who triggered prediction
    application_id VARCHAR(100), -- Calling application
    
    -- Input data
    input_features JSONB NOT NULL, -- Raw input features
    feature_vector JSONB, -- Processed features used by model
    feature_hash VARCHAR(64), -- Hash for deduplication
    
    -- Prediction results
    prediction_output JSONB NOT NULL, -- Raw model output
    prediction_class VARCHAR(100), -- For classification models
    prediction_score DECIMAL(10,6), -- Confidence/probability score
    prediction_probabilities JSONB, -- Class probabilities for multi-class
    prediction_value DECIMAL(18,6), -- For regression models
    
    -- Model execution metadata
    model_version_used VARCHAR(100), -- Actual version used (for A/B testing)
    execution_time_ms INTEGER NOT NULL,
    memory_usage_mb DECIMAL(8,2),
    cpu_usage_percent DECIMAL(5,2),
    
    -- Data quality and validation
    input_validation_status VARCHAR(20) DEFAULT 'pending', -- 'passed', 'failed', 'warning'
    input_validation_errors JSONB, -- Validation error details
    data_drift_score DECIMAL(5,4), -- Input drift from training data
    prediction_confidence_level VARCHAR(20), -- 'high', 'medium', 'low'
    
    -- Business context
    business_context JSONB, -- Additional business metadata
    prediction_reason TEXT, -- Why this prediction was requested
    downstream_impact VARCHAR(100), -- How prediction will be used
    
    -- Actual outcomes (for feedback loop)
    actual_outcome JSONB, -- Ground truth when available
    actual_outcome_timestamp TIMESTAMP, -- When actual outcome was recorded
    outcome_source VARCHAR(100), -- How actual outcome was obtained
    prediction_accuracy DECIMAL(5,4), -- Accuracy of this specific prediction
    
    -- Timestamps
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    request_received_at TIMESTAMP,
    prediction_completed_at TIMESTAMP,
    
    -- Partitioning and indexing
    prediction_date DATE GENERATED ALWAYS AS (DATE(prediction_timestamp)) STORED,
    
    INDEX idx_prediction_logs_model_date (model_id, prediction_date DESC),
    INDEX idx_prediction_logs_timestamp (prediction_timestamp DESC),
    INDEX idx_prediction_logs_user (user_id, prediction_timestamp DESC),
    INDEX idx_prediction_logs_session (session_id),
    INDEX idx_prediction_logs_request (request_id),
    INDEX idx_prediction_logs_feature_hash (feature_hash),
    INDEX idx_prediction_logs_validation (input_validation_status),
    INDEX idx_prediction_logs_confidence (prediction_confidence_level)
) PARTITION BY RANGE (prediction_date);

-- Create monthly partitions for prediction logs
CREATE TABLE ml_prediction_logging.prediction_logs_2024_01 
PARTITION OF ml_prediction_logging.prediction_logs
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE ml_prediction_logging.prediction_logs_2024_02 
PARTITION OF ml_prediction_logging.prediction_logs
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Performance metrics aggregated by time periods
CREATE TABLE ml_prediction_logging.prediction_metrics_hourly (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES ml_prediction_logging.model_registry(model_id),
    metric_hour TIMESTAMP NOT NULL, -- Hour bucket for aggregation
    
    -- Volume metrics
    total_predictions INTEGER NOT NULL,
    unique_users INTEGER,
    unique_sessions INTEGER,
    
    -- Performance metrics
    avg_execution_time_ms DECIMAL(8,2),
    p50_execution_time_ms INTEGER,
    p95_execution_time_ms INTEGER,
    p99_execution_time_ms INTEGER,
    max_execution_time_ms INTEGER,
    
    -- Quality metrics
    avg_confidence_score DECIMAL(5,4),
    high_confidence_predictions INTEGER,
    low_confidence_predictions INTEGER,
    validation_failure_count INTEGER,
    validation_failure_rate DECIMAL(5,4),
    
    -- Data drift metrics
    avg_data_drift_score DECIMAL(5,4),
    max_data_drift_score DECIMAL(5,4),
    high_drift_predictions INTEGER,
    
    -- Resource utilization
    avg_memory_usage_mb DECIMAL(8,2),
    max_memory_usage_mb DECIMAL(8,2),
    avg_cpu_usage_percent DECIMAL(5,2),
    
    -- Accuracy metrics (when ground truth available)
    accuracy_sample_size INTEGER DEFAULT 0,
    accuracy_rate DECIMAL(5,4),
    false_positive_rate DECIMAL(5,4),
    false_negative_rate DECIMAL(5,4),
    
    -- Computed timestamp
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_id, metric_hour),
    INDEX idx_metrics_hourly_model_time (model_id, metric_hour DESC),
    INDEX idx_metrics_hourly_computed (computed_at DESC)
);

-- Error and exception logging
CREATE TABLE ml_prediction_logging.prediction_errors (
    error_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ml_prediction_logging.model_registry(model_id),
    request_id UUID,
    
    -- Error details
    error_type VARCHAR(100) NOT NULL, -- 'validation_error', 'model_error', 'timeout', 'resource_exhausted'
    error_code VARCHAR(50),
    error_message TEXT NOT NULL,
    error_stack_trace TEXT,
    
    -- Context
    input_features JSONB,
    model_version VARCHAR(100),
    execution_stage VARCHAR(50), -- 'preprocessing', 'inference', 'postprocessing'
    
    -- Resolution
    resolution_status VARCHAR(20) DEFAULT 'open', -- 'open', 'investigating', 'resolved'
    resolution_notes TEXT,
    resolved_at TIMESTAMP,
    resolved_by UUID,
    
    -- Timestamps
    error_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_prediction_errors_model (model_id, error_timestamp DESC),
    INDEX idx_prediction_errors_type (error_type),
    INDEX idx_prediction_errors_status (resolution_status)
);

-- A/B testing and experimental tracking
CREATE TABLE ml_prediction_logging.prediction_experiments (
    experiment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Experiment configuration
    control_model_id UUID NOT NULL REFERENCES ml_prediction_logging.model_registry(model_id),
    treatment_model_ids UUID[] NOT NULL, -- Array of treatment model IDs
    traffic_allocation JSONB NOT NULL, -- {"control": 50, "treatment_a": 25, "treatment_b": 25}
    
    -- Experiment criteria
    target_user_segments JSONB, -- Which users to include
    geographic_restrictions JSONB, -- Geographic limitations
    time_restrictions JSONB, -- Time-based restrictions
    
    -- Status and timeline
    experiment_status VARCHAR(20) DEFAULT 'draft', -- 'draft', 'active', 'paused', 'completed'
    start_date TIMESTAMP,
    planned_end_date TIMESTAMP,
    actual_end_date TIMESTAMP,
    
    -- Success criteria
    primary_metric VARCHAR(100), -- Primary metric to optimize
    success_criteria JSONB, -- Detailed success criteria
    statistical_significance_threshold DECIMAL(3,2) DEFAULT 0.05,
    minimum_sample_size INTEGER DEFAULT 1000,
    
    -- Results
    experiment_results JSONB, -- Final results when completed
    winner_model_id UUID REFERENCES ml_prediction_logging.model_registry(model_id),
    
    -- Audit
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID NOT NULL,
    
    INDEX idx_experiments_status (experiment_status),
    INDEX idx_experiments_dates (start_date, planned_end_date)
);

-- Link predictions to experiments
CREATE TABLE ml_prediction_logging.prediction_experiment_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID NOT NULL REFERENCES ml_prediction_logging.prediction_logs(prediction_id),
    experiment_id UUID NOT NULL REFERENCES ml_prediction_logging.prediction_experiments(experiment_id),
    assigned_model_id UUID NOT NULL REFERENCES ml_prediction_logging.model_registry(model_id),
    assignment_group VARCHAR(50) NOT NULL, -- 'control', 'treatment_a', 'treatment_b'
    assignment_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_experiment_assignments_experiment (experiment_id, assignment_timestamp),
    INDEX idx_experiment_assignments_prediction (prediction_id)
);
```

#### 2. Prediction Logging and Monitoring Functions

```sql
-- Comprehensive prediction logging function
CREATE OR REPLACE FUNCTION ml_prediction_logging.log_prediction(
    p_model_name VARCHAR(255),
    p_model_version VARCHAR(100),
    p_input_features JSONB,
    p_prediction_output JSONB,
    p_user_id UUID DEFAULT NULL,
    p_session_id VARCHAR(255) DEFAULT NULL,
    p_request_id UUID DEFAULT NULL,
    p_execution_time_ms INTEGER DEFAULT NULL,
    p_business_context JSONB DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_model_id UUID;
    v_prediction_id UUID;
    v_feature_hash VARCHAR(64);
    v_prediction_class VARCHAR(100);
    v_prediction_score DECIMAL(10,6);
    v_prediction_value DECIMAL(18,6);
    v_validation_status VARCHAR(20);
    v_validation_errors JSONB;
    v_confidence_level VARCHAR(20);
    v_data_drift_score DECIMAL(5,4);
BEGIN
    -- Get model ID
    SELECT model_id INTO v_model_id
    FROM ml_prediction_logging.model_registry
    WHERE model_name = p_model_name 
      AND model_version = p_model_version 
      AND deployment_status = 'active';
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Active model not found: % version %', p_model_name, p_model_version;
    END IF;
    
    -- Generate feature hash for deduplication
    v_feature_hash := encode(digest(p_input_features::text, 'sha256'), 'hex');
    
    -- Extract prediction components from output
    v_prediction_class := p_prediction_output->>'predicted_class';
    v_prediction_score := (p_prediction_output->>'confidence_score')::DECIMAL(10,6);
    v_prediction_value := (p_prediction_output->>'predicted_value')::DECIMAL(18,6);
    
    -- Validate input features (simplified validation)
    SELECT ml_prediction_logging.validate_input_features(v_model_id, p_input_features) 
    INTO v_validation_status, v_validation_errors;
    
    -- Calculate confidence level
    v_confidence_level := CASE 
        WHEN v_prediction_score >= 0.8 THEN 'high'
        WHEN v_prediction_score >= 0.6 THEN 'medium'
        ELSE 'low'
    END;
    
    -- Calculate data drift score (simplified)
    SELECT ml_prediction_logging.calculate_data_drift(v_model_id, p_input_features) 
    INTO v_data_drift_score;
    
    -- Insert prediction log
    INSERT INTO ml_prediction_logging.prediction_logs (
        model_id,
        request_id,
        session_id,
        user_id,
        input_features,
        feature_hash,
        prediction_output,
        prediction_class,
        prediction_score,
        prediction_value,
        execution_time_ms,
        input_validation_status,
        input_validation_errors,
        data_drift_score,
        prediction_confidence_level,
        business_context,
        request_received_at,
        prediction_completed_at
    ) VALUES (
        v_model_id,
        p_request_id,
        p_session_id,
        p_user_id,
        p_input_features,
        v_feature_hash,
        p_prediction_output,
        v_prediction_class,
        v_prediction_score,
        v_prediction_value,
        p_execution_time_ms,
        v_validation_status,
        v_validation_errors,
        v_data_drift_score,
        v_confidence_level,
        p_business_context,
        CURRENT_TIMESTAMP - INTERVAL '1 millisecond' * COALESCE(p_execution_time_ms, 0),
        CURRENT_TIMESTAMP
    ) RETURNING prediction_id INTO v_prediction_id;
    
    RETURN v_prediction_id;
    
EXCEPTION
    WHEN OTHERS THEN
        -- Log error
        INSERT INTO ml_prediction_logging.prediction_errors (
            model_id,
            request_id,
            error_type,
            error_message,
            input_features,
            execution_stage
        ) VALUES (
            v_model_id,
            p_request_id,
            'logging_error',
            SQLERRM,
            p_input_features,
            'logging'
        );
        
        RAISE;
END;
$$ LANGUAGE plpgsql;

-- Function to update prediction with actual outcome
CREATE OR REPLACE FUNCTION ml_prediction_logging.update_prediction_outcome(
    p_prediction_id UUID,
    p_actual_outcome JSONB,
    p_outcome_source VARCHAR(100) DEFAULT 'manual'
) RETURNS BOOLEAN AS $$
DECLARE
    v_prediction_record RECORD;
    v_accuracy DECIMAL(5,4);
BEGIN
    -- Get prediction details
    SELECT prediction_class, prediction_value, prediction_score
    INTO v_prediction_record
    FROM ml_prediction_logging.prediction_logs
    WHERE prediction_id = p_prediction_id;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Prediction not found: %', p_prediction_id;
    END IF;
    
    -- Calculate accuracy based on prediction type
    IF v_prediction_record.prediction_class IS NOT NULL THEN
        -- Classification accuracy
        v_accuracy := CASE 
            WHEN v_prediction_record.prediction_class = (p_actual_outcome->>'actual_class') THEN 1.0 
            ELSE 0.0 
        END;
    ELSIF v_prediction_record.prediction_value IS NOT NULL THEN
        -- Regression accuracy (inverse of relative error, capped at 1.0)
        v_accuracy := GREATEST(0.0, 1.0 - ABS(
            v_prediction_record.prediction_value - (p_actual_outcome->>'actual_value')::DECIMAL
        ) / NULLIF(ABS((p_actual_outcome->>'actual_value')::DECIMAL), 0));
    END IF;
    
    -- Update prediction with actual outcome
    UPDATE ml_prediction_logging.prediction_logs
    SET 
        actual_outcome = p_actual_outcome,
        actual_outcome_timestamp = CURRENT_TIMESTAMP,
        outcome_source = p_outcome_source,
        prediction_accuracy = v_accuracy
    WHERE prediction_id = p_prediction_id;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Automated metrics aggregation
CREATE OR REPLACE FUNCTION ml_prediction_logging.aggregate_hourly_metrics(
    p_target_hour TIMESTAMP DEFAULT NULL
) RETURNS INTEGER AS $$
DECLARE
    v_target_hour TIMESTAMP;
    v_aggregated_count INTEGER := 0;
    v_model_record RECORD;
BEGIN
    -- Default to previous hour if not specified
    v_target_hour := COALESCE(p_target_hour, date_trunc('hour', CURRENT_TIMESTAMP - INTERVAL '1 hour'));
    
    -- Aggregate metrics for each model
    FOR v_model_record IN 
        SELECT DISTINCT model_id 
        FROM ml_prediction_logging.prediction_logs 
        WHERE prediction_timestamp >= v_target_hour 
          AND prediction_timestamp < v_target_hour + INTERVAL '1 hour'
    LOOP
        INSERT INTO ml_prediction_logging.prediction_metrics_hourly (
            model_id,
            metric_hour,
            total_predictions,
            unique_users,
            unique_sessions,
            avg_execution_time_ms,
            p50_execution_time_ms,
            p95_execution_time_ms,
            p99_execution_time_ms,
            max_execution_time_ms,
            avg_confidence_score,
            high_confidence_predictions,
            low_confidence_predictions,
            validation_failure_count,
            validation_failure_rate,
            avg_data_drift_score,
            max_data_drift_score,
            high_drift_predictions,
            avg_memory_usage_mb,
            max_memory_usage_mb,
            avg_cpu_usage_percent,
            accuracy_sample_size,
            accuracy_rate,
            false_positive_rate,
            false_negative_rate
        )
        SELECT 
            v_model_record.model_id,
            v_target_hour,
            COUNT(*) as total_predictions,
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(DISTINCT session_id) as unique_sessions,
            AVG(execution_time_ms) as avg_execution_time_ms,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY execution_time_ms) as p50_execution_time_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time_ms,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY execution_time_ms) as p99_execution_time_ms,
            MAX(execution_time_ms) as max_execution_time_ms,
            AVG(prediction_score) as avg_confidence_score,
            COUNT(*) FILTER (WHERE prediction_confidence_level = 'high') as high_confidence_predictions,
            COUNT(*) FILTER (WHERE prediction_confidence_level = 'low') as low_confidence_predictions,
            COUNT(*) FILTER (WHERE input_validation_status = 'failed') as validation_failure_count,
            COUNT(*) FILTER (WHERE input_validation_status = 'failed')::DECIMAL / COUNT(*) as validation_failure_rate,
            AVG(data_drift_score) as avg_data_drift_score,
            MAX(data_drift_score) as max_data_drift_score,
            COUNT(*) FILTER (WHERE data_drift_score > 0.3) as high_drift_predictions,
            AVG(memory_usage_mb) as avg_memory_usage_mb,
            MAX(memory_usage_mb) as max_memory_usage_mb,
            AVG(cpu_usage_percent) as avg_cpu_usage_percent,
            COUNT(*) FILTER (WHERE actual_outcome IS NOT NULL) as accuracy_sample_size,
            AVG(prediction_accuracy) FILTER (WHERE actual_outcome IS NOT NULL) as accuracy_rate,
            COUNT(*) FILTER (WHERE prediction_class != (actual_outcome->>'actual_class') AND prediction_class IS NOT NULL)::DECIMAL / 
            NULLIF(COUNT(*) FILTER (WHERE actual_outcome IS NOT NULL AND prediction_class IS NOT NULL), 0) as false_positive_rate,
            COUNT(*) FILTER (WHERE prediction_class != (actual_outcome->>'actual_class') AND (actual_outcome->>'actual_class') IS NOT NULL)::DECIMAL / 
            NULLIF(COUNT(*) FILTER (WHERE actual_outcome IS NOT NULL AND prediction_class IS NOT NULL), 0) as false_negative_rate
        FROM ml_prediction_logging.prediction_logs
        WHERE model_id = v_model_record.model_id
          AND prediction_timestamp >= v_target_hour
          AND prediction_timestamp < v_target_hour + INTERVAL '1 hour'
        ON CONFLICT (model_id, metric_hour) DO UPDATE SET
            total_predictions = EXCLUDED.total_predictions,
            unique_users = EXCLUDED.unique_users,
            unique_sessions = EXCLUDED.unique_sessions,
            avg_execution_time_ms = EXCLUDED.avg_execution_time_ms,
            p50_execution_time_ms = EXCLUDED.p50_execution_time_ms,
            p95_execution_time_ms = EXCLUDED.p95_execution_time_ms,
            p99_execution_time_ms = EXCLUDED.p99_execution_time_ms,
            max_execution_time_ms = EXCLUDED.max_execution_time_ms,
            avg_confidence_score = EXCLUDED.avg_confidence_score,
            high_confidence_predictions = EXCLUDED.high_confidence_predictions,
            low_confidence_predictions = EXCLUDED.low_confidence_predictions,
            validation_failure_count = EXCLUDED.validation_failure_count,
            validation_failure_rate = EXCLUDED.validation_failure_rate,
            avg_data_drift_score = EXCLUDED.avg_data_drift_score,
            max_data_drift_score = EXCLUDED.max_data_drift_score,
            high_drift_predictions = EXCLUDED.high_drift_predictions,
            avg_memory_usage_mb = EXCLUDED.avg_memory_usage_mb,
            max_memory_usage_mb = EXCLUDED.max_memory_usage_mb,
            avg_cpu_usage_percent = EXCLUDED.avg_cpu_usage_percent,
            accuracy_sample_size = EXCLUDED.accuracy_sample_size,
            accuracy_rate = EXCLUDED.accuracy_rate,
            false_positive_rate = EXCLUDED.false_positive_rate,
            false_negative_rate = EXCLUDED.false_negative_rate,
            computed_at = CURRENT_TIMESTAMP;
        
        v_aggregated_count := v_aggregated_count + 1;
    END LOOP;
    
    RETURN v_aggregated_count;
END;
$$ LANGUAGE plpgsql;
```

#### 3. Advanced Analytics and Monitoring Queries

```sql
-- Real-time model performance dashboard
CREATE OR REPLACE VIEW ml_prediction_logging.model_performance_dashboard AS
WITH recent_performance AS (
    SELECT 
        mr.model_name,
        mr.model_version,
        mr.model_type,
        mr.deployment_environment,
        
        -- Current hour metrics
        COUNT(*) as predictions_last_hour,
        AVG(pl.execution_time_ms) as avg_latency_ms,
        COUNT(DISTINCT pl.user_id) as active_users,
        AVG(pl.prediction_score) as avg_confidence,
        
        -- Quality indicators
        COUNT(*) FILTER (WHERE pl.input_validation_status = 'failed') as validation_failures,
        COUNT(*) FILTER (WHERE pl.prediction_confidence_level = 'low') as low_confidence_predictions,
        AVG(pl.data_drift_score) as avg_drift_score,
        
        -- Error metrics
        COUNT(pe.error_id) as error_count_last_hour
        
    FROM ml_prediction_logging.model_registry mr
    LEFT JOIN ml_prediction_logging.prediction_logs pl 
        ON mr.model_id = pl.model_id 
        AND pl.prediction_timestamp >= date_trunc('hour', CURRENT_TIMESTAMP)
    LEFT JOIN ml_prediction_logging.prediction_errors pe 
        ON mr.model_id = pe.model_id 
        AND pe.error_timestamp >= date_trunc('hour', CURRENT_TIMESTAMP)
    WHERE mr.deployment_status = 'active'
    GROUP BY mr.model_id, mr.model_name, mr.model_version, mr.model_type, mr.deployment_environment
),
daily_trends AS (
    SELECT 
        pmh.model_id,
        AVG(pmh.total_predictions) as avg_daily_predictions,
        AVG(pmh.avg_execution_time_ms) as avg_daily_latency,
        AVG(pmh.accuracy_rate) as avg_daily_accuracy,
        STDDEV(pmh.avg_execution_time_ms) as latency_volatility
    FROM ml_prediction_logging.prediction_metrics_hourly pmh
    WHERE pmh.metric_hour >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    GROUP BY pmh.model_id
),
health_assessment AS (
    SELECT 
        rp.*,
        dt.avg_daily_predictions,
        dt.avg_daily_latency,
        dt.avg_daily_accuracy,
        dt.latency_volatility,
        
        -- Health scores (0-100)
        CASE 
            WHEN rp.error_count_last_hour = 0 AND rp.validation_failures = 0 THEN 100
            WHEN rp.error_count_last_hour <= 5 AND rp.validation_failures <= 10 THEN 80
            WHEN rp.error_count_last_hour <= 20 AND rp.validation_failures <= 50 THEN 60
            ELSE 40
        END as reliability_score,
        
        CASE 
            WHEN rp.avg_latency_ms <= 100 THEN 100
            WHEN rp.avg_latency_ms <= 500 THEN 80
            WHEN rp.avg_latency_ms <= 1000 THEN 60
            ELSE 40
        END as performance_score,
        
        CASE 
            WHEN rp.avg_confidence >= 0.8 AND rp.low_confidence_predictions <= rp.predictions_last_hour * 0.1 THEN 100
            WHEN rp.avg_confidence >= 0.6 AND rp.low_confidence_predictions <= rp.predictions_last_hour * 0.2 THEN 80
            ELSE 60
        END as confidence_score,
        
        CASE 
            WHEN rp.avg_drift_score <= 0.1 THEN 100
            WHEN rp.avg_drift_score <= 0.3 THEN 80
            WHEN rp.avg_drift_score <= 0.5 THEN 60
            ELSE 40
        END as stability_score
        
    FROM recent_performance rp
    LEFT JOIN daily_trends dt ON rp.model_name = dt.model_id::text -- Simplified join
)
SELECT 
    model_name,
    model_version,
    model_type,
    deployment_environment,
    
    -- Volume metrics
    predictions_last_hour,
    active_users,
    COALESCE(avg_daily_predictions, 0) as avg_daily_volume,
    
    -- Performance metrics
    ROUND(avg_latency_ms::NUMERIC, 2) as avg_latency_ms,
    ROUND(COALESCE(avg_daily_latency, 0)::NUMERIC, 2) as baseline_latency_ms,
    ROUND(avg_confidence::NUMERIC, 4) as avg_confidence_score,
    
    -- Quality metrics
    validation_failures,
    low_confidence_predictions,
    error_count_last_hour,
    ROUND(avg_drift_score::NUMERIC, 4) as data_drift_score,
    ROUND(COALESCE(avg_daily_accuracy, 0)::NUMERIC, 4) as baseline_accuracy,
    
    -- Health assessment
    reliability_score,
    performance_score,
    confidence_score,
    stability_score,
    ROUND((reliability_score + performance_score + confidence_score + stability_score) / 4.0, 1) as overall_health_score,
    
    -- Status indicators
    CASE 
        WHEN (reliability_score + performance_score + confidence_score + stability_score) / 4.0 >= 90 THEN 'EXCELLENT'
        WHEN (reliability_score + performance_score + confidence_score + stability_score) / 4.0 >= 80 THEN 'GOOD'
        WHEN (reliability_score + performance_score + confidence_score + stability_score) / 4.0 >= 70 THEN 'FAIR'
        WHEN (reliability_score + performance_score + confidence_score + stability_score) / 4.0 >= 60 THEN 'POOR'
        ELSE 'CRITICAL'
    END as health_status,
    
    -- Recommendations
    CASE 
        WHEN error_count_last_hour > 50 THEN 'INVESTIGATE_ERRORS'
        WHEN avg_latency_ms > 1000 THEN 'OPTIMIZE_PERFORMANCE'
        WHEN avg_drift_score > 0.5 THEN 'RETRAIN_MODEL'
        WHEN low_confidence_predictions > predictions_last_hour * 0.3 THEN 'REVIEW_CONFIDENCE_THRESHOLDS'
        ELSE 'MONITOR_NORMALLY'
    END as recommendation
    
FROM health_assessment
ORDER BY overall_health_score DESC, predictions_last_hour DESC;

-- Prediction trend analysis
CREATE OR REPLACE FUNCTION ml_prediction_logging.analyze_prediction_trends(
    p_model_name VARCHAR(255),
    p_days_back INTEGER DEFAULT 30
) RETURNS TABLE (
    analysis_period VARCHAR(20),
    prediction_volume INTEGER,
    avg_confidence DECIMAL(5,4),
    accuracy_rate DECIMAL(5,4),
    error_rate DECIMAL(5,4),
    trend_direction VARCHAR(20)
) AS $$
BEGIN
    RETURN QUERY
    WITH daily_metrics AS (
        SELECT 
            DATE(pl.prediction_timestamp) as metric_date,
            COUNT(*) as daily_predictions,
            AVG(pl.prediction_score) as daily_avg_confidence,
            AVG(pl.prediction_accuracy) FILTER (WHERE pl.actual_outcome IS NOT NULL) as daily_accuracy,
            COUNT(*) FILTER (WHERE pe.error_id IS NOT NULL)::DECIMAL / COUNT(*) as daily_error_rate
        FROM ml_prediction_logging.prediction_logs pl
        JOIN ml_prediction_logging.model_registry mr ON pl.model_id = mr.model_id
        LEFT JOIN ml_prediction_logging.prediction_errors pe ON pl.prediction_id::text = pe.request_id
        WHERE mr.model_name = p_model_name
          AND pl.prediction_timestamp >= CURRENT_DATE - INTERVAL '1 day' * p_days_back
        GROUP BY DATE(pl.prediction_timestamp)
        ORDER BY DATE(pl.prediction_timestamp)
    ),
    trend_analysis AS (
        SELECT 
            'LAST_7_DAYS' as period,
            AVG(daily_predictions)::INTEGER as avg_volume,
            AVG(daily_avg_confidence) as avg_conf,
            AVG(daily_accuracy) as avg_acc,
            AVG(daily_error_rate) as avg_err,
            -- Simplified trend calculation
            CASE 
                WHEN AVG(daily_predictions) FILTER (WHERE metric_date >= CURRENT_DATE - 7) > 
                     AVG(daily_predictions) FILTER (WHERE metric_date < CURRENT_DATE - 7) THEN 'INCREASING'
                ELSE 'DECREASING'
            END as trend
        FROM daily_metrics
        WHERE metric_date >= CURRENT_DATE - 14
        
        UNION ALL
        
        SELECT 
            'LAST_30_DAYS' as period,
            AVG(daily_predictions)::INTEGER as avg_volume,
            AVG(daily_avg_confidence) as avg_conf,
            AVG(daily_accuracy) as avg_acc,
            AVG(daily_error_rate) as avg_err,
            CASE 
                WHEN AVG(daily_predictions) FILTER (WHERE metric_date >= CURRENT_DATE - 15) > 
                     AVG(daily_predictions) FILTER (WHERE metric_date < CURRENT_DATE - 15) THEN 'INCREASING'
                ELSE 'DECREASING'
            END as trend
        FROM daily_metrics
    )
    SELECT 
        ta.period,
        ta.avg_volume,
        ta.avg_conf,
        ta.avg_acc,
        ta.avg_err,
        ta.trend
    FROM trend_analysis ta;
END;
$$ LANGUAGE plpgsql;
```

### Best Practices for ML Prediction Logging

#### Key Implementation Strategies:

1. **Comprehensive Logging**: Capture input features, outputs, metadata, and context
2. **Performance Monitoring**: Track execution time, resource usage, and throughput
3. **Quality Assurance**: Monitor data drift, validation failures, and confidence levels
4. **Real-time Analytics**: Implement efficient aggregation and monitoring systems
5. **A/B Testing Support**: Enable controlled experimentation and comparison
6. **Audit Trail**: Maintain complete lineage for compliance and debugging
7. **Automated Alerting**: Set up intelligent monitoring and alert systems

#### Storage and Performance Optimization:

1. **Partitioning**: Partition by date for efficient time-based queries
2. **Indexing**: Create strategic indexes for common query patterns
3. **Compression**: Use compression for historical prediction data
4. **Archival**: Implement automated data lifecycle management
5. **Real-time Processing**: Use streaming for immediate metric updates

This comprehensive system provides robust infrastructure for tracking, monitoring, and analyzing machine learning predictions in production environments, enabling data-driven model optimization and reliable ML operations.

---

## Question 8

**Discuss how to manage the entire lifecycle of a Machine Learning model using SQL tools.**

**Answer:**

### Comprehensive ML Model Lifecycle Management with SQL

Managing the complete machine learning model lifecycle using SQL requires a sophisticated framework that encompasses data preparation, model training coordination, deployment management, monitoring, and maintenance operations.

#### 1. Model Lifecycle Management Schema

```sql
-- Core ML lifecycle management infrastructure
CREATE SCHEMA ml_lifecycle;

-- Model projects and experiments
CREATE TABLE ml_lifecycle.model_projects (
    project_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    business_objective TEXT NOT NULL,
    success_criteria JSONB NOT NULL,
    
    -- Project categorization
    domain VARCHAR(100) NOT NULL, -- 'fraud_detection', 'recommendation', 'forecasting'
    project_type VARCHAR(50) NOT NULL, -- 'classification', 'regression', 'clustering'
    priority_level VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    
    -- Timeline and ownership
    project_owner UUID NOT NULL,
    data_scientist_assigned UUID,
    ml_engineer_assigned UUID,
    stakeholder_list JSONB, -- Array of stakeholder user IDs
    
    -- Status tracking
    project_status VARCHAR(30) DEFAULT 'planning', -- 'planning', 'development', 'testing', 'deployed', 'retired'
    start_date DATE,
    target_completion_date DATE,
    actual_completion_date DATE,
    
    -- Budget and resources
    allocated_budget DECIMAL(12,2),
    spent_budget DECIMAL(12,2) DEFAULT 0,
    compute_resource_allocation JSONB,
    
    -- Governance and compliance
    compliance_requirements JSONB, -- Regulatory requirements
    data_privacy_level VARCHAR(20) DEFAULT 'internal', -- 'public', 'internal', 'confidential', 'restricted'
    approval_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
    approved_by UUID,
    approval_date TIMESTAMP,
    
    -- Audit trail
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_projects_status (project_status),
    INDEX idx_projects_owner (project_owner),
    INDEX idx_projects_domain (domain, project_type)
);

-- Model development experiments
CREATE TABLE ml_lifecycle.model_experiments (
    experiment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES ml_lifecycle.model_projects(project_id),
    experiment_name VARCHAR(255) NOT NULL,
    experiment_description TEXT,
    
    -- Experiment configuration
    algorithm_name VARCHAR(100) NOT NULL,
    algorithm_version VARCHAR(50),
    hyperparameters JSONB NOT NULL,
    feature_selection_strategy JSONB,
    
    -- Data configuration
    training_dataset_id UUID NOT NULL,
    validation_dataset_id UUID,
    test_dataset_id UUID,
    data_preprocessing_steps JSONB,
    feature_engineering_pipeline JSONB,
    
    -- Training configuration
    training_start_time TIMESTAMP,
    training_end_time TIMESTAMP,
    training_duration_minutes INTEGER,
    compute_resources_used JSONB, -- CPU, memory, GPU usage
    training_cost DECIMAL(10,2),
    
    -- Model artifacts
    model_artifact_path VARCHAR(500),
    model_size_mb DECIMAL(8,2),
    model_format VARCHAR(50), -- 'pickle', 'pmml', 'onnx', 'h5'
    model_metadata JSONB,
    
    -- Performance metrics
    training_metrics JSONB, -- Training performance
    validation_metrics JSONB, -- Validation performance
    test_metrics JSONB, -- Final test performance
    cross_validation_scores JSONB,
    feature_importance JSONB,
    
    -- Experiment status
    experiment_status VARCHAR(20) DEFAULT 'running', -- 'queued', 'running', 'completed', 'failed', 'cancelled'
    error_message TEXT,
    experiment_notes TEXT,
    
    -- Reproducibility
    random_seed INTEGER,
    environment_snapshot JSONB, -- Package versions, system info
    code_version_hash VARCHAR(64),
    
    -- Evaluation and selection
    is_baseline BOOLEAN DEFAULT FALSE,
    is_selected_for_deployment BOOLEAN DEFAULT FALSE,
    selection_rationale TEXT,
    peer_review_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
    reviewed_by UUID,
    review_date TIMESTAMP,
    
    -- Audit
    created_by UUID NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_experiments_project (project_id, created_at DESC),
    INDEX idx_experiments_status (experiment_status),
    INDEX idx_experiments_algorithm (algorithm_name, algorithm_version),
    INDEX idx_experiments_selection (is_selected_for_deployment, project_id)
);

-- Model versions and deployment history
CREATE TABLE ml_lifecycle.model_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID NOT NULL REFERENCES ml_lifecycle.model_experiments(experiment_id),
    project_id UUID NOT NULL REFERENCES ml_lifecycle.model_projects(project_id),
    
    -- Version identification
    version_number VARCHAR(50) NOT NULL, -- Semantic versioning: major.minor.patch
    version_name VARCHAR(255),
    version_description TEXT,
    
    -- Model specifications
    model_type VARCHAR(50) NOT NULL,
    input_schema JSONB NOT NULL, -- Expected input format
    output_schema JSONB NOT NULL, -- Expected output format
    api_specification JSONB, -- REST API contract
    
    -- Performance characteristics
    expected_latency_ms INTEGER,
    expected_throughput_rps INTEGER,
    expected_accuracy DECIMAL(5,4),
    expected_memory_usage_mb INTEGER,
    performance_benchmarks JSONB,
    
    -- Deployment configuration
    deployment_environment VARCHAR(30), -- 'development', 'staging', 'production'
    deployment_strategy VARCHAR(30), -- 'blue_green', 'canary', 'rolling', 'immediate'
    resource_requirements JSONB, -- CPU, memory, storage requirements
    scaling_configuration JSONB, -- Auto-scaling rules
    
    -- Quality gates
    quality_gate_results JSONB, -- Automated testing results
    security_scan_results JSONB, -- Security vulnerability scan
    performance_test_results JSONB, -- Load testing results
    
    -- Deployment status
    deployment_status VARCHAR(30) DEFAULT 'created', -- 'created', 'tested', 'deployed', 'retired'
    deployment_date TIMESTAMP,
    retirement_date TIMESTAMP,
    traffic_percentage INTEGER DEFAULT 0, -- Percentage of traffic served
    
    -- Monitoring and alerting
    monitoring_configuration JSONB, -- Monitoring rules and thresholds
    alert_configuration JSONB, -- Alert rules and recipients
    sla_requirements JSONB, -- Service level agreements
    
    -- Rollback capability
    rollback_version_id UUID REFERENCES ml_lifecycle.model_versions(version_id),
    can_rollback BOOLEAN DEFAULT TRUE,
    rollback_reason TEXT,
    
    -- Approval workflow
    deployment_approval_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
    approved_by UUID,
    approval_date TIMESTAMP,
    deployment_checklist JSONB, -- Pre-deployment checklist items
    
    -- Audit
    created_by UUID NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(project_id, version_number),
    INDEX idx_versions_project (project_id, version_number),
    INDEX idx_versions_deployment (deployment_status, deployment_environment),
    INDEX idx_versions_traffic (traffic_percentage DESC, deployment_date DESC)
);

-- Model performance monitoring
CREATE TABLE ml_lifecycle.model_performance_monitoring (
    monitoring_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES ml_lifecycle.model_versions(version_id),
    
    -- Time period
    monitoring_period_start TIMESTAMP NOT NULL,
    monitoring_period_end TIMESTAMP NOT NULL,
    aggregation_level VARCHAR(20) NOT NULL, -- 'hourly', 'daily', 'weekly'
    
    -- Volume metrics
    total_predictions INTEGER NOT NULL,
    successful_predictions INTEGER NOT NULL,
    failed_predictions INTEGER NOT NULL,
    unique_users INTEGER,
    
    -- Performance metrics
    avg_latency_ms DECIMAL(8,2),
    p50_latency_ms INTEGER,
    p95_latency_ms INTEGER,
    p99_latency_ms INTEGER,
    error_rate DECIMAL(5,4),
    timeout_rate DECIMAL(5,4),
    
    -- Model quality metrics
    accuracy_rate DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_score DECIMAL(5,4),
    
    -- Data quality metrics
    input_validation_failure_rate DECIMAL(5,4),
    missing_feature_rate DECIMAL(5,4),
    data_drift_score DECIMAL(5,4),
    concept_drift_score DECIMAL(5,4),
    
    -- Resource utilization
    avg_cpu_usage_percent DECIMAL(5,2),
    avg_memory_usage_mb DECIMAL(8,2),
    avg_disk_io_mb DECIMAL(8,2),
    cost_per_prediction DECIMAL(8,4),
    
    -- Business metrics
    business_impact_metrics JSONB, -- Custom business KPIs
    revenue_impact DECIMAL(12,2),
    cost_savings DECIMAL(12,2),
    
    -- Alert status
    alert_count INTEGER DEFAULT 0,
    critical_alert_count INTEGER DEFAULT 0,
    alert_details JSONB,
    
    -- Computed metrics
    health_score DECIMAL(5,2), -- Overall health score 0-100
    performance_trend VARCHAR(20), -- 'improving', 'stable', 'degrading'
    
    INDEX idx_monitoring_version_period (version_id, monitoring_period_start DESC),
    INDEX idx_monitoring_health (health_score DESC, monitoring_period_start DESC),
    INDEX idx_monitoring_alerts (critical_alert_count DESC, alert_count DESC)
);

-- Model incident and issue tracking
CREATE TABLE ml_lifecycle.model_incidents (
    incident_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID REFERENCES ml_lifecycle.model_versions(version_id),
    project_id UUID REFERENCES ml_lifecycle.model_projects(project_id),
    
    -- Incident classification
    incident_type VARCHAR(50) NOT NULL, -- 'performance_degradation', 'accuracy_drop', 'system_failure', 'security_breach'
    severity_level VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    priority_level VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'urgent'
    
    -- Incident details
    incident_title VARCHAR(255) NOT NULL,
    incident_description TEXT NOT NULL,
    impact_description TEXT,
    root_cause TEXT,
    
    -- Detection and timing
    detected_at TIMESTAMP NOT NULL,
    detection_method VARCHAR(50), -- 'automated_monitoring', 'user_report', 'manual_check'
    incident_start_time TIMESTAMP,
    incident_end_time TIMESTAMP,
    
    -- Resolution tracking
    incident_status VARCHAR(30) DEFAULT 'open', -- 'open', 'investigating', 'resolved', 'closed'
    assigned_to UUID,
    resolution_summary TEXT,
    resolution_steps JSONB, -- Detailed resolution steps taken
    
    -- Impact assessment
    affected_users_count INTEGER,
    business_impact_level VARCHAR(20), -- 'none', 'low', 'medium', 'high', 'severe'
    financial_impact DECIMAL(12,2),
    reputation_impact VARCHAR(20),
    
    -- Communication
    stakeholders_notified JSONB, -- List of notified stakeholders
    communication_log JSONB, -- Communication timeline
    
    -- Prevention measures
    prevention_measures JSONB, -- Steps taken to prevent recurrence
    monitoring_improvements JSONB, -- Monitoring enhancements made
    
    -- Post-incident review
    post_incident_review_completed BOOLEAN DEFAULT FALSE,
    lessons_learned TEXT,
    action_items JSONB,
    
    -- Audit
    reported_by UUID NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_incidents_version (version_id, detected_at DESC),
    INDEX idx_incidents_status (incident_status, severity_level),
    INDEX idx_incidents_detection (detected_at DESC, detection_method)
);

-- Model retraining and updates
CREATE TABLE ml_lifecycle.model_retraining_jobs (
    retraining_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES ml_lifecycle.model_projects(project_id),
    current_version_id UUID REFERENCES ml_lifecycle.model_versions(version_id),
    
    -- Retraining trigger
    trigger_type VARCHAR(50) NOT NULL, -- 'scheduled', 'performance_degradation', 'data_drift', 'manual'
    trigger_details JSONB,
    trigger_threshold_exceeded JSONB, -- Which thresholds were exceeded
    
    -- Retraining configuration
    retraining_strategy VARCHAR(50), -- 'full_retrain', 'incremental', 'transfer_learning'
    new_training_data_id UUID,
    additional_data_sources JSONB,
    hyperparameter_tuning_enabled BOOLEAN DEFAULT TRUE,
    
    -- Execution status
    job_status VARCHAR(30) DEFAULT 'queued', -- 'queued', 'running', 'completed', 'failed', 'cancelled'
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    execution_duration_minutes INTEGER,
    
    -- Resource usage
    compute_resources_allocated JSONB,
    compute_resources_used JSONB,
    estimated_cost DECIMAL(10,2),
    actual_cost DECIMAL(10,2),
    
    -- Results
    new_model_experiment_id UUID REFERENCES ml_lifecycle.model_experiments(experiment_id),
    performance_improvement JSONB, -- Comparison with current model
    recommendation VARCHAR(50), -- 'deploy', 'further_testing', 'reject'
    recommendation_confidence DECIMAL(3,2),
    
    -- Approval and deployment
    auto_deployment_enabled BOOLEAN DEFAULT FALSE,
    deployment_approval_required BOOLEAN DEFAULT TRUE,
    approved_for_deployment BOOLEAN DEFAULT FALSE,
    approval_notes TEXT,
    
    -- Scheduling
    scheduled_execution_time TIMESTAMP,
    recurring_schedule VARCHAR(100), -- Cron expression for recurring jobs
    next_scheduled_execution TIMESTAMP,
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Audit
    initiated_by UUID NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_retraining_project (project_id, created_at DESC),
    INDEX idx_retraining_status (job_status, start_time),
    INDEX idx_retraining_schedule (next_scheduled_execution, recurring_schedule)
);
```

#### 2. Lifecycle Management Functions

```sql
-- Model project creation and management
CREATE OR REPLACE FUNCTION ml_lifecycle.create_model_project(
    p_project_name VARCHAR(255),
    p_description TEXT,
    p_business_objective TEXT,
    p_success_criteria JSONB,
    p_domain VARCHAR(100),
    p_project_type VARCHAR(50),
    p_project_owner UUID,
    p_priority_level VARCHAR(20) DEFAULT 'medium',
    p_target_completion_date DATE DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_project_id UUID;
BEGIN
    INSERT INTO ml_lifecycle.model_projects (
        project_name,
        description,
        business_objective,
        success_criteria,
        domain,
        project_type,
        project_owner,
        priority_level,
        target_completion_date,
        start_date
    ) VALUES (
        p_project_name,
        p_description,
        p_business_objective,
        p_success_criteria,
        p_domain,
        p_project_type,
        p_project_owner,
        p_priority_level,
        p_target_completion_date,
        CURRENT_DATE
    ) RETURNING project_id INTO v_project_id;
    
    -- Create initial experiment tracking
    INSERT INTO ml_lifecycle.model_experiments (
        project_id,
        experiment_name,
        experiment_description,
        algorithm_name,
        hyperparameters,
        training_dataset_id,
        experiment_status,
        created_by
    ) VALUES (
        v_project_id,
        'Baseline Experiment',
        'Initial baseline model experiment',
        'baseline',
        '{}',
        gen_random_uuid(), -- Placeholder dataset ID
        'queued',
        p_project_owner
    );
    
    RETURN v_project_id;
END;
$$ LANGUAGE plpgsql;

-- Automated model deployment function
CREATE OR REPLACE FUNCTION ml_lifecycle.deploy_model_version(
    p_version_id UUID,
    p_deployment_environment VARCHAR(30),
    p_traffic_percentage INTEGER DEFAULT 100,
    p_deployment_strategy VARCHAR(30) DEFAULT 'blue_green',
    p_deployed_by UUID
) RETURNS BOOLEAN AS $$
DECLARE
    v_project_id UUID;
    v_current_version_id UUID;
    v_quality_gates_passed BOOLEAN;
    v_approval_status VARCHAR(20);
BEGIN
    -- Validate version and get project info
    SELECT project_id, deployment_approval_status
    INTO v_project_id, v_approval_status
    FROM ml_lifecycle.model_versions
    WHERE version_id = p_version_id;
    
    IF v_project_id IS NULL THEN
        RAISE EXCEPTION 'Model version not found: %', p_version_id;
    END IF;
    
    IF v_approval_status != 'approved' THEN
        RAISE EXCEPTION 'Model version not approved for deployment: %', p_version_id;
    END IF;
    
    -- Check quality gates
    SELECT ml_lifecycle.check_quality_gates(p_version_id) INTO v_quality_gates_passed;
    
    IF NOT v_quality_gates_passed THEN
        RAISE EXCEPTION 'Quality gates not passed for version: %', p_version_id;
    END IF;
    
    -- Handle deployment strategy
    IF p_deployment_strategy = 'blue_green' THEN
        -- Get current production version
        SELECT version_id INTO v_current_version_id
        FROM ml_lifecycle.model_versions
        WHERE project_id = v_project_id
          AND deployment_environment = p_deployment_environment
          AND deployment_status = 'deployed'
          AND traffic_percentage > 0;
        
        -- Update current version to reduce traffic
        IF v_current_version_id IS NOT NULL THEN
            UPDATE ml_lifecycle.model_versions
            SET traffic_percentage = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE version_id = v_current_version_id;
        END IF;
    ELSIF p_deployment_strategy = 'canary' THEN
        -- For canary deployment, start with lower traffic
        p_traffic_percentage := LEAST(p_traffic_percentage, 10);
    END IF;
    
    -- Update version deployment status
    UPDATE ml_lifecycle.model_versions
    SET deployment_status = 'deployed',
        deployment_environment = p_deployment_environment,
        deployment_date = CURRENT_TIMESTAMP,
        traffic_percentage = p_traffic_percentage,
        updated_at = CURRENT_TIMESTAMP
    WHERE version_id = p_version_id;
    
    -- Log deployment event
    INSERT INTO ml_lifecycle.model_incidents (
        version_id,
        project_id,
        incident_type,
        severity_level,
        priority_level,
        incident_title,
        incident_description,
        detected_at,
        incident_status,
        reported_by
    ) VALUES (
        p_version_id,
        v_project_id,
        'deployment',
        'low',
        'medium',
        'Model Version Deployed',
        format('Model version %s deployed to %s with %s%% traffic using %s strategy',
               p_version_id, p_deployment_environment, p_traffic_percentage, p_deployment_strategy),
        CURRENT_TIMESTAMP,
        'resolved',
        p_deployed_by
    );
    
    RETURN TRUE;
    
EXCEPTION
    WHEN OTHERS THEN
        -- Log deployment failure
        INSERT INTO ml_lifecycle.model_incidents (
            version_id,
            project_id,
            incident_type,
            severity_level,
            priority_level,
            incident_title,
            incident_description,
            detected_at,
            incident_status,
            reported_by
        ) VALUES (
            p_version_id,
            v_project_id,
            'deployment_failure',
            'high',
            'high',
            'Model Deployment Failed',
            'Deployment failed: ' || SQLERRM,
            CURRENT_TIMESTAMP,
            'open',
            p_deployed_by
        );
        
        RAISE;
END;
$$ LANGUAGE plpgsql;

-- Automated model monitoring and alerting
CREATE OR REPLACE FUNCTION ml_lifecycle.evaluate_model_health(
    p_version_id UUID,
    p_monitoring_period_hours INTEGER DEFAULT 1
) RETURNS JSONB AS $$
DECLARE
    v_health_report JSONB;
    v_performance_metrics RECORD;
    v_alert_count INTEGER;
    v_health_score DECIMAL(5,2);
    v_recommendations JSONB;
BEGIN
    -- Calculate performance metrics for the period
    SELECT 
        COUNT(*) as total_predictions,
        AVG(execution_time_ms) as avg_latency,
        COUNT(*) FILTER (WHERE prediction_accuracy >= 0.8) * 100.0 / COUNT(*) as accuracy_rate,
        COUNT(*) FILTER (WHERE input_validation_status = 'failed') * 100.0 / COUNT(*) as error_rate,
        AVG(data_drift_score) as avg_drift_score
    INTO v_performance_metrics
    FROM ml_prediction_logging.prediction_logs pl
    JOIN ml_lifecycle.model_versions mv ON pl.model_id::text = mv.version_id::text
    WHERE mv.version_id = p_version_id
      AND pl.prediction_timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 hour' * p_monitoring_period_hours;
    
    -- Count active alerts
    SELECT COUNT(*) INTO v_alert_count
    FROM ml_lifecycle.model_incidents
    WHERE version_id = p_version_id
      AND incident_status IN ('open', 'investigating')
      AND severity_level IN ('high', 'critical');
    
    -- Calculate health score (0-100)
    v_health_score := LEAST(100, GREATEST(0,
        CASE 
            WHEN v_performance_metrics.total_predictions = 0 THEN 50
            ELSE (
                -- Latency component (max 25 points)
                LEAST(25, 25 * (1000 - COALESCE(v_performance_metrics.avg_latency, 1000)) / 1000) +
                -- Accuracy component (max 30 points)
                LEAST(30, 30 * COALESCE(v_performance_metrics.accuracy_rate, 0) / 100) +
                -- Error rate component (max 25 points)
                LEAST(25, 25 * (1 - COALESCE(v_performance_metrics.error_rate, 100) / 100)) +
                -- Data drift component (max 20 points)
                LEAST(20, 20 * (1 - COALESCE(v_performance_metrics.avg_drift_score, 1)))
            )
        END
    ));
    
    -- Reduce score based on active alerts
    v_health_score := v_health_score - (v_alert_count * 10);
    v_health_score := GREATEST(0, v_health_score);
    
    -- Generate recommendations
    v_recommendations := jsonb_build_array();
    
    IF v_performance_metrics.avg_latency > 1000 THEN
        v_recommendations := v_recommendations || jsonb_build_object(
            'type', 'performance',
            'priority', 'high',
            'message', 'Average latency exceeds 1000ms - consider model optimization'
        );
    END IF;
    
    IF v_performance_metrics.accuracy_rate < 80 THEN
        v_recommendations := v_recommendations || jsonb_build_object(
            'type', 'accuracy',
            'priority', 'critical',
            'message', 'Accuracy below 80% - model retraining may be required'
        );
    END IF;
    
    IF v_performance_metrics.avg_drift_score > 0.3 THEN
        v_recommendations := v_recommendations || jsonb_build_object(
            'type', 'data_drift',
            'priority', 'medium',
            'message', 'Data drift detected - monitor and consider retraining'
        );
    END IF;
    
    -- Build health report
    v_health_report := jsonb_build_object(
        'version_id', p_version_id,
        'evaluation_timestamp', CURRENT_TIMESTAMP,
        'monitoring_period_hours', p_monitoring_period_hours,
        'health_score', v_health_score,
        'health_status', CASE 
            WHEN v_health_score >= 90 THEN 'excellent'
            WHEN v_health_score >= 80 THEN 'good'
            WHEN v_health_score >= 70 THEN 'fair'
            WHEN v_health_score >= 60 THEN 'poor'
            ELSE 'critical'
        END,
        'performance_metrics', jsonb_build_object(
            'total_predictions', v_performance_metrics.total_predictions,
            'avg_latency_ms', v_performance_metrics.avg_latency,
            'accuracy_rate', v_performance_metrics.accuracy_rate,
            'error_rate', v_performance_metrics.error_rate,
            'avg_drift_score', v_performance_metrics.avg_drift_score
        ),
        'active_alerts', v_alert_count,
        'recommendations', v_recommendations
    );
    
    -- Store monitoring record
    INSERT INTO ml_lifecycle.model_performance_monitoring (
        version_id,
        monitoring_period_start,
        monitoring_period_end,
        aggregation_level,
        total_predictions,
        avg_latency_ms,
        accuracy_rate,
        error_rate,
        data_drift_score,
        health_score,
        alert_count,
        alert_details
    ) VALUES (
        p_version_id,
        CURRENT_TIMESTAMP - INTERVAL '1 hour' * p_monitoring_period_hours,
        CURRENT_TIMESTAMP,
        'hourly',
        v_performance_metrics.total_predictions,
        v_performance_metrics.avg_latency,
        v_performance_metrics.accuracy_rate / 100,
        v_performance_metrics.error_rate / 100,
        v_performance_metrics.avg_drift_score,
        v_health_score,
        v_alert_count,
        v_recommendations
    );
    
    RETURN v_health_report;
END;
$$ LANGUAGE plpgsql;

-- Automated retraining trigger system
CREATE OR REPLACE FUNCTION ml_lifecycle.evaluate_retraining_triggers(
    p_project_id UUID DEFAULT NULL
) RETURNS TABLE (
    project_id UUID,
    version_id UUID,
    trigger_type VARCHAR(50),
    trigger_details JSONB,
    recommendation VARCHAR(50)
) AS $$
DECLARE
    v_project RECORD;
    v_version RECORD;
    v_recent_performance RECORD;
    v_triggers JSONB;
BEGIN
    -- Loop through projects (all or specific)
    FOR v_project IN 
        SELECT mp.project_id, mp.project_name, mp.project_status
        FROM ml_lifecycle.model_projects mp
        WHERE (p_project_id IS NULL OR mp.project_id = p_project_id)
          AND mp.project_status = 'deployed'
    LOOP
        -- Get current deployed version
        SELECT mv.version_id, mv.version_number, mv.deployment_date
        INTO v_version
        FROM ml_lifecycle.model_versions mv
        WHERE mv.project_id = v_project.project_id
          AND mv.deployment_status = 'deployed'
          AND mv.traffic_percentage > 0
        ORDER BY mv.deployment_date DESC
        LIMIT 1;
        
        IF v_version.version_id IS NOT NULL THEN
            -- Check performance degradation
            SELECT 
                AVG(mpm.accuracy_rate) as avg_accuracy,
                AVG(mpm.data_drift_score) as avg_drift,
                AVG(mpm.health_score) as avg_health,
                COUNT(*) FILTER (WHERE mpm.critical_alert_count > 0) as critical_alert_days
            INTO v_recent_performance
            FROM ml_lifecycle.model_performance_monitoring mpm
            WHERE mpm.version_id = v_version.version_id
              AND mpm.monitoring_period_start >= CURRENT_TIMESTAMP - INTERVAL '7 days';
            
            v_triggers := jsonb_build_array();
            
            -- Check accuracy degradation
            IF v_recent_performance.avg_accuracy < 0.75 THEN
                v_triggers := v_triggers || jsonb_build_object(
                    'type', 'accuracy_degradation',
                    'threshold', 0.75,
                    'current_value', v_recent_performance.avg_accuracy,
                    'severity', 'high'
                );
            END IF;
            
            -- Check data drift
            IF v_recent_performance.avg_drift > 0.4 THEN
                v_triggers := v_triggers || jsonb_build_object(
                    'type', 'data_drift',
                    'threshold', 0.4,
                    'current_value', v_recent_performance.avg_drift,
                    'severity', 'medium'
                );
            END IF;
            
            -- Check health score
            IF v_recent_performance.avg_health < 70 THEN
                v_triggers := v_triggers || jsonb_build_object(
                    'type', 'health_degradation',
                    'threshold', 70,
                    'current_value', v_recent_performance.avg_health,
                    'severity', 'high'
                );
            END IF;
            
            -- Check scheduled retraining (monthly)
            IF v_version.deployment_date <= CURRENT_DATE - INTERVAL '30 days' THEN
                v_triggers := v_triggers || jsonb_build_object(
                    'type', 'scheduled_retraining',
                    'last_retrain_date', v_version.deployment_date,
                    'frequency', 'monthly',
                    'severity', 'low'
                );
            END IF;
            
            -- Return recommendations if triggers found
            IF jsonb_array_length(v_triggers) > 0 THEN
                RETURN QUERY SELECT 
                    v_project.project_id,
                    v_version.version_id,
                    (trigger->>'type')::VARCHAR(50),
                    trigger,
                    CASE 
                        WHEN trigger->>'severity' = 'high' THEN 'immediate_retrain'
                        WHEN trigger->>'severity' = 'medium' THEN 'schedule_retrain'
                        ELSE 'monitor'
                    END::VARCHAR(50)
                FROM jsonb_array_elements(v_triggers) AS trigger;
            END IF;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

#### 3. Comprehensive Lifecycle Monitoring Views

```sql
-- Executive dashboard view
CREATE OR REPLACE VIEW ml_lifecycle.executive_dashboard AS
WITH project_summary AS (
    SELECT 
        domain,
        COUNT(*) as total_projects,
        COUNT(*) FILTER (WHERE project_status = 'deployed') as deployed_projects,
        COUNT(*) FILTER (WHERE project_status = 'development') as development_projects,
        COUNT(*) FILTER (WHERE project_status = 'planning') as planning_projects,
        AVG(EXTRACT(days FROM COALESCE(actual_completion_date, CURRENT_DATE) - start_date)) as avg_development_days,
        SUM(spent_budget) as total_spent_budget,
        SUM(allocated_budget) as total_allocated_budget
    FROM ml_lifecycle.model_projects
    GROUP BY domain
),
performance_summary AS (
    SELECT 
        mp.domain,
        COUNT(DISTINCT mv.version_id) as active_models,
        AVG(mpm.health_score) as avg_health_score,
        SUM(mpm.total_predictions) as total_predictions_24h,
        COUNT(*) FILTER (WHERE mi.severity_level IN ('high', 'critical')) as critical_incidents_24h
    FROM ml_lifecycle.model_projects mp
    JOIN ml_lifecycle.model_versions mv ON mp.project_id = mv.project_id
    LEFT JOIN ml_lifecycle.model_performance_monitoring mpm ON mv.version_id = mpm.version_id
        AND mpm.monitoring_period_start >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    LEFT JOIN ml_lifecycle.model_incidents mi ON mv.version_id = mi.version_id
        AND mi.detected_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    WHERE mv.deployment_status = 'deployed'
    GROUP BY mp.domain
)
SELECT 
    ps.domain,
    ps.total_projects,
    ps.deployed_projects,
    ps.development_projects,
    ps.planning_projects,
    ROUND(ps.avg_development_days::NUMERIC, 1) as avg_development_days,
    ps.total_spent_budget,
    ps.total_allocated_budget,
    ROUND((ps.total_spent_budget / NULLIF(ps.total_allocated_budget, 0) * 100)::NUMERIC, 1) as budget_utilization_percent,
    
    -- Performance metrics
    COALESCE(perf.active_models, 0) as active_models,
    ROUND(COALESCE(perf.avg_health_score, 0)::NUMERIC, 1) as avg_health_score,
    COALESCE(perf.total_predictions_24h, 0) as total_predictions_24h,
    COALESCE(perf.critical_incidents_24h, 0) as critical_incidents_24h,
    
    -- Status assessment
    CASE 
        WHEN COALESCE(perf.avg_health_score, 0) >= 90 THEN 'EXCELLENT'
        WHEN COALESCE(perf.avg_health_score, 0) >= 80 THEN 'GOOD'
        WHEN COALESCE(perf.avg_health_score, 0) >= 70 THEN 'FAIR'
        WHEN COALESCE(perf.avg_health_score, 0) >= 60 THEN 'POOR'
        ELSE 'CRITICAL'
    END as domain_health_status,
    
    -- ROI indicators
    CASE 
        WHEN ps.total_spent_budget > 0 AND perf.total_predictions_24h > 0 THEN
            ROUND((perf.total_predictions_24h::NUMERIC / ps.total_spent_budget)::NUMERIC, 2)
        ELSE 0
    END as predictions_per_dollar_invested
    
FROM project_summary ps
LEFT JOIN performance_summary perf ON ps.domain = perf.domain
ORDER BY ps.total_projects DESC, perf.avg_health_score DESC;

-- Model lifecycle status report
CREATE OR REPLACE VIEW ml_lifecycle.model_lifecycle_status AS
WITH model_status AS (
    SELECT 
        mp.project_id,
        mp.project_name,
        mp.domain,
        mp.project_status,
        mp.project_owner,
        mp.start_date,
        mp.target_completion_date,
        
        -- Latest version info
        mv.version_id,
        mv.version_number,
        mv.deployment_status,
        mv.deployment_environment,
        mv.deployment_date,
        mv.traffic_percentage,
        
        -- Performance metrics
        mpm.health_score,
        mpm.total_predictions,
        mpm.accuracy_rate,
        mpm.avg_latency_ms,
        mpm.error_rate,
        
        -- Incident counts
        COUNT(mi.incident_id) FILTER (WHERE mi.incident_status = 'open') as open_incidents,
        COUNT(mi.incident_id) FILTER (WHERE mi.severity_level IN ('high', 'critical')) as critical_incidents
        
    FROM ml_lifecycle.model_projects mp
    LEFT JOIN ml_lifecycle.model_versions mv ON mp.project_id = mv.project_id
        AND mv.deployment_status = 'deployed'
        AND mv.traffic_percentage > 0
    LEFT JOIN ml_lifecycle.model_performance_monitoring mpm ON mv.version_id = mpm.version_id
        AND mpm.monitoring_period_start >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    LEFT JOIN ml_lifecycle.model_incidents mi ON mv.version_id = mi.version_id
        AND mi.detected_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    GROUP BY mp.project_id, mp.project_name, mp.domain, mp.project_status, mp.project_owner,
             mp.start_date, mp.target_completion_date, mv.version_id, mv.version_number,
             mv.deployment_status, mv.deployment_environment, mv.deployment_date,
             mv.traffic_percentage, mpm.health_score, mpm.total_predictions, mpm.accuracy_rate,
             mpm.avg_latency_ms, mpm.error_rate
)
SELECT 
    project_name,
    domain,
    project_status,
    COALESCE(version_number, 'No deployment') as current_version,
    deployment_environment,
    deployment_status,
    
    -- Performance indicators
    COALESCE(health_score, 0) as health_score,
    CASE 
        WHEN health_score >= 90 THEN '🟢 Excellent'
        WHEN health_score >= 80 THEN '🟡 Good'
        WHEN health_score >= 70 THEN '🟠 Fair'
        WHEN health_score >= 60 THEN '🔴 Poor'
        ELSE '⚫ Critical'
    END as health_indicator,
    
    -- Operational metrics
    COALESCE(total_predictions, 0) as predictions_24h,
    ROUND(COALESCE(accuracy_rate * 100, 0)::NUMERIC, 1) as accuracy_percent,
    ROUND(COALESCE(avg_latency_ms, 0)::NUMERIC, 1) as avg_latency_ms,
    ROUND(COALESCE(error_rate * 100, 0)::NUMERIC, 2) as error_rate_percent,
    
    -- Issue tracking
    open_incidents,
    critical_incidents,
    
    -- Timeline information
    start_date,
    target_completion_date,
    deployment_date,
    CASE 
        WHEN deployment_date IS NOT NULL THEN 
            EXTRACT(days FROM CURRENT_DATE - deployment_date)
        ELSE NULL
    END as days_since_deployment,
    
    -- Status flags
    CASE 
        WHEN project_status = 'deployed' AND health_score < 70 THEN '⚠️ Needs attention'
        WHEN project_status = 'deployed' AND critical_incidents > 0 THEN '🚨 Critical issues'
        WHEN target_completion_date < CURRENT_DATE AND project_status != 'deployed' THEN '📅 Overdue'
        ELSE '✅ Normal'
    END as status_flag
    
FROM model_status
ORDER BY 
    CASE WHEN critical_incidents > 0 THEN 1 ELSE 2 END,
    CASE WHEN health_score < 70 THEN 1 ELSE 2 END,
    health_score DESC,
    project_name;
```

### Best Practices for ML Lifecycle Management

#### Key Success Factors:

1. **Comprehensive Tracking**: Monitor all stages from development to retirement
2. **Automated Workflows**: Implement CI/CD pipelines for model deployment
3. **Quality Gates**: Enforce quality checks before deployment
4. **Performance Monitoring**: Continuous monitoring of model health and performance
5. **Incident Management**: Rapid detection and resolution of issues
6. **Governance Framework**: Clear approval processes and compliance tracking
7. **Cost Management**: Track and optimize resource utilization and costs

#### Implementation Guidelines:

1. **Standardization**: Establish consistent processes across all ML projects
2. **Automation**: Automate repetitive tasks and monitoring activities
3. **Documentation**: Maintain comprehensive documentation and audit trails
4. **Collaboration**: Enable cross-functional team collaboration
5. **Scalability**: Design systems to handle multiple concurrent projects
6. **Security**: Implement proper access controls and data protection
7. **Continuous Improvement**: Regular review and enhancement of processes

This comprehensive SQL-based ML lifecycle management system provides robust infrastructure for managing machine learning projects from conception through retirement, ensuring operational excellence and business value delivery.

---

