# Sql Ml Interview Questions - Theory Questions

## Question 1

**What are the different types of JOIN operations in SQL?**

**Answer:**

### Theory

SQL JOIN operations are fundamental for combining data from multiple tables based on related columns. In machine learning contexts, JOINs are essential for feature engineering, data preparation, and creating comprehensive datasets by merging features from different sources. Understanding JOIN types and their performance implications is crucial for building efficient ML data pipelines.

### JOIN Types and Implementation

#### 1. INNER JOIN

```sql
-- Basic INNER JOIN for ML feature combination
SELECT 
    c.customer_id,
    c.age,
    c.income,
    t.total_transactions,
    t.avg_transaction_amount,
    p.product_preferences
FROM customers c
INNER JOIN transaction_summary t ON c.customer_id = t.customer_id
INNER JOIN product_analytics p ON c.customer_id = p.customer_id
WHERE c.active_status = 'ACTIVE';

-- INNER JOIN with aggregation for feature engineering
CREATE VIEW customer_features AS
SELECT 
    c.customer_id,
    c.demographics_score,
    COUNT(t.transaction_id) as transaction_count,
    AVG(t.amount) as avg_transaction_value,
    MAX(t.transaction_date) as last_transaction_date,
    STDDEV(t.amount) as transaction_variability
FROM customers c
INNER JOIN transactions t ON c.customer_id = t.customer_id
WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '365 days'
GROUP BY c.customer_id, c.demographics_score;

-- Complex INNER JOIN for time-series features
SELECT 
    f.customer_id,
    f.feature_date,
    f.behavioral_score,
    LAG(f.behavioral_score, 7) OVER (
        PARTITION BY f.customer_id 
        ORDER BY f.feature_date
    ) as score_week_ago,
    AVG(t.amount) as recent_avg_spending
FROM customer_features f
INNER JOIN transactions t ON f.customer_id = t.customer_id
    AND t.transaction_date BETWEEN f.feature_date - 7 AND f.feature_date
ORDER BY f.customer_id, f.feature_date;
```

#### 2. LEFT JOIN (LEFT OUTER JOIN)

```sql
-- LEFT JOIN for handling missing features gracefully
SELECT 
    c.customer_id,
    c.registration_date,
    c.risk_category,
    COALESCE(t.transaction_count, 0) as transaction_count,
    COALESCE(t.total_amount, 0) as total_spent,
    COALESCE(r.avg_rating, 3.0) as customer_satisfaction,
    -- Handle missing values for ML models
    CASE 
        WHEN t.customer_id IS NULL THEN 'NEW_CUSTOMER'
        WHEN t.transaction_count < 5 THEN 'LOW_ACTIVITY'
        ELSE 'ACTIVE'
    END as customer_segment
FROM customers c
LEFT JOIN (
    SELECT 
        customer_id,
        COUNT(*) as transaction_count,
        SUM(amount) as total_amount
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY customer_id
) t ON c.customer_id = t.customer_id
LEFT JOIN customer_reviews r ON c.customer_id = r.customer_id;

-- LEFT JOIN for feature completeness analysis
SELECT 
    'Customer Demographics' as feature_source,
    COUNT(*) as total_records,
    COUNT(c.age) as age_filled,
    COUNT(c.income) as income_filled,
    COUNT(c.education) as education_filled,
    ROUND(COUNT(c.age)::NUMERIC / COUNT(*) * 100, 2) as age_completeness_pct,
    ROUND(COUNT(c.income)::NUMERIC / COUNT(*) * 100, 2) as income_completeness_pct
FROM customers c
LEFT JOIN customer_profiles p ON c.customer_id = p.customer_id
WHERE c.created_date >= CURRENT_DATE - INTERVAL '30 days';

-- LEFT JOIN for temporal feature engineering
SELECT 
    c.customer_id,
    c.signup_date,
    COALESCE(f.first_transaction_date, c.signup_date) as first_activity,
    COALESCE(l.last_transaction_date, c.signup_date) as last_activity,
    EXTRACT(DAYS FROM (
        COALESCE(l.last_transaction_date, CURRENT_DATE) - c.signup_date
    )) as customer_lifetime_days,
    COALESCE(s.transaction_frequency, 0) as avg_monthly_transactions
FROM customers c
LEFT JOIN (
    SELECT customer_id, MIN(transaction_date) as first_transaction_date
    FROM transactions GROUP BY customer_id
) f ON c.customer_id = f.customer_id
LEFT JOIN (
    SELECT customer_id, MAX(transaction_date) as last_transaction_date
    FROM transactions GROUP BY customer_id
) l ON c.customer_id = l.customer_id
LEFT JOIN (
    SELECT 
        customer_id,
        COUNT(*)::NUMERIC / NULLIF(
            EXTRACT(MONTHS FROM (MAX(transaction_date) - MIN(transaction_date)) + INTERVAL '1 month'), 0
        ) as transaction_frequency
    FROM transactions
    GROUP BY customer_id
) s ON c.customer_id = s.customer_id;
```

#### 3. RIGHT JOIN (RIGHT OUTER JOIN)

```sql
-- RIGHT JOIN for ensuring all products are represented in features
SELECT 
    p.product_id,
    p.category,
    p.price,
    COALESCE(s.total_sales, 0) as total_sales,
    COALESCE(s.unique_customers, 0) as customer_count,
    COALESCE(r.avg_rating, 0) as average_rating,
    -- Calculate product performance metrics
    CASE 
        WHEN s.total_sales IS NULL THEN 'NO_SALES'
        WHEN s.total_sales < 100 THEN 'LOW_PERFORMER'
        WHEN s.total_sales < 1000 THEN 'MEDIUM_PERFORMER'
        ELSE 'HIGH_PERFORMER'
    END as performance_category
FROM (
    SELECT 
        product_id,
        SUM(quantity * price) as total_sales,
        COUNT(DISTINCT customer_id) as unique_customers
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY product_id
) s
RIGHT JOIN products p ON s.product_id = p.product_id
LEFT JOIN (
    SELECT 
        product_id,
        AVG(rating) as avg_rating
    FROM product_reviews
    GROUP BY product_id
) r ON p.product_id = r.product_id;

-- RIGHT JOIN for inventory and demand analysis
SELECT 
    i.product_id,
    i.current_stock,
    i.reorder_level,
    COALESCE(d.predicted_demand, 0) as ml_predicted_demand,
    COALESCE(h.avg_weekly_sales, 0) as historical_avg_sales,
    -- Stock sufficiency analysis
    CASE 
        WHEN i.current_stock < i.reorder_level THEN 'REORDER_NEEDED'
        WHEN i.current_stock < COALESCE(d.predicted_demand, 0) * 2 THEN 'LOW_STOCK'
        ELSE 'SUFFICIENT'
    END as stock_status
FROM ml_demand_predictions d
RIGHT JOIN inventory i ON d.product_id = i.product_id
LEFT JOIN (
    SELECT 
        product_id,
        AVG(weekly_sales) as avg_weekly_sales
    FROM sales_history
    WHERE week_ending >= CURRENT_DATE - INTERVAL '52 weeks'
    GROUP BY product_id
) h ON i.product_id = h.product_id;
```

#### 4. FULL OUTER JOIN

```sql
-- FULL OUTER JOIN for comprehensive data reconciliation
SELECT 
    COALESCE(o.customer_id, r.customer_id) as customer_id,
    COALESCE(o.total_orders, 0) as order_count,
    COALESCE(o.total_spent, 0) as total_spent,
    COALESCE(r.total_returns, 0) as return_count,
    COALESCE(r.return_value, 0) as return_value,
    -- Calculate return rate and customer value metrics
    CASE 
        WHEN o.customer_id IS NULL THEN 'RETURNS_ONLY'
        WHEN r.customer_id IS NULL THEN 'NO_RETURNS'
        ELSE 'MIXED_ACTIVITY'
    END as customer_type,
    ROUND(
        COALESCE(r.return_value, 0)::NUMERIC / 
        NULLIF(COALESCE(o.total_spent, 0), 0) * 100, 2
    ) as return_rate_percentage
FROM (
    SELECT 
        customer_id,
        COUNT(*) as total_orders,
        SUM(total_amount) as total_spent
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY customer_id
) o
FULL OUTER JOIN (
    SELECT 
        customer_id,
        COUNT(*) as total_returns,
        SUM(return_amount) as return_value
    FROM returns
    WHERE return_date >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY customer_id
) r ON o.customer_id = r.customer_id;

-- FULL OUTER JOIN for feature completeness across multiple sources
SELECT 
    COALESCE(d.customer_id, t.customer_id, s.customer_id) as customer_id,
    d.demographic_score,
    t.transaction_features,
    s.social_features,
    -- Data completeness indicators
    CASE WHEN d.customer_id IS NOT NULL THEN 1 ELSE 0 END as has_demographics,
    CASE WHEN t.customer_id IS NOT NULL THEN 1 ELSE 0 END as has_transactions,
    CASE WHEN s.customer_id IS NOT NULL THEN 1 ELSE 0 END as has_social_data,
    -- Combined feature vector with nulls handled
    jsonb_build_object(
        'demographics', COALESCE(d.demographic_score, 0),
        'transaction_volume', COALESCE(t.transaction_count, 0),
        'social_engagement', COALESCE(s.engagement_score, 0),
        'completeness_score', 
            (CASE WHEN d.customer_id IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN t.customer_id IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN s.customer_id IS NOT NULL THEN 1 ELSE 0 END)::NUMERIC / 3
    ) as feature_vector
FROM demographic_features d
FULL OUTER JOIN transaction_features t ON d.customer_id = t.customer_id
FULL OUTER JOIN social_features s ON COALESCE(d.customer_id, t.customer_id) = s.customer_id;
```

#### 5. CROSS JOIN

```sql
-- CROSS JOIN for creating feature combinations
SELECT 
    p.product_id,
    p.category as product_category,
    c.customer_segment,
    c.risk_score,
    -- Create interaction features
    CASE 
        WHEN p.category = 'PREMIUM' AND c.customer_segment = 'HIGH_VALUE' THEN 1
        ELSE 0
    END as premium_high_value_interaction,
    CASE 
        WHEN p.price_tier = 'EXPENSIVE' AND c.risk_score = 'LOW' THEN 1
        ELSE 0
    END as expensive_low_risk_interaction
FROM products p
CROSS JOIN customer_segments c
WHERE p.active = TRUE AND c.active = TRUE;

-- CROSS JOIN for time-series grid generation
SELECT 
    c.customer_id,
    d.date_value,
    COALESCE(t.transaction_count, 0) as daily_transactions,
    COALESCE(t.total_amount, 0) as daily_amount,
    -- Forward fill missing values
    COALESCE(t.total_amount, 
        LAG(t.total_amount) IGNORE NULLS OVER (
            PARTITION BY c.customer_id 
            ORDER BY d.date_value
        ), 0
    ) as amount_filled
FROM (
    SELECT DISTINCT customer_id FROM customers WHERE active = TRUE
) c
CROSS JOIN (
    SELECT generate_series(
        CURRENT_DATE - INTERVAL '30 days',
        CURRENT_DATE,
        INTERVAL '1 day'
    )::DATE as date_value
) d
LEFT JOIN (
    SELECT 
        customer_id,
        DATE(transaction_timestamp) as transaction_date,
        COUNT(*) as transaction_count,
        SUM(amount) as total_amount
    FROM transactions
    GROUP BY customer_id, DATE(transaction_timestamp)
) t ON c.customer_id = t.customer_id AND d.date_value = t.transaction_date
ORDER BY c.customer_id, d.date_value;

-- CROSS JOIN for A/B testing feature combinations
SELECT 
    e.experiment_id,
    e.variant_name,
    f.feature_name,
    f.feature_type,
    -- Create experiment-feature interaction matrix
    CASE 
        WHEN e.variant_name = 'CONTROL' THEN f.baseline_value
        WHEN e.variant_name = 'VARIANT_A' THEN f.baseline_value * 1.1
        WHEN e.variant_name = 'VARIANT_B' THEN f.baseline_value * 0.9
    END as adjusted_feature_value
FROM experiment_variants e
CROSS JOIN model_features f
WHERE e.active = TRUE AND f.feature_type IN ('NUMERIC', 'CONTINUOUS');
```

### Machine Learning Applications

#### 1. Feature Engineering with JOINs

```sql
-- Multi-table feature engineering pipeline
CREATE OR REPLACE VIEW ml_customer_features AS
WITH customer_base AS (
    SELECT customer_id, registration_date, customer_tier
    FROM customers WHERE active = TRUE
),
transaction_features AS (
    SELECT 
        customer_id,
        COUNT(*) as transaction_count_90d,
        AVG(amount) as avg_transaction_amount,
        STDDEV(amount) as transaction_amount_std,
        MAX(transaction_date) as last_transaction_date
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY customer_id
),
product_affinity AS (
    SELECT 
        t.customer_id,
        STRING_AGG(DISTINCT p.category, ',') as preferred_categories,
        COUNT(DISTINCT p.category) as category_diversity
    FROM transactions t
    JOIN order_items oi ON t.transaction_id = oi.transaction_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '180 days'
    GROUP BY t.customer_id
),
engagement_metrics AS (
    SELECT 
        customer_id,
        COUNT(*) as website_visits,
        AVG(session_duration) as avg_session_duration,
        SUM(pages_viewed) as total_pages_viewed
    FROM web_analytics
    WHERE visit_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY customer_id
)
SELECT 
    cb.customer_id,
    cb.customer_tier,
    EXTRACT(DAYS FROM (CURRENT_DATE - cb.registration_date)) as customer_age_days,
    COALESCE(tf.transaction_count_90d, 0) as transaction_count,
    COALESCE(tf.avg_transaction_amount, 0) as avg_transaction_amount,
    COALESCE(tf.transaction_amount_std, 0) as transaction_variability,
    COALESCE(pa.category_diversity, 0) as product_diversity_score,
    COALESCE(em.avg_session_duration, 0) as engagement_score,
    -- Derived features
    CASE 
        WHEN tf.last_transaction_date >= CURRENT_DATE - INTERVAL '7 days' THEN 'RECENT'
        WHEN tf.last_transaction_date >= CURRENT_DATE - INTERVAL '30 days' THEN 'ACTIVE'
        WHEN tf.last_transaction_date >= CURRENT_DATE - INTERVAL '90 days' THEN 'DORMANT'
        ELSE 'INACTIVE'
    END as activity_status
FROM customer_base cb
LEFT JOIN transaction_features tf ON cb.customer_id = tf.customer_id
LEFT JOIN product_affinity pa ON cb.customer_id = pa.customer_id
LEFT JOIN engagement_metrics em ON cb.customer_id = em.customer_id;
```

### Performance Optimization Strategies

#### 1. Index Optimization for JOINs

```sql
-- Optimized indexes for JOIN operations
CREATE INDEX CONCURRENTLY idx_customers_id_active 
ON customers(customer_id) WHERE active = TRUE;

CREATE INDEX CONCURRENTLY idx_transactions_customer_date 
ON transactions(customer_id, transaction_date DESC);

CREATE INDEX CONCURRENTLY idx_order_items_covering 
ON order_items(order_id, product_id) 
INCLUDE (quantity, price);

-- Composite index for multi-column JOINs
CREATE INDEX CONCURRENTLY idx_features_customer_date_type
ON customer_features(customer_id, feature_date, feature_type);

-- Partial index for filtered JOINs
CREATE INDEX CONCURRENTLY idx_transactions_recent
ON transactions(customer_id, amount)
WHERE transaction_date >= CURRENT_DATE - INTERVAL '365 days';
```

#### 2. JOIN Performance Monitoring

```sql
-- Monitor JOIN performance
CREATE OR REPLACE VIEW join_performance_analysis AS
SELECT 
    schemaname,
    tablename,
    seq_scan as sequential_scans,
    seq_tup_read as sequential_tuples_read,
    idx_scan as index_scans,
    idx_tup_fetch as index_tuples_fetched,
    ROUND(
        idx_tup_fetch::NUMERIC / NULLIF(idx_scan, 0), 2
    ) as avg_tuples_per_index_scan,
    CASE 
        WHEN seq_scan > idx_scan THEN 'INDEX_NEEDED'
        WHEN idx_scan = 0 THEN 'UNUSED_TABLE'
        ELSE 'OPTIMIZED'
    END as optimization_status
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY seq_tup_read DESC;
```

### Real-World Applications

1. **Customer 360 Views**: Combining data from CRM, transactions, and web analytics
2. **Feature Engineering**: Creating comprehensive ML features from multiple data sources
3. **Data Quality Assessment**: Identifying missing data across related tables
4. **Time-Series Analysis**: Joining temporal data with static attributes
5. **A/B Testing**: Combining experiment data with user behavior metrics

### Best Practices

1. **Index Strategy**: Create appropriate indexes on JOIN columns
2. **Join Order**: Place most selective conditions first
3. **Data Types**: Ensure matching data types for JOIN columns
4. **NULL Handling**: Plan for NULL values in outer JOINs
5. **Performance Testing**: Monitor query execution plans

### Common Pitfalls

1. **Cartesian Products**: Unintended CROSS JOINs causing exponential row growth
2. **Missing Indexes**: Poor performance on large table JOINs
3. **Data Type Mismatches**: Implicit type conversions affecting performance
4. **NULL Confusion**: Unexpected results with NULL values in JOINs
5. **Nested Loop Joins**: Inefficient join algorithms for large datasets

### Performance Considerations

- **INNER JOIN**: Fastest, filters out non-matching rows
- **LEFT/RIGHT JOIN**: Moderate performance, preserves all rows from one side
- **FULL OUTER JOIN**: Slowest, preserves all rows from both sides
- **Index Usage**: Properly indexed JOIN columns can improve performance by 10-100x
- **Memory Requirements**: Complex JOINs may require significant work_mem allocation

---

## Question 2

**Explain the difference between WHERE and HAVING clauses.**

**Answer:**

### Theory

WHERE and HAVING clauses are both used for filtering data in SQL, but they operate at different stages of query execution and serve distinct purposes. In machine learning contexts, understanding when to use each clause is crucial for efficient data preprocessing, feature engineering, and model training data preparation.

**Key Differences:**
- **WHERE**: Filters individual rows before grouping and aggregation
- **HAVING**: Filters groups after grouping and aggregation
- **Execution Order**: WHERE → GROUP BY → HAVING → SELECT → ORDER BY

### WHERE Clause - Row-Level Filtering

#### 1. Basic Row Filtering for ML Data Preparation

```sql
-- Filter training data for model building
SELECT 
    customer_id,
    age,
    income,
    transaction_amount,
    risk_score
FROM customer_transactions
WHERE 
    age BETWEEN 18 AND 80                    -- Remove outlier ages
    AND income > 0                           -- Valid income values
    AND transaction_amount > 0               -- Valid transactions
    AND risk_score IS NOT NULL              -- Complete feature data
    AND transaction_date >= '2023-01-01'    -- Recent data only
    AND customer_status = 'ACTIVE';         -- Active customers only

-- Complex WHERE conditions for feature engineering
SELECT 
    customer_id,
    product_category,
    purchase_date,
    amount,
    -- Create binary features based on conditions
    CASE WHEN amount > 1000 THEN 1 ELSE 0 END as high_value_purchase,
    CASE WHEN EXTRACT(DOW FROM purchase_date) IN (6, 0) THEN 1 ELSE 0 END as weekend_purchase
FROM purchases
WHERE 
    purchase_date >= CURRENT_DATE - INTERVAL '2 years'  -- Training window
    AND amount BETWEEN 10 AND 10000                     -- Remove extreme outliers
    AND product_category IN ('ELECTRONICS', 'CLOTHING', 'BOOKS')  -- Target categories
    AND payment_method != 'REFUND'                      -- Exclude refunds
    AND customer_id NOT IN (                            -- Exclude test customers
        SELECT customer_id FROM test_customers
    );

-- Time-based filtering for temporal ML features
SELECT 
    customer_id,
    event_timestamp,
    event_type,
    event_value,
    LAG(event_timestamp) OVER (
        PARTITION BY customer_id 
        ORDER BY event_timestamp
    ) as previous_event_time
FROM customer_events
WHERE 
    event_timestamp >= CURRENT_DATE - INTERVAL '1 year'
    AND event_type IN ('LOGIN', 'PURCHASE', 'VIEW_PRODUCT', 'ADD_TO_CART')
    AND event_value IS NOT NULL
    AND customer_id IN (
        SELECT customer_id 
        FROM active_customers 
        WHERE signup_date <= CURRENT_DATE - INTERVAL '30 days'  -- Established customers
    )
ORDER BY customer_id, event_timestamp;
```

#### 2. Data Quality Filtering with WHERE

```sql
-- Comprehensive data quality filtering for ML pipeline
SELECT 
    transaction_id,
    customer_id,
    merchant_id,
    amount,
    transaction_timestamp,
    payment_method,
    -- Data quality indicators
    'PASSED_QC' as data_quality_status
FROM transactions
WHERE 
    -- Remove null and invalid values
    customer_id IS NOT NULL
    AND merchant_id IS NOT NULL
    AND amount IS NOT NULL
    AND transaction_timestamp IS NOT NULL
    
    -- Business logic constraints
    AND amount > 0
    AND amount <= 50000  -- Reasonable transaction limit
    
    -- Temporal constraints
    AND transaction_timestamp >= '2020-01-01'
    AND transaction_timestamp <= CURRENT_TIMESTAMP
    
    -- Valid payment methods
    AND payment_method IN ('CREDIT_CARD', 'DEBIT_CARD', 'PAYPAL', 'BANK_TRANSFER')
    
    -- Exclude known fraudulent patterns
    AND NOT (amount = 1.00 AND payment_method = 'CREDIT_CARD')  -- Common test pattern
    
    -- Geographic constraints
    AND merchant_id NOT IN (
        SELECT merchant_id FROM blocked_merchants
    );

-- Feature completeness filtering
SELECT 
    customer_id,
    feature_vector,
    target_variable
FROM ml_dataset
WHERE 
    -- Ensure feature completeness
    jsonb_array_length(feature_vector) = 25  -- Expected feature count
    AND target_variable IS NOT NULL
    
    -- Remove sparse feature vectors
    AND (
        SELECT COUNT(*) 
        FROM jsonb_array_elements_text(feature_vector) as elem 
        WHERE elem::numeric != 0
    ) >= 5  -- At least 5 non-zero features
    
    -- Temporal validation
    AND created_date BETWEEN 
        CURRENT_DATE - INTERVAL '18 months' AND 
        CURRENT_DATE - INTERVAL '1 month';  -- Avoid data leakage
```

### HAVING Clause - Group-Level Filtering

#### 1. Aggregate-Based Filtering for ML Features

```sql
-- Filter customer segments based on aggregated behavior
SELECT 
    customer_id,
    COUNT(*) as transaction_count,
    AVG(amount) as avg_transaction_amount,
    SUM(amount) as total_spent,
    STDDEV(amount) as spending_variability,
    -- Derived ML features
    COUNT(DISTINCT DATE_TRUNC('month', transaction_date)) as active_months,
    MAX(transaction_date) as last_transaction_date
FROM transactions
WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY customer_id
HAVING 
    COUNT(*) >= 5                           -- Minimum transaction history
    AND SUM(amount) > 100                   -- Minimum total spending
    AND COUNT(DISTINCT DATE_TRUNC('month', transaction_date)) >= 3  -- Multi-month activity
    AND MAX(transaction_date) >= CURRENT_DATE - INTERVAL '90 days'  -- Recent activity
    AND STDDEV(amount) > 0;                 -- Spending variation exists

-- Product performance analysis with HAVING
SELECT 
    product_category,
    product_subcategory,
    COUNT(DISTINCT customer_id) as unique_customers,
    COUNT(*) as total_sales,
    AVG(sale_amount) as avg_sale_amount,
    SUM(sale_amount) as total_revenue,
    -- Performance metrics
    COUNT(*)::NUMERIC / COUNT(DISTINCT customer_id) as purchases_per_customer,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sale_amount) as median_sale_amount
FROM product_sales
WHERE sale_date >= CURRENT_DATE - INTERVAL '6 months'
GROUP BY product_category, product_subcategory
HAVING 
    COUNT(DISTINCT customer_id) >= 50      -- Sufficient customer base
    AND COUNT(*) >= 100                    -- Sufficient transaction volume
    AND SUM(sale_amount) >= 10000          -- Minimum revenue threshold
    AND AVG(sale_amount) >= 20             -- Reasonable average sale
ORDER BY total_revenue DESC;

-- Customer lifetime value segmentation
SELECT 
    customer_acquisition_channel,
    customer_tier,
    COUNT(*) as customer_count,
    AVG(total_clv) as avg_customer_lifetime_value,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_clv) as clv_q1,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_clv) as clv_q3,
    -- Segment characteristics
    AVG(months_active) as avg_months_active,
    AVG(total_transactions) as avg_transactions
FROM customer_lifetime_metrics
GROUP BY customer_acquisition_channel, customer_tier
HAVING 
    COUNT(*) >= 20                         -- Statistical significance
    AND AVG(total_clv) > 0                 -- Positive CLV
    AND AVG(months_active) >= 3            -- Established customers
    AND AVG(total_transactions) >= 2       -- Minimum engagement
ORDER BY avg_customer_lifetime_value DESC;
```

#### 2. Statistical Filtering with HAVING

```sql
-- Feature selection based on statistical properties
SELECT 
    feature_name,
    feature_category,
    COUNT(*) as sample_size,
    AVG(feature_value) as mean_value,
    STDDEV(feature_value) as standard_deviation,
    MIN(feature_value) as min_value,
    MAX(feature_value) as max_value,
    -- Statistical measures for feature selection
    STDDEV(feature_value) / NULLIF(ABS(AVG(feature_value)), 0) as coefficient_of_variation,
    COUNT(DISTINCT feature_value) as unique_values,
    COUNT(DISTINCT feature_value)::NUMERIC / COUNT(*) as uniqueness_ratio
FROM model_features
WHERE feature_value IS NOT NULL
GROUP BY feature_name, feature_category
HAVING 
    COUNT(*) >= 1000                       -- Sufficient sample size
    AND STDDEV(feature_value) > 0          -- Feature has variation
    AND COUNT(DISTINCT feature_value) > 1  -- Not constant
    AND COUNT(DISTINCT feature_value) <= COUNT(*) * 0.95  -- Not too sparse
    AND ABS(AVG(feature_value)) > 0.001    -- Non-zero mean (for ratio features)
ORDER BY coefficient_of_variation DESC;

-- Model performance by segment analysis
SELECT 
    model_name,
    customer_segment,
    prediction_date,
    COUNT(*) as prediction_count,
    AVG(CASE WHEN actual_value = predicted_value THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(confidence_score) as avg_confidence,
    -- Performance distribution
    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY confidence_score) as confidence_p10,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY confidence_score) as confidence_p90
FROM model_predictions
WHERE prediction_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY model_name, customer_segment, prediction_date
HAVING 
    COUNT(*) >= 50                         -- Sufficient predictions per segment
    AND AVG(confidence_score) >= 0.6       -- Minimum confidence threshold
    AND AVG(CASE WHEN actual_value = predicted_value THEN 1.0 ELSE 0.0 END) >= 0.7  -- Accuracy threshold
ORDER BY model_name, customer_segment, prediction_date;
```

### Combined WHERE and HAVING Usage

#### 1. Complex ML Data Pipeline

```sql
-- Comprehensive customer feature engineering with both WHERE and HAVING
WITH customer_transactions AS (
    -- WHERE filters individual transactions
    SELECT 
        customer_id,
        transaction_date,
        amount,
        product_category,
        payment_method
    FROM transactions
    WHERE 
        transaction_date >= CURRENT_DATE - INTERVAL '24 months'  -- Recent data
        AND amount > 0                                           -- Valid amounts
        AND customer_id IS NOT NULL                              -- Complete records
        AND transaction_status = 'COMPLETED'                     -- Successful transactions
),
customer_aggregates AS (
    -- GROUP BY with HAVING filters customer groups
    SELECT 
        customer_id,
        COUNT(*) as transaction_count,
        AVG(amount) as avg_amount,
        SUM(amount) as total_amount,
        STDDEV(amount) as amount_variability,
        COUNT(DISTINCT product_category) as category_diversity,
        COUNT(DISTINCT payment_method) as payment_method_diversity,
        MAX(transaction_date) as last_transaction,
        MIN(transaction_date) as first_transaction,
        -- Advanced features
        COUNT(*) / NULLIF(
            EXTRACT(DAYS FROM MAX(transaction_date) - MIN(transaction_date)) + 1, 0
        ) as daily_transaction_rate
    FROM customer_transactions
    GROUP BY customer_id
    HAVING 
        COUNT(*) >= 10                      -- Minimum transaction history
        AND SUM(amount) >= 500              -- Minimum spending
        AND COUNT(DISTINCT product_category) >= 2  -- Multi-category customers
        AND MAX(transaction_date) >= CURRENT_DATE - INTERVAL '6 months'  -- Recent activity
        AND EXTRACT(DAYS FROM MAX(transaction_date) - MIN(transaction_date)) >= 30  -- Sustained activity
)
SELECT 
    ca.*,
    -- Additional derived features
    CASE 
        WHEN total_amount > 5000 THEN 'HIGH_VALUE'
        WHEN total_amount > 1000 THEN 'MEDIUM_VALUE'
        ELSE 'LOW_VALUE'
    END as value_segment,
    CASE 
        WHEN daily_transaction_rate > 0.1 THEN 'HIGH_FREQUENCY'
        WHEN daily_transaction_rate > 0.05 THEN 'MEDIUM_FREQUENCY'
        ELSE 'LOW_FREQUENCY'
    END as frequency_segment
FROM customer_aggregates ca;

-- Time-based cohort analysis
SELECT 
    DATE_TRUNC('month', first_purchase_date) as cohort_month,
    EXTRACT(MONTH FROM AGE(activity_month, DATE_TRUNC('month', first_purchase_date))) as months_since_first_purchase,
    COUNT(DISTINCT customer_id) as active_customers,
    AVG(monthly_revenue) as avg_monthly_revenue_per_customer,
    SUM(monthly_revenue) as total_cohort_revenue
FROM (
    -- WHERE clause filters base data
    SELECT 
        customer_id,
        DATE_TRUNC('month', transaction_date) as activity_month,
        MIN(DATE_TRUNC('month', transaction_date)) OVER (PARTITION BY customer_id) as first_purchase_date,
        SUM(amount) as monthly_revenue
    FROM transactions
    WHERE 
        transaction_date >= '2022-01-01'
        AND transaction_status = 'COMPLETED'
        AND amount > 0
    GROUP BY customer_id, DATE_TRUNC('month', transaction_date)
) monthly_activity
GROUP BY 
    DATE_TRUNC('month', first_purchase_date),
    EXTRACT(MONTH FROM AGE(activity_month, DATE_TRUNC('month', first_purchase_date)))
HAVING 
    COUNT(DISTINCT customer_id) >= 20       -- Statistical significance
    AND SUM(monthly_revenue) > 0            -- Positive revenue
ORDER BY cohort_month, months_since_first_purchase;
```

#### 2. Model Training Data Preparation

```sql
-- ML training dataset preparation with quality controls
CREATE TABLE ml_training_data AS
WITH filtered_base_data AS (
    -- WHERE: Filter individual records for quality
    SELECT 
        customer_id,
        feature_date,
        feature_vector,
        target_variable,
        data_source
    FROM raw_ml_features
    WHERE 
        feature_date BETWEEN '2022-01-01' AND '2023-12-31'  -- Training period
        AND feature_vector IS NOT NULL                       -- Complete features
        AND target_variable IS NOT NULL                      -- Complete targets
        AND jsonb_array_length(feature_vector) = 50          -- Expected feature count
        AND data_source IN ('PRODUCTION', 'VALIDATED')       -- Trusted sources
),
customer_feature_stats AS (
    -- GROUP BY and HAVING: Ensure sufficient data per customer
    SELECT 
        customer_id,
        COUNT(*) as record_count,
        MIN(feature_date) as first_feature_date,
        MAX(feature_date) as last_feature_date,
        AVG(target_variable::numeric) as avg_target,
        STDDEV(target_variable::numeric) as target_variance
    FROM filtered_base_data
    GROUP BY customer_id
    HAVING 
        COUNT(*) >= 30                       -- Minimum records per customer
        AND COUNT(DISTINCT feature_date) >= 25  -- Temporal diversity
        AND STDDEV(target_variable::numeric) > 0  -- Target variation exists
        AND MAX(feature_date) - MIN(feature_date) >= INTERVAL '90 days'  -- Time span
)
SELECT 
    fbd.customer_id,
    fbd.feature_date,
    fbd.feature_vector,
    fbd.target_variable,
    cfs.record_count as customer_sample_size,
    cfs.target_variance as customer_target_variance
FROM filtered_base_data fbd
JOIN customer_feature_stats cfs ON fbd.customer_id = cfs.customer_id;
```

### Performance Optimization

#### 1. Index Strategies for WHERE vs HAVING

```sql
-- Indexes optimized for WHERE clause filtering
CREATE INDEX CONCURRENTLY idx_transactions_where_filters
ON transactions(transaction_date, customer_id, amount)
WHERE transaction_status = 'COMPLETED' AND amount > 0;

-- Covering index for GROUP BY operations used with HAVING
CREATE INDEX CONCURRENTLY idx_transactions_groupby_having
ON transactions(customer_id, transaction_date)
INCLUDE (amount, product_category);

-- Partial index for specific WHERE conditions
CREATE INDEX CONCURRENTLY idx_active_customers_recent
ON customer_features(customer_id, feature_date)
WHERE customer_status = 'ACTIVE' 
AND feature_date >= CURRENT_DATE - INTERVAL '1 year';
```

#### 2. Query Optimization Examples

```sql
-- Optimized query showing WHERE before HAVING
EXPLAIN (ANALYZE, BUFFERS) 
SELECT 
    customer_segment,
    COUNT(*) as customer_count,
    AVG(clv_score) as avg_clv
FROM customer_metrics
WHERE 
    calculation_date = '2024-01-01'          -- Early filtering
    AND clv_score IS NOT NULL                -- Remove nulls early
    AND customer_status = 'ACTIVE'           -- Filter active only
GROUP BY customer_segment
HAVING 
    COUNT(*) >= 100                          -- Segment size threshold
    AND AVG(clv_score) > 0;                  -- Positive CLV segments
```

### Real-World Applications

1. **Data Quality Control**: WHERE for record-level validation, HAVING for aggregate validation
2. **Customer Segmentation**: WHERE for basic filters, HAVING for segment criteria
3. **Feature Engineering**: WHERE for data preparation, HAVING for feature selection
4. **Model Training**: WHERE for data filtering, HAVING for sample size requirements
5. **Performance Analysis**: WHERE for time periods, HAVING for statistical significance

### Best Practices

1. **Filter Early**: Use WHERE to reduce data volume before GROUP BY
2. **Index Appropriately**: Create indexes that support WHERE conditions
3. **Logical Ordering**: Apply WHERE before HAVING in query logic
4. **Performance Testing**: Monitor execution plans for both clause types
5. **Documentation**: Clearly document filtering logic for maintainability

### Common Pitfalls

1. **Wrong Clause Choice**: Using HAVING for non-aggregate conditions
2. **Performance Issues**: Not filtering early with WHERE clause
3. **Logical Errors**: Misunderstanding execution order
4. **Index Misalignment**: Indexes not matching WHERE conditions
5. **Aggregate Confusion**: Applying aggregate functions in WHERE clause

### Performance Considerations

- **WHERE**: Processed before GROUP BY, more efficient for large datasets
- **HAVING**: Processed after GROUP BY, operates on smaller result sets
- **Index Usage**: WHERE conditions can use indexes more effectively
- **Memory Usage**: Early WHERE filtering reduces GROUP BY memory requirements
- **Execution Time**: WHERE filters can significantly reduce processing time

---

## Question 3

**Describe a subquery and its typical use case.**

**Answer:**

### Theory

A subquery is a SQL query nested inside another query (outer query). Also known as inner queries or nested queries, subqueries can appear in SELECT, FROM, WHERE, and HAVING clauses. In machine learning contexts, subqueries are essential for complex feature engineering, data filtering, statistical calculations, and creating sophisticated training datasets.

**Types of Subqueries:**
- **Scalar Subqueries**: Return single value
- **Row Subqueries**: Return single row
- **Column Subqueries**: Return single column
- **Table Subqueries**: Return multiple rows and columns

### Scalar Subqueries - Single Value Returns

#### 1. Statistical Feature Engineering

```sql
-- Create normalized features using scalar subqueries
SELECT 
    customer_id,
    transaction_amount,
    -- Z-score normalization using subqueries
    (transaction_amount - (SELECT AVG(transaction_amount) FROM transactions)) / 
    (SELECT STDDEV(transaction_amount) FROM transactions) as amount_zscore,
    
    -- Percentile ranking
    (SELECT PERCENT_RANK() OVER (ORDER BY t2.transaction_amount) 
     FROM transactions t2 
     WHERE t2.customer_id = t1.customer_id 
     AND t2.transaction_amount <= t1.transaction_amount 
     LIMIT 1) as amount_percentile_rank,
    
    -- Customer-specific deviation from average
    transaction_amount - (
        SELECT AVG(transaction_amount) 
        FROM transactions t3 
        WHERE t3.customer_id = t1.customer_id
    ) as amount_deviation_from_personal_avg,
    
    -- Market comparison features
    CASE 
        WHEN transaction_amount > (
            SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_amount) 
            FROM transactions
        ) THEN 'HIGH_SPENDER'
        WHEN transaction_amount > (
            SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_amount) 
            FROM transactions
        ) THEN 'MEDIUM_SPENDER'
        ELSE 'LOW_SPENDER'
    END as spending_category
FROM transactions t1
WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months';

-- Advanced feature engineering with correlated subqueries
SELECT 
    customer_id,
    product_id,
    purchase_date,
    price,
    quantity,
    -- Price comparison features
    price - (
        SELECT AVG(price) 
        FROM purchases p2 
        WHERE p2.product_id = p1.product_id
    ) as price_vs_product_avg,
    
    -- Customer loyalty features
    (SELECT COUNT(*) 
     FROM purchases p3 
     WHERE p3.customer_id = p1.customer_id 
     AND p3.purchase_date <= p1.purchase_date) as cumulative_purchases,
    
    -- Temporal features
    EXTRACT(DAYS FROM p1.purchase_date - (
        SELECT MAX(purchase_date) 
        FROM purchases p4 
        WHERE p4.customer_id = p1.customer_id 
        AND p4.purchase_date < p1.purchase_date
    )) as days_since_last_purchase,
    
    -- Product popularity features
    (SELECT COUNT(DISTINCT customer_id) 
     FROM purchases p5 
     WHERE p5.product_id = p1.product_id 
     AND p5.purchase_date >= p1.purchase_date - INTERVAL '30 days') as product_popularity_30d
FROM purchases p1
WHERE purchase_date >= CURRENT_DATE - INTERVAL '2 years';
```

#### 2. Dynamic Threshold and Benchmark Calculations

```sql
-- Dynamic fraud detection thresholds using scalar subqueries
SELECT 
    transaction_id,
    customer_id,
    amount,
    transaction_timestamp,
    merchant_category,
    -- Dynamic fraud scoring based on customer behavior
    CASE 
        WHEN amount > (
            SELECT AVG(amount) + 3 * STDDEV(amount)
            FROM transactions t2
            WHERE t2.customer_id = t1.customer_id
            AND t2.transaction_timestamp >= t1.transaction_timestamp - INTERVAL '90 days'
        ) THEN 5  -- High anomaly score
        WHEN amount > (
            SELECT AVG(amount) + 2 * STDDEV(amount)
            FROM transactions t3
            WHERE t3.customer_id = t1.customer_id
            AND t3.transaction_timestamp >= t1.transaction_timestamp - INTERVAL '30 days'
        ) THEN 3  -- Medium anomaly score
        ELSE 1    -- Normal score
    END as amount_anomaly_score,
    
    -- Merchant category risk assessment
    CASE 
        WHEN merchant_category IN (
            SELECT category 
            FROM high_risk_categories 
            WHERE risk_score > (SELECT AVG(risk_score) FROM high_risk_categories)
        ) THEN 'HIGH_RISK'
        ELSE 'NORMAL_RISK'
    END as merchant_risk_level,
    
    -- Time-based anomaly detection
    CASE 
        WHEN EXTRACT(HOUR FROM transaction_timestamp) NOT IN (
            SELECT DISTINCT EXTRACT(HOUR FROM transaction_timestamp)
            FROM transactions t4
            WHERE t4.customer_id = t1.customer_id
            AND t4.transaction_timestamp >= t1.transaction_timestamp - INTERVAL '60 days'
            GROUP BY EXTRACT(HOUR FROM transaction_timestamp)
            HAVING COUNT(*) >= 2
        ) THEN 'UNUSUAL_TIME'
        ELSE 'NORMAL_TIME'
    END as temporal_anomaly_flag
FROM transactions t1
WHERE transaction_timestamp >= CURRENT_DATE - INTERVAL '1 day';
```

### Column Subqueries - Multiple Row, Single Column

#### 1. Feature Selection and Filtering

```sql
-- Feature selection using column subqueries
SELECT 
    customer_id,
    feature_name,
    feature_value,
    feature_importance_score
FROM customer_features
WHERE feature_name IN (
    -- Select top features by importance
    SELECT feature_name
    FROM feature_importance_rankings
    WHERE importance_score > 0.05
    ORDER BY importance_score DESC
    LIMIT 20
)
AND customer_id IN (
    -- Select customers with sufficient data
    SELECT customer_id
    FROM customer_data_completeness
    WHERE completeness_score >= 0.8
    AND total_features >= 15
)
AND feature_value IS NOT NULL;

-- Customer segmentation using subquery filters
SELECT 
    c.customer_id,
    c.age,
    c.income,
    c.registration_date,
    'HIGH_VALUE' as customer_segment
FROM customers c
WHERE c.customer_id IN (
    -- High-value customers based on CLV
    SELECT customer_id
    FROM customer_lifetime_value
    WHERE predicted_clv > (
        SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY predicted_clv)
        FROM customer_lifetime_value
    )
)
AND c.customer_id NOT IN (
    -- Exclude churned customers
    SELECT customer_id
    FROM churned_customers
    WHERE churn_date >= CURRENT_DATE - INTERVAL '12 months'
)
AND c.customer_id IN (
    -- Include only active customers
    SELECT DISTINCT customer_id
    FROM recent_transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
);

-- Feature engineering with existence checks
SELECT 
    t.customer_id,
    t.transaction_date,
    t.amount,
    -- Binary features based on customer history
    CASE 
        WHEN t.customer_id IN (
            SELECT customer_id FROM premium_customers
        ) THEN 1 ELSE 0 
    END as is_premium_customer,
    
    CASE 
        WHEN t.customer_id IN (
            SELECT customer_id 
            FROM complaint_history 
            WHERE complaint_date >= CURRENT_DATE - INTERVAL '6 months'
        ) THEN 1 ELSE 0 
    END as has_recent_complaints,
    
    CASE 
        WHEN t.customer_id IN (
            SELECT customer_id 
            FROM loyalty_program_members
            WHERE status = 'ACTIVE'
        ) THEN 1 ELSE 0 
    END as is_loyalty_member
FROM transactions t
WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '12 months';
```

#### 2. Data Quality and Validation

```sql
-- Data quality assessment using column subqueries
SELECT 
    table_name,
    column_name,
    data_type,
    -- Completeness metrics
    total_records - null_count as complete_records,
    ROUND((total_records - null_count)::NUMERIC / total_records * 100, 2) as completeness_percentage,
    -- Quality flags
    CASE 
        WHEN column_name IN (
            SELECT column_name 
            FROM critical_features 
            WHERE model_type = 'CLASSIFICATION'
        ) AND (total_records - null_count)::NUMERIC / total_records < 0.95 
        THEN 'CRITICAL_MISSING_DATA'
        ELSE 'ACCEPTABLE'
    END as quality_status
FROM data_quality_summary
WHERE table_name IN (
    -- Focus on ML-relevant tables
    SELECT table_name 
    FROM ml_data_sources 
    WHERE is_active = TRUE
)
AND column_name NOT IN (
    -- Exclude known optional columns
    SELECT column_name 
    FROM optional_features
);
```

### Table Subqueries - Complex Result Sets

#### 1. Feature Engineering with Derived Tables

```sql
-- Complex feature engineering using table subqueries
SELECT 
    main.customer_id,
    main.base_features,
    rolling_stats.avg_monthly_spending,
    rolling_stats.spending_trend,
    seasonal_patterns.seasonal_index,
    peer_comparison.peer_group,
    peer_comparison.relative_rank
FROM (
    -- Base customer features
    SELECT 
        customer_id,
        jsonb_build_object(
            'age', age,
            'income', income,
            'tenure_months', EXTRACT(MONTHS FROM AGE(CURRENT_DATE, registration_date))
        ) as base_features
    FROM customers
    WHERE active_status = TRUE
) main
JOIN (
    -- Rolling statistics subquery
    SELECT 
        customer_id,
        AVG(monthly_spending) as avg_monthly_spending,
        CASE 
            WHEN CORR(month_number, monthly_spending) > 0.3 THEN 'INCREASING'
            WHEN CORR(month_number, monthly_spending) < -0.3 THEN 'DECREASING'
            ELSE 'STABLE'
        END as spending_trend
    FROM (
        SELECT 
            customer_id,
            DATE_TRUNC('month', transaction_date) as month_date,
            ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY DATE_TRUNC('month', transaction_date)) as month_number,
            SUM(amount) as monthly_spending
        FROM transactions
        WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
        GROUP BY customer_id, DATE_TRUNC('month', transaction_date)
    ) monthly_data
    GROUP BY customer_id
    HAVING COUNT(*) >= 6  -- At least 6 months of data
) rolling_stats ON main.customer_id = rolling_stats.customer_id
JOIN (
    -- Seasonal pattern analysis
    SELECT 
        customer_id,
        AVG(CASE WHEN month_num IN (11, 12, 1) THEN spending_ratio ELSE NULL END) as seasonal_index
    FROM (
        SELECT 
            customer_id,
            EXTRACT(MONTH FROM transaction_date) as month_num,
            SUM(amount) / AVG(SUM(amount)) OVER (PARTITION BY customer_id) as spending_ratio
        FROM transactions
        WHERE transaction_date >= CURRENT_DATE - INTERVAL '24 months'
        GROUP BY customer_id, EXTRACT(MONTH FROM transaction_date)
    ) seasonal_data
    GROUP BY customer_id
) seasonal_patterns ON main.customer_id = seasonal_patterns.customer_id
JOIN (
    -- Peer group comparison
    SELECT 
        customer_id,
        peer_group,
        PERCENT_RANK() OVER (PARTITION BY peer_group ORDER BY total_clv) as relative_rank
    FROM (
        SELECT 
            c.customer_id,
            CASE 
                WHEN c.age < 30 THEN 'YOUNG'
                WHEN c.age < 50 THEN 'MIDDLE'
                ELSE 'SENIOR'
            END || '_' || 
            CASE 
                WHEN c.income < 50000 THEN 'LOW_INCOME'
                WHEN c.income < 100000 THEN 'MID_INCOME'
                ELSE 'HIGH_INCOME'
            END as peer_group,
            clv.predicted_clv as total_clv
        FROM customers c
        JOIN customer_lifetime_value clv ON c.customer_id = clv.customer_id
    ) peer_groups
) peer_comparison ON main.customer_id = peer_comparison.customer_id;
```

#### 2. Model Training Data Preparation

```sql
-- Comprehensive ML dataset creation using table subqueries
CREATE TABLE ml_training_dataset AS
WITH feature_base AS (
    -- Base features subquery
    SELECT 
        customer_id,
        age,
        income,
        tenure_days,
        account_type
    FROM customer_demographics
    WHERE data_quality_score >= 0.9
),
behavioral_features AS (
    -- Behavioral aggregations
    SELECT 
        customer_id,
        COUNT(*) as transaction_count_12m,
        AVG(amount) as avg_transaction_amount,
        STDDEV(amount) as transaction_amount_stddev,
        COUNT(DISTINCT product_category) as category_diversity,
        MAX(transaction_date) as last_transaction_date
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY customer_id
    HAVING COUNT(*) >= 5  -- Minimum activity requirement
),
engagement_features AS (
    -- Digital engagement metrics
    SELECT 
        customer_id,
        COUNT(*) as login_count_90d,
        AVG(session_duration) as avg_session_duration,
        COUNT(DISTINCT DATE(login_timestamp)) as active_days_90d
    FROM user_sessions
    WHERE login_timestamp >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY customer_id
),
target_variable AS (
    -- Target variable calculation
    SELECT 
        customer_id,
        CASE 
            WHEN churn_date IS NOT NULL AND churn_date <= CURRENT_DATE + INTERVAL '30 days' THEN 1
            ELSE 0
        END as will_churn_30d
    FROM customer_churn_predictions
)
SELECT 
    fb.customer_id,
    -- Demographic features
    fb.age,
    fb.income,
    fb.tenure_days,
    fb.account_type,
    -- Behavioral features
    COALESCE(bf.transaction_count_12m, 0) as transaction_count,
    COALESCE(bf.avg_transaction_amount, 0) as avg_transaction_amount,
    COALESCE(bf.transaction_amount_stddev, 0) as transaction_variability,
    COALESCE(bf.category_diversity, 0) as product_diversity,
    EXTRACT(DAYS FROM CURRENT_DATE - bf.last_transaction_date) as days_since_last_transaction,
    -- Engagement features
    COALESCE(ef.login_count_90d, 0) as recent_logins,
    COALESCE(ef.avg_session_duration, 0) as avg_session_duration,
    COALESCE(ef.active_days_90d, 0) as active_days,
    -- Target variable
    tv.will_churn_30d as target
FROM feature_base fb
LEFT JOIN behavioral_features bf ON fb.customer_id = bf.customer_id
LEFT JOIN engagement_features ef ON fb.customer_id = ef.customer_id
JOIN target_variable tv ON fb.customer_id = tv.customer_id
WHERE tv.will_churn_30d IS NOT NULL;  -- Complete target data only
```

### EXISTS and NOT EXISTS Subqueries

#### 1. Complex Filtering Logic

```sql
-- Advanced customer filtering using EXISTS
SELECT 
    c.customer_id,
    c.customer_name,
    c.registration_date,
    c.customer_tier
FROM customers c
WHERE EXISTS (
    -- Has made a purchase in the last 90 days
    SELECT 1 
    FROM transactions t 
    WHERE t.customer_id = c.customer_id 
    AND t.transaction_date >= CURRENT_DATE - INTERVAL '90 days'
    AND t.amount > 0
)
AND EXISTS (
    -- Has interacted with customer service
    SELECT 1 
    FROM customer_interactions ci 
    WHERE ci.customer_id = c.customer_id 
    AND ci.interaction_type IN ('SUPPORT', 'COMPLAINT', 'INQUIRY')
    AND ci.interaction_date >= CURRENT_DATE - INTERVAL '180 days'
)
AND NOT EXISTS (
    -- Has not been flagged for fraud
    SELECT 1 
    FROM fraud_alerts fa 
    WHERE fa.customer_id = c.customer_id 
    AND fa.alert_date >= CURRENT_DATE - INTERVAL '12 months'
    AND fa.alert_status = 'CONFIRMED'
)
AND EXISTS (
    -- Has sufficient feature data quality
    SELECT 1 
    FROM customer_feature_completeness cfc 
    WHERE cfc.customer_id = c.customer_id 
    AND cfc.completeness_score >= 0.8
    AND cfc.last_updated >= CURRENT_DATE - INTERVAL '7 days'
);

-- Model eligibility assessment using EXISTS patterns
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    'ELIGIBLE_FOR_RECOMMENDATION_MODEL' as model_eligibility
FROM products p
WHERE EXISTS (
    -- Product has sufficient sales history
    SELECT 1 
    FROM order_items oi 
    JOIN orders o ON oi.order_id = o.order_id
    WHERE oi.product_id = p.product_id 
    AND o.order_date >= CURRENT_DATE - INTERVAL '6 months'
    GROUP BY oi.product_id
    HAVING COUNT(DISTINCT o.customer_id) >= 20  -- At least 20 unique customers
)
AND EXISTS (
    -- Product has review data
    SELECT 1 
    FROM product_reviews pr 
    WHERE pr.product_id = p.product_id 
    AND pr.review_date >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY pr.product_id
    HAVING COUNT(*) >= 5 AND AVG(rating) >= 3.0  -- Minimum reviews and rating
)
AND NOT EXISTS (
    -- Product is not discontinued
    SELECT 1 
    FROM discontinued_products dp 
    WHERE dp.product_id = p.product_id
)
AND p.active_status = TRUE;
```

#### 2. Data Validation and Quality Checks

```sql
-- Data quality validation using NOT EXISTS
SELECT 
    'ORPHANED_TRANSACTIONS' as issue_type,
    COUNT(*) as issue_count,
    'Transactions without corresponding customer records' as description
FROM transactions t
WHERE NOT EXISTS (
    SELECT 1 
    FROM customers c 
    WHERE c.customer_id = t.customer_id
)
AND t.transaction_date >= CURRENT_DATE - INTERVAL '30 days'

UNION ALL

SELECT 
    'INCOMPLETE_FEATURES' as issue_type,
    COUNT(*) as issue_count,
    'Customers missing critical feature data' as description
FROM customers c
WHERE c.active_status = TRUE
AND NOT EXISTS (
    SELECT 1 
    FROM customer_features cf 
    WHERE cf.customer_id = c.customer_id 
    AND cf.feature_completeness >= 0.8
    AND cf.last_updated >= CURRENT_DATE - INTERVAL '7 days'
)

UNION ALL

SELECT 
    'STALE_PREDICTIONS' as issue_type,
    COUNT(*) as issue_count,
    'Customers without recent model predictions' as description
FROM customers c
WHERE c.active_status = TRUE
AND EXISTS (
    SELECT 1 
    FROM transactions t 
    WHERE t.customer_id = c.customer_id 
    AND t.transaction_date >= CURRENT_DATE - INTERVAL '30 days'
)
AND NOT EXISTS (
    SELECT 1 
    FROM model_predictions mp 
    WHERE mp.customer_id = c.customer_id 
    AND mp.prediction_date >= CURRENT_DATE - INTERVAL '7 days'
);
```

### Performance Optimization for Subqueries

#### 1. Correlated vs Non-Correlated Subqueries

```sql
-- Optimized non-correlated subquery (better performance)
SELECT 
    customer_id,
    transaction_amount,
    CASE 
        WHEN transaction_amount > high_value_threshold.threshold THEN 'HIGH_VALUE'
        ELSE 'NORMAL_VALUE'
    END as transaction_category
FROM transactions t
CROSS JOIN (
    SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY transaction_amount) as threshold
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
) high_value_threshold
WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '7 days';

-- Convert correlated subquery to window function (better performance)
SELECT 
    customer_id,
    transaction_date,
    transaction_amount,
    -- Instead of correlated subquery, use window function
    transaction_amount - AVG(transaction_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as amount_vs_30day_avg,
    
    ROW_NUMBER() OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date
    ) as transaction_sequence_number
FROM transactions
WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
ORDER BY customer_id, transaction_date;

-- Materialized subquery results for better performance
CREATE MATERIALIZED VIEW customer_statistics AS
SELECT 
    customer_id,
    AVG(transaction_amount) as avg_amount,
    STDDEV(transaction_amount) as stddev_amount,
    COUNT(*) as transaction_count,
    MAX(transaction_date) as last_transaction_date
FROM transactions
WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY customer_id;

-- Use materialized view instead of repeated subqueries
SELECT 
    t.customer_id,
    t.transaction_amount,
    (t.transaction_amount - cs.avg_amount) / NULLIF(cs.stddev_amount, 0) as z_score,
    cs.transaction_count as customer_transaction_count
FROM transactions t
JOIN customer_statistics cs ON t.customer_id = cs.customer_id
WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '7 days';
```

### Real-World Applications

1. **Feature Engineering**: Complex derived features using statistical subqueries
2. **Data Filtering**: Multi-level filtering for training data preparation
3. **Anomaly Detection**: Dynamic threshold calculations using historical data
4. **Customer Segmentation**: Sophisticated grouping based on behavior patterns
5. **Model Validation**: Data quality checks and completeness assessment

### Best Practices

1. **Use Window Functions**: Replace correlated subqueries when possible
2. **Materialize Complex Subqueries**: Cache frequently used calculations
3. **Index Appropriately**: Ensure subquery joins can use indexes
4. **Limit Subquery Scope**: Use WHERE clauses in subqueries
5. **Consider CTEs**: Use Common Table Expressions for readability

### Common Pitfalls

1. **N+1 Problem**: Correlated subqueries executing for each row
2. **Missing Indexes**: Subqueries without proper index support
3. **Cartesian Products**: Unintended joins in subqueries
4. **NULL Handling**: Unexpected NULL behavior in subquery results
5. **Performance Issues**: Complex nested subqueries without optimization

### Performance Considerations

- **Correlated Subqueries**: Can be expensive, execute once per outer row
- **Non-Correlated Subqueries**: Execute once, generally more efficient
- **EXISTS vs IN**: EXISTS often performs better with NULL values
- **Subquery vs JOIN**: JOINs often perform better than subqueries
- **Materialized Views**: Cache complex subquery results for repeated use

---

## Question 4

**Can you explain the use of indexes in databases and how they relate to Machine Learning?**

**Answer:**

### Theory

Database indexes are data structures that improve query performance by creating fast access paths to table data. In machine learning contexts, indexes are crucial for efficient feature engineering, large-scale data processing, model training data retrieval, and real-time inference operations. Proper indexing strategy can dramatically reduce query execution time from hours to seconds, enabling practical ML workflows on large datasets.

**Index Types:**
- **B-Tree Indexes**: Most common, optimal for equality and range queries
- **Hash Indexes**: Fast equality lookups
- **Bitmap Indexes**: Efficient for low-cardinality columns
- **Partial Indexes**: Index subset of rows meeting conditions
- **Composite Indexes**: Multi-column indexes for complex queries
- **Functional Indexes**: Index expressions or function results

### B-Tree Indexes for ML Feature Engineering

#### 1. Customer Analytics and Segmentation

```sql
-- Customer dimension table with strategic indexes
CREATE TABLE customers (
    customer_id BIGINT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    registration_date DATE NOT NULL,
    birth_date DATE,
    country_code CHAR(2) NOT NULL,
    city VARCHAR(100),
    income_bracket VARCHAR(20),
    customer_tier VARCHAR(20) DEFAULT 'BRONZE',
    lifetime_value DECIMAL(12,2) DEFAULT 0.00,
    churn_risk_score DECIMAL(5,4),
    last_activity_date DATE,
    account_status VARCHAR(20) DEFAULT 'ACTIVE',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Strategic B-Tree indexes for ML feature queries
CREATE INDEX idx_customers_registration_date ON customers(registration_date);
CREATE INDEX idx_customers_country_income ON customers(country_code, income_bracket);
CREATE INDEX idx_customers_tier_clv ON customers(customer_tier, lifetime_value);
CREATE INDEX idx_customers_churn_risk ON customers(churn_risk_score) WHERE churn_risk_score IS NOT NULL;
CREATE INDEX idx_customers_activity_date ON customers(last_activity_date) WHERE last_activity_date IS NOT NULL;

-- Demonstrate index usage for ML feature engineering
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    country_code,
    customer_tier,
    COUNT(*) as customer_count,
    AVG(lifetime_value) as avg_clv,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lifetime_value) as median_clv,
    AVG(churn_risk_score) as avg_churn_risk,
    COUNT(*) FILTER (WHERE churn_risk_score > 0.7) as high_risk_customers,
    COUNT(*) FILTER (WHERE last_activity_date >= CURRENT_DATE - INTERVAL '30 days') as active_30d
FROM customers
WHERE country_code IN ('US', 'CA', 'GB', 'DE', 'FR')
  AND customer_tier IN ('GOLD', 'PLATINUM')
  AND account_status = 'ACTIVE'
GROUP BY country_code, customer_tier
ORDER BY country_code, customer_tier;

-- Complex feature engineering query leveraging multiple indexes
WITH customer_segments AS (
    SELECT 
        customer_id,
        country_code,
        customer_tier,
        lifetime_value,
        churn_risk_score,
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date)) as age,
        EXTRACT(DAYS FROM AGE(CURRENT_DATE, registration_date)) as tenure_days,
        EXTRACT(DAYS FROM AGE(CURRENT_DATE, last_activity_date)) as days_since_activity
    FROM customers
    WHERE account_status = 'ACTIVE'
      AND birth_date IS NOT NULL
      AND last_activity_date IS NOT NULL
      AND churn_risk_score IS NOT NULL
),
feature_engineering AS (
    SELECT 
        customer_id,
        country_code,
        customer_tier,
        -- Demographic features
        CASE 
            WHEN age < 25 THEN 'YOUNG'
            WHEN age < 40 THEN 'MIDDLE'
            WHEN age < 60 THEN 'MATURE'
            ELSE 'SENIOR'
        END as age_group,
        
        -- Tenure segmentation
        CASE 
            WHEN tenure_days < 30 THEN 'NEW'
            WHEN tenure_days < 365 THEN 'GROWING'
            WHEN tenure_days < 1095 THEN 'ESTABLISHED'
            ELSE 'VETERAN'
        END as tenure_segment,
        
        -- Value-based features
        CASE 
            WHEN lifetime_value < 100 THEN 'LOW_VALUE'
            WHEN lifetime_value < 1000 THEN 'MEDIUM_VALUE'
            WHEN lifetime_value < 5000 THEN 'HIGH_VALUE'
            ELSE 'PREMIUM_VALUE'
        END as value_segment,
        
        -- Risk-based features
        CASE 
            WHEN churn_risk_score < 0.3 THEN 'LOW_RISK'
            WHEN churn_risk_score < 0.7 THEN 'MEDIUM_RISK'
            ELSE 'HIGH_RISK'
        END as risk_segment,
        
        -- Engagement features
        CASE 
            WHEN days_since_activity <= 7 THEN 'HIGHLY_ACTIVE'
            WHEN days_since_activity <= 30 THEN 'ACTIVE'
            WHEN days_since_activity <= 90 THEN 'MODERATELY_ACTIVE'
            ELSE 'INACTIVE'
        END as activity_segment,
        
        -- Normalized features
        lifetime_value,
        churn_risk_score,
        age,
        tenure_days,
        days_since_activity
    FROM customer_segments
)
SELECT 
    country_code,
    age_group,
    tenure_segment,
    value_segment,
    risk_segment,
    activity_segment,
    COUNT(*) as segment_size,
    AVG(lifetime_value) as avg_clv,
    AVG(churn_risk_score) as avg_churn_risk,
    AVG(age) as avg_age,
    AVG(tenure_days) as avg_tenure_days
FROM feature_engineering
GROUP BY ROLLUP(
    country_code,
    age_group,
    tenure_segment,
    value_segment,
    risk_segment,
    activity_segment
)
ORDER BY country_code NULLS LAST, segment_size DESC;
```

#### 2. Transaction Analysis with Optimized Indexing

```sql
-- Transaction fact table with comprehensive indexing strategy
CREATE TABLE transactions (
    transaction_id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    transaction_date DATE NOT NULL,
    transaction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    amount DECIMAL(12,2) NOT NULL,
    currency_code CHAR(3) NOT NULL DEFAULT 'USD',
    merchant_id VARCHAR(50),
    merchant_category VARCHAR(20) NOT NULL,
    product_category VARCHAR(30),
    payment_method VARCHAR(20) NOT NULL,
    channel VARCHAR(20) NOT NULL,
    is_online BOOLEAN NOT NULL DEFAULT FALSE,
    transaction_status VARCHAR(20) NOT NULL DEFAULT 'COMPLETED',
    fraud_score DECIMAL(5,4),
    
    -- Add foreign key constraint
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Comprehensive indexing strategy for ML workloads
CREATE INDEX idx_trans_customer_date ON transactions(customer_id, transaction_date);
CREATE INDEX idx_trans_date_amount ON transactions(transaction_date, amount);
CREATE INDEX idx_trans_timestamp ON transactions(transaction_timestamp);
CREATE INDEX idx_trans_merchant_category ON transactions(merchant_category, product_category);
CREATE INDEX idx_trans_payment_channel ON transactions(payment_method, channel);
CREATE INDEX idx_trans_fraud_score ON transactions(fraud_score) WHERE fraud_score > 0.1;
CREATE INDEX idx_trans_high_value ON transactions(amount) WHERE amount > 1000;
CREATE INDEX idx_trans_recent ON transactions(transaction_date) WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days';

-- Partial index for online transactions (specific ML use case)
CREATE INDEX idx_trans_online_recent ON transactions(customer_id, transaction_timestamp) 
WHERE is_online = TRUE 
  AND transaction_date >= CURRENT_DATE - INTERVAL '180 days';

-- Functional index for ML feature engineering
CREATE INDEX idx_trans_hour_of_day ON transactions(EXTRACT(HOUR FROM transaction_timestamp));
CREATE INDEX idx_trans_day_of_week ON transactions(EXTRACT(DOW FROM transaction_timestamp));
CREATE INDEX idx_trans_amount_log ON transactions(LOG(amount + 1)) WHERE amount > 0;

-- Complex ML feature query leveraging indexes
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
WITH customer_transaction_features AS (
    SELECT 
        customer_id,
        -- Temporal features using functional indexes
        EXTRACT(HOUR FROM transaction_timestamp) as hour_of_day,
        EXTRACT(DOW FROM transaction_timestamp) as day_of_week,
        EXTRACT(MONTH FROM transaction_date) as month,
        
        -- Amount-based features
        amount,
        LOG(amount + 1) as log_amount,
        
        -- Categorical features
        merchant_category,
        product_category,
        payment_method,
        channel,
        is_online,
        
        -- Risk features
        fraud_score,
        
        -- Derived features
        CASE 
            WHEN EXTRACT(HOUR FROM transaction_timestamp) BETWEEN 22 AND 6 THEN TRUE
            ELSE FALSE
        END as is_night_transaction,
        
        CASE 
            WHEN EXTRACT(DOW FROM transaction_timestamp) IN (0, 6) THEN TRUE
            ELSE FALSE
        END as is_weekend_transaction,
        
        -- Date-based features
        transaction_date,
        transaction_timestamp
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
      AND transaction_status = 'COMPLETED'
      AND amount > 0
),
aggregated_features AS (
    SELECT 
        customer_id,
        -- Transaction count features
        COUNT(*) as total_transactions,
        COUNT(*) FILTER (WHERE is_online) as online_transactions,
        COUNT(*) FILTER (WHERE is_night_transaction) as night_transactions,
        COUNT(*) FILTER (WHERE is_weekend_transaction) as weekend_transactions,
        COUNT(DISTINCT merchant_category) as unique_merchant_categories,
        COUNT(DISTINCT payment_method) as unique_payment_methods,
        
        -- Amount-based features
        SUM(amount) as total_amount,
        AVG(amount) as avg_amount,
        STDDEV(amount) as stddev_amount,
        MIN(amount) as min_amount,
        MAX(amount) as max_amount,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) as median_amount,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) as p75_amount,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) as p95_amount,
        
        -- Frequency features
        COUNT(DISTINCT transaction_date) as active_days,
        COUNT(*) / GREATEST(COUNT(DISTINCT transaction_date), 1) as avg_transactions_per_day,
        
        -- Risk features
        AVG(fraud_score) FILTER (WHERE fraud_score IS NOT NULL) as avg_fraud_score,
        COUNT(*) FILTER (WHERE fraud_score > 0.5) as high_risk_transactions,
        
        -- Temporal pattern features
        AVG(hour_of_day) as avg_hour_of_day,
        MODE() WITHIN GROUP (ORDER BY day_of_week) as most_common_day,
        
        -- Channel preferences
        COUNT(*) FILTER (WHERE channel = 'WEB') * 1.0 / COUNT(*) as web_ratio,
        COUNT(*) FILTER (WHERE channel = 'MOBILE') * 1.0 / COUNT(*) as mobile_ratio,
        COUNT(*) FILTER (WHERE is_online) * 1.0 / COUNT(*) as online_ratio,
        
        -- Date range features
        MIN(transaction_date) as first_transaction_date,
        MAX(transaction_date) as last_transaction_date,
        MAX(transaction_date) - MIN(transaction_date) as customer_lifespan_days
    FROM customer_transaction_features
    GROUP BY customer_id
    HAVING COUNT(*) >= 5  -- Minimum transaction threshold
)
SELECT 
    af.customer_id,
    c.country_code,
    c.customer_tier,
    c.lifetime_value,
    c.churn_risk_score,
    
    -- Transaction behavior features
    af.total_transactions,
    af.total_amount,
    af.avg_amount,
    af.stddev_amount,
    af.median_amount,
    af.unique_merchant_categories,
    af.unique_payment_methods,
    
    -- Engagement features
    af.active_days,
    af.avg_transactions_per_day,
    af.customer_lifespan_days,
    
    -- Risk features
    af.avg_fraud_score,
    af.high_risk_transactions,
    
    -- Behavioral patterns
    af.online_ratio,
    af.night_transactions * 1.0 / af.total_transactions as night_transaction_ratio,
    af.weekend_transactions * 1.0 / af.total_transactions as weekend_transaction_ratio,
    
    -- Channel preferences
    af.web_ratio,
    af.mobile_ratio,
    
    -- Temporal features
    af.avg_hour_of_day,
    af.most_common_day,
    
    -- Derived features for ML
    CASE 
        WHEN af.stddev_amount / NULLIF(af.avg_amount, 0) > 2 THEN 'HIGH_VARIABILITY'
        WHEN af.stddev_amount / NULLIF(af.avg_amount, 0) > 1 THEN 'MEDIUM_VARIABILITY'
        ELSE 'LOW_VARIABILITY'
    END as spending_variability,
    
    CASE 
        WHEN af.avg_transactions_per_day > 3 THEN 'HIGH_FREQUENCY'
        WHEN af.avg_transactions_per_day > 1 THEN 'MEDIUM_FREQUENCY'
        ELSE 'LOW_FREQUENCY'
    END as transaction_frequency_category
    
FROM aggregated_features af
JOIN customers c ON af.customer_id = c.customer_id
WHERE c.account_status = 'ACTIVE'
ORDER BY af.total_amount DESC;
```

### Composite Indexes for Complex ML Queries

#### 1. Multi-Dimensional Feature Engineering

```sql
-- Create composite indexes for complex filtering and sorting
CREATE INDEX idx_customer_multi_dim ON customers(
    country_code, 
    customer_tier, 
    account_status, 
    registration_date
);

CREATE INDEX idx_transaction_complex ON transactions(
    customer_id, 
    transaction_date, 
    merchant_category, 
    amount
);

-- Advanced time-series feature engineering using composite indexes
WITH daily_customer_metrics AS (
    SELECT 
        t.customer_id,
        t.transaction_date,
        COUNT(*) as daily_transaction_count,
        SUM(t.amount) as daily_amount,
        AVG(t.amount) as daily_avg_amount,
        COUNT(DISTINCT t.merchant_category) as daily_category_diversity,
        COUNT(*) FILTER (WHERE t.is_online) as daily_online_count,
        COUNT(*) FILTER (WHERE t.fraud_score > 0.5) as daily_risky_transactions,
        
        -- Intraday features
        COUNT(*) FILTER (
            WHERE EXTRACT(HOUR FROM t.transaction_timestamp) BETWEEN 9 AND 17
        ) as business_hours_transactions,
        
        COUNT(*) FILTER (
            WHERE EXTRACT(HOUR FROM t.transaction_timestamp) BETWEEN 18 AND 23
        ) as evening_transactions,
        
        -- Amount distribution features
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.amount) as daily_median_amount,
        MAX(t.amount) as daily_max_amount,
        STDDEV(t.amount) as daily_amount_stddev
    FROM transactions t
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '90 days'
      AND t.transaction_status = 'COMPLETED'
    GROUP BY t.customer_id, t.transaction_date
),
rolling_window_features AS (
    SELECT 
        customer_id,
        transaction_date,
        daily_transaction_count,
        daily_amount,
        daily_avg_amount,
        
        -- 7-day rolling window features
        AVG(daily_transaction_count) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as avg_7d_transaction_count,
        
        AVG(daily_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as avg_7d_amount,
        
        STDDEV(daily_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as stddev_7d_amount,
        
        -- 30-day rolling window features
        AVG(daily_transaction_count) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as avg_30d_transaction_count,
        
        AVG(daily_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as avg_30d_amount,
        
        -- Trend detection using linear regression slope
        REGR_SLOPE(daily_amount, EXTRACT(EPOCH FROM transaction_date)) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) as amount_trend_14d,
        
        -- Volatility measures
        STDDEV(daily_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) / NULLIF(AVG(daily_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ), 0) as coefficient_of_variation_30d,
        
        -- Activity consistency features
        COUNT(*) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as active_days_7d,
        
        -- Lag features
        LAG(daily_amount, 1) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date
        ) as prev_day_amount,
        
        LAG(daily_amount, 7) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date
        ) as week_ago_amount
    FROM daily_customer_metrics
)
SELECT 
    rwf.customer_id,
    rwf.transaction_date,
    c.country_code,
    c.customer_tier,
    
    -- Base daily features
    rwf.daily_transaction_count,
    rwf.daily_amount,
    rwf.daily_avg_amount,
    
    -- Rolling window features
    rwf.avg_7d_transaction_count,
    rwf.avg_7d_amount,
    rwf.avg_30d_transaction_count,
    rwf.avg_30d_amount,
    rwf.stddev_7d_amount,
    
    -- Trend and volatility features
    rwf.amount_trend_14d,
    rwf.coefficient_of_variation_30d,
    rwf.active_days_7d,
    
    -- Change features
    CASE 
        WHEN rwf.prev_day_amount > 0 THEN 
            (rwf.daily_amount - rwf.prev_day_amount) / rwf.prev_day_amount
        ELSE NULL
    END as daily_amount_change_pct,
    
    CASE 
        WHEN rwf.week_ago_amount > 0 THEN 
            (rwf.daily_amount - rwf.week_ago_amount) / rwf.week_ago_amount
        ELSE NULL
    END as weekly_amount_change_pct,
    
    -- Deviation from personal average
    CASE 
        WHEN rwf.avg_30d_amount > 0 THEN 
            (rwf.daily_amount - rwf.avg_30d_amount) / rwf.avg_30d_amount
        ELSE NULL
    END as deviation_from_30d_avg,
    
    -- Anomaly detection features
    CASE 
        WHEN ABS(rwf.daily_amount - rwf.avg_30d_amount) > 3 * rwf.stddev_7d_amount 
        THEN TRUE
        ELSE FALSE
    END as is_amount_anomaly,
    
    -- Trend categorization
    CASE 
        WHEN rwf.amount_trend_14d > 0.1 THEN 'INCREASING'
        WHEN rwf.amount_trend_14d < -0.1 THEN 'DECREASING'
        ELSE 'STABLE'
    END as spending_trend_category,
    
    -- Activity level classification
    CASE 
        WHEN rwf.active_days_7d >= 6 THEN 'DAILY_ACTIVE'
        WHEN rwf.active_days_7d >= 4 THEN 'REGULAR_ACTIVE'
        WHEN rwf.active_days_7d >= 2 THEN 'OCCASIONAL_ACTIVE'
        ELSE 'INACTIVE'
    END as activity_level
    
FROM rolling_window_features rwf
JOIN customers c ON rwf.customer_id = c.customer_id
WHERE rwf.transaction_date >= CURRENT_DATE - INTERVAL '30 days'
  AND c.account_status = 'ACTIVE'
ORDER BY rwf.customer_id, rwf.transaction_date;
```

### Specialized Indexes for ML Workloads

#### 1. Partial Indexes for Specific ML Use Cases

```sql
-- Partial indexes for active, high-value customers
CREATE INDEX idx_customers_high_value_active ON customers(customer_id, lifetime_value) 
WHERE account_status = 'ACTIVE' 
  AND lifetime_value >= 1000 
  AND churn_risk_score IS NOT NULL;

-- Partial index for recent high-risk transactions
CREATE INDEX idx_transactions_recent_high_risk ON transactions(
    customer_id, 
    transaction_timestamp,
    amount
) WHERE transaction_date >= CURRENT_DATE - INTERVAL '30 days'
  AND fraud_score > 0.7;

-- Partial index for large online transactions
CREATE INDEX idx_transactions_large_online ON transactions(
    customer_id,
    transaction_date,
    amount,
    merchant_category
) WHERE is_online = TRUE 
  AND amount >= 500
  AND transaction_status = 'COMPLETED';

-- Functional indexes for ML feature engineering
CREATE INDEX idx_customer_age_computed ON customers(
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date))
) WHERE birth_date IS NOT NULL;

CREATE INDEX idx_customer_tenure_months ON customers(
    EXTRACT(MONTHS FROM AGE(CURRENT_DATE, registration_date))
);

-- Expression index for log-transformed amounts
CREATE INDEX idx_transaction_log_amount ON transactions(LOG(amount + 1)) 
WHERE amount > 0;

-- Example query leveraging partial indexes
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    c.customer_id,
    c.customer_tier,
    c.lifetime_value,
    c.churn_risk_score,
    
    -- Age feature using functional index
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) as age,
    
    -- Tenure feature using functional index
    EXTRACT(MONTHS FROM AGE(CURRENT_DATE, c.registration_date)) as tenure_months,
    
    -- Recent transaction features
    COUNT(t.transaction_id) as recent_transaction_count,
    SUM(t.amount) as recent_total_amount,
    AVG(t.amount) as recent_avg_amount,
    COUNT(DISTINCT t.merchant_category) as recent_category_diversity,
    
    -- Risk features
    COUNT(*) FILTER (WHERE t.fraud_score > 0.5) as recent_risky_transactions,
    AVG(t.fraud_score) FILTER (WHERE t.fraud_score IS NOT NULL) as recent_avg_fraud_score,
    
    -- Log-transformed features
    AVG(LOG(t.amount + 1)) as recent_avg_log_amount
    
FROM customers c
LEFT JOIN transactions t ON c.customer_id = t.customer_id 
    AND t.transaction_date >= CURRENT_DATE - INTERVAL '30 days'
    AND t.transaction_status = 'COMPLETED'
WHERE c.account_status = 'ACTIVE'
  AND c.lifetime_value >= 1000
  AND c.churn_risk_score IS NOT NULL
  AND c.birth_date IS NOT NULL
GROUP BY c.customer_id, c.customer_tier, c.lifetime_value, c.churn_risk_score, c.birth_date, c.registration_date
HAVING COUNT(t.transaction_id) > 0
ORDER BY c.lifetime_value DESC, c.churn_risk_score DESC;
```

#### 2. GIN and GiST Indexes for JSON and Array Data

```sql
-- Customer features table with JSON and array columns
CREATE TABLE customer_ml_features (
    customer_id BIGINT PRIMARY KEY REFERENCES customers(customer_id),
    feature_vector FLOAT8[] NOT NULL,
    feature_names TEXT[] NOT NULL,
    behavioral_features JSONB NOT NULL,
    demographic_features JSONB NOT NULL,
    temporal_features JSONB NOT NULL,
    model_scores JSONB,
    feature_importance JSONB,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure array consistency
    CONSTRAINT chk_feature_vector_consistency 
        CHECK (array_length(feature_vector, 1) = array_length(feature_names, 1))
);

-- GIN indexes for JSON columns (supports containment queries)
CREATE INDEX idx_customer_features_behavioral_gin ON customer_ml_features 
    USING GIN (behavioral_features);

CREATE INDEX idx_customer_features_demographic_gin ON customer_ml_features 
    USING GIN (demographic_features);

CREATE INDEX idx_customer_features_temporal_gin ON customer_ml_features 
    USING GIN (temporal_features);

-- GIN index for array columns
CREATE INDEX idx_customer_features_names_gin ON customer_ml_features 
    USING GIN (feature_names);

-- Specific JSON path indexes for common queries
CREATE INDEX idx_behavioral_spending_category ON customer_ml_features 
    USING GIN ((behavioral_features->'spending_category'));

CREATE INDEX idx_demographic_age_group ON customer_ml_features 
    USING GIN ((demographic_features->'age_group'));

-- Example queries using GIN indexes
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    customer_id,
    behavioral_features->'avg_transaction_amount' as avg_transaction_amount,
    behavioral_features->'transaction_frequency' as transaction_frequency,
    demographic_features->'age_group' as age_group,
    temporal_features->'most_active_hour' as most_active_hour
FROM customer_ml_features
WHERE behavioral_features @> '{"spending_category": "HIGH_SPENDER"}'
  AND demographic_features @> '{"age_group": "MIDDLE_AGE"}'
  AND temporal_features ? 'weekend_activity_ratio'
  AND 'avg_transaction_amount' = ANY(feature_names);

-- Complex JSON containment query for model selection
SELECT 
    cmf.customer_id,
    c.customer_tier,
    cmf.behavioral_features->'churn_probability' as churn_probability,
    cmf.model_scores->'classification_confidence' as classification_confidence
FROM customer_ml_features cmf
JOIN customers c ON cmf.customer_id = c.customer_id
WHERE cmf.behavioral_features @> '{"high_value_customer": true}'
  AND cmf.model_scores @> '{"model_version": "v2.1"}'
  AND (cmf.model_scores->'classification_confidence')::NUMERIC > 0.8
  AND c.account_status = 'ACTIVE'
ORDER BY (cmf.behavioral_features->'churn_probability')::NUMERIC DESC;
```

### Index Maintenance and Performance Monitoring

#### 1. Index Usage Analysis

```sql
-- Create function to analyze index usage for ML workloads
CREATE OR REPLACE FUNCTION analyze_ml_index_usage()
RETURNS TABLE(
    schemaname TEXT,
    tablename TEXT,
    indexname TEXT,
    num_scans BIGINT,
    tuples_read BIGINT,
    tuples_fetched BIGINT,
    index_size TEXT,
    table_size TEXT,
    usage_ratio NUMERIC,
    recommendation TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.schemaname::TEXT,
        s.tablename::TEXT,
        s.indexname::TEXT,
        s.idx_scan as num_scans,
        s.idx_tup_read as tuples_read,
        s.idx_tup_fetch as tuples_fetched,
        pg_size_pretty(pg_relation_size(s.indexrelid)) as index_size,
        pg_size_pretty(pg_relation_size(s.relid)) as table_size,
        CASE 
            WHEN s.idx_scan = 0 THEN 0
            ELSE ROUND(s.idx_tup_read::NUMERIC / s.idx_scan, 2)
        END as usage_ratio,
        CASE 
            WHEN s.idx_scan = 0 THEN 'UNUSED - Consider dropping'
            WHEN s.idx_scan < 100 AND pg_relation_size(s.indexrelid) > 1048576 THEN 'LOW_USAGE - Review necessity'
            WHEN s.idx_tup_read::NUMERIC / GREATEST(s.idx_scan, 1) > 1000 THEN 'HIGH_SELECTIVITY - Good performance'
            ELSE 'NORMAL_USAGE'
        END as recommendation
    FROM pg_stat_user_indexes s
    JOIN pg_index i ON s.indexrelid = i.indexrelid
    WHERE s.schemaname = 'public'
      AND (s.tablename LIKE '%customer%' OR s.tablename LIKE '%transaction%' OR s.tablename LIKE '%ml_%')
    ORDER BY s.idx_scan DESC, pg_relation_size(s.indexrelid) DESC;
END;
$$ LANGUAGE plpgsql;

-- Run index usage analysis
SELECT * FROM analyze_ml_index_usage();

-- Index bloat analysis for maintenance planning
WITH index_bloat_analysis AS (
    SELECT 
        schemaname,
        tablename,
        indexname,
        pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
        pg_relation_size(indexrelid) as index_size_bytes,
        CASE 
            WHEN idx_scan = 0 THEN 'UNUSED'
            WHEN idx_scan < 1000 THEN 'LOW_USAGE'
            WHEN idx_scan < 10000 THEN 'MEDIUM_USAGE'
            ELSE 'HIGH_USAGE'
        END as usage_category,
        idx_scan,
        idx_tup_read,
        idx_tup_fetch
    FROM pg_stat_user_indexes
    WHERE schemaname = 'public'
)
SELECT 
    *,
    CASE 
        WHEN usage_category = 'UNUSED' AND index_size_bytes > 1048576 THEN 'DROP_CANDIDATE'
        WHEN usage_category = 'LOW_USAGE' AND index_size_bytes > 10485760 THEN 'REVIEW_NEEDED'
        WHEN usage_category IN ('MEDIUM_USAGE', 'HIGH_USAGE') THEN 'KEEP'
        ELSE 'MONITOR'
    END as maintenance_action
FROM index_bloat_analysis
ORDER BY index_size_bytes DESC;
```

#### 2. Automated Index Recommendation System

```sql
-- Create table to track query patterns for index recommendations
CREATE TABLE query_performance_log (
    log_id BIGSERIAL PRIMARY KEY,
    query_hash VARCHAR(64) NOT NULL,
    query_text TEXT NOT NULL,
    execution_time_ms NUMERIC(10,3) NOT NULL,
    rows_examined BIGINT,
    rows_returned BIGINT,
    tables_accessed TEXT[],
    columns_accessed TEXT[],
    where_conditions TEXT[],
    order_by_columns TEXT[],
    group_by_columns TEXT[],
    join_columns TEXT[],
    execution_plan JSONB,
    timestamp_executed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Classification
    query_type VARCHAR(20), -- SELECT, INSERT, UPDATE, DELETE
    workload_type VARCHAR(30), -- FEATURE_ENGINEERING, MODEL_TRAINING, PREDICTION, ANALYTICS
    performance_category VARCHAR(20) -- FAST, MEDIUM, SLOW, VERY_SLOW
);

-- Function to suggest indexes based on query patterns
CREATE OR REPLACE FUNCTION suggest_ml_indexes()
RETURNS TABLE(
    suggested_index TEXT,
    justification TEXT,
    estimated_benefit TEXT,
    implementation_sql TEXT
) AS $$
BEGIN
    -- Suggest indexes for frequently accessed columns in WHERE clauses
    RETURN QUERY
    WITH frequent_where_columns AS (
        SELECT 
            unnest(where_conditions) as column_name,
            COUNT(*) as frequency,
            AVG(execution_time_ms) as avg_execution_time
        FROM query_performance_log
        WHERE timestamp_executed >= CURRENT_TIMESTAMP - INTERVAL '7 days'
          AND workload_type IN ('FEATURE_ENGINEERING', 'MODEL_TRAINING')
          AND performance_category IN ('SLOW', 'VERY_SLOW')
        GROUP BY unnest(where_conditions)
        HAVING COUNT(*) >= 10
    ),
    frequent_order_columns AS (
        SELECT 
            unnest(order_by_columns) as column_name,
            COUNT(*) as frequency,
            AVG(execution_time_ms) as avg_execution_time
        FROM query_performance_log
        WHERE timestamp_executed >= CURRENT_TIMESTAMP - INTERVAL '7 days'
          AND workload_type = 'FEATURE_ENGINEERING'
          AND performance_category IN ('SLOW', 'VERY_SLOW')
        GROUP BY unnest(order_by_columns)
        HAVING COUNT(*) >= 5
    )
    SELECT 
        'B-Tree Index on ' || fwc.column_name as suggested_index,
        'Frequently used in WHERE clauses (' || fwc.frequency || ' times) with avg execution time ' || 
        ROUND(fwc.avg_execution_time, 2) || 'ms' as justification,
        'Could improve query performance by 50-80%' as estimated_benefit,
        'CREATE INDEX idx_' || REPLACE(fwc.column_name, '.', '_') || ' ON ' || 
        SPLIT_PART(fwc.column_name, '.', 1) || '(' || SPLIT_PART(fwc.column_name, '.', 2) || ');' as implementation_sql
    FROM frequent_where_columns fwc
    WHERE fwc.avg_execution_time > 1000  -- Only suggest for slow queries
    
    UNION ALL
    
    SELECT 
        'B-Tree Index on ' || foc.column_name as suggested_index,
        'Frequently used in ORDER BY clauses (' || foc.frequency || ' times) with avg execution time ' || 
        ROUND(foc.avg_execution_time, 2) || 'ms' as justification,
        'Could improve sorting performance by 60-90%' as estimated_benefit,
        'CREATE INDEX idx_' || REPLACE(foc.column_name, '.', '_') || '_sort ON ' || 
        SPLIT_PART(foc.column_name, '.', 1) || '(' || SPLIT_PART(foc.column_name, '.', 2) || ');' as implementation_sql
    FROM frequent_order_columns foc
    WHERE foc.avg_execution_time > 2000;  -- Only suggest for very slow queries
END;
$$ LANGUAGE plpgsql;

-- Run index suggestions
SELECT * FROM suggest_ml_indexes();
```

### Real-World Applications

1. **Feature Engineering Acceleration**: Speed up complex aggregations and window functions
2. **Model Training Data Retrieval**: Fast access to training datasets with complex filters
3. **Real-time Inference**: Quick lookups for prediction serving
4. **A/B Testing**: Efficient customer segmentation and metric calculations
5. **Data Quality Monitoring**: Fast validation queries across large datasets

### Best Practices

1. **Index Selectivity**: Create indexes on columns with high selectivity
2. **Composite Index Order**: Place most selective columns first
3. **Partial Indexes**: Use for frequently filtered subsets of data
4. **Monitor Usage**: Regularly analyze index usage statistics
5. **Maintenance Windows**: Schedule index rebuilding during low-traffic periods

### Common Pitfalls

1. **Over-indexing**: Too many indexes slow down INSERT/UPDATE operations
2. **Wrong Column Order**: Incorrect composite index order reduces effectiveness
3. **Unused Indexes**: Maintaining indexes that are never used
4. **Index Bloat**: Large, fragmented indexes that need rebuilding
5. **Missing Statistics**: Outdated table statistics leading to poor query plans

### Performance Considerations

- **Index Size**: Large indexes may not fit in memory
- **Write Performance**: More indexes slow down data modifications
- **Query Complexity**: Complex queries may need multiple specialized indexes
- **Data Distribution**: Skewed data may reduce index effectiveness
- **Memory Usage**: Index caching strategy affects overall performance

---

## Question 5

**Explain the importance of data normalization in SQL and how it affects Machine Learning models.**

**Answer:**

### Theory

Data normalization in SQL refers to organizing database schema to eliminate redundancy and improve data integrity. In machine learning contexts, normalization has two critical aspects: **database normalization** (schema design) and **feature normalization** (data preprocessing). Both significantly impact ML model performance, training efficiency, and data quality. Proper normalization ensures clean, consistent data while enabling efficient feature engineering and model training workflows.

**Database Normalization Forms:**
- **1NF (First Normal Form)**: Atomic values, no repeating groups
- **2NF (Second Normal Form)**: 1NF + no partial dependencies
- **3NF (Third Normal Form)**: 2NF + no transitive dependencies
- **BCNF (Boyce-Codd Normal Form)**: 3NF + every determinant is a candidate key

**Feature Normalization Techniques:**
- **Min-Max Scaling**: Scale to [0,1] range
- **Z-Score Normalization**: Mean=0, standard deviation=1
- **Robust Scaling**: Use median and IQR
- **Unit Vector Scaling**: L2 normalization

### Database Normalization for ML Data Architecture

#### 1. Normalized Schema Design for ML Pipelines

```sql
-- Denormalized table (problematic for ML)
CREATE TABLE customer_transactions_denormalized (
    transaction_id BIGINT PRIMARY KEY,
    customer_id BIGINT,
    customer_name VARCHAR(200),
    customer_email VARCHAR(255),
    customer_age INTEGER,
    customer_country VARCHAR(50),
    customer_tier VARCHAR(20),
    customer_registration_date DATE,
    
    transaction_date DATE,
    transaction_amount DECIMAL(12,2),
    
    merchant_id VARCHAR(50),
    merchant_name VARCHAR(200),
    merchant_category VARCHAR(50),
    merchant_country VARCHAR(50),
    
    product_id VARCHAR(50),
    product_name VARCHAR(200),
    product_category VARCHAR(50),
    product_subcategory VARCHAR(50),
    product_price DECIMAL(12,2),
    
    -- Many other denormalized columns...
    payment_method VARCHAR(50),
    currency_code CHAR(3)
);

-- Problems with denormalized structure for ML:
-- 1. Data redundancy leads to inconsistencies
-- 2. Update anomalies affect multiple records
-- 3. Storage inefficiency
-- 4. Difficult feature engineering
-- 5. Poor data quality control

-- Properly normalized schema for ML
CREATE TABLE customers (
    customer_id BIGINT PRIMARY KEY,
    customer_name VARCHAR(200) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    birth_date DATE,
    registration_date DATE NOT NULL,
    country_code CHAR(2) NOT NULL,
    city VARCHAR(100),
    customer_tier VARCHAR(20) DEFAULT 'BRONZE',
    account_status VARCHAR(20) DEFAULT 'ACTIVE',
    lifetime_value DECIMAL(12,2) DEFAULT 0.00,
    churn_risk_score DECIMAL(5,4),
    last_activity_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT chk_customer_tier CHECK (customer_tier IN ('BRONZE', 'SILVER', 'GOLD', 'PLATINUM')),
    CONSTRAINT chk_account_status CHECK (account_status IN ('ACTIVE', 'SUSPENDED', 'CLOSED')),
    CONSTRAINT chk_churn_risk CHECK (churn_risk_score BETWEEN 0.0000 AND 1.0000)
);

CREATE TABLE merchants (
    merchant_id VARCHAR(50) PRIMARY KEY,
    merchant_name VARCHAR(200) NOT NULL,
    merchant_category VARCHAR(50) NOT NULL,
    merchant_subcategory VARCHAR(50),
    country_code CHAR(2) NOT NULL,
    city VARCHAR(100),
    is_online_only BOOLEAN DEFAULT FALSE,
    risk_rating VARCHAR(20) DEFAULT 'LOW',
    average_transaction_amount DECIMAL(12,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT chk_risk_rating CHECK (risk_rating IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'))
);

CREATE TABLE products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    product_category VARCHAR(50) NOT NULL,
    product_subcategory VARCHAR(50),
    base_price DECIMAL(12,2) NOT NULL,
    cost_price DECIMAL(12,2),
    margin_percent DECIMAL(5,2),
    is_active BOOLEAN DEFAULT TRUE,
    launch_date DATE,
    
    CONSTRAINT chk_base_price CHECK (base_price > 0),
    CONSTRAINT chk_margin CHECK (margin_percent BETWEEN 0.00 AND 100.00)
);

CREATE TABLE transactions (
    transaction_id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES customers(customer_id),
    merchant_id VARCHAR(50) NOT NULL REFERENCES merchants(merchant_id),
    product_id VARCHAR(50) REFERENCES products(product_id),
    
    transaction_date DATE NOT NULL,
    transaction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    amount DECIMAL(12,2) NOT NULL,
    quantity INTEGER DEFAULT 1,
    currency_code CHAR(3) NOT NULL DEFAULT 'USD',
    exchange_rate DECIMAL(10,6) DEFAULT 1.000000,
    amount_usd DECIMAL(12,2) GENERATED ALWAYS AS (amount * exchange_rate) STORED,
    
    payment_method VARCHAR(20) NOT NULL,
    channel VARCHAR(20) NOT NULL,
    transaction_type VARCHAR(20) NOT NULL,
    transaction_status VARCHAR(20) NOT NULL DEFAULT 'COMPLETED',
    
    -- ML-specific columns
    fraud_score DECIMAL(5,4),
    risk_flags TEXT[],
    feature_vector JSONB,
    
    CONSTRAINT chk_amount CHECK (amount > 0),
    CONSTRAINT chk_quantity CHECK (quantity > 0),
    CONSTRAINT chk_fraud_score CHECK (fraud_score BETWEEN 0.0000 AND 1.0000),
    CONSTRAINT chk_payment_method CHECK (payment_method IN ('CREDIT_CARD', 'DEBIT_CARD', 'BANK_TRANSFER', 'DIGITAL_WALLET', 'CASH')),
    CONSTRAINT chk_channel CHECK (channel IN ('WEB', 'MOBILE', 'STORE', 'PHONE', 'API')),
    CONSTRAINT chk_transaction_type CHECK (transaction_type IN ('PURCHASE', 'REFUND', 'ADJUSTMENT', 'FEE')),
    CONSTRAINT chk_transaction_status CHECK (transaction_status IN ('PENDING', 'COMPLETED', 'FAILED', 'CANCELLED'))
);

-- Create indexes for efficient ML queries
CREATE INDEX idx_customers_country_tier ON customers(country_code, customer_tier);
CREATE INDEX idx_customers_churn_risk ON customers(churn_risk_score) WHERE churn_risk_score IS NOT NULL;
CREATE INDEX idx_merchants_category ON merchants(merchant_category, merchant_subcategory);
CREATE INDEX idx_products_category ON products(product_category, is_active);
CREATE INDEX idx_transactions_customer_date ON transactions(customer_id, transaction_date);
CREATE INDEX idx_transactions_merchant_date ON transactions(merchant_id, transaction_date);
CREATE INDEX idx_transactions_amount_usd ON transactions(amount_usd);

-- Benefits of normalized structure for ML:
-- 1. Consistent data updates
-- 2. Efficient storage
-- 3. Better data quality
-- 4. Easier feature engineering
-- 5. Simplified maintenance
```

#### 2. Feature Engineering with Normalized Schema

```sql
-- Complex feature engineering leveraging normalized structure
WITH customer_demographics AS (
    SELECT 
        customer_id,
        customer_name,
        email,
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date)) as age,
        EXTRACT(DAYS FROM AGE(CURRENT_DATE, registration_date)) as tenure_days,
        country_code,
        customer_tier,
        lifetime_value,
        churn_risk_score,
        EXTRACT(DAYS FROM AGE(CURRENT_DATE, last_activity_date)) as days_since_last_activity
    FROM customers
    WHERE account_status = 'ACTIVE'
      AND birth_date IS NOT NULL
),
transaction_aggregates AS (
    SELECT 
        t.customer_id,
        
        -- Transaction count features
        COUNT(*) as total_transactions_12m,
        COUNT(DISTINCT t.merchant_id) as unique_merchants_12m,
        COUNT(DISTINCT t.product_id) as unique_products_12m,
        COUNT(DISTINCT m.merchant_category) as unique_merchant_categories_12m,
        COUNT(DISTINCT p.product_category) as unique_product_categories_12m,
        
        -- Amount-based features
        SUM(t.amount_usd) as total_amount_usd_12m,
        AVG(t.amount_usd) as avg_amount_usd_12m,
        STDDEV(t.amount_usd) as stddev_amount_usd_12m,
        MIN(t.amount_usd) as min_amount_usd_12m,
        MAX(t.amount_usd) as max_amount_usd_12m,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.amount_usd) as median_amount_usd_12m,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY t.amount_usd) as p95_amount_usd_12m,
        
        -- Frequency features
        COUNT(DISTINCT t.transaction_date) as active_days_12m,
        COUNT(*) / GREATEST(COUNT(DISTINCT t.transaction_date), 1) as avg_transactions_per_day,
        
        -- Channel and payment features
        COUNT(*) FILTER (WHERE t.channel = 'MOBILE') * 1.0 / COUNT(*) as mobile_ratio,
        COUNT(*) FILTER (WHERE t.channel = 'WEB') * 1.0 / COUNT(*) as web_ratio,
        COUNT(*) FILTER (WHERE t.payment_method = 'CREDIT_CARD') * 1.0 / COUNT(*) as credit_card_ratio,
        COUNT(*) FILTER (WHERE t.payment_method = 'DIGITAL_WALLET') * 1.0 / COUNT(*) as digital_wallet_ratio,
        
        -- Merchant risk features
        AVG(CASE m.risk_rating 
            WHEN 'LOW' THEN 1 
            WHEN 'MEDIUM' THEN 2 
            WHEN 'HIGH' THEN 3 
            WHEN 'VERY_HIGH' THEN 4 
            ELSE 1 
        END) as avg_merchant_risk_score,
        
        -- Product features
        AVG(p.margin_percent) FILTER (WHERE p.margin_percent IS NOT NULL) as avg_product_margin,
        COUNT(*) FILTER (WHERE p.product_category = 'ELECTRONICS') as electronics_purchases,
        COUNT(*) FILTER (WHERE p.product_category = 'FASHION') as fashion_purchases,
        
        -- Risk features
        AVG(t.fraud_score) FILTER (WHERE t.fraud_score IS NOT NULL) as avg_fraud_score,
        COUNT(*) FILTER (WHERE t.fraud_score > 0.5) as high_risk_transactions,
        
        -- Temporal features
        MIN(t.transaction_date) as first_transaction_date,
        MAX(t.transaction_date) as last_transaction_date
        
    FROM transactions t
    JOIN merchants m ON t.merchant_id = m.merchant_id
    LEFT JOIN products p ON t.product_id = p.product_id
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
      AND t.transaction_status = 'COMPLETED'
    GROUP BY t.customer_id
    HAVING COUNT(*) >= 3  -- Minimum transaction threshold
),
seasonal_features AS (
    SELECT 
        t.customer_id,
        
        -- Seasonal spending patterns
        AVG(t.amount_usd) FILTER (
            WHERE EXTRACT(MONTH FROM t.transaction_date) IN (12, 1, 2)
        ) as avg_winter_spending,
        
        AVG(t.amount_usd) FILTER (
            WHERE EXTRACT(MONTH FROM t.transaction_date) IN (3, 4, 5)
        ) as avg_spring_spending,
        
        AVG(t.amount_usd) FILTER (
            WHERE EXTRACT(MONTH FROM t.transaction_date) IN (6, 7, 8)
        ) as avg_summer_spending,
        
        AVG(t.amount_usd) FILTER (
            WHERE EXTRACT(MONTH FROM t.transaction_date) IN (9, 10, 11)
        ) as avg_fall_spending,
        
        -- Weekend vs weekday patterns
        COUNT(*) FILTER (
            WHERE EXTRACT(DOW FROM t.transaction_date) IN (0, 6)
        ) * 1.0 / COUNT(*) as weekend_transaction_ratio,
        
        -- Holiday spending (November-December)
        AVG(t.amount_usd) FILTER (
            WHERE EXTRACT(MONTH FROM t.transaction_date) IN (11, 12)
        ) as avg_holiday_spending
        
    FROM transactions t
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
      AND t.transaction_status = 'COMPLETED'
    GROUP BY t.customer_id
)
SELECT 
    cd.customer_id,
    cd.customer_name,
    cd.email,
    
    -- Demographic features
    cd.age,
    cd.tenure_days,
    cd.country_code,
    cd.customer_tier,
    cd.lifetime_value,
    cd.churn_risk_score,
    cd.days_since_last_activity,
    
    -- Transaction behavior features
    COALESCE(ta.total_transactions_12m, 0) as total_transactions,
    COALESCE(ta.unique_merchants_12m, 0) as unique_merchants,
    COALESCE(ta.unique_products_12m, 0) as unique_products,
    COALESCE(ta.unique_merchant_categories_12m, 0) as merchant_category_diversity,
    COALESCE(ta.unique_product_categories_12m, 0) as product_category_diversity,
    
    -- Financial features
    COALESCE(ta.total_amount_usd_12m, 0) as total_spending,
    COALESCE(ta.avg_amount_usd_12m, 0) as avg_transaction_amount,
    COALESCE(ta.stddev_amount_usd_12m, 0) as spending_volatility,
    COALESCE(ta.median_amount_usd_12m, 0) as median_transaction_amount,
    COALESCE(ta.p95_amount_usd_12m, 0) as p95_transaction_amount,
    
    -- Engagement features
    COALESCE(ta.active_days_12m, 0) as active_days,
    COALESCE(ta.avg_transactions_per_day, 0) as transaction_frequency,
    
    -- Channel preferences
    COALESCE(ta.mobile_ratio, 0) as mobile_preference,
    COALESCE(ta.web_ratio, 0) as web_preference,
    COALESCE(ta.credit_card_ratio, 0) as credit_card_preference,
    COALESCE(ta.digital_wallet_ratio, 0) as digital_wallet_preference,
    
    -- Risk features
    COALESCE(ta.avg_merchant_risk_score, 1) as avg_merchant_risk,
    COALESCE(ta.avg_fraud_score, 0) as avg_fraud_score,
    COALESCE(ta.high_risk_transactions, 0) as high_risk_transaction_count,
    
    -- Product preferences
    COALESCE(ta.avg_product_margin, 0) as avg_product_margin,
    COALESCE(ta.electronics_purchases, 0) as electronics_affinity,
    COALESCE(ta.fashion_purchases, 0) as fashion_affinity,
    
    -- Seasonal patterns
    COALESCE(sf.avg_winter_spending, 0) as winter_spending_avg,
    COALESCE(sf.avg_spring_spending, 0) as spring_spending_avg,
    COALESCE(sf.avg_summer_spending, 0) as summer_spending_avg,
    COALESCE(sf.avg_fall_spending, 0) as fall_spending_avg,
    COALESCE(sf.weekend_transaction_ratio, 0) as weekend_activity_ratio,
    COALESCE(sf.avg_holiday_spending, 0) as holiday_spending_avg,
    
    -- Derived features
    CASE 
        WHEN ta.stddev_amount_usd_12m > 0 AND ta.avg_amount_usd_12m > 0 THEN 
            ta.stddev_amount_usd_12m / ta.avg_amount_usd_12m
        ELSE 0
    END as coefficient_of_variation,
    
    CASE 
        WHEN ta.active_days_12m > 0 THEN 
            ta.total_transactions_12m * 1.0 / ta.active_days_12m
        ELSE 0
    END as transactions_per_active_day,
    
    CASE 
        WHEN cd.lifetime_value > 0 AND ta.total_amount_usd_12m > 0 THEN 
            ta.total_amount_usd_12m / cd.lifetime_value
        ELSE 0
    END as recent_spending_vs_lifetime_ratio

FROM customer_demographics cd
LEFT JOIN transaction_aggregates ta ON cd.customer_id = ta.customer_id
LEFT JOIN seasonal_features sf ON cd.customer_id = sf.customer_id
ORDER BY cd.lifetime_value DESC, ta.total_amount_usd_12m DESC;
```

### Feature Normalization Techniques in SQL

#### 1. Min-Max Scaling (0-1 Normalization)

```sql
-- Min-Max scaling for ML features
WITH feature_statistics AS (
    SELECT 
        MIN(age) as min_age,
        MAX(age) as max_age,
        MIN(tenure_days) as min_tenure,
        MAX(tenure_days) as max_tenure,
        MIN(total_spending) as min_spending,
        MAX(total_spending) as max_spending,
        MIN(transaction_frequency) as min_frequency,
        MAX(transaction_frequency) as max_frequency,
        MIN(merchant_category_diversity) as min_diversity,
        MAX(merchant_category_diversity) as max_diversity
    FROM (
        SELECT 
            EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) as age,
            EXTRACT(DAYS FROM AGE(CURRENT_DATE, c.registration_date)) as tenure_days,
            COALESCE(SUM(t.amount_usd), 0) as total_spending,
            COUNT(t.transaction_id) * 1.0 / GREATEST(COUNT(DISTINCT t.transaction_date), 1) as transaction_frequency,
            COUNT(DISTINCT m.merchant_category) as merchant_category_diversity
        FROM customers c
        LEFT JOIN transactions t ON c.customer_id = t.customer_id 
            AND t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
            AND t.transaction_status = 'COMPLETED'
        LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
        WHERE c.account_status = 'ACTIVE'
          AND c.birth_date IS NOT NULL
        GROUP BY c.customer_id, c.birth_date, c.registration_date
    ) raw_features
),
normalized_features AS (
    SELECT 
        c.customer_id,
        c.customer_name,
        c.country_code,
        c.customer_tier,
        
        -- Original features
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) as age_original,
        EXTRACT(DAYS FROM AGE(CURRENT_DATE, c.registration_date)) as tenure_days_original,
        COALESCE(SUM(t.amount_usd), 0) as total_spending_original,
        COUNT(t.transaction_id) * 1.0 / GREATEST(COUNT(DISTINCT t.transaction_date), 1) as frequency_original,
        COUNT(DISTINCT m.merchant_category) as diversity_original,
        
        -- Min-Max normalized features (0-1 scale)
        CASE 
            WHEN fs.max_age - fs.min_age > 0 THEN 
                (EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) - fs.min_age) * 1.0 / 
                (fs.max_age - fs.min_age)
            ELSE 0
        END as age_normalized,
        
        CASE 
            WHEN fs.max_tenure - fs.min_tenure > 0 THEN 
                (EXTRACT(DAYS FROM AGE(CURRENT_DATE, c.registration_date)) - fs.min_tenure) * 1.0 / 
                (fs.max_tenure - fs.min_tenure)
            ELSE 0
        END as tenure_normalized,
        
        CASE 
            WHEN fs.max_spending - fs.min_spending > 0 THEN 
                (COALESCE(SUM(t.amount_usd), 0) - fs.min_spending) * 1.0 / 
                (fs.max_spending - fs.min_spending)
            ELSE 0
        END as spending_normalized,
        
        CASE 
            WHEN fs.max_frequency - fs.min_frequency > 0 THEN 
                ((COUNT(t.transaction_id) * 1.0 / GREATEST(COUNT(DISTINCT t.transaction_date), 1)) - fs.min_frequency) / 
                (fs.max_frequency - fs.min_frequency)
            ELSE 0
        END as frequency_normalized,
        
        CASE 
            WHEN fs.max_diversity - fs.min_diversity > 0 THEN 
                (COUNT(DISTINCT m.merchant_category) - fs.min_diversity) * 1.0 / 
                (fs.max_diversity - fs.min_diversity)
            ELSE 0
        END as diversity_normalized
        
    FROM customers c
    CROSS JOIN feature_statistics fs
    LEFT JOIN transactions t ON c.customer_id = t.customer_id 
        AND t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
        AND t.transaction_status = 'COMPLETED'
    LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
    WHERE c.account_status = 'ACTIVE'
      AND c.birth_date IS NOT NULL
    GROUP BY c.customer_id, c.customer_name, c.country_code, c.customer_tier, 
             c.birth_date, c.registration_date, fs.min_age, fs.max_age, 
             fs.min_tenure, fs.max_tenure, fs.min_spending, fs.max_spending,
             fs.min_frequency, fs.max_frequency, fs.min_diversity, fs.max_diversity
)
SELECT 
    customer_id,
    customer_name,
    country_code,
    customer_tier,
    
    -- Original values for reference
    age_original,
    tenure_days_original,
    total_spending_original,
    frequency_original,
    diversity_original,
    
    -- Normalized values (0-1 scale)
    ROUND(age_normalized::NUMERIC, 4) as age_normalized,
    ROUND(tenure_normalized::NUMERIC, 4) as tenure_normalized,
    ROUND(spending_normalized::NUMERIC, 4) as spending_normalized,
    ROUND(frequency_normalized::NUMERIC, 4) as frequency_normalized,
    ROUND(diversity_normalized::NUMERIC, 4) as diversity_normalized,
    
    -- Create feature vector for ML models
    ARRAY[
        age_normalized,
        tenure_normalized,
        spending_normalized,
        frequency_normalized,
        diversity_normalized
    ] as feature_vector_minmax
    
FROM normalized_features
ORDER BY spending_normalized DESC, age_normalized DESC;
```

#### 2. Z-Score Normalization (Standardization)

```sql
-- Z-Score normalization for ML features
WITH feature_statistics AS (
    SELECT 
        AVG(age) as mean_age,
        STDDEV(age) as stddev_age,
        AVG(tenure_days) as mean_tenure,
        STDDEV(tenure_days) as stddev_tenure,
        AVG(total_spending) as mean_spending,
        STDDEV(total_spending) as stddev_spending,
        AVG(transaction_frequency) as mean_frequency,
        STDDEV(transaction_frequency) as stddev_frequency,
        AVG(avg_transaction_amount) as mean_avg_amount,
        STDDEV(avg_transaction_amount) as stddev_avg_amount
    FROM (
        SELECT 
            EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) as age,
            EXTRACT(DAYS FROM AGE(CURRENT_DATE, c.registration_date)) as tenure_days,
            COALESCE(SUM(t.amount_usd), 0) as total_spending,
            COUNT(t.transaction_id) * 1.0 / GREATEST(COUNT(DISTINCT t.transaction_date), 1) as transaction_frequency,
            COALESCE(AVG(t.amount_usd), 0) as avg_transaction_amount
        FROM customers c
        LEFT JOIN transactions t ON c.customer_id = t.customer_id 
            AND t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
            AND t.transaction_status = 'COMPLETED'
        WHERE c.account_status = 'ACTIVE'
          AND c.birth_date IS NOT NULL
        GROUP BY c.customer_id, c.birth_date, c.registration_date
    ) raw_features
),
standardized_features AS (
    SELECT 
        c.customer_id,
        c.customer_name,
        c.country_code,
        c.customer_tier,
        c.churn_risk_score,
        
        -- Original features
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) as age_original,
        EXTRACT(DAYS FROM AGE(CURRENT_DATE, c.registration_date)) as tenure_days_original,
        COALESCE(SUM(t.amount_usd), 0) as total_spending_original,
        COUNT(t.transaction_id) * 1.0 / GREATEST(COUNT(DISTINCT t.transaction_date), 1) as frequency_original,
        COALESCE(AVG(t.amount_usd), 0) as avg_amount_original,
        
        -- Z-Score standardized features (mean=0, std=1)
        CASE 
            WHEN fs.stddev_age > 0 THEN 
                (EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) - fs.mean_age) / fs.stddev_age
            ELSE 0
        END as age_standardized,
        
        CASE 
            WHEN fs.stddev_tenure > 0 THEN 
                (EXTRACT(DAYS FROM AGE(CURRENT_DATE, c.registration_date)) - fs.mean_tenure) / fs.stddev_tenure
            ELSE 0
        END as tenure_standardized,
        
        CASE 
            WHEN fs.stddev_spending > 0 THEN 
                (COALESCE(SUM(t.amount_usd), 0) - fs.mean_spending) / fs.stddev_spending
            ELSE 0
        END as spending_standardized,
        
        CASE 
            WHEN fs.stddev_frequency > 0 THEN 
                ((COUNT(t.transaction_id) * 1.0 / GREATEST(COUNT(DISTINCT t.transaction_date), 1)) - fs.mean_frequency) / 
                fs.stddev_frequency
            ELSE 0
        END as frequency_standardized,
        
        CASE 
            WHEN fs.stddev_avg_amount > 0 THEN 
                (COALESCE(AVG(t.amount_usd), 0) - fs.mean_avg_amount) / fs.stddev_avg_amount
            ELSE 0
        END as avg_amount_standardized
        
    FROM customers c
    CROSS JOIN feature_statistics fs
    LEFT JOIN transactions t ON c.customer_id = t.customer_id 
        AND t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
        AND t.transaction_status = 'COMPLETED'
    WHERE c.account_status = 'ACTIVE'
      AND c.birth_date IS NOT NULL
    GROUP BY c.customer_id, c.customer_name, c.country_code, c.customer_tier, 
             c.churn_risk_score, c.birth_date, c.registration_date,
             fs.mean_age, fs.stddev_age, fs.mean_tenure, fs.stddev_tenure,
             fs.mean_spending, fs.stddev_spending, fs.mean_frequency, fs.stddev_frequency,
             fs.mean_avg_amount, fs.stddev_avg_amount
)
SELECT 
    customer_id,
    customer_name,
    country_code,
    customer_tier,
    churn_risk_score,
    
    -- Original values
    age_original,
    tenure_days_original,
    total_spending_original,
    frequency_original,
    avg_amount_original,
    
    -- Standardized values (z-scores)
    ROUND(age_standardized::NUMERIC, 4) as age_z_score,
    ROUND(tenure_standardized::NUMERIC, 4) as tenure_z_score,
    ROUND(spending_standardized::NUMERIC, 4) as spending_z_score,
    ROUND(frequency_standardized::NUMERIC, 4) as frequency_z_score,
    ROUND(avg_amount_standardized::NUMERIC, 4) as avg_amount_z_score,
    
    -- Outlier detection using z-scores
    CASE 
        WHEN ABS(age_standardized) > 3 OR 
             ABS(tenure_standardized) > 3 OR 
             ABS(spending_standardized) > 3 OR 
             ABS(frequency_standardized) > 3 OR 
             ABS(avg_amount_standardized) > 3 
        THEN TRUE
        ELSE FALSE
    END as is_outlier,
    
    -- Create standardized feature vector
    ARRAY[
        age_standardized,
        tenure_standardized,
        spending_standardized,
        frequency_standardized,
        avg_amount_standardized
    ] as feature_vector_standardized
    
FROM standardized_features
ORDER BY ABS(spending_standardized) DESC;

-- Validation of standardization (should be ~0 and ~1)
SELECT 
    'Standardization Validation' as check_type,
    ROUND(AVG(age_z_score)::NUMERIC, 6) as mean_age_z_score,
    ROUND(STDDEV(age_z_score)::NUMERIC, 6) as stddev_age_z_score,
    ROUND(AVG(spending_z_score)::NUMERIC, 6) as mean_spending_z_score,
    ROUND(STDDEV(spending_z_score)::NUMERIC, 6) as stddev_spending_z_score
FROM (
    SELECT 
        ROUND(age_standardized::NUMERIC, 4) as age_z_score,
        ROUND(spending_standardized::NUMERIC, 4) as spending_z_score
    FROM standardized_features
) validation_data;
```

#### 3. Robust Scaling (Median and IQR)

```sql
-- Robust scaling using median and IQR (less sensitive to outliers)
WITH robust_statistics AS (
    SELECT 
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY age) as median_age,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY age) - 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY age) as iqr_age,
        
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_spending) as median_spending,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_spending) - 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_spending) as iqr_spending,
        
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY transaction_frequency) as median_frequency,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_frequency) - 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_frequency) as iqr_frequency,
        
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_transaction_amount) as median_avg_amount,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY avg_transaction_amount) - 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY avg_transaction_amount) as iqr_avg_amount
    FROM (
        SELECT 
            EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) as age,
            COALESCE(SUM(t.amount_usd), 0) as total_spending,
            COUNT(t.transaction_id) * 1.0 / GREATEST(COUNT(DISTINCT t.transaction_date), 1) as transaction_frequency,
            COALESCE(AVG(t.amount_usd), 0) as avg_transaction_amount
        FROM customers c
        LEFT JOIN transactions t ON c.customer_id = t.customer_id 
            AND t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
            AND t.transaction_status = 'COMPLETED'
        WHERE c.account_status = 'ACTIVE'
          AND c.birth_date IS NOT NULL
        GROUP BY c.customer_id, c.birth_date, c.registration_date
        HAVING COALESCE(SUM(t.amount_usd), 0) > 0  -- Filter zero-spending customers for robust scaling
    ) raw_features
),
robust_scaled_features AS (
    SELECT 
        c.customer_id,
        c.customer_name,
        c.country_code,
        c.customer_tier,
        
        -- Original features
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) as age_original,
        COALESCE(SUM(t.amount_usd), 0) as total_spending_original,
        COUNT(t.transaction_id) * 1.0 / GREATEST(COUNT(DISTINCT t.transaction_date), 1) as frequency_original,
        COALESCE(AVG(t.amount_usd), 0) as avg_amount_original,
        
        -- Robust scaled features (using median and IQR)
        CASE 
            WHEN rs.iqr_age > 0 THEN 
                (EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) - rs.median_age) / rs.iqr_age
            ELSE 0
        END as age_robust_scaled,
        
        CASE 
            WHEN rs.iqr_spending > 0 THEN 
                (COALESCE(SUM(t.amount_usd), 0) - rs.median_spending) / rs.iqr_spending
            ELSE 0
        END as spending_robust_scaled,
        
        CASE 
            WHEN rs.iqr_frequency > 0 THEN 
                ((COUNT(t.transaction_id) * 1.0 / GREATEST(COUNT(DISTINCT t.transaction_date), 1)) - rs.median_frequency) / 
                rs.iqr_frequency
            ELSE 0
        END as frequency_robust_scaled,
        
        CASE 
            WHEN rs.iqr_avg_amount > 0 THEN 
                (COALESCE(AVG(t.amount_usd), 0) - rs.median_avg_amount) / rs.iqr_avg_amount
            ELSE 0
        END as avg_amount_robust_scaled
        
    FROM customers c
    CROSS JOIN robust_statistics rs
    LEFT JOIN transactions t ON c.customer_id = t.customer_id 
        AND t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
        AND t.transaction_status = 'COMPLETED'
    WHERE c.account_status = 'ACTIVE'
      AND c.birth_date IS NOT NULL
    GROUP BY c.customer_id, c.customer_name, c.country_code, c.customer_tier, 
             c.birth_date, rs.median_age, rs.iqr_age, rs.median_spending, rs.iqr_spending,
             rs.median_frequency, rs.iqr_frequency, rs.median_avg_amount, rs.iqr_avg_amount
    HAVING COALESCE(SUM(t.amount_usd), 0) > 0
)
SELECT 
    customer_id,
    customer_name,
    country_code,
    customer_tier,
    
    -- Original values
    age_original,
    total_spending_original,
    frequency_original,
    avg_amount_original,
    
    -- Robust scaled values
    ROUND(age_robust_scaled::NUMERIC, 4) as age_robust_scaled,
    ROUND(spending_robust_scaled::NUMERIC, 4) as spending_robust_scaled,
    ROUND(frequency_robust_scaled::NUMERIC, 4) as frequency_robust_scaled,
    ROUND(avg_amount_robust_scaled::NUMERIC, 4) as avg_amount_robust_scaled,
    
    -- Outlier detection using robust scaling (values > 3 IQRs from median)
    CASE 
        WHEN ABS(age_robust_scaled) > 3 OR 
             ABS(spending_robust_scaled) > 3 OR 
             ABS(frequency_robust_scaled) > 3 OR 
             ABS(avg_amount_robust_scaled) > 3
        THEN TRUE
        ELSE FALSE
    END as is_robust_outlier,
    
    -- Create robust scaled feature vector
    ARRAY[
        age_robust_scaled,
        spending_robust_scaled,
        frequency_robust_scaled,
        avg_amount_robust_scaled
    ] as feature_vector_robust_scaled
    
FROM robust_scaled_features
ORDER BY ABS(spending_robust_scaled) DESC;
```

### Impact on ML Model Performance

#### 1. Normalization Effects on Different ML Algorithms

```sql
-- Create table to demonstrate normalization impact on ML algorithms
CREATE TABLE ml_algorithm_sensitivity (
    algorithm_type VARCHAR(50),
    normalization_required BOOLEAN,
    sensitivity_to_scale VARCHAR(20),
    recommended_normalization VARCHAR(30),
    explanation TEXT
);

INSERT INTO ml_algorithm_sensitivity VALUES
('Linear Regression', TRUE, 'HIGH', 'Z-Score or Min-Max', 'Coefficients directly affected by feature scales'),
('Logistic Regression', TRUE, 'HIGH', 'Z-Score or Min-Max', 'Gradient descent converges faster with normalized features'),
('Neural Networks', TRUE, 'VERY_HIGH', 'Z-Score or Min-Max', 'Prevents saturation of activation functions'),
('SVM', TRUE, 'VERY_HIGH', 'Min-Max or Z-Score', 'Distance-based algorithm highly sensitive to scale'),
('K-Means Clustering', TRUE, 'VERY_HIGH', 'Z-Score or Min-Max', 'Euclidean distance affected by feature scales'),
('KNN', TRUE, 'VERY_HIGH', 'Min-Max or Z-Score', 'Distance calculations dominated by large-scale features'),
('Random Forest', FALSE, 'LOW', 'None or Robust', 'Tree-based splits not affected by monotonic transformations'),
('XGBoost', FALSE, 'LOW', 'None or Robust', 'Tree-based algorithm handles different scales well'),
('Decision Trees', FALSE, 'NONE', 'None', 'Splits based on thresholds, not absolute values'),
('Naive Bayes', FALSE, 'MEDIUM', 'Log Transform', 'Assumes feature independence, may benefit from normalization');

-- Example: Impact of normalization on distance-based algorithms
WITH unnormalized_data AS (
    SELECT 
        customer_id,
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date)) as age,  -- Range: 18-80
        lifetime_value,  -- Range: 0-50000
        EXTRACT(DAYS FROM AGE(CURRENT_DATE, registration_date)) as tenure_days,  -- Range: 0-3650
        churn_risk_score * 100 as churn_risk_percent  -- Range: 0-100
    FROM customers
    WHERE account_status = 'ACTIVE' 
      AND birth_date IS NOT NULL 
      AND churn_risk_score IS NOT NULL
    LIMIT 10
),
normalized_data AS (
    SELECT 
        customer_id,
        -- Min-Max normalized features (0-1 scale)
        (age - 18.0) / (80.0 - 18.0) as age_normalized,
        (lifetime_value - 0.0) / (50000.0 - 0.0) as lifetime_value_normalized,
        (tenure_days - 0.0) / (3650.0 - 0.0) as tenure_normalized,
        (churn_risk_percent - 0.0) / (100.0 - 0.0) as churn_risk_normalized,
        
        -- Original values for comparison
        age,
        lifetime_value,
        tenure_days,
        churn_risk_percent
    FROM unnormalized_data
),
distance_comparison AS (
    SELECT 
        a.customer_id as customer_a,
        b.customer_id as customer_b,
        
        -- Euclidean distance with unnormalized features (dominated by lifetime_value)
        SQRT(
            POWER(a.age - b.age, 2) + 
            POWER(a.lifetime_value - b.lifetime_value, 2) + 
            POWER(a.tenure_days - b.tenure_days, 2) + 
            POWER(a.churn_risk_percent - b.churn_risk_percent, 2)
        ) as distance_unnormalized,
        
        -- Euclidean distance with normalized features (balanced contribution)
        SQRT(
            POWER(a.age_normalized - b.age_normalized, 2) + 
            POWER(a.lifetime_value_normalized - b.lifetime_value_normalized, 2) + 
            POWER(a.tenure_normalized - b.tenure_normalized, 2) + 
            POWER(a.churn_risk_normalized - b.churn_risk_normalized, 2)
        ) as distance_normalized,
        
        -- Individual feature differences
        ABS(a.age - b.age) as age_diff,
        ABS(a.lifetime_value - b.lifetime_value) as ltv_diff,
        ABS(a.tenure_days - b.tenure_days) as tenure_diff,
        ABS(a.churn_risk_percent - b.churn_risk_percent) as churn_diff
        
    FROM normalized_data a
    CROSS JOIN normalized_data b
    WHERE a.customer_id < b.customer_id  -- Avoid duplicate pairs
)
SELECT 
    customer_a,
    customer_b,
    ROUND(distance_unnormalized::NUMERIC, 2) as distance_unnormalized,
    ROUND(distance_normalized::NUMERIC, 4) as distance_normalized,
    
    -- Show how lifetime_value dominates unnormalized distance
    ROUND((ltv_diff / distance_unnormalized * 100)::NUMERIC, 1) as ltv_contribution_pct,
    
    age_diff,
    ltv_diff,
    tenure_diff,
    churn_diff
    
FROM distance_comparison
ORDER BY distance_unnormalized
LIMIT 15;

-- Show algorithm sensitivity summary
SELECT 
    algorithm_type,
    normalization_required,
    sensitivity_to_scale,
    recommended_normalization,
    explanation
FROM ml_algorithm_sensitivity
ORDER BY 
    CASE sensitivity_to_scale 
        WHEN 'VERY_HIGH' THEN 1 
        WHEN 'HIGH' THEN 2 
        WHEN 'MEDIUM' THEN 3 
        WHEN 'LOW' THEN 4 
        ELSE 5 
    END;
```

### Data Quality and Normalization

#### 1. Handling Missing Values During Normalization

```sql
-- Advanced missing value handling during normalization
CREATE OR REPLACE FUNCTION normalize_features_with_missing_values()
RETURNS TABLE(
    customer_id BIGINT,
    age_original NUMERIC,
    age_normalized NUMERIC,
    age_imputation_method TEXT,
    spending_original NUMERIC,
    spending_normalized NUMERIC,
    spending_imputation_method TEXT,
    feature_completeness_score NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH raw_features AS (
        SELECT 
            c.customer_id,
            EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) as age,
            COALESCE(SUM(t.amount_usd), 0) as total_spending,
            COUNT(t.transaction_id) as transaction_count,
            -- Track data completeness
            CASE WHEN c.birth_date IS NOT NULL THEN 1 ELSE 0 END as has_age,
            CASE WHEN COUNT(t.transaction_id) > 0 THEN 1 ELSE 0 END as has_transactions
        FROM customers c
        LEFT JOIN transactions t ON c.customer_id = t.customer_id 
            AND t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
            AND t.transaction_status = 'COMPLETED'
        WHERE c.account_status = 'ACTIVE'
        GROUP BY c.customer_id, c.birth_date
    ),
    feature_statistics AS (
        SELECT 
            AVG(age) FILTER (WHERE age IS NOT NULL) as mean_age,
            STDDEV(age) FILTER (WHERE age IS NOT NULL) as stddev_age,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY age) FILTER (WHERE age IS NOT NULL) as median_age,
            
            AVG(total_spending) FILTER (WHERE total_spending > 0) as mean_spending,
            STDDEV(total_spending) FILTER (WHERE total_spending > 0) as stddev_spending,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_spending) FILTER (WHERE total_spending > 0) as median_spending,
            
            MIN(age) FILTER (WHERE age IS NOT NULL) as min_age,
            MAX(age) FILTER (WHERE age IS NOT NULL) as max_age,
            MIN(total_spending) as min_spending,
            MAX(total_spending) as max_spending
        FROM raw_features
    ),
    imputed_features AS (
        SELECT 
            rf.customer_id,
            
            -- Age imputation strategy
            CASE 
                WHEN rf.age IS NOT NULL THEN rf.age
                WHEN rf.transaction_count > 10 THEN fs.mean_age  -- Active customers get mean
                ELSE fs.median_age  -- Others get median (more conservative)
            END as age_imputed,
            
            CASE 
                WHEN rf.age IS NOT NULL THEN 'ORIGINAL'
                WHEN rf.transaction_count > 10 THEN 'MEAN_IMPUTATION'
                ELSE 'MEDIAN_IMPUTATION'
            END as age_imputation_method,
            
            -- Spending imputation (0 is valid, so only handle truly missing cases)
            rf.total_spending as spending_imputed,
            'ORIGINAL' as spending_imputation_method,
            
            -- Calculate feature completeness score
            (rf.has_age + rf.has_transactions) * 1.0 / 2 as completeness_score,
            
            rf.age as age_original,
            rf.total_spending as spending_original,
            
            -- Get statistics for normalization
            fs.mean_age, fs.stddev_age, fs.min_age, fs.max_age,
            fs.mean_spending, fs.stddev_spending, fs.min_spending, fs.max_spending
            
        FROM raw_features rf
        CROSS JOIN feature_statistics fs
    )
    SELECT 
        if_data.customer_id,
        if_data.age_original,
        
        -- Z-score normalization for age
        CASE 
            WHEN if_data.stddev_age > 0 THEN 
                ROUND(((if_data.age_imputed - if_data.mean_age) / if_data.stddev_age)::NUMERIC, 4)
            ELSE 0
        END as age_normalized,
        
        if_data.age_imputation_method,
        if_data.spending_original,
        
        -- Min-Max normalization for spending
        CASE 
            WHEN if_data.max_spending - if_data.min_spending > 0 THEN 
                ROUND(((if_data.spending_imputed - if_data.min_spending) / 
                      (if_data.max_spending - if_data.min_spending))::NUMERIC, 4)
            ELSE 0
        END as spending_normalized,
        
        if_data.spending_imputation_method,
        ROUND(if_data.completeness_score::NUMERIC, 2) as feature_completeness_score
        
    FROM imputed_features if_data
    ORDER BY if_data.completeness_score DESC, if_data.customer_id;
END;
$$ LANGUAGE plpgsql;

-- Execute the function to see normalization with missing value handling
SELECT * FROM normalize_features_with_missing_values()
LIMIT 20;

-- Summary of imputation methods used
SELECT 
    age_imputation_method,
    COUNT(*) as customer_count,
    AVG(feature_completeness_score) as avg_completeness_score,
    AVG(age_normalized) as avg_normalized_age,
    STDDEV(age_normalized) as stddev_normalized_age
FROM normalize_features_with_missing_values()
GROUP BY age_imputation_method
ORDER BY customer_count DESC;
```

### Real-World Applications

1. **Schema Design**: Properly normalized database structure for ML data pipelines
2. **Feature Engineering**: Efficient feature extraction from normalized tables
3. **Data Preprocessing**: SQL-based feature scaling and normalization
4. **Model Training**: Consistent, high-quality training data preparation
5. **Data Quality**: Improved data integrity and consistency

### Best Practices

1. **Balance Normalization**: Don't over-normalize if it hurts query performance
2. **Document Relationships**: Maintain clear foreign key relationships
3. **Standardize Features**: Use consistent normalization across training and inference
4. **Handle Missing Values**: Implement robust imputation strategies
5. **Monitor Data Quality**: Regular validation of normalized data

### Common Pitfalls

1. **Over-Normalization**: Excessive joins hurt query performance
2. **Inconsistent Scaling**: Different normalization between training and production
3. **Ignoring Outliers**: Not handling extreme values during normalization
4. **Missing Value Issues**: Poor handling of NULL values in normalization
5. **Feature Leakage**: Using future information in normalization statistics

### Performance Considerations

- **Query Complexity**: Normalized schemas may require complex joins
- **Caching Strategy**: Materialize frequently used normalized features
- **Batch Processing**: Use efficient SQL for large-scale normalization
- **Index Design**: Optimize indexes for normalized table joins
- **Storage Efficiency**: Balance normalization with storage and query performance

---

## Question 6

**What are SQL Window Functions and how can they be used for Machine Learning feature engineering?**

**Answer:**

### Theory

SQL Window Functions are analytical functions that perform calculations across a set of table rows related to the current row, without collapsing the result set like aggregate functions. They operate over a "window" of rows defined by PARTITION BY, ORDER BY, and frame specifications. For machine learning, window functions are essential for creating time-series features, lag variables, rolling statistics, ranking features, and sequential patterns that capture temporal and positional relationships in data.

**Window Function Components:**
- **PARTITION BY**: Divides result set into partitions
- **ORDER BY**: Defines row order within partitions  
- **Frame Specification**: Defines which rows to include (ROWS/RANGE)
- **Window Functions**: ROW_NUMBER(), RANK(), LAG(), LEAD(), SUM() OVER(), etc.

### Ranking and Positional Functions

#### 1. Customer Behavior Ranking Features

```sql
-- Customer transaction ranking and percentile features
WITH customer_transaction_rankings AS (
    SELECT 
        t.customer_id,
        t.transaction_id,
        t.transaction_date,
        t.amount_usd,
        t.merchant_category,
        
        -- Ranking functions for customer transaction history
        ROW_NUMBER() OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_date, t.transaction_id
        ) as transaction_sequence_number,
        
        ROW_NUMBER() OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.amount_usd DESC
        ) as amount_rank_desc,
        
        ROW_NUMBER() OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_date DESC
        ) as recency_rank,
        
        -- Dense ranking for handling ties
        DENSE_RANK() OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.amount_usd DESC
        ) as amount_dense_rank,
        
        -- Percentile rankings
        PERCENT_RANK() OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.amount_usd
        ) as amount_percentile_rank,
        
        CUME_DIST() OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.amount_usd
        ) as amount_cumulative_distribution,
        
        -- Global rankings (across all customers)
        PERCENT_RANK() OVER (ORDER BY t.amount_usd) as global_amount_percentile,
        
        -- Category-specific rankings
        ROW_NUMBER() OVER (
            PARTITION BY t.customer_id, t.merchant_category 
            ORDER BY t.amount_usd DESC
        ) as category_amount_rank,
        
        -- Quartile classification
        NTILE(4) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.amount_usd
        ) as amount_quartile,
        
        -- Decile classification for more granular segmentation
        NTILE(10) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.amount_usd
        ) as amount_decile
        
    FROM transactions t
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
      AND t.transaction_status = 'COMPLETED'
),
enriched_rankings AS (
    SELECT 
        *,
        -- Binary features based on rankings
        CASE WHEN amount_rank_desc <= 3 THEN 1 ELSE 0 END as is_top3_transaction,
        CASE WHEN amount_percentile_rank >= 0.9 THEN 1 ELSE 0 END as is_top10_percent_amount,
        CASE WHEN recency_rank <= 5 THEN 1 ELSE 0 END as is_recent_top5,
        CASE WHEN amount_quartile = 4 THEN 1 ELSE 0 END as is_highest_quartile,
        
        -- Derived ranking features
        1.0 / (recency_rank + 1) as recency_weight,  -- Higher weight for recent transactions
        amount_percentile_rank * 100 as amount_percentile_score,
        
        -- Category dominance features
        CASE 
            WHEN category_amount_rank = 1 THEN 'CATEGORY_DOMINANT'
            WHEN category_amount_rank <= 3 THEN 'CATEGORY_FREQUENT'
            ELSE 'CATEGORY_OCCASIONAL'
        END as category_engagement_level
        
    FROM customer_transaction_rankings
)
SELECT 
    customer_id,
    transaction_date,
    amount_usd,
    merchant_category,
    
    -- Sequential features
    transaction_sequence_number,
    amount_rank_desc,
    recency_rank,
    
    -- Percentile features
    ROUND(amount_percentile_rank::NUMERIC, 4) as amount_percentile_rank,
    ROUND(global_amount_percentile::NUMERIC, 4) as global_amount_percentile,
    
    -- Classification features
    amount_quartile,
    amount_decile,
    
    -- Binary indicator features
    is_top3_transaction,
    is_top10_percent_amount,
    is_recent_top5,
    is_highest_quartile,
    
    -- Derived features
    ROUND(recency_weight::NUMERIC, 4) as recency_weight,
    ROUND(amount_percentile_score::NUMERIC, 2) as amount_percentile_score,
    category_engagement_level
    
FROM enriched_rankings
ORDER BY customer_id, transaction_sequence_number;

-- Customer-level ranking aggregations for ML features
SELECT 
    customer_id,
    COUNT(*) as total_transactions,
    
    -- Top transaction features
    MAX(CASE WHEN amount_rank_desc = 1 THEN amount_usd ELSE 0 END) as highest_transaction_amount,
    AVG(CASE WHEN amount_rank_desc <= 3 THEN amount_usd ELSE NULL END) as avg_top3_transaction_amount,
    
    -- Percentile-based features
    AVG(amount_percentile_rank) as avg_personal_amount_percentile,
    AVG(global_amount_percentile) as avg_global_amount_percentile,
    
    -- Distribution features
    SUM(is_top10_percent_amount) as high_value_transaction_count,
    SUM(is_top3_transaction) as top3_transaction_count,
    
    -- Consistency features
    STDDEV(amount_percentile_rank) as amount_percentile_consistency,
    
    -- Recency-weighted features
    SUM(amount_usd * recency_weight) / SUM(recency_weight) as recency_weighted_avg_amount,
    
    -- Category engagement diversity
    COUNT(DISTINCT category_engagement_level) as category_engagement_diversity
    
FROM enriched_rankings
GROUP BY customer_id
ORDER BY total_transactions DESC;
```

#### 2. Product Popularity and Market Position Features

```sql
-- Product and merchant ranking features for recommendation systems
WITH product_performance_rankings AS (
    SELECT 
        t.product_id,
        p.product_name,
        p.product_category,
        t.merchant_id,
        m.merchant_name,
        t.transaction_date,
        t.amount_usd,
        t.customer_id,
        
        -- Product popularity rankings
        COUNT(*) OVER (
            PARTITION BY t.product_id
        ) as product_total_purchases,
        
        ROW_NUMBER() OVER (
            PARTITION BY t.product_id 
            ORDER BY t.transaction_date DESC
        ) as product_purchase_recency_rank,
        
        -- Product sales ranking within category
        DENSE_RANK() OVER (
            PARTITION BY p.product_category 
            ORDER BY COUNT(*) OVER (PARTITION BY t.product_id) DESC
        ) as product_popularity_rank_in_category,
        
        -- Customer product affinity
        ROW_NUMBER() OVER (
            PARTITION BY t.customer_id, t.product_id 
            ORDER BY t.transaction_date
        ) as customer_product_purchase_sequence,
        
        -- Merchant performance rankings
        AVG(t.amount_usd) OVER (
            PARTITION BY t.merchant_id
        ) as merchant_avg_transaction_amount,
        
        COUNT(DISTINCT t.customer_id) OVER (
            PARTITION BY t.merchant_id
        ) as merchant_unique_customers,
        
        -- Price positioning
        PERCENT_RANK() OVER (
            PARTITION BY p.product_category 
            ORDER BY t.amount_usd
        ) as price_percentile_in_category,
        
        -- Market share approximation
        COUNT(*) OVER (PARTITION BY t.product_id) * 1.0 / 
        COUNT(*) OVER (PARTITION BY p.product_category) as product_category_market_share
        
    FROM transactions t
    JOIN products p ON t.product_id = p.product_id
    JOIN merchants m ON t.merchant_id = m.merchant_id
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '6 months'
      AND t.transaction_status = 'COMPLETED'
),
product_features AS (
    SELECT 
        product_id,
        product_name,
        product_category,
        
        -- Popularity features
        product_total_purchases,
        product_popularity_rank_in_category,
        ROUND(product_category_market_share::NUMERIC, 6) as category_market_share,
        
        -- Customer engagement features
        COUNT(DISTINCT customer_id) as unique_customers,
        COUNT(DISTINCT customer_id) * 1.0 / product_total_purchases as customer_diversity_ratio,
        
        -- Repeat purchase behavior
        COUNT(*) FILTER (WHERE customer_product_purchase_sequence > 1) as repeat_purchases,
        COUNT(*) FILTER (WHERE customer_product_purchase_sequence > 1) * 1.0 / 
        product_total_purchases as repeat_purchase_ratio,
        
        -- Price positioning
        AVG(price_percentile_in_category) as avg_price_percentile,
        STDDEV(price_percentile_in_category) as price_volatility,
        
        -- Merchant distribution
        COUNT(DISTINCT merchant_id) as unique_merchants,
        
        -- Performance classifications
        CASE 
            WHEN product_popularity_rank_in_category <= 10 THEN 'TOP_SELLER'
            WHEN product_popularity_rank_in_category <= 50 THEN 'POPULAR'
            WHEN product_popularity_rank_in_category <= 100 THEN 'MODERATE'
            ELSE 'NICHE'
        END as popularity_category,
        
        CASE 
            WHEN AVG(price_percentile_in_category) >= 0.8 THEN 'PREMIUM'
            WHEN AVG(price_percentile_in_category) >= 0.6 THEN 'HIGH_PRICE'
            WHEN AVG(price_percentile_in_category) >= 0.4 THEN 'MEDIUM_PRICE'
            WHEN AVG(price_percentile_in_category) >= 0.2 THEN 'LOW_PRICE'
            ELSE 'BUDGET'
        END as price_category
        
    FROM product_performance_rankings
    GROUP BY product_id, product_name, product_category, 
             product_total_purchases, product_popularity_rank_in_category, 
             product_category_market_share
)
SELECT 
    product_id,
    product_name,
    product_category,
    product_total_purchases,
    product_popularity_rank_in_category,
    category_market_share,
    unique_customers,
    ROUND(customer_diversity_ratio::NUMERIC, 4) as customer_diversity_ratio,
    repeat_purchases,
    ROUND(repeat_purchase_ratio::NUMERIC, 4) as repeat_purchase_ratio,
    ROUND(avg_price_percentile::NUMERIC, 4) as avg_price_percentile,
    ROUND(price_volatility::NUMERIC, 4) as price_volatility,
    unique_merchants,
    popularity_category,
    price_category
FROM product_features
ORDER BY product_total_purchases DESC, category_market_share DESC;
```

### Lag and Lead Functions for Temporal Features

#### 1. Customer Journey and Temporal Pattern Analysis

```sql
-- Advanced lag/lead functions for customer journey analysis
WITH customer_transaction_timeline AS (
    SELECT 
        t.customer_id,
        t.transaction_id,
        t.transaction_date,
        t.transaction_timestamp,
        t.amount_usd,
        t.merchant_category,
        t.payment_method,
        t.channel,
        
        -- Previous transaction features
        LAG(t.amount_usd, 1) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_timestamp
        ) as prev_transaction_amount,
        
        LAG(t.transaction_date, 1) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_timestamp
        ) as prev_transaction_date,
        
        LAG(t.merchant_category, 1) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_timestamp
        ) as prev_merchant_category,
        
        LAG(t.payment_method, 1) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_timestamp
        ) as prev_payment_method,
        
        -- Next transaction features (for pattern completion)
        LEAD(t.amount_usd, 1) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_timestamp
        ) as next_transaction_amount,
        
        LEAD(t.transaction_date, 1) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_timestamp
        ) as next_transaction_date,
        
        LEAD(t.merchant_category, 1) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_timestamp
        ) as next_merchant_category,
        
        -- Multiple lag features for sequence patterns
        LAG(t.amount_usd, 2) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_timestamp
        ) as amount_lag2,
        
        LAG(t.amount_usd, 3) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_timestamp
        ) as amount_lag3,
        
        -- First and last values in window
        FIRST_VALUE(t.amount_usd) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_timestamp 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as first_transaction_amount,
        
        LAST_VALUE(t.amount_usd) OVER (
            PARTITION BY t.customer_id 
            ORDER BY t.transaction_timestamp 
            ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
        ) as last_transaction_amount
        
    FROM transactions t
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
      AND t.transaction_status = 'COMPLETED'
),
temporal_features AS (
    SELECT 
        *,
        -- Time gap features
        CASE 
            WHEN prev_transaction_date IS NOT NULL THEN 
                EXTRACT(DAYS FROM transaction_date - prev_transaction_date)
            ELSE NULL
        END as days_since_prev_transaction,
        
        CASE 
            WHEN next_transaction_date IS NOT NULL THEN 
                EXTRACT(DAYS FROM next_transaction_date - transaction_date)
            ELSE NULL
        END as days_to_next_transaction,
        
        -- Amount change features
        CASE 
            WHEN prev_transaction_amount IS NOT NULL AND prev_transaction_amount > 0 THEN 
                (amount_usd - prev_transaction_amount) / prev_transaction_amount
            ELSE NULL
        END as amount_change_pct_from_prev,
        
        CASE 
            WHEN next_transaction_amount IS NOT NULL AND amount_usd > 0 THEN 
                (next_transaction_amount - amount_usd) / amount_usd
            ELSE NULL
        END as amount_change_pct_to_next,
        
        -- Behavioral consistency features
        CASE 
            WHEN prev_merchant_category = merchant_category THEN 1 
            ELSE 0 
        END as same_category_as_prev,
        
        CASE 
            WHEN next_merchant_category = merchant_category THEN 1 
            ELSE 0 
        END as same_category_as_next,
        
        CASE 
            WHEN prev_payment_method = payment_method THEN 1 
            ELSE 0 
        END as same_payment_method_as_prev,
        
        -- Sequence pattern features
        CASE 
            WHEN amount_lag2 IS NOT NULL AND prev_transaction_amount IS NOT NULL THEN
                CASE 
                    WHEN amount_usd > prev_transaction_amount AND prev_transaction_amount > amount_lag2 
                    THEN 'INCREASING_TREND'
                    WHEN amount_usd < prev_transaction_amount AND prev_transaction_amount < amount_lag2 
                    THEN 'DECREASING_TREND'
                    ELSE 'FLUCTUATING'
                END
            ELSE NULL
        END as spending_trend_3_transactions,
        
        -- Relative position features
        amount_usd / NULLIF(first_transaction_amount, 0) as amount_vs_first_ratio,
        amount_usd / NULLIF(last_transaction_amount, 0) as amount_vs_last_ratio
        
    FROM customer_transaction_timeline
)
SELECT 
    customer_id,
    transaction_date,
    amount_usd,
    merchant_category,
    payment_method,
    
    -- Previous transaction context
    prev_transaction_amount,
    days_since_prev_transaction,
    ROUND(amount_change_pct_from_prev::NUMERIC, 4) as amount_change_pct_from_prev,
    same_category_as_prev,
    same_payment_method_as_prev,
    
    -- Next transaction context (for pattern analysis)
    next_transaction_amount,
    days_to_next_transaction,
    ROUND(amount_change_pct_to_next::NUMERIC, 4) as amount_change_pct_to_next,
    same_category_as_next,
    
    -- Sequence patterns
    spending_trend_3_transactions,
    
    -- Position in customer journey
    ROUND(amount_vs_first_ratio::NUMERIC, 4) as amount_vs_first_ratio,
    ROUND(amount_vs_last_ratio::NUMERIC, 4) as amount_vs_last_ratio,
    
    -- Derived features for ML
    CASE 
        WHEN days_since_prev_transaction IS NULL THEN 'FIRST_TRANSACTION'
        WHEN days_since_prev_transaction <= 1 THEN 'SAME_DAY_REPEAT'
        WHEN days_since_prev_transaction <= 7 THEN 'WEEKLY_REPEAT'
        WHEN days_since_prev_transaction <= 30 THEN 'MONTHLY_REPEAT'
        WHEN days_since_prev_transaction <= 90 THEN 'QUARTERLY_REPEAT'
        ELSE 'LONG_GAP'
    END as transaction_frequency_pattern,
    
    CASE 
        WHEN amount_change_pct_from_prev IS NULL THEN NULL
        WHEN amount_change_pct_from_prev > 0.5 THEN 'LARGE_INCREASE'
        WHEN amount_change_pct_from_prev > 0.1 THEN 'MODERATE_INCREASE'
        WHEN amount_change_pct_from_prev > -0.1 THEN 'STABLE'
        WHEN amount_change_pct_from_prev > -0.5 THEN 'MODERATE_DECREASE'
        ELSE 'LARGE_DECREASE'
    END as spending_change_category
    
FROM temporal_features
ORDER BY customer_id, transaction_date;
```

### Rolling Aggregations and Moving Statistics

#### 1. Rolling Window Statistics for Trend Analysis

```sql
-- Advanced rolling window aggregations for ML feature engineering
WITH daily_customer_metrics AS (
    SELECT 
        t.customer_id,
        t.transaction_date,
        COUNT(*) as daily_transaction_count,
        SUM(t.amount_usd) as daily_total_amount,
        AVG(t.amount_usd) as daily_avg_amount,
        STDDEV(t.amount_usd) as daily_amount_stddev,
        MIN(t.amount_usd) as daily_min_amount,
        MAX(t.amount_usd) as daily_max_amount,
        COUNT(DISTINCT t.merchant_category) as daily_category_count,
        COUNT(*) FILTER (WHERE t.channel = 'MOBILE') as daily_mobile_transactions,
        COUNT(*) FILTER (WHERE t.fraud_score > 0.5) as daily_risky_transactions
    FROM transactions t
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '6 months'
      AND t.transaction_status = 'COMPLETED'
    GROUP BY t.customer_id, t.transaction_date
),
rolling_statistics AS (
    SELECT 
        customer_id,
        transaction_date,
        daily_transaction_count,
        daily_total_amount,
        daily_avg_amount,
        
        -- 7-day rolling windows
        SUM(daily_transaction_count) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as rolling_7d_transaction_count,
        
        AVG(daily_total_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as rolling_7d_avg_daily_amount,
        
        SUM(daily_total_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as rolling_7d_total_amount,
        
        STDDEV(daily_total_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as rolling_7d_amount_volatility,
        
        -- 30-day rolling windows
        AVG(daily_transaction_count) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_avg_daily_transactions,
        
        AVG(daily_total_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_avg_daily_amount,
        
        STDDEV(daily_total_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_amount_stddev,
        
        -- Rolling maximums and minimums
        MAX(daily_total_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_max_daily_amount,
        
        MIN(daily_total_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_min_daily_amount,
        
        -- Rolling trend analysis using linear regression
        REGR_SLOPE(daily_total_amount, EXTRACT(EPOCH FROM transaction_date)) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) as rolling_14d_spending_trend_slope,
        
        REGR_R2(daily_total_amount, EXTRACT(EPOCH FROM transaction_date)) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) as rolling_14d_trend_r_squared,
        
        -- Rolling percentiles
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY daily_total_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_median_daily_amount,
        
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY daily_total_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_p90_daily_amount,
        
        -- Rolling activity patterns
        COUNT(*) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as rolling_7d_active_days,
        
        COUNT(*) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_active_days,
        
        -- Rolling category diversity
        AVG(daily_category_count) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as rolling_7d_avg_category_diversity,
        
        -- Rolling mobile usage
        SUM(daily_mobile_transactions) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) * 1.0 / NULLIF(SUM(daily_transaction_count) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 0) as rolling_7d_mobile_ratio,
        
        -- Rolling risk metrics
        SUM(daily_risky_transactions) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) * 1.0 / NULLIF(SUM(daily_transaction_count) OVER (
            PARTITION BY customer_id 
            ORDER BY transaction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 0) as rolling_7d_risk_ratio
        
    FROM daily_customer_metrics
),
enhanced_features AS (
    SELECT 
        *,
        -- Derived volatility features
        CASE 
            WHEN rolling_7d_avg_daily_amount > 0 THEN 
                rolling_7d_amount_volatility / rolling_7d_avg_daily_amount
            ELSE NULL
        END as rolling_7d_coefficient_of_variation,
        
        CASE 
            WHEN rolling_30d_avg_daily_amount > 0 THEN 
                rolling_30d_amount_stddev / rolling_30d_avg_daily_amount
            ELSE NULL
        END as rolling_30d_coefficient_of_variation,
        
        -- Range-based volatility
        CASE 
            WHEN rolling_30d_min_daily_amount > 0 THEN 
                (rolling_30d_max_daily_amount - rolling_30d_min_daily_amount) / rolling_30d_min_daily_amount
            ELSE NULL
        END as rolling_30d_range_ratio,
        
        -- Anomaly detection features
        CASE 
            WHEN rolling_30d_amount_stddev > 0 THEN 
                ABS(daily_total_amount - rolling_30d_avg_daily_amount) / rolling_30d_amount_stddev
            ELSE 0
        END as daily_amount_z_score_30d,
        
        -- Activity consistency features
        rolling_7d_active_days * 1.0 / 7 as rolling_7d_activity_ratio,
        rolling_30d_active_days * 1.0 / 30 as rolling_30d_activity_ratio,
        
        -- Trend strength indicators
        CASE 
            WHEN rolling_14d_spending_trend_slope > 0 AND rolling_14d_trend_r_squared > 0.5 
            THEN 'STRONG_INCREASING'
            WHEN rolling_14d_spending_trend_slope > 0 AND rolling_14d_trend_r_squared > 0.2 
            THEN 'MODERATE_INCREASING'
            WHEN rolling_14d_spending_trend_slope < 0 AND rolling_14d_trend_r_squared > 0.5 
            THEN 'STRONG_DECREASING'
            WHEN rolling_14d_spending_trend_slope < 0 AND rolling_14d_trend_r_squared > 0.2 
            THEN 'MODERATE_DECREASING'
            ELSE 'NO_CLEAR_TREND'
        END as spending_trend_strength
        
    FROM rolling_statistics
)
SELECT 
    customer_id,
    transaction_date,
    
    -- Daily metrics
    daily_transaction_count,
    ROUND(daily_total_amount::NUMERIC, 2) as daily_total_amount,
    ROUND(daily_avg_amount::NUMERIC, 2) as daily_avg_amount,
    
    -- Rolling aggregations
    rolling_7d_transaction_count,
    ROUND(rolling_7d_avg_daily_amount::NUMERIC, 2) as rolling_7d_avg_daily_amount,
    ROUND(rolling_7d_total_amount::NUMERIC, 2) as rolling_7d_total_amount,
    ROUND(rolling_30d_avg_daily_amount::NUMERIC, 2) as rolling_30d_avg_daily_amount,
    
    -- Volatility features
    ROUND(rolling_7d_coefficient_of_variation::NUMERIC, 4) as rolling_7d_coeff_of_variation,
    ROUND(rolling_30d_coefficient_of_variation::NUMERIC, 4) as rolling_30d_coeff_of_variation,
    ROUND(rolling_30d_range_ratio::NUMERIC, 4) as rolling_30d_range_ratio,
    
    -- Trend features
    ROUND(rolling_14d_spending_trend_slope::NUMERIC, 8) as rolling_14d_trend_slope,
    ROUND(rolling_14d_trend_r_squared::NUMERIC, 4) as rolling_14d_trend_r_squared,
    spending_trend_strength,
    
    -- Activity features
    rolling_7d_active_days,
    rolling_30d_active_days,
    ROUND(rolling_7d_activity_ratio::NUMERIC, 4) as rolling_7d_activity_ratio,
    ROUND(rolling_30d_activity_ratio::NUMERIC, 4) as rolling_30d_activity_ratio,
    
    -- Behavioral features
    ROUND(rolling_7d_avg_category_diversity::NUMERIC, 2) as rolling_7d_avg_category_diversity,
    ROUND(rolling_7d_mobile_ratio::NUMERIC, 4) as rolling_7d_mobile_ratio,
    ROUND(rolling_7d_risk_ratio::NUMERIC, 4) as rolling_7d_risk_ratio,
    
    -- Anomaly features
    ROUND(daily_amount_z_score_30d::NUMERIC, 4) as daily_amount_z_score_30d,
    CASE 
        WHEN daily_amount_z_score_30d > 3 THEN TRUE
        ELSE FALSE
    END as is_daily_amount_anomaly,
    
    -- Percentile features
    ROUND(rolling_30d_median_daily_amount::NUMERIC, 2) as rolling_30d_median,
    ROUND(rolling_30d_p90_daily_amount::NUMERIC, 2) as rolling_30d_p90,
    
    -- Relative position features
    CASE 
        WHEN rolling_30d_median_daily_amount > 0 THEN 
            ROUND((daily_total_amount / rolling_30d_median_daily_amount)::NUMERIC, 4)
        ELSE NULL
    END as daily_vs_median_ratio
    
FROM enhanced_features
WHERE rolling_30d_active_days >= 5  -- Ensure sufficient history for reliable features
ORDER BY customer_id, transaction_date;
```

### Real-World Applications

1. **Time-Series Features**: Create lag variables and rolling statistics for temporal modeling
2. **Customer Journey Analysis**: Track progression and patterns in customer behavior
3. **Anomaly Detection**: Identify unusual patterns using rolling window comparisons
4. **Ranking Features**: Create relative position and percentile-based features
5. **Sequential Patterns**: Detect trends and behavioral sequences

### Best Practices

1. **Appropriate Partitioning**: Use logical partitions (customer, product, etc.)
2. **Efficient Ordering**: Order by meaningful temporal or logical sequences
3. **Frame Specification**: Choose appropriate window frames (ROWS vs RANGE)
4. **Handle NULLs**: Consider NULL handling in lag/lead operations
5. **Performance Optimization**: Use appropriate indexes for window function queries

### Common Pitfalls

1. **Inappropriate Partitioning**: Wrong partition keys leading to incorrect calculations
2. **Memory Usage**: Large windows consuming excessive memory
3. **Performance Issues**: Complex window functions without proper indexing
4. **Data Leakage**: Using future information in historical features
5. **Frame Confusion**: Misunderstanding ROWS vs RANGE behavior

### Performance Considerations

- **Index Strategy**: Index on partition and order columns
- **Window Size**: Balance feature richness with computational cost
- **Parallel Processing**: Window functions can often be parallelized
- **Memory Management**: Large partitions may require memory optimization
- **Query Planning**: Understand execution plans for window function queries

---

## Question 7

**Explain how to discretize a continuous variable in SQL.**

**Answer:**

### Theory

Discretization is the process of converting continuous variables into categorical or discrete intervals. In machine learning, discretization is crucial for handling continuous features in algorithms that work better with categorical data, reducing noise, handling outliers, and creating interpretable features. SQL provides several methods for discretization including equal-width binning, equal-frequency binning, and custom business logic-based binning.

**Discretization Benefits:**
- **Noise Reduction**: Smooths out minor variations in continuous data
- **Outlier Handling**: Extreme values are contained within bins
- **Algorithm Compatibility**: Some ML algorithms prefer categorical features
- **Interpretability**: Binned features are easier to understand and explain
- **Non-linear Relationships**: Can capture non-linear patterns between features and targets

### Equal-Width Binning (Fixed Intervals)

#### 1. Basic Equal-Width Discretization

```sql
-- Simple equal-width binning for customer age
SELECT 
    customer_id,
    age,
    -- Basic age groups with equal width
    CASE 
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 45 THEN '35-44'
        WHEN age < 55 THEN '45-54'
        WHEN age < 65 THEN '55-64'
        ELSE '65+'
    END as age_group,
    
    -- Numeric bin assignment
    CASE 
        WHEN age < 25 THEN 1
        WHEN age < 35 THEN 2
        WHEN age < 45 THEN 3
        WHEN age < 55 THEN 4
        WHEN age < 65 THEN 5
        ELSE 6
    END as age_bin
FROM customers
WHERE age IS NOT NULL AND age BETWEEN 18 AND 100;

-- Dynamic equal-width binning using statistics
WITH age_stats AS (
    SELECT 
        MIN(age) as min_age,
        MAX(age) as max_age,
        (MAX(age) - MIN(age)) / 5.0 as bin_width
    FROM customers
    WHERE age IS NOT NULL AND age BETWEEN 18 AND 100
),
discretized_ages AS (
    SELECT 
        c.customer_id,
        c.age,
        s.min_age,
        s.max_age,
        s.bin_width,
        
        -- Calculate bin number (0-based)
        FLOOR((c.age - s.min_age) / s.bin_width) as bin_number,
        
        -- Create bin labels
        CONCAT(
            ROUND(s.min_age + FLOOR((c.age - s.min_age) / s.bin_width) * s.bin_width),
            '-',
            ROUND(s.min_age + (FLOOR((c.age - s.min_age) / s.bin_width) + 1) * s.bin_width - 1)
        ) as age_bin_label,
        
        -- Bin boundaries
        s.min_age + FLOOR((c.age - s.min_age) / s.bin_width) * s.bin_width as bin_lower,
        s.min_age + (FLOOR((c.age - s.min_age) / s.bin_width) + 1) * s.bin_width as bin_upper
        
    FROM customers c
    CROSS JOIN age_stats s
    WHERE c.age IS NOT NULL AND c.age BETWEEN 18 AND 100
)
SELECT 
    customer_id,
    age,
    bin_number + 1 as age_bin,  -- 1-based bins
    age_bin_label,
    ROUND(bin_lower::NUMERIC, 1) as bin_lower_bound,
    ROUND(bin_upper::NUMERIC, 1) as bin_upper_bound
FROM discretized_ages
ORDER BY customer_id;

-- Income discretization with business logic
SELECT 
    customer_id,
    annual_income,
    
    -- Income brackets with meaningful business ranges
    CASE 
        WHEN annual_income < 25000 THEN 'LOW_INCOME'
        WHEN annual_income < 50000 THEN 'LOWER_MIDDLE'
        WHEN annual_income < 75000 THEN 'MIDDLE'
        WHEN annual_income < 100000 THEN 'UPPER_MIDDLE'
        WHEN annual_income < 150000 THEN 'HIGH'
        WHEN annual_income < 250000 THEN 'VERY_HIGH'
        ELSE 'ULTRA_HIGH'
    END as income_bracket,
    
    -- Numeric encoding for ML models
    CASE 
        WHEN annual_income < 25000 THEN 1
        WHEN annual_income < 50000 THEN 2
        WHEN annual_income < 75000 THEN 3
        WHEN annual_income < 100000 THEN 4
        WHEN annual_income < 150000 THEN 5
        WHEN annual_income < 250000 THEN 6
        ELSE 7
    END as income_bin_numeric,
    
    -- Log-scale binning for highly skewed income data
    CASE 
        WHEN LOG(annual_income + 1) < 10 THEN 'LOG_BIN_1'
        WHEN LOG(annual_income + 1) < 11 THEN 'LOG_BIN_2'
        WHEN LOG(annual_income + 1) < 12 THEN 'LOG_BIN_3'
        WHEN LOG(annual_income + 1) < 13 THEN 'LOG_BIN_4'
        ELSE 'LOG_BIN_5'
    END as income_log_bin
    
FROM customers
WHERE annual_income IS NOT NULL AND annual_income > 0;
```

#### 2. Advanced Equal-Width Binning with Width Functions

```sql
-- Transaction amount discretization using WIDTH_BUCKET
WITH amount_bins AS (
    SELECT 
        transaction_id,
        customer_id,
        amount_usd,
        transaction_date,
        
        -- WIDTH_BUCKET function for automatic binning
        WIDTH_BUCKET(
            amount_usd, 
            0,      -- Lower bound
            5000,   -- Upper bound
            10      -- Number of bins
        ) as amount_bin,
        
        -- Custom bin calculation showing the logic
        CASE 
            WHEN amount_usd = 5000 THEN 10  -- Handle edge case
            ELSE LEAST(FLOOR(amount_usd / 500) + 1, 10)
        END as amount_bin_manual,
        
        -- Percentile-based binning
        NTILE(10) OVER (ORDER BY amount_usd) as amount_percentile_bin
        
    FROM transactions
    WHERE amount_usd BETWEEN 0 AND 5000
      AND transaction_date >= CURRENT_DATE - INTERVAL '12 months'
),
bin_analysis AS (
    SELECT 
        amount_bin,
        COUNT(*) as transaction_count,
        MIN(amount_usd) as bin_min_amount,
        MAX(amount_usd) as bin_max_amount,
        AVG(amount_usd) as bin_avg_amount,
        STDDEV(amount_usd) as bin_stddev_amount,
        
        -- Bin boundaries for equal-width
        (amount_bin - 1) * 500 as theoretical_bin_lower,
        amount_bin * 500 as theoretical_bin_upper
        
    FROM amount_bins
    GROUP BY amount_bin
)
SELECT 
    ab.*,
    ba.transaction_count,
    ba.bin_min_amount,
    ba.bin_max_amount,
    ROUND(ba.bin_avg_amount::NUMERIC, 2) as bin_avg_amount,
    ROUND(ba.bin_stddev_amount::NUMERIC, 2) as bin_stddev_amount,
    ba.theoretical_bin_lower,
    ba.theoretical_bin_upper,
    
    -- Create categorical labels
    CONCAT('$', ba.theoretical_bin_lower, '-$', ba.theoretical_bin_upper) as amount_range_label
    
FROM amount_bins ab
JOIN bin_analysis ba ON ab.amount_bin = ba.amount_bin
ORDER BY ab.customer_id, ab.transaction_date;
```

### Equal-Frequency Binning (Quantile-Based)

#### 1. Percentile-Based Discretization

```sql
-- Customer lifetime value discretization using quantiles
WITH clv_percentiles AS (
    SELECT 
        PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY customer_lifetime_value) as p20,
        PERCENTILE_CONT(0.4) WITHIN GROUP (ORDER BY customer_lifetime_value) as p40,
        PERCENTILE_CONT(0.6) WITHIN GROUP (ORDER BY customer_lifetime_value) as p60,
        PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY customer_lifetime_value) as p80,
        MIN(customer_lifetime_value) as min_clv,
        MAX(customer_lifetime_value) as max_clv
    FROM customers
    WHERE customer_lifetime_value IS NOT NULL AND customer_lifetime_value > 0
),
clv_discretized AS (
    SELECT 
        c.customer_id,
        c.customer_lifetime_value,
        p.p20, p.p40, p.p60, p.p80,
        
        -- Quintile-based binning (equal frequency)
        CASE 
            WHEN c.customer_lifetime_value <= p.p20 THEN 1
            WHEN c.customer_lifetime_value <= p.p40 THEN 2
            WHEN c.customer_lifetime_value <= p.p60 THEN 3
            WHEN c.customer_lifetime_value <= p.p80 THEN 4
            ELSE 5
        END as clv_quintile,
        
        -- Descriptive labels
        CASE 
            WHEN c.customer_lifetime_value <= p.p20 THEN 'BOTTOM_20PCT'
            WHEN c.customer_lifetime_value <= p.p40 THEN 'LOW_20PCT'
            WHEN c.customer_lifetime_value <= p.p60 THEN 'MIDDLE_20PCT'
            WHEN c.customer_lifetime_value <= p.p80 THEN 'HIGH_20PCT'
            ELSE 'TOP_20PCT'
        END as clv_segment,
        
        -- NTILE function for validation
        NTILE(5) OVER (ORDER BY c.customer_lifetime_value) as clv_ntile_quintile
        
    FROM customers c
    CROSS JOIN clv_percentiles p
    WHERE c.customer_lifetime_value IS NOT NULL AND c.customer_lifetime_value > 0
)
SELECT 
    customer_id,
    ROUND(customer_lifetime_value::NUMERIC, 2) as customer_lifetime_value,
    clv_quintile,
    clv_segment,
    clv_ntile_quintile,
    
    -- Validation: both methods should give same result
    CASE 
        WHEN clv_quintile = clv_ntile_quintile THEN 'MATCH'
        ELSE 'MISMATCH'
    END as validation_check,
    
    -- Percentile boundaries for reference
    ROUND(p20::NUMERIC, 2) as quintile_1_upper,
    ROUND(p40::NUMERIC, 2) as quintile_2_upper,
    ROUND(p60::NUMERIC, 2) as quintile_3_upper,
    ROUND(p80::NUMERIC, 2) as quintile_4_upper
    
FROM clv_discretized
ORDER BY customer_lifetime_value;

-- Validate equal frequency distribution
SELECT 
    clv_quintile,
    clv_segment,
    COUNT(*) as customer_count,
    MIN(customer_lifetime_value) as segment_min_clv,
    MAX(customer_lifetime_value) as segment_max_clv,
    AVG(customer_lifetime_value) as segment_avg_clv,
    ROUND(COUNT(*)::NUMERIC / SUM(COUNT(*)) OVER () * 100, 2) as percentage_of_total
FROM clv_discretized
GROUP BY clv_quintile, clv_segment
ORDER BY clv_quintile;
```

#### 2. Advanced Quantile-Based Binning

```sql
-- Multi-variable quantile-based discretization
WITH transaction_statistics AS (
    SELECT 
        customer_id,
        COUNT(*) as transaction_count,
        AVG(amount_usd) as avg_transaction_amount,
        STDDEV(amount_usd) as transaction_amount_stddev,
        SUM(amount_usd) as total_spending,
        MAX(transaction_date) as last_transaction_date,
        MIN(transaction_date) as first_transaction_date
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
      AND amount_usd > 0
    GROUP BY customer_id
    HAVING COUNT(*) >= 3  -- Minimum transaction requirement
),
multi_variable_quantiles AS (
    SELECT 
        customer_id,
        transaction_count,
        avg_transaction_amount,
        transaction_amount_stddev,
        total_spending,
        
        -- Multiple quantile-based features
        NTILE(4) OVER (ORDER BY transaction_count) as frequency_quartile,
        NTILE(4) OVER (ORDER BY avg_transaction_amount) as avg_amount_quartile,
        NTILE(4) OVER (ORDER BY total_spending) as spending_quartile,
        NTILE(4) OVER (ORDER BY transaction_amount_stddev) as volatility_quartile,
        
        -- Decile-based features for finer granularity
        NTILE(10) OVER (ORDER BY total_spending) as spending_decile,
        
        -- Combined scoring using multiple variables
        NTILE(5) OVER (
            ORDER BY 
                transaction_count * 0.3 + 
                avg_transaction_amount * 0.4 + 
                total_spending * 0.3
        ) as composite_value_quintile
        
    FROM transaction_statistics
),
customer_segments AS (
    SELECT 
        *,
        -- Create customer value segments using multiple dimensions
        CASE 
            WHEN spending_quartile = 4 AND frequency_quartile >= 3 THEN 'HIGH_VALUE_FREQUENT'
            WHEN spending_quartile = 4 AND frequency_quartile < 3 THEN 'HIGH_VALUE_OCCASIONAL'
            WHEN spending_quartile = 3 AND frequency_quartile >= 3 THEN 'MEDIUM_VALUE_FREQUENT'
            WHEN spending_quartile = 3 AND frequency_quartile < 3 THEN 'MEDIUM_VALUE_OCCASIONAL'
            WHEN spending_quartile <= 2 AND frequency_quartile >= 3 THEN 'LOW_VALUE_FREQUENT'
            ELSE 'LOW_VALUE_OCCASIONAL'
        END as customer_segment,
        
        -- Risk-based segmentation using volatility
        CASE 
            WHEN volatility_quartile = 4 THEN 'HIGH_VOLATILITY'
            WHEN volatility_quartile = 3 THEN 'MEDIUM_VOLATILITY'
            ELSE 'LOW_VOLATILITY'
        END as spending_pattern,
        
        -- Combine multiple quartiles into single score
        (frequency_quartile + avg_amount_quartile + spending_quartile + (5 - volatility_quartile)) / 4.0 as combined_score
        
    FROM multi_variable_quantiles
)
SELECT 
    customer_id,
    ROUND(transaction_count::NUMERIC, 0) as transaction_count,
    ROUND(avg_transaction_amount::NUMERIC, 2) as avg_transaction_amount,
    ROUND(total_spending::NUMERIC, 2) as total_spending,
    ROUND(transaction_amount_stddev::NUMERIC, 2) as spending_volatility,
    
    -- Quartile-based features
    frequency_quartile,
    avg_amount_quartile,
    spending_quartile,
    volatility_quartile,
    spending_decile,
    composite_value_quintile,
    
    -- Derived segments
    customer_segment,
    spending_pattern,
    ROUND(combined_score::NUMERIC, 2) as combined_score,
    
    -- Final discretized score
    CASE 
        WHEN combined_score >= 4.0 THEN 'PREMIUM'
        WHEN combined_score >= 3.0 THEN 'HIGH_VALUE'
        WHEN combined_score >= 2.0 THEN 'MEDIUM_VALUE'
        ELSE 'LOW_VALUE'
    END as final_customer_tier
    
FROM customer_segments
ORDER BY combined_score DESC, total_spending DESC;
```

### Custom Business Logic Discretization

#### 1. Domain-Specific Binning

```sql
-- Credit score discretization based on industry standards
SELECT 
    customer_id,
    credit_score,
    
    -- Standard credit score ranges
    CASE 
        WHEN credit_score >= 800 THEN 'EXCELLENT'
        WHEN credit_score >= 740 THEN 'VERY_GOOD'
        WHEN credit_score >= 670 THEN 'GOOD'
        WHEN credit_score >= 580 THEN 'FAIR'
        WHEN credit_score >= 300 THEN 'POOR'
        ELSE 'NO_SCORE'
    END as credit_rating,
    
    -- Risk-based grouping for lending decisions
    CASE 
        WHEN credit_score >= 720 THEN 'LOW_RISK'
        WHEN credit_score >= 650 THEN 'MEDIUM_RISK'
        WHEN credit_score >= 580 THEN 'HIGH_RISK'
        ELSE 'VERY_HIGH_RISK'
    END as risk_category,
    
    -- Numeric encoding for ML models
    CASE 
        WHEN credit_score >= 800 THEN 5
        WHEN credit_score >= 740 THEN 4
        WHEN credit_score >= 670 THEN 3
        WHEN credit_score >= 580 THEN 2
        WHEN credit_score >= 300 THEN 1
        ELSE 0
    END as credit_score_bin
    
FROM customers
WHERE credit_score IS NOT NULL;

-- Purchase recency discretization using business rules
WITH customer_recency AS (
    SELECT 
        customer_id,
        MAX(purchase_date) as last_purchase_date,
        EXTRACT(DAYS FROM CURRENT_DATE - MAX(purchase_date)) as days_since_last_purchase
    FROM purchases
    GROUP BY customer_id
)
SELECT 
    customer_id,
    last_purchase_date,
    days_since_last_purchase,
    
    -- RFM-style recency binning
    CASE 
        WHEN days_since_last_purchase <= 30 THEN 'RECENT'
        WHEN days_since_last_purchase <= 90 THEN 'ACTIVE'
        WHEN days_since_last_purchase <= 180 THEN 'LAPSED'
        WHEN days_since_last_purchase <= 365 THEN 'DORMANT'
        ELSE 'INACTIVE'
    END as recency_segment,
    
    -- Numeric scoring (higher is better for recency)
    CASE 
        WHEN days_since_last_purchase <= 30 THEN 5
        WHEN days_since_last_purchase <= 90 THEN 4
        WHEN days_since_last_purchase <= 180 THEN 3
        WHEN days_since_last_purchase <= 365 THEN 2
        ELSE 1
    END as recency_score,
    
    -- Churn risk based on recency
    CASE 
        WHEN days_since_last_purchase > 365 THEN 'HIGH_CHURN_RISK'
        WHEN days_since_last_purchase > 180 THEN 'MEDIUM_CHURN_RISK'
        WHEN days_since_last_purchase > 90 THEN 'LOW_CHURN_RISK'
        ELSE 'ACTIVE_CUSTOMER'
    END as churn_risk_category
    
FROM customer_recency
ORDER BY days_since_last_purchase;
```

#### 2. Time-Based Discretization

```sql
-- Temporal feature discretization for time-series analysis
SELECT 
    transaction_id,
    customer_id,
    transaction_timestamp,
    amount_usd,
    
    -- Hour of day discretization
    CASE 
        WHEN EXTRACT(HOUR FROM transaction_timestamp) BETWEEN 6 AND 11 THEN 'MORNING'
        WHEN EXTRACT(HOUR FROM transaction_timestamp) BETWEEN 12 AND 17 THEN 'AFTERNOON'
        WHEN EXTRACT(HOUR FROM transaction_timestamp) BETWEEN 18 AND 22 THEN 'EVENING'
        ELSE 'NIGHT'
    END as time_of_day,
    
    -- Granular hour bins
    CASE 
        WHEN EXTRACT(HOUR FROM transaction_timestamp) < 6 THEN 'LATE_NIGHT'
        WHEN EXTRACT(HOUR FROM transaction_timestamp) < 9 THEN 'EARLY_MORNING'
        WHEN EXTRACT(HOUR FROM transaction_timestamp) < 12 THEN 'LATE_MORNING'
        WHEN EXTRACT(HOUR FROM transaction_timestamp) < 15 THEN 'EARLY_AFTERNOON'
        WHEN EXTRACT(HOUR FROM transaction_timestamp) < 18 THEN 'LATE_AFTERNOON'
        WHEN EXTRACT(HOUR FROM transaction_timestamp) < 21 THEN 'EARLY_EVENING'
        ELSE 'LATE_EVENING'
    END as detailed_time_period,
    
    -- Day of week grouping
    CASE 
        WHEN EXTRACT(DOW FROM transaction_timestamp) IN (1, 2, 3, 4, 5) THEN 'WEEKDAY'
        ELSE 'WEEKEND'
    END as day_type,
    
    -- Month seasonality
    CASE 
        WHEN EXTRACT(MONTH FROM transaction_timestamp) IN (12, 1, 2) THEN 'WINTER'
        WHEN EXTRACT(MONTH FROM transaction_timestamp) IN (3, 4, 5) THEN 'SPRING'
        WHEN EXTRACT(MONTH FROM transaction_timestamp) IN (6, 7, 8) THEN 'SUMMER'
        ELSE 'FALL'
    END as season,
    
    -- Holiday periods (US-centric example)
    CASE 
        WHEN EXTRACT(MONTH FROM transaction_timestamp) = 11 
             AND EXTRACT(DAY FROM transaction_timestamp) >= 20 THEN 'THANKSGIVING_PERIOD'
        WHEN EXTRACT(MONTH FROM transaction_timestamp) = 12 
             AND EXTRACT(DAY FROM transaction_timestamp) >= 15 THEN 'CHRISTMAS_PERIOD'
        WHEN EXTRACT(MONTH FROM transaction_timestamp) = 1 
             AND EXTRACT(DAY FROM transaction_timestamp) <= 5 THEN 'NEW_YEAR_PERIOD'
        WHEN EXTRACT(MONTH FROM transaction_timestamp) = 7 
             AND EXTRACT(DAY FROM transaction_timestamp) BETWEEN 1 AND 7 THEN 'JULY_4TH_PERIOD'
        ELSE 'REGULAR_PERIOD'
    END as holiday_period,
    
    -- Week of month
    CASE 
        WHEN EXTRACT(DAY FROM transaction_timestamp) <= 7 THEN 'WEEK_1'
        WHEN EXTRACT(DAY FROM transaction_timestamp) <= 14 THEN 'WEEK_2'
        WHEN EXTRACT(DAY FROM transaction_timestamp) <= 21 THEN 'WEEK_3'
        ELSE 'WEEK_4_PLUS'
    END as week_of_month
    
FROM transactions
WHERE transaction_timestamp >= CURRENT_DATE - INTERVAL '12 months'
ORDER BY transaction_timestamp;
```

### Advanced Discretization Techniques

#### 1. Optimal Binning Using Statistical Methods

```sql
-- Chi-square based optimal binning for target variable
WITH target_statistics AS (
    SELECT 
        customer_id,
        annual_income,
        CASE WHEN churned = TRUE THEN 1 ELSE 0 END as churn_target
    FROM customers
    WHERE annual_income IS NOT NULL AND churned IS NOT NULL
),
income_bins AS (
    SELECT 
        customer_id,
        annual_income,
        churn_target,
        
        -- Initial equal-frequency binning
        NTILE(10) OVER (ORDER BY annual_income) as initial_bin,
        
        -- Business-driven binning
        CASE 
            WHEN annual_income < 30000 THEN 1
            WHEN annual_income < 50000 THEN 2
            WHEN annual_income < 75000 THEN 3
            WHEN annual_income < 100000 THEN 4
            WHEN annual_income < 150000 THEN 5
            ELSE 6
        END as business_bin
        
    FROM target_statistics
),
bin_performance AS (
    SELECT 
        business_bin,
        COUNT(*) as total_customers,
        SUM(churn_target) as churned_customers,
        AVG(churn_target) as churn_rate,
        MIN(annual_income) as bin_min_income,
        MAX(annual_income) as bin_max_income,
        
        -- Chi-square contribution calculation
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as expected_proportion,
        SUM(churn_target) * 1.0 / SUM(SUM(churn_target)) OVER () as actual_proportion
        
    FROM income_bins
    GROUP BY business_bin
),
chi_square_analysis AS (
    SELECT 
        *,
        -- Chi-square statistic component
        POWER(actual_proportion - expected_proportion, 2) / 
        NULLIF(expected_proportion, 0) as chi_square_component,
        
        -- Information value calculation for feature selection
        (actual_proportion - expected_proportion) * 
        LN(NULLIF(actual_proportion, 0) / NULLIF(expected_proportion, 0)) as iv_component
        
    FROM bin_performance
)
SELECT 
    business_bin,
    total_customers,
    churned_customers,
    ROUND(churn_rate::NUMERIC, 4) as churn_rate,
    ROUND(bin_min_income::NUMERIC, 0) as bin_min_income,
    ROUND(bin_max_income::NUMERIC, 0) as bin_max_income,
    ROUND(chi_square_component::NUMERIC, 6) as chi_square_contribution,
    ROUND(iv_component::NUMERIC, 6) as information_value_contribution,
    
    -- Bin quality assessment
    CASE 
        WHEN total_customers >= 100 AND ABS(churn_rate - 0.15) > 0.05 THEN 'GOOD_SEPARATION'
        WHEN total_customers >= 50 THEN 'ADEQUATE'
        ELSE 'INSUFFICIENT_DATA'
    END as bin_quality
    
FROM chi_square_analysis
ORDER BY business_bin;

-- Summary statistics for binning evaluation
SELECT 
    'BINNING_SUMMARY' as analysis_type,
    COUNT(DISTINCT business_bin) as number_of_bins,
    SUM(chi_square_component) as total_chi_square,
    SUM(iv_component) as total_information_value,
    
    -- Interpretation
    CASE 
        WHEN SUM(iv_component) > 0.3 THEN 'STRONG_PREDICTIVE_POWER'
        WHEN SUM(iv_component) > 0.1 THEN 'MEDIUM_PREDICTIVE_POWER'
        WHEN SUM(iv_component) > 0.02 THEN 'WEAK_PREDICTIVE_POWER'
        ELSE 'NO_PREDICTIVE_POWER'
    END as feature_strength
    
FROM chi_square_analysis;
```

#### 2. Tree-Based Discretization

```sql
-- Decision tree-inspired binning using recursive splits
WITH recursive_splits AS (
    -- First split: median split
    SELECT 
        customer_id,
        transaction_frequency,
        is_high_value_customer,
        CASE 
            WHEN transaction_frequency >= (
                SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY transaction_frequency)
                FROM customer_metrics
            ) THEN 'HIGH_FREQ'
            ELSE 'LOW_FREQ'
        END as frequency_split_1
    FROM customer_metrics
    WHERE transaction_frequency IS NOT NULL
),
second_level_splits AS (
    SELECT 
        *,
        -- Split each half again
        CASE 
            WHEN frequency_split_1 = 'HIGH_FREQ' THEN
                CASE 
                    WHEN transaction_frequency >= (
                        SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY transaction_frequency)
                        FROM customer_metrics
                    ) THEN 'VERY_HIGH_FREQ'
                    ELSE 'MEDIUM_HIGH_FREQ'
                END
            ELSE -- LOW_FREQ
                CASE 
                    WHEN transaction_frequency >= (
                        SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY transaction_frequency)
                        FROM customer_metrics
                    ) THEN 'MEDIUM_LOW_FREQ'
                    ELSE 'VERY_LOW_FREQ'
                END
        END as frequency_split_2
        
    FROM recursive_splits
),
final_tree_bins AS (
    SELECT 
        *,
        -- Create final bins based on tree structure
        CASE frequency_split_2
            WHEN 'VERY_HIGH_FREQ' THEN 4
            WHEN 'MEDIUM_HIGH_FREQ' THEN 3
            WHEN 'MEDIUM_LOW_FREQ' THEN 2
            WHEN 'VERY_LOW_FREQ' THEN 1
        END as tree_based_bin,
        
        -- Validate splits maintain target separation
        AVG(is_high_value_customer::INTEGER) OVER (
            PARTITION BY frequency_split_2
        ) as bin_target_rate
        
    FROM second_level_splits
)
SELECT 
    tree_based_bin,
    frequency_split_2 as bin_label,
    COUNT(*) as customer_count,
    MIN(transaction_frequency) as bin_min_frequency,
    MAX(transaction_frequency) as bin_max_frequency,
    AVG(transaction_frequency) as bin_avg_frequency,
    AVG(is_high_value_customer::INTEGER) as high_value_rate,
    
    -- Bin homogeneity measure
    STDDEV(is_high_value_customer::INTEGER) as target_stddev_in_bin
    
FROM final_tree_bins
GROUP BY tree_based_bin, frequency_split_2, bin_target_rate
ORDER BY tree_based_bin;
```

### One-Hot Encoding After Discretization

#### 1. Creating Binary Features from Bins

```sql
-- Convert discretized features to one-hot encoded features
WITH discretized_features AS (
    SELECT 
        customer_id,
        
        -- Age group discretization
        CASE 
            WHEN age < 25 THEN 'AGE_18_24'
            WHEN age < 35 THEN 'AGE_25_34'
            WHEN age < 45 THEN 'AGE_35_44'
            WHEN age < 55 THEN 'AGE_45_54'
            WHEN age < 65 THEN 'AGE_55_64'
            ELSE 'AGE_65_PLUS'
        END as age_group,
        
        -- Income discretization
        CASE 
            WHEN annual_income < 50000 THEN 'INCOME_LOW'
            WHEN annual_income < 100000 THEN 'INCOME_MEDIUM'
            ELSE 'INCOME_HIGH'
        END as income_bracket
        
    FROM customers
    WHERE age IS NOT NULL AND annual_income IS NOT NULL
)
SELECT 
    customer_id,
    
    -- One-hot encoding for age groups
    CASE WHEN age_group = 'AGE_18_24' THEN 1 ELSE 0 END as age_18_24,
    CASE WHEN age_group = 'AGE_25_34' THEN 1 ELSE 0 END as age_25_34,
    CASE WHEN age_group = 'AGE_35_44' THEN 1 ELSE 0 END as age_35_44,
    CASE WHEN age_group = 'AGE_45_54' THEN 1 ELSE 0 END as age_45_54,
    CASE WHEN age_group = 'AGE_55_64' THEN 1 ELSE 0 END as age_55_64,
    CASE WHEN age_group = 'AGE_65_PLUS' THEN 1 ELSE 0 END as age_65_plus,
    
    -- One-hot encoding for income brackets
    CASE WHEN income_bracket = 'INCOME_LOW' THEN 1 ELSE 0 END as income_low,
    CASE WHEN income_bracket = 'INCOME_MEDIUM' THEN 1 ELSE 0 END as income_medium,
    CASE WHEN income_bracket = 'INCOME_HIGH' THEN 1 ELSE 0 END as income_high,
    
    -- Alternative: JSON representation for ML frameworks
    jsonb_build_object(
        'age_features', jsonb_build_object(
            'age_18_24', CASE WHEN age_group = 'AGE_18_24' THEN 1 ELSE 0 END,
            'age_25_34', CASE WHEN age_group = 'AGE_25_34' THEN 1 ELSE 0 END,
            'age_35_44', CASE WHEN age_group = 'AGE_35_44' THEN 1 ELSE 0 END,
            'age_45_54', CASE WHEN age_group = 'AGE_45_54' THEN 1 ELSE 0 END,
            'age_55_64', CASE WHEN age_group = 'AGE_55_64' THEN 1 ELSE 0 END,
            'age_65_plus', CASE WHEN age_group = 'AGE_65_PLUS' THEN 1 ELSE 0 END
        ),
        'income_features', jsonb_build_object(
            'income_low', CASE WHEN income_bracket = 'INCOME_LOW' THEN 1 ELSE 0 END,
            'income_medium', CASE WHEN income_bracket = 'INCOME_MEDIUM' THEN 1 ELSE 0 END,
            'income_high', CASE WHEN income_bracket = 'INCOME_HIGH' THEN 1 ELSE 0 END
        )
    ) as encoded_features_json
    
FROM discretized_features
ORDER BY customer_id;
```

### Real-World Applications

1. **Customer Segmentation**: Age, income, and spending discretization
2. **Risk Assessment**: Credit score and behavior pattern binning
3. **Feature Engineering**: Converting continuous variables for tree-based models
4. **A/B Testing**: Creating test groups based on discretized metrics
5. **Marketing Campaigns**: Target audience definition using binned demographics

### Best Practices

1. **Domain Knowledge**: Use business-meaningful bin boundaries
2. **Statistical Validation**: Ensure bins have sufficient sample sizes
3. **Target Separation**: Validate that bins differentiate target classes
4. **Consistency**: Apply same binning logic to training and production data
5. **Documentation**: Clearly document binning logic and boundaries

### Common Pitfalls

1. **Too Many Bins**: Creating bins with insufficient data
2. **Arbitrary Boundaries**: Using cut-offs without business justification
3. **Data Leakage**: Using future information in historical binning
4. **Overfitting**: Creating bins that work only on training data
5. **Loss of Information**: Excessive discretization losing predictive power

### Performance Considerations

- **Computational Efficiency**: Binning can reduce model complexity
- **Memory Usage**: Categorical features may require more storage
- **Query Performance**: Binning logic should be optimized for production
- **Maintenance**: Regular validation of bin boundaries as data evolves
- **Scalability**: Ensure binning approaches work with large datasets

---

## Question 8

**Explain how to perform binning of categorical variables in SQL for use in a Machine Learning model.**

**Answer:**

### Theory

Categorical variable binning is the process of grouping or combining categorical values into fewer, more meaningful categories. This technique is essential for machine learning as it helps reduce dimensionality, handle rare categories, improve model performance, and create more interpretable features. SQL provides powerful tools for categorical binning through CASE statements, aggregation functions, and pattern matching.

**Binning Benefits:**
- **Dimensionality Reduction**: Reduces the number of categories for algorithms
- **Rare Category Handling**: Groups infrequent categories to prevent overfitting
- **Domain Knowledge Integration**: Creates business-meaningful category groups
- **Improved Generalization**: Reduces model complexity and improves robustness
- **Memory Efficiency**: Fewer categories require less storage and processing

### Basic Categorical Binning

#### 1. Manual Category Grouping

```sql
-- Product category binning for e-commerce ML features
SELECT 
    product_id,
    original_category,
    subcategory,
    
    -- Group detailed categories into broader bins
    CASE 
        WHEN original_category IN ('smartphones', 'tablets', 'laptops', 'smartwatches') 
            THEN 'ELECTRONICS'
        WHEN original_category IN ('shirts', 'pants', 'dresses', 'shoes', 'accessories') 
            THEN 'CLOTHING'
        WHEN original_category IN ('fiction', 'non_fiction', 'textbooks', 'magazines') 
            THEN 'BOOKS_MEDIA'
        WHEN original_category IN ('furniture', 'decor', 'kitchen', 'bathroom') 
            THEN 'HOME_GARDEN'
        WHEN original_category IN ('toys', 'games', 'sports', 'outdoor') 
            THEN 'RECREATION'
        ELSE 'OTHER'
    END as category_bin,
    
    -- Create hierarchical binning
    CASE 
        WHEN original_category IN ('smartphones', 'tablets') THEN 'MOBILE_DEVICES'
        WHEN original_category IN ('laptops', 'desktops') THEN 'COMPUTERS'
        WHEN original_category IN ('headphones', 'speakers', 'smartwatches') THEN 'ACCESSORIES'
        WHEN original_category IN ('shirts', 'pants', 'dresses') THEN 'APPAREL'
        WHEN original_category IN ('shoes', 'bags', 'jewelry') THEN 'FASHION_ACCESSORIES'
        ELSE 'GENERAL'
    END as detailed_category_bin,
    
    -- Price-influenced category binning
    CASE 
        WHEN original_category IN ('jewelry', 'luxury_watches', 'designer_clothing') 
            THEN 'LUXURY'
        WHEN original_category IN ('smartphones', 'laptops', 'furniture') 
            THEN 'HIGH_VALUE'
        WHEN original_category IN ('books', 'accessories', 'decor') 
            THEN 'MEDIUM_VALUE'
        ELSE 'LOW_VALUE'
    END as value_category_bin
    
FROM products
WHERE original_category IS NOT NULL;

-- Customer location binning for geographic features
SELECT 
    customer_id,
    city,
    state,
    country,
    
    -- Geographic region binning
    CASE 
        WHEN state IN ('CA', 'OR', 'WA', 'NV', 'AZ') THEN 'WEST_COAST'
        WHEN state IN ('NY', 'NJ', 'CT', 'MA', 'PA') THEN 'NORTHEAST'
        WHEN state IN ('FL', 'GA', 'SC', 'NC', 'VA') THEN 'SOUTHEAST'
        WHEN state IN ('TX', 'OK', 'AR', 'LA', 'NM') THEN 'SOUTH_CENTRAL'
        WHEN state IN ('IL', 'IN', 'OH', 'MI', 'WI') THEN 'MIDWEST'
        ELSE 'OTHER_US'
    END as us_region,
    
    -- Metropolitan area binning
    CASE 
        WHEN city IN ('New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix') 
            THEN 'MAJOR_METRO'
        WHEN city IN ('Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose') 
            THEN 'LARGE_METRO'
        ELSE 'SMALLER_METRO'
    END as metro_size,
    
    -- Climate-based binning
    CASE 
        WHEN state IN ('FL', 'TX', 'CA', 'AZ', 'NV', 'HI') THEN 'WARM_CLIMATE'
        WHEN state IN ('MN', 'WI', 'MI', 'ME', 'VT', 'NH', 'AK') THEN 'COLD_CLIMATE'
        ELSE 'TEMPERATE_CLIMATE'
    END as climate_zone
    
FROM customers
WHERE city IS NOT NULL AND state IS NOT NULL;
```

#### 2. Frequency-Based Binning

```sql
-- Brand binning based on frequency and market share
WITH brand_statistics AS (
    SELECT 
        brand,
        COUNT(*) as product_count,
        COUNT(DISTINCT customer_id) as unique_customers,
        SUM(revenue) as total_brand_revenue,
        AVG(price) as avg_brand_price,
        
        -- Calculate brand market share
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as market_share_by_products,
        SUM(revenue) * 1.0 / SUM(SUM(revenue)) OVER () as market_share_by_revenue
        
    FROM sales_data
    WHERE brand IS NOT NULL
    GROUP BY brand
),
brand_ranking AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (ORDER BY total_brand_revenue DESC) as revenue_rank,
        ROW_NUMBER() OVER (ORDER BY product_count DESC) as volume_rank,
        
        -- Percentile ranking
        PERCENT_RANK() OVER (ORDER BY total_brand_revenue) as revenue_percentile,
        PERCENT_RANK() OVER (ORDER BY product_count) as volume_percentile
        
    FROM brand_statistics
),
brand_bins AS (
    SELECT 
        brand,
        product_count,
        total_brand_revenue,
        market_share_by_revenue,
        
        -- Frequency-based binning
        CASE 
            WHEN product_count >= 1000 THEN 'HIGH_VOLUME_BRAND'
            WHEN product_count >= 100 THEN 'MEDIUM_VOLUME_BRAND'
            WHEN product_count >= 10 THEN 'LOW_VOLUME_BRAND'
            ELSE 'RARE_BRAND'
        END as volume_bin,
        
        -- Revenue-based binning
        CASE 
            WHEN revenue_rank <= 10 THEN 'TOP_10_BRAND'
            WHEN revenue_rank <= 50 THEN 'TOP_50_BRAND'
            WHEN revenue_rank <= 200 THEN 'MID_TIER_BRAND'
            ELSE 'LONG_TAIL_BRAND'
        END as revenue_tier_bin,
        
        -- Market share binning
        CASE 
            WHEN market_share_by_revenue >= 0.05 THEN 'MAJOR_BRAND'      -- 5%+ market share
            WHEN market_share_by_revenue >= 0.01 THEN 'SIGNIFICANT_BRAND' -- 1-5% market share
            WHEN market_share_by_revenue >= 0.001 THEN 'NICHE_BRAND'     -- 0.1-1% market share
            ELSE 'MICRO_BRAND'                                            -- <0.1% market share
        END as market_share_bin,
        
        -- Combined scoring bin
        CASE 
            WHEN revenue_percentile >= 0.9 AND volume_percentile >= 0.8 THEN 'PREMIUM_HIGH_VOLUME'
            WHEN revenue_percentile >= 0.9 THEN 'PREMIUM_BRAND'
            WHEN volume_percentile >= 0.9 THEN 'HIGH_VOLUME_BRAND'
            WHEN revenue_percentile >= 0.7 OR volume_percentile >= 0.7 THEN 'ESTABLISHED_BRAND'
            WHEN revenue_percentile >= 0.3 OR volume_percentile >= 0.3 THEN 'EMERGING_BRAND'
            ELSE 'SMALL_BRAND'
        END as composite_brand_bin
        
    FROM brand_ranking
)
SELECT 
    sd.product_id,
    sd.brand,
    bb.volume_bin,
    bb.revenue_tier_bin,
    bb.market_share_bin,
    bb.composite_brand_bin,
    
    -- Create numeric encoding for ML models
    CASE bb.composite_brand_bin
        WHEN 'PREMIUM_HIGH_VOLUME' THEN 6
        WHEN 'PREMIUM_BRAND' THEN 5
        WHEN 'HIGH_VOLUME_BRAND' THEN 4
        WHEN 'ESTABLISHED_BRAND' THEN 3
        WHEN 'EMERGING_BRAND' THEN 2
        WHEN 'SMALL_BRAND' THEN 1
        ELSE 0
    END as brand_score
    
FROM sales_data sd
JOIN brand_bins bb ON sd.brand = bb.brand
ORDER BY sd.product_id;
```

### Advanced Categorical Binning Techniques

#### 1. Statistical-Based Binning

```sql
-- Skill-based employee binning using statistical measures
WITH skill_analysis AS (
    SELECT 
        employee_id,
        department,
        job_title,
        years_experience,
        performance_rating,
        salary,
        
        -- Extract skill keywords from job descriptions
        CASE 
            WHEN job_description ILIKE '%python%' OR job_description ILIKE '%sql%' 
                OR job_description ILIKE '%machine learning%' THEN 'DATA_SKILLS'
            WHEN job_description ILIKE '%java%' OR job_description ILIKE '%javascript%' 
                OR job_description ILIKE '%react%' THEN 'WEB_DEV_SKILLS'
            WHEN job_description ILIKE '%aws%' OR job_description ILIKE '%kubernetes%' 
                OR job_description ILIKE '%docker%' THEN 'CLOUD_SKILLS'
            WHEN job_description ILIKE '%project management%' OR job_description ILIKE '%agile%' 
                OR job_description ILIKE '%scrum%' THEN 'MANAGEMENT_SKILLS'
            ELSE 'GENERAL_SKILLS'
        END as skill_category
        
    FROM employees
    WHERE department IS NOT NULL AND job_title IS NOT NULL
),
department_statistics AS (
    SELECT 
        department,
        COUNT(*) as dept_size,
        AVG(salary) as avg_dept_salary,
        STDDEV(salary) as salary_stddev,
        AVG(years_experience) as avg_experience,
        AVG(performance_rating) as avg_performance
        
    FROM skill_analysis
    GROUP BY department
),
employee_binning AS (
    SELECT 
        sa.*,
        ds.dept_size,
        ds.avg_dept_salary,
        
        -- Department size binning
        CASE 
            WHEN ds.dept_size >= 100 THEN 'LARGE_DEPARTMENT'
            WHEN ds.dept_size >= 30 THEN 'MEDIUM_DEPARTMENT'
            WHEN ds.dept_size >= 10 THEN 'SMALL_DEPARTMENT'
            ELSE 'MICRO_DEPARTMENT'
        END as department_size_bin,
        
        -- Salary percentile within department
        PERCENT_RANK() OVER (
            PARTITION BY sa.department 
            ORDER BY sa.salary
        ) as salary_percentile_in_dept,
        
        -- Experience level binning
        CASE 
            WHEN sa.years_experience <= 2 THEN 'ENTRY_LEVEL'
            WHEN sa.years_experience <= 5 THEN 'JUNIOR_LEVEL'
            WHEN sa.years_experience <= 10 THEN 'MID_LEVEL'
            WHEN sa.years_experience <= 15 THEN 'SENIOR_LEVEL'
            ELSE 'EXPERT_LEVEL'
        END as experience_bin,
        
        -- Performance-based binning
        CASE 
            WHEN sa.performance_rating >= 4.5 THEN 'TOP_PERFORMER'
            WHEN sa.performance_rating >= 4.0 THEN 'HIGH_PERFORMER'
            WHEN sa.performance_rating >= 3.5 THEN 'GOOD_PERFORMER'
            WHEN sa.performance_rating >= 3.0 THEN 'AVERAGE_PERFORMER'
            ELSE 'NEEDS_IMPROVEMENT'
        END as performance_bin
        
    FROM skill_analysis sa
    JOIN department_statistics ds ON sa.department = ds.department
),
composite_employee_bins AS (
    SELECT 
        *,
        -- Create composite bins for ML features
        CASE 
            WHEN salary_percentile_in_dept >= 0.8 AND performance_bin IN ('TOP_PERFORMER', 'HIGH_PERFORMER') 
                THEN 'HIGH_VALUE_EMPLOYEE'
            WHEN salary_percentile_in_dept >= 0.6 OR performance_bin = 'HIGH_PERFORMER' 
                THEN 'VALUED_EMPLOYEE'
            WHEN salary_percentile_in_dept >= 0.4 AND performance_bin != 'NEEDS_IMPROVEMENT' 
                THEN 'STANDARD_EMPLOYEE'
            WHEN performance_bin = 'NEEDS_IMPROVEMENT' 
                THEN 'AT_RISK_EMPLOYEE'
            ELSE 'DEVELOPING_EMPLOYEE'
        END as employee_value_segment,
        
        -- Career stage binning
        CONCAT(experience_bin, '_', skill_category) as career_stage_skill_bin,
        
        -- Retention risk binning
        CASE 
            WHEN salary_percentile_in_dept < 0.3 AND performance_bin IN ('TOP_PERFORMER', 'HIGH_PERFORMER') 
                THEN 'HIGH_FLIGHT_RISK'
            WHEN salary_percentile_in_dept < 0.5 AND performance_bin = 'HIGH_PERFORMER' 
                THEN 'MEDIUM_FLIGHT_RISK'
            WHEN performance_bin = 'NEEDS_IMPROVEMENT' 
                THEN 'PERFORMANCE_RISK'
            ELSE 'LOW_RISK'
        END as retention_risk_bin
        
    FROM employee_binning
)
SELECT 
    employee_id,
    department,
    job_title,
    skill_category,
    department_size_bin,
    experience_bin,
    performance_bin,
    employee_value_segment,
    career_stage_skill_bin,
    retention_risk_bin,
    
    -- Numeric encodings for ML
    CASE department_size_bin
        WHEN 'LARGE_DEPARTMENT' THEN 4
        WHEN 'MEDIUM_DEPARTMENT' THEN 3
        WHEN 'SMALL_DEPARTMENT' THEN 2
        WHEN 'MICRO_DEPARTMENT' THEN 1
    END as dept_size_numeric,
    
    CASE employee_value_segment
        WHEN 'HIGH_VALUE_EMPLOYEE' THEN 5
        WHEN 'VALUED_EMPLOYEE' THEN 4
        WHEN 'STANDARD_EMPLOYEE' THEN 3
        WHEN 'DEVELOPING_EMPLOYEE' THEN 2
        WHEN 'AT_RISK_EMPLOYEE' THEN 1
    END as value_score
    
FROM composite_employee_bins
ORDER BY employee_id;
```

#### 2. Pattern-Based Binning

```sql
-- Product description binning using text patterns
WITH product_features AS (
    SELECT 
        product_id,
        product_name,
        product_description,
        category,
        
        -- Extract features from product names/descriptions
        CASE 
            WHEN product_description ILIKE '%wireless%' OR product_description ILIKE '%bluetooth%' 
                THEN 'WIRELESS'
            WHEN product_description ILIKE '%wired%' OR product_description ILIKE '%cable%' 
                THEN 'WIRED'
            ELSE 'CONNECTION_UNSPECIFIED'
        END as connectivity_feature,
        
        -- Size-based binning from descriptions
        CASE 
            WHEN product_description ILIKE '%large%' OR product_description ILIKE '%xl%' 
                OR product_description ILIKE '%big%' THEN 'LARGE_SIZE'
            WHEN product_description ILIKE '%medium%' OR product_description ILIKE '%standard%' 
                THEN 'MEDIUM_SIZE'
            WHEN product_description ILIKE '%small%' OR product_description ILIKE '%compact%' 
                OR product_description ILIKE '%mini%' THEN 'SMALL_SIZE'
            ELSE 'SIZE_UNSPECIFIED'
        END as size_feature,
        
        -- Material binning
        CASE 
            WHEN product_description ILIKE '%leather%' THEN 'LEATHER'
            WHEN product_description ILIKE '%metal%' OR product_description ILIKE '%steel%' 
                OR product_description ILIKE '%aluminum%' THEN 'METAL'
            WHEN product_description ILIKE '%plastic%' OR product_description ILIKE '%polymer%' 
                THEN 'PLASTIC'
            WHEN product_description ILIKE '%wood%' OR product_description ILIKE '%wooden%' 
                THEN 'WOOD'
            WHEN product_description ILIKE '%glass%' THEN 'GLASS'
            WHEN product_description ILIKE '%fabric%' OR product_description ILIKE '%cotton%' 
                OR product_description ILIKE '%silk%' THEN 'FABRIC'
            ELSE 'MATERIAL_UNSPECIFIED'
        END as material_feature,
        
        -- Color binning
        CASE 
            WHEN product_description ILIKE '%black%' THEN 'BLACK'
            WHEN product_description ILIKE '%white%' THEN 'WHITE'
            WHEN product_description ILIKE '%red%' THEN 'RED'
            WHEN product_description ILIKE '%blue%' THEN 'BLUE'
            WHEN product_description ILIKE '%green%' THEN 'GREEN'
            WHEN product_description ILIKE '%silver%' OR product_description ILIKE '%grey%' 
                OR product_description ILIKE '%gray%' THEN 'NEUTRAL'
            ELSE 'COLOR_UNSPECIFIED'
        END as color_feature,
        
        -- Quality indicators
        CASE 
            WHEN product_description ILIKE '%premium%' OR product_description ILIKE '%luxury%' 
                OR product_description ILIKE '%high-end%' THEN 'PREMIUM'
            WHEN product_description ILIKE '%professional%' OR product_description ILIKE '%commercial%' 
                THEN 'PROFESSIONAL'
            WHEN product_description ILIKE '%basic%' OR product_description ILIKE '%standard%' 
                OR product_description ILIKE '%economy%' THEN 'BASIC'
            ELSE 'QUALITY_UNSPECIFIED'
        END as quality_tier,
        
        -- Brand positioning from descriptions
        CASE 
            WHEN product_description ILIKE '%innovative%' OR product_description ILIKE '%cutting-edge%' 
                OR product_description ILIKE '%advanced%' THEN 'INNOVATION_FOCUSED'
            WHEN product_description ILIKE '%reliable%' OR product_description ILIKE '%durable%' 
                OR product_description ILIKE '%sturdy%' THEN 'RELIABILITY_FOCUSED'
            WHEN product_description ILIKE '%affordable%' OR product_description ILIKE '%budget%' 
                OR product_description ILIKE '%value%' THEN 'VALUE_FOCUSED'
            ELSE 'POSITIONING_UNSPECIFIED'
        END as brand_positioning
        
    FROM products
    WHERE product_description IS NOT NULL
),
feature_combinations AS (
    SELECT 
        *,
        -- Create composite features
        CONCAT(connectivity_feature, '_', size_feature) as connectivity_size_combo,
        CONCAT(material_feature, '_', quality_tier) as material_quality_combo,
        
        -- Multi-level categorization
        CASE 
            WHEN quality_tier = 'PREMIUM' AND material_feature IN ('LEATHER', 'METAL') 
                THEN 'LUXURY_PRODUCT'
            WHEN quality_tier = 'PROFESSIONAL' AND connectivity_feature = 'WIRELESS' 
                THEN 'PRO_WIRELESS_PRODUCT'
            WHEN brand_positioning = 'VALUE_FOCUSED' AND quality_tier = 'BASIC' 
                THEN 'BUDGET_PRODUCT'
            WHEN connectivity_feature = 'WIRELESS' AND size_feature = 'SMALL_SIZE' 
                THEN 'PORTABLE_WIRELESS_PRODUCT'
            ELSE 'GENERAL_PRODUCT'
        END as product_segment,
        
        -- Target market binning
        CASE 
            WHEN quality_tier = 'PREMIUM' OR brand_positioning = 'INNOVATION_FOCUSED' 
                THEN 'PREMIUM_MARKET'
            WHEN quality_tier = 'PROFESSIONAL' OR connectivity_feature = 'WIRELESS' 
                THEN 'PROFESSIONAL_MARKET'
            WHEN brand_positioning = 'VALUE_FOCUSED' OR quality_tier = 'BASIC' 
                THEN 'MASS_MARKET'
            ELSE 'GENERAL_MARKET'
        END as target_market
        
    FROM product_features
)
SELECT 
    product_id,
    product_name,
    category,
    connectivity_feature,
    size_feature,
    material_feature,
    color_feature,
    quality_tier,
    brand_positioning,
    product_segment,
    target_market,
    
    -- Create feature vectors for ML models
    ARRAY[
        CASE connectivity_feature WHEN 'WIRELESS' THEN 1 ELSE 0 END,
        CASE size_feature WHEN 'LARGE_SIZE' THEN 1 ELSE 0 END,
        CASE material_feature WHEN 'PREMIUM' THEN 1 ELSE 0 END,
        CASE quality_tier WHEN 'PREMIUM' THEN 1 ELSE 0 END
    ] as feature_vector,
    
    -- Numeric scoring
    (CASE connectivity_feature WHEN 'WIRELESS' THEN 2 WHEN 'WIRED' THEN 1 ELSE 0 END +
     CASE quality_tier WHEN 'PREMIUM' THEN 3 WHEN 'PROFESSIONAL' THEN 2 WHEN 'BASIC' THEN 1 ELSE 0 END +
     CASE material_feature WHEN 'LEATHER' THEN 2 WHEN 'METAL' THEN 2 ELSE 1 END) as composite_feature_score
     
FROM feature_combinations
ORDER BY product_id;
```

### Time-Based Categorical Binning

#### 1. Temporal Pattern Binning

```sql
-- Customer activity temporal binning for behavioral features
WITH customer_activity AS (
    SELECT 
        customer_id,
        activity_type,
        activity_timestamp,
        activity_value,
        
        -- Extract temporal components
        EXTRACT(HOUR FROM activity_timestamp) as activity_hour,
        EXTRACT(DOW FROM activity_timestamp) as day_of_week,
        EXTRACT(MONTH FROM activity_timestamp) as activity_month,
        EXTRACT(QUARTER FROM activity_timestamp) as activity_quarter,
        
        -- Time-based categorization
        CASE 
            WHEN EXTRACT(HOUR FROM activity_timestamp) BETWEEN 9 AND 17 THEN 'BUSINESS_HOURS'
            WHEN EXTRACT(HOUR FROM activity_timestamp) BETWEEN 6 AND 8 
                OR EXTRACT(HOUR FROM activity_timestamp) BETWEEN 18 AND 22 THEN 'PEAK_PERSONAL_HOURS'
            ELSE 'OFF_HOURS'
        END as time_category,
        
        -- Day type binning
        CASE 
            WHEN EXTRACT(DOW FROM activity_timestamp) IN (1, 2, 3, 4, 5) THEN 'WEEKDAY'
            WHEN EXTRACT(DOW FROM activity_timestamp) = 6 THEN 'SATURDAY'
            WHEN EXTRACT(DOW FROM activity_timestamp) = 0 THEN 'SUNDAY'
        END as day_type,
        
        -- Season binning
        CASE 
            WHEN EXTRACT(MONTH FROM activity_timestamp) IN (12, 1, 2) THEN 'WINTER'
            WHEN EXTRACT(MONTH FROM activity_timestamp) IN (3, 4, 5) THEN 'SPRING'
            WHEN EXTRACT(MONTH FROM activity_timestamp) IN (6, 7, 8) THEN 'SUMMER'
            WHEN EXTRACT(MONTH FROM activity_timestamp) IN (9, 10, 11) THEN 'FALL'
        END as season,
        
        -- Shopping period binning
        CASE 
            WHEN EXTRACT(MONTH FROM activity_timestamp) = 11 
                AND EXTRACT(DAY FROM activity_timestamp) >= 25 THEN 'BLACK_FRIDAY'
            WHEN EXTRACT(MONTH FROM activity_timestamp) = 12 
                AND EXTRACT(DAY FROM activity_timestamp) BETWEEN 15 AND 25 THEN 'CHRISTMAS_SHOPPING'
            WHEN EXTRACT(MONTH FROM activity_timestamp) = 1 
                AND EXTRACT(DAY FROM activity_timestamp) <= 15 THEN 'NEW_YEAR_SALES'
            WHEN EXTRACT(MONTH FROM activity_timestamp) = 2 
                AND EXTRACT(DAY FROM activity_timestamp) BETWEEN 10 AND 16 THEN 'VALENTINES'
            WHEN EXTRACT(MONTH FROM activity_timestamp) = 5 
                AND EXTRACT(DAY FROM activity_timestamp) BETWEEN 5 AND 12 THEN 'MOTHERS_DAY'
            WHEN EXTRACT(MONTH FROM activity_timestamp) = 6 
                AND EXTRACT(DAY FROM activity_timestamp) BETWEEN 15 AND 22 THEN 'FATHERS_DAY'
            WHEN EXTRACT(MONTH FROM activity_timestamp) = 7 
                AND EXTRACT(DAY FROM activity_timestamp) BETWEEN 1 AND 7 THEN 'JULY_4TH'
            WHEN EXTRACT(MONTH FROM activity_timestamp) = 8 
                AND EXTRACT(DAY FROM activity_timestamp) BETWEEN 15 AND 31 THEN 'BACK_TO_SCHOOL'
            ELSE 'REGULAR_PERIOD'
        END as shopping_period
        
    FROM customer_activities
    WHERE activity_timestamp >= CURRENT_DATE - INTERVAL '24 months'
),
activity_patterns AS (
    SELECT 
        customer_id,
        
        -- Aggregate temporal patterns
        COUNT(*) as total_activities,
        COUNT(*) FILTER (WHERE time_category = 'BUSINESS_HOURS') as business_hour_activities,
        COUNT(*) FILTER (WHERE day_type = 'WEEKDAY') as weekday_activities,
        COUNT(*) FILTER (WHERE shopping_period != 'REGULAR_PERIOD') as holiday_activities,
        
        -- Calculate percentages
        ROUND(
            COUNT(*) FILTER (WHERE time_category = 'BUSINESS_HOURS') * 100.0 / COUNT(*), 2
        ) as business_hour_percentage,
        
        ROUND(
            COUNT(*) FILTER (WHERE day_type = 'WEEKDAY') * 100.0 / COUNT(*), 2
        ) as weekday_percentage,
        
        -- Most common patterns
        MODE() WITHIN GROUP (ORDER BY time_category) as primary_time_pattern,
        MODE() WITHIN GROUP (ORDER BY day_type) as primary_day_pattern,
        MODE() WITHIN GROUP (ORDER BY season) as primary_season,
        
        -- Activity distribution
        COUNT(DISTINCT activity_type) as activity_type_diversity,
        ARRAY_AGG(DISTINCT activity_type ORDER BY activity_type) as activity_types,
        
        -- Seasonal behavior
        COUNT(*) FILTER (WHERE season = 'WINTER') as winter_activities,
        COUNT(*) FILTER (WHERE season = 'SPRING') as spring_activities,
        COUNT(*) FILTER (WHERE season = 'SUMMER') as summer_activities,
        COUNT(*) FILTER (WHERE season = 'FALL') as fall_activities
        
    FROM customer_activity
    GROUP BY customer_id
    HAVING COUNT(*) >= 10  -- Minimum activity threshold
),
customer_temporal_bins AS (
    SELECT 
        customer_id,
        total_activities,
        
        -- Behavioral time pattern binning
        CASE 
            WHEN business_hour_percentage >= 70 THEN 'BUSINESS_FOCUSED'
            WHEN business_hour_percentage <= 30 THEN 'PERSONAL_TIME_FOCUSED'
            ELSE 'MIXED_TIME_PATTERN'
        END as time_behavior_bin,
        
        -- Weekend vs weekday preference
        CASE 
            WHEN weekday_percentage >= 80 THEN 'WEEKDAY_DOMINANT'
            WHEN weekday_percentage <= 50 THEN 'WEEKEND_DOMINANT'
            ELSE 'BALANCED_WEEK_PATTERN'
        END as weekly_behavior_bin,
        
        -- Activity diversity binning
        CASE 
            WHEN activity_type_diversity >= 5 THEN 'HIGH_DIVERSITY'
            WHEN activity_type_diversity >= 3 THEN 'MEDIUM_DIVERSITY'
            ELSE 'LOW_DIVERSITY'
        END as activity_diversity_bin,
        
        -- Seasonal preference binning
        CASE 
            WHEN winter_activities = GREATEST(winter_activities, spring_activities, summer_activities, fall_activities) 
                THEN 'WINTER_PEAK'
            WHEN spring_activities = GREATEST(winter_activities, spring_activities, summer_activities, fall_activities) 
                THEN 'SPRING_PEAK'
            WHEN summer_activities = GREATEST(winter_activities, spring_activities, summer_activities, fall_activities) 
                THEN 'SUMMER_PEAK'
            WHEN fall_activities = GREATEST(winter_activities, spring_activities, summer_activities, fall_activities) 
                THEN 'FALL_PEAK'
            ELSE 'NO_SEASONAL_PREFERENCE'
        END as seasonal_preference_bin,
        
        -- Overall activity level binning
        CASE 
            WHEN total_activities >= 100 THEN 'HIGHLY_ACTIVE'
            WHEN total_activities >= 50 THEN 'MODERATELY_ACTIVE'
            WHEN total_activities >= 20 THEN 'SOMEWHAT_ACTIVE'
            ELSE 'LOW_ACTIVITY'
        END as activity_level_bin,
        
        -- Create composite behavioral segment
        CONCAT(
            CASE 
                WHEN business_hour_percentage >= 70 THEN 'BUS'
                WHEN business_hour_percentage <= 30 THEN 'PER'
                ELSE 'MIX'
            END,
            '_',
            CASE 
                WHEN weekday_percentage >= 80 THEN 'WKD'
                WHEN weekday_percentage <= 50 THEN 'WKE'
                ELSE 'BAL'
            END,
            '_',
            CASE 
                WHEN activity_type_diversity >= 5 THEN 'HDV'
                WHEN activity_type_diversity >= 3 THEN 'MDV'
                ELSE 'LDV'
            END
        ) as behavioral_segment_code
        
    FROM activity_patterns
)
SELECT 
    customer_id,
    total_activities,
    time_behavior_bin,
    weekly_behavior_bin,
    activity_diversity_bin,
    seasonal_preference_bin,
    activity_level_bin,
    behavioral_segment_code,
    
    -- Create final customer temporal segment
    CASE 
        WHEN time_behavior_bin = 'BUSINESS_FOCUSED' AND activity_level_bin = 'HIGHLY_ACTIVE' 
            THEN 'PROFESSIONAL_POWER_USER'
        WHEN time_behavior_bin = 'PERSONAL_TIME_FOCUSED' AND activity_diversity_bin = 'HIGH_DIVERSITY' 
            THEN 'LIFESTYLE_ENTHUSIAST'
        WHEN weekly_behavior_bin = 'WEEKEND_DOMINANT' AND activity_level_bin IN ('HIGHLY_ACTIVE', 'MODERATELY_ACTIVE') 
            THEN 'WEEKEND_WARRIOR'
        WHEN activity_level_bin = 'HIGHLY_ACTIVE' AND activity_diversity_bin = 'HIGH_DIVERSITY' 
            THEN 'SUPER_ENGAGED_USER'
        WHEN activity_level_bin = 'LOW_ACTIVITY' 
            THEN 'MINIMAL_ENGAGEMENT_USER'
        ELSE 'STANDARD_USER'
    END as final_temporal_segment,
    
    -- Numeric encoding for ML models
    CASE time_behavior_bin
        WHEN 'BUSINESS_FOCUSED' THEN 3
        WHEN 'MIXED_TIME_PATTERN' THEN 2
        WHEN 'PERSONAL_TIME_FOCUSED' THEN 1
    END as time_behavior_score,
    
    CASE activity_level_bin
        WHEN 'HIGHLY_ACTIVE' THEN 4
        WHEN 'MODERATELY_ACTIVE' THEN 3
        WHEN 'SOMEWHAT_ACTIVE' THEN 2
        WHEN 'LOW_ACTIVITY' THEN 1
    END as activity_level_score
    
FROM customer_temporal_bins
ORDER BY total_activities DESC;
```

### One-Hot Encoding and Feature Engineering

#### 1. Converting Binned Categories to ML Features

```sql
-- Create one-hot encoded features from categorical bins
WITH customer_segments AS (
    SELECT 
        customer_id,
        
        -- Geographic binning
        CASE 
            WHEN state IN ('CA', 'NY', 'TX', 'FL') THEN 'MAJOR_STATE'
            WHEN state IN ('WA', 'OR', 'NV', 'AZ', 'CO') THEN 'WEST_REGION'
            WHEN state IN ('IL', 'OH', 'MI', 'PA') THEN 'MIDWEST_REGION'
            ELSE 'OTHER_STATES'
        END as geographic_segment,
        
        -- Age group binning
        CASE 
            WHEN age BETWEEN 18 AND 25 THEN 'GEN_Z'
            WHEN age BETWEEN 26 AND 40 THEN 'MILLENNIAL'
            WHEN age BETWEEN 41 AND 55 THEN 'GEN_X'
            WHEN age BETWEEN 56 AND 70 THEN 'BOOMER'
            ELSE 'SENIOR'
        END as age_segment,
        
        -- Income binning
        CASE 
            WHEN annual_income < 40000 THEN 'LOW_INCOME'
            WHEN annual_income < 80000 THEN 'MIDDLE_INCOME'
            WHEN annual_income < 120000 THEN 'UPPER_MIDDLE_INCOME'
            ELSE 'HIGH_INCOME'
        END as income_segment
        
    FROM customers
    WHERE age IS NOT NULL AND annual_income IS NOT NULL AND state IS NOT NULL
)
SELECT 
    customer_id,
    
    -- One-hot encoding for geographic segments
    CASE WHEN geographic_segment = 'MAJOR_STATE' THEN 1 ELSE 0 END as is_major_state,
    CASE WHEN geographic_segment = 'WEST_REGION' THEN 1 ELSE 0 END as is_west_region,
    CASE WHEN geographic_segment = 'MIDWEST_REGION' THEN 1 ELSE 0 END as is_midwest_region,
    CASE WHEN geographic_segment = 'OTHER_STATES' THEN 1 ELSE 0 END as is_other_states,
    
    -- One-hot encoding for age segments
    CASE WHEN age_segment = 'GEN_Z' THEN 1 ELSE 0 END as is_gen_z,
    CASE WHEN age_segment = 'MILLENNIAL' THEN 1 ELSE 0 END as is_millennial,
    CASE WHEN age_segment = 'GEN_X' THEN 1 ELSE 0 END as is_gen_x,
    CASE WHEN age_segment = 'BOOMER' THEN 1 ELSE 0 END as is_boomer,
    CASE WHEN age_segment = 'SENIOR' THEN 1 ELSE 0 END as is_senior,
    
    -- One-hot encoding for income segments
    CASE WHEN income_segment = 'LOW_INCOME' THEN 1 ELSE 0 END as is_low_income,
    CASE WHEN income_segment = 'MIDDLE_INCOME' THEN 1 ELSE 0 END as is_middle_income,
    CASE WHEN income_segment = 'UPPER_MIDDLE_INCOME' THEN 1 ELSE 0 END as is_upper_middle_income,
    CASE WHEN income_segment = 'HIGH_INCOME' THEN 1 ELSE 0 END as is_high_income,
    
    -- Create feature vectors as arrays
    ARRAY[
        CASE WHEN geographic_segment = 'MAJOR_STATE' THEN 1 ELSE 0 END,
        CASE WHEN geographic_segment = 'WEST_REGION' THEN 1 ELSE 0 END,
        CASE WHEN geographic_segment = 'MIDWEST_REGION' THEN 1 ELSE 0 END,
        CASE WHEN geographic_segment = 'OTHER_STATES' THEN 1 ELSE 0 END
    ] as geographic_feature_vector,
    
    ARRAY[
        CASE WHEN age_segment = 'GEN_Z' THEN 1 ELSE 0 END,
        CASE WHEN age_segment = 'MILLENNIAL' THEN 1 ELSE 0 END,
        CASE WHEN age_segment = 'GEN_X' THEN 1 ELSE 0 END,
        CASE WHEN age_segment = 'BOOMER' THEN 1 ELSE 0 END,
        CASE WHEN age_segment = 'SENIOR' THEN 1 ELSE 0 END
    ] as age_feature_vector,
    
    -- JSON representation for ML frameworks
    jsonb_build_object(
        'geographic_features', jsonb_build_object(
            'is_major_state', CASE WHEN geographic_segment = 'MAJOR_STATE' THEN 1 ELSE 0 END,
            'is_west_region', CASE WHEN geographic_segment = 'WEST_REGION' THEN 1 ELSE 0 END,
            'is_midwest_region', CASE WHEN geographic_segment = 'MIDWEST_REGION' THEN 1 ELSE 0 END,
            'is_other_states', CASE WHEN geographic_segment = 'OTHER_STATES' THEN 1 ELSE 0 END
        ),
        'demographic_features', jsonb_build_object(
            'is_gen_z', CASE WHEN age_segment = 'GEN_Z' THEN 1 ELSE 0 END,
            'is_millennial', CASE WHEN age_segment = 'MILLENNIAL' THEN 1 ELSE 0 END,
            'is_gen_x', CASE WHEN age_segment = 'GEN_X' THEN 1 ELSE 0 END,
            'is_boomer', CASE WHEN age_segment = 'BOOMER' THEN 1 ELSE 0 END,
            'is_senior', CASE WHEN age_segment = 'SENIOR' THEN 1 ELSE 0 END
        ),
        'income_features', jsonb_build_object(
            'is_low_income', CASE WHEN income_segment = 'LOW_INCOME' THEN 1 ELSE 0 END,
            'is_middle_income', CASE WHEN income_segment = 'MIDDLE_INCOME' THEN 1 ELSE 0 END,
            'is_upper_middle_income', CASE WHEN income_segment = 'UPPER_MIDDLE_INCOME' THEN 1 ELSE 0 END,
            'is_high_income', CASE WHEN income_segment = 'HIGH_INCOME' THEN 1 ELSE 0 END
        )
    ) as ml_features_json,
    
    -- Interaction features (combinations of categories)
    CONCAT(geographic_segment, '_', age_segment) as geo_age_interaction,
    CONCAT(age_segment, '_', income_segment) as age_income_interaction,
    CONCAT(geographic_segment, '_', income_segment) as geo_income_interaction,
    
    -- Multi-hot encoding for interaction features
    CASE WHEN geographic_segment = 'MAJOR_STATE' AND age_segment = 'MILLENNIAL' THEN 1 ELSE 0 END as major_state_millennial,
    CASE WHEN income_segment = 'HIGH_INCOME' AND age_segment IN ('GEN_X', 'BOOMER') THEN 1 ELSE 0 END as high_income_mature,
    CASE WHEN geographic_segment = 'WEST_REGION' AND income_segment IN ('UPPER_MIDDLE_INCOME', 'HIGH_INCOME') THEN 1 ELSE 0 END as west_affluent
    
FROM customer_segments
ORDER BY customer_id;
```

### Real-World Applications

1. **Customer Segmentation**: Grouping customer attributes for targeted marketing
2. **Product Categorization**: Organizing products for recommendation systems
3. **Geographic Analysis**: Binning locations for regional modeling
4. **Behavioral Analysis**: Grouping user actions for pattern recognition
5. **Risk Assessment**: Categorizing factors for credit scoring models

### Best Practices

1. **Domain Expertise**: Use business knowledge to create meaningful bins
2. **Statistical Validation**: Ensure bins have sufficient sample sizes
3. **Consistency**: Apply same binning logic across training and production
4. **Interpretability**: Create bins that business stakeholders can understand
5. **Regular Review**: Update binning logic as business requirements evolve

### Common Pitfalls

1. **Over-Binning**: Creating too many categories with sparse data
2. **Arbitrary Grouping**: Combining categories without logical reasoning
3. **Data Leakage**: Using future information in historical binning
4. **Imbalanced Bins**: Creating categories with vastly different frequencies
5. **Loss of Signal**: Combining categories that have different predictive power

### Performance Considerations

- **Index Strategy**: Ensure binned columns are properly indexed
- **Memory Efficiency**: Consider storage implications of categorical expansions
- **Query Optimization**: Optimize CASE statements for large datasets
- **Maintenance**: Regular monitoring of bin distributions
- **Scalability**: Design binning approaches that scale with data growth

---

## Question 9

**Describe SQL techniques to perform data sampling.**

**Answer:**

### Theory

Data sampling in SQL is the process of selecting a subset of data from a larger dataset for analysis, testing, or model training. Effective sampling is crucial for machine learning applications where working with massive datasets can be computationally expensive or when creating balanced training sets. SQL provides various techniques for random sampling, stratified sampling, systematic sampling, and cluster sampling, each serving different analytical purposes.

**Sampling Benefits:**
- **Performance**: Reduces computation time and memory usage
- **Development Efficiency**: Enables faster model prototyping and testing
- **Statistical Validity**: Maintains representativeness of original data
- **Cost Reduction**: Minimizes processing costs for large-scale analytics
- **Exploratory Analysis**: Facilitates quick data exploration and validation

### Random Sampling Techniques

#### 1. Simple Random Sampling

```sql
-- Basic random sampling using TABLESAMPLE
SELECT 
    customer_id,
    age,
    annual_income,
    purchase_frequency,
    total_spent
FROM customers 
TABLESAMPLE SYSTEM (10);  -- 10% sample using system sampling

-- Alternative: Random sampling with RANDOM() function
SELECT 
    customer_id,
    age,
    annual_income,
    purchase_frequency,
    total_spent
FROM customers 
WHERE RANDOM() < 0.1  -- 10% probability sampling
ORDER BY RANDOM()
LIMIT 1000;  -- Limit to 1000 records

-- Deterministic random sampling with seed for reproducibility
SELECT 
    customer_id,
    age,
    annual_income,
    purchase_frequency,
    total_spent,
    -- Create deterministic hash-based sampling
    ABS(HASHTEXT(customer_id::TEXT)) % 100 as hash_bucket
FROM customers 
WHERE ABS(HASHTEXT(customer_id::TEXT)) % 100 < 10  -- 10% sample
ORDER BY customer_id;

-- ROW_NUMBER() based sampling for exact counts
WITH random_ordered AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (ORDER BY RANDOM()) as random_rank,
        COUNT(*) OVER () as total_count
    FROM customers
)
SELECT 
    customer_id,
    age,
    annual_income,
    purchase_frequency,
    total_spent,
    random_rank,
    total_count,
    ROUND(random_rank * 100.0 / total_count, 2) as percentile_rank
FROM random_ordered
WHERE random_rank <= (total_count * 0.1)  -- Exact 10% sample
ORDER BY random_rank;

-- Advanced random sampling with multiple criteria
WITH eligible_customers AS (
    SELECT 
        customer_id,
        age,
        annual_income,
        purchase_frequency,
        total_spent,
        registration_date,
        last_purchase_date
    FROM customers
    WHERE age BETWEEN 18 AND 80
      AND annual_income > 0
      AND total_spent > 0
      AND last_purchase_date >= CURRENT_DATE - INTERVAL '24 months'
),
stratified_sample AS (
    SELECT 
        *,
        -- Create age-based strata
        CASE 
            WHEN age BETWEEN 18 AND 30 THEN 'YOUNG'
            WHEN age BETWEEN 31 AND 50 THEN 'MIDDLE_AGE'
            ELSE 'MATURE'
        END as age_stratum,
        
        -- Random number for sampling within each stratum
        ROW_NUMBER() OVER (
            PARTITION BY CASE 
                WHEN age BETWEEN 18 AND 30 THEN 'YOUNG'
                WHEN age BETWEEN 31 AND 50 THEN 'MIDDLE_AGE'
                ELSE 'MATURE'
            END 
            ORDER BY RANDOM()
        ) as stratum_rank,
        
        -- Count within each stratum
        COUNT(*) OVER (
            PARTITION BY CASE 
                WHEN age BETWEEN 18 AND 30 THEN 'YOUNG'
                WHEN age BETWEEN 31 AND 50 THEN 'MIDDLE_AGE'
                ELSE 'MATURE'
            END
        ) as stratum_size
        
    FROM eligible_customers
)
SELECT 
    customer_id,
    age,
    annual_income,
    purchase_frequency,
    total_spent,
    age_stratum,
    stratum_rank,
    stratum_size,
    ROUND(stratum_rank * 100.0 / stratum_size, 2) as stratum_percentile
FROM stratified_sample
WHERE stratum_rank <= GREATEST(stratum_size * 0.1, 10)  -- At least 10 samples per stratum
ORDER BY age_stratum, stratum_rank;
```

#### 2. Bernoulli and System Sampling

```sql
-- Bernoulli sampling (row-level probability)
SELECT 
    transaction_id,
    customer_id,
    product_id,
    transaction_amount,
    transaction_date
FROM transactions 
TABLESAMPLE BERNOULLI (5)  -- 5% probability per row
WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months';

-- System sampling (page-level sampling)
SELECT 
    transaction_id,
    customer_id,
    product_id,
    transaction_amount,
    transaction_date
FROM transactions 
TABLESAMPLE SYSTEM (5)  -- ~5% of table blocks
WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months';

-- Comparison of sampling methods with statistics
WITH bernoulli_sample AS (
    SELECT 
        'BERNOULLI' as sampling_method,
        COUNT(*) as sample_size,
        AVG(transaction_amount) as avg_amount,
        STDDEV(transaction_amount) as stddev_amount,
        MIN(transaction_amount) as min_amount,
        MAX(transaction_amount) as max_amount
    FROM transactions 
    TABLESAMPLE BERNOULLI (1)
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
),
system_sample AS (
    SELECT 
        'SYSTEM' as sampling_method,
        COUNT(*) as sample_size,
        AVG(transaction_amount) as avg_amount,
        STDDEV(transaction_amount) as stddev_amount,
        MIN(transaction_amount) as min_amount,
        MAX(transaction_amount) as max_amount
    FROM transactions 
    TABLESAMPLE SYSTEM (1)
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
),
random_sample AS (
    SELECT 
        'RANDOM' as sampling_method,
        COUNT(*) as sample_size,
        AVG(transaction_amount) as avg_amount,
        STDDEV(transaction_amount) as stddev_amount,
        MIN(transaction_amount) as min_amount,
        MAX(transaction_amount) as max_amount
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
      AND RANDOM() < 0.01  -- 1% sample
),
full_population AS (
    SELECT 
        'FULL_POPULATION' as sampling_method,
        COUNT(*) as sample_size,
        AVG(transaction_amount) as avg_amount,
        STDDEV(transaction_amount) as stddev_amount,
        MIN(transaction_amount) as min_amount,
        MAX(transaction_amount) as max_amount
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
)
SELECT 
    sampling_method,
    sample_size,
    ROUND(avg_amount::NUMERIC, 2) as avg_amount,
    ROUND(stddev_amount::NUMERIC, 2) as stddev_amount,
    ROUND(min_amount::NUMERIC, 2) as min_amount,
    ROUND(max_amount::NUMERIC, 2) as max_amount,
    
    -- Calculate sampling efficiency
    ROUND(sample_size * 100.0 / (SELECT sample_size FROM full_population), 4) as sample_percentage
    
FROM (
    SELECT * FROM bernoulli_sample
    UNION ALL SELECT * FROM system_sample
    UNION ALL SELECT * FROM random_sample
    UNION ALL SELECT * FROM full_population
) combined_results
ORDER BY sample_size DESC;
```

### Stratified Sampling

#### 1. Proportional Stratified Sampling

```sql
-- Customer segmentation with proportional stratified sampling
WITH customer_segments AS (
    SELECT 
        customer_id,
        age,
        annual_income,
        geographic_region,
        customer_tier,
        
        -- Create stratification variables
        CASE 
            WHEN annual_income < 40000 THEN 'LOW_INCOME'
            WHEN annual_income < 80000 THEN 'MIDDLE_INCOME'
            ELSE 'HIGH_INCOME'
        END as income_segment,
        
        CASE 
            WHEN age < 30 THEN 'YOUNG'
            WHEN age < 50 THEN 'MIDDLE_AGE'
            ELSE 'MATURE'
        END as age_segment
        
    FROM customers
    WHERE age IS NOT NULL 
      AND annual_income IS NOT NULL
      AND geographic_region IS NOT NULL
),
segment_statistics AS (
    SELECT 
        income_segment,
        age_segment,
        geographic_region,
        COUNT(*) as segment_size,
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as segment_proportion
    FROM customer_segments
    GROUP BY income_segment, age_segment, geographic_region
),
stratified_sampling AS (
    SELECT 
        cs.*,
        ss.segment_size,
        ss.segment_proportion,
        
        -- Assign random numbers within each stratum
        ROW_NUMBER() OVER (
            PARTITION BY cs.income_segment, cs.age_segment, cs.geographic_region 
            ORDER BY RANDOM()
        ) as stratum_rank,
        
        -- Calculate target sample size for each stratum (10% overall sample)
        GREATEST(
            ROUND(ss.segment_size * 0.1)::INTEGER,  -- 10% of stratum
            CASE WHEN ss.segment_size >= 50 THEN 5 ELSE 1 END  -- Minimum samples
        ) as target_stratum_sample_size
        
    FROM customer_segments cs
    JOIN segment_statistics ss ON 
        cs.income_segment = ss.income_segment 
        AND cs.age_segment = ss.age_segment 
        AND cs.geographic_region = ss.geographic_region
)
SELECT 
    customer_id,
    age,
    annual_income,
    geographic_region,
    income_segment,
    age_segment,
    segment_size,
    ROUND(segment_proportion::NUMERIC, 4) as segment_proportion,
    stratum_rank,
    target_stratum_sample_size,
    
    -- Indicate if this record is selected in the sample
    CASE 
        WHEN stratum_rank <= target_stratum_sample_size THEN 'SELECTED'
        ELSE 'NOT_SELECTED'
    END as sample_status,
    
    -- Calculate actual sampling rate for this stratum
    ROUND(
        target_stratum_sample_size * 100.0 / segment_size, 2
    ) as actual_sampling_rate_percent
    
FROM stratified_sampling
WHERE stratum_rank <= target_stratum_sample_size  -- Only return selected samples
ORDER BY income_segment, age_segment, geographic_region, stratum_rank;

-- Validate stratified sampling maintains proportions
WITH sample_validation AS (
    SELECT 
        income_segment,
        age_segment,
        geographic_region,
        COUNT(*) as sample_count
    FROM (
        -- Repeat the stratified sampling logic above
        SELECT 
            cs.*,
            ROW_NUMBER() OVER (
                PARTITION BY cs.income_segment, cs.age_segment, cs.geographic_region 
                ORDER BY RANDOM()
            ) as stratum_rank,
            GREATEST(
                ROUND(ss.segment_size * 0.1)::INTEGER,
                CASE WHEN ss.segment_size >= 50 THEN 5 ELSE 1 END
            ) as target_stratum_sample_size
        FROM customer_segments cs
        JOIN segment_statistics ss ON 
            cs.income_segment = ss.income_segment 
            AND cs.age_segment = ss.age_segment 
            AND cs.geographic_region = ss.geographic_region
    ) stratified_data
    WHERE stratum_rank <= target_stratum_sample_size
    GROUP BY income_segment, age_segment, geographic_region
),
original_proportions AS (
    SELECT 
        income_segment,
        age_segment,
        geographic_region,
        COUNT(*) as original_count,
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as original_proportion
    FROM customer_segments
    GROUP BY income_segment, age_segment, geographic_region
),
sample_proportions AS (
    SELECT 
        income_segment,
        age_segment,
        geographic_region,
        sample_count,
        sample_count * 1.0 / SUM(sample_count) OVER () as sample_proportion
    FROM sample_validation
)
SELECT 
    op.income_segment,
    op.age_segment,
    op.geographic_region,
    op.original_count,
    sp.sample_count,
    ROUND(op.original_proportion::NUMERIC, 4) as original_proportion,
    ROUND(sp.sample_proportion::NUMERIC, 4) as sample_proportion,
    ROUND(ABS(op.original_proportion - sp.sample_proportion)::NUMERIC, 4) as proportion_difference,
    
    -- Quality check
    CASE 
        WHEN ABS(op.original_proportion - sp.sample_proportion) < 0.01 THEN 'GOOD'
        WHEN ABS(op.original_proportion - sp.sample_proportion) < 0.02 THEN 'ACCEPTABLE'
        ELSE 'NEEDS_ADJUSTMENT'
    END as proportionality_quality
    
FROM original_proportions op
JOIN sample_proportions sp ON 
    op.income_segment = sp.income_segment 
    AND op.age_segment = sp.age_segment 
    AND op.geographic_region = sp.geographic_region
ORDER BY proportion_difference DESC;
```

#### 2. Disproportional Stratified Sampling

```sql
-- Fraud detection dataset with oversampling of rare fraud cases
WITH transaction_analysis AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount,
        merchant_category,
        is_fraud,
        transaction_timestamp,
        
        -- Create risk-based strata
        CASE 
            WHEN transaction_amount > 5000 THEN 'HIGH_AMOUNT'
            WHEN transaction_amount > 1000 THEN 'MEDIUM_AMOUNT'
            ELSE 'LOW_AMOUNT'
        END as amount_stratum,
        
        CASE 
            WHEN merchant_category IN ('cash_advance', 'gambling', 'money_transfer') THEN 'HIGH_RISK_MERCHANT'
            WHEN merchant_category IN ('gas_station', 'grocery', 'restaurant') THEN 'LOW_RISK_MERCHANT'
            ELSE 'MEDIUM_RISK_MERCHANT'
        END as merchant_risk_stratum,
        
        -- Time-based stratification
        CASE 
            WHEN EXTRACT(HOUR FROM transaction_timestamp) BETWEEN 2 AND 6 THEN 'UNUSUAL_HOURS'
            WHEN EXTRACT(HOUR FROM transaction_timestamp) BETWEEN 9 AND 17 THEN 'BUSINESS_HOURS'
            ELSE 'EVENING_HOURS'
        END as time_stratum
        
    FROM transactions
    WHERE transaction_timestamp >= CURRENT_DATE - INTERVAL '6 months'
),
stratum_statistics AS (
    SELECT 
        is_fraud,
        amount_stratum,
        merchant_risk_stratum,
        time_stratum,
        COUNT(*) as stratum_size,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
        AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END) as fraud_rate
    FROM transaction_analysis
    GROUP BY is_fraud, amount_stratum, merchant_risk_stratum, time_stratum
),
disproportional_sampling AS (
    SELECT 
        ta.*,
        ss.stratum_size,
        ss.fraud_count,
        ss.fraud_rate,
        
        -- Define different sampling rates based on fraud status and risk
        CASE 
            WHEN ta.is_fraud = TRUE THEN 0.8  -- 80% of all fraud cases
            WHEN ta.amount_stratum = 'HIGH_AMOUNT' AND ta.merchant_risk_stratum = 'HIGH_RISK_MERCHANT' THEN 0.3  -- 30% of high-risk non-fraud
            WHEN ta.amount_stratum = 'HIGH_AMOUNT' OR ta.merchant_risk_stratum = 'HIGH_RISK_MERCHANT' THEN 0.1  -- 10% of medium-risk
            WHEN ta.time_stratum = 'UNUSUAL_HOURS' THEN 0.05  -- 5% of unusual hours
            ELSE 0.01  -- 1% of normal transactions
        END as sampling_rate,
        
        ROW_NUMBER() OVER (
            PARTITION BY ta.is_fraud, ta.amount_stratum, ta.merchant_risk_stratum, ta.time_stratum 
            ORDER BY RANDOM()
        ) as stratum_rank
        
    FROM transaction_analysis ta
    JOIN stratum_statistics ss ON 
        ta.is_fraud = ss.is_fraud
        AND ta.amount_stratum = ss.amount_stratum 
        AND ta.merchant_risk_stratum = ss.merchant_risk_stratum 
        AND ta.time_stratum = ss.time_stratum
)
SELECT 
    transaction_id,
    customer_id,
    transaction_amount,
    merchant_category,
    is_fraud,
    amount_stratum,
    merchant_risk_stratum,
    time_stratum,
    stratum_size,
    sampling_rate,
    stratum_rank,
    ROUND(stratum_size * sampling_rate) as target_sample_size,
    
    -- Sampling weight for model training
    CASE 
        WHEN is_fraud = TRUE THEN 1.0 / 0.8  -- Down-weight oversampled fraud
        WHEN amount_stratum = 'HIGH_AMOUNT' AND merchant_risk_stratum = 'HIGH_RISK_MERCHANT' THEN 1.0 / 0.3
        WHEN amount_stratum = 'HIGH_AMOUNT' OR merchant_risk_stratum = 'HIGH_RISK_MERCHANT' THEN 1.0 / 0.1
        WHEN time_stratum = 'UNUSUAL_HOURS' THEN 1.0 / 0.05
        ELSE 1.0 / 0.01
    END as sample_weight
    
FROM disproportional_sampling
WHERE stratum_rank <= ROUND(stratum_size * sampling_rate)
ORDER BY is_fraud DESC, transaction_amount DESC;

-- Analyze the resulting sample composition
WITH sample_composition AS (
    SELECT 
        is_fraud,
        COUNT(*) as sample_count,
        AVG(transaction_amount) as avg_amount,
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as sample_proportion
    FROM (
        -- Repeat disproportional sampling query above
        SELECT 
            ta.*,
            ROW_NUMBER() OVER (
                PARTITION BY ta.is_fraud, ta.amount_stratum, ta.merchant_risk_stratum, ta.time_stratum 
                ORDER BY RANDOM()
            ) as stratum_rank,
            CASE 
                WHEN ta.is_fraud = TRUE THEN 0.8
                WHEN ta.amount_stratum = 'HIGH_AMOUNT' AND ta.merchant_risk_stratum = 'HIGH_RISK_MERCHANT' THEN 0.3
                WHEN ta.amount_stratum = 'HIGH_AMOUNT' OR ta.merchant_risk_stratum = 'HIGH_RISK_MERCHANT' THEN 0.1
                WHEN ta.time_stratum = 'UNUSUAL_HOURS' THEN 0.05
                ELSE 0.01
            END as sampling_rate,
            ss.stratum_size
        FROM transaction_analysis ta
        JOIN stratum_statistics ss ON 
            ta.is_fraud = ss.is_fraud
            AND ta.amount_stratum = ss.amount_stratum 
            AND ta.merchant_risk_stratum = ss.merchant_risk_stratum 
            AND ta.time_stratum = ss.time_stratum
    ) ds
    WHERE stratum_rank <= ROUND(stratum_size * sampling_rate)
    GROUP BY is_fraud
)
SELECT 
    is_fraud,
    sample_count,
    ROUND(avg_amount::NUMERIC, 2) as avg_transaction_amount,
    ROUND(sample_proportion::NUMERIC, 4) as sample_proportion,
    
    -- Compare to original population
    ROUND(sample_proportion / LAG(sample_proportion) OVER (ORDER BY is_fraud), 2) as fraud_to_normal_ratio
FROM sample_composition
ORDER BY is_fraud;
```

### Systematic Sampling

#### 1. Ordered Systematic Sampling

```sql
-- Time-based systematic sampling for time-series data
WITH time_ordered_data AS (
    SELECT 
        measurement_id,
        sensor_id,
        measurement_value,
        measurement_timestamp,
        ROW_NUMBER() OVER (ORDER BY measurement_timestamp) as time_sequence,
        COUNT(*) OVER () as total_measurements
    FROM sensor_measurements
    WHERE measurement_timestamp >= CURRENT_DATE - INTERVAL '30 days'
      AND measurement_value IS NOT NULL
),
systematic_sample AS (
    SELECT 
        *,
        -- Calculate sampling interval for 5% sample
        ROUND(total_measurements / (total_measurements * 0.05)) as sampling_interval,
        
        -- Determine if this record should be included
        time_sequence % ROUND(total_measurements / (total_measurements * 0.05)) as remainder_check
        
    FROM time_ordered_data
)
SELECT 
    measurement_id,
    sensor_id,
    measurement_value,
    measurement_timestamp,
    time_sequence,
    sampling_interval,
    
    -- Include every Kth record where K is the sampling interval
    CASE 
        WHEN remainder_check = 0 THEN 'SELECTED'
        ELSE 'NOT_SELECTED'
    END as sample_status,
    
    -- Calculate time gaps between selected samples
    LAG(measurement_timestamp) OVER (
        ORDER BY time_sequence
    ) as previous_selected_timestamp,
    
    measurement_timestamp - LAG(measurement_timestamp) OVER (
        ORDER BY time_sequence
    ) as time_gap_from_previous
    
FROM systematic_sample
WHERE remainder_check = 0  -- Only return selected samples
ORDER BY measurement_timestamp;

-- Customer ID-based systematic sampling
WITH customer_systematic AS (
    SELECT 
        customer_id,
        registration_date,
        total_orders,
        lifetime_value,
        
        -- Convert customer_id to integer for systematic sampling
        ABS(HASHTEXT(customer_id::TEXT)) as customer_hash,
        
        -- Create ordered sequence based on hash
        ROW_NUMBER() OVER (ORDER BY ABS(HASHTEXT(customer_id::TEXT))) as hash_sequence,
        COUNT(*) OVER () as total_customers
        
    FROM customers
    WHERE status = 'ACTIVE'
),
systematic_selection AS (
    SELECT 
        *,
        -- 1-in-20 systematic sample (5%)
        20 as sampling_interval,
        hash_sequence % 20 as systematic_remainder,
        
        -- Alternative: 1-in-K where K is calculated
        ROUND(total_customers * 1.0 / (total_customers * 0.05)) as calculated_interval,
        hash_sequence % ROUND(total_customers * 1.0 / (total_customers * 0.05)) as calculated_remainder
        
    FROM customer_systematic
)
SELECT 
    customer_id,
    registration_date,
    total_orders,
    lifetime_value,
    hash_sequence,
    sampling_interval,
    systematic_remainder,
    
    -- Validation: check distribution of selected customers
    CASE 
        WHEN systematic_remainder = 0 THEN 'SELECTED'
        ELSE 'NOT_SELECTED'
    END as sample_status,
    
    -- Show sampling pattern
    hash_sequence - LAG(hash_sequence) OVER (
        WHERE systematic_remainder = 0 
        ORDER BY hash_sequence
    ) as gap_to_previous_selected
    
FROM systematic_selection
WHERE systematic_remainder = 0
ORDER BY hash_sequence
LIMIT 100;
```

### Cluster Sampling

#### 1. Geographic Cluster Sampling

```sql
-- Store-based cluster sampling for retail analysis
WITH store_clusters AS (
    SELECT 
        store_id,
        city,
        state,
        region,
        store_size_category,
        annual_revenue,
        
        -- Create geographic clusters
        CASE 
            WHEN state IN ('CA', 'OR', 'WA') THEN 'WEST_COAST'
            WHEN state IN ('NY', 'NJ', 'CT', 'MA') THEN 'NORTHEAST'
            WHEN state IN ('TX', 'OK', 'AR', 'LA') THEN 'SOUTH_CENTRAL'
            WHEN state IN ('FL', 'GA', 'SC', 'NC') THEN 'SOUTHEAST'
            ELSE 'OTHER_REGIONS'
        END as geographic_cluster,
        
        -- Urban vs rural clustering
        CASE 
            WHEN city IN ('New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix') THEN 'MAJOR_METRO'
            WHEN annual_revenue > 5000000 THEN 'HIGH_REVENUE_AREA'
            ELSE 'STANDARD_AREA'
        END as market_cluster
        
    FROM stores
    WHERE status = 'ACTIVE'
),
cluster_statistics AS (
    SELECT 
        geographic_cluster,
        market_cluster,
        COUNT(*) as cluster_size,
        AVG(annual_revenue) as avg_cluster_revenue,
        STDDEV(annual_revenue) as revenue_stddev,
        MIN(annual_revenue) as min_revenue,
        MAX(annual_revenue) as max_revenue
    FROM store_clusters
    GROUP BY geographic_cluster, market_cluster
),
cluster_selection AS (
    SELECT 
        geographic_cluster,
        market_cluster,
        cluster_size,
        avg_cluster_revenue,
        
        -- Randomly select clusters (not individual stores)
        ROW_NUMBER() OVER (ORDER BY RANDOM()) as cluster_rank,
        COUNT(*) OVER () as total_clusters,
        
        -- Determine if cluster is selected (select 30% of clusters)
        CASE 
            WHEN ROW_NUMBER() OVER (ORDER BY RANDOM()) <= COUNT(*) OVER () * 0.3 
            THEN 'SELECTED_CLUSTER'
            ELSE 'NOT_SELECTED_CLUSTER'
        END as cluster_selection_status
        
    FROM cluster_statistics
),
sampled_stores AS (
    SELECT 
        sc.*,
        cs.cluster_selection_status,
        cs.cluster_rank,
        
        -- Within selected clusters, sample all stores or a subset
        CASE 
            WHEN cs.cluster_selection_status = 'SELECTED_CLUSTER' THEN 'INCLUDED_IN_SAMPLE'
            ELSE 'EXCLUDED_FROM_SAMPLE'
        END as store_sample_status,
        
        -- Calculate cluster-level weights for analysis
        1.0 / 0.3 as cluster_weight  -- Inverse of cluster selection probability
        
    FROM store_clusters sc
    JOIN cluster_selection cs ON 
        sc.geographic_cluster = cs.geographic_cluster 
        AND sc.market_cluster = cs.market_cluster
)
SELECT 
    store_id,
    city,
    state,
    geographic_cluster,
    market_cluster,
    annual_revenue,
    cluster_selection_status,
    store_sample_status,
    cluster_weight,
    
    -- Store characteristics within selected clusters
    COUNT(*) OVER (PARTITION BY geographic_cluster, market_cluster) as stores_in_cluster,
    AVG(annual_revenue) OVER (PARTITION BY geographic_cluster, market_cluster) as cluster_avg_revenue,
    
    -- Calculate store's contribution to cluster
    annual_revenue / SUM(annual_revenue) OVER (PARTITION BY geographic_cluster, market_cluster) as store_revenue_share_in_cluster
    
FROM sampled_stores
WHERE store_sample_status = 'INCLUDED_IN_SAMPLE'
ORDER BY geographic_cluster, market_cluster, store_id;

-- Analysis of cluster sampling quality
WITH cluster_sample_analysis AS (
    SELECT 
        geographic_cluster,
        market_cluster,
        COUNT(CASE WHEN cs.cluster_selection_status = 'SELECTED_CLUSTER' THEN 1 END) as selected_stores,
        COUNT(*) as total_stores_in_cluster,
        AVG(sc.annual_revenue) as original_avg_revenue,
        AVG(CASE WHEN cs.cluster_selection_status = 'SELECTED_CLUSTER' THEN sc.annual_revenue END) as sample_avg_revenue
    FROM store_clusters sc
    JOIN cluster_selection cs ON 
        sc.geographic_cluster = cs.geographic_cluster 
        AND sc.market_cluster = cs.market_cluster
    GROUP BY sc.geographic_cluster, sc.market_cluster
)
SELECT 
    geographic_cluster,
    market_cluster,
    selected_stores,
    total_stores_in_cluster,
    ROUND(original_avg_revenue::NUMERIC, 0) as original_avg_revenue,
    ROUND(sample_avg_revenue::NUMERIC, 0) as sample_avg_revenue,
    ROUND(selected_stores * 100.0 / total_stores_in_cluster, 1) as cluster_selection_rate,
    
    -- Quality metrics
    ROUND(ABS(original_avg_revenue - sample_avg_revenue) / original_avg_revenue * 100, 2) as revenue_bias_percent,
    
    CASE 
        WHEN selected_stores > 0 THEN 'CLUSTER_SELECTED'
        ELSE 'CLUSTER_NOT_SELECTED'
    END as cluster_status
    
FROM cluster_sample_analysis
ORDER BY geographic_cluster, market_cluster;
```

### Temporal Sampling

#### 1. Time-Series and Seasonal Sampling

```sql
-- Multi-period temporal sampling for seasonal analysis
WITH temporal_structure AS (
    SELECT 
        sale_id,
        sale_date,
        customer_id,
        product_id,
        sale_amount,
        
        -- Extract temporal components
        EXTRACT(YEAR FROM sale_date) as sale_year,
        EXTRACT(QUARTER FROM sale_date) as sale_quarter,
        EXTRACT(MONTH FROM sale_date) as sale_month,
        EXTRACT(WEEK FROM sale_date) as sale_week,
        EXTRACT(DOW FROM sale_date) as day_of_week,
        
        -- Create seasonal periods
        CASE 
            WHEN EXTRACT(MONTH FROM sale_date) IN (12, 1, 2) THEN 'WINTER'
            WHEN EXTRACT(MONTH FROM sale_date) IN (3, 4, 5) THEN 'SPRING'
            WHEN EXTRACT(MONTH FROM sale_date) IN (6, 7, 8) THEN 'SUMMER'
            ELSE 'FALL'
        END as season,
        
        -- Holiday periods
        CASE 
            WHEN EXTRACT(MONTH FROM sale_date) = 12 AND EXTRACT(DAY FROM sale_date) BETWEEN 15 AND 31 THEN 'CHRISTMAS'
            WHEN EXTRACT(MONTH FROM sale_date) = 11 AND EXTRACT(DAY FROM sale_date) BETWEEN 22 AND 30 THEN 'BLACK_FRIDAY'
            WHEN EXTRACT(MONTH FROM sale_date) = 2 AND EXTRACT(DAY FROM sale_date) BETWEEN 10 AND 16 THEN 'VALENTINES'
            ELSE 'REGULAR'
        END as holiday_period,
        
        -- Business cycles
        CASE 
            WHEN day_of_week IN (1, 2, 3, 4, 5) THEN 'WEEKDAY'
            ELSE 'WEEKEND'
        END as day_type
        
    FROM sales
    WHERE sale_date >= CURRENT_DATE - INTERVAL '36 months'
      AND sale_amount > 0
),
temporal_sampling_strategy AS (
    SELECT 
        ts.*,
        
        -- Different sampling rates for different periods
        CASE 
            WHEN holiday_period != 'REGULAR' THEN 0.5  -- 50% sample during holidays
            WHEN season = 'WINTER' THEN 0.3  -- 30% sample in winter
            WHEN day_type = 'WEEKEND' THEN 0.4  -- 40% sample on weekends
            ELSE 0.1  -- 10% sample for regular periods
        END as temporal_sampling_rate,
        
        -- Stratified temporal sampling
        ROW_NUMBER() OVER (
            PARTITION BY sale_year, sale_quarter, season, holiday_period, day_type
            ORDER BY RANDOM()
        ) as temporal_stratum_rank,
        
        COUNT(*) OVER (
            PARTITION BY sale_year, sale_quarter, season, holiday_period, day_type
        ) as temporal_stratum_size
        
    FROM temporal_structure
),
temporal_sample AS (
    SELECT 
        *,
        ROUND(temporal_stratum_size * temporal_sampling_rate) as target_stratum_sample_size,
        
        CASE 
            WHEN temporal_stratum_rank <= ROUND(temporal_stratum_size * temporal_sampling_rate) 
            THEN 'SELECTED'
            ELSE 'NOT_SELECTED'
        END as temporal_sample_status,
        
        -- Calculate temporal weight for analysis
        1.0 / temporal_sampling_rate as temporal_weight
        
    FROM temporal_sampling_strategy
)
SELECT 
    sale_id,
    sale_date,
    customer_id,
    product_id,
    sale_amount,
    sale_year,
    sale_quarter,
    season,
    holiday_period,
    day_type,
    temporal_sampling_rate,
    temporal_sample_status,
    temporal_weight,
    
    -- Temporal context
    temporal_stratum_rank,
    target_stratum_sample_size,
    temporal_stratum_size,
    
    -- Quality metrics
    ROUND(temporal_stratum_rank * 100.0 / temporal_stratum_size, 2) as percentile_within_period
    
FROM temporal_sample
WHERE temporal_sample_status = 'SELECTED'
ORDER BY sale_date;

-- Validate temporal sampling maintains seasonal patterns
WITH seasonal_validation AS (
    SELECT 
        season,
        holiday_period,
        day_type,
        COUNT(*) as sample_count,
        AVG(sale_amount) as avg_sale_amount,
        SUM(temporal_weight) as weighted_sample_size,
        
        -- Compare sample to population
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as sample_proportion
        
    FROM temporal_sample
    WHERE temporal_sample_status = 'SELECTED'
    GROUP BY season, holiday_period, day_type
),
population_baseline AS (
    SELECT 
        season,
        holiday_period,
        day_type,
        COUNT(*) as population_count,
        AVG(sale_amount) as population_avg_sale_amount,
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as population_proportion
        
    FROM temporal_structure
    GROUP BY season, holiday_period, day_type
)
SELECT 
    sv.season,
    sv.holiday_period,
    sv.day_type,
    sv.sample_count,
    pb.population_count,
    ROUND(sv.avg_sale_amount::NUMERIC, 2) as sample_avg_amount,
    ROUND(pb.population_avg_sale_amount::NUMERIC, 2) as population_avg_amount,
    ROUND(sv.sample_proportion::NUMERIC, 4) as sample_proportion,
    ROUND(pb.population_proportion::NUMERIC, 4) as population_proportion,
    ROUND(ABS(sv.sample_proportion - pb.population_proportion)::NUMERIC, 4) as proportion_difference,
    
    -- Bias assessment
    CASE 
        WHEN ABS(sv.sample_proportion - pb.population_proportion) < 0.01 THEN 'LOW_BIAS'
        WHEN ABS(sv.sample_proportion - pb.population_proportion) < 0.02 THEN 'MODERATE_BIAS'
        ELSE 'HIGH_BIAS'
    END as temporal_bias_assessment
    
FROM seasonal_validation sv
JOIN population_baseline pb ON 
    sv.season = pb.season 
    AND sv.holiday_period = pb.holiday_period 
    AND sv.day_type = pb.day_type
ORDER BY proportion_difference DESC;
```

### Advanced Sampling Techniques

#### 1. Reservoir Sampling for Streaming Data

```sql
-- Reservoir sampling implementation for large datasets
WITH streaming_data AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_amount,
        transaction_timestamp,
        ROW_NUMBER() OVER (ORDER BY transaction_timestamp) as stream_position
    FROM transactions
    WHERE transaction_timestamp >= CURRENT_DATE - INTERVAL '7 days'
),
reservoir_sampling AS (
    SELECT 
        *,
        -- Reservoir sampling probability: k/n where k=1000 (reservoir size)
        CASE 
            WHEN stream_position <= 1000 THEN 1.0  -- Always include first 1000
            ELSE 1000.0 / stream_position  -- Probability decreases as 1000/n
        END as inclusion_probability,
        
        -- Random value for inclusion decision
        RANDOM() as random_value,
        
        -- Decision logic
        CASE 
            WHEN stream_position <= 1000 THEN 'INCLUDE'
            WHEN RANDOM() <= 1000.0 / stream_position THEN 'INCLUDE'
            ELSE 'EXCLUDE'
        END as reservoir_decision
        
    FROM streaming_data
)
SELECT 
    transaction_id,
    customer_id,
    transaction_amount,
    transaction_timestamp,
    stream_position,
    ROUND(inclusion_probability::NUMERIC, 6) as inclusion_probability,
    reservoir_decision,
    
    -- Running count of included samples
    COUNT(*) FILTER (WHERE reservoir_decision = 'INCLUDE') OVER (
        ORDER BY stream_position 
        ROWS UNBOUNDED PRECEDING
    ) as running_sample_size,
    
    -- Quality metrics
    stream_position - LAG(stream_position) FILTER (WHERE reservoir_decision = 'INCLUDE') OVER (
        ORDER BY stream_position
    ) as gap_since_last_inclusion
    
FROM reservoir_sampling
WHERE reservoir_decision = 'INCLUDE'
ORDER BY stream_position;
```

#### 2. Bootstrap Sampling

```sql
-- Bootstrap sampling for statistical inference
WITH customer_metrics AS (
    SELECT 
        customer_id,
        age,
        annual_income,
        total_purchases,
        avg_order_value
    FROM customers
    WHERE age IS NOT NULL 
      AND annual_income IS NOT NULL
      AND total_purchases > 0
),
bootstrap_samples AS (
    SELECT 
        1 as bootstrap_iteration,
        customer_id,
        age,
        annual_income,
        total_purchases,
        avg_order_value
    FROM (
        SELECT 
            cm.*,
            ROW_NUMBER() OVER (ORDER BY RANDOM()) as random_rank,
            COUNT(*) OVER () as population_size
        FROM customer_metrics cm
    ) random_ordered
    WHERE random_rank <= population_size  -- Sample with replacement (conceptually)
    
    UNION ALL
    
    SELECT 
        2 as bootstrap_iteration,
        customer_id,
        age,
        annual_income,
        total_purchases,
        avg_order_value
    FROM (
        SELECT 
            cm.*,
            ROW_NUMBER() OVER (ORDER BY RANDOM()) as random_rank,
            COUNT(*) OVER () as population_size
        FROM customer_metrics cm
    ) random_ordered
    WHERE random_rank <= population_size
    
    -- Add more iterations as needed...
),
bootstrap_statistics AS (
    SELECT 
        bootstrap_iteration,
        COUNT(*) as sample_size,
        AVG(age) as mean_age,
        STDDEV(age) as stddev_age,
        AVG(annual_income) as mean_income,
        STDDEV(annual_income) as stddev_income,
        AVG(avg_order_value) as mean_order_value,
        STDDEV(avg_order_value) as stddev_order_value,
        
        -- Percentiles
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY annual_income) as income_q1,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY annual_income) as income_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY annual_income) as income_q3
        
    FROM bootstrap_samples
    GROUP BY bootstrap_iteration
)
SELECT 
    bootstrap_iteration,
    sample_size,
    ROUND(mean_age::NUMERIC, 2) as mean_age,
    ROUND(stddev_age::NUMERIC, 2) as stddev_age,
    ROUND(mean_income::NUMERIC, 2) as mean_income,
    ROUND(stddev_income::NUMERIC, 2) as stddev_income,
    ROUND(mean_order_value::NUMERIC, 2) as mean_order_value,
    ROUND(income_median::NUMERIC, 2) as income_median,
    
    -- Confidence intervals (across bootstrap samples)
    ROUND(AVG(mean_income) OVER ()::NUMERIC, 2) as overall_mean_income,
    ROUND(STDDEV(mean_income) OVER ()::NUMERIC, 2) as bootstrap_se_income
    
FROM bootstrap_statistics
ORDER BY bootstrap_iteration;
```

### Real-World Applications

1. **Model Training**: Creating balanced datasets for machine learning
2. **A/B Testing**: Selecting representative test groups
3. **Data Exploration**: Quick analysis of large datasets
4. **Performance Testing**: Creating test datasets of manageable size
5. **Statistical Inference**: Estimating population parameters

### Best Practices

1. **Representativeness**: Ensure samples represent the target population
2. **Reproducibility**: Use seeds for deterministic sampling
3. **Stratification**: Maintain important subgroup proportions
4. **Sample Size**: Calculate adequate sample sizes for statistical power
5. **Bias Minimization**: Use appropriate sampling methods to reduce selection bias

### Common Pitfalls

1. **Selection Bias**: Non-random sampling leading to biased results
2. **Insufficient Sample Size**: Samples too small for reliable inference
3. **Temporal Bias**: Not accounting for time-based patterns
4. **Survivorship Bias**: Only sampling from existing/active records
5. **Cluster Correlation**: Ignoring dependencies within clusters

### Performance Considerations

- **Query Optimization**: Use efficient sampling methods for large tables
- **Index Usage**: Leverage indexes for ordered sampling
- **Memory Management**: Consider memory implications of large samples
- **Parallel Processing**: Utilize parallel query execution for sampling
- **Storage Efficiency**: Optimize storage for sampled datasets

---

## Question 10

**How does SQL play a role in ML model deployment?**

**Answer:**

### Theory

SQL plays a crucial role in machine learning model deployment by serving as the bridge between trained models and production data systems. In production environments, SQL handles real-time feature engineering, model scoring, result storage, monitoring, and integration with existing data infrastructure. Modern SQL databases support various deployment patterns including in-database model execution, real-time feature serving, batch prediction pipelines, and model performance monitoring.

**SQL's Role in ML Deployment:**
- **Feature Engineering**: Real-time computation of model features from raw data
- **Model Scoring**: Executing model predictions within the database
- **Data Integration**: Connecting models with existing data infrastructure
- **Performance Monitoring**: Tracking model accuracy and data drift
- **Scalability**: Handling high-volume prediction requests efficiently

### Real-Time Feature Engineering

#### 1. Production Feature Pipelines

```sql
-- Real-time customer feature engineering for model scoring
CREATE OR REPLACE FUNCTION get_customer_features(p_customer_id INTEGER)
RETURNS TABLE (
    customer_id INTEGER,
    age_group TEXT,
    income_tier TEXT,
    transaction_frequency NUMERIC,
    avg_transaction_amount NUMERIC,
    days_since_last_purchase INTEGER,
    seasonal_spending_pattern TEXT,
    risk_score NUMERIC,
    lifetime_value_quartile INTEGER,
    preferred_category TEXT,
    churn_risk_indicators JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH customer_base AS (
        SELECT 
            c.customer_id,
            c.age,
            c.annual_income,
            c.registration_date,
            c.status
        FROM customers c
        WHERE c.customer_id = p_customer_id
          AND c.status = 'ACTIVE'
    ),
    transaction_features AS (
        SELECT 
            cb.customer_id,
            COUNT(*) as transaction_count,
            AVG(t.amount) as avg_amount,
            STDDEV(t.amount) as amount_volatility,
            MAX(t.transaction_date) as last_transaction_date,
            MIN(t.transaction_date) as first_transaction_date,
            
            -- Recency metrics
            EXTRACT(DAYS FROM CURRENT_DATE - MAX(t.transaction_date)) as days_since_last,
            
            -- Frequency metrics (last 6 months)
            COUNT(*) FILTER (
                WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '6 months'
            ) as recent_transaction_count,
            
            -- Monetary metrics
            SUM(t.amount) as lifetime_value,
            SUM(t.amount) FILTER (
                WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '3 months'
            ) as recent_spending,
            
            -- Category preferences
            MODE() WITHIN GROUP (ORDER BY p.category) as preferred_category,
            COUNT(DISTINCT p.category) as category_diversity
            
        FROM customer_base cb
        LEFT JOIN transactions t ON cb.customer_id = t.customer_id
        LEFT JOIN products p ON t.product_id = p.product_id
        WHERE t.transaction_date >= cb.registration_date
        GROUP BY cb.customer_id
    ),
    seasonal_patterns AS (
        SELECT 
            cb.customer_id,
            
            -- Seasonal spending analysis
            AVG(t.amount) FILTER (
                WHERE EXTRACT(MONTH FROM t.transaction_date) IN (12, 1, 2)
            ) as winter_avg_spending,
            
            AVG(t.amount) FILTER (
                WHERE EXTRACT(MONTH FROM t.transaction_date) IN (6, 7, 8)
            ) as summer_avg_spending,
            
            -- Holiday behavior
            COUNT(*) FILTER (
                WHERE EXTRACT(MONTH FROM t.transaction_date) = 12 
                  AND EXTRACT(DAY FROM t.transaction_date) BETWEEN 15 AND 31
            ) as christmas_transactions,
            
            COUNT(*) FILTER (
                WHERE EXTRACT(MONTH FROM t.transaction_date) = 11 
                  AND EXTRACT(DAY FROM t.transaction_date) BETWEEN 22 AND 30
            ) as black_friday_transactions
            
        FROM customer_base cb
        LEFT JOIN transactions t ON cb.customer_id = t.customer_id
        WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '24 months'
        GROUP BY cb.customer_id
    ),
    risk_indicators AS (
        SELECT 
            cb.customer_id,
            
            -- Calculate risk score components
            CASE 
                WHEN tf.days_since_last > 180 THEN 3
                WHEN tf.days_since_last > 90 THEN 2
                WHEN tf.days_since_last > 30 THEN 1
                ELSE 0
            END as recency_risk,
            
            CASE 
                WHEN tf.recent_transaction_count = 0 THEN 3
                WHEN tf.recent_transaction_count <= 2 THEN 2
                WHEN tf.recent_transaction_count <= 5 THEN 1
                ELSE 0
            END as frequency_risk,
            
            CASE 
                WHEN tf.recent_spending = 0 THEN 3
                WHEN tf.recent_spending < tf.lifetime_value * 0.1 THEN 2
                WHEN tf.recent_spending < tf.lifetime_value * 0.3 THEN 1
                ELSE 0
            END as monetary_risk
            
        FROM customer_base cb
        JOIN transaction_features tf ON cb.customer_id = tf.customer_id
    ),
    feature_assembly AS (
        SELECT 
            cb.customer_id,
            
            -- Demographic features
            CASE 
                WHEN cb.age < 25 THEN 'YOUNG'
                WHEN cb.age < 40 THEN 'MIDDLE_AGE'
                WHEN cb.age < 60 THEN 'MATURE'
                ELSE 'SENIOR'
            END as age_group,
            
            CASE 
                WHEN cb.annual_income < 40000 THEN 'LOW'
                WHEN cb.annual_income < 80000 THEN 'MEDIUM'
                WHEN cb.annual_income < 120000 THEN 'HIGH'
                ELSE 'PREMIUM'
            END as income_tier,
            
            -- Behavioral features
            COALESCE(tf.transaction_count, 0) as transaction_frequency,
            ROUND(COALESCE(tf.avg_amount, 0)::NUMERIC, 2) as avg_transaction_amount,
            COALESCE(tf.days_since_last, 999) as days_since_last_purchase,
            
            -- Seasonal patterns
            CASE 
                WHEN sp.winter_avg_spending > sp.summer_avg_spending * 1.2 THEN 'WINTER_SHOPPER'
                WHEN sp.summer_avg_spending > sp.winter_avg_spending * 1.2 THEN 'SUMMER_SHOPPER'
                WHEN sp.christmas_transactions >= 3 THEN 'HOLIDAY_SHOPPER'
                ELSE 'CONSISTENT_SHOPPER'
            END as seasonal_spending_pattern,
            
            -- Risk assessment
            (ri.recency_risk + ri.frequency_risk + ri.monetary_risk) / 3.0 as risk_score,
            
            -- Lifetime value quartile
            NTILE(4) OVER (ORDER BY tf.lifetime_value) as lifetime_value_quartile,
            
            -- Preferred category
            COALESCE(tf.preferred_category, 'UNKNOWN') as preferred_category,
            
            -- Churn risk indicators as JSON
            jsonb_build_object(
                'recency_risk', ri.recency_risk,
                'frequency_risk', ri.frequency_risk,
                'monetary_risk', ri.monetary_risk,
                'days_since_last', tf.days_since_last,
                'recent_transaction_count', tf.recent_transaction_count,
                'spending_decline', CASE 
                    WHEN tf.recent_spending < tf.lifetime_value * 0.2 THEN TRUE 
                    ELSE FALSE 
                END
            ) as churn_risk_indicators
            
        FROM customer_base cb
        LEFT JOIN transaction_features tf ON cb.customer_id = tf.customer_id
        LEFT JOIN seasonal_patterns sp ON cb.customer_id = sp.customer_id
        LEFT JOIN risk_indicators ri ON cb.customer_id = ri.customer_id
    )
    SELECT 
        fa.customer_id,
        fa.age_group,
        fa.income_tier,
        fa.transaction_frequency,
        fa.avg_transaction_amount,
        fa.days_since_last_purchase,
        fa.seasonal_spending_pattern,
        fa.risk_score,
        fa.lifetime_value_quartile,
        fa.preferred_category,
        fa.churn_risk_indicators
    FROM feature_assembly fa;
END;
$$ LANGUAGE plpgsql;

-- Usage: Real-time feature extraction for model scoring
SELECT * FROM get_customer_features(12345);

-- Batch feature extraction for multiple customers
WITH customer_batch AS (
    SELECT customer_id 
    FROM customers 
    WHERE status = 'ACTIVE' 
      AND last_login_date >= CURRENT_DATE - INTERVAL '7 days'
    LIMIT 1000
)
SELECT 
    cf.*
FROM customer_batch cb
CROSS JOIN LATERAL get_customer_features(cb.customer_id) cf;
```

#### 2. Real-Time Model Scoring Pipeline

```sql
-- Model scoring function that applies trained model logic
CREATE OR REPLACE FUNCTION score_churn_model(
    p_customer_id INTEGER,
    p_model_version TEXT DEFAULT 'v1.2'
)
RETURNS TABLE (
    customer_id INTEGER,
    churn_probability NUMERIC,
    risk_category TEXT,
    confidence_score NUMERIC,
    feature_contributions JSONB,
    model_version TEXT,
    scored_at TIMESTAMP
) AS $$
DECLARE
    v_features RECORD;
    v_churn_prob NUMERIC;
    v_confidence NUMERIC;
    v_risk_category TEXT;
BEGIN
    -- Get customer features
    SELECT * INTO v_features 
    FROM get_customer_features(p_customer_id) 
    LIMIT 1;
    
    -- Apply model logic (simplified decision tree example)
    -- In production, this would call your actual model
    IF v_features.risk_score >= 2.5 THEN
        IF v_features.days_since_last_purchase > 120 THEN
            v_churn_prob := 0.85;
            v_confidence := 0.92;
            v_risk_category := 'HIGH_RISK';
        ELSE
            v_churn_prob := 0.65;
            v_confidence := 0.78;
            v_risk_category := 'MEDIUM_RISK';
        END IF;
    ELSIF v_features.risk_score >= 1.5 THEN
        v_churn_prob := 0.35;
        v_confidence := 0.71;
        v_risk_category := 'MEDIUM_RISK';
    ELSE
        v_churn_prob := 0.15;
        v_confidence := 0.89;
        v_risk_category := 'LOW_RISK';
    END IF;
    
    -- Return scored results
    RETURN QUERY
    SELECT 
        p_customer_id,
        v_churn_prob,
        v_risk_category,
        v_confidence,
        jsonb_build_object(
            'risk_score', v_features.risk_score,
            'days_since_last', v_features.days_since_last_purchase,
            'transaction_frequency', v_features.transaction_frequency,
            'income_tier', v_features.income_tier,
            'age_group', v_features.age_group
        ),
        p_model_version,
        CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Batch scoring with parallel processing
WITH scoring_batch AS (
    SELECT 
        customer_id,
        score_churn_model(customer_id, 'v1.2') as model_results
    FROM customers
    WHERE status = 'ACTIVE'
      AND last_activity_date >= CURRENT_DATE - INTERVAL '30 days'
    LIMIT 10000
)
INSERT INTO model_scores (
    customer_id,
    model_name,
    model_version,
    prediction_value,
    confidence_score,
    risk_category,
    feature_contributions,
    scored_at,
    batch_id
)
SELECT 
    (model_results).customer_id,
    'churn_prediction',
    (model_results).model_version,
    (model_results).churn_probability,
    (model_results).confidence_score,
    (model_results).risk_category,
    (model_results).feature_contributions,
    (model_results).scored_at,
    gen_random_uuid()
FROM scoring_batch;
```

### In-Database Model Execution

#### 1. SQL-Based Model Implementation

```sql
-- Linear regression model implemented in SQL
CREATE OR REPLACE FUNCTION predict_customer_ltv(
    p_customer_id INTEGER
)
RETURNS TABLE (
    customer_id INTEGER,
    predicted_ltv NUMERIC,
    prediction_confidence TEXT,
    feature_weights JSONB
) AS $$
DECLARE
    -- Model coefficients (learned from training)
    c_intercept NUMERIC := 150.0;
    c_age_coeff NUMERIC := 2.5;
    c_income_coeff NUMERIC := 0.003;
    c_frequency_coeff NUMERIC := 45.0;
    c_recency_coeff NUMERIC := -0.8;
    c_category_diversity_coeff NUMERIC := 25.0;
    
    v_features RECORD;
    v_prediction NUMERIC;
    v_confidence TEXT;
BEGIN
    -- Extract customer features
    WITH customer_features AS (
        SELECT 
            c.customer_id,
            c.age,
            c.annual_income,
            COUNT(t.transaction_id) as transaction_frequency,
            EXTRACT(DAYS FROM CURRENT_DATE - MAX(t.transaction_date)) as days_since_last,
            COUNT(DISTINCT p.category) as category_diversity,
            AVG(t.amount) as avg_transaction_amount
        FROM customers c
        LEFT JOIN transactions t ON c.customer_id = t.customer_id
        LEFT JOIN products p ON t.product_id = p.product_id
        WHERE c.customer_id = p_customer_id
          AND t.transaction_date >= CURRENT_DATE - INTERVAL '24 months'
        GROUP BY c.customer_id, c.age, c.annual_income
    )
    SELECT * INTO v_features FROM customer_features;
    
    -- Apply linear regression formula
    v_prediction := c_intercept +
                   (c_age_coeff * v_features.age) +
                   (c_income_coeff * v_features.annual_income) +
                   (c_frequency_coeff * v_features.transaction_frequency) +
                   (c_recency_coeff * v_features.days_since_last) +
                   (c_category_diversity_coeff * v_features.category_diversity);
    
    -- Determine confidence based on feature ranges
    v_confidence := CASE 
        WHEN v_features.transaction_frequency >= 10 AND v_features.days_since_last <= 90 THEN 'HIGH'
        WHEN v_features.transaction_frequency >= 5 AND v_features.days_since_last <= 180 THEN 'MEDIUM'
        ELSE 'LOW'
    END;
    
    RETURN QUERY
    SELECT 
        p_customer_id,
        ROUND(GREATEST(v_prediction, 0)::NUMERIC, 2),
        v_confidence,
        jsonb_build_object(
            'intercept', c_intercept,
            'age_contribution', c_age_coeff * v_features.age,
            'income_contribution', c_income_coeff * v_features.annual_income,
            'frequency_contribution', c_frequency_coeff * v_features.transaction_frequency,
            'recency_contribution', c_recency_coeff * v_features.days_since_last,
            'diversity_contribution', c_category_diversity_coeff * v_features.category_diversity
        );
END;
$$ LANGUAGE plpgsql;

-- Ensemble model combining multiple predictions
CREATE OR REPLACE FUNCTION ensemble_customer_predictions(
    p_customer_id INTEGER
)
RETURNS TABLE (
    customer_id INTEGER,
    churn_probability NUMERIC,
    lifetime_value NUMERIC,
    risk_category TEXT,
    value_segment TEXT,
    ensemble_confidence NUMERIC,
    model_agreement JSONB
) AS $$
DECLARE
    v_churn_result RECORD;
    v_ltv_result RECORD;
    v_ensemble_confidence NUMERIC;
    v_model_agreement JSONB;
BEGIN
    -- Get predictions from multiple models
    SELECT * INTO v_churn_result 
    FROM score_churn_model(p_customer_id) 
    LIMIT 1;
    
    SELECT * INTO v_ltv_result 
    FROM predict_customer_ltv(p_customer_id) 
    LIMIT 1;
    
    -- Calculate ensemble confidence
    v_ensemble_confidence := (v_churn_result.confidence_score + 
                             CASE v_ltv_result.prediction_confidence
                                 WHEN 'HIGH' THEN 0.9
                                 WHEN 'MEDIUM' THEN 0.7
                                 WHEN 'LOW' THEN 0.5
                             END) / 2.0;
    
    -- Create model agreement object
    v_model_agreement := jsonb_build_object(
        'churn_model_confidence', v_churn_result.confidence_score,
        'ltv_model_confidence', v_ltv_result.prediction_confidence,
        'models_agree', CASE 
            WHEN v_churn_result.risk_category = 'HIGH_RISK' AND v_ltv_result.predicted_ltv < 200 THEN TRUE
            WHEN v_churn_result.risk_category = 'LOW_RISK' AND v_ltv_result.predicted_ltv > 500 THEN TRUE
            ELSE FALSE
        END
    );
    
    RETURN QUERY
    SELECT 
        p_customer_id,
        v_churn_result.churn_probability,
        v_ltv_result.predicted_ltv,
        v_churn_result.risk_category,
        CASE 
            WHEN v_ltv_result.predicted_ltv > 1000 THEN 'HIGH_VALUE'
            WHEN v_ltv_result.predicted_ltv > 500 THEN 'MEDIUM_VALUE'
            ELSE 'LOW_VALUE'
        END,
        ROUND(v_ensemble_confidence::NUMERIC, 3),
        v_model_agreement;
END;
$$ LANGUAGE plpgsql;
```

### Model Performance Monitoring

#### 1. Prediction Tracking and Validation

```sql
-- Create comprehensive model monitoring tables
CREATE TABLE model_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id INTEGER NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    predicted_value NUMERIC NOT NULL,
    confidence_score NUMERIC,
    feature_values JSONB,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    batch_id UUID,
    environment VARCHAR(20) DEFAULT 'production'
);

CREATE TABLE model_actuals (
    actual_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id INTEGER NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    actual_value NUMERIC,
    event_timestamp TIMESTAMP NOT NULL,
    source_system VARCHAR(100)
);

CREATE TABLE model_performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC NOT NULL,
    calculation_period_start TIMESTAMP NOT NULL,
    calculation_period_end TIMESTAMP NOT NULL,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Function to calculate model performance metrics
CREATE OR REPLACE FUNCTION calculate_model_performance(
    p_model_name TEXT,
    p_model_version TEXT,
    p_days_back INTEGER DEFAULT 30
)
RETURNS TABLE (
    model_name TEXT,
    model_version TEXT,
    total_predictions INTEGER,
    total_actuals INTEGER,
    coverage_rate NUMERIC,
    mae NUMERIC,
    rmse NUMERIC,
    accuracy NUMERIC,
    precision_score NUMERIC,
    recall_score NUMERIC,
    f1_score NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH prediction_actual_pairs AS (
        SELECT 
            mp.customer_id,
            mp.predicted_value,
            mp.confidence_score,
            ma.actual_value,
            mp.prediction_timestamp,
            ma.event_timestamp,
            
            -- For binary classification (churn prediction)
            CASE WHEN mp.predicted_value >= 0.5 THEN 1 ELSE 0 END as predicted_class,
            CASE WHEN ma.actual_value >= 0.5 THEN 1 ELSE 0 END as actual_class
            
        FROM model_predictions mp
        JOIN model_actuals ma ON mp.customer_id = ma.customer_id
        WHERE mp.model_name = p_model_name
          AND mp.model_version = p_model_version
          AND mp.prediction_timestamp >= CURRENT_DATE - INTERVAL '1 day' * p_days_back
          AND ma.event_timestamp >= mp.prediction_timestamp
          AND ma.event_timestamp <= mp.prediction_timestamp + INTERVAL '90 days'
    ),
    performance_calculations AS (
        SELECT 
            COUNT(*) as total_predictions_with_actuals,
            
            -- Regression metrics
            AVG(ABS(predicted_value - actual_value)) as mean_absolute_error,
            SQRT(AVG(POWER(predicted_value - actual_value, 2))) as root_mean_square_error,
            
            -- Classification metrics
            SUM(CASE WHEN predicted_class = actual_class THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as accuracy,
            
            -- Precision: TP / (TP + FP)
            SUM(CASE WHEN predicted_class = 1 AND actual_class = 1 THEN 1 ELSE 0 END) * 1.0 / 
            NULLIF(SUM(CASE WHEN predicted_class = 1 THEN 1 ELSE 0 END), 0) as precision,
            
            -- Recall: TP / (TP + FN) 
            SUM(CASE WHEN predicted_class = 1 AND actual_class = 1 THEN 1 ELSE 0 END) * 1.0 / 
            NULLIF(SUM(CASE WHEN actual_class = 1 THEN 1 ELSE 0 END), 0) as recall
            
        FROM prediction_actual_pairs
    ),
    total_counts AS (
        SELECT 
            COUNT(*) as total_preds,
            (SELECT COUNT(*) FROM model_actuals 
             WHERE event_timestamp >= CURRENT_DATE - INTERVAL '1 day' * p_days_back) as total_acts
        FROM model_predictions
        WHERE model_name = p_model_name
          AND model_version = p_model_version
          AND prediction_timestamp >= CURRENT_DATE - INTERVAL '1 day' * p_days_back
    )
    SELECT 
        p_model_name,
        p_model_version,
        tc.total_preds,
        tc.total_acts,
        ROUND((pc.total_predictions_with_actuals * 1.0 / tc.total_preds)::NUMERIC, 4) as coverage_rate,
        ROUND(pc.mean_absolute_error::NUMERIC, 4) as mae,
        ROUND(pc.root_mean_square_error::NUMERIC, 4) as rmse,
        ROUND(pc.accuracy::NUMERIC, 4) as accuracy,
        ROUND(pc.precision::NUMERIC, 4) as precision_score,
        ROUND(pc.recall::NUMERIC, 4) as recall_score,
        ROUND((2 * pc.precision * pc.recall / NULLIF(pc.precision + pc.recall, 0))::NUMERIC, 4) as f1_score
    FROM performance_calculations pc
    CROSS JOIN total_counts tc;
END;
$$ LANGUAGE plpgsql;

-- Daily model performance monitoring
WITH daily_performance AS (
    SELECT * FROM calculate_model_performance('churn_prediction', 'v1.2', 7)
)
INSERT INTO model_performance_metrics (
    model_name, model_version, metric_name, metric_value,
    calculation_period_start, calculation_period_end
)
SELECT 
    model_name, model_version, 'accuracy', accuracy,
    CURRENT_DATE - INTERVAL '7 days', CURRENT_DATE
FROM daily_performance
UNION ALL
SELECT 
    model_name, model_version, 'precision', precision_score,
    CURRENT_DATE - INTERVAL '7 days', CURRENT_DATE
FROM daily_performance
UNION ALL
SELECT 
    model_name, model_version, 'recall', recall_score,
    CURRENT_DATE - INTERVAL '7 days', CURRENT_DATE
FROM daily_performance;
```

#### 2. Data Drift Detection

```sql
-- Feature drift monitoring
CREATE OR REPLACE FUNCTION detect_feature_drift(
    p_feature_name TEXT,
    p_days_back INTEGER DEFAULT 30,
    p_baseline_days INTEGER DEFAULT 90
)
RETURNS TABLE (
    feature_name TEXT,
    current_mean NUMERIC,
    baseline_mean NUMERIC,
    mean_drift_percent NUMERIC,
    current_stddev NUMERIC,
    baseline_stddev NUMERIC,
    stddev_drift_percent NUMERIC,
    drift_severity TEXT,
    requires_attention BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH current_period AS (
        SELECT 
            AVG((feature_values->>p_feature_name)::NUMERIC) as curr_mean,
            STDDEV((feature_values->>p_feature_name)::NUMERIC) as curr_stddev,
            COUNT(*) as curr_count
        FROM model_predictions
        WHERE prediction_timestamp >= CURRENT_DATE - INTERVAL '1 day' * p_days_back
          AND feature_values ? p_feature_name
    ),
    baseline_period AS (
        SELECT 
            AVG((feature_values->>p_feature_name)::NUMERIC) as base_mean,
            STDDEV((feature_values->>p_feature_name)::NUMERIC) as base_stddev,
            COUNT(*) as base_count
        FROM model_predictions
        WHERE prediction_timestamp >= CURRENT_DATE - INTERVAL '1 day' * (p_baseline_days + p_days_back)
          AND prediction_timestamp < CURRENT_DATE - INTERVAL '1 day' * p_days_back
          AND feature_values ? p_feature_name
    ),
    drift_analysis AS (
        SELECT 
            cp.curr_mean,
            bp.base_mean,
            ABS(cp.curr_mean - bp.base_mean) / NULLIF(bp.base_mean, 0) * 100 as mean_drift_pct,
            cp.curr_stddev,
            bp.base_stddev,
            ABS(cp.curr_stddev - bp.base_stddev) / NULLIF(bp.base_stddev, 0) * 100 as stddev_drift_pct
        FROM current_period cp
        CROSS JOIN baseline_period bp
    )
    SELECT 
        p_feature_name,
        ROUND(da.curr_mean::NUMERIC, 4),
        ROUND(da.base_mean::NUMERIC, 4),
        ROUND(da.mean_drift_pct::NUMERIC, 2),
        ROUND(da.curr_stddev::NUMERIC, 4),
        ROUND(da.base_stddev::NUMERIC, 4),
        ROUND(da.stddev_drift_pct::NUMERIC, 2),
        CASE 
            WHEN da.mean_drift_pct > 20 OR da.stddev_drift_pct > 30 THEN 'HIGH'
            WHEN da.mean_drift_pct > 10 OR da.stddev_drift_pct > 20 THEN 'MEDIUM'
            WHEN da.mean_drift_pct > 5 OR da.stddev_drift_pct > 10 THEN 'LOW'
            ELSE 'MINIMAL'
        END as drift_severity,
        CASE 
            WHEN da.mean_drift_pct > 15 OR da.stddev_drift_pct > 25 THEN TRUE
            ELSE FALSE
        END as requires_attention
    FROM drift_analysis da;
END;
$$ LANGUAGE plpgsql;

-- Automated drift monitoring for all features
WITH feature_drift_check AS (
    SELECT 
        unnest(ARRAY['age', 'annual_income', 'transaction_frequency', 
                    'days_since_last_purchase', 'risk_score']) as feature_name
),
drift_results AS (
    SELECT 
        fdc.feature_name,
        dfd.*
    FROM feature_drift_check fdc
    CROSS JOIN LATERAL detect_feature_drift(fdc.feature_name, 7, 30) dfd
)
SELECT 
    feature_name,
    current_mean,
    baseline_mean,
    mean_drift_percent,
    drift_severity,
    requires_attention,
    CASE 
        WHEN requires_attention THEN 'ALERT: Feature drift detected'
        ELSE 'OK: Feature stable'
    END as monitoring_status
FROM drift_results
ORDER BY mean_drift_percent DESC;
```

### A/B Testing and Model Comparison

#### 1. Model Experimentation Framework

```sql
-- A/B testing framework for model versions
CREATE TABLE model_experiments (
    experiment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_name VARCHAR(200) NOT NULL,
    control_model_version VARCHAR(50) NOT NULL,
    treatment_model_version VARCHAR(50) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP,
    traffic_split NUMERIC DEFAULT 0.5,
    status VARCHAR(20) DEFAULT 'ACTIVE',
    success_metric VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE experiment_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES model_experiments(experiment_id),
    customer_id INTEGER NOT NULL,
    assigned_group VARCHAR(20) NOT NULL, -- 'control' or 'treatment'
    assignment_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Function to assign customers to experiment groups
CREATE OR REPLACE FUNCTION assign_experiment_group(
    p_customer_id INTEGER,
    p_experiment_id UUID
)
RETURNS TEXT AS $$
DECLARE
    v_traffic_split NUMERIC;
    v_hash_value NUMERIC;
    v_assigned_group TEXT;
BEGIN
    -- Get experiment configuration
    SELECT traffic_split INTO v_traffic_split
    FROM model_experiments
    WHERE experiment_id = p_experiment_id
      AND status = 'ACTIVE';
    
    -- Create deterministic hash-based assignment
    v_hash_value := (ABS(HASHTEXT(p_customer_id::TEXT || p_experiment_id::TEXT)) % 100) / 100.0;
    
    -- Assign group based on hash and traffic split
    IF v_hash_value < v_traffic_split THEN
        v_assigned_group := 'treatment';
    ELSE
        v_assigned_group := 'control';
    END IF;
    
    -- Record assignment
    INSERT INTO experiment_assignments (experiment_id, customer_id, assigned_group)
    VALUES (p_experiment_id, p_customer_id, v_assigned_group)
    ON CONFLICT (experiment_id, customer_id) DO NOTHING;
    
    RETURN v_assigned_group;
END;
$$ LANGUAGE plpgsql;

-- Model scoring with experiment awareness
CREATE OR REPLACE FUNCTION score_with_experiment(
    p_customer_id INTEGER,
    p_experiment_id UUID DEFAULT NULL
)
RETURNS TABLE (
    customer_id INTEGER,
    model_version TEXT,
    experiment_group TEXT,
    churn_probability NUMERIC,
    confidence_score NUMERIC
) AS $$
DECLARE
    v_experiment_group TEXT;
    v_model_version TEXT;
    v_score_result RECORD;
BEGIN
    -- Determine experiment group and model version
    IF p_experiment_id IS NOT NULL THEN
        v_experiment_group := assign_experiment_group(p_customer_id, p_experiment_id);
        
        SELECT 
            CASE WHEN v_experiment_group = 'control' 
                 THEN control_model_version 
                 ELSE treatment_model_version 
            END INTO v_model_version
        FROM model_experiments
        WHERE experiment_id = p_experiment_id;
    ELSE
        v_experiment_group := 'production';
        v_model_version := 'v1.2';  -- Default production model
    END IF;
    
    -- Get model prediction
    SELECT * INTO v_score_result
    FROM score_churn_model(p_customer_id, v_model_version)
    LIMIT 1;
    
    RETURN QUERY
    SELECT 
        p_customer_id,
        v_model_version,
        v_experiment_group,
        v_score_result.churn_probability,
        v_score_result.confidence_score;
END;
$$ LANGUAGE plpgsql;

-- Experiment results analysis
CREATE OR REPLACE FUNCTION analyze_experiment_results(
    p_experiment_id UUID
)
RETURNS TABLE (
    experiment_name TEXT,
    control_model TEXT,
    treatment_model TEXT,
    control_customers INTEGER,
    treatment_customers INTEGER,
    control_accuracy NUMERIC,
    treatment_accuracy NUMERIC,
    accuracy_lift NUMERIC,
    statistical_significance BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH experiment_data AS (
        SELECT 
            me.experiment_name,
            me.control_model_version,
            me.treatment_model_version,
            ea.assigned_group,
            ea.customer_id,
            mp.predicted_value,
            ma.actual_value
        FROM model_experiments me
        JOIN experiment_assignments ea ON me.experiment_id = ea.experiment_id
        JOIN model_predictions mp ON ea.customer_id = mp.customer_id
        LEFT JOIN model_actuals ma ON ea.customer_id = ma.customer_id
        WHERE me.experiment_id = p_experiment_id
          AND mp.prediction_timestamp >= me.start_date
          AND ma.event_timestamp IS NOT NULL
    ),
    group_performance AS (
        SELECT 
            experiment_name,
            control_model_version,
            treatment_model_version,
            assigned_group,
            COUNT(*) as customer_count,
            AVG(CASE WHEN ABS(predicted_value - actual_value) < 0.5 THEN 1.0 ELSE 0.0 END) as accuracy
        FROM experiment_data
        GROUP BY experiment_name, control_model_version, treatment_model_version, assigned_group
    ),
    comparison AS (
        SELECT 
            experiment_name,
            control_model_version,
            treatment_model_version,
            MAX(CASE WHEN assigned_group = 'control' THEN customer_count END) as control_count,
            MAX(CASE WHEN assigned_group = 'treatment' THEN customer_count END) as treatment_count,
            MAX(CASE WHEN assigned_group = 'control' THEN accuracy END) as control_acc,
            MAX(CASE WHEN assigned_group = 'treatment' THEN accuracy END) as treatment_acc
        FROM group_performance
        GROUP BY experiment_name, control_model_version, treatment_model_version
    )
    SELECT 
        c.experiment_name,
        c.control_model_version,
        c.treatment_model_version,
        c.control_count,
        c.treatment_count,
        ROUND(c.control_acc::NUMERIC, 4),
        ROUND(c.treatment_acc::NUMERIC, 4),
        ROUND(((c.treatment_acc - c.control_acc) / c.control_acc * 100)::NUMERIC, 2) as accuracy_lift,
        -- Simplified significance test (z-test approximation)
        CASE WHEN ABS(c.treatment_acc - c.control_acc) > 0.02 AND 
                  c.control_count > 100 AND c.treatment_count > 100 
             THEN TRUE ELSE FALSE END
    FROM comparison c;
END;
$$ LANGUAGE plpgsql;
```

### Real-World Applications

1. **Real-Time Recommendations**: Feature engineering and scoring for product recommendations
2. **Fraud Detection**: Real-time transaction scoring and risk assessment
3. **Dynamic Pricing**: Price optimization based on customer and market features
4. **Churn Prevention**: Proactive customer retention based on churn predictions
5. **Content Personalization**: Personalized content serving based on user models

### Best Practices

1. **Feature Consistency**: Maintain identical feature engineering between training and production
2. **Model Versioning**: Track model versions and enable rollbacks
3. **Performance Monitoring**: Continuous monitoring of model accuracy and data drift
4. **Scalability Planning**: Design for high-volume, low-latency scoring requirements
5. **A/B Testing**: Validate model improvements through controlled experiments

### Common Pitfalls

1. **Training-Serving Skew**: Differences between training and production feature engineering
2. **Data Leakage**: Using future information in production features
3. **Performance Degradation**: Models becoming stale without retraining
4. **Scaling Issues**: Inadequate infrastructure for production workloads
5. **Monitoring Gaps**: Insufficient tracking of model performance in production

### Performance Considerations

- **Query Optimization**: Efficient feature engineering for real-time scoring
- **Caching Strategies**: Cache frequently accessed features and predictions
- **Parallel Processing**: Utilize parallel execution for batch scoring
- **Resource Management**: Monitor and optimize database resource utilization
- **Latency Requirements**: Design for millisecond-level response times when needed

---

## Question 11

**What is the significance of in-database analytics for Machine Learning?**

**Answer:**

### Theory

In-database analytics refers to performing analytical computations directly within the database system rather than extracting data to external processing engines. For machine learning, this approach offers significant advantages including reduced data movement, improved performance, enhanced security, real-time processing capabilities, and simplified architecture. Modern databases support sophisticated analytical functions, statistical computations, and even machine learning model execution within the database engine.

**Key Benefits:**
- **Performance**: Eliminate data transfer bottlenecks
- **Security**: Keep sensitive data within database boundaries  
- **Scalability**: Leverage database's distributed computing capabilities
- **Real-time**: Enable real-time feature engineering and scoring
- **Cost Efficiency**: Reduce infrastructure and data movement costs

### Statistical Functions and ML Computations

#### 1. Advanced Statistical Analysis

```sql
-- Comprehensive statistical analysis for feature engineering
CREATE OR REPLACE FUNCTION customer_statistical_profile(p_customer_id INTEGER)
RETURNS TABLE (
    customer_id INTEGER,
    transaction_stats JSONB,
    temporal_patterns JSONB,
    behavioral_metrics JSONB,
    risk_indicators JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH transaction_base AS (
        SELECT 
            t.customer_id,
            t.amount,
            t.transaction_date,
            t.merchant_category,
            EXTRACT(HOUR FROM t.transaction_timestamp) as hour_of_day,
            EXTRACT(DOW FROM t.transaction_date) as day_of_week,
            EXTRACT(MONTH FROM t.transaction_date) as month_of_year
        FROM transactions t
        WHERE t.customer_id = p_customer_id
          AND t.transaction_date >= CURRENT_DATE - INTERVAL '24 months'
          AND t.status = 'COMPLETED'
    ),
    statistical_measures AS (
        SELECT 
            customer_id,
            -- Central tendency measures
            AVG(amount) as mean_amount,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) as median_amount,
            MODE() WITHIN GROUP (ORDER BY amount) as mode_amount,
            
            -- Dispersion measures
            STDDEV(amount) as stddev_amount,
            VAR_POP(amount) as variance_amount,
            MAX(amount) - MIN(amount) as range_amount,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) - 
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) as iqr_amount,
            
            -- Distribution shape measures
            -- Skewness approximation using Pearson's moment coefficient
            (AVG(amount) - PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)) / 
            NULLIF(STDDEV(amount), 0) as skewness_approx,
            
            -- Coefficient of variation
            STDDEV(amount) / NULLIF(AVG(amount), 0) as coefficient_of_variation,
            
            -- Percentile-based measures
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY amount) as p5_amount,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) as p95_amount,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount) as p99_amount,
            
            -- Count-based measures
            COUNT(*) as total_transactions,
            COUNT(DISTINCT merchant_category) as category_diversity,
            COUNT(DISTINCT DATE_TRUNC('month', transaction_date)) as active_months
            
        FROM transaction_base
        GROUP BY customer_id
    ),
    temporal_analysis AS (
        SELECT 
            customer_id,
            -- Time-based patterns
            AVG(CASE WHEN hour_of_day BETWEEN 9 AND 17 THEN amount ELSE NULL END) as business_hours_avg,
            AVG(CASE WHEN day_of_week IN (6, 0) THEN amount ELSE NULL END) as weekend_avg,
            AVG(CASE WHEN month_of_year IN (11, 12) THEN amount ELSE NULL END) as holiday_season_avg,
            
            -- Frequency patterns
            COUNT(*) FILTER (WHERE hour_of_day BETWEEN 9 AND 17) as business_hours_count,
            COUNT(*) FILTER (WHERE day_of_week IN (6, 0)) as weekend_count,
            COUNT(*) FILTER (WHERE month_of_year IN (11, 12)) as holiday_season_count,
            
            -- Temporal clustering using autocorrelation approximation
            CORR(
                amount, 
                LAG(amount) OVER (ORDER BY transaction_date)
            ) as temporal_autocorrelation
            
        FROM transaction_base
        GROUP BY customer_id
    ),
    behavioral_patterns AS (
        SELECT 
            customer_id,
            -- Spending velocity
            AVG(amount) / NULLIF(
                EXTRACT(DAYS FROM MAX(transaction_date) - MIN(transaction_date)) + 1, 0
            ) as daily_spending_velocity,
            
            -- Transaction regularity (inverse of coefficient of variation of intervals)
            1.0 / NULLIF(
                STDDEV(EXTRACT(DAYS FROM transaction_date - LAG(transaction_date) OVER (ORDER BY transaction_date))) /
                NULLIF(AVG(EXTRACT(DAYS FROM transaction_date - LAG(transaction_date) OVER (ORDER BY transaction_date))), 0), 
                0
            ) as transaction_regularity,
            
            -- Category preference concentration (Herfindahl index)
            SUM(POWER(category_share, 2)) as category_concentration_index
            
        FROM (
            SELECT 
                customer_id,
                transaction_date,
                amount,
                merchant_category,
                COUNT(*) OVER (PARTITION BY customer_id, merchant_category) * 1.0 / 
                COUNT(*) OVER (PARTITION BY customer_id) as category_share
            FROM transaction_base
        ) enriched
        GROUP BY customer_id
    ),
    risk_assessment AS (
        SELECT 
            customer_id,
            -- Volatility measures
            STDDEV(amount) / NULLIF(AVG(amount), 0) as spending_volatility,
            
            -- Outlier detection using IQR method
            COUNT(*) FILTER (
                WHERE amount > (
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) + 
                    1.5 * (PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) - 
                           PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount))
                )
            ) as outlier_transaction_count,
            
            -- Trend analysis using linear regression approximation
            REGR_SLOPE(amount, EXTRACT(EPOCH FROM transaction_date)) * 86400 as daily_trend_slope,
            REGR_R2(amount, EXTRACT(EPOCH FROM transaction_date)) as trend_r_squared,
            
            -- Risk flags
            CASE 
                WHEN MAX(amount) > AVG(amount) + 3 * STDDEV(amount) THEN TRUE 
                ELSE FALSE 
            END as has_extreme_outliers,
            
            CASE 
                WHEN STDDEV(amount) / NULLIF(AVG(amount), 0) > 2.0 THEN TRUE 
                ELSE FALSE 
            END as high_volatility_flag
            
        FROM transaction_base
        GROUP BY customer_id
    )
    SELECT 
        sm.customer_id,
        
        -- Statistical measures as JSON
        jsonb_build_object(
            'mean_amount', ROUND(sm.mean_amount::NUMERIC, 2),
            'median_amount', ROUND(sm.median_amount::NUMERIC, 2),
            'stddev_amount', ROUND(sm.stddev_amount::NUMERIC, 2),
            'coefficient_of_variation', ROUND(sm.coefficient_of_variation::NUMERIC, 4),
            'skewness_approx', ROUND(sm.skewness_approx::NUMERIC, 4),
            'p5_amount', ROUND(sm.p5_amount::NUMERIC, 2),
            'p95_amount', ROUND(sm.p95_amount::NUMERIC, 2),
            'total_transactions', sm.total_transactions,
            'category_diversity', sm.category_diversity,
            'active_months', sm.active_months
        ) as transaction_stats,
        
        -- Temporal patterns as JSON
        jsonb_build_object(
            'business_hours_avg', ROUND(COALESCE(ta.business_hours_avg, 0)::NUMERIC, 2),
            'weekend_avg', ROUND(COALESCE(ta.weekend_avg, 0)::NUMERIC, 2),
            'holiday_season_avg', ROUND(COALESCE(ta.holiday_season_avg, 0)::NUMERIC, 2),
            'business_hours_ratio', ROUND(ta.business_hours_count::NUMERIC / sm.total_transactions, 3),
            'weekend_ratio', ROUND(ta.weekend_count::NUMERIC / sm.total_transactions, 3),
            'temporal_autocorrelation', ROUND(COALESCE(ta.temporal_autocorrelation, 0)::NUMERIC, 4)
        ) as temporal_patterns,
        
        -- Behavioral metrics as JSON
        jsonb_build_object(
            'daily_spending_velocity', ROUND(COALESCE(bp.daily_spending_velocity, 0)::NUMERIC, 4),
            'transaction_regularity', ROUND(COALESCE(bp.transaction_regularity, 0)::NUMERIC, 4),
            'category_concentration', ROUND(COALESCE(bp.category_concentration_index, 0)::NUMERIC, 4)
        ) as behavioral_metrics,
        
        -- Risk indicators as JSON
        jsonb_build_object(
            'spending_volatility', ROUND(COALESCE(ra.spending_volatility, 0)::NUMERIC, 4),
            'outlier_count', ra.outlier_transaction_count,
            'daily_trend_slope', ROUND(COALESCE(ra.daily_trend_slope, 0)::NUMERIC, 6),
            'trend_r_squared', ROUND(COALESCE(ra.trend_r_squared, 0)::NUMERIC, 4),
            'has_extreme_outliers', ra.has_extreme_outliers,
            'high_volatility_flag', ra.high_volatility_flag
        ) as risk_indicators
        
    FROM statistical_measures sm
    LEFT JOIN temporal_analysis ta ON sm.customer_id = ta.customer_id
    LEFT JOIN behavioral_patterns bp ON sm.customer_id = bp.customer_id
    LEFT JOIN risk_assessment ra ON sm.customer_id = ra.customer_id;
END;
$$ LANGUAGE plpgsql;

-- Usage: Get comprehensive statistical profile
SELECT * FROM customer_statistical_profile(12345);

-- Batch processing for model training dataset
CREATE MATERIALIZED VIEW customer_ml_features AS
SELECT 
    csp.customer_id,
    (csp.transaction_stats->>'mean_amount')::NUMERIC as avg_transaction_amount,
    (csp.transaction_stats->>'coefficient_of_variation')::NUMERIC as spending_consistency,
    (csp.temporal_patterns->>'business_hours_ratio')::NUMERIC as business_hours_preference,
    (csp.behavioral_metrics->>'transaction_regularity')::NUMERIC as purchase_regularity,
    (csp.risk_indicators->>'spending_volatility')::NUMERIC as risk_score
FROM (
    SELECT customer_id FROM customers WHERE active = TRUE LIMIT 10000
) active_customers
CROSS JOIN LATERAL customer_statistical_profile(active_customers.customer_id) csp;

-- Refresh materialized view for batch updates
REFRESH MATERIALIZED VIEW CONCURRENTLY customer_ml_features;
```

#### 2. Time Series Analysis and Forecasting

```sql
-- In-database time series analysis for demand forecasting
CREATE OR REPLACE FUNCTION time_series_decomposition(
    p_product_id INTEGER,
    p_days_back INTEGER DEFAULT 365
)
RETURNS TABLE (
    date_value DATE,
    actual_sales NUMERIC,
    trend_component NUMERIC,
    seasonal_component NUMERIC,
    residual_component NUMERIC,
    forecasted_sales NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH daily_sales AS (
        SELECT 
            DATE(order_date) as sale_date,
            SUM(quantity) as daily_quantity,
            ROW_NUMBER() OVER (ORDER BY DATE(order_date)) as day_number
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        WHERE oi.product_id = p_product_id
          AND o.order_date >= CURRENT_DATE - INTERVAL '1 day' * p_days_back
          AND o.status = 'COMPLETED'
        GROUP BY DATE(order_date)
        ORDER BY DATE(order_date)
    ),
    complete_date_series AS (
        SELECT 
            generate_series(
                CURRENT_DATE - INTERVAL '1 day' * p_days_back,
                CURRENT_DATE - INTERVAL '1 day',
                INTERVAL '1 day'
            )::DATE as date_value
    ),
    filled_series AS (
        SELECT 
            cds.date_value,
            COALESCE(ds.daily_quantity, 0) as sales_quantity,
            ROW_NUMBER() OVER (ORDER BY cds.date_value) as day_number
        FROM complete_date_series cds
        LEFT JOIN daily_sales ds ON cds.date_value = ds.sale_date
    ),
    trend_calculation AS (
        SELECT 
            *,
            -- Moving average for trend (7-day window)
            AVG(sales_quantity) OVER (
                ORDER BY day_number 
                ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
            ) as moving_avg_7d,
            
            -- Linear trend using window functions
            REGR_SLOPE(sales_quantity, day_number) OVER () * day_number + 
            REGR_INTERCEPT(sales_quantity, day_number) OVER () as linear_trend
            
        FROM filled_series
    ),
    seasonal_calculation AS (
        SELECT 
            *,
            -- Day of week seasonality
            AVG(sales_quantity) OVER (
                PARTITION BY EXTRACT(DOW FROM date_value)
            ) as dow_seasonal,
            
            -- Day of month seasonality  
            AVG(sales_quantity) OVER (
                PARTITION BY EXTRACT(DAY FROM date_value)
            ) as dom_seasonal,
            
            -- Overall average for normalization
            AVG(sales_quantity) OVER () as overall_avg
            
        FROM trend_calculation
    ),
    decomposed_series AS (
        SELECT 
            date_value,
            sales_quantity as actual_sales,
            linear_trend as trend_component,
            (dow_seasonal + dom_seasonal) / 2.0 - overall_avg as seasonal_component,
            sales_quantity - linear_trend - ((dow_seasonal + dom_seasonal) / 2.0 - overall_avg) as residual_component
        FROM seasonal_calculation
    ),
    forecasting AS (
        SELECT 
            *,
            -- Simple additive forecast
            trend_component + seasonal_component + 
            AVG(residual_component) OVER (
                ORDER BY date_value 
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) as forecast_additive,
            
            -- Exponential smoothing approximation
            sales_quantity * 0.3 + 
            LAG(sales_quantity) OVER (ORDER BY date_value) * 0.7 as forecast_exponential
            
        FROM decomposed_series
    )
    SELECT 
        date_value,
        ROUND(actual_sales::NUMERIC, 2),
        ROUND(trend_component::NUMERIC, 2),
        ROUND(seasonal_component::NUMERIC, 2),
        ROUND(residual_component::NUMERIC, 2),
        ROUND(COALESCE(forecast_additive, actual_sales)::NUMERIC, 2)
    FROM forecasting
    ORDER BY date_value;
END;
$$ LANGUAGE plpgsql;

-- Advanced cohort analysis with retention prediction
WITH cohort_base AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', MIN(order_date)) as cohort_month,
        MIN(order_date) as first_order_date
    FROM orders
    WHERE order_date >= '2023-01-01'
    GROUP BY customer_id
),
cohort_data AS (
    SELECT 
        cb.cohort_month,
        DATE_TRUNC('month', o.order_date) as activity_month,
        EXTRACT(MONTH FROM AGE(DATE_TRUNC('month', o.order_date), cb.cohort_month)) as period_number,
        COUNT(DISTINCT o.customer_id) as active_customers,
        SUM(o.total_amount) as cohort_revenue
    FROM cohort_base cb
    JOIN orders o ON cb.customer_id = o.customer_id
    WHERE o.order_date >= cb.first_order_date
    GROUP BY cb.cohort_month, DATE_TRUNC('month', o.order_date)
),
cohort_sizes AS (
    SELECT 
        cohort_month,
        COUNT(DISTINCT customer_id) as cohort_size
    FROM cohort_base
    GROUP BY cohort_month
),
retention_rates AS (
    SELECT 
        cd.cohort_month,
        cd.period_number,
        cd.active_customers,
        cs.cohort_size,
        cd.active_customers::NUMERIC / cs.cohort_size as retention_rate,
        cd.cohort_revenue / cd.active_customers as revenue_per_customer,
        
        -- Retention prediction using exponential decay
        cs.cohort_size * EXP(-0.1 * cd.period_number) as predicted_retention,
        
        -- Churn rate calculation
        1 - (cd.active_customers::NUMERIC / cs.cohort_size) as churn_rate
    FROM cohort_data cd
    JOIN cohort_sizes cs ON cd.cohort_month = cs.cohort_month
)
SELECT 
    cohort_month,
    period_number,
    cohort_size,
    active_customers,
    ROUND(retention_rate::NUMERIC, 4) as retention_rate,
    ROUND(churn_rate::NUMERIC, 4) as churn_rate,
    ROUND(predicted_retention::NUMERIC, 0) as predicted_active_customers,
    ROUND(revenue_per_customer::NUMERIC, 2) as avg_revenue_per_customer,
    
    -- Lifetime value projection
    ROUND(
        (revenue_per_customer * retention_rate / churn_rate)::NUMERIC, 2
    ) as projected_clv
FROM retention_rates
WHERE period_number <= 12  -- First year analysis
ORDER BY cohort_month, period_number;
```

### Machine Learning Model Implementation

#### 1. Logistic Regression in SQL

```sql
-- In-database logistic regression for churn prediction
CREATE OR REPLACE FUNCTION logistic_regression_predict(
    p_customer_id INTEGER
)
RETURNS TABLE (
    customer_id INTEGER,
    churn_probability NUMERIC,
    feature_contributions JSONB,
    prediction_confidence TEXT
) AS $$
DECLARE
    -- Model coefficients (learned from training data)
    v_intercept NUMERIC := -2.1;
    v_age_coeff NUMERIC := -0.02;
    v_tenure_coeff NUMERIC := -0.01;
    v_transaction_freq_coeff NUMERIC := -0.5;
    v_avg_amount_coeff NUMERIC := 0.0001;
    v_support_contacts_coeff NUMERIC := 0.3;
    v_last_login_days_coeff NUMERIC := 0.05;
    
    v_features RECORD;
    v_linear_combination NUMERIC;
    v_probability NUMERIC;
BEGIN
    -- Extract features for the customer
    WITH customer_features AS (
        SELECT 
            c.customer_id,
            EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) as age,
            EXTRACT(DAYS FROM AGE(CURRENT_DATE, c.registration_date)) as tenure_days,
            COALESCE(t.transaction_count, 0) as transaction_frequency,
            COALESCE(t.avg_transaction_amount, 0) as avg_transaction_amount,
            COALESCE(s.support_contact_count, 0) as support_contacts,
            COALESCE(EXTRACT(DAYS FROM AGE(CURRENT_DATE, l.last_login_date)), 999) as days_since_last_login
        FROM customers c
        LEFT JOIN (
            SELECT 
                customer_id,
                COUNT(*) as transaction_count,
                AVG(amount) as avg_transaction_amount
            FROM transactions
            WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY customer_id
        ) t ON c.customer_id = t.customer_id
        LEFT JOIN (
            SELECT 
                customer_id,
                COUNT(*) as support_contact_count
            FROM support_contacts
            WHERE contact_date >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY customer_id
        ) s ON c.customer_id = s.customer_id
        LEFT JOIN (
            SELECT 
                customer_id,
                MAX(login_date) as last_login_date
            FROM user_logins
            GROUP BY customer_id
        ) l ON c.customer_id = l.customer_id
        WHERE c.customer_id = p_customer_id
    )
    SELECT * INTO v_features FROM customer_features;
    
    -- Calculate linear combination
    v_linear_combination := v_intercept +
        (v_age_coeff * v_features.age) +
        (v_tenure_coeff * v_features.tenure_days) +
        (v_transaction_freq_coeff * v_features.transaction_frequency) +
        (v_avg_amount_coeff * v_features.avg_transaction_amount) +
        (v_support_contacts_coeff * v_features.support_contacts) +
        (v_last_login_days_coeff * v_features.days_since_last_login);
    
    -- Apply logistic function: 1 / (1 + exp(-x))
    v_probability := 1.0 / (1.0 + EXP(-v_linear_combination));
    
    RETURN QUERY
    SELECT 
        p_customer_id,
        ROUND(v_probability::NUMERIC, 4),
        jsonb_build_object(
            'age_contribution', v_age_coeff * v_features.age,
            'tenure_contribution', v_tenure_coeff * v_features.tenure_days,
            'transaction_freq_contribution', v_transaction_freq_coeff * v_features.transaction_frequency,
            'avg_amount_contribution', v_avg_amount_coeff * v_features.avg_transaction_amount,
            'support_contribution', v_support_contacts_coeff * v_features.support_contacts,
            'login_recency_contribution', v_last_login_days_coeff * v_features.days_since_last_login,
            'linear_combination', v_linear_combination
        ),
        CASE 
            WHEN v_probability >= 0.8 THEN 'HIGH_CONFIDENCE'
            WHEN v_probability >= 0.6 THEN 'MEDIUM_CONFIDENCE'
            WHEN v_probability >= 0.4 THEN 'MODERATE_CONFIDENCE'
            WHEN v_probability >= 0.2 THEN 'LOW_CONFIDENCE'
            ELSE 'VERY_LOW_CONFIDENCE'
        END;
END;
$$ LANGUAGE plpgsql;

-- Decision tree implementation for customer segmentation
CREATE OR REPLACE FUNCTION decision_tree_segment(
    p_customer_id INTEGER
)
RETURNS TABLE (
    customer_id INTEGER,
    segment_name TEXT,
    decision_path JSONB,
    segment_characteristics JSONB
) AS $$
DECLARE
    v_features RECORD;
    v_segment TEXT;
    v_path JSONB := '[]'::JSONB;
BEGIN
    -- Get customer features
    WITH customer_data AS (
        SELECT 
            c.customer_id,
            COALESCE(c.annual_income, 0) as annual_income,
            EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.birth_date)) as age,
            COALESCE(t.total_spent_12m, 0) as total_spent_12m,
            COALESCE(t.transaction_count_12m, 0) as transaction_count_12m,
            COALESCE(l.loyalty_score, 0) as loyalty_score
        FROM customers c
        LEFT JOIN (
            SELECT 
                customer_id,
                SUM(amount) as total_spent_12m,
                COUNT(*) as transaction_count_12m
            FROM transactions
            WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
            GROUP BY customer_id
        ) t ON c.customer_id = t.customer_id
        LEFT JOIN customer_loyalty_scores l ON c.customer_id = l.customer_id
        WHERE c.customer_id = p_customer_id
    )
    SELECT * INTO v_features FROM customer_data;
    
    -- Decision tree logic
    IF v_features.annual_income >= 75000 THEN
        v_path := v_path || jsonb_build_object('node', 'income', 'condition', 'income >= 75000', 'result', true);
        
        IF v_features.total_spent_12m >= 5000 THEN
            v_path := v_path || jsonb_build_object('node', 'spending', 'condition', 'spending >= 5000', 'result', true);
            v_segment := 'PREMIUM_HIGH_SPENDER';
        ELSE
            v_path := v_path || jsonb_build_object('node', 'spending', 'condition', 'spending < 5000', 'result', false);
            v_segment := 'PREMIUM_MODERATE_SPENDER';
        END IF;
        
    ELSIF v_features.annual_income >= 40000 THEN
        v_path := v_path || jsonb_build_object('node', 'income', 'condition', '40000 <= income < 75000', 'result', true);
        
        IF v_features.age <= 35 THEN
            v_path := v_path || jsonb_build_object('node', 'age', 'condition', 'age <= 35', 'result', true);
            v_segment := 'YOUNG_PROFESSIONAL';
        ELSE
            v_path := v_path || jsonb_build_object('node', 'age', 'condition', 'age > 35', 'result', false);
            
            IF v_features.loyalty_score >= 7 THEN
                v_path := v_path || jsonb_build_object('node', 'loyalty', 'condition', 'loyalty >= 7', 'result', true);
                v_segment := 'MIDDLE_CLASS_LOYAL';
            ELSE
                v_path := v_path || jsonb_build_object('node', 'loyalty', 'condition', 'loyalty < 7', 'result', false);
                v_segment := 'MIDDLE_CLASS_CASUAL';
            END IF;
        END IF;
        
    ELSE
        v_path := v_path || jsonb_build_object('node', 'income', 'condition', 'income < 40000', 'result', false);
        
        IF v_features.transaction_count_12m >= 20 THEN
            v_path := v_path || jsonb_build_object('node', 'frequency', 'condition', 'transactions >= 20', 'result', true);
            v_segment := 'BUDGET_FREQUENT_BUYER';
        ELSE
            v_path := v_path || jsonb_build_object('node', 'frequency', 'condition', 'transactions < 20', 'result', false);
            v_segment := 'BUDGET_OCCASIONAL_BUYER';
        END IF;
    END IF;
    
    RETURN QUERY
    SELECT 
        p_customer_id,
        v_segment,
        v_path,
        jsonb_build_object(
            'annual_income', v_features.annual_income,
            'age', v_features.age,
            'total_spent_12m', v_features.total_spent_12m,
            'transaction_count_12m', v_features.transaction_count_12m,
            'loyalty_score', v_features.loyalty_score,
            'segment_rules', CASE v_segment
                WHEN 'PREMIUM_HIGH_SPENDER' THEN 'High income (≥$75k) + High spending (≥$5k)'
                WHEN 'PREMIUM_MODERATE_SPENDER' THEN 'High income (≥$75k) + Moderate spending (<$5k)'
                WHEN 'YOUNG_PROFESSIONAL' THEN 'Middle income ($40k-$75k) + Young (≤35)'
                WHEN 'MIDDLE_CLASS_LOYAL' THEN 'Middle income ($40k-$75k) + Mature (>35) + High loyalty (≥7)'
                WHEN 'MIDDLE_CLASS_CASUAL' THEN 'Middle income ($40k-$75k) + Mature (>35) + Low loyalty (<7)'
                WHEN 'BUDGET_FREQUENT_BUYER' THEN 'Low income (<$40k) + Frequent buyer (≥20 transactions)'
                WHEN 'BUDGET_OCCASIONAL_BUYER' THEN 'Low income (<$40k) + Occasional buyer (<20 transactions)'
            END
        );
END;
$$ LANGUAGE plpgsql;
```

### Real-Time Analytics and Feature Serving

#### 1. Streaming Analytics Processing

```sql
-- Real-time feature computation for streaming data
CREATE OR REPLACE FUNCTION real_time_fraud_score(
    p_transaction_id BIGINT,
    p_customer_id INTEGER,
    p_amount NUMERIC,
    p_merchant_id INTEGER,
    p_transaction_timestamp TIMESTAMP
)
RETURNS TABLE (
    transaction_id BIGINT,
    fraud_score NUMERIC,
    risk_level TEXT,
    risk_factors JSONB,
    requires_review BOOLEAN
) AS $$
DECLARE
    v_customer_profile RECORD;
    v_merchant_profile RECORD;
    v_fraud_score NUMERIC := 0;
    v_risk_factors JSONB := '{}'::JSONB;
BEGIN
    -- Get customer profile (cached in materialized view)
    SELECT 
        avg_transaction_amount,
        stddev_transaction_amount,
        max_single_transaction,
        typical_merchants,
        usual_transaction_hours
    INTO v_customer_profile
    FROM customer_transaction_profiles
    WHERE customer_id = p_customer_id;
    
    -- Get merchant risk profile
    SELECT 
        risk_category,
        fraud_rate,
        avg_transaction_amount as merchant_avg_amount
    INTO v_merchant_profile
    FROM merchant_risk_profiles
    WHERE merchant_id = p_merchant_id;
    
    -- Amount-based risk scoring
    IF p_amount > COALESCE(v_customer_profile.max_single_transaction * 2, 1000) THEN
        v_fraud_score := v_fraud_score + 30;
        v_risk_factors := v_risk_factors || jsonb_build_object('unusual_amount', 'Transaction amount significantly higher than customer history');
    ELSIF p_amount > COALESCE(v_customer_profile.avg_transaction_amount + 3 * v_customer_profile.stddev_transaction_amount, 500) THEN
        v_fraud_score := v_fraud_score + 15;
        v_risk_factors := v_risk_factors || jsonb_build_object('elevated_amount', 'Transaction amount above normal range');
    END IF;
    
    -- Merchant risk scoring
    IF v_merchant_profile.risk_category = 'HIGH_RISK' THEN
        v_fraud_score := v_fraud_score + 25;
        v_risk_factors := v_risk_factors || jsonb_build_object('high_risk_merchant', 'Merchant flagged as high risk');
    ELSIF v_merchant_profile.fraud_rate > 0.05 THEN
        v_fraud_score := v_fraud_score + 10;
        v_risk_factors := v_risk_factors || jsonb_build_object('merchant_fraud_history', 'Merchant has elevated fraud rate');
    END IF;
    
    -- Temporal risk scoring
    IF EXTRACT(HOUR FROM p_transaction_timestamp) NOT IN (
        SELECT UNNEST(string_to_array(v_customer_profile.usual_transaction_hours, ','))::INTEGER
    ) THEN
        v_fraud_score := v_fraud_score + 10;
        v_risk_factors := v_risk_factors || jsonb_build_object('unusual_time', 'Transaction outside customer''s usual hours');
    END IF;
    
    -- Velocity-based risk scoring
    WITH recent_transactions AS (
        SELECT COUNT(*) as recent_count
        FROM transactions
        WHERE customer_id = p_customer_id
          AND transaction_timestamp >= p_transaction_timestamp - INTERVAL '1 hour'
          AND transaction_id != p_transaction_id
    )
    SELECT recent_count INTO v_fraud_score
    FROM recent_transactions;
    
    IF v_fraud_score >= 3 THEN
        v_fraud_score := v_fraud_score + 20;
        v_risk_factors := v_risk_factors || jsonb_build_object('high_velocity', 'Multiple transactions in short time period');
    END IF;
    
    RETURN QUERY
    SELECT 
        p_transaction_id,
        v_fraud_score,
        CASE 
            WHEN v_fraud_score >= 50 THEN 'HIGH_RISK'
            WHEN v_fraud_score >= 25 THEN 'MEDIUM_RISK'
            WHEN v_fraud_score >= 10 THEN 'LOW_RISK'
            ELSE 'NORMAL'
        END,
        v_risk_factors,
        CASE WHEN v_fraud_score >= 25 THEN TRUE ELSE FALSE END;
END;
$$ LANGUAGE plpgsql;

-- Batch processing for model features with window functions
CREATE MATERIALIZED VIEW customer_feature_store AS
WITH customer_metrics AS (
    SELECT 
        customer_id,
        -- Recency features
        EXTRACT(DAYS FROM CURRENT_DATE - MAX(transaction_date)) as days_since_last_transaction,
        
        -- Frequency features
        COUNT(*) as total_transactions,
        COUNT(*) / NULLIF(EXTRACT(DAYS FROM MAX(transaction_date) - MIN(transaction_date)) + 1, 0) as transaction_frequency,
        
        -- Monetary features
        SUM(amount) as total_amount,
        AVG(amount) as avg_amount,
        STDDEV(amount) as stddev_amount,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) as median_amount,
        
        -- Diversity features
        COUNT(DISTINCT merchant_id) as merchant_diversity,
        COUNT(DISTINCT product_category) as category_diversity,
        
        -- Temporal features
        MODE() WITHIN GROUP (ORDER BY EXTRACT(HOUR FROM transaction_timestamp)) as preferred_hour,
        MODE() WITHIN GROUP (ORDER BY EXTRACT(DOW FROM transaction_date)) as preferred_day,
        
        -- Trend features
        REGR_SLOPE(amount, EXTRACT(EPOCH FROM transaction_date)) as spending_trend,
        
        -- Seasonality features
        AVG(CASE WHEN EXTRACT(MONTH FROM transaction_date) IN (11, 12) THEN amount END) as holiday_avg_spending,
        AVG(CASE WHEN EXTRACT(DOW FROM transaction_date) IN (6, 0) THEN amount END) as weekend_avg_spending
        
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '24 months'
      AND status = 'COMPLETED'
    GROUP BY customer_id
    HAVING COUNT(*) >= 5  -- Minimum transaction requirement
),
feature_engineering AS (
    SELECT 
        *,
        -- Derived features
        CASE 
            WHEN days_since_last_transaction <= 30 THEN 'ACTIVE'
            WHEN days_since_last_transaction <= 90 THEN 'RECENT'
            WHEN days_since_last_transaction <= 180 THEN 'DORMANT'
            ELSE 'INACTIVE'
        END as activity_status,
        
        -- RFM scoring
        NTILE(5) OVER (ORDER BY days_since_last_transaction DESC) as recency_score,
        NTILE(5) OVER (ORDER BY transaction_frequency) as frequency_score,
        NTILE(5) OVER (ORDER BY total_amount) as monetary_score,
        
        -- Risk indicators
        CASE WHEN stddev_amount / NULLIF(avg_amount, 0) > 2 THEN 1 ELSE 0 END as high_variability_flag,
        CASE WHEN spending_trend < -0.01 THEN 1 ELSE 0 END as declining_spending_flag
        
    FROM customer_metrics
)
SELECT 
    customer_id,
    days_since_last_transaction,
    total_transactions,
    transaction_frequency,
    total_amount,
    avg_amount,
    stddev_amount,
    median_amount,
    merchant_diversity,
    category_diversity,
    activity_status,
    recency_score,
    frequency_score,
    monetary_score,
    (recency_score + frequency_score + monetary_score) as rfm_combined_score,
    high_variability_flag,
    declining_spending_flag,
    CURRENT_TIMESTAMP as feature_computed_at
FROM feature_engineering;

-- Create indexes for fast feature serving
CREATE INDEX idx_customer_feature_store_customer_id 
ON customer_feature_store(customer_id);

CREATE INDEX idx_customer_feature_store_rfm_score 
ON customer_feature_store(rfm_combined_score DESC);

-- Auto-refresh materialized view
CREATE OR REPLACE FUNCTION refresh_customer_features()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY customer_feature_store;
END;
$$ LANGUAGE plpgsql;

-- Schedule automatic refresh (using pg_cron extension)
-- SELECT cron.schedule('refresh-customer-features', '0 2 * * *', 'SELECT refresh_customer_features();');
```

### Performance Optimization and Scalability

#### 1. Parallel Processing and Partitioning

```sql
-- Partitioned tables for large-scale analytics
CREATE TABLE transactions_partitioned (
    transaction_id BIGINT NOT NULL,
    customer_id INTEGER NOT NULL,
    amount NUMERIC(10,2) NOT NULL,
    transaction_date DATE NOT NULL,
    merchant_id INTEGER,
    status VARCHAR(20) DEFAULT 'PENDING'
) PARTITION BY RANGE (transaction_date);

-- Create monthly partitions
CREATE TABLE transactions_2024_01 PARTITION OF transactions_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE transactions_2024_02 PARTITION OF transactions_partitioned
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Parallel processing for feature computation
CREATE OR REPLACE FUNCTION parallel_customer_analysis()
RETURNS TABLE (
    customer_id INTEGER,
    analysis_results JSONB
) AS $$
BEGIN
    -- Set parallel processing parameters
    SET max_parallel_workers_per_gather = 4;
    SET parallel_tuple_cost = 0.1;
    SET parallel_setup_cost = 1000;
    
    RETURN QUERY
    SELECT 
        c.customer_id,
        jsonb_build_object(
            'transaction_summary', t.summary,
            'behavioral_patterns', b.patterns,
            'risk_assessment', r.assessment
        )
    FROM customers c
    PARALLEL (
        -- Parallel transaction analysis
        LEFT JOIN (
            SELECT 
                customer_id,
                jsonb_build_object(
                    'total_count', COUNT(*),
                    'total_amount', SUM(amount),
                    'avg_amount', AVG(amount)
                ) as summary
            FROM transactions_partitioned
            WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
            GROUP BY customer_id
        ) t ON c.customer_id = t.customer_id
    )
    PARALLEL (
        -- Parallel behavioral analysis
        LEFT JOIN (
            SELECT 
                customer_id,
                jsonb_build_object(
                    'preferred_categories', array_agg(DISTINCT category),
                    'peak_hours', MODE() WITHIN GROUP (ORDER BY EXTRACT(HOUR FROM created_at))
                ) as patterns
            FROM user_activities
            WHERE created_at >= CURRENT_DATE - INTERVAL '6 months'
            GROUP BY customer_id
        ) b ON c.customer_id = b.customer_id
    )
    PARALLEL (
        -- Parallel risk assessment
        LEFT JOIN (
            SELECT 
                customer_id,
                jsonb_build_object(
                    'risk_score', AVG(risk_score),
                    'alert_count', COUNT(*) FILTER (WHERE risk_score > 0.7)
                ) as assessment
            FROM risk_assessments
            WHERE assessment_date >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY customer_id
        ) r ON c.customer_id = r.customer_id
    )
    WHERE c.active = TRUE;
END;
$$ LANGUAGE plpgsql;
```

### Real-World Applications

1. **Real-Time Recommendations**: Feature computation and model scoring within database
2. **Fraud Detection**: Real-time transaction analysis and risk scoring
3. **Customer Analytics**: Comprehensive behavioral analysis and segmentation
4. **Demand Forecasting**: Time series analysis and predictive modeling
5. **A/B Testing**: Statistical analysis and experiment evaluation

### Best Practices

1. **Materialize Complex Calculations**: Use materialized views for expensive computations
2. **Partition Large Tables**: Improve query performance and maintenance
3. **Use Parallel Processing**: Leverage database parallelism for analytics
4. **Index Strategically**: Create indexes that support analytical queries
5. **Monitor Performance**: Track query execution and resource usage

### Common Pitfalls

1. **Over-Processing**: Doing too much computation in database vs. application
2. **Memory Limitations**: Complex analytics exceeding available memory
3. **Blocking Operations**: Long-running analytics blocking transactional workloads
4. **Data Skew**: Uneven data distribution affecting parallel processing
5. **Resource Contention**: Analytics competing with operational queries

### Performance Considerations

- **CPU Usage**: Complex analytical functions can be CPU-intensive
- **Memory Requirements**: Window functions and aggregations need sufficient memory
- **I/O Patterns**: Sequential vs. random access patterns for analytics
- **Concurrency**: Balance between analytical and transactional workloads
- **Scalability**: Design for growing data volumes and user base

---

## Question 12

**Explain recursive SQL queries and how they can be used to prepare data for hierarchical Machine Learning algorithms.**

**Answer:**

### Theory

Recursive SQL queries use Common Table Expressions (CTEs) with the `WITH RECURSIVE` clause to process hierarchical or graph-like data structures. These queries are particularly valuable for machine learning applications involving organizational structures, social networks, decision trees, recommendation systems, and any scenario where relationships form hierarchies or networks. Recursive queries enable the transformation of hierarchical data into formats suitable for ML algorithms, feature extraction from graph structures, and preparation of training datasets with ancestral or descendant relationships.

**Key Applications in ML:**
- **Hierarchical Classification**: Preparing taxonomic or organizational data
- **Graph Neural Networks**: Feature extraction from network structures  
- **Recommendation Systems**: User-item relationship traversal
- **Decision Tree Analysis**: Path extraction and rule generation
- **Social Network Analysis**: Influence propagation and community detection

### Hierarchical Data Structures

#### 1. Organizational Hierarchy Analysis

```sql
-- Recursive CTE for organizational hierarchy with ML features
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: Top-level managers (no supervisor)
    SELECT 
        employee_id,
        name,
        title,
        department,
        supervisor_id,
        salary,
        hire_date,
        1 as hierarchy_level,
        name as top_manager,
        employee_id::TEXT as path,
        ARRAY[employee_id] as path_array,
        0 as depth_from_root,
        TRUE as is_manager
    FROM employees
    WHERE supervisor_id IS NULL
    
    UNION ALL
    
    -- Recursive case: Employees with supervisors
    SELECT 
        e.employee_id,
        e.name,
        e.title,
        e.department,
        e.supervisor_id,
        e.salary,
        e.hire_date,
        eh.hierarchy_level + 1,
        eh.top_manager,
        eh.path || ' -> ' || e.name,
        eh.path_array || e.employee_id,
        eh.depth_from_root + 1,
        EXISTS(
            SELECT 1 FROM employees sub 
            WHERE sub.supervisor_id = e.employee_id
        ) as is_manager
    FROM employees e
    INNER JOIN employee_hierarchy eh ON e.supervisor_id = eh.employee_id
),
ml_features AS (
    SELECT 
        eh.*,
        -- Hierarchical features for ML
        CASE 
            WHEN hierarchy_level = 1 THEN 'C_LEVEL'
            WHEN hierarchy_level = 2 THEN 'VP_LEVEL'
            WHEN hierarchy_level = 3 THEN 'DIRECTOR_LEVEL'
            WHEN hierarchy_level = 4 THEN 'MANAGER_LEVEL'
            ELSE 'INDIVIDUAL_CONTRIBUTOR'
        END as executive_band,
        
        -- Team size (direct and indirect reports)
        (SELECT COUNT(*) 
         FROM employee_hierarchy sub 
         WHERE eh.employee_id = ANY(sub.path_array)
           AND sub.employee_id != eh.employee_id
        ) as total_team_size,
        
        -- Direct reports count
        (SELECT COUNT(*) 
         FROM employees sub 
         WHERE sub.supervisor_id = eh.employee_id
        ) as direct_reports,
        
        -- Span of control (management complexity)
        CASE 
            WHEN hierarchy_level <= 2 THEN 'STRATEGIC'
            WHEN hierarchy_level = 3 THEN 'TACTICAL'
            ELSE 'OPERATIONAL'
        END as management_level,
        
        -- Salary relative to level
        salary / NULLIF(
            (SELECT AVG(salary) 
             FROM employee_hierarchy avg_eh 
             WHERE avg_eh.hierarchy_level = eh.hierarchy_level), 0
        ) as salary_level_ratio,
        
        -- Career progression potential
        CASE 
            WHEN hierarchy_level >= 5 THEN 'LIMITED'
            WHEN hierarchy_level >= 3 THEN 'MODERATE'
            ELSE 'HIGH'
        END as promotion_potential,
        
        -- Department diversity in path
        (SELECT COUNT(DISTINCT department) 
         FROM employees path_emp 
         WHERE path_emp.employee_id = ANY(eh.path_array)
        ) as path_department_diversity,
        
        -- Tenure-based features
        EXTRACT(DAYS FROM AGE(CURRENT_DATE, hire_date)) as tenure_days,
        
        -- Performance indicators (derived)
        CASE 
            WHEN salary > (SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) 
                          FROM employee_hierarchy level_eh 
                          WHERE level_eh.hierarchy_level = eh.hierarchy_level) 
            THEN 'HIGH_PERFORMER'
            WHEN salary < (SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary) 
                          FROM employee_hierarchy level_eh 
                          WHERE level_eh.hierarchy_level = eh.hierarchy_level)
            THEN 'UNDERPERFORMER'
            ELSE 'AVERAGE_PERFORMER'
        END as performance_tier
        
    FROM employee_hierarchy eh
),
graph_features AS (
    SELECT 
        mf.*,
        -- Network centrality measures
        ROW_NUMBER() OVER (
            PARTITION BY hierarchy_level 
            ORDER BY total_team_size DESC
        ) as influence_rank_in_level,
        
        -- Structural features
        CASE 
            WHEN total_team_size = 0 THEN 'LEAF_NODE'
            WHEN direct_reports > 0 AND total_team_size = direct_reports THEN 'BRANCH_NODE'
            ELSE 'INTERNAL_NODE'
        END as node_type,
        
        -- Management efficiency ratio
        CASE 
            WHEN total_team_size > 0 
            THEN salary / total_team_size 
            ELSE salary 
        END as cost_per_managed_employee,
        
        -- Hierarchical distance features
        hierarchy_level::NUMERIC / (SELECT MAX(hierarchy_level) FROM ml_features) as normalized_level,
        
        -- Path-based features for sequence models
        array_length(path_array, 1) as path_length,
        path_array[1] as root_manager_id,
        path_array[array_length(path_array, 1)] as immediate_supervisor_id
        
    FROM ml_features mf
)
SELECT 
    employee_id,
    name,
    department,
    executive_band,
    hierarchy_level,
    total_team_size,
    direct_reports,
    management_level,
    ROUND(salary_level_ratio::NUMERIC, 3) as salary_competitiveness,
    promotion_potential,
    performance_tier,
    node_type,
    ROUND(cost_per_managed_employee::NUMERIC, 2) as management_efficiency,
    ROUND(normalized_level::NUMERIC, 3) as hierarchy_position,
    path,
    path_array,
    
    -- Features for classification models
    CASE 
        WHEN performance_tier = 'HIGH_PERFORMER' AND promotion_potential = 'HIGH' THEN 1 
        ELSE 0 
    END as high_potential_flag,
    
    CASE 
        WHEN salary_level_ratio < 0.8 AND performance_tier != 'UNDERPERFORMER' THEN 1 
        ELSE 0 
    END as retention_risk_flag,
    
    -- Features for regression models
    total_team_size as target_team_size,
    salary as target_salary,
    tenure_days as experience_feature
    
FROM graph_features
ORDER BY hierarchy_level, total_team_size DESC;

-- Materialized view for efficient ML feature serving
CREATE MATERIALIZED VIEW employee_ml_features AS
SELECT * FROM (
    -- Above query here
) hierarchy_analysis;

-- Create indexes for ML pipelines
CREATE INDEX idx_employee_ml_level ON employee_ml_features(hierarchy_level);
CREATE INDEX idx_employee_ml_performance ON employee_ml_features(performance_tier);
CREATE INDEX idx_employee_ml_potential ON employee_ml_features(high_potential_flag);
```

#### 2. Product Category Hierarchy for Recommendation Systems

```sql
-- Recursive category hierarchy for collaborative filtering
WITH RECURSIVE category_tree AS (
    -- Root categories
    SELECT 
        category_id,
        category_name,
        parent_category_id,
        1 as level,
        category_name as root_category,
        category_id::TEXT as category_path,
        ARRAY[category_id] as path_ids,
        0 as ancestors_count
    FROM product_categories
    WHERE parent_category_id IS NULL
    
    UNION ALL
    
    -- Child categories
    SELECT 
        pc.category_id,
        pc.category_name,
        pc.parent_category_id,
        ct.level + 1,
        ct.root_category,
        ct.category_path || ' > ' || pc.category_name,
        ct.path_ids || pc.category_id,
        ct.ancestors_count + 1
    FROM product_categories pc
    INNER JOIN category_tree ct ON pc.parent_category_id = ct.category_id
),
category_metrics AS (
    SELECT 
        ct.*,
        -- Product count at each level
        (SELECT COUNT(*) FROM products p WHERE p.category_id = ct.category_id) as direct_product_count,
        
        -- Total products in subtree
        (SELECT COUNT(*) 
         FROM products p 
         WHERE p.category_id IN (
             SELECT descendant.category_id 
             FROM category_tree descendant 
             WHERE ct.category_id = ANY(descendant.path_ids)
         )
        ) as subtree_product_count,
        
        -- Sales performance metrics
        COALESCE(
            (SELECT SUM(oi.quantity * oi.unit_price)
             FROM order_items oi
             JOIN products p ON oi.product_id = p.product_id
             WHERE p.category_id = ct.category_id
               AND oi.order_date >= CURRENT_DATE - INTERVAL '90 days'
            ), 0
        ) as category_revenue_90d,
        
        -- Customer engagement metrics
        (SELECT COUNT(DISTINCT o.customer_id)
         FROM orders o
         JOIN order_items oi ON o.order_id = oi.order_id
         JOIN products p ON oi.product_id = p.product_id
         WHERE p.category_id = ct.category_id
           AND o.order_date >= CURRENT_DATE - INTERVAL '90 days'
        ) as unique_customers_90d
        
    FROM category_tree ct
),
ml_recommendation_features AS (
    SELECT 
        cm.*,
        -- Hierarchical features for recommendation algorithms
        CASE 
            WHEN level = 1 THEN 'L1_ROOT'
            WHEN level = 2 THEN 'L2_MAJOR'
            WHEN level = 3 THEN 'L3_MINOR'
            ELSE 'L4_LEAF'
        END as hierarchy_tier,
        
        -- Popularity features
        subtree_product_count::NUMERIC / NULLIF(
            (SELECT SUM(subtree_product_count) FROM category_metrics), 0
        ) as category_market_share,
        
        -- Revenue concentration
        category_revenue_90d::NUMERIC / NULLIF(
            (SELECT SUM(category_revenue_90d) FROM category_metrics), 0
        ) as revenue_share,
        
        -- Customer affinity
        unique_customers_90d::NUMERIC / NULLIF(
            (SELECT COUNT(DISTINCT customer_id) FROM orders 
             WHERE order_date >= CURRENT_DATE - INTERVAL '90 days'), 0
        ) as customer_penetration,
        
        -- Average order value in category
        CASE 
            WHEN unique_customers_90d > 0 
            THEN category_revenue_90d / unique_customers_90d 
            ELSE 0 
        END as avg_customer_value,
        
        -- Category depth score (for hierarchical embeddings)
        level::NUMERIC / (SELECT MAX(level) FROM category_tree) as normalized_depth,
        
        -- Sibling competition (categories at same level with same parent)
        (SELECT COUNT(*) 
         FROM category_tree siblings 
         WHERE siblings.parent_category_id = cm.parent_category_id
           AND siblings.category_id != cm.category_id
        ) as sibling_competition,
        
        -- Path-based features for neural networks
        array_length(path_ids, 1) as path_length,
        path_ids[1] as root_category_id,
        CASE 
            WHEN array_length(path_ids, 1) > 1 
            THEN path_ids[array_length(path_ids, 1) - 1] 
            ELSE NULL 
        END as parent_category_id_extracted
        
    FROM category_metrics cm
),
cross_category_analysis AS (
    SELECT 
        mrf.*,
        -- Cross-category purchase patterns
        (SELECT ARRAY_AGG(DISTINCT other_cat.category_id)
         FROM orders o
         JOIN order_items oi ON o.order_id = oi.order_id
         JOIN products p ON oi.product_id = p.product_id
         JOIN products other_p ON other_p.product_id != p.product_id
         JOIN category_tree other_cat ON other_p.category_id = other_cat.category_id
         WHERE p.category_id = mrf.category_id
           AND o.order_date >= CURRENT_DATE - INTERVAL '90 days'
           AND other_cat.category_id != mrf.category_id
        ) as frequently_bought_with_categories,
        
        -- Seasonal patterns
        (SELECT AVG(monthly_revenue) 
         FROM (
             SELECT 
                 EXTRACT(MONTH FROM oi.order_date) as month,
                 SUM(oi.quantity * oi.unit_price) as monthly_revenue
             FROM order_items oi
             JOIN products p ON oi.product_id = p.product_id
             WHERE p.category_id = mrf.category_id
               AND oi.order_date >= CURRENT_DATE - INTERVAL '12 months'
             GROUP BY EXTRACT(MONTH FROM oi.order_date)
         ) monthly_data
        ) as avg_monthly_revenue,
        
        -- Customer lifetime value in category
        (SELECT AVG(customer_clv)
         FROM (
             SELECT 
                 o.customer_id,
                 SUM(oi.quantity * oi.unit_price) as customer_clv
             FROM orders o
             JOIN order_items oi ON o.order_id = oi.order_id
             JOIN products p ON oi.product_id = p.product_id
             WHERE p.category_id = mrf.category_id
             GROUP BY o.customer_id
         ) customer_values
        ) as avg_customer_lifetime_value
        
    FROM ml_recommendation_features mrf
)
SELECT 
    category_id,
    category_name,
    level as hierarchy_level,
    root_category,
    hierarchy_tier,
    category_path,
    path_ids,
    
    -- ML Features for recommendation systems
    ROUND(category_market_share::NUMERIC, 6) as market_share,
    ROUND(revenue_share::NUMERIC, 6) as revenue_contribution,
    ROUND(customer_penetration::NUMERIC, 6) as customer_reach,
    ROUND(avg_customer_value::NUMERIC, 2) as avg_order_value,
    ROUND(normalized_depth::NUMERIC, 3) as depth_score,
    
    -- Structural features
    subtree_product_count as product_inventory,
    sibling_competition as competitive_pressure,
    path_length as hierarchy_depth,
    
    -- Engagement features
    unique_customers_90d as active_customers,
    category_revenue_90d as recent_revenue,
    
    -- Cross-sell features
    array_length(frequently_bought_with_categories, 1) as cross_category_connections,
    frequently_bought_with_categories as related_categories,
    
    -- Performance features
    ROUND(avg_monthly_revenue::NUMERIC, 2) as revenue_stability,
    ROUND(avg_customer_lifetime_value::NUMERIC, 2) as customer_value_potential,
    
    -- Binary flags for classification
    CASE WHEN revenue_share > 0.05 THEN 1 ELSE 0 END as high_revenue_category,
    CASE WHEN customer_penetration > 0.1 THEN 1 ELSE 0 END as mass_market_category,
    CASE WHEN level >= 3 THEN 1 ELSE 0 END as specialized_category
    
FROM cross_category_analysis
ORDER BY level, revenue_share DESC;
```

### Graph Traversal for Network Analysis

#### 1. Social Network Analysis

```sql
-- Recursive query for social influence analysis
WITH RECURSIVE influence_network AS (
    -- Starting points: Influential users (high follower count)
    SELECT 
        user_id,
        username,
        follower_count,
        user_id as influencer_root,
        1 as degree_of_separation,
        0 as path_length,
        ARRAY[user_id] as influence_path,
        follower_count as accumulated_influence,
        username as influence_source
    FROM users
    WHERE follower_count >= 10000
      AND account_type = 'ACTIVE'
    
    UNION ALL
    
    -- Recursive traversal: Users influenced by influential users
    SELECT 
        f.follower_id as user_id,
        u.username,
        u.follower_count,
        inet.influencer_root,
        inet.degree_of_separation + 1,
        inet.path_length + 1,
        inet.influence_path || f.follower_id,
        inet.accumulated_influence + u.follower_count,
        inet.influence_source
    FROM influence_network inet
    JOIN follows f ON inet.user_id = f.following_id
    JOIN users u ON f.follower_id = u.user_id
    WHERE inet.degree_of_separation < 6  -- Six degrees of separation limit
      AND f.follower_id != ALL(inet.influence_path)  -- Prevent cycles
      AND u.account_type = 'ACTIVE'
      AND f.created_at >= CURRENT_DATE - INTERVAL '12 months'  -- Recent follows only
),
network_metrics AS (
    SELECT 
        inet.*,
        -- Network position metrics
        COUNT(*) OVER (
            PARTITION BY influencer_root, degree_of_separation
        ) as peers_at_same_level,
        
        -- Influence decay model
        follower_count * POWER(0.7, degree_of_separation) as discounted_influence,
        
        -- Path diversity
        (SELECT COUNT(DISTINCT influencer_root) 
         FROM influence_network other 
         WHERE other.user_id = inet.user_id
        ) as influence_source_diversity,
        
        -- Centrality approximations
        ROW_NUMBER() OVER (
            PARTITION BY degree_of_separation 
            ORDER BY accumulated_influence DESC
        ) as influence_rank_at_level,
        
        -- User engagement potential
        CASE 
            WHEN degree_of_separation = 1 THEN 'DIRECT_FOLLOWER'
            WHEN degree_of_separation <= 2 THEN 'SECOND_DEGREE'
            WHEN degree_of_separation <= 3 THEN 'THIRD_DEGREE'
            ELSE 'DISTANT_CONNECTION'
        END as connection_strength,
        
        -- Network bridge potential
        CASE 
            WHEN influence_source_diversity > 3 THEN 'BRIDGE_USER'
            WHEN influence_source_diversity > 1 THEN 'CONNECTOR_USER'
            ELSE 'ENDPOINT_USER'
        END as network_role
        
    FROM influence_network inet
),
ml_network_features AS (
    SELECT 
        user_id,
        username,
        follower_count,
        
        -- Network position features
        MIN(degree_of_separation) as min_separation_from_influencer,
        MAX(degree_of_separation) as max_separation_from_influencer,
        AVG(degree_of_separation) as avg_separation_from_influencer,
        COUNT(DISTINCT influencer_root) as unique_influence_sources,
        
        -- Influence metrics
        SUM(discounted_influence) as total_discounted_influence,
        MAX(accumulated_influence) as max_influence_path,
        AVG(accumulated_influence) as avg_influence_path,
        
        -- Network diversity
        MAX(influence_source_diversity) as max_influence_diversity,
        COUNT(DISTINCT connection_strength) as connection_type_diversity,
        
        -- Centrality approximations
        MIN(influence_rank_at_level) as best_influence_rank,
        AVG(peers_at_same_level) as avg_peer_competition,
        
        -- Network role classification
        MODE() WITHIN GROUP (ORDER BY network_role) as primary_network_role,
        
        -- Path characteristics
        MIN(path_length) as shortest_influence_path,
        MAX(path_length) as longest_influence_path,
        AVG(path_length) as avg_influence_path_length,
        
        -- Binary features for ML models
        CASE WHEN MIN(degree_of_separation) <= 2 THEN 1 ELSE 0 END as direct_influence_flag,
        CASE WHEN COUNT(DISTINCT influencer_root) >= 3 THEN 1 ELSE 0 END as multi_source_influence_flag,
        CASE WHEN MAX(influence_source_diversity) >= 3 THEN 1 ELSE 0 END as bridge_user_flag
        
    FROM network_metrics
    GROUP BY user_id, username, follower_count
),
engagement_prediction_features AS (
    SELECT 
        mnf.*,
        -- Engagement prediction features
        LOG(1 + total_discounted_influence) as log_influence_score,
        follower_count::NUMERIC / NULLIF(avg_peer_competition, 0) as influence_efficiency,
        
        -- Virality potential
        CASE 
            WHEN direct_influence_flag = 1 AND bridge_user_flag = 1 THEN 'HIGH_VIRAL'
            WHEN direct_influence_flag = 1 OR bridge_user_flag = 1 THEN 'MEDIUM_VIRAL'
            ELSE 'LOW_VIRAL'
        END as virality_potential,
        
        -- Targeting score for marketing
        (total_discounted_influence * unique_influence_sources * direct_influence_flag) as marketing_target_score,
        
        -- Network stability
        CASE 
            WHEN avg_influence_path_length <= 2 THEN 'STABLE_CONNECTION'
            WHEN avg_influence_path_length <= 4 THEN 'MODERATE_CONNECTION'
            ELSE 'WEAK_CONNECTION'
        END as connection_stability
        
    FROM ml_network_features mnf
)
SELECT 
    user_id,
    username,
    follower_count,
    
    -- Core influence metrics
    ROUND(total_discounted_influence::NUMERIC, 2) as influence_score,
    unique_influence_sources,
    min_separation_from_influencer,
    
    -- Network position
    primary_network_role,
    virality_potential,
    connection_stability,
    
    -- ML features (normalized for models)
    ROUND(log_influence_score::NUMERIC, 4) as log_influence_feature,
    ROUND(influence_efficiency::NUMERIC, 4) as efficiency_feature,
    ROUND(marketing_target_score::NUMERIC, 2) as marketing_score,
    
    -- Classification features
    direct_influence_flag,
    multi_source_influence_flag,
    bridge_user_flag,
    
    -- Regression targets
    follower_count as target_followers,
    total_discounted_influence as target_influence
    
FROM engagement_prediction_features
WHERE total_discounted_influence > 0
ORDER BY marketing_target_score DESC, influence_score DESC;
```

### Decision Tree Path Analysis

#### 1. Customer Journey Analysis

```sql
-- Recursive analysis of customer decision paths
WITH RECURSIVE customer_journey AS (
    -- Entry points: First website visit or app open
    SELECT 
        session_id,
        customer_id,
        event_type,
        event_timestamp,
        page_url,
        product_id,
        1 as step_number,
        event_type as journey_start,
        ARRAY[event_type] as event_sequence,
        ARRAY[event_timestamp] as timestamp_sequence,
        event_timestamp as session_start,
        0 as total_duration_seconds
    FROM user_events
    WHERE event_type IN ('page_view', 'app_open', 'search')
      AND event_timestamp >= CURRENT_DATE - INTERVAL '30 days'
      AND customer_id IS NOT NULL
    
    UNION ALL
    
    -- Subsequent events in the same session
    SELECT 
        ue.session_id,
        ue.customer_id,
        ue.event_type,
        ue.event_timestamp,
        ue.page_url,
        ue.product_id,
        cj.step_number + 1,
        cj.journey_start,
        cj.event_sequence || ue.event_type,
        cj.timestamp_sequence || ue.event_timestamp,
        cj.session_start,
        EXTRACT(EPOCH FROM ue.event_timestamp - cj.session_start)::INTEGER
    FROM user_events ue
    INNER JOIN customer_journey cj 
        ON ue.session_id = cj.session_id 
        AND ue.event_timestamp > cj.event_timestamp
    WHERE cj.step_number < 50  -- Limit journey length
      AND EXTRACT(EPOCH FROM ue.event_timestamp - cj.session_start) < 7200  -- 2-hour session max
),
journey_analysis AS (
    SELECT 
        customer_id,
        session_id,
        journey_start,
        
        -- Journey completion metrics
        MAX(step_number) as journey_length,
        MAX(total_duration_seconds) as session_duration,
        
        -- Event patterns
        event_sequence,
        timestamp_sequence,
        
        -- Conversion events
        CASE 
            WHEN 'purchase' = ANY(event_sequence) THEN 1 
            ELSE 0 
        END as converted_flag,
        
        CASE 
            WHEN 'add_to_cart' = ANY(event_sequence) THEN 1 
            ELSE 0 
        END as cart_addition_flag,
        
        CASE 
            WHEN 'checkout_start' = ANY(event_sequence) THEN 1 
            ELSE 0 
        END as checkout_attempt_flag,
        
        -- Journey abandonment analysis
        CASE 
            WHEN event_sequence[array_length(event_sequence, 1)] = 'checkout_start' 
                 AND 'purchase' != ANY(event_sequence) 
            THEN 1 ELSE 0 
        END as checkout_abandonment_flag,
        
        -- Product interaction depth
        (SELECT COUNT(*) 
         FROM unnest(event_sequence) as event 
         WHERE event IN ('product_view', 'product_details', 'reviews_view')
        ) as product_interaction_count,
        
        -- Search behavior
        (SELECT COUNT(*) 
         FROM unnest(event_sequence) as event 
         WHERE event = 'search'
        ) as search_count,
        
        -- Navigation patterns
        (SELECT COUNT(*) 
         FROM unnest(event_sequence) as event 
         WHERE event = 'page_view'
        ) as page_view_count
        
    FROM customer_journey
    GROUP BY customer_id, session_id, journey_start, event_sequence, timestamp_sequence
),
ml_journey_features AS (
    SELECT 
        customer_id,
        session_id,
        
        -- Sequence features for RNN/LSTM models
        journey_length,
        session_duration,
        event_sequence,
        
        -- Time-based features
        CASE 
            WHEN session_duration <= 300 THEN 'QUICK_SESSION'      -- 5 minutes
            WHEN session_duration <= 1800 THEN 'MEDIUM_SESSION'    -- 30 minutes
            ELSE 'LONG_SESSION'
        END as session_duration_category,
        
        -- Interaction intensity
        journey_length::NUMERIC / NULLIF(session_duration / 60.0, 0) as events_per_minute,
        
        -- Conversion funnel features
        converted_flag,
        cart_addition_flag,
        checkout_attempt_flag,
        checkout_abandonment_flag,
        
        -- Engagement depth
        product_interaction_count,
        product_interaction_count::NUMERIC / NULLIF(journey_length, 0) as product_interaction_ratio,
        
        -- Search behavior
        search_count,
        search_count::NUMERIC / NULLIF(journey_length, 0) as search_ratio,
        
        -- Navigation efficiency
        CASE 
            WHEN converted_flag = 1 THEN journey_length::NUMERIC 
            ELSE NULL 
        END as conversion_journey_length,
        
        -- Bounce analysis
        CASE WHEN journey_length = 1 THEN 1 ELSE 0 END as bounce_flag,
        
        -- Journey start influence
        CASE 
            WHEN journey_start = 'search' THEN 'SEARCH_DRIVEN'
            WHEN journey_start = 'page_view' THEN 'BROWSE_DRIVEN'
            ELSE 'APP_DRIVEN'
        END as acquisition_channel,
        
        -- Pattern classification for clustering
        CASE 
            WHEN converted_flag = 1 AND journey_length <= 5 THEN 'EFFICIENT_BUYER'
            WHEN converted_flag = 1 AND journey_length > 10 THEN 'RESEARCH_BUYER'
            WHEN cart_addition_flag = 1 AND converted_flag = 0 THEN 'CART_ABANDONER'
            WHEN product_interaction_count >= 3 AND converted_flag = 0 THEN 'BROWSER'
            WHEN search_count >= 2 AND converted_flag = 0 THEN 'SEARCHER'
            WHEN bounce_flag = 1 THEN 'BOUNCER'
            ELSE 'CASUAL_VISITOR'
        END as journey_persona
        
    FROM journey_analysis
),
customer_journey_summary AS (
    SELECT 
        customer_id,
        
        -- Aggregated journey metrics
        COUNT(*) as total_sessions,
        AVG(journey_length) as avg_journey_length,
        AVG(session_duration) as avg_session_duration,
        SUM(converted_flag) as total_conversions,
        SUM(cart_addition_flag) as total_cart_additions,
        SUM(checkout_abandonment_flag) as total_checkout_abandonments,
        
        -- Conversion rates
        SUM(converted_flag)::NUMERIC / COUNT(*) as conversion_rate,
        SUM(cart_addition_flag)::NUMERIC / COUNT(*) as cart_addition_rate,
        SUM(checkout_abandonment_flag)::NUMERIC / NULLIF(SUM(checkout_attempt_flag), 0) as abandonment_rate,
        
        -- Behavioral consistency
        STDDEV(journey_length) as journey_length_consistency,
        STDDEV(events_per_minute) as interaction_pace_consistency,
        
        -- Journey persona distribution
        MODE() WITHIN GROUP (ORDER BY journey_persona) as primary_persona,
        COUNT(DISTINCT journey_persona) as persona_diversity,
        
        -- Acquisition channel preferences
        MODE() WITHIN GROUP (ORDER BY acquisition_channel) as preferred_channel,
        
        -- Engagement trends
        AVG(product_interaction_ratio) as avg_product_engagement,
        AVG(search_ratio) as avg_search_propensity,
        
        -- Experience quality indicators
        AVG(CASE WHEN bounce_flag = 0 THEN events_per_minute END) as quality_session_pace,
        SUM(bounce_flag)::NUMERIC / COUNT(*) as bounce_rate
        
    FROM ml_journey_features
    GROUP BY customer_id
)
SELECT 
    customer_id,
    total_sessions,
    ROUND(avg_journey_length::NUMERIC, 2) as avg_path_length,
    ROUND(avg_session_duration::NUMERIC, 0) as avg_duration_seconds,
    
    -- Performance metrics
    ROUND(conversion_rate::NUMERIC, 4) as conversion_rate,
    ROUND(cart_addition_rate::NUMERIC, 4) as cart_rate,
    ROUND(abandonment_rate::NUMERIC, 4) as abandonment_rate,
    ROUND(bounce_rate::NUMERIC, 4) as bounce_rate,
    
    -- Behavioral insights
    primary_persona,
    preferred_channel,
    ROUND(avg_product_engagement::NUMERIC, 4) as engagement_intensity,
    ROUND(quality_session_pace::NUMERIC, 2) as interaction_efficiency,
    
    -- Consistency measures
    ROUND(journey_length_consistency::NUMERIC, 2) as behavior_consistency,
    persona_diversity,
    
    -- ML features for prediction
    CASE WHEN conversion_rate >= 0.1 THEN 1 ELSE 0 END as high_converter_flag,
    CASE WHEN bounce_rate <= 0.3 THEN 1 ELSE 0 END as engaged_user_flag,
    CASE WHEN abandonment_rate <= 0.2 THEN 1 ELSE 0 END as low_abandonment_flag,
    
    -- Targets for ML models
    total_conversions as target_conversions,
    conversion_rate as target_conversion_rate
    
FROM customer_journey_summary
WHERE total_sessions >= 3  -- Minimum sessions for reliable patterns
ORDER BY conversion_rate DESC, total_sessions DESC;
```

### Real-World Applications

1. **Hierarchical Classification**: Organize products/content into taxonomies
2. **Social Network Analysis**: Influence propagation and community detection
3. **Recommendation Systems**: Content similarity and collaborative filtering
4. **Decision Tree Analysis**: Extract rules and patterns from tree models
5. **Customer Journey Mapping**: Path analysis for conversion optimization

### Best Practices

1. **Limit Recursion Depth**: Prevent infinite loops and excessive computation
2. **Use Proper Termination Conditions**: Define clear stopping criteria
3. **Index Hierarchical Columns**: Optimize performance for recursive queries
4. **Handle Cycles**: Implement cycle detection in graph traversals
5. **Consider Memory Usage**: Recursive queries can consume significant memory

### Common Pitfalls

1. **Infinite Recursion**: Missing or incorrect termination conditions
2. **Performance Issues**: Deep recursions without proper indexing
3. **Memory Exhaustion**: Large hierarchies consuming too much memory
4. **Cycle Handling**: Not accounting for circular relationships
5. **Incorrect Aggregations**: Misunderstanding how to aggregate in recursive contexts

### Performance Considerations

- **Depth Limitations**: Set reasonable limits on recursion depth
- **Index Strategy**: Create indexes on parent-child relationship columns
- **Memory Management**: Monitor memory usage for large hierarchies
- **Query Optimization**: Use EXPLAIN to analyze recursive query performance
- **Materialized Views**: Consider caching results for frequently accessed hierarchies

---

## Question 13

**Describe how graph-based features can be generated from SQL data.**

**Answer:**

### Theory

Graph-based features extract structural and relational information from data by modeling entities as nodes and relationships as edges. In SQL databases, these features capture network properties, centrality measures, community structures, and path-based insights that are crucial for machine learning applications. Graph features are particularly valuable for social networks, recommendation systems, fraud detection, knowledge graphs, and any domain where relationships between entities provide predictive power.

**Key Graph Feature Categories:**
- **Centrality Measures**: Node importance (degree, betweenness, closeness, PageRank)
- **Structural Features**: Clustering coefficients, path lengths, connectivity
- **Community Features**: Group membership, modularity, bridge detection
- **Temporal Features**: Evolution patterns, growth trends, activity cycles
- **Similarity Features**: Neighborhood overlap, structural equivalence

### Node-Level Features

#### 1. Centrality Measures

```sql
-- Comprehensive centrality analysis for social networks
WITH user_connections AS (
    SELECT 
        user_id,
        follower_id as connected_user,
        'OUTGOING' as connection_type,
        created_at as connection_date
    FROM user_follows
    WHERE active = TRUE
    
    UNION ALL
    
    SELECT 
        follower_id as user_id,
        user_id as connected_user,
        'INCOMING' as connection_type,
        created_at as connection_date
    FROM user_follows
    WHERE active = TRUE
),
degree_centrality AS (
    SELECT 
        user_id,
        -- Degree centrality measures
        COUNT(*) as total_degree,
        COUNT(*) FILTER (WHERE connection_type = 'OUTGOING') as out_degree,
        COUNT(*) FILTER (WHERE connection_type = 'INCOMING') as in_degree,
        
        -- Normalized degree centrality
        COUNT(*)::NUMERIC / (
            SELECT COUNT(DISTINCT user_id) - 1 
            FROM users WHERE active = TRUE
        ) as normalized_degree_centrality,
        
        -- Asymmetry measure
        ABS(
            COUNT(*) FILTER (WHERE connection_type = 'OUTGOING') - 
            COUNT(*) FILTER (WHERE connection_type = 'INCOMING')
        )::NUMERIC / NULLIF(COUNT(*), 0) as degree_asymmetry,
        
        -- Connection recency
        AVG(EXTRACT(DAYS FROM CURRENT_DATE - connection_date)) as avg_connection_age,
        
        -- Connection velocity
        COUNT(*) FILTER (WHERE connection_date >= CURRENT_DATE - INTERVAL '30 days') as recent_connections
        
    FROM user_connections
    GROUP BY user_id
),
neighborhood_analysis AS (
    SELECT 
        uc1.user_id,
        -- Two-hop neighborhood size
        COUNT(DISTINCT uc2.connected_user) as two_hop_neighbors,
        
        -- Clustering coefficient approximation
        COUNT(DISTINCT uc2.connected_user) FILTER (
            WHERE EXISTS (
                SELECT 1 FROM user_connections uc3 
                WHERE uc3.user_id = uc1.connected_user 
                  AND uc3.connected_user = uc2.connected_user
            )
        )::NUMERIC / NULLIF(
            COUNT(DISTINCT uc1.connected_user) * (COUNT(DISTINCT uc1.connected_user) - 1) / 2, 0
        ) as local_clustering_coefficient,
        
        -- Neighborhood diversity
        COUNT(DISTINCT u.user_category) as neighbor_category_diversity,
        COUNT(DISTINCT u.location) as neighbor_location_diversity,
        
        -- Neighbor influence
        AVG(u.follower_count) as avg_neighbor_influence,
        MAX(u.follower_count) as max_neighbor_influence,
        
        -- Activity level of neighbors
        AVG(
            EXTRACT(DAYS FROM CURRENT_DATE - u.last_activity_date)
        ) as avg_neighbor_activity_recency
        
    FROM user_connections uc1
    JOIN user_connections uc2 ON uc1.connected_user = uc2.user_id
    JOIN users u ON uc1.connected_user = u.user_id
    WHERE uc1.connection_type = 'OUTGOING'
      AND uc2.connection_type = 'OUTGOING'
    GROUP BY uc1.user_id
),
pagerank_approximation AS (
    SELECT 
        user_id,
        -- Simple PageRank approximation using iterative calculation
        SUM(
            (1.0 / NULLIF(source_out_degree, 0)) * 0.85 + 0.15
        ) as pagerank_score
    FROM (
        SELECT 
            uc.user_id,
            dc_source.total_degree as source_out_degree
        FROM user_connections uc
        JOIN degree_centrality dc_source ON uc.connected_user = dc_source.user_id
        WHERE uc.connection_type = 'INCOMING'
    ) pagerank_calc
    GROUP BY user_id
),
bridging_analysis AS (
    SELECT 
        uc.user_id,
        -- Structural holes and bridging potential
        COUNT(DISTINCT c1.cluster_id) as connected_clusters,
        
        -- Bridge score (connects different communities)
        CASE 
            WHEN COUNT(DISTINCT c1.cluster_id) > COUNT(DISTINCT uc.connected_user) * 0.5 
            THEN 1 ELSE 0 
        END as bridge_user_flag,
        
        -- Brokerage position
        COUNT(*) FILTER (
            WHERE c1.cluster_id != c2.cluster_id
        )::NUMERIC / NULLIF(COUNT(*), 0) as brokerage_ratio,
        
        -- Information flow potential
        AVG(c1.cluster_size) as avg_connected_cluster_size,
        
        -- Cross-cluster influence
        SUM(
            CASE WHEN c1.cluster_id != (
                SELECT cluster_id FROM user_clusters WHERE user_id = uc.user_id
            ) THEN 1 ELSE 0 END
        ) as cross_cluster_connections
        
    FROM user_connections uc
    JOIN user_clusters c1 ON uc.connected_user = c1.user_id
    JOIN user_clusters c2 ON uc.user_id = c2.user_id
    WHERE uc.connection_type = 'OUTGOING'
    GROUP BY uc.user_id
)
SELECT 
    dc.user_id,
    u.username,
    u.user_category,
    
    -- Basic centrality features
    dc.total_degree,
    dc.out_degree,
    dc.in_degree,
    ROUND(dc.normalized_degree_centrality::NUMERIC, 6) as norm_degree_centrality,
    ROUND(dc.degree_asymmetry::NUMERIC, 4) as degree_asymmetry,
    
    -- Neighborhood features
    COALESCE(na.two_hop_neighbors, 0) as neighborhood_size,
    ROUND(COALESCE(na.local_clustering_coefficient, 0)::NUMERIC, 4) as clustering_coefficient,
    COALESCE(na.neighbor_category_diversity, 0) as neighbor_diversity,
    ROUND(COALESCE(na.avg_neighbor_influence, 0)::NUMERIC, 2) as neighbor_influence,
    
    -- Authority and hub scores
    ROUND(COALESCE(pr.pagerank_score, 0.15)::NUMERIC, 6) as pagerank_authority,
    dc.out_degree::NUMERIC / NULLIF(dc.in_degree, 0) as hub_authority_ratio,
    
    -- Bridging and brokerage
    COALESCE(ba.connected_clusters, 1) as cluster_connections,
    COALESCE(ba.bridge_user_flag, 0) as bridge_flag,
    ROUND(COALESCE(ba.brokerage_ratio, 0)::NUMERIC, 4) as brokerage_score,
    COALESCE(ba.cross_cluster_connections, 0) as inter_cluster_edges,
    
    -- Temporal features
    ROUND(dc.avg_connection_age::NUMERIC, 1) as avg_relationship_age,
    dc.recent_connections as recent_activity,
    
    -- Derived features for ML
    CASE 
        WHEN dc.total_degree >= (SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY total_degree) FROM degree_centrality) 
        THEN 1 ELSE 0 
    END as high_degree_flag,
    
    CASE 
        WHEN COALESCE(pr.pagerank_score, 0) >= (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY pagerank_score) FROM pagerank_approximation) 
        THEN 1 ELSE 0 
    END as high_authority_flag,
    
    -- Network position classification
    CASE 
        WHEN COALESCE(ba.bridge_user_flag, 0) = 1 THEN 'BRIDGE'
        WHEN dc.out_degree > dc.in_degree * 2 THEN 'BROADCASTER'
        WHEN dc.in_degree > dc.out_degree * 2 THEN 'RECEIVER'
        WHEN COALESCE(na.local_clustering_coefficient, 0) > 0.7 THEN 'COMMUNITY_CORE'
        ELSE 'REGULAR'
    END as network_role
    
FROM degree_centrality dc
JOIN users u ON dc.user_id = u.user_id
LEFT JOIN neighborhood_analysis na ON dc.user_id = na.user_id
LEFT JOIN pagerank_approximation pr ON dc.user_id = pr.user_id
LEFT JOIN bridging_analysis ba ON dc.user_id = ba.user_id
WHERE u.active = TRUE
ORDER BY dc.total_degree DESC;
```

#### 2. Path-Based Features

```sql
-- Path analysis for recommendation systems
WITH RECURSIVE user_paths AS (
    -- Base case: Direct connections
    SELECT 
        f1.user_id as source_user,
        f1.following_id as target_user,
        1 as path_length,
        ARRAY[f1.user_id, f1.following_id] as path_nodes,
        f1.created_at as relationship_start
    FROM user_follows f1
    WHERE f1.active = TRUE
    
    UNION ALL
    
    -- Recursive case: Extended paths (up to 4 hops)
    SELECT 
        up.source_user,
        f2.following_id as target_user,
        up.path_length + 1,
        up.path_nodes || f2.following_id,
        up.relationship_start
    FROM user_paths up
    JOIN user_follows f2 ON up.target_user = f2.user_id
    WHERE up.path_length < 4
      AND f2.following_id != ALL(up.path_nodes)  -- Prevent cycles
      AND f2.active = TRUE
),
shortest_paths AS (
    SELECT 
        source_user,
        target_user,
        MIN(path_length) as shortest_distance,
        COUNT(*) as path_count,
        AVG(path_length) as avg_path_length
    FROM user_paths
    GROUP BY source_user, target_user
),
path_features AS (
    SELECT 
        u1.user_id,
        
        -- Reachability features
        COUNT(DISTINCT sp.target_user) as reachable_users,
        COUNT(*) FILTER (WHERE sp.shortest_distance = 1) as direct_connections,
        COUNT(*) FILTER (WHERE sp.shortest_distance = 2) as two_hop_connections,
        COUNT(*) FILTER (WHERE sp.shortest_distance = 3) as three_hop_connections,
        COUNT(*) FILTER (WHERE sp.shortest_distance = 4) as four_hop_connections,
        
        -- Path diversity
        AVG(sp.path_count) as avg_path_redundancy,
        STDDEV(sp.shortest_distance) as path_length_variance,
        
        -- Network diameter approximation
        MAX(sp.shortest_distance) as max_distance_reachable,
        AVG(sp.shortest_distance) as avg_distance_to_reachable,
        
        -- Structural features
        COUNT(*) FILTER (WHERE sp.path_count = 1) as bridge_dependent_connections,
        COUNT(*) FILTER (WHERE sp.path_count > 3) as highly_redundant_connections,
        
        -- Connectivity ratio
        COUNT(DISTINCT sp.target_user)::NUMERIC / (
            SELECT COUNT(*) FROM users WHERE active = TRUE
        ) as network_coverage_ratio
        
    FROM users u1
    LEFT JOIN shortest_paths sp ON u1.user_id = sp.source_user
    WHERE u1.active = TRUE
    GROUP BY u1.user_id
)
SELECT 
    user_id,
    reachable_users,
    direct_connections,
    two_hop_connections,
    three_hop_connections,
    four_hop_connections,
    
    -- Path efficiency features
    ROUND(avg_path_redundancy::NUMERIC, 2) as connection_redundancy,
    ROUND(path_length_variance::NUMERIC, 2) as path_diversity,
    max_distance_reachable as network_reach,
    ROUND(avg_distance_to_reachable::NUMERIC, 2) as avg_separation,
    
    -- Structural importance
    bridge_dependent_connections as critical_path_count,
    highly_redundant_connections as robust_connection_count,
    ROUND(network_coverage_ratio::NUMERIC, 6) as network_coverage,
    
    -- Classification features
    CASE 
        WHEN network_coverage_ratio > 0.1 THEN 'HIGH_REACH'
        WHEN network_coverage_ratio > 0.05 THEN 'MEDIUM_REACH'
        ELSE 'LOW_REACH'
    END as reach_category,
    
    CASE 
        WHEN avg_distance_to_reachable <= 2.0 THEN 'CLOSE_NETWORK'
        WHEN avg_distance_to_reachable <= 3.0 THEN 'MEDIUM_NETWORK'
        ELSE 'DISTANT_NETWORK'
    END as network_density,
    
    -- Binary flags for ML models
    CASE WHEN bridge_dependent_connections > direct_connections * 0.1 THEN 1 ELSE 0 END as network_vulnerability_flag,
    CASE WHEN network_coverage_ratio >= 0.05 THEN 1 ELSE 0 END as influential_user_flag
    
FROM path_features
ORDER BY network_coverage DESC, reachable_users DESC;
```

### Edge-Level Features

#### 1. Relationship Strength and Properties

```sql
-- Edge-level analysis for relationship prediction
WITH interaction_base AS (
    SELECT 
        user_id,
        target_user_id,
        interaction_type,
        interaction_timestamp,
        interaction_value,
        platform
    FROM user_interactions
    WHERE interaction_timestamp >= CURRENT_DATE - INTERVAL '12 months'
      AND interaction_type IN ('like', 'comment', 'share', 'message', 'mention')
),
edge_statistics AS (
    SELECT 
        user_id,
        target_user_id,
        
        -- Interaction frequency
        COUNT(*) as total_interactions,
        COUNT(DISTINCT interaction_type) as interaction_type_diversity,
        COUNT(DISTINCT DATE(interaction_timestamp)) as active_days,
        
        -- Interaction intensity
        SUM(interaction_value) as total_interaction_value,
        AVG(interaction_value) as avg_interaction_value,
        MAX(interaction_value) as max_interaction_value,
        
        -- Temporal patterns
        MIN(interaction_timestamp) as first_interaction,
        MAX(interaction_timestamp) as last_interaction,
        EXTRACT(DAYS FROM MAX(interaction_timestamp) - MIN(interaction_timestamp)) + 1 as relationship_duration_days,
        
        -- Interaction type breakdown
        COUNT(*) FILTER (WHERE interaction_type = 'like') as like_count,
        COUNT(*) FILTER (WHERE interaction_type = 'comment') as comment_count,
        COUNT(*) FILTER (WHERE interaction_type = 'share') as share_count,
        COUNT(*) FILTER (WHERE interaction_type = 'message') as message_count,
        COUNT(*) FILTER (WHERE interaction_type = 'mention') as mention_count,
        
        -- Recency features
        EXTRACT(DAYS FROM CURRENT_DATE - MAX(interaction_timestamp)) as days_since_last_interaction,
        COUNT(*) FILTER (WHERE interaction_timestamp >= CURRENT_DATE - INTERVAL '30 days') as recent_interactions,
        
        -- Platform diversity
        COUNT(DISTINCT platform) as platform_diversity,
        
        -- Consistency measures
        STDDEV(EXTRACT(DAYS FROM interaction_timestamp - LAG(interaction_timestamp) OVER (ORDER BY interaction_timestamp))) as interaction_interval_variance
        
    FROM interaction_base
    GROUP BY user_id, target_user_id
    HAVING COUNT(*) >= 3  -- Minimum interactions for reliable analysis
),
mutual_features AS (
    SELECT 
        es1.user_id,
        es1.target_user_id,
        es1.*,
        
        -- Reciprocity features
        COALESCE(es2.total_interactions, 0) as reverse_interactions,
        CASE 
            WHEN es2.total_interactions > 0 THEN 1 
            ELSE 0 
        END as reciprocal_relationship_flag,
        
        -- Interaction balance
        es1.total_interactions::NUMERIC / NULLIF(
            es1.total_interactions + COALESCE(es2.total_interactions, 0), 0
        ) as interaction_dominance_ratio,
        
        -- Temporal synchronization
        ABS(
            EXTRACT(DAYS FROM es1.last_interaction - COALESCE(es2.last_interaction, es1.last_interaction))
        ) as last_interaction_time_diff,
        
        -- Value balance
        es1.total_interaction_value::NUMERIC / NULLIF(
            es1.total_interaction_value + COALESCE(es2.total_interaction_value, 0), 0
        ) as value_dominance_ratio
        
    FROM edge_statistics es1
    LEFT JOIN edge_statistics es2 
        ON es1.user_id = es2.target_user_id 
        AND es1.target_user_id = es2.user_id
),
common_neighbors AS (
    SELECT 
        mf.user_id,
        mf.target_user_id,
        
        -- Mutual connections
        COUNT(DISTINCT cn.common_user) as mutual_connections,
        
        -- Jaccard similarity of neighborhoods
        COUNT(DISTINCT cn.common_user)::NUMERIC / NULLIF(
            (SELECT COUNT(DISTINCT following_id) FROM user_follows WHERE user_id = mf.user_id) +
            (SELECT COUNT(DISTINCT following_id) FROM user_follows WHERE user_id = mf.target_user_id) -
            COUNT(DISTINCT cn.common_user), 0
        ) as jaccard_similarity,
        
        -- Adamic-Adar index approximation
        SUM(
            1.0 / NULLIF(LOG(
                (SELECT COUNT(*) FROM user_follows WHERE following_id = cn.common_user) + 1
            ), 0)
        ) as adamic_adar_score,
        
        -- Preferential attachment
        (SELECT COUNT(*) FROM user_follows WHERE user_id = mf.user_id) *
        (SELECT COUNT(*) FROM user_follows WHERE user_id = mf.target_user_id) as preferential_attachment_score
        
    FROM mutual_features mf
    LEFT JOIN (
        SELECT 
            uf1.user_id,
            uf2.user_id as target_user_id,
            uf1.following_id as common_user
        FROM user_follows uf1
        JOIN user_follows uf2 ON uf1.following_id = uf2.following_id
        WHERE uf1.user_id != uf2.user_id
          AND uf1.active = TRUE 
          AND uf2.active = TRUE
    ) cn ON mf.user_id = cn.user_id AND mf.target_user_id = cn.target_user_id
    GROUP BY mf.user_id, mf.target_user_id
),
relationship_strength AS (
    SELECT 
        mf.*,
        cn.mutual_connections,
        cn.jaccard_similarity,
        cn.adamic_adar_score,
        cn.preferential_attachment_score,
        
        -- Composite relationship strength score
        (
            -- Frequency component (30%)
            (mf.total_interactions::NUMERIC / NULLIF(mf.relationship_duration_days, 0)) * 0.3 +
            
            -- Reciprocity component (25%)
            CASE WHEN mf.reciprocal_relationship_flag = 1 
                 THEN (1 - ABS(0.5 - mf.interaction_dominance_ratio)) * 0.25
                 ELSE 0 END +
            
            -- Diversity component (20%)
            (mf.interaction_type_diversity::NUMERIC / 5.0) * 0.2 +
            
            -- Recency component (15%)
            CASE WHEN mf.days_since_last_interaction <= 30 THEN 0.15
                 WHEN mf.days_since_last_interaction <= 90 THEN 0.10
                 WHEN mf.days_since_last_interaction <= 180 THEN 0.05
                 ELSE 0 END +
            
            -- Common neighbors component (10%)
            LEAST(cn.mutual_connections::NUMERIC / 10.0, 1.0) * 0.1
            
        ) as relationship_strength_score,
        
        -- Relationship classification
        CASE 
            WHEN mf.total_interactions >= 50 AND mf.reciprocal_relationship_flag = 1 
                 AND mf.interaction_type_diversity >= 3 THEN 'STRONG_BIDIRECTIONAL'
            WHEN mf.total_interactions >= 20 AND mf.reciprocal_relationship_flag = 1 THEN 'MODERATE_BIDIRECTIONAL'
            WHEN mf.total_interactions >= 30 AND mf.interaction_type_diversity >= 2 THEN 'STRONG_UNIDIRECTIONAL'
            WHEN mf.total_interactions >= 10 THEN 'MODERATE_UNIDIRECTIONAL'
            ELSE 'WEAK'
        END as relationship_type,
        
        -- Communication pattern
        CASE 
            WHEN mf.message_count >= mf.total_interactions * 0.3 THEN 'COMMUNICATION_HEAVY'
            WHEN mf.share_count >= mf.total_interactions * 0.2 THEN 'CONTENT_SHARING'
            WHEN mf.comment_count >= mf.total_interactions * 0.4 THEN 'DISCUSSION_BASED'
            WHEN mf.like_count >= mf.total_interactions * 0.6 THEN 'PASSIVE_ENGAGEMENT'
            ELSE 'MIXED_INTERACTION'
        END as interaction_pattern
        
    FROM mutual_features mf
    LEFT JOIN common_neighbors cn ON mf.user_id = cn.user_id AND mf.target_user_id = cn.target_user_id
)
SELECT 
    user_id,
    target_user_id,
    
    -- Core relationship metrics
    total_interactions,
    interaction_type_diversity,
    relationship_duration_days,
    ROUND(relationship_strength_score::NUMERIC, 4) as strength_score,
    
    -- Interaction patterns
    ROUND(total_interactions::NUMERIC / NULLIF(relationship_duration_days, 0), 3) as interaction_frequency,
    days_since_last_interaction,
    recent_interactions,
    
    -- Reciprocity and balance
    reciprocal_relationship_flag,
    ROUND(interaction_dominance_ratio::NUMERIC, 3) as dominance_ratio,
    
    -- Similarity and common ground
    mutual_connections,
    ROUND(COALESCE(jaccard_similarity, 0)::NUMERIC, 4) as neighborhood_similarity,
    ROUND(COALESCE(adamic_adar_score, 0)::NUMERIC, 4) as connection_strength,
    
    -- Classification features
    relationship_type,
    interaction_pattern,
    
    -- Binary features for ML
    CASE WHEN relationship_strength_score >= 0.6 THEN 1 ELSE 0 END as strong_relationship_flag,
    CASE WHEN mutual_connections >= 5 THEN 1 ELSE 0 END as well_connected_flag,
    CASE WHEN days_since_last_interaction <= 7 THEN 1 ELSE 0 END as recently_active_flag,
    
    -- Detailed interaction breakdown for specialized models
    like_count,
    comment_count,
    share_count,
    message_count,
    mention_count,
    platform_diversity
    
FROM relationship_strength
WHERE relationship_strength_score > 0.1  -- Filter out very weak relationships
ORDER BY relationship_strength_score DESC;
```

### Community Detection Features

#### 1. Clustering and Community Analysis

```sql
-- Community detection using modularity-based clustering
WITH user_interactions_graph AS (
    SELECT 
        user_id,
        target_user_id,
        COUNT(*) as interaction_weight,
        AVG(interaction_value) as avg_interaction_strength
    FROM user_interactions
    WHERE interaction_timestamp >= CURRENT_DATE - INTERVAL '6 months'
    GROUP BY user_id, target_user_id
    HAVING COUNT(*) >= 3
),
community_initialization AS (
    -- Initialize each user as their own community
    SELECT 
        user_id,
        user_id as community_id,
        1 as community_size,
        0 as modularity_gain,
        1 as iteration
    FROM users
    WHERE active = TRUE
),
-- Simplified community detection (single iteration for demonstration)
community_assignment AS (
    SELECT 
        ci.user_id,
        -- Assign to community of strongest connected neighbor
        COALESCE(
            (SELECT uig.target_user_id
             FROM user_interactions_graph uig
             WHERE uig.user_id = ci.user_id
             ORDER BY uig.interaction_weight DESC
             LIMIT 1),
            ci.user_id
        ) as community_id
    FROM community_initialization ci
),
community_stats AS (
    SELECT 
        community_id,
        COUNT(*) as community_size,
        -- Calculate internal edges
        COALESCE(
            (SELECT SUM(uig.interaction_weight)
             FROM user_interactions_graph uig
             JOIN community_assignment ca1 ON uig.user_id = ca1.user_id
             JOIN community_assignment ca2 ON uig.target_user_id = ca2.user_id
             WHERE ca1.community_id = community_assignment.community_id
               AND ca2.community_id = community_assignment.community_id
            ), 0
        ) as internal_edges,
        
        -- Calculate external edges
        COALESCE(
            (SELECT SUM(uig.interaction_weight)
             FROM user_interactions_graph uig
             JOIN community_assignment ca1 ON uig.user_id = ca1.user_id
             JOIN community_assignment ca2 ON uig.target_user_id = ca2.user_id
             WHERE ca1.community_id = community_assignment.community_id
               AND ca2.community_id != community_assignment.community_id
            ), 0
        ) as external_edges,
        
        -- Community density
        COALESCE(
            (SELECT SUM(uig.interaction_weight)
             FROM user_interactions_graph uig
             JOIN community_assignment ca1 ON uig.user_id = ca1.user_id
             JOIN community_assignment ca2 ON uig.target_user_id = ca2.user_id
             WHERE ca1.community_id = community_assignment.community_id
               AND ca2.community_id = community_assignment.community_id
            )::NUMERIC / NULLIF(COUNT(*) * (COUNT(*) - 1), 0), 0
        ) as community_density
        
    FROM community_assignment
    GROUP BY community_id
),
user_community_features AS (
    SELECT 
        ca.user_id,
        ca.community_id,
        cs.community_size,
        cs.internal_edges,
        cs.external_edges,
        cs.community_density,
        
        -- User's contribution to community
        COALESCE(
            (SELECT SUM(uig.interaction_weight)
             FROM user_interactions_graph uig
             JOIN community_assignment ca2 ON uig.target_user_id = ca2.user_id
             WHERE uig.user_id = ca.user_id
               AND ca2.community_id = ca.community_id
            ), 0
        ) as internal_connections,
        
        COALESCE(
            (SELECT SUM(uig.interaction_weight)
             FROM user_interactions_graph uig
             JOIN community_assignment ca2 ON uig.target_user_id = ca2.user_id
             WHERE uig.user_id = ca.user_id
               AND ca2.community_id != ca.community_id
            ), 0
        ) as external_connections,
        
        -- Community centrality
        COALESCE(
            (SELECT COUNT(DISTINCT uig.target_user_id)
             FROM user_interactions_graph uig
             JOIN community_assignment ca2 ON uig.target_user_id = ca2.user_id
             WHERE uig.user_id = ca.user_id
               AND ca2.community_id = ca.community_id
            ), 0
        ) as community_degree,
        
        -- Bridge potential
        COALESCE(
            (SELECT COUNT(DISTINCT ca2.community_id)
             FROM user_interactions_graph uig
             JOIN community_assignment ca2 ON uig.target_user_id = ca2.user_id
             WHERE uig.user_id = ca.user_id
               AND ca2.community_id != ca.community_id
            ), 0
        ) as connected_communities
        
    FROM community_assignment ca
    JOIN community_stats cs ON ca.community_id = cs.community_id
),
community_roles AS (
    SELECT 
        ucf.*,
        
        -- Participation coefficient
        CASE 
            WHEN ucf.internal_connections + ucf.external_connections > 0
            THEN ucf.internal_connections::NUMERIC / (ucf.internal_connections + ucf.external_connections)
            ELSE 0
        END as participation_coefficient,
        
        -- Within-community z-score (centrality relative to community)
        (ucf.community_degree - AVG(ucf.community_degree) OVER (PARTITION BY ucf.community_id)) / 
        NULLIF(STDDEV(ucf.community_degree) OVER (PARTITION BY ucf.community_id), 0) as within_community_zscore,
        
        -- Community loyalty
        ucf.internal_connections::NUMERIC / NULLIF(ucf.internal_connections + ucf.external_connections, 0) as community_loyalty,
        
        -- Modularity contribution
        (ucf.internal_connections::NUMERIC / NULLIF(cs.internal_edges, 0)) - 
        POWER(
            (ucf.internal_connections + ucf.external_connections)::NUMERIC / 
            NULLIF((SELECT SUM(internal_edges + external_edges) FROM community_stats), 0), 2
        ) as modularity_contribution
        
    FROM user_community_features ucf
    JOIN community_stats cs ON ucf.community_id = cs.community_id
)
SELECT 
    user_id,
    community_id,
    community_size,
    
    -- Community engagement features
    internal_connections,
    external_connections,
    community_degree,
    connected_communities,
    
    -- Role classification features
    ROUND(participation_coefficient::NUMERIC, 4) as participation_coeff,
    ROUND(COALESCE(within_community_zscore, 0)::NUMERIC, 4) as community_centrality_zscore,
    ROUND(community_loyalty::NUMERIC, 4) as loyalty_score,
    ROUND(modularity_contribution::NUMERIC, 6) as modularity_contribution,
    
    -- Community characteristics
    ROUND(community_density::NUMERIC, 4) as cluster_density,
    
    -- Role classification
    CASE 
        WHEN COALESCE(within_community_zscore, 0) > 1.5 AND participation_coefficient > 0.8 THEN 'COMMUNITY_HUB'
        WHEN COALESCE(within_community_zscore, 0) > 1.0 AND participation_coefficient > 0.6 THEN 'COMMUNITY_LEADER'
        WHEN connected_communities >= 3 AND community_loyalty < 0.7 THEN 'BRIDGE_USER'
        WHEN participation_coefficient > 0.9 THEN 'COMMUNITY_CORE'
        WHEN connected_communities >= 2 THEN 'BOUNDARY_SPANNER'
        ELSE 'COMMUNITY_MEMBER'
    END as community_role,
    
    -- Size-based community classification
    CASE 
        WHEN community_size >= 100 THEN 'LARGE_COMMUNITY'
        WHEN community_size >= 20 THEN 'MEDIUM_COMMUNITY'
        WHEN community_size >= 5 THEN 'SMALL_COMMUNITY'
        ELSE 'MICRO_COMMUNITY'
    END as community_size_category,
    
    -- Binary features for ML models
    CASE WHEN connected_communities >= 3 THEN 1 ELSE 0 END as multi_community_flag,
    CASE WHEN COALESCE(within_community_zscore, 0) >= 1.0 THEN 1 ELSE 0 END as community_influencer_flag,
    CASE WHEN participation_coefficient >= 0.8 THEN 1 ELSE 0 END as highly_engaged_flag,
    CASE WHEN community_loyalty >= 0.8 THEN 1 ELSE 0 END as loyal_member_flag
    
FROM community_roles
ORDER BY modularity_contribution DESC, community_centrality_zscore DESC;
```

### Temporal Graph Features

#### 1. Evolution and Dynamic Analysis

```sql
-- Temporal graph analysis for trend detection
WITH monthly_snapshots AS (
    SELECT 
        DATE_TRUNC('month', interaction_timestamp) as month,
        user_id,
        target_user_id,
        COUNT(*) as monthly_interactions,
        SUM(interaction_value) as monthly_interaction_value,
        COUNT(DISTINCT interaction_type) as monthly_interaction_diversity
    FROM user_interactions
    WHERE interaction_timestamp >= CURRENT_DATE - INTERVAL '24 months'
    GROUP BY DATE_TRUNC('month', interaction_timestamp), user_id, target_user_id
),
user_monthly_activity AS (
    SELECT 
        month,
        user_id,
        COUNT(DISTINCT target_user_id) as active_connections,
        SUM(monthly_interactions) as total_monthly_interactions,
        AVG(monthly_interaction_value) as avg_monthly_value,
        MAX(monthly_interactions) as max_connection_intensity
    FROM monthly_snapshots
    GROUP BY month, user_id
),
temporal_features AS (
    SELECT 
        user_id,
        
        -- Activity trend analysis
        REGR_SLOPE(active_connections, EXTRACT(EPOCH FROM month)) * 86400 * 30 as connection_growth_rate,
        REGR_SLOPE(total_monthly_interactions, EXTRACT(EPOCH FROM month)) * 86400 * 30 as interaction_growth_rate,
        REGR_R2(active_connections, EXTRACT(EPOCH FROM month)) as connection_trend_consistency,
        
        -- Activity patterns
        COUNT(DISTINCT month) as active_months,
        AVG(active_connections) as avg_monthly_connections,
        STDDEV(active_connections) as connection_volatility,
        AVG(total_monthly_interactions) as avg_monthly_interactions,
        STDDEV(total_monthly_interactions) as interaction_volatility,
        
        -- Recent vs historical comparison
        AVG(CASE WHEN month >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '3 months') 
                 THEN active_connections END) as recent_avg_connections,
        AVG(CASE WHEN month < DATE_TRUNC('month', CURRENT_DATE - INTERVAL '3 months') 
                 THEN active_connections END) as historical_avg_connections,
        
        -- Peak activity analysis
        MAX(active_connections) as peak_connections,
        MAX(total_monthly_interactions) as peak_interactions,
        
        -- Consistency measures
        COUNT(*) FILTER (WHERE active_connections > 0) as consistent_activity_months,
        
        -- Seasonal patterns (simplified)
        AVG(CASE WHEN EXTRACT(MONTH FROM month) IN (12, 1, 2) THEN active_connections END) as winter_activity,
        AVG(CASE WHEN EXTRACT(MONTH FROM month) IN (6, 7, 8) THEN active_connections END) as summer_activity,
        
        -- Growth phases
        COUNT(*) FILTER (
            WHERE active_connections > LAG(active_connections) OVER (
                PARTITION BY user_id ORDER BY month
            )
        ) as growth_months,
        
        -- Activity lifecycle stage
        CASE 
            WHEN COUNT(DISTINCT month) <= 3 THEN 'NEW_USER'
            WHEN MAX(month) < DATE_TRUNC('month', CURRENT_DATE - INTERVAL '6 months') THEN 'DORMANT_USER'
            ELSE 'ACTIVE_USER'
        END as lifecycle_stage
        
    FROM user_monthly_activity
    GROUP BY user_id
    HAVING COUNT(DISTINCT month) >= 3  -- Minimum months for trend analysis
),
network_evolution AS (
    SELECT 
        tf.user_id,
        tf.*,
        
        -- Derived temporal features
        CASE 
            WHEN tf.recent_avg_connections > tf.historical_avg_connections * 1.2 THEN 'EXPANDING'
            WHEN tf.recent_avg_connections < tf.historical_avg_connections * 0.8 THEN 'CONTRACTING'
            ELSE 'STABLE'
        END as network_trajectory,
        
        -- Activity consistency score
        tf.consistent_activity_months::NUMERIC / tf.active_months as activity_consistency_ratio,
        
        -- Growth momentum
        CASE 
            WHEN tf.connection_growth_rate > 0 AND tf.connection_trend_consistency > 0.5 THEN 'STRONG_GROWTH'
            WHEN tf.connection_growth_rate > 0 THEN 'MODERATE_GROWTH'
            WHEN tf.connection_growth_rate < 0 AND tf.connection_trend_consistency > 0.5 THEN 'STRONG_DECLINE'
            WHEN tf.connection_growth_rate < 0 THEN 'MODERATE_DECLINE'
            ELSE 'STABLE'
        END as growth_momentum,
        
        -- Volatility classification
        CASE 
            WHEN tf.connection_volatility / NULLIF(tf.avg_monthly_connections, 0) > 0.5 THEN 'HIGH_VOLATILITY'
            WHEN tf.connection_volatility / NULLIF(tf.avg_monthly_connections, 0) > 0.3 THEN 'MEDIUM_VOLATILITY'
            ELSE 'LOW_VOLATILITY'
        END as activity_volatility_level,
        
        -- Seasonal behavior
        ABS(COALESCE(tf.winter_activity, 0) - COALESCE(tf.summer_activity, 0)) / 
        NULLIF(GREATEST(COALESCE(tf.winter_activity, 0), COALESCE(tf.summer_activity, 0)), 0) as seasonality_index,
        
        -- Peak performance relative to average
        tf.peak_connections::NUMERIC / NULLIF(tf.avg_monthly_connections, 0) as peak_to_average_ratio
        
    FROM temporal_features tf
)
SELECT 
    user_id,
    active_months,
    lifecycle_stage,
    
    -- Growth and trend features
    ROUND(connection_growth_rate::NUMERIC, 4) as monthly_connection_growth,
    ROUND(interaction_growth_rate::NUMERIC, 4) as monthly_interaction_growth,
    ROUND(connection_trend_consistency::NUMERIC, 4) as trend_reliability,
    growth_momentum,
    network_trajectory,
    
    -- Activity level features
    ROUND(avg_monthly_connections::NUMERIC, 2) as typical_network_size,
    ROUND(avg_monthly_interactions::NUMERIC, 2) as typical_activity_level,
    peak_connections as max_network_size,
    peak_interactions as max_activity_level,
    
    -- Consistency and stability
    ROUND(activity_consistency_ratio::NUMERIC, 4) as consistency_score,
    activity_volatility_level,
    ROUND(connection_volatility::NUMERIC, 2) as network_variability,
    
    -- Seasonal and cyclical patterns
    ROUND(seasonality_index::NUMERIC, 4) as seasonal_variation,
    ROUND(peak_to_average_ratio::NUMERIC, 2) as peak_performance_ratio,
    
    -- Recent performance comparison
    ROUND(COALESCE(recent_avg_connections, 0)::NUMERIC, 2) as recent_network_size,
    ROUND(COALESCE(historical_avg_connections, 0)::NUMERIC, 2) as historical_network_size,
    
    -- Binary classification features
    CASE WHEN connection_growth_rate > 0 THEN 1 ELSE 0 END as growing_network_flag,
    CASE WHEN activity_consistency_ratio >= 0.8 THEN 1 ELSE 0 END as consistent_user_flag,
    CASE WHEN lifecycle_stage = 'ACTIVE_USER' THEN 1 ELSE 0 END as currently_active_flag,
    CASE WHEN network_trajectory = 'EXPANDING' THEN 1 ELSE 0 END as expanding_influence_flag,
    
    -- Engagement quality indicators
    growth_months,
    ROUND(
        (connection_growth_rate + interaction_growth_rate) * activity_consistency_ratio::NUMERIC, 4
    ) as overall_engagement_score
    
FROM network_evolution
ORDER BY overall_engagement_score DESC, monthly_connection_growth DESC;
```

### Real-World Applications

1. **Social Network Analysis**: User influence, community detection, viral marketing
2. **Recommendation Systems**: Collaborative filtering, graph-based recommendations
3. **Fraud Detection**: Anomalous connection patterns, suspicious network behavior
4. **Knowledge Graphs**: Entity relationships, semantic similarity, information retrieval
5. **Supply Chain Analysis**: Supplier networks, risk propagation, optimization

### Best Practices

1. **Feature Scaling**: Normalize graph metrics for ML algorithms
2. **Temporal Windows**: Use appropriate time windows for dynamic features
3. **Sampling Strategies**: Handle large graphs with appropriate sampling
4. **Memory Management**: Optimize recursive queries and graph traversals
5. **Index Optimization**: Create indexes for graph relationship queries

### Common Pitfalls

1. **Scalability Issues**: Graph queries can become computationally expensive
2. **Feature Correlation**: Many graph features are highly correlated
3. **Dynamic Graphs**: Handling temporal changes in graph structure
4. **Sparse Graphs**: Dealing with disconnected components and low-degree nodes
5. **Interpretation Challenges**: Complex graph features can be difficult to interpret

### Performance Considerations

- **Query Optimization**: Use appropriate indexes and query plans
- **Memory Usage**: Graph computations can require significant memory
- **Computation Complexity**: Some centrality measures are computationally expensive
- **Data Freshness**: Balance between feature accuracy and computation time
- **Parallel Processing**: Leverage database parallelism for graph computations

---

## Question 14

**What are SQL Common Table Expressions (CTEs) and how can they be used for feature generation?**

**Answer:**

### Theory

Common Table Expressions (CTEs) are temporary named result sets that exist only within the scope of a single SQL statement. They provide a powerful mechanism for structuring complex queries, breaking down feature engineering pipelines into manageable steps, and creating reusable intermediate results. For machine learning applications, CTEs enable sophisticated feature transformations, hierarchical data processing, recursive calculations, and multi-step aggregations that would be difficult or impossible to achieve in a single query level.

**Key Advantages for ML Feature Engineering:**
- **Modularity**: Break complex feature calculations into logical steps
- **Readability**: Make complex transformations easier to understand and maintain  
- **Recursion**: Handle hierarchical and graph-like data structures
- **Performance**: Optimize query execution through logical decomposition
- **Reusability**: Reference intermediate results multiple times within the same query

### Basic CTE Structure and Feature Engineering

#### 1. Sequential Feature Transformations

```sql
-- Multi-stage customer feature engineering pipeline
WITH customer_base AS (
    -- Stage 1: Basic customer information
    SELECT 
        customer_id,
        registration_date,
        birth_date,
        location,
        account_type,
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date)) as age,
        EXTRACT(DAYS FROM AGE(CURRENT_DATE, registration_date)) as tenure_days
    FROM customers
    WHERE active = TRUE
      AND registration_date >= '2020-01-01'
),
transaction_summary AS (
    -- Stage 2: Transaction-based features
    SELECT 
        customer_id,
        COUNT(*) as total_transactions,
        SUM(amount) as total_spent,
        AVG(amount) as avg_transaction_amount,
        STDDEV(amount) as transaction_volatility,
        MIN(transaction_date) as first_transaction_date,
        MAX(transaction_date) as last_transaction_date,
        
        -- Percentile-based features
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) as q1_amount,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) as median_amount,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) as q3_amount,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) as p95_amount,
        
        -- Frequency patterns
        COUNT(DISTINCT DATE_TRUNC('month', transaction_date)) as active_months,
        COUNT(DISTINCT merchant_id) as unique_merchants,
        COUNT(DISTINCT product_category) as category_diversity,
        
        -- Recency features
        EXTRACT(DAYS FROM AGE(CURRENT_DATE, MAX(transaction_date))) as days_since_last_transaction,
        
        -- Seasonal patterns
        AVG(CASE WHEN EXTRACT(MONTH FROM transaction_date) IN (11, 12, 1) THEN amount END) as holiday_avg_spending,
        AVG(CASE WHEN EXTRACT(DOW FROM transaction_date) IN (6, 0) THEN amount END) as weekend_avg_spending
        
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '24 months'
      AND status = 'COMPLETED'
    GROUP BY customer_id
),
behavioral_patterns AS (
    -- Stage 3: Advanced behavioral analysis
    SELECT 
        ts.customer_id,
        ts.*,
        cb.age,
        cb.tenure_days,
        cb.location,
        cb.account_type,
        
        -- Derived behavioral features
        ts.total_spent::NUMERIC / NULLIF(ts.tenure_days, 0) as daily_spending_rate,
        ts.total_transactions::NUMERIC / NULLIF(ts.active_months, 0) as monthly_transaction_frequency,
        ts.transaction_volatility / NULLIF(ts.avg_transaction_amount, 0) as spending_consistency_ratio,
        
        -- Comparative features
        ts.avg_transaction_amount / NULLIF(ts.median_amount, 0) as mean_median_ratio,
        (ts.q3_amount - ts.q1_amount) / NULLIF(ts.median_amount, 0) as relative_iqr,
        ts.p95_amount / NULLIF(ts.avg_transaction_amount, 0) as outlier_ratio,
        
        -- Engagement patterns
        ts.category_diversity::NUMERIC / NULLIF(ts.unique_merchants, 0) as category_merchant_ratio,
        ts.active_months::NUMERIC / GREATEST(
            EXTRACT(MONTHS FROM AGE(ts.last_transaction_date, ts.first_transaction_date)) + 1, 1
        ) as activity_consistency_ratio,
        
        -- Seasonal behavior
        COALESCE(ts.holiday_avg_spending, 0) / NULLIF(ts.avg_transaction_amount, 0) as holiday_spending_ratio,
        COALESCE(ts.weekend_avg_spending, 0) / NULLIF(ts.avg_transaction_amount, 0) as weekend_spending_ratio,
        
        -- Age-based features
        CASE 
            WHEN cb.age BETWEEN 18 AND 25 THEN 'GEN_Z'
            WHEN cb.age BETWEEN 26 AND 41 THEN 'MILLENNIAL'
            WHEN cb.age BETWEEN 42 AND 57 THEN 'GEN_X'
            WHEN cb.age BETWEEN 58 AND 76 THEN 'BOOMER'
            ELSE 'SILENT'
        END as generation_cohort,
        
        -- Tenure-based features
        CASE 
            WHEN cb.tenure_days <= 90 THEN 'NEW_CUSTOMER'
            WHEN cb.tenure_days <= 365 THEN 'RECENT_CUSTOMER'
            WHEN cb.tenure_days <= 1095 THEN 'ESTABLISHED_CUSTOMER'
            ELSE 'VETERAN_CUSTOMER'
        END as customer_maturity
        
    FROM transaction_summary ts
    INNER JOIN customer_base cb ON ts.customer_id = cb.customer_id
),
risk_assessment AS (
    -- Stage 4: Risk and propensity scoring
    SELECT 
        bp.*,
        
        -- Churn risk indicators
        CASE 
            WHEN bp.days_since_last_transaction > 90 THEN 'HIGH_CHURN_RISK'
            WHEN bp.days_since_last_transaction > 30 THEN 'MEDIUM_CHURN_RISK'
            ELSE 'LOW_CHURN_RISK'
        END as churn_risk_category,
        
        -- Value-based segmentation
        CASE 
            WHEN bp.total_spent >= 5000 AND bp.avg_transaction_amount >= 100 THEN 'HIGH_VALUE'
            WHEN bp.total_spent >= 1000 AND bp.avg_transaction_amount >= 50 THEN 'MEDIUM_VALUE'
            WHEN bp.total_spent >= 200 THEN 'LOW_VALUE'
            ELSE 'MINIMAL_VALUE'
        END as customer_value_segment,
        
        -- Engagement level
        CASE 
            WHEN bp.monthly_transaction_frequency >= 5 AND bp.category_diversity >= 3 THEN 'HIGH_ENGAGEMENT'
            WHEN bp.monthly_transaction_frequency >= 2 AND bp.category_diversity >= 2 THEN 'MEDIUM_ENGAGEMENT'
            WHEN bp.monthly_transaction_frequency >= 0.5 THEN 'LOW_ENGAGEMENT'
            ELSE 'MINIMAL_ENGAGEMENT'
        END as engagement_level,
        
        -- Predictive scores (simplified)
        LEAST(100, 
            (bp.monthly_transaction_frequency * 10) +
            (bp.category_diversity * 5) +
            (CASE WHEN bp.days_since_last_transaction <= 7 THEN 20 ELSE 0 END) +
            (bp.activity_consistency_ratio * 30)
        ) as engagement_score,
        
        LEAST(100,
            (bp.total_spent / 100) +
            (bp.avg_transaction_amount / 10) +
            (bp.unique_merchants * 2) +
            (CASE WHEN bp.customer_maturity = 'VETERAN_CUSTOMER' THEN 15 ELSE 0 END)
        ) as lifetime_value_score,
        
        -- Behavioral flags
        CASE WHEN bp.spending_consistency_ratio <= 0.5 THEN 1 ELSE 0 END as consistent_spender_flag,
        CASE WHEN bp.holiday_spending_ratio >= 1.2 THEN 1 ELSE 0 END as holiday_shopper_flag,
        CASE WHEN bp.weekend_spending_ratio >= 1.1 THEN 1 ELSE 0 END as weekend_shopper_flag,
        CASE WHEN bp.outlier_ratio >= 3.0 THEN 1 ELSE 0 END as high_variance_spender_flag
        
    FROM behavioral_patterns bp
),
ml_features AS (
    -- Stage 5: Final ML feature preparation
    SELECT 
        ra.*,
        
        -- Interaction features
        ra.engagement_score * ra.lifetime_value_score / 100.0 as composite_value_score,
        
        -- Normalized features for ML models
        ra.total_spent::NUMERIC / (
            SELECT MAX(total_spent) FROM risk_assessment
        ) as normalized_total_spent,
        
        ra.avg_transaction_amount::NUMERIC / (
            SELECT MAX(avg_transaction_amount) FROM risk_assessment
        ) as normalized_avg_amount,
        
        ra.monthly_transaction_frequency::NUMERIC / (
            SELECT MAX(monthly_transaction_frequency) FROM risk_assessment
        ) as normalized_frequency,
        
        -- Percentile ranks
        PERCENT_RANK() OVER (ORDER BY ra.total_spent) as spending_percentile,
        PERCENT_RANK() OVER (ORDER BY ra.engagement_score) as engagement_percentile,
        PERCENT_RANK() OVER (ORDER BY ra.lifetime_value_score) as value_percentile,
        
        -- Clustering features (using standardized values)
        (ra.total_spent - AVG(ra.total_spent) OVER ()) / 
        NULLIF(STDDEV(ra.total_spent) OVER (), 0) as standardized_spending,
        
        (ra.monthly_transaction_frequency - AVG(ra.monthly_transaction_frequency) OVER ()) / 
        NULLIF(STDDEV(ra.monthly_transaction_frequency) OVER (), 0) as standardized_frequency,
        
        (ra.category_diversity - AVG(ra.category_diversity) OVER ()) / 
        NULLIF(STDDEV(ra.category_diversity) OVER (), 0) as standardized_diversity
        
    FROM risk_assessment ra
)
SELECT 
    customer_id,
    age,
    tenure_days,
    generation_cohort,
    customer_maturity,
    
    -- Core behavioral features
    total_transactions,
    total_spent,
    avg_transaction_amount,
    monthly_transaction_frequency,
    category_diversity,
    days_since_last_transaction,
    
    -- Derived features
    ROUND(daily_spending_rate::NUMERIC, 4) as spending_velocity,
    ROUND(spending_consistency_ratio::NUMERIC, 4) as consistency_score,
    ROUND(activity_consistency_ratio::NUMERIC, 4) as activity_reliability,
    
    -- Segmentation
    customer_value_segment,
    engagement_level,
    churn_risk_category,
    
    -- Predictive scores
    ROUND(engagement_score::NUMERIC, 2) as engagement_index,
    ROUND(lifetime_value_score::NUMERIC, 2) as value_index,
    ROUND(composite_value_score::NUMERIC, 2) as composite_score,
    
    -- Percentile features
    ROUND(spending_percentile::NUMERIC, 4) as spending_rank,
    ROUND(engagement_percentile::NUMERIC, 4) as engagement_rank,
    ROUND(value_percentile::NUMERIC, 4) as value_rank,
    
    -- Standardized features for clustering
    ROUND(standardized_spending::NUMERIC, 4) as z_spending,
    ROUND(standardized_frequency::NUMERIC, 4) as z_frequency,
    ROUND(standardized_diversity::NUMERIC, 4) as z_diversity,
    
    -- Binary flags
    consistent_spender_flag,
    holiday_shopper_flag,
    weekend_shopper_flag,
    high_variance_spender_flag,
    
    -- Target variables for different models
    CASE WHEN churn_risk_category = 'HIGH_CHURN_RISK' THEN 1 ELSE 0 END as churn_target,
    CASE WHEN customer_value_segment IN ('HIGH_VALUE', 'MEDIUM_VALUE') THEN 1 ELSE 0 END as valuable_customer_target,
    total_spent as regression_target_ltv
    
FROM ml_features
ORDER BY composite_score DESC;
```

#### 2. Recursive CTEs for Hierarchical Features

```sql
-- Recursive CTE for organizational hierarchy feature extraction
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: Top-level executives
    SELECT 
        employee_id,
        manager_id,
        name,
        department,
        salary,
        hire_date,
        job_level,
        0 as hierarchy_depth,
        ARRAY[employee_id] as management_path,
        employee_id as top_level_manager,
        salary as inherited_budget
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: Subordinates
    SELECT 
        e.employee_id,
        e.manager_id,
        e.name,
        e.department,
        e.salary,
        e.hire_date,
        e.job_level,
        eh.hierarchy_depth + 1,
        eh.management_path || e.employee_id,
        eh.top_level_manager,
        eh.inherited_budget + e.salary
    FROM employees e
    INNER JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
    WHERE eh.hierarchy_depth < 10  -- Prevent infinite recursion
),
hierarchy_aggregates AS (
    -- Aggregate features at each hierarchy level
    SELECT 
        eh.*,
        -- Team size calculations
        (SELECT COUNT(*) 
         FROM employee_hierarchy sub 
         WHERE eh.employee_id = ANY(sub.management_path)
           AND sub.employee_id != eh.employee_id
        ) as total_subordinates,
        
        (SELECT COUNT(*) 
         FROM employees direct 
         WHERE direct.manager_id = eh.employee_id
        ) as direct_reports,
        
        -- Salary analytics
        (SELECT AVG(salary) 
         FROM employee_hierarchy sub 
         WHERE eh.employee_id = ANY(sub.management_path)
           AND sub.hierarchy_depth = eh.hierarchy_depth + 1
        ) as avg_direct_report_salary,
        
        (SELECT MAX(salary) 
         FROM employee_hierarchy sub 
         WHERE eh.employee_id = ANY(sub.management_path)
        ) as max_team_salary,
        
        -- Span of control
        (SELECT MAX(hierarchy_depth) 
         FROM employee_hierarchy sub 
         WHERE eh.employee_id = ANY(sub.management_path)
        ) - eh.hierarchy_depth as max_reporting_depth,
        
        -- Department diversity
        (SELECT COUNT(DISTINCT department) 
         FROM employee_hierarchy sub 
         WHERE eh.employee_id = ANY(sub.management_path)
        ) as managed_departments,
        
        -- Tenure analysis within hierarchy
        (SELECT AVG(EXTRACT(DAYS FROM CURRENT_DATE - hire_date)) 
         FROM employee_hierarchy sub 
         WHERE eh.employee_id = ANY(sub.management_path)
        ) as avg_team_tenure,
        
        -- Performance indicators
        eh.salary / NULLIF(
            (SELECT AVG(salary) 
             FROM employee_hierarchy level_peers 
             WHERE level_peers.hierarchy_depth = eh.hierarchy_depth), 0
        ) as salary_vs_peers_ratio
        
    FROM employee_hierarchy eh
),
management_features AS (
    -- Calculate management-specific features
    SELECT 
        ha.*,
        
        -- Management efficiency metrics
        ha.total_subordinates::NUMERIC / NULLIF(ha.salary, 0) * 1000 as subordinates_per_salary_k,
        ha.inherited_budget / NULLIF(ha.salary, 0) as budget_leverage_ratio,
        
        -- Leadership span
        CASE 
            WHEN ha.hierarchy_depth = 0 THEN 'C_LEVEL'
            WHEN ha.hierarchy_depth = 1 THEN 'VP_LEVEL'
            WHEN ha.hierarchy_depth = 2 THEN 'DIRECTOR_LEVEL'
            WHEN ha.hierarchy_depth >= 3 AND ha.direct_reports > 0 THEN 'MANAGER_LEVEL'
            ELSE 'INDIVIDUAL_CONTRIBUTOR'
        END as management_tier,
        
        -- Management complexity
        (ha.total_subordinates * ha.managed_departments * ha.max_reporting_depth) as management_complexity_score,
        
        -- Career progression potential
        CASE 
            WHEN ha.hierarchy_depth >= 4 THEN 'LIMITED_PROGRESSION'
            WHEN ha.hierarchy_depth = 3 THEN 'MODERATE_PROGRESSION'
            WHEN ha.hierarchy_depth <= 2 THEN 'HIGH_PROGRESSION'
            ELSE 'EXECUTIVE_LEVEL'
        END as career_trajectory,
        
        -- Team performance indicators
        ha.avg_direct_report_salary / NULLIF(ha.salary, 0) as team_compensation_ratio,
        ha.max_team_salary / NULLIF(ha.salary, 0) as max_subordinate_ratio,
        
        -- Organizational influence
        array_length(ha.management_path, 1) as influence_path_length,
        ha.total_subordinates::NUMERIC / (
            SELECT SUM(total_subordinates) FROM hierarchy_aggregates 
            WHERE hierarchy_depth = 0
        ) as organizational_influence_share
        
    FROM hierarchy_aggregates ha
)
SELECT 
    employee_id,
    name,
    department,
    management_tier,
    hierarchy_depth,
    
    -- Team metrics
    total_subordinates,
    direct_reports,
    managed_departments,
    max_reporting_depth,
    
    -- Performance features
    ROUND(salary_vs_peers_ratio::NUMERIC, 3) as peer_salary_ratio,
    ROUND(budget_leverage_ratio::NUMERIC, 2) as budget_efficiency,
    ROUND(management_complexity_score::NUMERIC, 0) as complexity_index,
    ROUND(organizational_influence_share::NUMERIC, 6) as influence_percentage,
    
    -- Career features
    career_trajectory,
    ROUND(avg_team_tenure::NUMERIC, 0) as team_avg_tenure_days,
    ROUND(team_compensation_ratio::NUMERIC, 3) as team_salary_alignment,
    
    -- Binary features for classification
    CASE WHEN total_subordinates >= 10 THEN 1 ELSE 0 END as large_team_manager_flag,
    CASE WHEN managed_departments > 1 THEN 1 ELSE 0 END as cross_functional_manager_flag,
    CASE WHEN salary_vs_peers_ratio >= 1.2 THEN 1 ELSE 0 END as high_performer_flag,
    CASE WHEN max_subordinate_ratio >= 0.8 THEN 1 ELSE 0 END as talent_developer_flag,
    
    -- Hierarchical path features
    management_path,
    top_level_manager,
    influence_path_length,
    
    -- Target variables
    salary as target_salary,
    total_subordinates as target_team_size,
    management_complexity_score as target_complexity
    
FROM management_features
ORDER BY organizational_influence_share DESC, hierarchy_depth, total_subordinates DESC;
```

### Window Functions with CTEs

#### 1. Time Series Feature Engineering

```sql
-- Advanced time series features using CTEs and window functions
WITH daily_sales AS (
    -- Stage 1: Aggregate to daily level
    SELECT 
        product_id,
        DATE(order_date) as sale_date,
        SUM(quantity) as daily_quantity,
        SUM(quantity * unit_price) as daily_revenue,
        COUNT(*) as daily_order_count,
        COUNT(DISTINCT customer_id) as daily_unique_customers
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '2 years'
      AND o.status = 'COMPLETED'
    GROUP BY product_id, DATE(order_date)
),
time_series_base AS (
    -- Stage 2: Fill missing dates and add temporal features
    SELECT 
        p.product_id,
        d.calendar_date as sale_date,
        COALESCE(ds.daily_quantity, 0) as daily_quantity,
        COALESCE(ds.daily_revenue, 0) as daily_revenue,
        COALESCE(ds.daily_order_count, 0) as daily_order_count,
        COALESCE(ds.daily_unique_customers, 0) as daily_unique_customers,
        
        -- Temporal features
        EXTRACT(DOW FROM d.calendar_date) as day_of_week,
        EXTRACT(DAY FROM d.calendar_date) as day_of_month,
        EXTRACT(MONTH FROM d.calendar_date) as month_of_year,
        EXTRACT(QUARTER FROM d.calendar_date) as quarter_of_year,
        CASE WHEN EXTRACT(DOW FROM d.calendar_date) IN (0, 6) THEN 1 ELSE 0 END as is_weekend,
        CASE WHEN d.calendar_date IN (SELECT holiday_date FROM holidays) THEN 1 ELSE 0 END as is_holiday
        
    FROM (SELECT DISTINCT product_id FROM daily_sales) p
    CROSS JOIN (
        SELECT generate_series(
            CURRENT_DATE - INTERVAL '2 years',
            CURRENT_DATE - INTERVAL '1 day',
            INTERVAL '1 day'
        )::DATE as calendar_date
    ) d
    LEFT JOIN daily_sales ds ON p.product_id = ds.product_id AND d.calendar_date = ds.sale_date
),
rolling_features AS (
    -- Stage 3: Rolling window calculations
    SELECT 
        tsb.*,
        
        -- Moving averages (multiple windows)
        AVG(daily_quantity) OVER (
            PARTITION BY product_id 
            ORDER BY sale_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as ma_7d_quantity,
        
        AVG(daily_quantity) OVER (
            PARTITION BY product_id 
            ORDER BY sale_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as ma_30d_quantity,
        
        AVG(daily_quantity) OVER (
            PARTITION BY product_id 
            ORDER BY sale_date 
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        ) as ma_90d_quantity,
        
        -- Exponential moving averages (approximated)
        AVG(daily_quantity) OVER (
            PARTITION BY product_id 
            ORDER BY sale_date 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) as ema_14d_quantity,
        
        -- Rolling standard deviations
        STDDEV(daily_quantity) OVER (
            PARTITION BY product_id 
            ORDER BY sale_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_volatility,
        
        -- Rolling min/max
        MIN(daily_quantity) OVER (
            PARTITION BY product_id 
            ORDER BY sale_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_min,
        
        MAX(daily_quantity) OVER (
            PARTITION BY product_id 
            ORDER BY sale_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_max,
        
        -- Lag features
        LAG(daily_quantity, 1) OVER (PARTITION BY product_id ORDER BY sale_date) as lag_1d_quantity,
        LAG(daily_quantity, 7) OVER (PARTITION BY product_id ORDER BY sale_date) as lag_7d_quantity,
        LAG(daily_quantity, 30) OVER (PARTITION BY product_id ORDER BY sale_date) as lag_30d_quantity,
        LAG(daily_quantity, 365) OVER (PARTITION BY product_id ORDER BY sale_date) as lag_1y_quantity,
        
        -- Lead features (for target creation)
        LEAD(daily_quantity, 1) OVER (PARTITION BY product_id ORDER BY sale_date) as lead_1d_quantity,
        LEAD(daily_quantity, 7) OVER (PARTITION BY product_id ORDER BY sale_date) as lead_7d_quantity,
        
        -- Percentile-based features
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY daily_quantity) OVER (
            PARTITION BY product_id 
            ORDER BY sale_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_median,
        
        -- Cumulative features
        SUM(daily_quantity) OVER (
            PARTITION BY product_id 
            ORDER BY sale_date 
            ROWS UNBOUNDED PRECEDING
        ) as cumulative_quantity,
        
        AVG(daily_quantity) OVER (
            PARTITION BY product_id 
            ORDER BY sale_date 
            ROWS UNBOUNDED PRECEDING
        ) as cumulative_avg_quantity
        
    FROM time_series_base tsb
),
trend_features AS (
    -- Stage 4: Trend and momentum indicators
    SELECT 
        rf.*,
        
        -- Trend indicators
        daily_quantity - lag_1d_quantity as day_over_day_change,
        daily_quantity - lag_7d_quantity as week_over_week_change,
        daily_quantity - lag_30d_quantity as month_over_month_change,
        daily_quantity - lag_1y_quantity as year_over_year_change,
        
        -- Percentage changes
        CASE 
            WHEN lag_1d_quantity > 0 
            THEN (daily_quantity - lag_1d_quantity)::NUMERIC / lag_1d_quantity * 100
            ELSE NULL 
        END as pct_change_1d,
        
        CASE 
            WHEN lag_7d_quantity > 0 
            THEN (daily_quantity - lag_7d_quantity)::NUMERIC / lag_7d_quantity * 100
            ELSE NULL 
        END as pct_change_7d,
        
        CASE 
            WHEN lag_30d_quantity > 0 
            THEN (daily_quantity - lag_30d_quantity)::NUMERIC / lag_30d_quantity * 100
            ELSE NULL 
        END as pct_change_30d,
        
        -- Moving average convergence/divergence
        ma_7d_quantity - ma_30d_quantity as macd_signal,
        
        -- Bollinger bands approximation
        daily_quantity - (ma_30d_quantity + 2 * rolling_30d_volatility) as upper_band_distance,
        daily_quantity - (ma_30d_quantity - 2 * rolling_30d_volatility) as lower_band_distance,
        
        -- Relative strength index approximation
        CASE 
            WHEN rolling_30d_volatility > 0
            THEN (daily_quantity - rolling_30d_min) / (rolling_30d_max - rolling_30d_min) * 100
            ELSE 50
        END as rsi_approximation,
        
        -- Momentum indicators
        daily_quantity - ma_7d_quantity as momentum_vs_short_ma,
        daily_quantity - ma_30d_quantity as momentum_vs_long_ma,
        
        -- Volatility-adjusted returns
        CASE 
            WHEN rolling_30d_volatility > 0
            THEN (daily_quantity - ma_30d_quantity) / rolling_30d_volatility
            ELSE 0
        END as volatility_adjusted_position,
        
        -- Rank-based features
        PERCENT_RANK() OVER (
            PARTITION BY product_id 
            ORDER BY daily_quantity
        ) as historical_percentile_rank
        
    FROM rolling_features rf
),
seasonal_features AS (
    -- Stage 5: Seasonal decomposition features
    SELECT 
        tf.*,
        
        -- Seasonal averages
        AVG(daily_quantity) OVER (
            PARTITION BY product_id, day_of_week
        ) as avg_quantity_by_dow,
        
        AVG(daily_quantity) OVER (
            PARTITION BY product_id, day_of_month
        ) as avg_quantity_by_dom,
        
        AVG(daily_quantity) OVER (
            PARTITION BY product_id, month_of_year
        ) as avg_quantity_by_month,
        
        AVG(daily_quantity) OVER (
            PARTITION BY product_id, quarter_of_year
        ) as avg_quantity_by_quarter,
        
        -- Seasonal deviations
        daily_quantity - AVG(daily_quantity) OVER (
            PARTITION BY product_id, day_of_week
        ) as seasonal_deviation_dow,
        
        daily_quantity - AVG(daily_quantity) OVER (
            PARTITION BY product_id, month_of_year
        ) as seasonal_deviation_month,
        
        -- Holiday effects
        AVG(daily_quantity) OVER (
            PARTITION BY product_id, is_holiday
        ) as avg_quantity_by_holiday_flag,
        
        -- Weekend effects
        AVG(daily_quantity) OVER (
            PARTITION BY product_id, is_weekend
        ) as avg_quantity_by_weekend_flag,
        
        -- Cyclical patterns (simplified)
        daily_quantity - LAG(daily_quantity, 7) OVER (
            PARTITION BY product_id ORDER BY sale_date
        ) as weekly_cycle_deviation,
        
        daily_quantity - LAG(daily_quantity, 30) OVER (
            PARTITION BY product_id ORDER BY sale_date
        ) as monthly_cycle_deviation
        
    FROM trend_features tf
)
SELECT 
    product_id,
    sale_date,
    daily_quantity,
    daily_revenue,
    
    -- Temporal context
    day_of_week,
    month_of_year,
    is_weekend,
    is_holiday,
    
    -- Moving averages (rounded for readability)
    ROUND(ma_7d_quantity::NUMERIC, 2) as ma_7d,
    ROUND(ma_30d_quantity::NUMERIC, 2) as ma_30d,
    ROUND(ma_90d_quantity::NUMERIC, 2) as ma_90d,
    
    -- Trend features
    ROUND(pct_change_1d::NUMERIC, 2) as daily_growth_pct,
    ROUND(pct_change_7d::NUMERIC, 2) as weekly_growth_pct,
    ROUND(pct_change_30d::NUMERIC, 2) as monthly_growth_pct,
    
    -- Technical indicators
    ROUND(macd_signal::NUMERIC, 2) as trend_signal,
    ROUND(rsi_approximation::NUMERIC, 2) as strength_index,
    ROUND(volatility_adjusted_position::NUMERIC, 2) as vol_adj_position,
    
    -- Volatility measures
    ROUND(rolling_30d_volatility::NUMERIC, 2) as volatility_30d,
    
    -- Seasonal features
    ROUND(seasonal_deviation_dow::NUMERIC, 2) as dow_seasonality,
    ROUND(seasonal_deviation_month::NUMERIC, 2) as monthly_seasonality,
    
    -- Position indicators
    ROUND(historical_percentile_rank::NUMERIC, 4) as historical_rank,
    
    -- Binary indicators
    CASE WHEN daily_quantity > ma_30d_quantity THEN 1 ELSE 0 END as above_long_ma_flag,
    CASE WHEN pct_change_7d > 10 THEN 1 ELSE 0 END as strong_growth_flag,
    CASE WHEN rsi_approximation > 70 THEN 1 ELSE 0 END as overbought_flag,
    CASE WHEN rsi_approximation < 30 THEN 1 ELSE 0 END as oversold_flag,
    
    -- Target variables for forecasting
    lead_1d_quantity as target_next_day,
    lead_7d_quantity as target_next_week,
    
    -- Lag features for sequence models
    lag_1d_quantity as prev_1d,
    lag_7d_quantity as prev_7d,
    lag_30d_quantity as prev_30d
    
FROM seasonal_features
WHERE sale_date >= CURRENT_DATE - INTERVAL '1 year'  -- Recent data for training
ORDER BY product_id, sale_date;
```

### Advanced CTE Patterns

#### 1. Multi-Level Aggregations

```sql
-- Complex multi-level feature aggregation using CTEs
WITH transaction_level AS (
    -- Level 1: Individual transactions with enrichments
    SELECT 
        t.transaction_id,
        t.customer_id,
        t.merchant_id,
        t.amount,
        t.transaction_date,
        t.transaction_timestamp,
        c.customer_segment,
        c.registration_date,
        m.merchant_category,
        m.risk_level as merchant_risk,
        
        -- Transaction timing features
        EXTRACT(HOUR FROM t.transaction_timestamp) as hour_of_day,
        EXTRACT(DOW FROM t.transaction_date) as day_of_week,
        EXTRACT(DAY FROM t.transaction_date) as day_of_month,
        EXTRACT(MONTH FROM t.transaction_date) as month_of_year,
        
        -- Customer context at transaction time
        EXTRACT(DAYS FROM t.transaction_date - c.registration_date) as customer_age_at_transaction,
        
        -- Amount categories
        CASE 
            WHEN t.amount <= 10 THEN 'MICRO'
            WHEN t.amount <= 50 THEN 'SMALL'
            WHEN t.amount <= 200 THEN 'MEDIUM'
            WHEN t.amount <= 1000 THEN 'LARGE'
            ELSE 'XLARGE'
        END as amount_category
        
    FROM transactions t
    JOIN customers c ON t.customer_id = c.customer_id
    JOIN merchants m ON t.merchant_id = m.merchant_id
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '18 months'
      AND t.status = 'COMPLETED'
),
customer_daily AS (
    -- Level 2: Customer daily aggregations
    SELECT 
        customer_id,
        transaction_date,
        customer_segment,
        
        -- Daily transaction features
        COUNT(*) as daily_transaction_count,
        SUM(amount) as daily_total_amount,
        AVG(amount) as daily_avg_amount,
        STDDEV(amount) as daily_amount_stddev,
        MIN(amount) as daily_min_amount,
        MAX(amount) as daily_max_amount,
        
        -- Merchant diversity
        COUNT(DISTINCT merchant_id) as daily_unique_merchants,
        COUNT(DISTINCT merchant_category) as daily_unique_categories,
        
        -- Risk indicators
        COUNT(*) FILTER (WHERE merchant_risk = 'HIGH') as daily_high_risk_transactions,
        SUM(amount) FILTER (WHERE merchant_risk = 'HIGH') as daily_high_risk_amount,
        
        -- Timing patterns
        COUNT(*) FILTER (WHERE hour_of_day BETWEEN 9 AND 17) as business_hours_transactions,
        COUNT(*) FILTER (WHERE day_of_week IN (0, 6)) as weekend_transactions,
        
        -- Amount distribution
        COUNT(*) FILTER (WHERE amount_category = 'MICRO') as micro_transactions,
        COUNT(*) FILTER (WHERE amount_category = 'SMALL') as small_transactions,
        COUNT(*) FILTER (WHERE amount_category = 'MEDIUM') as medium_transactions,
        COUNT(*) FILTER (WHERE amount_category = 'LARGE') as large_transactions,
        COUNT(*) FILTER (WHERE amount_category = 'XLARGE') as xlarge_transactions,
        
        -- Temporal concentration
        MAX(hour_of_day) - MIN(hour_of_day) as daily_time_span,
        COUNT(DISTINCT hour_of_day) as active_hours_count
        
    FROM transaction_level
    GROUP BY customer_id, transaction_date, customer_segment
),
customer_weekly AS (
    -- Level 3: Customer weekly aggregations
    SELECT 
        customer_id,
        DATE_TRUNC('week', transaction_date) as week_start,
        customer_segment,
        
        -- Weekly activity patterns
        COUNT(*) as weekly_active_days,
        SUM(daily_transaction_count) as weekly_total_transactions,
        SUM(daily_total_amount) as weekly_total_amount,
        AVG(daily_avg_amount) as weekly_avg_daily_amount,
        
        -- Consistency measures
        STDDEV(daily_transaction_count) as weekly_transaction_consistency,
        STDDEV(daily_total_amount) as weekly_amount_consistency,
        
        -- Peak activity
        MAX(daily_transaction_count) as weekly_peak_transactions,
        MAX(daily_total_amount) as weekly_peak_amount,
        
        -- Diversity measures
        MAX(daily_unique_merchants) as weekly_max_merchants,
        MAX(daily_unique_categories) as weekly_max_categories,
        
        -- Risk patterns
        SUM(daily_high_risk_transactions) as weekly_risk_transactions,
        SUM(daily_high_risk_amount) as weekly_risk_amount,
        
        -- Behavioral consistency
        AVG(business_hours_transactions::NUMERIC / NULLIF(daily_transaction_count, 0)) as weekly_business_hours_ratio,
        AVG(weekend_transactions::NUMERIC / NULLIF(daily_transaction_count, 0)) as weekly_weekend_ratio
        
    FROM customer_daily
    GROUP BY customer_id, DATE_TRUNC('week', transaction_date), customer_segment
),
customer_monthly AS (
    -- Level 4: Customer monthly aggregations
    SELECT 
        customer_id,
        DATE_TRUNC('month', week_start) as month_start,
        customer_segment,
        
        -- Monthly activity
        COUNT(*) as monthly_active_weeks,
        SUM(weekly_total_transactions) as monthly_total_transactions,
        SUM(weekly_total_amount) as monthly_total_amount,
        AVG(weekly_avg_daily_amount) as monthly_avg_transaction_amount,
        
        -- Growth patterns
        weekly_total_amount - LAG(weekly_total_amount) OVER (
            PARTITION BY customer_id 
            ORDER BY week_start
        ) as week_over_week_amount_change,
        
        -- Behavioral stability
        STDDEV(weekly_total_transactions) as monthly_activity_volatility,
        STDDEV(weekly_total_amount) as monthly_spending_volatility,
        
        -- Peak performance
        MAX(weekly_peak_transactions) as monthly_peak_weekly_transactions,
        MAX(weekly_peak_amount) as monthly_peak_weekly_amount,
        
        -- Risk aggregation
        SUM(weekly_risk_transactions) as monthly_risk_transactions,
        SUM(weekly_risk_amount) as monthly_risk_amount,
        
        -- Consistency scores
        AVG(weekly_business_hours_ratio) as monthly_business_preference,
        AVG(weekly_weekend_ratio) as monthly_weekend_preference
        
    FROM customer_weekly
    GROUP BY customer_id, DATE_TRUNC('month', week_start), customer_segment
),
customer_features AS (
    -- Level 5: Final customer-level features
    SELECT 
        customer_id,
        customer_segment,
        
        -- Activity summary
        COUNT(*) as total_active_months,
        SUM(monthly_total_transactions) as total_transactions_18m,
        SUM(monthly_total_amount) as total_amount_18m,
        AVG(monthly_avg_transaction_amount) as avg_transaction_amount,
        
        -- Growth analysis
        REGR_SLOPE(monthly_total_amount, EXTRACT(EPOCH FROM month_start)) * 86400 * 30 as monthly_spending_trend,
        REGR_R2(monthly_total_amount, EXTRACT(EPOCH FROM month_start)) as spending_trend_consistency,
        
        -- Volatility measures
        AVG(monthly_spending_volatility) as avg_monthly_volatility,
        STDDEV(monthly_total_amount) as overall_spending_volatility,
        
        -- Peak analysis
        MAX(monthly_peak_weekly_amount) as highest_weekly_amount,
        AVG(monthly_peak_weekly_transactions) as avg_peak_weekly_transactions,
        
        -- Risk profile
        SUM(monthly_risk_transactions) as total_risk_transactions,
        SUM(monthly_risk_amount) as total_risk_amount,
        SUM(monthly_risk_amount) / NULLIF(SUM(monthly_total_amount), 0) as risk_amount_ratio,
        
        -- Behavioral consistency
        STDDEV(monthly_business_preference) as business_preference_consistency,
        STDDEV(monthly_weekend_preference) as weekend_preference_consistency,
        AVG(monthly_business_preference) as overall_business_preference,
        AVG(monthly_weekend_preference) as overall_weekend_preference,
        
        -- Activity patterns
        AVG(monthly_active_weeks) as avg_weekly_activity_rate,
        SUM(monthly_active_weeks) / (COUNT(*) * 4.33) as monthly_consistency_ratio,
        
        -- Recent vs historical performance
        AVG(CASE WHEN month_start >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '3 months') 
                 THEN monthly_total_amount END) as recent_3m_avg_spending,
        AVG(CASE WHEN month_start < DATE_TRUNC('month', CURRENT_DATE - INTERVAL '3 months') 
                 THEN monthly_total_amount END) as historical_avg_spending
        
    FROM customer_monthly
    GROUP BY customer_id, customer_segment
    HAVING COUNT(*) >= 6  -- Minimum 6 months of data
)
SELECT 
    customer_id,
    customer_segment,
    total_active_months,
    
    -- Core spending features
    total_transactions_18m,
    ROUND(total_amount_18m::NUMERIC, 2) as total_spending,
    ROUND(avg_transaction_amount::NUMERIC, 2) as avg_transaction_size,
    
    -- Growth and trend features
    ROUND(monthly_spending_trend::NUMERIC, 4) as spending_growth_rate,
    ROUND(spending_trend_consistency::NUMERIC, 4) as growth_consistency,
    
    -- Volatility features
    ROUND(overall_spending_volatility::NUMERIC, 2) as spending_volatility,
    ROUND(avg_monthly_volatility::NUMERIC, 2) as monthly_volatility,
    
    -- Risk features
    total_risk_transactions,
    ROUND(total_risk_amount::NUMERIC, 2) as risk_spending,
    ROUND(risk_amount_ratio::NUMERIC, 4) as risk_propensity,
    
    -- Behavioral features
    ROUND(overall_business_preference::NUMERIC, 4) as business_hours_preference,
    ROUND(overall_weekend_preference::NUMERIC, 4) as weekend_preference,
    ROUND(monthly_consistency_ratio::NUMERIC, 4) as activity_consistency,
    
    -- Performance comparison
    ROUND(COALESCE(recent_3m_avg_spending, 0)::NUMERIC, 2) as recent_monthly_avg,
    ROUND(COALESCE(historical_avg_spending, 0)::NUMERIC, 2) as historical_monthly_avg,
    
    -- Derived features
    ROUND(
        (COALESCE(recent_3m_avg_spending, 0) - COALESCE(historical_avg_spending, 0)) / 
        NULLIF(COALESCE(historical_avg_spending, 0), 0)::NUMERIC, 4
    ) as recent_vs_historical_ratio,
    
    -- Classification features
    CASE 
        WHEN monthly_spending_trend > 0 AND spending_trend_consistency > 0.5 THEN 'GROWING'
        WHEN monthly_spending_trend < 0 AND spending_trend_consistency > 0.5 THEN 'DECLINING'
        ELSE 'STABLE'
    END as spending_trajectory,
    
    CASE 
        WHEN overall_spending_volatility / NULLIF(avg_transaction_amount, 0) > 2 THEN 'HIGH_VOLATILITY'
        WHEN overall_spending_volatility / NULLIF(avg_transaction_amount, 0) > 1 THEN 'MEDIUM_VOLATILITY'
        ELSE 'LOW_VOLATILITY'
    END as volatility_segment,
    
    -- Binary flags for ML models
    CASE WHEN risk_amount_ratio > 0.1 THEN 1 ELSE 0 END as high_risk_user_flag,
    CASE WHEN monthly_spending_trend > 0 THEN 1 ELSE 0 END as growing_spender_flag,
    CASE WHEN activity_consistency >= 0.8 THEN 1 ELSE 0 END as consistent_user_flag,
    CASE WHEN COALESCE(recent_3m_avg_spending, 0) > COALESCE(historical_avg_spending, 0) * 1.2 THEN 1 ELSE 0 END as accelerating_flag
    
FROM customer_features
ORDER BY total_spending DESC;
```

### Real-World Applications

1. **Feature Engineering Pipelines**: Multi-stage transformation workflows
2. **Data Quality**: Cleansing and validation in structured steps
3. **Recursive Analysis**: Hierarchical data processing for ML features
4. **Performance Optimization**: Breaking complex queries into manageable parts
5. **Code Maintainability**: Improving readability and debugging capabilities

### Best Practices

1. **Logical Decomposition**: Break complex logic into clear, named steps
2. **Reusability**: Design CTEs that can be referenced multiple times
3. **Performance Optimization**: Consider materialization vs. repeated computation
4. **Naming Conventions**: Use descriptive names for CTEs and columns
5. **Documentation**: Comment complex transformations within CTEs

### Common Pitfalls

1. **Over-Engineering**: Creating unnecessarily complex CTE hierarchies
2. **Performance Overhead**: Repeated CTE evaluation without optimization
3. **Memory Usage**: Large intermediate results consuming excessive memory
4. **Recursive Limits**: Infinite recursion or excessive depth
5. **Maintainability**: Overly complex CTEs that are difficult to debug

### Performance Considerations

- **Materialization**: Database optimizer decisions on CTE evaluation
- **Index Usage**: Ensure proper indexing for CTE join conditions
- **Memory Management**: Monitor memory usage for large intermediate results
- **Query Planning**: Analyze execution plans for CTE optimization opportunities
- **Recursive Depth**: Set appropriate limits for recursive CTEs to prevent resource exhaustion

---

## Question 15

**Explain the role of partitioning in large-scale SQL databases.**

**Answer:**

### Theory

Database partitioning is a technique that divides large tables into smaller, more manageable pieces called partitions while maintaining the logical view of a single table. For machine learning applications dealing with massive datasets, partitioning provides crucial benefits including improved query performance, parallel processing capabilities, efficient data lifecycle management, and enhanced maintenance operations. Partitioning enables ML pipelines to process large-scale data efficiently by allowing queries to access only relevant data segments, reducing I/O operations and improving overall system performance.

**Key Benefits for ML Applications:**
- **Query Performance**: Partition pruning reduces data scan volumes
- **Parallel Processing**: Enable concurrent operations across partitions
- **Data Management**: Efficient archiving and deletion of historical data
- **Maintenance Operations**: Faster indexing, statistics updates, and backups
- **Resource Optimization**: Better memory and storage utilization

### Partition Types and Strategies

#### 1. Range Partitioning for Time-Series Data

```sql
-- Range partitioning for transaction data
CREATE TABLE transactions_partitioned (
    transaction_id BIGINT NOT NULL,
    customer_id INTEGER NOT NULL,
    merchant_id INTEGER NOT NULL,
    amount NUMERIC(10,2) NOT NULL,
    transaction_date DATE NOT NULL,
    transaction_timestamp TIMESTAMP NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING',
    payment_method VARCHAR(50),
    merchant_category VARCHAR(100),
    location_data JSONB,
    
    -- Partitioning constraints
    CONSTRAINT pk_transactions_partitioned PRIMARY KEY (transaction_id, transaction_date),
    CONSTRAINT chk_amount_positive CHECK (amount > 0)
) PARTITION BY RANGE (transaction_date);

-- Create monthly partitions for current and future data
CREATE TABLE transactions_2024_01 PARTITION OF transactions_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE transactions_2024_02 PARTITION OF transactions_partitioned
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE transactions_2024_03 PARTITION OF transactions_partitioned
FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

-- Historical data partitions (quarterly for older data)
CREATE TABLE transactions_2023_q4 PARTITION OF transactions_partitioned
FOR VALUES FROM ('2023-10-01') TO ('2024-01-01');

CREATE TABLE transactions_2023_q3 PARTITION OF transactions_partitioned
FOR VALUES FROM ('2023-07-01') TO ('2023-10-01');

-- Default partition for future dates
CREATE TABLE transactions_default PARTITION OF transactions_partitioned
DEFAULT;

-- Create indexes on partitions for ML query optimization
CREATE INDEX idx_trans_2024_01_customer_date 
ON transactions_2024_01 (customer_id, transaction_date);

CREATE INDEX idx_trans_2024_01_merchant_category 
ON transactions_2024_01 (merchant_id, merchant_category);

CREATE INDEX idx_trans_2024_01_amount_method 
ON transactions_2024_01 (amount, payment_method);

-- Partition-wise join optimization
CREATE INDEX idx_trans_2024_01_customer_hash 
ON transactions_2024_01 USING HASH (customer_id);

-- ML feature extraction leveraging partitioning
WITH monthly_features AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', transaction_date) as feature_month,
        COUNT(*) as monthly_transactions,
        SUM(amount) as monthly_spending,
        AVG(amount) as avg_transaction_amount,
        STDDEV(amount) as spending_volatility,
        COUNT(DISTINCT merchant_id) as merchant_diversity,
        COUNT(DISTINCT merchant_category) as category_diversity,
        
        -- Time-based patterns
        COUNT(*) FILTER (
            WHERE EXTRACT(HOUR FROM transaction_timestamp) BETWEEN 9 AND 17
        ) as business_hours_transactions,
        
        COUNT(*) FILTER (
            WHERE EXTRACT(DOW FROM transaction_date) IN (0, 6)
        ) as weekend_transactions,
        
        -- Payment method preferences
        COUNT(*) FILTER (WHERE payment_method = 'CREDIT_CARD') as credit_card_usage,
        COUNT(*) FILTER (WHERE payment_method = 'DEBIT_CARD') as debit_card_usage,
        COUNT(*) FILTER (WHERE payment_method = 'DIGITAL_WALLET') as digital_wallet_usage,
        
        -- Amount distribution
        COUNT(*) FILTER (WHERE amount <= 50) as small_transactions,
        COUNT(*) FILTER (WHERE amount > 50 AND amount <= 200) as medium_transactions,
        COUNT(*) FILTER (WHERE amount > 200) as large_transactions,
        
        -- Location diversity (if using location data)
        COUNT(DISTINCT location_data->>'city') as city_diversity
        
    FROM transactions_partitioned
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
      AND status = 'COMPLETED'
    GROUP BY customer_id, DATE_TRUNC('month', transaction_date)
),
customer_trajectory AS (
    SELECT 
        customer_id,
        
        -- Temporal evolution features
        REGR_SLOPE(monthly_spending, EXTRACT(EPOCH FROM feature_month)) * 86400 * 30 as spending_trend,
        REGR_R2(monthly_spending, EXTRACT(EPOCH FROM feature_month)) as trend_consistency,
        
        -- Activity patterns
        AVG(monthly_transactions) as avg_monthly_activity,
        STDDEV(monthly_transactions) as activity_volatility,
        COUNT(DISTINCT feature_month) as active_months,
        
        -- Behavioral consistency
        STDDEV(avg_transaction_amount) as amount_consistency,
        STDDEV(merchant_diversity) as merchant_preference_stability,
        
        -- Growth metrics
        MAX(monthly_spending) - MIN(monthly_spending) as spending_range,
        LAST_VALUE(monthly_spending) OVER (
            PARTITION BY customer_id 
            ORDER BY feature_month 
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) - FIRST_VALUE(monthly_spending) OVER (
            PARTITION BY customer_id 
            ORDER BY feature_month 
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as total_spending_change,
        
        -- Channel preferences
        AVG(business_hours_transactions::NUMERIC / NULLIF(monthly_transactions, 0)) as business_hours_preference,
        AVG(weekend_transactions::NUMERIC / NULLIF(monthly_transactions, 0)) as weekend_preference,
        AVG(digital_wallet_usage::NUMERIC / NULLIF(monthly_transactions, 0)) as digital_adoption_rate
        
    FROM monthly_features
    GROUP BY customer_id
    HAVING COUNT(DISTINCT feature_month) >= 6  -- Minimum 6 months for reliable trends
)
SELECT 
    customer_id,
    active_months,
    ROUND(avg_monthly_activity::NUMERIC, 2) as typical_monthly_transactions,
    ROUND(spending_trend::NUMERIC, 4) as monthly_spending_growth,
    ROUND(trend_consistency::NUMERIC, 4) as growth_predictability,
    ROUND(activity_volatility::NUMERIC, 2) as activity_stability,
    ROUND(business_hours_preference::NUMERIC, 4) as business_preference,
    ROUND(digital_adoption_rate::NUMERIC, 4) as digital_engagement,
    
    -- Classification features
    CASE 
        WHEN spending_trend > 0 AND trend_consistency > 0.7 THEN 'GROWING'
        WHEN spending_trend < 0 AND trend_consistency > 0.7 THEN 'DECLINING'
        ELSE 'STABLE'
    END as spending_trajectory,
    
    CASE 
        WHEN digital_adoption_rate > 0.3 THEN 'DIGITAL_NATIVE'
        WHEN digital_adoption_rate > 0.1 THEN 'DIGITAL_ADOPTER'
        ELSE 'TRADITIONAL'
    END as digital_persona,
    
    -- Binary flags for ML models
    CASE WHEN spending_trend > 0 THEN 1 ELSE 0 END as growth_flag,
    CASE WHEN activity_volatility < avg_monthly_activity * 0.3 THEN 1 ELSE 0 END as stable_activity_flag
    
FROM customer_trajectory
ORDER BY spending_trend DESC;

-- Partition maintenance automation
CREATE OR REPLACE FUNCTION create_monthly_partition(target_date DATE)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    start_date := DATE_TRUNC('month', target_date);
    end_date := start_date + INTERVAL '1 month';
    partition_name := 'transactions_' || TO_CHAR(start_date, 'YYYY_MM');
    
    EXECUTE format('
        CREATE TABLE %I PARTITION OF transactions_partitioned
        FOR VALUES FROM (%L) TO (%L)',
        partition_name, start_date, end_date
    );
    
    -- Create partition-specific indexes
    EXECUTE format('
        CREATE INDEX %I ON %I (customer_id, transaction_date)',
        'idx_' || partition_name || '_customer_date', partition_name
    );
    
    EXECUTE format('
        CREATE INDEX %I ON %I (merchant_id, merchant_category)',
        'idx_' || partition_name || '_merchant', partition_name
    );
    
    RAISE NOTICE 'Created partition % for period % to %', partition_name, start_date, end_date;
END;
$$ LANGUAGE plpgsql;

-- Schedule partition creation (using pg_cron if available)
-- SELECT cron.schedule('create-monthly-partitions', '0 0 25 * *', 
--   'SELECT create_monthly_partition(CURRENT_DATE + INTERVAL ''1 month'')');
```

#### 2. Hash Partitioning for Load Distribution

```sql
-- Hash partitioning for customer data distribution
CREATE TABLE customers_partitioned (
    customer_id INTEGER NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    registration_date DATE NOT NULL,
    birth_date DATE,
    location VARCHAR(100),
    customer_segment VARCHAR(50),
    lifetime_value NUMERIC(12,2),
    risk_score NUMERIC(5,2),
    last_activity_date DATE,
    preferences JSONB,
    
    CONSTRAINT pk_customers_partitioned PRIMARY KEY (customer_id)
) PARTITION BY HASH (customer_id);

-- Create hash partitions for load distribution
CREATE TABLE customers_p0 PARTITION OF customers_partitioned
FOR VALUES WITH (modulus 8, remainder 0);

CREATE TABLE customers_p1 PARTITION OF customers_partitioned
FOR VALUES WITH (modulus 8, remainder 1);

CREATE TABLE customers_p2 PARTITION OF customers_partitioned
FOR VALUES WITH (modulus 8, remainder 2);

CREATE TABLE customers_p3 PARTITION OF customers_partitioned
FOR VALUES WITH (modulus 8, remainder 3);

CREATE TABLE customers_p4 PARTITION OF customers_partitioned
FOR VALUES WITH (modulus 8, remainder 4);

CREATE TABLE customers_p5 PARTITION OF customers_partitioned
FOR VALUES WITH (modulus 8, remainder 5);

CREATE TABLE customers_p6 PARTITION OF customers_partitioned
FOR VALUES WITH (modulus 8, remainder 6);

CREATE TABLE customers_p7 PARTITION OF customers_partitioned
FOR VALUES WITH (modulus 8, remainder 7);

-- Parallel feature computation across hash partitions
WITH parallel_customer_analysis AS (
    SELECT 
        cp.customer_id,
        cp.customer_segment,
        cp.registration_date,
        cp.lifetime_value,
        cp.risk_score,
        
        -- Concurrent transaction analysis per partition
        COALESCE(t.total_transactions, 0) as total_transactions,
        COALESCE(t.total_amount, 0) as total_spending,
        COALESCE(t.avg_amount, 0) as avg_transaction_amount,
        COALESCE(t.merchant_count, 0) as unique_merchants,
        COALESCE(t.days_since_last, 999) as days_since_last_transaction,
        
        -- Cross-partition join optimization
        COALESCE(o.order_count, 0) as total_orders,
        COALESCE(o.product_diversity, 0) as product_categories_purchased,
        COALESCE(o.return_rate, 0) as return_frequency,
        
        -- Support interaction analysis
        COALESCE(s.support_tickets, 0) as support_interactions,
        COALESCE(s.avg_resolution_time, 0) as avg_support_resolution_hours
        
    FROM customers_partitioned cp
    
    -- Partition-wise join with transactions
    LEFT JOIN (
        SELECT 
            customer_id,
            COUNT(*) as total_transactions,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount,
            COUNT(DISTINCT merchant_id) as merchant_count,
            EXTRACT(DAYS FROM CURRENT_DATE - MAX(transaction_date)) as days_since_last
        FROM transactions_partitioned
        WHERE transaction_date >= CURRENT_DATE - INTERVAL '24 months'
          AND status = 'COMPLETED'
        GROUP BY customer_id
    ) t ON cp.customer_id = t.customer_id
    
    -- Partition-wise join with orders
    LEFT JOIN (
        SELECT 
            customer_id,
            COUNT(*) as order_count,
            COUNT(DISTINCT product_category) as product_diversity,
            COUNT(*) FILTER (WHERE status = 'RETURNED')::NUMERIC / COUNT(*) as return_rate
        FROM orders_partitioned
        WHERE order_date >= CURRENT_DATE - INTERVAL '12 months'
        GROUP BY customer_id
    ) o ON cp.customer_id = o.customer_id
    
    -- Support ticket analysis
    LEFT JOIN (
        SELECT 
            customer_id,
            COUNT(*) as support_tickets,
            AVG(EXTRACT(HOURS FROM resolution_timestamp - created_timestamp)) as avg_resolution_time
        FROM support_tickets
        WHERE created_date >= CURRENT_DATE - INTERVAL '12 months'
        GROUP BY customer_id
    ) s ON cp.customer_id = s.customer_id
),
enriched_features AS (
    SELECT 
        pca.*,
        
        -- Behavioral scoring
        CASE 
            WHEN pca.total_transactions >= 50 AND pca.days_since_last <= 30 THEN 'HIGH_ACTIVITY'
            WHEN pca.total_transactions >= 20 AND pca.days_since_last <= 60 THEN 'MEDIUM_ACTIVITY'
            WHEN pca.total_transactions >= 5 AND pca.days_since_last <= 120 THEN 'LOW_ACTIVITY'
            ELSE 'DORMANT'
        END as activity_segment,
        
        -- Value segmentation
        CASE 
            WHEN pca.lifetime_value >= 5000 THEN 'PREMIUM'
            WHEN pca.lifetime_value >= 1000 THEN 'VALUABLE'
            WHEN pca.lifetime_value >= 200 THEN 'REGULAR'
            ELSE 'STARTER'
        END as value_tier,
        
        -- Engagement complexity
        (pca.unique_merchants * pca.product_categories_purchased) as engagement_complexity,
        
        -- Support efficiency
        CASE 
            WHEN pca.support_interactions > 0 AND pca.avg_support_resolution_hours <= 24 THEN 'EFFICIENT_SUPPORT'
            WHEN pca.support_interactions > 0 AND pca.avg_support_resolution_hours <= 72 THEN 'STANDARD_SUPPORT'
            WHEN pca.support_interactions > 0 THEN 'SLOW_SUPPORT'
            ELSE 'NO_SUPPORT_NEEDED'
        END as support_efficiency,
        
        -- Cross-partition computed features
        pca.total_spending / NULLIF(pca.total_transactions, 0) as computed_avg_transaction,
        pca.total_orders::NUMERIC / NULLIF(pca.total_transactions, 0) as order_to_transaction_ratio
        
    FROM parallel_customer_analysis pca
)
SELECT 
    customer_id,
    customer_segment,
    activity_segment,
    value_tier,
    support_efficiency,
    
    -- Core metrics
    total_transactions,
    total_spending,
    total_orders,
    unique_merchants,
    product_categories_purchased,
    
    -- Derived features
    ROUND(computed_avg_transaction::NUMERIC, 2) as avg_transaction_value,
    ROUND(order_to_transaction_ratio::NUMERIC, 4) as order_efficiency,
    engagement_complexity,
    days_since_last_transaction,
    
    -- Risk and quality indicators
    risk_score,
    ROUND(return_frequency::NUMERIC, 4) as return_propensity,
    support_interactions,
    ROUND(avg_support_resolution_hours::NUMERIC, 1) as support_efficiency_hours,
    
    -- Binary classification features
    CASE WHEN activity_segment IN ('HIGH_ACTIVITY', 'MEDIUM_ACTIVITY') THEN 1 ELSE 0 END as active_customer_flag,
    CASE WHEN value_tier IN ('PREMIUM', 'VALUABLE') THEN 1 ELSE 0 END as high_value_flag,
    CASE WHEN risk_score <= 0.3 THEN 1 ELSE 0 END as low_risk_flag,
    CASE WHEN return_frequency <= 0.05 THEN 1 ELSE 0 END as low_return_flag,
    
    -- ML model targets
    CASE WHEN days_since_last_transaction > 90 THEN 1 ELSE 0 END as churn_target,
    lifetime_value as ltv_target,
    total_spending as spending_target
    
FROM enriched_features
ORDER BY lifetime_value DESC, total_spending DESC;
```

#### 3. List Partitioning for Categorical Data

```sql
-- List partitioning by geographic region
CREATE TABLE orders_regional (
    order_id BIGINT NOT NULL,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    order_date DATE NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10,2) NOT NULL,
    shipping_address JSONB,
    region VARCHAR(50) NOT NULL,
    order_status VARCHAR(20),
    
    CONSTRAINT pk_orders_regional PRIMARY KEY (order_id, region)
) PARTITION BY LIST (region);

-- Regional partitions
CREATE TABLE orders_north_america PARTITION OF orders_regional
FOR VALUES IN ('US', 'CA', 'MX');

CREATE TABLE orders_europe PARTITION OF orders_regional
FOR VALUES IN ('UK', 'DE', 'FR', 'IT', 'ES', 'NL');

CREATE TABLE orders_asia_pacific PARTITION OF orders_regional
FOR VALUES IN ('JP', 'AU', 'SG', 'HK', 'IN');

CREATE TABLE orders_other PARTITION OF orders_regional
DEFAULT;

-- Regional ML feature analysis
WITH regional_performance AS (
    SELECT 
        region,
        product_id,
        
        -- Sales metrics by region
        COUNT(*) as total_orders,
        SUM(quantity) as total_quantity,
        SUM(quantity * unit_price) as total_revenue,
        AVG(quantity * unit_price) as avg_order_value,
        COUNT(DISTINCT customer_id) as unique_customers,
        
        -- Regional temporal patterns
        COUNT(*) FILTER (
            WHERE EXTRACT(MONTH FROM order_date) IN (11, 12)
        ) as holiday_season_orders,
        
        COUNT(*) FILTER (
            WHERE EXTRACT(DOW FROM order_date) IN (0, 6)
        ) as weekend_orders,
        
        -- Customer behavior by region
        AVG(quantity) as avg_quantity_per_order,
        STDDEV(quantity * unit_price) as order_value_volatility,
        
        -- Market penetration
        COUNT(DISTINCT customer_id)::NUMERIC / (
            SELECT COUNT(DISTINCT customer_id) 
            FROM customers_partitioned 
            WHERE location LIKE '%' || orders_regional.region || '%'
        ) as market_penetration_rate
        
    FROM orders_regional
    WHERE order_date >= CURRENT_DATE - INTERVAL '12 months'
      AND order_status = 'COMPLETED'
    GROUP BY region, product_id
),
cross_regional_analysis AS (
    SELECT 
        product_id,
        
        -- Cross-regional performance metrics
        COUNT(DISTINCT region) as regions_sold,
        SUM(total_revenue) as global_revenue,
        AVG(avg_order_value) as avg_regional_order_value,
        STDDEV(avg_order_value) as regional_price_variation,
        
        -- Regional market share
        MAX(total_revenue) - MIN(total_revenue) as revenue_range_across_regions,
        MAX(market_penetration_rate) as best_regional_penetration,
        MIN(market_penetration_rate) as weakest_regional_penetration,
        
        -- Seasonal consistency across regions
        STDDEV(holiday_season_orders::NUMERIC / NULLIF(total_orders, 0)) as holiday_seasonality_variance,
        
        -- Regional preference indicators
        SUM(total_orders) / COUNT(DISTINCT region) as avg_regional_demand,
        
        -- Market diversification
        1 - (MAX(total_revenue) / SUM(total_revenue)) as market_diversification_index
        
    FROM regional_performance
    GROUP BY product_id
    HAVING COUNT(DISTINCT region) >= 2  -- Products sold in multiple regions
),
product_regional_features AS (
    SELECT 
        rp.product_id,
        rp.region,
        
        -- Regional performance
        rp.total_orders,
        rp.total_revenue,
        rp.avg_order_value,
        rp.unique_customers,
        rp.market_penetration_rate,
        
        -- Cross-regional context
        cra.regions_sold,
        cra.global_revenue,
        cra.regional_price_variation,
        cra.market_diversification_index,
        
        -- Regional ranking
        RANK() OVER (
            PARTITION BY rp.product_id 
            ORDER BY rp.total_revenue DESC
        ) as regional_revenue_rank,
        
        ROW_NUMBER() OVER (
            PARTITION BY rp.region 
            ORDER BY rp.total_revenue DESC
        ) as product_rank_in_region,
        
        -- Market share within region
        rp.total_revenue / SUM(rp.total_revenue) OVER (PARTITION BY rp.region) as regional_market_share,
        
        -- Performance relative to global average
        rp.avg_order_value / cra.avg_regional_order_value as relative_order_value,
        rp.market_penetration_rate / cra.best_regional_penetration as penetration_efficiency,
        
        -- Seasonal behavior
        rp.holiday_season_orders::NUMERIC / NULLIF(rp.total_orders, 0) as holiday_preference,
        rp.weekend_orders::NUMERIC / NULLIF(rp.total_orders, 0) as weekend_preference
        
    FROM regional_performance rp
    JOIN cross_regional_analysis cra ON rp.product_id = cra.product_id
)
SELECT 
    product_id,
    region,
    
    -- Regional performance metrics
    total_orders,
    ROUND(total_revenue::NUMERIC, 2) as revenue,
    ROUND(avg_order_value::NUMERIC, 2) as aov,
    unique_customers,
    ROUND(market_penetration_rate::NUMERIC, 4) as penetration_rate,
    
    -- Global context
    regions_sold as global_presence,
    regional_revenue_rank,
    product_rank_in_region,
    ROUND(regional_market_share::NUMERIC, 4) as market_share,
    
    -- Performance indicators
    ROUND(relative_order_value::NUMERIC, 3) as aov_vs_global,
    ROUND(penetration_efficiency::NUMERIC, 3) as penetration_vs_best,
    ROUND(market_diversification_index::NUMERIC, 4) as diversification_score,
    
    -- Behavioral patterns
    ROUND(holiday_preference::NUMERIC, 4) as holiday_seasonality,
    ROUND(weekend_preference::NUMERIC, 4) as weekend_activity,
    
    -- Classification features
    CASE 
        WHEN regional_revenue_rank = 1 THEN 'TOP_REGIONAL_PERFORMER'
        WHEN regional_revenue_rank <= 3 THEN 'STRONG_REGIONAL_PERFORMER'
        ELSE 'STANDARD_REGIONAL_PERFORMER'
    END as regional_performance_tier,
    
    CASE 
        WHEN market_penetration_rate > 0.1 THEN 'HIGH_PENETRATION'
        WHEN market_penetration_rate > 0.05 THEN 'MEDIUM_PENETRATION'
        ELSE 'LOW_PENETRATION'
    END as penetration_category,
    
    -- Binary flags for ML models
    CASE WHEN regional_revenue_rank <= 3 THEN 1 ELSE 0 END as top_performer_flag,
    CASE WHEN market_penetration_rate >= 0.05 THEN 1 ELSE 0 END as well_penetrated_flag,
    CASE WHEN relative_order_value >= 1.1 THEN 1 ELSE 0 END as premium_market_flag,
    CASE WHEN holiday_preference >= 0.3 THEN 1 ELSE 0 END as holiday_sensitive_flag,
    
    -- Target variables
    total_revenue as target_revenue,
    market_penetration_rate as target_penetration
    
FROM product_regional_features
ORDER BY total_revenue DESC, regional_revenue_rank;
```

### Performance Optimization Strategies

#### 1. Partition Pruning and Query Optimization

```sql
-- Partition-aware query optimization examples
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    customer_id,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount
FROM transactions_partitioned
WHERE transaction_date BETWEEN '2024-01-01' AND '2024-01-31'  -- Partition pruning
  AND status = 'COMPLETED'
GROUP BY customer_id
HAVING SUM(amount) > 1000;

-- Partition-wise joins for better performance
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    c.customer_id,
    c.customer_segment,
    t.monthly_spending,
    t.transaction_count
FROM customers_partitioned c
JOIN (
    SELECT 
        customer_id,
        SUM(amount) as monthly_spending,
        COUNT(*) as transaction_count
    FROM transactions_partitioned
    WHERE transaction_date >= DATE_TRUNC('month', CURRENT_DATE)
    GROUP BY customer_id
) t ON c.customer_id = t.customer_id
WHERE c.customer_segment IN ('PREMIUM', 'GOLD');

-- Parallel partition processing
SET max_parallel_workers_per_gather = 4;
SET parallel_tuple_cost = 0.1;

SELECT 
    DATE_TRUNC('month', transaction_date) as month,
    merchant_category,
    COUNT(*) as transaction_count,
    SUM(amount) as total_revenue,
    AVG(amount) as avg_transaction_amount
FROM transactions_partitioned
WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', transaction_date), merchant_category
ORDER BY month DESC, total_revenue DESC;
```

#### 2. Partition Maintenance and Lifecycle Management

```sql
-- Automated partition management functions
CREATE OR REPLACE FUNCTION archive_old_partitions(
    table_name TEXT,
    retention_months INTEGER DEFAULT 24
)
RETURNS TABLE (
    partition_name TEXT,
    action_taken TEXT,
    rows_affected BIGINT
) AS $$
DECLARE
    partition_record RECORD;
    cutoff_date DATE;
    partition_sql TEXT;
    row_count BIGINT;
BEGIN
    cutoff_date := CURRENT_DATE - INTERVAL '1 month' * retention_months;
    
    FOR partition_record IN
        SELECT schemaname, tablename, 
               regexp_replace(tablename, '.*_(\d{4}_\d{2})$', '\1') as date_part
        FROM pg_tables
        WHERE tablename LIKE table_name || '_%'
          AND schemaname = 'public'
          AND regexp_replace(tablename, '.*_(\d{4}_\d{2})$', '\1') ~ '^\d{4}_\d{2}$'
    LOOP
        IF TO_DATE(partition_record.date_part, 'YYYY_MM') < cutoff_date THEN
            -- Get row count before archiving
            EXECUTE format('SELECT COUNT(*) FROM %I.%I', 
                          partition_record.schemaname, 
                          partition_record.tablename) INTO row_count;
            
            -- Archive to separate schema or external storage
            EXECUTE format('CREATE TABLE archived.%I AS SELECT * FROM %I.%I',
                          partition_record.tablename,
                          partition_record.schemaname,
                          partition_record.tablename);
            
            -- Drop the original partition
            EXECUTE format('DROP TABLE %I.%I',
                          partition_record.schemaname,
                          partition_record.tablename);
            
            RETURN QUERY SELECT partition_record.tablename, 'ARCHIVED', row_count;
        END IF;
    END LOOP;
    
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- Partition statistics maintenance
CREATE OR REPLACE FUNCTION update_partition_statistics(
    table_pattern TEXT
)
RETURNS VOID AS $$
DECLARE
    partition_record RECORD;
BEGIN
    FOR partition_record IN
        SELECT schemaname, tablename
        FROM pg_tables
        WHERE tablename LIKE table_pattern
          AND schemaname = 'public'
    LOOP
        EXECUTE format('ANALYZE %I.%I', partition_record.schemaname, partition_record.tablename);
        RAISE NOTICE 'Updated statistics for %', partition_record.tablename;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Partition health monitoring
CREATE OR REPLACE VIEW partition_health_monitoring AS
WITH partition_info AS (
    SELECT 
        schemaname,
        tablename as partition_name,
        regexp_replace(tablename, '^(.+)_\d{4}_\d{2}$', '\1') as base_table,
        regexp_replace(tablename, '.*_(\d{4}_\d{2})$', '\1') as time_period
    FROM pg_tables
    WHERE tablename ~ '_\d{4}_\d{2}$'
      AND schemaname = 'public'
),
partition_stats AS (
    SELECT 
        pi.partition_name,
        pi.base_table,
        pi.time_period,
        pg_size_pretty(pg_total_relation_size(quote_ident(pi.schemaname) || '.' || quote_ident(pi.partition_name))) as partition_size,
        pg_total_relation_size(quote_ident(pi.schemaname) || '.' || quote_ident(pi.partition_name)) as size_bytes,
        
        -- Get row count estimate from pg_class
        COALESCE(c.reltuples, 0)::BIGINT as estimated_rows,
        
        -- Last analyze time
        COALESCE(s.last_analyze, s.last_autoanalyze) as last_stats_update
        
    FROM partition_info pi
    LEFT JOIN pg_class c ON c.relname = pi.partition_name
    LEFT JOIN pg_stat_user_tables s ON s.relname = pi.partition_name
)
SELECT 
    base_table,
    partition_name,
    time_period,
    partition_size,
    estimated_rows,
    last_stats_update,
    
    -- Health indicators
    CASE 
        WHEN estimated_rows = 0 THEN 'EMPTY'
        WHEN last_stats_update < CURRENT_DATE - INTERVAL '7 days' THEN 'STALE_STATS'
        WHEN size_bytes > 10737418240 THEN 'LARGE_PARTITION'  -- > 10GB
        ELSE 'HEALTHY'
    END as health_status,
    
    -- Maintenance recommendations
    CASE 
        WHEN last_stats_update < CURRENT_DATE - INTERVAL '7 days' THEN 'ANALYZE_NEEDED'
        WHEN size_bytes > 21474836480 THEN 'CONSIDER_FURTHER_PARTITIONING'  -- > 20GB
        WHEN estimated_rows = 0 AND time_period < TO_CHAR(CURRENT_DATE - INTERVAL '1 month', 'YYYY_MM') THEN 'CANDIDATE_FOR_ARCHIVAL'
        ELSE 'NO_ACTION_REQUIRED'
    END as maintenance_recommendation
    
FROM partition_stats
ORDER BY base_table, time_period DESC;
```

### Real-World Applications

1. **Time-Series Analytics**: Historical data analysis with efficient range queries
2. **Geographic Analysis**: Regional performance comparison and market analysis
3. **Customer Segmentation**: Parallel processing of large customer datasets
4. **A/B Testing**: Temporal experiment analysis with partition pruning
5. **Data Lifecycle Management**: Efficient archiving and purging of old data

### Best Practices

1. **Partition Key Selection**: Choose keys that align with query patterns
2. **Partition Size Management**: Keep partitions reasonably sized (typically 1-10GB)
3. **Index Strategy**: Create appropriate indexes on each partition
4. **Constraint Exclusion**: Use CHECK constraints for partition pruning
5. **Maintenance Automation**: Automate partition creation and cleanup

### Common Pitfalls

1. **Over-Partitioning**: Creating too many small partitions
2. **Wrong Partition Key**: Choosing keys that don't align with queries
3. **Cross-Partition Queries**: Inefficient queries spanning many partitions
4. **Maintenance Overhead**: Neglecting partition lifecycle management
5. **Constraint Violations**: Issues with partition constraints and data integrity

### Performance Considerations

- **Query Planning**: Partition pruning effectiveness and plan complexity
- **Join Performance**: Partition-wise joins vs. cross-partition operations
- **Parallel Processing**: Leveraging parallelism across partitions
- **Memory Usage**: Buffer pool efficiency with partitioned data
- **Storage Layout**: Optimal storage organization for partitioned tables

---

## Question 16

**Describe how you could use SQL to report the performance metrics of a Machine Learning model.**

**Answer:**

### Theory

SQL can be effectively used to calculate and report comprehensive machine learning model performance metrics by leveraging statistical functions, aggregations, and analytical capabilities. This approach enables real-time monitoring, batch evaluation, historical performance tracking, and automated reporting of model effectiveness. SQL-based performance reporting provides benefits including integration with existing data infrastructure, real-time dashboards, automated alerting systems, and the ability to perform sophisticated statistical analysis without moving data out of the database.

**Key Advantages of SQL-Based Model Evaluation:**
- **Real-Time Monitoring**: Continuous performance tracking as new data arrives
- **Historical Analysis**: Trend analysis and performance degradation detection
- **Statistical Rigor**: Comprehensive metric calculations with confidence intervals
- **Integration**: Seamless integration with BI tools and dashboards
- **Efficiency**: In-database computation avoiding data movement

### Classification Model Metrics

#### 1. Binary Classification Performance

```sql
-- Comprehensive binary classification evaluation
WITH model_predictions AS (
    SELECT 
        prediction_id,
        model_id,
        model_version,
        customer_id,
        prediction_timestamp,
        predicted_probability,
        predicted_class,
        actual_class,
        prediction_date,
        
        -- Binary predictions (using 0.5 threshold)
        CASE WHEN predicted_probability >= 0.5 THEN 1 ELSE 0 END as binary_prediction,
        
        -- Feature context
        feature_vector,
        model_confidence
        
    FROM ml_predictions
    WHERE model_id = 'churn_prediction_v2'
      AND prediction_date >= CURRENT_DATE - INTERVAL '30 days'
      AND actual_class IS NOT NULL  -- Only evaluate where we have ground truth
),
confusion_matrix AS (
    SELECT 
        model_id,
        model_version,
        prediction_date,
        
        -- Confusion matrix components
        COUNT(*) FILTER (WHERE actual_class = 1 AND binary_prediction = 1) as true_positives,
        COUNT(*) FILTER (WHERE actual_class = 0 AND binary_prediction = 0) as true_negatives,
        COUNT(*) FILTER (WHERE actual_class = 0 AND binary_prediction = 1) as false_positives,
        COUNT(*) FILTER (WHERE actual_class = 1 AND binary_prediction = 0) as false_negatives,
        COUNT(*) as total_predictions,
        
        -- Class distribution
        COUNT(*) FILTER (WHERE actual_class = 1) as actual_positives,
        COUNT(*) FILTER (WHERE actual_class = 0) as actual_negatives,
        COUNT(*) FILTER (WHERE binary_prediction = 1) as predicted_positives,
        COUNT(*) FILTER (WHERE binary_prediction = 0) as predicted_negatives
        
    FROM model_predictions
    GROUP BY model_id, model_version, prediction_date
),
basic_metrics AS (
    SELECT 
        cm.*,
        
        -- Core performance metrics
        (true_positives + true_negatives)::NUMERIC / total_predictions as accuracy,
        true_positives::NUMERIC / NULLIF(actual_positives, 0) as sensitivity_recall,
        true_negatives::NUMERIC / NULLIF(actual_negatives, 0) as specificity,
        true_positives::NUMERIC / NULLIF(predicted_positives, 0) as precision,
        false_positives::NUMERIC / NULLIF(actual_negatives, 0) as false_positive_rate,
        false_negatives::NUMERIC / NULLIF(actual_positives, 0) as false_negative_rate,
        
        -- Derived metrics
        (2.0 * true_positives) / NULLIF(2.0 * true_positives + false_positives + false_negatives, 0) as f1_score,
        
        -- Matthews Correlation Coefficient
        (true_positives * true_negatives - false_positives * false_negatives)::NUMERIC / 
        NULLIF(
            SQRT((true_positives + false_positives) * (true_positives + false_negatives) * 
                 (true_negatives + false_positives) * (true_negatives + false_negatives)), 0
        ) as matthews_correlation,
        
        -- Balanced accuracy
        ((true_positives::NUMERIC / NULLIF(actual_positives, 0)) + 
         (true_negatives::NUMERIC / NULLIF(actual_negatives, 0))) / 2.0 as balanced_accuracy,
        
        -- Diagnostic odds ratio
        (true_positives::NUMERIC * true_negatives) / 
        NULLIF(false_positives::NUMERIC * false_negatives, 0) as diagnostic_odds_ratio
        
    FROM confusion_matrix cm
),
threshold_analysis AS (
    SELECT 
        mp.model_id,
        mp.model_version,
        mp.prediction_date,
        threshold_value,
        
        -- Metrics at different thresholds
        COUNT(*) FILTER (WHERE mp.actual_class = 1 AND mp.predicted_probability >= threshold_value) as tp_at_threshold,
        COUNT(*) FILTER (WHERE mp.actual_class = 0 AND mp.predicted_probability < threshold_value) as tn_at_threshold,
        COUNT(*) FILTER (WHERE mp.actual_class = 0 AND mp.predicted_probability >= threshold_value) as fp_at_threshold,
        COUNT(*) FILTER (WHERE mp.actual_class = 1 AND mp.predicted_probability < threshold_value) as fn_at_threshold,
        
        -- ROC curve points
        COUNT(*) FILTER (WHERE mp.actual_class = 1 AND mp.predicted_probability >= threshold_value)::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE mp.actual_class = 1), 0) as tpr_at_threshold,
        
        COUNT(*) FILTER (WHERE mp.actual_class = 0 AND mp.predicted_probability >= threshold_value)::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE mp.actual_class = 0), 0) as fpr_at_threshold,
        
        -- Precision-Recall curve points
        COUNT(*) FILTER (WHERE mp.actual_class = 1 AND mp.predicted_probability >= threshold_value)::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE mp.predicted_probability >= threshold_value), 0) as precision_at_threshold
        
    FROM model_predictions mp
    CROSS JOIN (
        SELECT generate_series(0.1, 0.9, 0.1) as threshold_value
    ) thresholds
    GROUP BY mp.model_id, mp.model_version, mp.prediction_date, threshold_value
),
auc_calculation AS (
    SELECT 
        model_id,
        model_version,
        prediction_date,
        
        -- AUC-ROC approximation using trapezoidal rule
        SUM(
            (tpr_at_threshold + LAG(tpr_at_threshold) OVER (ORDER BY fpr_at_threshold)) / 2.0 *
            (fpr_at_threshold - LAG(fpr_at_threshold) OVER (ORDER BY fpr_at_threshold))
        ) as auc_roc_approx,
        
        -- AUC-PR approximation
        SUM(
            (precision_at_threshold + LAG(precision_at_threshold) OVER (ORDER BY tpr_at_threshold)) / 2.0 *
            (tpr_at_threshold - LAG(tpr_at_threshold) OVER (ORDER BY tpr_at_threshold))
        ) as auc_pr_approx
        
    FROM threshold_analysis
    WHERE fpr_at_threshold IS NOT NULL AND tpr_at_threshold IS NOT NULL
    GROUP BY model_id, model_version, prediction_date
),
calibration_analysis AS (
    SELECT 
        mp.model_id,
        mp.model_version,
        mp.prediction_date,
        
        -- Calibration bins
        CASE 
            WHEN mp.predicted_probability <= 0.1 THEN '0.0-0.1'
            WHEN mp.predicted_probability <= 0.2 THEN '0.1-0.2'
            WHEN mp.predicted_probability <= 0.3 THEN '0.2-0.3'
            WHEN mp.predicted_probability <= 0.4 THEN '0.3-0.4'
            WHEN mp.predicted_probability <= 0.5 THEN '0.4-0.5'
            WHEN mp.predicted_probability <= 0.6 THEN '0.5-0.6'
            WHEN mp.predicted_probability <= 0.7 THEN '0.6-0.7'
            WHEN mp.predicted_probability <= 0.8 THEN '0.7-0.8'
            WHEN mp.predicted_probability <= 0.9 THEN '0.8-0.9'
            ELSE '0.9-1.0'
        END as probability_bin,
        
        COUNT(*) as bin_count,
        AVG(mp.predicted_probability) as avg_predicted_probability,
        AVG(mp.actual_class) as actual_positive_rate,
        
        -- Calibration error for each bin
        ABS(AVG(mp.predicted_probability) - AVG(mp.actual_class)) as bin_calibration_error
        
    FROM model_predictions mp
    GROUP BY mp.model_id, mp.model_version, mp.prediction_date,
             CASE 
                WHEN mp.predicted_probability <= 0.1 THEN '0.0-0.1'
                WHEN mp.predicted_probability <= 0.2 THEN '0.1-0.2'
                WHEN mp.predicted_probability <= 0.3 THEN '0.2-0.3'
                WHEN mp.predicted_probability <= 0.4 THEN '0.3-0.4'
                WHEN mp.predicted_probability <= 0.5 THEN '0.4-0.5'
                WHEN mp.predicted_probability <= 0.6 THEN '0.5-0.6'
                WHEN mp.predicted_probability <= 0.7 THEN '0.6-0.7'
                WHEN mp.predicted_probability <= 0.8 THEN '0.7-0.8'
                WHEN mp.predicted_probability <= 0.9 THEN '0.8-0.9'
                ELSE '0.9-1.0'
             END
),
model_performance_report AS (
    SELECT 
        bm.model_id,
        bm.model_version,
        bm.prediction_date,
        bm.total_predictions,
        
        -- Confusion matrix
        bm.true_positives,
        bm.true_negatives,
        bm.false_positives,
        bm.false_negatives,
        
        -- Core metrics (rounded for readability)
        ROUND(bm.accuracy::NUMERIC, 4) as accuracy,
        ROUND(bm.sensitivity_recall::NUMERIC, 4) as recall,
        ROUND(bm.specificity::NUMERIC, 4) as specificity,
        ROUND(bm.precision::NUMERIC, 4) as precision,
        ROUND(bm.f1_score::NUMERIC, 4) as f1_score,
        ROUND(bm.balanced_accuracy::NUMERIC, 4) as balanced_accuracy,
        ROUND(bm.matthews_correlation::NUMERIC, 4) as mcc,
        
        -- Advanced metrics
        ROUND(bm.false_positive_rate::NUMERIC, 4) as fpr,
        ROUND(bm.false_negative_rate::NUMERIC, 4) as fnr,
        ROUND(bm.diagnostic_odds_ratio::NUMERIC, 2) as diagnostic_or,
        
        -- AUC metrics
        ROUND(auc.auc_roc_approx::NUMERIC, 4) as auc_roc,
        ROUND(auc.auc_pr_approx::NUMERIC, 4) as auc_pr,
        
        -- Calibration metrics
        ROUND(AVG(ca.bin_calibration_error)::NUMERIC, 4) as avg_calibration_error,
        ROUND(MAX(ca.bin_calibration_error)::NUMERIC, 4) as max_calibration_error,
        
        -- Model quality indicators
        CASE 
            WHEN bm.accuracy >= 0.9 AND bm.f1_score >= 0.85 THEN 'EXCELLENT'
            WHEN bm.accuracy >= 0.8 AND bm.f1_score >= 0.75 THEN 'GOOD'
            WHEN bm.accuracy >= 0.7 AND bm.f1_score >= 0.6 THEN 'ACCEPTABLE'
            ELSE 'POOR'
        END as performance_rating,
        
        CASE 
            WHEN AVG(ca.bin_calibration_error) <= 0.05 THEN 'WELL_CALIBRATED'
            WHEN AVG(ca.bin_calibration_error) <= 0.1 THEN 'MODERATELY_CALIBRATED'
            ELSE 'POORLY_CALIBRATED'
        END as calibration_quality,
        
        -- Bias analysis
        CASE 
            WHEN ABS(bm.sensitivity_recall - bm.specificity) <= 0.05 THEN 'BALANCED'
            WHEN bm.sensitivity_recall > bm.specificity THEN 'POSITIVE_BIASED'
            ELSE 'NEGATIVE_BIASED'
        END as bias_assessment
        
    FROM basic_metrics bm
    LEFT JOIN auc_calculation auc ON bm.model_id = auc.model_id 
                                 AND bm.model_version = auc.model_version 
                                 AND bm.prediction_date = auc.prediction_date
    LEFT JOIN calibration_analysis ca ON bm.model_id = ca.model_id 
                                      AND bm.model_version = ca.model_version 
                                      AND bm.prediction_date = ca.prediction_date
    GROUP BY bm.model_id, bm.model_version, bm.prediction_date, bm.total_predictions,
             bm.true_positives, bm.true_negatives, bm.false_positives, bm.false_negatives,
             bm.accuracy, bm.sensitivity_recall, bm.specificity, bm.precision, bm.f1_score,
             bm.balanced_accuracy, bm.matthews_correlation, bm.false_positive_rate,
             bm.false_negative_rate, bm.diagnostic_odds_ratio, auc.auc_roc_approx, auc.auc_pr_approx
)
SELECT 
    model_id,
    model_version,
    prediction_date,
    total_predictions,
    
    -- Primary metrics
    accuracy,
    precision,
    recall,
    f1_score,
    auc_roc,
    
    -- Detailed analysis
    specificity,
    balanced_accuracy,
    mcc,
    auc_pr,
    
    -- Quality assessments
    performance_rating,
    calibration_quality,
    bias_assessment,
    
    -- Calibration metrics
    avg_calibration_error,
    max_calibration_error,
    
    -- Confusion matrix for detailed analysis
    true_positives,
    true_negatives,
    false_positives,
    false_negatives,
    
    -- Error rates
    fpr,
    fnr,
    
    -- Statistical significance (simplified)
    CASE 
        WHEN total_predictions >= 1000 THEN 'STATISTICALLY_SIGNIFICANT'
        WHEN total_predictions >= 100 THEN 'MODERATELY_SIGNIFICANT'
        ELSE 'INSUFFICIENT_SAMPLE'
    END as statistical_confidence
    
FROM model_performance_report
ORDER BY prediction_date DESC, f1_score DESC;
```

#### 2. Multi-Class Classification Metrics

```sql
-- Multi-class classification evaluation
WITH multiclass_predictions AS (
    SELECT 
        prediction_id,
        model_id,
        model_version,
        customer_id,
        prediction_timestamp,
        predicted_class,
        actual_class,
        prediction_probabilities,  -- JSON with class probabilities
        prediction_date
    FROM ml_predictions
    WHERE model_id = 'customer_segment_classifier_v3'
      AND prediction_date >= CURRENT_DATE - INTERVAL '7 days'
      AND actual_class IS NOT NULL
),
class_level_metrics AS (
    SELECT 
        model_id,
        model_version,
        prediction_date,
        class_label,
        
        -- Per-class confusion matrix
        COUNT(*) FILTER (WHERE actual_class = class_label AND predicted_class = class_label) as true_positives,
        COUNT(*) FILTER (WHERE actual_class != class_label AND predicted_class != class_label) as true_negatives,
        COUNT(*) FILTER (WHERE actual_class != class_label AND predicted_class = class_label) as false_positives,
        COUNT(*) FILTER (WHERE actual_class = class_label AND predicted_class != class_label) as false_negatives,
        
        -- Class distribution
        COUNT(*) FILTER (WHERE actual_class = class_label) as actual_class_count,
        COUNT(*) FILTER (WHERE predicted_class = class_label) as predicted_class_count,
        COUNT(*) as total_predictions
        
    FROM multiclass_predictions mp
    CROSS JOIN (
        SELECT DISTINCT unnest(ARRAY[actual_class, predicted_class]) as class_label
        FROM multiclass_predictions
    ) classes
    GROUP BY model_id, model_version, prediction_date, class_label
),
per_class_performance AS (
    SELECT 
        clm.*,
        
        -- Per-class metrics
        true_positives::NUMERIC / NULLIF(actual_class_count, 0) as class_recall,
        true_positives::NUMERIC / NULLIF(predicted_class_count, 0) as class_precision,
        (true_positives + true_negatives)::NUMERIC / total_predictions as class_accuracy,
        
        -- F1 score per class
        (2.0 * true_positives) / NULLIF(2.0 * true_positives + false_positives + false_negatives, 0) as class_f1_score,
        
        -- Support (number of actual instances)
        actual_class_count as support
        
    FROM class_level_metrics clm
),
overall_multiclass_metrics AS (
    SELECT 
        model_id,
        model_version,
        prediction_date,
        
        -- Overall accuracy
        SUM(true_positives)::NUMERIC / SUM(total_predictions) as overall_accuracy,
        
        -- Macro averages (unweighted)
        AVG(class_precision) as macro_avg_precision,
        AVG(class_recall) as macro_avg_recall,
        AVG(class_f1_score) as macro_avg_f1,
        
        -- Weighted averages (by support)
        SUM(class_precision * support) / SUM(support) as weighted_avg_precision,
        SUM(class_recall * support) / SUM(support) as weighted_avg_recall,
        SUM(class_f1_score * support) / SUM(support) as weighted_avg_f1,
        
        -- Class balance analysis
        MAX(support) - MIN(support) as class_imbalance_range,
        STDDEV(support) as class_distribution_stddev,
        COUNT(DISTINCT class_label) as num_classes,
        SUM(support) as total_samples
        
    FROM per_class_performance
    GROUP BY model_id, model_version, prediction_date
),
confusion_matrix_detailed AS (
    SELECT 
        mp.model_id,
        mp.model_version,
        mp.prediction_date,
        mp.actual_class,
        mp.predicted_class,
        COUNT(*) as prediction_count,
        
        -- Confusion matrix percentage
        COUNT(*)::NUMERIC / SUM(COUNT(*)) OVER (PARTITION BY mp.model_id, mp.model_version, mp.prediction_date) as confusion_percentage
        
    FROM multiclass_predictions mp
    GROUP BY mp.model_id, mp.model_version, mp.prediction_date, mp.actual_class, mp.predicted_class
)
SELECT 
    omm.model_id,
    omm.model_version,
    omm.prediction_date,
    omm.num_classes,
    omm.total_samples,
    
    -- Overall performance
    ROUND(omm.overall_accuracy::NUMERIC, 4) as accuracy,
    
    -- Macro metrics (treat all classes equally)
    ROUND(omm.macro_avg_precision::NUMERIC, 4) as macro_precision,
    ROUND(omm.macro_avg_recall::NUMERIC, 4) as macro_recall,
    ROUND(omm.macro_avg_f1::NUMERIC, 4) as macro_f1,
    
    -- Weighted metrics (weight by class frequency)
    ROUND(omm.weighted_avg_precision::NUMERIC, 4) as weighted_precision,
    ROUND(omm.weighted_avg_recall::NUMERIC, 4) as weighted_recall,
    ROUND(omm.weighted_avg_f1::NUMERIC, 4) as weighted_f1,
    
    -- Class balance indicators
    ROUND(omm.class_distribution_stddev::NUMERIC, 2) as class_imbalance_score,
    omm.class_imbalance_range,
    
    -- Performance assessment
    CASE 
        WHEN omm.overall_accuracy >= 0.9 AND omm.macro_avg_f1 >= 0.85 THEN 'EXCELLENT'
        WHEN omm.overall_accuracy >= 0.8 AND omm.macro_avg_f1 >= 0.75 THEN 'GOOD'
        WHEN omm.overall_accuracy >= 0.7 AND omm.macro_avg_f1 >= 0.6 THEN 'ACCEPTABLE'
        ELSE 'POOR'
    END as performance_grade,
    
    -- Class balance assessment
    CASE 
        WHEN omm.class_distribution_stddev / (omm.total_samples / omm.num_classes) <= 0.2 THEN 'WELL_BALANCED'
        WHEN omm.class_distribution_stddev / (omm.total_samples / omm.num_classes) <= 0.5 THEN 'MODERATELY_IMBALANCED'
        ELSE 'HIGHLY_IMBALANCED'
    END as class_balance_status,
    
    -- Per-class performance summary (JSON)
    jsonb_agg(
        jsonb_build_object(
            'class', pcp.class_label,
            'precision', ROUND(pcp.class_precision::NUMERIC, 4),
            'recall', ROUND(pcp.class_recall::NUMERIC, 4),
            'f1_score', ROUND(pcp.class_f1_score::NUMERIC, 4),
            'support', pcp.support
        ) ORDER BY pcp.class_label
    ) as per_class_metrics
    
FROM overall_multiclass_metrics omm
LEFT JOIN per_class_performance pcp ON omm.model_id = pcp.model_id 
                                   AND omm.model_version = pcp.model_version 
                                   AND omm.prediction_date = pcp.prediction_date
GROUP BY omm.model_id, omm.model_version, omm.prediction_date, omm.num_classes, omm.total_samples,
         omm.overall_accuracy, omm.macro_avg_precision, omm.macro_avg_recall, omm.macro_avg_f1,
         omm.weighted_avg_precision, omm.weighted_avg_recall, omm.weighted_avg_f1,
         omm.class_distribution_stddev, omm.class_imbalance_range
ORDER BY omm.prediction_date DESC, omm.weighted_avg_f1 DESC;
```

### Regression Model Metrics

#### 1. Regression Performance Analysis

```sql
-- Comprehensive regression model evaluation
WITH regression_predictions AS (
    SELECT 
        prediction_id,
        model_id,
        model_version,
        customer_id,
        prediction_timestamp,
        predicted_value,
        actual_value,
        prediction_date,
        
        -- Calculate residuals
        actual_value - predicted_value as residual,
        ABS(actual_value - predicted_value) as absolute_error,
        POWER(actual_value - predicted_value, 2) as squared_error,
        
        -- Percentage error (handle division by zero)
        CASE 
            WHEN actual_value != 0 
            THEN ABS(actual_value - predicted_value) / ABS(actual_value) * 100
            ELSE NULL 
        END as absolute_percentage_error,
        
        -- Feature context for analysis
        feature_importance_scores
        
    FROM ml_predictions
    WHERE model_id = 'customer_ltv_predictor_v4'
      AND prediction_date >= CURRENT_DATE - INTERVAL '14 days'
      AND actual_value IS NOT NULL
      AND predicted_value IS NOT NULL
),
basic_regression_metrics AS (
    SELECT 
        model_id,
        model_version,
        prediction_date,
        COUNT(*) as total_predictions,
        
        -- Central tendency of actuals
        AVG(actual_value) as mean_actual,
        STDDEV(actual_value) as stddev_actual,
        MIN(actual_value) as min_actual,
        MAX(actual_value) as max_actual,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY actual_value) as median_actual,
        
        -- Central tendency of predictions
        AVG(predicted_value) as mean_predicted,
        STDDEV(predicted_value) as stddev_predicted,
        MIN(predicted_value) as min_predicted,
        MAX(predicted_value) as max_predicted,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY predicted_value) as median_predicted,
        
        -- Error metrics
        AVG(residual) as mean_residual,
        STDDEV(residual) as stddev_residual,
        AVG(absolute_error) as mae,  -- Mean Absolute Error
        SQRT(AVG(squared_error)) as rmse,  -- Root Mean Squared Error
        AVG(squared_error) as mse,  -- Mean Squared Error
        
        -- Percentage-based metrics
        AVG(absolute_percentage_error) as mape,  -- Mean Absolute Percentage Error
        STDDEV(absolute_percentage_error) as stddev_ape,
        
        -- Quantile-based error analysis
        PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY absolute_error) as p10_absolute_error,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY absolute_error) as q1_absolute_error,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY absolute_error) as median_absolute_error,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY absolute_error) as q3_absolute_error,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY absolute_error) as p90_absolute_error,
        
        -- Correlation and R-squared
        CORR(actual_value, predicted_value) as correlation_coefficient,
        POWER(CORR(actual_value, predicted_value), 2) as r_squared,
        
        -- Total Sum of Squares and Residual Sum of Squares for adjusted R²
        SUM(POWER(actual_value - AVG(actual_value) OVER (), 2)) as total_sum_squares,
        SUM(squared_error) as residual_sum_squares
        
    FROM regression_predictions
    GROUP BY model_id, model_version, prediction_date
),
advanced_regression_metrics AS (
    SELECT 
        brm.*,
        
        -- Adjusted R-squared (assuming we know number of features)
        1 - ((residual_sum_squares / (total_predictions - 2)) / 
             (total_sum_squares / (total_predictions - 1))) as adjusted_r_squared,
        
        -- Normalized metrics
        rmse / NULLIF(stddev_actual, 0) as normalized_rmse,
        mae / NULLIF(mean_actual, 0) as relative_mae,
        
        -- Bias indicators
        ABS(mean_residual) / NULLIF(stddev_actual, 0) as bias_ratio,
        
        -- Prediction interval coverage (approximate)
        COUNT(*) FILTER (
            WHERE ABS(residual) <= 1.96 * stddev_residual
        )::NUMERIC / total_predictions as coverage_95_percent,
        
        -- Performance rating
        CASE 
            WHEN r_squared >= 0.9 AND mape <= 10 THEN 'EXCELLENT'
            WHEN r_squared >= 0.8 AND mape <= 20 THEN 'GOOD'
            WHEN r_squared >= 0.6 AND mape <= 30 THEN 'ACCEPTABLE'
            ELSE 'POOR'
        END as performance_rating,
        
        -- Prediction quality assessment
        CASE 
            WHEN ABS(mean_residual) / NULLIF(stddev_actual, 0) <= 0.05 THEN 'UNBIASED'
            WHEN mean_residual > 0 THEN 'OVERESTIMATING'
            ELSE 'UNDERESTIMATING'
        END as bias_assessment
        
    FROM basic_regression_metrics brm
    CROSS JOIN (
        SELECT COUNT(*) as coverage_count
        FROM regression_predictions rp
        JOIN basic_regression_metrics brm2 ON rp.model_id = brm2.model_id 
                                           AND rp.model_version = brm2.model_version 
                                           AND rp.prediction_date = brm2.prediction_date
        WHERE ABS(rp.residual) <= 1.96 * brm2.stddev_residual
    ) coverage_calc
),
residual_analysis AS (
    SELECT 
        rp.model_id,
        rp.model_version,
        rp.prediction_date,
        
        -- Residual distribution analysis
        CASE 
            WHEN residual < -2 * stddev_residual THEN 'LARGE_NEGATIVE'
            WHEN residual < -stddev_residual THEN 'MODERATE_NEGATIVE'
            WHEN residual < stddev_residual THEN 'NORMAL'
            WHEN residual < 2 * stddev_residual THEN 'MODERATE_POSITIVE'
            ELSE 'LARGE_POSITIVE'
        END as residual_category,
        
        COUNT(*) as residual_count,
        AVG(actual_value) as avg_actual_in_category,
        AVG(predicted_value) as avg_predicted_in_category,
        AVG(absolute_error) as avg_error_in_category
        
    FROM regression_predictions rp
    JOIN basic_regression_metrics brm ON rp.model_id = brm.model_id 
                                      AND rp.model_version = brm.model_version 
                                      AND rp.prediction_date = brm.prediction_date
    GROUP BY rp.model_id, rp.model_version, rp.prediction_date,
             CASE 
                WHEN residual < -2 * stddev_residual THEN 'LARGE_NEGATIVE'
                WHEN residual < -stddev_residual THEN 'MODERATE_NEGATIVE'
                WHEN residual < stddev_residual THEN 'NORMAL'
                WHEN residual < 2 * stddev_residual THEN 'MODERATE_POSITIVE'
                ELSE 'LARGE_POSITIVE'
             END
),
prediction_interval_analysis AS (
    SELECT 
        rp.model_id,
        rp.model_version,
        rp.prediction_date,
        
        -- Value range analysis
        CASE 
            WHEN actual_value <= PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY actual_value) OVER () THEN 'LOW_VALUE'
            WHEN actual_value <= PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY actual_value) OVER () THEN 'MEDIUM_VALUE'
            ELSE 'HIGH_VALUE'
        END as value_range,
        
        COUNT(*) as range_count,
        AVG(absolute_percentage_error) as avg_ape_in_range,
        STDDEV(absolute_percentage_error) as stddev_ape_in_range,
        AVG(ABS(residual)) as avg_absolute_error_in_range
        
    FROM regression_predictions rp
    GROUP BY rp.model_id, rp.model_version, rp.prediction_date,
             CASE 
                WHEN actual_value <= PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY actual_value) OVER () THEN 'LOW_VALUE'
                WHEN actual_value <= PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY actual_value) OVER () THEN 'MEDIUM_VALUE'
                ELSE 'HIGH_VALUE'
             END
)
SELECT 
    arm.model_id,
    arm.model_version,
    arm.prediction_date,
    arm.total_predictions,
    
    -- Core regression metrics
    ROUND(arm.r_squared::NUMERIC, 4) as r_squared,
    ROUND(arm.adjusted_r_squared::NUMERIC, 4) as adjusted_r_squared,
    ROUND(arm.mae::NUMERIC, 2) as mean_absolute_error,
    ROUND(arm.rmse::NUMERIC, 2) as root_mean_squared_error,
    ROUND(arm.mape::NUMERIC, 2) as mean_absolute_percentage_error,
    
    -- Normalized metrics
    ROUND(arm.normalized_rmse::NUMERIC, 4) as normalized_rmse,
    ROUND(arm.relative_mae::NUMERIC, 4) as relative_mae,
    
    -- Distribution comparison
    ROUND(arm.mean_actual::NUMERIC, 2) as actual_mean,
    ROUND(arm.mean_predicted::NUMERIC, 2) as predicted_mean,
    ROUND(arm.correlation_coefficient::NUMERIC, 4) as correlation,
    
    -- Error distribution
    ROUND(arm.median_absolute_error::NUMERIC, 2) as median_error,
    ROUND(arm.q1_absolute_error::NUMERIC, 2) as q1_error,
    ROUND(arm.q3_absolute_error::NUMERIC, 2) as q3_error,
    ROUND(arm.p90_absolute_error::NUMERIC, 2) as p90_error,
    
    -- Bias analysis
    ROUND(arm.mean_residual::NUMERIC, 4) as mean_bias,
    ROUND(arm.bias_ratio::NUMERIC, 4) as bias_magnitude,
    arm.bias_assessment,
    
    -- Quality indicators
    arm.performance_rating,
    ROUND(arm.coverage_95_percent::NUMERIC, 4) as prediction_interval_coverage,
    
    -- Residual distribution summary
    jsonb_agg(
        jsonb_build_object(
            'residual_type', ra.residual_category,
            'count', ra.residual_count,
            'percentage', ROUND((ra.residual_count::NUMERIC / arm.total_predictions * 100), 2)
        ) ORDER BY ra.residual_category
    ) as residual_distribution,
    
    -- Performance by value range
    jsonb_agg(DISTINCT
        jsonb_build_object(
            'value_range', pia.value_range,
            'count', pia.range_count,
            'avg_ape', ROUND(pia.avg_ape_in_range::NUMERIC, 2),
            'avg_absolute_error', ROUND(pia.avg_absolute_error_in_range::NUMERIC, 2)
        ) ORDER BY pia.value_range
    ) as performance_by_range,
    
    -- Model diagnostics
    CASE 
        WHEN arm.r_squared >= 0.8 AND arm.bias_ratio <= 0.1 THEN 'PRODUCTION_READY'
        WHEN arm.r_squared >= 0.6 AND arm.bias_ratio <= 0.2 THEN 'NEEDS_IMPROVEMENT'
        ELSE 'REQUIRES_RETRAINING'
    END as model_status
    
FROM advanced_regression_metrics arm
LEFT JOIN residual_analysis ra ON arm.model_id = ra.model_id 
                               AND arm.model_version = ra.model_version 
                               AND arm.prediction_date = ra.prediction_date
LEFT JOIN prediction_interval_analysis pia ON arm.model_id = pia.model_id 
                                           AND arm.model_version = pia.model_version 
                                           AND arm.prediction_date = pia.prediction_date
GROUP BY arm.model_id, arm.model_version, arm.prediction_date, arm.total_predictions,
         arm.r_squared, arm.adjusted_r_squared, arm.mae, arm.rmse, arm.mape,
         arm.normalized_rmse, arm.relative_mae, arm.mean_actual, arm.mean_predicted,
         arm.correlation_coefficient, arm.median_absolute_error, arm.q1_absolute_error,
         arm.q3_absolute_error, arm.p90_absolute_error, arm.mean_residual, arm.bias_ratio,
         arm.bias_assessment, arm.performance_rating, arm.coverage_95_percent
ORDER BY arm.prediction_date DESC, arm.r_squared DESC;
```

### Model Monitoring and Drift Detection

#### 1. Performance Monitoring Over Time

```sql
-- Model performance monitoring and drift detection
WITH performance_trends AS (
    SELECT 
        model_id,
        model_version,
        prediction_date,
        
        -- Daily performance metrics
        COUNT(*) as daily_predictions,
        AVG(CASE WHEN predicted_class = actual_class THEN 1.0 ELSE 0.0 END) as daily_accuracy,
        
        -- Classification metrics (assuming binary classification)
        COUNT(*) FILTER (WHERE actual_class = 1 AND predicted_class = 1)::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE actual_class = 1), 0) as daily_recall,
        
        COUNT(*) FILTER (WHERE actual_class = 1 AND predicted_class = 1)::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE predicted_class = 1), 0) as daily_precision,
        
        -- Feature drift indicators
        AVG(model_confidence) as avg_confidence,
        STDDEV(model_confidence) as confidence_stability,
        
        -- Distribution shift indicators
        STDDEV(predicted_probability) as prediction_distribution_spread,
        AVG(predicted_probability) as avg_predicted_probability,
        
        -- Data quality indicators
        COUNT(*) FILTER (WHERE predicted_probability IS NULL OR actual_class IS NULL) as missing_data_count
        
    FROM ml_predictions
    WHERE model_id IN ('churn_prediction_v2', 'customer_segment_classifier_v3')
      AND prediction_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY model_id, model_version, prediction_date
),
rolling_performance AS (
    SELECT 
        pt.*,
        
        -- 7-day rolling averages
        AVG(daily_accuracy) OVER (
            PARTITION BY model_id, model_version 
            ORDER BY prediction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as rolling_7d_accuracy,
        
        AVG(daily_recall) OVER (
            PARTITION BY model_id, model_version 
            ORDER BY prediction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as rolling_7d_recall,
        
        AVG(daily_precision) OVER (
            PARTITION BY model_id, model_version 
            ORDER BY prediction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as rolling_7d_precision,
        
        AVG(avg_confidence) OVER (
            PARTITION BY model_id, model_version 
            ORDER BY prediction_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as rolling_7d_confidence,
        
        -- 30-day baselines
        AVG(daily_accuracy) OVER (
            PARTITION BY model_id, model_version 
            ORDER BY prediction_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_30d_accuracy,
        
        -- Trend detection
        daily_accuracy - LAG(daily_accuracy, 7) OVER (
            PARTITION BY model_id, model_version 
            ORDER BY prediction_date
        ) as week_over_week_accuracy_change,
        
        -- Volatility measures
        STDDEV(daily_accuracy) OVER (
            PARTITION BY model_id, model_version 
            ORDER BY prediction_date 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) as accuracy_volatility_14d
        
    FROM performance_trends pt
),
drift_detection AS (
    SELECT 
        rp.*,
        
        -- Performance degradation flags
        CASE 
            WHEN rolling_7d_accuracy < rolling_30d_accuracy - 0.05 THEN 'ACCURACY_DEGRADATION'
            WHEN rolling_7d_accuracy > rolling_30d_accuracy + 0.05 THEN 'ACCURACY_IMPROVEMENT'
            ELSE 'STABLE_ACCURACY'
        END as accuracy_trend,
        
        -- Confidence drift
        CASE 
            WHEN confidence_stability > 0.2 THEN 'HIGH_CONFIDENCE_VARIANCE'
            WHEN avg_confidence < 0.6 THEN 'LOW_CONFIDENCE'
            ELSE 'STABLE_CONFIDENCE'
        END as confidence_assessment,
        
        -- Data quality alerts
        CASE 
            WHEN missing_data_count > daily_predictions * 0.05 THEN 'DATA_QUALITY_ISSUE'
            WHEN daily_predictions < 100 THEN 'LOW_VOLUME'
            ELSE 'NORMAL_DATA_QUALITY'
        END as data_quality_status,
        
        -- Statistical significance of changes
        CASE 
            WHEN ABS(week_over_week_accuracy_change) > 2 * accuracy_volatility_14d 
                 AND daily_predictions >= 100 
            THEN 'STATISTICALLY_SIGNIFICANT_CHANGE'
            ELSE 'NORMAL_VARIATION'
        END as change_significance,
        
        -- Overall model health
        CASE 
            WHEN rolling_7d_accuracy >= 0.8 AND avg_confidence >= 0.7 AND missing_data_count = 0 
            THEN 'HEALTHY'
            WHEN rolling_7d_accuracy >= 0.7 AND avg_confidence >= 0.6 
            THEN 'ACCEPTABLE'
            WHEN rolling_7d_accuracy >= 0.6 
            THEN 'NEEDS_ATTENTION'
            ELSE 'CRITICAL'
        END as model_health_status
        
    FROM rolling_performance rp
),
alert_summary AS (
    SELECT 
        model_id,
        model_version,
        prediction_date,
        
        -- Current performance
        ROUND(daily_accuracy::NUMERIC, 4) as current_accuracy,
        ROUND(rolling_7d_accuracy::NUMERIC, 4) as week_avg_accuracy,
        ROUND(rolling_30d_accuracy::NUMERIC, 4) as month_avg_accuracy,
        
        -- Trend indicators
        accuracy_trend,
        confidence_assessment,
        data_quality_status,
        change_significance,
        model_health_status,
        
        -- Detailed metrics
        daily_predictions,
        ROUND(avg_confidence::NUMERIC, 4) as model_confidence,
        ROUND(confidence_stability::NUMERIC, 4) as confidence_variability,
        missing_data_count,
        
        -- Change indicators
        ROUND(week_over_week_accuracy_change::NUMERIC, 4) as weekly_change,
        ROUND(accuracy_volatility_14d::NUMERIC, 4) as recent_volatility,
        
        -- Alert flags
        CASE 
            WHEN model_health_status IN ('NEEDS_ATTENTION', 'CRITICAL') THEN 1 
            ELSE 0 
        END as performance_alert,
        
        CASE 
            WHEN change_significance = 'STATISTICALLY_SIGNIFICANT_CHANGE' 
                 AND week_over_week_accuracy_change < -0.02 
            THEN 1 ELSE 0 
        END as degradation_alert,
        
        CASE 
            WHEN data_quality_status = 'DATA_QUALITY_ISSUE' THEN 1 
            ELSE 0 
        END as data_quality_alert
        
    FROM drift_detection
)
SELECT 
    model_id,
    model_version,
    prediction_date,
    
    -- Performance summary
    current_accuracy,
    week_avg_accuracy,
    month_avg_accuracy,
    model_confidence,
    
    -- Health assessment
    model_health_status,
    accuracy_trend,
    confidence_assessment,
    
    -- Volume and quality
    daily_predictions,
    data_quality_status,
    missing_data_count,
    
    -- Change analysis
    weekly_change,
    recent_volatility,
    change_significance,
    
    -- Alert indicators
    performance_alert,
    degradation_alert,
    data_quality_alert,
    
    -- Overall alert level
    CASE 
        WHEN performance_alert + degradation_alert + data_quality_alert >= 2 THEN 'HIGH'
        WHEN performance_alert + degradation_alert + data_quality_alert = 1 THEN 'MEDIUM'
        ELSE 'LOW'
    END as alert_level,
    
    -- Recommendations
    CASE 
        WHEN model_health_status = 'CRITICAL' THEN 'IMMEDIATE_RETRAINING_REQUIRED'
        WHEN degradation_alert = 1 THEN 'INVESTIGATE_PERFORMANCE_DECLINE'
        WHEN data_quality_alert = 1 THEN 'CHECK_DATA_PIPELINE'
        WHEN confidence_assessment = 'LOW_CONFIDENCE' THEN 'REVIEW_MODEL_CALIBRATION'
        ELSE 'MONITOR_NORMALLY'
    END as recommended_action
    
FROM alert_summary
WHERE prediction_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY alert_level DESC, prediction_date DESC, model_id;
```

### Real-World Applications

1. **Production Monitoring**: Real-time model performance tracking
2. **A/B Testing**: Comparing model versions and configurations
3. **Regulatory Reporting**: Compliance and audit trail documentation
4. **Business Reporting**: Performance dashboards and executive summaries
5. **Model Validation**: Automated testing and quality assurance

### Best Practices

1. **Automated Reporting**: Schedule regular performance evaluations
2. **Threshold Setting**: Define clear performance boundaries and alerts
3. **Historical Tracking**: Maintain comprehensive performance history
4. **Multi-Metric Evaluation**: Use diverse metrics for comprehensive assessment
5. **Business Context**: Include business-relevant metrics alongside technical ones

### Common Pitfalls

1. **Single Metric Focus**: Relying on accuracy alone for model evaluation
2. **Sample Size Issues**: Drawing conclusions from insufficient data
3. **Data Leakage**: Including future information in performance calculations
4. **Survivorship Bias**: Only evaluating successful predictions
5. **Metric Gaming**: Optimizing for metrics rather than business outcomes

### Performance Considerations

- **Query Optimization**: Efficient aggregation and window function usage
- **Data Partitioning**: Partition by date for efficient time-based queries
- **Index Strategy**: Create indexes on model_id, prediction_date, and performance columns
- **Materialized Views**: Cache expensive metric calculations
- **Real-Time vs Batch**: Balance between real-time monitoring and computational efficiency

---

## Question 17

**Describe how you would version control the datasets used for building Machine Learning models in SQL.**

**Answer:**

### Theory

Dataset version control for machine learning involves systematically tracking, managing, and maintaining different versions of training data, validation sets, and test datasets throughout the ML lifecycle. This practice ensures reproducibility, enables rollback capabilities, facilitates collaboration, and maintains data lineage for compliance and debugging purposes. SQL-based version control systems provide robust capabilities for managing large datasets with efficient storage, querying, and versioning mechanisms.

**Key Components of Dataset Version Control:**
- **Data Lineage**: Track data sources, transformations, and dependencies
- **Version History**: Maintain historical snapshots of datasets
- **Metadata Management**: Store comprehensive information about each version
- **Change Tracking**: Monitor modifications, additions, and deletions
- **Access Control**: Secure versioned datasets with appropriate permissions
- **Storage Efficiency**: Optimize storage through deduplication and compression

### SQL-Based Version Control Implementation

#### 1. Core Versioning Schema

```sql
-- Core versioning infrastructure
CREATE SCHEMA ml_data_version_control;

-- Dataset metadata and version tracking
CREATE TABLE ml_data_version_control.datasets (
    dataset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    owner_user_id UUID NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Dataset classification
    dataset_type VARCHAR(50) NOT NULL CHECK (dataset_type IN ('TRAINING', 'VALIDATION', 'TEST', 'PRODUCTION')),
    domain VARCHAR(100) NOT NULL,  -- e.g., 'customer_analytics', 'fraud_detection'
    
    -- Schema information
    schema_definition JSONB NOT NULL,  -- Column definitions, types, constraints
    primary_key_columns TEXT[] NOT NULL,
    
    -- Status and lifecycle
    status VARCHAR(20) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'DEPRECATED', 'ARCHIVED')),
    retention_policy INTERVAL,  -- How long to keep versions
    
    -- Indexes for efficient queries
    INDEX idx_datasets_name_type (dataset_name, dataset_type),
    INDEX idx_datasets_owner (owner_user_id),
    INDEX idx_datasets_domain (domain)
);

-- Version tracking for each dataset
CREATE TABLE ml_data_version_control.dataset_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL REFERENCES ml_data_version_control.datasets(dataset_id),
    version_number SERIAL NOT NULL,
    version_name VARCHAR(255),  -- Human-readable version name
    
    -- Version metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID NOT NULL,
    commit_message TEXT,
    
    -- Data characteristics
    row_count BIGINT NOT NULL,
    column_count INTEGER NOT NULL,
    data_size_bytes BIGINT,
    checksum_md5 VARCHAR(32),  -- Data integrity validation
    
    -- Schema evolution
    schema_version INTEGER NOT NULL DEFAULT 1,
    schema_changes JSONB,  -- Track schema modifications
    
    -- Statistics and quality metrics
    data_quality_score NUMERIC(5,4),  -- 0.0 to 1.0
    completeness_percentage NUMERIC(5,2),
    uniqueness_percentage NUMERIC(5,2),
    validity_percentage NUMERIC(5,2),
    
    -- Lineage and provenance
    parent_version_id UUID REFERENCES ml_data_version_control.dataset_versions(version_id),
    source_query TEXT,  -- Query used to generate this version
    transformation_type VARCHAR(50),  -- 'FULL_LOAD', 'INCREMENTAL', 'FILTERED', 'TRANSFORMED'
    
    -- ML-specific metadata
    ml_metadata JSONB,  -- Feature engineering info, target distributions, etc.
    
    -- Performance tracking
    build_duration_seconds INTEGER,
    validation_status VARCHAR(20) DEFAULT 'PENDING' CHECK (validation_status IN ('PENDING', 'PASSED', 'FAILED', 'SKIPPED')),
    
    -- Unique constraint ensuring one version number per dataset
    UNIQUE(dataset_id, version_number),
    
    -- Indexes for efficient access
    INDEX idx_versions_dataset_version (dataset_id, version_number DESC),
    INDEX idx_versions_created (created_at DESC),
    INDEX idx_versions_parent (parent_version_id)
);

-- Change tracking for granular modifications
CREATE TABLE ml_data_version_control.version_changes (
    change_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES ml_data_version_control.dataset_versions(version_id),
    change_type VARCHAR(20) NOT NULL CHECK (change_type IN ('INSERT', 'UPDATE', 'DELETE', 'SCHEMA_CHANGE')),
    
    -- Record-level changes
    record_identifier TEXT,  -- Primary key or unique identifier
    column_name VARCHAR(255),
    old_value TEXT,
    new_value TEXT,
    
    -- Change metadata
    change_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    changed_by UUID,
    change_reason TEXT,
    
    -- Batch processing
    batch_id UUID,  -- Group related changes
    
    INDEX idx_changes_version_type (version_id, change_type),
    INDEX idx_changes_timestamp (change_timestamp),
    INDEX idx_changes_batch (batch_id)
);

-- Dataset relationships and dependencies
CREATE TABLE ml_data_version_control.dataset_dependencies (
    dependency_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dependent_dataset_id UUID NOT NULL REFERENCES ml_data_version_control.datasets(dataset_id),
    source_dataset_id UUID NOT NULL REFERENCES ml_data_version_control.datasets(dataset_id),
    
    -- Dependency metadata
    dependency_type VARCHAR(50) NOT NULL,  -- 'SOURCE', 'DERIVED', 'JOINED', 'AGGREGATED'
    relationship_description TEXT,
    
    -- Version constraints
    source_version_constraint VARCHAR(50),  -- 'LATEST', 'FIXED', 'RANGE'
    min_source_version INTEGER,
    max_source_version INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Prevent circular dependencies
    CHECK (dependent_dataset_id != source_dataset_id),
    
    INDEX idx_dependencies_dependent (dependent_dataset_id),
    INDEX idx_dependencies_source (source_dataset_id)
);

-- Model-dataset associations for ML lifecycle tracking
CREATE TABLE ml_data_version_control.model_dataset_usage (
    usage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    dataset_id UUID NOT NULL REFERENCES ml_data_version_control.datasets(dataset_id),
    version_id UUID NOT NULL REFERENCES ml_data_version_control.dataset_versions(version_id),
    
    -- Usage context
    usage_type VARCHAR(20) NOT NULL CHECK (usage_type IN ('TRAINING', 'VALIDATION', 'TESTING', 'INFERENCE')),
    usage_start_time TIMESTAMP NOT NULL,
    usage_end_time TIMESTAMP,
    
    -- Performance tracking
    model_performance_metrics JSONB,  -- Accuracy, F1, etc. achieved with this dataset
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    
    INDEX idx_model_usage_model (model_id, model_version),
    INDEX idx_model_usage_dataset (dataset_id, version_id),
    INDEX idx_model_usage_type_time (usage_type, usage_start_time)
);

-- Data validation rules and quality checks
CREATE TABLE ml_data_version_control.validation_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL REFERENCES ml_data_version_control.datasets(dataset_id),
    rule_name VARCHAR(255) NOT NULL,
    
    -- Rule definition
    rule_type VARCHAR(50) NOT NULL,  -- 'COLUMN_NOT_NULL', 'RANGE_CHECK', 'REGEX_MATCH', 'CUSTOM_SQL'
    rule_definition JSONB NOT NULL,  -- Flexible rule configuration
    
    -- Rule execution
    is_active BOOLEAN DEFAULT TRUE,
    severity VARCHAR(20) DEFAULT 'ERROR' CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID NOT NULL,
    
    INDEX idx_validation_rules_dataset (dataset_id),
    INDEX idx_validation_rules_active (dataset_id, is_active)
);

-- Validation results for each version
CREATE TABLE ml_data_version_control.validation_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES ml_data_version_control.dataset_versions(version_id),
    rule_id UUID NOT NULL REFERENCES ml_data_version_control.validation_rules(rule_id),
    
    -- Validation outcome
    validation_status VARCHAR(20) NOT NULL CHECK (validation_status IN ('PASSED', 'FAILED', 'WARNING', 'SKIPPED')),
    error_count INTEGER DEFAULT 0,
    warning_count INTEGER DEFAULT 0,
    
    -- Details
    validation_message TEXT,
    validation_details JSONB,  -- Specific failures, sample records, etc.
    
    -- Execution metadata
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validation_duration_ms INTEGER,
    
    INDEX idx_validation_results_version (version_id),
    INDEX idx_validation_results_status (validation_status)
);
```

#### 2. Dataset Creation and Versioning Functions

```sql
-- Comprehensive dataset versioning functions
CREATE OR REPLACE FUNCTION ml_data_version_control.create_dataset_version(
    p_dataset_name VARCHAR(255),
    p_version_name VARCHAR(255) DEFAULT NULL,
    p_commit_message TEXT DEFAULT NULL,
    p_source_query TEXT,
    p_created_by UUID,
    p_transformation_type VARCHAR(50) DEFAULT 'FULL_LOAD',
    p_parent_version_id UUID DEFAULT NULL,
    p_validation_enabled BOOLEAN DEFAULT TRUE
) RETURNS UUID AS $$
DECLARE
    v_dataset_id UUID;
    v_version_id UUID;
    v_version_number INTEGER;
    v_temp_table_name TEXT;
    v_row_count BIGINT;
    v_column_count INTEGER;
    v_data_size_bytes BIGINT;
    v_checksum_md5 VARCHAR(32);
    v_schema_definition JSONB;
    v_validation_result BOOLEAN;
BEGIN
    -- Get or validate dataset
    SELECT dataset_id INTO v_dataset_id
    FROM ml_data_version_control.datasets
    WHERE dataset_name = p_dataset_name;
    
    IF v_dataset_id IS NULL THEN
        RAISE EXCEPTION 'Dataset % not found', p_dataset_name;
    END IF;
    
    -- Generate next version number
    SELECT COALESCE(MAX(version_number), 0) + 1 
    INTO v_version_number
    FROM ml_data_version_control.dataset_versions
    WHERE dataset_id = v_dataset_id;
    
    -- Create temporary table with versioned data
    v_temp_table_name := format('temp_dataset_v%s_%s', v_version_number, extract(epoch from now())::bigint);
    
    -- Execute source query to create dataset version
    EXECUTE format('CREATE TEMP TABLE %I AS %s', v_temp_table_name, p_source_query);
    
    -- Gather dataset statistics
    EXECUTE format('SELECT COUNT(*) FROM %I', v_temp_table_name) INTO v_row_count;
    
    -- Get column count and schema information
    SELECT jsonb_agg(
        jsonb_build_object(
            'column_name', column_name,
            'data_type', data_type,
            'is_nullable', is_nullable,
            'ordinal_position', ordinal_position
        ) ORDER BY ordinal_position
    ), COUNT(*)
    INTO v_schema_definition, v_column_count
    FROM information_schema.columns
    WHERE table_name = v_temp_table_name
      AND table_schema = 'pg_temp';
    
    -- Calculate data size (approximate)
    EXECUTE format('SELECT pg_total_relation_size(''%I'')', v_temp_table_name) INTO v_data_size_bytes;
    
    -- Generate checksum for data integrity
    EXECUTE format(
        'SELECT md5(string_agg(md5(t.*::text), '''' ORDER BY md5(t.*::text))) FROM %I t',
        v_temp_table_name
    ) INTO v_checksum_md5;
    
    -- Create version record
    INSERT INTO ml_data_version_control.dataset_versions (
        dataset_id,
        version_number,
        version_name,
        created_by,
        commit_message,
        row_count,
        column_count,
        data_size_bytes,
        checksum_md5,
        schema_version,
        parent_version_id,
        source_query,
        transformation_type
    ) VALUES (
        v_dataset_id,
        v_version_number,
        COALESCE(p_version_name, format('v%s_%s', v_version_number, to_char(now(), 'YYYY-MM-DD'))),
        p_created_by,
        p_commit_message,
        v_row_count,
        v_column_count,
        v_data_size_bytes,
        v_checksum_md5,
        1,  -- Default schema version
        p_parent_version_id,
        p_source_query,
        p_transformation_type
    ) RETURNING version_id INTO v_version_id;
    
    -- Create actual versioned table
    EXECUTE format(
        'CREATE TABLE ml_data_version_control.dataset_%s_v%s AS SELECT * FROM %I',
        replace(p_dataset_name, '-', '_'),
        v_version_number,
        v_temp_table_name
    );
    
    -- Add version metadata columns
    EXECUTE format(
        'ALTER TABLE ml_data_version_control.dataset_%s_v%s 
         ADD COLUMN _version_id UUID DEFAULT ''%s'',
         ADD COLUMN _created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
        replace(p_dataset_name, '-', '_'),
        v_version_number,
        v_version_id
    );
    
    -- Run validation if enabled
    IF p_validation_enabled THEN
        SELECT ml_data_version_control.validate_dataset_version(v_version_id) INTO v_validation_result;
        
        UPDATE ml_data_version_control.dataset_versions
        SET validation_status = CASE WHEN v_validation_result THEN 'PASSED' ELSE 'FAILED' END
        WHERE version_id = v_version_id;
    END IF;
    
    -- Clean up temporary table
    EXECUTE format('DROP TABLE %I', v_temp_table_name);
    
    -- Log version creation
    INSERT INTO ml_data_version_control.version_changes (
        version_id,
        change_type,
        change_reason,
        changed_by
    ) VALUES (
        v_version_id,
        'INSERT',
        format('Created dataset version %s with %s rows', v_version_number, v_row_count),
        p_created_by
    );
    
    RETURN v_version_id;
    
EXCEPTION
    WHEN OTHERS THEN
        -- Cleanup on error
        IF v_temp_table_name IS NOT NULL THEN
            EXECUTE format('DROP TABLE IF EXISTS %I', v_temp_table_name);
        END IF;
        RAISE;
END;
$$ LANGUAGE plpgsql;

-- Function to validate dataset version
CREATE OR REPLACE FUNCTION ml_data_version_control.validate_dataset_version(
    p_version_id UUID
) RETURNS BOOLEAN AS $$
DECLARE
    v_dataset_id UUID;
    v_version_number INTEGER;
    v_dataset_name VARCHAR(255);
    v_table_name TEXT;
    v_rule RECORD;
    v_validation_passed BOOLEAN := TRUE;
    v_error_count INTEGER;
    v_rule_result RECORD;
BEGIN
    -- Get version information
    SELECT dv.dataset_id, dv.version_number, d.dataset_name
    INTO v_dataset_id, v_version_number, v_dataset_name
    FROM ml_data_version_control.dataset_versions dv
    JOIN ml_data_version_control.datasets d ON dv.dataset_id = d.dataset_id
    WHERE dv.version_id = p_version_id;
    
    v_table_name := format('ml_data_version_control.dataset_%s_v%s', 
                          replace(v_dataset_name, '-', '_'), v_version_number);
    
    -- Run all active validation rules for this dataset
    FOR v_rule IN 
        SELECT rule_id, rule_name, rule_type, rule_definition, severity
        FROM ml_data_version_control.validation_rules
        WHERE dataset_id = v_dataset_id AND is_active = TRUE
    LOOP
        v_error_count := 0;
        
        -- Execute validation based on rule type
        CASE v_rule.rule_type
            WHEN 'COLUMN_NOT_NULL' THEN
                EXECUTE format(
                    'SELECT COUNT(*) FROM %s WHERE %I IS NULL',
                    v_table_name,
                    v_rule.rule_definition->>'column_name'
                ) INTO v_error_count;
                
            WHEN 'RANGE_CHECK' THEN
                EXECUTE format(
                    'SELECT COUNT(*) FROM %s WHERE %I NOT BETWEEN %s AND %s',
                    v_table_name,
                    v_rule.rule_definition->>'column_name',
                    v_rule.rule_definition->>'min_value',
                    v_rule.rule_definition->>'max_value'
                ) INTO v_error_count;
                
            WHEN 'CUSTOM_SQL' THEN
                EXECUTE replace(v_rule.rule_definition->>'sql_query', '{table_name}', v_table_name)
                INTO v_error_count;
                
            -- Add more rule types as needed
        END CASE;
        
        -- Record validation result
        INSERT INTO ml_data_version_control.validation_results (
            version_id,
            rule_id,
            validation_status,
            error_count,
            validation_message
        ) VALUES (
            p_version_id,
            v_rule.rule_id,
            CASE WHEN v_error_count = 0 THEN 'PASSED' ELSE 'FAILED' END,
            v_error_count,
            CASE WHEN v_error_count = 0 
                THEN format('Rule %s passed', v_rule.rule_name)
                ELSE format('Rule %s failed with %s errors', v_rule.rule_name, v_error_count)
            END
        );
        
        -- Update overall validation status
        IF v_error_count > 0 AND v_rule.severity IN ('ERROR', 'CRITICAL') THEN
            v_validation_passed := FALSE;
        END IF;
    END LOOP;
    
    RETURN v_validation_passed;
END;
$$ LANGUAGE plpgsql;
```

#### 3. Dataset Comparison and Diff Analysis

```sql
-- Function to compare two dataset versions
CREATE OR REPLACE FUNCTION ml_data_version_control.compare_dataset_versions(
    p_dataset_name VARCHAR(255),
    p_version_1 INTEGER,
    p_version_2 INTEGER
) RETURNS TABLE (
    comparison_type VARCHAR(50),
    comparison_detail TEXT,
    count_difference BIGINT,
    percentage_change NUMERIC
) AS $$
DECLARE
    v_table_1 TEXT;
    v_table_2 TEXT;
    v_pk_columns TEXT[];
    v_comparison_result RECORD;
BEGIN
    -- Build table names
    v_table_1 := format('ml_data_version_control.dataset_%s_v%s', 
                       replace(p_dataset_name, '-', '_'), p_version_1);
    v_table_2 := format('ml_data_version_control.dataset_%s_v%s', 
                       replace(p_dataset_name, '-', '_'), p_version_2);
    
    -- Get primary key columns
    SELECT primary_key_columns INTO v_pk_columns
    FROM ml_data_version_control.datasets
    WHERE dataset_name = p_dataset_name;
    
    -- Row count comparison
    EXECUTE format('
        WITH counts AS (
            SELECT COUNT(*) as count_v1 FROM %s
            UNION ALL
            SELECT COUNT(*) as count_v2 FROM %s
        )
        SELECT 
            ''ROW_COUNT'' as comparison_type,
            format(''Version %s: %%s rows, Version %s: %%s rows'', 
                   (SELECT count_v1 FROM counts LIMIT 1), 
                   (SELECT count_v2 FROM counts OFFSET 1 LIMIT 1)) as comparison_detail,
            (SELECT count_v2 FROM counts OFFSET 1 LIMIT 1) - (SELECT count_v1 FROM counts LIMIT 1) as count_difference,
            CASE WHEN (SELECT count_v1 FROM counts LIMIT 1) > 0 
                THEN ((SELECT count_v2 FROM counts OFFSET 1 LIMIT 1) - (SELECT count_v1 FROM counts LIMIT 1))::NUMERIC / (SELECT count_v1 FROM counts LIMIT 1) * 100
                ELSE 0 
            END as percentage_change
    ', v_table_1, v_table_2, p_version_1, p_version_2) INTO v_comparison_result;
    
    RETURN QUERY SELECT v_comparison_result.comparison_type, v_comparison_result.comparison_detail, 
                        v_comparison_result.count_difference, v_comparison_result.percentage_change;
    
    -- Added records (in version 2 but not in version 1)
    EXECUTE format('
        WITH added_records AS (
            SELECT COUNT(*) as added_count
            FROM %s t2
            WHERE NOT EXISTS (
                SELECT 1 FROM %s t1 
                WHERE %s
            )
        )
        SELECT 
            ''ADDED_RECORDS'' as comparison_type,
            format(''%%s records added in version %s'', added_count) as comparison_detail,
            added_count as count_difference,
            0::NUMERIC as percentage_change
        FROM added_records
    ', v_table_2, v_table_1, 
       array_to_string(
           array(SELECT format('t1.%I = t2.%I', col, col) FROM unnest(v_pk_columns) col),
           ' AND '
       ), p_version_2) INTO v_comparison_result;
    
    RETURN QUERY SELECT v_comparison_result.comparison_type, v_comparison_result.comparison_detail, 
                        v_comparison_result.count_difference, v_comparison_result.percentage_change;
    
    -- Deleted records (in version 1 but not in version 2)
    EXECUTE format('
        WITH deleted_records AS (
            SELECT COUNT(*) as deleted_count
            FROM %s t1
            WHERE NOT EXISTS (
                SELECT 1 FROM %s t2 
                WHERE %s
            )
        )
        SELECT 
            ''DELETED_RECORDS'' as comparison_type,
            format(''%%s records deleted from version %s'', deleted_count) as comparison_detail,
            -deleted_count as count_difference,
            0::NUMERIC as percentage_change
        FROM deleted_records
    ', v_table_1, v_table_2,
       array_to_string(
           array(SELECT format('t1.%I = t2.%I', col, col) FROM unnest(v_pk_columns) col),
           ' AND '
       ), p_version_1) INTO v_comparison_result;
    
    RETURN QUERY SELECT v_comparison_result.comparison_type, v_comparison_result.comparison_detail, 
                        v_comparison_result.count_difference, v_comparison_result.percentage_change;
    
END;
$$ LANGUAGE plpgsql;

-- Comprehensive dataset lineage and usage report
CREATE OR REPLACE VIEW ml_data_version_control.dataset_lineage_report AS
WITH RECURSIVE dataset_hierarchy AS (
    -- Base case: datasets with no dependencies
    SELECT 
        d.dataset_id,
        d.dataset_name,
        d.dataset_type,
        0 as hierarchy_level,
        ARRAY[d.dataset_id] as lineage_path,
        d.dataset_name as root_dataset
    FROM ml_data_version_control.datasets d
    WHERE NOT EXISTS (
        SELECT 1 FROM ml_data_version_control.dataset_dependencies dep
        WHERE dep.dependent_dataset_id = d.dataset_id
    )
    
    UNION ALL
    
    -- Recursive case: datasets that depend on others
    SELECT 
        d.dataset_id,
        d.dataset_name,
        d.dataset_type,
        dh.hierarchy_level + 1,
        dh.lineage_path || d.dataset_id,
        dh.root_dataset
    FROM ml_data_version_control.datasets d
    JOIN ml_data_version_control.dataset_dependencies dep ON d.dataset_id = dep.dependent_dataset_id
    JOIN dataset_hierarchy dh ON dep.source_dataset_id = dh.dataset_id
    WHERE NOT (d.dataset_id = ANY(dh.lineage_path))  -- Prevent cycles
),
version_stats AS (
    SELECT 
        dv.dataset_id,
        COUNT(*) as total_versions,
        MAX(dv.version_number) as latest_version,
        MIN(dv.created_at) as first_version_date,
        MAX(dv.created_at) as latest_version_date,
        SUM(dv.row_count) as total_rows_across_versions,
        AVG(dv.data_quality_score) as avg_quality_score
    FROM ml_data_version_control.dataset_versions dv
    GROUP BY dv.dataset_id
),
model_usage_stats AS (
    SELECT 
        mdu.dataset_id,
        COUNT(DISTINCT mdu.model_id) as models_using_dataset,
        COUNT(*) as total_model_usages,
        array_agg(DISTINCT mdu.usage_type) as usage_types
    FROM ml_data_version_control.model_dataset_usage mdu
    GROUP BY mdu.dataset_id
)
SELECT 
    dh.dataset_name,
    dh.dataset_type,
    dh.hierarchy_level,
    dh.root_dataset,
    vs.total_versions,
    vs.latest_version,
    vs.first_version_date,
    vs.latest_version_date,
    vs.avg_quality_score,
    COALESCE(mus.models_using_dataset, 0) as models_using_dataset,
    COALESCE(mus.total_model_usages, 0) as total_model_usages,
    COALESCE(mus.usage_types, ARRAY[]::VARCHAR[]) as usage_types,
    
    -- Dependency information
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'source_dataset', sd.dataset_name,
                'dependency_type', dep.dependency_type,
                'relationship', dep.relationship_description
            )
        )
        FROM ml_data_version_control.dataset_dependencies dep
        JOIN ml_data_version_control.datasets sd ON dep.source_dataset_id = sd.dataset_id
        WHERE dep.dependent_dataset_id = dh.dataset_id
    ) as dependencies,
    
    -- Dependent datasets
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'dependent_dataset', dd.dataset_name,
                'dependency_type', dep2.dependency_type
            )
        )
        FROM ml_data_version_control.dataset_dependencies dep2
        JOIN ml_data_version_control.datasets dd ON dep2.dependent_dataset_id = dd.dataset_id
        WHERE dep2.source_dataset_id = dh.dataset_id
    ) as dependents
    
FROM dataset_hierarchy dh
LEFT JOIN version_stats vs ON dh.dataset_id = vs.dataset_id
LEFT JOIN model_usage_stats mus ON dh.dataset_id = mus.dataset_id
ORDER BY dh.root_dataset, dh.hierarchy_level, dh.dataset_name;
```

### Advanced Dataset Management

#### 1. Automated Backup and Archive System

```sql
-- Automated dataset archival and cleanup
CREATE OR REPLACE FUNCTION ml_data_version_control.archive_old_versions(
    p_dataset_name VARCHAR(255) DEFAULT NULL,
    p_retention_days INTEGER DEFAULT 365,
    p_keep_minimum_versions INTEGER DEFAULT 5,
    p_dry_run BOOLEAN DEFAULT TRUE
) RETURNS TABLE (
    action VARCHAR(20),
    dataset_name VARCHAR(255),
    version_number INTEGER,
    version_id UUID,
    size_bytes BIGINT,
    reason TEXT
) AS $$
DECLARE
    v_dataset RECORD;
    v_version RECORD;
    v_table_name TEXT;
    v_archive_table_name TEXT;
BEGIN
    FOR v_dataset IN 
        SELECT d.dataset_id, d.dataset_name, d.retention_policy
        FROM ml_data_version_control.datasets d
        WHERE (p_dataset_name IS NULL OR d.dataset_name = p_dataset_name)
          AND d.status = 'ACTIVE'
    LOOP
        -- Find versions eligible for archival
        FOR v_version IN
            SELECT dv.version_id, dv.version_number, dv.created_at, dv.data_size_bytes
            FROM ml_data_version_control.dataset_versions dv
            WHERE dv.dataset_id = v_dataset.dataset_id
              AND dv.created_at < CURRENT_DATE - INTERVAL '1 day' * p_retention_days
              AND dv.version_number <= (
                  SELECT MAX(version_number) - p_keep_minimum_versions
                  FROM ml_data_version_control.dataset_versions
                  WHERE dataset_id = v_dataset.dataset_id
              )
            ORDER BY dv.version_number
        LOOP
            v_table_name := format('ml_data_version_control.dataset_%s_v%s',
                                  replace(v_dataset.dataset_name, '-', '_'),
                                  v_version.version_number);
            
            v_archive_table_name := format('ml_data_archive.dataset_%s_v%s',
                                          replace(v_dataset.dataset_name, '-', '_'),
                                          v_version.version_number);
            
            IF NOT p_dry_run THEN
                -- Create archive schema if it doesn't exist
                CREATE SCHEMA IF NOT EXISTS ml_data_archive;
                
                -- Move table to archive
                EXECUTE format('ALTER TABLE %s SET SCHEMA ml_data_archive', v_table_name);
                
                -- Update version status
                UPDATE ml_data_version_control.dataset_versions
                SET validation_status = 'ARCHIVED'
                WHERE version_id = v_version.version_id;
            END IF;
            
            RETURN QUERY SELECT 
                'ARCHIVE'::VARCHAR(20),
                v_dataset.dataset_name,
                v_version.version_number,
                v_version.version_id,
                v_version.data_size_bytes,
                format('Version older than %s days', p_retention_days);
        END LOOP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

### Real-World Applications

1. **Reproducible Research**: Enable exact recreation of experiments and model training
2. **Compliance and Auditing**: Maintain detailed data lineage for regulatory requirements
3. **A/B Testing**: Track different dataset versions used in model comparisons
4. **Data Quality Management**: Monitor data quality evolution over time
5. **Collaboration**: Enable team access to specific dataset versions

### Best Practices

1. **Semantic Versioning**: Use meaningful version naming conventions (major.minor.patch)
2. **Automated Validation**: Implement comprehensive data quality checks
3. **Storage Optimization**: Use compression and deduplication for large datasets
4. **Access Control**: Implement role-based permissions for dataset access
5. **Documentation**: Maintain detailed metadata and change logs

### Common Pitfalls

1. **Storage Explosion**: Not implementing proper retention policies
2. **Missing Validation**: Inadequate data quality checks on version creation
3. **Circular Dependencies**: Creating dependency loops between datasets
4. **Insufficient Metadata**: Lacking comprehensive dataset documentation
5. **Manual Processes**: Not automating version creation and validation

### Performance Considerations

- **Partitioning**: Partition large datasets by version or date ranges
- **Compression**: Use table compression for archived versions
- **Indexing**: Create appropriate indexes for version queries
- **Parallel Processing**: Leverage parallel processing for large dataset operations
- **Storage Tiering**: Move older versions to cheaper storage tiers

---

## Question 18

**What is Data Lineage, and how can you track it using SQL?**

**Answer:**

### Theory

Data lineage is the comprehensive tracking of data flow, transformations, and dependencies throughout its lifecycle, from source systems to final outputs. It provides a complete audit trail showing how data moves and changes across systems, tables, views, and processes. For machine learning applications, data lineage is crucial for understanding feature engineering pipelines, model reproducibility, compliance requirements, impact analysis, and debugging data quality issues.

**Key Components of Data Lineage:**
- **Source-to-Target Mapping**: Track data from origin to final destination
- **Transformation Logic**: Document all data modifications and processing steps
- **Dependency Analysis**: Understand upstream and downstream relationships
- **Impact Assessment**: Analyze effects of changes on dependent systems
- **Compliance Tracking**: Maintain audit trails for regulatory requirements

### SQL-Based Data Lineage Implementation

#### 1. Core Lineage Tracking Schema

```sql
-- Comprehensive data lineage tracking system
CREATE SCHEMA data_lineage;

-- Data sources and systems
CREATE TABLE data_lineage.data_sources (
    source_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name VARCHAR(255) NOT NULL UNIQUE,
    source_type VARCHAR(50) NOT NULL, -- 'DATABASE', 'API', 'FILE', 'STREAM', 'EXTERNAL'
    connection_string TEXT,
    description TEXT,
    owner_team VARCHAR(100),
    
    -- Classification and metadata
    data_classification VARCHAR(50), -- 'PUBLIC', 'INTERNAL', 'CONFIDENTIAL', 'RESTRICTED'
    compliance_tags TEXT[], -- ['GDPR', 'PCI', 'HIPAA', 'SOX']
    geographic_region VARCHAR(50),
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'DEPRECATED', 'OFFLINE')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_sources_type_status (source_type, status),
    INDEX idx_sources_classification (data_classification)
);

-- Data entities (tables, views, files, etc.)
CREATE TABLE data_lineage.data_entities (
    entity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES data_lineage.data_sources(source_id),
    entity_name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL, -- 'TABLE', 'VIEW', 'MATERIALIZED_VIEW', 'FILE', 'API_ENDPOINT'
    schema_name VARCHAR(255),
    full_path TEXT NOT NULL, -- Complete path/identifier
    
    -- Entity metadata
    description TEXT,
    business_purpose TEXT,
    data_owner VARCHAR(100),
    steward_contact VARCHAR(255),
    
    -- Schema and structure
    column_definitions JSONB,
    primary_keys TEXT[],
    indexes_info JSONB,
    constraints_info JSONB,
    
    -- Data characteristics
    estimated_row_count BIGINT,
    data_size_bytes BIGINT,
    update_frequency VARCHAR(50), -- 'REAL_TIME', 'HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY', 'ON_DEMAND'
    last_updated TIMESTAMP,
    
    -- Quality and compliance
    data_quality_score NUMERIC(5,4),
    contains_pii BOOLEAN DEFAULT FALSE,
    retention_period INTERVAL,
    
    -- Lineage metadata
    lineage_level INTEGER DEFAULT 0, -- Distance from source (0 = raw source)
    is_derived BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(source_id, schema_name, entity_name),
    INDEX idx_entities_source_schema (source_id, schema_name),
    INDEX idx_entities_type_level (entity_type, lineage_level),
    INDEX idx_entities_pii (contains_pii),
    INDEX idx_entities_updated (last_updated DESC)
);

-- Data transformations and processes
CREATE TABLE data_lineage.data_transformations (
    transformation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transformation_name VARCHAR(255) NOT NULL,
    transformation_type VARCHAR(50) NOT NULL, -- 'ETL', 'SQL_QUERY', 'FUNCTION', 'PROCEDURE', 'ML_PIPELINE', 'API_CALL'
    
    -- Process definition
    process_definition TEXT NOT NULL, -- SQL query, code, or process description
    execution_engine VARCHAR(50), -- 'POSTGRESQL', 'SPARK', 'AIRFLOW', 'DBT', 'PYTHON'
    
    -- Business context
    business_rule_description TEXT,
    transformation_purpose TEXT,
    owner_team VARCHAR(100),
    
    -- Technical details
    execution_frequency VARCHAR(50),
    average_runtime_seconds INTEGER,
    resource_requirements JSONB, -- Memory, CPU, etc.
    
    -- Dependencies and scheduling
    depends_on_transformations UUID[],
    schedule_expression VARCHAR(100), -- Cron expression or schedule
    
    -- Quality and monitoring
    data_quality_checks JSONB,
    alert_conditions JSONB,
    
    -- Version control
    version VARCHAR(50) DEFAULT '1.0',
    git_commit_hash VARCHAR(40),
    deployment_environment VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_transformations_type (transformation_type),
    INDEX idx_transformations_team (owner_team),
    INDEX idx_transformations_environment (deployment_environment)
);

-- Lineage relationships between entities
CREATE TABLE data_lineage.lineage_relationships (
    relationship_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID NOT NULL REFERENCES data_lineage.data_entities(entity_id),
    target_entity_id UUID NOT NULL REFERENCES data_lineage.data_entities(entity_id),
    transformation_id UUID REFERENCES data_lineage.data_transformations(transformation_id),
    
    -- Relationship details
    relationship_type VARCHAR(50) NOT NULL, -- 'DIRECT_COPY', 'TRANSFORMATION', 'AGGREGATION', 'JOIN', 'FILTER', 'UNION'
    confidence_score NUMERIC(3,2) DEFAULT 1.0, -- 0.0 to 1.0
    
    -- Column-level lineage
    column_mappings JSONB, -- Source to target column mappings
    transformation_logic TEXT, -- Specific transformation applied
    
    -- Metadata
    discovered_method VARCHAR(50), -- 'QUERY_PARSING', 'METADATA_EXTRACTION', 'MANUAL', 'ML_INFERENCE'
    discovery_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_verified TIMESTAMP,
    
    -- Data flow characteristics
    data_volume_per_day BIGINT,
    latency_minutes INTEGER, -- How long data takes to flow from source to target
    
    -- Quality and validation
    is_active BOOLEAN DEFAULT TRUE,
    validation_status VARCHAR(20) DEFAULT 'UNVERIFIED', -- 'VERIFIED', 'UNVERIFIED', 'INVALID'
    validation_notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Prevent self-references and duplicates
    CHECK (source_entity_id != target_entity_id),
    UNIQUE(source_entity_id, target_entity_id, transformation_id),
    
    INDEX idx_lineage_source (source_entity_id),
    INDEX idx_lineage_target (target_entity_id),
    INDEX idx_lineage_transformation (transformation_id),
    INDEX idx_lineage_type_active (relationship_type, is_active)
);

-- Execution history for transformations
CREATE TABLE data_lineage.execution_history (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transformation_id UUID NOT NULL REFERENCES data_lineage.data_transformations(transformation_id),
    
    -- Execution details
    execution_start TIMESTAMP NOT NULL,
    execution_end TIMESTAMP,
    execution_status VARCHAR(20) NOT NULL CHECK (execution_status IN ('RUNNING', 'SUCCESS', 'FAILED', 'CANCELLED')),
    
    -- Performance metrics
    rows_processed BIGINT,
    rows_inserted BIGINT,
    rows_updated BIGINT,
    rows_deleted BIGINT,
    data_volume_bytes BIGINT,
    
    -- Resource utilization
    cpu_time_seconds INTEGER,
    memory_peak_mb INTEGER,
    io_read_bytes BIGINT,
    io_write_bytes BIGINT,
    
    -- Error handling
    error_message TEXT,
    error_code VARCHAR(50),
    retry_count INTEGER DEFAULT 0,
    
    -- Data quality results
    quality_check_results JSONB,
    data_quality_score NUMERIC(5,4),
    
    -- Lineage impact
    affected_entities UUID[], -- Entities modified by this execution
    
    -- Execution context
    triggered_by VARCHAR(50), -- 'SCHEDULE', 'MANUAL', 'EVENT', 'DEPENDENCY'
    execution_environment VARCHAR(50),
    job_id VARCHAR(255), -- External job/batch ID
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_execution_transformation_time (transformation_id, execution_start DESC),
    INDEX idx_execution_status_time (execution_status, execution_start DESC),
    INDEX idx_execution_job (job_id)
);

-- Data lineage events and changes
CREATE TABLE data_lineage.lineage_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL, -- 'ENTITY_CREATED', 'ENTITY_MODIFIED', 'RELATIONSHIP_ADDED', 'TRANSFORMATION_EXECUTED'
    entity_id UUID REFERENCES data_lineage.data_entities(entity_id),
    transformation_id UUID REFERENCES data_lineage.data_transformations(transformation_id),
    relationship_id UUID REFERENCES data_lineage.lineage_relationships(relationship_id),
    
    -- Event details
    event_description TEXT,
    change_details JSONB, -- Before/after states, specific changes
    
    -- Context
    user_id UUID,
    session_id VARCHAR(255),
    application_name VARCHAR(100),
    
    -- Impact analysis
    impact_scope TEXT[], -- List of affected downstream entities
    change_risk_level VARCHAR(20), -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_events_type_time (event_type, event_timestamp DESC),
    INDEX idx_events_entity (entity_id),
    INDEX idx_events_transformation (transformation_id)
);
```

#### 2. Automated Lineage Discovery and Tracking

```sql
-- Function to automatically discover lineage from SQL queries
CREATE OR REPLACE FUNCTION data_lineage.parse_sql_lineage(
    p_sql_query TEXT,
    p_transformation_id UUID,
    p_target_entity_id UUID
) RETURNS TABLE (
    source_table TEXT,
    source_columns TEXT[],
    target_columns TEXT[],
    transformation_type VARCHAR(50)
) AS $$
DECLARE
    v_parsed_info RECORD;
    v_source_entity_id UUID;
    v_relationship_id UUID;
BEGIN
    -- Simplified SQL parsing (in production, use proper SQL parser)
    -- This is a basic example - real implementation would use advanced parsing
    
    -- Extract source tables using regex patterns
    -- Note: This is simplified - production systems should use proper SQL parsers
    
    WITH sql_analysis AS (
        SELECT 
            regexp_matches(
                upper(p_sql_query), 
                'FROM\s+([a-zA-Z_][a-zA-Z0-9_.]*)', 
                'g'
            ) as from_matches,
            regexp_matches(
                upper(p_sql_query), 
                'JOIN\s+([a-zA-Z_][a-zA-Z0-9_.]*)', 
                'g'
            ) as join_matches,
            CASE 
                WHEN upper(p_sql_query) LIKE '%GROUP BY%' THEN 'AGGREGATION'
                WHEN upper(p_sql_query) LIKE '%UNION%' THEN 'UNION'
                WHEN upper(p_sql_query) LIKE '%JOIN%' THEN 'JOIN'
                WHEN upper(p_sql_query) LIKE '%WHERE%' THEN 'FILTER'
                ELSE 'DIRECT_COPY'
            END as detected_transformation_type
    ),
    source_tables AS (
        SELECT 
            unnest(
                COALESCE(sa.from_matches, ARRAY[]::TEXT[]) || 
                COALESCE(sa.join_matches, ARRAY[]::TEXT[])
            ) as table_name,
            sa.detected_transformation_type
        FROM sql_analysis sa
    )
    SELECT 
        st.table_name,
        ARRAY[]::TEXT[] as source_cols, -- Simplified - would extract actual columns
        ARRAY[]::TEXT[] as target_cols, -- Simplified - would extract actual columns
        st.detected_transformation_type
    FROM source_tables st;
    
    -- Create lineage relationships for discovered sources
    FOR v_parsed_info IN 
        SELECT * FROM data_lineage.parse_sql_lineage(p_sql_query, p_transformation_id, p_target_entity_id)
    LOOP
        -- Find source entity
        SELECT entity_id INTO v_source_entity_id
        FROM data_lineage.data_entities
        WHERE entity_name = v_parsed_info.source_table
        LIMIT 1;
        
        IF v_source_entity_id IS NOT NULL THEN
            -- Create lineage relationship
            INSERT INTO data_lineage.lineage_relationships (
                source_entity_id,
                target_entity_id,
                transformation_id,
                relationship_type,
                column_mappings,
                transformation_logic,
                discovered_method
            ) VALUES (
                v_source_entity_id,
                p_target_entity_id,
                p_transformation_id,
                v_parsed_info.transformation_type,
                jsonb_build_object(
                    'source_columns', v_parsed_info.source_columns,
                    'target_columns', v_parsed_info.target_columns
                ),
                p_sql_query,
                'QUERY_PARSING'
            ) ON CONFLICT (source_entity_id, target_entity_id, transformation_id) 
              DO NOTHING
            RETURNING relationship_id INTO v_relationship_id;
            
            -- Log lineage discovery event
            INSERT INTO data_lineage.lineage_events (
                event_type,
                relationship_id,
                transformation_id,
                event_description
            ) VALUES (
                'RELATIONSHIP_DISCOVERED',
                v_relationship_id,
                p_transformation_id,
                format('Discovered lineage relationship from %s via SQL parsing', v_parsed_info.source_table)
            );
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to track ML pipeline lineage
CREATE OR REPLACE FUNCTION data_lineage.track_ml_pipeline_lineage(
    p_pipeline_name VARCHAR(255),
    p_input_datasets UUID[],
    p_output_dataset UUID,
    p_feature_engineering_steps JSONB,
    p_model_id UUID,
    p_executed_by UUID
) RETURNS UUID AS $$
DECLARE
    v_transformation_id UUID;
    v_input_dataset UUID;
    v_relationship_id UUID;
BEGIN
    -- Create transformation record for ML pipeline
    INSERT INTO data_lineage.data_transformations (
        transformation_name,
        transformation_type,
        process_definition,
        execution_engine,
        business_rule_description,
        owner_team,
        data_quality_checks
    ) VALUES (
        p_pipeline_name,
        'ML_PIPELINE',
        p_feature_engineering_steps::TEXT,
        'PYTHON_SKLEARN', -- Example
        'Machine learning feature engineering and model training pipeline',
        'DATA_SCIENCE_TEAM',
        jsonb_build_object(
            'feature_validation', true,
            'data_drift_detection', true,
            'model_performance_check', true
        )
    ) RETURNING transformation_id INTO v_transformation_id;
    
    -- Create lineage relationships for each input dataset
    FOREACH v_input_dataset IN ARRAY p_input_datasets
    LOOP
        INSERT INTO data_lineage.lineage_relationships (
            source_entity_id,
            target_entity_id,
            transformation_id,
            relationship_type,
            transformation_logic,
            discovered_method
        ) VALUES (
            v_input_dataset,
            p_output_dataset,
            v_transformation_id,
            'ML_FEATURE_ENGINEERING',
            format('ML pipeline: %s', p_pipeline_name),
            'ML_PIPELINE_TRACKING'
        ) RETURNING relationship_id INTO v_relationship_id;
        
        -- Log ML lineage event
        INSERT INTO data_lineage.lineage_events (
            event_type,
            relationship_id,
            transformation_id,
            event_description,
            user_id,
            change_details
        ) VALUES (
            'ML_PIPELINE_EXECUTED',
            v_relationship_id,
            v_transformation_id,
            format('ML pipeline %s executed with model %s', p_pipeline_name, p_model_id),
            p_executed_by,
            jsonb_build_object(
                'model_id', p_model_id,
                'input_datasets', p_input_datasets,
                'feature_engineering_steps', p_feature_engineering_steps
            )
        );
    END LOOP;
    
    RETURN v_transformation_id;
END;
$$ LANGUAGE plpgsql;
```

#### 3. Lineage Visualization and Analysis

```sql
-- Comprehensive lineage analysis views and functions
CREATE OR REPLACE VIEW data_lineage.lineage_summary AS
WITH RECURSIVE lineage_tree AS (
    -- Base case: source entities (no upstream dependencies)
    SELECT 
        de.entity_id,
        de.entity_name,
        de.source_id,
        ds.source_name,
        de.lineage_level,
        ARRAY[de.entity_id] as lineage_path,
        de.entity_name as root_source,
        0 as depth_from_source
    FROM data_lineage.data_entities de
    JOIN data_lineage.data_sources ds ON de.source_id = ds.source_id
    WHERE NOT EXISTS (
        SELECT 1 FROM data_lineage.lineage_relationships lr
        WHERE lr.target_entity_id = de.entity_id
          AND lr.is_active = TRUE
    )
    
    UNION ALL
    
    -- Recursive case: entities with upstream dependencies
    SELECT 
        de.entity_id,
        de.entity_name,
        de.source_id,
        ds.source_name,
        de.lineage_level,
        lt.lineage_path || de.entity_id,
        lt.root_source,
        lt.depth_from_source + 1
    FROM data_lineage.data_entities de
    JOIN data_lineage.data_sources ds ON de.source_id = ds.source_id
    JOIN data_lineage.lineage_relationships lr ON de.entity_id = lr.target_entity_id
    JOIN lineage_tree lt ON lr.source_entity_id = lt.entity_id
    WHERE lr.is_active = TRUE
      AND NOT (de.entity_id = ANY(lt.lineage_path)) -- Prevent cycles
),
lineage_stats AS (
    SELECT 
        entity_id,
        COUNT(*) FILTER (WHERE relationship_type = 'DIRECT_COPY') as direct_copy_count,
        COUNT(*) FILTER (WHERE relationship_type = 'TRANSFORMATION') as transformation_count,
        COUNT(*) FILTER (WHERE relationship_type = 'AGGREGATION') as aggregation_count,
        COUNT(*) FILTER (WHERE relationship_type = 'JOIN') as join_count,
        COUNT(*) as total_upstream_dependencies,
        AVG(confidence_score) as avg_confidence_score
    FROM data_lineage.lineage_relationships lr
    WHERE lr.is_active = TRUE
    GROUP BY entity_id
),
downstream_stats AS (
    SELECT 
        source_entity_id as entity_id,
        COUNT(*) as downstream_dependent_count,
        COUNT(DISTINCT target_entity_id) as unique_downstream_entities
    FROM data_lineage.lineage_relationships lr
    WHERE lr.is_active = TRUE
    GROUP BY source_entity_id
)
SELECT 
    lt.entity_id,
    lt.entity_name,
    lt.source_name,
    lt.root_source,
    lt.depth_from_source,
    lt.lineage_level,
    
    -- Upstream lineage statistics
    COALESCE(ls.total_upstream_dependencies, 0) as upstream_dependencies,
    COALESCE(ls.direct_copy_count, 0) as direct_copies,
    COALESCE(ls.transformation_count, 0) as transformations,
    COALESCE(ls.aggregation_count, 0) as aggregations,
    COALESCE(ls.join_count, 0) as joins,
    ROUND(COALESCE(ls.avg_confidence_score, 0)::NUMERIC, 3) as avg_lineage_confidence,
    
    -- Downstream impact
    COALESCE(ds.downstream_dependent_count, 0) as downstream_dependencies,
    COALESCE(ds.unique_downstream_entities, 0) as unique_downstream_entities,
    
    -- Complexity indicators
    CASE 
        WHEN COALESCE(ls.total_upstream_dependencies, 0) = 0 THEN 'SOURCE'
        WHEN COALESCE(ls.total_upstream_dependencies, 0) <= 3 THEN 'SIMPLE'
        WHEN COALESCE(ls.total_upstream_dependencies, 0) <= 10 THEN 'MODERATE'
        ELSE 'COMPLEX'
    END as complexity_level,
    
    -- Impact assessment
    CASE 
        WHEN COALESCE(ds.downstream_dependent_count, 0) = 0 THEN 'LEAF'
        WHEN COALESCE(ds.downstream_dependent_count, 0) <= 5 THEN 'LOW_IMPACT'
        WHEN COALESCE(ds.downstream_dependent_count, 0) <= 20 THEN 'MEDIUM_IMPACT'
        ELSE 'HIGH_IMPACT'
    END as impact_level,
    
    -- Lineage quality
    CASE 
        WHEN COALESCE(ls.avg_confidence_score, 0) >= 0.9 THEN 'HIGH_CONFIDENCE'
        WHEN COALESCE(ls.avg_confidence_score, 0) >= 0.7 THEN 'MEDIUM_CONFIDENCE'
        WHEN COALESCE(ls.avg_confidence_score, 0) >= 0.5 THEN 'LOW_CONFIDENCE'
        ELSE 'UNVERIFIED'
    END as lineage_quality,
    
    -- Lineage path for visualization
    array_to_string(
        ARRAY(
            SELECT de2.entity_name 
            FROM unnest(lt.lineage_path) WITH ORDINALITY t(entity_id, ord)
            JOIN data_lineage.data_entities de2 ON t.entity_id = de2.entity_id
            ORDER BY t.ord
        ), 
        ' → '
    ) as lineage_path_display
    
FROM lineage_tree lt
LEFT JOIN lineage_stats ls ON lt.entity_id = ls.entity_id
LEFT JOIN downstream_stats ds ON lt.entity_id = ds.entity_id;

-- Function for impact analysis when changes occur
CREATE OR REPLACE FUNCTION data_lineage.analyze_change_impact(
    p_entity_id UUID,
    p_change_type VARCHAR(50) DEFAULT 'SCHEMA_CHANGE',
    p_max_depth INTEGER DEFAULT 10
) RETURNS TABLE (
    impacted_entity_id UUID,
    impacted_entity_name VARCHAR(255),
    impact_distance INTEGER,
    impact_type VARCHAR(50),
    risk_level VARCHAR(20),
    estimated_records_affected BIGINT,
    transformation_names TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE impact_analysis AS (
        -- Base case: directly impacted entities
        SELECT 
            lr.target_entity_id as entity_id,
            de.entity_name,
            1 as distance,
            lr.relationship_type as impact_type,
            ARRAY[dt.transformation_name] as transformation_path,
            de.estimated_row_count
        FROM data_lineage.lineage_relationships lr
        JOIN data_lineage.data_entities de ON lr.target_entity_id = de.entity_id
        LEFT JOIN data_lineage.data_transformations dt ON lr.transformation_id = dt.transformation_id
        WHERE lr.source_entity_id = p_entity_id
          AND lr.is_active = TRUE
        
        UNION ALL
        
        -- Recursive case: downstream impacts
        SELECT 
            lr.target_entity_id,
            de.entity_name,
            ia.distance + 1,
            lr.relationship_type,
            ia.transformation_path || dt.transformation_name,
            de.estimated_row_count
        FROM impact_analysis ia
        JOIN data_lineage.lineage_relationships lr ON ia.entity_id = lr.source_entity_id
        JOIN data_lineage.data_entities de ON lr.target_entity_id = de.entity_id
        LEFT JOIN data_lineage.data_transformations dt ON lr.transformation_id = dt.transformation_id
        WHERE lr.is_active = TRUE
          AND ia.distance < p_max_depth
          AND NOT (lr.target_entity_id = ANY(SELECT entity_id FROM impact_analysis)) -- Prevent cycles
    )
    SELECT 
        ia.entity_id,
        ia.entity_name,
        ia.distance,
        ia.impact_type,
        CASE 
            WHEN ia.distance = 1 THEN 'HIGH'
            WHEN ia.distance <= 3 THEN 'MEDIUM'
            ELSE 'LOW'
        END as risk_level,
        COALESCE(ia.estimated_row_count, 0),
        ia.transformation_path
    FROM impact_analysis ia
    ORDER BY ia.distance, ia.entity_name;
END;
$$ LANGUAGE plpgsql;

-- Data lineage health monitoring
CREATE OR REPLACE VIEW data_lineage.lineage_health_report AS
WITH lineage_metrics AS (
    SELECT 
        COUNT(*) as total_entities,
        COUNT(*) FILTER (WHERE de.lineage_level = 0) as source_entities,
        COUNT(*) FILTER (WHERE de.is_derived = TRUE) as derived_entities,
        COUNT(*) FILTER (WHERE de.contains_pii = TRUE) as pii_entities,
        COUNT(*) FILTER (WHERE de.data_quality_score < 0.7) as low_quality_entities,
        AVG(de.data_quality_score) as avg_quality_score
    FROM data_lineage.data_entities de
    WHERE de.created_at >= CURRENT_DATE - INTERVAL '30 days'
),
relationship_metrics AS (
    SELECT 
        COUNT(*) as total_relationships,
        COUNT(*) FILTER (WHERE lr.validation_status = 'VERIFIED') as verified_relationships,
        COUNT(*) FILTER (WHERE lr.validation_status = 'UNVERIFIED') as unverified_relationships,
        COUNT(*) FILTER (WHERE lr.validation_status = 'INVALID') as invalid_relationships,
        AVG(lr.confidence_score) as avg_confidence_score,
        COUNT(*) FILTER (WHERE lr.last_verified < CURRENT_DATE - INTERVAL '90 days') as stale_relationships
    FROM data_lineage.lineage_relationships lr
    WHERE lr.is_active = TRUE
),
transformation_metrics AS (
    SELECT 
        COUNT(*) as total_transformations,
        COUNT(*) FILTER (WHERE dt.updated_at >= CURRENT_DATE - INTERVAL '30 days') as recently_updated,
        COUNT(DISTINCT dt.owner_team) as active_teams,
        AVG(eh.data_quality_score) as avg_execution_quality
    FROM data_lineage.data_transformations dt
    LEFT JOIN (
        SELECT 
            transformation_id,
            AVG(data_quality_score) as data_quality_score
        FROM data_lineage.execution_history
        WHERE execution_start >= CURRENT_DATE - INTERVAL '7 days'
          AND execution_status = 'SUCCESS'
        GROUP BY transformation_id
    ) eh ON dt.transformation_id = eh.transformation_id
),
coverage_metrics AS (
    SELECT 
        COUNT(*) FILTER (WHERE de.entity_type = 'TABLE') as tables_total,
        COUNT(*) FILTER (WHERE de.entity_type = 'TABLE' AND EXISTS (
            SELECT 1 FROM data_lineage.lineage_relationships lr 
            WHERE lr.source_entity_id = de.entity_id OR lr.target_entity_id = de.entity_id
        )) as tables_with_lineage,
        COUNT(*) FILTER (WHERE de.entity_type = 'VIEW') as views_total,
        COUNT(*) FILTER (WHERE de.entity_type = 'VIEW' AND EXISTS (
            SELECT 1 FROM data_lineage.lineage_relationships lr 
            WHERE lr.source_entity_id = de.entity_id OR lr.target_entity_id = de.entity_id
        )) as views_with_lineage
    FROM data_lineage.data_entities de
)
SELECT 
    -- Entity health
    lm.total_entities,
    lm.source_entities,
    lm.derived_entities,
    lm.pii_entities,
    lm.low_quality_entities,
    ROUND(lm.avg_quality_score::NUMERIC, 3) as avg_entity_quality,
    
    -- Relationship health  
    rm.total_relationships,
    rm.verified_relationships,
    rm.unverified_relationships,
    rm.invalid_relationships,
    rm.stale_relationships,
    ROUND(rm.avg_confidence_score::NUMERIC, 3) as avg_relationship_confidence,
    
    -- Transformation health
    tm.total_transformations,
    tm.recently_updated,
    tm.active_teams,
    ROUND(tm.avg_execution_quality::NUMERIC, 3) as avg_transformation_quality,
    
    -- Coverage metrics
    cm.tables_total,
    cm.tables_with_lineage,
    ROUND(cm.tables_with_lineage * 100.0 / NULLIF(cm.tables_total, 0), 1) as table_lineage_coverage_pct,
    cm.views_total,
    cm.views_with_lineage,
    ROUND(cm.views_with_lineage * 100.0 / NULLIF(cm.views_total, 0), 1) as view_lineage_coverage_pct,
    
    -- Overall health score
    ROUND((
        (rm.verified_relationships * 100.0 / NULLIF(rm.total_relationships, 0)) * 0.3 +
        (lm.avg_quality_score * 100) * 0.3 +
        (rm.avg_confidence_score * 100) * 0.2 +
        (cm.tables_with_lineage * 100.0 / NULLIF(cm.tables_total, 0)) * 0.2
    )::NUMERIC, 1) as overall_lineage_health_score,
    
    -- Health assessment
    CASE 
        WHEN (rm.verified_relationships * 100.0 / NULLIF(rm.total_relationships, 0)) >= 80 
             AND lm.avg_quality_score >= 0.8 
             AND rm.avg_confidence_score >= 0.8 THEN 'EXCELLENT'
        WHEN (rm.verified_relationships * 100.0 / NULLIF(rm.total_relationships, 0)) >= 60 
             AND lm.avg_quality_score >= 0.7 
             AND rm.avg_confidence_score >= 0.7 THEN 'GOOD'
        WHEN (rm.verified_relationships * 100.0 / NULLIF(rm.total_relationships, 0)) >= 40 
             AND lm.avg_quality_score >= 0.6 
             AND rm.avg_confidence_score >= 0.6 THEN 'FAIR'
        ELSE 'POOR'
    END as health_assessment,
    
    -- Recommendations
    ARRAY[
        CASE WHEN rm.unverified_relationships > rm.verified_relationships 
             THEN 'Verify unverified lineage relationships' END,
        CASE WHEN rm.stale_relationships > rm.total_relationships * 0.2 
             THEN 'Update stale relationship validations' END,
        CASE WHEN lm.low_quality_entities > lm.total_entities * 0.1 
             THEN 'Improve data quality for low-scoring entities' END,
        CASE WHEN cm.tables_with_lineage * 100.0 / NULLIF(cm.tables_total, 0) < 70 
             THEN 'Increase lineage coverage for tables' END
    ]::TEXT[] as improvement_recommendations
    
FROM lineage_metrics lm
CROSS JOIN relationship_metrics rm  
CROSS JOIN transformation_metrics tm
CROSS JOIN coverage_metrics cm;
```

### Advanced Lineage Applications

#### 1. ML Model Lineage and Feature Attribution

```sql
-- ML model lineage tracking for feature attribution
CREATE OR REPLACE FUNCTION data_lineage.track_ml_feature_lineage(
    p_model_id UUID,
    p_feature_importances JSONB, -- {"feature_name": importance_score}
    p_training_dataset_id UUID,
    p_validation_dataset_id UUID,
    p_feature_engineering_transformations JSONB
) RETURNS TABLE (
    feature_name TEXT,
    importance_score NUMERIC,
    source_entities TEXT[],
    transformation_chain TEXT[],
    data_freshness_hours INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH feature_analysis AS (
        SELECT 
            key as feature_name,
            value::NUMERIC as importance_score
        FROM jsonb_each_text(p_feature_importances)
    ),
    feature_lineage AS (
        SELECT 
            fa.feature_name,
            fa.importance_score,
            
            -- Find source entities for each feature through lineage
            ARRAY_AGG(DISTINCT de_source.entity_name ORDER BY de_source.entity_name) as source_entities,
            
            -- Build transformation chain
            ARRAY_AGG(DISTINCT dt.transformation_name ORDER BY dt.transformation_name) as transformations,
            
            -- Calculate data freshness
            EXTRACT(HOURS FROM (CURRENT_TIMESTAMP - MAX(de_source.last_updated))) as freshness_hours
            
        FROM feature_analysis fa
        LEFT JOIN data_lineage.data_entities de_target ON de_target.entity_id = p_training_dataset_id
        LEFT JOIN data_lineage.lineage_relationships lr ON lr.target_entity_id = de_target.entity_id
        LEFT JOIN data_lineage.data_entities de_source ON lr.source_entity_id = de_source.entity_id
        LEFT JOIN data_lineage.data_transformations dt ON lr.transformation_id = dt.transformation_id
        WHERE lr.is_active = TRUE
        GROUP BY fa.feature_name, fa.importance_score
    )
    SELECT 
        fl.feature_name,
        ROUND(fl.importance_score::NUMERIC, 4),
        fl.source_entities,
        fl.transformations,
        COALESCE(fl.freshness_hours, 0)::INTEGER
    FROM feature_lineage fl
    ORDER BY fl.importance_score DESC;
END;
$$ LANGUAGE plpgsql;

-- Data quality lineage propagation
CREATE OR REPLACE FUNCTION data_lineage.propagate_data_quality_issues(
    p_source_entity_id UUID,
    p_quality_issue_type VARCHAR(100),
    p_issue_description TEXT,
    p_severity VARCHAR(20) DEFAULT 'MEDIUM'
) RETURNS TABLE (
    affected_entity_id UUID,
    affected_entity_name VARCHAR(255),
    propagation_distance INTEGER,
    estimated_impact_percentage NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE quality_propagation AS (
        -- Base case: direct downstream entities
        SELECT 
            lr.target_entity_id as entity_id,
            de.entity_name,
            1 as distance,
            CASE lr.relationship_type
                WHEN 'DIRECT_COPY' THEN 100.0
                WHEN 'FILTER' THEN 90.0
                WHEN 'TRANSFORMATION' THEN 75.0
                WHEN 'AGGREGATION' THEN 60.0
                WHEN 'JOIN' THEN 50.0
                ELSE 30.0
            END as impact_percentage
        FROM data_lineage.lineage_relationships lr
        JOIN data_lineage.data_entities de ON lr.target_entity_id = de.entity_id
        WHERE lr.source_entity_id = p_source_entity_id
          AND lr.is_active = TRUE
        
        UNION ALL
        
        -- Recursive case: further downstream propagation
        SELECT 
            lr.target_entity_id,
            de.entity_name,
            qp.distance + 1,
            qp.impact_percentage * 
            CASE lr.relationship_type
                WHEN 'DIRECT_COPY' THEN 0.9
                WHEN 'FILTER' THEN 0.8
                WHEN 'TRANSFORMATION' THEN 0.7
                WHEN 'AGGREGATION' THEN 0.6
                WHEN 'JOIN' THEN 0.5
                ELSE 0.3
            END as impact_percentage
        FROM quality_propagation qp
        JOIN data_lineage.lineage_relationships lr ON qp.entity_id = lr.source_entity_id
        JOIN data_lineage.data_entities de ON lr.target_entity_id = de.entity_id
        WHERE lr.is_active = TRUE
          AND qp.distance < 5 -- Limit propagation depth
          AND qp.impact_percentage > 10 -- Only propagate significant impacts
    )
    SELECT 
        qp.entity_id,
        qp.entity_name,
        qp.distance,
        ROUND(qp.impact_percentage, 1)
    FROM quality_propagation qp
    ORDER BY qp.distance, qp.impact_percentage DESC;
    
    -- Log quality issue propagation
    INSERT INTO data_lineage.lineage_events (
        event_type,
        entity_id,
        event_description,
        change_details,
        change_risk_level
    ) 
    SELECT 
        'DATA_QUALITY_ISSUE_PROPAGATED',
        qp.entity_id,
        format('Data quality issue propagated from source entity: %s', p_quality_issue_type),
        jsonb_build_object(
            'source_entity_id', p_source_entity_id,
            'issue_type', p_quality_issue_type,
            'issue_description', p_issue_description,
            'propagation_distance', qp.distance,
            'estimated_impact_percentage', qp.impact_percentage
        ),
        p_severity
    FROM (
        SELECT entity_id, distance, impact_percentage 
        FROM data_lineage.propagate_data_quality_issues(p_source_entity_id, p_quality_issue_type, p_issue_description, p_severity)
    ) qp;
END;
$$ LANGUAGE plpgsql;
```

### Real-World Applications

1. **Regulatory Compliance**: Track data flow for GDPR, CCPA, and other privacy regulations
2. **Impact Analysis**: Assess downstream effects of schema changes or data issues
3. **Root Cause Analysis**: Trace data quality problems back to their sources
4. **Model Governance**: Understand feature dependencies and model reproducibility
5. **Data Discovery**: Help users find relevant datasets and understand data relationships

### Best Practices

1. **Automated Discovery**: Implement automated lineage extraction from SQL logs and metadata
2. **Real-Time Updates**: Keep lineage information current with data pipeline changes
3. **Validation and Verification**: Regularly validate lineage relationships for accuracy
4. **Business Context**: Include business terminology and descriptions in lineage metadata
5. **Performance Optimization**: Design efficient storage and querying for large lineage graphs

### Common Pitfalls

1. **Incomplete Coverage**: Missing lineage for critical data transformations
2. **Stale Information**: Outdated lineage not reflecting current data flows
3. **Over-Complexity**: Creating overly detailed lineage that's difficult to navigate
4. **Manual Maintenance**: Relying too heavily on manual lineage documentation
5. **Circular Dependencies**: Not properly handling circular references in data flows

### Performance Considerations

- **Graph Traversal**: Optimize recursive queries for large lineage graphs
- **Indexing Strategy**: Create appropriate indexes for lineage relationship queries
- **Materialized Views**: Cache complex lineage calculations for performance
- **Partitioning**: Partition large lineage tables by time or entity type
- **Query Optimization**: Use efficient algorithms for graph analysis and traversal

---

