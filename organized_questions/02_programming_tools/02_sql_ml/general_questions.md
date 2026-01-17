# SQL ML Interview Questions - General Questions

## Question 1

**What does GROUP BY do in a SQL query?**

### Definition
GROUP BY partitions rows into groups based on column values. Aggregate functions then operate on each group separately.

### Syntax
```sql
SELECT column, AGG_FUNCTION(column)
FROM table
GROUP BY column
HAVING AGG_CONDITION;
```

### Code Example
```sql
-- Basic aggregation
SELECT 
    department,
    COUNT(*) AS employee_count,
    AVG(salary) AS avg_salary,
    MAX(salary) AS max_salary
FROM employees
GROUP BY department
HAVING COUNT(*) >= 5;

-- Multiple column grouping
SELECT 
    department, job_level,
    COUNT(*) AS count,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY department, job_level
ORDER BY department, job_level;
```

### ML Feature Engineering Example
```sql
-- Create customer features
SELECT 
    customer_id,
    COUNT(*) AS total_purchases,
    SUM(amount) AS total_spent,
    AVG(amount) AS avg_purchase,
    MAX(purchase_date) AS last_purchase,
    COUNT(DISTINCT product_category) AS categories_bought
FROM purchases
GROUP BY customer_id;
```

### Rules
- SELECT can only include GROUP BY columns or aggregates
- Use HAVING for filtering on aggregates (not WHERE)
- NULL values form their own group

---

## Question 2

**Explain indexes and their importance for ML pipelines.**

### Definition
An index is a B-Tree data structure that enables O(log n) lookups instead of O(n) full table scans.

### Creating Indexes
```sql
-- Single column
CREATE INDEX idx_customer ON orders(customer_id);

-- Composite (order matters!)
CREATE INDEX idx_cust_date ON orders(customer_id, order_date);
```

### ML Impact

| Operation | Without Index | With Index |
|-----------|---------------|------------|
| Feature lookup | Full scan | Direct access |
| JOIN operations | Nested loops | Index scan |
| Real-time prediction | Seconds | Milliseconds |

### Best Practices
- Index columns used in WHERE, JOIN, ORDER BY
- Avoid over-indexing (slows writes)
- Don't index low-cardinality columns

---

## Question 3

**How do you handle NULL values in SQL?**

### Definition
NULL represents missing/unknown data. It's not equal to anything, including itself.

### Checking for NULL
```sql
-- Correct
WHERE column IS NULL
WHERE column IS NOT NULL

-- Wrong (never matches!)
WHERE column = NULL
```

### Handling NULLs
```sql
-- Replace NULL with default
SELECT COALESCE(column, 'default_value') AS column;

-- CASE statement
SELECT CASE WHEN column IS NULL THEN 'Unknown' ELSE column END;

-- NULL-safe comparison (MySQL)
WHERE column <=> other_column;
```

### ML Considerations
- Decide imputation strategy before querying
- Track NULL percentages for data quality
- Some models can't handle NULLs - handle in SQL or Python

---

## Question 4

**What is a Common Table Expression (CTE)?**

### Definition
A CTE is a named temporary result set defined within a query using WITH clause. Makes complex queries readable.

### Syntax
```sql
WITH cte_name AS (
    SELECT ...
)
SELECT * FROM cte_name;
```

### Code Example
```sql
-- Multiple CTEs for feature engineering
WITH customer_orders AS (
    SELECT 
        customer_id,
        COUNT(*) AS order_count,
        SUM(amount) AS total_spent
    FROM orders
    GROUP BY customer_id
),
customer_returns AS (
    SELECT 
        customer_id,
        COUNT(*) AS return_count
    FROM returns
    GROUP BY customer_id
)
SELECT 
    co.customer_id,
    co.order_count,
    co.total_spent,
    COALESCE(cr.return_count, 0) AS return_count,
    COALESCE(cr.return_count, 0) * 1.0 / co.order_count AS return_rate
FROM customer_orders co
LEFT JOIN customer_returns cr ON co.customer_id = cr.customer_id;
```

### Benefits
- Improved readability over nested subqueries
- Can reference CTE multiple times in query
- Recursive CTEs for hierarchical data

---

## Question 5

**What is the difference between UNION and UNION ALL?**

### Definition
Both combine results from multiple SELECT statements.

| Operator | Duplicates | Performance |
|----------|------------|-------------|
| UNION | Removed | Slower (sorts) |
| UNION ALL | Kept | Faster |

### Code Example
```sql
-- UNION: Remove duplicates
SELECT city FROM customers
UNION
SELECT city FROM suppliers;

-- UNION ALL: Keep all rows
SELECT 'train' AS dataset, * FROM train_data
UNION ALL
SELECT 'test' AS dataset, * FROM test_data;
```

### Best Practice
Use UNION ALL unless you specifically need deduplication - it's more efficient.

---

## Question 6

**How do you calculate running totals and moving averages in SQL?**

### Code Examples
```sql
-- Running total
SELECT 
    date,
    sales,
    SUM(sales) OVER (ORDER BY date) AS running_total
FROM daily_sales;

-- 7-day moving average
SELECT 
    date,
    sales,
    AVG(sales) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM daily_sales;

-- Running total per customer
SELECT 
    customer_id,
    transaction_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date
    ) AS cumulative_spent
FROM transactions;
```

### ML Applications
- Cumulative features (lifetime value)
- Smoothed features (reduce noise)
- Trend detection

