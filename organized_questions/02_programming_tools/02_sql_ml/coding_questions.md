# SQL ML Interview Questions - Coding Questions

## Question 1

**Write a SQL query that joins two tables and retrieves only matching rows.**

### Code
```sql
-- Basic INNER JOIN
SELECT 
    c.customer_id,
    c.name,
    o.order_id,
    o.order_date,
    o.amount
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id;

-- Multiple JOINs with aggregation
SELECT 
    c.customer_id,
    c.name,
    COUNT(o.order_id) AS total_orders,
    SUM(o.amount) AS total_spent
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
HAVING COUNT(o.order_id) > 1;
```

---

## Question 2

**Write a SQL query to pivot rows into columns.**

### Code
```sql
-- Manual PIVOT using CASE (works in all databases)
SELECT 
    customer_id,
    SUM(CASE WHEN category = 'Electronics' THEN amount ELSE 0 END) AS electronics,
    SUM(CASE WHEN category = 'Clothing' THEN amount ELSE 0 END) AS clothing,
    SUM(CASE WHEN category = 'Books' THEN amount ELSE 0 END) AS books,
    SUM(amount) AS total
FROM sales
GROUP BY customer_id;

-- Time series pivot (monthly sales)
SELECT 
    product_id,
    SUM(CASE WHEN MONTH(sale_date) = 1 THEN amount ELSE 0 END) AS jan_sales,
    SUM(CASE WHEN MONTH(sale_date) = 2 THEN amount ELSE 0 END) AS feb_sales,
    SUM(CASE WHEN MONTH(sale_date) = 3 THEN amount ELSE 0 END) AS mar_sales
FROM sales
WHERE YEAR(sale_date) = 2024
GROUP BY product_id;
```

---

## Question 3

**Write a SQL query to find duplicate records.**

### Code
```sql
-- Find duplicates
SELECT 
    email,
    COUNT(*) AS count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- See all duplicate rows
WITH Duplicates AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY email ORDER BY created_at) AS rn
    FROM users
)
SELECT * FROM Duplicates WHERE rn > 1;

-- Delete duplicates (keep first)
DELETE FROM users
WHERE id NOT IN (
    SELECT MIN(id) 
    FROM users 
    GROUP BY email
);
```

---

## Question 4

**Write a SQL query to calculate cumulative sum.**

### Code
```sql
SELECT 
    customer_id,
    transaction_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date
        ROWS UNBOUNDED PRECEDING
    ) AS cumulative_amount
FROM transactions;
```

---

## Question 5

**Write a SQL query to find the second highest salary.**

### Code
```sql
-- Method 1: LIMIT OFFSET
SELECT salary FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 1;

-- Method 2: Subquery
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Method 3: DENSE_RANK (handles ties)
SELECT salary FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rank
    FROM employees
) ranked
WHERE rank = 2;
```

---

## Question 6

**Write a SQL query to calculate year-over-year growth.**

### Code
```sql
WITH YearlySales AS (
    SELECT 
        YEAR(sale_date) AS year,
        SUM(amount) AS total_sales
    FROM sales
    GROUP BY YEAR(sale_date)
)
SELECT 
    year,
    total_sales,
    LAG(total_sales) OVER (ORDER BY year) AS prev_year_sales,
    ROUND(
        (total_sales - LAG(total_sales) OVER (ORDER BY year)) * 100.0 
        / LAG(total_sales) OVER (ORDER BY year), 
        2
    ) AS yoy_growth_pct
FROM YearlySales;
```

---

## Question 7

**Write a SQL query to create ML features from transaction data.**

### Code
```sql
-- Comprehensive feature engineering
SELECT 
    customer_id,
    
    -- Recency
    DATEDIFF(CURRENT_DATE, MAX(transaction_date)) AS days_since_last_purchase,
    
    -- Frequency
    COUNT(*) AS total_transactions,
    COUNT(DISTINCT DATE(transaction_date)) AS unique_purchase_days,
    
    -- Monetary
    SUM(amount) AS total_spent,
    AVG(amount) AS avg_transaction,
    MAX(amount) AS max_transaction,
    MIN(amount) AS min_transaction,
    STDDEV(amount) AS transaction_stddev,
    
    -- Time patterns
    COUNT(CASE WHEN DAYOFWEEK(transaction_date) IN (1,7) THEN 1 END) AS weekend_purchases,
    COUNT(CASE WHEN HOUR(transaction_date) < 12 THEN 1 END) AS morning_purchases,
    
    -- Category diversity
    COUNT(DISTINCT product_category) AS categories_bought,
    
    -- Tenure
    DATEDIFF(MAX(transaction_date), MIN(transaction_date)) AS customer_tenure_days

FROM transactions
WHERE transaction_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
GROUP BY customer_id;
```

---

## Question 8

**Write a SQL query to detect outliers using IQR method.**

### Code
```sql
WITH Stats AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) AS q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) AS q3
    FROM transactions
),
Bounds AS (
    SELECT 
        q1,
        q3,
        q3 - q1 AS iqr,
        q1 - 1.5 * (q3 - q1) AS lower_bound,
        q3 + 1.5 * (q3 - q1) AS upper_bound
    FROM Stats
)
SELECT t.*
FROM transactions t, Bounds b
WHERE t.amount < b.lower_bound OR t.amount > b.upper_bound;
```

---

## Question 9

**Write a SQL query to create a train/test split.**

### Code
```sql
-- Random 80/20 split
SELECT 
    *,
    CASE 
        WHEN RAND() < 0.8 THEN 'train'
        ELSE 'test'
    END AS dataset
FROM features;

-- Reproducible split using modulo
SELECT 
    *,
    CASE 
        WHEN MOD(customer_id, 5) = 0 THEN 'test'  -- 20%
        ELSE 'train'                               -- 80%
    END AS dataset
FROM features;

-- Stratified split
WITH Ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY target_class ORDER BY RAND()) AS rn,
        COUNT(*) OVER (PARTITION BY target_class) AS class_count
    FROM features
)
SELECT *,
    CASE 
        WHEN rn <= class_count * 0.8 THEN 'train'
        ELSE 'test'
    END AS dataset
FROM Ranked;
```

---

## Question 10

**Write a SQL query to one-hot encode a categorical column.**

### Code
```sql
-- One-hot encoding for category column
SELECT 
    id,
    CASE WHEN category = 'A' THEN 1 ELSE 0 END AS category_A,
    CASE WHEN category = 'B' THEN 1 ELSE 0 END AS category_B,
    CASE WHEN category = 'C' THEN 1 ELSE 0 END AS category_C,
    CASE WHEN category NOT IN ('A', 'B', 'C') THEN 1 ELSE 0 END AS category_other
FROM products;
```

