# SQL ML Interview Questions - Theory Questions

## Question 1

**What are the different types of JOIN operations in SQL?**

### Definition
JOINs combine rows from two or more tables based on related columns. The JOIN type determines which rows appear in the result set.

### Types of JOINs

| JOIN Type | Description | Returns |
|-----------|-------------|---------|
| INNER JOIN | Match in both tables | Only matching rows |
| LEFT JOIN | All from left + matches from right | All left, NULLs for non-matches |
| RIGHT JOIN | All from right + matches from left | All right, NULLs for non-matches |
| FULL OUTER JOIN | All from both tables | All rows, NULLs where no match |
| CROSS JOIN | Cartesian product | Every combination |
| SELF JOIN | Table joined with itself | Hierarchical/comparative queries |

### Code Examples
```sql
-- INNER JOIN: Customers with orders only
SELECT c.name, o.order_date
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id;

-- LEFT JOIN: All customers, including those without orders
SELECT c.name, o.order_date
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id;

-- Find customers with NO orders
SELECT c.name
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.order_id IS NULL;
```

### ML Use Cases
- **INNER**: Create feature matrix with complete data only
- **LEFT**: Find inactive users for churn prediction
- **CROSS**: Generate all user-product pairs for recommendations

---

## Question 2

**Explain the difference between WHERE and HAVING clauses.**

### Definition
- **WHERE**: Filters individual rows BEFORE grouping
- **HAVING**: Filters groups AFTER aggregation

### SQL Execution Order
```
1. FROM / JOIN
2. WHERE        ← Row filtering (no aggregates)
3. GROUP BY
4. HAVING       ← Group filtering (aggregates allowed)
5. SELECT
6. ORDER BY
7. LIMIT
```

### Code Example
```sql
-- Find customers with >5 orders where each order was >$10
SELECT 
    customer_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total_spent
FROM orders
WHERE amount > 10              -- Filter rows first
  AND order_date >= '2024-01-01'
GROUP BY customer_id
HAVING COUNT(*) > 5;           -- Filter groups after
```

### Best Practice
**Filter early with WHERE** - reduces data before aggregation, improving performance.

---

## Question 3

**Describe a subquery and its typical use case.**

### Definition
A subquery (inner query) is a SELECT statement nested inside another query.

### Types

| Type | Runs | Description |
|------|------|-------------|
| Non-correlated | Once | Independent of outer query |
| Correlated | Per row | References outer query |

### Code Examples
```sql
-- Non-correlated: Find employees in Engineering dept
SELECT name FROM employees
WHERE dept_id IN (
    SELECT id FROM departments WHERE name = 'Engineering'
);

-- Correlated: Each employee vs their dept average
SELECT name, salary,
    (SELECT AVG(salary) FROM employees e2 
     WHERE e2.dept_id = e1.dept_id) AS dept_avg
FROM employees e1;
```

### Performance Tip
Correlated subqueries run per row - consider rewriting as JOINs for better performance.

---

## Question 4

**What are SQL Window Functions and how are they used for ML feature engineering?**

### Definition
Window functions perform calculations across a set of rows related to the current row, without collapsing rows like GROUP BY.

### Syntax
```sql
function() OVER (
    PARTITION BY column    -- Groups (keeps rows)
    ORDER BY column        -- Order within partition
)
```

### Key Window Functions

| Function | Purpose | ML Use Case |
|----------|---------|-------------|
| LAG(col, n) | Previous row value | Previous purchase amount |
| LEAD(col, n) | Next row value | Target variable creation |
| ROW_NUMBER() | Sequential rank | Deduplication |
| RANK() | Rank with gaps | Rank within category |
| SUM() OVER | Running total | Cumulative spending |
| AVG() OVER | Moving average | Smoothed features |

### Code Example
```sql
SELECT
    customer_id,
    transaction_date,
    amount,
    
    -- Previous transaction
    LAG(amount, 1) OVER (
        PARTITION BY customer_id ORDER BY transaction_date
    ) AS prev_amount,
    
    -- Running total
    SUM(amount) OVER (
        PARTITION BY customer_id ORDER BY transaction_date
    ) AS cumulative_spent,
    
    -- Rank by amount
    RANK() OVER (
        PARTITION BY customer_id ORDER BY amount DESC
    ) AS amount_rank

FROM transactions;
```

---

## Question 5

**Explain how to discretize a continuous variable in SQL.**

### Definition
Discretization (binning) converts continuous variables into categorical by grouping values into intervals.

### Two Approaches

```sql
-- Equal-width: Custom bins with CASE
SELECT 
    user_id,
    age,
    CASE 
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 50 THEN '35-49'
        ELSE '50+'
    END AS age_group
FROM users;

-- Equal-frequency: Quartiles with NTILE
SELECT 
    user_id,
    spending,
    NTILE(4) OVER (ORDER BY spending) AS spending_quartile
FROM user_stats;
-- Returns 1 (lowest 25%), 2, 3, 4 (highest 25%)
```

### When to Use Each
- **CASE**: Domain-specific bins (age groups, income brackets)
- **NTILE**: Statistical bins (percentiles, deciles)

---

## Question 6

**Describe SQL techniques to perform data sampling.**

### Sampling Methods

```sql
-- Method 1: Random sample (slow for large tables)
SELECT * FROM users
ORDER BY RAND()
LIMIT 1000;

-- Method 2: TABLESAMPLE (fast, database-specific)
SELECT * FROM users TABLESAMPLE BERNOULLI(10);  -- 10% sample

-- Method 3: Stratified sampling
WITH Ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY customer_segment ORDER BY RAND()
        ) AS rn
    FROM customers
)
SELECT * FROM Ranked WHERE rn <= 100;  -- 100 from each segment
```

### Interview Tip
For ML, stratified sampling is critical for imbalanced datasets.

