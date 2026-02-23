# 04_pandas — Exhaustive Audit Report

**Generated:** 2026-02-20
**Source:** Pandas.md (45 questions)
**Category files audited:** theory_questions.md, general_questions.md, coding_questions.md, scenario_based_questions.md
**Cross-checked against:** PythonMl.md, SQL in ML.md, NumPy.md, MATLAB.md

---

## Summary

| Metric | Count |
|--------|-------|
| Total questions in Pandas.md | 45 |
| Answered questions across all category files | 45 (unique) |
| Questions with full answers | 45 |
| WRONG topic questions (misplaced) | **0** |
| Missing questions (in Pandas.md but not in any category file) | **0** |
| Stub entries ("Answer to be added") | **19** |
| Duplicate entries (same question answered + stub in same or other file) | **16** |

**Verdict:** All questions are correctly attributed to Pandas. No misplaced questions from other topics. However, there are 19 stubs (16 of which are duplicates of already-answered questions in the same or other files).

---

## 1. theory_questions.md (33 entries: 17 answered + 16 stubs)

### Answered Questions

| # | Question Title | Pandas.md # | Status |
|---|---------------|-------------|--------|
| 1 | What is Pandas in Python and why is it used for data analysis? | Q1 | ✅ CORRECT |
| 2 | Explain the difference between a Series and a DataFrame in Pandas | Q2 | ✅ CORRECT |
| 3 | What are Pandas indexes, and how are they used? | Q4 | ✅ CORRECT |
| 4 | Explain the concept of data alignment and broadcasting in Pandas | Q7 | ✅ CORRECT |
| 5 | What is data slicing vs filtering in Pandas? | Q8 | ✅ CORRECT |
| 6 | Describe how joining and merging data works in Pandas | Q9 | ✅ CORRECT |
| 7 | How do you convert categorical data to numerical? | Q12 | ✅ CORRECT |
| 8 | What is the purpose of the apply() function in Pandas? | Q15 | ✅ CORRECT |
| 9 | Explain astype, to_numeric, and pd.to_datetime | Q17 | ✅ CORRECT |
| 10 | Explain data ranking in Pandas | Q20 | ✅ CORRECT |
| 11 | What is a crosstab in Pandas? | Q22 | ✅ CORRECT |
| 12 | How do you perform a MultiIndex query? | Q23 | ✅ CORRECT |
| 13 | How do you export a DataFrame to different file formats? | Q27 | ✅ CORRECT |
| 14 | How to handle larger-than-memory data with Dask or Modin? | Q30 | ✅ CORRECT |
| 15 | How to use Pandas to preprocess data for ML? | Q37 | ✅ CORRECT |
| 16 | What are some strategies for optimizing Pandas code performance? | Q43 | ✅ CORRECT |
| 17 | Explain the importance of categorical data types | Q45 | ✅ CORRECT |

### Stub Entries (answer says "Answer to be added")

| # | Stub Title | Pandas.md # | Duplicate of |
|---|-----------|-------------|--------------|
| 18 | Discuss the use of groupby in Pandas and provide an example | Q6 | scenario_based Q1 (answered) |
| 19 | What is data slicing in Pandas, and how does it differ from filtering? | Q8 | theory Q5 (answered) |
| 20 | Describe how you would convert categorical data into numeric format | Q12 | theory Q7 (answered) |
| 21 | Show how to apply conditional logic to columns using the where() method | Q14 | general Q6 (answered) |
| 22 | Explain the usage and differences between astype, to_numeric, and pd.to_datetime | Q17 | theory Q9 (answered) |
| 23 | Discuss how to deal with time series data in Pandas | Q18 | scenario_based Q2 (answered) |
| 24 | Explain the different types of data ranking available in Pandas | Q20 | theory Q10 (answered) |
| 25 | What is a crosstab in Pandas, and when would you use it? | Q22 | theory Q11 (answered) |
| 26 | Describe how to perform a multi-index query on a DataFrame | Q23 | theory Q12 (answered) |
| 27 | Provide an example of how to normalize data within a DataFrame column | Q24 | general Q10 (answered) |
| 28 | Show how to create simple plots from a DataFrame using Pandas' visualization tools | Q25 | general Q11 (answered) |
| 29 | Discuss how Pandas integrates with Matplotlib and Seaborn for data visualization | Q26 | scenario_based Q3 (answered) |
| 30 | Explain how you would export a DataFrame to different file formats for reporting purposes | Q27 | theory Q13 (answered) |
| 31 | How does one use Dask or Modin to handle larger-than-memory data in Pandas? | Q30 | theory Q14 (answered) |
| 32 | Discuss the advantages of vectorized operations in Pandas over iteration | Q41 | scenario_based Q5 (answered) |
| 33 | Explain the importance of using categorical data types, especially when working with a large number of unique values | Q45 | theory Q17 (answered) |

> **Note:** All 16 stubs are duplicates of questions already answered elsewhere in the category files.

---

## 2. general_questions.md (15 entries: all answered)

| # | Question Title | Pandas.md # | Status |
|---|---------------|-------------|--------|
| 1 | How can you read and write data from and to a CSV file in Pandas? | Q3 | ✅ CORRECT |
| 2 | How do you handle missing data in a DataFrame? | Q5 | ✅ CORRECT |
| 3 | How do you apply a function to all elements in a DataFrame column? | Q10 | ✅ CORRECT |
| 4 | Demonstrate how to handle duplicate rows in a DataFrame | Q11 | ✅ CORRECT |
| 5 | How can you pivot data in a DataFrame? | Q13 | ✅ CORRECT |
| 6 | How do you apply conditional logic using where()? | Q14 | ✅ CORRECT |
| 7 | How do you reshape a DataFrame using stack and unstack methods? | Q16 | ✅ CORRECT |
| 8 | How can you perform statistical aggregation on DataFrame groups? | Q19 | ✅ CORRECT |
| 9 | How do you use window functions in Pandas for running calculations? | Q21 | ✅ CORRECT |
| 10 | How do you normalize data within a DataFrame column? | Q24 | ✅ CORRECT |
| 11 | How do you create simple plots from a DataFrame? | Q25 | ✅ CORRECT |
| 12 | What techniques can you use to improve the performance of Pandas operations? | Q28 | ✅ CORRECT |
| 13 | Compare and contrast the memory usage in Pandas for categories vs. objects | Q29 | ✅ CORRECT |
| 14 | How do you manage memory usage when working with large DataFrames? | Q42 | ✅ CORRECT |
| 15 | How can you use chunking to process large CSV files with Pandas? | Q44 | ✅ CORRECT |

> **No issues.** All 15 questions are answered and correctly mapped to Pandas.md.

---

## 3. coding_questions.md (9 entries: 8 answered + 1 stub)

| # | Question Title | Pandas.md # | Status |
|---|---------------|-------------|--------|
| 1 | Write a Pandas script to filter rows based on a column's value being higher than a specified percentile | Q31 | ✅ CORRECT |
| 2 | Code a function that concatenates two DataFrames and handles overlapping indices correctly | Q32 | ✅ CORRECT |
| 3 | Implement a data cleaning function that drops columns with more than 50% missing values and fills remaining with column mean | Q33 | ✅ CORRECT |
| 4 | Create a Pandas pipeline that ingests, processes, and summarizes time-series data from a CSV file | Q34 | ✅ CORRECT |
| 5 | Write a Python function that computes the correlation matrix and visualizes it using Seaborn's heatmap | Q35 | ✅ CORRECT |
| 6 | Given a DataFrame with multiple datetime columns, create a new column with the earliest datetime | Q36 | ✅ CORRECT |
| 7 | Develop a routine to detect and flag rows deviating by more than three standard deviations from the mean | Q38 | ✅ CORRECT |
| 8 | Outline how to merge multiple time series datasets effectively in Pandas | Q40 | ✅ CORRECT |

### Stub Entry

| # | Stub Title | Pandas.md # | Duplicate of |
|---|-----------|-------------|--------------|
| 9 | If you have a DataFrame with multiple datetime columns, detail how you would create a new column combining them into the earliest datetime | Q36 | coding Q6 (answered) |

---

## 4. scenario_based_questions.md (7 entries: 5 answered + 2 stubs)

| # | Question Title | Pandas.md # | Status |
|---|---------------|-------------|--------|
| 1 | Discuss the use of groupby in Pandas and provide an example | Q6 | ✅ CORRECT |
| 2 | Discuss how to deal with time series data in Pandas | Q18 | ✅ CORRECT |
| 3 | Discuss how Pandas integrates with Matplotlib and Seaborn for visualization | Q26 | ✅ CORRECT |
| 4 | How would you use Pandas to prepare and clean e-commerce sales data for insights into customer purchasing patterns? | Q39 | ✅ CORRECT |
| 5 | Discuss the advantages of vectorized operations in Pandas over iteration | Q41 | ✅ CORRECT |

### Stub Entries

| # | Stub Title | Pandas.md # | Duplicate of |
|---|-----------|-------------|--------------|
| S1 | Describe how you could use Pandas to preprocess data for a machine learning model | Q37 | theory Q15 (answered) |
| S2 | How would you use Pandas to prepare and clean ecommerce sales data for better insight into customer purchasing patterns? | Q39 | scenario Q4 (answered) |

---

## 5. Missing Questions from Pandas.md

**None.** All 45 questions from Pandas.md appear in at least one category file with a full answer.

### Full Coverage Map

| Pandas.md Q# | Answered In | Also Stub In |
|--------------|-------------|--------------|
| Q1 | theory Q1 | — |
| Q2 | theory Q2 | — |
| Q3 | general Q1 | — |
| Q4 | theory Q3 | — |
| Q5 | general Q2 | — |
| Q6 | scenario Q1 | theory Q18 |
| Q7 | theory Q4 | — |
| Q8 | theory Q5 | theory Q19 |
| Q9 | theory Q6 | — |
| Q10 | general Q3 | — |
| Q11 | general Q4 | — |
| Q12 | theory Q7 | theory Q20 |
| Q13 | general Q5 | — |
| Q14 | general Q6 | theory Q21 |
| Q15 | theory Q8 | — |
| Q16 | general Q7 | — |
| Q17 | theory Q9 | theory Q22 |
| Q18 | scenario Q2 | theory Q23 |
| Q19 | general Q8 | — |
| Q20 | theory Q10 | theory Q24 |
| Q21 | general Q9 | — |
| Q22 | theory Q11 | theory Q25 |
| Q23 | theory Q12 | theory Q26 |
| Q24 | general Q10 | theory Q27 |
| Q25 | general Q11 | theory Q28 |
| Q26 | scenario Q3 | theory Q29 |
| Q27 | theory Q13 | theory Q30 |
| Q28 | general Q12 | — |
| Q29 | general Q13 | — |
| Q30 | theory Q14 | theory Q31 |
| Q31 | coding Q1 | — |
| Q32 | coding Q2 | — |
| Q33 | coding Q3 | — |
| Q34 | coding Q4 | — |
| Q35 | coding Q5 | — |
| Q36 | coding Q6 | coding Q9 |
| Q37 | theory Q15 | scenario S1 |
| Q38 | coding Q7 | — |
| Q39 | scenario Q4 | scenario S2 |
| Q40 | coding Q8 | — |
| Q41 | scenario Q5 | theory Q32 |
| Q42 | general Q14 | — |
| Q43 | theory Q16 | — |
| Q44 | general Q15 | — |
| Q45 | theory Q17 | theory Q33 |

---

## 6. Cross-Topic Check

All questions in the four category files were verified against PythonMl.md, SQL in ML.md, NumPy.md, and MATLAB.md.

**Result: 0 questions belong to another topic.** Every question is specific to Pandas.

---

## 7. Issues Summary

### Issue 1: 16 Duplicate Stubs in theory_questions.md
theory_questions.md contains 16 stub entries (Q18–Q33) that are exact duplicates of questions already answered with full content in other files (or even within theory_questions.md itself). These should be **removed** to avoid confusion.

### Issue 2: 1 Duplicate Stub in coding_questions.md
coding_questions.md Q9 is a stub duplicate of coding_questions.md Q6 (both map to Pandas.md Q36). Should be **removed**.

### Issue 3: 2 Duplicate Stubs in scenario_based_questions.md
scenario_based_questions.md has 2 trailing stubs (S1, S2) that duplicate scenario Q4 and theory Q15. Should be **removed**.

### Issue 4: Category Placement Inconsistencies
Some questions are answered in a different category than their Pandas.md section suggests:
- Pandas.md Q6 ("groupby") is in section "Pandas Fundamentals" but answered in **scenario_based** (not theory/general)
- Pandas.md Q12 ("categorical to numeric") is in section "Data Manipulation and Cleaning" but answered in **theory**
- Pandas.md Q37 ("preprocess for ML") is in section "Scenario-Based" but answered in **theory**
- Pandas.md Q41 ("vectorized operations") is in section "Advanced Topics" but answered in **scenario_based**

These are minor classification preferences, not errors.

---

## 8. Recommendations

1. **Remove all 19 stubs** — they duplicate already-answered questions and add no value
2. **No content is missing** — all 45 Pandas.md questions have full answers somewhere
3. **No misplaced questions** — everything is Pandas-specific
4. **Consider reorganizing** the 4 category-placement inconsistencies noted above if strict alignment with Pandas.md sections is desired
