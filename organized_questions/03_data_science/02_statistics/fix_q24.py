file_path = r'c:\Users\damod\OneDrive\Desktop\Machine-Learning\organized_questions\03_data_science\02_statistics\theory_questions.md'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old_q24 = '''Question 24

**Explain the concepts ofeffect sizeandCohen's d.**

**Answer:** _[To be filled]_'''

new_q24 = '''Question 24

**Explain the concepts ofeffect sizeandCohen's d.**

**Answer:**

### What is Effect Size?
Effect size quantifies the **magnitude** of a phenomenon, independent of sample size.

### Cohen's d Formula
d = (Mean1 - Mean2) / Pooled_SD

### Interpretation Guidelines
| d Value | Effect Size |
|---------|-------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

### Python Implementation
```python
import numpy as np

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

control = np.random.normal(100, 15, 50)
treatment = np.random.normal(108, 15, 50)
d = cohens_d(treatment, control)
print(f"Cohen's d = {d:.2f}")
```

### Interview Tip
Always report effect size alongside p-values for practical significance.'''

if old_q24 in content:
    content = content.replace(old_q24, new_q24)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Q24 replaced successfully')
else:
    print('Not found')
