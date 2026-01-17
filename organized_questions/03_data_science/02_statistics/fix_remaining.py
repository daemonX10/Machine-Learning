import re

# Read file
with open('theory_questions.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Q24 - find the exact text around line 830
q24_new = '''## Question 24

**Explain the concepts of effect size and Cohen's d.**

**Answer:**

### Definition
**Effect size** quantifies the magnitude of a phenomenon independent of sample size. Unlike p-values, it tells you HOW BIG the difference is.

### Cohen's d Formula
$$d = \\frac{\\bar{X}_1 - \\bar{X}_2}{s_{pooled}}$$

Where: $s_{pooled} = \\sqrt{\\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$

### Interpretation Guidelines
| Cohen's d | Interpretation |
|-----------|----------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

### Python Implementation
```python
import numpy as np
from scipy import stats

# Two groups
group1 = [85, 90, 88, 92, 87, 91]
group2 = [78, 82, 80, 75, 79, 81]

# Calculate Cohen's d
def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / pooled_std

d = cohens_d(group1, group2)
print(f"Cohen's d = {d:.2f}")
```

### Other Effect Size Measures
| Measure | Use Case | Formula |
|---------|----------|---------|
| r (correlation) | Relationship strength | r itself |
| eta-squared | ANOVA | SS_between / SS_total |
| Odds Ratio | Categorical data | (a*d)/(b*c) |
| Hedges g | Small samples | Bias-corrected d |

### Why Effect Size Matters
- p-values depend on sample size (large n -> small p)
- Effect size is independent of n
- Needed for meta-analyses
- Shows practical significance

### Interview Tip
Say: "Effect size tells you the magnitude of an effect, while p-value only tells you if it exists. A statistically significant result with tiny effect size may be practically meaningless."'''

# Manual search around line 830
lines = content.split('\n')
found_idx = -1
for i, line in enumerate(lines):
    if 'Question 24' in line and i < 850 and i > 800:
        found_idx = i
        print(f"Found Question 24 at line {i}")
        break

if found_idx >= 0:
    # Find the start and end of Q24
    start_idx = found_idx
    # Find Q25
    end_idx = start_idx + 1
    while end_idx < len(lines) and 'Question 25' not in lines[end_idx]:
        end_idx += 1
    
    print(f"Q24 spans lines {start_idx} to {end_idx-1}")
    print(f"Current content:")
    for i in range(start_idx, min(end_idx, start_idx+10)):
        print(f"  {i}: {repr(lines[i][:80])}")
    
    # Replace lines
    new_lines = lines[:start_idx] + q24_new.split('\n') + lines[end_idx:]
    content = '\n'.join(new_lines)
    
    with open('theory_questions.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Q24 replaced and file saved")
else:
    print("Q24 not found in expected location")
