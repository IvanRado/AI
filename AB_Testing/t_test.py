import numpy as np
from scipy import stats

# Generate data with separate means
N = 10
a = np.random.randn(N) + 2 # Centered at 2
b = np.random.randn(N) # Centered at 0

# Calculate distribution variances and pooled variance 
var_a = a.var(ddof = 1)
var_b = b.var(ddof = 1)
s = np.sqrt( (var_a + var_b)/ 2)

# Calculate test statistic, dof and p value
t = (a.mean() - b.mean()) / (s * np.sqrt(2.0/N))
df = 2*N - 2
p = 1- stats.t.cdf(t, df =df)

print("t:\t", t, "p:\t", 2*p)

# Compare with scipy values
t2, p2 = stats.ttest_ind(a,b)
print("t2:\t", t2, "p2:\t", p2)