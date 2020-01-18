# Example of using the t-test with click conversion data
# Note we are purposely violating some assumptions of the t-test
# We want to see what happens if we apply this test in this situation

import numpy as np
import pandas as pd
from scipy import stats

# Load the data
data = pd.read_csv("advertisement_clicks.csv")

N = data.shape[0]/2

# Separate the results into two sets of observations
# We are asking "Does one design lead to a greater conversion rate?"
a = data[data['advertisement_id'] == 'A']
b = data[data['advertisement_id'] == 'B']

a = a['action']
b = b['action']

# Calculate variance and pooled std dev
var_a = np.var(a)
print(var_a)
var_b = np.var(b)
print(var_b)
sp = np.sqrt((var_a + var_b)/2) # I verified each class had an equal # of samples beforehand

# Calculate t, dof, and p-value
t = (a.mean() - b.mean())/(sp * np.sqrt(2.0/N))
df = 2*N - 2
p = (1-stats.t.cdf(np.abs(t), df = df))*2

print("t value:", t)
print("p-value:", p)

t, p = stats.ttest_ind(a, b)
print("Built in Scipy:")
print("t value:", t)
print("p-value:", p)

# Welch's t-test:
t, p = stats.ttest_ind(a, b, equal_var=False)
print("Welch's t-test:")
print("t:\t", t, "p:\t", p)

# Welch's t-test manual:
N1 = len(a)
s1_sq = a.var()
N2 = len(b)
s2_sq = b.var()
t = (a.mean() - b.mean()) / np.sqrt(s1_sq / N1 + s2_sq / N2)

nu1 = N1 - 1
nu2 = N2 - 1
df = (s1_sq / N1 + s2_sq / N2)**2 / ( (s1_sq*s1_sq) / (N1*N1 * nu1) + (s2_sq*s2_sq) / (N2*N2 * nu2) )
p = (1 - stats.t.cdf(np.abs(t), df=df))*2
print("Manual Welch t-test")
print("t:\t", t, "p:\t", p)