# Example of using the t-test with click conversion data
# Note we are purposely violating some assumptions of the chi2-test
# We want to see what happens if we apply this test in this situation

import numpy as np
import pandas as pd
from scipy.stats import chi2

# Load the data
data = pd.read_csv("advertisement_clicks.csv")

N = data.shape[0]/2

# Separate the results into two sets of observations
# We are asking "Does one design lead to a greater conversion rate?"
a = data[data['advertisement_id'] == 'A']
b = data[data['advertisement_id'] == 'B']

a = a['action']
b = b['action']

# Create the observation table and find determinant
T = np.array([[np.sum(a == 1), np.sum(a ==0)], [np.sum(b == 1), np.sum(b == 0)]])
det = det = T[0,0]*T[1,1] - T[0,1]*T[1,0]

# Calculate the chi2 and p values
c2 = float(det) / T[0].sum() * det/T[1].sum() * T.sum() / T[:,0].sum() / T[:,1].sum()
p = 1 - chi2.cdf(x=c2, df = 1)

print("p-value:", p)

