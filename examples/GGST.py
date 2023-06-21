"""
Example of usage of GGST.
"""

import numpy as np
from sequential_test import GGST
import matplotlib.pyplot as plt

n_groups = 10
size_group = 10

test = GGST(name="OF", n_groups=n_groups, size_group = size_group, alpha = 0.05, student_approx=True)
X = np.array([])
Y = np.array([])
test_stats = np.array([])
for k in range(n_groups):
    X = np.append(np.random.normal(size=size_group), X)
    Y = np.append(np.random.normal(size=size_group)+0.5, Y)
    test.step(X, Y)
    if test.decision == "reject":
        break
print("Final decision is ", test.decision)
print("Number of interim before decition is ", k+1)
