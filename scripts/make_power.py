"""
Estimation of power of Gaussian GST using Monte-Carlo methods.
We choose the number of simulations such that the error is lower than 0.5% with high probability.

i.e. n_mc was chosen with

from scipy import stats
error = 1
n = 1
while error > 0.005:
    n += 10
    error = (stats.binom( n, 1/2).ppf(0.95)-n/2)/n # 1/2 is the worst case in term of variance for bernoulli
    print(n, error)
print(n)

"""

import numpy as np
from sequential_test import GGST
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
import itertools


drifts = [0.1,0.5,1,2,3] # multiples of std for the drift

size_groups = np.arange(1, 51)
Ks = np.arange(1, 21)
alphas = [0.001, 0.01, 0.05, 0.1] # Usual values for alpha

names = ["OF", "PK"]

n_mc = 27000 # number of monte-carlo simulations used

def make_one_size(size_group, n_groups, name, alpha):
    df = pd.DataFrame()
    for drift in drifts:
        decisions = []
        X_all = np.random.normal(size = [n_mc, n_groups, size_group])
        Y_all = np.random.normal(size = [n_mc, n_groups, size_group]) + drift
        test = GGST(n_groups=n_groups, size_group = size_group, alpha = alpha)
        for simu in range(n_mc):
            X = np.array([])
            Y = np.array([])
            test.reset() # reset test to zero
            for k in range(n_groups):
                X = np.append(X_all[simu, k], X)
                Y = np.append(Y_all[simu, k], Y)
                test.step(X, Y)
                if test.decision == "reject":
                    break
            decisions.append(test.decision)
        df = pd.concat([df, pd.DataFrame({ "size_group":[size_group],
                                           "n_groups":[n_groups],
                                           "alpha":[alpha],
                                           "name":[name],
                                           "drift":[drift],
                                           "power":[np.mean(np.array(decisions) == "reject")]})], ignore_index = True)
    return df

dfs = Parallel(n_jobs=6)(delayed(make_one_size)(size_group, n_groups, name, alpha) for n_groups, size_group, name, alpha in tqdm(itertools.product(Ks,size_groups,names, alphas), total = len(Ks)*len(size_groups)*len(names)*len(alphas)))

df = pd.concat(dfs, ignore_index = True)
df.to_csv("power_dataframe.csv")

