import numpy as np
from sequential_test import GGST
from tqdm import tqdm
import pandas as pd


def test_level():
    n_mc = 5000 # number of monte-carlo simulations used
    n_groups = 5
    size_group = 5
    alpha = 0.05

    decisions = []
    X_all = np.random.normal(size = [n_mc, n_groups, size_group])
    Y_all = np.random.normal(size = [n_mc, n_groups, size_group])
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
        decisions.append(test.decision=="reject")

    assert np.mean(decisions) < 0.06


def test_power():
    n_mc = 5000 # number of monte-carlo simulations used
    size_group = 10
    drift = 0.5
    alpha = 0.05

    decisions = []
    test = GGST(size_group=size_group, alpha = alpha, power=0.9, drift=drift)
    n_groups = test.n_groups
    X_all = np.random.normal(size = [n_mc, n_groups, size_group])
    Y_all = np.random.normal(size = [n_mc, n_groups, size_group])
    for simu in range(n_mc):
        X = np.array([])
        Y = np.array([])
        test.reset() # reset test to zero
        for k in range(n_groups):
            X = np.append(X_all[simu, k], X)+drift
            Y = np.append(Y_all[simu, k], Y)
            test.step(X, Y)
            if test.decision == "reject":
                break
        decisions.append(test.decision=="reject")

    assert np.mean(decisions) > 0.89

