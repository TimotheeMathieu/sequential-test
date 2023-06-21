import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from tqdm import tqdm
from tqdm import tqdm

import logging
from pathlib import Path
import os
import itertools


LIB_DIR = os.path.dirname(__file__)

class GST():
    """
    Base class for Group Sequential testing.
    """
    def __init__(self, size_group=5, n_groups=5, alpha=0.05, **kwargs):
        self.size_group = size_group
        self.n_groups = n_groups
        self.alpha = alpha  # level of the test


class GGST(GST):
    """
    Group-Sequential two sample bilateral tests. Data are supposed Gaussian with equal variance.
    With K equally sized groups. Level should be small.

    Parameters
    ----------

    n_groups: int
        maximum number of interims. If None, use pow to infer it.
    size_group: int
        size of a block. Used if env is None.
    alpha: float, default=0.05
        level of the test
    power: float, default=0.9
        If K is None, power of the test.
    name: str, default="PK"
        name of the boundary used. "PK" for Pocock, "OF" for O'Brien-Fleming, numerical value
        use the lan-demetz level-spending functions.
    sigma: float, default=1
        std of the samples. Unused if student_approx = True (the default).
    drift: float, default=True
        multiplicative factor of the std used for the drift when computing number of interims to achieve a given
        power when n_groups = None.
    student_approx: bool, default=True
        if True, use studentized Z-score and student ppf, this is a a heuristic and may have level different from alpha.

    Examples
    --------
    >>> X = np.array([])
    >>> Y = np.array([])
    >>> n_groups, size_group = 5,10
    >>> test = GGST(n_group, size_group)
    >>> for k in range(n_group):
    >>>    X_g, Y_g = get_data() # get the data from whatever process you want to test
    >>>    X = np.append(X_g, X)
    >>>    Y = np.append(Y_g, Y)
    >>>    test.step(X, Y)
    >>>    if test.decision == "reject":
    >>>        break
    >>> print("Decision is ", test.decision)

    Remarks
    -------

    If you have to make a large number of tests, do not reinstantiate the test each time instead use test.reset() which is a lot faster.
    """

    def __init__(
        self,
        n_groups=None,
        size_group=10,
        alpha=0.05,
        power=0.9,
        name="PK",
        sigma=1,
        drift=1,
        student_approx=True,
    ):

        self.name = name
        self.alpha = alpha
        self.size_group = size_group
        if n_groups is None:
            n_groups_ = self._find_K_power(power, drift)
        else:
            n_groups_ = n_groups

        GST.__init__(self, n_groups = n_groups_ , size_group = size_group, alpha = alpha)

        self.decision = "accept"
        self.student_approx = student_approx
        self.sigma = sigma  # Only used if student_approx == False
        self.k = 0
        self.get_boundary()

    def _find_K_power(self, power, drift = 1):
        df = pd.read_csv(os.path.join(LIB_DIR,"power_dataframe.csv"))
        df = df.loc[df["name"] == self.name]
        df = df.loc[np.abs(df["alpha"] - self.alpha) < 1e-5]
        df = df.loc[df["size_group"] == self.size_group]
        df = df.loc[df["drift"] == drift]
        df = df.sort_values(by="n_groups")
        Ks = df["n_groups"].values
        powers = df["power"].values
        assert np.sum(powers > power)>0, "Please choose a larger drift or a larger size_group"
        return Ks[np.min(np.where(powers > power))]

    def get_boundary(self):
        
        df = pd.read_csv(os.path.join(LIB_DIR,"boundaries.csv"), sep=";")
        df = df.loc[df["name"] == self.name]
        df = df.loc[df["K"] == self.n_groups]
        
        df = df.loc[np.abs(df["alpha"] - self.alpha) < 1e-5]  # allow for small approximation
        if len(df) == 0:
            raise NotImplementedError("The boundary for the specified values of alpha and K has not been computed")
        else:
            df = df.sort_values("t", axis=0)
            self.boundary = df["up"].values # symmetric boundaries, we need only the upper bound
        return df

    def reset(self):
        self.k = 0
        self.decision = "accept"

    def step(self, X, Y):
        k = self.k

        # Compute the test statistic, with student approximation if necessary
        if not self.student_approx:
            Zk = np.sum(X - Y) / np.sqrt(2 * len(X) * self.sigma**2)
        else:
            Zk = np.sum(X - Y) / np.sqrt(
                len(X) * (np.std(X, ddof=1) ** 2 + np.std(Y, ddof=1) ** 2)
            )

        # value of the threshold for this step in gaussian model
        ck = self.get_ck(k)

        if not self.student_approx:
            threshold = ck
        else:
            dof = 2 * len(X) - 2 # number of degrees of freedom we use
            threshold = -stats.t.ppf(q = 1 - stats.norm.cdf(ck), df=dof)

        self.n_iter = (k + 1) * self.size_group

        if np.abs(Zk) > threshold:
            self.decision = "reject"
        self.k += 1

    def get_ck(self, k):
        return self.boundary[k]

