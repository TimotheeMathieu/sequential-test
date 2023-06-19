import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from tqdm import tqdm
from statsmodels.stats.power import tt_ind_solve_power as power_tt
from tqdm import tqdm

import logging
from pathlib import Path
import os

import itertools

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
        name values of 1 and 2 correspond to alpha spending functions which give O’Brien Fleming and
        Pocock type boundaries, respectively. A value of 3 is the power family. Here, the spending function
        is αtφ, where φ must be greater than 0. A value of 4 is the Hwang-Shih-DeCani family, with
        spending function α(1 − e−φt)/(1 − e−φ), where φ cannot be 0.
    sigma: float, default=1
        std of the samples. Unused if student_approx = True
    student_approx: bool, default=False
        if True, use studentized Z-score and student ppf, this is a a heuristic and may have level different from alpha

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
    """

    def __init__(
        self,
        n_groups=None,
        size_group=10,
        alpha=0.05,
        power=0.9,
        name="PK",
        sigma=1,
        student_approx=False,
        boundary=None,
        **kwargs
    ):

        self.alpha = alpha
        self.n = size_group

        if K is None:
            K_ = self._find_K_power(power)
        else:
            K_ = n_groups
            
        self.K = K_
        GST.__init__(self, self.K, self.n, self.alpha)

        self.decision = "accept"
        self.student_approx = student_approx
        self.sigma = sigma  # Only used if student_approx == False
        self.k = 0

    def _find_K_power(self, power):
        # Find K such that the power is at least `power` at a drift of one time the std.
        df = pd.read_csv("power_dataframe.csv")
        df = df.loc[df["name"] == self.bname]
        df = df.loc[np.abs(df["alpha"] - self.alpha) < 1e-5]
        df = df.loc[df["n"] == self.n]
        df = df.sort_values(by="K")
        Ks = df["K"].values
        print(df)
        powers = df["power"].values
        return Ks[np.min(np.where(powers > power))]

    def get_boundary(self, alpha, K):
        df = pd.read_csv("boundaries.csv", sep=";")
        df = df.loc[df["name"] == self.bname]
        df = df.loc[df["K"] == K]
        df = df.loc[np.abs(df["alpha"] - alpha) < 1e-5]  # allow for small approximation
        if len(df) == 0:
            raise NotImplemented("The boundary for the specified values of alpha and K has not been computed")
        else:
            self.boundary = df
        return df

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
            dof = 2 * len(X) - 2 # number of degrees of freedom, use
            threshold = -stats.t.ppf(q=1 - stats.norm.cdf(ck), df=dof)

        self.n_iter = (k + 1) * self.n
        if np.abs(Zk) > threshold:
            self.decision = "reject"
        self.k += 1

    def get_ck(self, k):
        return self.boundary[k]

    def draw_region(self, ax=None):
        """
        Plot the rejection region into the axis ax. If ax is None, create an axis.
        """
        assert not self.student_approx, "region for student is not defined properly"

        x = np.arange(1, self.K + 1) / self.K
        y = self.boundary

        if ax is None:
            fig, ax = plt.subplots()

        if not ax.lines:
            p1 = ax.plot(
                x,
                stats.norm.ppf(self.alpha / 2) * np.ones(self.K),
                "--",
                label="Non seq test",
                alpha=0.7,
            )
            ax.plot(
                x,
                -stats.norm.ppf(self.alpha / 2) * np.ones(self.K),
                "--",
                color=p1[0].get_color(),
                alpha=0.7,
            )

        p2 = ax.plot(x, y, "o-", label="Seq test " + self.bname, alpha=0.7)
        ax.plot(x, -np.array(y), "o-", color=p2[0].get_color(), alpha=0.7)

        plt.legend()
        plt.xlabel("portion of sample size")
        plt.ylabel("Z-stat")
        return y

