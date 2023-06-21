import numpy as np
import subprocess
from tqdm import tqdm

Ks = np.arange(1, 21)
alphas = np.geomspace(1e-4, 0.3,num=50) # several values of alpha
alphas = np.hstack([alphas, np.array([0.001, 0.01, 0.05, 0.1])]) # add usual values of alpha
names = ["PK", "OF"]

for K in tqdm(Ks):
    for alpha in tqdm(alphas):
        for name in names:
            subprocess.run(["Rscript", "boundary.R", "-n", str(name),"-a",str(alpha), "-l",str(K)])
