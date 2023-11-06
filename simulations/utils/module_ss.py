"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
- Source seeking module -
"""

import numpy as np

class module_ss:
    def __init__(self, simulation_engine, sigma_field):
        self.sim_eng = simulation_engine
        self.sigma_field = sigma_field

        n = self.sim_eng.p.shape[1]
        self.l_sigma_hat = np.zeros(n)
        self.l_sigma_hat_norm = np.zeros(n)
        
        # Make the first computation of L_sigma
        self.compute_Lsigma()
    
    """\
    Funtion to compute L_sigma.
    * X: (n_agents x n) matrix of agents position from the centroid
    * sigma: (n_agents) vector of simgma_values on each row of X
    """
    def compute_Lsigma(self):
        X = self.sim_eng.p - self.sim_eng.get_pc()
        sigma = self.sigma_field.value(self.sim_eng.p)
        n_agents = X.shape[0]

        # Compute max distance from centroid
        d = np.linalg.norm(X, axis=1)
        D_sqr = np.max(d)

        # Compute l_sigma
        l_sigma_hat = sigma[:,None].T @ X / (n_agents * D_sqr)
        
        # Normalize l_sigma
        norm = np.linalg.norm(l_sigma_hat)
        if norm != 0:
            l_sigma_hat_norm = l_sigma_hat / norm
        else:
            l_sigma_hat_norm = np.zeros(X.shape[1])

        
        self.l_sigma_hat = l_sigma_hat.flatten()
        self.l_sigma_hat_norm = l_sigma_hat_norm.flatten()