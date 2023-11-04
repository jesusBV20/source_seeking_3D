"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
- Scalar field functions -
"""

import numpy as np

from simulations.source_seeking.toolbox import filter_X_3D

# Matrix Q dot each row of X
Q_prod_xi = lambda Q,X: (Q @ X.T).T

# Exponential function with quadratic form: exp(r) = e^((r - mu)^t @ Q @ (r - mu)).
exp = lambda X,Q,mu: np.exp(np.sum((X - mu) * Q_prod_xi(Q,X - mu), axis=1))

# ----------------------------------------------------------------------
# Scalar fields used in simulations
# (all these classes needs eval and grad functions)
# ----------------------------------------------------------------------

"""\
Gaussian function (3D).
  * x0: center of the Gaussian.
  * max_intensity: maximum intensity of the Gaussian.
  * dev: models the width of the Gaussian.
"""
class sigma_gauss:
  def __init__(self, x0=[0,0,0], max_intensity=100, dev=10, S=None, R=None):
    self.n = len(x0)
    self.max_intensity = max_intensity
    self.dev = dev

    # Variables required by the sigma class
    self.x0  = np.array(x0)
    # ---

    if S is None:
      S = -np.eye(self.n)
    if R is None:
      R = np.eye(self.n)
    self.Q = R.T@S@R/(2*self.dev**2)

  # Functions required by the sigma class
  def eval(self, X):
    X = filter_X_3D(X)
    print(X)
    sigma = self.max_intensity * exp(X,self.Q,self.x0) / np.sqrt(2*np.pi*self.dev**2)
    return sigma

  def grad(self, X):
    X = filter_X_3D(X)
    return Q_prod_xi(self.Q,X-self.x0) * self.eval(X)
  # ---