"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Source seeking toolbox -
"""

import numpy as np
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------
# Mathematical tools
# ----------------------------------------------------------------------

# Affine transformation
M_scale = lambda s, n=2: np.eye(n)*s

# Affine transformation
M_scale = lambda s, n=2: np.eye(n)*s

# Rotation matrix
M_rot = lambda psi: np.array([[np.cos(psi), -np.sin(psi)], \
                              [np.sin(psi),  np.cos(psi)]])

# Angle between two vectors (matrix computation)
def angle_of_vectors(A,B):
    cosTh = np.sum(A*B, axis=1)
    sinTh = np.cross(A,B, axis=1)
    theta = np.arctan2(sinTh,cosTh)
    return theta

"""\
- Funtion to compute L_sigma -
"""
def L_sigma(X, sigma, denom=None):
    # X: (N x 2) matrix of agents position from the centroid
    # sigma: (N) vector of simgma_values on each row of X

    N = X.shape[0]
    l_sigma_hat = sigma[:,None].T @ X
    if denom == None:
        x_norms = np.zeros((N))
        for i in range(N):
            x_norms[i] = X[i,:] @ X[i,:].T
            D_sqr = np.max(x_norms)
        l_sigma_hat = l_sigma_hat / (N * D_sqr)
    else:
        l_sigma_hat = l_sigma_hat/denom
    return l_sigma_hat.flatten()

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

"""
- Check if the dimensions are correct and adapt the input to 3D (N x 3) -
"""
def filter_X_3D(X):
    if type(X) == list:
        return np.array([[X]])
    elif len(X.shape) < 2:
        return np.array([X])
    else:
        return X
  
"""
- Apply alpha to the desired color map -
https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
----------------------------------------------------------------------

When using pcolormesh, directly applying alpha can cause many problems.
The ideal approach is to generate and use a pre-diluted color map on a white background.
"""
def alpha_cmap(cmap, alpha):
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Define the alphas in the range from 0 to 1
    alphas = np.linspace(alpha, alpha, cmap.N)

    # Define the background as white
    BG = np.asarray([1., 1., 1.,])

    # Mix the colors with the background
    for i in range(cmap.N):
        my_cmap[i,:-1] = my_cmap[i,:-1] * alphas[i] + BG * (1.-alphas[i])

    # Create new colormap which mimics the alpha values
    my_cmap = ListedColormap(my_cmap)
    return my_cmap