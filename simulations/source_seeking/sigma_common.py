"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
- Common class for scalar fields -
"""

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from simulations.source_seeking.toolbox import alpha_cmap

# Scalar field color map
MY_CMAP = alpha_cmap(plt.cm.jet, 0.3)

# Matrix Q dot each row of X
Q_prod_xi = lambda Q,X: (Q @ X.T).T

# ----------------------------------------------------------------------
# General class for scalar field
#  * mu: central point of the scalar field.
# ----------------------------------------------------------------------

class sigma:
    def __init__(self, sigma_func, mu=None):
        self.sigma_func = sigma_func
        self.rot = None # Variable to rotate the field from outside
        if mu is None:
            x0 = self.sigma_func.x0 # Ask for help to find minimum
            self.mu = minimize(lambda x: -self.value(np.array([x])), x0).x
        else:
            self.mu = minimize(lambda x: norm(self.grad(np.array([x]))), mu).x
        print(self.mu)

    """
    - Evaluation of the scalar field for a vector of values -
    """
    def value(self, X):
        if self.rot is not None:
            X = Q_prod_xi(self.rot, X-self.mu) + self.mu
        return self.sigma_func.eval(X)

    """
    - Gradient vector of the scalar field for a vector of values -
    """
    def grad(self, X):
        if self.rot is not None:
            X = Q_prod_xi(self.rot, X-self.mu) + self.mu
            grad = self.sigma_func.grad(X)
            return Q_prod_xi(self.rot.T, grad)
        else:
            return self.sigma_func.grad(X)

    """
    - Function to draw a 2D scalar field -
    """
    def draw_2D(self, fig=None, ax=None, xlim=30, ylim=30, cmap=MY_CMAP, n=256, contour_levels=0, contour_lw=0.3, cbar_sw=True):
        if fig == None:
            fig = plt.figure(figsize=(16, 9), dpi=100)
            ax = fig.subplots()
        elif ax == None:
            ax = fig.subplots()

        # Calculate
        x = np.linspace(self.mu[0] - xlim, self.mu[0] + xlim, n)
        y = np.linspace(self.mu[1] - ylim, self.mu[1] + ylim, n)
        X, Y = np.meshgrid(x, y)

        P = np.array([list(X.flatten()), list(Y.flatten())]).T
        Z = self.value(P).reshape(n,n)

        # Draw
        ax.plot(self.mu[0], self.mu[1], "+k")
        color_map = ax.pcolormesh(X, Y, Z, cmap=cmap)

        if cbar_sw:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.05)

            cbar = fig.colorbar(color_map, cax=cax)
            cbar.set_label(label='$\sigma$ [u]', labelpad=10)

        if contour_levels != 0:
            contr_map = ax.contour(X, Y, Z, contour_levels, colors="k", linewidths=contour_lw, linestyles="-", alpha=0.2)
            return color_map, contr_map
        else:
            return color_map,

    """
    - Function to draw a 2D scalar field -
    """
    def draw_3D(self, fig=None, ax=None, lim=30, cmap=MY_CMAP, n=30, contour_levels=0, contour_lw=0.3, cbar_sw=True):
        if fig == None:
            fig = plt.figure(figsize=(16, 9), dpi=100)
            ax = fig.subplots(projection="3d")
        elif ax == None:
            ax = fig.subplots(projection="3d")

        ax.set_xlabel(r"$X$")
        ax.set_ylabel(r"$Y$")  
        ax.set_zlabel(r"$Z$")

        ax.set_proj_type('ortho')
        ax.set_xlim([-lim,lim])
        ax.set_ylim([-lim,lim])
        ax.set_zlim([-lim,lim])

        # Calculate
        x = np.linspace(self.mu[0] - lim, self.mu[0] + lim, n)
        y = np.linspace(self.mu[1] - lim, self.mu[1] + lim, n)
        z = np.linspace(self.mu[2] - lim, self.mu[2] + lim, n)

        Xx, Yx, Zx = np.meshgrid(self.mu[0], y, z)
        Xy, Yy, Zy = np.meshgrid(x, self.mu[1], z)
        Xz, Yz, Zz = np.meshgrid(x, y, self.mu[2])

        Px = np.array([list(Xx.flatten()), list(Yx.flatten()), list(Zx.flatten())]).T
        Py = np.array([list(Xy.flatten()), list(Yy.flatten()), list(Zy.flatten())]).T
        Pz = np.array([list(Xz.flatten()), list(Yz.flatten()), list(Zz.flatten())]).T

        #sigma_x = self.value(Px).reshape(n,n)
        sigma_y = self.value(Py)
        #sigma_z = self.value(Pz).reshape(n,n)

        # Draw
        ax.contourf(Xy[0,:,:], Yy[0,:,:], sigma_y.reshape(n,n), zdir="z", offset=0)


    """\
    - Function to draw the gradient at a point -
    * x: point where the gradient is to be drawn. [x,y]
    * ax: axis to plot on.
    * width: arrow size.
    * scale: scales the length of the arrow (smaller for larger scale values).
    * zorder: overlay order in the plot.
    """
    def draw_grad(self, x, ax, width=0.002, scale=30, zorder=2, alpha=1, ret_arr=True):
        if type(x) == list:
            grad_x = self.grad(np.array(x))[0]
        else:
            grad_x = self.grad(x)[0]
        grad_x_unit = grad_x/norm(grad_x)
        quiver = ax.quiver(x[0], x[1], grad_x_unit[0], grad_x_unit[1],
                            width=width, scale=scale, zorder=zorder, alpha=alpha)
        if ret_arr:
            return quiver
        else:
            return grad_x_unit


    """
    - Funtion to compute and draw L^1 -
    * pc: [x,y] position of the centroid
    * X: (N x 2) matrix of agents position from the centroid
    """
    def draw_L1(self, pc, P):
        N = P.shape[0]
        X = P - pc

        grad_pc = self.grad(np.array(pc))[0]
        l1_sigma_hat = (grad_pc[:,None].T @ X.T) @ X

        x_norms = np.zeros((N))
        for i in range(N):
            x_norms[i] = (X[i,:]) @ X[i,:].T
            D_sqr = np.max(x_norms)

        l1_sigma_hat = l1_sigma_hat / (N * D_sqr)
        return l1_sigma_hat.flatten()