"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
- Common class for scalar fields -
"""

import numpy as np
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
            self.mu = minimize(lambda x: np.linalg.norm(self.grad(np.array([x]))), mu).x

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
    - Function to draw a 3D scalar field -
    """
    def draw_3D(self, fig=None, ax=None, lim=30, cmap=MY_CMAP, n=30, contour_levels=30, contour_lw=0.15, offsets=[-1,1,-1]):
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
        x = np.linspace(- lim, lim, n)
        y = np.linspace(- lim, lim, n)
        z = np.linspace(- lim, lim, n)
 
        Yx, Zx, Xx = np.meshgrid(y, z, self.mu[0])
        Xy, Zy, Yy = np.meshgrid(x, z, self.mu[1])
        Xz, Yz, Zz = np.meshgrid(x, y, self.mu[2])

        Px = np.array([list(Xx.flatten()), list(Yx.flatten()), list(Zx.flatten())]).T
        Py = np.array([list(Xy.flatten()), list(Yy.flatten()), list(Zy.flatten())]).T
        Pz = np.array([list(Xz.flatten()), list(Yz.flatten()), list(Zz.flatten())]).T

        sigma_x = self.value(Px).reshape(n,n)
        sigma_y = self.value(Py).reshape(n,n)
        sigma_z = self.value(Pz).reshape(n,n)

        # Draw colormaps
        kw_cf = {"levels":contour_levels, "cmap":cmap, "alpha":0.5}
        kw_c  = {"levels":contour_levels, "colors":"k", "linewidths":contour_lw, 
                 "linestyles":"-", "alpha":0.8}
        
        ax.contourf(sigma_x, Yx[:,:,0], Zx[:,:,0], **kw_cf, zdir="x", offset=offsets[0]*lim)
        ax.contour(sigma_x, Yx[:,:,0], Zx[:,:,0], **kw_c, zdir="x", offset=offsets[0]*lim)
        
        ax.contourf(Xy[:,:,0], sigma_y, Zy[:,:,0], **kw_cf, zdir="y", offset=offsets[1]*lim)
        ax.contour(Xy[:,:,0], sigma_y, Zy[:,:,0], **kw_c, zdir="y", offset=offsets[1]*lim)
        
        ax.contourf(Xz[:,:,0], Yz[:,:,0], sigma_z, **kw_cf, zdir="z", offset=offsets[2]*lim)
        ax.contour(Xz[:,:,0], Yz[:,:,0], sigma_z, **kw_c, zdir="z", offset=offsets[2]*lim)
        
        # Draw the center of the distribution
        x0 = self.sigma_func.x0
        ax.plot(x0[0], x0[1], x0[2], "xk")
        ax.plot(x0[0], x0[1], x0[2], "ok", alpha=0.2)

        ax.plot(x0[1], x0[2], "xk", zdir="x", zs= offsets[0]*lim)
        ax.plot(x0[0], x0[2], "xk", zdir="y", zs= offsets[1]*lim)
        ax.plot(x0[0], x0[1], "xk", zdir="z", zs= offsets[2]*lim)

        ax.plot([x0[0], offsets[0]*lim], [x0[1],x0[1]], [x0[2],x0[2]], "-k", lw=0.5, alpha=0.5)
        ax.plot([x0[0],x0[0]], [x0[1], offsets[1]*lim], [x0[2],x0[2]], "-k", lw=0.5, alpha=0.5)
        ax.plot([x0[0],x0[0]], [x0[1],x0[1]], [x0[2], offsets[2]*lim], "-k", lw=0.5, alpha=0.5)

    """\
    - Function to draw the gradient at a point -
    * x: point where the gradient is to be drawn. [x,y,z]
    * ax: axis to plot on.
    * length: arrow length.
    * zorder: overlay order in the plot.
    """
    def draw_grad_3D(self, x, ax, length=1, zorder=2, alpha=0.8, lim=50):
        if type(x) == list:
            grad_x = self.grad(np.array(x))[0]
        else:
            grad_x = self.grad(x)[0]
        grad_x_unit = grad_x/np.linalg.norm(grad_x)

        # 2D projection quivers
        ax.quiver(x[0], x[1], -lim, grad_x_unit[0], grad_x_unit[1], 0,
                  color="k", length=0.7*length, normalize=True, alpha=alpha, zorder=zorder)
        
        ax.quiver(x[0], lim, x[2], grad_x_unit[0], 0, grad_x_unit[2],
                  color="k", length=0.7*length, normalize=True, alpha=alpha, zorder=zorder)
        
        ax.quiver(-lim, x[1], x[2], 0, grad_x_unit[1], grad_x_unit[2],
                  color="k", length=0.7*length, normalize=True, alpha=alpha, zorder=zorder)
        
        # # 3D quiver
        # ax.quiver(x[0], x[1], x[2], grad_x_unit[0], grad_x_unit[1], grad_x_unit[2],
        #           color="k", length=length, normalize=True, alpha=alpha, zorder=zorder)