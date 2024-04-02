"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
-  -
"""

import numpy as np
import matplotlib.pyplot as plt

# -- Numerical tools --
from simulations.utils.tools_math import *

# DPI resolution dictionary
resolution_dic = {
    "480p"   : 640,
    "HD"     : 1280,
    "FullHD" : 1920,
    "2K"     : 2560,
    "4K"     : 3880
    }

def plot_3d_sphere_wf(ax,r,lim):
    # Generate the 3D sphere wireframe
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    X = r*np.outer(np.cos(theta), np.sin(phi))
    Y = r*np.outer(np.sin(theta), np.sin(phi))
    Z = r*np.outer(np.ones(np.size(theta)), np.cos(phi))

    # Plot the 3D sphere and its 2D projections
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, alpha=0.3, linewidth=0.8, color="k")
    ax.plot_wireframe(X, Y, Z, rstride=50, cstride=50, alpha=1, linewidth=1, color="gray")
    
    ax.contour(X, Y, Z, alpha=0.2, linewidths=0.7, colors=["k"], zdir='z', offset=-lim)
    ax.contour(X, Y, Z, alpha=0.2, linewidths=0.7, colors=["k"], zdir='y', offset= lim)
    ax.contour(X, Y, Z, alpha=0.2, linewidths=0.7, colors=["k"], zdir='x', offset=-lim)

def filter_R_data(R_data):
    # The final R_adata shape should be: (time_frames, agents, 3, 3)

    if len(R_data.shape) == 2: # One agent SO3 state
        return R_data[None,None,...]
    
    if R_data.shape[2] != 3 or R_data.shape[3] != 3:
        print("ERROR: The dimensionality of R_data {0:s} is wrong.".format(str(R_data.shape)),
              "Remember that the correct shape is (time_frames, agents, 3, 3).")
        return None
    
    if len(R_data.shape) == 4:
        return R_data
    
    # If the R_data dimensionality < 4 then we suppose the
    elif len(R_data.shape) == 3: # One agent trajectory
        return R_data[None,...]

    else:
        print("ERROR: The dimensionality of R_data {0:s} is wrong.".format(str(R_data.shape)),
              "Remember that the correct shape is (time_frames, agents, 3, 3).")
        return None
    

"""\
- Funtion to visualize the 3D heading trajectory -
"""
def plot_heading_traj(R_data, view=(25,-50), lim=1.6, ax=None):
    # Filtering the input
    R_data = filter_R_data(R_data)
    if R_data is None:
        return

    # Generate the 3D heading trajectory
    v = np.array([[1,0,0]])
    u = v @ R_data

    # -- Figure init --
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        grid = plt.GridSpec(1, 1, hspace=0, wspace=0)
        main_ax = fig.add_subplot(grid[:, 0], projection='3d')
    
    else:
        main_ax = ax

    # Format of the axis
    main_ax.set_xlabel(r"$X$")
    main_ax.set_ylabel(r"$Y$")  
    main_ax.set_zlabel(r"$Z$")
    main_ax.view_init(*view)

    main_ax.set_proj_type('ortho')
    main_ax.set_xlim([-lim,lim])
    main_ax.set_ylim([-lim,lim])
    main_ax.set_zlim([-lim,lim])
    main_ax.set_title("", fontsize=14)
    main_ax.grid(True)
    

    # Plot the 3D sphere and its 2D projections
    plot_3d_sphere_wf(main_ax,1,lim)
    
    if R_data.shape[1] > 1:
        for n in range(R_data.shape[1]):
            # Plot the 3D heading trajectories
            main_ax.plot(u[0,n,0,0], u[0,n,0,1], u[0,n,0,2], "or", markersize=2, alpha=0.5)
            main_ax.plot(u[:,n,0,0], u[:,n,0,1], u[:,n,0,2], "b", lw=0.8, alpha=0.8)
            if R_data.shape[0] > 1:
                main_ax.plot(u[-1,n,0,0], u[-1,n,0,1], u[-1,n,0,2], "or", markersize=3, alpha=1)
            
            # Plot the 2D heading trajectories projections
            main_ax.plot(u[0,n,0,0], u[0,n,0,1], "og", zdir='z', zs=-lim, markersize=2, alpha=0.5)
            main_ax.plot(u[0,n,0,0], u[0,n,0,2], "og", zdir='y', zs= lim, markersize=2, alpha=0.5)
            main_ax.plot(u[0,n,0,1], u[0,n,0,2], "og", zdir='x', zs=-lim, markersize=2, alpha=0.5)

            main_ax.plot(u[:,n,0,0], u[:,n,0,1], "r", zdir='z', zs=-lim, alpha=0.5)
            main_ax.plot(u[:,n,0,0], u[:,n,0,2], "r", zdir='y', zs= lim, alpha=0.5)
            main_ax.plot(u[:,n,0,1], u[:,n,0,2], "r", zdir='x', zs=-lim, alpha=0.5)

            if R_data.shape[0] > 1:
                main_ax.plot(u[-1,n,0,0], u[-1,n,0,1], "og", zdir='z', zs=-lim, markersize=2, alpha=1)
                main_ax.plot(u[-1,n,0,0], u[-1,n,0,2], "og", zdir='y', zs= lim, markersize=2, alpha=1)
                main_ax.plot(u[-1,n,0,1], u[-1,n,0,2], "og", zdir='x', zs=-lim, markersize=2, alpha=1)

    else:
        n = 0
        # Plot the 3D heading point
        main_ax.plot(u[0,n,0,0], u[0,n,0,1], u[0,n,0,2], "or", markersize=4, alpha=1)

        # Plot the 2D heading point projections
        main_ax.plot(u[0,n,0,0], u[0,n,0,1], "or", zdir='z', zs=-lim, markersize=2, alpha=1)
        main_ax.plot(u[0,n,0,0], u[0,n,0,2], "or", zdir='y', zs= lim, markersize=2, alpha=1)
        main_ax.plot(u[0,n,0,1], u[0,n,0,2], "or", zdir='x', zs=-lim, markersize=2, alpha=1)


"""\
- Funtion to visualize a trajectory in SO(3) using \omega \in so(3) (associated Lie algebra) -
"""
def plot_so3_traj(R_data, view=(25,-50), lim=5, ax=None):
    # Filtering the input
    R_data = filter_R_data(R_data)
    if R_data is None:
        return

    # Generate the SO(3) points
    omega = np.zeros(R_data.shape[0:3])
    for n in range(R_data.shape[1]):
        for t in range(R_data.shape[0]):
            log_R = log_map_of_R(R_data[t,n,...])
            omega[t,n,:] = np.array([log_R[2,1],log_R[0,2],log_R[1,0]])

    # -- Figure init --
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        grid = plt.GridSpec(1, 1, hspace=0, wspace=0)
        main_ax = fig.add_subplot(grid[:, 0], projection='3d')
    else:
        main_ax = ax

    # Format of the axis
    main_ax.set_xlabel(r"$w_x$")
    main_ax.set_ylabel(r"$w_y$")
    main_ax.set_zlabel(r"$w_z$")
    main_ax.view_init(*view)

    main_ax.set_proj_type('ortho')
    main_ax.set_xlim([-lim,lim])
    main_ax.set_ylim([-lim,lim])
    main_ax.set_zlim([-lim,lim])
    main_ax.set_title("", fontsize=14)
    main_ax.grid(True)

    # 3D sphere plot and its 2D projections
    plot_3d_sphere_wf(main_ax,np.pi,lim)

    for n in range(R_data.shape[1]):
        # 3D SO(3) trajectory plot
        main_ax.plot(omega[0,n,0], omega[0,n,1], omega[0,n,2], "or", markersize=2, alpha=0.5)
        main_ax.plot(omega[:,n,0], omega[:,n,1], omega[:,n,2], ".b", markersize=0.5, lw=0.8, alpha=0.4)
        if R_data.shape[0] > 1:
            main_ax.plot(omega[-1,n,0], omega[-1,n,1], omega[-1,n,2], "og", markersize=3, alpha=0.9)

        # 2D SO(3) projection plots
        main_ax.plot(omega[:,n,0], omega[:,n,1], ".r", zdir='z', zs=-lim, markersize=0.5, alpha=0.7)
        main_ax.plot(omega[:,n,0], omega[:,n,2], ".r", zdir='y', zs= lim, markersize=0.5, alpha=0.7)
        main_ax.plot(omega[:,n,1], omega[:,n,2], ".r", zdir='x', zs=-lim, markersize=0.5, alpha=0.7)
    
