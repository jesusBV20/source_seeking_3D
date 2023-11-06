"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import os
import time
import numpy as np
from tqdm import tqdm

# -- Graphic tools --
import matplotlib.pyplot as plt

# -- Animation tools --
from matplotlib.animation import FuncAnimation, FFMpegWriter

# -- Numerical tools --
from simulations.utils.tools_math import *
from simulations.utils.simulator import simulator
from simulations.utils.module_ss import module_ss

from simulations.source_seeking.sigma_common import sigma
from simulations.source_seeking.sigma_funcs import sigma_gauss

class sim_ss_test:
    def __init__(self, n_agents=4, tf = 20, dt = 1/60, wx = 6*np.pi, 
                 fb_control = True, sim_kw={}):
        self.n_agents = n_agents
        self.tf = tf
        self.dt = dt
        self.data = {"R": None, "p": None, "theta_e": None, "pc":None}

        # Generate the scalar field
        sigma_func = sigma_gauss(x0=[50,40,40], max_intensity=100, dev=[20,20,30])
        self.sigma = sigma(sigma_func)

        # Simulation frame parameters
        self.wx = wx
        self.fb_control = fb_control

        # Initial spacial position of the agents
        pc = np.array([-55,-55,-55])
        p0 = 2*(np.random.random((n_agents,3)) - 0.49) * 15 + pc
        v0 = 5

        # Generation the initial orientation of the body frames
        alfa_0  = 2*(np.random.rand((n_agents)) - 0.49) * np.pi # YAW
        beta_0  = 2*(np.random.rand((n_agents)) - 0.49) * np.pi # PITCH
        gamma_0 = 2*(np.random.rand((n_agents)) - 0.49) * np.pi # ROLL

        R = np.repeat(np.eye(3)[None,:,:], n_agents, axis=0)
        for n in range(n_agents):
            R[n,:,:] = rot_3d_matrix(alfa_0[n], beta_0[n], gamma_0[n])

        # -------------------------------------------------
        # Generate the simulation engine
        self.sim = simulator(p0=p0, R0=R, v0=v0, dt=self.dt, **sim_kw)
        self.ss_module = module_ss(self.sim, self.sigma)

        # Set the initial derired common orientation
        self.Lsgima = self.ss_module.l_sigma_hat_norm
        self.sim.set_R_desired(get_R_from_v(self.Lsgima))

        # -------------------------------------------------
        # Plotting configurable parameters
        self.title = ""
        self.ax_cols = ["r","g","b"]
        self.n_tail = 50
        self.lim = 80

        self.arr_len = self.lim*0.05
        


    """\
    - Function to launch the numerical simulation -
    """
    def numerical_simulation(self):
        its = int(self.tf/self.dt) + 1

        # Initialise the data dictionary with empty arrays
        for data_key in self.data:
            self.data[data_key] = np.empty((its, *self.sim.data[data_key].shape))

        # As Y and Z are not fixed, we will apply an \omega_x rotation to the
        # reference desired rotation matrix
        omega_hat_x = np.array([[0,0,0],[0,0,-self.wx],[0,self.wx,0]])
        omega_hat_xi = omega_hat_x * 0

        # Numerical simulation loop
        for i in tqdm(range(its)):
            # - Collect data from the simulation engine
            for data_key in self.data:
                self.data[data_key][i] = self.sim.data[data_key]

            # - Set a new derired common orientation Re
            self.Lsgima = self.ss_module.l_sigma_hat_norm

            # Generate the ny and nz (ortogonal vector to Lsgima)
            R = get_R_from_v(self.Lsgima)

            # Rotate the resultant action R with w = w_x
            omega_hat_xi = omega_hat_xi + self.dt*omega_hat_x
            
            # Ensure that \omega \in [0,2\pi)
            if omega_hat_xi[2,1] > 2*np.pi:
                omega_hat_xi = omega_hat_xi[2,1] % (2*np.pi) * omega_hat_x / self.wx
            
            # Since our computation of Exp(Ω) is an approximation, next we restrict 
            # the maximum rotation to a fixed step (π/6). E.g., it means that if we need 
            # to perform a π-radian rotation, we will execute six rotations of π/6 each.
            step = np.pi/6
            if omega_hat_xi[2,1] >= step:
                for k in range(int(omega_hat_xi[2,1] // (step))):
                    R = (R.T @ exp_map_of_R((step) * omega_hat_x / self.wx)).T
                
                R = (R.T @ exp_map_of_R(omega_hat_xi[2,1] % (step) * omega_hat_x / self.wx)).T
            else:
                R = (R.T @ exp_map_of_R(omega_hat_xi)).T

            # Once the rotation is applied, now we set the desired Re
            self.sim.set_R_desired(R)

            # Inform to the controller how Rd will change next
            if self.fb_control:
                self.sim.set_R_desired_dot((R.T @ (omega_hat_x)).T)


            # - Simulator euler step integration
            self.sim.int_euler()
            self.ss_module.compute_Lsigma()


    """\
    - Function to generate the summary graphical plot of the whole simulation -
    """
    def plot_summary(self, t_list=None, dpi=100):
        if t_list is None:
            t_list = [0, self.tf]

        ti, tf = t_list[0], t_list[-1]
        li, lf = int(ti/self.dt), int(tf/self.dt)

        # -- Extract data fields from data dictonary --
        p_data = self.data["p"]
        R_data = self.data["R"]
        error_data = self.data["theta_e"]
        pc_data = self.data["pc"]

        # -- Plotting the summary --
        # Figure and grid init
        fig = plt.figure(figsize=(16,9))
        grid = plt.GridSpec(3, 5, hspace=0, wspace=0.6)

        main_ax = fig.add_subplot(grid[:, 0:3], projection='3d', computed_zorder=False)
        error_ax = fig.add_subplot(grid[2, 3:5])

        # Format of the axis
        main_ax.set_xlim([-self.lim,self.lim])
        main_ax.set_ylim([-self.lim,self.lim])
        main_ax.set_zlim([-self.lim,self.lim])
        main_ax.set_title(self.title, fontsize=14)
        main_ax.set_ylabel(r"$p_y$ (L)")  
        main_ax.set_xlabel(r"$p_x$ (L)")
        main_ax.set_zlabel(r"$p_z$ (L)")
        main_ax.grid(True)

        error_ax.set_ylabel(r"$|\theta|$")
        error_ax.set_xlabel(r"t [T]")
        error_ax.grid(True)
        
        # - 3D main plot -
        # Draw the scalar field
        self.sigma.draw_3D(fig=fig, ax=main_ax, lim=self.lim, contour_levels=40)

        # Icons
        main_ax.scatter(self.data["p"][li,:,0], self.data["p"][li,:,1], self.data["p"][li,:,2], 
                        marker="o", color="k", alpha=0.5, s=5)
        # main_ax.scatter(self.data["p"][lf,:,0], self.data["p"][lf,:,1], self.data["p"][lf,:,2], 
        #                 marker="o", color="k")

        # Body frame axes
        for n in range(R_data.shape[1]):
            for i in range(3):
                main_ax.quiver(p_data[li,n,0], p_data[li,n,1], p_data[li,n,2],
                            R_data[li,n,i,0], R_data[li,n,i,1], R_data[li,n,i,2],
                            color=self.ax_cols[i], length=self.arr_len, normalize=True, alpha=0.3)
                
                main_ax.quiver(p_data[lf,n,0], p_data[lf,n,1], p_data[lf,n,2],
                            R_data[lf,n,i,0], R_data[lf,n,i,1], R_data[lf,n,i,2],
                            color=self.ax_cols[i], length=self.arr_len, normalize=True, alpha=1)
        
            # Tail
            main_ax.plot(p_data[:,n,0], p_data[:,n,1], p_data[:,n,2], "k", lw=0.5, alpha=0.5)

        # Centroid tail 2D projection
        main_ax.plot(pc_data[:,1], pc_data[:,2], "--k", lw=1, zdir="x", zs=-self.lim)
        main_ax.plot(pc_data[:,0], pc_data[:,2], "--k", lw=1, zdir="y", zs= self.lim)
        main_ax.plot(pc_data[:,0], pc_data[:,1], "--k", lw=1, zdir="z", zs=-self.lim)

        # - Error plot -
        error_ax.grid(True)
        error_ax.axhline(0, c="k", ls="--", lw=1)
        time_vec = np.linspace(0, self.tf, int(self.tf/self.dt) + 1)

        for n in range(R_data.shape[1]):
            error_ax.plot(time_vec, error_data[:,n], "b", lw=1)

        plt.show()


    """"\
    - Animation function update -
    """
    def animate(self, i):
        # Update icons
        self.icons._offsets3d = (self.data["p"][i,:,0], self.data["p"][i,:,1], self.data["p"][i,:,2])
        
        for n in range(self.data["R"].shape[1]):
            #Update trace
            if i > self.n_tail:
                self.tails[n].set_data_3d(self.data["p"][i-self.n_tail:i,n,0], self.data["p"][i-self.n_tail:i,n,1], 
                                          self.data["p"][i-self.n_tail:i,n,2])
            else:
                self.tails[n].set_data_3d(self.data["p"][0:i,n,0], self.data["p"][0:i,n,1], self.data["p"][0:i,n,2])

            # Update axis quivers
            for k in range(3):
                uvw = self.data["p"][i,n,:] + self.data["R"][i,n,k,:]*self.arr_len
                new_segs = [[self.data["p"][i,n,:].tolist(), uvw.tolist()]]
                self.ax_arrows[n,k].set_segments(new_segs)
        
        # return self.ax_arrows


    """"\
    - Funtion to generate the full animation of the simulation -
    """
    def generate_animation(self, output_folder, tf_anim=None, res=1920, n_tail=200):
        if tf_anim is None:
            tf_anim = self.tf

        fps = 1/self.dt
        frames = int(tf_anim/self.dt-1)
        self.n_tail = n_tail

        print("Animation parameters: ", {"fps":fps, "tf":tf_anim, "frames":frames})

        # -- Plotting the summary --
        # Figure and grid init
        fig = plt.figure()
        grid = plt.GridSpec(1, 1, hspace=0.1, wspace=0.4)

        main_ax  = fig.add_subplot(grid[:, :], projection='3d')

        # Format of the axis
        main_ax.set_xlim([-7,7])
        main_ax.set_ylim([-7,7])
        main_ax.set_zlim([-7,7])
        main_ax.set_title(self.title, fontsize=14)
        main_ax.set_ylabel(r"$p_y$ (L)")  
        main_ax.set_xlabel(r"$p_x$ (L)")
        main_ax.set_zlabel(r"$p_z$ (L)")
        main_ax.grid(True)

        # Draw icons and body frame quivers
        self.icons = main_ax.scatter(self.data["p"][0,:,0], self.data["p"][0,:,1], self.data["p"][0,:,2], 
                                     marker="o", color="k", alpha=0.5)
        
        self.ax_arrows = np.empty((self.data["R"].shape[1],3), dtype=object)
        self.tails = np.empty((self.data["R"].shape[1]), dtype=object)
        for n in range(self.data["R"].shape[1]):
            # Body frame axis
            for i in range(3):
                arr = main_ax.quiver(self.data["p"][0,n,0], self.data["p"][0,n,1], self.data["p"][0,n,2],
                            self.data["R"][0,n,i,0], self.data["R"][0,n,i,1], self.data["R"][0,n,i,2],
                            color=self.ax_cols[i], length=self.arr_len, normalize=True)
                self.ax_arrows[n,i] = arr

            # Tail
            tail, = main_ax.plot(self.data["p"][0,n,0], self.data["p"][0,n,1], self.data["p"][0,n,2], "k", lw=0.5, alpha=0.5)
            self.tails[n] = tail

        # -- Animation --
        # Init of the animation class
        anim = FuncAnimation(fig, self.animate, frames=tqdm(range(frames), initial=1, position=0), interval=1/fps*1000/4)

        # Generate and save the animation
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Jesús Bautista Villar'), bitrate=10000)
        anim.save(os.path.join(output_folder, "anim__{0}_{1}_{2}__{3}_{4}_{5}.mp4".format(*time.localtime()[0:6])), 
                writer = writer)