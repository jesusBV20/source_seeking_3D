"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import os
import numpy as np
from tqdm import tqdm

# -- Graphic tools --
import matplotlib.pyplot as plt

# -- Animation tools --
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# -- Numerical tools --
from sourceseeking_3d.utils.tools_math import *
from sourceseeking_3d.utils.simulator import simulator

class sim_thm2:
    def __init__(self, wx, wd, mu_re_star, L1, 
                 v = 0.5, tf = 20, dt = 1/60, fb_control = True, 
                 arr_len = 0.3):
        
        self.n_agents = 1
        self.tf = tf
        self.dt = dt
        self.data = {"R": None, "p": None, "theta_e": None, "delta": None, 
                     "Re": None, "log_Re_vee": None}

        # Simulation frame parameters
        self.wx = wx
        self.fb_control = fb_control

        self.wd = wd
        self.mu_re_star = mu_re_star

        # Initial spacial position of the agents
        self.p0 = np.array([[-2, 2.5, 0]])
        self.v0 = v

        # Generation the initial orientation of the body frames
        alfa_0  = 2*(np.random.rand((self.n_agents)) - 0.49) * np.pi # YAW
        beta_0  = 2*(np.random.rand((self.n_agents)) - 0.49) * np.pi # PITCH
        gamma_0 = 2*(np.random.rand((self.n_agents)) - 0.49) * np.pi # ROLL

        self.R = np.repeat(np.eye(3)[None,:,:], self.n_agents, axis=0)
        for n in range(self.n_agents):
            self.R[n,:,:] = rot_3d_matrix(alfa_0[n], beta_0[n], gamma_0[n])

        # Set the initial derired common orientation
        self.L1_1 = L1

        # -------------------------------------------------
        # Plotting configurable parameters
        self.ax_cols = ["r","g","b"]
        self.n_tail = 50
        self.arr_len = arr_len


    def numerical_simulation(self):
        """\
        - Function to launch the numerical simulation -
        """
        its = int(self.tf/self.dt) + 1

        # Generate the simulation engine
        self.sim = simulator(p0=self.p0, R0=self.R, v0=self.v0, dt=self.dt, 
                             kw=np.sqrt(2)*abs(self.wd)/self.mu_re_star)

        # Set the initial derired common orientation
        self.L1 = np.copy(self.L1_1)
        Ra_0 = get_R_from_v(self.L1)
        self.sim.set_R_desired(Ra_0)

        # Initialise the data dictionary with empty arrays
        for data_key in self.data:
            self.data[data_key] = np.empty((its, *self.sim.data[data_key].shape))

        # Generate the "unknown" omega_hat (earth-fixed)
        omega_hat_u = Ra_0 @ so3_hat([0,0,self.wd]) @ Ra_0.T
        omega_hat_ui = np.zeros_like(omega_hat_u)

        # Generate the "known" omega_hat (body-fixed)
        omega_hat_k = np.array([[0,0,0],[0,0,-self.wx],[0,self.wx,0]])
        omega_hat_ki = np.zeros_like(omega_hat_k)

        # Numerical simulation loop
        for i in tqdm(range(its)):
            # - Collect data from the simulation engine
            for data_key in self.data:
                self.data[data_key][i] = self.sim.data[data_key]

            # - Set a new derired common orientation Ra
            Ra = np.copy(Ra_0)

            # Compute the new rotation matrix
            omega_hat_ki = omega_hat_ki + self.dt*omega_hat_k
            omega_hat_ui = omega_hat_ui + self.dt*omega_hat_u
            
            theta_k = np.linalg.norm(so3_vee(omega_hat_ki))
            theta_u = np.linalg.norm(so3_vee(omega_hat_ui))
            
            # Ensure that theta is in [0,2\pi)
            if theta_k > 2*np.pi:
                omega_hat_ki = theta_k % (2*np.pi) * (omega_hat_ki / theta_k)
            if theta_u > 2*np.pi:
                omega_hat_ui = theta_u % (2*np.pi) * (omega_hat_ui / theta_u)

            # Since our computation of Exp(Ω) is an approximation, next we restrict 
            # the maximum rotation to a fixed step (π/6). E.g., it means that if we need 
            # to perform a π-radian rotation, we will execute six rotations of π/6 each.
            step = np.pi/10

            # We integrate first the earth-fixed rotation
            if theta_u >= step:
                for k in range(int(theta_u // (step))):
                    Ra = (Ra.T @ exp_map((step) * (omega_hat_ui / theta_u) )).T
                
                Ra = (Ra.T @  exp_map(theta_u % (step) * (omega_hat_ui / theta_u) )).T
            else:
                Ra = (Ra.T @ exp_map(omega_hat_ui)).T

            # and then the body-fixed rotation
            if theta_k >= step:
                for k in range(int(theta_k // (step))):
                    Ra = (Ra.T @ exp_map((step) * (omega_hat_ki / theta_k) )).T
                
                Ra = (Ra.T @  exp_map(theta_k % (step) * (omega_hat_ki / theta_k) )).T
            else:
                Ra = (Ra.T @ exp_map(omega_hat_ki)).T

            # Once the rotation is applied, we pass the desired Ra to the simulator
            self.sim.set_R_desired(np.copy(Ra))

            # and the known time variation of Ra to the feedforward controller
            if self.fb_control:
                self.sim.set_R_desired_dot((Ra.T @ (omega_hat_k)).T)

            # - Simulator euler step integration
            self.sim.int_euler()

    def plot_article_figure(self, output_folder=None, lims=None):
        """
        - Function to generate the article figure -
        """ 
        n = 0

        # -- Extract data fields from data dictonary --
        p_data = self.data["p"]
        R_data = self.data["R"]
        error_data = self.data["theta_e"]
        delta_data = self.data["delta"]
        # Re_data = self.data["Re"]
        # log_Re_vee_data = self.data["log_Re_vee"]

        # -- Plotting the summary --
        # Figure and grid init
        fig = plt.figure(figsize=(12,4), dpi=300)
        grid = plt.GridSpec(2, 3, hspace=0.1, wspace=0)

        error_ax = fig.add_subplot(grid[0, 0:2], xticklabels=[])
        delta_ax = fig.add_subplot(grid[1, 0:2])
        main_ax = fig.add_subplot(grid[:, 2], projection='3d')

        # Format of the axis
        if lims is None:
            lims = [-3.5,3.5]
        main_ax.set_xlim(lims)
        main_ax.set_ylim(lims)
        main_ax.set_zlim(lims)
        main_ax.set_xlabel(r"$X$", fontsize=11)
        main_ax.set_ylabel(r"$Y$", fontsize=11)  
        main_ax.set_zlabel(r"$Z$", fontsize=11)
        main_ax.set_box_aspect(aspect=None, zoom=0.8)
        main_ax.grid(True)

        error_ax.set_ylabel(r"$\mu_{R_e}$")
        error_ax.grid(True)

        delta_ax.set_ylabel(r"$\delta$")
        delta_ax.set_xlabel(r"$t$ [T]")
        delta_ax.grid(True)

        error_ax.set_ylim([-0.2,np.pi+0.2])
        delta_ax.set_ylim([-0.2,np.pi+0.2])
        
        # -> 3D main plot
        ti, t2, tf = 0, self.tf/2, self.tf
        li, l2, lf = int(0/self.dt), int(t2/self.dt), int(self.tf/self.dt)
        
        l_list = [li, l2, lf]
        for l in l_list:
            # Icons
            main_ax.scatter(self.data["p"][l,:,0], self.data["p"][l,:,1], self.data["p"][l,:,2], 
                            marker="o", color="k")

            # Body frame axes
            for i in range(3):
                main_ax.quiver(p_data[l,n,0], p_data[l,n,1], p_data[l,n,2],
                            R_data[l,n,i,0], R_data[l,n,i,1], R_data[l,n,i,2],
                            color=self.ax_cols[i], length=self.arr_len, normalize=True, alpha=1)
            
        # Text labels
        main_ax.text(p_data[li,n,0]-1, p_data[li,n,1], p_data[li,n,2]+0.5, r"$t$ = {0:.0f}".format(ti))
        main_ax.text(p_data[l2,n,0]-1, p_data[l2,n,1], p_data[l2,n,2]-3, r"$t$ = {0:.0f}".format(t2))
        main_ax.text(p_data[lf,n,0]-1, p_data[lf,n,1], p_data[lf,n,2]-2.5, r"$t$ = {0:.0f}".format(tf))
        
        # Tail
        main_ax.plot(p_data[:,n,0], p_data[:,n,1], p_data[:,n,2], "k", lw=1.5, alpha=0.5)

        # -> Error plot
        error_ax.axvline(0, c="k", ls="-", lw=1)
        error_ax.axhline(0, c="k", ls="-", lw=1)

        # Time evolution of the attitude error
        time_vec = np.linspace(self.dt, self.tf, int(self.tf/self.dt))
        for n in range(R_data.shape[1]):
            error_ax.plot(time_vec, error_data[1:,n], "b", lw=1)

        # Desired attitude error dashed line
        error_ax.text(self.tf-2.7, self.mu_re_star+0.2, r"$\mu_{R_e}^*$", color="r")
        error_ax.text(self.tf-1.6, self.mu_re_star+0.2, r"= {0:.1f}".format(self.mu_re_star), color="r")
        error_ax.axhline(self.mu_re_star, c="r", ls="--", lw=1, alpha=1)

        # -> Delta plot
        delta_ax.axvline(0, c="k", ls="-", lw=1)
        delta_ax.axhline(0, c="k", ls="-", lw=1)

        # Time evolution of delta
        time_vec = np.linspace(self.dt, self.tf, int(self.tf/self.dt))
        for n in range(R_data.shape[1]):
            delta_ax.plot(time_vec, delta_data[1:,n], "b", lw=1)

        # Desired delta dashed line
        delta_ax.text(self.tf-2.7, self.mu_re_star+0.2, r"$\delta*$", color="r")
        delta_ax.text(self.tf-1.8, self.mu_re_star+0.2, r"= {0:.1f}".format(self.mu_re_star), color="r")
        delta_ax.axhline(self.mu_re_star, c="r", ls="--", lw=1, alpha=1)

        # Save and show the plot ----------------
        if output_folder is not None:
            fig.savefig(os.path.join(output_folder, "anim_thm2.png"))
            
        plt.show()


    def animate(self, i):
        """"
        - Animation function update -
        """
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

        self.txt_title.set_text("N = {0:} | t = {1:>5.2f} [T] \n".format(self.n_agents, i*self.dt, int(self.wx/np.pi)) +
                                "$w^k$ = $[${}$\pi,0,0]$ | $k_\omega$ = {:.1f} | ".format(int(self.wx/np.pi), self.sim.kw[0]) +
                                "$w^u$ = $[0,0,-\pi/${}$]$".format(int(np.pi/abs(self.wd))))


    def generate_animation(self, output_folder, tf_anim=None, dpi=100, n_tail=200, lims=None, gif=False, fps=None):
        """
        - Funtion to generate the full animation of the simulation -
        """
        if tf_anim is None:
            tf_anim = self.tf

        if fps is None:
            fps = 1/self.dt

        frames = int(tf_anim/self.dt-1)
        self.n_tail = n_tail

        print("Animation parameters: ", {"fps":fps, "tf":tf_anim, "frames":frames})

        # -- Plotting the summary --
        # Figure and grid init
        fig = plt.figure(figsize=(6,7), dpi=dpi)
        grid = plt.GridSpec(1, 1, hspace=0.1, wspace=0.4)

        main_ax  = fig.add_subplot(grid[:, :], projection='3d')

        # Format of the axis
        if lims is None:
            lims = [-3.5,3.5]
        main_ax.set_xlim(lims)
        main_ax.set_ylim(lims)
        main_ax.set_zlim(lims)
        main_ax.set_xlabel(r"$X$", fontsize=11)
        main_ax.set_ylabel(r"$Y$", fontsize=11)  
        main_ax.set_zlabel(r"$Z$", fontsize=11)
        main_ax.grid(True)

        self.txt_title = main_ax.set_title("N = {0:} | t = {1:>5.2f} [T] \n".format(self.n_agents, 0, int(self.wx/np.pi)) +
                                "$w^k$ = $[${}$\pi,0,0]$ | $k_\omega$ = {:.1f} | ".format(int(self.wx/np.pi), self.sim.kw[0]) +
                                "$w^u$ = $[0,0,${}$\pi/${}$]$".format(int(np.sign(self.wd)),int(np.pi/abs(self.wd))))   

        # Draw icons and body frame quivers
        self.icons = main_ax.scatter(self.data["p"][0,:,0], self.data["p"][0,:,1], self.data["p"][0,:,2], 
                                     marker="o", color="k", alpha=0.5)
        
        self.ax_arrows = np.empty((self.data["R"].shape[1],3), dtype=object)
        self.tails = np.empty((self.data["R"].shape[1]), dtype=object)
        for n in range(self.data["R"].shape[1]):
            # Body frame axes
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
        if gif:
            writer = PillowWriter(fps=15, bitrate=1800)
            anim.save(os.path.join(output_folder, "anim_thm2.gif"),
                    writer = writer)
        else:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='Jesús Bautista Villar'), bitrate=10000)
            anim.save(os.path.join(output_folder, "anim_thm2.mp4"), 
                    writer = writer)
