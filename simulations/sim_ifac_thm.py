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
from ssl_simulator.math import cov_matrix
from sourceseeking_3d.utils.tools_math import *
from sourceseeking_3d.utils.simulator import simulator

class sim_ifac_thm:
    def __init__(
            self,
            p0, s,
            md, omega_s, omega_k, mu_re_star,
            tf = 20, dt = 1/60, seed = None,
            fb_control = True, arr_len = 0.3
        ):

        self.tf = tf
        self.dt = dt
        self.data = {"R": None, "p": None, "theta_e": None, "delta": None, 
                     "Re": None, "log_Re_vee": None}

        self.n_agents = p0.shape[0] # number of agents
        self.s = s                  # speed (constant)

        # Initial state
        self.p0 = p0                        # spacial position of the agents
        self.md_0 = md                      # derired common heading
        self.Rd_0 = get_R_from_v(self.md_0) # desired attitude matrix (complete right-hand orthonormal basis)

        # Initial orientation of the body frames
        if seed is not None:
            np.random.seed(seed)
        alfa_0  = 2*(np.random.rand((self.n_agents)) - 0.49) * np.pi # YAW
        beta_0  = 2*(np.random.rand((self.n_agents)) - 0.49) * np.pi # PITCH
        gamma_0 = 2*(np.random.rand((self.n_agents)) - 0.49) * np.pi # ROLL

        self.R0 = np.repeat(np.eye(3)[None,:,:], self.n_agents, axis=0)
        for n in range(self.n_agents):
            self.R0[n,:,:] = rot_3d_matrix(alfa_0[n], beta_0[n], gamma_0[n])

        # Simulation frame parameters
        self.omega_s = omega_s
        self.omega_k = omega_k

        self.fb_control = fb_control
        self.mu_re_star = mu_re_star

        # -------------------------------------------------
        # Plotting configurable parameters
        self.ax_cols = ["r","g","b"]
        self.n_tail = 50
        self.arr_len = arr_len

        # -------------------------------------------------
        # Extra calculations for the paper
        lambdas_0 = np.linalg.eigvals(cov_matrix(self.p0))
        lambda_min_0, lambda_max_0 = np.min(lambdas_0), np.max(lambdas_0)

        D0 = np.max(np.linalg.norm(self.p0 - np.mean(self.p0, axis=1)[:,None], axis=1))
        epsilon_max1 = -D0 + np.sqrt(D0**2 + lambda_min_0)
        epsilon_max2 = - np.sqrt(lambda_max_0) + np.sqrt(lambda_max_0 + lambda_min_0)

        k1 = np.sqrt(2)*abs(self.omega_s)/self.mu_re_star # Proposition 2
        k2 = 2*np.pi*self.s / epsilon_max1 # Proposition 3
        
        print("Initial conditions summary:")
        print("D0 = {}, lambda_min_0 = {}".format(D0,lambda_min_0))
        print("epsilon_max: (1) = {}, (2) = {}". format(epsilon_max1, epsilon_max2))
        print("k1 = {}, k2 = {}".format(k1,k2))

    def numerical_simulation(self):
        """\
        - Function to launch the numerical simulation -
        """
        its = int(self.tf/self.dt) + 1

        # Generate the simulation engine
        kw = np.sqrt(2)*abs(self.omega_s)/self.mu_re_star # Just consider Proposition 2
        self.sim = simulator(p0=self.p0, R0=np.copy(self.R0), v0=self.s, dt=self.dt, kw=kw)

        # Set the initial derired common orientation
        self.md = np.copy(self.md_0)
        self.sim.set_R_desired(self.Rd_0)

        # Initialise the data dictionary with empty arrays
        for data_key in self.data:
            self.data[data_key] = np.empty((its, *self.sim.data[data_key].shape))

        # Generate the "unknown" omega_hat (earth-fixed)
        omega_hat_u = self.Rd_0 @ so3_hat([0,0,self.omega_s]) @ self.Rd_0.T
        omega_hat_ui = np.zeros_like(omega_hat_u)

        # Generate the "known" omega_hat (body-fixed)
        omega_hat_k = np.array([[0,0,0],[0,0,-self.omega_k],[0,self.omega_k,0]])
        omega_hat_ki = np.zeros_like(omega_hat_k)

        # Numerical simulation loop
        for i in tqdm(range(its)):
            # - Collect data from the simulation engine
            for data_key in self.data:
                self.data[data_key][i] = self.sim.data[data_key]

            # - Set a new derired common orientation Rd
            Ra = np.copy(self.Rd_0)

            # Compute the new rotation matrix
            omega_hat_ki = omega_hat_ki + self.dt*omega_hat_k
            omega_hat_ui = omega_hat_ui + self.dt*omega_hat_u
            
            theta_k = np.linalg.norm(so3_vee(omega_hat_ki))
            theta_u = np.linalg.norm(so3_vee(omega_hat_ui))
            
            # Ensure that theta is in [0,2\pi)
            if abs(theta_k) > 2*np.pi:
                omega_hat_ki = abs(theta_k) % (2*np.pi) * (omega_hat_ki / abs(theta_k))
            if abs(theta_u) > 2*np.pi:
                omega_hat_ui = abs(theta_u) % (2*np.pi) * (omega_hat_ui / abs(theta_u))

            # Since our computation of Exp(Ω) is an approximation, next we restrict 
            # the maximum rotation to a fixed step (π/6). E.g., it means that if we need 
            # to perform a π-radian rotation, we will execute six rotations of π/6 each.
            step = np.pi/10

            # We integrate first the earth-fixed rotation
            if abs(theta_u) >= step:
                for k in range(int(abs(theta_u) // (step))):
                    Ra = (Ra.T @ exp_map((step) * (omega_hat_ui / abs(theta_u)) )).T
                
                Ra = (Ra.T @  exp_map(abs(theta_u) % (step) * (omega_hat_ui / abs(theta_u)) )).T
            else:
                Ra = (Ra.T @ exp_map(omega_hat_ui)).T

            # and then the body-fixed rotation
            if abs(theta_k) >= step:
                for k in range(int(abs(theta_k) // (step))):
                    Ra = (Ra.T @ exp_map((step) * (omega_hat_ki / abs(theta_k)) )).T
                
                Ra = (Ra.T @  exp_map(abs(theta_k) % (step) * (omega_hat_ki / abs(theta_k)) )).T
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

        # -- Extract data fields from data dictonary --
        p_data = self.data["p"]
        R_data = self.data["R"]
        error_data = self.data["theta_e"]
        delta_data = self.data["delta"]
        # Re_data = self.data["Re"]
        # log_Re_vee_data = self.data["log_Re_vee"]
        
        # -- Post-processing --
        time_vec = np.linspace(self.dt, self.tf, int(self.tf/self.dt))
        covariances = cov_matrix(p_data)
        lambda_min = np.min(np.linalg.eigvals(covariances), axis=1)

        # -- Plotting the summary --
        # Figure and grid init
        fig = plt.figure(figsize=(13,4), dpi=300)
        grid = plt.GridSpec(2, 2, hspace=0.1, wspace=0)

        ax_delta = fig.add_subplot(grid[0, 0:1], xticklabels=[])
        ax_lambda = fig.add_subplot(grid[1, 0:1])
        ax_main = fig.add_subplot(grid[:, 1], projection='3d')

        # Format of the axis
        if lims is None:
            lims = [-3.5,3.5]
        ax_main.set_xlim(lims)
        ax_main.set_ylim(lims)
        ax_main.set_zlim(lims)
        ax_main.set_xlabel(r"$p_x [L]$", fontsize=11)
        ax_main.set_ylabel(r"$p_y [L]$", fontsize=11)  
        ax_main.set_zlabel(r"$p_z [L]$", fontsize=11)
        ax_main.set_box_aspect(aspect=None, zoom=1.2)
        ax_main.grid(True)

        ax_lambda.set_ylabel(r"$\Delta\lambda_{min}(t) [L^2]$")
        ax_lambda.set_xlabel(r"$t$ [T]")
        ax_lambda.grid(True)
        ax_delta.set_ylabel(r"$\delta_i(t)$ [rad]")
        ax_delta.grid(True)

        # ax_lambda.set_ylim([-0.2,np.pi+0.2])
        ax_delta.set_ylim([-0.2,np.max(delta_data[1,:])+0.2])
        
        # -> 3D main plot
        ti, t2, tf = 0, self.tf/2, self.tf
        li, l2, lf = int(0/self.dt), int(t2/self.dt), int(self.tf/self.dt)
        
        l_list = [li, lf]
        for n in range(self.n_agents):
            # Icons
            for l in l_list:
                ax_main.scatter(self.data["p"][l,n,0], self.data["p"][l,n,1], self.data["p"][l,n,2], 
                                marker="o", color="k")

            # Body frame axes
            for l in l_list:
                for i in range(3):
                    ax_main.quiver(p_data[l,n,0], p_data[l,n,1], p_data[l,n,2],
                                R_data[l,n,i,0], R_data[l,n,i,1], R_data[l,n,i,2],
                                color=self.ax_cols[i], length=self.arr_len, normalize=True, alpha=1)
                    
            # Tail
            ax_main.plot(p_data[:,n,0], p_data[:,n,1], p_data[:,n,2], "k", lw=1.5, alpha=0.5)

        # -> Delta plot
        ax_delta.axvline(0, c="k", ls="-", lw=1)
        ax_delta.axhline(0, c="k", ls="-", lw=1)

        ax_delta.plot(time_vec, delta_data[1:,:], "b", lw=1.2)

        label_delta = r"$\mu_{R_e}^* = \delta^*$"
        label_delta += r"= {0:.1f}".format(self.mu_re_star)
        ax_delta.text(self.tf-6, self.mu_re_star+0.2, label_delta, color="r")
        ax_delta.axhline(self.mu_re_star, c="r", ls="--", lw=1, alpha=1)

        # -> Lambda plot
        ax_lambda.axvline(0, c="k", ls="-", lw=1)
        ax_lambda.axhline(0, c="k", ls="-", lw=1)

        ax_lambda.plot(time_vec, lambda_min[1:] - lambda_min[1], "b", lw=1)
    
        ax_lambda.text(self.tf-5.5, -lambda_min[1] + lambda_min[1]*0.1, r"$-\lambda_{min}(P(t_0))$", color="r")
        ax_lambda.axhline(-lambda_min[1], c="r", ls="--", lw=1.2, alpha=1)

        # Save and show the plot ----------------
        if output_folder is not None:
            fig.savefig(os.path.join(output_folder, "sim_ifac_thm.png"))
            
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

        self.txt_title.set_text(
            "N = {0:} | t = {1:>5.2f} [T] \n".format(self.n_agents, i*self.dt) +
            "$\omega^k$ = $[${}$\pi,0,0]$ | ".format(int(self.omega_k/np.pi)) +
            "$k_\omega$ = {:.1f} | ".format(self.sim.kw[0]) +
            "$\omega^u$ = $[0,0,-\pi/${}$]$".format(int(np.pi/abs(self.omega_s)))
        )

    def generate_animation(self, output_folder, tf_anim=None, dpi=100, n_tail=200, lims=None, gif=False, fps=None):
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

        self.txt_title = main_ax.set_title(
            "N = {0:} | t = {1:>5.2f} [T] \n".format(self.n_agents, 0) +
            "$\omega^k$ = $[${}$\pi,0,0]$ | ".format(int(self.omega_k/np.pi)) +
            "$k_\omega$ = {:.1f} | ".format(self.sim.kw[0]) +
            "$\omega^u$ = $[0,0,-\pi/${}$]$".format(int(np.pi/abs(self.omega_s)))
        )

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
            tail, = main_ax.plot(self.data["p"][0,n,0], self.data["p"][0,n,1], self.data["p"][0,n,2], "k", lw=1.5, alpha=0.5)
            self.tails[n] = tail

        # -- Animation --
        # Init of the animation class
        anim = FuncAnimation(fig, self.animate, frames=tqdm(range(frames), initial=1, position=0), interval=1/fps*1000/4)

        # Generate and save the animation
        if gif:
            writer = PillowWriter(fps=15, bitrate=1800)
            anim.save(os.path.join(output_folder, "anim_ifac_thm.gif"),
                    writer = writer)
        else:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='Jesús Bautista Villar'), bitrate=10000)
            anim.save(os.path.join(output_folder, "anim_ifac_thm.mp4"), 
                    writer = writer)
