"""\
# Copyright (C) 2025 Jesús Bautista Villar <jesbauti20@gmail.com>
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
from sourceseeking_3d.utils.tools_plot import resolution_dic

from sourceseeking_3d.utils.simulator import simulator
from sourceseeking_3d.utils.module_ss import module_ss

from sourceseeking_3d.sigma_common import sigma
from sourceseeking_3d.sigma_funcs import sigma_gauss

class sim_ifac_ss:
    def __init__(
            self, 
            n_agents=4, tf = 20, dt = 1/60, omega_k = 6*np.pi, v = 15,
            omega_s = np.pi/4, mu_re_star = 0.4, 
            fb_control = True, sim_kw={}, seed=None
        ):

        self.n_agents = n_agents
        self.tf = tf
        self.dt = dt
        self.data = {"R": None, "p": None, "theta_e": None, "pc":None, 
                     "Lsigma":None, "grad_pc":None}

        # Generate the scalar field
        sigma_func = sigma_gauss(x0=[50,20,40], max_intensity=100, dev=[20,20,30])
        self.sigma = sigma(sigma_func)

        # Simulation frame parameters
        self.omega_k = omega_k
        self.fb_control = fb_control
        self.sim_kw = sim_kw

        self.omega_s = omega_s
        self.mu_re_star = mu_re_star

        # Initial spacial position of the agents
        pc = np.array([-55,-55,-55])
        self.p0 = 2*(np.random.random((n_agents,3)) - 0.49) * 15 + pc
        self.v0 = v

        # Generation the initial orientation of the body frames
        if seed is not None:
            np.random.seed(seed)
        alfa_0  = 2*(np.random.rand((n_agents)) - 0.49) * np.pi # YAW
        beta_0  = 2*(np.random.rand((n_agents)) - 0.49) * np.pi # PITCH
        gamma_0 = 2*(np.random.rand((n_agents)) - 0.49) * np.pi # ROLL

        self.R0 = np.repeat(np.eye(3)[None,:,:], n_agents, axis=0)
        for n in range(n_agents):
            self.R0[n,:,:] = rot_3d_matrix(alfa_0[n], beta_0[n], gamma_0[n])
            
        # -------------------------------------------------
        # Plotting configurable parameters
        self.title = ""
        self.ax_cols = ["r","g","b"]
        self.n_tail = 50
        self.lim = 80

        self.arr_len = self.lim*0.05
        

    def numerical_simulation(self):
        """\
        - Function to launch the numerical simulation -
        """

        its = int(self.tf/self.dt) + 1

        # Generate the simulation engine
        self.sim = simulator(p0=self.p0, R0=self.R0, v0=self.v0, dt=self.dt, 
                             kw=np.sqrt(2)*abs(self.omega_s)/self.mu_re_star)
        self.ss_module = module_ss(self.sim, self.sigma)

        # Set the initial derired common orientation
        self.Lsgima = self.ss_module.l_sigma_hat_norm
        self.sim.set_R_desired(get_R_from_v(self.Lsgima))

        # Initialise the data dictionary with empty arrays
        for data_key in self.data:
            if data_key in self.sim.data:
                self.data[data_key] = np.empty((its, *self.sim.data[data_key].shape))
            elif data_key in self.ss_module.data:
                self.data[data_key] = np.empty((its, *self.ss_module.data[data_key].shape))

        # As Y and Z are not fixed, we will apply an \omega_x rotation to the
        # reference desired rotation matrix
        omega_hat_x = np.array([[0,0,0],[0,0,-self.omega_k],[0,self.omega_k,0]])
        omega_hat_xi = omega_hat_x * 0

        # Numerical simulation loop
        for i in tqdm(range(its)):
            # - Collect data from the simulation engine
            for data_key in self.data:
                if data_key in self.sim.data:
                    self.data[data_key][i] = self.sim.data[data_key]
                elif data_key in self.ss_module.data:
                    self.data[data_key][i] = self.ss_module.data[data_key]

            # - Set a new derired common orientation Re
            self.Lsgima = self.ss_module.l_sigma_hat_norm

            # Generate the ny and nz (ortogonal vector to Lsgima)
            R = get_R_from_v(self.Lsgima)

            # Rotate the resultant action R with w = w_x
            omega_hat_xi = omega_hat_xi + self.dt*omega_hat_x
            
            # Ensure that \omega \in [0,2\pi)
            if omega_hat_xi[2,1] > 2*np.pi:
                omega_hat_xi = omega_hat_xi[2,1] % (2*np.pi) * omega_hat_x / self.omega_k
            
            # Since our computation of Exp(Ω) is an approximation, next we restrict 
            # the maximum rotation to a fixed step (π/6). E.g., it means that if we need 
            # to perform a π-radian rotation, we will execute six rotations of π/6 each.
            step = np.pi/6
            if omega_hat_xi[2,1] >= step:
                for k in range(int(omega_hat_xi[2,1] // (step))):
                    R = (R.T @ exp_map((step) * omega_hat_x / self.omega_k)).T
                
                R = (R.T @ exp_map(omega_hat_xi[2,1] % (step) * omega_hat_x / self.omega_k)).T
            else:
                R = (R.T @ exp_map(omega_hat_xi)).T

            # Once the rotation is applied, now we set the desired Re
            self.sim.set_R_desired(R)

            # Inform to the controller how Rd will change next
            if self.fb_control:
                self.sim.set_R_desired_dot((R.T @ (omega_hat_x)).T)


            # - Simulator euler step integration
            self.sim.int_euler()
            self.ss_module.compute_Lsigma()


    def plot_article_figure(self, output_folder=None, dpi=100, ):
        """
        - Function to generate the article figure -
        """ 

        ti, tf = 0, self.tf
        li, lf = int(ti/self.dt), int(tf/self.dt)

        # -- Extract data fields from data dictonary --
        p_data = self.data["p"]
        R_data = self.data["R"]
        error_data = self.data["theta_e"]
        pc_data = self.data["pc"]
        # lsigma_data = self.data["Lsigma"]
        # grad_data = self.data["grad_pc"]

        dist_data = np.linalg.norm(p_data[:,:,:] - np.array(self.ss_module.p_star), axis=2)
        
        # -- Plotting the summary --
        # Figure and grid init
        fig = plt.figure(figsize=(13,4), dpi=dpi)
        grid = plt.GridSpec(2, 2, hspace=0.1, wspace=0)

        ax_error = fig.add_subplot(grid[0, 0], xticklabels=[])
        ax_dist = fig.add_subplot(grid[1, 0])
        ax_main = fig.add_subplot(grid[:, 1], projection='3d', computed_zorder=False)

        # Format of the axis

     
        # if lims is None:
        #     lims = [-3.5,3.5]
        # ax_main.set_xlim(lims)
        # ax_main.set_ylim(lims)
        # ax_main.set_zlim(lims)
        ax_main.set_title(self.title, fontsize=14)
        ax_main.view_init(azim=-120)
        ax_main.set_box_aspect(aspect=None, zoom=1.2)

        ax_main.grid(True)
        ax_error.set_ylabel(r"$\mu_{R_e}(t)$")
        ax_error.grid(True)

        ax_dist.set_ylabel(r"$| |p_i(t) - p_\sigma| |$ [L]")
        ax_dist.set_xlabel(r"t [T]")
        ax_dist.grid(True)
        
        # - 3D main plot -
        # Draw the scalar field
        self.sigma.draw_3D(fig=fig, ax=ax_main, lim=self.lim, contour_levels=30, offsets=[1,1,-1])

        # Text
        n=0
        ax_main.text(-50, -100, -40, r"$t$ = {0:.0f}".format(ti), fontsize=18)
        ax_main.text(10, 50+30, 40-40, r"$t$ = {0:.0f}".format(tf), fontsize=18)

        # Icons
        ax_main.scatter(self.data["p"][li,:,0], self.data["p"][li,:,1], self.data["p"][li,:,2], 
                        marker="o", color="k", alpha=0.5, s=5)

        # Body frame axes
        for n in range(R_data.shape[1]):
            for i in range(3):
                ax_main.quiver(p_data[li,n,0], p_data[li,n,1], p_data[li,n,2],
                            R_data[li,n,i,0], R_data[li,n,i,1], R_data[li,n,i,2],
                            color=self.ax_cols[i], length=self.arr_len, normalize=True, alpha=0.3)
                
                ax_main.quiver(p_data[lf,n,0], p_data[lf,n,1], p_data[lf,n,2],
                            R_data[lf,n,i,0], R_data[lf,n,i,1], R_data[lf,n,i,2],
                            color=self.ax_cols[i], length=self.arr_len, normalize=True, alpha=1)
        
            # Tail
            ax_main.plot(p_data[:,n,0], p_data[:,n,1], p_data[:,n,2], "k", lw=0.5, alpha=0.5)

        # Centroid tail 2D projection
        ax_main.plot(pc_data[:,1], pc_data[:,2], "--k", lw=1, zdir="x", zs= self.lim)
        ax_main.plot(pc_data[:,0], pc_data[:,2], "--k", lw=1, zdir="y", zs= self.lim)
        ax_main.plot(pc_data[:,0], pc_data[:,1], "--k", lw=1, zdir="z", zs=-self.lim)

        ax_main.plot([pc_data[0,0], self.lim], [pc_data[0,1],pc_data[0,1]], [pc_data[0,2],pc_data[0,2]], "-k", lw=0.5, alpha=0.8)
        ax_main.plot([pc_data[0,0], pc_data[0,0]], [pc_data[0,1],self.lim], [pc_data[0,2],pc_data[0,2]], "-k", lw=0.5, alpha=0.8)
        ax_main.plot([pc_data[0,0], pc_data[0,0]], [pc_data[0,1],pc_data[0,1]], [pc_data[0,2],-self.lim], "-k", lw=0.5, alpha=0.8)
    
        ax_main.set_xlabel(r"$p_x [L]$", fontsize=11)
        ax_main.set_ylabel(r"$p_y [L]$", fontsize=11)  
        ax_main.set_zlabel(r"", fontsize=11)

        # - Error plot -
        ax_error.grid(True)
        ax_error.axvline(0, c="k", ls="-", lw=1)
        ax_error.axhline(0, c="k", ls="-", lw=1)
        time_vec = np.linspace(self.dt, self.tf, int(self.tf/self.dt))

        for n in range(R_data.shape[1]):
            ax_error.plot(time_vec, error_data[1:,n], "b", lw=1, alpha=0.3)

        # Desired attitude error dashed line
        ax_error.text(1.6, self.mu_re_star+0.3, r"$\mu_{R_e}^*$", color="r")
        ax_error.text(2.6, self.mu_re_star+0.2, r"= {0:.1f}".format(self.mu_re_star), color="r")
        ax_error.axhline(self.mu_re_star, c="r", ls="--", lw=1, alpha=1)
        
        # - Distance plot -
        ax_dist.grid(True)
        ax_dist.axvline(0, c="k", ls="-", lw=1)
        ax_dist.axhline(0, c="k", ls="-", lw=1)
        time_vec = np.linspace(self.dt, self.tf, int(self.tf/self.dt))

        for n in range(R_data.shape[1]):
            ax_dist.plot(time_vec, dist_data[1:,n], "b", lw=1, alpha=0.3)

        # Save and show the plot ----------------
        if output_folder is not None:
            fig.savefig(os.path.join(output_folder, "anim_ss.png"))
            
        plt.show()


    def animate(self, i):
        """
        - Animation function update -
        """

        # Update icons
        self.icons._offsets3d = (self.data["p"][i,:,0], self.data["p"][i,:,1], self.data["p"][i,:,2])
        
        # Update pc tail
        zeros_array =np.zeros(i)
        self.pc_tail[0].set_data_3d(zeros_array - self.lim, self.data["pc"][0:i,1], self.data["pc"][0:i,2])
        self.pc_tail[1].set_data_3d(self.data["pc"][0:i,0], zeros_array + self.lim, self.data["pc"][0:i,2])
        self.pc_tail[2].set_data_3d(self.data["pc"][0:i,0], self.data["pc"][0:i,1], zeros_array - self.lim)
        
        self.pc_lines[0].set_data_3d([self.data["pc"][i,0], -self.lim], [self.data["pc"][i,1],self.data["pc"][i,1]], 
                                     [self.data["pc"][i,2],self.data["pc"][i,2]])
        self.pc_lines[1].set_data_3d([self.data["pc"][i,0], self.data["pc"][i,0]], [self.data["pc"][i,1], self.lim], 
                                     [self.data["pc"][i,2],self.data["pc"][i,2]])
        self.pc_lines[2].set_data_3d([self.data["pc"][i,0], self.data["pc"][i,0]], [self.data["pc"][i,1],self.data["pc"][i,1]], 
                                     [self.data["pc"][i,2], -self.lim])

        for n in range(self.data["R"].shape[1]):
            # Update axis quivers
            for k in range(3):
                uvw = self.data["p"][i,n,:] + self.data["R"][i,n,k,:]*self.arr_len
                new_segs = [[self.data["p"][i,n,:].tolist(), uvw.tolist()]]
                self.ax_arrows[n,k].set_segments(new_segs)
        
        # return self.ax_arrows


    def generate_animation(self, output_folder, dpi=100, tf_anim=None, gif=False, fps=None):
        """
        - Funtion to generate the full animation of the simulation -
        """

        if tf_anim is None:
            tf_anim = self.tf

        if fps is None:
            fps = 1/self.dt

        frames = int(tf_anim/self.dt-1)

        print("Animation parameters: ", {"fps":fps, "tf":tf_anim, "frames":frames})

        # -- Extract data fields from data dictonary --
        p_data = self.data["p"]
        R_data = self.data["R"]
        pc_data = self.data["pc"]
        # lsigma_data = self.data["Lsigma"]
        # grad_data = self.data["grad_pc"]

        # -- Initial state of the animation --
        # Figure and grid init
        fig = plt.figure(figsize=(6,5), dpi=dpi)
        grid = plt.GridSpec(1, 1, hspace=0.1, wspace=0.4)

        ax_main  = fig.add_subplot(grid[:, :], projection='3d', computed_zorder=False)

        # Format of the axis
        ax_main.set_title(self.title, fontsize=14)
        ax_main.grid(True)

        # Draw the scalar field
        self.sigma.draw_3D(fig=fig, ax=ax_main, lim=self.lim, contour_levels=30)

        # Draw icons and body frame quivers
        self.icons = ax_main.scatter(p_data[0,:,0], p_data[0,:,1], p_data[0,:,2], 
                                     marker="o", color="k", alpha=0.5, s=5)
        
        self.ax_arrows = np.empty((R_data.shape[1],3), dtype=object)
        self.tails = np.empty((R_data.shape[1]), dtype=object)
        for n in range(R_data.shape[1]):
            # Body frame axis
            for i in range(3):
                arr = ax_main.quiver(p_data[0,n,0], p_data[0,n,1], p_data[0,n,2],
                            R_data[0,n,i,0], R_data[0,n,i,1], R_data[0,n,i,2],
                            color=self.ax_cols[i], length=self.arr_len, normalize=True)
                self.ax_arrows[n,i] = arr

        # Centroid tail 2D projection
        tail_x, = ax_main.plot(-self.lim, pc_data[0,1], pc_data[0,2], "--k", lw=1)
        tail_y, = ax_main.plot(pc_data[0,0], self.lim, pc_data[0,2],  "--k", lw=1)
        tail_z, = ax_main.plot(pc_data[0,0], pc_data[0,1], self.lim,  "--k", lw=1)
        self.pc_tail = np.array([tail_x, tail_y, tail_z], dtype=object)

        pcline_x, = ax_main.plot([pc_data[0,0], -self.lim], [pc_data[0,1],pc_data[0,1]], [pc_data[0,2],pc_data[0,2]], "-k", lw=0.5, alpha=0.5)
        pcline_y, = ax_main.plot([pc_data[0,0], pc_data[0,0]], [pc_data[0,1],self.lim], [pc_data[0,2],pc_data[0,2]], "-k", lw=0.5, alpha=0.5)
        pcline_z, = ax_main.plot([pc_data[0,0], pc_data[0,0]], [pc_data[0,1],pc_data[0,1]], [pc_data[0,2],-self.lim], "-k", lw=0.5, alpha=0.5)
        self.pc_lines = np.array([pcline_x, pcline_y, pcline_z], dtype=object)

        # # 2D quiver projections of the grad of sigma in pc and Lsigma
        # ax_main.quiver(pc_data[lf,0], pc_data[lf,1], -self.lim    , 
        #                0              , lsigma_data[lf,1], lsigma_data[lf,2],
        #                color="darkred", length=self.arr_len*2, normalize=True, alpha=1, zorder=2)
        # ax_main.quiver(pc_data[lf,0], self.lim     , pc_data[lf,2], 
        #                lsigma_data[lf,0], 0                 , lsigma_data[lf,2],
        #                color="darkred", length=self.arr_len*2, normalize=True, alpha=1, zorder=2)
        # ax_main.quiver(-self.lim    , pc_data[lf,1], pc_data[lf,2], 
        #                0              , lsigma_data[lf,1], lsigma_data[lf,2],
        #                color="darkred", length=self.arr_len*2, normalize=True, alpha=1, zorder=2)
        
        # ax_main.quiver(pc_data[lf,0], pc_data[lf,1], -self.lim    , 
        #                0            , grad_data[lf,1], grad_data[lf,2],
        #                color="k", length=self.arr_len*2, normalize=True, alpha=1, zorder=1)
        # ax_main.quiver(pc_data[lf,0], self.lim       , pc_data[lf,2], 
        #                grad_data[lf,0], 0            , grad_data[lf,2],
        #                color="k", length=self.arr_len*2, normalize=True, alpha=1, zorder=1)
        # ax_main.quiver(-self.lim    , pc_data[lf,1], pc_data[lf,2], 
        #                0            , grad_data[lf,1], grad_data[lf,2],
        #                color="k", length=self.arr_len*2, normalize=True, alpha=1, zorder=1)
        
        # -- Animation --
        # Init of the animation class
        anim = FuncAnimation(fig, self.animate, frames=tqdm(range(frames), initial=1, position=0), interval=1/fps*1000/4)

        # Generate and save the animation
        if gif:
            writer = PillowWriter(fps=15, bitrate=1800)
            anim.save(os.path.join(output_folder, "anim_ss.gif"),
                    writer = writer)
        else:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='Jesús Bautista Villar'), bitrate=10000)
            anim.save(os.path.join(output_folder, "anim_ss.mp4"), 
                    writer = writer)