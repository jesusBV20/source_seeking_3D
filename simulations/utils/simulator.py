"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np

# -- Numerical tools --
from simulations.utils.tools_math import *

class simulator:
    def __init__(self, p0, R0, v0, dt=1/60, kw=1):
        # Initial state
        self.R = np.array(R0)
        self.p = np.array(p0)
        self.v = v0
        self.N = self.R.shape[0]

        # Controller variables and parameters
        self.kw = kw
        self.set_R_desired(self.R)
        self.set_R_desired_dot(np.zeros((3,3)))

        # Integrator parameters
        self.dt = dt
        self.data = {"R": None, "p": None, "theta_e": None, "pc":None}
        self.update_data()

    """\
    - Compute and give the centroid position -
    """
    def get_pc(self):
        return np.mean(self.p, axis=0)

    """\
    - Update the data dictionary -
    """
    def update_data(self):
        self.data["R"] = self.R
        self.data["p"] = self.p
        self.data["theta_e"] = self.theta_e
        self.data["pc"] = self.get_pc()

    """\
    - Set the desired body frame orientation -
    """
    def set_R_desired(self, input_matrix, n=None): 
        if n is None and len(input_matrix.shape) == 3:
            self.Rd = np.copy(input_matrix)
            self.error_rot()
        elif n is None and len(input_matrix.shape) == 2:
            self.Rd = np.ones(self.R.shape) * np.copy(input_matrix)
            self.error_rot()
        elif n is not None and len(input_matrix.shape) == 2:
            self.Rd[n,:] = np.copy(input_matrix)
            self.error_rot()
        else:
            print("ERROR: can not set the desired Rd matrix. Wrong shape! -")

    """\
    - Set the rate of change of the desired body frame orientation -
    """
    def set_R_desired_dot(self, input_matrix, n=None): 
        if n is None and len(input_matrix.shape) == 3:
            self.Rd_dot = input_matrix
        elif n is None and len(input_matrix.shape) == 2:
            self.Rd_dot = np.ones(self.R.shape) * input_matrix
        elif n is not None and len(input_matrix.shape) == 2:
            self.Rd_dot[n,:] = input_matrix
        else:
            print("ERROR: can not set the desired Rd_dot matrix. Wrong shape! -")

    """\
    - Computate the orientation error for every body frame  -
    """
    def error_rot(self):
        self.theta_e = np.zeros(self.N)
        self.Re = np.zeros(self.R.shape)
        for n in range(self.N):
            # Rotation error matrix
            self.Re[n,...] = self.Rd[n,...].T @ self.R[n,...]

            # Get the angle error by computing the angle distance of Re
            self.theta_e[n] = theta_distance_from_R(self.Re[n,...])
    
    """\
    - 3D rotation controller  -
    """
    def rot_controller(self):
        log_Re = np.zeros(self.R.shape)
        omega_hat = np.zeros(self.R.shape)
        for n in range(self.N):
            log_Re[n,...] = log_map_of_R(self.Re[n,...])

            # If Rd_dot != I then apply the feedback controller 
            if not np.allclose(self.Rd_dot[n,...],np.zeros((3,3))):
                omega_hat[n,...] = self.R[n,...].T @ self.Rd_dot[n,...] @ self.Re[n,...]
                
        # Proportional controller
        omega_hat = - self.kw * log_Re + omega_hat

        return omega_hat

    """\
    - Euler step integration  -
    """
    def int_euler(self):
        self.error_rot()
        omega_hat = self.rot_controller()

        # Rotation integrator
        for n in range(self.N):
            # Compute the exponential map
            exp_dt_omega = exp_map_of_R(self.dt*omega_hat[n,...])
            # Apply the omega rotation matrix
            self.R[n,...] = self.R[n,...] @ exp_dt_omega

            # Position integrator
            self.p[n,:] = self.p[n,:] + self.dt*self.v * self.R[n,0,:]

        # Update data
        self.update_data()