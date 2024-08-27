"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np

# -- Numerical tools --
from .tools_math import *

class simulator:
    def __init__(self, p0, R0, v0, dt=1/60, kw=1):
        # Initial state
        self.R = np.array(R0)
        self.p = np.array(p0)
        self.v = v0
        self.N = self.R.shape[0]

        # Controller variables and parameters
        if isinstance(kw,list):
            self.kw = np.array(kw)
        else:
            self.kw = np.array([kw for n in range(self.N)])

        self.set_R_desired(self.R)
        self.set_R_desired_dot(np.zeros((3,3)))

        # Integrator parameters
        self.dt = dt
        self.data = {"R": None, "p": None, "theta_e": None, "delta": None,
                     "pc": None, "Re": None, "log_Re_vee": None}
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
        self.data["delta"] = self.delta
        self.data["pc"] = self.get_pc()
        self.data["Re"] = self.Re
        self.data["log_Re_vee"] = self.log_Re_vee

    """\
    - Set the desired body frame orientation -
    """
    def set_R_desired(self, input_matrix, n=None): 
        if n is None and len(input_matrix.shape) == 3:
            self.Ra = np.copy(input_matrix)
            self.error_rot()
        elif n is None and len(input_matrix.shape) == 2:
            self.Ra = np.ones(self.R.shape) * np.copy(input_matrix)
            self.error_rot()
        elif n is not None and len(input_matrix.shape) == 2:
            self.Ra[n,:] = np.copy(input_matrix)
            self.error_rot()
        else:
            print("ERROR: can not set the desired Ra matrix. Wrong shape! -")

    """\
    - Set the rate of change of the desired body frame orientation -
    """
    def set_R_desired_dot(self, input_matrix, n=None): 
        if n is None and len(input_matrix.shape) == 3:
            self.Ra_dot = input_matrix
        elif n is None and len(input_matrix.shape) == 2:
            self.Ra_dot = np.ones(self.R.shape) * input_matrix
        elif n is not None and len(input_matrix.shape) == 2:
            self.Ra_dot[n,:] = input_matrix
        else:
            print("ERROR: can not set the desired Ra_dot matrix. Wrong shape! -")

    """\
    - Computate the orientation error for every body frame  -
    """
    def error_rot(self):
        self.Re = np.zeros(self.R.shape)
        self.log_Re = np.zeros(self.R.shape)
        self.log_Re_vee = np.zeros(self.p.shape)
        self.theta_e = np.zeros(self.N)
        self.delta = np.zeros(self.N)

        for n in range(self.N):
            # Rotation error matrix
            self.Re[n,...] = self.Ra[n,...].T @ self.R[n,...]
            self.log_Re[n,...] = log_map_of_R(self.Re[n,...])
            self.log_Re_vee[n,...] = so3_vee(self.log_Re[n,...])

            # Get the angle error by computing the angle distance of Re
            self.theta_e[n] = theta_distance_from_R(self.Re[n,...])

            # Get the angle between x and x_a (delta)
            cos_delta = self.Ra[n,:,0].T @ self.R[n,:,0]
            cos_delta = np.where(cos_delta > 1, 1, cos_delta)
            cos_delta = np.where(cos_delta < -1, -1, cos_delta)
            self.delta[n] = np.arccos(cos_delta)

    """\
    - 3D rotation controller  -
    """
    def rot_controller(self):
        omega_hat = np.zeros(self.R.shape)
        for n in range(self.N):
            # If Ra_dot != I then apply the feedback controller 
            if not np.allclose(self.Ra_dot[n,...],np.zeros((3,3))):
                omega_hat[n,...] = self.R[n,...].T @ self.Ra_dot[n,...] @ self.Re[n,...]
                
            # Proportional controller
            omega_hat[n,...] = - self.kw[n] * self.log_Re[n,...] + omega_hat[n,...]

        return omega_hat

    """\
    - Euler step integration  -
    """
    def int_euler(self):
        self.error_rot()
        omega_hat = self.rot_controller()

        # Euler integration
        for n in range(self.N):
            # Compute the exponential map
            exp_dt_omega = exp_map(self.dt*omega_hat[n,...])
            
            # Rotation integrator
            self.R[n,...] = self.R[n,...] @ exp_dt_omega

            # Position integrator
            self.p[n,:] = self.p[n,:] + self.dt*self.v * self.R[n,0,:]

        # Update data
        self.update_data()