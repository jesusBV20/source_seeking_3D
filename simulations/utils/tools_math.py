"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np


def rot_3d_matrix(psi, theta, phi, dec=2):
    """
    - Generate R ∈ SO(3) composing ROLL, PITCH and YAW rotation matrices -
    """
    # ROLL
    Rx = np.array([[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])
    # PITCH
    Ry = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]) 
    # YAW
    Rz = np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]]) 
    
    R = Rx @ Ry @ Rz
    return np.round(R, decimals=dec)

def get_orthonormal_to_v(v):
    """
    - Select one of the possible perpendicular vector to v ∈ R^3 -
    """
    vx, vy, vz = v[0], v[1], v[2]

    # Capture the singularity
    if (vz < -0.99999999):
        n = np.array([0,-1,0])
    
    # Perpendicular vector computation
    else:
        a = 1/(1 + vz)
        b = -vx*vy*a
        n = np.array([1 - a*vx**2, b, -vx])
    
    return n

def get_R_from_v(v):
    """
    - Given the input vector v, build an orthonormal basis and codify into a rotation matrix R ∈ SO(3) -
    """
    # Normalization of v
    md = v / np.linalg.norm(v)

    # Get an arbitrary (fixed) perperdicular vector
    md_z = -get_orthonormal_to_v(md)

    # Compute the las orthogonal vector
    md_y = np.cross(md_z, md)

    # Build the rotation matrix
    R = np.array([md, md_y, md_z])
    return R / np.linalg.det(R)

###################################################################
## Isomorphism computation: rotation vector \omega <-> so(3) ######

def so3_hat(omega):
    """
    - Generate \omega_\hat ∈ so(3) from the \omega vector -
    """
    wx, wy, wz = omega[0], omega[1], omega[2]
    return  np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]])

# Generate \omega vector from \omega_\hat ∈ so(3)
def so3_vee(omega_hat):
    wx, wy, wz = omega_hat[2,1], omega_hat[0,2], omega_hat[1,0]
    return  wx, wy, wz

###################################################################

def theta_distance_from_R(R):
    """
    - Compute the distance in the tangent plane (theta) corresponding to a given R ∈ SO(3) -
    """
    # We know that Tr(R) = 1 - 2 cosθ  if R ∈ SO(3)
    cos_theta = (np.trace(R) - 1)/2

    # The approximation of the exponential map may yield to 
    # rotation matrices outside SO(3). This is why we can fin
    # values of |\cos(θ)| greather than 1. We capture this
    # error as follows
    cos_theta = np.where(cos_theta > 1, 1, cos_theta)
    cos_theta = np.where(cos_theta < -1, -1, cos_theta)

    # So we can compute the distance |θ| as follows
    theta = np.arccos(cos_theta)
    return theta

def exp_map(A,n=6):
    """
    - Compute the exponencial map corresponding to a given tau^\wedge ∈ so(3) -
    """
    # Given Exp(A) = \sum_{k=0}^{\infty} \frac{A^k}{k!} as one 
    # possible formal definition of the exponential. We can easily
    # aproximate the exponential as follows:
    #           Exp(A) = I + A + A^2/2! + ... + A^n/n! + O(A^n)
    exp_A, A_i = np.eye(3), np.eye(3)
    for i in range(n):
        A_i = A_i @ A
        exp_A = exp_A + A_i / np.math.factorial(i+1)
        
    return exp_A

def log_map_of_R(R, n=5):
    """
    - Compute the logaritmic map corresponding to a given R ∈ SO(3) -
    """
    log_R = np.zeros((3,3))
    theta = theta_distance_from_R(R)
    
    if theta > 0.96*np.pi:
        omega_pi = np.sqrt((np.array([R[0,0], R[1,1], R[2,2]])+1)/2)
        log_R = so3_hat(omega_pi)

    elif theta > np.pi/6:
        # Compute the logarithmic map by making small rotations and calculating the log map centered
        # on different tangent spaces
        
        Ri = np.eye(3)
        for i in range(n):
            # Since we compute the rotation vector \omega_i ()
            Reval = Ri.T@R
            theta_i = theta_distance_from_R(Reval)
            
            # Compute \theta_i / 2sin(\theta_i) -- (Rotation angle from Ri to R)
            theta_sin = theta_i/2 / np.sin(theta_i)

            # But rotate just (n - i)^-1 times \theta_i. Note that when i = n-1 we rotate \theta_i
            theta_sin = theta_sin / (n - i)

            # Then compute the \omega_hat of this rotation (from the tangent plane of Ri)
            log_Ri_Ri = theta_sin * (Reval - Reval.T)
            
            # Move this \omega_hat to the tangent plane of I (Lie Algebra): compute the Adjoint map
            log_Ri_I = Ri.T@log_Ri_Ri@Ri

            # Add the computed rotation to the last one (both are now in the Lie Algebra)
            log_R = log_R + log_Ri_I

            # Apply the whole rotation and compute the new rotation matrix
            Ri = Ri@exp_map(log_Ri_I)

    else:
        # Compute the log map directrly into the tanget plane of I

        # Since theta is small, we can compute \theta / 2sin(\theta)
        # by approximating the expression using a truncated Taylor series:
        theta_sin = 1/2 + theta**2/12 + 7*theta**4/720 # O(θ^6)
        log_R = theta_sin * (R - R.T)

    # Clean the diagonal (elements may not be zero due to computational errors)
    log_R[0,0], log_R[1,1], log_R[2,2] = 0, 0, 0

    return log_R