"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np

"""\
- Generate R ∈ SO(3) composing ROLL, PITCH and YAW rotation matrices -
"""
def rot_3d_matrix(alfa, beta, gamma, dec=2):
    # ROLL
    Rx = np.array([[1,0,0],[0,np.cos(gamma),-np.sin(gamma)],[0,np.sin(gamma),np.cos(gamma)]])
    # PITCH
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]]) 
    # YAW
    Rz = np.array([[np.cos(alfa),-np.sin(alfa),0],[np.sin(alfa),np.cos(alfa),0],[0,0,1]]) 
    
    R = Rx @ Ry @ Rz
    return np.round(R, decimals=dec)

"""\
- Compute the vector \omega corresponding to a given R ∈ SO(3) -
"""
def omega_from_R(R):
    # Compute the eigenvalues and eigenvectors Rv = λ v
    eigval, eigvec = np.linalg.eig(R)

    # It is known that \omega lies in the null space of (R - I),
    # so we can find \omega looking for the eigenvector of R
    # corresponding to the eigenvalue λ = 1 
    omega = eigvec[:,np.abs(eigval.real - 1) < 0.01].real
    return  omega

"""\
- Generate \omega_\hat ∈ so(3) from the \omega vector -
"""
def omega_hat_from_omega(omega):
    wx, wy, wz = omega[0], omega[1], omega[2]
    return  np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]])

"""\
- Compute the distance in the tangent plane (theta) corresponding to a given R ∈ SO(3) -
"""
def theta_distance_from_R(R):
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

"""\
- Compute the exponencial map corresponding to a given R ∈ SO(3) -
"""
def exp_map_of_R(R,n=6):
    # Given Exp(R) = \sum_{k=0}^{\infty} \frac{R^k}{k!} as one 
    # possible formal definition of the exponential. We can easily
    # aproximate the exponential as follows:
    #           Exp(R) = I + R + R^2/2! + ... + R^n/n! + O(R^n)
    exp_R, R_i = np.eye(3), np.eye(3)
    for i in range(n):
        R_i = R_i @ R
        exp_R = exp_R + R_i / np.math.factorial(i+1)
        
    return exp_R

"""\
- Compute the logaritmic map corresponding to a given R ∈ SO(3) -
"""
def log_map_of_R(R):
    # Fist, we compute the angular distance |θ|
    theta = theta_distance_from_R(R)

    if theta > np.pi*0.93:
        return log_map_pi(R)
        # omega = np.sqrt(np.abs((np.array([R[0,0],R[1,1],R[2,2]]) + 1))/2)
        # return omega_hat_from_omega(omega)

    # Let us approximate \frac{θ}{2 sin θ} with Euler if θ is small
    if theta < np.pi/4:
        theta_sin = 1/2 + theta**2/12 + 7*theta**4/720 # O(θ^6)
    else:
        theta_sin = theta/2/np.sin(theta)
    
    # And then we can easily compute the log(R) = \frac{θ}{2 sin θ} * (R - R^T)
    log_R = theta_sin * (R - R.T)
    return log_R
    

def log_map_pi(R):
    #source: https://github.com/nurlanov-zh/so3_log_map/blob/main/so3_log_map_analysis.ipynb

    trR = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = max(min(0.5 * (trR - 1), 1), -1)
    sin_theta = 0.5 * np.sqrt(max(0, (3 - trR) * (1 + trR)))
    theta = np.arctan2(sin_theta, cos_theta)
    R_minus_R_T_vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    S = R + R.transpose() + (1 - trR) * np.eye(3)
    rest_tr = 3 - trR
    n = np.ones(3)

    # Fix modules of n_i
    for i in range(3):
        n[i] = np.sqrt(max(0, S[i, i] / rest_tr))
    max_i = np.argmax(n)

    # Fix signs according to the sign of max element
    for i in range(3):
        if i != max_i:
            n[i] *= np.sign(S[max_i, i])

    # Fix an overall sign
    if any(np.sign(n) * np.sign(R_minus_R_T_vee) < 0):
        n = -n
    omega = theta * n

    # Build \omega_\hat
    return omega_hat_from_omega(omega) 

"""\
- Select one of the possible perpendicular vector to v ∈ R^3 -
"""
def get_orthonormal_to_v(v):
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

"""\
- Given the input vector v, build an orthonormal basis and codify into a rotation matrix R ∈ SO(3) -
"""
def get_R_from_v(v):
    # Normalization of v
    md = v / np.linalg.norm(v)

    # Get an arbitrary (fixed) perperdicular vector
    md_z = -get_orthonormal_to_v(md)

    # Compute the las orthogonal vector
    md_y = np.cross(md_z, md)

    # Build the rotation matrix
    R = np.array([md, md_y, md_z])
    return R / np.linalg.det(R)