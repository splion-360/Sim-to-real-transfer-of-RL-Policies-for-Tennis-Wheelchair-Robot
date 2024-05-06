### Estimate the trajectory of the ball using Extended Kalman Fitler 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Input States, (p,v) 6X1 vector
## Magnus effect is ignored and Inertia body model is used in mujoco instead of ellipsoid model

class Trajectory:
    def __init__(self,n_controls = 3, sensor_variance = 1, Cd = 0.55, \
                 rho = 1.29, radius = 0.0343, process_variance = np.ones(3)):
        assert n_controls == process_variance.shape[0]
        
        self.n_controls = n_controls
        self.R = sensor_variance
        self.Cd = Cd ## Tennis Ball drag coefficient
        self.rho = rho ## Air density at NTP
        self.radius = radius ## Radius of a standard tennis ball 
        self.Q = np.diag(process_variance)

        ## Some constants which remain the same during the update
        self.g = np.array([0,0,-9.81])
        self.H = np.zeros((n_controls,6)) ## Measurement model jacobian
        self.H[:,:3] = np.eye(n_controls)
        self.L = np.zeros((6,n_controls))
        self.L[3:,:] = np.eye(n_controls)

        ## Other data structures
        self.P_prev = np.zeros((6,6))
        self.p_prev = np.zeros(6)

    def initialize(self, p_init, P_init = None):
        self.p_prev= p_init
        if P_init is not None:
            self.P_prev = P_init

    def correct(self,y_k:np.array,p_check:np.array, P_check:np.array):
        ## Kalman Gain
        K = P_check @ self.H.T @ np.linalg.inv(self.H @ P_check @ self.H.T + self.R * np.eye(self.n_controls))
        ## Compute Error State
        x_check,v_check = p_check[:3], p_check[3:]
        delta_x = K @ (y_k - x_check)

        ## Compute the corrected state 
        x_hat = x_check + delta_x[:3]
        v_hat = v_check + delta_x[3:]

        ## Corrected Covariance Matrix
        P_hat = (np.eye(6) - (K @ self.H)) @ P_check

        self.P_prev = P_hat
        self.p_prev = np.concatenate((x_hat,v_hat))
        return np.concatenate((x_hat,v_hat)), P_hat

    def predict(self, delta_t):
        ## Update the states using the motion model
        x_prev, v_prev = self.p_prev[:3],self.p_prev[3:]
        x_check,v_check = np.zeros(3), np.zeros(3)

        kD = 0.5 * self.Cd * self.rho * np.pi * (self.radius**2)
        x_check[:] = x_prev + (v_prev * delta_t)
        v_check[:] = v_prev[:] - kD*delta_t*np.linalg.norm(v_prev)*v_prev[:] + self.g * delta_t

        ## Linearize motion model and compute jacobian
        F = np.zeros((6,6))
        F[:3,:3] = np.eye(3)
        F[:3:,3:] = np.eye(3) * delta_t
        
        norm = np.linalg.norm(v_prev) + 1e-7
        F[3][3] = 1 - kD * delta_t * (norm * v_prev[0]**2) * (1/norm)
        F[4][4] = 1 - kD * delta_t * (norm * v_prev[1]**2) * (1/norm)
        F[5][5] = 1 - kD * delta_t * (norm * v_prev[2]**2) * (1/norm)
        F[3][4] = - kD * delta_t * v_prev[1] * (1/norm) 
        F[3][5] = - kD * delta_t * v_prev[2] * (1/norm)
        F[4][3] = - kD * delta_t * v_prev[0] * (1/norm) 
        F[4][5] = - kD * delta_t * v_prev[2] * (1/norm)
        F[5][3] = - kD * delta_t * v_prev[0] * (1/norm) 
        F[5][4] = - kD * delta_t * v_prev[1] * (1/norm) 
        ## Control input will be the velocities
        P_check = F @ self.P_prev @ F.T + self.L @ self.Q @ self.L.T

        return np.concatenate((x_check,v_check)), P_check

if __name__ == "__main__":
    filter = Trajectory()
    breakpoint()