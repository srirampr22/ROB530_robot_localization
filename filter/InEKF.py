
from mimetypes import init
from os import stat
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi, wedge

# import InEKF lib
from scipy.linalg import logm, expm


class InEKF:
    # InEKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        # self.hfun = system.hfun  # measurement model
        # self.Gfun = init.Gfun  # Jocabian of motion model
        # self.Vfun = init.Vfun  
        # self.Hfun = init.Hfun  # Jocabian of measurement model
        self.W = system.W # motion noise covariance
        self.V = system.V # measurement noise covariance
        
        self.mu = init.mu
        self.Sigma = init.Sigma
        self.mu_pred = np.zeros((3,3))
        self.sigma_pred = np.zeros((3,3))
        self.X_pred = np.zeros((3,3))
        self.P_pred = np.zeros((3,3))

        self.state_ = RobotState()
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])])
        # print("X", X)
        # print("mu", self.mu)
        self.state_.setState(X)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u, Sigma, mu, step):
        if step != 0 :
            self.Sigma = Sigma
            self.mu = mu
        state_vector = np.zeros(3)
        state_vector[0] = self.mu[0,2]
        state_vector[1] = self.mu[1,2]
        state_vector[2] = np.arctan2(self.mu[1,0], self.mu[0,0])
        H_prev = self.pose_mat(state_vector)
        state_pred = self.gfun(state_vector, u)
        H_pred = self.pose_mat(state_pred)

        u_se2 = logm(np.linalg.inv(H_prev) @ H_pred)

        ###############################################################################
        # TODO: Propagate mean and covairance (You need to compute adjoint AdjX)      #
        ###############################################################################
        # print(H_pred, H_pred.shape)
        # print("mu", self.mu.shape)
        # print("Sigma", self.Sigma.shape)

        # Extract R11, R12, R21, R22, x, and y from H_pred
        R11, R12, x = H_pred[0, 0], H_pred[0, 1], H_pred[0, 2]
        R21, R22, y = H_pred[1, 0], H_pred[1, 1], H_pred[1, 2]

        adjX = np.array([[R11, R12, x], [R21, R22, y], [0, 0, 1]])

        # Propagate mean and covarince
        self.mu_pred, self.sigma_pred = self.propagation(u_se2, adjX, self.mu, self.Sigma, self.W)
        
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        return np.copy(self.mu_pred), np.copy(self.sigma_pred)

    def propagation(self, u, adjX, mu, Sigma , W):
        self.mu = mu
        self.Sigma = Sigma
        self.W = W
        ###############################################################################
        # TODO: Complete propagation function                                         #
        # Hint: you can save predicted state and cov as self.X_pred and self.P_pred   #
        #       and use them in the correction function                               #
        ###############################################################################

        self.mu_pred = self.mu @ expm(u)
        self.X_pred = np.array([self.mu_pred[0,2], self.mu_pred[1,2], np.arctan2(self.mu_pred[1,0], self.mu_pred[0,0])])
        # how do i compute A which is the state transition matrix (or error dynamics matrix) this is right inavriant ekf
        self.sigma_pred = self.Sigma + np.dot(np.dot(adjX, self.W), adjX.T)
        self.P_pred = self.sigma_pred
        # print("mu_pred", mu_pred)
        # print("sigma_pred", sigma_pred)
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        return np.copy(self.mu_pred), np.copy(self.sigma_pred)
        
    def correction(self, Y1, Y2, z, landmarks, mu_pred, sigma_pred):
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))
        self.mu_pred = mu_pred
        self.sigma_pred = sigma_pred
        ###############################################################################
        # TODO: Implement the correction step for InEKF                               #
        # Hint: save your corrected state and cov as X and self.Sigma                 #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################

        m1 = landmark1.getPosition()  # [m1_x, m1_y]
        m2 = landmark2.getPosition()  # [m2_x, m2_y]
        # Construct the stacked measurement Jacobian matrix H
        # print("mu_pred", mu_pred.shape)
        # print("X_pred", self.X_pred.shape)
        H = np.array([
            [m1[1], -1, 0],
            [-m1[0], 0, -1],
            [m2[1], -1, 0],
            [-m2[0], 0, -1]
        ])
        # print("V", self.V)
        V = block_diag(self.V, 0)
        N = np.dot(np.dot(self.mu_pred, V), self.mu_pred.T)
        N = block_diag(N[0:2, 0:2], N[0:2, 0:2])
        # print("N", N.shape)

        S = np.dot(np.dot(H, self.sigma_pred), H.T) + N
        L = np.dot(np.dot(self.sigma_pred, H.T), np.linalg.inv(S))

        # Inverse of the predicted state
        mu_inv = np.linalg.inv(self.mu_pred)

        # Compute the expected measurement for each landmark
        b1 = mu_inv @ np.array([landmark1.getPosition()[0], landmark1.getPosition()[1], 1])
        b2 = mu_inv @ np.array([landmark2.getPosition()[0], landmark2.getPosition()[1], 1])

        # Stacked expected measurements (only take the first two components for each landmark)
        expected_Y = np.vstack((b1[:2], b2[:2])).reshape(-1, 1)

        # Actual measurements stacked
        Y1_positions = Y1[:-1].reshape(-1, 1)  # This selects only the x, y positions from Y1
        Y1_positions = Y2[:-1].reshape(-1, 1) 
        actual_Y = np.vstack((Y1_positions, Y1_positions))
        # print("Y1", Y1.reshape(-1, 1))

        # Compute the innovation vector (nu)
        # Here, you need to consider the measurement noise Vk, which is typically added to the covariance matrix during the update
        nu = actual_Y - expected_Y

        delta = wedge(np.dot(L, nu)) # innovation in the spatial frame
        self.mu = np.dot(expm(delta), self.mu_pred)
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])])

        # Update Covariance
        I = np.eye(np.shape(self.sigma_pred)[0])
        temp = I - np.dot(L, H)
        self.Sigma = np.dot(np.dot(temp, self.sigma_pred), temp.T) + np.dot(np.dot(L, N), L.T)
        # Compute the stacked innovation
        print("self.Sigma", self.Sigma)

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        self.state_.setState(X)
        self.state_.setCovariance(self.Sigma)
        return np.copy(X), np.copy(self.Sigma), np.copy(self.mu)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

    def pose_mat(self, X):
        x = X[0]
        y = X[1]
        h = X[2]
        H = np.array([[np.cos(h),-np.sin(h),x],\
                      [np.sin(h),np.cos(h),y],\
                      [0,0,1]])
        return H
    
    # def wedge(self, x):
    #     # wedge operation for se(2) to put an R^3 vector to the Lie algebra basis
    #     G1 = np.array([[0, -1, 0],
    #     [1, 0, 0],
    #     [0, 0, 0]]) # omega
    #     G2 = np.array([[0, 0, 1],
    #     [0, 0, 0],
    #     [0, 0, 0]]) # v_1
    #     G3 = np.array([[0, 0, 0],
    #     [0, 0, 1],
    #     [0, 0, 0]]) # v_2
    #     xhat = G1 * x[0] + G2 * x[1] + G3 * x[2]
    #     return xhat
