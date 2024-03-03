import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy, copy

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

class EKF:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.Gfun = init.Gfun  # Jocabian of motion model
        self.Vfun = init.Vfun  # Jocabian of motion model
        self.Hfun = init.Hfun  # Jocabian of measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance

        self.state_ = RobotState()

        # init state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)


    ## Do prediction and set state in RobotState()
    def prediction(self, u, X, P, step):
        if step == 0:
            X = self.state_.getState()
            P = self.state_.getCovariance()
        else:
            X = X
            P = P
        ###############################################################################
        # TODO: Implement the prediction step for EKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################

        X_pred = self.gfun(X, u)
        G = self.Gfun(X, u)
        V = self.Vfun(X, u)
        P_pred = G @ P @ G.T + V @ self.M(u) @ V.T

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)
        return np.copy(X_pred), np.copy(P_pred)


    def correction(self, z, landmarks, X, P):
        # EKF correction step
        #
        # Inputs:
        #   z:  measurement
        X_predict = X
        P_predict = P
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        z_hat_k1 = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], X_predict)
        z_hat_k2 = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], X_predict)
        H_1 = self.Hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], X_predict, z_hat_k1)
        H_2 = self.Hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], X_predict, z_hat_k2)

        #need to compute the stacked measurement jacobian H here
        H = np.vstack((H_1, H_2))
        z_hat = np.concatenate((z_hat_k1.reshape(-1, 1), z_hat_k2.reshape(-1, 1)), axis=0)

        z_reformatted = np.array([z[0], z[1], z[3], z[4]])

        innovaiton = z_reformatted - z_hat.flatten()

        Q = block_diag(self.Q, self.Q)
        S = H @ P_predict @ H.T + Q

        K = P_predict @ H.T @ np.linalg.inv(S)

        X = X_predict + K @ innovaiton

        P = (np.eye(3) - K @ H) @ P_predict

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X)
        self.state_.setCovariance(P)
        return np.copy(X), np.copy(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state