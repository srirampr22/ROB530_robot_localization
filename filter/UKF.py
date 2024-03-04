
from scipy.linalg import block_diag
from copy import deepcopy, copy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi


class UKF:
    # UKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        self.n = 6
        self.kappa_g = init.kappa_g
        # self.Y_temp = []
        
        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)
        


    def prediction(self, u, X, P , step):
        # prior belief
        if step == 0:
            mean = self.state_.getState()
            sigma = self.state_.getCovariance()
        else:
            mean = X
            sigma = P
        # print("mean", mean.shape)
        # print("sigma", sigma.shape)
        ###############################################################################
        # TODO: Implement the prediction step for UKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        self.Y = []
        X_pred = np.zeros(3,)
        P_pred = np.zeros((3,3))
        M = self.M(u)
        Q = self.Q
        augmented_mean = np.hstack((mean, np.zeros((M.shape[0],))))
        n = len(mean)   
        m = M.shape[0]  
        augmented_cov = np.zeros((n + m, n + m))
        augmented_cov = block_diag(sigma, M)

        self.sigma_point(augmented_mean.reshape(-1,1), augmented_cov, self.kappa_g)
        
        for i in range(2*self.n + 1):
            Yvalue = self.gfun(self.X[:3, i], self.X[3:, i] + u)
            self.Y.append(Yvalue)
        
        # print("self.w", self.w.shape)
        self.Y = np.array(self.Y).T
        # print("self.Y", self.Y.shape)
        predicted_mean = self.w * self.Y
        # print("predicted_mean", predicted_mean.shape)
        predicted_mean = np.sum(predicted_mean, axis=1)
        # print(predicted_mean.shape)

        
        temp = self.Y - predicted_mean.reshape(-1,1)
        predicted_Cov = np.dot(np.dot(temp, np.diag(self.w)), temp.T)
        P_pred = predicted_Cov
        X_pred = predicted_mean
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)
        return np.copy(self.Y), np.copy(self.w), np.copy(X_pred), np.copy(P_pred)

    def correction(self, z, landmarks, Y, w, X, P):

        X_predict = X
        P_predict = P
        self.Y = Y
        self.w = w        
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        Z = []
        print("Y", Y.shape)
        for i in range(2*self.n + 1):
            Z_dash_1 = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], Y[:,i])
            Z_dash_2 = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], Y[:,i])
            Z_dash = np.concatenate((Z_dash_1, Z_dash_2))
            Z.append(Z_dash)
            # z_hat = X_predict + self.w[i] * Z_dash
            # print("z_hat", z_hat.shape)

        Z = np.array(Z).T
        z_hat = self.w * Z
        z_hat = np.sum(z_hat, axis=1)
        # print("predicted_measurement_mean", z_hat.shape)

        temp_S = Z - z_hat.reshape(-1,1)
        S = np.dot(np.dot(temp_S, np.diag(self.w)), temp_S.T) + block_diag(self.Q,self.Q)
        # print("predicted_measurement_cov", S.shape)

        temp_C = self.Y - X_predict.reshape(-1,1)
        C = np.dot(np.dot(temp_C, np.diag(self.w)), temp_S.T)
        # print("Cross covariance", C.shape)

        K = np.dot(C, np.linalg.inv(S))
        # print("Kalman Gain", K.shape)
        # print("z", z.shape)
        z_measurements = np.array([z[0], z[1], z[3], z[4]])
        # print("z_hat", z_hat.shape)
        innovation = z_measurements - z_hat
        updated_mean = X_predict + np.dot(K, innovation)
        updated_cov = P_predict - np.dot(np.dot(K, S), K.T)
        X = updated_mean
        P = updated_cov

        # Z = np.array(Z).T
        # temp = Z - z_hat.reshape(-1,1)
        # predicted_measurement_cov = np.dot(np.dot(temp, np.diag(self.w)), temp.T)
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X)
        self.state_.setCovariance(P)
        return np.copy(X), np.copy(P)

    def sigma_point(self, mean, cov, kappa):
        self.n = len(mean) # dim of state
        L = np.sqrt(self.n + kappa) * np.linalg.cholesky(cov)
        Y = mean.repeat(len(mean), axis=1)
        self.X = np.hstack((mean, Y+L, Y-L))
        self.w = np.zeros([2 * self.n + 1, 1])
        self.w[0] = kappa / (self.n + kappa)
        self.w[1:] = 1 / (2 * (self.n + kappa))
        self.w = self.w.reshape(-1)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state