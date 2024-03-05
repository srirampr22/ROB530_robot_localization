
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

from scipy.stats import multivariate_normal
from numpy.random import default_rng
# rng = default_rng()      # - If students want to see different results, they can unset the seed.
rng = default_rng(seed=3)  # - Set seed to get deterministic random number generator

class PF:
    # PF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        # PF parameters
        self.n = init.n
        self.Sigma = init.Sigma
        self.particles = init.particles
        self.particle_weight = init.particle_weight

        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u, particles, step):
        if step != 0:
            self.particles = particles
        ###############################################################################
        # TODO: Implement the prediction step for PF, remove pass                     #
        # Hint: Propagate your particles. Particles are saved in self.particles       #
        # Hint: Use rng.standard_normal instead of np.random.randn.                   #
        #       It is statistically more random.                                      #
        ###############################################################################
        # print("u: ", u.shape)
        # print("particles: ", particles)
        for i in range(self.n):
            # print(self.particles[:,i].shape)
            u = u + rng.standard_normal(3) @ self.M(u)
            # print("u: ", u.shape)
            self.particles[:,i] = self.gfun(self.particles[:,i], u)
            self.particles[:,i] = np.clip(self.particles[:,i], 0, self.n) 

        # print("particles: ", self.particles.size)

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        return self.particles


    def correction(self, z, landmarks, particles, particle_weight, step):
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))
        
        self.particles = particles
        if step != 0 :
            self.particle_weight = particle_weight
        
        ###############################################################################
        # TODO: Implement the correction step for PF                                  #
        # Hint: self.mean_variance() will update the mean and covariance              #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        # for i in range(self.n):
        w = self.importance_measurement(z, self.particles, landmark1, landmark2)
        # print("w: ", w.shape)
        # print("before particle_weight_sum: ", np.sum(self.particle_weight))
        self.particle_weight = np.multiply(self.particle_weight, w)
        self.particle_weight = self.particle_weight / np.sum(self.particle_weight)
        self.Neff = 1 / np.sum(np.power(self.particle_weight, 2))
        if self.Neff < self.n / 5:
            self.resampling()


        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        X, P = self.mean_variance()
        return np.copy(X), np.copy(P), np.copy(self.particles), np.copy(self.particle_weight)


    def resample(self):
        new_samples = np.zeros_like(self.particles)
        new_weight = np.zeros_like(self.particle_weight)
        W = np.cumsum(self.particle_weight)
        r = np.random.rand(1) / self.n
        count = 0
        for j in range(self.n):
            u = r + j/self.n
            while u > W[count]:
                count += 1
            new_samples[:,j] = self.particles[:,count]
            new_weight[j] = 1 / self.n
        self.particles = new_samples
        self.particle_weight = new_weight
    

    def mean_variance(self):
        X = np.mean(self.particles, axis=1)
        sinSum = 0
        cosSum = 0
        for s in range(self.n):
            cosSum += np.cos(self.particles[2,s])
            sinSum += np.sin(self.particles[2,s])
        X[2] = np.arctan2(sinSum, cosSum)
        zero_mean = np.zeros_like(self.particles)
        for s in range(self.n):
            zero_mean[:,s] = self.particles[:,s] - X
            zero_mean[2,s] = wrap2Pi(zero_mean[2,s])
        P = zero_mean @ zero_mean.T / self.n
        self.state_.setState(X)
        self.state_.setCovariance(P)
        return np.copy(X), np.copy(P)
    
    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

    def importance_measurement(self, z, particles, landmark1, landmark2):
        # compare important weight for each particle based on the obtain range and bearing measurements
        # Inputs:
        # z: measurement
        w = np.zeros([self.n, 1]) # importance weights
        for i in range(self.n):
            z_hat1 = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], particles[:,i])
            z_hat2 = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], particles[:,i])
            z_hat = np.concatenate((z_hat1.reshape(-1, 1), z_hat2.reshape(-1, 1)), axis=0)
            z_reformatted = np.array([z[0], z[1], z[3], z[4]])
            v = z_reformatted - z_hat.flatten()
            w[i] = multivariate_normal.pdf(v,np.array([0,0,0,0]), block_diag(self.Q, self.Q))
            # print("w: ", w[i].shape)

        return w

