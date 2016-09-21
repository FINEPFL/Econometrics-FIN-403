from __future__ import print_function
from scipy.stats import norm
from numpy.linalg import inv

import numpy as np
import matplotlib.pyplot as plt

class unbiasednessSimulate:
    ''' a monte carlo simulation for proofing the
    unbiasedness of the least square estimator '''

    def __init__(self, numObs = 10, numSims = 10000):
        ''' parameters initialisation '''

        self.numSims = numSims
        self.numObs = numObs

    def DGP(self):
        ''' define data generation processing '''

        x2 = norm.rvs(loc = 0, scale = 1, size = self.numObs).T
        self.epsl = norm.rvs(loc = 0, scale = 1, size = self.numObs).T
        self.beta = np.asarray([1, 2]).T
        self.X = np.ones((self.numObs, 2))
        self.X[:, 1] = x2
        self.y = np.dot(self.X, self.beta) + self.epsl

    def similuting(self):
        ''' repeat simulationg for numSims times '''
        b2 = []
        for _ in xrange(0, self.numSims):
            b = np.dot(np.dot(inv(np.dot(self.X.T, self.X)),self.X.T), self.y)
            self.DGP()
            b2.append(b[1])
        self.b2 = b2

    def plotting(self):
        # plt.scatter(np.linspace(1, self.numSims, num=self.numSims), self.b2)
        _, bins, _ = plt.hist(self.b2, 50, normed=1, facecolor='green', alpha=0.75)
        plt.xlabel('b2')
        plt.ylabel('frequency')
        plt.title('distribution of the estimator b2')
        plt.show()


if __name__ == '__main__':
    ubp = unbiasednessSimulate()
    ubp.DGP()
    ubp.similuting()
    ubp.plotting()
