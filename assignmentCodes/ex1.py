from numpy.random import uniform
from numpy.random import normal
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np

''' Computational exercise ii of problem set 1, econometrics, EPFL '''
''' Default parameters:
        number of observations n = 20
        number of dimensions d = 3
        error term epsl ~ N(0, 4)
        variables ~ U(0, 10)
        beta = [0.5, 0.8, 1.3]'
'''
class DGP():
    ''' Data generator and estimator of beta'''
    def __init__(self, beta, numOb=4, numDim=3, epslMean=0, epslVar=4,
                    variMean=0, varVar=10):
        self.beta = beta
        self.numOb = numOb
        self.numDim = len(beta)
        self.epslMean = epslMean
        self.epslVar = epslVar
        self.variMean = variMean
        self.varVar = varVar

    def generator(self):
        ''' y is generate as the summation of error term and estimation '''
        errorTerm = normal(self.epslMean, self.epslVar, (self.numOb, 1))
        X = uniform(-10, 10, (self.numOb, self.numDim))
        X[:, 0] = 1
        y = np.dot(X, np.reshape(self.beta, (self.numDim, 1))) + errorTerm
        self.X = X
        self.y = y

    def olsEstimator(self):
        ''' pour this section, esstimate beta via using normal equation
            beta = inv((X.T * X)) * X.T * y     '''
        self.estiBeta = np.dot(np.dot(inv(np.dot(self.X.T, self.X)),
                                        self.X.T), self.y)

    def checker(self):
        b0 = np.reshape(np.asarray(self.beta), (self.numDim, 1))
        bTXTy = np.dot(np.dot(b0.T, self.X.T), self.y)
        yTXb = np.dot(np.dot(self.y.T, self.X), b0)

        print 'Comparing b\'X\'y = y\'Xb (we use close since'
        print 'they will not be strictly equals to each other):'
        print np.isclose(bTXTy, yTXb)[0][0]

        P = np.dot(np.dot(self.X, inv(np.dot(self.X.T, self.X))), self.X.T)
        M = np.eye(P.shape[0]) - P
        # it can be easily identified that P and M are symmetric and idepometic

if __name__ == '__main__':
    data = DGP(beta=[0.5, 0.8, 1.3])
    data.generator()
    data.olsEstimator()
    data.checker()
