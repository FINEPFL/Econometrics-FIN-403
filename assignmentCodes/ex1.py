from numpy.random import uniform, normal
from numpy.linalg import inv, matrix_rank, norm
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
    def __init__(self, beta, numOb=20, numDim=3, epslMean=0, epslVar=4,
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

    def getMP(self, X):
        ''' obtain the residual maker M and projector P '''
        P = np.dot(np.dot(X, inv(np.dot(X.T, X))), X.T)
        M = np.eye(P.shape[0]) - P
        return P, M

    def checker(self):
        b0 = np.reshape(np.asarray(self.beta), (self.numDim, 1))
        bTXTy = np.dot(np.dot(b0.T, self.X.T), self.y)
        yTXb = np.dot(np.dot(self.y.T, self.X), b0)

        print 'Comparing b\'X\'y = y\'Xb (we use close since'
        print 'they will not be strictly equals to each other)(c):'
        print np.isclose(bTXTy, yTXb)[0][0]

        '''it can be easily identified that P and M are symmetric and
           idepomtent  (d) '''
        P, M = self.getMP(self.X)

        ''' verify the Frisch-Waugh Theorem (e)'''
        X1 = self.X[:,0:(self.numDim-1)]
        X2 = np.reshape(self.X[:, -1], (self.numOb, 1))
        P1, M1 = self.getMP(X1)
        b2 = np.dot(np.dot(np.dot(inv(np.dot(np.dot(X2.T, M1), X2)), X2.T), M1), self.y)
        print 'The approximation and actual value of b2 is:', b2[0][0], self.beta[-1]

        ''' Compute SSE of a constant+x model and complete model (f) '''
        sseFull = np.dot(M, self.y)
        ssePart = np.dot(M1, self.y)
        print norm(sseFull)**2, norm(ssePart)**2 # observe that error sseFull is much smaller than ssePart

        ''' compute R squre, e = My, SST = SSR + SSE '''
        M0 = np.eye(self.numOb) - np.dot(np.ones((self.numOb, 1)), np.ones((1, self.numOb)))/self.numOb
        sst = np.dot(np.dot(self.y.T, M0), self.y)

        Rsqre = (sst-norm(sseFull)**2)/sst
        print 'R^2: ', Rsqre[0][0]

if __name__ == '__main__':
    data = DGP(beta=[0.5, 0.8, 1.3])
    data.generator()
    data.olsEstimator()
    data.checker()
