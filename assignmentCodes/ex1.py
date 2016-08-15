from numpy.random import uniform
from numpy.random import normal
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
    ''' Data generator '''
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
        y = np.dot(X, np.reshape(self.beta, (self.numDim, 1))) + errorTerm
        self.data = y
        print self.data

if __name__ == '__main__':
    data = DGP(beta=[0.5, 0.8, 1.3])
    data.generator()
