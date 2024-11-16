import numpy as np

def triggeringFunction(alphai, di, zetaij, hzij, Gamma, eij, thetaij):
    Phiij = (alphai / (2 * di)) * zetaij * np.dot(hzij.T, np.dot(Gamma, hzij)) \
            + zetaij * np.dot(hzij.T, np.dot(Gamma, eij)) \
            - thetaij * np.dot(eij.T, np.dot(Gamma, eij))
    return Phiij
