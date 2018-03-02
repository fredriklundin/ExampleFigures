import sys
from array import *
from math import *
from numpy import linalg as la
import numpy as np
from numpy.ma import mean
import csv
import matplotlib.pyplot as plt


exampleCovariance = np.empty((2,2))
exampleCovariance[0][0] = 5
exampleCovariance[0][1] = 1
exampleCovariance[1][0] = 1
exampleCovariance[1][1] = 2

badWeights = np.empty((1,2))
badWeights[0][0] = 0.7
badWeights[0][1] = 0.3

worseWeights = np.empty((1,2))
worseWeights[0][0] = 0.9
worseWeights[0][1] = 0.1

worstWeights = np.empty((1,2))
worstWeights[0][0] = 0.99
worstWeights[0][1] = 0.01

goodWeights = np.empty((1,2))
goodWeights[0][0] = 0.2
goodWeights[0][1] = 0.8

testResult = np.asarray(np.dot(exampleCovariance, goodWeights.T))
testResult_final = np.dot(goodWeights, testResult)

testResult1 = np.asmatrix(np.dot(exampleCovariance, badWeights.T))
testResult1_final = np.dot(badWeights, testResult1)

testResult2 = np.asarray(np.dot(exampleCovariance, worseWeights.T))
testResult2_final = np.dot(worseWeights, testResult2)

testResult3 = np.asarray(np.dot(exampleCovariance, worstWeights.T))
testResult3_final = np.dot(worstWeights, testResult3)

# test = la.eig(exampleCovariance)

plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 20)

ax = plt.gca()
plt.plot(goodWeights[0][0], goodWeights[0][1],'ro', markeredgewidth = 0.5, markersize = 12, label = 'Good weights')
plt.plot(testResult[0],testResult[1], 'r*', markeredgewidth = 0.5, markersize = 12, label = 'Transformation of good weights')

plt.plot(badWeights[0][0], badWeights[0][1], 'co', markeredgewidth = 0.5, markersize = 12, label = 'Bad weights')
plt.plot(testResult1[0], testResult1[1], 'c*', markeredgewidth = 0.5, markersize = 12, label = 'Transformation of bad weights')

plt.plot(worseWeights[0][0], worseWeights[0][1], 'bo', markeredgewidth = 0.5, markersize = 12, label = 'Worse weights')
plt.plot(testResult2[0], testResult2[1], 'b*', markeredgewidth = 0.5, markersize = 12, label = 'Transformation of worse weights')

plt.plot(worstWeights[0][0], worstWeights[0][1], 'go', markeredgewidth = 0.5, markersize = 12, label = 'Worst weights')
plt.plot(testResult3[0], testResult3[1], 'g*', markeredgewidth = 0.5, markersize = 12, label = 'Transformation of worst weights')

plt.axis([0, 5.5, 0, 5.5])
plt.xlabel("Weight w_0", fontsize = 20)
plt.ylabel("Weight w_1", fontsize = 20)
plt.legend(loc = 'best', numpoints = 1)
plt.tight_layout()
plt.savefig('unequalCovariance.eps', format = 'eps', dpi = 2000)
plt.show()

exampleCovariance = np.empty((2,2))
exampleCovariance[0][0] = 2
exampleCovariance[0][1] = 1
exampleCovariance[1][0] = 1
exampleCovariance[1][1] = 2

badWeights = np.empty((1,2))
badWeights[0][0] = 0.7
badWeights[0][1] = 0.3

worseWeights = np.empty((1,2))
worseWeights[0][0] = 0.9
worseWeights[0][1] = 0.1

worstWeights = np.empty((1,2))
worstWeights[0][0] = 0.99
worstWeights[0][1] = 0.01

goodWeights = np.empty((1,2))
goodWeights[0][0] = 0.5
goodWeights[0][1] = 0.5

testResult = np.asarray(np.dot(exampleCovariance, goodWeights.T))
testResult_final = np.dot(goodWeights, testResult)

testResult1 = np.asmatrix(np.dot(exampleCovariance, badWeights.T))
testResult1_final = np.dot(badWeights, testResult1)

testResult2 = np.asarray(np.dot(exampleCovariance, worseWeights.T))
testResult2_final = np.dot(worseWeights, testResult2)

testResult3 = np.asarray(np.dot(exampleCovariance, worstWeights.T))
testResult3_final = np.dot(worstWeights, testResult3)

# test = la.eig(exampleCovariance)

plt.plot(goodWeights[0][0], goodWeights[0][1],'ro', markeredgewidth = 0.5, markersize = 12, label = 'Good weights')
plt.plot(testResult[0],testResult[1], 'r*', markeredgewidth = 0.5, markersize = 12, label = 'Transformation of good weights')

plt.plot(badWeights[0][0], badWeights[0][1], 'co', markeredgewidth = 0.5, markersize = 12, label = 'Bad weights')
plt.plot(testResult1[0], testResult1[1], 'c*', markeredgewidth = 0.5, markersize = 12, label = 'Transformation of bad weights')

plt.plot(worseWeights[0][0], worseWeights[0][1], 'bo', markeredgewidth = 0.5, markersize = 12, label = 'Worse weights')
plt.plot(testResult2[0], testResult2[1], 'b*', markeredgewidth = 0.5, markersize = 12, label = 'Transformation of worse weights')

plt.plot(worstWeights[0][0], worstWeights[0][1], 'go', markeredgewidth = 0.5, markersize = 12, label = 'Worst weights')
plt.plot(testResult3[0], testResult3[1], 'g*', markeredgewidth = 0.5, markersize = 12, label = 'Transformation of worst weights')

plt.axis([0, 3.5, 0, 3.5])
plt.xlabel("Weight w_0", fontsize = 20)
plt.ylabel("Weight w_1", fontsize = 20)
plt.legend(loc = 'best', numpoints = 1)
plt.tight_layout()
plt.savefig('equalCovariance.eps', format = 'eps', dpi = 2000)
plt.show()