import sys
from array import *
from math import *
from numpy import linalg as la
import numpy as np
from numpy.ma import mean
import csv
import matplotlib.pyplot as plt


def covariance(X):
	vectorSize = X.shape
	rows = vectorSize[0]
	if len(vectorSize) == 1:
		columns = 1
	else:
		columns = vectorSize[1]
	meanVector = averageReturn(X)

	covariance = np.empty((rows,rows))

	for i in np.arange(0,rows):
		for j in np.arange(0,rows):
			covariance[i, j] = (1 / (float(vectorSize[1]) - 1)) * np.sum(np.asarray(X[i,:] - meanVector[i]) * np.asarray(X[j,:] - meanVector[j]))
	return covariance

# Start date and end date must match with tthe data in the .csv files.
startDate = "2012-11-30"
endDate = "2018-01-29"

# Several stocks at Stockholm OMX large cap are listed below. Only the 41 first stocks were selected because it is tiresome to import the data manually.
# Only stocks having data during the complete time between startDate end endDate have been selected.
numberOfStocks = 41
stocks = [None] * numberOfStocks
stocks[0] = "AAK"
stocks[1] = "ABB"
stocks[2] = "AHSL"
stocks[3] = "ALFA"
stocks[4] = "ALIV-SDB"
stocks[5] = "AM1S"
stocks[6] = "ASSA-B"
stocks[7] = "ATCO-A"
stocks[8] = "ATCO-B"
stocks[9] = "ATRLJ-B"
stocks[10] = "AXFO"
stocks[11] = "AXIS"
stocks[12] = "AZA"
stocks[13] = "AZN"
stocks[14] = "BALD-B"
stocks[15] = "BETS-B"
stocks[16] = "BILL"
stocks[17] = "BOL"
stocks[18] = "CAST"
stocks[19] = "EKTA-B"
stocks[20] = "ELUX-A"
stocks[21] = "ELUX-B"
stocks[22] = "ERIC-A"
stocks[23] = "ERIC-B"
stocks[24] = "FABG"
stocks[25] = "FING-B"
stocks[26] = "GETI-B"
stocks[27] = "HEXA-B"
stocks[28] = "HM-B"
stocks[29] = "HOLM-A"
stocks[30] = "HOLM-B"
stocks[31] = "HPOL-B"
stocks[32] = "HUFV-A"
stocks[33] = "HUFV-C"
stocks[34] = "HUSQ-A"
stocks[35] = "HUSQ-B"
stocks[36] = "ICA"
stocks[37] = "INDU-A"
stocks[38] = "INDU-C"
stocks[39] = "SHB-A"
stocks[40] = "SHB-B"


# The data (.csv files) have been manually downloaded from this website: http://www.nasdaqomxnordic.com/aktier/historiskakurser/?languageId=3.
dataMatrix = []
for i in np.arange(0,numberOfStocks):
    temporaryData = np.genfromtxt(stocks[i] + "-" + startDate + "-" + endDate + ".csv", delimiter=";", dtype = None, skip_header=1,usecols=(7,)).astype("str")
    
    # Remove the header "Average price" present in the .csv file.
    if temporaryData[0] == "Average price":
        temporaryData = temporaryData[1:len(temporaryData)]
    
    dataMatrix.append(temporaryData)

for i in np.arange(0,numberOfStocks):
    for j in np.arange(0,dataMatrix[0].size):
        dataMatrix[i][j] = float(dataMatrix[i][j].replace("'","").replace(",","."))


# Required so that the first sample is the oldest sample, and the last sample is the newest sample.
dataMatrix = np.fliplr(dataMatrix)


# Calculate the average return for each stock.
def averageReturn(x):
    numberOfStocks = x.shape[0]
    average = np.empty((numberOfStocks,1))
    for i in np.arange(0,len(x)):
        average[i] = mean(x[i,:],axis=0)
    return average


calculatedReturn = np.empty((numberOfStocks,dataMatrix[0].size))


for i in np.arange(0,numberOfStocks):
    for j in np.arange(0,dataMatrix[0].size-1):
        
        calculatedReturn[i,j+1] = (float(dataMatrix[i][j+1]) - float(dataMatrix[i][j])) / float(dataMatrix[i][j])
        
returnMatrix = (calculatedReturn[:,1:dataMatrix[0].size])


# Print the average returns of the data.
mu = averageReturn(returnMatrix).T
for i in np.arange(0,numberOfStocks):
    print("mu[" + str(i) + "]: " + str(100*mu[0][i]) + "\n")


covarianceOfReturns = np.asmatrix(covariance(returnMatrix))

   
def costFunction(weights):
    return -portfolioExpectedValue(weights) + portfolioCovariance(weights)

def portfolioExpectedValue(weights):
    return float(np.dot(weights,mu.T))

def portfolioCovariance(weights):
    return float(np.dot(weights,covarianceOfReturns) * np.matrix(weights).T)

def costFunctionWithPenalty(weights, v):
    h = np.dot(np.ones(numberOfStocks), weights.T) - 1

    inequalityPenalty = 0
    equalityPenalty = h**2
    for j in np.arange(0, numberOfStocks):
        inequalityPenalty = inequalityPenalty + (max(0, -weights[0][j]))**2

    return costFunction(weights) + v * (inequalityPenalty + equalityPenalty)


gradient = np.empty((1, numberOfStocks))
def calculateGradient(weights, v):
    delta = 0.000000001
    for k in np.arange(0,numberOfStocks):
        point = weights
        point[0][k] = weights[0][k] + delta
        temporary1 = costFunctionWithPenalty(point, v)

        point[0][k] = weights[0][k] - delta
        temporary2 = costFunctionWithPenalty(point, v)

        # Derivative is calculated in each direction k as:
        # f(w + delta) - f(w - delta) / (2 * delta).
        gradient[0][k] = (temporary1 - temporary2) / (2 * delta)
    return gradient


# Initialise infeasible weights as a starting point to the exterior penalty method.
weights = 3 + np.random.uniform(-1,1,(1,numberOfStocks))

v = 1 # Initialise the penalty parameter.

# Define some different step sizes.
alpha = np.empty((1,7))
alpha[0][0] = 0.00000100
alpha[0][1] = 0.00000500
alpha[0][2] = 0.00000010
alpha[0][3] = 0.00000050
alpha[0][4] = 0.00000005
alpha[0][5] = 0.00001000
alpha[0][6] = 0.000000008

ax = plt.gca()
for i in np.arange(1,6000):
    # To ensure convergence, the penalty parameter must approach infinity.
    # Since it will be harder and harder to reduce the objective value, after some time, it must be a large number.
    v = v + 400
    gradient = calculateGradient(weights, v)
    bestCost = 1000000

    for j in np.arange(0, 7):
        # Update the weights by going in the opposite direction f the gradient at the current objective value.
        temporaryWeights = weights - alpha[0][j] * gradient

        # Try some different step sizes, alpha, to approximately find the best step size.
        if costFunction(temporaryWeights) < bestCost:
            bestCost = costFunctionWithPenalty(temporaryWeights, v)
            bestWeights = temporaryWeights

    weights = bestWeights
    if (i % 200 == 0):
        print("i :", i)

        # Make it visible how the cost function is reduced almost every iteration.
        print("costFunction: ", costFunction(weights))

        # Make it visible how the sum of the weights approach 1 from above.
        print("sum of weights: ", sum(sum(weights)))
        print("portfolioExpectedValue: ", portfolioExpectedValue(weights))
        print("square root of variance: ", sqrt(portfolioCovariance(weights)))

        # make it visible how the algorithm tries to lower the standard deviation while keeping a high expected return.
        plt.plot(100 * sqrt(252) * sqrt(portfolioCovariance(weights)), 100 * 252 * portfolioExpectedValue(weights), marker = 'o')
        plt.xlabel("Portfolio standard deviation per year [%]")
        plt.ylabel("Portfolio expected return per year [%]")
plt.show()

# make sure that the weights approximately sum to 1.
print("Sum of weights, final result: ", sum(sum(weights)))
print("costFunction, final result :", costFunction(weights))

# Print the result.
for i in np.arange(0,numberOfStocks):
    print("Stock: " + str(stocks[i]) + ", weight: " + str(100*weights[0][i]) + "\n")