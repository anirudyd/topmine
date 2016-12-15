"""
CITATION: The following code is a python implementation of some of the functions
taken from https://github.com/mimno/Mallet
"""
from __future__ import division
import sys, math
EULER_MASCHERONI = -0.5772156649015328606065121
PI_SQUARED_OVER_SIX = math.pi * math.pi / 6
HALF_LOG_TWO_PI = math.log(2 * math.pi) / 2
DIGAMMA_COEF_1 = 1 / 12
DIGAMMA_COEF_2 = 1 / 120
DIGAMMA_COEF_3 = 1 / 252
DIGAMMA_COEF_4 = 1 / 240
DIGAMMA_COEF_5 = 1 / 132
DIGAMMA_COEF_6 = 691 / 32760
DIGAMMA_COEF_7 = 1 / 12
DIGAMMA_COEF_8 = 3617 / 8160
DIGAMMA_COEF_9 = 43867 / 14364
DIGAMMA_COEF_10 = 174611 / 6600
DIGAMMA_LARGE = 9.5
DIGAMMA_SMALL = 0.000001

def learn_symmetric_concentration(countHistogram, observationLengths, numDimensions, currentValue):
    currentDigamma = float()
    largestNonZeroCount = 0
    nonZeroLengthIndex = [int()]*len(observationLengths)

    index = 0
    while index < len(countHistogram):
        if countHistogram[index] > 0:
            largestNonZeroCount = index
        index += 1

    denseIndex = 0
    index = 0
    while index < len(observationLengths):
        if observationLengths[index] > 0:
            nonZeroLengthIndex[denseIndex] = index
            denseIndex += 1
        index += 1

    denseIndexSize = denseIndex

    iteration = 1
    while iteration <= 200:
        currentParameter = currentValue / numDimensions
        currentDigamma = 0
        numerator = 0.0

        index = 1
        while index <= largestNonZeroCount:
            currentDigamma += 1.0 / (currentParameter + index - 1)
            numerator += countHistogram[index] * currentDigamma
            index += 1

        currentDigamma = 0
        denominator = 0.0
        previousLength = 0

        cachedDigamma = digamma(currentValue)

        denseIndex = 0
        while denseIndex < denseIndexSize:
            length = nonZeroLengthIndex[denseIndex]
            if length - previousLength > 20:
                currentDigamma = digamma(currentValue + length) - cachedDigamma
            else:
                index = previousLength
                while index < length:
                    currentDigamma += 1.0 / (currentValue + index)
                    index += 1
            denominator += currentDigamma * observationLengths[length]
            denseIndex += 1

        currentValue = currentParameter * numerator / denominator
        iteration += 1
    return currentValue


def learn_parameters(parameters, observations, observationLengths):
    return learn_params(parameters, observations, observationLengths, 1.00001, 1.0, 200)


def learn_params(parameters, observations, observationLengths, shape, scale, numIterations):
    i = int()
    k = int()
    parametersSum = 0
    k = 0
    while k < len(parameters):
        parametersSum += parameters[k]
        k += 1

    oldParametersK = float()
    currentDigamma = float()
    denominator = float()

    nonZeroLimit = int()
    nonZeroLimits = [-1]*len(observations)

    histogram = []

    i = 0
    while i < len(observations):
        histogram = observations[i]

        k = 0
        while k < len(histogram):
            if histogram[k] > 0:
                nonZeroLimits[i] = k
            k += 1
        i += 1

    iteration = 0
    while iteration < numIterations:
        denominator = 0
        currentDigamma = 0

        i = 1
        while i < len(observationLengths):
            currentDigamma += 1 / (parametersSum + i - 1)
            denominator += observationLengths[i] * currentDigamma
            i += 1

        denominator -= 1 / scale
        parametersSum = 0

        k = 0
        while k < len(parameters):
            nonZeroLimit = nonZeroLimits[k]

            oldParametersK = parameters[k]
            parameters[k] = 0
            currentDigamma = 0
            
            histogram = observations[k]
            
            i = 1
            while i <= nonZeroLimit:
                currentDigamma += 1 / (oldParametersK + i - 1)
                parameters[k] += histogram[i] * currentDigamma
                i += 1
            
            parameters[k] = oldParametersK * (parameters[k] + shape) / denominator
            parametersSum += parameters[k]
            k += 1
        iteration += 1
    if parametersSum < 0.0:
        print parametersSum
        print("sum not valid")
        sys.exit(1)
    return parametersSum

def digamma(z):
    psi = 0.0
    if z < DIGAMMA_SMALL:
        psi = EULER_MASCHERONI - (1 / z)

        return psi
    while z < DIGAMMA_LARGE:
        psi -= 1 / z
        z += 1
    invZ = 1 / z
    invZSquared = invZ * invZ
    psi += math.log(z) - 0.5 * invZ - invZSquared * (DIGAMMA_COEF_1 - invZSquared * (DIGAMMA_COEF_2 - invZSquared * (DIGAMMA_COEF_3 - invZSquared * (DIGAMMA_COEF_4 - invZSquared * (DIGAMMA_COEF_5 - invZSquared * (DIGAMMA_COEF_6 - invZSquared * DIGAMMA_COEF_7))))))
    return psi


