import numpy as np
from numpy import random

DEBUG_MODE = False # True = weights of hidden and output layers are
                    # not initialized to random values
lr = 0.5
MSE_EXPECTED = 1 / (1 * 1000)# * 1000)# * 1000)

# XOR gate inputs and corresponding output
i0 = np.array([0, 0, 1, 1], np.float16)
i1 = np.array([0, 1, 0, 1], np.float16)
inputs = np.array([i0, i1], np.float16)
o0 = np.array([0, 1, 1, 0], np.float16)

#                    _
#    _              |_|
#   |_|              _               _
#    _              |_|             |_|
#   |_|              _
#                   |_|
#
#  Inputs       Hidden Layer      Output
#  (i1, i2)     (h1, h2, h3)       (o1)
#

# Weights for hidden layer
#
#       i0  i1
#
# wh = w00 w01      h0
#      w10 w11      h1
#      w20 w21      h2
wh = np.array([ [0, 0], \
                [0, 0], \
                [0, 0]], np.float16)
if DEBUG_MODE:
    wh[0, 0] = 0.2316
    wh[0, 1] = 0.6753
    wh[1, 0] = 0.1956
    wh[1, 1] = 0.7686
    wh[2, 0] = 0.6577
    wh[2, 1] = 0.6978
else:
    wh_rows = len(wh)
    wh_cols = len(wh[0])
    for i in range(wh_rows):
        for j in range(wh_cols):
            wh[i, j] = random.rand() * 1 # 1 is upper value of 0/1 input set

def sigmoid(x):
    val = 1 / (1 + np.exp(-x))
    return val

def derivativeOfSigmoid(x):
    val = sigmoid(x)
    val = val * (1 - val)
    return val

# Weights for output layer
#
#       h0   h1   h2
#
# wo = w0_o w1_o w2_o       o
wo = np.array([0, 0, 0], np.float16)
if DEBUG_MODE:
    wo[0] = 0.0543
    wo[1] = 0.04202
    wo[2] = 0.2502
else:
    wo_rows = len(wo)
    for i in range(wo_rows):
        wo[i] = random.rand() * 1 # 1 is upper value of 0/1 hidden set

# Calculating partial derivatives for output layer
def partialErrorOverOutputOfOutputNode(output, target):
    val = np.subtract(output, target)
    return val

def partialOutputOverInputOfOutputNode(output):
    val = derivativeOfSigmoid(output)
    return val

def partialInputOverWeightOfOutputNode(outputOfHiddenLayer):
    val = outputOfHiddenLayer
    return val

def partialErrorOverWeightAtOutputNode(dErrOverOut, dOutOverIn, dInOverW):
    val = np.multiply(dErrOverOut, dOutOverIn)
    val = np.matmul(dInOverW, np.transpose(val))
    return val

# Calculating partial derivatives for hidden layer
def partialInputOfOutputNodeOverOutputOfHiddenNode(weightsOfOutputLayer):
    val = weightsOfOutputLayer
    return val

def partialErrorOverOutputOfHiddenNode(dErrOverOutOfOutputLayer, \
                                        dOutOverInOfOutputLayer, \
                                        dWeightedInputAtOutputNodeOverOutputOfHiddenLayer):
    val = np.multiply(dErrOverOutOfOutputLayer, dOutOverInOfOutputLayer)
    val = np.matmul( \
            np.reshape(dWeightedInputAtOutputNodeOverOutputOfHiddenLayer, (3, 1)), \
            val.reshape((1,4)))
    return val

def partialOutputOverInputOfHiddenNode(outputOfHiddenLayer):
    val = np.vectorize(derivativeOfSigmoid)(outputOfHiddenLayer)
    return val

def partialInputOverWeightOfHiddenNode(inputOfHiddenLayer):
    val = inputOfHiddenLayer
    return val

def partialErrorOverWeightAtHiddenNode(dErrOverOut, dOutOverIn, dInOverW):
    val = np.multiply(dErrOverOut, dOutOverIn)
    val = np.matmul(dInOverW, np.transpose(val))
    return val

# Calculate value of new weights for output layer
def newValueOfWeightsForOutputLayer(lr, oldWeights, fr):
    val = -1
    fr = np.transpose(fr)

    if (oldWeights.shape == fr.shape):
        val = np.subtract(oldWeights, np.multiply(lr, fr))
    else:
        errMsg = "Order of matrices differ."
        errMsg += " "+"Will not compute new weights for output layer."
        print("ERROR: ",errMsg)
        print("oldWeights: \n",oldWeights)
        print("fr: \n",fr)

    return val

# Calculate value of new weights for hidden layer
def newValueOfWeightsForHiddenLayer(lr, oldWeights, fr):
    val = -1
    fr = np.transpose(fr)

    if (oldWeights.shape == fr.shape):
        val = np.subtract(oldWeights, np.multiply(lr, fr))
    else:
        errMsg = "Order of matrices differ."
        errMsg += " "+"Will not compute new weights for hidden layer."
        print("ERROR: ",errMsg)
        print("oldWeights: \n",oldWeights)
        print("fr: \n",fr)

    return val

def oneIteration(inputs, o0, wo, wh, lr):
    if DEBUG_MODE:
        print("inputs: \n",inputs)
        print("o0: \n",o0)
        print("wo: \n",wo)
        print("wh: \n",wh)
        print("lr: \n",lr)
    woNew = np.array([0, 0, 0], np.float16)
    if DEBUG_MODE:
        print("init: woNew: \n",woNew)
    whNew = np.array([  [0, 0], \
                        [0, 0], \
                        [0, 0]], np.float16)
    if DEBUG_MODE:
        print("init: whNew: \n",whNew)
    mse = 0.0
    if DEBUG_MODE:
        print("init: mse: \n",mse)
    # Inputs for hidden layer after applying weights
    in_h = np.dot(wh, inputs)
    if DEBUG_MODE:
        print("in_h: \n",in_h)
    # Outputs of hidden layer
    out_h = np.vectorize(sigmoid)(in_h)
    if DEBUG_MODE:
        print("out_h: \n",out_h)
    # Inputs for output layer after applying weights
    in_o = np.dot(wo, out_h)
    if DEBUG_MODE:
        print("in_o: \n",in_o)
    # Outputs of output layer
    out_o = np.vectorize(sigmoid)(in_o)
    if DEBUG_MODE:
        print("out_o: \n",out_o)
    # Error in each output compared to what was expected
    error = np.subtract(o0, out_o)
    if DEBUG_MODE:
        print("error: \n",error)
    # Calculating mean squared error
    mse_arr = 0.5 * np.square(error)
    if DEBUG_MODE:
        print("mse_arr: \n",mse_arr)
    mse = np.sum(mse_arr)
    if DEBUG_MODE:
        print("mse: \n",mse)
    # Calculate new weights of output layer
    dErrorOverOutput_o = partialErrorOverOutputOfOutputNode(out_o, o0)
    if DEBUG_MODE:
        print("dErrorOverOutput_o: \n",dErrorOverOutput_o)
    dOutputOverInput_o = partialOutputOverInputOfOutputNode(out_o)
    if DEBUG_MODE:
        print("dOutputOverInput_o: \n",dOutputOverInput_o)
    dInputOverWeight_o = partialInputOverWeightOfOutputNode(out_h)
    if DEBUG_MODE:
        print("dInputOverWeight_o: \n",dInputOverWeight_o)
    dErrorOverWeight_o = partialErrorOverWeightAtOutputNode( \
                                dErrorOverOutput_o, \
                                dOutputOverInput_o, \
                                dInputOverWeight_o)
    if DEBUG_MODE:
        print("dErrorOverWeight_o: \n",dErrorOverWeight_o)
    woNew = newValueOfWeightsForOutputLayer(lr, wo, dErrorOverWeight_o)
    if DEBUG_MODE:
        print("woNew: \n",woNew)
    # Calculate new weights of hidden layer
    dWeightInAtOutputOverOutOfHidden = \
        partialInputOfOutputNodeOverOutputOfHiddenNode(wo)
    if DEBUG_MODE:
        print("dWeightInAtOutputOverOutOfHidden: \n",\
            dWeightInAtOutputOverOutOfHidden)
    dErrorOverOutput_h = partialErrorOverOutputOfHiddenNode( \
                            dErrorOverOutput_o, \
                            dOutputOverInput_o, \
                            dWeightInAtOutputOverOutOfHidden)
    if DEBUG_MODE:
        print("dErrorOverOutput_h: \n",dErrorOverOutput_h)
    dOutputOverInput_h = partialOutputOverInputOfHiddenNode(out_h)
    if DEBUG_MODE:
        print("dOutputOverInput_h: \n",dOutputOverInput_h)
    dInputOverWeight_h = partialInputOverWeightOfHiddenNode(inputs)
    if DEBUG_MODE:
        print("dInputOverWeight_h: \n",dInputOverWeight_h)
    dErrorOverWeight_h = partialErrorOverWeightAtHiddenNode( \
                            dErrorOverOutput_h, \
                            dOutputOverInput_h, \
                            dInputOverWeight_h)
    if DEBUG_MODE:
        print("dErrorOverWeight_h: \n",dErrorOverWeight_h)
    whNew = newValueOfWeightsForHiddenLayer(lr, wh, dErrorOverWeight_h)
    if DEBUG_MODE:
        print("whNew: \n",whNew)
    # Return mse of this iteration, and the new weights calculated
    if DEBUG_MODE:
        print("mse: \n",mse)
        print("whNew: \n",whNew)
        print("woNew: \n",woNew)
    return mse, whNew, woNew

whStart = wh
woStart = wo
itrCount = 0
continueIterating = True
mseAfter = 0.0
whAfter = 0.0
woAfter = 0.0
while continueIterating:
    itrCount += 1
    mseAfter, whAfter, woAfter = oneIteration(inputs, o0, wo, wh, lr)
    showInfo = False
    if itrCount > 100000:
        if (itrCount % 100000) == 0:
            showInfo = True
    elif itrCount > 10000:
        if (itrCount % 10000) == 0:
            showInfo = True
    elif itrCount > 1000:
        if (itrCount % 1000) == 0:
            showInfo = True
    else:
        showInfo = True
    if showInfo:
        print("\n")
        print("iteration: ",itrCount)
        print("mseAfter: ",mseAfter)
        print("whAfter: \n",whAfter)
        print("woAfter: \n",woAfter)
    if mseAfter < MSE_EXPECTED:
        continueIterating = False
    else:
        wo = woAfter
        wh = whAfter

print("\n")
print("iteration: ",itrCount)
print("mseAfter: ",mseAfter)
print("whAfter: \n",whAfter)
print("woAfter: \n",woAfter)
print("whStart: \n",whStart)
print("woStart: \n",woStart)