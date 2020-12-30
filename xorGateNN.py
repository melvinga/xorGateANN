import numpy as np
from numpy import random

lr = 0.5
MSE_EXPECTED = 1 / (1 * 1000)

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
wh_rows = len(wh)
wh_cols = len(wh[0])
for i in range(wh_rows):
    for j in range(wh_cols):
        wh[i,j] = random.rand() * 1 # 1 is upper value of 0/1 input set

## Inputs for hidden layer after applying weights
#in_h = np.dot(wh, inputs)
#print("in_h: ",in_h)

def sigmoid(x):
    val = 1 / (1 + np.exp(-x))
    return val

def derivativeOfSigmoid(x):
    val = sigmoid(x)
    val = val * (1 - val)
    return val

#out_h = np.vectorize(sigmoid)(in_h)
#print("out_h: ",out_h)

# Weights for output layer
#
#       h0   h1   h2
#
# wo = w0_o w1_o w2_o       o
wo = np.array([0, 0, 0], np.float16)
wo_rows = len(wo)
for i in range(wo_rows):
    wo[i] = random.rand() * 1 # 1 is upper value of 0/1 hidden set

'''
# Inputs for output layer after applying weights
in_o = np.dot(wo, out_h)
print("in_o: ",in_o)

out_o = np.vectorize(sigmoid)(in_o)
print("out_o: ",out_o)

# Error in each output compared to what was expected
error = np.subtract(o0, out_o)
print("error: ",error)

# Calculating mean squared error
mse_arr = 0.5 * np.square(error)
print("mse_arr: ",mse_arr)

MSE = np.sum(mse_arr)
print("MSE: ",MSE)
'''

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
    val = np.multiply(dInOverW, np.transpose(val))
    return val

# Calculating partial derivatives for hidden layer
def partialInputOfOutputNodeOverOutputOfHiddenNode(weightsOfOutputLayer):
    val = weightsOfOutputLayer
    return val

def partialErrorOverOutputOfHiddenNode(dErrOverOutOfOutputLayer, \
                                        dOutOverInOfOutputLayer, \
                                        dWeightedInputAtOutputNodeOverOutputOfHiddenLayer):
    val = np.multiply(dErrOverOutOfOutputLayer * dOutOverInOfOutputLayer)
    val = (np.transpose(dWeightedInputAtOutputNodeOverOutputOfHiddenLayer)).dot(val)
    return val

def partialOutputOverInputOfHiddenNode(outputOfHiddenLayer):
    val = derivativeOfSigmoid(outputOfHiddenLayer)
    return val

def partialInputOverWeightOfHiddenNode(inputOfHiddenLayer):
    val = inputOfHiddenLayer
    return val

def partialErrorOverWeightAtHiddenNode(dErrOverOut, dOutOverIn, dInOverW):
    val = val = np.multiply(dErrOverOut, dOutOverIn)
    val = np.multiply(dInOverW, np.transpose(val))
    return val

# Calculate value of new weights for output layer
def newValueOfWeightsForOutputLayer(lr, oldWeights, fr):
    val = -1

    if (oldWeights.shape == fr.shape):
        val = np.subtract(oldWeights, np.multiply(lr, fr))
        val = np.transpose(val)
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

    if (oldWeights.shape == fr.shape):
        val = np.subtract(oldWeights, np.multiply(lr, fr))
        val = val.reshape(1, 2)
    else:
        errMsg = "Order of matrices differ."
        errMsg += " "+"Will not compute new weights for hidden layer."
        print("ERROR: ",errMsg)
        print("oldWeights: \n",oldWeights)
        print("fr: \n",fr)

    return val

def oneIteration(inputs, o0, wo, wh, lr):
    print("inputs: \n",inputs) # MEL_DEBUG
    print("o0: \n",o0) # MEL_DEBUG
    print("wo: \n",wo) # MEL_DEBUG
    print("wh: \n",wh) # MEL_DEBUG
    print("lr: \n",lr) # MEL_DEBUG
    woNew = np.array([0, 0, 0], np.float16)
    print("init: woNew: \n",woNew) # MEL_DEBUG
    whNew = np.array([  [0, 0], \
                        [0, 0], \
                        [0, 0]], np.float16)
    print("init: whNew: \n",whNew) # MEL_DEBUG
    mse = 0.0
    print("init: mse: \n",mse) # MEL_DEBUG
    # Inputs for hidden layer after applying weights
    in_h = np.dot(wh, inputs)
    print("in_h: \n",in_h) # MEL_DEBUG
    # Outputs of hidden layer
    out_h = np.vectorize(sigmoid)(in_h)
    print("out_h: \n",out_h) # MEL_DEBUG
    # Inputs for output layer after applying weights
    in_o = np.dot(wo, out_h)
    print("in_o: \n",in_o) # MEL_DEBUG
    # Outputs of output layer
    out_o = np.vectorize(sigmoid)(in_o)
    print("out_o: \n",out_o) # MEL_DEBUG
    # Error in each output compared to what was expected
    error = np.subtract(o0, out_o)
    print("error: \n",error) # MEL_DEBUG
    # Calculating mean squared error
    mse_arr = 0.5 * np.square(error)
    print("mse_arr: \n",mse_arr) # MEL_DEBUG
    mse = np.sum(mse_arr)
    print("mse: \n",mse) # MEL_DEBUG
    # Calculate new weights of output layer
    dErrorOverOutput_o = partialErrorOverOutputOfOutputNode(out_o, o0)
    print("dErrorOverOutput_o: \n",dErrorOverOutput_o) # MEL_DEBUG
    dOutputOverInput_o = partialOutputOverInputOfOutputNode(out_o)
    print("dOutputOverInput_o: \n",dOutputOverInput_o) # MEL_DEBUG
    dInputOverWeight_o = partialInputOverWeightOfOutputNode(in_o)
    print("dInputOverWeight_o: \n",dInputOverWeight_o) # MEL_DEBUG
    dErrorOverWeight_o = partialErrorOverWeightAtHiddenNode( \
                                dErrorOverOutput_o, \
                                dOutputOverInput_o, \
                                dInputOverWeight_o)
    print("dErrorOverWeight_o: \n",dErrorOverWeight_o) # MEL_DEBUG
    woNew = newValueOfWeightsForOutputLayer(lr, wo, dErrorOverWeight_o)
    print("woNew: \n",woNew) # MEL_DEBUG
    # Calculate new weights of hidden layer
    dWeightInAtOutputOverOutOfHidden = \
        partialInputOfOutputNodeOverOutputOfHiddenNode(wo)
    print("dWeightInAtOutputOverOutOfHidden: \n",\
        dWeightInAtOutputOverOutOfHidden) # MEL_DEBUG
    dErrorOverOutput_h = partialErrorOverOutputOfHiddenNode( \
                            dErrorOverOutput_o, \
                            dOutputOverInput_o, \
                            dWeightInAtOutputOverOutOfHidden)
    print("dErrorOverOutput_h: \n",dErrorOverOutput_h) # MEL_DEBUG
    dOutputOverInput_h = partialOutputOverInputOfHiddenNode(out_h)
    print("dOutputOverInput_h: \n",dOutputOverInput_h) # MEL_DEBUG
    dInputOverWeight_h = partialInputOverWeightOfHiddenNode(inputs)
    print("dInputOverWeight_h: \n",dInputOverWeight_h) # MEL_DEBUG
    dErrorOverWeight_h = partialErrorOverWeightAtHiddenNode( \
                            dErrorOverOutput_h, \
                            dOutputOverInput_h, \
                            dInputOverWeight_h)
    print("dErrorOverWeight_h: \n",dErrorOverWeight_h) # MEL_DEBUG
    whNew = newValueOfWeightsForHiddenLayer(lr, wh, dErrorOverWeight_h)
    print("whNew: \n",whNew) # MEL_DEBUG
    # Return mse of this iteration, and the new weights calculated
    print("mse: \n",mse) # MEL_DEBUG
    print("whNew: \n",whNew) # MEL_DEBUG
    print("woNew: \n",woNew) # MEL_DEBUG
    return mse, whNew, woNew

mseAfter, whAfter, woAfter = oneIteration(inputs, o0, wo, wh, lr)

'''
# weights between i1 node and hidden layer nodes
w_iA_h0 = np.random.rand(0, 1)
w_iA_h1 = np.random.rand(0, 1)
w_iA_h2 = np.random.rand(0, 1)

# weights between i2 node and hidden layer nodes
w_iB_h0 = np.random.rand(0, 1)
w_iB_h1 = np.random.rand(0, 1)
w_iB_h2 = np.random.rand(0, 1)

# weights between hidden layer nodes and output node
w_h0_o1 = np.random.rand(0, 1)
w_h1_o1 = np.random.rand(0, 1)
w_h2_o1 = np.random.rand(0, 1)

def sigmoid(x):
    val = 1 / (1 + np.exp(-x))
    return val

def inputGivenToHiddenNode(inA, inB, wA, wB):
    val = (wA * inA) + (wB + inB)
    return val

def outputPutIntoHiddenNode(input):
    val = sigmoid(input)
    return val

def inputGivenToOutputNode(in0, in1, in2, w0, w1, w2):
    val = (in0 * w0) + (in1 * w1) + (in2 * w2)
    return val

def outputPutIntoOutputNode(input):
    val = sigmoid(input)
    return val

def meanSquaredErrorTerm(targetOutput, predictedOutput):
    d = targetOutput - predictedOutput
    val = 0.5 * np.power(d, 2)
    return val

# partial derivatives for output layer error trend
def partialErrorOverOutputOfOutputNode(predictedOutput, targetOutput):
    val = predictedOutput - targetOutput
    return val

def partialOutputOverInputOfOutputNode(predictedOutput):
    val = predictedOutput(1 - predictedOutput)
    return val

def partialInputOverWeightOfOutputNode(input):
    val = input
    return val

def partialErrorOverWeightAtOutputNode(fr1, fr2, fr3): # fr = fraction
    val = fr1 * fr2 * fr3
    return val

# partial derivatives for hidden layer error trend
def partialErrorOverOutputOfHiddenNode(predictedOutput, targetOutput, w):
    val = ((predictedOutput - targetOutput) * (predictedOutput(1 - predictedOutput))) * w
    return val

def partialOutputOverInputOfHiddenNode(hiddenNodeOutput):
    val = hiddenNodeOutput(1 - hiddenNodeOutput)
    return val

def partialInputOverWeightOfHiddenNode(input):
    val = input
    return val

def partialErrorOverWeightAtHiddenNode(fr1, fr2, fr3): # fr = fraction
    val = fr1 * fr2 * fr3
    return val

def newValueOfWeight(w, lr, fr): # lr = learning rate
    val = w - (lr * fr)
    return val

def oneIteration(inputA, inputB, hidden, \
            w_iA_h1, w_iA_h2, w_iA_h3, \
            w_iB_h1, w_iB_h2, w_iB_h3, \
            w_h0_o1, w_h1_o1, w_h2_o1):
    # Things to return after every iteration
    w_iA_h0_new = 0.0
    w_iA_h1_new = 0.0
    w_iA_h2_new = 0.0
    w_iB_h0_new = 0.0
    w_iB_h1_new = 0.0
    w_iB_h2_new = 0.0
    w_h0_o1_new = 0.0
    w_h1_o1_new = 0.0
    w_h2_o1_new = 0.0
    mse = 0.0
    # Values needed to be stored to complete this iteration
    predictedOutput = np.array(0, 0, 0, 0)
    # For each element in input layer, do something
    for i in range(len(inputA)):
        print("i: ",i)
        inputToHiddenNode0 = inputGivenToHiddenNode( \
                                    inputA[i], inputB[i], \
                                    w_iA_h0, w_iB_h0)
        inputToHiddenNode1 = inputGivenToHiddenNode( \
                                    inputA[i], inputB[i], \
                                    w_iA_h1, w_iB_h1)
        inputToHiddenNode2 = inputGivenToHiddenNode( \
                                    inputA[i], inputB[i], \
                                    w_iA_h2, w_iB_h2)
        print("Input at hidden layer: ",hidden)
        hidden[0] = outputPutIntoHiddenNode(inputToHiddenNode0)
        hidden[1] = outputPutIntoHiddenNode(inputToHiddenNode1)
        hidden[2] = outputPutIntoHiddenNode(inputToHiddenNode2)
        print("Output at hidden layer: ",hidden)
        # Predict output of output node
        inputOfOutputNode = inputGivenToOutputNode( \
                                hidden[0], hidden[1], hidden[2],
                                w_h0_o1, w_h1_o1, w_h2_o1)
        print("Input at output node: ",inputOfOutputNode)
        predictedOutput[i] = outputPutIntoOutputNode(inputOfOutputNode)
        print("Output at output node: ",predictedOutput[i])
    print("Predicted output: ",predictedOutput)
    print("Target output   : ",targetOutput)
    # Calculate mean squared error across all input features
    for i in range(len(predictedOutput)):
        mse += meanSquaredErrorTerm(targetOutput[i], predictedOutput[i])
    print("Mean squared error: ",mse)
    
    # Note: We have to use matrices from here on. Because, as said in the reference
    # article, the input-output combo is non-linearly separable data.
    # What if we don't use matrices? There are three weights and only two input
    # combinations (0, 1) and (1, 0) giving output of 1; we can use to find new
    # these input-output pairs to find values of two weights.
    # For the third weight, there are two more input combinations
    # - (0, 0) and (1, 1) - and corresponding outputs 0 and 0 in both cases.
    # Which of these input-output pair will give value of third weight?
    # Should the third weight be 0? Such confusing questions can be avoided by
    # using matrices. Besides, using matrices makes it easier more useful
    # neural networks that are larger; such neural networks solve problem that can
    # never be solved by small neural networks.

    # Partial derivatives to calculate change in error w.r.t. change in weights
'''