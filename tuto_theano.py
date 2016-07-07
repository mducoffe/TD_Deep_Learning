'''
Created on 7 juil. 2016

@author: lvanni
'''
from numpy import array
import numpy
import theano
from theano.compile import function
from theano.compile.io import In

import theano.tensor as T


def tuto():
    
    print "\nLogistic Function 1"
    print "---------------------"
    x = T.dmatrix('x')
    s = 1 / (1 + T.exp(-x))
    logistic = theano.function([x], s)
    print logistic([[0, 1], [-1, -2]])

    print "\nLogistic Function 2"
    print "---------------------"
    s2 = (1 + T.tanh(x / 2)) / 2
    logistic2 = theano.function([x], s2)
    print logistic2([[0, 1], [-1, -2]])
    
    print "\nComputing More than one Thing at the Same Time"
    print "------------------------------------------------"
    a, b = T.dmatrices('a', 'b')
    diff = a - b
    abs_diff = abs(diff)
    diff_squared = diff**2
    f = theano.function([a, b], [diff, abs_diff, diff_squared])
    print f([[1, 1], [1, 1]], [[0, 1], [2, 3]]) 
    
    print "\nSetting a Default Value for an Argument"
    print "---------------------------------------"
    x, y = T.dscalars('x', 'y')
    z = x + y
    f = function([x, In(y, value=1)], z)
    print f(33)
    print f(33, 2)
    
    print "A Real Example: Logistic Regression"
    print "-----------------------------------"
    rng = numpy.random
    N = 400                                   # training sample size
    feats = 784                               # number of input variables
    
    # generate a dataset: D = (input_values, target_class)
    D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
    training_steps = 10000
    
    # Declare Theano symbolic variables
    x = T.dmatrix("x")
    y = T.dvector("y")
    
    # initialize the weight vector w randomly
    #
    # this and the following bias variable b
    # are shared so they keep their values
    # between training iterations (updates)
    w = theano.shared(rng.randn(feats), name="w")
    
    # initialize the bias term
    b = theano.shared(0., name="b")
    
    print("Initial model:")
    print(w.get_value())
    print(b.get_value())
    
    # Construct Theano expression graph
    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
    prediction = p_1 > 0.5                    # The prediction thresholded
    xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
    cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
    gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                              # w.r.t weight vector w and
                                              # bias term b
                                              # (we shall return to this in a
                                              # following section of this tutorial)
    
    # Compile
    train = theano.function(
              inputs=[x,y],
              outputs=[prediction, xent],
              updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
    predict = theano.function(inputs=[x], outputs=prediction)
    
    # Train
    for i in range(training_steps):
        pred, err = train(D[0], D[1])
    
    print("Final model:")
    print(w.get_value())
    print(b.get_value())
    print("target values for D:")
    print(D[1])
    print("prediction on D:")
    print(predict(D[0]))

if __name__ == '__main__':
    tuto()