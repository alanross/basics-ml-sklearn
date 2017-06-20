# Back-Propagation Neural Networks
#
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

import math
import random
import string

random.seed( 0 )


# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random( ) + a


# Make a matrix (we could use NumPy to speed this up)
def makeArray2D(I, J, fill=0.0):
    m = [ ]
    for i in range( I ):
        m.append( [ fill ] * J )
    return m


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh( x )


# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y ** 2


class NeuralNetwork:
    def __init__(self, numInputs, numHidden, numOutputs):
        # number of input, hidden, and output nodes
        self.numInputs = numInputs + 1  # +1 for bias node
        self.numHidden = numHidden
        self.numOutputs = numOutputs

        # activations for nodes
        self.inputNeurons = [ 1.0 ] * self.numInputs
        self.hiddenNeurons = [ 1.0 ] * self.numHidden
        self.outputNeurons = [ 1.0 ] * self.numOutputs

        # create weights
        self.weightsInput = makeArray2D( self.numInputs, self.numHidden )
        self.weightsOutput = makeArray2D( self.numHidden, self.numOutputs )
        # set them to random vaules
        for i in range( self.numInputs ):
            for j in range( self.numHidden ):
                self.weightsInput[ i ][ j ] = rand( -0.2, 0.2 )
        for j in range( self.numHidden ):
            for k in range( self.numOutputs ):
                self.weightsOutput[ j ][ k ] = rand( -2.0, 2.0 )

        # last change in weights for momentum
        self.ci = makeArray2D( self.numInputs, self.numHidden )
        self.co = makeArray2D( self.numHidden, self.numOutputs )

    def update(self, inputs):
        if len( inputs ) != self.numInputs - 1:
            raise ValueError( 'wrong number of inputs' )

        # input activations
        for i in range( self.numInputs - 1 ):
            self.inputNeurons[ i ] = inputs[ i ]

        # hidden activations
        for j in range( self.numHidden ):
            sum = 0.0
            for i in range( self.numInputs ):
                sum = sum + self.inputNeurons[ i ] * self.weightsInput[ i ][ j ]
            self.hiddenNeurons[ j ] = sigmoid( sum )

        # output activations
        for k in range( self.numOutputs ):
            sum = 0.0
            for j in range( self.numHidden ):
                sum = sum + self.hiddenNeurons[ j ] * self.weightsOutput[ j ][ k ]
            self.outputNeurons[ k ] = sigmoid( sum )

        return self.outputNeurons[ : ]

    def backPropagate(self, targets, N, M):
        if len( targets ) != self.numOutputs:
            raise ValueError( 'wrong number of target values' )

        # calculate error terms for output
        output_deltas = [ 0.0 ] * self.numOutputs
        for k in range( self.numOutputs ):
            error = targets[ k ] - self.outputNeurons[ k ]
            output_deltas[ k ] = dsigmoid( self.outputNeurons[ k ] ) * error

        # calculate error terms for hidden
        hidden_deltas = [ 0.0 ] * self.numHidden
        for j in range( self.numHidden ):
            error = 0.0
            for k in range( self.numOutputs ):
                error = error + output_deltas[ k ] * self.weightsOutput[ j ][ k ]
            hidden_deltas[ j ] = dsigmoid( self.hiddenNeurons[ j ] ) * error

        # update output weights
        for j in range( self.numHidden ):
            for k in range( self.numOutputs ):
                change = output_deltas[ k ] * self.hiddenNeurons[ j ]
                self.weightsOutput[ j ][ k ] = self.weightsOutput[ j ][ k ] + N * change + M * self.co[ j ][ k ]
                self.co[ j ][ k ] = change

        # update input weights
        for i in range( self.numInputs ):
            for j in range( self.numHidden ):
                change = hidden_deltas[ j ] * self.inputNeurons[ i ]
                self.weightsInput[ i ][ j ] = self.weightsInput[ i ][ j ] + N * change + M * self.ci[ i ][ j ]
                self.ci[ i ][ j ] = change

        # calculate error
        error = 0.0
        for k in range( len( targets ) ):
            error += 0.5 * (targets[ k ] - self.outputNeurons[ k ]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[ 0 ], '->', self.update( p[ 0 ] ))

    def weights(self):
        print('Input weights:')

        for i in range( self.numInputs ):
            print(self.weightsInput[ i ])

        print()
        print('Output weights:')

        for j in range( self.numHidden ):
            print(self.weightsOutput[ j ])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range( iterations ):
            error = 0.0
            for p in patterns:
                inputs = p[ 0 ]
                targets = p[ 1 ]
                self.update( inputs )
                error += self.backPropagate( targets, N, M )
            if i % 100 == 0:
                print('error %-.5f' % error)