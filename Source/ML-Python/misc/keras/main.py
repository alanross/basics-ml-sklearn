# coding=utf-8
# http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# Keras is a Deep Learning, high-level neural networks library
# capable of running on top of either TensorFlow or Theano.

# We can summarize the construction of deep learning models in Keras as follows:
# 1 Define your model. Create a sequence and add layers.
# 2 Compile your model. Specify loss functions and optimizers.
# 3 Fit your model. Execute the model using data.
# 4 Make predictions. Use the model to generate predictions on new data.

from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed( seed )

# load pima indians dataset
dataset = numpy.loadtxt( "pima-indians-diabetes.csv", delimiter="," )

# There are 8 input variables and 1 output variable (the last column).
# Once loaded we can split the dataset into input variables (X) and
# the output class variable (Y).

X = dataset[ :, 0:8 ]
Y = dataset[ :, 8 ]

# The first thing to get right is to ensure the input layer has the right number of inputs.
# This can be specified when creating the first layer with the input_dim argument and setting it to 8 for the 8 input variables.

# create model
model = Sequential( )
model.add( Dense( 12, input_dim=8, init='uniform', activation='relu' ) )
model.add( Dense( 8, init='uniform', activation='relu' ) )
model.add( Dense( 1, init='uniform', activation='sigmoid' ) )

# In this case, we will use logarithmic loss, which for a binary classification problem
# is defined in Keras as “binary_crossentropy“. We will also use the efficient
# gradient descent algorithm “adam” for no other reason that it is an efficient
# default. Learn more about the Adam optimization algorithm in the paper
# “Adam: A Method for Stochastic Optimization“.
# Finally, because it is a classification problem, we will collect and report
# the classification accuracy as the metric.

# Compile model
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=[ 'accuracy' ] )

# Fit the model
model.fit( X, Y, nb_epoch=150, batch_size=10 )

# evaluate the model
scores = model.evaluate( X, Y )

print("%s: %.2f%%" % (model.metrics_names[ 1 ], scores[ 1 ] * 100))

# calculate predictions
predictions = model.predict( X )

# round predictions
rounded = [ round( x ) for x in predictions ]

print(rounded)
