import tensorflow as tf
import sys
from tensorflow.keras import backend as K
from tensorflow.python.ops import array_ops

@tf.function
def in_Range(range, value):
    return  tf.math.logical_and(tf.math.greater_equal(value, -range), tf.math.less_equal(value, range))

class StimulusLayer(tf.keras.layers.Layer):
  def __init__(self, units=32, threshold=20., range=0.06):
    super(StimulusLayer, self).__init__()
    self.threshold = threshold
    self.range = range
    self.units = units

  def build(self, input_shape):
    w_init = tf.initializers.Zeros()
    # create shape from given partially known input_shape
    dims = []
    
    for dim in input_shape:
        if dim is not None:
            dims.append(dim)
        else:
            dims.append(self.units)
    
    # initialization of state variables
    self.last_inputs = tf.Variable(initial_value=w_init(shape=dims, dtype='float32'), shape=input_shape,  trainable=False)
    self.counters = tf.Variable(initial_value=w_init(shape=dims, dtype='float32'), shape=input_shape, trainable=False)

  def call(self, inputs, training=None):
    trainings_outputs = []
    
    # training variable is assigned by the keras backend
    if training:
        # Change shape of state variables according to batch_size (first dimension of shape)
        if tf.shape(inputs)[0] != tf.shape(self.last_inputs)[0]:
            self.last_inputs.assign(tf.zeros(K.shape(inputs)))
            self.counters.assign(tf.zeros(K.shape(inputs)))
        
        # Change counters according to inputs - last_inputs (check if inputs stay the same)
        # tf.where(condition, true return, false return)
        # tensorflow internal functions work elementwise for the given inputs
        self.counters.assign(tf.where(in_Range(self.range, inputs - self.last_inputs), tf.math.add(self.counters, 1), tf.math.multiply(self.counters, 0)))
        self.counters.assign(tf.where(tf.math.greater_equal(self.counters, 2.*self.threshold), 0., self.counters))
        
        self.last_inputs.assign(inputs)
        # resistence as returning zero for given input
        trainings_outputs = tf.where(tf.math.greater_equal(self.counters, self.threshold), 0., inputs)
    
    # training true => trainings_outputs else inputs
    return K.in_train_phase(trainings_outputs, inputs, training=training)
    
#local test

#x = tf.ones((2, 2, 2, 2))
#layer = StimulusLayer()
#epoch = 23

#for i in range(epoch):
#    x = layer(x)
#    print(x)