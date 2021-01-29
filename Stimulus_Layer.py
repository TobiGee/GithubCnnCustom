import tensorflow as tf
import sys
from tensorflow.keras import backend as K
from tensorflow.python.ops import array_ops

@tf.function
def in_Range(range, value):
    return  tf.math.logical_and(tf.math.greater_equal(value, -range), tf.math.less_equal(value, range))

class StimulusLayer(tf.keras.layers.Layer):
  def __init__(self, units=32, threshold=20., range=0.1):
    super(StimulusLayer, self).__init__()
    self.threshold = threshold
    self.range = range
    self.units = units

  def build(self, input_shape):
    w_init = tf.initializers.Zeros()
    # Bekannte Dimensionen finden
    dims = []
    dims.append(self.units)
    for dim in input_shape:
        if dim is not None:
            dims.append(dim)
    
    # inital comparison layer
    self.last_inputs = tf.Variable(initial_value=w_init(shape=dims, dtype='float32'), shape=input_shape,  trainable=False)
    self.counter_layer = tf.Variable(initial_value=w_init(shape=dims, dtype='float32'), shape=input_shape, trainable=False)

  def call(self, inputs, training=None):
    trainings_outputs = []
    
    if training:
        if tf.shape(inputs)[0] != tf.shape(self.last_inputs)[0]:
            self.last_inputs.assign(tf.zeros(K.shape(inputs)))
            self.counter_layer.assign(tf.zeros(K.shape(inputs)))
        
        self.counter_layer.assign(tf.where(in_Range(self.range, inputs - self.last_inputs), tf.math.add(self.counter_layer, 1), tf.math.multiply(self.counter_layer, 0)))
        self.counter_layer.assign(tf.where(tf.math.greater_equal(self.counter_layer, 2.*self.threshold), 0., self.counter_layer))
        
        self.last_inputs.assign(inputs)
        trainings_outputs = tf.where(tf.math.greater_equal(self.counter_layer, self.threshold), 0., inputs)
    
    return K.in_train_phase(trainings_outputs, inputs, training=training)
    
#local test

#x = tf.ones((2, 2, 2, 2))
#layer = StimulusLayer()
#epoch = 23

#for i in range(epoch):
#    x = layer(x)
#    print(x)