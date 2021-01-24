import tensorflow as tf
from tensorflow.keras import backend as backend

class StimulusLayer(tf.keras.layers.Layer):
  def __init__(self, threshold=20, range=0.5):
    super(StimulusLayer, self).__init__()
    self.threshold = threshold
    self.range = range


  def build(self, input_shape):
  
    # inital comparison layer
    self.last_layer = input_shape
    # counting repeats
    self.counter_layer = input_shape
    

  def call(self, inputs, training=None):
    # reset mit 0
    # inkrement mit 1 + 1/(current_counter_value)
    # test as is
    # test to ignore neurons that are clo
    
    resistant_input = self.resistant_inputs(inputs)
    resistent_input = resistent_input

    self.last_layer = inputs
    
    return backend.in_train_phase(resistant_input, inputs, training=training)

  def resistant_inputs(self, inputs):
    temp = (inputs - self.last_layer)
    
    for index, value in enumerate(temp):
        if self.in_Range(value):
            self.counter_layer[index] += 1
        else:
            self.counter_layer[index] = 0
      
    for index, value in enumerate(self.counter_layer):
        if value >= self.threshold:
            inputs[index] = 0
            
    return inputs

  def in_Range(value):
    if value > -self.range and value < self.range:
        return true
    
    return false
    
layer = StimulusLayer()