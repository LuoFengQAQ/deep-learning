import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping

class Focus(Layer):
    def __init__(self):
        super(Focus,self).__init__()

    def compute_output_shape(self,input_shape):
        pass
