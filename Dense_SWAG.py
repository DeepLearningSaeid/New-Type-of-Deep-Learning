
# coding: utf-8

# In[4]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras import backend as K
from keras import initializers,regularizers,constraints



from keras.callbacks import ReduceLROnPlateau
from keras.engine import Layer
from keras.initializers import RandomNormal,glorot_normal
from keras.layers import AveragePooling1D,AveragePooling2D,AveragePooling3D

from keras.layers import Input, Embedding, LSTM, Dense,concatenate,  Dropout, Flatten,  MaxPool2D, Activation
from keras.layers import InputSpec,MaxPooling1D,MaxPooling2D,MaxPooling3D,activations,concatenate,InputSpec

from keras.legacy import interfaces
from keras.legacy.layers import AtrousConvolution1D
from keras.legacy.layers import AtrousConvolution2D


from keras.utils import conv_utils
from keras.utils import np_utils
from keras.utils import plot_model
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_dump,func_load,deserialize_keras_object,has_arg
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import get_custom_objects
from keras.utils.generic_utils import has_arg


# In[5]:


#Defining New Activation functions
############################################################################
def X_1(x):
    return (0*K.pow(x,1))
get_custom_objects().update({'X_1': Activation(X_1)})
############################################################################
def X_2(x):
    return (K.pow(x,2))/2
get_custom_objects().update({'X_2': Activation(X_2)})
############################################################################
def X_3(x):
    return (K.pow(x,3))/6
get_custom_objects().update({'X_3': Activation(X_3)})
############################################################################
def X_4(x):
    return (K.pow(x,4))/24
get_custom_objects().update({'X_4': Activation(X_4)})
############################################################################
def X_5(x):
    return (K.pow(x,5))/120
get_custom_objects().update({'X_5': Activation(X_5)})
###############################################################################
def X_6(x):
    return (K.pow(x,6))/720
get_custom_objects().update({'X_6': Activation(X_6)})
############################################################################
def X_7(x):
    return (K.pow(x,7))/5040
get_custom_objects().update({'X_7': Activation(X_7)})
############################################################################
def X_8(x):
    return (K.pow(x,8))/40320
get_custom_objects().update({'X_8': Activation(X_8)})
###############################################################################
def X_9(x):
    return (K.pow(x,8))/362880
get_custom_objects().update({'X_9': Activation(X_9)})
###############################################################################


# In[6]:



class Dense_Co(Layer):
    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 hidden_dim=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense_Co, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        if hidden_dim!=None :
            self.hidden_dim = hidden_dim
        else :
            self.hidden_dim=self.units
                

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]     
##########################################################################      
        self.kernel = self.add_weight(shape=(input_dim, self.hidden_dim*6),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
    
    
        self.kernel1 = self.kernel[:, :self.hidden_dim]
        self.kernel2 = self.kernel[:, self.hidden_dim: self.hidden_dim * 2]
        self.kernel3 = self.kernel[:, self.hidden_dim * 2: self.hidden_dim * 3]
        self.kernel4 = self.kernel[:, self.hidden_dim * 3: self.hidden_dim * 4]   
        self.kernel5 = self.kernel[:, self.hidden_dim * 4: self.hidden_dim * 5]   
        self.kernel6 = self.kernel[:, self.hidden_dim * 5:]   

    
    
    
##########################################################################
        self.kernel_all = self.add_weight(shape=(6*self.hidden_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
##########################################################################    
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.hidden_dim*6,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.bias1 = self.bias[:self.hidden_dim]
            self.bias2 = self.bias[self.hidden_dim: self.hidden_dim * 2]
            self.bias3 = self.bias[self.hidden_dim * 2: self.hidden_dim * 3]
            self.bias4 = self.bias[self.hidden_dim * 3: self.hidden_dim * 4]
            self.bias5 = self.bias[self.hidden_dim * 4: self.hidden_dim * 5]
            self.bias6 = self.bias[self.hidden_dim * 5:]
            
            
###########################################################################
            self.bias_all = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
    
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output1 = K.dot(inputs, self.kernel1)
        output2 = K.dot(inputs, self.kernel2) 
        output3 = K.dot(inputs, self.kernel3)
        output4 = K.dot(inputs, self.kernel4)
        output5 = K.dot(inputs, self.kernel5) 
        output6 = K.dot(inputs, self.kernel6)
 
        if self.use_bias:
            output1 = K.bias_add(output1, self.bias1, data_format='channels_last')
            output2 = K.bias_add(output2, self.bias2, data_format='channels_last')
            output3 = K.bias_add(output3, self.bias3, data_format='channels_last')
            output4 = K.bias_add(output4, self.bias4, data_format='channels_last')
            output5 = K.bias_add(output5, self.bias5, data_format='channels_last')
            output6 = K.bias_add(output6, self.bias6, data_format='channels_last')
            
        self.activation= activations.get('X_1')
        output1 = self.activation(output1)

        self.activation= activations.get('X_2')
        output2 = self.activation(output2)

        self.activation= activations.get('X_3')            
        output3 = self.activation(output3) 

        self.activation= activations.get('X_4')
        output4 = self.activation(output4)

        self.activation= activations.get('X_5')
        output5 = self.activation(output5)

        self.activation= activations.get('X_6')            
        output6 = self.activation(output6) 
        output_all=concatenate([output1,output2,output3,output4,output5,output6])

        output_all = K.dot(output_all, self.kernel_all)  
        output_all = K.bias_add(output_all, self.bias_all, data_format='channels_last')
        self.activation= activations.get('linear')
        output_all = self.activation(output_all)
            
        return output_all

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


