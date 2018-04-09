#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: conv2d.py


import tensorflow as tf
from .common import layer_register, VariableHolder
from ..tfutils.common import get_tf_version_number
from ..utils.argtools import shape2d, shape4d, get_data_format
from .tflayer import rename_get_variable, convert_to_tflayer_args
import numpy as np

__all__ = ['sy_quantization']
@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['after'],
    name_mapping={'after': 'after'})
def sy_quantization(
        inputs,
        after = 32,
	activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None
        ):

    if after == 0:
       return inputs

    else:
        before = 32
        tmp = 0.0 
        after_div2=after/2
	
        for i in range(after-1):
	        if((i-after_div2)<0):
		        tmp+= 1.0/np.power(2,-i+after_div2)
	        else:
		        tmp += np.power(2,i-after_div2)

        max_range = tmp
        min_range = -1.0*np.power(2,after_div2-1) 
        range_T = np.power(2,after-1) * 2.0 - 1.0
        range_T_add_1_div_2 = (range_T + 1.0)/2.0
        range_target = max_range - min_range
        range_div_range_T = range_target/range_T
        one_over_range_div_range_T =1/ range_div_range_T

        inputs = tf.round((inputs - min_range) * (one_over_range_div_range_T) - range_T_add_1_div_2)
        inputs = (inputs+range_T_add_1_div_2)
        inputs = min_range+(inputs*range_div_range_T)
        inputs = tf.clip_by_value(inputs,min_range,max_range)
        return tf.identity(inputs, name='output')


