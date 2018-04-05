#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: conv2d.py


import tensorflow as tf
from .common import layer_register, VariableHolder
from ..tfutils.common import get_tf_version_number
from ..utils.argtools import shape2d, shape4d, get_data_format
from .tflayer import rename_get_variable, convert_to_tflayer_args
import numpy as np

__all__ = ['Conv2D', 'Deconv2D', 'Conv2DTranspose']


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Conv2D(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        split=1):
    """
    A wrapper around `tf.layers.Conv2D`.
    Some differences to maintain backward-compatibility:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'.
    3. Support 'split' argument to do group conv.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    if split == 1:
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            layer = tf.layers.Conv2D(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer)
            ret = layer.apply(inputs, scope=tf.get_variable_scope())
            ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=layer.kernel)
        if use_bias:
            ret.variables.b = layer.bias

    else:
        # group conv implementation
        data_format = get_data_format(data_format, tfmode=False)
        in_shape = inputs.get_shape().as_list()
        channel_axis = 3 if data_format == 'NHWC' else 1
        in_channel = in_shape[channel_axis]
        assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
        assert in_channel % split == 0

        assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
            "Not supported by group conv now!"

        out_channel = filters
        #assert out_channel % split == 0
        assert dilation_rate == (1, 1) or get_tf_version_number() >= 1.5, 'TF>=1.5 required for group dilated conv'

        kernel_shape = shape2d(kernel_size)
        filter_shape = kernel_shape + [in_channel, out_channel]
        stride = shape4d(strides, data_format=data_format)

        kwargs = dict(data_format=data_format)
        if get_tf_version_number() >= 1.5:
            kwargs['dilations'] = shape4d(dilation_rate, data_format=data_format)

        W = tf.get_variable(
            'W', filter_shape, initializer=kernel_initializer)

        if use_bias:
            b = tf.get_variable('b', [out_channel], initializer=bias_initializer)

        inputs = tf.split(inputs, in_channel, channel_axis)
        #print(inputs)
        kernels = W
        #print("\nkernels1")
        #print(kernels)
        kernels = tf.transpose(kernels, perm=[0,1,3,2])
        #print("\nkernels2")
        #print(kernels)
        kernels = tf.split(kernels, in_channel, 3)
        #print("\nkernels3")
        #print(kernels)
        #print(kernels)
        #outputs = [tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, padding.upper(), **kwargs)
                   #for i, k in zip(inputs, kernels)]
        count = 0
        before = 32
        after = 16
        tmp = 0 
        
        for i in range(after):
	        if(i-3<0):
		        tmp+= 1/np.power(2,-i+3)
	        else:
		        tmp += np.power(2,i-3)

        key = (before - after)/2
        tmp2 = 1/np.power(2,key)
        tmp3 = np.power(2,key+1)

        for i, k in zip(inputs, kernels):
            if(count==0):
		with tf.Session() as sess:
  			print("hello")
                #b = tf.add(i,-1*tf.mod(i,(tf.div(i,i) * tmp2)))
                #c = tf.floor(tf.div(i,tmp3))
                #c = tf.round((tf.div(c,c+0.1)))
                #c = tf.add(c*tmp,c*b*-1)
                #i = tf.add(b,c)
                
                #b = tf.add(k,-1*tf.mod(k,(tf.div(k,k) * tmp2)))
                #c = tf.floor(tf.div(k,tmp3))
                #c = tf.round((tf.div(c,c+0.1)))
                #c = tf.add(c*tmp,c*b*-1)
                #k = tf.add(b,c)
                
                outputs = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, padding.upper(), **kwargs)
                
                
                #b = tf.add(outputs,-1*tf.mod(outputs,(tf.div(outputs,outputs) * tmp2)))
                #c = tf.floor(tf.div(outputs,tmp3))
                #c = tf.round((tf.div(c,c+0.1)))
                #c = tf.add(c*tmp,c*b*-1)
                #outputs = tf.add(b,c)
		
            else:
                
                #b = tf.add(i,-1*tf.mod(i,(tf.div(i,i) * tmp2)))
                #c = tf.floor(tf.div(i,tmp3))
                #c = tf.round((tf.div(c,c+0.1)))
                #c = tf.add(c*tmp,c*b*-1)
                #i = tf.add(b,c)
               
                #b = tf.add(k,-1*tf.mod(k,(tf.div(k,k) * tmp2)))
                #c = tf.floor(tf.div(k,tmp3))
                #c = tf.round((tf.div(c,c+0.1)))
                #c = tf.add(c*tmp,c*b*-1)
                #k = tf.add(b,c)
                
                outputs2 = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, padding.upper(), **kwargs)
                
                #b = tf.add(outputs2,-1*tf.mod(outputs2,(tf.div(outputs2,outputs2) * tmp2)))
                #c = tf.floor(tf.div(outputs2,tmp3))
                #c = tf.round((tf.div(c,c+0.1)))
                #c = tf.add(c*tmp,c*b*-1)
                #outputs2 = tf.add(b,c)
                outputs = tf.add(outputs, outputs2)
                
                #b = tf.add(outputs,-1*tf.mod(outputs,(tf.div(outputs,outputs) * tmp2)))
                #c = tf.floor(tf.div(outputs,tmp3))
                #c = tf.round((tf.div(c,c+0.1)))
                #c = tf.add(c*tmp,c*b*-1)
                #outputs = tf.add(b,c)
            count+=1
            
        #print("\noutputs")
        #print(outputs)
        conv = outputs#tf.concat(outputs, channel_axis)
        #conv = tf.reduce_sum(outputs,0)
        #print("\nconv")
        #print(conv)
        if activation is None:
            activation = tf.identity
        ret = activation(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')

        ret.variables = VariableHolder(W=W)
        if use_bias:
            ret.variables.b = b
    return ret


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size', 'strides'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Conv2DTranspose(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):
    """
    A wrapper around `tf.layers.Conv2DTranspose`.
    Some differences to maintain backward-compatibility:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """

    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())

    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return tf.identity(ret, name='output')


Deconv2D = Conv2DTranspose
