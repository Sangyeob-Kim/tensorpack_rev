#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: conv2d.py


import tensorflow as tf
from .common import layer_register, VariableHolder
from ..tfutils.common import get_tf_version_number
from ..utils.argtools import shape2d, shape4d, get_data_format
from .tflayer import rename_get_variable, convert_to_tflayer_args
import numpy as np
from tensorflow.python.framework import function
from tensorflow.python.framework import ops

__all__ = ['Conv2D', 'Deconv2D', 'Conv2DTranspose']

@tf.RegisterGradient("CustomGrad_for_conv_32bit")
def customGrad(op, x):
    before = 32
    after = 32
    tmp = 0.0 
    after2 = after
    after_div2=after2/2
    tmp = 0
    for i in range(after2-1):
        tmp+= 1.0/np.power(2,i)

    y = tf.sign(x)
    x = tf.abs(x)
    x = tf.floor(x / (1/np.power(2,after2-2)))
    x = x * (1.0/np.power(2,after2-2))
    x = tf.clip_by_value(x,1.0/np.power(2,after2-2),tmp)
    x = y*x
    return x
    #return [tf.ones(tf.shpae(grad)), tf.zeros(tf.shape(op.inputs[1]))]

@tf.RegisterGradient("CustomGrad_for_conv_24bit")
def customGrad(op, x):
    before = 32
    after = 24
    tmp = 0.0 
    after2 = after
    after_div2=after2/2
    tmp = 0
    for i in range(after2-1):
        tmp+= 1.0/np.power(2,i)

    y = tf.sign(x)
    x = tf.abs(x)
    x = tf.floor(x / (1/np.power(2,after2-2)))
    x = x * (1.0/np.power(2,after2-2))
    x = tf.clip_by_value(x,1.0/np.power(2,after2-2),tmp)
    x = y*x
    return x
    #return [tf.ones(tf.shpae(grad)), tf.zeros(tf.shape(op.inputs[1]))]
	
@tf.RegisterGradient("CustomGrad_for_conv_16bit")
def customGrad(op, x):
    before = 32
    after = 16
    tmp = 0.0 
    after2 = after
    after_div2=after2/2
    tmp = 0
    for i in range(after2-1):
        tmp+= 1.0/np.power(2,i)

    y = tf.sign(x)
    x = tf.abs(x)
    x = tf.floor(x / (1/np.power(2,after2-2)))
    x = x * (1.0/np.power(2,after2-2))
    x = tf.clip_by_value(x,1.0/np.power(2,after2-2),tmp)
    x = y*x
    return x

@tf.RegisterGradient("CustomGrad_for_conv_8bit")
def customGrad(op, x):
    before = 32
    after = 8
    tmp = 0.0 
    after2 = after
    after_div2=after2/2
    tmp = 0
    for i in range(after2-1):
        tmp+= 1.0/np.power(2,i)

    y = tf.sign(x)
    x = tf.abs(x)
    x = tf.floor(x / (1/np.power(2,after2-2)))
    x = x * (1.0/np.power(2,after2-2))
    x = tf.clip_by_value(x,1.0/np.power(2,after2-2),tmp)
    x = x*y
    return x

@tf.RegisterGradient("CustomGrad_for_conv_4bit")
def customGrad(op, x):
    before = 32
    after = 4
    tmp = 0.0 
    after2 = after
    after_div2=after2/2
    tmp = 0
    for i in range(after2-1):
        tmp+= 1.0/np.power(2,i)

    y = tf.sign(x)
    x = tf.abs(x)
    x = tf.floor(x / (1/np.power(2,after2-2)))
    x = x * (1.0/np.power(2,after2-2))
    x = tf.clip_by_value(x,1.0/np.power(2,after2-2),tmp)
    x = x*y
    return x
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
        split=1,
        after = 32,
        f_part = after-2,
	g_after = 32
        ):
	
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
        G = tf.get_default_graph()
        # group conv implementation
        data_format = get_data_format(data_format, tfmode=False)
        in_shape = inputs.get_shape().as_list()
        channel_axis = 3 if data_format == 'NHWC' else 1
        in_channel = in_shape[channel_axis]
        assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
        #assert in_channel % split == 0

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
       
        before = 32
        tmp = 0.0 
	
        after2 = after
        after_div2=after2/2

        for i in range(after2-1):
            if(i-f_part<0):
                tmp+= 1.0/np.power(2,-i+f_part)
            else:
                tmp+= np.power(2,i-f_part)

        min = 1/np.power(1,f_part)
	
        W = tf.get_variable(
            'W', filter_shape, initializer=kernel_initializer)

        if use_bias:
            b = tf.get_variable('b', [out_channel], initializer=bias_initializer)
            #b = tf.round((b - min_range) * (one_over_range_div_range_T) - range_T_add_1_div_2)
            #b = (b+range_T_add_1_div_2)
            #b = min_range+(b*range_div_range_T)
            #b = tf.clip_by_value(b,min_range,max_range)
        #with g.gradient_override_map({"Round": "Identity"}), g.gradient_override_map({"Clip_by_value": "Identity"}):
        #    inputs = tf.round((inputs - min_range) * (one_over_range_div_range_T) - range_T_add_1_div_2)
        #    inputs = (inputs+range_T_add_1_div_2)
        #    inputs = min_range+(inputs*range_div_range_T)
        #   inputs = tf.clip_by_value(inputs,min_range,max_range)

        #ith g.gradient_override_map({"Round": "Identity"}), g.gradient_override_map({"Clip_by_value": "Identity"}):
        #   W = tf.round((W - min_range) * (one_over_range_div_range_T) - range_T_add_1_div_2)
        #   W = (W+range_T_add_1_div_2)
        #   W = min_range+(W*range_div_range_T)
        #   W = tf.clip_by_value(W,min_range,max_range)
		
        inputs = tf.split(inputs, in_channel, channel_axis)
        kernels = W
        kernels = tf.transpose(kernels, perm=[0,1,3,2])
        kernels = tf.split(kernels, in_channel, 3)
        count = 0	
	
        for i, k in zip(inputs, kernels):
            if(count==0):
                with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(g_after)+"bit"}):
                    i = tf.identity(i)
                    k = tf.identity(k)
		
                outputs = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, padding.upper(), **kwargs)

                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Add",
                                "Maximum" : "Add",
                                "LessEqual" : "Add",
                                "GreaterEqual" : "Add",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Add",
                                "Div": "Add",
                                "Add": "Add",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Mul": "Add"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / min)
                    outputs = outputs * min
                    outputs = tf.clip_by_value(outputs,min,tmp)
                    outputs = outputs*y
		
            else:
                with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(g_after)+"bit"}):
                    i = tf.identity(i)
                    k = tf.identity(k)
		
                outputs2 = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, padding.upper(), **kwargs)

                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Add",
                                "Maximum" : "Add",
                                "LessEqual" : "Add",
                                "GreaterEqual" : "Add",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Add",
                                "Div": "Add",
                                "Add": "Add",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Mul": "Add"}):
                    y = tf.sign(outputs2)
                    outputs2 = tf.abs(outputs2)
                    outputs2 = tf.floor(outputs2 / min)
                    outputs2 = outputs2 * min
                    outputs2 = tf.clip_by_value(outputs2,min,tmp)
                    outputs2 = outputs2*y
                #with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(after)+"bit"}):
                #    outputs = tf.identity(outputs)
		
                outputs = tf.add(outputs, outputs2)
                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Add",
                                "Maximum" : "Add",
                                "LessEqual" : "Add",
                                "GreaterEqual" : "Add",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Add",
                                "Div": "Add",
                                "Add": "Add",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Mul": "Add"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / min)
                    outputs = outputs * min
                    outputs = tf.clip_by_value(outputs,min,tmp)
                    outputs = outputs*y
            count+=1

        conv = outputs
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
        activity_regularizer=None,
        quantization=1):
	
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
