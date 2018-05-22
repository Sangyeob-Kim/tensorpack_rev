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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.ops import nn_ops

__all__ = ['Conv2D', 'Deconv2D', 'Conv2DTranspose']

@ops.RegisterGradient("Jump")
def customGrad(op, grad):
    return [None, grad]

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
    x = tf.floor(x / (1.0/np.power(2,after2-2)))
    x = x * (1.0/np.power(2,after2-2))
    x = tf.clip_by_value(x,0,tmp)
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
    x = tf.floor(x / (1.0/np.power(2,after2-2)))
    x = x * (1.0/np.power(2,after2-2))
    x = tf.clip_by_value(x,0,tmp)
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
    x = tf.floor(x / (1.0/np.power(2,after2-2)))
    x = x * (1.0/np.power(2,after2-2))
    x = tf.clip_by_value(x,0,tmp)
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
    x = tf.floor(x / (1.0/np.power(2,after2-2)))
    x = x * (1.0/np.power(2,after2-2))
    x = tf.clip_by_value(x,0,tmp)
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
    x = tf.floor(x / (1.0/np.power(2,after2-2)))
    x = x * (1.0/np.power(2,after2-2))
    x = tf.clip_by_value(x,0,tmp)
    x = x*y
    return x

# #For padding conv2d
# @ops.RegisterGradient("Conv2D_rev")
# def Conv2D_rev(op, grad):
# 	dilations = op.get_attr("dilations")
# 	strides = op.get_attr("strides")
# 	padding = op.get_attr("padding")
# 	use_cudnn_on_gpu = op.get_attr("use_cudnn_on_gpu")
# 	data_format = op.get_attr("data_format")
# 	shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])

# 	shape0 = op.inputs[0].get_shape().as_list()

# 	inputs = tf.pad(op.inputs[0], tf.constant([[0,0],[1,1],[1,1],[0,0]]),"constant")

# 	for i in range(shape0[0]):
# 		temp_input = [inputs[i,:,:,:]]
# 		temp_input = tf.transpose(temp_input, perm=[3,1,2,0])	
# 		temp_grad = tf.transpose([grad[i,:,:,:]],perm=[1,2,0,3])
		
# 		temp_out = tf.nn.conv2d(temp_input,temp_grad,strides,"VALID")
	
# 		if(i==0):
# 			shape3 = temp_input.get_shape().as_list()
# 			shape1 = temp_out.get_shape().as_list()
# 			grad_w = tf.zeros(shape1,tf.float32)

# 		grad_w = grad_w + temp_out

# 	grad_w = tf.transpose(grad_w, perm=[1,2,0,3])


# 	kernel = op.inputs[1]	

# 	kernel_T = tf.transpose(kernel,perm=[3,0,1,2]) 
# 	kernel_T = tf.image.rot90(kernel_T,k=2)
# 	kernel_T = tf.transpose(kernel_T,perm=[1,2,0,3])
# 	pad_size = shape1[1]-2
# 	pad = tf.constant([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
# 	grad_rev = tf.pad(grad, pad, "constant")

# 	grad_x = tf.nn.conv2d(grad_rev,kernel_T,strides,"VALID")

# 	return[
# 		grad_x,
# 		grad_w
# 	]

#For no padding conv2d
@ops.RegisterGradient("Conv2D_no_padding")
def Conv2D_no_padding(op, grad):
	dilations = op.get_attr("dilations")
	strides = op.get_attr("strides")
	padding = op.get_attr("padding")
	use_cudnn_on_gpu = op.get_attr("use_cudnn_on_gpu")
	data_format = op.get_attr("data_format")
	shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])

	shape0 = op.inputs[0].get_shape().as_list()

	inputs = op.inputs[0]

# 	for i in range(shape0[0]):
# 		temp_input = [inputs[i,:,:,:]]
# 		temp_input = tf.transpose(temp_input, perm=[3,1,2,0])	
# 		temp_grad = tf.transpose([grad[i,:,:,:]],perm=[1,2,0,3])
		
# 		temp_out = tf.nn.conv2d(temp_input,temp_grad,strides,"VALID")
		
# 		if(i==0):
# 			shape1 = temp_out.get_shape().as_list()
# 			grad_w = tf.zeros(shape1,tf.float32)

# 		grad_w = grad_w + temp_out

	
	temp_input = inputs
	temp_input = tf.transpose(temp_input, perm=[3,1,2,0])	
	temp_grad = tf.transpose(grad,perm=[1,2,0,3])
		
	temp_out = tf.nn.conv2d(temp_input,temp_grad,strides,"VALID")
		
	
	shape1 = temp_out.get_shape().as_list()
	grad_w = tf.zeros(shape1,tf.float32)
	
	grad_w = grad_w + temp_out
	grad_w = tf.transpose(grad_w, perm=[1,2,0,3])

	kernel = op.inputs[1]	
	kernel_T = tf.zeros(shape1,tf.float32)

	 
	kernel_T = tf.transpose(kernel,perm=[3,0,1,2]) 
	kernel_T = tf.image.rot90(kernel_T,k=2)
# 	kernel_T = tf.transpose(kernel_T,perm=[1,2,0,3])
# 	pad_size = shape1[1]-1
# 	pad = tf.constant([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
# 	grad_rev = tf.pad(grad, pad, "constant")

# 	grad_x = tf.nn.conv2d(grad_rev,kernel_T,strides,"VALID")

	return[
		#grad_x,
		#grad_w
		nn_ops.conv2d_backprop_input(
		shape_0,
		op.inputs[1],
		grad,
		dilations=dilations,
		strides=strides,
		padding=padding,
		use_cudnn_on_gpu=use_cudnn_on_gpu,
		data_format=data_format
		),

# 		nn_ops.conv2d_backprop_filter(
# 		op.inputs[0],
# 		shape_1,
# 		grad,
# 		dilations=dilations,
# 		strides=strides,
# 		padding=padding,
# 		use_cudnn_on_gpu=use_cudnn_on_gpu,
# 		data_format=data_format
# 		)
		grad_w
	]


#For padding conv2d
@ops.RegisterGradient("Conv2D_with_padding")
def Conv2D_with_padding(op, grad):
	dilations = op.get_attr("dilations")
	strides = op.get_attr("strides")
	padding = op.get_attr("padding")
	use_cudnn_on_gpu = op.get_attr("use_cudnn_on_gpu")
	data_format = op.get_attr("data_format")
	shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])

	shape0 = op.inputs[0].get_shape().as_list()
	shape2 = grad.get_shape().as_list()

	inputs = tf.pad(op.inputs[0], tf.constant([[0,0],[1,1],[1,1],[0,0]]),"constant")

# 	for i in range(shape0[0]):
# 		temp_input = [inputs[i,:,:,:]]
# 		temp_input = tf.transpose(temp_input, perm=[3,1,2,0])	
# 		temp_grad = tf.transpose([grad[i,:,:,:]],perm=[1,2,0,3])
			
# 		temp_out = tf.nn.conv2d(temp_input,temp_grad,strides,"VALID")
		
# 		if(i==0):
# 			shape3 = temp_input.get_shape().as_list()
# 			shape1 = temp_out.get_shape().as_list()
# 			grad_w = tf.zeros(shape1,tf.float32)

# 		grad_w = grad_w + temp_out

	temp_input = inputs
	temp_input = tf.transpose(temp_input, perm=[3,1,2,0])	
	temp_grad = tf.transpose(grad,perm=[1,2,0,3])
		
	temp_out = tf.nn.conv2d(temp_input,temp_grad,strides,"VALID")
# 	shape3 = temp_input.get_shape().as_list()
	shape1 = temp_out.get_shape().as_list()
	grad_w = tf.zeros(shape1,tf.float32)

	grad_w = grad_w + temp_out
		
	grad_w = tf.transpose(grad_w, perm=[1,2,0,3])

	kernel = op.inputs[1]	
	
	kernel_T = tf.transpose(kernel,perm=[3,0,1,2]) 
# 	kernel_T = tf.image.rot90(kernel_T,k=2)

	kernel_T = tf.image.flip_up_down(kernel_T)
	kernel_T = tf.image.flip_left_right(kernel_T)

# 	kernel_T = tf.transpose(kernel_T,perm=[1,2,0,3])
# 	pad_size = shape1[1]-2
# 	pad = tf.constant([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
# 	grad_rev = tf.pad(grad, pad, "constant")

# 	grad_x = tf.nn.conv2d(grad_rev,kernel_T,strides,"VALID")

	return[
		#grad_x,
		#grad_w
		nn_ops.conv2d_backprop_input(
		shape_0,
		op.inputs[1],
		grad,
		dilations=dilations,
		strides=strides,
		padding=padding,
		use_cudnn_on_gpu=use_cudnn_on_gpu,
		data_format=data_format
		),

# 		nn_ops.conv2d_backprop_filter(
# 		op.inputs[0],
# 		shape_1,
# 		grad,
# 		dilations=dilations,
# 		strides=strides,
# 		padding=padding,
# 		use_cudnn_on_gpu=use_cudnn_on_gpu,
# 		data_format=data_format
# 		)
		grad_w
	]


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
        f_part = 30,
	g_after = 32,
	activate_after = 0,
	activate_f_part = 0
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
        filter_shape = [1,1] + [in_channel, out_channel]
        filter_shape2 = kernel_shape + [in_channel, out_channel]
        stride = shape4d(strides, data_format=data_format)

        kwargs = dict(data_format=data_format)
        if get_tf_version_number() >= 1.5:
            kwargs['dilations'] = shape4d(dilation_rate, data_format=data_format)
       
        W = tf.get_variable(
            'W', filter_shape2, initializer=kernel_initializer)

        kernels = W
	
        if use_bias:
            b = tf.get_variable('b', [out_channel], initializer=bias_initializer)

        #shape = tf.shape(inputs)
        if(padding.upper()=="VALID"):
          with G.gradient_override_map({"Conv2D" : "Conv2D_no_padding"}):
            outputs = tf.nn.conv2d(inputs, kernels, stride, padding.upper(), **kwargs)
        elif(padding.upper()=="SAME"):
          with G.gradient_override_map({"Conv2D" : "Conv2D_with_padding"}):
            outputs = tf.nn.conv2d(inputs, kernels, stride, padding.upper(), **kwargs)
	
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
