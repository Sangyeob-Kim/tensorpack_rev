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
       
        before = 32
        tmp = 0.0 
	
        after2 = after
        before = 32
        tmp = 0.0 
        after_div2=after/2
	
#         for i in range(after-1):
# 	        if((i-after_div2)<0):
# 		        tmp+= 1.0/np.power(2,-i+after_div2)
# 	        else:
# 		        tmp += np.power(2,i-after_div2)

#         max_range = tmp
#         min_range = -1.0*np.power(2,after_div2-1) 
#         range_T = np.power(2,after-1) * 2.0 - 1.0
#         range_T_add_1_div_2 = (range_T + 1.0)/2.0
#         range_target = max_range - min_range
#         range_div_range_T = range_target/range_T
#         one_over_range_div_range_T =1/ range_div_range_T
        for i in range(after2-1):
            if(i-f_part<0):
                tmp+= 1.0/np.power(2,-i+f_part)
            else:
                tmp+= np.power(2,i-f_part)

        min = 1.0/np.power(2,f_part)
        print(tmp)
        print(min)
        W = tf.get_variable(
            'W', filter_shape2, initializer=kernel_initializer)
        W1 = W[0:1,0:1,:,:]
        W2 = W[1:2,0:1,:,:]
        W3 = W[2:3,0:1,:,:]
        W4 = W[0:1,1:2,:,:]
        W5 = W[1:2,1:2,:,:]
        W6 = W[2:3,1:2,:,:]
        W7 = W[0:1,2:3,:,:]
        W8 = W[1:2,2:3,:,:]
        W9 = W[2:3,2:3,:,:]
#         W1 = tf.get_variable(
#             'W1', filter_shape, initializer=kernel_initializer)
#         W2 = tf.get_variable(
#             'W2', filter_shape, initializer=kernel_initializer)
#         W3 = tf.get_variable(
#             'W3', filter_shape, initializer=kernel_initializer)
#         W4 = tf.get_variable(
#             'W4', filter_shape, initializer=kernel_initializer)
#         W5 = tf.get_variable(
#             'W5', filter_shape, initializer=kernel_initializer)
#         W6 = tf.get_variable(
#             'W6', filter_shape, initializer=kernel_initializer)
#         W7 = tf.get_variable(
#             'W7', filter_shape, initializer=kernel_initializer)
#         W8 = tf.get_variable(
#             'W8', filter_shape, initializer=kernel_initializer)
#         W9 = tf.get_variable(
#             'W9', filter_shape, initializer=kernel_initializer)	
	
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
        with G.gradient_override_map({"Round": "Identity",
                        "Minimum" : "Jump",
                        "Maximum" : "Jump",
                        "LessEqual" : "Jump",
                        "GreaterEqual" : "Jump",
                        "Select" : "Identity",
                        "Reshape" : "Identity",
                        "Sub": "Jump",
                        "Div": "Jump",
                        "Add": "Jump",
                        "Sign" : "Identity",
                        "Abs" : "Identity",
                        "Floor" : "Identity",
                       	"Div" : "Jump",
                       	"RealDiv" : "Jump",      
                        "Mul": "Jump"}):
            #inputs = tf.Print(inputs,[inputs[0]])
            #tf.summary.histogram(inputs.name, inputs)
            y = tf.sign(inputs)
            inputs = tf.abs(inputs)
            inputs = tf.floor(inputs / min)
            inputs = inputs * (min)
            inputs = tf.clip_by_value(inputs,0,tmp)
            inputs = inputs*y
            #inputs = tf.Print(inputs,[inputs[0]])

        #ith g.gradient_override_map({"Round": "Identity"}), g.gradient_override_map({"Clip_by_value": "Identity"}):
        #   W = tf.round((W - min_range) * (one_over_range_div_range_T) - range_T_add_1_div_2)
        #   W = (W+range_T_add_1_div_2)
        #   W = min_range+(W*range_div_range_T)
        #   W = tf.clip_by_value(W,min_range,max_range)
        shape = inputs.shape
        b = shape[0]
        w = shape[1]
        h = shape[2]
        c = shape[3]
        
        inputs = tf.transpose(inputs, perm=[0,3,1,2])
        if(padding.upper()=="SAME"):
          paddings = tf.constant([[0,0],[0,0],[1,1],[1,1]])
          inputs = tf.pad(inputs,paddings,"constant")
        inputs = tf.transpose(inputs, perm=[0,2,3,1])

        inputs1 = tf.split(inputs[:,0:w-1,0:h-1,:], in_channel, channel_axis)
        inputs2 = tf.split(inputs[:,0:w-1,1:h,:], in_channel, channel_axis)
        inputs3 = tf.split(inputs[:,0:w-1,2:h+1,:], in_channel, channel_axis)
        inputs4 = tf.split(inputs[:,1:w,0:h-1,:], in_channel, channel_axis)
        inputs5 = tf.split(inputs[:,1:w,1:h,:], in_channel, channel_axis)
        inputs6 = tf.split(inputs[:,1:w,2:h+1,:], in_channel, channel_axis)
        inputs7 = tf.split(inputs[:,2:w+1,0:h-1,:], in_channel, channel_axis)
        inputs8 = tf.split(inputs[:,2:w+1,1:h,:], in_channel, channel_axis)
        inputs9 = tf.split(inputs[:,2:w+1,2:h+1,:], in_channel, channel_axis)
	
        kernels1 = W1
        kernels1 = tf.transpose(kernels1, perm=[0,1,3,2])
        kernels1 = tf.split(kernels1, in_channel, 3)

        kernels2 = W2
        kernels2 = tf.transpose(kernels2, perm=[0,1,3,2])
        kernels2 = tf.split(kernels2, in_channel, 3)

        kernels3 = W3
        kernels3 = tf.transpose(kernels3, perm=[0,1,3,2])
        kernels3 = tf.split(kernels3, in_channel, 3)
	
        kernels4 = W4
        kernels4 = tf.transpose(kernels4, perm=[0,1,3,2])
        kernels4 = tf.split(kernels4, in_channel, 3)
	
        kernels5 = W5
        kernels5 = tf.transpose(kernels5, perm=[0,1,3,2])
        kernels5 = tf.split(kernels5, in_channel, 3)
	
        kernels6 = W6
        kernels6 = tf.transpose(kernels6, perm=[0,1,3,2])
        kernels6 = tf.split(kernels6, in_channel, 3)
	
        kernels7 = W7
        kernels7 = tf.transpose(kernels7, perm=[0,1,3,2])
        kernels7 = tf.split(kernels7, in_channel, 3)
	
        kernels8 = W8
        kernels8 = tf.transpose(kernels8, perm=[0,1,3,2])
        kernels8 = tf.split(kernels8, in_channel, 3)
	
        kernels9 = W9
        kernels9 = tf.transpose(kernels9, perm=[0,1,3,2])
        kernels9 = tf.split(kernels9, in_channel, 3)
        count = 0	

        #shape = tf.shape(inputs)

        #print(shape,shape[0],shape[1],shape[2],shape[3])
        for i, k in zip(inputs1, kernels1):
            if(count==0):
                with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(g_after)+"bit"}):
                    i = tf.identity(i)
                    k = tf.identity(k)
		
                outputs = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, "VALID", **kwargs)

                with G.gradient_override_map({"Round": "Jump",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / min)
                    outputs = outputs * min
                    outputs = tf.clip_by_value(outputs,0,tmp)
                    outputs = outputs*y
		
            else:
                with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(g_after)+"bit"}):
                    i = tf.identity(i)
                    k = tf.identity(k)
		
                outputs2 = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, "VALID", **kwargs)

                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs2)
                    outputs2 = tf.abs(outputs2)
                    outputs2 = tf.floor(outputs2 / min)
                    outputs2 = outputs2 * min
                    outputs2 = tf.clip_by_value(outputs2,0,tmp)
                    outputs2 = outputs2*y
                #with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(after)+"bit"}):
                #    outputs = tf.identity(outputs)
		
                outputs = tf.add(outputs, outputs2)
                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / min)
                    outputs = outputs * min
                    outputs = tf.clip_by_value(outputs,0,tmp)
                    outputs = outputs*y
            count+=1
	
        for i, k in zip(inputs2, kernels2):
		
                with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(g_after)+"bit"}):
                    i = tf.identity(i)
                    k = tf.identity(k)
		
                outputs2 = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, "VALID", **kwargs)

                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs2)
                    outputs2 = tf.abs(outputs2)
                    outputs2 = tf.floor(outputs2 / min)
                    outputs2 = outputs2 * min
                    outputs2 = tf.clip_by_value(outputs2,0,tmp)
                    outputs2 = outputs2*y
                #with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(after)+"bit"}):
                #    outputs = tf.identity(outputs)
		
                outputs = tf.add(outputs, outputs2)
                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / min)
                    outputs = outputs * min
                    outputs = tf.clip_by_value(outputs,0,tmp)
                    outputs = outputs*y	
			
        for i, k in zip(inputs3, kernels3):
                with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(g_after)+"bit"}):
                    i = tf.identity(i)
                    k = tf.identity(k)
		
                outputs2 = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, "VALID", **kwargs)

                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs2)
                    outputs2 = tf.abs(outputs2)
                    outputs2 = tf.floor(outputs2 / min)
                    outputs2 = outputs2 * min
                    outputs2 = tf.clip_by_value(outputs2,0,tmp)
                    outputs2 = outputs2*y
                #with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(after)+"bit"}):
                #    outputs = tf.identity(outputs)
		
                outputs = tf.add(outputs, outputs2)
                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / min)
                    outputs = outputs * min
                    outputs = tf.clip_by_value(outputs,0,tmp)
                    outputs = outputs*y
			
        for i, k in zip(inputs4, kernels4):

                with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(g_after)+"bit"}):
                    i = tf.identity(i)
                    k = tf.identity(k)
		
                outputs2 = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, "VALID", **kwargs)

                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs2)
                    outputs2 = tf.abs(outputs2)
                    outputs2 = tf.floor(outputs2 / min)
                    outputs2 = outputs2 * min
                    outputs2 = tf.clip_by_value(outputs2,0,tmp)
                    outputs2 = outputs2*y
                #with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(after)+"bit"}):
                #    outputs = tf.identity(outputs)
		
                outputs = tf.add(outputs, outputs2)
                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / min)
                    outputs = outputs * min
                    outputs = tf.clip_by_value(outputs,0,tmp)
                    outputs = outputs*y
			
        for i, k in zip(inputs5, kernels5):
                with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(g_after)+"bit"}):
                    i = tf.identity(i)
                    k = tf.identity(k)
		
                outputs2 = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, "VALID", **kwargs)

                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs2)
                    outputs2 = tf.abs(outputs2)
                    outputs2 = tf.floor(outputs2 / min)
                    outputs2 = outputs2 * min
                    outputs2 = tf.clip_by_value(outputs2,0,tmp)
                    outputs2 = outputs2*y
                #with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(after)+"bit"}):
                #    outputs = tf.identity(outputs)
		
                outputs = tf.add(outputs, outputs2)
                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / min)
                    outputs = outputs * min
                    outputs = tf.clip_by_value(outputs,0,tmp)
                    outputs = outputs*y
			
        for i, k in zip(inputs6, kernels6):
                with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(g_after)+"bit"}):
                    i = tf.identity(i)
                    k = tf.identity(k)
		
                outputs2 = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, "VALID", **kwargs)

                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs2)
                    outputs2 = tf.abs(outputs2)
                    outputs2 = tf.floor(outputs2 / min)
                    outputs2 = outputs2 * min
                    outputs2 = tf.clip_by_value(outputs2,0,tmp)
                    outputs2 = outputs2*y
                #with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(after)+"bit"}):
                #    outputs = tf.identity(outputs)
		
                outputs = tf.add(outputs, outputs2)
                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / min)
                    outputs = outputs * min
                    outputs = tf.clip_by_value(outputs,0,tmp)
                    outputs = outputs*y

        for i, k in zip(inputs7, kernels7):
                with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(g_after)+"bit"}):
                    i = tf.identity(i)
                    k = tf.identity(k)
		
                outputs2 = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, "VALID", **kwargs)

                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs2)
                    outputs2 = tf.abs(outputs2)
                    outputs2 = tf.floor(outputs2 / min)
                    outputs2 = outputs2 * min
                    outputs2 = tf.clip_by_value(outputs2,0,tmp)
                    outputs2 = outputs2*y
                #with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(after)+"bit"}):
                #    outputs = tf.identity(outputs)
		
                outputs = tf.add(outputs, outputs2)
                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / min)
                    outputs = outputs * min
                    outputs = tf.clip_by_value(outputs,0,tmp)
                    outputs = outputs*y
			
        for i, k in zip(inputs8, kernels8):
                with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(g_after)+"bit"}):
                    i = tf.identity(i)
                    k = tf.identity(k)
		
                outputs2 = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride,"VALID", **kwargs)

                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs2)
                    outputs2 = tf.abs(outputs2)
                    outputs2 = tf.floor(outputs2 / min)
                    outputs2 = outputs2 * min
                    outputs2 = tf.clip_by_value(outputs2,0,tmp)
                    outputs2 = outputs2*y
                #with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(after)+"bit"}):
                #    outputs = tf.identity(outputs)
		
                outputs = tf.add(outputs, outputs2)
                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / min)
                    outputs = outputs * min
                    outputs = tf.clip_by_value(outputs,0,tmp)
                    outputs = outputs*y
			
        for i, k in zip(inputs9, kernels9):
                with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(g_after)+"bit"}):
                    i = tf.identity(i)
                    k = tf.identity(k)
		
                outputs2 = tf.nn.conv2d(i, tf.transpose(k, perm=[0,1,3,2]), stride, "VALID", **kwargs)

                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs2)
                    outputs2 = tf.abs(outputs2)
                    outputs2 = tf.floor(outputs2 / min)
                    outputs2 = outputs2 * min
                    outputs2 = tf.clip_by_value(outputs2,0,tmp)
                    outputs2 = outputs2*y
                #with G.gradient_override_map({"Identity" : "CustomGrad_for_conv_"+str(after)+"bit"}):
                #    outputs = tf.identity(outputs)
		
                outputs = tf.add(outputs, outputs2)
                with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / min)
                    outputs = outputs * min
                    outputs = tf.clip_by_value(outputs,0,tmp)
                    outputs = outputs*y	
        activate_before = 32
        activate_tmp = 0.0 
     
        for i in range(activate_after-1):
            if(i-activate_f_part<0):
                activate_tmp+= 1.0/np.power(2,-i+activate_f_part)
            else:
                activate_tmp+= np.power(2,i-activate_f_part)

        activate_min = 1.0/np.power(2,activate_f_part)
	
        with G.gradient_override_map({"Round": "Identity",
                                "Minimum" : "Jump",
                                "Maximum" : "Jump",
                                "LessEqual" : "Jump",
                                "GreaterEqual" : "Jump",
                                "Select" : "Identity",
                                "Reshape" : "Identity",
                                "Sub": "Jump",
                                "Div": "Jump",
                                "Add": "Jump",
                                "Sign" : "Identity",
                                "Abs" : "Identity",
                                "Floor" : "Identity",
                                "Div" : "Jump",
                       	        "RealDiv" : "Jump",
                                "Mul": "Jump"}):
                    y = tf.sign(outputs)
                    outputs = tf.abs(outputs)
                    outputs = tf.floor(outputs / activate_min)
                    outputs = outputs * activate_min
                    outputs = tf.clip_by_value(outputs,0,activate_tmp)
                    outputs = outputs*y
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
