#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi

from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Lambda, Layer, InputSpec, Convolution1D, Convolution2D, add, multiply, Activation, Input, concatenate
from keras.layers.convolutional import _Conv
from keras.layers.merge import _Merge
from keras.layers.recurrent import Recurrent
from keras.utils import conv_utils
from keras.models import Model
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from .fft import fft, ifft, fft2, ifft2
from .bn import ComplexBN as complex_normalization
from .bn import sqrt_init
from .init import ComplexInit, ComplexIndependentFilters
from .norm import LayerNormalization, ComplexLayerNorm



def sanitizedInitGet(init):
	"""
 	根据输入的初始化器名称返回对应的初始化器函数。如果输入的名称是预定义的几种情况之一，则直接返回相应的函数或名称；
  	否则，通过initializers.get(init)获取对应的初始化器函数。
 	"""
	# 如果输入的初始化器名称是 "sqrt_init"，返回 sqrt_init 函数
	if   init in ["sqrt_init"]:
		return sqrt_init
	# 如果输入的初始化器名称是以下任意一个，直接返回该名称
	elif init in ["complex", "complex_independent",
	              "glorot_complex", "he_complex"]:
		return init
	# 如果输入的初始化器名称不属于以上任何一种，通过 initializers.get(init) 获取对应的初始化器函数
	else:
		return initializers.get(init)


def sanitizedInitSer(init):
	"""
	这个函数的作用是根据输入的初始化器函数，返回相应的字符串表示。如果输入的初始化器函数是预定义的几种情况之一，则直接返回相应的字符串；
 	否则，通过initializers.serialize(init)将输入的初始化器函数序列化为字符串并返回。
 	"""
	if init in [sqrt_init]:
		return "sqrt_init"
	elif init == "complex" or isinstance(init, ComplexInit):
		return "complex"
	elif init == "complex_independent" or isinstance(init, ComplexIndependentFilters):
		return "complex_independent"
	else:
		return initializers.serialize(init)



class ComplexConv(Layer):
    """
    这是一个 nD 复数卷积层的抽象类。
    此层创建一个复数卷积核，将其与层输入进行卷积，产生输出张量。
    如果 `use_bias` 为 True，则创建一个偏置向量并添加到输出中。
    最后，如果 `activation` 不是 `None`，则也应用于输出。
    # Arguments
    # 参数
        rank: An integer, the rank of the convolution,
            e.g. "2" for 2D convolution.
        rank: 一个整数，卷积的秩，
            例如，"2" 表示二维卷积。
        filters: Integer, the dimensionality of the output space, i.e,
            the number of complex feature maps. It is also the effective number
            of feature maps for each of the real and imaginary parts.
            (i.e. the number of complex filters in the convolution)
        filters: 整数，输出空间的维度，即
            复数特征映射的数量。 它也是每个实部和虚部的特征映射的有效数量。
            （即卷积中的复数滤波器数量）
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        kernel_size: 一个整数或包含 n 个整数的元组/列表，指定
            卷积窗口的维度。
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        strides: 一个整数或包含 n 个整数的元组/列表，
            指定卷积的步幅。
            指定任何不等于 1 的步幅值与指定
            任何不等于 1 的 `dilation_rate` 值不兼容。
        padding: One of `"valid"` or `"same"` (case-insensitive).
        padding: `"valid"` 或 `"same"` 中的一个（不区分大小写）。
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        data_format: 一个字符串，
            其中之一为 `channels_last`（默认值）或 `channels_first`。
            输入中维度的排序。
            `channels_last` 对应于形状为
            `(batch, ..., channels)` 的输入，而 `channels_first` 对应于
            形状为 `(batch, channels, ...)` 的输入。
            默认值为您在 `~/.keras/keras.json` 中找到的 `image_data_format` 值。
            如果从未设置过，则为 "channels_last"。
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        dilation_rate: 一个整数或包含 n 个整数的元组/列表，指定
            用于膨胀卷积的膨胀率。
            目前，指定任何不等于 1 的 `dilation_rate` 值与指定任何不等于 1 的 `strides` 值不兼容。
        activation: Activation function to use
            (see keras.activations).
            If you don't specify anything, no activation is applied
            (i.e. "linear" activation: `a(x) = x`).
        activation: 要使用的激活函数
            (参见 keras.activations)。
            如果没有指定任何内容，则不应用激活函数
            （即“线性”激活：`a(x) = x`）。
        use_bias: Boolean, whether the layer uses a bias vector.
        use_bias: 布尔值，该层是否使用偏置向量。
        normalize_weight: Boolean, whether the layer normalizes its complex
            weights before convolving the complex input.
            The complex normalization performed is similar to the one
            for the batchnorm. Each of the complex kernels are centred and multiplied by
            the inverse square root of covariance matrix.
            Then, a complex multiplication is perfromed as the normalized weights are
            multiplied by the complex scaling factor gamma.
        normalize_weight: 布尔值，该层是否在卷积复数输入之前对其复数权重进行归一化。
            执行的复数归一化与批量归一化类似。 每个复数核都是居中的并且乘以
            协方差矩阵的倒数平方根。
            然后，由于归一化的权重与复数缩放因子 gamma 相乘，因此执行了复数乘法。
        kernel_initializer: Initializer for the complex `kernel` weights matrix.
            By default it is 'complex'. The 'complex_independent'
            and the usual initializers could also be used.
            (see keras.initializers and init.py).
        kernel_initializer: 复数 `kernel` 权重矩阵的初始化器。
            默认情况下为 'complex'。 也可以使用 'complex_independent'
            和通常的初始化器。
            (参见 keras.initializers 和 init.py)。
        bias_initializer: Initializer for the bias vector
            (see keras.initializers).
        bias_initializer: 偏置向量的初始化器
            (参见 keras.initializers)。
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see keras.regularizers).
        kernel_regularizer: 应用于 `kernel` 权重矩阵的正则化函数
            (参见 keras.regularizers)。
        bias_regularizer: Regularizer function applied to the bias vector
            (see keras.regularizers).
        bias_regularizer: 应用于偏置向量的正则化函数
            (参见 keras.regularizers)。
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see keras.regularizers).
        activity_regularizer: 应用于该层输出（其“激活”）的正则化函数
            (参见 keras.regularizers)。
        kernel_constraint: Constraint function applied to the kernel matrix
            (see keras.constraints).
        kernel_constraint: 应用于内核矩阵的约束函数
            (参见 keras.constraints)。
        bias_constraint: Constraint function applied to the bias vector
            (see keras.constraints).
        bias_constraint: 应用于偏置向量的约束函数
            (参见 keras.constraints)。
        spectral_parametrization: Whether or not to use a spectral
            parametrization of the parameters.
        spectral_parametrization: 是否使用参数的频谱参数化。
    """

    def __init__(self, rank,
                 filters,   # filters: 整数，输出空间的维度，
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 normalize_weight=False,
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 gamma_diag_initializer=sqrt_init,
                 gamma_off_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 gamma_diag_regularizer=None,
                 gamma_off_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 gamma_diag_constraint=None,
                 gamma_off_constraint=None,
                 init_criterion='he',
                 seed=None,
                 spectral_parametrization=False,
                 epsilon=1e-7,
                 **kwargs):
        super(ComplexConv, self).__init__(**kwargs)
        self.rank = rank   # rank: 一个整数，卷积的秩，例如，"2" 表示二维卷积。
        self.filters = filters   # filters: 整数，输出空间的维度，
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')   # 卷积窗口的维度
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')   # 卷积的步幅
        self.padding = conv_utils.normalize_padding(padding)   # 填充方式
        self.data_format = 'channels_last' if rank == 1 else conv_utils.normalize_data_format(data_format)   # 数据格式
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')   # 膨胀率
        self.activation = activations.get(activation)   # 激活函数
        self.use_bias = use_bias   # 是否使用偏置
        self.normalize_weight = normalize_weight   # 是否对权重进行初始化
        self.init_criterion = init_criterion    # 初始化准则
        self.spectral_parametrization = spectral_parametrization   # 是否使用频谱参数化
        self.epsilon = epsilon   # 用于数值稳定性
        self.kernel_initializer = sanitizedInitGet(kernel_initializer)   # 内核权重矩阵的初始化器
        self.bias_initializer = sanitizedInitGet(bias_initializer)   # 偏置向量的初始化器
        self.gamma_diag_initializer = sanitizedInitGet(gamma_diag_initializer)   # gamma 对角线的初始化器
        self.gamma_off_initializer = sanitizedInitGet(gamma_off_initializer)   # gamma 非对角线的初始化器
        self.kernel_regularizer = regularizers.get(kernel_regularizer)   # 内核权重矩阵的正则化器
        self.bias_regularizer = regularizers.get(bias_regularizer)   # 偏置向量的正则化器
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)   # gamma 对角线的正则化器
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)   # gammma 非对角线的正则化器
        self.activity_regularizer = regularizers.get(activity_regularizer)   # 输出的正则化器
        self.kernel_constraint = constraints.get(kernel_constraint)   # 内核矩阵的约束函数
        self.bias_constraint = constraints.get(bias_constraint)   # 偏置向量的约束函数
        self.gamma_diag_constraint = constraints.get(gamma_diag_constraint)   # gamma 对角线的约束函数
        self.gamma_off_constraint = constraints.get(gamma_off_constraint)   # gamma 非对角线的约束函数

			 
        if seed is None:    # 如果未提供随机种子
            self.seed = np.random.randint(1, 10e6)   # 生成一个位于1到10e6（即1000000）之间的随机整数作为种子
        else:   # 如果提供了随机种子
            self.seed = seed   # 直接使用提供的种子
        self.input_spec = InputSpec(ndim=self.rank + 2)   # 设置输入规范，ndim 表示输入的维度，为卷积层的维度加上2

     def build(self, input_shape):
        # 确定通道轴的位置
        if self.data_format == 'channels_first':  # 如果数据格式为'channels_first'
            channel_axis = 1  # 通道轴在第二个位置
        else:
            channel_axis = -1  # 通道轴在最后一个位置
        # 检查输入形状是否已定义
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        # 计算输入维度
        input_dim = input_shape[channel_axis] // 2
        # 计算卷积核的形状
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)
        # 如果初始化器为复数或独立复数初始化器
        if self.kernel_initializer in {'complex', 'complex_independent'}:
            # 选择相应的初始化器类
            kls = {'complex': ComplexInit, 'complex_independent': ComplexIndependentFilters}[self.kernel_initializer]
            # 初始化卷积核
            kern_init = kls(
                kernel_size=self.kernel_size,
                input_dim=input_dim,
                weight_dim=self.rank,
                nb_filters=self.filters,
                criterion=self.init_criterion
            )
        else:  # 如果不是复数或独立复数初始化器
            kern_init = self.kernel_initializer  # 使用指定的初始化器
        # 添加权重（卷积核）
        self.kernel = self.add_weight(
            self.kernel_shape,
            initializer=kern_init,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        # 如果需要对权重进行归一化
        if self.normalize_weight:
            gamma_shape = (input_dim * self.filters,)
            # 添加 gamma 对角线权重
            self.gamma_rr = self.add_weight(
                shape=gamma_shape,
                name='gamma_rr',
                initializer=self.gamma_diag_initializer,
                regularizer=self.gamma_diag_regularizer,
                constraint=self.gamma_diag_constraint
            )
            # 添加 gamma 对角线权重
            self.gamma_ii = self.add_weight(
                shape=gamma_shape,
                name='gamma_ii',
                initializer=self.gamma_diag_initializer,
                regularizer=self.gamma_diag_regularizer,
                constraint=self.gamma_diag_constraint
            )
            # 添加 gamma 非对角线权重
            self.gamma_ri = self.add_weight(
                shape=gamma_shape,
                name='gamma_ri',
                initializer=self.gamma_off_initializer,
                regularizer=self.gamma_off_regularizer,
                constraint=self.gamma_off_constraint
            )
        else:  # 如果不需要对权重进行归一化
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None
        # 如果使用偏置
        if self.use_bias:
            bias_shape = (2 * self.filters,)
            # 添加偏置
            self.bias = self.add_weight(
                bias_shape,
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:  # 如果不使用偏置
            self.bias = None
        # 设置输入规范
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim * 2})
        # 标记模型已构建完成
        self.built = True


   def call(self, inputs):
    # 确定通道轴的位置
    channel_axis = 1 if self.data_format == 'channels_first' else -1
    # 计算输入数据的通道数（除以2是因为输入是复数信号，每个通道有实部和虚部）
    input_dim = K.shape(inputs)[channel_axis] // 2
    
    # 根据卷积的rank不同，从卷积核中提取实部和虚部
	"""
 	如果卷积的rank为1（一维卷积），则self.kernel的形状为 (kernel_size, input_dim * 2, filters * 2)，其中kernel_size表示卷积核的大小，input_dim表示输入信号的维度，filters表示输出的特征图数量。
 	如果卷积的rank为1（一维卷积），则复数卷积核的实部和虚部从self.kernel中提取的方式为：
		实部 (f_real)：取所有行、所有列、前filters个通道的部分。
		虚部 (f_imag)：取所有行、所有列、从第filters个通道到最后的部分。

  	如果卷积的rank为2（二维卷积），则self.kernel的形状为 (kernel_size[0], kernel_size[1], input_dim * 2, filters * 2)。在这种情况下，f_real和f_imag同样表示复数卷积核的实部和虚部。
	如果卷积的rank为2（二维卷积），则复数卷积核的实部和虚部从self.kernel中提取的方式为：
		实部 (f_real)：取所有行、所有列、所有深度、前filters个通道的部分。
		虚部 (f_imag)：取所有行、所有列、所有深度、从第filters个通道到最后的部分。

  	如果卷积的rank为3（三维卷积），则self.kernel的形状为 (kernel_size[0], kernel_size[1], kernel_size[2], input_dim * 2, filters * 2)。在这种情况下，f_real和f_imag同样表示复数卷积核的实部和虚部。
	如果卷积的rank为3（三维卷积），则复数卷积核的实部和虚部从self.kernel中提取的方式为：
		实部 (f_real)：取所有行、所有列、所有深度、所有通道、前filters个通道的部分。
		虚部 (f_imag)：取所有行、所有列、所有深度、所有通道、从第filters个通道到最后的部分。
 	"""
    if self.rank == 1:
        f_real = self.kernel[:, :, :self.filters]
        f_imag = self.kernel[:, :, self.filters:]
    elif self.rank == 2:
        f_real = self.kernel[:, :, :, :self.filters]
        f_imag = self.kernel[:, :, :, self.filters:]
    elif self.rank == 3:
        f_real = self.kernel[:, :, :, :, :self.filters]
        f_imag = self.kernel[:, :, :, :, self.filters:]
    
    # 构建卷积参数， convArgs 是一个字典，用于存储卷积操作的参数配置
    convArgs = {
        "strides": self.strides[0] if self.rank == 1 else self.strides,
        "padding": self.padding,
        "data_format": self.data_format,
        "dilation_rate": self.dilation_rate[0] if self.rank == 1 else self.dilation_rate
    }
    """
    创建了一个字典 convFunc，其中键是卷积的维度（1、2 或 3），对应的值是 Keras 中对应维度的卷积函数。
    根据当前卷积层的 rank 属性的值，选择相应维度的卷积函数，并将其赋值给 convFunc 变量，以便后续在调用卷积操作时使用。
    """
    convFunc = {1: K.conv1d, 2: K.conv2d, 3: K.conv3d}[self.rank]  # 根据rank选择卷积函数

    # 如果假设权重是在频谱域中表示的，则对其进行处理
    if self.spectral_parametrization:
        if self.rank == 1:
            # 处理一维卷积核
        elif self.rank == 2:
            # 处理二维卷积核
        elif self.rank == 3:
            # 处理三维卷积核

    # 如果进行权重归一化，则对实部和虚部进行归一化
    if self.normalize_weight:
        # 计算归一化的权重
        
    # 执行复数卷积
    f_real._keras_shape = self.kernel_shape
    f_imag._keras_shape = self.kernel_shape
    cat_kernels_4_real = K.concatenate([f_real, -f_imag], axis=-2)
    cat_kernels_4_imag = K.concatenate([f_imag, f_real], axis=-2)
    cat_kernels_4_complex = K.concatenate([cat_kernels_4_real, cat_kernels_4_imag], axis=-1)
    cat_kernels_4_complex._keras_shape = self.kernel_size + (2 * input_dim, 2 * self.filters)
    output = convFunc(inputs, cat_kernels_4_complex, **convArgs)  # 执行卷积操作

    # 如果使用偏置，则添加偏置
    if self.use_bias:
        output = K.bias_add(output, self.bias, data_format=self.data_format)

    # 如果指定了激活函数，则应用激活函数
    if self.activation is not None:
        output = self.activation(output)

    return output


    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i]
                )
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (2 * self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + (2 * self.filters,) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'normalize_weight': self.normalize_weight,
            'kernel_initializer': sanitizedInitSer(self.kernel_initializer),
            'bias_initializer': sanitizedInitSer(self.bias_initializer),
            'gamma_diag_initializer': sanitizedInitSer(self.gamma_diag_initializer),
            'gamma_off_initializer': sanitizedInitSer(self.gamma_off_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'gamma_diag_regularizer': regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer': regularizers.serialize(self.gamma_off_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'gamma_diag_constraint': constraints.serialize(self.gamma_diag_constraint),
            'gamma_off_constraint': constraints.serialize(self.gamma_off_constraint),
            'init_criterion': self.init_criterion,
            'spectral_parametrization': self.spectral_parametrization,
        }
        base_config = super(ComplexConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ComplexConv1D(ComplexConv):
    """1D complex convolution layer.
    This layer creates a complex convolution kernel that is convolved
    with a complex input layer over a single complex spatial (or temporal) dimension
    to produce a complex output tensor.
    If `use_bias` is True, a bias vector is created and added to the complex output.
    Finally, if `activation` is not `None`,
    it is applied each of the real and imaginary parts of the output.
    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.
    # Arguments
        filters: Integer, the dimensionality of the output space, i.e,
            the number of complex feature maps. It is also the effective number
            of feature maps for each of the real and imaginary parts.
            (i.e. the number of complex filters in the convolution)
            The total effective number of filters is 2 x filters.
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
            `"causal"` results in causal (dilated) convolutions, e.g. output[t]
            does not depend on input[t+1:]. Useful when modeling temporal data
            where the model should not violate the temporal order.
            See [WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499).
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see keras.activations).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        normalize_weight: Boolean, whether the layer normalizes its complex
            weights before convolving the complex input.
            The complex normalization performed is similar to the one
            for the batchnorm. Each of the complex kernels are centred and multiplied by
            the inverse square root of covariance matrix.
            Then, a complex multiplication is perfromed as the normalized weights are
            multiplied by the complex scaling factor gamma.
        kernel_initializer: Initializer for the complex `kernel` weights matrix.
			By default it is 'complex'. The 'complex_independent' 
			and the usual initializers could also be used.
            (see keras.initializers and init.py).
        bias_initializer: Initializer for the bias vector
            (see keras.initializers).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see keras.regularizers).
        bias_regularizer: Regularizer function applied to the bias vector
            (see keras.regularizers).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see keras.regularizers).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see keras.constraints).
        bias_constraint: Constraint function applied to the bias vector
            (see keras.constraints).
        spectral_parametrization: Whether or not to use a spectral
            parametrization of the parameters.
    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`
    # Output shape
        3D tensor with shape: `(batch_size, new_steps, 2 x filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 init_criterion='he',
                 spectral_parametrization=False,
                 **kwargs):
        super(ComplexConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            init_criterion=init_criterion,
            spectral_parametrization=spectral_parametrization,
            **kwargs)

    def get_config(self):
        config = super(ComplexConv1D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config


class ComplexConv2D(ComplexConv):
    """2D Complex convolution layer (e.g. spatial convolution over images).
    This layer creates a complex convolution kernel that is convolved
    with a complex input layer to produce a complex output tensor. If `use_bias` 
    is True, a complex bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to both the
    real and imaginary parts of the output.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    # Arguments
        filters: Integer, the dimensionality of the complex output space
            (i.e, the number complex feature maps in the convolution).
            The total effective number of filters or feature maps is 2 x filters.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see keras.activations).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        normalize_weight: Boolean, whether the layer normalizes its complex
            weights before convolving the complex input.
            The complex normalization performed is similar to the one
            for the batchnorm. Each of the complex kernels are centred and multiplied by
            the inverse square root of covariance matrix.
            Then, a complex multiplication is perfromed as the normalized weights are
            multiplied by the complex scaling factor gamma.
        kernel_initializer: Initializer for the complex `kernel` weights matrix.
			By default it is 'complex'. The 'complex_independent' 
			and the usual initializers could also be used.
            (see keras.initializers and init.py).
        bias_initializer: Initializer for the bias vector
            (see keras.initializers).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see keras.regularizers).
        bias_regularizer: Regularizer function applied to the bias vector
            (see keras.regularizers).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see keras.regularizers).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see keras.constraints).
        bias_constraint: Constraint function applied to the bias vector
            (see keras.constraints).
        spectral_parametrization: Whether or not to use a spectral
            parametrization of the parameters.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, 2 x filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, 2 x filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 init_criterion='he',
                 spectral_parametrization=False,
                 **kwargs):
        super(ComplexConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
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
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            init_criterion=init_criterion,
            spectral_parametrization=spectral_parametrization,
            **kwargs)

    def get_config(self):
        config = super(ComplexConv2D, self).get_config()
        config.pop('rank')
        return config


class ComplexConv3D(ComplexConv):
    """3D convolution layer (e.g. spatial convolution over volumes).
    This layer creates a complex convolution kernel that is convolved
    with a complex layer input to produce a complex output tensor.
    If `use_bias` is True,
    a complex bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to each of the real and imaginary
    parts of the output.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(2, 128, 128, 128, 3)` for 128x128x128 volumes
    with 3 channels,
    in `data_format="channels_last"`.
    # Arguments
        filters: Integer, the dimensionality of the complex output space
            (i.e, the number complex feature maps in the convolution).
            The total effective number of filters or feature maps is 2 x filters.
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            width and height of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along each spatial dimension.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `channels_first` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 3 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see keras.activations).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        normalize_weight: Boolean, whether the layer normalizes its complex
            weights before convolving the complex input.
            The complex normalization performed is similar to the one
            for the batchnorm. Each of the complex kernels are centred and multiplied by
            the inverse square root of covariance matrix.
            Then, a complex multiplication is perfromed as the normalized weights are
            multiplied by the complex scaling factor gamma.
        kernel_initializer: Initializer for the complex `kernel` weights matrix.
			By default it is 'complex'. The 'complex_independent' 
			and the usual initializers could also be used.
            (see keras.initializers and init.py).
        bias_initializer: Initializer for the bias vector
            (see keras.initializers).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see keras.regularizers).
        bias_regularizer: Regularizer function applied to the bias vector
            (see keras.regularizers).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see keras.regularizers).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see keras.constraints).
        bias_constraint: Constraint function applied to the bias vector
            (see keras.constraints).
        spectral_parametrization: Whether or not to use a spectral
            parametrization of the parameters.
    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if data_format='channels_last'.
    # Output shape
        5D tensor with shape:
        `(samples, 2 x filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, 2 x filters)` if data_format='channels_last'.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 init_criterion='he',
                 spectral_parametrization=False,
                 **kwargs):
        super(ComplexConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
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
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            init_criterion=init_criterion,
            spectral_parametrization=spectral_parametrization,
            **kwargs)

    def get_config(self):
        config = super(ComplexConv3D, self).get_config()
        config.pop('rank')
        return config


class WeightNorm_Conv(_Conv):
	# Real-valued Convolutional Layer that normalizes its weights
	# before convolving the input.
	# The weight Normalization performed the one
	# described in the following paper:
	# Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
	# (see https://arxiv.org/abs/1602.07868)

    def __init__(self,
                 gamma_initializer='ones',
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 epsilon=1e-07,
                 **kwargs):
        super(WeightNorm_Conv, self).__init__(**kwargs)
        if self.rank == 1:
            self.data_format = 'channels_last'
        self.gamma_initializer = sanitizedInitGet(gamma_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.epsilon = epsilon

    def build(self, input_shape):
        super(WeightNorm_Conv, self).build(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        gamma_shape = (input_dim * self.filters,)
        self.gamma = self.add_weight(
            shape=gamma_shape,
            name='gamma',
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint
        )

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        ker_shape = self.kernel_size + (input_dim, self.filters)
        nb_kernels = ker_shape[-2] * ker_shape[-1]
        kernel_shape_4_norm = (np.prod(self.kernel_size), nb_kernels)
        reshaped_kernel = K.reshape(self.kernel, kernel_shape_4_norm)
        normalized_weight = K.l2_normalize(reshaped_kernel, axis=0, epsilon=self.epsilon)
        normalized_weight = K.reshape(self.gamma, (1, ker_shape[-2] * ker_shape[-1])) * normalized_weight
        shaped_kernel = K.reshape(normalized_weight, ker_shape)
        shaped_kernel._keras_shape = ker_shape
        
        convArgs = {"strides":       self.strides[0]       if self.rank == 1 else self.strides,
                    "padding":       self.padding,
                    "data_format":   self.data_format,
                    "dilation_rate": self.dilation_rate[0] if self.rank == 1 else self.dilation_rate}
        convFunc = {1: K.conv1d,
                    2: K.conv2d,
                    3: K.conv3d}[self.rank]
        output = convFunc(inputs, shaped_kernel, **convArgs)

        if self.use_bias:
            output = K.bias_add(
                output,
                self.bias,
                data_format=self.data_format
            )

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = {
            'gamma_initializer': sanitizedInitSer(self.gamma_initializer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'epsilon': self.epsilon
        }
        base_config = super(WeightNorm_Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# Aliases

ComplexConvolution1D = ComplexConv1D
ComplexConvolution2D = ComplexConv2D
ComplexConvolution3D = ComplexConv3D
