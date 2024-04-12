#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi

#
# Implementation of Layer Normalization and Complex Layer Normalization
#

import numpy as np
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
import keras.backend as K
from .bn import ComplexBN as complex_normalization
from .bn import sqrt_init 


def layernorm(x, axis, epsilon, gamma, beta):
    # 确定输入张量的形状和归一化轴
    input_shape = K.shape(x)
    reduction_axes = list(range(K.ndim(x)))  # 获取张量的所有维度
    del reduction_axes[axis]  # 删除归一化轴
    del reduction_axes[0]  # 删除 batch 维度
    broadcast_shape = [1] * K.ndim(x)  # 初始化广播形状
    broadcast_shape[axis] = input_shape[axis]  # 设置广播形状的归一化轴维度为输入张量的归一化轴维度
    broadcast_shape[0] = K.shape(x)[0]  # 设置广播形状的 batch 维度

    # 计算均值
    mean = K.mean(x, axis=reduction_axes)
    broadcast_mean = K.reshape(mean, broadcast_shape)  # 将均值进行广播以与输入张量相同的形状
    x_centred = x - broadcast_mean  # 中心化
    variance = K.mean(x_centred ** 2, axis=reduction_axes) + epsilon  # 计算方差，添加 epsilon 避免除零
    broadcast_variance = K.reshape(variance, broadcast_shape)  # 将方差进行广播以与输入张量相同的形状

    x_normed = x_centred / K.sqrt(broadcast_variance)  # 归一化操作

    # 缩放和平移
    broadcast_shape_params = [1] * K.ndim(x)  # 初始化广播形状
    broadcast_shape_params[axis] = K.shape(x)[axis]  # 设置广播形状的归一化轴维度为输入张量的归一化轴维度
    broadcast_gamma = K.reshape(gamma, broadcast_shape_params)  # 将缩放因子进行广播以与输入张量相同的形状
    broadcast_beta = K.reshape(beta, broadcast_shape_params)  # 将平移因子进行广播以与输入张量相同的形状

    x_LN = broadcast_gamma * x_normed + broadcast_beta  # 缩放和平移

    return x_LN

class LayerNormalization(Layer):
    
    def __init__(self,
                 epsilon=1e-4,
                 axis=-1,
                 beta_init='zeros',
                 gamma_init='ones',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 **kwargs):

        self.supports_masking = True
        self.beta_init = initializers.get(beta_init)  # 初始化平移因子
        self.gamma_init = initializers.get(gamma_init)  # 初始化缩放因子
        self.epsilon = epsilon  # 设置 epsilon，用于计算方差时避免除零
        self.axis = axis  # 设置归一化轴
        self.gamma_regularizer = regularizers.get(gamma_regularizer)  # 缩放因子的正则化器
        self.beta_regularizer = regularizers.get(beta_regularizer)  # 平移因子的正则化器

        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: input_shape[self.axis]})  # 设置输入张量的规范化
        shape = (input_shape[self.axis],)  # 设置缩放和平移因子的形状

        self.gamma = self.add_weight(shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name))  # 添加缩放因子参数
        self.beta = self.add_weight(shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name))  # 添加平移因子参数

        self.built = True  # 设置构建标志为已构建

    def call(self, x, mask=None):
        assert self.built, 'Layer must be built before being called'  # 断言层已构建
        return layernorm(x, self.axis, self.epsilon, self.gamma, self.beta)  # 调用自定义的归一化函数进行归一化操作

    def get_config(self):
        config = {'epsilon':           self.epsilon,  # epsilon 参数
                  'axis':              self.axis,  # 归一化轴参数
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,  # 缩放因子的正则化配置
                  'beta_regularizer':  self.beta_regularizer.get_config()  if self.beta_regularizer  else None  # 平移因子的正则化配置
                  }
        base_config = super(LayerNormalization, self).get_config()  # 获取基础配置
        return dict(list(base_config.items()) + list(config.items()))  # 返回配置字典


class ComplexLayerNorm(Layer):
    def __init__(self,
                 epsilon=1e-4,  # 初始化参数：归一化过程中的epsilon，用于防止除以零
                 axis=-1,  # 归一化操作的轴，默认为最后一个轴
                 center=True,  # 是否进行中心化，默认为True
                 scale=True,  # 是否进行缩放，默认为True
                 beta_initializer='zeros',  # beta参数的初始化方法，默认为全零初始化
                 gamma_diag_initializer=sqrt_init,  # gamma对角元素的初始化方法，默认为sqrt_init
                 gamma_off_initializer='zeros',  # gamma非对角元素的初始化方法，默认为全零初始化
                 beta_regularizer=None,  # beta参数的正则化方法，默认为None
                 gamma_diag_regularizer=None,  # gamma对角元素的正则化方法，默认为None
                 gamma_off_regularizer=None,  # gamma非对角元素的正则化方法，默认为None
                 beta_constraint=None,  # beta参数的约束条件，默认为None
                 gamma_diag_constraint=None,  # gamma对角元素的约束条件，默认为None
                 gamma_off_constraint=None,  # gamma非对角元素的约束条件，默认为None
                 **kwargs):

        self.supports_masking = True  # 是否支持掩码操作
        self.epsilon = epsilon  # 初始化参数：归一化过程中的epsilon
        self.axis = axis  # 归一化操作的轴
        self.center = center  # 是否进行中心化
        self.scale = scale  # 是否进行缩放
        self.beta_initializer = initializers.get(beta_initializer)  # 初始化方法转化为函数对象
        self.gamma_diag_initializer = initializers.get(gamma_diag_initializer)  # 初始化方法转化为函数对象
        self.gamma_off_initializer = initializers.get(gamma_off_initializer)  # 初始化方法转化为函数对象
        self.beta_regularizer = regularizers.get(beta_regularizer)  # 正则化方法转化为函数对象
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)  # 正则化方法转化为函数对象
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)  # 正则化方法转化为函数对象
        self.beta_constraint = constraints.get(beta_constraint)  # 约束条件转化为函数对象
        self.gamma_diag_constraint = constraints.get(gamma_diag_constraint)  # 约束条件转化为函数对象
        self.gamma_off_constraint = constraints.get(gamma_off_constraint)  # 约束条件转化为函数对象
        super(ComplexLayerNorm, self).__init__(**kwargs)  # 调用父类的初始化方法

    def build(self, input_shape):
        ndim = len(input_shape)  # 输入张量的维度数
        dim = input_shape[self.axis]  # 归一化操作轴上的维度大小
        if dim is None:  # 如果轴上的维度大小未定义，抛出异常
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})  # 输入规范化对象

        gamma_shape = (input_shape[self.axis] // 2,)  # gamma参数的形状，实部和虚部维度大小一半
        if self.scale:  # 如果进行缩放操作
            self.gamma_rr = self.add_weight(
                shape=gamma_shape,
                name='gamma_rr',
                initializer=self.gamma_diag_initializer,  # 对角元素初始化方法
                regularizer=self.gamma_diag_regularizer,  # 对角元素正则化方法
                constraint=self.gamma_diag_constraint  # 对角元素约束条件
            )
            self.gamma_ii = self.add_weight(
                shape=gamma_shape,
                name='gamma_ii',
                initializer=self.gamma_diag_initializer,  # 对角元素初始化方法
                regularizer=self.gamma_diag_regularizer,  # 对角元素正则化方法
                constraint=self.gamma_diag_constraint  # 对角元素约束条件
            )
            self.gamma_ri = self.add_weight(
                shape=gamma_shape,
                name='gamma_ri',
                initializer=self.gamma_off_initializer,  # 非对角元素初始化方法
                regularizer=self.gamma_off_regularizer,  # 非对角元素正则化方法
                constraint=self.gamma_off_constraint  # 非对角元素约束条件
            )
        else:
            self.gamma_rr = None  # 如果不进行缩放，则设置为None
            self.gamma_ii = None  # 如果不进行缩放，则设置为None
            self.gamma_ri = None  # 如果不进行缩放，则设置为None

        if self.center:  # 如果进行中心化操作
            self.beta = self.add_weight(shape=(input_shape[self.axis],),
                                        name='beta',
                                        initializer=self.beta_initializer,  # beta参数初始化方法
                                        regularizer=self.beta_regularizer,  # beta参数正则化方法
                                        constraint=self.beta_constraint)  # beta参数约束条件
        else:
            self.beta = None  # 如果不进行中心化，则设置为None

        self.built = True  # 设置构建标志为True

    def call(self, inputs):
        input_shape = K.shape(inputs)  # 输入张量的形状
        ndim = K.ndim(inputs)  # 输入张量的维度数
        reduction_axes = list(range(ndim))  # 归一化操作中需要减少的轴列表
        del reduction_axes[self.axis]  # 删除归一化轴
        del reduction_axes[0]  # 删除batch轴
        input_dim = input_shape[self.axis] // 2  # 输入维度大小的一半（因为是复数，所以实部和虚部维度相同）
        mu = K.mean(inputs, axis=reduction_axes)  # 计算均值
        broadcast_mu_shape = [1] * ndim  # 广播均值的形状
        broadcast_mu_shape[self.axis] = input_shape[self.axis]  # 设置广播均值的形状
        broadcast_mu_shape[0] = K.shape(inputs)[0]  # 设置batch的大小
        broadcast_mu = K.reshape(mu, broadcast_mu_shape)  # 广播均值
        if self.center:  # 如果进行中心化操作
            input_centred = inputs - broadcast_mu  # 中心化
        else:
            input_centred = inputs  # 不进行中心化
        centred_squared = input_centred ** 2  # 中心化后的张量的平方
        if (self.axis == 1 and ndim != 3) or ndim == 2:  # 如果轴是1且维度不是3，或者维度为2
            centred_squared_real = centred_squared[:, :input_dim]  # 实部的平方
            centred_squared_imag = centred_squared[:, input_dim:]  # 虚部的平方
            centred_real = input_centred[:, :input_dim]  # 实部
            centred_imag = input_centred[:, input_dim:]  # 虚部
        elif ndim == 3:  # 如果维度为3
            centred_squared_real = centred_squared[:, :, :input_dim]  # 实部的平方
            centred_squared_imag = centred_squared[:, :, input_dim:]  # 虚部的平方
            centred_real = input_centred[:, :, :input_dim]  # 实部
            centred_imag = input_centred[:, :, input_dim:]  # 虚部
        elif self.axis == -1 and ndim == 4:  # 如果轴是-1且维度为4
            centred_squared_real = centred_squared[:, :, :, :input_dim]  # 实部的平方
            centred_squared_imag = centred_squared[:, :, :, input_dim:]  # 虚部的平方
            centred_real = input_centred[:, :, :, :input_dim]  # 实部
            centred_imag = input_centred[:, :, :, input_dim:]  # 虚部
        elif self.axis == -1 and ndim == 5:  # 如果轴是-1且维度为5
            centred_squared_real = centred_squared[:, :, :, :, :input_dim]  # 实部的平方
            centred_squared_imag = centred_squared[:, :, :, :, input_dim:]  # 虚部的平方
            centred_real = input_centred[:, :, :, :, :input_dim]  # 实部
            centred_imag = input_centred[:, :, :, :, input_dim:]  # 虚部
        else:  # 其他情况
            raise ValueError(
                'Incorrect Layernorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
            )
        if self.scale:  # 如果进行缩放
            Vrr = K.mean(
                centred_squared_real,
                axis=reduction_axes
            ) + self.epsilon  # 计算实部方差
            Vii = K.mean(
                centred_squared_imag,
                axis=reduction_axes
            ) + self.epsilon  # 计算虚部方差
            # Vri contains the real and imaginary covariance for each feature map.
            Vri = K.mean(
                centred_real * centred_imag,
                axis=reduction_axes,
            )  # 计算实部和虚部的协方差
        elif self.center:  # 如果进行中心化但不进行缩放
            Vrr = None  # 实部方差为None
            Vii = None  # 虚部方差为None
            Vri = None  # 协方差为None
        else:  # 其他情况
            raise ValueError('Error. Both scale and center in batchnorm are set to False.')

        return complex_normalization(  # 调用复数归一化函数
            input_centred, Vrr, Vii, Vri,
            self.beta, self.gamma_rr, self.gamma_ri,
            self.gamma_ii, self.scale, self.center,
            layernorm=True, axis=self.axis
        )

    def get_config(self):
        config = {
            'axis': self.axis,  # 归一化操作的轴
            'epsilon': self.epsilon,  # 归一化过程中的epsilon
            'center': self.center,  # 是否进行中心化
            'scale': self.scale,  # 是否进行缩放
            'beta_initializer': initializers.serialize(self.beta_initializer),  # beta参数的初始化方法
            'gamma_diag_initializer': initializers.serialize(self.gamma_diag_initializer),  # gamma对角元素的初始化方法
            'gamma_off_initializer': initializers.serialize(self.gamma_off_initializer),  # gamma非对角元素的初始化方法
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),  # beta参数的正则化方法
            'gamma_diag_regularizer': regularizers.serialize(self.gamma_diag_regularizer),  # gamma对角元素的正则化方法
            'gamma_off_regularizer': regularizers.serialize(self.gamma_off_regularizer),  # gamma非对角元素的正则化方法
            'beta_constraint': constraints.serialize(self.beta_constraint),  # beta参数的约束条件
            'gamma_diag_constraint': constraints.serialize(self.gamma_diag_constraint),  # gamma对角元素的约束条件
            'gamma_off_constraint': constraints.serialize(self.gamma_off_constraint),  # gamma非对角元素的约束条件
        }
        base_config = super(ComplexLayerNorm, self).get_config()  # 获取父类的配置信息
        return dict(list(base_config.items()) + list(config.items()))  # 返回合并后的配置信息
