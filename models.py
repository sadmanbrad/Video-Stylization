from tensorflow import keras
import numpy as np
import tensorflow as tf


def make_conv(filter_size, input_layer, kernel_size, strides):
    conv0 = keras.layers.Conv2D(filter_size, kernel_size=kernel_size, strides=strides, padding='same', use_bias=True)(
        input_layer)
    conv0 = keras.layers.BatchNormalization()(conv0)
    conv0 = keras.layers.LeakyReLU(alpha=0.2)(conv0)
    return conv0


def make_resnet(filter_size, input_layer, kernel_size, strides):
    relu0 = keras.layers.ReLU()(input_layer)

    conv0 = keras.layers.Conv2D(filter_size, kernel_size=kernel_size, strides=strides, padding='same', use_bias=True)(
        relu0)
    norm = keras.layers.BatchNormalization()(conv0)

    relu1 = keras.layers.ReLU()(norm)

    conv1 = keras.layers.Conv2D(filter_size, kernel_size=kernel_size, strides=strides, padding='same', use_bias=True)(
        relu1)
    return conv1


def make_upconv(filter_size, input_layer, kernel_size, strides):
    up = keras.layers.UpSampling2D()(input_layer)
    conv = keras.layers.Conv2D(filter_size, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(
        up)
    norm = keras.layers.BatchNormalization()(conv)
    relu = keras.layers.ReLU()(norm)
    return relu


def make_smoother(filter_size, input_layer, kernel_size):
    conv0 = keras.layers.Conv2D(filter_size, kernel_size=kernel_size, padding='same', use_bias=True, activation='relu')(
        input_layer)
    norm = keras.layers.BatchNormalization()(conv0)
    conv1 = keras.layers.Conv2D(filter_size, kernel_size=kernel_size, padding='same', use_bias=True, activation='relu')(
        norm)

    return conv1


def make_generator():
    resnet_blocks = 7
    filters = [32, 64, 128, 128, 128, 64]
    input_channels = 6
    patch_size = (32, 32)
    input_shape = (256, 256, input_channels)

    input_layer = keras.layers.Input(shape=input_shape)
    conv0 = make_conv(filters[0], input_layer, (7, 7), (1, 1))
    conv1 = make_conv(filters[1], conv0, (3, 3), (2, 2))
    conv2 = make_conv(filters[2], conv1, (3, 3), (2, 2))

    resnet_blocks_out = conv2
    print(conv2.shape)
    for i in range(resnet_blocks):
        print(resnet_blocks_out.shape)
        resnet_block = make_resnet(filters[2], resnet_blocks_out, (3, 3), (1, 1))
        print(resnet_block.shape)
        resnet_blocks_out = keras.layers.Add()([resnet_block, resnet_blocks_out])

    upconv2_in = keras.layers.Concatenate()([resnet_blocks_out, conv2])
    upconv2 = make_upconv(filters[3], upconv2_in, (3, 3), (1, 1))

    upconv1_in = keras.layers.Concatenate()([upconv2, conv1])
    upconv1 = make_upconv(filters[4], upconv1_in, (3, 3), (1, 1))

    conv11 = keras.layers.Conv2D(filters[5], kernel_size=(7, 7), strides=(1, 1), padding='same', use_bias=True,
                                 activation='relu')(keras.layers.Concatenate()([upconv1, conv0]))

    conv11a = make_smoother(filters[5], conv11, (3, 3))

    conv12 = keras.layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True,
                                 activation='tanh')(conv11a)
    return keras.Model(inputs=input_layer, outputs=conv12)


def make_discriminator():
    number_of_layers = 2
    filters = 12
    input_channels = 3
    patch_size = (32, 32)
    input_shape = (256, 256, input_channels)

    input_layer = keras.layers.Input(shape=input_shape)

    conv0 = make_conv(filters, input_layer, (4, 4), (2, 2))

    flt_mult, flt_mult_prev = 1, 1
    # n - 1 blocks
    last_layer = conv0
    for layer in range(1, number_of_layers):
        flt_mult = min(2 ** layer, 8)
        last_layer = make_conv(filters * flt_mult, last_layer, (4, 4), (2, 2))

    flt_mult = min(2 ** number_of_layers, 8)

    conv_last = make_conv(last_layer, filters * flt_mult, (4, 4), (1, 1))

    output = keras.layers.Conv2D(1, kernel_size=(4, 4), strides=(1, 1), padding='same', use_bias=True)(conv_last)

    return keras.Model(inputs=input_layer, outputs=output)

