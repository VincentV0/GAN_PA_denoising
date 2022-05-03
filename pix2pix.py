

import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt

OUTPUT_CHANNELS = 1

def downsample(filters, size, apply_batchnorm=True):
    """
    Define the downsampling steps that will be used in the discriminator and generator.
    Steps: Conv2D -> (BatchNorm) -> LeakyReLU
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))
    
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    """
    Define the downsampling steps that will be used in the discriminator and generator.
    Steps: Deconv2D -> BatchNorm -> (Dropout) -> ReLU
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator(input_size=(256,256), n_filters=64):
    """
    Define the generator of the Pix2Pix GAN model. Based on the U-Net model
    """
    inputs = tf.keras.layers.Input(shape=input_size)

    down_stack = [
        downsample(n_filters*1, 4, apply_batchnorm=False),
        downsample(n_filters*2, 4), 
        downsample(n_filters*4, 4), 
        downsample(n_filters*8, 4),
        downsample(n_filters*8, 4),  
        downsample(n_filters*8, 4),  
        downsample(n_filters*8, 4),  
        #downsample(n_filters*8, 4),  
    ]

    up_stack = [
        #upsample(n_filters*8, 4, apply_dropout=True), 
        upsample(n_filters*8, 4, apply_dropout=True), 
        upsample(n_filters*8, 4, apply_dropout=True), 
        upsample(n_filters*8, 4),
        upsample(n_filters*4, 4),
        upsample(n_filters*2, 4),
        upsample(n_filters*1, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)





def generator_loss(disc_generated_output, gen_output, target, lambda_value=100,\
                    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)):
    """
    Define the generator loss. Contains both the sigmoid cross-entropy loss of the 
    generated images compared to an array of ones, and the L1 loss between generated and target images.
    
    Total generator loss = GAN loss + LAMBDA * L1-loss.

    A value of 100 for LAMBDA was found by the authors of the Pix2Pix paper.
    """
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (lambda_value * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def Discriminator(input_shape=(256,256,1)):
    """
    Define the discriminator of the Pix2Pix GAN model.
    Structure:
    
    Conv2D (64; 4x4)  --> LeakyReLU
     v
    Conv2D (128; 4x4) --> BatchNorm --> LeakyReLU
     v
    Conv2D (256; 4x4) --> BatchNorm --> LeakyReLU
     v
    Zero padding
     V 
    Conv2D (516, 4, stride=1) --> BatchNorm --> LeakyReLU
     V
    Zero padding --> Conv2D (1, 4x4, stride=1)
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
    tar = tf.keras.layers.Input(shape=input_shape, name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output,\
                       loss_object= tf.keras.losses.BinaryCrossentropy(from_logits=True)):
    """
    Define the discriminator loss. Requires the real and generated images.
    
    Real loss: sigmoid cross-entropy loss of the real images compared to an array of ones ('real' images)
    Generated loss: sigmoid cross-entropy loss of the generated images and an array of zeros ('fake' images)
    Total discriminator loss = real loss + generated loss
    """

    # Real loss
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    # Generated loss
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    # Total loss
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generate_images(model, test_input, target=None, plot=False):
    prediction = model(test_input, training=True)

    if plot:
        plt.figure(figsize=(15, 15))
        display_list = [test_input[0], target[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()
    return prediction