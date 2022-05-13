###
### This file contains the Pix2Pix model, optimized for the purpose of image denoising 
### Largely based on https://www.tensorflow.org/tutorials/generative/pix2pix
### and https://opg.optica.org/boe/fulltext.cfm?uri=boe-12-10-6184&id=458664
### Last updated: 2022/05/04 9:15 AM
###

# Import required libraries
import tensorflow as tf
from matplotlib import pyplot as plt

# Set the number of output channels. In this case, there is 
# only a single output channel (no RGB signal or whatsoever)
OUTPUT_CHANNELS = 1


### Helper functions
def downsample(filters, size, apply_batchnorm=True):
    """
    Define the downsampling steps that will be used in the discriminator and generator.
    Steps: Conv2D -> (BatchNorm) -> LeakyReLU
    """
    # Set random initialization
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    
    # Single convolutional layer
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))
    
    # Followed by BatchNorm (optional)
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    # Finally, a Leaky ReLU layer
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    """
    Define the downsampling steps that will be used in the discriminator and generator.
    Steps: Deconv2D -> BatchNorm -> (Dropout) -> ReLU
    """
    # Set random initialization
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    
    # Add single deconvolution layer 
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    # Followed by Batchnorm and dropout (optionally)
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    # Finally, a ReLU layer
    result.add(tf.keras.layers.ReLU())
    return result


### Define generator
def Generator(input_size=(256,256), n_filters=64, kernel_size=4):
    """
    Define the generator of the Pix2Pix GAN model. Based on the U-Net model
    """
    # Define the inputs to the model
    inputs = tf.keras.layers.Input(shape=input_size)

    # Define the downwards (decoder) stream
    down_stack = [
        downsample(n_filters*1, kernel_size, apply_batchnorm=False),
        downsample(n_filters*2, kernel_size), 
        downsample(n_filters*4, kernel_size), 
        downsample(n_filters*8, kernel_size),
        downsample(n_filters*8, kernel_size),  
        downsample(n_filters*8, kernel_size),  
        #downsample(n_filters*8, kernel_size),  
        #downsample(n_filters*8, kernel_size),  
    ]

    # Define the upwards (encoder) stream
    up_stack = [
        #upsample(n_filters*8, kernel_size, apply_dropout=True), 
        #upsample(n_filters*8, kernel_size, apply_dropout=True), 
        upsample(n_filters*8, kernel_size, apply_dropout=True), 
        upsample(n_filters*8, kernel_size),
        upsample(n_filters*4, kernel_size),
        upsample(n_filters*2, kernel_size),
        upsample(n_filters*1, kernel_size),
    ]

    # Set random initialization
    initializer = tf.random_normal_initializer(0., 0.02)

    # Define the final (deconvolution) layer (single output channel)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, kernel_size,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')

    x = inputs

    # Downsampling the image through the model, holding onto the data 
    # at different stages for the skip connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling the image through the model, concatenating with
    # the 'skip'-connected images
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    # Last layer
    x = last(x)

    # Define as a Keras model and return
    return tf.keras.Model(inputs=inputs, outputs=x)


### Define generator loss
def generator_loss(disc_generated_output, gen_output, target, lambda_value=100,\
                    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)):
    """
    Define the generator loss. Contains both the sigmoid cross-entropy loss of the 
    generated images compared to an array of ones, and the L1 loss between generated and target images.
    
    Total generator loss = GAN loss + LAMBDA * L1-loss.

    A value of 100 for LAMBDA was found by the authors of the Pix2Pix paper.
    """
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # Mean absolute error
    total_gen_loss = gan_loss + (lambda_value * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


### Define discriminator
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
    Conv2D (512, 4, stride=1) --> BatchNorm --> LeakyReLU
     V
    Zero padding --> Conv2D (1, 4x4, stride=1)
    """
    # Set random initialization
    initializer = tf.random_normal_initializer(0., 0.02)

    # Define two types of input: the actual input image and the 'target' image.
    # The target image is either the reference image or the predicted image
    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
    tar = tf.keras.layers.Input(shape=input_shape, name='target_image')

    # Concatenate the inputs
    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    # Perform three downsampling steps
    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    # Followed by zero padding, and another set of Conv2D -> BatchNorm -> LeakyReLU
    # Difference with downsampling is that now, a stride of 1 is used instead of 2
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    # Do another zero padding and Conv2D, end up with a single channel
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    # Define as a Keras model and return
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


### Define discriminator loss
def discriminator_loss(disc_real_output, disc_generated_output,\
                       loss_object= tf.keras.losses.BinaryCrossentropy(from_logits=True)):
    """
    Define the discriminator loss. Requires the real and generated images.
    
    Real loss: sigmoid cross-entropy loss of the real images compared to an array of ones ('real' images)
    Generated loss: sigmoid cross-entropy loss of the generated images and an array of zeros ('fake' images)
    Total discriminator loss = real loss + generated loss
    """

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss