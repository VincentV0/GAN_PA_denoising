###
### This file contains the functions that are required for training the
### Pix2Pix GAN model. For that reason, it uses the functions from 
### pix2pix.py.
### Last updated: 2022/05/04 9:15 AM
###

# Import libraries
import tensorflow as tf
import os
import numpy as np
from datetime import datetime
from time import time
from pix2pix import Generator, Discriminator, generator_loss, discriminator_loss
from matplotlib import pyplot as plt
from tqdm import tqdm

from keras import backend as K_backend
K_backend.set_image_data_format('channels_last')
tf.config.run_functions_eagerly(True)

# Set constants
CHECKPOINT_PATH = './training_checkpoints/'
LOG_PATH        = './logs/'

# Define TensorBoard writer
SUMMARY_WRITER = tf.summary.create_file_writer(
  os.path.join(LOG_PATH, "tb", datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
)

# Define batch generator (used in training loop)
def batch_generator(x, n):
    """
    X: data
    n: batch size
    """
    start, stop = 0, n
    while True:
        if start < stop:
            yield x[start:stop]
        else:
            break
        start = stop
        stop = (stop + n) % len(x)


### Do a single training step
@tf.function
def train_step(input_image, target, step, generator, discriminator, gen_optim, disc_optim, L1_lambda=100):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Let the model generate an image based on an input image, let it train as well
        gen_output = generator(input_image, training=True)

        # Let the discriminator learn, first input an input image and the reference image ('real'), 
        # then the input image and the generated output ('fake')
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # Calculate the losses for both models 
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target, lambda_value=L1_lambda)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # Calculate the gradients for both models based on their losses
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)

    # Optimize the models by applying the calculated gradients
    gen_optim.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    disc_optim.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    # Write the losses to TensorBoard
    with SUMMARY_WRITER.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
        tf.summary.scalar('disc_loss', disc_loss, step=step)
    
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss





### Main training function
def model_train(X_tr, y_tr, X_val, y_val, epochs, img_shape=(128,128), n_layers=64, lr=2e-4, \
                L1_lambda = 100, batch_size=5):
    
    # Initialize models
    G = Generator(img_shape, n_layers)
    D = Discriminator(img_shape)

    # Initialize Adam optimizers
    gen_optim = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    disc_optim = tf.keras.optimizers.Adam(lr, beta_1=0.5)

    # Set up checkpoints for saving the model
    checkpoint_prefix = os.path.join(CHECKPOINT_PATH, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optim,
                                    discriminator_optimizer=disc_optim,
                                    generator=G,
                                    discriminator=D)
    
    # Set up dictionary to save the losses
    global losses_dict 
    losses_dict = {'gen_total_loss': [], \
                   'gen_gan_loss'  : [], \
                   'gen_l1_loss'   : [], \
                   'disc_loss'     : []}

    # Start the loop for X epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch}: ", end='')
        # Set a timer
        start = time()

        # initiate batch generator
        batches_X = batch_generator(X_tr, batch_size)
        batches_y = batch_generator(y_tr, batch_size)

        for image_batch, ref_batch in tqdm(zip(batches_X, batches_y)):
            # Change data types for compatibility
            image_batch = image_batch.astype('float32')
            ref_batch   = ref_batch.astype('float32')
            gen_tot, gen_gan, gen_l1, disc = train_step(image_batch, ref_batch, epoch, G, D, gen_optim, disc_optim, L1_lambda)

        # Write the losses from the end of the epoch to a dictionary
        losses_dict['gen_total_loss'].append(gen_tot.numpy())
        losses_dict['gen_gan_loss'].append(gen_gan.numpy())
        losses_dict['gen_l1_loss'].append(gen_l1.numpy())
        losses_dict['disc_loss'].append(disc.numpy())

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time()-start))

    return G, D, losses_dict