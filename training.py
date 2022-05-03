# Import libraries
import tensorflow as tf
import os
import numpy as np
from datetime import datetime
from time import time
from pix2pix import Generator, Discriminator, generator_loss, discriminator_loss, generate_images
from matplotlib import pyplot as plt
from IPython import display

# Set constants
CHECKPOINT_PATH = './training_checkpoints/'
LOG_PATH        = './logs/'

SUMMARY_WRITER = tf.summary.create_file_writer(
  os.path.join(LOG_PATH, "fit", datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
)



@tf.function
def train_step(input_image, target, step, generator, discriminator, gen_optim, disc_optim):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)

    gen_optim.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    disc_optim.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with SUMMARY_WRITER.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
        tf.summary.scalar('disc_loss', disc_loss, step=step)











def train(X_tr, y_tr, X_val, y_val, epochs, img_shape=(128,896,1), n_layers=64):
    
    # Initialize
    G = Generator(img_shape, n_layers)
    D = Discriminator(img_shape)
    
    gen_optim = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disc_optim = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_prefix = os.path.join(CHECKPOINT_PATH, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optim,
                                    discriminator_optimizer=disc_optim,
                                    generator=G,
                                    discriminator=D)

    for epoch in range(epochs):
        start = time()

        for image_batch, ref_batch in zip(X_tr, y_tr):
            train_step(image_batch[np.newaxis], ref_batch[np.newaxis], epoch, G, D, gen_optim, disc_optim)

        # Produce images as you go
        display.clear_output(wait=True)
        generate_images(G, X_val, y_val)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_images(G, X_val, y_val, plot=True)
