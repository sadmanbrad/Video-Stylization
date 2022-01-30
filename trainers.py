import time

from tensorflow import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


max_training_time = 1200


class Trainer(object):
    def __init__(self, generator_optimizer, discriminator_optimizer, train_dataset, perception_loss_model=None):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.use_image_loss = perception_loss_model is not None
        self.perception_loss_model = perception_loss_model
        self.train_dataset = train_dataset

        self.adversarial_criterion = keras.losses.MeanSquaredError()
        self.adversarial_weight = 0.5
        self.reconstruction_weight = 4.
        self.perception_weight = 6.

        pass

    def compute_discriminator_loss(self, fake_labels, true_labels):
        discriminator_loss = self.adversarial_criterion(fake_labels, np.zeros_like(fake_labels))
        discriminator_loss += self.adversarial_criterion(true_labels, np.ones_like(true_labels))

        return discriminator_loss

    def compute_generator_loss(self, generated, discriminator, batch_y):
        image_loss = 0
        perception_loss = 0

        if self.use_image_loss:
            if generated[0][0].shape != batch_y[0][0].shape:
                if ((batch_y[0][0].shape[0] - generated[0][0].shape[0]) % 2) != 0:
                    raise RuntimeError("batch['post'][0][0].shape[0] - generated[0][0].shape[0] must be even number")
                if generated[0][0].shape[0] != generated[0][0].shape[1] or batch_y[0][0].shape[0] != \
                        batch_y[0][0].shape[1]:
                    raise RuntimeError("And also it is expected to be exact square ... fix it if you want")
                boundary_size = int((batch_y[0][0].shape[0] - generated[0][0].shape[0]) / 2)
                cropped_batch_post = batch_y[:, :, boundary_size: -1 * boundary_size, boundary_size: -1 * boundary_size]
                image_loss = keras.losses.MeanAbsoluteError()(generated, cropped_batch_post)
            else:
                image_loss = keras.losses.MeanAbsoluteError()(generated, batch_y)

        if self.perception_loss_model is not None:
            fake_features = self.perception_loss_model(generated)
            target_features = self.perception_loss_model(batch_y)
            perception_loss = keras.losses.MeanSquaredError()(target_features, fake_features).numpy()

        fake_smiling_labels = discriminator(generated)
        adversarial_loss = self.adversarial_criterion(fake_smiling_labels, np.ones_like(fake_smiling_labels))

        return image_loss, perception_loss, adversarial_loss

    def train(self, generator, discriminator, epochs):
        generator.compile()
        discriminator.compile()
        batch_num = 0
        start = time.time()
        for e in range(epochs):
            np.random.seed()
            for step, (x_train, y_train, y_random) in enumerate(self.train_dataset):

                generated = generator(x_train)
                with tf.GradientTape() as dtape:
                    fake_labels = discriminator(generated, training=True)
                    true_labels = discriminator(y_random, training=True)
                    discriminator_loss = self.compute_discriminator_loss(fake_labels, true_labels)

                dgrads = dtape.gradient(discriminator_loss, discriminator.trainable_weights)

                self.discriminator_optimizer.apply_gradients(zip(dgrads, discriminator.trainable_weights))

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as gtape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    # Output for this mini batch
                    generated = generator(x_train, training=True)

                    # Compute the loss value for this mini batch.
                    image_loss, perception_loss, adversarial_loss = self.compute_generator_loss(generated,
                                                                                                discriminator, y_train)
                    generator_loss = self.reconstruction_weight * image_loss + \
                        self.adversarial_weight * adversarial_loss + \
                        self.perception_weight * perception_loss

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                ggrads = gtape.gradient(generator_loss, generator.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                self.generator_optimizer.apply_gradients(zip(ggrads, generator.trainable_weights))

                batch_num += 1

                if batch_num % 100 == 0:
                    total_time = time.time() - start
                    print(f"Batch num: {batch_num}, totally elapsed {total_time}", flush=True)
                    if total_time > max_training_time and batch_num > 1000:
                        print(f"Finishing training", flush=True)
                        generator.save('generator')
                        return

                if batch_num % 1000 == 0:  # (time.time() - start) > 16:
                    eval_start = time.time()
                    full_image = self.train_dataset.get_single_full_image()
                    generated = generator(full_image.reshape((1,) + full_image.shape))
                    keras.utils.save_img(f'generated/{batch_num//1000:03}.png', generated[0],
                                         data_format='channels_last')
                    print(f"Eval of batch: {batch_num} took {(time.time() - eval_start)}", flush=True)
                    generator.save('generator')
