from tensorflow import keras
import numpy as np
import tensorflow as tf


class Trainer(object):
    def __init__(self, generator_optimizer, discriminator_optimizer, perception_loss_model=None):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.use_image_loss = perception_loss_model is not None
        self.perception_loss_model = perception_loss_model

        pass

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
            perception_loss = ((fake_features - target_features) ** 2).mean()

        fake_smiling_labels = discriminator(generated)
        adversarial_loss = keras.losses.MeanSquaredError()(fake_smiling_labels, np.ones_like(fake_smiling_labels))

        return image_loss, perception_loss, adversarial_loss, generated

    def train(self, generator, discriminator, epochs):
        np.random.seed()
        for e in range(epochs):
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    # Output for this mini batch
                    generated = generator(x_batch_train, training=True)

                    # Compute the loss value for this mini batch.
                    loss_value = self.compute_generator_loss(generated, discriminator,  y_batch_train)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, generator.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                self.generator_optimizer.apply_gradients(
                    zip(grads, generator.trainable_weights))

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * 64))
                pass
