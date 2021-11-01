from tensorflow import keras
import numpy as np
import tensorflow as tf


class Trainer(object):
    def __init__(self, generator_optimizer, discriminator_optimizer, perception_loss_model=None):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.use_image_loss = perception_loss_model is not None

        pass

    def train(self, generator, discriminator, epochs):
        for e in range(epochs):
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    # Logits for this minibatch
                    logits = generator(x_batch_train, training=True)

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(y_batch_train, logits)

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
