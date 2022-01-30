import sys

import tensorflow as tf
from tensorflow import keras

import data_providers
import models
import trainers


def train(frames_path, stylized_frames_path, aux_frames_path):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    regularizer = tf.keras.regularizers.l2(0.00001)
    decay_attributes = ['kernel_regularizer', 'bias_regularizer',
                        'beta_regularizer', 'gamma_regularizer']

    generator = models.make_generator()
    discriminator = models.make_discriminator()
    for layer in generator.layers:
        for attr in decay_attributes:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)
    for layer in discriminator.layers:
        for attr in decay_attributes:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)
    perception_loss_model = models.make_perception_loss_model([0, 3, 5])
    generator_optimizer = keras.optimizers.Adam(learning_rate=0.0004)
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0004)
    train_dataset = data_providers.PatchedDataProvider(frames_path, stylized_frames_path, aux_frames_path, 32)
    data_provider = data_providers.BatchProvider(train_dataset, batch_size=40)

    trainer = trainers.Trainer(generator_optimizer, discriminator_optimizer, data_provider,
                               perception_loss_model)

    trainer.train(generator, discriminator, 1)


if __name__ == '__main__':
    train(sys.argv[1], sys.argv[2], sys.argv[3])
