import os

import numpy as np
import tensorflow as tf

import data_providers
import models
import trainers
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import sys
import PIL


def test():
    generator = keras.models.load_model("generator")
    paths = sorted(os.listdir(sys.argv[2]))

    images = []
    for p in paths:
        image = PIL.Image.open(os.path.join(sys.argv[2], p))
        aux = PIL.Image.open(os.path.join(sys.argv[3], p))
        aux = keras.preprocessing.image.img_to_array(aux)
        image = keras.preprocessing.image.img_to_array(image)
        aux = aux[:, :, 0:3] / 255.0
        image = image[:, :, 0:3] / 255.0
        image = np.dstack([image, aux])
        images.append(image)
    for i, image in enumerate(images):
        generated = generator(image.reshape((1,) + image.shape))
        keras.utils.save_img(f'generated/{i:03}.png', generated[0], data_format='channels_last')
    pass


def train():
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
    train_dataset = data_providers.PatchedDataProvider(sys.argv[2], sys.argv[3], sys.argv[4], 32)
    data_provider = data_providers.BatchProvider(train_dataset, batch_size=40)

    trainer = trainers.Trainer(generator_optimizer, discriminator_optimizer, data_provider,
                               perception_loss_model)

    trainer.train(generator, discriminator, 1)


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    if sys.argv[1] == 'test':
        test()
