from tensorflow import keras
import numpy as np


class Trainer(object):
    def __init__(self, generator, discriminator, perception_loss_model=None):
        self.discriminator = discriminator
        self.generator = generator

        self.use_image_loss = perception_loss_model is not None

        pass

    def train(self, epochs):
        for e in range(epochs):
            for i, batch in enumerate(self.train_loader):
                # train
                pass
