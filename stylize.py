import os

import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras


def test(frames_path, aux_frames_path, output_directory, model_path="generator"):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    generator = keras.models.load_model(model_path)
    paths = sorted(os.listdir(frames_path))

    images = []
    for p in paths:
        image = PIL.Image.open(os.path.join(frames_path, p))
        aux = PIL.Image.open(os.path.join(aux_frames_path, p))
        aux = keras.preprocessing.image.img_to_array(aux)
        image = keras.preprocessing.image.img_to_array(image)
        aux = (aux[:, :, 0:3] / 255.0 - 0.5) * 2
        image = (image[:, :, 0:3] / 255.0 - 0.5) * 2
        image = np.dstack([image, aux])
        images.append(image)
    for i, image in enumerate(images):
        generated = generator(image.reshape((1,) + image.shape))
        keras.utils.save_img(f'{output_directory}/{i:03}.png', generated[0].numpy() / 2 + 0.5, data_format='channels_last')
    pass