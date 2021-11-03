import models
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

def main():
    generator = models.make_generator()
    plot_model(generator, "model.png", show_shapes=True, show_layer_names=True)
    plt.show()


if __name__ == '__main__':
    main()
