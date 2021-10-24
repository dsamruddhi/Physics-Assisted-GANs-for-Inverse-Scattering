import os
import random
import numpy as np
import tensorflow as tf

from config import Config
from data.data import Data
from utils.utils import PlotUtils
from model.generator import UNet
from model.discriminator import PatchGAN


def get_data():

    input_path = os.path.join(Config.data_path, "python_guess")
    output_path = os.path.join(Config.data_path, "scatterer_data_inverse")

    train_input, train_output, test_input, test_output = Data.get_data(input_path, output_path)

    return train_input, train_output, test_input, test_output


def check_results(test_input, y_pred, test_output):

    for i in random.sample(range(0, test_input.shape[0]), 10):
        print(i)
        PlotUtils.plot_results(test_output[i, :, :], test_input[i, :, :, 0], test_input[i, :, :, 1], y_pred[i, :, :])


if __name__ == '__main__':

    """" GPU config """
    tf.keras.backend.clear_session()

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = InteractiveSession(config=config)

    """ Load Model / Checkpoint """
    checkpoint_dir = os.path.join(Config.model_path, "training_checkpoints", f"{Config.EXPERIMENT_NAME}")
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    generator = UNet.get_model()
    discriminator = PatchGAN.get_model()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    checkpoint.restore(latest)

    """ Generate Training and Test Data """
    train_input, train_output, test_input, test_output = get_data()

    """ Model Test """
    y_pred = checkpoint.generator(test_input, training=True)
    y_pred = np.asarray(y_pred)

    """ Plot results """
    check_results(test_input, y_pred, test_output)
