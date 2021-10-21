import time
import os
import datetime
import numpy as np
from scipy.io import loadmat
import tensorflow as tf

from config import Config
from data.data import Data
from utils.utils import PlotUtils
from model.generator import UNet
from model.discriminator import PatchGAN


def get_data():

    input_path = os.path.join(Config.data_path, "python_guess")
    output_path = os.path.join(Config.data_path, "scatterer_data_inverse")

    train_inputn, train_outputn, test_inputn, test_outputn = Data.get_data(input_path, output_path)

    train_inputf = np.flip(train_inputn, 2)
    train_outputf = np.flip(train_outputn, 2)
    test_inputf = np.flip(test_inputn, 2)
    test_outputf = np.flip(test_outputn, 2)

    train_input = np.concatenate((train_inputn, train_inputf), axis=0)
    train_output = np.concatenate((train_outputn, train_outputf), axis=0)
    test_input = np.concatenate((test_inputn, test_inputf), axis=0)
    test_output = np.concatenate((test_outputn, test_outputf), axis=0)

    # Shuffle data
    np.random.seed(1234)

    randomize = np.arange(len(train_input))
    np.random.shuffle(randomize)
    train_input = train_input[randomize]
    train_output = train_output[randomize]

    randomize = np.arange(len(test_input))
    np.random.shuffle(randomize)
    test_input = test_input[randomize]
    test_output = test_output[randomize]

    return train_input, train_output, test_input, test_output


if __name__ == '__main__':

    """" Global parameters """

    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 100

    images_to_save = [0, 7, 10, 20, 50, 100, 150]

    """" GPU config """
    # tf.random.set_seed(1234)
    # tf.keras.backend.clear_session()
    # from tensorflow.compat.v1 import ConfigProto
    # from tensorflow.compat.v1 import InteractiveSession
    # config = ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # session = InteractiveSession(config=config)

    model_path = Config.model_path

    """" Models """
    generator = UNet.get_model()
    discriminator = PatchGAN.get_model()

    """" Optimizers """
    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

    """" Checkpoint """
    checkpoint_dir = os.path.join(model_path, "training_checkpoints", f"{Config.EXPERIMENT_NAME}")
    checkpoint_prefix = os.path.join(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    """" Tensorboard """
    log_dir = os.path.join(model_path, "logs/")
    summary_writer = tf.summary.create_file_writer(log_dir + f"{Config.EXPERIMENT_NAME}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    """" Data """
    train_input, train_output, test_input, test_output = get_data()
    train_output = np.real(train_output)
    test_output = np.real(test_output)
    PlotUtils.check_data(train_input, train_output)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_output))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_output))

    TRAIN_SIZE = train_input.shape[0]
    TEST_SIZE = test_input.shape[0]
    epoch_steps = int(TRAIN_SIZE / BATCH_SIZE)
    print("Steps in one epoch: ", epoch_steps)

    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)


    def gradient_penalty(input_images, real_images, fake_images):
        alpha = tf.random.normal([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = discriminator([input_images, interpolated], training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    """" Train discriminator using one batch of data """
    @tf.function()
    def disc_train_step(input_batch, target_batch, step):
        with tf.GradientTape() as disc_tape:
            gen_output = generator(input_batch, training=True)

            disc_real_output = discriminator([input_batch, target_batch], training=True)
            disc_generated_output = discriminator([input_batch, gen_output], training=True)
            gp = gradient_penalty(input_batch, target_batch[..., np.newaxis], gen_output)

            disc_loss = PatchGAN.discriminator_loss(disc_real_output, disc_generated_output)
            disc_loss = disc_loss + 10 * gp

        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        tf.print(discriminator_gradients[-1])
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        if step % epoch_steps == 0:
            tf.print("Discriminator loss:  ", disc_loss)

        with summary_writer.as_default():
            tf.summary.scalar('disc_loss', disc_loss, step=step//epoch_steps)

    """" Train generator using one batch of data """
    @tf.function
    def gen_train_step(input_batch, target_batch, step):
        with tf.GradientTape() as gen_tape:
            gen_output = generator(input_batch, training=True)

            disc_generated_output = discriminator([input_batch, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = UNet.generator_loss(disc_generated_output, gen_output, target_batch)

        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

        if step % epoch_steps == 0:
            tf.print("Generator GAN loss:  ", gen_gan_loss)
            tf.print("Generator L1 loss:  ", gen_l1_loss)

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//epoch_steps)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//epoch_steps)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//epoch_steps)

    """" Train both generator and discriminator together"""
    @tf.function
    def train_step(input_batch, target_batch, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_batch, training=True)

            disc_real_output = discriminator([input_batch, target_batch], training=True)
            disc_generated_output = discriminator([input_batch, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = UNet.generator_loss(disc_generated_output, gen_output, target_batch)
            disc_loss = PatchGAN.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        if step % epoch_steps == 0:
            tf.print("Generator GAN loss:  ", gen_gan_loss)
            tf.print("Generator L1 loss:  ", gen_l1_loss)
            tf.print("Discriminator loss:  ", disc_loss)

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//epoch_steps)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//epoch_steps)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//epoch_steps)
            tf.summary.scalar('disc_loss', disc_loss, step=step//epoch_steps)


    """" Test generator on test dataset """
    def check_on_test_images(generator, test_ds, epoch):
        y_pred = None
        targets = None
        test_inputs = None
        for (test_input, target) in test_ds.take(20):
            prediction = generator(test_input, training=False)
            if y_pred is None:
                test_inputs = test_input
                y_pred = prediction
                targets = target
            else:
                test_inputs = np.concatenate((test_inputs, test_input), axis=0)
                y_pred = np.concatenate((y_pred, prediction), axis=0)
                targets = np.concatenate((targets, target), axis=0)

        gen_test_loss = np.mean(np.abs(targets[..., np.newaxis] - y_pred))
        with summary_writer.as_default():
            tf.summary.scalar('gen_test_loss', gen_test_loss, step=epoch)

        for j in images_to_save:
            display_list = [targets[j, :, :], test_inputs[j, :, :, 0], test_inputs[j, :, :, 1], y_pred[j, :, :, :]]
            plot_buf = PlotUtils.plot_results(display_list[0], display_list[1], display_list[2], display_list[3], mode="save")
            image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            with summary_writer.as_default():
                tf.summary.image(f'test_{j}', image, step=epoch)
            del image

        return gen_test_loss

    """" Test generator on out of sample dataset """
    def check_on_oos_images(generator, epoch):
        main_dir = Config.oos_data_path
        test_dirs = os.listdir(main_dir)
        for index, test_dir in enumerate(test_dirs):
            dir_path = os.path.join(main_dir, test_dir)
            real_rec = loadmat(os.path.join(dir_path, "real_rec"))["real_rec"]
            imag_rec = loadmat(os.path.join(dir_path, "imag_rec"))["imag_rec"]
            scatterer = loadmat(os.path.join(dir_path, "scatterer"))["scatterer"]
            scatterer = np.real(scatterer)

            test_input = np.asarray([real_rec, imag_rec])
            test_input = np.moveaxis(test_input, 0, -1)
            test_input = test_input[np.newaxis, ...]

            """ Model Test """
            y_pred = generator(test_input, training=True)

            plot_buf = PlotUtils.plot_results(scatterer, real_rec, imag_rec, y_pred[0, :, :], mode="save")
            image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            with summary_writer.as_default():
                tf.summary.image(f'oos_{index}', image, step=epoch)
            del image

    """" Train """
    def fit(train_ds, test_ds, steps):

        start = time.time()

        gen_loss = 1000
        gen_test_loss = 1000

        for step, (input_batch, target_batch) in train_ds.repeat().take(steps).enumerate():

            if (step) % epoch_steps == 0:
                if step != 0:
                    print(f'Time taken for {epoch_steps} steps/ one epoch: {time.time() - start:.2f} sec\n')
                gen_test_loss = check_on_test_images(generator, test_ds, step//epoch_steps)
                check_on_oos_images(generator, step//epoch_steps)
                print(f"Epoch : {step // epoch_steps}")
                start = time.time()

            train_step(np.float32(input_batch), np.float32(target_batch), step)

            if (step + 1) % 10 == 0:
                print('.', end='', flush=True)
            if gen_test_loss < gen_loss:
                if (step + 1) % epoch_steps == 0:
                    gen_loss = gen_test_loss
                    checkpoint.save(file_prefix=f"{checkpoint_prefix}-{step//epoch_steps}-{gen_test_loss:.4f}")


    total_steps = Config.EPOCHS*epoch_steps
    print("Steps: ", total_steps)
    fit(train_dataset, test_dataset, steps=total_steps)
