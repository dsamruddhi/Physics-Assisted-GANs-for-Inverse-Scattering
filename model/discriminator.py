import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization


class PatchGAN:

    @staticmethod
    def one_cnn_layer(input, num_filters, kernel_size, strides, padding, batch_norm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        layer = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=initializer)(input)
        if batch_norm:
            layer = BatchNormalization()(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        return layer

    @staticmethod
    def get_model():
        initializer = tf.random_normal_initializer(0., 0.02)
        input_layer = tf.keras.layers.Input(shape=[50, 50, 2], name='input_image')
        target_layer = tf.keras.layers.Input(shape=[50, 50, 1], name='target_image')
        x = tf.keras.layers.concatenate([input_layer, target_layer])

        conv1 = PatchGAN.one_cnn_layer(x, 64, 3, 2, "SAME", batch_norm=False)
        conv2 = PatchGAN.one_cnn_layer(conv1, 128, 3, 2, "SAME", batch_norm=False)
        conv3 = PatchGAN.one_cnn_layer(conv2, 256, 3, 2, "SAME", batch_norm=False)
        output_layer = tf.keras.layers.Conv2D(1, 3, strides=2, padding="SAME", kernel_initializer=initializer)(conv3)

        return tf.keras.Model(inputs=[input_layer, target_layer], outputs=output_layer)

    @staticmethod
    def discriminator_loss(disc_real_output, disc_generated_output):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    @staticmethod
    def wasserstein_discriminator_loss(disc_real_output, disc_generated_output):
        total_disc_loss = tf.reduce_mean(disc_real_output - disc_generated_output)
        return total_disc_loss


if __name__ == '__main__':

    model = PatchGAN.get_model()
    print(model.summary())
