import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, UpSampling2D, BatchNormalization, \
    Activation, Conv2DTranspose


class UNet:

    LAMBDA = 100

    @staticmethod
    def one_cnn_layer(input, num_filters, kernel_size, padding, batch_norm=True):
        layer = Conv2D(num_filters, kernel_size=kernel_size, padding=padding)(input)
        if batch_norm:
            layer = BatchNormalization()(layer)
        layer = Activation("relu")(layer)
        return layer

    @staticmethod
    def get_model(pretrained_weights=None):

        input_layer = Input(shape=(50, 50, 2))

        """ Down-sampling """

        conv1 = UNet.one_cnn_layer(input_layer, 64, 3, "VALID")
        conv1 = UNet.one_cnn_layer(conv1, 64, 3, "SAME")
        conv1 = UNet.one_cnn_layer(conv1, 64, 3, "SAME")
        pool1 = MaxPooling2D(pool_size=2)(conv1)                            # 24 x 24

        conv2 = UNet.one_cnn_layer(pool1, 128, 3, "SAME")
        conv2 = UNet.one_cnn_layer(conv2, 128, 3, "SAME")
        conv2 = UNet.one_cnn_layer(conv2, 128, 3, "SAME")
        pool2 = MaxPooling2D(pool_size=2)(conv2)                            # 12 x 12

        conv3 = UNet.one_cnn_layer(pool2, 256, 3, "SAME")
        conv3 = UNet.one_cnn_layer(conv3, 256, 3, "SAME")
        conv3 = UNet.one_cnn_layer(conv3, 256, 3, "SAME")
        pool3 = MaxPooling2D(pool_size=2)(conv3)                            # 6 x 6

        conv4 = UNet.one_cnn_layer(pool3, 512, 3, "SAME")
        conv4 = UNet.one_cnn_layer(conv4, 512, 3, "SAME")
        conv4 = UNet.one_cnn_layer(conv4, 512, 3, "SAME")

        """ Upsampling """
        up5 = (UpSampling2D(size=(2, 2))(conv4))                            # 12 x 12
        merge5 = Concatenate()([conv3, up5])

        conv5 = UNet.one_cnn_layer(merge5, 256, 2, "SAME")
        conv5 = UNet.one_cnn_layer(conv5, 256, 3, "SAME")
        conv5 = UNet.one_cnn_layer(conv5, 256, 3, "SAME")

        up6 = (UpSampling2D(size=(2, 2))(conv5))                            # 24 x 24
        merge6 = Concatenate()([conv2, up6])

        conv6 = UNet.one_cnn_layer(merge6, 128, 2, "SAME")
        conv6 = UNet.one_cnn_layer(conv6, 128, 3, "SAME")
        conv6 = UNet.one_cnn_layer(conv6, 128, 3, "SAME")

        up7 = (UpSampling2D(size=(2, 2))(conv6))                            # 48 x 48
        merge7 = Concatenate()([conv1, up7])

        conv7 = UNet.one_cnn_layer(merge7, 64, 2, "SAME")
        conv7 = UNet.one_cnn_layer(conv7, 64, 3, "SAME")
        conv7 = UNet.one_cnn_layer(conv7, 64, 3, "SAME")

        conv8 = Conv2DTranspose(1, kernel_size=3, padding="VALID")(conv7)  # 50 x 50
        merge9 = Concatenate()([input_layer, conv8])

        """ Final layer """
        conv10 = Conv2D(1, kernel_size=1)(merge9)
        conv10 = Activation("relu")(conv10)

        model = Model(inputs=input_layer, outputs=conv10)
        return model

    @staticmethod
    def generator_loss(disc_generated_output, gen_output, target):
        gan_loss = -tf.reduce_mean(disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target[..., np.newaxis] - gen_output))
        total_gen_loss = gan_loss + UNet.LAMBDA * l1_loss
        return total_gen_loss, gan_loss, l1_loss


if __name__ == '__main__':

    model = UNet.get_model()
    print(model.summary())
