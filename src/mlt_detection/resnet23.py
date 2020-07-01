"""
Implementation of ResNet-23 to generate a single cell probability map as in:
Rempfler, Markus, et al. "Cell lineage tracing in lens-free microscopy videos." 2017
"""

import logging
from keras.losses import binary_crossentropy
from keras.optimizers import adam
from keras.layers import Input, Conv2D, BatchNormalization,\
    Activation, Add, ZeroPadding2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.initializers import glorot_uniform
from keras.engine import Model


def identity_block(X, k, filters, stage, block, final_activation=True):
    """
    Implementation of the ResNet identity block

    :param X: (tensor) input tensor with shape (batch, H, W, C)
    :param k: (int) kernel size of conv layer in main path
    :param filters: (list) number of filters the three convolutional_block components
    :param stage: (string) name of convolutional_block stage
    :param block: (string) name of convolutional_block block
    :param final_activation: (bool) Use ReLu activation in the end if True
    :return: (tensor) shape (H, W, C)
    """

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    f1, f2, f3 = filters

    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=conv_name_base + "2a",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    # Second component of main path
    X = Conv2D(filters=f2, kernel_size=k, strides=(1, 1), padding="same", name=conv_name_base + "2b",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    # Third component of main path
    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=conv_name_base + "2c",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    # Final step: Add shortcut value to main path
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X) if final_activation else X

    return X


def convolutional_block(X, k, filters, stage, block, s=2):
    """
    Implementation of the ResNet convolutional block

    :param X: (tensor) input tensor with shape (batch, H, W, C)
    :param k: (int) kernel size of conv layer in main path
    :param filters: (list) number of filters the three convolutional_block components
    :param stage: (string) name of convolutional_block stage
    :param block: (string) name of convolutional_block block
    :param s: (int) stride
    :return: (tensor) shape (H, W, C)
    """

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    f1, f2, f3 = filters

    X_shortcut = X

    # MAIN PATH
    # First component of main path
    X = Conv2D(f1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=f2, kernel_size=k, strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # SHORTCUT PATH
    X_shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def build_model(input_shape=(224, 224, 3), classes=1, training=True):
    """
    Builds the ResNet23 graph

    :param input_shape: (tuple) image shape as (W, H, C)
    :param classes: (int) number of output probability maps
    :param training: (bool) run in training mode
    :return: (model) ResNet23 graph
    """

    inputs = Input(input_shape)
    x = ZeroPadding2D((3, 3))(inputs)

    # Stage 1
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = convolutional_block(x, k=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    x = convolutional_block(x, k=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', final_activation=False)

    # Custom layers
    x = Conv2D(512, (1, 1))(x)
    x = Dropout(rate=0.5)(x) if training else x
    outputs = Conv2DTranspose(classes, (8, 8), strides=(8, 8), activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ResNet23")
    return model


class ResNet23:
    """Main functionalitys of ResNet23"""

    def __init__(self, image_shape=(224, 224, 3), classes=1, training=True):
        self.training = training
        self.classes = classes
        self.image_shape = image_shape
        self.model = build_model(image_shape, classes, training)
        self.compiled = False
        self.weights_path = None

    def load_weights(self, weights_path):
        self.weights_path = weights_path
        self.model.load_weights(str(weights_path))

    def compile(self, optimizer=adam(learning_rate=0.02), loss=binary_crossentropy, **kwargs):
        self.model.compile(optimizer=optimizer, loss=loss, **kwargs)
        self.compiled = True

    def train(self, train, val, **kwargs):
        if not self.compiled:
            logging.warning("Model not compiled. Autocompiling!")
            self.compile()
        return self.model.fit_generator(generator=train,
                                        validation_data=val,
                                        **kwargs)

    def predict(self, img, **kwargs):
        out = self.model.predict(img, **kwargs)
        return out[0, ..., 0]
