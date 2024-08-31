"""
Network definitions
"""

import numpy as np
np.random.seed(0)
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense, Reshape

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def base_network(input_size, descriptor_size, trainable=False):
    '''
    Define the base network model
    :param input_size: shape of input images
    :param descriptor_size: embedding size used to encode input images
    :return: network model
    '''
    base_model = ResNet50(input_shape=input_size, weights='imagenet', include_top=False)
    # base_model.trainable = trainable

    x = GlobalMaxPooling2D(name='global_max_1')(base_model.get_layer('conv5_block3_out').output)
    # x = GlobalMaxPooling2D(name='global_max_1')(base_model.get_layer('block4_pool').output)
    x = Dense(descriptor_size * 4, kernel_regularizer=l2(1e-3), activation='relu', kernel_initializer='he_uniform', name='dense_descriptor_1')(x)
    x = Dense(descriptor_size * 2, kernel_regularizer=l2(1e-3), activation='relu', kernel_initializer='he_uniform', name='dense_descriptor_2')(x)
    descriptor = Dense(descriptor_size, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform', name='dense_descriptor_3')(x)
    norm_descriptor = Lambda(lambda x: K.l2_normalize(x, axis=-1))(descriptor)
    network = Model(inputs=[base_model.input], outputs=[norm_descriptor])
    # return norm_descriptor
    # network.summary()

    for layer in base_model.layers:
        layer.trainable = trainable

    return network


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square( anchor -positive), axis=-1)
        n_dist = K.sum(K.square( anchor -negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

class TripletLossLayerKL(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayerKL, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = keras.losses.kl_divergence(anchor, positive)
        n_dist = keras.losses.kl_divergence(anchor, negative)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

def build_neural_embedding(input_shape, base_network, margin_loss):
    '''
    :param input_shape: shape of input images
    :param network: descriptor/embedding size
    :param margin_loss: margin in triplet loss
    :return: network definition for training
    '''
    # Define input tensor
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')

    anchor_merge = Concatenate(axis=-1)([anchor_input, anchor_input, anchor_input])
    pos_merge = Concatenate(axis=-1)([positive_input, positive_input, positive_input])
    neg_merge = Concatenate(axis=-1)([negative_input, negative_input, negative_input])

    net_anchor = base_network(anchor_merge)
    net_positive = base_network(pos_merge)
    net_negative = base_network(neg_merge)

    # TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin_loss, name='triplet_loss_layer')([net_anchor, net_positive, net_negative])

    # Connect the inputs with the outputs
    network_training = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)

    return network_training