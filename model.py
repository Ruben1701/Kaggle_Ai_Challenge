import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_hub as hub
from collections import deque
import random
import math
from tensorflow.keras import backend as K

"""
we add two lines:

e= tf.keras.backend.max(y_true,axis = -1)
y_pred*= K.stack([e]*8, axis=-1)
    
to make the positions which doesn't contain neither unit or city by zero in the prediction probabilities, in order to focus only on the main occupied positions.
"""

def custom_mean_squared_error(y_true, y_pred):
    y_units_true = y_true[:,:,:,:6]
    y_cities_true = y_true[:,:,:,6:]

    y_units_pred = y_pred[:,:,:,:6]
    y_cities_pred = y_pred[:,:,:,6:]


    is_unit = tf.keras.backend.max(y_units_true,axis = -1)
    is_city = tf.keras.backend.max(y_cities_true,axis = -1)

    y_units_pred*= K.stack([is_unit]*6, axis=-1)
    y_cities_pred*= K.stack([is_city]*2, axis=-1)

    loss1 = K.square(y_units_pred - y_units_true)#/K.sum(is_unit)
    loss2 = K.square(y_cities_pred - y_cities_true)#/K.sum(is_city)
    return K.concatenate([loss1,loss2])

def units_accuracy(y_true, y_pred):
    y_units_true = y_true[:,:,:,:6]
    y_cities_true = y_true[:,:,:,6:]

    y_units_pred = y_pred[:,:,:,:6]
    y_cities_pred = y_pred[:,:,:,6:]

    is_unit = tf.keras.backend.max(y_units_true,axis = -1)
    y_units_pred*= K.stack([is_unit]*6, axis=-1)
    return K.cast(K.equal(y_units_true, K.round(y_units_pred)), "float32")/K.sum(is_unit)

def cities_accuracy(y_true, y_pred):
    y_units_true = y_true[:,:,:,:6]
    y_cities_true = y_true[:,:,:,6:]

    y_units_pred = y_pred[:,:,:,:6]
    y_cities_pred = y_pred[:,:,:,6:]

    is_city = tf.keras.backend.max(y_cities_true,axis = -1)
    y_cities_pred*= K.stack([is_city]*2, axis=-1)

    return K.cast(K.equal(y_cities_true, K.round(y_cities_pred)), "float32")/K.sum(is_city)


def get_model(s):
    inputs = keras.Input(shape=(s,s,17),name = 'The game map')
    f = layers.Flatten()(inputs)
    h,w= s,s
    f = layers.Dense(w*h,activation = "sigmoid")(f)
    f = layers.Reshape((h,w,-1))(f)
    units = layers.Dense(6,activation = "softmax",name = "Units_actions")(f)
    cities = layers.Dense(2,activation = "sigmoid",name = "Cities_actions")(f)
    output = layers.Concatenate()([units,cities])
    model = keras.Model(inputs = inputs, outputs = output)
    model.compile(optimizer= "adam", loss= custom_mean_squared_error ,metrics = ["accuracy"])

    return model


model =get_model(12)
model.summary()

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=1,
    show_dtype=1,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96)