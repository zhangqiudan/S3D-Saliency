import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import load_model
from dataset_s3d import *
from S3DModel import *
import math
import pickle
import os

SCALE_LIST = [1, 3, 5]
IMG_SIZE = 224
NUM_CHANNEL=3
batch_size = 1

def create_model(input_shape, scale_list,learn_rate, momentum, decay, loss='crossentropy'):

    model = S3DModel().model
    loss_fn = keras.losses.binary_crossentropy
    sgd = keras.optimizers.SGD(lr=learn_rate, momentum=momentum, decay=decay, nesterov=True)
    model.compile(loss=loss_fn, optimizer=sgd, metrics=['accuracy', 'mae'])
    return model


def train(learn_rate=0.01, loss='crossentropy', num_epochs=100):

    X_train, Y_train, X_val, Y_val = get_3D_data()
    checkpoint_path = "./models_checkpoint/s3d-{epoch:04d}.h5"

    model = create_model(input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNEL), scale_list=SCALE_LIST, learn_rate=learn_rate, momentum=0.9, decay=0.0005, loss=loss)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_weights_only=False,save_freq=1*batch_size)

    history = model.fit(X_train, Y_train,
        batch_size=1,
        epochs=num_epochs,
        validation_data=(X_val, Y_val),
        shuffle=True,
        callbacks=[checkpoint],verbose=1)

    model.save(checkpoint_path.format(epoch=0))
    return model



if __name__ == "__main__":
    model = train()
