#!/usr/bin/python3
''' '''

import matplotlib.pyplot as plot

import yaml
import munch

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow import ConfigProto, Session

config = ConfigProto()

config.gpu_options.allow_growth = True

session = Session(config=config)

with open('./configuration.yaml') as file:
    cg = munch.munchify(yaml.load(file, Loader=yaml.Loader))


def construct_model():

    inputs = Input(cg.hyperparameters.input.dimensions + [1, ])

    conv = Conv2D(32, (3, 3), activation='relu')(inputs)
    conv = Conv2D(32, (3, 3), activation='relu')(conv)
    max_pool = MaxPooling2D((2, 2))(conv)

    conv = Conv2D(32, (3, 3), activation='relu')(max_pool)
    conv = Conv2D(32, (3, 3), activation='relu')(conv)
    max_pool = MaxPooling2D((2, 2))(conv)

    conv = Conv2D(64, (3, 3), activation='relu')(max_pool)
    conv = Conv2D(64, (3, 3), activation='relu')(conv)
    max_pool = MaxPooling2D((2, 2))(conv)

    conv = Conv2D(64, (3, 3), activation='relu')(max_pool)
    conv = Conv2D(64, (3, 3), activation='relu')(conv)
    max_pool = MaxPooling2D((2, 2))(conv)

    flatten = Flatten()(max_pool)

    dense = Dense(128, 'relu')(flatten)

    dropout = Dropout(cg.hyperparameters.dropout.rate)(dense)

    outputs = Dense(cg.hyperparameters.output.classes, cg.hyperparameters.output.activation)(dropout)

    model = Model(inputs, outputs)

    kwargs = {
        'optimizer': Adam(cg.hyperparameters.optimizer.learning_rate),
        'loss': binary_crossentropy,
        'metrics': ['accuracy'],
    }

    model.compile(**kwargs)

    return model


def standardization(array):

    array = array.copy()

    array = array.astype(np.float32)

    array -= cg.hyperparameters.standardization.mu
    array /= cg.hyperparameters.standardization.sigma

    return array


def plot_results(history):

    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']


def main():

    kwargs = {
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'brightness_range': [0.9, 1.1],
        'zoom_range': 0.2,
        'fill_mode': 'constant',
        'horizontal_flip': True,
        'preprocessing_function': standardization,
    }

    data_generator = ImageDataGenerator(**kwargs)

    kwargs = {
        'directory': cg.paths.training_directory,
        'target_size': cg.hyperparameters.input.dimensions,
        'color_mode': 'grayscale',
        'class_mode': 'binary',
        'batch_size': cg.hyperparameters.batch_size,
    }

    training_data_generator = data_generator.flow_from_directory(**kwargs)

    kwargs = {
        'preprocessing_function': standardization,
    }

    data_generator = ImageDataGenerator(**kwargs)

    kwargs = {
        'directory': cg.paths.validation_directory,
        'target_size': cg.hyperparameters.input.dimensions,
        'color_mode': 'grayscale',
        'class_mode': 'binary',
        'batch_size': cg.hyperparameters.batch_size,
    }

    validation_data_generator = data_generator.flow_from_directory(**kwargs)

    model = construct_model()

    kwargs = {
        'filepath': './models/model.h5',
        'monitor': 'val_acc',
        'save_best_only': True,
    }

    callbacks = [ModelCheckpoint(**kwargs)]

    kwargs = {
        'generator': training_data_generator,
        'epochs': cg.hyperparameters.epochs,
        'callbacks': callbacks,
        'validation_data': validation_data_generator,
    }

    history = model.fit_generator(**kwargs)

    plot_results(history)


if __name__ == '__main__':
    main()
