#!/usr/bin/python3
''' '''


from tensorflow.python.keras.models import load_model

import random

import numpy as np

import cv2

from pathlib import Path

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('model')

args = parser.parse_args()

import yaml
import munch

with open('./configuration.yaml') as file:
    cg = munch.munchify(yaml.load(file, Loader=yaml.Loader))

from tensorflow import ConfigProto, Session

config = ConfigProto()

config.gpu_options.allow_growth = True

session = Session(config=config)


def inference(model, image):

    image = image.copy()

    image = cv2.resize(image, tuple(cg.hyperparameters.input.dimensions))

    image = image.reshape(*cg.hyperparameters.input.dimensions, 1)

    image = image.astype(np.float32)

    image -= cg.hyperparameters.standardization.mu
    image /= cg.hyperparameters.standardization.sigma

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    prediction = float(np.squeeze(prediction, axis=0))

    return False if prediction > 0.5 else True


def main():

    model = load_model(args.model)

    paths = [str(path) for path in Path('./images/').glob('*.png')]
    random.shuffle(paths)

    # reader = cv2.VideoCapture(0)

    for path in paths:
    # while reader.isOpened():

        image = cv2.imread(path, 0)

        # proceed, image = reader.read()

        # if not proceed:
            # break

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        prediction = inference(model, image)

        print(prediction)

        cv2.imshow('', image)

        key = cv2.waitKey(1)

        if key == 27:
            break

        base_directory = './images/squatting/' if prediction else './images/standing/'

        filename = path.split('/')[-1]

        Path(path).replace(base_directory + filename)


if __name__ == '__main__':
    main()

