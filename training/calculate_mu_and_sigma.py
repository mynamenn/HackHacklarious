#!/usr/bin/python3
''' '''

import sys
import yaml
import munch

import numpy as np

import cv2

from pathlib import Path


def main():

    with open('./configuration.yaml') as file:
        cg = munch.munchify(yaml.load(file, Loader=yaml.Loader))

    mu, sigma = 0, 0

    paths = (str(path) for path in Path('./data/training/').rglob('*.png'))

    files = 0
    for path in paths:

        image = cv2.imread(path, 0)

        mu += float(image.mean())
        sigma += float(image.std())

        files += 1

    cg.hyperparameters.standardization.mu = mu / files
    cg.hyperparameters.standardization.sigma = sigma / files

    with open('./configuration.yaml', 'w') as file:
        yaml.safe_dump(cg, file, default_flow_style=False)


if __name__ == '__main__':
    main()
