import os
import sys   # to exit program
import pygame
from pygame.locals import *

from tensorflow.python.keras.models import load_model

import numpy as np
import cv2

from Player import Player
from Pipe import Pipe

from tensorflow import ConfigProto, Session

config = ConfigProto()

config.gpu_options.allow_growth = True

session = Session(config=config)

import yaml
import munch

with open('./configuration.yaml') as file:
    cg = munch.munchify(yaml.load(file, Loader=yaml.Loader))

# Global Variable for the game
SCREEN = pygame.display.set_mode((cg.settings.screen.width, cg.settings.screen.height))
GAME_SPRITES = {}
GAME_SOUNDS = {}

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join('img', 'bird1.png'))),
             pygame.transform.scale2x(pygame.image.load(os.path.join('img', 'bird2.png'))),
             pygame.transform.scale2x(pygame.image.load(os.path.join('img', 'bird3.png')))]


def inference(model, image):

    image = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (128, 128))

    image = image.reshape(128, 128, 1)

    image = image.astype(np.float32)

    image -= cg.model.standardization.mu
    image /= cg.model.standardization.sigma

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    prediction = float(np.squeeze(prediction, axis=0))

    return False if prediction > cg.model.confidence_threshold else True


def welcomeScreen():
     """
     Shows welcome image
     """

     playerx = int(cg.settings.screen.width / 5)                                             # player x position
     playery = int(cg.settings.screen.height - GAME_SPRITES['player'].get_height()) / 2    # player y position
     messagex = int(cg.settings.screen.width - GAME_SPRITES['message'].get_width()) / 2    # message x position
     messagey = int(cg.settings.screen.height - GAME_SPRITES['message'].get_height()) / 2  # message y position
     basex = 0
     while True:
         for event in pygame.event.get():

             # quit if x is pressed or hold esc
             if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                 pygame.quit()
                 sys.exit()

             # start if start screen is clicked
             elif event.type == pygame.MOUSEBUTTONUP:
                 return
             else:
                 SCREEN.blit(GAME_SPRITES['background'], (0, 0))
                 SCREEN.blit(GAME_SPRITES['player'], (playerx, playery))
                 SCREEN.blit(GAME_SPRITES['message'], (messagex, messagey))
                 SCREEN.blit(GAME_SPRITES['base'], (basex, cg.ground.y))
                 pygame.display.update()
                 FPSCLOCK.tick(cg.settings.fps)

def mainGame():
    score = 0

    IMGS = BIRD_IMGS
    image = IMGS[0]
    ANIMATION_TIME = 5
    img_count = 0

    player = Player(int(cg.settings.screen.width / 5), int(cg.settings.screen.height / 2), -9, 10, -8, 0.3, BIRD_IMGS, -8, False)
    basex = 0

    # create 2 random pipes
    newPipe1 = Pipe(-4, GAME_SPRITES, cg.settings.screen.height, cg.settings.screen.width)
    newPipe1r = newPipe1.getRandomPipe()
    newPipe2 = Pipe(-4, GAME_SPRITES, cg.settings.screen.height, cg.settings.screen.width)
    newPipe2r = newPipe2.getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': cg.settings.screen.width + 200, 'y': newPipe1r[0]['y']},
        {'x': cg.settings.screen.width + 200 + 650, 'y': newPipe2r[0]['y']},
    ]
    # list of lower pipes
    lowerPipes = [
        {'x': cg.settings.screen.width + 200, 'y': newPipe1r[1]['y']},
        {'x': cg.settings.screen.width + 200 + 650, 'y': newPipe2r[1]['y']},
    ]

    model = load_model(cg.paths.model)

    webcam = cv2.VideoCapture(0)

    while webcam.isOpened():

        proceed, frame = webcam.read()

        if not proceed:
            break

        img_count += 1

        if player.velocityY<0:
            if img_count < ANIMATION_TIME:
                image = IMGS[0]
            elif img_count < ANIMATION_TIME*2:
                image = IMGS[1]
            elif img_count < ANIMATION_TIME*3:
                image = IMGS[2]
            elif img_count < ANIMATION_TIME*4:
                image = IMGS[1]
            elif img_count < ANIMATION_TIME*4 + 1:
                image = IMGS[0]
                img_count = 0
        else:
            image = IMGS[0]
            img_count = 0

        if inference(model, frame):
            if player.positionY > 0:
                player.velocityY = player.flappedAccVel
                player.flapped = True
                image = IMGS[2]
                GAME_SOUNDS['wing'].play()

        if cg.webcam.display:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128))
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, flipCode=0)
            frame = pygame.surfarray.make_surface(frame)

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        if isCollide(player.positionX, player.positionY, upperPipes, lowerPipes):
            return

        # score check
        playerMidPos = player.positionX + GAME_SPRITES['player'].get_width()/2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + GAME_SPRITES['pipe'][0].get_width()/2
            if pipeMidPos <= playerMidPos < pipeMidPos +4:
                score += 1
                GAME_SOUNDS['point'].play()

        if player.velocityY < player.maxVelocityY and not player.flapped:
            player.velocityY += player.accY

        if player.flapped:
            player.flapped = False
        playerHeight = GAME_SPRITES['player'].get_height()
        player.positionY = player.positionY + min(player.velocityY, cg.ground.y - player.positionY - playerHeight)

        # move pipes to left
        for upperPipe, lowerPipe in zip(upperPipes, lowerPipes):
            upperPipe['x'] += newPipe1.velocityX
            lowerPipe['x'] += newPipe1.velocityX

        # add new pipe when the first pipe about to cross leftmost part of screen
        if 0 < upperPipes[0]['x'] < 5:
            newpipe = Pipe(-4, GAME_SPRITES, cg.settings.screen.height, cg.settings.screen.width).getRandomPipe()
            upperPipes.append(newpipe[0])
            lowerPipes.append(newpipe[1])

        # remove pipe out of screen
        if upperPipes[0]['x'] < -GAME_SPRITES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # blit sprites
        SCREEN.blit(GAME_SPRITES['background'], (0, 0))
        for upperPipe, lowerPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(GAME_SPRITES['pipe'][0], (upperPipe['x'], upperPipe['y']))
            SCREEN.blit(GAME_SPRITES['pipe'][1], (lowerPipe['x'], lowerPipe['y']))

        SCREEN.blit(GAME_SPRITES['base'], (basex, cg.ground.y))

        SCREEN.blit(image, (player.positionX, player.positionY))

        if cg.webcam.display:
            SCREEN.blit(frame, (cg.settings.screen.width - 136, 8))

        myDigits = [int(x) for x in list(str(score))]

        width = 0

        for digit in myDigits:
            width += GAME_SPRITES['numbers'][digit].get_width()

        Xoffset = (cg.settings.screen.width - width) / 2

        for digit in myDigits:
            SCREEN.blit(GAME_SPRITES['numbers'][digit], (Xoffset, cg.settings.screen.height * 0.12))
            Xoffset += GAME_SPRITES['numbers'][digit].get_width()

        pygame.display.update()
        FPSCLOCK.tick(cg.settings.fps)

    webcam.release()
    cv2.destroyAllWindows()


def isCollide(playerx, playery, upperPipes, lowerPipes):
    if playery > cg.ground.y - GAME_SPRITES['player'].get_height() - 1 or playery < 0:
        GAME_SOUNDS['hit'].play()
        return True

    for pipe in upperPipes:
        pipeHeight = GAME_SPRITES['pipe'][0].get_height()
        if (playery < pipeHeight + pipe['y'] and abs(playerx - pipe['x']) < GAME_SPRITES['player'].get_width() - 50):
            GAME_SOUNDS['hit'].play()
            return True

    for pipe in lowerPipes:
        if (playery + GAME_SPRITES['player'].get_height() > pipe['y']+40) and abs(playerx - pipe['x']) < GAME_SPRITES['player'].get_width()-50:
            GAME_SOUNDS['hit'].play()
            return True
    return False


def initialize_sprites():

    GAME_SPRITES['numbers'] = [pygame.image.load(f'gallery/sprites/{_}.png').convert_alpha() for _ in range(10)]

    GAME_SPRITES['message'] = pygame.image.load(cg.paths.sprites.message).convert_alpha()
    GAME_SPRITES['base'] = pygame.image.load(cg.paths.sprites.base).convert_alpha()

    GAME_SPRITES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(cg.paths.sprites.pipe).convert_alpha(), 180),
            pygame.image.load(cg.paths.sprites.pipe).convert_alpha())

    GAME_SPRITES['background'] = pygame.image.load(cg.paths.sprites.background).convert()
    GAME_SPRITES['player'] = pygame.image.load(cg.paths.sprites.player).convert_alpha()


def initialize_sound_effects():

    GAME_SOUNDS['die'] = pygame.mixer.Sound(cg.paths.audio.die)
    GAME_SOUNDS['hit'] = pygame.mixer.Sound(cg.paths.audio.hit)
    GAME_SOUNDS['point'] = pygame.mixer.Sound(cg.paths.audio.point)
    GAME_SOUNDS['swoosh'] = pygame.mixer.Sound(cg.paths.audio.swoosh)
    GAME_SOUNDS['wing'] = pygame.mixer.Sound(cg.paths.audio.wing)


if __name__ == '__main__':

    pygame.init()

    FPSCLOCK = pygame.time.Clock()

    pygame.display.set_caption('Squatty Birds')

    initialize_sprites()
    initialize_sound_effects()

    while True:
        welcomeScreen()
        mainGame()
