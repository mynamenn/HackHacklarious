import os
import random
import sys   # to exit program
import pygame
from pygame.locals import *
import pyautogui

from tensorflow.python.keras.models import load_model

import numpy as np
import cv2

from tensorflow import ConfigProto, Session

config = ConfigProto()

config.gpu_options.allow_growth = True

session = Session(config=config)

# Global Variable for the game
FPS = 32
SCREENWIDTH = 1280
SCREENHEIGHT = 720
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
GROUNDY = 720   # Height of ground
GAME_SPRITES = {}
GAME_SOUNDS = {}
PLAYER = 'gallery/sprites/bird.png'
BACKGROUND = 'gallery/sprites/background.png'
PIPE = 'gallery/sprites/pipe.png'
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join('img', 'bird1.png'))),
             pygame.transform.scale2x(pygame.image.load(os.path.join('img', 'bird2.png'))),
             pygame.transform.scale2x(pygame.image.load(os.path.join('img', 'bird3.png')))]


def inference(model, image):

    image = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (128, 128))

    image = image.reshape(128, 128, 1)

    image = image.astype(np.float32)

    image -= 91.14755214376875
    image /= 63.70613171447146

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    prediction = float(np.squeeze(prediction, axis=0))

    return False if prediction > 0.5 else True


def welcomeScreen():
     """
     Shows welcome image
     """

     playerx = int(SCREENWIDTH/5)                                             # player x position
     playery = int(SCREENHEIGHT - GAME_SPRITES['player'].get_height()) / 2    # player y position
     messagex = int(SCREENWIDTH - GAME_SPRITES['message'].get_width()) / 2    # message x position
     messagey = int(SCREENHEIGHT - GAME_SPRITES['message'].get_height()) / 2  # message y position
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
                 SCREEN.blit(GAME_SPRITES['player'], (playerx,playery))
                 SCREEN.blit(GAME_SPRITES['message'], (messagex,messagey))
                 SCREEN.blit(GAME_SPRITES['base'], (basex,GROUNDY))
                 pygame.display.update()
                 FPSCLOCK.tick(FPS)

def mainGame():
    score = 0

    IMGS = BIRD_IMGS
    image = IMGS[0]
    ANIMATION_TIME = 5
    img_count = 0
    img = IMGS[0]
    tilt = 0

    playerx = int(SCREENWIDTH/5)
    playery = int(SCREENHEIGHT/2)
    basex = 0

    # create 2 random pipes
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + 650, 'y': newPipe2[0]['y']},
    ]
    # list of lower pipes
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + 650, 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4   # pipe vel

    playerVelY = -9
    playerMaxVelY = 10
    playerMinVelY = -8
    playerAccY = 0.3

    playerFlapAccv = -8 #velocity while flapping
    playerFlapped = False

    model = load_model('./training/models/model.7900.h5')

    webcam = cv2.VideoCapture(0)

    while webcam.isOpened():

        proceed, frame = webcam.read()

        if not proceed:
            break

        img_count += 1

        if playerVelY<0:
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
            if playery > 0:
                playerVelY = playerFlapAccv
                playerFlapped = True
                image = IMGS[2]
                GAME_SOUNDS['wing'].play()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, flipCode=0)
        frame = pygame.surfarray.make_surface(frame)

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        if isCollide(playerx, playery, upperPipes, lowerPipes):
            return

        # score check
        playerMidPos = playerx + GAME_SPRITES['player'].get_width()/2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + GAME_SPRITES['pipe'][0].get_width()/2
            if pipeMidPos <= playerMidPos < pipeMidPos +4:
                score += 1
                print("Your score is " + str(score))
                GAME_SOUNDS['point'].play()

        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY

        if playerFlapped:
            playerFlapped = False
        playerHeight = GAME_SPRITES['player'].get_height()
        playery = playery + min(playerVelY, GROUNDY - playery - playerHeight)

        # move pipes to left
        for upperPipe, lowerPipe in zip(upperPipes, lowerPipes):
            upperPipe['x'] += pipeVelX
            lowerPipe['x'] += pipeVelX

        # add new pipe when the first pipe about to cross leftmost part of screen
        if 0 < upperPipes[0]['x'] < 5:
            newpipe = getRandomPipe()
            upperPipes.append(newpipe[0])
            lowerPipes.append(newpipe[1])

        # remove pipe out of screen
        if upperPipes[0]['x'] < -GAME_SPRITES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # If bird is falling down, no flapping
        if tilt <= -80:
            img = IMGS[1]
            img_count = ANIMATION_TIME * 2
        #rotated_image = pygame.transform.rotate(img, tilt)
        #image = rotated_image.get_rect(center=img.get_rect(topleft=(playerx, playery)).center)

        # blit sprites
        SCREEN.blit(GAME_SPRITES['background'], (0, 0))
        for upperPipe, lowerPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(GAME_SPRITES['pipe'][0], (upperPipe['x'], upperPipe['y']))
            SCREEN.blit(GAME_SPRITES['pipe'][1], (lowerPipe['x'], lowerPipe['y']))

        SCREEN.blit(GAME_SPRITES['base'], (basex, GROUNDY))
        SCREEN.blit(image, (playerx, playery))

        SCREEN.blit(frame, (SCREENWIDTH - 136, 8))

        myDigits = [int(x) for x in list(str(score))]

        width = 0

        for digit in myDigits:
            width += GAME_SPRITES['numbers'][digit].get_width()

        Xoffset = (SCREENWIDTH - width)/2

        for digit in myDigits:
            SCREEN.blit(GAME_SPRITES['numbers'][digit], (Xoffset, SCREENHEIGHT*0.12))
            Xoffset += GAME_SPRITES['numbers'][digit].get_width()

        pygame.display.update()
        FPSCLOCK.tick(FPS)

    capture.release()
    cv2.destroyAllWindows()


def isCollide(playerx, playery, upperPipes, lowerPipes):
    if playery > GROUNDY - GAME_SPRITES['player'].get_height() - 1 or playery < 0:
        GAME_SOUNDS['hit'].play()
        return True

    for pipe in upperPipes:
        pipeHeight = GAME_SPRITES['pipe'][0].get_height()
        if (playery< pipeHeight + pipe['y'] and abs(playerx - pipe['x']) < GAME_SPRITES['player'].get_width()-50):
            GAME_SOUNDS['hit'].play()
            return True

    for pipe in lowerPipes:
        if (playery + GAME_SPRITES['player'].get_height() > pipe['y']+40) and abs(playerx - pipe['x']) < GAME_SPRITES['player'].get_width()-50:
            GAME_SOUNDS['hit'].play()
            return True
    return False


def getRandomPipe():
    pipeHeight = GAME_SPRITES['pipe'][0].get_height()
    offset = SCREENHEIGHT / 3
    y2 = offset + random.randrange(0, int(SCREENHEIGHT - GAME_SPRITES['base'].get_height() - 1.2 * offset))
    pipeX = SCREENWIDTH + 10
    y1 = pipeHeight - y2 + offset
    pipe = [
        {'x': pipeX, 'y': -y1},  # upper Pipe
        {'x': pipeX, 'y': y2}  # lower Pipe
    ]
    return pipe


if __name__ == '__main__':
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    pygame.display.set_caption('Squatty Bird')
    GAME_SPRITES['numbers'] = (
        pygame.image.load('gallery/sprites/0.png').convert_alpha(),
        pygame.image.load('gallery/sprites/1.png').convert_alpha(),
        pygame.image.load('gallery/sprites/2.png').convert_alpha(),
        pygame.image.load('gallery/sprites/3.png').convert_alpha(),
        pygame.image.load('gallery/sprites/4.png').convert_alpha(),
        pygame.image.load('gallery/sprites/5.png').convert_alpha(),
        pygame.image.load('gallery/sprites/6.png').convert_alpha(),
        pygame.image.load('gallery/sprites/7.png').convert_alpha(),
        pygame.image.load('gallery/sprites/8.png').convert_alpha(),
        pygame.image.load('gallery/sprites/9.png').convert_alpha(),
    )

    GAME_SPRITES['message'] = pygame.image.load('gallery/sprites/message.png').convert_alpha()
    GAME_SPRITES['base'] = pygame.image.load('gallery/sprites/base.png').convert_alpha()
    GAME_SPRITES['pipe'] = (pygame.transform.rotate(pygame.image.load(PIPE).convert_alpha(), 180),
                            pygame.image.load(PIPE).convert_alpha())

    # Game sounds
    GAME_SOUNDS['die'] = pygame.mixer.Sound('gallery/audio/die.wav')
    GAME_SOUNDS['hit'] = pygame.mixer.Sound('gallery/audio/hit.wav')
    GAME_SOUNDS['point'] = pygame.mixer.Sound('gallery/audio/point.wav')
    GAME_SOUNDS['swoosh'] = pygame.mixer.Sound('gallery/audio/swoosh.wav')
    GAME_SOUNDS['wing'] = pygame.mixer.Sound('gallery/audio/wing.wav')

    GAME_SPRITES['background'] = pygame.image.load(BACKGROUND).convert()
    GAME_SPRITES['player'] = pygame.image.load(PLAYER).convert_alpha()

    while True:
        welcomeScreen()
        mainGame()
