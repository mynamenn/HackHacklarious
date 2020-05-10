''' '''
import random

class Pipe:

    def __init__(self, velocityX, GAME_SPRITES, SCREENHEIGHT):

        self.velocityX = velocityX

        self.GAME_SPRITES = GAME_SPRITES

        self.SCREENHEIGHT = SCREENHEIGHT


    def getRandomPipe(self):
        pipeHeight = self.GAME_SPRITES['pipe'][0].get_height()
        offset = self.SCREENHEIGHT / 3
        y2 = offset + random.randrange(0, int(self.SCREENHEIGHT - self.GAME_SPRITES['base'].get_height() - 1.2 * offset))
        pipeX = self.SCREENHEIGHT + 10
        y1 = pipeHeight - y2 + offset
        pipe = [
            {'x': pipeX, 'y': -y1},  # upper Pipe
            {'x': pipeX, 'y': y2}  # lower Pipe
        ]
        return pipe

