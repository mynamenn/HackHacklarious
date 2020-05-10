''' '''

class Player:

    def __init__(self, positionX, positionY, velocityY, maxVelocityY, minVelocityY, accY, BIRD_IMGS, flappedAccVel, flapped):

        self.positionX = positionX

        self.positionY = positionY

        self.velocityY = velocityY

        self.maxVelocityY = maxVelocityY

        self.minVelocityY = minVelocityY

        self.accY = accY

        self.BIRD_IMGS = BIRD_IMGS

        self.flapped = flapped

        self.flappedAccVel = flappedAccVel #velocity while flapping
