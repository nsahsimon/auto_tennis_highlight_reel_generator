import numpy as np
import time

# this class will hold information about the each frame

class Frame:
    count = None # the position of the frame in the list of all frames in the video
    timestamp = None # the timestamp of the frame given the current frame rate
    fps = None # the frame rate of the video
    data = [None, None] # a tuple containing the game score contained in the frame (upDigits, downDigits, edgeXCoord)
    image = None
    
    def __init__(self, count : np.random.randint(1, 100) = 10, timestamp: int = np.random.randint(1, 100), fps : float= 0.1 * np.random.randint(1, 100), data: list = [], image: np.ndarray = np.zeros((3,3),dtype=np.uint8)):
        self.count = count
        self.timestamp = timestamp
        self.fps = fps
        self.data = data
        self.image = image

    def classicTimestamp(self):
        return time.strftime("%H:%M:%S", time.gmtime(self.timestamp))