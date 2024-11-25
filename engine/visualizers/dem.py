import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage


class VisTool:
    """
    Generate a video for visualization
    """
    def __init__(self, embed=True, vis_height=None):
        """
        vis_height is the resolution of output frame
        """

        self._vis_height = vis_height

        self.interval = 50
        self.blit = True
        self.repeat_delay = 1000
        self.animated = True

        self.fig, self.ax = plt.subplots()
        self.images = []


    def merge_geotiffs(self, images, north: list[int], west: list[int]):
        
        north_images = []

        for n in range(north[-1] - north[0]):
            count = n * (west[-1] - west[0])
            merged = images[count]
            for w in range(west[-1] - west[0] - 1):
                merged = np.concatenate((images[count + w + 1], merged), axis=1)

            north_images.append(merged)

        merged = north_images[0]
        for i in range(len(north_images) - 1):
            merged = np.concatenate((north_images[i + 1], merged), axis=0)

        return merged


    def show(self, image):
        if type(image) == np.ndarray:
            plt.imshow(image, interpolation='nearest')
            plt.show()
        elif type(image) == str:
            img = PILImage.open(image)
            plt.imshow(img, interpolation='nearest')
            plt.show()

