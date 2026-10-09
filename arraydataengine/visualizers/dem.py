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

        self.images = []


    def merge_geotiffs(self, images, north: list[int], west: list[int], overlap: int = 1):
        """Stitch north-up tiles in `DEMSource` order (south-east tile first).

        Adjacent SRTM tiles share `overlap` edge rows/columns (1 for SRTM), so
        the shared edge is kept once instead of duplicated; pass `overlap=0`
        for rasters that do not overlap. `arraydataengine.ops.mosaic_dem_tiles`
        also handles missing tiles and voids.
        """

        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        columns = west[-1] - west[0]

        def without_shared_edge(tile, axis):
            # A tile prepended to the west/north shares its last column/row
            # with the first column/row of what is already merged.
            if overlap == 0:
                return tile
            return tile[:, :-overlap] if axis == 1 else tile[:-overlap]

        north_images = []
        for n in range(north[-1] - north[0]):
            count = n * columns
            merged = images[count]
            for w in range(columns - 1):
                merged = np.concatenate((without_shared_edge(images[count + w + 1], 1), merged), axis=1)
            north_images.append(merged)

        merged = north_images[0]
        for i in range(len(north_images) - 1):
            merged = np.concatenate((without_shared_edge(north_images[i + 1], 0), merged), axis=0)

        return merged


    def show(self, image):
        if type(image) == np.ndarray:
            plt.imshow(image, interpolation='nearest')
            plt.show()
        elif type(image) == str:
            img = PILImage.open(image)
            plt.imshow(img, interpolation='nearest')
            plt.show()

