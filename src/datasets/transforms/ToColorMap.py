import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ToColorMap:
    """Applies a color map to the given sample.

    Args:
        colormap: a valid plt.get_cmap
    """

    def __init__(self, colormap=plt.get_cmap("magma")):
        self.colormap = colormap

    def __call__(self, sample):
        sample_colored = self.colormap(np.array(sample))

        return Image.fromarray((sample_colored[:, :, :3] * 255).astype(np.uint8))
