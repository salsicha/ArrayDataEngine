import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm

from PIL import Image as PILImage

# BoxList (torch), OpenCV (box drawing), and IPython.display are imported
# lazily so the visualizer stays usable without the full ML/notebook stack.


class VisTool:
    """
    Generate a video for visualization
    """
    def __init__(self, embed=True, vis_height=None, output_path=None):
        """
        vis_height is the resolution of output frame.
        output_path is where the embedded animation GIF is saved; by default
        it goes to a temporary file that is removed after display.
        """

        self.show_animation = self.get_show_animation(embed)
        self.output_path = output_path

        self._vis_height = vis_height
        # by default, 50 colors
        self.num_colors = 50
        self.colors = self.get_n_colors(self.num_colors)
        # use coco class name order
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']

        self.interval = 50
        self.blit = True
        self.repeat_delay = 1000
        self.animated = True

        self.fig, self.ax = plt.subplots()
        self.images = []


    def get_show_animation(self, embed):
        if embed:
            return self._show_animation_embed
        else:
            return self._show_animation_native


    def append_img(self, frame):
        if len(self.images) == 0:
            self.ax.imshow(frame)
        im = self.ax.imshow(frame, animated=self.animated)
        self.images.append([im])


    def update(self, frame, *args, **kwargs):
        # The Visualizer facade delegates update(); collect frames for the
        # animation shown by show_animation().
        self.append_img(frame)


    def show(self, image=None):
        """Show one image (array or file path), or, with no argument, play
        the frames collected with update()/append_img() as an animation."""
        if image is None:
            if not self.images:
                raise ValueError(
                    "show() needs an image (a NumPy array or an image file path), "
                    "or frames added first with update()/append_img()"
                )
            return self.show_animation()
        if isinstance(image, np.ndarray):
            plt.imshow(image, interpolation='nearest')
            plt.show()
        elif isinstance(image, (str, os.PathLike)):
            img = PILImage.open(image)
            plt.imshow(img, interpolation='nearest')
            plt.show()
        else:
            raise TypeError(
                f"show() expects a NumPy array or an image file path, got {type(image).__name__}"
            )


    def _show_animation_embed(self, output_path=None):
        ## Notebook embedded, called from library:
        from IPython.display import display, Image

        output_path = self.output_path if output_path is None else output_path
        ani = animation.ArtistAnimation(self.fig, self.images, interval=self.interval, blit=self.blit, repeat_delay=self.repeat_delay)
        writer = "ffmpeg" if animation.writers.is_available("ffmpeg") else "pillow"
        if output_path is None:
            # No explicit destination: never write into the working directory.
            with tempfile.TemporaryDirectory(prefix="ade_animation_") as tmp:
                path = Path(tmp) / "animation.gif"
                ani.save(path, writer=writer)
                gif = path.read_bytes()
        else:
            path = Path(output_path)
            ani.save(path, writer=writer)
            gif = path.read_bytes()
        plt.close(self.fig)
        display(Image(gif))
        return None if output_path is None else path


    def _show_animation_native(self, output_path=None):
        ## Pop out:
        # Keep a reference so the animation is not garbage-collected, and
        # show the figure — the animation only starts on its first draw.
        self._animation = animation.ArtistAnimation(
            self.fig, self.images, interval=self.interval, blit=self.blit, repeat_delay=self.repeat_delay
        )
        output_path = self.output_path if output_path is None else output_path
        if output_path is not None:
            writer = "ffmpeg" if animation.writers.is_available("ffmpeg") else "pillow"
            self._animation.save(Path(output_path), writer=writer)
        plt.show()


    @staticmethod
    def get_n_colors(n, colormap="gist_ncar"):
        # Get n color samples from the colormap, derived from: https://stackoverflow.com/a/25730396/583620
        # gist_ncar is the default colormap as it appears to have the highest number of color transitions.
        # tab20 also seems like it would be a good option but it can only show a max of 20 distinct colors.
        # For more options see:
        # https://matplotlib.org/examples/color/colormaps_reference.html
        # and https://matplotlib.org/users/colormaps.html

        colors = matplotlib.colormaps[colormap](np.linspace(0, 1, n))
        # Randomly shuffle the colors
        np.random.shuffle(colors)
        # Opencv expects bgr while cm returns rgb, so we swap to match the colormap (though it also works fine without)
        # Also multiply by 255 since cm returns values in the range [0, 1]
        colors = colors[:, (2, 1, 0)] * 255
        return colors


    def normalize_output(self, frame, results: "BoxList"):
        if self._vis_height is not None:
            import cv2

            boxlist_height = results.size[1]
            frame_height, frame_width = frame.shape[:2]
            assert (boxlist_height == frame_height)

            rescale_ratio = float(self._vis_height) / float(frame_height)
            new_height = int(round(frame_height * rescale_ratio))
            new_width = int(round(frame_width * rescale_ratio))

            frame = cv2.resize(frame, (new_width, new_height))
            results = results.resize((new_width, new_height))

        return frame, results


    def frame_vis_generator(self, frame, results: "BoxList" = None):
        if results is None:
            return frame
        import cv2

        frame, results = self.normalize_output(frame, results)
        ids = results.get_field('ids')
        results = results[ids >= 0]
        results = results.convert('xyxy')
        bbox = results.bbox.detach().cpu().numpy()
        ids = results.get_field('ids').tolist()
        labels = results.get_field('labels').tolist()

        for i, entity_id in enumerate(ids):
            color = self.colors[entity_id % self.num_colors]
            label = labels[i]
            if 0 < label <= len(self.class_names):
                class_name = self.class_names[label - 1]
            else:
                class_name = f"class_{label}"
            text_width = len(class_name) * 20
            x1, y1, x2, y2 = (np.round(bbox[i, :])).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)
            cv2.putText(frame, str(entity_id), (x1 + 5, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)
            # Draw black background rectangle for test
            cv2.rectangle(frame, (x1-5, y1-25), (x1+text_width, y1), color, -1)
            cv2.putText(frame, '{}'.format(class_name), (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
        return frame
