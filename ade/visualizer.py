
## Visualization class


import numpy as np
import open3d as o3d

from .visualizers.point_cloud import VisTool as PCVisTool
from .visualizers.video_segment import VisGenerator as ImgVisTool


class Visualizer:
    """Visualizer Class
    
    All visualizer classes should implement:
    vis_tool = VisTool()
    vis_tool.update()
    vis_tool.show() ???
    vis_tool.destroy() ???

    
    Attributes:
    Args:
    Returns:
    """

    def __init__(self, vis_type, embed=True, **kwargs):
        """Constructor

        """
        self.vis_type = vis_type
        self.embed = embed

        if vis_type == "pointcloud":
            self.vis_tool = PCVisTool(embed, **kwargs)
        elif vis_type == "image":
            self.vis_tool = ImgVisTool(embed, **kwargs)
        else:
            raise Exception(f"Visualization type not supported: ['image', 'pointcloud']")

        # self.init()


    def update(self, *args, **kwargs):
        self.vis_tool.update(*args, **kwargs)


    def show(self, *args, **kwargs):
        self.vis_tool.show(*args, **kwargs) # this calls destroy


    # def show_ego(self, *args, **kwargs):
    #     self.vis_tool.show(*args, **kwargs) # this calls destroy


