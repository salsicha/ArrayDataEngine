
## Visualization class


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
            from .visualizers.point_cloud import VisTool as PCVisTool

            self.vis_tool = PCVisTool(embed, **kwargs)
        elif vis_type == "image":
            from .visualizers.video_segment import VisTool as ImgVisTool

            self.vis_tool = ImgVisTool(embed, **kwargs)
        else:
            raise ValueError(f"Visualization type not supported: ['image', 'pointcloud']")

        # self.init()


    def update(self, *args, **kwargs):
        self.vis_tool.update(*args, **kwargs)


    def show(self, *args, **kwargs):
        self.vis_tool.show(*args, **kwargs) # this calls destroy


    # def show_ego(self, *args, **kwargs):
    #     self.vis_tool.show(*args, **kwargs) # this calls destroy
