
## Visualization class


import numpy as np
import open3d as o3d

from open3d.web_visualizer import draw


class VisTool:
    """
    """

    def __init__(self, embed=True, **kwargs):

        if embed:
            self._init_embeded()
            self.show_point_cloud = self._show_point_cloud_embedded
            self.update_point_cloud = self._update_point_cloud_embedded
            self.destroy = self._destroy_embedded
            self.add_pose_arrow = self._add_pose_arrow_embedded
        else:
            self._init_native()
            self.show_point_cloud = self._show_point_cloud_native
            self.update_point_cloud = self._update_point_cloud_native
            self.destroy = self._destroy_native
            self.add_pose_arrow = self._add_pose_arrow_native


    def _init_native(self):
        ## Pop out
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.new_pcd = o3d.geometry.PointCloud()


    def _init_embeded(self):
        ## Embedded
        self.shapes = []


    def _show_point_cloud_embedded(self, data):
        ## Embedded

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(data)
        new_pcd.colors = o3d.utility.Vector3dVector(self.get_colors(data))

        draw(new_pcd)


    def _show_point_cloud_native(self, data):
        ## Pop out:
        self.new_pcd.points = o3d.utility.Vector3dVector(data)
        self.new_pcd.colors = o3d.utility.Vector3dVector(self.get_colors(data))

        self.vis.add_geometry(self.new_pcd)
        self.vis.run()
        self.vis.destroy_window()


    def calculate_zy_rotation_for_arrow(self, vec):
        gamma = np.arctan2(vec[1], vec[0])
        Rz = np.array([
                        [np.cos(gamma), -np.sin(gamma), 0],
                        [np.sin(gamma), np.cos(gamma), 0],
                        [0, 0, 1]
                    ])
    
        vec = Rz.T @ vec
    
        beta = np.arctan2(vec[0], vec[2])
        Ry = np.array([
                        [np.cos(beta), 0, np.sin(beta)],
                        [0, 1, 0],
                        [-np.sin(beta), 0, np.cos(beta)]
                    ])
        return Rz, Ry

    
    def get_arrow(self, end, origin, scale):
        assert(not np.all(end == origin))
        vec = end - origin
        size = np.sqrt(np.sum(vec**2))
    
        Rz, Ry = self.calculate_zy_rotation_for_arrow(vec)
        mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
            cone_height=size*0.2 * scale,
            cylinder_radius=size/30 * scale,
            cylinder_height=size*(1 - 0.2*scale))
        mesh.rotate(Ry, center=np.array([0, 0, 0]))
        mesh.rotate(Rz, center=np.array([0, 0, 0]))
        mesh.translate(origin)
        return(mesh)


    def _add_pose_arrow_embedded(self, curr_se3):
        origin = curr_se3[:3, -1]
        end = np.matmul(np.array([10, 0, 0]), curr_se3[:3, :3])
        scale = 1 / np.sqrt(3)
        arrow = self.get_arrow(end, origin, scale)
        self.shapes.append(arrow)


    def _add_pose_arrow_native(self, curr_se3):
        origin = curr_se3[:3, -1]
        end = np.matmul(np.array([10, 0, 0]), curr_se3[:3, :3])
        scale = 1 / np.sqrt(3)
        arrow = self.get_arrow(end, origin, scale)
        self.vis.add_geometry(arrow)
        self.vis.poll_events()
        self.vis.update_renderer()


    def _update_point_cloud_embedded(self, last_message, odom_transform):
        ## Embedded
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(last_message)
        new_pcd.colors = o3d.utility.Vector3dVector(np.zeros((last_message.shape[0], last_message.shape[1])))
        new_pcd.transform(odom_transform)
        self.shapes.append(new_pcd)


    def get_colors(self, data):
        # TODO: proper color map

        channel = 2
        colors = np.zeros((data.shape[0], data.shape[1]), dtype=int)
        min = int(data[:, -1].min())
        colors[:, channel] = data[:, -1].astype(int) - min
        max = colors.max()
        colors[:, channel] = (colors[:, channel] / max) * 254
        return colors


    def _update_point_cloud_native(self, last_message, odom_transform):
        ## Pop out
        self.new_pcd.points = o3d.utility.Vector3dVector(last_message)
        self.new_pcd.colors = o3d.utility.Vector3dVector(self.get_colors(last_message))

        self.new_pcd.transform(odom_transform)
        self.vis.add_geometry(self.new_pcd)
        self.vis.poll_events()
        self.vis.update_renderer()


    def _destroy_embedded(self):
        draw(self.shapes)


    def _destroy_native(self):
        self.vis.destroy_window()

