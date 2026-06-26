
## Visualization class


import json
from pathlib import Path

import numpy as np

o3d = None
draw = None


def _ensure_open3d():
    global o3d, draw
    if o3d is not None:
        return True
    try:
        import open3d as imported_open3d
        from open3d.web_visualizer import draw as imported_draw
    except ImportError:
        return False
    o3d = imported_open3d
    draw = imported_draw
    return True


class VisTool:
    """
    """

    def __init__(self, embed=True, **kwargs):
        self.output_path = Path(kwargs.get("output_path", "ade_pointcloud_viewer.html"))
        backend = str(kwargs.get("backend", "auto")).lower()

        has_open3d = False if backend == "html" else _ensure_open3d()

        if backend == "html" or not has_open3d:
            self._init_html()
            self.show_point_cloud = self._show_point_cloud_html
            self.update_point_cloud = self._update_point_cloud_html
            self.destroy = self._destroy_html
            self.add_pose_arrow = self._add_pose_arrow_html
            self.add_point_cloud = self._add_point_cloud_html
            self.show = self._show_html
            self.update = self._update_point_cloud_html
        elif embed:
            self._init_embeded()
            self.show_point_cloud = self._show_point_cloud_embedded
            self.update_point_cloud = self._update_point_cloud_embedded
            self.destroy = self._destroy_embedded
            self.add_pose_arrow = self._add_pose_arrow_embedded
            self.add_point_cloud = self._add_point_cloud_embedded
            self.show = self._show_embedded
            self.update = self._update_point_cloud_embedded
        else:
            self._init_native()
            self.show_point_cloud = self._show_point_cloud_native
            self.update_point_cloud = self._update_point_cloud_native
            self.destroy = self._destroy_native
            self.add_pose_arrow = self._add_pose_arrow_native
            self.add_point_cloud = self._add_point_cloud_native
            self.show = self._show_native
            self.update = self._update_point_cloud_native


    def _show_embedded(self, data=None):
        if data is not None:
            self._show_point_cloud_embedded(data)
            return
        self._destroy_embedded()


    def _show_native(self, data=None):
        if data is not None:
            self._show_point_cloud_native(data)
            return
        self.vis.run()
        self._destroy_native()


    def _init_html(self):
        self.point_sets = []
        self.pose_segments = []


    def _add_point_cloud_html(self, points, colors=None):
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
            return
        self.point_sets.append(arr[:, :3].copy())


    def _add_pose_arrow_html(self, curr_se3):
        transform = np.asarray(curr_se3, dtype=np.float64)
        origin = transform[:3, 3]
        end = transform[:3, :3] @ np.array([1.0, 0.0, 0.0]) + origin
        self.pose_segments.append((origin.copy(), end.copy()))


    def _update_point_cloud_html(self, last_message, odom_transform):
        points = np.asarray(last_message, dtype=np.float64)
        transform = np.asarray(odom_transform, dtype=np.float64)
        transformed = points.copy()
        transformed[:, :3] = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        self._add_point_cloud_html(transformed)


    def _show_point_cloud_html(self, data):
        self._add_point_cloud_html(data)
        self._destroy_html()


    def _show_html(self, data=None):
        if data is not None:
            self._add_point_cloud_html(data)
        self._destroy_html()


    def _destroy_html(self):
        points = np.vstack(self.point_sets) if self.point_sets else np.empty((0, 3), dtype=np.float64)
        if points.shape[0] > 100000:
            indices = np.linspace(0, points.shape[0] - 1, 100000).astype(np.int64)
            points = points[indices]
        payload = {
            "points": points.round(5).tolist(),
            "poses": [
                [origin.round(5).tolist(), end.round(5).tolist()]
                for origin, end in self.pose_segments
            ],
        }
        self.output_path.write_text(_html_viewer(json.dumps(payload)), encoding="utf-8")
        print(f"Wrote point-cloud viewer to {self.output_path}")


    def _init_native(self):
        ## Pop out
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.new_pcd = o3d.geometry.PointCloud()


    def _init_embeded(self):
        ## Embedded
        self.shapes = []


    def _add_point_cloud_embedded(self, points, colors=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if colors is None:
            colors = self.get_colors(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        self.shapes.append(pcd)


    def _add_point_cloud_native(self, points, colors=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if colors is None:
            colors = self.get_colors(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()


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
        end = np.matmul(curr_se3[:3, :3], np.array([10, 0, 0])) + origin
        scale = 1 / np.sqrt(3)
        arrow = self.get_arrow(end, origin, scale)
        self.shapes.append(arrow)


    def _add_pose_arrow_native(self, curr_se3):
        origin = curr_se3[:3, -1]
        end = np.matmul(curr_se3[:3, :3], np.array([10, 0, 0])) + origin
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
        channel = 2
        colors = np.zeros((data.shape[0], 3), dtype=np.float64)
        z_vals = data[:, -1]
        z_min = z_vals.min()
        z_max = z_vals.max()
        z_range = z_max - z_min
        if z_range > 0:
            colors[:, channel] = (z_vals - z_min) / z_range
        else:
            colors[:, channel] = 0.5
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




def _html_viewer(payload: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>ArrayDataEngine Point Cloud</title>
<style>
html, body {{ margin: 0; height: 100%; background: #111; color: #eee; font-family: system-ui, sans-serif; }}
canvas {{ display: block; width: 100vw; height: 100vh; }}
#hud {{ position: fixed; left: 12px; top: 10px; font-size: 13px; color: #ddd; }}
</style>
</head>
<body>
<canvas id=\"view\"></canvas><div id=\"hud\"></div>
<script>
const data = {payload};
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const hud = document.getElementById('hud');
let yaw = -0.7, pitch = 0.45, scale = 120, drag = false, lastX = 0, lastY = 0;
const pts = data.points || [];
const poses = data.poses || [];
let center = [0,0,0];
if (pts.length) {{
  for (const p of pts) {{ center[0]+=p[0]; center[1]+=p[1]; center[2]+=p[2]; }}
  center = center.map(v => v / pts.length);
}}
function resize() {{ canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio; draw(); }}
function project(p) {{
  const x = p[0]-center[0], y = p[1]-center[1], z = p[2]-center[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy*x - sy*y, y1 = sy*x + cy*y, z1 = z;
  const y2 = cp*y1 - sp*z1, z2 = sp*y1 + cp*z1;
  const f = scale / (1 + Math.max(-0.8, z2) * 0.015);
  return [canvas.width/2 + x1*f*devicePixelRatio, canvas.height/2 - y2*f*devicePixelRatio, z2];
}}
function draw() {{
  ctx.fillStyle = '#111'; ctx.fillRect(0,0,canvas.width,canvas.height);
  hud.textContent = `${{pts.length}} points, ${{poses.length}} poses | drag rotate, wheel zoom`;
  const projected = pts.map(p => [p, project(p)]).sort((a,b) => a[1][2]-b[1][2]);
  for (const [p, q] of projected) {{
    const c = Math.max(60, Math.min(255, 120 + (p[2]-center[2])*40));
    ctx.fillStyle = `rgb(${{c}},${{180}},${{255-c/3}})`;
    ctx.fillRect(q[0], q[1], 1.6*devicePixelRatio, 1.6*devicePixelRatio);
  }}
  ctx.strokeStyle = '#ffcc33'; ctx.lineWidth = 2 * devicePixelRatio;
  for (const seg of poses) {{ const a=project(seg[0]), b=project(seg[1]); ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke(); }}
}}
canvas.addEventListener('mousedown', e => {{ drag = true; lastX = e.clientX; lastY = e.clientY; }});
addEventListener('mouseup', () => drag = false);
addEventListener('mousemove', e => {{ if (!drag) return; yaw += (e.clientX-lastX)*0.006; pitch += (e.clientY-lastY)*0.006; lastX=e.clientX; lastY=e.clientY; draw(); }});
canvas.addEventListener('wheel', e => {{ e.preventDefault(); scale *= e.deltaY > 0 ? 0.9 : 1.1; draw(); }}, {{passive:false}});
addEventListener('resize', resize); resize();
</script>
</body></html>"""
