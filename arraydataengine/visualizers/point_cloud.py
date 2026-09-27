
## Visualization class


import json
import os
from pathlib import Path

import numpy as np

o3d = None
draw = None

# Points embedded in the standalone HTML viewer (uniformly subsampled).
HTML_MAX_POINTS = 100000


def _ensure_open3d():
    global o3d, draw
    if o3d is not None:
        return True
    try:
        import open3d as imported_open3d
    except ImportError:
        return False
    try:
        from open3d.web_visualizer import draw as imported_draw
    except ImportError:
        imported_draw = None
    o3d = imported_open3d
    draw = imported_draw
    return True


def _configure_native_glfw_platform():
    if os.environ.get("GLFW_PLATFORM"):
        return
    if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
        os.environ["GLFW_PLATFORM"] = "x11"


class VisTool:
    """
    """

    def __init__(self, embed=True, **kwargs):
        self.output_path = Path(kwargs.get("output_path", "ade_pointcloud_viewer.html"))
        backend = str(kwargs.get("backend", "auto")).lower()
        native_requested = backend in {"native", "open3d", "desktop"} or (backend == "auto" and not embed)

        if native_requested:
            _configure_native_glfw_platform()
        has_open3d = False if backend == "html" else _ensure_open3d()
        if native_requested and not has_open3d:
            raise RuntimeError(
                "The native point-cloud visualizer requires the optional `open3d` dependency. "
                "Install it with `pip install open3d` or use backend='html'."
            )

        # The embedded backend needs open3d.web_visualizer.draw; when open3d
        # is present without the Jupyter stack, fall back to the html viewer.
        embedded_available = has_open3d and draw is not None
        if backend == "html" or not has_open3d or (embed and not native_requested and not embedded_available):
            self._init_html()
            self.show_point_cloud = self._show_point_cloud_html
            self.update_point_cloud = self._update_point_cloud_html
            self.destroy = self._destroy_html
            self.add_pose_arrow = self._add_pose_arrow_html
            self.add_point_cloud = self._add_point_cloud_html
            self.show = self._show_html
            self.update = self._update_point_cloud_html
        elif embed and not native_requested:
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
        # Streaming uniform decimation keeps memory bounded for long inputs:
        # only incoming points whose running index is a multiple of the
        # stride are kept, and the stride doubles whenever the kept set
        # exceeds twice the display budget.
        self._html_stride = 1
        self._html_seen = 0
        self._html_kept = 0


    def _add_point_cloud_html(self, points, colors=None):
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
            return
        xyz = arr[:, :3]
        # NaN/Inf would serialize as invalid JSON and blank the viewer.
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if xyz.shape[0] == 0:
            return
        start = (-self._html_seen) % self._html_stride
        kept = xyz[start::self._html_stride].copy()
        self._html_seen += xyz.shape[0]
        if kept.shape[0]:
            self.point_sets.append(kept)
            self._html_kept += kept.shape[0]
        while self._html_kept > 2 * HTML_MAX_POINTS:
            merged = np.vstack(self.point_sets)[::2]
            self.point_sets = [merged]
            self._html_kept = merged.shape[0]
            self._html_stride *= 2


    def _add_pose_arrow_html(self, curr_se3):
        transform = np.asarray(curr_se3, dtype=np.float64)
        origin = transform[:3, 3]
        end = transform[:3, :3] @ np.array([1.0, 0.0, 0.0]) + origin
        if not (np.isfinite(origin).all() and np.isfinite(end).all()):
            return
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
        if points.shape[0] > HTML_MAX_POINTS:
            indices = np.linspace(0, points.shape[0] - 1, HTML_MAX_POINTS).astype(np.int64)
            points = points[indices]
        payload = {
            "points": points.round(5).tolist(),
            "poses": [
                [origin.round(5).tolist(), end.round(5).tolist()]
                for origin, end in self.pose_segments
            ],
        }
        self.output_path.write_text(_html_viewer(json.dumps(payload, allow_nan=False)), encoding="utf-8")
        print(f"Wrote point-cloud viewer to {self.output_path}")


    def _init_native(self):
        ## Pop out
        self.vis = o3d.visualization.Visualizer()
        if not self.vis.create_window():
            raise RuntimeError(
                "Open3D failed to create a native GLFW window. On Wayland this usually means "
                "the Open3D legacy visualizer/GLEW path could not use a compatible OpenGL "
                "context. Make sure XWayland is running and try `GLFW_PLATFORM=x11 "
                "python example/mapeverything.py`."
            )
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
        points = np.asarray(last_message)
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        new_pcd.colors = o3d.utility.Vector3dVector(self.get_colors(points))
        new_pcd.transform(odom_transform)
        self.shapes.append(new_pcd)


    def get_colors(self, data):
        channel = 2
        colors = np.zeros((data.shape[0], 3), dtype=np.float64)
        if data.shape[0] == 0:
            return colors
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
        # A fresh geometry per update so successive scans accumulate into a
        # map; reusing one object made every call replace the previous scan.
        points = np.asarray(last_message)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(self.get_colors(points))

        pcd.transform(odom_transform)
        self.vis.add_geometry(pcd)
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
let yaw = -0.7, pitch = 0.45, drag = false, lastX = 0, lastY = 0, pending = false;
const pts = data.points || [];
const poses = data.poses || [];
const all = pts.concat(poses.flat());
let center = [0,0,0];
if (all.length) {{
  for (const p of all) {{ center[0]+=p[0]; center[1]+=p[1]; center[2]+=p[2]; }}
  center = center.map(v => v / all.length);
}}
function quantile(values, q) {{
  if (!values.length) return 0;
  const sorted = Float64Array.from(values).sort();
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}}
// Fit the initial zoom, perspective, and height colors to the data extent
// (robust percentiles, so a few stray points do not shrink the view).
const radius = quantile(all.map(p => Math.hypot(p[0]-center[0], p[1]-center[1], p[2]-center[2])), 0.98) || 1;
const heights = pts.map(p => p[2]);
const zLo = quantile(heights, 0.02), zHi = quantile(heights, 0.98);
const zSpan = zHi > zLo ? zHi - zLo : 1;
let scale = 0.45 * Math.min(innerWidth, innerHeight) / radius;
function resize() {{ canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio; requestDraw(); }}
function project(p) {{
  const x = p[0]-center[0], y = p[1]-center[1], z = p[2]-center[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy*x - sy*y, y1 = sy*x + cy*y, z1 = z;
  const y2 = cp*y1 - sp*z1, z2 = sp*y1 + cp*z1;
  const f = scale / (1 + Math.max(-0.8, z2 / radius * 0.2));
  return [canvas.width/2 + x1*f*devicePixelRatio, canvas.height/2 - y2*f*devicePixelRatio, z2];
}}
function draw() {{
  ctx.fillStyle = '#111'; ctx.fillRect(0,0,canvas.width,canvas.height);
  hud.textContent = `${{pts.length}} points, ${{poses.length}} poses | drag rotate, wheel zoom`;
  const projected = pts.map(p => [p, project(p)]).sort((a,b) => a[1][2]-b[1][2]);
  for (const [p, q] of projected) {{
    const c = Math.round(60 + 195 * Math.max(0, Math.min(1, (p[2]-zLo) / zSpan)));
    ctx.fillStyle = `rgb(${{c}},${{180}},${{255-c/3}})`;
    ctx.fillRect(q[0], q[1], 1.6*devicePixelRatio, 1.6*devicePixelRatio);
  }}
  ctx.strokeStyle = '#ffcc33'; ctx.lineWidth = 2 * devicePixelRatio;
  for (const seg of poses) {{ const a=project(seg[0]), b=project(seg[1]); ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke(); }}
}}
// Coalesce bursts of input events into at most one redraw per frame.
function requestDraw() {{ if (pending) return; pending = true; requestAnimationFrame(() => {{ pending = false; draw(); }}); }}
canvas.addEventListener('mousedown', e => {{ drag = true; lastX = e.clientX; lastY = e.clientY; }});
addEventListener('mouseup', () => drag = false);
addEventListener('mousemove', e => {{ if (!drag) return; yaw += (e.clientX-lastX)*0.006; pitch += (e.clientY-lastY)*0.006; lastX=e.clientX; lastY=e.clientY; requestDraw(); }});
canvas.addEventListener('wheel', e => {{ e.preventDefault(); scale *= e.deltaY > 0 ? 0.9 : 1.1; requestDraw(); }}, {{passive:false}});
addEventListener('resize', resize); resize();
</script>
</body></html>"""
