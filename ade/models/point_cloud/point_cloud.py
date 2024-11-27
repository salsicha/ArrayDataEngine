
import open3d as o3d

def o3d_icp(last_message, prev_message, icp_initial):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(last_message)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(prev_message)
    reg_p2p = o3d.pipelines.registration.registration_icp(
                                                        source = source, 
                                                        target = target, 
                                                        max_correspondence_distance = 10, 
                                                        init = icp_initial, 
                                                        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
                                                        )
    odom_transform = reg_p2p.transformation 
    return odom_transform
