import numpy as np
from scipy.spatial.transform import Rotation as R


def axis_local_rotate(axis_homogeneous_matrix, axis, angle_degree):
    # Rotate the axis frame locally around one of its own axes (x, y, or z) by the given angle
    angle_rad = np.deg2rad(angle_degree)
    R_old = axis_homogeneous_matrix[:3, :3]
    t_old = axis_homogeneous_matrix[:3, 3]
    
    euler_angles = [0.0, 0.0, 0.0]
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(axis.lower(), 0)
    euler_angles[axis_idx] = angle_rad
    
    local_rotation = R.from_euler('xyz', euler_angles, degrees=False).as_matrix()
    R_new = R_old @ local_rotation
    
    result = np.eye(4)
    result[:3, :3] = R_new
    result[:3, 3] = t_old
    return result


def axis_local_transform(axis_homo, axis, transform_scalar):
    # Translate the axis frame locally along one of its own axes (x, y, or z) by the scalar amount
    R_frame = axis_homo[:3, :3]
    t_old = axis_homo[:3, 3]
    
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(axis.lower(), 0)
    
    local_direction = R_frame[:, axis_idx]
    t_new = t_old + local_direction * transform_scalar
    
    result = axis_homo.copy()
    result[:3, 3] = t_new
    return result


def axis_global_rotate(axis_homo, axis_in_global, angle_degree):
    # Rotate the axis frame globally around a given axis vector in world coordinates
    angle_rad = np.deg2rad(angle_degree)
    R_old = axis_homo[:3, :3]
    t_old = axis_homo[:3, 3]
    
    axis_normalized = np.array(axis_in_global) / np.linalg.norm(axis_in_global)
    global_rotation = R.from_rotvec(axis_normalized * angle_rad).as_matrix()
    
    R_new = global_rotation @ R_old
    t_new = global_rotation @ t_old
    
    result = np.eye(4)
    result[:3, :3] = R_new
    result[:3, 3] = t_new
    return result


def axis_global_transform(axis_homo, axis_in_global, transform_scalar):
    # Translate the axis frame globally along a given axis vector in world coordinates
    t_old = axis_homo[:3, 3]
    
    axis_normalized = np.array(axis_in_global) / np.linalg.norm(axis_in_global)
    t_new = t_old + axis_normalized * transform_scalar
    
    result = axis_homo.copy()
    result[:3, 3] = t_new
    return result


def wrist_to_homogeneous(wrist_translation, wrist_axes):
    translation = np.array(wrist_translation)
    
    x_axis = np.array(wrist_axes['x_axis'])
    y_axis = np.array(wrist_axes['y_axis'])
    z_axis = np.array(wrist_axes['z_axis'])
    
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation
    
    return homogeneous_matrix


def transform_device_to_camera(wrist_pose, T_Device_Camera):
    """Transform wrist pose from device frame to camera frame."""
    T_Camera_Device = np.linalg.inv(T_Device_Camera)
    T_Camera_Wrist = T_Camera_Device @ wrist_pose
    return T_Camera_Wrist


def transform_camera_to_base(wrist_pose, T_Camera_Base):
    """Transform wrist pose from camera frame to base frame."""
    T_Base_Wrist = T_Camera_Base @ wrist_pose
    return T_Base_Wrist


def project_point_to_axis(point, axis):
    point_homo = np.array([point[0], point[1], point[2], 1.0])
    point_new_homo = np.linalg.inv(axis) @ point_homo
    return point_new_homo[:3]


def rotate_point_around_axis(point, axis, angle):
    point = np.array(point)
    axis = np.array(axis)
    axis_normalized = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle)
    rotation = R.from_rotvec(axis_normalized * angle_rad)
    rotation_matrix = rotation.as_matrix()
    rotated_point = rotation_matrix @ point
    return rotated_point


