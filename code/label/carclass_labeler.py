import numpy as np
import open3d as o3d

LINES = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]]
COLORS = [[0, 1, 0] for i in range(len(LINES))]
LINES = o3d.utility.Vector2iVector(LINES)
    
def get_box_corners(pcd):
    pc = np.array(pcd.points)
    max_x = pc[:,0].max()
    min_x = pc[:,0].min()
    max_y = pc[:,1].max()
    min_y = pc[:,1].min()
    max_z = pc[:,2].max()
    min_z = pc[:,2].min()
    corners = [
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [min_x, max_y, min_z],
        [max_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [min_x, max_y, max_z],
        [max_x, max_y, max_z]
    ]
    
    return corners

def draw_box(pcd, corners, label_mask):
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=LINES
    )
    line_set.colors = o3d.utility.Vector3dVector(COLORS)
    
    pcd.paint_uniform_color([0, 0, 0])
    pc_color = np.array(pcd.colors)
    label_color = np.array([1, 0, 0])
    for i in label_mask:
        pc_color[i] = label_color
    pcd.colors = o3d.utility.Vector3dVector(pc_color)
    
    o3d.visualization.draw_geometries([pcd, line_set])

def rotate(pcd, angle):
    angle = np.deg2rad(angle)
    R = pcd.get_rotation_matrix_from_xyz((0,0,angle))
    
    return pcd.rotate(R, center=(0,0,0))

def height_filter(pcd, height_ratio=[2/5, 3/5]):
    lidar = np.array(pcd.points)
    min_z = lidar.min(0)[2]
    max_z = lidar.max(0)[2] - min_z
    translated_lidar = lidar[:,2] - min_z
    mask = (translated_lidar > height_ratio[0]*max_z) & \
           (translated_lidar < height_ratio[1]*max_z)
    ind = np.where(mask)[0].tolist()
    pcd = pcd.select_by_index(ind)
    
    return pcd

def outlier_filter(pcd, nb_neighbors=32, std_ratio=1.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    
    return pcd.select_by_index(ind)

def filter_z(corners, pc, ratio_min_z, ratio_max_z):
    min_z = corners[0][2]
    max_z = corners[7][2]
    new_min_z = min_z + ratio_min_z * (max_z - min_z)
    new_max_z = min_z + ratio_max_z * (max_z - min_z)
    new_corners = np.copy(corners)
    for i in range(4):
        new_corners[i][2] = new_min_z
        new_corners[i + 4][2] = new_max_z
    
    new_pc = np.copy(pc)    
    mask_1 = np.where(new_pc[:,2] >= new_min_z)
    new_pc = new_pc[mask_1]
    mask_2 = (new_pc[:,2] <= new_max_z)
    new_pc = new_pc[mask_2]
    
    mask_1 = np.array(mask_1).squeeze()
    mask_1 = mask_1[mask_2]
    
    return new_corners, new_pc, mask_1

def point_distance_line(point,line_point1,line_point2):
    vec1 = np.copy(point)
    vec1 = -vec1
    vec1[:,0] += line_point1[0]
    vec1[:,1] += line_point1[1]
    
    vec2 = np.copy(point)
    vec2 = -vec2
    vec2[:,0] += line_point2[0]
    vec2[:,1] += line_point2[1]
    
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    
    return distance

def distance_mask(corners, pc, mask, distance_scope = 0.05):
    pc_xy = pc[:,:2]
    line_point1 = corners[1,:]
    line_point1 = line_point1[:2]
    line_point2 = corners[2,:]
    line_point2 = line_point2[:2]
    
    distance = point_distance_line(pc_xy,line_point1,line_point2)
    distance_mask = np.where(distance <= distance_scope)
    distance_mask = np.array(distance_mask).squeeze()
    label_mask = np.copy(mask)
    label_mask = label_mask[distance_mask]
    
    return label_mask

def closeness_criterion(c1, c2, d0):
    d1 = np.minimum(c1.max()-c1, c1-c1.min())
    d2 = np.minimum(c2.max()-c2, c2-c2.min())
    
    return (1/np.maximum(np.minimum(d1, d2), d0)).sum()

def project_to_unit(theta, x):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    c1 = (x * np.array([cos_theta, sin_theta])).sum(1)
    c2 = (x * np.array([-sin_theta, cos_theta])).sum(1)
    
    return c1, c2

def rectangle_fitting(pcd, num_bins=100, d0=1e-3):
    x = np.array(pcd.points)[:,:2]
    thetas = [np.pi*i/(2*num_bins) for i in range(num_bins)]

    max_value = 0
    best_theta = None
    for theta in thetas:
        c1, c2 = project_to_unit(theta, x)
        value = closeness_criterion(c1, c2, d0)
        if value > max_value:
            max_value = value
            best_theta = theta
    
    return best_theta
