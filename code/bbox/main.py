import numpy as np
import os
import yaml
import open3d as o3d
from scipy.linalg import pascal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm 

from dataset import AlignedKITTI
from rectangle_fitting import rectangle_fitting
from label import filter_z, distance_mask

LINES = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]]
COLORS = [[0, 1, 0] for i in range(len(LINES))]
LINES = o3d.utility.Vector2iVector(LINES)

def center_velo(velo):
    return velo - velo.mean(0)

def draw(pcd):
    o3d.visualization.draw_geometries([pcd])
    
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
    
    pcd.paint_uniform_color([0.827, 0.827, 0.827])
    pc_color = np.array(pcd.colors)
    # label_color = np.array([1, 0, 0])
    # for i in label_mask:
    #     pc_color[i] = label_color
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

if __name__ == '__main__':
    ds = AlignedKITTI('data', '04')
    velo = ds.concat_velo_based_on_label(sem_label=10,
                                         inst_label=8,
                                         idx=0,
                                         search_len=1000)
    centered_xyz = center_velo(velo[:,:3])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centered_xyz)

    pcd = rotate(pcd, 30)
    # draw(pcd)
    pcd_ = height_filter(pcd)
    # draw(pcd_)
    pcd_ = outlier_filter(pcd_)
    # draw(pcd_)

    theta = rectangle_fitting(pcd_)
    print(np.rad2deg(theta))
    pcd = rotate(pcd, -np.rad2deg(theta))
    # draw(pcd)
    
    corners = get_box_corners(pcd)
    pc = np.array(pcd.points)
    
    ratio_min_z, ratio_max_z = 0.2, 1.0
    new_corners, new_pc, new_pc_mask = filter_z(corners, pc, ratio_min_z, ratio_max_z)
    # print(new_pc_mask)
    label_mask = distance_mask(new_corners, new_pc, new_pc_mask, distance_scope=0.05)
    # print(label_mask)
    
    draw_box(pcd, corners, label_mask)
    
    # pc = np.array(pcd.points)
    # plt.scatter(pc[:,0], pc[:,1])
    # left = pc[:,0].min()
    # bottom = pc[:,1].min()
    # width = pc[:,0].max() - left
    # height = pc[:,1].max() - bottom
    
    # rect=mpatches.Rectangle((left, bottom), width, height, linewidth=1, edgecolor='r', facecolor='none')
    # plt.gca().add_patch(rect)
    # plt.show()

    # angle, pcd_2 = iterative_box(pcd)
    # print(np.rad2deg(angle))

    # import pdb; pdb.set_trace()
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # normals = np.array(pcd.normals)[z_mask]
    # angles, filtered_normals = normal_to_unit_quadrant(normals[:,:2])

    # draw(pcd_2)
    # densest = dense_finder(angles, np.deg2rad(10))
    # pc = np.array(pcd_2.points)
    # plt.scatter(pc[:,0], pc[:,1])
    # left = pc[:,0].min()
    # bottom = pc[:,1].min()
    # width = pc[:,0].max() - left
    # height = pc[:,1].max() - bottom
    # rect=mpatches.Rectangle((left, bottom), width, height,
    #                         alpha=0.1,
    #                         facecolor="red")
    # plt.gca().add_patch(rect)
    # plt.show()