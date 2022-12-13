import numpy as np

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
