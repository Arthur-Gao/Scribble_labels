import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from carclass_labeler import (distance_mask, filter_z, get_box_corners,
                              height_filter, outlier_filter, rectangle_fitting,
                              rotate)
from class_labeler import (buildingLabeler, parkingLabeler, roadLabeler,
                           sidewalkLabeler, trunkLabeler, vegetationLabeler)

null = np.array([])
class AutoLabeler:
    def __init__(self, xyz, sem_mask, label_filter, distance=0.6):
        self.xyz = xyz
        self.sem_mask = sem_mask
        self.label = label_filter
        self.padding = distance
     
    def labeler(self):
        if self.label == 40: # road
            roadlabeler = roadLabeler(self.xyz, self.sem_mask, self.label, self.padding)
            mask = roadlabeler.get_road_label()
        elif self.label == 44: # parking
            parkinglabeler = parkingLabeler(self.xyz, self.sem_mask, self.label, self.padding)
            mask = parkinglabeler.get_parking_label()
        elif self.label == 48: # sidewalk
            sidewalklabeler = sidewalkLabeler(self.xyz, self.sem_mask, self.label, self.padding)
            mask = sidewalklabeler.get_sidewalk_label()
        elif self.label in [50, 51]: # building / fence
            buildinglabeler = buildingLabeler(self.xyz, self.sem_mask, self.label, self.padding)
            mask = buildinglabeler.get_building_label()
        elif self.label in [70, 81]: # vegetation / traffic sign
            vegetationlabeler = vegetationLabeler(self.xyz, self.sem_mask, self.label, self.padding)
            min_points = 5 if self.label == 81 else 10
            mask = vegetationlabeler.get_vegetation_label(min_points)
        elif self.label in [71, 80]: # trunk / pole
            trunklabeler = trunkLabeler(self.xyz, self.sem_mask, self.label, self.padding)
            mask = trunklabeler.get_trunk_label()
        
        return mask
    
    # 10: car / 11: bicycle / 13: bus / 15: motorcycle / 18: truck / 20: other-vehicle
    def get_car_label(self, single_car_xyz, ratio_min_z=0.2, ratio_max_z=1.0):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(single_car_xyz)
        
        pcd = rotate(pcd, 30)
        pcd_ = height_filter(pcd)
        pcd_ = outlier_filter(pcd_)

        theta = rectangle_fitting(pcd_)
        pcd = rotate(pcd, -np.rad2deg(theta))
        
        corners = get_box_corners(pcd)
        pc = np.array(pcd.points)
        
        new_corners, new_pc, new_pc_mask = filter_z(corners, pc, ratio_min_z, ratio_max_z)
        label_mask = distance_mask(new_corners, new_pc, new_pc_mask, distance_scope=self.padding)
        
        return label_mask
    
    def concat_labels_separation(self, frame_idx, points_num, label_mask):
        total_points_num = 0;
        sem_mask_per_frame = []
        for i in range(frame_idx.shape[0]):
            semi_separation_mask_1 = np.where(self.sem_mask >= total_points_num)
            semi_separation_mask_1 = np.array(semi_separation_mask_1).squeeze()
            semi_separation_mask_1 = semi_separation_mask_1.flatten()
            # if (semi_separation_mask_1.shape[1] == 1):
            #     semi_separation_mask_1 = semi_separation_mask_1[0]
            # else:
            #     semi_separation_mask_1 = np.squeeze(semi_separation_mask_1)
            if (semi_separation_mask_1.shape[0] == 0):
                sem_mask_per_frame.append(null)
                continue;
            semi_mask_1 = np.copy(self.sem_mask)
            semi_mask_1 = semi_mask_1[semi_separation_mask_1]
            
            semi_separation_mask_2 = np.where(semi_mask_1 < total_points_num + points_num[i])
            semi_separation_mask_2 = np.array(semi_separation_mask_2).squeeze()
            semi_separation_mask_2 = semi_separation_mask_2.flatten()
            # if (semi_separation_mask_2.shape[1] == 1):
            #     semi_separation_mask_2 = semi_separation_mask_2[0]
            # else:
            #     semi_separation_mask_2 = np.squeeze(semi_separation_mask_2)
            
            # if (isinstance(semi_separation_mask_2, int)):
            #     semi_separation_mask_2 = np.array([semi_separation_mask_2])
            # elif (semi_separation_mask_2.shape[0] == 0):
            #     sem_mask_per_frame.append(null)
            #     continue;
            if (semi_separation_mask_2.shape[0] == 0):
                sem_mask_per_frame.append(null)
                continue;
            semi_separation_mask_1 = semi_separation_mask_1[semi_separation_mask_2]
            
            separation_mask = np.copy(self.sem_mask)
            separation_mask = separation_mask[semi_separation_mask_1]
            
            separation_mask -= total_points_num
            sem_mask_per_frame.append(separation_mask)
            
            total_points_num += points_num[i]
            
        label_points_num = np.copy(points_num)
        for i in range(label_points_num.shape[0]):
            label_points_num[i] = sem_mask_per_frame[i].shape[0]
            
        total_label_points_num = 0
        label_mask_per_frame = []
        for i in range(frame_idx.shape[0]):
            if (label_points_num[i] == 0):
                label_mask_per_frame.append(null)
                continue;
            
            semi_separation_mask_1 = np.where(label_mask >= total_label_points_num)
            semi_separation_mask_1 = np.array(semi_separation_mask_1).squeeze()
            semi_separation_mask_1 = semi_separation_mask_1.flatten()
            semi_mask_1 = np.copy(label_mask)
            semi_mask_1 = semi_mask_1[semi_separation_mask_1]
            
            semi_separation_mask_2 = np.where(semi_mask_1 < total_label_points_num + label_points_num[i])
            semi_separation_mask_2 = np.array(semi_separation_mask_2).squeeze()
            semi_separation_mask_2 = semi_separation_mask_2.flatten()
            semi_separation_mask_1 = semi_separation_mask_1[semi_separation_mask_2]
            
            separation_mask = np.copy(label_mask)
            separation_mask = separation_mask[semi_separation_mask_1]
            separation_mask -= total_label_points_num
            
            separation_mask = sem_mask_per_frame[i][separation_mask]
            label_mask_per_frame.append(separation_mask)
            
            total_label_points_num += label_points_num[i]
        
        return label_mask_per_frame
    
    def concat_car_labels_separation(self, frame_idx, points_num, label_mask):
        total_points_num = 0;
        sem_mask_per_frame = []
        for i in range(frame_idx.shape[0]):
            semi_separation_mask_1 = np.where(label_mask >= total_points_num)
            semi_separation_mask_1 = np.array(semi_separation_mask_1).squeeze()
            semi_separation_mask_1 = semi_separation_mask_1.flatten()
            if (semi_separation_mask_1.shape[0] == 0):
                sem_mask_per_frame.append(null)
                continue;
            semi_mask_1 = np.copy(label_mask)
            semi_mask_1 = semi_mask_1[semi_separation_mask_1]
            
            semi_separation_mask_2 = np.where(semi_mask_1 < total_points_num + points_num[i])
            semi_separation_mask_2 = np.array(semi_separation_mask_2).squeeze()
            semi_separation_mask_2 = semi_separation_mask_2.flatten()
            if (semi_separation_mask_2.shape[0] == 0):
                sem_mask_per_frame.append(null)
                continue;
            semi_separation_mask_1 = semi_separation_mask_1[semi_separation_mask_2]
            
            separation_mask = np.copy(label_mask)
            separation_mask = separation_mask[semi_separation_mask_1]
            
            separation_mask -= total_points_num
            sem_mask_per_frame.append(separation_mask)
            
            total_points_num += points_num[i]
        
        return sem_mask_per_frame
