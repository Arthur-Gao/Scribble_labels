import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from centerline.geometry import Centerline
from polylidar import MatrixDouble, Polylidar3D
from polylidar.polylidarutil import (apply_rotation,
                                     convert_to_shapely_polygons,
                                     generate_3d_plane, generate_test_points,
                                     get_colored_planar_segments,
                                     get_estimated_lmax,
                                     get_triangles_from_list, plot_planes_3d,
                                     plot_polygons, plot_polygons_3d,
                                     plot_triangle_meshes, plot_triangles,
                                     rotation_matrix, scale_points,
                                     set_axes_equal)
from shapely.geometry import MultiPoint, Point

from carlabel import (distance_mask, draw_box, filter_z, get_box_corners,
                      height_filter, outlier_filter, rotate)
from rectangle_fitting import rectangle_fitting

null = np.array([])
class AutoLabeler:
    def __init__(self, xyz, sem_mask, label_filter, distance=0.6):
        self.xyz = xyz
        self.sem_mask = sem_mask
        self.label = label_filter
        self.padding = distance
        
    def labeler(self):
        if self.label == 40:
            mask = self.get_polygon()
        elif self.label in [70, 81]:
            min_points = 5 if self.label == 81 else 10
            mask = self.get_cluster(min_points)
        
        return mask
    
    def get_centre(self, polygon):
        attributes = {"id": 1, "name": "polygon", "valid": True}
        centerline = Centerline(polygon, **attributes)
        return centerline

    def get_polygon(self):
        points = self.xyz
        points_mat = MatrixDouble(points, copy=True)
        polylidar_kwargs = dict(alpha=0.0, lmax=6, min_triangles=0, z_thresh=5, norm_thresh_min=0.1)
        polylidar = Polylidar3D(**polylidar_kwargs)

        _, _, polygons = polylidar.extract_planes_and_polygons(points_mat)
        for poly in polygons:
            poly.holes = []
        sp = convert_to_shapely_polygons(polygons, points, return_first=True, mp=True)
        centre_geom = self.get_centre(sp)
        gdf = gpd.GeoSeries(centre_geom.geoms)

        centre_poly = gdf.buffer(self.padding).unary_union

        geo_points = gpd.GeoSeries(MultiPoint(points)).explode(index_parts=True)
        mask = [point.covered_by(centre_poly) for point in geo_points]
        mask = np.array(mask).squeeze()
        mask = np.where(mask == True)
        mask = np.array(mask).squeeze()
        return mask
    
    def get_cluster(self, min_points=10):
        points = self.xyz
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        labels = np.array(pcd.cluster_dbscan(eps=1, min_points=min_points))
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label
                                                if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        bboxes = []
        mask = []
        for l in np.unique(labels):
            if l == -1:
                continue
            p = o3d.geometry.PointCloud()
            p.points = o3d.utility.Vector3dVector(points[labels == l])
            indexes = np.array(self.sem_mask[labels == l])

            bbox = o3d.geometry.OrientedBoundingBox().create_from_points(p.points)

            min_indice = np.argmin(bbox.extent)
            shrink_extent = bbox.extent.copy()
            shrink_extent[min_indice] = min(self.padding, min(bbox.extent))

            shrink_bbox = o3d.geometry.OrientedBoundingBox(bbox.center, bbox.R, shrink_extent)
            within_index = shrink_bbox.get_point_indices_within_bounding_box(pcd.points)

            mask.extend(within_index)
            bboxes.extend([bbox, shrink_bbox])

        mask = np.array(mask).squeeze()
        return mask
    
    def get_car_label(self, single_car_xyz, ratio_min_z=0.2, ratio_max_z=1.0):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(single_car_xyz)
        
        pcd = rotate(pcd, 30)
        pcd_ = height_filter(pcd)
        pcd_ = outlier_filter(pcd_)

        theta = rectangle_fitting(pcd_)
        # print(np.rad2deg(theta))
        pcd = rotate(pcd, -np.rad2deg(theta))
        
        corners = get_box_corners(pcd)
        pc = np.array(pcd.points)
        
        new_corners, new_pc, new_pc_mask = filter_z(corners, pc, ratio_min_z, ratio_max_z)
        label_mask = distance_mask(new_corners, new_pc, new_pc_mask, distance_scope=self.padding)
        # draw_box(pcd, corners, label_mask)
        return label_mask
        
    def concat_labels_separation(self, frame_idx, points_num, label_mask):
        total_points_num = 0;
        sem_mask_per_frame = []
        for i in range(frame_idx.shape[0]):
            semi_separation_mask_1 = np.where(self.sem_mask >= total_points_num)
            semi_separation_mask_1 = np.array(semi_separation_mask_1).squeeze()
            if (isinstance(semi_separation_mask_1,int)):
                semi_separation_mask_1 = np.array([semi_separation_mask_1])
            elif (semi_separation_mask_1.shape[0] == 0):
                sem_mask_per_frame.append(null)
                continue;
            semi_mask_1 = np.copy(self.sem_mask)
            semi_mask_1 = semi_mask_1[semi_separation_mask_1]
            
            semi_separation_mask_2 = np.where(semi_mask_1 < total_points_num + points_num[i])
            semi_separation_mask_2 = np.array(semi_separation_mask_2).squeeze()
            if (isinstance(semi_separation_mask_2,int)):
                semi_separation_mask_2 = np.array([semi_separation_mask_2])
            elif (semi_separation_mask_2.shape[0] == 0):
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
            # print(label_points_num[i])
            # print(sem_mask_per_frame[i])
            
        total_label_points_num = 0
        label_mask_per_frame = []
        for i in range(frame_idx.shape[0]):
            if (label_points_num[i] == 0):
                label_mask_per_frame.append(null)
                continue;
            
            semi_separation_mask_1 = np.where(label_mask >= total_label_points_num)
            semi_separation_mask_1 = np.array(semi_separation_mask_1).squeeze()
            semi_mask_1 = np.copy(label_mask)
            semi_mask_1 = semi_mask_1[semi_separation_mask_1]
            
            semi_separation_mask_2 = np.where(semi_mask_1 < total_label_points_num + label_points_num[i])
            semi_separation_mask_2 = np.array(semi_separation_mask_2).squeeze()
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
            if (isinstance(semi_separation_mask_1,int)):
                semi_separation_mask_1 = np.array([semi_separation_mask_1])
            elif (semi_separation_mask_1.shape[0] == 0):
                sem_mask_per_frame.append(null)
                continue;
            semi_mask_1 = np.copy(label_mask)
            semi_mask_1 = semi_mask_1[semi_separation_mask_1]
            
            semi_separation_mask_2 = np.where(semi_mask_1 < total_points_num + points_num[i])
            semi_separation_mask_2 = np.array(semi_separation_mask_2).squeeze()
            if (isinstance(semi_separation_mask_2,int)):
                semi_separation_mask_2 = np.array([semi_separation_mask_2])
            elif (semi_separation_mask_2.shape[0] == 0):
                sem_mask_per_frame.append(null)
                continue;
            semi_separation_mask_1 = semi_separation_mask_1[semi_separation_mask_2]
            
            separation_mask = np.copy(label_mask)
            separation_mask = separation_mask[semi_separation_mask_1]
            
            separation_mask -= total_points_num
            sem_mask_per_frame.append(separation_mask)
            
            total_points_num += points_num[i]
        return sem_mask_per_frame
