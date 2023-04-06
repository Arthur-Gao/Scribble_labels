"""
class need label:
    10: car / 11: bicycle / 13: bus / 15: motorcycle / 18: truck / 20: other-vehicle
    40: road
    44: parking
    48: sidewalk
    50: building / 51: fence
    70: vegetation / 81: traffic-sign
    71: trunk / 80: pole
    30: person / 31: bicyclist / 32: motorcyclist
    Above classes have already done. 

    72: terrain
    49: other_ground
    Above classed have not been finished.
    class_can_label = [40, 44, 48, 50, 51, 70, 71, 80, 81, 10, 11, 13, 15, 18, 20, 30, 31, 32]
    class_distance = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1]
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from centerline.geometry import Centerline
from polylidar import MatrixDouble, Polylidar3D
from polylidar.polylidarutil import convert_to_shapely_polygons
from shapely.geometry import MultiPoint


def get_centreline(polygon):
    attributes = {"id": 1, "name": "polygon", "valid": True}
    centerline = Centerline(polygon, interpolation_distance=0.1, **attributes)
    
    return centerline

def render_pcd(pcd, labels):
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

def get_bboxes(points, labels):
    bboxes = []
    for l in np.unique(labels):
        if l == -1:
            continue
        
        if (points[labels == l].shape[0] <= 10):
            continue
        
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(points[labels == l])
        
        bbox = o3d.geometry.OrientedBoundingBox().create_from_points(p.points)
        bboxes.append(bbox)

    return bboxes

def get_cluster(points, eps=1, min_points=10):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    render_pcd(pcd, labels)
    bboxes = get_bboxes(points, labels)

    return pcd, bboxes, labels


# 40: road
class roadLabeler:
    def __init__(self, xyz, sem_mask, label_filter, distance=0.6):
        self.xyz = xyz
        self.sem_mask = sem_mask
        self.label = label_filter
        self.padding = distance
        
    def get_road_label(self):
        points = self.xyz
        points_mat = MatrixDouble(points, copy=True)
        polylidar_kwargs = dict(alpha=0.0, lmax=6, min_triangles=0, z_thresh=5, norm_thresh_min=0.1)
        polylidar = Polylidar3D(**polylidar_kwargs)

        _, _, polygons = polylidar.extract_planes_and_polygons(points_mat)
        for poly in polygons:
            poly.holes = []
        sp = convert_to_shapely_polygons(polygons, points, return_first=True, mp=True)
        centre_geom = get_centreline(sp)
        gdf = gpd.GeoSeries(centre_geom.geoms)

        centre_poly = gdf.buffer(self.padding).unary_union

        geo_points = gpd.GeoSeries(MultiPoint(points)).explode(index_parts=True)
        mask = [point.covered_by(centre_poly) for point in geo_points]
        mask = np.array(mask).squeeze()
        mask = np.where(mask == True)
        mask = np.array(mask).squeeze()
        
        return mask


# 44: parking
class parkingLabeler:
    def __init__(self, xyz, sem_mask, label_filter, distance=0.6):
        self.xyz = xyz
        self.sem_mask = sem_mask
        self.label = label_filter
        self.padding = distance
        
    def get_polygon(self, points, polylidar_kwargs):
        points_mat = MatrixDouble(points, copy=True)
        polylidar = Polylidar3D(**polylidar_kwargs)

        _, _, polygons = polylidar.extract_planes_and_polygons(points_mat)
        for poly in polygons:
            poly.holes = []
        sp = convert_to_shapely_polygons(polygons, points, return_first=True, mp=True)

        return sp
    
    def shrink_cluster_middle_convex(self, pcd, bboxes, convex, label):
        shrink_bboxes = []
        selected_indices = []
        for bbox in bboxes:
            min_indice = np.argmin(bbox.extent)
            max_indice = np.argmax(bbox.extent)
            for i in range(3):
                if i not in [min_indice, max_indice]:
                    middle_indice = i
                    break
            shrink_extent_middle = bbox.extent.copy()
            shrink_extent_middle[middle_indice] = min(self.padding, min(bbox.extent))
            shrink_bbox_middle = o3d.geometry.OrientedBoundingBox(bbox.center, bbox.R, shrink_extent_middle)
            within_index_middle = shrink_bbox_middle.get_point_indices_within_bounding_box(pcd.points)

            shrink_bboxes.append(shrink_bbox_middle)
            selected_indices.extend(within_index_middle)

        selected_indices = np.asarray(selected_indices)
        selected_points = np.asarray(pcd.points)[selected_indices]
        geo_points = gpd.GeoSeries(MultiPoint(selected_points)).explode(index_parts=True)
        mask_convex = [point.covered_by(convex) for point in geo_points]
        shrinked_indices = selected_indices[mask_convex]

        return shrinked_indices, bboxes

    def get_parking_label(self):
        polylidar_kwargs = dict(alpha=0.0, lmax=2, min_triangles=0, z_thresh=5, norm_thresh_min=0.1)
        full_points = self.xyz
        min_points = 10
        
        eps = 2
        if (full_points.shape[0] >= 5000000):
            eps = 0.1
        elif (full_points.shape[0] >= 120000):
            eps = 0.2
        # elif (full_points.shape[0] >= 500000):
        #     eps = 0.1
        # elif (full_points.shape[0] >= 1000000):
        #     eps = 0.05
        # elif (full_points.shape[0] >= 2000000):
        #     eps = 0.02
        print("eps = %f" % (eps))
        polygons = []
        pcd, bboxes, labels = get_cluster(self.xyz, eps=eps, min_points=min_points)
        uni_labels = np.unique(labels)
        for l in uni_labels:
            if l == -1:
                continue
            part_points = full_points[labels == l]
            
            try:
                polygon = gpd.GeoSeries(self.get_polygon(part_points, polylidar_kwargs))
            except Exception as e:
                print(e)
                continue

            polygon_shrink = polygon.scale(0.9, 0.9)
            polygons.append(polygon_shrink[0])

        convex_se = gpd.GeoSeries(polygons)
        convex = convex_se.unary_union
        mask, bboxes = self.shrink_cluster_middle_convex(pcd, bboxes, convex, self.label)
        
        return mask


# 48: sidewalk  
class sidewalkLabeler:
    def __init__(self, xyz, sem_mask, label_filter, distance=0.6):
        self.xyz = xyz
        self.sem_mask = sem_mask
        self.label = label_filter
        self.padding = distance

    def get_center_polygon(self, points, polylidar_kwargs):
        points_mat = MatrixDouble(points, copy=True)
        polylidar = Polylidar3D(**polylidar_kwargs)

        _, _, polygons = polylidar.extract_planes_and_polygons(points_mat)
        for poly in polygons:
            poly.holes = []
        sp = convert_to_shapely_polygons(polygons, points, return_first=True, mp=True)
        
        try:
            center_geom = get_centreline(sp)
        except Exception as e:
            print(e)
            return False, False
        
        gdf = gpd.GeoSeries(center_geom.geoms)
        center_poly = gdf.buffer(self.padding).unary_union

        return center_poly, True

    def get_center_points(self, points, center_poly, label):
        geo_points = gpd.GeoSeries(MultiPoint(points)).explode(index_parts=True)
        mask = [point.covered_by(center_poly) for point in geo_points]
        mask = np.array(mask).squeeze()
        mask = np.where(mask == True)
        mask = np.array(mask).squeeze()
        
        return mask, geo_points, center_poly

    def get_sidewalk_label(self):
        polylidar_kwargs = dict(alpha=0.0, lmax=6, min_triangles=0, z_thresh=5, norm_thresh_min=0.1)
        full_points = self.xyz

        eps = 1
        if (full_points.shape[0] >= 5000000):
            eps = 0.1
        elif (full_points.shape[0] >= 120000):
            eps = 0.2
        # elif (full_points.shape[0] >= 500000):
        #     eps = 0.1
        # elif (full_points.shape[0] >= 1000000):
        #     eps = 0.05
        # elif (full_points.shape[0] >= 2000000):
        #     eps = 0.02
        print("eps = %f" % (eps))
        pcd, bboxes, labels = get_cluster(self.xyz, eps=eps, min_points=10)
        print("clustering success")
        
        center_polys = []
        sidewalks = np.unique(labels)
        
        flag = False
        
        # print(sidewalks.shape[0])
        # print(sidewalks)
        for sidewalk in sidewalks:
            if sidewalk == -1:
                continue
            
            part_points = full_points[labels == sidewalk]
            
            if (part_points.shape[0] / full_points.shape[0] < 0.05):
                continue
            
            print(part_points.shape[0])
            
            center_poly, center_flag = self.get_center_polygon(part_points, polylidar_kwargs)
            
            if (center_flag == False):
                continue
            
            center_polys.append(center_poly)
        
        flag = True
        if (flag):
            print("center_polygon success")
        
        print("center polygon的数量为 %0.f" % (len(center_polys)))    
        if (len(center_polys) == 0):
            return np.asarray([])
        
        center_poly_union = gpd.GeoSeries(center_polys).unary_union
        mask, geo_points, center_poly = self.get_center_points(
            full_points, center_poly=center_poly_union, label=self.label)
        print("mask success")
        return mask


# 50: building / 51: fence
class buildingLabeler:
    def __init__(self, xyz, sem_mask, label_filter, distance=0.6):
        self.xyz = xyz
        self.sem_mask = sem_mask
        self.label = label_filter
        self.padding = distance
    
    def seg_plane(self, points):
        rest_cloud = o3d.geometry.PointCloud()
        rest_cloud.points = o3d.utility.Vector3dVector(points)

        segments = []
        segments_index = []

        p = 1
        full_num = len(rest_cloud.points)
        rest_indices = np.array(range(full_num))

        while p > 0.2:
            plane_model, inliers = rest_cloud.segment_plane(distance_threshold=0.3, ransac_n=6, num_iterations=500)
            inlier_cloud = rest_cloud.select_by_index(inliers)
            rest_cloud = rest_cloud.select_by_index(inliers, invert=True)

            p = len(rest_cloud.points) / full_num

            segments.append(inlier_cloud)
            segments_index.append(rest_indices[inliers])

            mask = np.ones(rest_indices.size, dtype=bool)
            mask[inliers] = False
            rest_indices = rest_indices[mask]

        return segments_index

    def shrink_cluster_diagonal(self, pcd, bboxes, label):
        shrink_bboxes = []
        selected_indices = []
        p = self.padding
        
        if (len(bboxes) == 0):
            return np.asarray([])
        
        for bbox in bboxes:
            l, w, h = bbox.extent

            bounding_box = np.array([
                [-l/2,  -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
                [w/2-p,  w/2, -w/2+p, -w/2, w/2-p, w/2, -w/2+p, -w/2],
                [-h/2,  -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])
            eight_points = np.tile(bbox.center, (8, 1))

            corner_box = np.dot(
                bbox.R, bounding_box) + eight_points.transpose() # box平移
            corner_box = corner_box.transpose()

            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                    [4, 5], [5, 6], [6, 7], [4, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7]]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corner_box)
            line_set.lines = o3d.utility.Vector2iVector(lines)

            shrink_bbox = line_set.get_oriented_bounding_box()
            within_index = shrink_bbox.get_point_indices_within_bounding_box(pcd.points)

            shrink_bboxes.append(shrink_bbox)
            selected_indices.extend(within_index)

        bboxes.extend(shrink_bboxes)
        selected_indices = np.asarray(selected_indices)
        
        return selected_indices, bboxes

    def get_building_label(self):
        full_points = self.xyz
        # pcd, bboxes, labels = get_cluster(self.xyz, eps=1, min_points=10)
        
        eps = 1
        if (full_points.shape[0] >= 4000000):
            eps = 0.1
        elif (full_points.shape[0] >= 120000):
            eps = 0.2
        # elif (full_points.shape[0] >= 500000):
        #     eps = 0.1
        # elif (full_points.shape[0] >= 1000000):
        #     eps = 0.05
        # elif (full_points.shape[0] >= 2000000):
        #     eps = 0.02
        print("eps = %f" % (eps))    
        pcd, bboxes, labels = get_cluster(self.xyz, eps=eps, min_points=10)
        
        uni_labels = np.unique(labels)
        for l in uni_labels:
            if l == -1:
                continue
            part_points = full_points[labels == l]
            
            if (part_points.shape[0] <= 500):
                continue
            
            planes_indexes = self.seg_plane(part_points)

            part_label_index = np.argwhere(labels == l)
            max_label = labels.max()
            for i in range(len(planes_indexes)):
                plane_indexes = planes_indexes[i]
                label_indexes = part_label_index[plane_indexes]
                labels[label_indexes] = max_label+(i+1)

            labels[labels == l] = -1

        render_pcd(pcd, labels)
        bboxes = get_bboxes(full_points, labels)
        
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd)
        # for bb in bboxes:
        #     # vis.add_geometry(bb)
        #     line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(bb)
        #     line_set.paint_uniform_color(np.asarray([1,0,0]))
        #     # 在 Visualizer 中绘制 Box
        #     vis.add_geometry(line_set)
        # vis.get_render_option().background_color = np.asarray([1, 1, 1])
        # vis.run()
        # # vis.destroy_window()
        
        mask, bboxes = self.shrink_cluster_diagonal(pcd, bboxes, self.label)
        
        return mask           


# 70: vegetation / 81: traffic-sign   
class vegetationLabeler:
    def __init__(self, xyz, sem_mask, label_filter, distance=0.6):
        self.xyz = xyz
        self.sem_mask = sem_mask
        self.label = label_filter
        self.padding = distance
    
    def get_vegetation_label(self, min_points=10):
        points = self.xyz
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        eps = 1
        if (points.shape[0] >= 4500000):
            eps = 0.1
        elif (points.shape[0] >= 120000):
            eps = 0.2
        # elif (points.shape[0] >= 500000):
        #     eps = 0.1
        # elif (points.shape[0] >= 1000000):
        #     eps = 0.05
        # elif (points.shape[0] >= 2000000):
        #     eps = 0.02
        print("eps = %f" % (eps))
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
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
            
            if (points[labels == l].shape[0] <= 100):
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


# 71: trunk / 80: pole       
class trunkLabeler:
    def __init__(self, xyz, sem_mask, label_filter, distance=0.6):
        self.xyz = xyz
        self.sem_mask = sem_mask
        self.label = label_filter
        self.padding = distance
        
    def shrink_cluster_middle_min(self, pcd, bboxes, label):
        shrink_bboxes = []
        selected_indices = []
        for bbox in bboxes:
            min_indice = np.argmin(bbox.extent)
            max_indice = np.argmax(bbox.extent)
            for i in range(3):
                if i not in [min_indice, max_indice]:
                    middle_indice = i
                    break
            shrink_extent_middle = bbox.extent.copy()
            shrink_extent_middle[middle_indice] = min(self.padding, min(bbox.extent))
            shrink_bbox_middle = o3d.geometry.OrientedBoundingBox(bbox.center, bbox.R, shrink_extent_middle)
            within_index_middle = shrink_bbox_middle.get_point_indices_within_bounding_box(pcd.points)

            shrink_extent_min = bbox.extent.copy()
            shrink_extent_min[min_indice] = min(self.padding, min(bbox.extent))
            shrink_bbox_min = o3d.geometry.OrientedBoundingBox(bbox.center, bbox.R, shrink_extent_min)
            within_index_min = shrink_bbox_min.get_point_indices_within_bounding_box(pcd.points)

            if len(within_index_middle) > len(within_index_min):
                within_index = within_index_middle
                shrink_bbox = shrink_bbox_middle
            else:
                within_index = within_index_min
                shrink_bbox = shrink_bbox_min

            shrink_bboxes.append(shrink_bbox)
            selected_indices.extend(within_index)

        bboxes.extend(shrink_bboxes)
        selected_indices = np.asarray(selected_indices)
        
        return selected_indices, bboxes

    def get_trunk_label(self):
        full_points = self.xyz
        min_points = 5
        
        eps = 1
        if (full_points.shape[0] >= 4000000):
            eps = 0.1
        elif (full_points.shape[0] >= 120000):
            eps = 0.2
        # elif (full_points.shape[0] >= 500000):
        #     eps = 0.1
        # elif (full_points.shape[0] >= 1000000):
        #     eps = 0.05
        # elif (full_points.shape[0] >= 2000000):
        #     eps = 0.02
        print("eps = %f" % (eps))
        pcd, bboxes, _ = get_cluster(self.xyz, eps=eps, min_points=min_points)
        mask, bboxes = self.shrink_cluster_middle_min(pcd, bboxes, self.label)
        
        return mask
