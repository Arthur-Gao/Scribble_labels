import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
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
import open3d as o3d


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