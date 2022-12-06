#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
# from mpl_toolkits import mplot3d
from laserscan import LaserScan, SemLaserScan

import geopandas as gpd
import time
from polylidar import MatrixDouble, Polylidar3D
from polylidar.polylidarutil import (generate_test_points, plot_triangles, get_estimated_lmax,
                                     plot_triangle_meshes, get_triangles_from_list, get_colored_planar_segments, plot_polygons,convert_to_shapely_polygons)

from polylidar.polylidarutil import (plot_polygons_3d, generate_3d_plane, set_axes_equal, plot_planes_3d,
                                     scale_points, rotation_matrix, apply_rotation)
import matplotlib.pyplot as plt
from centerline.geometry import Centerline

import open3d as o3d
# from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN

class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, scan, scan_names, label_names, offset=0,
               semantics=True, instances=False):
    self.scan = scan
    self.scan_names = scan_names
    self.label_names = label_names
    self.offset = offset
    self.total = len(self.scan_names)
    self.semantics = semantics
    self.instances = instances
    # sanity check
    if not self.semantics and self.instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError

    self.reset()
    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True)
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    # grid
    self.grid = self.canvas.central_widget.add_grid()

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)
    # add semantics
    if self.semantics:
      print("Using semantics in visualizer")
      self.sem_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.sem_view, 0, 1)
      self.sem_vis = visuals.Markers()
      self.sem_view.camera = 'turntable'
      self.sem_view.add(self.sem_vis)
      visuals.XYZAxis(parent=self.sem_view.scene)
      # self.sem_view.camera.link(self.scan_view.camera)

    if self.instances:
      print("Using instances in visualizer")
      self.inst_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.inst_view, 0, 2)
      self.inst_vis = visuals.Markers()
      self.inst_view.camera = 'turntable'
      self.inst_view.add(self.inst_vis)
      visuals.XYZAxis(parent=self.inst_view.scene)
      # self.inst_view.camera.link(self.scan_view.camera)

    # img canvas size
    self.multiplier = 1
    self.canvas_W = 1024
    self.canvas_H = 64
    if self.semantics:
      self.multiplier += 1
    if self.instances:
      self.multiplier += 1

    # new canvas for img
    self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                  size=(self.canvas_W, self.canvas_H * self.multiplier))
    # grid
    self.img_grid = self.img_canvas.central_widget.add_grid()
    # interface (n next, b back, q quit, very simple)
    self.img_canvas.events.key_press.connect(self.key_press)
    self.img_canvas.events.draw.connect(self.draw)

    # add a view for the depth
    self.img_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.img_canvas.scene)
    self.img_grid.add_widget(self.img_view, 0, 0)
    self.img_vis = visuals.Image(cmap='viridis')
    self.img_view.add(self.img_vis)

    # add semantics
    if self.semantics:
      self.sem_img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.sem_img_view, 1, 0)
      self.sem_img_vis = visuals.Image(cmap='viridis')
      self.sem_img_view.add(self.sem_img_vis)

    # add instances
    if self.instances:
      self.inst_img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.inst_img_view, 2, 0)
      self.inst_img_vis = visuals.Image(cmap='viridis')
      self.inst_img_view.add(self.inst_img_vis)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def get_polygon(self):
    points = self.scan.points
    points_mat = MatrixDouble(points, copy=True)
    polylidar_kwargs = dict(alpha=0.0, lmax=6, min_triangles=0, z_thresh=5, norm_thresh_min=0.1)
    polylidar = Polylidar3D(**polylidar_kwargs)

    t1 = time.time()
    mesh, planes, polygons = polylidar.extract_planes_and_polygons(points_mat)
    for poly in polygons:
      poly.holes = []
    sp = convert_to_shapely_polygons(polygons, points, return_first=True, mp=True)
    geom = self.get_centre(sp)
    gdf = gpd.GeoSeries(geom.geoms)
    gdf.plot()

    t2 = time.time()
    print("Took {:.2f} milliseconds".format((t2 - t1) * 1000))
    print("Should see two planes extracted, please rotate.")

    triangles = np.asarray(mesh.triangles)
    fig, ax = plt.subplots(nrows=1, ncols=1,
                          subplot_kw=dict(projection='3d'))
    # plot all triangles
    # plot_planes_3d(points, triangles, planes, ax)
    plot_polygons_3d(points, polygons, ax)
    # plot points
    ax.scatter(*scale_points(points), c='k', s=0.1)
    set_axes_equal(ax)
    ax.view_init(elev=15., azim=-35)
    plt.show()

  def get_centre(self, polygon):
    attributes = {"id": 1, "name": "polygon", "valid": True}
    centerline = Centerline(polygon, **attributes)
    return centerline

  def get_cluster(self, min_points=10):    
    points = self.scan.points
    # kmeans = KMeans(n_clusters=1).fit(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(pcd.cluster_dbscan(eps=1, min_points=min_points))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label 
    if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    bboxes = []

    for l in np.unique(labels):
      if l == -1:
        continue
      p = o3d.geometry.PointCloud()
      p.points = o3d.utility.Vector3dVector(points[labels==l])
      bbox = o3d.geometry.OrientedBoundingBox().create_from_points(p.points)
      bboxes.append(bbox)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for bb in bboxes:
      vis.add_geometry(bb)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()

  def update_scan(self):
    # first open data
    if self.semantics:
      if self.scan.label_filter is not None:
        self.scan.open_scan(self.scan_names[self.offset])
        self.scan.open_label(self.label_names[self.offset])
        self.scan.filter()
      if self.scan.label_filter[0] == 40:
        self.get_polygon()
      elif self.scan.label_filter[0] in [70, 81]:
        min_points = 3 if self.scan.label_filter[0] == 81 else 10
        self.get_cluster(min_points)
      
      self.scan.colorize()

    else:
      self.scan.open_scan(self.scan_names[self.offset])

    # then change names
    title = "scan " + str(self.offset)
    self.canvas.title = title
    self.img_canvas.title = title

    # then do all the point cloud stuff

    # plot scan
    power = 16
    # print()
    range_data = np.copy(self.scan.unproj_range)
    # print(range_data.max(), range_data.min())
    range_data = range_data**(1 / power)
    # print(range_data.max(), range_data.min())
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = self.get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]
    self.scan_vis.set_data(self.scan.points,
                           face_color=viridis_colors[..., ::-1],
                           edge_color=viridis_colors[..., ::-1],
                           size=1)

    # plot semantics
    if self.semantics:
      self.sem_vis.set_data(self.scan.points,
                            face_color=self.scan.sem_label_color[..., ::-1],
                            edge_color=self.scan.sem_label_color[..., ::-1],
                            size=1)

    # plot instances
    if self.instances:
      self.inst_vis.set_data(self.scan.points,
                             face_color=self.scan.inst_label_color[..., ::-1],
                             edge_color=self.scan.inst_label_color[..., ::-1],
                             size=1)

    # now do all the range image stuff
    # plot range image
    data = np.copy(self.scan.proj_range)
    # print(data[data > 0].max(), data[data > 0].min())
    data[data > 0] = data[data > 0]**(1 / power)
    data[data < 0] = data[data > 0].min()
    # print(data.max(), data.min())
    data = (data - data[data > 0].min()) / \
        (data.max() - data[data > 0].min())
    # print(data.max(), data.min())
    self.img_vis.set_data(data)
    self.img_vis.update()

    if self.semantics:
      self.sem_img_vis.set_data(self.scan.proj_sem_color[..., ::-1])
      self.sem_img_vis.update()

    if self.instances:
      self.inst_img_vis.set_data(self.scan.proj_inst_color[..., ::-1])
      self.inst_img_vis.update()

  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    self.img_canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      if self.offset >= self.total:
        self.offset = 0
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      if self.offset < 0:
        self.offset = self.total - 1
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    self.img_canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()
