import vispy
from vispy.scene import visuals, SceneCanvas
import os
import numpy as np
from pykitti import odometry
import pickle
import yaml
from tqdm import tqdm

BASE_DIR = '/media/ozan/hdd_backup/dataset/semantickitti'
SEQ =  '00'

class SemanticKITTI(odometry):
   @staticmethod
   def map_label(label, map_dict):
       maxkey = 0
       for key, data in map_dict.items():
           if isinstance(data, list):
               nel = len(data)
           else:
               nel = 1
           if key > maxkey:
               maxkey = key
       if nel > 1:
           lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
       else:
           lut = np.zeros((maxkey + 100), dtype=np.int32)
       for key, data in map_dict.items():
           try:
               lut[key] = data
           except IndexError:
               print("Wrong key ", key)
       return lut[label]

   def get_semantic_label(self, idx, learning_map=None):
       filename = self.velo_files[idx].replace('velodyne', 'labels').replace('.bin', '.label')
       label = np.fromfile(filename, dtype=np.int32)
       label = label.reshape((-1)) & 0xFFFF
       return self.map_label(label, learning_map) if learning_map is not None else label

   def get_velo_pose(self, idx):
       pose_ = np.matmul(self.poses[idx], self.calib.T_cam0_velo)
       Tr_inv = np.linalg.inv(self.calib.T_cam0_velo)
       return np.matmul(Tr_inv, pose_)

   def get_aligned_velo(self, idx, align_idx=0):
       velo = self.get_velo(idx)
       velo[:,3] = 1
       pose = self.get_velo_pose(idx)
       align_pose = self.poses[align_idx]
       diff_pose = np.matmul(np.linalg.inv(align_pose), pose)
       return np.matmul(diff_pose, velo.T).T

class Visualizer():
   def __init__(self):
       self.canvas = SceneCanvas(keys='interactive', show=True, bgcolor='white')
       self.canvas.events.key_press.connect(self.key_press)
       self.canvas.events.draw.connect(self.draw)
       self.grid = self.canvas.central_widget.add_grid()
       self.view = vispy.scene.widgets.ViewBox(parent=self.canvas.scene) # border_color='white',
       self.grid.add_widget(self.view, 0, 0)

       # Point Cloud Visualizer
       self.sem_vis = visuals.Markers()
       self.sem_vis.antialias = 0
       self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
       self.view.add(self.sem_vis)
       visuals.XYZAxis(parent=self.view.scene)

   def update(self, points, colors):
       self.sem_vis.set_data(points, face_color=colors, edge_color=colors, size=3)

   def draw(self, event):
       if self.canvas.events.key_press.blocked():
           self.canvas.events.key_press.unblock()

   def key_press(self, event):
       self.canvas.events.key_press.block()
       if event.key == 'Q' or event.key == 'Escape':
           self.destroy()


def main():
   ds = SemanticKITTI(BASE_DIR, SEQ)

   config = yaml.safe_load(open(os.path.join(BASE_DIR, 'semantic-kitti.yaml'), 'r'))
   num_classes = len(config['learning_map_inv'])
   color_map = np.zeros((num_classes, 3))
   for i in range(num_classes):
       color_idx = config['learning_map_inv'][i]
       color_map[i,:] = np.array(config['color_map'][color_idx][::-1])

   # Load data
   all_xyz = []
   all_color = []

   for i in tqdm(range(50,51)):
       xyz = ds.get_aligned_velo(i)[:,:3]
       label = ds.get_semantic_label(i, config['learning_map'])
       color = color_map[label.astype(np.int32)]/255

       all_xyz.append(xyz)
       all_color.append(color)

   visualizer = Visualizer()
   visualizer.update(np.concatenate(all_xyz), np.concatenate(all_color))
   vispy.app.run()

if __name__ == '__main__':
   main()