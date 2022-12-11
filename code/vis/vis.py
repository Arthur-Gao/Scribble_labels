import numpy as np
import vispy
from pykitti import odometry
from vispy.scene import SceneCanvas, visuals

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

   def get_semantic_label_1(self, idx, learning_map=None): # 得到每一个点的label
       filename = self.velo_files[idx].replace('velodyne', 'labels').replace('.bin', '.label')
       label = np.fromfile(filename, dtype=np.int32)
       sem_label = label.reshape((-1)) & 0xFFFF # semantic label in lower half
       return self.map_label(sem_label, learning_map) if learning_map is not None else sem_label, sem_label
   
   def get_semantic_label_2(self, idx, learning_map=None):
       filename = self.velo_files[idx].replace('velodyne', 'labels').replace('.bin', '.label')
       label = np.fromfile(filename, dtype=np.int32)
       sem_label = label.reshape((-1)) & 0xFFFF # semantic label in lower half
       return sem_label
   
   def get_instance_label(self, label_path, label_map=None):
       label = np.fromfile(label_path, dtype=np.int32)
       inst_label = label >> 16 # instance id in upper half
       return inst_label

   def get_velo_pose(self, idx):
       pose_ = np.matmul(self.poses[idx], self.calib.T_cam0_velo) # 从velo到cam0
       Tr_inv = np.linalg.inv(self.calib.T_cam0_velo)
       return np.matmul(Tr_inv, pose_)
   
   def get_aligned_center(self, idx, align_idx=0):
       center = np.array([0,0,0,1]).T
       pose = self.get_velo_pose(idx)
       align_pose = self.poses[align_idx]
       diff_pose = np.matmul(np.linalg.inv(align_pose), pose)
       center = np.matmul(diff_pose, center.T).T
       return center

   def get_aligned_velo(self, idx, align_idx=0): # 得到对齐的velo points
       velo = self.get_velo(idx)
       velo[:,3] = 1 # 反射率都设为1
       pose = self.get_velo_pose(idx) # 得到第idx帧velo的pose
       align_pose = self.poses[align_idx] # 默认为第0帧的基准位置
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

    def destroy(self):
        self.canvas.close()
        vispy.app.quit()
    
    def run(self):
        vispy.app.run()