import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import os
import yaml
import open3d as o3d

class Visualizer():
    def __init__(self):
        config =  yaml.safe_load(open('semantickitti.yaml', 'r'))
        max_key = [*config['learning_map']][-1]
        self.learning_map = np.zeros((max_key+1,))
        for k, v in config['learning_map'].items():
            self.learning_map[k] = v
        num_classes = len(config['learning_map_inv'])
        self.color_map = np.zeros((num_classes, 3))
        for i in range(num_classes):
            i_ = config['learning_map_inv'][i]
            self.color_map[i,:] = np.array(config['color_map'][i_][::-1])

        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)
        self.vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.vis)
        visuals.XYZAxis(parent=self.view.scene)

        self.point_size = 3

    def update(self, points, label=None):
        if points.shape[1] > 3:
            points = points[:,:3]
        if label is not None:
            label_ = self.learning_map[label].astype(int)
            color = self.color_map[label_]/255
            self.vis.set_data(points,
                              face_color=color,
                              edge_color=color,
                              size=self.point_size)
        else:
            self.vis.set_data(points, size=self.point_size)

    def show(self):
        vispy.app.run()

def center_velo(velo):
    return velo - velo.mean(0)

if __name__ == '__main__':
    vis = Visualizer()

    from dataset import AlignedKITTI
    ds = AlignedKITTI('data', '04')
    idx = [30]
    velo = ds.get_aligned_velos(idx)
    velo = center_velo(velo)
    label = ds.get_labels(idx)[0]
    vis.update(velo, label)
    vispy.app.run()
