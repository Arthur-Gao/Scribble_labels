import argparse
import os

import numpy as np
import vispy
import yaml
from tqdm import tqdm

from vis import SemanticKITTI, Visualizer
from labeler import AutoLabeler

BASE_DIR = '/Users/chenguang.gao/Desktop/Dataset/kitti'
SEQ =  '00'

label_color = np.array([0.0, 0.0, 0.0])

parser = argparse.ArgumentParser("./main.py")
parser.add_argument(
    '--type', '-t',
    type=int,
    default=40,
    required=False,
    help='40, 70, 81 -> "road", "vege", "sign" Defaults to %(default)s',
)
parser.add_argument(
    '--distance', '-ds',
    type=float,
    default=0.2,
    required=False,
    help='Width of road. Defaults to %(default)s',
)
FLAGS, unparsed = parser.parse_known_args()

if __name__ == '__main__':
    ds = SemanticKITTI(BASE_DIR, SEQ)

    config = yaml.safe_load(open(os.path.join(BASE_DIR, 'semantic-kitti.yaml'), 'r'))
    num_classes = len(config['learning_map_inv'])
    color_map = np.zeros((num_classes, 3))
    for i in range(num_classes):
        color_idx = config['learning_map_inv'][i]
        color_map[i,:] = np.array(config['color_map'][color_idx][::-1]) # transfer color bgr to rgb

    # Load data
    all_xyz = []
    all_sem_label = []
    all_color = []

    for i in tqdm(range(0,1)):
        xyz = ds.get_aligned_velo(i)[:,:3]
        color_label, sem_label = ds.get_semantic_label_1(i, config['learning_map'])
        color = color_map[color_label.astype(np.int32)]/255

        all_xyz.append(xyz)
        all_sem_label.append(sem_label)
        all_color.append(color)
        
    all_xyz = np.concatenate(all_xyz)
    all_sem_label = np.concatenate(all_sem_label)
    all_color = np.concatenate(all_color)

    sem_mask = np.where(all_sem_label == FLAGS.type)
    sem_mask = np.array(sem_mask).squeeze()
    all_xyz = all_xyz[sem_mask]
    all_color = all_color[sem_mask]
    
    autolabeler = AutoLabeler(xyz=all_xyz, sem_mask=sem_mask, label_filter=FLAGS.type, distance=FLAGS.distance)
    label_mask = autolabeler.labeler()
    all_color[label_mask] = label_color
    
    visualizer = Visualizer()
    visualizer.update(all_xyz, all_color)
    visualizer.run()
