import argparse
import os

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from labeler import AutoLabeler

BASE_DIR = '/Users/chenguang.gao/Desktop/Dataset/kitti'
SEQ = '10'

def get_scribble_mask_by_teil(all_xyz, all_sem_label, all_inst_label):
    class_can_label = [40, 44, 48, 50, 51, 70, 71, 80, 81, 10, 11, 13, 15, 18, 20, 30, 31, 32] # 18 classes
    class_distance = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1]
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type', '-t',
        type=int,
        default=10,
        required=False,
        help='Labels. Defaults to %(default)s',
    )
    parser.add_argument(
        '--distance', '-ds',
        type=float,
        default=0.2,
        required=False,
        help='Width of label line. Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    config = yaml.safe_load(open(os.path.join(BASE_DIR, 'semantic-kitti.yaml'), 'r'))
    
    scribble_mask = []
    
    for i, label in enumerate(class_can_label):
        FLAGS.type = label
        FLAGS.distance = class_distance[i]
        print("--- " + config['labels'][FLAGS.type] + " ---")
        if FLAGS.type in [40, 44, 48, 50, 51, 70, 71, 80, 81]:
            # get semantic mask and point that belong to the given type
            sem_mask = np.where(all_sem_label == FLAGS.type)
            sem_mask = np.array(sem_mask).squeeze()
            sem_mask = sem_mask.flatten()

            single_class_xyz = all_xyz[sem_mask]
            print(single_class_xyz.shape[0])
            
            # if (FLAGS.type == 48 & single_class_xyz.shape[0] >= 50000):
            #     continue
            
            # if (FLAGS.type != 70):
            #     continue
            
            if (single_class_xyz.shape[0] < 10):
                continue
            
            autolabeler = AutoLabeler(xyz=single_class_xyz, sem_mask=sem_mask, label_filter=FLAGS.type, distance=FLAGS.distance)
            # get contact label according to filtered points all_xyz[sem_mask]
            concat_label_mask = autolabeler.labeler()
            print(concat_label_mask)
            # get concat label according to teil
            if (concat_label_mask.shape[0] == 0):
                continue
            concat_label_mask_wrt_teil = sem_mask[concat_label_mask]
            print(concat_label_mask_wrt_teil)
            concat_label_mask_wrt_teil = concat_label_mask_wrt_teil.flatten()
            if (concat_label_mask_wrt_teil.shape[0] != 0):
                scribble_mask.append(concat_label_mask_wrt_teil)
        
        elif FLAGS.type in [10, 11, 13, 15, 18, 20]:
            all_concat_car_label = []
            sem_mask = np.where(all_sem_label == FLAGS.type)
            sem_mask = np.array(sem_mask).squeeze()
            sem_mask = sem_mask.flatten()
            all_car_inst_label = all_inst_label[sem_mask]
            
            if (all_car_inst_label.shape[0] < 100):
                continue
            
            all_car_inst_id = list(set(all_car_inst_label))
            
            autocarlabeler = AutoLabeler(xyz=all_xyz, sem_mask=all_sem_label, label_filter=FLAGS.type, distance=FLAGS.distance)
            for car in all_car_inst_id:
                single_car_inst_mask = np.where(all_car_inst_label == car)
                single_car_inst_mask = np.array(single_car_inst_mask).squeeze()
                single_car_inst_mask = single_car_inst_mask.flatten()
                
                single_car_sem_mask = sem_mask[single_car_inst_mask]
                single_car_xyz = all_xyz[single_car_sem_mask]
                
                # if car instances points number is less than 40, we do not label it
                if (single_car_xyz.shape[0] < 40):
                    continue;
                single_car_label = autocarlabeler.get_car_label(single_car_xyz)
                
                if isinstance(single_car_label, np.int64):
                    single_car_label = np.array([single_car_label])
                if (single_car_label.shape[0] != 0):
                    single_car_label_towards_full_pcd = single_car_sem_mask[single_car_label]
                    all_concat_car_label.append(single_car_label_towards_full_pcd)
            
            # get all car label according to the whole contact given frame
            if len(all_concat_car_label) == 0:
                continue;
            all_concat_car_label = np.concatenate(all_concat_car_label)
            scribble_mask.append(all_concat_car_label)     
            
    if (len(scribble_mask) == 0):
        return np.array([])       
    scribble_mask = np.concatenate(scribble_mask)
    scribble_mask = np.squeeze(scribble_mask)
    scribble_mask = scribble_mask.flatten()
    
    return scribble_mask
