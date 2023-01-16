import argparse
import os

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from labeler import AutoLabeler

BASE_DIR = '/Users/chenguang.gao/Desktop/Dataset/kitti'
SEQ = '00'

def scribble_mask(start_idx, ds):
    class_can_label = [40, 44, 48, 50, 51, 70, 71, 80, 81, 10, 11, 13, 15, 18, 20, 30, 31, 32] # 15 classes
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
    parser.add_argument(
        '--startidx', '-idx',
        type=int,
        default=0,
        required=False,
        help='starting index from list. Defaults to %(default)s',
    )
    parser.add_argument(
        '--searchlen', '-l',
        type=int,
        default=2,
        required=False,
        help='search length on either direction. Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()
    
    # ds = SemanticKITTI(BASE_DIR, SEQ)

    config = yaml.safe_load(open(os.path.join(BASE_DIR, 'semantic-kitti.yaml'), 'r'))
    num_classes = len(config['learning_map_inv'])
    color_map = np.zeros((num_classes, 3))
    for i in range(num_classes):
        color_idx = config['learning_map_inv'][i]
        color_map[i,:] = np.array(config['color_map'][color_idx][::-1]) # transfer color bgr to rgb
        
    FLAGS.startidx = start_idx
    first_idx = max(0, FLAGS.startidx - FLAGS.searchlen + 1)
    files_num = ds.get_total_velo_files_num()
    last_idx = min(files_num - 1, FLAGS.startidx + FLAGS.searchlen - 1)
    search_idx = list(range(first_idx, last_idx + 1))
    print("Search frames [ %.0f, %.0f ]" % (first_idx, last_idx))
    
    # Load data
    all_xyz = []
    all_sem_label = []
    all_inst_label = []
    all_color = []
    
    frame_idx = []
    points_num = []
    
    for i in tqdm(search_idx):
        xyz = ds.get_aligned_velo(i)[:,:3]
        color_label, sem_label = ds.get_semantic_label(i, config['learning_map'])
        ins_label = ds.get_instance_label(i)
        color = color_map[color_label.astype(np.int32)]/255

        all_xyz.append(xyz)
        all_sem_label.append(sem_label)
        all_inst_label.append(ins_label)
        all_color.append(color)
        
        frame_idx.append(i)
        points_num.append(xyz.shape[0])
        
    all_xyz = np.concatenate(all_xyz)
    all_sem_label = np.concatenate(all_sem_label)
    all_inst_label = np.concatenate(all_inst_label)
    all_color = np.concatenate(all_color)
    
    frame_idx = np.array(frame_idx)
    points_num = np.array(points_num)
    
    '''
    Only car, bicycle, motorcycle, other_vehicle,
         person, moving-person, moving-bicyclist, moving-motorcyclist
    have instance label. (i.e. car class and person class)
    '''
    # get all instances in the given frames
    # all_inst_id = ds.get_all_inst_id(all_sem_label, all_inst_label, config['labels'])
    # print(all_inst_id)
    
    scribble_mask = []
    
    for i, label in enumerate(class_can_label):
        FLAGS.type = label
        FLAGS.distance = class_distance[i]
        print("--- " + config['labels'][FLAGS.type] + " ---")
        if FLAGS.type in [40, 44, 48, 50, 51, 70, 71, 80, 81]:
            # get semantic mask and point that belong to the given type
            sem_mask = np.where(all_sem_label == FLAGS.type)
            sem_mask = np.array(sem_mask).squeeze()
            single_class_xyz = all_xyz[sem_mask]
            single_class_color = all_color[sem_mask]
            
            if (single_class_xyz.shape[0] < 10):
                continue
            
            autolabeler = AutoLabeler(xyz=single_class_xyz, sem_mask=sem_mask, label_filter=FLAGS.type, distance=FLAGS.distance)
            # get contact label according to filtered points all_xyz[sem_mask]
            concat_label_mask = autolabeler.labeler()
            # get separate label according to each frame
            separate_label_mask = autolabeler.concat_labels_separation(frame_idx=frame_idx, points_num=points_num, label_mask=concat_label_mask)
            
            # save label indices of start_idx
            assert len(separate_label_mask) == len(frame_idx)
            selected_mask = separate_label_mask[start_idx - first_idx].flatten()
            if (selected_mask.shape[0] != 0):
                scribble_mask.append(selected_mask)
            print(selected_mask.shape)
            print(selected_mask)
        
        elif FLAGS.type in [10, 11, 13, 15, 18, 20]:
            all_concat_car_label = []
            sem_mask = np.where(all_sem_label == FLAGS.type)
            sem_mask = np.array(sem_mask).squeeze()
            all_car_inst_label = all_inst_label[sem_mask]
            
            if (all_car_inst_label.shape[0] == 0):
                continue
            
            all_car_inst_id = ds.get_all_car_inst_id(all_car_inst_label)
            
            autocarlabeler = AutoLabeler(xyz=all_xyz, sem_mask=all_sem_label, label_filter=FLAGS.type, distance=FLAGS.distance)
            for car in all_car_inst_id:
                single_car_inst_mask = np.where(all_car_inst_label == car)
                single_car_inst_mask = np.array(single_car_inst_mask).squeeze()
                
                single_car_sem_mask = sem_mask[single_car_inst_mask]
                single_car_xyz = all_xyz[single_car_sem_mask]
                
                # if car instances points number is less than 100, we do not label it
                if (single_car_xyz.shape[0] < 100):
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
            # get separate car label according to each frame        
            all_separate_car_label = autocarlabeler.concat_car_labels_separation(frame_idx=frame_idx, points_num=points_num, label_mask=all_concat_car_label)
            
            # save car label indices of start_idx
            assert len(all_separate_car_label) == len(frame_idx)
            selected_mask = all_separate_car_label[start_idx - first_idx].flatten()
            if (selected_mask.shape[0] != 0):
                scribble_mask.append(selected_mask)
            print(selected_mask.shape)
            print(selected_mask)
            
    scribble_mask = np.concatenate(scribble_mask)
    scribble_mask = np.squeeze(scribble_mask)
    scribble_mask = scribble_mask.flatten()
    
    return scribble_mask
