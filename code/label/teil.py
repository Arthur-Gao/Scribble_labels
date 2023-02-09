import argparse
import os
import pathlib
import pdb

import numpy as np
import yaml
from scipy import ndimage
from tqdm import tqdm

from kitti import SemanticKITTI
from scribble_mask_teil import get_scribble_mask_by_teil

BASE_DIR = '/Users/chenguang.gao/Desktop/Dataset/kitti'
SEQ = '04'
save_name = 'scribbles'

scribble_label_per_frame = []

def concat_labels_separation(self, frame_idx, points_num, label_mask, files_num):
    total_points_num = 0
    for i in range(files_num):
        frame_mask = np.where((label_mask >= total_points_num) & (label_mask < total_points_num + points_num[i]))
        frame_mask = np.asarray(frame_mask).squeeze()
        frame_mask = frame_mask.flatten()
        
        frame_mask -= total_points_num
        scribble_label_per_frame[i].append(frame_mask)
        
        total_points_num += points_num[i]


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config_path', default='/Users/chenguang.gao/Desktop/Dataset/kitti/config/semantic-kitti.yaml')
    parser.add_argument('--save_dir', default='/Users/chenguang.gao/Desktop/Dataset/kitti/sequences')
    args = parser.parse_args()
    
    ds = SemanticKITTI(BASE_DIR, SEQ)
    config = yaml.safe_load(open(args.dataset_config_path, 'r'))
    num_classes = len(config['learning_map_inv'])
    
    files_num = ds.get_total_velo_files_num()
    print("There are total %.0f files." % files_num)
    
    all_xyz = []
    all_sem_label = []
    all_inst_label = []
    
    frame_idx = []
    points_num = []
    
    for i in tqdm(range(files_num)):
        xyz = ds.get_aligned_velo(i)[:,:3]
        color_label, sem_label = ds.get_semantic_label(i, config['learning_map'])
        ins_label = ds.get_instance_label(i)

        all_xyz.append(xyz)
        all_sem_label.append(sem_label)
        all_inst_label.append(ins_label)
        
        frame_idx.append(i)
        points_num.append(xyz.shape[0])
        
        scribble_label_per_frame.append([])
        
    all_xyz = np.concatenate(all_xyz)
    all_sem_label = np.concatenate(all_sem_label)
    all_inst_label = np.concatenate(all_inst_label)
    
    min_x = np.min(all_xyz[:,0])
    max_x = np.max(all_xyz[:,0])
    min_y = np.min(all_xyz[:,1])
    max_y = np.max(all_xyz[:,1])
    
    teil_scope = 50.0
    
    teil_num_x = np.ceil((max_x - min_x) / teil_scope)
    teil_num_x = int(teil_num_x)
    teil_num_y = np.ceil((max_y - min_y) / teil_scope)
    teil_num_y = int(teil_num_y)
    print(teil_num_x)
    print(teil_num_y)
    
    all_teil_concat_scribble_mask = []
    
    for i in range(teil_num_x):
        teil_min_x = min_x + i * teil_scope
        teil_max_x = min_x + (i + 1) * teil_scope
        if (max_x < teil_max_x):
            teil_max_x = max_x
        print("------x")
        for j in range(teil_num_y):
            teil_min_y = min_y + j * teil_scope
            teil_max_y = min_y + (j + 1) * teil_scope
            if (max_y < teil_max_y):
                teil_max_y = max_y
            
            teil_mask = np.where((all_xyz[:,0] >= teil_min_x) & (all_xyz[:,0] < teil_max_x) & 
                                 (all_xyz[:,1] >= teil_min_y) & (all_xyz[:,1] < teil_max_y))
            teil_mask = np.asarray(teil_mask)[0]
            # print(teil_mask)
            print(teil_mask.shape)
            
            if (teil_mask.shape[0] <= 100):
                continue;
            
            teil_xyz = all_xyz[teil_mask]
            teil_sem_label = all_sem_label[teil_mask]
            teil_inst_label = all_inst_label[teil_mask]
            
            teil_concat_scribble_mask = get_scribble_mask_by_teil(teil_xyz, teil_sem_label, teil_inst_label)
            if (teil_concat_scribble_mask.shape[0] == 0):
                print("----y")
                continue
            teil_concat_scribble_mask = teil_mask[teil_concat_scribble_mask]
            
            all_teil_concat_scribble_mask.append(teil_concat_scribble_mask)
            print("++++y")
            
    all_teil_concat_scribble_mask = np.concatenate(all_teil_concat_scribble_mask)
    all_teil_concat_scribble_mask = np.sort(all_teil_concat_scribble_mask)
    
    concat_labels_separation(frame_idx, points_num, all_teil_concat_scribble_mask, files_num)
    
    for i in tqdm(range(files_num)):
        # pdb.set_trace()
        true_label = ds.get_true_label(i)
        point_num = true_label.shape[0]
        
        frame = str(i)
        if len(frame) != 6:
            lost_num = 6 - len(frame)
            for k in range(lost_num):
                frame = "0" + frame
        print("********** " + "Get scribble label for velo " + frame + " **********") 
        
        scribble_label_mask = np.concatenate(scribble_label_per_frame[i])
        no_scribble_label_mask = np.array([k for k in range(point_num)])
        no_scribble_label_mask = np.delete(no_scribble_label_mask, scribble_label_mask)
        scribble_label = np.copy(true_label)
        scribble_label[no_scribble_label_mask] = 0
        
        save_dir = os.path.join(args.save_dir, SEQ, save_name)
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, frame + '.label')
        scribble_label.tofile(save_path)
        print("********** " + "Successfully save scribble label for velo " + frame + " **********")  
            