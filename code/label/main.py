import argparse
import os
import pathlib

import numpy as np
import yaml
from scipy import ndimage
from tqdm import tqdm

from kitti import SemanticKITTI
import pdb

BASE_DIR = '/Users/chenguang.gao/Desktop/Dataset/kitti'
SEQ = '00'
save_name = 'scribbles'

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
    
    for i in tqdm(range(files_num)):
        # pdb.set_trace()
        # true_label = ds.get_true_label(i)
        true_label = ds.get_true_label(2401)
        point_num = true_label.shape[0]
        
        # frame = str(i)   
        frame = str(2401)
        if len(frame) != 6:
            lost_num = 6 - len(frame)
            for k in range(lost_num):
                frame = "0" + frame
        print("********** " + "Get scribble label for velo " + frame + " **********") 
        
        # scribble_label_mask = ds.get_scribble_label_mask(i, ds)
        scribble_label_mask = ds.get_scribble_label_mask(2401, ds)
        no_scribble_label_mask = np.array([k for k in range(point_num)])
        no_scribble_label_mask = np.delete(no_scribble_label_mask, scribble_label_mask)
        scribble_label = np.copy(true_label)
        scribble_label[no_scribble_label_mask] = 0
        
        save_dir = os.path.join(args.save_dir, SEQ, save_name)
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, frame + '.label')
        scribble_label.tofile(save_path)
        print("********** " + "Successfully save scribble label for velo " + frame + " **********") 
        