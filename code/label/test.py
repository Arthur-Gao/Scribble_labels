import argparse
import os
import pathlib
import pdb

import numpy as np
import yaml
from scipy import ndimage
from tqdm import tqdm
import pandas as pd

from kitti import SemanticKITTI
from scribble_mask_teil import get_scribble_mask_by_teil

BASE_DIR = '/Users/chenguang.gao/Desktop/Dataset/kitti'
SEQ = '10'
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
        frame = str(i)
        if len(frame) != 6:
            lost_num = 6 - len(frame)
            for k in range(lost_num):
                frame = "0" + frame
        print("********** " + "Get scribble label for velo " + frame + " **********")
        
        scribble_dir = os.path.join(args.save_dir, SEQ, save_name)
        pathlib.Path(scribble_dir).mkdir(parents=True, exist_ok=True)
        scribble_path = os.path.join(scribble_dir, frame + '.label')
        scribble_true_label = np.fromfile(scribble_path, dtype=np.int32)
        
        scribble_label = scribble_true_label.reshape((-1)) & 0xFFFF
        new_scribble_path = "/Users/chenguang.gao/Desktop/scribbles"
        new_scribble_dir = os.path.join(new_scribble_path, SEQ, save_name)
        new_scribble_path = os.path.join(new_scribble_dir, frame + '.label')
        
        scribble_label.tofile(new_scribble_path)
        
        # arr1 = pd.DataFrame(scribble_true_label)
        # arr1.to_csv("old.csv", index=False)
        # arr2 = pd.DataFrame(scribble_label)
        # arr2.to_csv("new.csv", index=False)
        
        print("********** " + "Successfully save scribble label for velo " + frame + " **********") 