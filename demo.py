from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from functools import reduce
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
import os
from nuscenes.utils.data_io import load_bin_file

from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box

def anno_to_box(anno):
    return Box(anno['translation'], anno['size'], Quaternion(anno['rotation'], name=anno['category_name'], token=anno['token']))

def get_cross_matched_corners(box, bottom_corners, dense_bottom_corner_id):
    top_corners = box.corners()[:, [1, 0, 4, 5]]
    top_corner_ids = [(dense_bottom_corner_id-1)%4, (dense_bottom_corner_id+1)%4]
    bottom_corner_ids = [(top_corner_ids[0]+2)%4, (top_corner_ids[1]+2)%4]
    return top_corners[:, top_corner_ids], bottom_corners[:, bottom_corner_ids]

def get_lidar_from_anno(nusc, anno):
    sample = nusc.get('sample', anno['sample_token'])
    lidar_sample = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

    file_path = os.path.join(nusc.dataroot, lidar_sample['filename'])
    assert file_path.endswith('.bin'), 'Unsupported filetype {}'.format(file_path)
    scan = np.fromfile(file_path, dtype=np.float32)
    lidar = scan.reshape((-1, 5))[:,:4].T

    calibration_sample = nusc.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
    car_from_current = transform_matrix(calibration_sample['translation'], Quaternion(calibration_sample['rotation']), inverse=False)

    pose_sample = nusc.get('ego_pose', lidar_sample['ego_pose_token'])
    global_from_car = transform_matrix(pose_sample['translation'], Quaternion(pose_sample['rotation']), inverse=False)

    projection = reduce(np.dot, [global_from_car, car_from_current])
    global_lidar = projection.dot(np.vstack((lidar[:3,:], np.ones(lidar.shape[1]))))[:3,:]
    return global_lidar

def check_is_moving(all_boxes, threshold=1):
    distance_vector = all_boxes[0].center[:2] - all_boxes[-1].center[:2]
    return np.linalg.norm(distance_vector, ord=2) > threshold

def collect_instance(nusc, idx):
    instance = nusc.instance[idx]
    annotation_token = instance['first_annotation_token']

    all_lidar = []
    all_boxes = []
    while annotation_token != '':
        current_annotation_sample = nusc.get('sample_annotation', annotation_token)
        current_box = anno_to_box(current_annotation_sample)
        current_lidar = get_lidar_from_anno(nusc, current_annotation_sample)

        mask = points_in_box(current_box, current_lidar)
        assert mask.sum() == current_annotation_sample['num_lidar_pts']

        all_lidar.append(current_lidar.T[mask])
        all_boxes.append(current_box)
        annotation_token = current_annotation_sample['next']
        # is_moving = is_moving or 'moving' in current_annotation_sample['category_name']
    is_moving = check_is_moving(all_boxes)
    return all_lidar, all_boxes, is_moving

def annotate_static_object(lidar_list, box_list, height_bias=0.1, radius=0.5, scribble_width=0.1):
    lidar = np.concatenate(lidar_list, axis=0)

    # Filter points in the ground boundary
    mean_center = np.mean([box.center for box in box_list], axis=0)
    mean_center[2] += height_bias/2
    mean_size =  np.mean([box.wlh for box in box_list], axis=0)
    mean_size[2] -= height_bias/2
    filtered_box = Box(
        center = mean_center.tolist(),
        size = mean_size.tolist(),
        orientation = Quaternion(np.mean([box.orientation for box in box_list], axis=0))
    )
    filtering_mask = points_in_box(filtered_box, lidar.T)
    filtered_lidar = lidar[filtering_mask]

    # Find the heighest density front bottom corner
    bot_corners = filtered_box.bottom_corners()[:,:2]
    bot_distance_matrix = np.linalg.norm(filtered_lidar[:,:,None] - bot_corners[None], ord=2, axis=1)
    dense_bot_corner_id = (bot_distance_matrix < radius).sum(0).argmax()
    final_bot_corner = bot_corners[:,dense_bot_corner_id]

    # Find the cross top corner
    cross_top_corner_id = dense_bot_corner_id + 4
    final_top_corner = filtered_box.corners()[:, cross_top_corner_id]

    # Rotate box and points
    Rinv = filtered_box.orientation.rotation_matrix
    filtered_lidar = filtered_lidar.dot(Rinv)
    final_bot_corner = final_bot_corner.dot(Rinv)
    final_top_corner = final_top_corner.dot(Rinv)

    # Annotate along the width of the car
    axis = [0, 2]
    projected_bot_corner = final_bot_corner[axis]
    projected_top_corner = final_top_corner[axis] - projected_bot_corner 
    projected_lidar = filtered_lidar[:,axis] - projected_bot_corner
    theta = np.arctan((projected_top_corner[0])/projected_top_corner[1])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    distance = np.abs(np.dot(R, projected_lidar.T).T[:,0])
    painting_mask = distance <= scribble_width

    scribble_mask = np.full_like(filtering_mask, False)
    scribble_mask[filtering_mask] = painting_mask
    return scribble_mask

def annotate_dynamic_object(lidar_list, box_list, scribble_width=0.1):
    lidar = np.concatenate(lidar_list, axis=0)
    scribble_mask = np.full((lidar.shape[0],), False)

    # distance_vector = box_list[0].center - box_list[-1].center
    # axis = [int(distance_vector[0] > distance_vector[1]), 2]
    # Always scribble from BEV
    axis = [0,1]
    for cbox, nbox in zip(box_list[:-1], box_list[1:]):
        p1 = cbox.center
        p2 = nbox.center
        filtering_mask = points_in_box(cbox, lidar.T)
        pc = lidar[filtering_mask]

        # Rotate box and points
        Rinv = cbox.orientation.rotation_matrix
        pc = pc.dot(Rinv)[:,axis]
        p1 = p1.T.dot(Rinv).T[axis]
        p2 = p2.T.dot(Rinv).T[axis]

        # Find distance to center line between consecutive boxes
        dist = np.abs(np.cross(p2-p1,pc-p1))/np.linalg.norm(p2-p1)
        scribble_mask[filtering_mask] = dist <= scribble_width
    return scribble_mask

def main():
    nusc = NuScenes(version='v1.0-mini', dataroot='/media/ozan/hdd_backup/dataset/nuscenes/', verbose=True)
    offset = 500
    num_instances = 10
    multi_lidar = []
    multi_scribble = []
    while num_instances > 0:
        all_lidar, all_boxes, is_moving = collect_instance(nusc, offset)
        offset += 1
        concat_lidar = np.concatenate(all_lidar, axis=0)
        if concat_lidar.shape[0] < 400:
            continue
        if not is_moving:
            print('isNOTmoving', offset)
            scribble_mask = annotate_static_object(all_lidar, all_boxes)
        if is_moving:
            print('ismoving', offset)
            scribble_mask = annotate_dynamic_object(all_lidar, all_boxes)
        multi_lidar.append(concat_lidar)
        multi_scribble.append(scribble_mask)
        num_instances -= 1

    np.save('all_pc.npy', np.concatenate(multi_lidar, axis=0))
    np.save('scribble.npy', np.concatenate(multi_scribble, axis=0))

if __name__=='__main__':
    main()