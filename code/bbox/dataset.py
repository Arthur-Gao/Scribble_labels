from pykitti import odometry
import numpy as np
# https://github.com/alexkreimer/odometry/blob/master/devkit/readme.txt
# https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py

class AlignedKITTI(odometry):
    def get_label(self, idx):
        filename = self.velo_files[idx].replace('velodyne', 'labels').replace('.bin', '.label')
        label = np.fromfile(filename, dtype=np.int32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16    # instance id in upper half
        return sem_label, inst_label

    def get_labels(self, idx: list):
        sem_labels = []
        inst_labels = []
        for i in idx:
            sem_label, inst_label = self.get_label(i)
            # Filter moving
            sem_label[sem_label > 250] = 0
            sem_labels.extend(sem_label)
            inst_labels.extend(inst_label)
        return np.array(sem_labels), np.array(inst_labels)

    def get_velo_pose(self, idx):
        pose_ = np.matmul(self.poses[idx], self.calib.T_cam0_velo)
        Tr_inv = np.linalg.inv(self.calib.T_cam0_velo)
        return np.matmul(Tr_inv, pose_)

    def get_aligned_velo(self, idx, align_idx=0):
        velo = self.get_velo(idx)
        reflectance_ = velo[:,3]
        velo[:,3] = 1
        pose = self.get_velo_pose(idx)
        align_pose = self.poses[align_idx]
        diff_pose = np.matmul(np.linalg.inv(align_pose), pose)
        velo = np.matmul(diff_pose, velo.T).T
        velo[:,3] = reflectance_
        return velo

    def get_aligned_velos(self, idx: list, align_idx=0):
        velos = [self.get_aligned_velo(i, align_idx) for i in idx]
        return np.concatenate(velos, axis=0)

    def concat_velo_based_on_label(
            self,
            sem_label,     # semantic label of the object
            inst_label,    # instance label of the object on frame 0
            idx=0,         # starting index from list
            search_len=30, # search length on either direction
        ):
        first_idx = max(0, idx-search_len)
        last_idx = min(len(self.velo_files), idx+search_len)
        if last_idx != len(self.velo_files):
            search_idx = list(range(first_idx, last_idx + 1))
        else:
            search_idx = list(range(first_idx, last_idx))
        print(f'{len(search_idx)} search frames with indices: {search_idx}.')

        velo = self.get_aligned_velo(idx, idx)
        sem, inst = self.get_label(idx)

        initial_mask = (sem == sem_label) & (inst == inst_label)
        if not initial_mask.sum():
            sem_mask = sem == sem_label
            print('inst:', np.unique(inst[sem_mask]))
            assert False
        current_velo = velo[initial_mask]
        if not search_len:
            return current_velo
        initial_point_count = current_velo.shape[0]

        concat_frames = []
        search_idx.remove(idx)
        for e, i in enumerate(search_idx):
            min_boundary = current_velo.min(0)
            max_boundary = current_velo.max(0)

            velo = self.get_aligned_velo(i, idx)
            sem, inst = self.get_label(i)
            sem_mask = sem == sem_label
            masked_velo = velo[sem_mask]
            masked_inst = inst[sem_mask]

            matching_inst = -1
            for j in np.unique(masked_inst):
                # check if any points inside current boundaries
                inst_mask = masked_inst == j
                inst_velo = masked_velo[inst_mask]
                if ((inst_velo >= min_boundary).all(1) & \
                    (inst_velo <= max_boundary).all(1)).sum():
                    matching_inst = j
                    break

            if matching_inst != -1:
                concat_frames.append(i)
                current_velo = np.concatenate((current_velo, inst_velo))
        
        if last_idx != len(self.velo_files):
            print(f'Search range set between frames {first_idx} and {last_idx}.')
        else:
            print(f'Search range set between frames {first_idx} and {last_idx - 1}.')
        print(f'{len(concat_frames)} concatenated frames with indices: {concat_frames}.')
        print(f'Initial point count {initial_point_count}.')
        print(f'Concatenated point count {current_velo.shape[0]}.')
        return current_velo