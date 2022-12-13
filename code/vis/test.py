import numpy as np

# def concat_labels_separation(frame_idx, sem_mask, points_num, label_mask):
#     total_points_num = 0;
#     sem_mask_per_frame = []
#     for i in range(frame_idx.shape[0]):
#         semi_separation_mask_1 = np.where(sem_mask >= total_points_num)
#         semi_separation_mask_1 = np.array(semi_separation_mask_1).squeeze()
#         semi_mask_1 = np.copy(sem_mask)
#         semi_mask_1 = semi_mask_1[semi_separation_mask_1]
        
#         semi_separation_mask_2 = np.where(semi_mask_1 < total_points_num + points_num[i])
#         semi_separation_mask_2 = np.array(semi_separation_mask_2).squeeze()
#         semi_separation_mask_1 = semi_separation_mask_1[semi_separation_mask_2]
        
#         separation_mask = np.copy(sem_mask)
#         separation_mask = separation_mask[semi_separation_mask_1]
        
#         separation_mask -= total_points_num
#         sem_mask_per_frame.append(separation_mask)
        
#         total_points_num += points_num[i]
        
#     label_points_num = np.copy(points_num)
#     for i in range(label_points_num.shape[0]):
#         label_points_num[i] = sem_mask_per_frame[i].shape[0]
        
#     total_label_points_num = 0
#     label_mask_per_frame = []
#     for i in range(frame_idx.shape[0]):
#         semi_separation_mask_1 = np.where(label_mask >= total_label_points_num)
#         semi_separation_mask_1 = np.array(semi_separation_mask_1).squeeze()
#         semi_mask_1 = np.copy(label_mask)
#         semi_mask_1 = semi_mask_1[semi_separation_mask_1]
        
#         semi_separation_mask_2 = np.where(semi_mask_1 < total_label_points_num + label_points_num[i])
#         semi_separation_mask_2 = np.array(semi_separation_mask_2).squeeze()
#         semi_separation_mask_1 = semi_separation_mask_1[semi_separation_mask_2]
        
#         separation_mask = np.copy(label_mask)
#         separation_mask = separation_mask[semi_separation_mask_1]
#         separation_mask -= total_label_points_num
        
#         separation_mask = sem_mask_per_frame[i][separation_mask]
#         label_mask_per_frame.append(separation_mask)
        
#         total_label_points_num += label_points_num[i]
    
#     return label_mask_per_frame

# frame_idx = np.array([1,2,3])
# sem_mask = np.array([3,6,7,8,9,
#                      11,13,15,16,
#                      19,21,23])
# points_num = np.array([10,7,8])
# label_mask = np.array([1,2,
#                        5,7,8,
#                        9,10])

# mask = concat_labels_separation(frame_idx,sem_mask,points_num,label_mask)
# print(mask)


# l = [1,1,2,3,5,34,23,56,6,6,99]
# l2 = list(set(l))
 
# print(l2)

# l1 = {1:1,2:2,3:3}
# print(l1[1])

# a = np.array([1,1,1,1,2,3,3,4]);
# mask = np.where(a == 1)
# mask = np.array(mask)
# mask = 1

# def get_all_instances(self, all_sem_label, all_inst_label):
#     instances = {}
#     all_inst = np.array(set(all_inst_label))
#     for inst in all_inst:
#         inst_mask = np.where(all_inst_label == inst)
#         inst_mask = np.array(inst_mask).squeeze()
#         sem_label = all_sem_label[inst_mask]
#         if sem_label[0] in instances:
#             instances[sem_label[0]].append(inst)
#         else:
#             instances[sem_label[0]] = []
#             instances[sem_label[0]].append(inst)
#     return instances


null = np.array([3,4,5,6,7,8,9])
mask = np.where(null <= 1)
mask = np.array(mask).squeeze()
print(mask.shape)

a = np.array([1])
a = a.squeeze();
print(a.shape)
print(a)

single_car_label = 210
if isinstance(single_car_label,int):
    print(isinstance(single_car_label,int))
    single_car_label = np.array([single_car_label])
print(single_car_label)


a = [40,70]
print(40 in a)
