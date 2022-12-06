import numpy as np

a = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
z_mask = (a[:,2] <= 3)
a = a[z_mask]
z_mask = (a[:,2] >= 2)
a = a[z_mask]
print(a)

def point_distance_line(point,line_point1,line_point2):
	#计算向量
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance

point = np.array([5,2])
line_point1 = np.array([2,2])
line_point2 = np.array([3,3])
print(point_distance_line(point,line_point1,line_point2))

x = np.array([[5,2],[8,3]])
z = np.copy(x)
z = -z
z[:,0] += 2
print(x)
print(z)
y = np.array([1,3])

new_pc = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
new_pc_mask = np.array([1,2])
print(new_pc[new_pc_mask])
# mask_z = np.where(2 <= new_pc[:,2])
# new_pc = new_pc[mask_z]
# mask_k = np.where(3 >= new_pc[:,2])
# mask_z = np.array(mask_z).squeeze()
# print(mask_z)
# print(mask_z[mask_k])
