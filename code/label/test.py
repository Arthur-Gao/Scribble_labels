import numpy as np

x = np.array([[1,1],[-1,-1]])
c1 = np.array([1,0.4])
print((x*c1).sum(1))
print(c1.max() - c1)
print((1 / np.maximum(c1, 0.8)).sum())

# x = np.array([1,2,3,4,5,6,7,8,9])
x = np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
in_x = np.where((x[:,0] >= 1) & (x[:,0] < 5) & (x[:,1] < 4))
print(np.asarray(in_x).squeeze())

x = []
x.append([])
x.append([])
print(x[0])

x = np.array([5,4,3,2,1])
x = np.sort(x)
print(x)