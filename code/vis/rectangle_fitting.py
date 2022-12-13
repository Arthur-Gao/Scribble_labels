# Search-Based Rectangle Fitting
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7995698

import numpy as np

def closeness_criterion(c1, c2, d0):
    d1 = np.minimum(c1.max()-c1, c1-c1.min())
    d2 = np.minimum(c2.max()-c2, c2-c2.min())
    return (1/np.maximum(np.minimum(d1, d2), d0)).sum()

def project_to_unit(theta, x):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    c1 = (x * np.array([cos_theta, sin_theta])).sum(1)
    c2 = (x * np.array([-sin_theta, cos_theta])).sum(1)
    return c1, c2

def rectangle_fitting(pcd, num_bins=100, d0=1e-3):
    x = np.array(pcd.points)[:,:2]
    thetas = [np.pi*i/(2*num_bins) for i in range(num_bins)]

    max_value = 0
    best_theta = None
    for theta in thetas:
        c1, c2 = project_to_unit(theta, x)
        value = closeness_criterion(c1, c2, d0)
        if value > max_value:
            max_value = value
            best_theta = theta
    return best_theta