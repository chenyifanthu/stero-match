import numpy as np
import open3d as o3d

points = np.load("structure.npy")
colors = np.load("colors.npy") / 255
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])

print(colors)
print(points.shape)
print(colors.shape)
