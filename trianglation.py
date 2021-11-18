import cv2
import numpy as np
import open3d as o3d

# camK = np.array([[3.38048001e+03, 0.00000000e+00, 2.08729544e+03],
#                     [0.00000000e+00, 3.37789767e+03, 1.43524514e+03],
#                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# D = np.array([[2.66170916e-01, -1.66939104e+00, -5.80920399e-04, 1.99668672e-04, 3.64622788e+00]])

camK = np.array([3380, 0, 2016, 0, 3380, 1512, 0, 0, 1]).reshape(3, -1)
D = np.zeros((1, 5))



if __name__ == '__main__':
    left = cv2.imread("images/2-1.JPG")
    right = cv2.imread("images/2-2.JPG")
    print(left.shape)

    points1 = np.load("images/corr2.npz")['left']
    points2 = np.load("images/corr2.npz")['right']
    
    E, mask = cv2.findEssentialMat(points1, points2, camK)
    points1 = points1[np.where(mask == 1)[0]]
    points2 = points2[np.where(mask == 1)[0]]
    num, R, t, mask = cv2.recoverPose(E, points1, points2)

    # combine = np.concatenate((left, right), axis=0)
    # points1 = np.int32(points1)
    # points2 = np.int32(points2)
    # for i in range(points1.shape[0]):
    #     cv2.line(combine, (points1[i,0], points1[i,1]), (points2[i,0], points2[i,1]+left.shape[0]), color=(255, 0, 0), thickness=2)
    #     cv2.circle(combine, (points1[i,0], points1[i,1]), 5, (0, 0, 255), thickness=2)
    #     cv2.circle(combine, (points2[i,0], points2[i,1]+left.shape[0]), 5, (0, 0, 255), thickness=2)
    # cv2.imwrite("combine.jpg", combine)
    # exit()
    
    projMatr1 = camK @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    projMatr2 = camK @ np.concatenate((R, t), axis=1)
    points3d = cv2.triangulatePoints(projMatr1, projMatr2, points1.T, points2.T)
    points3d /= points3d[3]
    xyz = points3d[:3].T

    xyz = xyz[np.where(np.abs(xyz[:, 2]) >= 0)]
    print(xyz.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])
    print(pcd)
    print(points3d)
