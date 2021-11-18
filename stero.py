import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def get_sift_matches(image1, image2, ratio=0.8):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k = 2)
    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.85*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    return pts1, pts2

def get_extrinsics(points1, point2, camK):
    E, mask = cv2.findEssentialMat(points1, point2, camK)
    R1, R2, t = cv2.decomposeEssentialMat(E)
    return R1, t

def get_f_r(pos, arr):
    l = r = pos
    for i in range(pos, 1, -1):
        if arr[i] > 0:
            l = i
            break
    for i in range(pos, len(arr)-1):
        if arr[i] > 0:
            r = i
            break

    return l, r


if __name__ == '__main__':
    left = cv2.imread("images/2-1.JPG")
    right = cv2.imread("images/2-2.JPG")

    camK = np.array([[3.38048001e+03, 0.00000000e+00, 2.08729544e+03],
                    [0.00000000e+00, 3.37789767e+03, 1.43524514e+03],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    D = np.array([[2.66170916e-01, -1.66939104e+00, -5.80920399e-04, 1.99668672e-04, 3.64622788e+00]])
    
    points1, points2 = get_sift_matches(left, right)
    np.savez("images/corr2.npz", left=points1, right=points2)
    print(points1.shape[0])
    exit()
    
    R, t = get_extrinsics(points1, points2, camK)
    print(R, t)

    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(camK, D, camK, D, (4032, 3024), R, t, 1, (0, 0))
    left_stero_map = cv2.initUndistortRectifyMap(camK, D, RL, PL, (4032, 3024), cv2.CV_16SC2)
    left_rectified= cv2.remap(left, left_stero_map[0], left_stero_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    right_stero_map = cv2.initUndistortRectifyMap(camK, D, RR, PR, (4032, 3024), cv2.CV_16SC2)
    right_rectified= cv2.remap(right, right_stero_map[0], right_stero_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    combine = np.concatenate((left_rectified, right_rectified), axis=1)
    
    for i in range(0, combine.shape[0], 100):
        cv2.line(combine, (0, i), (combine.shape[1]-1, i), (0, 0, 255), thickness=1)
    cv2.imwrite("combine.png", combine)
    
    left_rectified = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_rectified = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    
    window_size = 9
    min_disp = 0
    num_disp = 240 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=8,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )

    disp = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0
    
    print(disp)
    # new_disp = (disp - min_disp) / num_disp
    cv2.imwrite("disparity.jpg", disp)
    
    
    
    
    # count = 0
    # for j in range(new_disp.shape[0]):
    #     if new_disp[0, j] > 0:
    #         count = j
    #         break
    # new_disp_pp = np.zeros_like(new_disp)
    # for i in range(new_disp.shape[0]):
    #     for j in range(count, new_disp.shape[1] - 1):
    #         if new_disp[i, j] <= 0:
    #             left, right = get_f_r(j, new_disp[i])
    #             #up, down  = get_f_r(i, new_disp[:, j])
    #             new_disp_pp[i, j] = (new_disp[i, left - 1] + new_disp[i, right + 1]) / 2
    #         else:
    #             new_disp_pp[i, j] = new_disp[i, j]
    # new_disp_pp = cv2.medianBlur(new_disp_pp, 7)
    
    # cv2.imshow('disp', disp)
    # cv2.imshow('org_disp', new_disp)
    # cv2.imshow("disp_pp", new_disp_pp)
    # cv2.waitKey(0)
