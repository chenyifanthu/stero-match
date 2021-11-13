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
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    
    return pts1, pts2

def get_extrinsics(points1, point2, camK):
    E, mask = cv2.findEssentialMat(points1, point2, camK)
    R1, R2, t = cv2.decomposeEssentialMat(E)
    return R1, t

if __name__ == '__main__':
    left = cv2.imread("images/1-1.jpeg")
    right = cv2.imread("images/1-2.jpeg")

    camK = np.array([[3.38048001e+03, 0.00000000e+00, 2.08729544e+03],
                    [0.00000000e+00, 3.37789767e+03, 1.43524514e+03],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    D = np.array([[2.66170916e-01, -1.66939104e+00, -5.80920399e-04, 1.99668672e-04, 3.64622788e+00]])
    
    points1, points2 = get_sift_matches(left, right)
    R, t = get_extrinsics(points1, points2, camK)
    print(R, t)

    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(camK, D, camK, D, (4032, 3024), R, t, 1, (0, 0))
    left_stero_map = cv2.initUndistortRectifyMap(camK, D, RL, PL, (4032, 3024), cv2.CV_16SC2)
    left_rectified= cv2.remap(left, left_stero_map[0], left_stero_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    right_stero_map = cv2.initUndistortRectifyMap(camK, D, RR, PR, (4032, 3024), cv2.CV_16SC2)
    right_rectified= cv2.remap(right, right_stero_map[0], right_stero_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    combine = np.concatenate((left_rectified, right_rectified), axis=1)
    cv2.imwrite("combine.png", combine)
    
    window_size = 3
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=240,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=3,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = sgbm.compute(left_rectified, right_rectified).astype(np.float32) / 16.0
    cv2.imwrite("disparity.jpg", disparity)
    
    
