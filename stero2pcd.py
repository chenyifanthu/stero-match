import cv2
import numpy as np

def extract_features_for_image(image):
    sift = cv2.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    keypoints = np.array([kp.pt for kp in keypoints])
    colors = np.array([image[int(kp[1]), int(kp[0])] for kp in keypoints])
    return keypoints, descriptor, colors

def match_features(query, train, ratio=0.7):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(query, train, k=2)
    matches = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            matches.append(m)
    return matches


def get_matched_points(p1, p2, matches):
    src_pts = np.asarray([p1[m.queryIdx].pt for m in matches])
    dst_pts = np.asarray([p2[m.trainIdx].pt for m in matches])
    return src_pts, dst_pts

def get_matched_colors(c1, c2, matches):
    color_src_pts = np.asarray([c1[m.queryIdx] for m in matches])
    color_dst_pts = np.asarray([c2[m.trainIdx] for m in matches])    
    return color_src_pts, color_dst_pts

def calculate_camera_motion(K, p1, p2):  
    focal_length = 0.5 * (K[0, 0] + K[1, 1])
    principle_point = (K[0, 2], K[1, 2])
    E,mask = cv2.findEssentialMat(p1, p2, focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)
    cameraMatrix = np.array([[focal_length, 0, principle_point[0]], [0, focal_length, principle_point[1]], [0, 0, 1]])
    pass_count, R, T, mask = cv2.recoverPose(E, p1, p2, cameraMatrix, mask)
    return R, T, mask
    
def reconstruct(camK, left_kp, right_kp, left_colors, right_colors, matches):
    p1, p2 = get_matched_colors(left_kp, right_kp, matches)
    c1, c2 = get_matched_colors(left_colors, right_colors, matches)
    
    R, T, mask = calculate_camera_motion(camK, p1, p2)
    p1 = p1[np.where(mask > 0)[0]]
    p2 = p2[np.where(mask > 0)[0]]
    c1 = c1[np.where(mask > 0)[0]]
    
    print(p1)
    

if __name__ == '__main__':
    left = cv2.imread("images/1-1.jpeg")
    right = cv2.imread("images/1-2.jpeg")
    camK = np.array([[3.38048001e+03, 0.00000000e+00, 2.08729544e+03],
                     [0.00000000e+00, 3.37789767e+03, 1.43524514e+03],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    left_keypoints, left_descriptors, left_colors = extract_features_for_image(left)
    right_keypoints, right_descriptors, right_colors = extract_features_for_image(right)
    matches = match_features(left_descriptors, right_descriptors)
    reconstruct(camK, left_keypoints, right_keypoints, left_colors, right_colors, matches)
    
    
    
    