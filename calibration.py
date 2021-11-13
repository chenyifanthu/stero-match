import cv2
import glob
import numpy as np
from tqdm import tqdm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 设置 object points, 形式为 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32) #我用的是6×7的棋盘格，可根据自己棋盘格自行修改相关参数
objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

# 用arrays存储所有图片的object points 和 image points
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#用glob匹配文件夹/home/song/pic_1/right/下所有文件名含有“.jpg"的图片
images = glob.glob(r"./biaoding/*.JPG")

for fname in tqdm(images[:20]):
    img = cv2.imread(fname)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (6, 8), None)
    # 如果找到了就添加 object points, image points
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # 对角点连接画线加以展示
#         cv2.drawChessboardCorners(img, (6,8), corners2, ret)
#         cv2.imshow('img', img)
#         cv2.waitKey(500)
# cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx, dist)