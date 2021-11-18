import cv2
import numpy as np
import matplotlib.pyplot as plt

DPI = 96

left = cv2.imread("images/1-1.jpeg", 0)
right = cv2.imread("images/1-2.jpeg", 0)

window_size = 9
min_disp = 0
num_disp = 112 - min_disp
left_matcher = cv2.StereoSGBM_create(minDisparity=min_disp,
                                numDisparities=num_disp,
                                blockSize=8,
                                P1=8 * 3 * window_size ** 2,
                                P2=32 * 3 * window_size ** 2,
                                disp12MaxDiff=1,
                                uniquenessRatio=10,
                                speckleWindowSize=100,
                                speckleRange=32
                                )

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(80000)
wls_filter.setSigmaColor(1.2)

disparity_left = np.int16(left_matcher.compute(left, right))
disparity_right = np.int16(right_matcher.compute(right, left) )

wls_image = wls_filter.filter(disparity_left, left, None, disparity_right)
wls_image = cv2.normalize(src=wls_image, dst=wls_image, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
wls_image = np.uint8(wls_image)

fig = plt.figure(figsize=(wls_image.shape[1]/DPI, wls_image.shape[0]/DPI), dpi=DPI, frameon=False);
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(wls_image, cmap='jet');
cv2.imshow('ll',wls_image)
cv2.waitKey(0)
print("llllll")
# plt.savefig(DATASET_DISPARITIES+name)
plt.close()




# disp = stereo.compute(left, right).astype(np.float32) / 16.0
# disp8u = disp.convertTo(cv2.CV_8U, 255/((num_disp*16+16)*16.))
# cv2.imwrite("disparity.jpg", disp8u)
# print(disp)
# print(np.max(disp))
# cv2.reprojectImageTo3D()