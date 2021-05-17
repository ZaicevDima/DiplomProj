from math import pi

import cv2 as cv
import numpy as np

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Arrays to store object points and image points from all the images.
objpoints = np.zeros((11, 1, 8, 3), dtype='float64') # 3d point in real world space
imgpoints = np.zeros((11, 1, 8, 2), dtype='float64')  # 2d points in image plane.
image = cv.imread('patterns/1.jpg')
alpha = pi/18
objpointRed = np.array([13, 77, 0], dtype='float64')
objpointYellow = np.array([17, 74, 50], dtype='float64')
objpointBlue = np.array([58, 81, 75], dtype='float64')
objpointPink = np.array([129, 97, 80], dtype='float64')
objpointPurple = np.array([109, 100, 0], dtype='float64')
objpointWhite = np.array([129, 97, -80], dtype='float64')
objpointBlack = np.array([58, 81, -75], dtype='float64')
objpointGreen = np.array([17, 74, -50], dtype='float64')

for i in range(11):
    objpoints[i] = (np.array(
        [[objpointRed, objpointYellow, objpointBlue, objpointPink, objpointPurple, objpointWhite, objpointBlack,
         objpointGreen]], np.float64))

# 1
imagepointRed = [533, 740]
imagepointYellow = [769, 717]
imagepointBlue = [842, 678]
imagepointPink = [738, 565]
imagepointPurple = [495, 574]
imagepointWhite = [252, 568]
imagepointBlack = [194, 686]
imagepointGreen = [290, 729]
imgpoints[0] = (np.array(
    [[imagepointRed, imagepointYellow, imagepointBlue, imagepointPink, imagepointPurple, imagepointWhite,
     imagepointBlack, imagepointGreen]], np.float64))

# 2
imagepointRed = [442, 615]
imagepointYellow = [570, 608]
imagepointBlue = [632, 601]
imagepointPink = [604, 543]
imagepointPurple = [448, 548]
imagepointWhite = [310, 536]
imagepointBlack = [276, 595]
imagepointGreen = [324, 611]
imgpoints[1] = (np.array(
    [[imagepointRed, imagepointYellow, imagepointBlue, imagepointPink, imagepointPurple, imagepointWhite,
      imagepointBlack, imagepointGreen]], np.float64))

# 3
imagepointRed = [434, 636]
imagepointYellow = [583, 626]
imagepointBlue = [648, 611]
imagepointPink = [620, 545]
imagepointPurple = [450, 550]
imagepointWhite = [293, 538]
imagepointBlack = [251, 610]
imagepointGreen = [301, 629]
imgpoints[2] = (np.array(
    [[imagepointRed, imagepointYellow, imagepointBlue, imagepointPink, imagepointPurple, imagepointWhite,
      imagepointBlack, imagepointGreen]], np.float64))

# 4
imagepointRed = [586, 773]
imagepointYellow = [846, 750]
imagepointBlue = [914, 697]
imagepointPink = [781, 568]
imagepointPurple = [527, 580]
imagepointWhite = [249, 567]
imagepointBlack = [194, 709]
imagepointGreen = [303, 760]
imgpoints[3] = (np.array(
    [[imagepointRed, imagepointYellow, imagepointBlue, imagepointPink, imagepointPurple, imagepointWhite,
      imagepointBlack, imagepointGreen]], np.float64))

# 5
imagepointRed = [797, 868]
imagepointYellow = [1089, 803]
imagepointBlue = [1121, 739]
imagepointPink = [896, 588]
imagepointPurple = [627, 601]
imagepointWhite = [243, 594]
imagepointBlack = [191, 795]
imagepointGreen = [370, 892]
imgpoints[4] = (np.array(
    [[imagepointRed, imagepointYellow, imagepointBlue, imagepointPink, imagepointPurple, imagepointWhite,
      imagepointBlack, imagepointGreen]], np.float64))

# 6
imagepointRed = [815, 825]
imagepointYellow = [1083, 767]
imagepointBlue = [1116, 722]
imagepointPink = [914, 580]
imagepointPurple = [656, 591]
imagepointWhite = [291, 582]
imagepointBlack = [253, 766]
imagepointGreen = [433, 844]
imgpoints[5] = (np.array(
    [[imagepointRed, imagepointYellow, imagepointBlue, imagepointPink, imagepointPurple, imagepointWhite,
      imagepointBlack, imagepointGreen]], np.float64))

# 7
imagepointRed = [600, 770]
imagepointYellow = [757, 725]
imagepointBlue = [872, 742]
imagepointPink = [807, 567]
imagepointPurple = [545, 576]
imagepointWhite = [264, 567]
imagepointBlack = [200, 707]
imagepointGreen = [308, 754]
imgpoints[6] = (np.array(
    [[imagepointRed, imagepointYellow, imagepointBlue, imagepointPink, imagepointPurple, imagepointWhite,
      imagepointBlack, imagepointGreen]], np.float64))

# 8
imagepointRed = [392, 678]
imagepointYellow = [600, 666]
imagepointBlue = [703, 659]
imagepointPink = [670, 555]
imagepointPurple = [434, 563]
imagepointWhite = [242, 553]
imagepointBlack = [165, 648]
imagepointGreen = [219, 678]
imgpoints[7] = (np.array(
    [[imagepointRed, imagepointYellow, imagepointBlue, imagepointPink, imagepointPurple, imagepointWhite,
      imagepointBlack, imagepointGreen]], np.float64))

# 9
imagepointRed = [331, 638]
imagepointYellow = [501, 644]
imagepointBlue = [594, 631]
imagepointPink = [602, 552]
imagepointPurple = [397, 551]
imagepointWhite = [254, 540]
imagepointBlack = [176, 616]
imagepointGreen = [211, 640]
imgpoints[8] = (np.array(
    [[imagepointRed, imagepointYellow, imagepointBlue, imagepointPink, imagepointPurple, imagepointWhite,
      imagepointBlack, imagepointGreen]], np.float64))

# 10
imagepointRed = [437, 635]
imagepointYellow = [592, 629]
imagepointBlue = [661, 618]
imagepointPink = [629, 550]
imagepointPurple = [449, 553]
imagepointWhite = [286, 541]
imagepointBlack = [245, 614]
imagepointGreen = [295, 639]
imgpoints[9] = (np.array(
    [[imagepointRed, imagepointYellow, imagepointBlue, imagepointPink, imagepointPurple, imagepointWhite,
      imagepointBlack, imagepointGreen]], np.float64))

# 11
imagepointRed = [451, 644]
imagepointYellow = [620, 642]
imagepointBlue = [691, 629]
imagepointPink = [645, 550]
imagepointPurple = [454, 555]
imagepointWhite = [278, 545]
imagepointBlack = [237, 625]
imagepointGreen = [291, 650]
imgpoints[10] = (np.array(
    [[imagepointRed, imagepointYellow, imagepointBlue, imagepointPink, imagepointPurple, imagepointWhite,
      imagepointBlack, imagepointGreen]], np.float64))

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
_img_shape = None
_img_shape = gray.shape[:2]
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]


ret, _, _, _, _ = cv.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], K, D, rvecs, tvecs,
                                       flags=cv.fisheye.CALIB_FIX_PRINCIPAL_POINT + cv.fisheye.CALIB_FIX_SKEW + cv.fisheye.CALIB_CHECK_COND)
print(_img_shape, gray.shape[::-1])
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

#K = np.array([[727.742365559, -185.31980097, 959.5], [0, 971.10064313, 539.5], [0, 0, 1]], np.float64)
DIM=_img_shape[::-1]
balance=1
dim2=None
dim3=None

img = image
dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
if not dim2:
    dim2 = dim1
if not dim3:
    dim3 = dim1
scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=0)

undistorted_img = cv.fisheye.undistortImage(image, K, D, None, new_K, dim3)
map1, map2 = cv.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv.CV_16SC2)
#undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

#newcameramtx, roi = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, (w,h), )
#print(dim3)
#newImage = cv.fisheye.undistortImage(image, K, D, new_K, dim3)

cv.imwrite("undistorted.jpg", undistorted_img)
cv.imwrite("noneundistorted.jpg", image)


