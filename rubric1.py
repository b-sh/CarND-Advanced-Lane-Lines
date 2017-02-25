import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

# pipline output images
path        = "output_images/"
path_calib  = "camera_cal/" 
shape       = None

nx = 9
ny = 6

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 0.001)
 
# prepare object points
# creating 3D object points and mapping to 2D
# http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

def get_corners(img_gray):
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)

    # when not found just return None
    if ret != True:
        return None

#    corners = cv2.cornerSubPix(img_gray,corners,(1,1),(-1,-1),criteria)
    return corners

objpoints = []
imgpoints = []
for image in os.listdir(path_calib):
    img            = cv2.imread(path_calib+image)
    # Convert to grayscale
    gray           = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgpoints_temp = get_corners(img)

    if imgpoints_temp is not None:
        objpoints.append(objp)
        imgpoints.append(imgpoints_temp)

        img  = cv2.drawChessboardCorners(img, (nx, ny), imgpoints_temp, True)
        cv2.imwrite(path+"corners"+image, img)
    else:
        print(image)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

calib = {}
calib['mtx'] = mtx
calib['dist'] = dist

pickle.dump(calib, open("calibration.p","wb"))

for image in os.listdir(path_calib):
    img          = cv2.imread(path_calib+image)

    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(path+"undistorted"+image, undistorted)

# http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
total_error = 0
for i in range(len(objpoints)):
     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
     error         = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
     total_error    += error
 
print("total error: ", total_error/len(objpoints))

