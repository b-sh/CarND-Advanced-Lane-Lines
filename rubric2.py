import os
import cv2
import pickle

# path images
path_image  = "test_images/"
path_out    = "output_images/"

# load calibration
calib = pickle.load(open("calibration.p","rb"))
mtx   = calib['mtx']
dist  = calib['dist']

for image in os.listdir(path_image):
    img          = cv2.imread(path_image+image)

    undistorted  = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(path_out+"undistorted"+image, undistorted)

