import numpy as np
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

def warper(img, src, dst):
    img_size = (img.shape[1],img.shape[0])
    M        = cv2.getPerspectiveTransform(src, dst)
    warped   = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M

for image in os.listdir(path_image):
    img          = cv2.imread(path_image+image)

    undistorted  = cv2.undistort(img, mtx, dist, None, mtx)

    imshape = img.shape
    upper_left=85
    upper_right=105
    height=85
    left_bottom=260
    right_bottom=200
    height_bottom=80
    vertices1 = np.array([[(left_bottom, imshape[0] - height_bottom),(imshape[1]/2 - upper_left, imshape[0]/2 + height), (imshape[1]/2 + upper_right, imshape[0]/2 + height), (imshape[1] - right_bottom ,imshape[0] - height_bottom)]], dtype=np.int32)
    src = np.float32([[left_bottom, imshape[0] - height_bottom], [imshape[1]/2 - upper_left, imshape[0]/2 + height], [imshape[1]/2 + upper_right, imshape[0]/2 + height], [imshape[1] - right_bottom ,imshape[0] - height_bottom]])

    upper_left=100
    upper_right=100
    height=0
    left_bottom=200
    right_bottom=200
    height_bottom=0
#    dst = np.float32([[left_bottom, imshape[0] - height_bottom], [imshape[1]/2 - upper_left, imshape[0]/2 + height], [imshape[1]/2 + upper_right, imshape[0]/2 + height], [imshape[1] - right_bottom ,imshape[0] - height_bottom]])
#    vertices2 = np.array([[(left_bottom, imshape[0] - height_bottom),(imshape[1]/2 - upper_left, imshape[0]/2 + height), (imshape[1]/2 + upper_right, imshape[0]/2 + height), (imshape[1] - right_bottom ,imshape[0] - height_bottom)]], dtype=np.int32)

    dst = np.float32([[left_bottom, imshape[0] - height_bottom], [left_bottom, 0], [imshape[1] - right_bottom, 0], [imshape[1] - right_bottom ,imshape[0] - height_bottom]])
    vertices2 = np.array([[(left_bottom, imshape[0] - height_bottom),(left_bottom, 0), (imshape[1] - right_bottom, 0), (imshape[1] - right_bottom ,imshape[0] - height_bottom)]], dtype=np.int32)

    top_down, perspective_M = warper(img, src, dst)

    if len(imshape) > 2:
        channel_count = imshape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

#    cv2.fillPoly(top_down, vertices2, ignore_mask_color)

    cv2.imwrite(path_out+"perspective"+image, top_down)

