import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

# pipline output images
path_image  = "test_images/"
path_out    = "output_images/"

def mag_thresh(img_gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx       = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely       = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, sobel_kernel)

    abs_sobelx   = np.square(sobelx)
    abs_sobely   = np.square(sobely)
    abs_sobelxy  = abs_sobelx + abs_sobely

    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))

    sbinary      = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return sbinary

def dir_threshold(img_gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx       = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely       = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, sobel_kernel)

    scaled_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    sbinary      = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sbinary

def abs_sobel_thresh(img_gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel       = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, sobel_kernel)
    else:
        sobel       = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, sobel_kernel)

    abs_sobel    = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sbinary

for image in os.listdir(path_image):
    img          = cv2.imread(path_image+image)
    gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Choose a Sobel kernel size
    ksize = 9 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx       = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(50, 100))
    grady       = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(50, 100))
    mag_binary  = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary  = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.9, 1.1))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    cv2.imwrite(path_out+"binary_sobelx"+image, gradx*255 )
    cv2.imwrite(path_out+"binary_sobely"+image, grady*255)
    cv2.imwrite(path_out+"binary_mag"+image, mag_binary*255 )
    cv2.imwrite(path_out+"binary_dir"+image, dir_binary*255)
    cv2.imwrite(path_out+"binary_comb"+image, combined*255 )
