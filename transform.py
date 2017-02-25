import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

# pipline output images
path_image  = "test_images/"
path_out    = "output_images/"

def abs_sobel_thresh(img_gray, orient='x', thresh_min=0, thresh_max=255):    
    if orient == 'x':
        sobel       = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)
    else:
        sobel       = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1)

    abs_sobel    = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return sbinary

for image in os.listdir(path_image):
    img          = cv2.imread(path_image+image)
    gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path_out+"grey"+image, gray)

    grad_binary = abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255)
    cv2.imwrite(path_out+"binary_sobelx"+image, grad_binary)

    grad_binary = abs_sobel_thresh(gray, orient='y', thresh_min=0, thresh_max=255)
    cv2.imwrite(path_out+"binary_sobely"+image, grad_binary)
