import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# pipline output images
path_image  = "test_images/"
path_out    = "output_images/"

def rgb_thresh(img, thresh=(0,255)):
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]

    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1

    return binary

def hls_thresh(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    channel  = s_channel
    s_binary = np.zeros_like(channel)
    s_binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1

    return s_binary

def color_sobel(img, c_thresh=(0,255), sx_thresh=(0,255)):
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hls  = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx          = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx      = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel    = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
#    channel  = h_channel
    channel  = s_channel
#    channel  = l_channel
    s_binary = np.zeros_like(channel)
    s_binary[(channel >= c_thresh[0]) & (channel <= c_thresh[1])] = 1
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary

def color_hsv_tresh(img, thresh=(0,255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]

    binary = np.zeros_like(s_channel)
    binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    return binary

def color_hls_thresh(img, thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    # H is from 0-179 fitting 8-bit image
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    binary = np.zeros_like(s_channel)
    binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    return binary

def mag_thresh(img_gray, sobel_kernel=3, thresh=(0, 255)):
    sobelx       = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely       = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelxy  = np.sqrt(sobelx**2 + sobely**2)

    scaled       = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))

    sbinary      = np.zeros_like(scaled)
    sbinary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1

    return sbinary

def dir_threshold(img_gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx       = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely       = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    direction    = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    sbinary      = np.zeros_like(direction)
    sbinary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    return sbinary

def abs_sobel_thresh(img_gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel       = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel       = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel    = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

#    print(sobel)
#    print(scaled_sobel)
#    print(np.max(abs_sobel))
#    print(np.max(scaled_sobel))

    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sbinary

if __name__ == "__main__":
    for image in os.listdir(path_image):
#        path_image   = "test_images/"
#        image        = "perspectivestraight_lines1.jpg"
#        image        = "perspectivestraight_lines2.jpg"
#        image        = "perspective_origtest5.jpg"
#        image        = "signs.png"
#        image        = "test5.jpg"
#        image        = "straight_lines1.jpg"
#        image        = "perspectivetest5.jpg"
        plt_img      = mpimg.imread(path_image+image)
        img          = cv2.imread(path_image+image)
#        img2         = cv2.imread(path_image+image)
        img_hls      = np.copy(img)
#        img_hls_sbl  = np.copy(img)
        hls          = cv2.cvtColor(img_hls, cv2.COLOR_BGR2HLS).astype(np.float)

        s_channel    = hls[:,:,2]
        r_channel    = img[:,:,2]

        channel_comb  = np.zeros_like(s_channel)
        channel_comb  = s_channel + r_channel

#        gray         = hls[:,:,2]
#        gray         = img[:,:,2]
        gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        gray         = channel_comb
        
        # Choose a Sobel kernel size
        ksize = 9 # Choose a larger odd number to smooth gradient measurements

        # Apply each of the thresholding functions
#        sobelx_binary           = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 40))
#        sobely_binary   = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(60, 80))
        rgb_binary      = rgb_thresh(img, thresh=(210,255))
        hls_binary      = hls_thresh(img, thresh=(180,255))
#        hls_binary      = hls_thresh(img, thresh=(180,255))
        sobelx_binary   = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 200))
        sobely_binary   = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 200))
        mag_binary      = mag_thresh(gray, sobel_kernel=ksize, thresh=(20, 100))
#        dir_binary      = dir_threshold(gray, sobel_kernel=15, thresh=(np.pi/4, np.pi/2))
        dir_binary      = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.0, 0.2))
#        dir_binary      = dir_threshold(gray, sobel_kernel=31, thresh=(4.0, 10.0))
#        dir_binary      = dir_threshold(gray, sobel_kernel=15, thresh=(1, 1.4))
#        color_hls_bin   = color_hls_thresh(img_hls, thresh=(170, 255))
#        hls_sbl_bin     = color_sobel(img_hls_sbl, c_thresh=(170, 255), sx_thresh=(50,100))
        
        combined = np.zeros_like(rgb_binary)
#        stack1 = np.zeros_like(rgb_binary)
#        stack2 = np.zeros_like(rgb_binary)
#        combined2 = np.zeros_like(rgb_binary)
#        combined = np.zeros_like(hls_binary)
#        combined[((sobelx_binary == 1) & (sobely_binary| ((mag_binary == 1) & (dir_binary == 1))] = 1
#        combined[((sobelx_binary == 1) & (sobely_binary = 1
#        combined[((sobelx_binary == 1) & (mag_binary == 1))] = 1
#        combined[((mag_binary == 1) & (dir_binary == 1))] = 1
#        combined[((sobelx_binary == 1) & (mag_binary == 1)) | ((sobely_binary (dir_binary == 1))] = 1
#        combined[((sobelx_binary == 1) & (sobely_binary| ((mag_binary == 1) & (dir_binary == 1))] = 1

#        h_channel = hls[:,:,0]
#        l_channel = hls[:,:,1]

#        combined = gray
#        combined = channel_comb
#        combined = r_channel
#        combined = s_channel
#        combined = sobelx_binary
#        combined = sobely_binary  combined = dir_binary
#        combined = mag_binary
#        combined = rgb_bina
#        combined = hls_binary

#        stack1[((sobelx_binary == 1 ) & (sobely_binary = 1
#        stack2[((hls_binary == 1) & (rgb_binary == 1))] = 1
#        stack1 = sobelx_binary
#        stack2 = rgb_bina
#        color_binary = np.dstack(( np.zeros_like(stack1), stack1, stack2))
        combined[((sobelx_binary == 1 ) & (sobely_binary == 1) | (hls_binary == 1) & (rgb_binary == 1) | ((mag_binary == 1) & (dir_binary == 1)))] = 1

#        color_binary     = color_sobel(img_hls_sbl, c_thresh=(170, 255), sx_thresh=(50,100))

#        B = img2[:,:,0]
#        G = img2[:,:,1]
#        R = img2[:,:,2]
        
#        print(R)
#        combined = rgb_bina
#        combined2[((hls_binary == 1) & ( rgb_binary == 1) | (mag_binary == 1) & (rgb_binary == 1)) ] = 1

#        plt.imshow(stack2, cmap='gray')
#        plt.show()

#        plt.imshow(color_binary, cmap='gray')
#        plt.show()
#        plt.title(image)
#        plt.imshow(combined, cmap='gray')
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        f.tight_layout()
        ax1.imshow(plt_img)
        ax1.set_title('Original Image ' + image , fontsize=12)
        ax2.imshow(combined, cmap='gray')
        ax2.set_title('Thresholded', fontsize=12)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#        plt.imshow(combined2, cmap='gray')
#        plt.imshow(combined)
#        plt.imshow(dir_binary, cmap='gray')
#        plt.imshow(combined, cmap='gray')
#        plt.savefig('test.jpg')
        plt.show()

#        cv2.imwrite(path_out+"binary_rgb"+image, rgb_thresh*255 )
#        cv2.imwrite(path_out+"binary_sobelx"+image, sobelx_binary*255 )
#        cv2.imwrite(path_out+"binary_sobely"+image, sobely_binary       cv2.imwrite(path_out+"binary_mag"+image, mag_binary*255 )
#        cv2.imwrite(path_out+"binary_dir"+image, dir_binary*255)
#        cv2.imwrite(path_out+"binary_comb"+image, combined*255 )
#        cv2.imwrite(path_out+"binary_hls"+image, color_hls_bin*255 )
#        cv2.imwrite(path_out+"binary_hls_sbl"+image, hls_sbl_bin*255 )
