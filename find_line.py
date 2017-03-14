import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import transform
import perspective
from numpy.linalg import inv

# code snippets taken from udacity examples
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = []
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # bad count
        self.bad_count = 0

left_line          = Line()
right_line         = Line()

def image_test(file_path):
    img           = cv2.imread(file_path)

    print(file_path)

    return process_frame(img)

def process_frame(img):
    top_down, top_down_vert, img_orig, undist, M  = perspective.perspective(img)

    gray         = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)
    # verbose mode is using plt which expects RGB instead of cv2 BGR
    if verbose:
        undist          = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
        top_down        = cv2.cvtColor(top_down, cv2.COLOR_BGR2RGB)
        img_orig        = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        top_down_vert   = cv2.cvtColor(top_down_vert, cv2.COLOR_BGR2RGB)
        plt.imshow(undist)
        plt.show()
        plt.imshow(img_orig)
        plt.show()
        plt.imshow(top_down)
        plt.show()
        plt.imshow(top_down_vert)
        plt.show()
    
    ksize = 9

    # thresholding input
    rgb_binary      = transform.rgb_thresh(top_down, thresh=(210,255))
    hls_binary      = transform.hls_thresh(top_down, thresh=(180,255))
    sobelx_binary   = transform.abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 200))
    sobely_binary   = transform.abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 200))
    mag_binary      = transform.mag_thresh(gray, sobel_kernel=ksize, thresh=(20, 100))
    dir_binary      = transform.dir_threshold(gray, sobel_kernel=ksize, thresh=(0.1, 0.4))

    binary_warped = np.zeros_like(rgb_binary)
    binary_warped[(((sobelx_binary == 1) & (sobely_binary == 1)) | ((hls_binary == 1) & ( dir_binary == 1)) | ((rgb_binary == 1)) | ((sobelx_binary == 1) & (mag_binary == 1) & (dir_binary == 1)))] = 1
#    binary_warped[(((hls_binary == 1) & ( dir_binary == 1)) | ((rgb_binary == 1)) | ((sobelx_binary == 1) & (mag_binary == 1)))] = 1
#    binary_warped[((hls_binary == 1) & (rgb_binary == 1))] = 1
#    binary_warped[hls_binary == 1] = 1
#    binary_warped = hls_binary
#    binary_warped = sobelx_binary
#    binary_warped = sobelx_binary
#    binary_warped = mag_binary
#    binary_warped = rgb_binary
#    binary_warped = dir_binary
#    binary_warped[((hls_binary == 1) & (dir_binary == 1))] = 1
#    binary_warped[((sobelx_binary == 1) & (mag_binary == 1))] = 1
#    binary_warped = ((sobely_binary == 1))
    midpoint      = np.int(binary_warped.shape[1]/2)

    if verbose:
        plt.imshow(binary_warped, cmap='gray')
        plt.savefig("result_binary.jpg")
        plt.show()

    if left_line.detected is False and right_line.detected is False:
        # histogram of the bottom half of the image
        histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)
        if verbose:
            plt.plot(histogram)
            plt.savefig("result_histogram.jpg")
            plt.show()

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        leftx_base  = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Current positions to be updated for each window
        leftx_current   = leftx_base
        rightx_current  = rightx_base

    # Create an output image to draw on and  visualize the result
    #out_img = np.array(cv2.merge((binary_warped, binary_warped, binary_warped)),np.uint8)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero   = binary_warped.nonzero()
    nonzeroy  = np.array(nonzero[0])
    nonzerox  = np.array(nonzero[1])

    # Set the width of the windows +/- margin
    margin                = 50
    margin_after_detected = 25
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds  = []
    right_lane_inds = []

    if left_line.detected is False and right_line.detected is False:
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low       = binary_warped.shape[0] - (window+1)*window_height
            win_y_high      = binary_warped.shape[0] - window*window_height
            win_xleft_low   = leftx_current - margin
            win_xleft_high  = leftx_current + margin
            win_xright_low  = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low, win_y_low),(win_xleft_high, win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    if left_line.detected and right_line.detected:
        left_fit  = left_line.current_fit
        right_fit = right_line.current_fit
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin_after_detected)) &
                          (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin_after_detected)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin_after_detected)) &
                           (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin_after_detected)))
    else:
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    left_line.allx  = nonzerox[left_lane_inds]
    left_line.ally  = nonzeroy[left_lane_inds]
    right_line.allx = nonzerox[right_lane_inds]
    right_line.ally = nonzeroy[right_lane_inds]

#    if len(left_line.recent_xfitted) > 10:
#        del left_line.recent_xfitted[0]
#    left_line.recent_xfitted.append(left_line.allx)
#    if left_line.recent_xfitted:
#        left_line.bestx = np.average(left_line.recent_xfitted, axis=1)

#    if len(right_line.recent_xfitted) > 10:
#        del right_line.recent_xfitted[0]
#    right_line.recent_xfitted.append(right_line.allx)
#    if right_line.recent_xfitted:
#        right_line.bestx = np.average(right_line.recent_xfitted, axis=1)

    # Fit a second order polynomial to each
    left_fit  = np.polyfit(left_line.ally, left_line.allx, 2)
    right_fit = np.polyfit(right_line.ally, right_line.allx, 2)

    if not left_line.detected and not right_line.detected:
        left_line.current_fit  = left_fit
        left_line.best_fit     = [left_fit]
        right_line.current_fit = right_fit
        right_line.best_fit    = [right_fit]
        left_line.detected     = True
        right_line.detected    = True

    show_img = False

    if left_line.best_fit:
        left_best_fit  = np.mean(left_line.best_fit, axis=0)
    if right_line.best_fit:
        right_best_fit = np.mean(right_line.best_fit, axis=0)

    # when fitted values of current frame are bad
    diff = np.absolute(left_best_fit[:2] - left_fit[:2])
    if np.average(diff) < 0.1:
        left_line.current_fit = left_best_fit
        left_line.best_fit.append(left_fit)
        if len(left_line.best_fit) > 10:
            del left_line.best_fit[0]
    else:
#        show_img = True
        left_line.bad_count += 1

    diff = np.absolute(right_best_fit[:2] - right_fit[:2])
    if np.average(diff) < 0.1:
        right_line.current_fit = right_best_fit
        right_line.best_fit.append(right_fit)
        if len(right_line.best_fit) > 10:
            del right_line.best_fit[0]
    else: 
#        show_img = True
        right_line.bad_count += 1

    if left_line.bad_count > 15 or right_line.bad_count > 15:
        left_line.detected  = False
        right_line.detected = False

    left_line.diffs  = left_line.current_fit - left_line.diffs
    right_line.diffs = right_line.current_fit - right_line.diffs

    # Generate x and y values for plotting

    ploty       = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx   = left_line.current_fit[0]*ploty**2 + \
                  left_line.current_fit[1]*ploty + \
                  left_line.current_fit[2]
    right_fitx  = right_line.current_fit[0]*ploty**2 + \
                  right_line.current_fit[1]*ploty + \
                  right_line.current_fit[2]
    
    # radius of curvature
    # http://www.intmath.com/applications-differentiation/8-radius-curvature.php
    y_eval      = np.max(ploty)
    ym_per_pix  = 30/720    # meters per pixel in y dimension
    xm_per_pix  = 3.7/1280  # meters per pixel in x dimension

    left_fit_cr     = np.polyfit(left_line.ally*ym_per_pix, left_line.allx*xm_per_pix, 2)
    right_fit_cr    = np.polyfit(right_line.ally*ym_per_pix, right_line.allx*xm_per_pix, 2)
    left_curverad   = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad  = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    left_line.radius_of_curvature  = left_curverad
    right_line.radius_of_curvature = right_curverad

    left_line.line_base_pos = (midpoint - left_fitx[0]) * xm_per_pix
    right_line.line_base_pos = (midpoint - right_fitx[0]) * xm_per_pix

    if verbose or show_img:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_line_window1   = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2   = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts       = np.hstack((left_line_window1, left_line_window2))

        right_line_window1  = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2  = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts      = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig("result_lines.jpg")
        plt.show()

        curvature_txt    = "Radius of curvature " + str.format("{0:.2f}",left_curverad) + " m"
        curvature_txt_r  = "Radius of curvature " + str.format("{0:.2f}",right_curverad) + " m"

        print(curvature_txt_r)

        # off center
        midpoint_lines = (right_fitx[0] - left_fitx[0])/2
        off_center     = np.absolute(midpoint - midpoint_lines) * xm_per_pix

        off_center_txt = "Off center " + str.format("{0:.2f}",off_center) + " m"

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left  = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts       = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0))

    Minv = inv(M)

    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    result  = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    if verbose or show_img:
        cv2.putText(result,curvature_txt,(1,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(result,off_center_txt,(1,200), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
        plt.imshow(result)
        plt.savefig("result.jpg")
        plt.show()

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced lane line finding')
    parser.add_argument(
        'file',
        type=str,
        help='path to image or video file.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        default=False,
        action='store_true',
        help='verbose mode with lots of plots.'
    )

    args    = parser.parse_args()
    verbose = args.verbose

    if "mp4" in args.file:
        from moviepy.editor import VideoFileClip
        from IPython.display import HTML

        output          = 'result_' + args.file
        video_file      = VideoFileClip(args.file)
        process_clip    = video_file.fl_image(process_frame)
        process_clip.write_videofile(output, audio=False)

    if "jpg" in args.file:
        image_test(args.file)