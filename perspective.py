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

def perspective(img):
    img_orig     = np.copy(img)
    undistorted  = cv2.undistort(img, mtx, dist, None, mtx)

#    imshape = img.shape
#    upper_left=90
#    upper_right=90
#    height=100
#    left_bottom=120
#    right_bottom=120
#    height_bottom=60

#    src = np.float32([[left_bottom, imshape[0] - height_bottom], [imshape[1]/2 - upper_left, imshape[0]/2 + height], [imshape[1]/2 + upper_right, imshape[0]/2 + height], [imshape[1] - right_bottom ,imshape[0] - height_bottom]])
#    vertices1 = np.array(
#        [[
#            (left_bottom, imshape[0] - height_bottom),
#            (imshape[1]/2 - upper_left, imshape[0]/2 + height), 
#            (imshape[1]/2 + upper_right, imshape[0]/2 + height), 
#            (imshape[1] - right_bottom ,imshape[0] - height_bottom)
#        ]]
#        , dtype=np.int32)

    imshape         = (img.shape[1],img.shape[0],img.shape[2])
    height          = 100
    length_top      = 70
    length_bottom   = 500

    p11 = imshape[0] / 2 - length_bottom
    p12 = imshape[1]
    p21 = imshape[0] / 2 - length_top
    p22 = imshape[1] / 2 + height
    p31 = imshape[0] / 2 + length_top
    p32 = imshape[1] / 2 + height
    p41 = imshape[0] / 2 + length_bottom
    p42 = imshape[1]

    vertices1 = np.array(
        [[
            (p11, p12),
            (p21, p22), 
            (p31, p32), 
            (p41, p42)
        ]]
        , dtype=np.int32)

    src = np.float32(
        [
            [p11, p12],
            [p21, p22],
            [p31, p32],
            [p41, p42]
        ]
        )

#    print("Img")
#    print(vertices1)
#    print(vertices1[0][0][0])
#    print(np.sqrt((vertices1[0][1][0] - vertices1[0][0][0])**2 + (vertices1[0][0][0] - vertices1[0][1][0])**2))

#    height = np.int(np.sqrt((vertices1[0][1][0] - vertices1[0][0][0])**2 + (vertices1[0][0][0] - vertices1[0][1][0])**2))
#    upper_left=50
#    upper_right=50
#    height=0
#    left_bottom=120
#    right_bottom=120
#    height_bottom=0
#    dst = np.float32([[left_bottom, imshape[0] - height_bottom], [imshape[1]/2 - upper_left, imshape[0]/2 + height], [imshape[1]/2 + upper_right, imshape[0]/2 + height], [imshape[1] - right_bottom ,imshape[0] - height_bottom]])
#    vertices2 = np.array([[(left_bottom, imshape[0] - height_bottom),(imshape[1]/2 - upper_left, imshape[0]/2 + height), (imshape[1]/2 + upper_right, imshape[0]/2 + height), (imshape[1] - right_bottom ,imshape[0] - height_bottom)]], dtype=np.int32)

#    dst = np.float32([[left_bottom, imshape[0] - height_bottom], [left_bottom, 0], [imshape[1] - right_bottom, 0], [imshape[1] - right_bottom ,imshape[0] - height_bottom]])
#    vertices2 = np.array([[(left_bottom, imshape[0] - height_bottom),(left_bottom, 0), (imshape[1] - right_bottom, 0), (imshape[1] - right_bottom ,imshape[0] - height_bottom)]], dtype=np.int32)
#    dst = np.float32([[left_bottom, imshape[0]], [left_bottom, imshape[0] - height], [imshape[1] - right_bottom, imshape[0] - height], [imshape[1] - right_bottom ,imshape[0]]])

#    vertices2 = np.array(
#        [[
#            (left_bottom, imshape[0]),
#            (left_bottom, imshape[0] - height), 
#            (imshape[1] - right_bottom, imshape[0] - height), 
#            (imshape[1] - right_bottom , imshape[0])
#        ]]
#        , dtype=np.int32)

    p11 = imshape[0] / 9
    p12 = imshape[1]
    p21 = imshape[0] / 9
    p22 = 0
    p31 = imshape[0] * 5 / 6
    p32 = 0
    p41 = imshape[0] * 5 / 6
    p42 = imshape[1]

    vertices2 = np.array(
        [[
            (p11, p12),
            (p21, p22), 
            (p31, p32), 
            (p41, p42)
        ]]
        , dtype=np.int32)

    dst = np.float32(
                [
            [p11, p12],
            [p21, p22],
            [p31, p32],
            [p41, p42]
        ]
        )

#    print(vertices2)

    top_down, M = warper(undistorted, src, dst)

    if len(imshape) > 2:
        channel_count = imshape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    polyimg1 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.fillPoly(polyimg1, vertices1, ignore_mask_color)

    polyimg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.fillPoly(polyimg, vertices2, ignore_mask_color)

    top_down_vert   = cv2.addWeighted(top_down, 1, polyimg, 0.3, 0)
    img_orig        = cv2.addWeighted(img_orig, 1, polyimg1, 0.3, 0)

    return top_down, top_down_vert, img_orig, undistorted, M

if __name__ == "__main__":
    for image in os.listdir(path_image):
        img          = cv2.imread(path_image+image)

        top_down, top_down_vert, img_orig, undist, M  = perspective(img)

        if "perspective" not in image:
            cv2.imwrite("test_images/perspective"+image, top_down)
            cv2.imwrite(path_out+"perspective_vert"+image, top_down_vert)
            cv2.imwrite(path_out+"perspective_orig_vert"+image, img_orig)
            cv2.imwrite(path_out+"perspective_orig"+image, undist)

