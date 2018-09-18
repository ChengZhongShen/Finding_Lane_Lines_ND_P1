
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import os

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2-x1 == 0:
                continue
            k = (y2-y1)/(x2-x1)
            if abs(k) < k_threshold_low:
                continue
                # print("exclude line k: ", k, "line: ", line)
            else:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    # check for the line drawing on 
#     left = []
#     right = []
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             if (x2-x1) == 0: # horizental line, exclude
#               pass
#             else:
#               k = (y2-y1)/(x2-x1)
#               if (y2-y1)/(x2-x1)< 0:  # note in the img coordinate, the k < 0 is left
#                   left.append((x1,y1,x2,y2,k))
#               else:
#                   right.append((x1,y1,x2,y2,k))
#     # print(left)

#     # for line in left:
#     #     print(line)
#     #     print(type(line))

#     #     x1,y1,x2,y2,k = line
#     #     cv2.line(img,(x1,y1),(x2,y2), (0,255,0),2)


def draw_lines_extrapolate(img, lines, color=[255, 0, 0], thickness=6):
    """
    draw the right/left lane
    """
    left, right = [],[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (x2-x1) == 0: # handle the zero issue
                pass
            else: #exclude the line which abs(k) < k_threshold_low
                k = (y2-y1)/(x2-x1)
                if k< -k_threshold_low and k > -k_threshold_high:   # note in the img coordinate, the k < 0 is left
                    left.append(line)               
                elif k > k_threshold_low and k < k_threshold_high:
                    right.append(line)

    # left, right = lines_check(left), lines_check(right)

    if not (left and right):    # handle the issue not catch the lines
        return img

    # print("left lines after check: ", left)

    left_lane, right_lane = line_fit(left, img), line_fit(right,img)
    # print("left lane is: ", left_lane)
    # print("right lane is : ", right_lane)

    cv2.line(img, left_lane[0], left_lane[1], color, thickness)
    cv2.line(img, right_lane[0], right_lane[1], color, thickness)


def lines_check(lines):
    # print("runging lines_check...")
    # print(lines)
    while lines:
        ks = [(line[0,1]-line[0,3])/(line[0,0]-line[0,2]) for line in lines]
        ks_error = [abs(k-np.mean(ks)) for k in ks]
        # print(ks_error)
        if max(ks_error) < k_diff_threshold:
            break

        idx = ks_error.index(max(ks_error))
        ks_error.pop(idx)
        lines.pop(idx)
    # print(lines)

    return lines

def line_fit(lines,img):
    """
    input: lines, img
    output: top, bot points of lane
    """
    print("runing line_fit...")
    x, y = [],[]
    print("input lines: ", lines)
    for line in lines:
        x.append(line[0,0])
        x.append(line[0,2])
        y.append(line[0,1])
        y.append(line[0,3])
    line_f = np.poly1d(np.polyfit(y,x,1)) # need use y to calcul x, so fit(y, x)
    x_top = int(line_f(0))
    x_bot = int(line_f(img.shape[0]))
    return ((x_top,0),(x_bot,img.shape[0]))


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    # draw_lines_extrapolate(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def HSV_mask(img, threshold):
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(HSV, threshold[0], threshold[1])
    return mask

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.


    
def test_img(file):
    
    img = mpimg.imread(file)

    img_gray = grayscale(img)
    blur_gray = gaussian_blur(img_gray, kernel_size)
    edges = canny(blur_gray, low_threshold, high_threshold)
    masked_edges = region_of_interest(edges, vertices)
    
    # masked_edges[color_mask==0] = 0 # apply color mask
    # white_mask = HSV_mask(img, white_threshold)
    # yellow_mask = HSV_mask(img, yellow_threshold)
    # color_mask = white_mask | yellow_mask
    # plt.figure()
    # plt.imshow(color_mask,cmap="gray")
    
    lined_edges = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    plt.figure()
    plt.imshow(lined_edges)

    masked_lined_edges = region_of_interest(lined_edges, vertices)
    plt.figure()
    plt.imshow(masked_lined_edges)

    lined_img = weighted_img(masked_lined_edges, img, α=0.8, β=1., γ=0.)

    plt.figure()
    plt.imshow(edges,cmap='gray')

    plt.figure()
    plt.imshow(masked_edges,cmap='gray')

    plt.figure()
    plt.imshow(lined_edges)

    plt.figure()
    plt.imshow(lined_img)

    plt.subplot(131),plt.imshow(lined_edges),plt.title("Left/Right Lane")
    plt.subplot(132),plt.imshow(masked_lined_edges),plt.title("Lanes Masked")
    plt.subplot(133),plt.imshow(lined_img),plt.title("Annotated Img")
        


    # plt.subplot(231),plt.imshow(img),plt.title("Original")
    # plt.subplot(232),plt.imshow(img_gray,cmap="gray"),plt.title("Gray Img")
    # plt.subplot(233),plt.imshow(blur_gray,cmap="gray"),plt.title("blur_gray")
    # plt.subplot(234),plt.imshow(edges,cmap="gray"),plt.title("Edges")
    # plt.subplot(235),plt.imshow(masked_edges,cmap="gray"),plt.title("Edges ROI")
    # plt.subplot(236),plt.imshow(masked_lined_edges),plt.title("Draw Lines")





    plt.show()

def img_lane_detect(img):
	"""
	This function is used to package the lane detect pipline, all the paramter should set/ajust out of this function.
	"""
	white_mask = HSV_mask(img, white_threshold)
	yellow_mask = HSV_mask(img, yellow_threshold)
	color_mask = white_mask | yellow_mask

	img_gray = grayscale(img)
	blur_gray = gaussian_blur(img, kernel_size)
	edges = canny(blur_gray, low_threshold, high_threshold)
	masked_edges = region_of_interest(edges, vertices)
	masked_edges[color_mask==0] = 0 # apply color mask
	lined_edges = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
	masked_lined_edges = region_of_interest(lined_edges, vertices)

	return weighted_img(masked_lined_edges, img, α=0.8, β=1., γ=0.)


def test_images(folder_src,folder_dst="test_images_output"):
    for file in os.listdir(folder_src):
        if file == "challenge": # skip the challenge folder
            continue
        imgpath = os.path.join(folder_src,file)
        img = mpimg.imread(imgpath)
        lined_edges = img_lane_detect(img)
        output_imgpath = os.path.join(folder_dst,file)
        print(output_imgpath)



        # swith the RGB to BGR
        r,g,b = lined_edges[:,:,0], lined_edges[:,:,1], lined_edges[:,:,2]
        lined_edges = np.dstack((b,g,r))
        
        #draw ROI line
        # cv2.line(lined_edges, (160, 720),(615,430), (255,0,0),2)
        # cv2.line(lined_edges, (615,430),(700,430), (255,0,0),2)
        # cv2.line(lined_edges, (700,430),(1210, 720), (255,0,0),2)

        cv2.imwrite(output_imgpath, lined_edges) # matplotlib.image not support save jpg file

#################################################################################################
# set the paramters
kernel_size = 5 # paramter for guassian_blur

low_threshold = 50 # paramter for edge finding
high_threshold = 150

# paramter for image masking
imshape = [540,960]
vertices = np.array([[(80, imshape[0]),(410,330),(550,330),(900, imshape[0])]],dtype=np.int32)

# parameter for hough function
rho = 1 # angular resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15 # minimum number of pixels making up a line
min_line_len = 40 # minumum number of pixels making up a line
max_line_gap = 20 # maximum gap in pixels making up a line

k_threshold_low = 0.5   # used to exclude the horizental line
k_threshold_high = 100  # setted not adjusted
k_diff_threshold = 0.1  # in lines check, to sorted out the lines which's slope is different with others

# parameter for color mask HSV value
white_threshold = ((60,0,130),(180,50,255))
yellow_threshold = ((10,60,0),(30,255,255))

#################################################################################################

# test_img("test_images/solidWhiteRight.jpg")

test_images("test_images")