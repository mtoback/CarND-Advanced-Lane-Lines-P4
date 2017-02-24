import cv2
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage.feature._canny import canny
from tracker import tracker

def get_video_reader(file_name):
    return imageio.get_reader(file_name, 'ffmpeg')
    
def get_image_from_mp4(reader , num):
    image = None
    try:
        image = reader.get_data(num)
    except:
        pass
    return image

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude 
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    arctan_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(arctan_sobel)
    binary_output[(arctan_sobel >= thresh[0]) & (arctan_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def color_threshold(image, s_threshold = (0, 255), v_threshold = (0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_threshold[0]) & (s_channel <= s_threshold[1])] = 1
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_threshold[0]) & (v_channel <= v_threshold[1])] = 1
    
    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - 
               (level+1)*height) : int(img_ref.shape[0] - level*height),
                                max(0, int(center - width)):
                                min(int(center + width), img_ref.shape[1])] = 1
    return output

def run_video(mtx, dist, video_name):
    ksize=3
    reader = get_video_reader(video_name)
    num = 0
    image = get_image_from_mp4(reader, num)
    while image != None:
        image = img_undistort(image, mtx, dist)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 100))
        grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
        c_binary = color_threshold(image, s_threshold=(100,255), v_threshold=(50, 255))
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        plt.plot(combined)
        plt.show()
        num = num + 1
        image = get_image_from_mp4(reader, num)
    
def process_images(mtx, dist, file_name, ksize=3):
    # get the list of names
    images = glob.glob(file_name)
    for idx, fname in enumerate(images):
        # read file and undistort it
        image = cv2.imread(fname)
        # A function that takes an image, object points, and image points
        # performs the camera calibration, image distortion correction and 
        # returns the undistorted image
        image = cv2.undistort(image, mtx, dist, None, mtx)
        preprocessImage = np.zeros_like(image[:,:,0])
        # calculate the various thresholds
        gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(12, 255))
        grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(25, 255))
        c_binary = color_threshold(image, s_threshold=(100,255), v_threshold=(50, 255))
        # form a combination of thresholds
        preprocessImage[((gradx == 1) & (grady == 1)) | (c_binary == 1)] = 255
        
        # work on defining perspective transformation area
        img_size = (image.shape[1], image.shape[0])
        bot_width = 0.85 # oercebt of bottom trapezoid height
        mid_width = 0.12 # percent of middle trapezoid height
        height_pct = 0.62 # percent for trapezoid height
        bottom_trim = 0.935 # percent from top to bottom to avoid car hood
        src = np.float32([[image.shape[1]*(0.5 - mid_width/2), image.shape[0]*height_pct],
                          [image.shape[1]*(0.5 + mid_width/2), image.shape[0]*height_pct],
                          [image.shape[1]*(0.5 + bot_width/2), image.shape[0]],
                          [image.shape[1]*(0.5 - bot_width/2), image.shape[0]]])
        '''
        cv2.line(image, (src[0][0],src[0][1] ), (src[1][0], src[1][1]),( 110, 220, 0 ),5)
        cv2.line(image, (src[1][0],src[1][1] ), (src[2][0], src[2][1]),( 110, 220, 0 ),5)
        cv2.line(image, (src[2][0],src[2][1] ), (src[3][0], src[3][1]),( 110, 220, 0 ),5)
        cv2.line(image, (src[3][0],src[3][1] ), (src[0][0], src[0][1]),( 110, 220, 0 ),5)
        '''
        offset = img_size[0]*0.25
        dst = np.float32([[offset,0], 
                          [img_size[0] - offset, 0],
                          [img_size[0] - offset, img_size[1]],
                          [offset, img_size[1]]])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)
        
        window_width = 25
        window_height = 80
        
        # set up the overall class to do all the tracking
        curve_centers = tracker(Mywindow_width = window_width,
                                Mywindow_height = window_height,
                                Mymargin = 25,
                                My_ym = 10/720,
                                My_xm = 4/384,
                                Mysmooth_factor = 15)
        window_centroids = curve_centers.find_window_centroids(warped)
        
        # points used to draw all  the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
        
        # points used to find the left and right lanes
        rightx = []
        leftx =  []
        
        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # window_mask is a function to draw window areas
            l_mask = window_mask(window_width,
                                 window_height,
                                 warped,
                                 window_centroids[level][0],
                                 level)
            r_mask = window_mask(window_width,
                                 window_height,
                                 warped,
                                 window_centroids[level][1],
                                 level)
            # add center value found in frame to the list of lane point per left, right
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])
            
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1)) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1)) ] = 255
        
        # Draw the results
        template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8) # making the original road pixels 3 color channels
        result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the original road image with window results
        
        # Fit the lane boundaries to the left, right center positions found
        # yvals is height
        yvals = range(0, warped.shape[0]) 
        
        # fit to box centers
        res_yvals = np.arange(warped.shape[0] - (window_height/2),
                              0, -window_height)
        
        # polynomial fit to a second order polynomial
        left_fit = np.polyfit(res_yvals, leftx, 2)
        left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
        left_fitx = np.array(left_fitx, np.int32)
        
        # polynomial fit to a second order polynomial
        right_fit = np.polyfit(res_yvals, rightx, 2)
        right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
        right_fitx = np.array(right_fitx, np.int32)


        left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2,
                            left_fitx[::-1] + window_width/2), axis=0),
                            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
        right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width/2, 
                            right_fitx[::-1] + window_width/2), axis=0),
                            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
        middle_marker = np.array(list(zip(np.concatenate((right_fitx - window_width/2, 
                            right_fitx[::-1] + window_width/2), axis=0),
                            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
        
        # color in the roads
        road = np.zeros_like(image)
        #road_bkg = np.zeros_like(img)
        cv2.fillPoly(road,[left_lane], color=[255, 0, 0])
        cv2.fillPoly(road,[right_lane], color=[0, 0, 255])
        #cv2.fillPoly(road_bkg,[left_lane], color=[255, 255, 255])
        #cv2.fillPoly(road,[right_lane], color=[255, 255, 255])

        write_name = './test_images/tracked' + str(idx) + '.jpg'
        cv2.imwrite(write_name, road)
        
def main():
    data = pickle.load( open('./calibration_pickle.p', 'rb'))
    mtx = data['mtx']
    dist = data['dist']
    process_images(mtx, dist,'./test_images/test*.jpg')
    
if __name__ == "__main__":
    main()
        
