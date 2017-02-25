import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage.feature._canny import canny
from threshold import Threshold
from tracker import tracker

class Image():
    def __init__(self, ksize=3, debug=False):
        data = pickle.load( open('./calibration_pickle.p', 'rb'))
        self.mtx = data['mtx']
        self.dist = data['dist']
        self.kernel = ksize
        self.debug = debug
        self.bot_width = 0.85 # oercebt of bottom trapezoid height
        self.mid_width = 0.12 # percent of middle trapezoid height
        self.height_pct = 0.62 # percent for trapezoid height
        self.bottom_trim = 0.935 # percent from top to bottom to avoid car hood
        self.threshold = Threshold(3)
        self.first_image = True
    
    def window_mask(self, width, height, img_ref, center, level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0] - 
                   (level+1)*height) : int(img_ref.shape[0] - level*height),
                                    max(0, int(center - width)):
                                    min(int(center + width), img_ref.shape[1])] = 1
        return output
            
    def draw_trapezoid(self,image, src):
        cv2.line(image, (src[0][0],src[0][1] ), (src[1][0], src[1][1]),( 110, 220, 0 ),5)
        cv2.line(image, (src[1][0],src[1][1] ), (src[2][0], src[2][1]),( 110, 220, 0 ),5)
        cv2.line(image, (src[2][0],src[2][1] ), (src[3][0], src[3][1]),( 110, 220, 0 ),5)
        cv2.line(image, (src[3][0],src[3][1] ), (src[0][0], src[0][1]),( 110, 220, 0 ),5)
        return image

    def warp_image(self,image):
        preprocessImage = np.zeros_like(image[:,:,0])
        # calculate the various thresholds
        gradx = self.threshold.abs_sobel_thresh(image, orient='x', thresh=(12, 255))
        grady = self.threshold.abs_sobel_thresh(image, orient='y',  thresh=(25, 255))
        c_binary = self.threshold.color_threshold(image, s_threshold=(100,255), v_threshold=(50, 255))
        # form a combination of thresholds
        preprocessImage[((gradx == 1) & (grady == 1)) | (c_binary == 1)] = 255
        
        # work on defining perspective transformation area
        img_size = (image.shape[1], image.shape[0])
        src = np.float32([[image.shape[1]*(0.535 - self.mid_width/2), image.shape[0]*self.height_pct],
                          [image.shape[1]*(0.5 + self.mid_width/2), image.shape[0]*self.height_pct],
                          [image.shape[1]*(0.495 + self.bot_width/2), self.bottom_trim*image.shape[0]],
                          [image.shape[1]*(0.56 - self.bot_width/2), self.bottom_trim*image.shape[0]]])
        
        #for debugging, draw the trapezoid on the image
        if self.debug:
            image = self.draw_trapezoid(image, src)
        offset = img_size[0]*0.25
        dst = np.float32([[offset,0], 
                          [img_size[0] - offset, 0],
                          [img_size[0] - offset, img_size[1]],
                          [offset, img_size[1]]])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)
        if self.first_image:
            self.first_image = False
            write_name = 'preprocessed_image.jpg'
            cv2.imwrite(write_name, preprocessImage)
            write_name = 'source_image.jpg'
            image = self.draw_trapezoid(image, src)
            cv2.imwrite(write_name, image)
            write_name = 'sample_warped_image.jpg'
            dest_warped = self.draw_trapezoid(warped, dst)
            cv2.imwrite(write_name, dest_warped)
        return (warped, M, Minv, src, dst)
    
    def calc_curves(self, image):
        # A function that takes an image, object points, and image points
        # performs the camera calibration, image distortion correction and 
        # returns the undistorted image
        image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        
        (warped, M, Minv, src, dst) = self.warp_image(image)
        img_size = (image.shape[1], image.shape[0])

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
            l_mask = self.window_mask(window_width,
                                 window_height,
                                 warped,
                                 window_centroids[level][0],
                                 level)
            r_mask = self.window_mask(window_width,
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
        inner_lane = np.array(list(zip(np.concatenate((left_fitx + window_width/2, 
                            right_fitx[::-1] - window_width/2), axis=0),
                            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
        middle_marker = np.array(list(zip(np.concatenate((right_fitx + window_width/2, 
                            right_fitx[::-1] + window_width/2), axis=0),
                            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
        
        # color in the lines (road) and clear out in the original lines (road_bkg)
        road = np.zeros_like(image)
        road_bkg = np.zeros_like(image)
        cv2.fillPoly(road,[left_lane], color=[255, 0, 0])
        cv2.fillPoly(road,[right_lane], color=[0, 0, 255])
        cv2.fillPoly(road,[inner_lane], color=[0, 255, 0])
        cv2.fillPoly(road_bkg,[left_lane], color=[255, 255, 255])
        cv2.fillPoly(road_bkg,[right_lane], color=[255, 255, 255])
        
        # Overlay lines onto the road surface
        road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
        road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)
    
        base = cv2.addWeighted(image, 1.0, road_warped_bkg, -1.0, 0.0)
        result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)
    
        # calculate the offset of the car on the road and 
        # figure out if it's in the right or left lane
        # and scale to meters
        # the -1 position is closest to the car
        ym_per_pix = curve_centers.ym_per_pix # meters per pixel in y direction
        xm_per_pix = curve_centers.xm_per_pix # meters per pixel in x direction
        
        # we want curvature in meters using the left lane to calculate the value
        # we could do the right, left + right average, or create a whole new middle line
        # see www.intmath.com/applications-differentiation/8-radius-curvature.php
        curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix,
                                  np.array(leftx, np.float32) *xm_per_pix, 2)
        curvead = ((1 + (2* curve_fit_cr[0] * yvals[-1] * ym_per_pix + 
                         curve_fit_cr[1])**2)**1.5)/np.absolute(2*curve_fit_cr[0])
        
        camera_center = (left_fitx[-1] + right_fitx[-1])/2
        center_diff = (camera_center - warped.shape[1]/2)*xm_per_pix
        
        # if camera center is positive we are on the left side of the road
        side_pos = 'left'
        if center_diff <= 0:
            side_pos = 'right'
        thumbnail = cv2.resize(template, (int(0.25*result.shape[0]), int(0.25*result.shape[1])))
        result[0:thumbnail.shape[0],result.shape[1]-thumbnail.shape[1]:result.shape[1]] = thumbnail
        cv2.putText(result, 'Radius of Curvature = ' + str(round(curvead, 3)) +
                    '(m) ', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.putText(result, 'Vehicle is  = ' + str(abs(round(center_diff, 3))) +
                    'm ' + side_pos + ' of center', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        return result
    
    def process_images(self,  file_name, ksize=3):
        # get the list of names
        images = glob.glob(file_name)
        for idx, fname in enumerate(images):
            # read file and undistort it
            image = cv2.imread(fname)
            result = self.calc_curves(image)
            write_name = './test_images/tracked' + str(idx) + '.jpg'
            cv2.imwrite(write_name, result)
        
def main():
    image_proc = Image(3, debug=True)
    image_proc.process_images('./test_images/test*.jpg')
if __name__ == "__main__":
    main()
        
