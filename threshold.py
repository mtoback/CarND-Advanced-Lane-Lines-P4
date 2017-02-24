import cv2
import numpy as np
class Threshold():
    def __init__(self, kernel=3):
        self.kernel = kernel
    def abs_sobel_thresh(self, img, orient='x', thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel)
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
    
    def mag_thresh(self, image, mag_thresh=(0, 255)):
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
    
    def dir_threshold(self, image, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, self.kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, self.kernel)
        # 3) Take the absolute value of the x and y gradients
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        arctan_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(arctan_sobel)
        binary_output[(arctan_sobel >= thresh[0]) & (arctan_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary_output
    
    def color_threshold(self, image, s_threshold = (0, 255), v_threshold = (0, 255)):
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
