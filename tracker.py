import numpy as np
import cv2
from ipywidgets.widgets.widget_controller import Axis
class tracker():
    
    # initialize the class
    # Mywindow_width - image window width
    # Mywindow_height - image window height
    # Mymargin - number of pixels that we are allowing for sliding the window
    # as part of the correlation analysis
    # My_ym - Meters per pixel along the y axis
    # My_xm - Meters per pixel along the x axis
    # Mysmooth_factor - how many prior values to be used to smooth out the center estimate
    def __init__(self, Mywindow_width, Mywindow_height, Mymargin,
                 My_ym = 1, My_xm = 1, Mysmooth_factor = 15):
        # list that stors all the past (left, right) center set values 
        # used for smoothing the output with the smoothing factor
        self.recent_centers = []
        
        # the window pixel length of the center values, used to count pixels 
        # inside center windows to determine curve values
        self.window_width = Mywindow_width
        
        # the windows pixel height of the center values, used to count pixels 
        # inside center windows tp determine curve values
        # breaks the image into vertical levels
        self.window_height = Mywindow_height
        
        # the pixel distance in both directions that the window is allowed to slide 
        # (left_window + right_window) template for searching
        self.margin = Mymargin
        
        self.ym_per_pix = My_ym # meters per pixel in vertical Axis
        
        self.xm_per_pix = My_xm # meters per pixel in horizontal Axis
        
        self.smooth_factor = Mysmooth_factor
        
    # the main tracking function for finding and storing lane segment positions
    # use convolution as a smoothing factor looking for peaks along a given axis
    # this is like a 1D convolution. Helpful for tracking down pixel ranges
    def find_window_centroids(self, warped):
        # initialize the window height and margin
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        
        window_centroids = [] # store the (left,right) window centroid positions per level
        # so breaking up the image into 9 vertical slices, so we will have 9 convolutional pairs
        # Create our own window template that will will use for convolutions
        window = np.ones(window_width) 
        
        # First find the two starting positions for the left and right line by using 
        # np.sum to get the vertical image slice, and then use np.convolve to convolve
        # the vertical image slice with the window template
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        # we create the histogram by squashing the signal using np.sum
        l_sum = np.sum(warped[int(3 * warped.shape[0]/4) : , :int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum )) - window_width/2
        r_sum = np.sum(warped[int(3 * warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum )) - window_width/2 + int(warped.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))
        
        # Go through the other layers looking for max pixel locations
        for level in range(1, (int)(warped.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0] - 
                            (level + 1) * window_height) : int(warped.shape[0] -
                                                level*window_height), :],axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a Reference
            # Use window_width/2 as offset because 
            # convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index : l_max_index]) + l_min_index - offset
            # Find the best right centroid by using past right center as a reference 
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index : r_max_index]) + r_min_index - offset 
            # add what we found for that layer 
            window_centroids.append((l_center, r_center))
            
        self.recent_centers.append(window_centroids)
        # return averaged values of the line centers, helps to keep the markers from jumping around too much
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)