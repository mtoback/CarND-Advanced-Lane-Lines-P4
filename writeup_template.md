##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration10.jpg "Original Distorted"
[image1]: ./undistored_calibration10.jpg "Undistorted checkerboard"
[image2]: ./test_images/test1.jpg "Distorted Test Image"
[image3]: ./preprocessed_image.jpg "Thresholded Image"
[image4]: ./source_image.jpg "Pre-Warp Image"
[image5]: ./sample_warped_image.jpg "Warp Image"
[image6]: ./test_images/tracked0.jpg "Undistorted Test Image"
[image7]: ./examples/color_fit_lines.jpg "Fit Visual"
[image8]: ./test_images/tracked0.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Camera Calibration

####1. Computing the camera matrix and distortion coefficients, providing an example of a distortion corrected calibration image.

The code for this step is contained in a file called `camera_calibrate.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Using the findChessboardCorners function to make sure that I could find the corners (a few of the images did not have the requisite 9x6 checkerboard visible and so were thrown out). For each valid image, I collected the corners and objpoints and appended them to an array for later processing. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

I applied this distortion correction to one test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image0] 
![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #41 through #43 in `image.py`).  Here's an example of my output for this step:

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 39 through 76 in the file `image.py` (./image.py).  The `warp_image()` function takes as inputs an image (`image`).  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
	[[image.shape[1]*(0.535 - self.mid_width/2), image.shape[0]*self.height_pct],
    [image.shape[1]*(0.5 + self.mid_width/2), image.shape[0]*self.height_pct],
    [image.shape[1]*(0.495 + self.bot_width/2), self.bottom_trim*image.shape[0]],
    [image.shape[1]*(0.56 - self.bot_width/2), self.bottom_trim*image.shape[0]]])

dst = np.float32([[offset,0], 
    [img_size[0] - offset, 0],
    [img_size[0] - offset, img_size[1]],
    [offset, img_size[1]]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 608.0, 446.4  | 320, 0        | 
| 716.8, 446.4  | 960, 0        |
| 1177.6, 673.2 | 960, 720      |
| 172.8, 673.2  | 320, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

See lines 85-166 in the calc_curves method in image.py. 

First we create a tracker object (see tracker.py) and clculate the window centroids for the image. What we are doing here is convolving a window across the warped image to find the lane lines. Once we do that, we fill in the lane lines and center lane.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #183 through #196 in my calc_curves method in `image.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image8]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result. the raw file found at output_images/project_video.mp4](https://youtu.be/HPieR5GHkVE)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

