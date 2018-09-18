# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./examples/pipline.png "Pipline"
---


### 1. The pipeline.

Pipeline as below:
1. convert the image from RGB image to gray
2. use gaussian_blur fucntion to blue the grayscale image
3. use Canny() function to detect the edges
4. use mask to get the ROI(region of interest) of edge
5. use Hough function to find lines in the ROI area
6. draw the left/right lane(detail in below) 
![alt text][image2]

### 2. Draw the line

The Hough function will return the pionts of the linse find in the ROI.

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
