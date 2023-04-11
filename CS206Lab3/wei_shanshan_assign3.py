import os
os.chdir(os.path.dirname(__file__))


import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import io
# part I

# Bugs
# No bugs encountered in this part

# Read the image and get the image shape
img = io.imread('PeppersBayerGray.bmp', as_gray=True)
h, w = img.shape

# our final image will be a 3 dimentional image with 3 channels
rgb = np.zeros((h, w, 3), np.uint8);

# reconstruction of the green channel IG

IG = np.copy(img)  # copy the image into each channel

for row in range(0, h, 4):  # loop step is 4 since our mask size is 4.
    # loop step is 4 since our mask size is 4
    for col in range(0, w, 4):
        # Base on the gridG and the formula in assignment submission page, calculate the pixel value for the corresponding points
        IG[row, col + 1] = (int(img[row, col]) + int(img[row, col + 2])) / 2
        IG[row + 1, col] = (int(img[row, col]) + int(img[row + 2, col])) / 2
        IG[row + 3, col] = (int(img[row + 2, col]) + int(img[row + 3, col + 1])) / 2
        IG[row, col + 3] = (int(img[row, col + 2]) + int(img[row + 1, col + 3])) / 2
        IG[row + 1, col + 2] = (int(img[row, col + 2]) + int(img[row + 1, col + 1]) + int(img[row + 1, col + 3]) + int(
            img[row + 2, col + 2])) / 4  # G
        IG[row + 2, col + 1] = (int(img[row + 2, col]) + int(img[row + 1, col + 1]) + int(img[row + 2, col + 2]) + int(
            img[row + 3, col + 1])) / 4  # J
        IG[row + 2, col + 3] = (int(img[row + 1, col + 3]) + int(img[row + 3, col + 3])) / 2
        IG[row + 3, col + 2] = (int(img[row + 3, col + 1]) + int(img[row + 3, col + 3])) / 2

    # reconstruction of the red channel IR

IR = np.copy(img)
IR = IR.astype(np.float32)
for row in range(0, h, 4):  # loop step is 4 since our mask size is 4.
    for col in range(0, w, 4):  # loop step is 4 since our mask size is 4.
        # Base on the gridR and the formula in assignment submission page, calculate the pixel value for the corresponding points
        IR[row + 1, col + 1] = (int(img[row, col + 1]) + int(img[row + 2, col + 1])) / 2
        IR[row, col + 2] = (int(img[row, col + 1]) + int(img[row, col + 3])) / 2
        IR[row + 1, col + 3] = (int(img[row, col + 3]) + int(img[row + 2, col + 3])) / 2
        IR[row + 2, col + 2] = (int(img[row + 2, col + 3]) + int(img[row + 2, col + 1])) / 2
        IR[row + 1, col + 2] = (int(img[row, col + 1]) + int(img[row, col + 3]) + int(img[row + 2, col + 1]) + int(
            img[row + 2, col + 3])) / 4
        IR[row, col] = IR[row, col + 1]
        IR[row + 1, col] = IR[row + 1, col + 1]
        IR[row + 2, col] = IR[row + 2, col + 1]
        IR[row + 3, col + 3] = IR[row + 2, col + 3]
        IR[row + 3, col + 2] = IR[row + 2, col + 2]
        IR[row + 3, col + 1] = IR[row + 2, col + 1]
        IR[row + 3, col] = IR[row + 2, col + 1]

    # reconstruction of the blue channel IB
IB = np.copy(img)  # copy the image into each channel

IB = IB.astype(np.float32)
for row in range(0, h, 4):  # loop step is 4 since our mask size is 4.
    for col in range(0, w, 4):  # loop step is 4 since our mask size is 4.
        # Base on the gridB and the formula in assignment submission page, calculate the pixel value for the corresponding points
        IB[row + 1, col + 1] = (int(img[row + 1, col]) + int(img[row + 1, col + 2])) / 2
        IB[row + 1, col + 3] = IB[row + 1, col + 2]
        IB[row, col] = IB[row + 1, col]
        IB[row, col + 1] = IB[row + 1, col + 1]
        IB[row, col + 2] = IB[row + 1, col + 2]
        IB[row, col + 3] = IB[row + 1, col + 3]
        IB[row + 2, col] = (int(img[row + 1, col]) + int(img[row + 3, col])) / 2
        IB[row + 2, col + 1] = (int(img[row + 1, col]) + int(img[row + 1, col + 2]) + int(img[row + 3, col]) + int(
            img[row + 3, col + 2])) / 4
        IB[row + 2, col + 2] = (int(img[row + 1, col + 2]) + int(img[row + 3, col + 2])) / 2
        IB[row + 2, col + 3] = IB[row + 2, col + 2]
        IB[row + 3, col + 1] = (int(img[row + 3, col]) + int(img[row + 3, col + 2])) / 2
        IB[row + 3, col + 3] = IB[row + 3, col + 2]

    # merge the channels
rgb[:, :, 0] = IR
rgb[:, :, 1] = IG
rgb[:, :, 2] = IB

# Show the merged image
plt.imshow(rgb), plt.title('rgb')
plt.show()




# part2

import os
from sklearn.cluster import KMeans
from scipy import spatial
from skimage import io, color, img_as_float
import numpy as np
import matplotlib.pyplot as plt
from math import floor
# Bugs
# There is index problem with FloydSteinbergDitherColor function. Then i found the index under two for loops will reach the last row and column, so it has to plus 1 for forloop range
# citation: https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering

# Finds the closest colour in the palette using kd-tree.
def nearest(palette, colour):
    dist, i = palette.query(colour)
    return palette.data[i]


# Make a kd-tree palette from the provided list of colours
def makePalette(colours):
    # print(colours)
    return spatial.KDTree(colours)


# Dynamically calculates and N-colour palette for the given image
# Uses the KMeans clustering algorithm to determine the best colours
# Returns a kd-tree palette with those colours
def findPalette(image, nColours):
    h, w, d = image.shape
    d2_image = image.reshape(h * w, d)
    kmeans = KMeans(n_clusters=nColours)
    kmeans.fit(d2_image)
    colours = kmeans.cluster_centers_
    return makePalette(colours)

    return makePalette(colours)


def FloydSteinbergDitherColor(image, palette):
    # ***** The following pseudo-code is grabbed from Wikipedia: https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering.
    pixel = np.copy(image)
    for y in range(image.shape[0] - 1):
        for x in range(image.shape[1] - 1):
            oldpixel = pixel[x, y]
            newpixel = nearest(palette, oldpixel)
            pixel[x][y] = newpixel
            quant_error = oldpixel - newpixel
            pixel[x + 1][y] = pixel[x + 1][y] + quant_error * (np.float(7 / 16))
            pixel[x - 1][y + 1] = pixel[x - 1][y + 1] + quant_error * (np.float(3 / 16))
            pixel[x][y + 1] = pixel[x][y + 1] + quant_error * (np.float(5 / 16))
            pixel[x + 1][y + 1] = pixel[x + 1][y + 1] + quant_error * (np.float(1 / 16))

    return pixel


if __name__ == "__main__":
    nColours = 8  # The number colours: change to generate a dynamic palette

    imfile = 'lena.png'

    image = io.imread(imfile)

    # Strip the alpha channel if it exists
    image = image[:, :, :3]

    # Convert the image from 8bits per channel to floats in each channel for precision
    image = img_as_float(image)

    # Dynamically generate an N colour palette for the given image
    palette = findPalette(image, nColours)
    colours = palette.data

    colours = img_as_float([colours.astype(np.ubyte)])[0]

    img = FloydSteinbergDitherColor(image, palette)

    plt.imshow(img)
    plt.show()








# part3
# Import libraries

# Bugs
# When i tried to use the function estimate_transform from lecture, the output image is just black.
# Then i went to the official documentation to check whether it is parameter problem
# I found another function AffineTransform from the skimage library whichis very efficient and allows to input the rotation and scaling as parameters
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from skimage.transform import warp

# read tbe image and find the surface shape
image1 = io.imread('lab5_img.jpeg')
h = image1.shape[0]
w = image1.shape[1]
# create the rotation matrix of clockwise 90 degree
T_r = np.array([[0, 1], [-1, 0]])
# create the scaling matrix with scale of 2
T_s = np.array([[2, 0], [0, 2]])
# dot product of two transformation matrix to get a combined transformation matrix
combined = T_s.dot(T_r)

# initialize an empty numpy array. Since this is color image, not gray image, the parameter should be (h,w,3)
trans = np.zeros((450, 450, 3), dtype=np.uint8)
# two for loops go through every pixel of the image and apply transformation to every pixel location by dot product of combined transformation and transpose of pixel coordinate
for i in range(h):
    for j in range(w):
        pixel = np.array([i, j])
        pix_val = image1[i, j]
        # print(pix_val)
        new_pix = combined.dot(pixel.T)
        a = new_pix[0]
        b = new_pix[1]
        # print((a,b))
        # assign the pixel value of original location to the new location pixel
        trans[a, b] = pix_val

# initialize a new numpy array for the color image
imageAffine = np.empty((h * 2, w * 2, image1.shape[2]), dtype=np.uint8)
# use AffinTransform function from skimage.tranform module to get a affine transform matrix
# input parameters are the rotations and scales we apply on the image
tform = transform.AffineTransform(scale=(2, 2), rotation=np.deg2rad(90), translation=(w * 2, 0))
# use warp from skimage.transform to Warp an image according to the given affine inverse transformation.
img_warped = warp(image1, tform.inverse, output_shape=imageAffine.shape, order=0)

# plot original image
plt.title('Original Image')
plt.imshow(image1, cmap='gray')
plt.show()

# plot transformed image using combine transformation
plt.title('Transformed')
plt.imshow(trans)
plt.show()

# plot affine tranformed image
plt.title('Affine')
plt.imshow(img_warped)
plt.show()



# part 4
# Import libraries
from skimage import io
from skimage import exposure
import skimage
from skimage.color import rgb2gray
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform



# Bugs
# I used other detectors, they cause crashes except ORB
# There is a lot of bugs encountered in this part, since this part needs me to study some new functions from library
# In ransac function, i messed up with the input parameters, since there is two data sets to be used. I solved this bug
# by clearly reading the official documentation about this function and some example code others write using this function


# For ploting the final image, i used the normal plot ar the beginning, but it failed to show the final image

# There is a problem with my code that i can not fix. I don't know why every time i execute my code, the output image will be different.
# Some output are very similar to the required output, some are very different.
# I though it may be some functions problem like ORB/ransac. I don't think i am able to fix it

# read the two images
image0 = io.imread('im1.jpg', True)
image1 = io.imread('im2.jpg', True)

# plot two images
plt.imshow(image0,cmap='gray')
plt.show()
plt.imshow(image1,cmap='gray')
plt.show()
#Feature detection and matching

# Initiate ORB detector
# ORB is feature function in skimage module. The parameter n_keypoints is the number of keypoints to be returned.
ORB_detector = ORB(n_keypoints=500)


# Find the keypoints and descriptors
# detect_and_extract with parameter of a image is a function that ORB dectector has. It is used to detect oriented fast keypoints and extract rBRIEF descriptors.
ORB_detector.detect_and_extract(image0)
keypoints1 = ORB_detector.keypoints
descriptors1 = ORB_detector.descriptors
ORB_detector.detect_and_extract(image1)
keypoints2 = ORB_detector.keypoints
descriptors2 = ORB_detector.descriptors

# initialize Brute-Force matcher and exclude outliers. See match descriptor function.
# The match descriptor function given here is to find the closest descriptor in the second set for each descriptor in the first set this matcher  (and vice-versa in the case of enabled cross-checking).
# It takes descriptors1, descriptors2 as main parameters. max_distance as its restriction to seek for the closest descripters in another set of descriptor.  cross_check is true, so it will return matched pairs
# citation https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.match_descriptors
match =  match_descriptors(descriptors1,descriptors2, max_distance = 10,cross_check=True)
#print(match)

# Compute homography matrix using ransac and ProjectiveTransform
# model_robust, inliers = ransac ...
# https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.ransac
# the robust estimation of parameters from a subset of inliers from the complete data set.
# parameter1: the data sets the model is fitted. since this is multiple data sets, it can be passed with tuple
# parameter2: transform methods 'ProjectiveTrans' as requested. There could be other transform like AffinTrans
# parameter3: min_samples: The minimum number of data points to fit a model to.
# parameter4: residual_threshold: Maximum distance for a data point to be classified as an inlier.
data1 = keypoints1[match[:,0]][:,::-1]
data2 = keypoints2[match[:,1]][:,::-1]
model_robust, inliers = ransac((data2,data1), ProjectiveTransform, min_samples=3, residual_threshold=2)


#Warping
#Next, we produce the panorama itself. The first step is to find the shape of the output image by considering the extents of all warped images.

r, c = image1.shape[:2]

# Note that transformations take coordinates in
# (x, y) format, not (row, column), in order to be
# consistent with most literature.
corners = np.array([[0, 0],
                    [0, r],
                    [c, 0],
                    [c, r]])

# Warp the image corners to their new positions.
warped_corners = model_robust(corners)

# Find the extents of both the reference image and
# the warped target image.
all_corners = np.vstack((warped_corners, corners))

corner_min = np.min(all_corners, axis=0)
corner_max = np.max(all_corners, axis=0)

output_shape = (corner_max - corner_min)
output_shape = np.ceil(output_shape[::-1])

#The images are now warped according to the estimated transformation model.

#A shift is added to ensure that both images are visible in their entirety. Note that warp takes the inverse mapping as input.

from skimage.exposure import rescale_intensity
from skimage.transform import warp
from skimage.transform import SimilarityTransform
from skimage.color import gray2rgb

offset = SimilarityTransform(translation=-corner_min)

image0_ = warp(image0, offset.inverse,
               output_shape=output_shape)

image1_ = warp(image1, (model_robust + offset).inverse,
               output_shape=output_shape)

#An alpha channel is added to the warped images before merging them into a single image:

def add_alpha(image, background=-1):
    """Add an alpha layer to the image.

    The alpha layer is set to 1 for foreground
    and 0 for background.
    """

    rgb = gray2rgb(image)
    alpha = (image != background)
    return np.dstack((rgb, alpha))


#add alpha to the image0 and image1
img0 = add_alpha(image0_)
img1 = add_alpha(image1_)

#your code
#merge the alpha added image
merged = (img0+img1)
#your code
#merged = ...

alpha = merged[..., 3]
merged /= np.maximum(alpha, 1)[..., np.newaxis]
# The summed alpha layers give us an indication of
# how many images were combined to make up each
# pixel.  Divide by the number of images to get
# an average.


#show and save the output image as '/content/gdrive/My Drive/CMPUT 206 Wi19/Lab5_Files/imgOut.png'
fig, ax = plt.subplots()
ax.imshow(merged,'gray')
plt.show()


#io.imsave("/content/gdrive/My Drive/CMPUT 206 Wi19/Lab5_Files/imgOut.png",merged)