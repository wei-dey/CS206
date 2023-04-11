
import matplotlib.pyplot as plt
from skimage import io, exposure, color
import numpy as np
import cv2
import math
import os
os.chdir(os.path.dirname(__file__))

def part1_histogram_compute():
    """add your code here"""
    # return a 2D numpy array
    image1 = cv2.imread('test.jpg', 0)
    # print(image1)
    # initialize an all-zero numpy array with int type elements
    arr = np.zeros(256, dtype=int)
    # go through all the pixels
    for x in range(0, image1.shape[0]):
        for y in range(0, image1.shape[1]):
            # store the frequency of the element in the numpy array at the index of element value
            index = image1[x][y]
            arr[index] += 1
            # using pyplot to visualize the histogram of the created numpy array
    # visualize the histogram
    plt.title("My Histogram")
    plt.plot(arr)
    # x-axis range
    plt.xlim([0, 256])
    plt.show()

    # create Numpy Histogram using calcHist and  np.histogram function to compute histogram and
    hist = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist, bins = np.histogram(image1.ravel(), 256, [0, 256])
    # pyplot to visualize the Numpy histogram
    plt.title("Numpy Histogram")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

    # create Skimage Histogram using skimage.exposure library function
    hist2 = exposure.histogram(image1, nbins=256, source_range='dtype')
    # print(hist2)
    # visualize the Skimage Histogram
    plt.title("Skimage Histogram")
    plt.plot(hist2[1], hist2[0])
    plt.xlim([0, 256])
    plt.show()


def part2_histogram_equalization():
    """add your code here"""
    ori = io.imread('test.jpg')
    plt.title("Original Image")
    plt.imshow(ori)
    plt.show()
    # print(image1)
    image1 = cv2.imread('test.jpg', 0)
    # initialize an all-zero numpy array with int type elements
    arr = np.zeros(256, dtype=int)
    # go through all the pixels

    for x in range(0, image1.shape[0]):
        for y in range(0, image1.shape[1]):
            # store the frequency of the element in the numpy array at the index of element value
            index = image1[x][y]
            arr[index] += 1
            # using pyplot to visualize the histogram of the created numpy array
    plt.title("My Histogram")
    plt.plot(arr)
    # x-axis range
    plt.xlim([0, 256])
    plt.show()
    # initialize an all zero 2D numpy array with same size and shape of original image
    eql = np.zeros((image1.shape[0], image1.shape[1]))
    # calculate the cumulative histogram of original histogram
    cum_arr = np.cumsum(arr)
    # go through all the elements and use formula to calculate the equalized elements value
    # store the equalized elements value in newly created 2D numpy array
    for i in range(0, image1.shape[0]):
        for j in range(0, image1.shape[1]):
            index = image1[i][j]
            eql[i][j] = math.floor(255 * cum_arr[index] / image1.size + 0.5)
    # initialized an all zero numpy array
    neweq = np.zeros(256, dtype=int)
    # go through all the elements in the new 2d numpy array and compute the new histogram
    for w in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            neweq[int(eql[w, y])] += 1
    # visualized the equalized histogram
    plt.title('Equalized Histogram')
    plt.plot(neweq)
    plt.xlim([0, 256])
    plt.show()
    # visualized the equalized image and convert it to grey scale
    plt.title("Equalized Image")
    plt.imshow(eql, 'gray')
    plt.show()


def part3_histogram_comparing():
    """add your code here"""
    # read two images and converts them to gray scale
    dayimg = cv2.imread('day.jpg', 0)
    nightimg = cv2.imread('night.jpg', 0)

    # compute two histogram of two images using numoy function
    dayhist = cv2.calcHist([dayimg], [0], None, [256], [0, 256])
    dayhist, bins = np.histogram(dayimg.ravel(), 256, [0, 256])
    nighthist = cv2.calcHist([nightimg], [0], None, [256], [0, 256])
    nighthist, bins = np.histogram(nightimg.ravel(), 256, [0, 256])

    # initialzie the bc coefficient
    bc = 0

    # initialize two list to store the partition
    day = []
    night = []
    # go through all the values in the two histograms
    for i in range(0, 256):
        # to normalize the histograms of two images
        day.append(dayhist[i] / dayimg.size)
        night.append(nighthist[i] / nightimg.size)
        # count the bc coefficient using formula
        bc += math.sqrt(day[i] * night[i])
    print(bc)
    return bc


def part4_histogram_matching():
    """add your code here"""
    # read day image and convert it to gray scale
    dayimg = cv2.imread('day.jpg', 0)
    # visualize the gray scale day image
    plt.title("Day")
    plt.imshow(dayimg, 'gray')
    plt.show()

    # read the night image and convert it to gray scale image
    nightimg = cv2.imread('night.jpg', 0)
    # visualize the gray scale night image
    plt.title("Night")
    plt.imshow(nightimg, 'gray')
    plt.show()

    # compute the histogram of the day image using numpy function
    dayhist = cv2.calcHist([dayimg], [0], None, [256], [0, 256])
    dayhist, bins = np.histogram(dayimg.ravel(), 256, [0, 256])
    # compute the cumulative histogram of day image using numpy function
    day_cum = np.cumsum(dayhist)/dayimg.size

    # compute the histogram of the night image using numpy function
    nighthist = cv2.calcHist([nightimg], [0], None, [256], [0, 256])
    nighthist, bins = np.histogram(nightimg.ravel(), 256, [0, 256])
    # compute the cumulative histogram of night image using numpy function
    night_cum = np.cumsum(nighthist)/nightimg.size

    # initialize an all zero numpy array
    new = np.zeros(dayimg.size)

    # histogram matching using algorithms and code in lecture slides
    for i in range(0, dayimg.shape[0]):
        for j in range(0, dayimg.shape[1]):
            a = dayimg[i, j]
            b = 0
            while day_cum[a] > night_cum[a]:
                b += 1
            new[i, j] = b

    # visualize the image
    plt.title('Matched')
    plt.imshow(new, 'gray')


if __name__ == '__main__':
    part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()
