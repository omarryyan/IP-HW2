# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import matplotlib.pyplot as plt
import numpy as np

def histogram_equalization(image, id):

    # creating a Histograms Equalization
    # of a image using cv2.equalizeHist()
    equ = cv2.equalizeHist(image)

    # stacking images side-by-side
    res = np.hstack((image, equ))

    # show image input vs output
    cv2.imshow('image',res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return res


def apply_fix(image, id):
    # Your code goes here
    return 5

for i in range(1,4):
    path = f'{i}.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    fixed_image = histogram_equalization(image, i)
    plt.imsave(f'{i}_fixed.jpg', fixed_image, cmap='gray', vmin=0,vmax=255)
