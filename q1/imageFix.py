import cv2
import matplotlib.pyplot as plt
import numpy as np


def contrast_stretch(image):
    alpha = 1.5
    beta = 50
    fixed_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    cumulative_hist = np.cumsum(cv2.calcHist([fixed_image], [0], None, [256], [0, 256]).flatten())
    return fixed_image, cumulative_hist


def gamma_correction(image, gamma):
    # Normalize pixels
    normalized_image = image / 255.0
    # Apply gamma correction
    corrected_image = np.power(normalized_image, gamma)
    # Scale back to [0, 255]
    corrected_image = (corrected_image * 255).astype(np.uint8)
    cumulative_hist = np.cumsum(cv2.calcHist([corrected_image], [0], None, [256], [0, 256]).flatten())
    return corrected_image, cumulative_hist


def hist_equalization(image):
    equalized_image = cv2.equalizeHist(image)
    cumulative_hist = np.cumsum(cv2.calcHist([equalized_image], [0], None, [256], [0, 256]).flatten())
    return equalized_image, cumulative_hist


def apply_fix(image):
    contrast_img, contrast_hist = contrast_stretch(image)
    gamma_img, gamma_hist = gamma_correction(image, gamma=1.5)  # Adjust gamma as needed
    equalized_img, equalized_hist = hist_equalization(image)

    return [contrast_img, contrast_hist], [gamma_img, gamma_hist], [equalized_img, equalized_hist]


for i in range(1, 4):
    path = f'{i}.jpg'

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Get the results from apply_fix function
    contrast_data, gamma_data, equalized_data = apply_fix(image)

    # Plot original image and histograms
    plt.figure(figsize=(18, 6))

    # Original Image
    plt.subplot(2, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Original Histogram
    plt.subplot(2, 4, 2)
    original_hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    cumulative_original_hist = np.cumsum(original_hist)
    plt.plot(cumulative_original_hist, color='b', label='Cumulative Histogram')
    plt.hist(image.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6, label='Histogram')
    plt.title('Original Histogram')
    plt.legend()

    # Contrast Image and Histogram
    plt.subplot(2, 4, 3)
    plt.imshow(contrast_data[0], cmap='gray')
    plt.title('Contrast Image')
    plt.axis('off')
    plt.subplot(2, 4, 4)
    contrast_hist = contrast_data[1]
    plt.plot(contrast_hist, color='b', label='Cumulative Histogram')
    plt.hist(contrast_data[0].flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6, label='Histogram')
    plt.title('Contrast Histogram')
    plt.legend()

    # Gamma Corrected Image and Histogram
    plt.subplot(2, 4, 5)
    plt.imshow(gamma_data[0], cmap='gray')
    plt.title('Gamma Corrected Image')
    plt.axis('off')
    plt.subplot(2, 4, 6)
    gamma_hist = gamma_data[1]
    plt.plot(gamma_hist, color='b', label='Cumulative Histogram')
    plt.hist(gamma_data[0].flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6, label='Histogram')
    plt.title('Gamma Histogram')
    plt.legend()

    # Equalized Image and Histogram
    plt.subplot(2, 4, 7)
    plt.imshow(equalized_data[0], cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')
    plt.subplot(2, 4, 8)
    equalized_hist = equalized_data[1]
    plt.plot(equalized_hist, color='b', label='Cumulative Histogram')
    plt.hist(equalized_data[0].flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6, label='Histogram')
    plt.title('Equalized Histogram')
    plt.legend()

    plt.tight_layout()
    plt.show()
