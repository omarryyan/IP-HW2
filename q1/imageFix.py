import cv2
import matplotlib.pyplot as plt
import numpy as np

def contrast_stretch(image):
    mean_val = np.mean(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = image[i, j] - mean_val  # minus the expectation
    min_val = np.min(image)
    max_val = np.max(image)
    contrast_parameter_a = 255.0 / (max_val - min_val)  # y_1 - y_0 / x_1 - x_0
    contrast_parameter_b = 0 - min_val * contrast_parameter_a  # y_0 - x_0 * a
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = image[i, j] * contrast_parameter_a + contrast_parameter_b + mean_val  # ax + b + m

    return image, np.cumsum(cv2.calcHist([image], [0], None, [256], [0, 256]))

def gamma_correction(image, gamma):
    # Normalize pixels
    normalized_image = image / 255.0

    # for every pixed i  i = i ^ gamma
    corrected_image = np.power(normalized_image, gamma)

    # Scale back to the range [0, 255]
    corrected_image = (corrected_image * 255).astype(np.uint8)
    return corrected_image, np.cumsum(cv2.calcHist([corrected_image], [0], None, [256], [0, 256]))

def hist_equalization(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image, np.cumsum(cv2.calcHist([equalized_image], [0], None, [256], [0, 256]))

def apply_fix(image):
    contrast_img, contrast_hist = contrast_stretch(image)
    gamma_img, gamma_hist = gamma_correction(image, gamma=1.5)  # Adjust gamma as needed
    equalized_img, equalized_hist = hist_equalization(image)

    return [contrast_img, contrast_hist], [gamma_img, gamma_hist], [equalized_img, equalized_hist]

for i in range(1, 4):
    if i == 1:
        path = f'{i}.png'
    else:
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

    # Original Histogram
    plt.subplot(2, 4, 2)
    plt.plot(np.cumsum(cv2.calcHist([image], [0], None, [256], [0, 256])) / np.sum(cv2.calcHist([image], [0], None, [256], [0, 256])) * 255, color='b')
    plt.hist(image.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6)
    plt.title('Histogram')

    # Contrast Image and Histogram
    plt.subplot(2, 4, 3)
    plt.imshow(contrast_data[0], cmap='gray')
    plt.title('Contrast Image')
    plt.subplot(2, 4, 4)
    plt.plot(contrast_data[1] / np.sum(contrast_data[1]) * 255, color='b')
    plt.hist(contrast_data[0].flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6)
    plt.title('Contrast Histogram')

    # Gamma Corrected Image and Histogram
    plt.subplot(2, 4, 5)
    plt.imshow(gamma_data[0], cmap='gray')
    plt.title('Gamma Corrected Image')
    plt.subplot(2, 4, 6)
    plt.plot(gamma_data[1] / np.sum(gamma_data[1]) * 255, color='b')
    plt.hist(gamma_data[0].flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6)
    plt.title('Gamma Histogram')

    # Equalized Image and Histogram
    plt.subplot(2, 4, 7)
    plt.imshow(equalized_data[0], cmap='gray')
    plt.title('Equalized Image')
    plt.subplot(2, 4, 8)
    plt.plot(equalized_data[1] / np.sum(equalized_data[1]) * 255, color='b')
    plt.hist(equalized_data[0].flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6)
    plt.title('Equalized Histogram')

    plt.tight_layout()
    plt.show()

    # Save the modified images
    plt.imsave(f'{i}_contrast_fixed.jpg', contrast_data[0], cmap='gray', vmin=0, vmax=255)
    plt.imsave(f'{i}_gamma_fixed.jpg', gamma_data[0], cmap='gray', vmin=0, vmax=255)
    plt.imsave(f'{i}_equalized_fixed.jpg', equalized_data[0], cmap='gray', vmin=0, vmax=255)
