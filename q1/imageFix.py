import cv2
import matplotlib.pyplot as plt
import numpy as np


def contrast_stretch(image):
    alpha = 1
    beta = 190
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
    if len(image.shape) == 2:  # Grayscale
        equalized_image = cv2.equalizeHist(image)
    else:  # Colored image
        channels = cv2.split(image)
        equalized_channels = [cv2.equalizeHist(channel) for channel in channels]
        equalized_image = cv2.merge(equalized_channels)

    cumulative_hist = np.cumsum(cv2.calcHist([cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]).flatten())
    return equalized_image, cumulative_hist


def apply_fix(image):
    if len(image.shape) == 2:  # Grayscale image
        contrast_img, contrast_hist = contrast_stretch(image)
        gamma_img, gamma_hist = gamma_correction(image, gamma=1.5)
        equalized_img, equalized_hist = hist_equalization(image)
    else:  # Colored image
        channels = cv2.split(image)
        # Apply transformations to each channel independently
        contrast_img_channels = [contrast_stretch(channel)[0] for channel in channels]
        gamma_img_channels = [gamma_correction(channel, gamma=1.2)[0] for channel in channels]
        equalized_img_channels = [cv2.equalizeHist(channel) for channel in channels]

        # Merge channels back to color images
        contrast_img = cv2.merge(contrast_img_channels)
        gamma_img = cv2.merge(gamma_img_channels)
        equalized_img = cv2.merge(equalized_img_channels)

        # Calculate histograms on grayscale versions of the enhanced images
        contrast_hist = np.cumsum(cv2.calcHist([cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]).flatten())
        gamma_hist = np.cumsum(cv2.calcHist([cv2.cvtColor(gamma_img, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]).flatten())
        equalized_hist = np.cumsum(cv2.calcHist([cv2.cvtColor(equalized_img, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]).flatten())

    return [contrast_img, contrast_hist], [gamma_img, gamma_hist], [equalized_img, equalized_hist]


for i in range(1, 4):
    path = f'{i}.jpg'

    image = cv2.imread(path)  # Load the image in color (default)

    # Get the results from apply_fix function
    contrast_data, gamma_data, equalized_data = apply_fix(image)

    # Display the gamma-corrected image (example)
    cv2.imshow("Gamma Corrected", gamma_data[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Plot original image and histograms
    plt.figure(figsize=(18, 6))

    # Original Image
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Original Histogram
    plt.subplot(2, 4, 2)
    original_hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]).flatten()
    cumulative_original_hist = np.cumsum(original_hist)
    plt.plot(cumulative_original_hist, color='b', label='Cumulative Histogram')
    plt.hist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6, label='Histogram')
    plt.title('Original Histogram')
    plt.legend()

    # Contrast Image and Histogram
    plt.subplot(2, 4, 3)
    plt.imshow(cv2.cvtColor(contrast_data[0], cv2.COLOR_BGR2RGB))
    plt.title('Contrast Image')
    plt.axis('off')
    plt.subplot(2, 4, 4)
    contrast_hist = contrast_data[1]
    plt.plot(contrast_hist, color='b', label='Cumulative Histogram')
    plt.hist(cv2.cvtColor(contrast_data[0], cv2.COLOR_BGR2GRAY).flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6, label='Histogram')
    plt.title('Contrast Histogram')
    plt.legend()

    # Gamma Corrected Image and Histogram
    plt.subplot(2, 4, 5)
    plt.imshow(cv2.cvtColor(gamma_data[0], cv2.COLOR_BGR2RGB))
    plt.title('Gamma Corrected Image')
    plt.axis('off')
    plt.subplot(2, 4, 6)
    gamma_hist = gamma_data[1]
    plt.plot(gamma_hist, color='b', label='Cumulative Histogram')
    plt.hist(cv2.cvtColor(gamma_data[0], cv2.COLOR_BGR2GRAY).flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6, label='Histogram')
    plt.title('Gamma Histogram')
    plt.legend()

    # Equalized Image and Histogram
    plt.subplot(2, 4, 7)
    plt.imshow(cv2.cvtColor(equalized_data[0], cv2.COLOR_BGR2RGB))
    plt.title('Equalized Image')
    plt.axis('off')
    plt.subplot(2, 4, 8)
    equalized_hist = equalized_data[1]
    plt.plot(equalized_hist, color='b', label='Cumulative Histogram')
    plt.hist(cv2.cvtColor(equalized_data[0], cv2.COLOR_BGR2GRAY).flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6, label='Histogram')
    plt.title('Equalized Histogram')
    plt.legend()

    plt.tight_layout()
    plt.show()
