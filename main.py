import os
import cv2
import numpy as np

# Apply Gaussian Blur to reduce noise
def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """Apply Gaussian Blur to the image."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                     np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    kernel = kernel / np.sum(kernel)  # Normalize the kernel
    return cv2.filter2D(image, -1, kernel)

def histogram_equalization(image: np.ndarray):
    """Apply manual histogram equalization to a grayscale image."""
    
    # Compute the histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Normalize the CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    # Use the normalized CDF to map the pixel values to new values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    
    # Reshape the flattened image back to its original shape
    image_equalized = image_equalized.reshape(image.shape).astype(np.uint8)
    
    return image_equalized

if __name__ == '__main__':
    folder_path = 'vehicle_dataset'

    # Get a list of all files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        # Construct the full file path
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply gaussian blur to reduce noise
        blurred_image = gaussian_blur(gray_image)

        # Apply histogram equalixzation to enhance the contrast
        equalized_image = histogram_equalization(blurred_image)

        # Apply Otsu's Thresholding
        _, otsu_thresholded = cv2.threshold(equalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform canny edge detection
        low_threshold = 50
        high_threshold = 150
        canny_edges = cv2.Canny(equalized_image, low_threshold, high_threshold)

        # Find Connected Components
        num_labels, labels = cv2.connectedComponents(canny_edges)

        # Filter the components based on size and aspect ratio
        plate_regions = []
        for i in range(1, num_labels):  # We start from 1 to avoid the background label
            # Extract the region of the component
            component = np.where(labels == i, 255, 0).astype(np.uint8)
            
            # Find contours of the component
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding box around the contour
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter based on aspect ratio and size
                aspect_ratio = float(w) / h
                if 0.2 < float(h) / w < 0.3 and 3000 < w * h < 10000:
                    plate_regions.append((x, y, w, h))

        # Draw bounding boxes around potential number plates
        for (x, y, w, h) in plate_regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the result
        # cv2.imshow(f"Number Plate Localization - {image_file}", image)
        cv2.waitKey(0)  # Wait for user input to close the window
        localized_path = 'localized'
        result_path = os.path.join(localized_path, image_file)
        cv2.imwrite(result_path, image)

    # close windows
    cv2.destroyAllWindows()
