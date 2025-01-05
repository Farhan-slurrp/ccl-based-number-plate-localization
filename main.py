import os
import cv2
import numpy as np

def ccl_connectivity_analysis(image, final_image, type):
    # Find Connected Components
    num_labels, labels = cv2.connectedComponents(final_image, connectivity=8 if type == "8" else 4)

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
            if 3 < aspect_ratio < 9:
                plate_regions.append((x, y, w, h))

    # Draw bounding boxes around potential number plates
    for (x, y, w, h) in plate_regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.waitKey(0)
    localized_path = f'localized_{"c8" if type == "8" else "c4"}'
    result_path = os.path.join(localized_path, image_file)
    cv2.imwrite(result_path, image)

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

        # Apply blackhat morphological operation
        filterSize =(13, 5) 
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                        filterSize)
        blackhat_image = cv2.morphologyEx(gray_image,  
                              cv2.MORPH_BLACKHAT, 
                              rectKern)
        
        # Get the light region from image and may contain plate number by:
        # Apply Closing morphological operation (Dilation, then erosion)
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, squareKern)
        # Apply otsu thresholding to binarize the image
        light_image = cv2.threshold(light_image, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


        # Compute the Scharr gradient representation of the blackhat image in the x-direction 
        gradX = cv2.Sobel(blackhat_image, ddepth=cv2.CV_32F,
			dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        # Scale the result back to the range [0, 255]
        gradX = gradX.astype("uint8")

        # Apply gaussian blur
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        #Binarize the image with Otsu thresholding
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        binary_image = cv2.threshold(gradX, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Perform a series of erotion and dilation
        binary_image = cv2.erode(binary_image, None, iterations=2)
        binary_image = cv2.dilate(binary_image, None, iterations=2)

        # Take the bitwise AND between our binary_image and light_image
        binary_image = cv2.bitwise_and(binary_image, binary_image, mask=light_image)

        # Perform another series of erotion and dilation and get the final image
        binary_image = cv2.dilate(binary_image, None, iterations=2)
        final_image = cv2.erode(binary_image, None, iterations=1)

        # CCL connectivity and pattern analysis
        ccl_connectivity_analysis(image, final_image, "4")
        ccl_connectivity_analysis(image, final_image, "8")

    # close windows
    cv2.destroyAllWindows()
