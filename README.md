# PerceptionChallengeUW


## Libraries Used

- `cv2` (OpenCV): For image processing and computer vision tasks.
- `numpy`: For numerical operations, particularly in data manipulation and line fitting.
- `sklearn.linear_model`: Specifically, the `LinearRegression` class is used for line fitting.

## Methodology

### Image Preprocessing

1. **Resizing**: The image is resized to 640x480 pixels to reduce computational load.
2. **Grayscale Conversion**: The image is converted to grayscale.
3. **Gaussian Blurring**: A Gaussian blur with a kernel size of 5x5 is applied to reduce noise.

### Cone Detection

1. **Color Space Conversion**: The image is converted to the HSV color space.
2. **Color Filtering**: A color filter is applied to isolate the cones based on manually tuned HSV values.
3. **Contour Finding**: OpenCV's `findContours` is used to find contours in the filtered image.
  
### Path Prediction

1. **Centroid Calculation**: The centroids of the detected cones are calculated.
2. **Line Fitting**: Linear regression is performed on the centroids to predict the path.

### Code Organization

The code is organized into functions for modularity:

- `preprocess_image(image)`: Handles the resizing, grayscaling, and blurring of the image.
- `detect_cones(image)`: Manages the cone detection process including color space conversion and contour finding.
- `fit_line(centroids)`: Uses the `LinearRegression` class from `sklearn.linear_model` to fit a line through the centroids.

# Usage

1. Run the python file `main.py`.
2. The script takes an input image named `input_image.png`.
3. The script writes the output to a file named `answer.png`.
