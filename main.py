import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

def preprocess_image(image):
    resized = cv2.resize(image, (640, 480)) # rescale to reduce computation
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred, resized

def detect_cones(image):
    lower_bound = np.array([0, 199, 150]) # HSV values found after tuning
    upper_bound = np.array([179, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours(contours):
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            filtered.append(contour)
    return filtered

def calculate_centroids(contours):
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append((cX, cY))
    return centroids

def perform_regression(centroids):
    left_points, right_points = [], []
    for cx, cy in centroids:
        if cx < 320:
            left_points.append([cx, cy])
        else:
            right_points.append([cx, cy])
    
    left_points, right_points = np.array(left_points), np.array(right_points)
    model_left, model_right = LinearRegression(), LinearRegression()
    
    model_left.fit(left_points[:, 0].reshape(-1, 1), left_points[:, 1])
    model_right.fit(right_points[:, 0].reshape(-1, 1), right_points[:, 1])
    
    return model_left, model_right

def visualize(image, model_left, model_right, scale_x, scale_y):
    y_min_left = model_left.predict([[0]])[0]
    y_max_left = model_left.predict([[640]])[0]
    y_min_right = model_right.predict([[0]])[0]
    y_max_right = model_right.predict([[640]])[0]

    points_left = [(0, int(y_min_left * scale_y)), (int(640 * scale_x), int(y_max_left * scale_y))]
    points_right = [(0, int(y_min_right * scale_y)), (int(640 * scale_x), int(y_max_right * scale_y))]

    cv2.line(image, points_left[0], points_left[1], [0, 0, 255], 10)
    cv2.line(image, points_right[0], points_right[1], [0, 0, 255], 10)

    return image

image = cv2.imread("input_image.png")
if image is None:
    print("Image not read properly.")
    exit()

original_h, original_w = image.shape[:2]
scale_x = original_w / 640
scale_y = original_h / 480

processed_image, resized_image = preprocess_image(image)

# Code for tuning HSV values for cone detection (not required once values are found)

# cv2.namedWindow("Trackbars")
# cv2.createTrackbar("LH", "Trackbars", 0, 179, lambda x: None)
# cv2.createTrackbar("LS", "Trackbars", 0, 255, lambda x: None)
# cv2.createTrackbar("LV", "Trackbars", 0, 255, lambda x: None)
# cv2.createTrackbar("UH", "Trackbars", 179, 179, lambda x: None)
# cv2.createTrackbar("US", "Trackbars", 255, 255, lambda x: None)
# cv2.createTrackbar("UV", "Trackbars", 255, 255, lambda x: None)

# while True:
#     hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    
#     l_h = cv2.getTrackbarPos("LH", "Trackbars")
#     l_s = cv2.getTrackbarPos("LS", "Trackbars")
#     l_v = cv2.getTrackbarPos("LV", "Trackbars")
#     u_h = cv2.getTrackbarPos("UH", "Trackbars")
#     u_s = cv2.getTrackbarPos("US", "Trackbars")
#     u_v = cv2.getTrackbarPos("UV", "Trackbars")
    
#     lower_bound = np.array([l_h, l_s, l_v])
#     upper_bound = np.array([u_h, u_s, u_v])
    
#     mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
#     cv2.imshow("Mask", mask)
#     cv2.imshow("Trackbars", np.zeros((1, 640, 3), np.uint8))  # Added this line
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

contours = detect_cones(resized_image)
filtered_contours = filter_contours(contours)
centroids = calculate_centroids(filtered_contours)
model_left, model_right = perform_regression(centroids)

final_image = visualize(image, model_left, model_right, scale_x, scale_y)

# save image
cv2.imwrite("answer.png", final_image)


# show image using cv2
# cv2.imshow("Final Image", final_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


