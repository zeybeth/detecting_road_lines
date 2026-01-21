import cv2
import numpy as np

def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    polygons = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.53)),
        (0, int(height * 0.55))
    ]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def average_slope_intercept(lines, frame_width, frame_height):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope < -0.1 and slope > -10:
            left_fit.append((slope, intercept))
        elif slope > 0.1 and slope < 10:
            right_fit.append((slope, intercept))

    def make_points(line, frame_width, frame_height):
        slope, intercept = line
        y1 = frame_height
        y2 = int(y1 * 0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        x1 = max(min(x1, frame_width - 1), 0)
        x2 = max(min(x2, frame_width - 1), 0)
        y1 = max(min(y1, frame_height - 1), 0)
        y2 = max(min(y2, frame_height - 1), 0)
        return [x1, y1, x2, y2]

    left_line = None
    right_line = None
    if left_fit:
        left_avg = np.mean(left_fit, axis=0)
        left_line = make_points(left_avg, frame_width, frame_height)
    if right_fit:
        right_avg = np.mean(right_fit, axis=0)
        right_line = make_points(right_avg, frame_width, frame_height)

    return left_line, right_line

video = cv2.VideoCapture("road_car_view.mp4")

while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture("road_car_view.mp4")
        continue

    frame_height, frame_width = orig_frame.shape[:2]

    frame = cv2.GaussianBlur(orig_frame, (3, 3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    low_yellow = np.array([15, 80, 120])
    up_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, low_yellow, up_yellow)

    low_white = np.array([0, 0, 180])
    up_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, low_white, up_white)

    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

    edges = cv2.Canny(combined_mask, 50, 150)
    cropped_edges = region_of_interest(edges)

    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=100)

    line_image = np.zeros_like(orig_frame)

    if lines is not None:
        left_line, right_line = average_slope_intercept(lines, frame_width, frame_height)
        if left_line is not None:
            cv2.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 8)
        if right_line is not None:
            cv2.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 8)

    combo = cv2.addWeighted(orig_frame, 0.8, line_image, 1, 1)

    cv2.imshow("frame", combo)
    cv2.imshow("edges", cropped_edges)

    key = cv2.waitKey(25)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
