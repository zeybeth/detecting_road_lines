Real-Time Lane Detection System 🚗
This project implements a real-time lane detection system using Python and OpenCV. It detects white and yellow lane markings on the road from a video feed and overlays the predicted path on the original footage.

The system relies on color masking (HSV), edge detection (Canny), and the Hough Transform technique to identify lane lines robustly.

 Tech Stack
Language: Python 3.x

Libraries:

OpenCV (cv2) - For image processing.

NumPy - For matrix operations and mathematical calculations.

 How It Works (Pipeline)
The algorithm processes the video frame-by-frame through the following steps:

Gaussian Blur: Applies smoothing to reduce image noise.

Color Thresholding (HSV): Converts the image to HSV color space to specifically mask Yellow and White colors, isolating the lane markings.

Canny Edge Detection: Detects the edges within the masked regions.

Region of Interest (ROI): Crops the image to focus only on the road area (ignoring the sky and surroundings) using a polygon mask.

Hough Transform: Uses Probabilistic Hough Transform (HoughLinesP) to detect line segments.

Average Slope & Intercept: Calculates the average slope and intercept of the detected segments to construct two single, smooth lines (Left Lane & Right Lane).

Overlay: Draws the calculated lines onto the original frame.
