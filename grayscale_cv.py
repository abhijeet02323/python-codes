import cv2
import numpy as np

# Open a connection to the default camera (usually camera 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initial filter setting
filter_type = 0

def apply_filter(frame, filter_type):
    if filter_type == 0:
        # No filter (original)
        return frame
    elif filter_type == 1:
        # Blur Filter
        return cv2.GaussianBlur(frame, (15, 15), 0)
    elif filter_type == 2:
        # Edge Detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges_frame = cv2.Canny(gray_frame, 50, 150)
        return cv2.cvtColor(edges_frame, cv2.COLOR_GRAY2BGR)
    elif filter_type == 3:
        # Sepia Filter
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_frame = cv2.transform(frame, sepia_filter)
        return np.clip(sepia_frame, 0, 255).astype(np.uint8)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was not captured correctly, exit the loop
    if not ret:
        print("Error: Could not read frame.")
        break

    # Apply the current filter
    filtered_frame = apply_filter(frame, filter_type)

    # Display the resulting frame
    cv2.imshow('Filtered Video', filtered_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        filter_type = 1  # Apply Blur Filter
    elif key == ord('2'):
        filter_type = 2  # Apply Edge Detection
    elif key == ord('3'):
        filter_type = 3  # Apply Sepia Filter
    elif key == ord('0'):
        filter_type = 0  # No Filter (Original)

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
