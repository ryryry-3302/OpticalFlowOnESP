import cv2
import numpy as np
import os

# Function to preprocess the frame (convert to grayscale and resize)
def preprocess_frame(frame):
    # Convert to grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get the center 16x16 region of the grayscale frame
    height, width = grayscale.shape
    start_x = (width - 16) // 2
    start_y = (height - 16) // 2
    center_16x16 = grayscale[start_y:start_y + 16, start_x:start_x + 16]
    return center_16x16

# Path to your video file
def validate(nameOfFile, datacount):
    src_path = os.path.join("src", nameOfFile)
    frame_count = 0

    cap = cv2.VideoCapture(src_path)

    # Initialize a list to store optical flow vectors over time
    flow_vectors = []

    # Read the first frame and preprocess it
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read the first frame from the video.")
    frame1 = preprocess_frame(frame)

    # Define the point at (8, 8) to track
    point_to_track = np.array([[8, 8]], dtype=np.float32).reshape(-1, 1, 2)

    # Loop over the frames in the video
    while frame_count < datacount:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames

        frame2 = preprocess_frame(frame)

        # Calculate optical flow using Lucas-Kanade method
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            frame1, frame2, point_to_track, None, winSize=(5, 5), maxLevel=2,
        )

        # Check if the point was successfully tracked
        if status[0][0] == 1:  # Point successfully tracked
            vx, vy = next_points[0][0] - point_to_track[0][0]
            vx = round(float(vx), 4)
            vy = round(float(vy), 4)
            flow_vectors.append((frame_count, vx, vy))
        else:  # If the point is not tracked, append (0, 0)
            flow_vectors.append((frame_count, 0, 0))

        # Update frame1 and point_to_track for the next iteration
        frame1 = frame2
        point_to_track = next_points

        frame_count += 1

    cap.release()
    return flow_vectors
