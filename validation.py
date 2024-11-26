import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to preprocess the frame (convert to grayscale and resize)
def preprocess_frame(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, (16, 16), interpolation=cv2.INTER_AREA)
    return resized

# Path to your video file
def validate(nameOfFile, datacount):
    src_path = os.path.join("src", nameOfFile)
    frame_count = 0

    cap = cv2.VideoCapture(src_path)


    # Initialize a list to store optical flow vectors over time
    flow_vectors = []
    frame1 = None
    # Loop over all frames in the video
    while frame_count < datacount:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames

        frame2 = np.array(preprocess_frame(frame), dtype=np.uint8)  # First 16x16 frame
        # Calculate dense optical flow between frame1 and frame2
        if frame1 is None:
            frame1 = frame2
            frame_count += 1
            continue
        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2, None,
            pyr_scale=0.5, levels=1, winsize=3, iterations=1, poly_n=5, poly_sigma=1.1, flags=0
        )

        # Extract flow vector for coordinate (8, 8)
        flow_at_8_8 = flow[8, 8]  # Flow vector at (8, 8)
        vx, vy = flow_at_8_8  # Horizontal (x) and vertical (y) flow components
        
        vx = round(float(vx),4)
        vy = round(float(vy),4)

        # Append frame count and flow components (vx, vy) to flow_vectors list
        flow_vectors.append((frame_count, vx, vy))

        # Read next frame and preprocess
        
        
        processed_frame = preprocess_frame(frame)
        frame1 = frame2  # Previous frame becomes current frame
        frame2 = np.array(processed_frame, dtype=np.uint8)  # Update the second frame

        frame_count += 1

    cap.release()
    return flow_vectors

