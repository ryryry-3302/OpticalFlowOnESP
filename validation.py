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
src_path = os.path.join("src", "DashcamFootage.mp4")
frame_count = 0

cap = cv2.VideoCapture(src_path)

# Read first frame and preprocess
ret, frame = cap.read()
processed_frame = preprocess_frame(frame)
frame1 = np.array(processed_frame, dtype=np.uint8)  # First 16x16 frame

# Initialize lists to store optical flow vectors over time
vx_values = []
vy_values = []

# Read second frame and preprocess
ret, frame = cap.read()
processed_frame = preprocess_frame(frame)
frame2 = np.array(processed_frame, dtype=np.uint8)  # Second 16x16 frame

# Loop over all frames in the video
while True:
    # Calculate dense optical flow between frame1 and frame2
    flow = cv2.calcOpticalFlowFarneback(
        frame1, frame2, None,
        pyr_scale=0.5, levels=3, winsize=5, iterations=3, poly_n=5, poly_sigma=1.1, flags=0
    )

    # Extract flow vector for coordinate (8, 8)
    flow_at_8_8 = flow[8, 8]  # Flow vector at (8, 8)
    vx, vy = flow_at_8_8  # Horizontal (x) and vertical (y) flow components
    vx_values.append(vx)  # Append horizontal flow component
    vy_values.append(vy)  # Append vertical flow component
    
    # Read next frame and preprocess
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames
    
    processed_frame = preprocess_frame(frame)
    frame1 = frame2  # Previous frame becomes current frame
    frame2 = np.array(processed_frame, dtype=np.uint8)  # Update the second frame

    frame_count += 1

cap.release()

# Plot the optical flow vectors over time
time_points = np.arange(len(vx_values))

plt.figure(figsize=(10, 6))
plt.quiver(time_points, vx_values, vy_values, angles='xy', scale_units='xy', scale=1, color='r')
plt.title("Optical Flow Vectors Over Time")
plt.xlabel("Frame Index")
plt.ylabel("Flow Vector Components (vx, vy)")
plt.grid(True)
plt.show()
