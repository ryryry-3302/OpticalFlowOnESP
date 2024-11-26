import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Constants for scaling
SCALE_FACTOR = 1000.0  # Used to scale the flow for better visualization

# Function to compute optical flow (SimpleFlow approach)
def compute_simpleflow(received_data1, received_data2, x, y):
    # Gradients (float for precision)
    I_x = 0.0
    I_y = 0.0
    I_t = 0.0

    # Local 3x3 window (gradient computation)
    for i in range(-1, 2):
        for j in range(-1, 2):
            xCoord = x + i
            yCoord = y + j

            if 0 <= xCoord < received_data1.shape[1] and 0 <= yCoord < received_data1.shape[0]:
                # Convert byte values to float for gradient computation
                I_x += float(received_data1[yCoord, xCoord + 1] - received_data1[yCoord, xCoord - 1])
                I_y += float(received_data1[yCoord + 1, xCoord] - received_data1[yCoord - 1, xCoord])
                I_t += float(received_data2[yCoord, xCoord] - received_data1[yCoord, xCoord])

    # Compute optical flow components u (horizontal) and v (vertical)
    flow_u = I_x * I_t / (I_x ** 2 + I_y ** 2 + 1e-6)
    flow_v = I_y * I_t / (I_x ** 2 + I_y ** 2 + 1e-6)

    # Scale the flow vectors for better visualization
    scaled_u = int(flow_u * SCALE_FACTOR) / SCALE_FACTOR
    scaled_v = int(flow_v * SCALE_FACTOR) / SCALE_FACTOR

    return scaled_u, scaled_v

# Function to preprocess the frame (convert to grayscale and resize)
def preprocess_frame(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, (16, 16), interpolation=cv2.INTER_AREA)
    return resized

# Path to your video file
src_path = os.path.join("src", "DashcamFootage.mp4")

# Function to load frames from a video
def load_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return frames  # Return empty list if the video can't be opened

    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()

    cap.release()
    return frames

# Load video frames
frames = load_frames(src_path)

# Check if any frames were loaded
if len(frames) == 0:
    print("No frames loaded. Check the video file path or format.")
else:
    print(f"Loaded {len(frames)} frames.")

# Process the frames to calculate optical flow for the first 100 frames
max_frames = min(100, len(frames) - 1)  # Process at most 100 frames or the number of available frames - 1

for i in range(max_frames):
    # Preprocess consecutive frame pairs
    frame1 = preprocess_frame(frames[i])
    frame2 = preprocess_frame(frames[i + 1])

    # Specify the coordinate (8, 8)
    x, y = 8, 8

    # Compute flow at (8, 8) for each consecutive pair of frames
    u, v = compute_simpleflow(frame1, frame2, x, y)

    # Print the optical flow for each frame pair
    print(f"Optical Flow at (8, 8) for frame {i} -> u: {u}, v: {v}")
