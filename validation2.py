import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Constants for scaling
SCALE_FACTOR = 100000.0

# Function to compute optical flow (equivalent to the ESP32 code)
def compute_optical_flow(received_data1, received_data2, x, y):
    # Gradients (float for precision)
    I_x = 0.0
    I_y = 0.0
    I_t = 0.0

    # Local 3x3 window (matrix G, vector b)
    G = np.zeros((2, 2))  # Gradient matrix
    b = np.zeros(2)       # RHS vector

    for i in range(-1, 2):
        for j in range(-1, 2):
            xCoord = x + i
            yCoord = y + j

            if 0 <= xCoord < 16 and 0 <= yCoord < 16:
                # Convert byte values to int16 to avoid overflow
                I_x = float(int(received_data1[yCoord][xCoord + 1]) - int(received_data1[yCoord][xCoord - 1]))
                I_y = float(int(received_data1[yCoord + 1][xCoord]) - int(received_data1[yCoord - 1][xCoord]))
                I_t = float(int(received_data2[yCoord][xCoord]) - int(received_data1[yCoord][xCoord]))

                # Update matrix G
                G[0][0] += I_x * I_x
                G[0][1] += I_x * I_y
                G[1][0] += I_x * I_y
                G[1][1] += I_y * I_y

                # Update vector b
                b[0] += I_x * I_t
                b[1] += I_y * I_t

    # Determinant of G
    det = G[0][0] * G[1][1] - G[0][1] * G[1][0]
    if det == 0:
        print("No flow detected (singular matrix).")
        return None, None

    # Solve for (u, v) using Cramer's rule
    u = -(b[0] * G[1][1] - b[1] * G[0][1]) / det
    v = -(b[1] * G[0][0] - b[0] * G[1][0]) / det

    # Scale the flow vectors by the SCALE_FACTOR and convert to integers
    scaled_u = int(u * SCALE_FACTOR) /SCALE_FACTOR
    scaled_v = int(v * SCALE_FACTOR) /SCALE_FACTOR

    return scaled_u, scaled_v


# Function to preprocess the frame (convert to grayscale and resize)
def preprocess_frame(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, (16, 16), interpolation=cv2.INTER_AREA)
    return resized


# Path to your video file
src_path = os.path.join("src", "DashcamFootage.mp4")
frame_count = 0

cap = cv2.VideoCapture(src_path)

# Initialize lists to store optical flow vectors over time
u_values = []
v_values = []

# Read first frame and preprocess
ret, frame = cap.read()
processed_frame = preprocess_frame(frame)
frame1 = np.array(processed_frame, dtype=np.uint8)  # First 16x16 frame

# Read second frame and preprocess
ret, frame = cap.read()
processed_frame = preprocess_frame(frame)
frame2 = np.array(processed_frame, dtype=np.uint8)  # Second 16x16 frame

# Loop over all frames in the video
while True:
    # Compute optical flow between frame1 and frame2
    u, v = compute_optical_flow(frame1, frame2, 8, 8)

    if u is not None and v is not None:
        u_values.append(u)  # Append horizontal flow component
        v_values.append(v)  # Append vertical flow component

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
time_points = np.arange(len(u_values))

plt.figure(figsize=(10, 6))
plt.quiver(time_points, u_values, v_values, angles='xy', scale_units='xy', scale=1, color='r')
plt.title("Optical Flow Vectors Over Time")
plt.xlabel("Frame Index")
plt.ylabel("Flow Vector Components (u, v)")
plt.grid(True)
plt.show()
