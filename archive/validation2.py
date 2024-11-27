import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Constants for scaling
SCALE_FACTOR = 1000.0  # Used to scale the flow for better visualization

# Gradient kernels
IxKernel = np.array([[-0.25, 0.25], [-0.25, 0.25]])
IyKernel = np.array([[-0.25, -0.25], [0.25, 0.25]])
ItKernel1 = np.array([[0.25, 0.25], [0.25, 0.25]])  # img1
ItKernel2 = np.array([[-0.25, -0.25], [-0.25, -0.25]])  # img2

def convolve2D(img, kernel, x, y):
    """
    Perform 2x2 convolution at (x, y) on the given image with the kernel.
    """
    submatrix = img[x:x+2, y:y+2]
    return np.sum(submatrix * kernel)

def calculate_optical_flow(frame1, frame2, x, y, scale_factor=SCALE_FACTOR):
    """
    Calculate optical flow at a specific (x, y) coordinate from two image frames.
    """
    if x < 2 or y < 2 or x > 13 or y > 13:
        print("Error: Coordinates out of bounds for a 5x5 neighborhood.")
        return None, None

    # Extract 5x5 windows
    window1 = frame1[x-2:x+3, y-2:y+3]
    window2 = frame2[x-2:x+3, y-2:y+3]

    # Initialize gradients and matrices
    Ix = np.zeros((4, 4))
    Iy = np.zeros((4, 4))
    It = np.zeros((4, 4))
    A = np.zeros((2, 2))
    b = np.zeros(2)

    # Compute gradients
    for i in range(4):
        for j in range(4):
            Ix[i, j] = convolve2D(window1, IxKernel, i, j) + convolve2D(window2, IxKernel, i, j)
            Iy[i, j] = convolve2D(window1, IyKernel, i, j) + convolve2D(window2, IyKernel, i, j)
            It[i, j] = convolve2D(window2, ItKernel2, i, j) + convolve2D(window1, ItKernel1, i, j)

    # Populate A and b matrices
    for i in range(4):
        for j in range(4):
            A[0, 0] += Ix[i, j] ** 2
            A[0, 1] += Ix[i, j] * Iy[i, j]
            A[1, 1] += Iy[i, j] ** 2
            b[0] += Ix[i, j] * It[i, j]
            b[1] += Iy[i, j] * It[i, j]
    A[1, 0] = A[0, 1]  # Symmetry

    # Solve for u, v
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if det != 0:
        u = (A[1, 1] * -b[0] - A[0, 1] * -b[1]) / det
        v = (A[0, 0] * -b[1] - A[0, 1] * -b[0]) / det
    else:
        u, v = 0.0, 0.0

    # Scale results
    scaled_u = int(u * scale_factor)
    scaled_v = int(v * scale_factor)

    # Print results for verification

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

for i in range(10):
    # Preprocess consecutive frame pairs
    frame1 = preprocess_frame(frames[i])
    frame2 = preprocess_frame(frames[i + 1])

    # Specify the coordinate (8, 8)
    x, y = 8, 8

    # Compute flow at (8, 8) for each consecutive pair of frames
    u, v = calculate_optical_flow(frame1, frame2, x, y)

    # Print the optical flow for each frame pair
    print(f"Optical Flow at (8, 8) for frame {i} -> u: {u/SCALE_FACTOR}, v: {v/SCALE_FACTOR}")
