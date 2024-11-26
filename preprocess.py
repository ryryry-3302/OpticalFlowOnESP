import cv2
import os
import serial
import numpy as np

# Used to process frame of video to grayscale and resize to 16x16
def preprocess_frame(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, (16, 16), interpolation=cv2.INTER_AREA)
    return resized.flatten()  # Flatten to 256 elements (1D array)

def send_frame_to_esp32(frame_data):
    ser.write(frame_data.tobytes())  # Send the flattened data

def read_ser():
    # Read 256 bytes (the full 16x16 array data) from the serial port
    data = ser.read(256)  # Expecting exactly 256 bytes from ESP32
    if len(data) == 256:
        print("Data received successfully!")
        return np.frombuffer(data, dtype=np.uint8)  # Return as a 1D array of size 256
    else:
        print(f"Failed to read 256 bytes, received {len(data)} bytes.")
        return None
    
def read_optical_flow_vector():
    # Read 4 bytes: 2 bytes for u and 2 bytes for v components of optical flow
    data = ser.read(4)  # Expecting 4 bytes: 2 for u and 2 for v
    if len(data) == 4:
        u = int.from_bytes(data[0:2], byteorder='big', signed=True)
        v = int.from_bytes(data[2:4], byteorder='big', signed=True)
        return u, v
    else:
        print(f"Failed to read 4 bytes, received {len(data)} bytes.")
        return None, None

# Path to your video file
src_path = os.path.join("src", "DashcamFootage.mp4")

cap = cv2.VideoCapture(src_path)

# Set up serial communication
ser = serial.Serial('COM5', 115200, timeout=5)

frame_count = 0

while frame_count < 2:  # Loop to send and compare up to 2 frames
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame = preprocess_frame(frame)
    print(f"Sending frame {frame_count + 1}:")
    print(processed_frame)

    # Send the processed frame to ESP32
    send_frame_to_esp32(processed_frame)

    # Wait to receive the optical flow after the frame is processed

    frame_count += 1

# Close the serial port after processing is done
u, v = read_optical_flow_vector()
scaled_u = u / 10000
scaled_v = v / 10000

if u is not None and v is not None:
    print(f"Optical flow at (8, 8): u={scaled_u}, v={scaled_v}")
else:
    print("Failed to receive optical flow data.")

ser.close()
