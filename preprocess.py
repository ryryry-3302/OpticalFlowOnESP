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
    # Read 2 bytes (u and v components of optical flow)
    data = ser.read(2)
    if len(data) == 2:
        u = int.from_bytes(data[0:1], byteorder='little', signed=True)
        v = int.from_bytes(data[1:2], byteorder='little', signed=True)
        return u, v
    else:
        print(f"Failed to read 2 bytes, received {len(data)} bytes.")
        return None, None


# Example usage:


def compare_data(sent_data, received_data):
    # Compare the two arrays and print the result
    if np.array_equal(sent_data, received_data):
        print("Sent and received data are the same.")
    else:
        print("Data mismatch!")
        # Print the differences
        diff = np.abs(sent_data - received_data)
        print(f"Differences: {diff}")

# Path to your video file
src_path = os.path.join("src", "DashcamFootage.mp4")

cap = cv2.VideoCapture(src_path)

# Set up serial communication
ser = serial.Serial('COM5', 115200, timeout=5)

frame_count = 0

while frame_count < 2:  # Loop to send and compare up to 100 frames
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame = preprocess_frame(frame)
    print(f"Sending frame {frame_count + 1}:")
    print(processed_frame)

    # Send the processed frame to ESP32
    send_frame_to_esp32(processed_frame)



    frame_count += 1

u, v = read_optical_flow_vector()
if u is not None and v is not None:
    print(f"Optical flow at (8, 8): u={u}, v={v}")