import cv2
import os
import serial
import numpy as np

# Used to process frame of video to grayscale and resize to 16x16
def preprocess_frame(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, (16, 16), interpolation=cv2.INTER_AREA)
    return resized.flatten()  # Flatten to 256 elements (1D array)

def send_frame_to_esp32(frame_data, ser):
    ser.write(frame_data.tobytes())  # Send the flattened data

def read_ser():
    # Read 256 bytes (the full 16x16 array data) from the serial port
    data = ser.read(256)  # Expecting exactly 256 bytes from ESP32
    if len(data) == 256:
        return np.frombuffer(data, dtype=np.uint8)  # Return as a 1D array of size 256
    else:
        return None
    
def read_optical_flow_vector(ser):
    # Read 4 bytes: 2 bytes for u and 2 bytes for v components of optical flow
    data = ser.read(4)  # Expecting 4 bytes: 2 for u and 2 for v
    if len(data) == 4:
        u = int.from_bytes(data[0:2], byteorder='big', signed=True)
        v = int.from_bytes(data[2:4], byteorder='big', signed=True)
        return u, v
    else:
        print(f"Failed to read 4 bytes, received {len(data)} bytes.")
        return None, None


def processdata(videoname, datacount):
    # Path to your video file
    src_path = os.path.join("src", videoname)

    cap = cv2.VideoCapture(src_path)

    # Set up serial communication
    ser = serial.Serial('COM5', 500000, timeout=2)

    frame_count = 0
    flow_vectors = []  # To store the optical flow vectors for each frame

    while frame_count < datacount:  # Loop to process up to 10 frames
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Process the frame
        processed_frame = preprocess_frame(frame)

        # Send the processed frame to ESP32
        send_frame_to_esp32(processed_frame, ser)

        # Wait to receive the optical flow after the frame is processed
        if (frame_count > 0):
            u, v = read_optical_flow_vector(ser)
            if u is not None and v is not None:
                # Store the u, v values
                scaled_u = u / 10000  # Scale the u component
                scaled_v = v / 10000  # Scale the v component
                flow_vectors.append((frame_count, scaled_u, scaled_v))
            else:
                print("Failed to receive optical flow data.")

        frame_count += 1

    # Close the serial port and video capture after processing is done
    cap.release()
    ser.close()
    return flow_vectors

