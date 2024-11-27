import cv2
import os
import serial
import numpy as np

# Used to process the center 16x16 region of the frame to grayscale
def preprocess_frame(frame):
    # Convert to grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get the center 16x16 region of the grayscale frame
    height, width = grayscale.shape
    start_x = (width - 16) // 2
    start_y = (height - 16) // 2
    center_16x16 = grayscale[start_y:start_y + 16, start_x:start_x + 16]
    blurred_center = cv2.GaussianBlur(center_16x16, (5, 5), 0)  # Kernel size (5, 5), you can adjust it
    
    # Flatten the 16x16 region to a 1D array and return it
    return blurred_center.flatten()

def send_frame_to_esp32(frame_data, ser):
    ser.write(frame_data.tobytes())  # Send the flattened data

def read_ser(ser):
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
                scaled_u = u/100.0  # Scale the u component
                scaled_v = v/100.0  # Scale the v component
                flow_vectors.append((frame_count, scaled_u, scaled_v))
            else:
                print("Failed to receive optical flow data.")

        frame_count += 1

    # Close the serial port and video capture after processing is done
    cap.release()
    ser.close()
    return flow_vectors
