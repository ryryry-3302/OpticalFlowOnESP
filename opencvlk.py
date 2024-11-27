import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def flowFarneback(videotitle,datacount,showVid):
    # Open the video stream
    path = os.path.join("src", videotitle)
    cap = cv.VideoCapture(path)

    # Read the first frame
    ret, frame1 = cap.read()
    if not ret:
        print('No frames grabbed!')
        exit()

    # Convert the first frame to grayscale
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    # Get the center of the frame and define the size of the window
    frame_height, frame_width = prvs.shape
    center = (frame_width // 2, frame_height // 2)
    window_size = 8  # 8x8 center window

    # Define the region of interest (ROI) as a 8x8 window around the center
    roi_top_left = (center[0] - window_size // 2, center[1] - window_size // 2)
    roi_bottom_right = (center[0] + window_size // 2, center[1] + window_size // 2)

    # List to store magnitudes at pixel (8, 8) of each frame
    magnitude_angle_at_8_8 =[]
    isFirst = True
    while len(magnitude_angle_at_8_8) < datacount-5:
        # Read the next frame
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        
        # Convert the frame to grayscale
        next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        if isFirst:
            isFirst = False
            prvs = next_frame
            continue
        # Extract the 8x8 window (ROI) from the center of both current and previous frames
        prvs_roi = prvs[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
        next_roi = next_frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

        # Calculate optical flow using Farneback method on the 8x8 window
        flow = cv.calcOpticalFlowFarneback(prvs_roi, next_roi, None, 0.5, 3, 5, 3, 5, 1.2, 0)

        # Convert flow to magnitude and angle
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Get the magnitude at position (8,8) within the 8x8 region (corresponding to index (4,4) in the flow array)
        magnitude_at_8_8 = mag[4, 4]  # Since (8,8) is in the middle of the 8x8 window, it's at (4,4) in the array
        angle_at_8_8 = ang[4, 4]  
        
        # Store the magnitude at (8, 8)
        magnitude_angle_at_8_8.append((magnitude_at_8_8, angle_at_8_8))
        if showVid:
            # Optional: Visualization code to show optical flow
            hsv_small = np.zeros((window_size, window_size, 3), dtype=np.uint8)
            hsv_small[..., 0] = ang * 180 / np.pi / 2  # Angle as hue
            hsv_small[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # Magnitude as value
            hsv_small[..., 1] = 255  # Full saturation

            # Convert HSV to BGR for visualization
            hsv = np.zeros_like(frame2)
            hsv[..., 1] = 255  # Set saturation to 255 for full saturation
            hsv[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] = hsv_small
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

            # Overlay the optical flow visualization on the original frame
            frame2_copy = frame2.copy()
            frame2_copy[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] = bgr[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
            
            # Display the result (optional)
            cv.imshow('Optical Flow in 8x8 Window', frame2_copy)

            # Wait for key press, exit on 'ESC' or save image on 's'
            k = cv.waitKey(30) & 0xff
            if k == 27:  # ESC key
                break
            elif k == ord('s'):  # 's' key to save image
                cv.imwrite('opticalfb_window.png', next_frame)
                cv.imwrite('opticalhsv_window.png', bgr)

        # Update previous frame for next iteration
        prvs = next_frame

    # Cleanup
    cap.release()
    cv.destroyAllWindows()
    return magnitude_angle_at_8_8
